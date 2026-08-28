use crate::inference::admission::{
    ActualUsage, AdmissionEstimate, AdmissionRejection, LogicalAdmissionTicket, RouteTurnPermit,
    TerminalDisposition,
};
use crate::inference::health::{ProbeClaimResult, ProbeLease, ShadowObservationMode};
use crate::inference::{
    AttemptFailure, AttemptFailureKind, AttemptOutcome, AttemptStage, AttemptTerminal,
    CompletionEvidence, InferenceAttempt, InferenceExecution, InferenceIntent, InferenceSurface,
    ReplaySafety, WorkloadClass,
};
use crate::inference_planning::{ProviderPreference, RoutePlan, RoutePlanningError};
use crate::model_config::{
    model_alias_requires_flag_lookup, model_catalog_response, openai_models_response,
    ModelAliasTargets, ModelPlan,
};
use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::provider_client::{
    ProviderClient, ProviderRequest, ProviderRequestError, ProviderResponse, ProviderSendTrace,
    UpstreamProviderError,
};
use crate::provider_registry::{ProviderId, PROVIDER_REGISTRY, SHADOW_ROUTING_POLICY_VERSION};
use crate::provider_routing::{
    compare_shadow_route, ProviderRouter, ProviderRoutingError, SelectedProviderRoute,
    ShadowRouteComparison,
};
use crate::proxy_config::{ProxyConfig, ProxyRouter};
use crate::sqs::UsageEvent;
use crate::web::audio_utils::{merge_transcriptions, AudioSplitter, TINFOIL_MAX_SIZE};
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use axum::http::{header, HeaderMap, HeaderName, StatusCode};
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use bigdecimal::BigDecimal;
use chrono::Utc;
use futures::StreamExt;
use reqwest::Method;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

// Maximum audio file size (100MB) - sanity check, CF already limits to 50MB
const MAX_AUDIO_SIZE: usize = 100 * 1024 * 1024;

// Timeout constants for provider requests
const REQUEST_TIMEOUT_SECS: u64 = 120; // Request timeout (generous for large non-streaming responses)
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 120; // Per-chunk timeout for streaming reads
const UNKNOWN_ROUTE_IMAGE_TOKEN_RESERVATION: u64 = 16_384;
const COMPLETION_PROMPT_BASE_TOKEN_OVERHEAD: u64 = 512;
const COMPLETION_PROMPT_MESSAGE_TOKEN_OVERHEAD: u64 = 64;
const COMPLETION_PROMPT_TOOL_TOKEN_OVERHEAD: u64 = 256;
const COMPLETION_PROMPT_TOOL_CALL_TOKEN_OVERHEAD: u64 = 64;
const COMPLETION_PROMPT_ARGUMENT_ENTRY_TOKEN_OVERHEAD: u64 = 16;
const COMPLETION_PROMPT_LINE_PREFIX_TOKEN_OVERHEAD: u64 = 4;
const MAX_COMPLETION_NAME_BYTES: usize = 64;
const MAX_COMPLETION_TOOL_ARGUMENT_BYTES: usize = 1024 * 1024;
const TTS_BILLING_CHECK_TIMEOUT: Duration = Duration::from_secs(5);
const TTS_PROVIDER_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_TTS_INPUT_CHARS: usize = 100_000;
const MAX_BOUNDED_PROVIDER_RESPONSE_BYTES: usize = 256 * 1024;

const PROVIDER_MANAGED_CACHE_SALT_FIELD: &str = "cache_salt";
const PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD: &str = "user_cache_secret";
const PROVIDER_MANAGED_KV_TRANSFER_PARAMS_FIELD: &str = "kv_transfer_params";
const TINFOIL_TOOL_AUTO_CONTINUE_FIELD: &str = "x-tinfoil-tool-auto-continue";
const TINFOIL_ROUTER_EXECUTION_FIELDS: &[&str] = &[
    "auto_model_options",
    "code_execution_options",
    "filters",
    "pii_check_options",
    "prompt_injection_check_options",
    "web_search_options",
];

#[derive(Clone, Default)]
struct CompletionRequestLogMetadata {
    body_bytes: usize,
    top_level_keys: usize,
    stream: Option<bool>,
    stream_options_include_usage: Option<bool>,
    max_tokens_present: bool,
    max_tokens_is_null: bool,
    max_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    tool_choice_kind: Option<&'static str>,
    parallel_tool_calls: Option<bool>,
    tools_count: usize,
    tools_json_bytes: usize,
    include_reasoning: Option<bool>,
    chat_template_enable_thinking: Option<bool>,
    messages_json_bytes: usize,
    messages: MessageLogMetadata,
}

impl std::fmt::Debug for CompletionRequestLogMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionRequestLogMetadata")
            .field("body_bytes", &self.body_bytes)
            .field("top_level_keys", &self.top_level_keys)
            .field("stream", &self.stream)
            .field(
                "stream_options_include_usage",
                &self.stream_options_include_usage,
            )
            .field("max_tokens_present", &self.max_tokens_present)
            .field("max_tokens_is_null", &self.max_tokens_is_null)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("tool_choice_kind", &self.tool_choice_kind)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("tools_count", &self.tools_count)
            .field("tools_json_bytes", &self.tools_json_bytes)
            .field("include_reasoning", &self.include_reasoning)
            .field(
                "chat_template_enable_thinking",
                &self.chat_template_enable_thinking,
            )
            .field("messages_json_bytes", &self.messages_json_bytes)
            .field("messages", &self.messages)
            .finish()
    }
}

#[derive(Debug, Clone, Default)]
struct MessageLogMetadata {
    total: usize,
    role_system: usize,
    role_user: usize,
    role_assistant: usize,
    role_tool: usize,
    role_other: usize,
    missing_role: usize,
    content_string: usize,
    content_array: usize,
    content_null: usize,
    content_missing: usize,
    content_other: usize,
    empty_string_content: usize,
    text_parts: usize,
    image_parts: usize,
    image_data_url_parts: usize,
    image_remote_url_parts: usize,
    image_other_url_parts: usize,
    file_parts: usize,
    unknown_parts: usize,
    assistant_tool_call_messages: usize,
    assistant_tool_calls: usize,
    tool_call_args_total_bytes: usize,
    tool_call_args_max_bytes: usize,
    tool_calls_missing_id: usize,
    tool_calls_duplicate_id: usize,
    tool_calls_missing_function: usize,
    tool_calls_missing_arguments: usize,
    tool_messages_missing_tool_call_id: usize,
    tool_messages_matched_tool_call_id: usize,
    tool_messages_unmatched_tool_call_id: usize,
    total_message_json_bytes: usize,
    max_message_json_bytes: usize,
    total_content_json_bytes: usize,
    max_content_json_bytes: usize,
}

impl CompletionRequestLogMetadata {
    fn from_body(body: &Value, body_bytes: usize) -> Self {
        let tools = body.get("tools");
        let messages = body.get("messages");

        Self {
            body_bytes,
            top_level_keys: body.as_object().map(|obj| obj.len()).unwrap_or_default(),
            stream: body.get("stream").and_then(Value::as_bool),
            stream_options_include_usage: body
                .get("stream_options")
                .and_then(|opts| opts.get("include_usage"))
                .and_then(Value::as_bool),
            max_tokens_present: body.get("max_tokens").is_some(),
            max_tokens_is_null: body.get("max_tokens").is_some_and(Value::is_null),
            max_tokens: body.get("max_tokens").and_then(Value::as_i64),
            temperature: body.get("temperature").and_then(Value::as_f64),
            top_p: body.get("top_p").and_then(Value::as_f64),
            tool_choice_kind: body.get("tool_choice").map(value_kind),
            parallel_tool_calls: body.get("parallel_tool_calls").and_then(Value::as_bool),
            tools_count: tools
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or_default(),
            tools_json_bytes: tools.map(json_value_len).unwrap_or_default(),
            include_reasoning: body.get("include_reasoning").and_then(Value::as_bool),
            chat_template_enable_thinking: body
                .get("chat_template_kwargs")
                .and_then(|kwargs| kwargs.get("enable_thinking"))
                .and_then(Value::as_bool),
            messages_json_bytes: messages.map(json_value_len).unwrap_or_default(),
            messages: MessageLogMetadata::from_messages(messages.and_then(Value::as_array)),
        }
    }
}

impl MessageLogMetadata {
    fn from_messages(messages: Option<&Vec<Value>>) -> Self {
        let mut metadata = Self::default();
        let mut tool_call_ids = HashSet::new();
        let mut tool_message_ids = Vec::new();

        let Some(messages) = messages else {
            return metadata;
        };

        metadata.total = messages.len();

        for message in messages {
            let message_json_bytes = json_value_len(message);
            metadata.total_message_json_bytes += message_json_bytes;
            metadata.max_message_json_bytes =
                metadata.max_message_json_bytes.max(message_json_bytes);

            match message.get("role").and_then(Value::as_str) {
                Some("system") => metadata.role_system += 1,
                Some("user") => metadata.role_user += 1,
                Some("assistant") => metadata.role_assistant += 1,
                Some("tool") => metadata.role_tool += 1,
                Some(_) => metadata.role_other += 1,
                None => metadata.missing_role += 1,
            }

            metadata.record_content(message.get("content"));
            metadata.record_assistant_tool_calls(message.get("tool_calls"), &mut tool_call_ids);

            if message.get("role").and_then(Value::as_str) == Some("tool") {
                if let Some(tool_call_id) = message.get("tool_call_id").and_then(Value::as_str) {
                    tool_message_ids.push(tool_call_id.to_string());
                } else {
                    metadata.tool_messages_missing_tool_call_id += 1;
                }
            }
        }

        for tool_message_id in tool_message_ids {
            if tool_call_ids.contains(&tool_message_id) {
                metadata.tool_messages_matched_tool_call_id += 1;
            } else {
                metadata.tool_messages_unmatched_tool_call_id += 1;
            }
        }

        metadata
    }

    fn record_content(&mut self, content: Option<&Value>) {
        let Some(content) = content else {
            self.content_missing += 1;
            return;
        };

        let content_json_bytes = json_value_len(content);
        self.total_content_json_bytes += content_json_bytes;
        self.max_content_json_bytes = self.max_content_json_bytes.max(content_json_bytes);

        match content {
            Value::Null => self.content_null += 1,
            Value::String(text) => {
                self.content_string += 1;
                if text.is_empty() {
                    self.empty_string_content += 1;
                }
            }
            Value::Array(parts) => {
                self.content_array += 1;
                for part in parts {
                    self.record_content_part(part);
                }
            }
            _ => self.content_other += 1,
        }
    }

    fn record_content_part(&mut self, part: &Value) {
        match part.get("type").and_then(Value::as_str) {
            Some("text") | Some("input_text") => self.text_parts += 1,
            Some("image_url") | Some("input_image") => {
                self.image_parts += 1;
                match image_part_url(part) {
                    Some(url) if url.starts_with("data:") => self.image_data_url_parts += 1,
                    Some(url) if url.starts_with("http://") || url.starts_with("https://") => {
                        self.image_remote_url_parts += 1;
                    }
                    Some(_) => self.image_other_url_parts += 1,
                    None => self.image_other_url_parts += 1,
                }
            }
            Some("file") | Some("input_file") => self.file_parts += 1,
            _ => self.unknown_parts += 1,
        }
    }

    fn record_assistant_tool_calls(
        &mut self,
        tool_calls: Option<&Value>,
        tool_call_ids: &mut HashSet<String>,
    ) {
        let Some(tool_calls) = tool_calls.and_then(Value::as_array) else {
            return;
        };

        if !tool_calls.is_empty() {
            self.assistant_tool_call_messages += 1;
        }

        for tool_call in tool_calls {
            self.assistant_tool_calls += 1;

            match tool_call.get("id").and_then(Value::as_str) {
                Some(id) => {
                    if !tool_call_ids.insert(id.to_string()) {
                        self.tool_calls_duplicate_id += 1;
                    }
                }
                None => self.tool_calls_missing_id += 1,
            }

            let Some(function) = tool_call.get("function") else {
                self.tool_calls_missing_function += 1;
                self.tool_calls_missing_arguments += 1;
                continue;
            };

            match function.get("arguments") {
                Some(Value::String(arguments)) => {
                    let argument_bytes = arguments.len();
                    self.tool_call_args_total_bytes += argument_bytes;
                    self.tool_call_args_max_bytes =
                        self.tool_call_args_max_bytes.max(argument_bytes);
                }
                Some(arguments) => {
                    let argument_bytes = json_value_len(arguments);
                    self.tool_call_args_total_bytes += argument_bytes;
                    self.tool_call_args_max_bytes =
                        self.tool_call_args_max_bytes.max(argument_bytes);
                }
                None => self.tool_calls_missing_arguments += 1,
            }
        }
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn json_value_len(value: &Value) -> usize {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or_default()
}

fn image_part_url(part: &Value) -> Option<&str> {
    part.get("image_url")
        .and_then(|image| {
            image
                .get("url")
                .and_then(Value::as_str)
                .or_else(|| image.as_str())
        })
        .or_else(|| part.get("image").and_then(Value::as_str))
}

/// Parameters for transcription requests
struct TranscriptionParams<'a> {
    audio_data: &'a [u8],
    filename: &'a str,
    content_type: &'a str,
    language: Option<&'a str>,
    prompt: Option<&'a str>,
    response_format: &'a str,
    temperature: Option<f64>,
}

/// Request structure for TTS (Text-to-Speech) endpoints
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(transparent)]
struct TTSRequest {
    provider_payload: serde_json::Map<String, Value>,
}

const VOXTRAL_TTS_MODEL: &str = "voxtral-tts";
const DEFAULT_VOXTRAL_TTS_VOICE: &str = "neutral_female";
const VOXTRAL_TTS_VOICE_CONDITIONING_FIELDS: [&str; 4] =
    ["voice", "speaker", "ref_audio", "references"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
enum TTSRequestValidationError {
    #[error("input text is empty")]
    EmptyInput,
    #[error("input text exceeds the maximum length")]
    InputTooLong,
    #[error("unsupported TTS model")]
    UnsupportedModel,
}

#[derive(Debug, PartialEq)]
struct PreparedTTSRequest {
    model: String,
    voice_for_log: String,
    provider_payload: Value,
}

fn prepare_tts_request(
    mut request: TTSRequest,
) -> Result<PreparedTTSRequest, TTSRequestValidationError> {
    let Some(input) = request
        .provider_payload
        .get("input")
        .and_then(Value::as_str)
    else {
        return Err(TTSRequestValidationError::EmptyInput);
    };
    if input.trim().is_empty() {
        return Err(TTSRequestValidationError::EmptyInput);
    }
    if input.chars().count() > MAX_TTS_INPUT_CHARS {
        return Err(TTSRequestValidationError::InputTooLong);
    }

    if let Some(model) = request.provider_payload.get("model") {
        if model.as_str() != Some(VOXTRAL_TTS_MODEL) {
            return Err(TTSRequestValidationError::UnsupportedModel);
        }
    }

    let provider_payload = &mut request.provider_payload;
    provider_payload
        .entry("model".to_string())
        .or_insert_with(|| Value::String(VOXTRAL_TTS_MODEL.to_string()));

    if !VOXTRAL_TTS_VOICE_CONDITIONING_FIELDS
        .iter()
        .any(|field| provider_payload.contains_key(*field))
    {
        provider_payload.insert(
            "voice".to_string(),
            Value::String(DEFAULT_VOXTRAL_TTS_VOICE.to_string()),
        );
    }

    let voice_for_log = match provider_payload.get("voice") {
        Some(Value::String(voice)) => voice.clone(),
        Some(Value::Null) => "<null>".to_string(),
        Some(_) => "<non-string>".to_string(),
        None => "<provider-conditioned>".to_string(),
    };

    Ok(PreparedTTSRequest {
        model: VOXTRAL_TTS_MODEL.to_string(),
        voice_for_log,
        provider_payload: Value::Object(request.provider_payload),
    })
}

fn build_tts_response_payload(body_bytes: &[u8], content_type: &str) -> (Value, bool) {
    let is_json_response = serde_json::from_slice::<Value>(body_bytes).is_ok();
    (
        json!({
            "content_base64": general_purpose::STANDARD.encode(body_bytes),
            "content_type": content_type,
        }),
        is_json_response,
    )
}

/// Request structure for transcription endpoints
#[derive(Debug, Clone, serde::Deserialize)]
struct TranscriptionRequest {
    file: String, // Base64 encoded audio file
    #[serde(default = "default_transcription_filename")]
    filename: String,
    #[serde(default = "default_transcription_content_type")]
    content_type: String,
    #[serde(default = "default_transcription_model")]
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    #[serde(default = "default_transcription_response_format")]
    response_format: String,
    temperature: Option<f64>,
}

fn default_transcription_filename() -> String {
    "audio.mp3".to_string()
}

fn default_transcription_content_type() -> String {
    "audio/mpeg".to_string()
}

fn default_transcription_model() -> String {
    "whisper-large-v3".to_string()
}

fn default_transcription_response_format() -> String {
    "json".to_string()
}

/// Request structure for embeddings endpoints
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct EmbeddingRequest {
    input: serde_json::Value, // string or array of strings
    #[serde(default = "default_embedding_model")]
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

fn default_embedding_model() -> String {
    "nomic-embed-text".to_string()
}

// ============================================================================
// Centralized Billing Architecture - New Types
// ============================================================================

/// Context needed for billing/usage tracking
#[derive(Debug, Clone)]
pub struct BillingContext {
    pub auth_method: AuthMethod,
    pub model_name: String,
}

impl BillingContext {
    pub fn new(auth_method: AuthMethod, model_name: String) -> Self {
        Self {
            auth_method,
            model_name,
        }
    }
}

/// A single, move-only billing settlement for one logical provider model turn.
///
/// `request_id` can span a complete Responses tool loop, while `execution_id`
/// identifies exactly one model turn. Reusing the execution UUID as the SQS
/// event id makes an accidentally repeated publish idempotent at the billing
/// consumer without introducing distributed coordination in OpenSecret.
#[derive(Debug)]
struct UsageSettlement {
    event_id: Uuid,
    request_id: crate::inference::InferenceRequestId,
    execution_id: crate::inference::InferenceExecutionId,
    attempt_id: crate::inference::InferenceAttemptId,
    requested_model_id: String,
    public_model_id: String,
    provider_name: String,
    auth_method: AuthMethod,
}

impl UsageSettlement {
    fn new(billing_context: BillingContext, attempt: &InferenceAttempt) -> Self {
        Self {
            event_id: attempt.execution_id.as_uuid(),
            request_id: attempt.request_id,
            execution_id: attempt.execution_id,
            attempt_id: attempt.attempt_id,
            requested_model_id: billing_context.model_name,
            public_model_id: attempt.route.public_model_id.clone(),
            provider_name: attempt.route.provider.as_str().to_string(),
            auth_method: billing_context.auth_method,
        }
    }
}

/// Usage statistics extracted from a completion
#[derive(Debug, Clone)]
pub struct CompletionUsage {
    pub prompt_tokens: i32,
    /// Whether the provider explicitly reported a prompt-token total.
    /// A synthesized zero is not sufficient for refunding reserved input capacity.
    pub prompt_tokens_observed: bool,
    pub completion_tokens: i32,
    /// Whether the provider explicitly reported a completion-token total.
    /// A synthesized zero is not sufficient for enforcing aggregate output budgets.
    pub completion_tokens_observed: bool,
    pub cached_prompt_tokens: Option<i32>,
}

#[derive(Debug, Default)]
struct CompletionUsageObservation {
    prompt_tokens: Option<i32>,
    completion_tokens: Option<i32>,
    cached_prompt_tokens: Option<i32>,
}

impl CompletionUsageObservation {
    fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cached_prompt_tokens.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamUsageFinalization {
    ProviderDone,
    EndOfStream,
    ProviderError,
    TransportError,
    Timeout,
    ConsumerDropped,
    InvalidData,
}

impl StreamUsageFinalization {
    fn is_provider_done(self) -> bool {
        self == Self::ProviderDone
    }
}

#[derive(Debug, Default)]
struct StreamUsageAccumulator {
    latest_usage: Option<CompletionUsage>,
    saw_terminal_signal: bool,
    finalized: bool,
}

impl StreamUsageAccumulator {
    fn observe(&mut self, json: &Value) {
        self.saw_terminal_signal |= has_terminal_stream_signal(json);

        let Some(observed) = extract_usage_observation(json) else {
            return;
        };
        if observed.is_empty() {
            return;
        }

        if let Some(previous) = &self.latest_usage {
            if observed
                .prompt_tokens
                .is_some_and(|tokens| tokens < previous.prompt_tokens)
                || observed
                    .completion_tokens
                    .is_some_and(|tokens| tokens < previous.completion_tokens)
            {
                warn!(
                    "Streaming usage totals regressed: previous_prompt_tokens={}, previous_completion_tokens={}, observed_prompt_tokens={:?}, observed_completion_tokens={:?}; preserving the highest cumulative totals",
                    previous.prompt_tokens,
                    previous.completion_tokens,
                    observed.prompt_tokens,
                    observed.completion_tokens
                );
            }
        }

        let prompt_tokens = observed
            .prompt_tokens
            .map(|tokens| {
                self.latest_usage
                    .as_ref()
                    .map_or(tokens, |usage| tokens.max(usage.prompt_tokens))
            })
            .or_else(|| self.latest_usage.as_ref().map(|usage| usage.prompt_tokens))
            .unwrap_or(0);
        let completion_tokens = observed
            .completion_tokens
            .map(|tokens| {
                self.latest_usage
                    .as_ref()
                    .map_or(tokens, |usage| tokens.max(usage.completion_tokens))
            })
            .or_else(|| {
                self.latest_usage
                    .as_ref()
                    .map(|usage| usage.completion_tokens)
            })
            .unwrap_or(0);
        self.latest_usage = Some(CompletionUsage {
            prompt_tokens,
            prompt_tokens_observed: observed.prompt_tokens.is_some()
                || self
                    .latest_usage
                    .as_ref()
                    .is_some_and(|usage| usage.prompt_tokens_observed),
            completion_tokens,
            completion_tokens_observed: observed.completion_tokens.is_some()
                || self
                    .latest_usage
                    .as_ref()
                    .is_some_and(|usage| usage.completion_tokens_observed),
            cached_prompt_tokens: observed.cached_prompt_tokens.or_else(|| {
                self.latest_usage
                    .as_ref()
                    .and_then(|usage| usage.cached_prompt_tokens)
            }),
        });
    }

    fn take_final_usage(
        &mut self,
        finalization: StreamUsageFinalization,
    ) -> Option<CompletionUsage> {
        let can_finalize = finalization.is_provider_done() || self.saw_terminal_signal;
        if self.finalized || !can_finalize {
            return None;
        }

        self.finalized = true;
        let mut usage = self.latest_usage.take()?;
        if usage.prompt_tokens_observed {
            usage.cached_prompt_tokens = usage
                .cached_prompt_tokens
                .map(|cached| cached.min(usage.prompt_tokens));
        }
        (usage.prompt_tokens_observed
            || usage.completion_tokens_observed
            || usage.cached_prompt_tokens.is_some())
        .then_some(usage)
    }
}

/// A chunk from the completion stream
#[derive(Clone, Debug)]
pub enum CompletionChunk {
    /// Streaming chunk with full JSON from upstream (includes all metadata)
    StreamChunk(Value),
    /// Complete response for non-streaming
    FullResponse(Value),
    /// Usage information (for streaming with include_usage)
    Usage(CompletionUsage),
    /// Exactly one typed terminal outcome for the upstream attempt.
    Terminal(AttemptTerminal),
}

/// Metadata about the completion
#[derive(Clone, Debug)]
pub struct CompletionMetadata {
    pub provider_name: String,
    pub model_name: String,
    pub is_streaming: bool,
    pub(crate) attempt: InferenceAttempt,
}

/// Processed completion stream - billing happens automatically
pub struct CompletionStream {
    /// The actual data stream for consumers
    pub stream: mpsc::Receiver<CompletionChunk>,
    /// Metadata about the completion
    pub metadata: CompletionMetadata,
}

#[derive(Debug, Clone)]
pub(crate) struct PinnedCompletionRequest {
    intent: InferenceIntent,
    /// A pure provisional decision used for model-compatible validation and
    /// context construction before scarce admission is acquired.
    route: SelectedProviderRoute,
    provider_preference: Option<ProviderPreference>,
    finalized_route: Arc<OnceLock<SelectedProviderRoute>>,
}

impl PinnedCompletionRequest {
    pub(crate) fn intent(&self) -> &InferenceIntent {
        &self.intent
    }

    pub(crate) fn public_model_id(&self) -> &str {
        &self.selected_route().public_model_id
    }

    fn begin_execution(&self) -> InferenceExecution {
        self.intent.begin_execution()
    }

    fn selected_route(&self) -> &SelectedProviderRoute {
        self.finalized_route.get().unwrap_or(&self.route)
    }

    fn finalize_route(&self, route: SelectedProviderRoute) -> bool {
        self.finalized_route.set(route).is_ok()
    }
}

fn pin_chat_request_model(body: &mut Value, pinned: &PinnedCompletionRequest) {
    body.as_object_mut()
        .expect("model was read from a JSON object")
        .insert("model".to_string(), json!(pinned.public_model_id()));
}

#[derive(Debug)]
pub(crate) enum CompletionExecutionError {
    Request(ApiError),
    Attempt {
        terminal: AttemptTerminal,
        public_error: ApiError,
    },
}

impl CompletionExecutionError {
    pub(crate) fn terminal(&self) -> Option<&AttemptTerminal> {
        match self {
            Self::Request(_) => None,
            Self::Attempt { terminal, .. } => Some(terminal),
        }
    }

    pub(crate) fn into_api_error(self) -> ApiError {
        let has_terminal = self.terminal().is_some();
        match self {
            Self::Request(error) => {
                debug_assert!(!has_terminal);
                error
            }
            Self::Attempt { public_error, .. } => {
                debug_assert!(has_terminal);
                public_error
            }
        }
    }

    pub(crate) fn into_pre_persistence_api_error(self) -> ApiError {
        self.into_api_error().with_client_replay_safe()
    }
}

impl From<ApiError> for CompletionExecutionError {
    fn from(error: ApiError) -> Self {
        Self::Request(error)
    }
}

#[cfg(test)]
fn failed_completion_execution(
    provider_router: &ProviderRouter,
    attempt: InferenceAttempt,
    failure: AttemptFailure,
    public_error: ApiError,
) -> CompletionExecutionError {
    let terminal = AttemptTerminal::Failed { attempt, failure };
    record_attempt_outcome(
        provider_router,
        &AttemptOutcome::Terminal(terminal.clone()),
        ShadowObservationMode::Update,
    );
    CompletionExecutionError::Attempt {
        terminal,
        public_error,
    }
}

fn attempt_failure_from_provider_error(error: &ProviderRequestError) -> AttemptFailure {
    match error {
        ProviderRequestError::TinfoilUnavailable => AttemptFailure::new(
            AttemptFailureKind::ProviderUnavailable,
            AttemptStage::BeforeSend,
            ReplaySafety::ProvenPreAcceptance,
        ),
        ProviderRequestError::Build(_) => AttemptFailure::new(
            AttemptFailureKind::RequestBuild,
            AttemptStage::BeforeSend,
            ReplaySafety::ProvenPreAcceptance,
        ),
        ProviderRequestError::Connect(_) => AttemptFailure::new(
            AttemptFailureKind::Connect,
            AttemptStage::BeforeSend,
            ReplaySafety::ProvenPreAcceptance,
        ),
        ProviderRequestError::Timeout(_) => AttemptFailure::new(
            AttemptFailureKind::ResponseStartTimeout,
            AttemptStage::AwaitingResponse,
            ReplaySafety::NotProvenPreAcceptance,
        ),
        ProviderRequestError::Send(_) => AttemptFailure::new(
            AttemptFailureKind::Transport,
            AttemptStage::AwaitingResponse,
            ReplaySafety::NotProvenPreAcceptance,
        ),
        ProviderRequestError::Upstream(upstream) => {
            let is_capacity_rejection = public_capacity_status(upstream.status).is_some();
            AttemptFailure::new(
                if is_capacity_rejection {
                    AttemptFailureKind::CapacityRejected
                } else {
                    AttemptFailureKind::HttpStatus
                },
                AttemptStage::AwaitingResponse,
                if is_capacity_rejection {
                    ReplaySafety::ProvenPreAcceptance
                } else {
                    ReplaySafety::NotProvenPreAcceptance
                },
            )
            .with_upstream_response(
                upstream.status,
                upstream.retry_after,
                upstream.upstream_request_id.clone(),
            )
        }
    }
}

fn public_capacity_status(upstream_status: u16) -> Option<StatusCode> {
    match upstream_status {
        429 => Some(StatusCode::TOO_MANY_REQUESTS),
        503 | 529 => Some(StatusCode::SERVICE_UNAVAILABLE),
        _ => None,
    }
}

fn public_completion_error(error: &ProviderRequestError, failure: &AttemptFailure) -> ApiError {
    if failure.kind == AttemptFailureKind::CapacityRejected {
        let status = failure
            .status
            .and_then(public_capacity_status)
            .expect("capacity failure must carry a supported upstream status");
        return ApiError::InferenceCapacity {
            status,
            retry_after: failure.retry_after,
            client_replay_safe: false,
        };
    }

    ApiError::from(error.clone())
}

fn terminalize_recovered_provider_failures(
    provider_router: &ProviderRouter,
    execution: InferenceExecution,
    route: &crate::inference::RouteIdentity,
    prior_failures: Vec<ProviderRequestError>,
) -> Vec<AttemptTerminal> {
    prior_failures
        .into_iter()
        .map(|prior_error| {
            let attempt = execution.begin_attempt(route.clone());
            let failure = attempt_failure_from_provider_error(&prior_error);
            let terminal = AttemptTerminal::Failed {
                attempt: attempt.clone(),
                failure: failure.clone(),
            };
            record_attempt_outcome(
                provider_router,
                &AttemptOutcome::Terminal(terminal.clone()),
                ShadowObservationMode::TelemetryOnly,
            );
            warn!(
                "Inference transport recovered a failed attempt: request_id={}, execution_id={}, attempt_id={}, kind={:?}, replay_safety={:?}",
                attempt.request_id,
                attempt.execution_id,
                attempt.attempt_id,
                failure.kind,
                failure.replay_safety
            );
            terminal
        })
        .collect()
}

fn record_attempt_outcome(
    provider_router: &ProviderRouter,
    outcome: &AttemptOutcome,
    shadow_mode: ShadowObservationMode,
) {
    match outcome {
        AttemptOutcome::ResponseStarted { attempt, status } => {
            let route_key = attempt.route.route_key();
            debug!(
                "Inference attempt response started: request_id={}, execution_id={}, attempt_id={}, provider={}, public_model={}, provider_model={}, response_model={}, route_provider={}, route_model={}, source={:?}, bucket={:?}, status={}",
                attempt.request_id,
                attempt.execution_id,
                attempt.attempt_id,
                attempt.route.provider.as_str(),
                attempt.route.public_model_id,
                attempt.route.provider_model_id,
                attempt.route.response_model_id,
                route_key.provider.as_str(),
                route_key.provider_model_id,
                attempt.route.selection_source,
                attempt.route.bucket,
                status
            );
        }
        AttemptOutcome::Terminal(terminal) => {
            let shadow_report = provider_router.observe_attempt_terminal(terminal, shadow_mode);
            log_attempt_terminal(terminal, &shadow_report);
        }
    }
}

fn record_attempt_terminal_with_probe(
    provider_router: &ProviderRouter,
    terminal: &AttemptTerminal,
    shadow_mode: ShadowObservationMode,
    probe: Option<ProbeLease>,
) {
    let shadow_report =
        provider_router.observe_attempt_terminal_with_probe(terminal, shadow_mode, probe);
    log_attempt_terminal(terminal, &shadow_report);
}

fn log_attempt_terminal(
    terminal: &AttemptTerminal,
    shadow_report: &crate::inference::health::ShadowObservationReport,
) {
    let attempt = terminal.attempt();
    debug!(
        "Inference shadow health observation: request_id={}, execution_id={}, attempt_id={}, policy_version={}, mode={:?}, route_provider={}, route_model={}, signal={:?}, capacity_pool={:?}, snapshot={:?}, mutated={}",
        attempt.request_id,
        attempt.execution_id,
        attempt.attempt_id,
        shadow_report.policy_version,
        shadow_report.mode,
        shadow_report.route.provider.as_str(),
        shadow_report.route.provider_model_id,
        shadow_report.signal,
        shadow_report.capacity_pool,
        shadow_report.snapshot,
        shadow_report.mutated
    );
    match terminal {
        AttemptTerminal::Completed { evidence, .. } => debug!(
            "Inference attempt completed: request_id={}, execution_id={}, attempt_id={}, provider={}, public_model={}, provider_model={}, evidence={:?}",
            attempt.request_id,
            attempt.execution_id,
            attempt.attempt_id,
            attempt.route.provider.as_str(),
            attempt.route.public_model_id,
            attempt.route.provider_model_id,
            evidence
        ),
        AttemptTerminal::Failed { failure, .. } => warn!(
            "Inference attempt failed: request_id={}, execution_id={}, attempt_id={}, provider={}, public_model={}, provider_model={}, kind={:?}, stage={:?}, replay_safety={:?}, status={:?}, retry_after_ms={:?}, upstream_request_id={:?}, upstream_code={:?}",
            attempt.request_id,
            attempt.execution_id,
            attempt.attempt_id,
            attempt.route.provider.as_str(),
            attempt.route.public_model_id,
            attempt.route.provider_model_id,
            failure.kind,
            failure.stage,
            failure.replay_safety,
            failure.status,
            failure.retry_after.map(|duration| duration.as_millis()),
            failure.upstream_request_id,
            failure.upstream_code
        ),
    }
}

#[cfg(test)]
async fn send_attempt_terminal(
    provider_router: &ProviderRouter,
    sender: &mpsc::Sender<CompletionChunk>,
    terminal: AttemptTerminal,
) {
    record_attempt_outcome(
        provider_router,
        &AttemptOutcome::Terminal(terminal.clone()),
        ShadowObservationMode::Update,
    );
    let _ = sender.send(CompletionChunk::Terminal(terminal)).await;
}

fn stream_end_terminal(
    attempt: InferenceAttempt,
    saw_finish_signal: bool,
    has_incomplete_frame: bool,
) -> AttemptTerminal {
    if saw_finish_signal && !has_incomplete_frame {
        AttemptTerminal::Completed {
            attempt,
            evidence: CompletionEvidence::FinishSignalThenEof,
        }
    } else {
        AttemptTerminal::Failed {
            attempt,
            failure: AttemptFailure::new(
                AttemptFailureKind::UnexpectedEof,
                AttemptStage::Stream,
                ReplaySafety::NotProvenPreAcceptance,
            ),
        }
    }
}

struct StreamProcessResult {
    terminal: AttemptTerminal,
    finalization: StreamUsageFinalization,
    usage: Option<CompletionUsage>,
}

fn finish_stream_processing(
    accumulator: &mut StreamUsageAccumulator,
    finalization: StreamUsageFinalization,
    terminal: AttemptTerminal,
) -> StreamProcessResult {
    StreamProcessResult {
        terminal,
        finalization,
        usage: accumulator.take_final_usage(finalization),
    }
}

fn bounded_upstream_error_code(error: &Value) -> Option<String> {
    let code = error.get("code")?;
    let code = match code {
        Value::String(code) => code.clone(),
        Value::Number(code) => code.to_string(),
        _ => return None,
    };
    let code = code.trim();
    (!code.is_empty() && code.len() <= 64 && code.chars().all(is_safe_identifier_char))
        .then(|| code.to_string())
}

fn upstream_payload_failure(
    json: &Value,
    kind: AttemptFailureKind,
    stage: AttemptStage,
) -> Option<AttemptFailure> {
    let error = json.get("error").filter(|error| !error.is_null())?;
    Some(
        AttemptFailure::new(kind, stage, ReplaySafety::NotProvenPreAcceptance)
            .with_upstream_code(bounded_upstream_error_code(error)),
    )
}

fn upstream_stream_failure(json: &Value) -> Option<AttemptFailure> {
    upstream_payload_failure(
        json,
        AttemptFailureKind::UpstreamStreamError,
        AttemptStage::Stream,
    )
}

async fn read_non_streaming_completion_response(
    response: ProviderResponse,
    response_model_id: &str,
    attempt: &InferenceAttempt,
    body_limit: Option<usize>,
    body_timeout: Duration,
) -> Result<Value, AttemptFailure> {
    let body_bytes = match timeout(body_timeout, async move {
        match body_limit {
            Some(limit_bytes) => collect_bounded_provider_response_body(
                response.bytes_stream(),
                limit_bytes,
            )
            .await
            .map(bytes::Bytes::from)
            .map_err(|error| {
                error!(
                    "Failed to read bounded inference response body: request_id={}, execution_id={}, attempt_id={}, error={}",
                    attempt.request_id, attempt.execution_id, attempt.attempt_id, error
                );
                let kind = match error {
                    BoundedProviderResponseBodyError::Read => AttemptFailureKind::ResponseBody,
                    BoundedProviderResponseBodyError::TooLarge { .. } => {
                        AttemptFailureKind::InvalidResponse
                    }
                };
                AttemptFailure::new(
                    kind,
                    AttemptStage::ResponseBody,
                    ReplaySafety::NotProvenPreAcceptance,
                )
            }),
            None => response.bytes().await.map_err(|error| {
                error!(
                    "Failed to read inference response body: request_id={}, execution_id={}, attempt_id={}, error={}",
                    attempt.request_id, attempt.execution_id, attempt.attempt_id, error
                );
                AttemptFailure::new(
                    AttemptFailureKind::ResponseBody,
                    AttemptStage::ResponseBody,
                    ReplaySafety::NotProvenPreAcceptance,
                )
            }),
        }
    })
    .await
    {
        Ok(result) => result?,
        Err(_) => {
            warn!(
                "Inference response body timed out: request_id={}, execution_id={}, attempt_id={}, timeout_ms={}",
                attempt.request_id,
                attempt.execution_id,
                attempt.attempt_id,
                body_timeout.as_millis()
            );
            return Err(AttemptFailure::new(
                AttemptFailureKind::ResponseBody,
                AttemptStage::ResponseBody,
                ReplaySafety::NotProvenPreAcceptance,
            ));
        }
    };

    let mut response_json: Value = serde_json::from_slice(&body_bytes).map_err(|error| {
        error!(
            "Failed to parse inference response JSON: request_id={}, execution_id={}, attempt_id={}, error={}",
            attempt.request_id, attempt.execution_id, attempt.attempt_id, error
        );
        AttemptFailure::new(
            AttemptFailureKind::InvalidResponse,
            AttemptStage::ResponseBody,
            ReplaySafety::NotProvenPreAcceptance,
        )
    })?;

    if let Some(failure) = upstream_payload_failure(
        &response_json,
        AttemptFailureKind::UpstreamResponseError,
        AttemptStage::ResponseBody,
    ) {
        warn!(
            "Inference provider emitted a non-streaming error payload: request_id={}, execution_id={}, attempt_id={}, upstream_code={:?}",
            attempt.request_id,
            attempt.execution_id,
            attempt.attempt_id,
            failure.upstream_code
        );
        return Err(failure);
    }

    canonicalize_response_model(&mut response_json, response_model_id);
    Ok(response_json)
}

async fn process_completion_stream(
    response: ProviderResponse,
    response_model_id: &str,
    attempt: InferenceAttempt,
    tx_consumer: &mpsc::Sender<CompletionChunk>,
    chunk_timeout: Duration,
) -> StreamProcessResult {
    let mut body_stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut usage_accumulator = StreamUsageAccumulator::default();

    loop {
        let next_chunk = tokio::select! {
            biased;
            _ = tx_consumer.closed() => {
                return finish_stream_processing(
                    &mut usage_accumulator,
                    StreamUsageFinalization::ConsumerDropped,
                    AttemptTerminal::Failed {
                        attempt,
                        failure: AttemptFailure::new(
                            AttemptFailureKind::ConsumerDropped,
                            AttemptStage::Stream,
                            ReplaySafety::NotProvenPreAcceptance,
                        ),
                    },
                );
            }
            result = timeout(chunk_timeout, body_stream.next()) => result,
        };
        match next_chunk {
            Ok(Some(Ok(bytes))) => {
                buffer.extend_from_slice(bytes.as_ref());

                while let Some(frame) = extract_sse_frame(&mut buffer) {
                    if frame == b"[DONE]" {
                        return finish_stream_processing(
                            &mut usage_accumulator,
                            StreamUsageFinalization::ProviderDone,
                            AttemptTerminal::Completed {
                                attempt,
                                evidence: CompletionEvidence::ProviderDone,
                            },
                        );
                    }

                    let mut json = match serde_json::from_slice::<Value>(&frame) {
                        Ok(json) => json,
                        Err(error) => {
                            error!(
                                "Received invalid inference stream data: request_id={}, execution_id={}, attempt_id={}, error={}",
                                attempt.request_id,
                                attempt.execution_id,
                                attempt.attempt_id,
                                error
                            );
                            return finish_stream_processing(
                                &mut usage_accumulator,
                                StreamUsageFinalization::InvalidData,
                                AttemptTerminal::Failed {
                                    attempt,
                                    failure: AttemptFailure::new(
                                        AttemptFailureKind::InvalidResponse,
                                        AttemptStage::Stream,
                                        ReplaySafety::NotProvenPreAcceptance,
                                    ),
                                },
                            );
                        }
                    };

                    if let Some(failure) = upstream_stream_failure(&json) {
                        warn!(
                            "Inference provider emitted an error stream frame: request_id={}, execution_id={}, attempt_id={}, upstream_code={:?}",
                            attempt.request_id,
                            attempt.execution_id,
                            attempt.attempt_id,
                            failure.upstream_code
                        );
                        return finish_stream_processing(
                            &mut usage_accumulator,
                            StreamUsageFinalization::ProviderError,
                            AttemptTerminal::Failed { attempt, failure },
                        );
                    }

                    usage_accumulator.observe(&json);
                    canonicalize_response_model(&mut json, response_model_id);

                    if !matches!(
                        timeout(
                            chunk_timeout,
                            tx_consumer.send(CompletionChunk::StreamChunk(json))
                        )
                        .await,
                        Ok(Ok(()))
                    ) {
                        return finish_stream_processing(
                            &mut usage_accumulator,
                            StreamUsageFinalization::ConsumerDropped,
                            AttemptTerminal::Failed {
                                attempt,
                                failure: AttemptFailure::new(
                                    AttemptFailureKind::ConsumerDropped,
                                    AttemptStage::Stream,
                                    ReplaySafety::NotProvenPreAcceptance,
                                ),
                            },
                        );
                    }
                }
            }
            Ok(Some(Err(error))) => {
                error!(
                    "Inference stream transport error: request_id={}, execution_id={}, attempt_id={}, error={}",
                    attempt.request_id, attempt.execution_id, attempt.attempt_id, error
                );
                return finish_stream_processing(
                    &mut usage_accumulator,
                    StreamUsageFinalization::TransportError,
                    AttemptTerminal::Failed {
                        attempt,
                        failure: AttemptFailure::new(
                            AttemptFailureKind::Transport,
                            AttemptStage::Stream,
                            ReplaySafety::NotProvenPreAcceptance,
                        ),
                    },
                );
            }
            Ok(None) => {
                let has_incomplete_frame = buffer.iter().any(|byte| !byte.is_ascii_whitespace());
                let terminal = stream_end_terminal(
                    attempt,
                    usage_accumulator.saw_terminal_signal,
                    has_incomplete_frame,
                );
                return finish_stream_processing(
                    &mut usage_accumulator,
                    StreamUsageFinalization::EndOfStream,
                    terminal,
                );
            }
            Err(_) => {
                error!(
                    "Inference stream chunk timeout: request_id={}, execution_id={}, attempt_id={}, timeout_ms={}",
                    attempt.request_id,
                    attempt.execution_id,
                    attempt.attempt_id,
                    chunk_timeout.as_millis()
                );
                return finish_stream_processing(
                    &mut usage_accumulator,
                    StreamUsageFinalization::Timeout,
                    AttemptTerminal::Failed {
                        attempt,
                        failure: AttemptFailure::new(
                            AttemptFailureKind::StreamTimeout,
                            AttemptStage::Stream,
                            ReplaySafety::NotProvenPreAcceptance,
                        ),
                    },
                );
            }
        }
    }
}

async fn publish_stream_usage(
    usage: Option<CompletionUsage>,
    finalization: StreamUsageFinalization,
    state: &Arc<AppState>,
    user: &User,
    settlement: UsageSettlement,
    tx_consumer: &mpsc::Sender<CompletionChunk>,
) {
    let Some(usage) = usage else {
        return;
    };

    if !finalization.is_provider_done() {
        warn!(
            "Finalizing streaming usage from terminal fallback: trigger={:?}, provider={}, model={}",
            finalization, settlement.provider_name, settlement.public_model_id
        );
    }

    settle_completion_usage(state, user, settlement, usage.clone()).await;

    // Billing is independent from the consumer. If it has gone away after a
    // terminal provider signal, the usage event must still be emitted once.
    let _ = tx_consumer.send(CompletionChunk::Usage(usage)).await;
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(proxy_openai).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<Value>,
            )),
        )
        .route(
            "/v1/models/catalog",
            get(proxy_model_catalog).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<()>,
            )),
        )
        .route(
            "/v1/audio/speech",
            post(proxy_tts).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<TTSRequest>,
            )),
        )
        .route(
            "/v1/audio/transcriptions",
            post(proxy_transcription).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<TranscriptionRequest>,
            )),
        )
        .route(
            "/v1/embeddings",
            post(proxy_embeddings).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<EmbeddingRequest>,
            )),
        )
        .with_state(app_state)
}

pub fn models_router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/v1/models",
            get(proxy_models).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<()>,
            )),
        )
        .with_state(app_state)
}

async fn proxy_openai(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(auth_method): axum::Extension<AuthMethod>,
    axum::Extension(mut body): axum::Extension<Value>,
) -> Result<Response, ApiError> {
    let billing_access = state
        .chat_billing_access(user.uuid, auth_method == AuthMethod::ApiKey)
        .await;
    let model_plan = ModelPlan::from_is_paid(
        billing_access.is_some_and(crate::billing::ChatBillingAccess::is_paid),
    );

    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() && !model_plan.is_paid() {
        error!(
            "Guest user without a paid plan attempted to use chat: {}",
            user.uuid
        );
        return Err(ApiError::Unauthorized);
    }

    if billing_access.is_some_and(|access| !access.can_use()) {
        error!("Usage limit reached for user: {}", user.uuid);
        return Err(ApiError::UsageLimitReached);
    }

    // Extract the model from the request
    let requested_model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?
        .to_string();

    let alias_targets = if model_alias_requires_flag_lookup(&requested_model_name) {
        state.model_alias_targets(user.uuid, model_plan).await
    } else {
        ModelAliasTargets::for_plan(model_plan)
    };
    let model_name = alias_targets.resolve(&requested_model_name).to_string();
    let intent = InferenceIntent::new(
        user.uuid,
        requested_model_name.clone(),
        model_name.clone(),
        model_plan,
        InferenceSurface::ChatCompletions,
        WorkloadClass::Interactive,
    );
    let pinned_completion = prepare_completion_request(&state, &user, intent).await?;
    let served_model_name = pinned_completion.public_model_id().to_string();
    if requested_model_name != served_model_name {
        debug!(
            "Resolved chat model {} to preferred {} and pinned {}",
            requested_model_name, model_name, served_model_name
        );
    }
    pin_chat_request_model(&mut body, &pinned_completion);

    let logical_ticket: LogicalAdmissionTicket = state
        .inference_admission
        .acquire_logical(user.uuid, WorkloadClass::Interactive)
        .await
        .map_err(admission_api_error)?;

    // Create billing context
    let billing_context = BillingContext::new(auth_method, requested_model_name);

    // Get the completion stream - billing happens automatically inside!
    let completion = get_chat_completion_response(
        &state,
        &user,
        body,
        &headers,
        billing_context,
        &pinned_completion,
    )
    .await
    .map_err(CompletionExecutionError::into_pre_persistence_api_error)?;

    debug!(
        "Received completion stream: request_id={}, execution_id={}, attempt_id={}, provider={}, streaming={}",
        completion.metadata.attempt.request_id,
        completion.metadata.attempt.execution_id,
        completion.metadata.attempt.attempt_id,
        completion.metadata.provider_name,
        completion.metadata.is_streaming
    );

    // Handle non-streaming vs streaming responses differently
    if !completion.metadata.is_streaming {
        // For non-streaming responses, get the full response chunk
        debug!("Handling non-streaming response");
        let mut rx = completion.stream;

        // Get the FullResponse chunk
        if let Some(CompletionChunk::FullResponse(response_json)) = rx.recv().await {
            // Billing already happened in get_chat_completion_response!
            // Just encrypt and return
            let encrypted_response = encrypt_response(&state, &session_id, &response_json).await?;
            drop(logical_ticket);
            return Ok(encrypted_response.into_response());
        } else {
            error!("Expected FullResponse chunk but got something else");
            return Err(ApiError::InternalServerError);
        }
    }

    // For streaming responses, process CompletionChunk stream
    debug!("Handling streaming response");
    let mut rx = completion.stream;

    let stream = async_stream::stream! {
        let _logical_ticket = logical_ticket;
        while let Some(chunk) = rx.recv().await {
            match chunk {
                CompletionChunk::StreamChunk(json) => {
                    // Pass through full JSON (includes all metadata from upstream)
                    match encrypt_sse_event(&state, &session_id, &json).await {
                        Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                        Err(e) => {
                            error!("Failed to encrypt event data: {:?}", e);
                            break;
                        }
                    }
                }
                CompletionChunk::Usage(_usage) => {
                    // Billing already handled internally, no need to send to client
                    trace!("Received usage chunk (billing already processed)");
                }
                CompletionChunk::Terminal(terminal) => {
                    match terminal {
                        AttemptTerminal::Completed { .. } => {
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        AttemptTerminal::Failed { failure, .. } => {
                            error!(
                                "Completion attempt failed: kind={:?}, stage={:?}",
                                failure.kind, failure.stage
                            );
                            let error_payload = completion_error_payload(failure.client_message());
                            match encrypt_sse_event(&state, &session_id, &error_payload).await {
                                Ok(event) => yield Ok(event),
                                Err(error) => {
                                    error!("Failed to encrypt terminal error event: {error:?}");
                                }
                            }
                        }
                    }
                    break;
                }
                CompletionChunk::FullResponse(_) => {
                    // Shouldn't happen in streaming mode
                    error!("Received FullResponse in streaming mode");
                    let error_payload = completion_error_payload("Invalid inference response");
                    match encrypt_sse_event(&state, &session_id, &error_payload).await {
                        Ok(event) => yield Ok(event),
                        Err(error) => {
                            error!("Failed to encrypt invalid-format error event: {error:?}");
                        }
                    }
                    break;
                }
            }
        }
    };
    Ok(Sse::new(stream).into_response())
}

fn retain_active_route_after_shadow_observation(
    intent: &InferenceIntent,
    active: Result<SelectedProviderRoute, ProviderRoutingError>,
    shadow: Result<RoutePlan, RoutePlanningError>,
) -> Result<SelectedProviderRoute, ProviderRoutingError> {
    let comparison = compare_shadow_route(&active, &shadow);
    let policy_version = shadow
        .as_ref()
        .map(|plan| plan.policy_version)
        .unwrap_or(SHADOW_ROUTING_POLICY_VERSION);
    let candidate_scope = shadow.as_ref().ok().map(|plan| plan.candidate_scope);

    match &comparison {
        ShadowRouteComparison::Match { .. } => debug!(
            "Shadow route matched active route: request_id={}, public_model={}, policy_version={}, candidate_scope={:?}, comparison={:?}",
            intent.request_id,
            intent.public_model_id,
            policy_version,
            candidate_scope,
            comparison
        ),
        ShadowRouteComparison::Mismatch { .. }
            if matches!(
                active,
                Err(ProviderRoutingError::CapacityUnavailable { .. })
            ) =>
        {
            debug!(
                "Health-aware routing found every configured policy candidate open: request_id={}, preferred_public_model={}, policy_version={}, candidate_scope={:?}",
                intent.request_id,
                intent.public_model_id,
                policy_version,
                candidate_scope
            )
        }
        ShadowRouteComparison::Mismatch { .. }
            if active.as_ref().is_ok_and(|route| {
                route.model_selection_source
                    == crate::provider_routing::ModelSelectionSource::AutoFallback
            }) =>
        {
            debug!(
                "Auto policy selected a compatible fallback model: request_id={}, preferred_public_model={}, selected_public_model={}, policy_version={}, comparison={:?}",
                intent.request_id,
                intent.public_model_id,
                active.as_ref().map(|route| route.public_model_id.as_str()).unwrap_or("unavailable"),
                crate::inference_planning::AUTO_MODEL_ROUTING_POLICY_VERSION,
                comparison
            )
        }
        ShadowRouteComparison::Mismatch { .. } if active.is_ok() => debug!(
            "Health-aware route differed from the baseline plan; retaining the active route: request_id={}, preferred_public_model={}, policy_version={}, candidate_scope={:?}, comparison={:?}",
            intent.request_id,
            intent.public_model_id,
            policy_version,
            candidate_scope,
            comparison
        ),
        ShadowRouteComparison::Mismatch { .. } => warn!(
            "Shadow route differed from active route; retaining active route: request_id={}, public_model={}, policy_version={}, candidate_scope={:?}, comparison={:?}",
            intent.request_id,
            intent.public_model_id,
            policy_version,
            candidate_scope,
            comparison
        ),
    }

    active
}

fn provider_routing_api_error(error: ProviderRoutingError) -> ApiError {
    match error {
        ProviderRoutingError::UnsupportedModel(model) => {
            error!("Unsupported completion model requested: {}", model);
            ApiError::BadRequest
        }
        ProviderRoutingError::NoEligibleRoute(model) => {
            error!("No eligible provider route for completion model: {}", model);
            ApiError::InternalServerError
        }
        ProviderRoutingError::CapacityUnavailable { model, retry_after } => {
            debug!(
                "All configured policy-permitted routes are temporarily unavailable: preferred_public_model={}, retry_after_seconds={}",
                model,
                retry_after.as_secs()
            );
            ApiError::InferenceCapacity {
                status: StatusCode::SERVICE_UNAVAILABLE,
                retry_after: Some(retry_after),
                client_replay_safe: true,
            }
        }
    }
}

fn select_prepared_completion_route(
    provider_router: &ProviderRouter,
    proxy_router: &ProxyRouter,
    intent: &InferenceIntent,
    provider_preference: Option<crate::inference_planning::ProviderPreference>,
    baseline_plan: Result<RoutePlan, RoutePlanningError>,
) -> Result<SelectedProviderRoute, ApiError> {
    let active =
        provider_router.select_active_completion_route(proxy_router, intent, provider_preference);
    retain_active_route_after_shadow_observation(intent, active, baseline_plan)
        .map_err(provider_routing_api_error)
}

pub(crate) async fn prepare_completion_request(
    state: &Arc<AppState>,
    user: &User,
    intent: InferenceIntent,
) -> Result<PinnedCompletionRequest, ApiError> {
    if intent.account_uuid != user.uuid {
        error!("Inference intent account did not match the authenticated user");
        return Err(ApiError::InternalServerError);
    }

    ensure_completion_model_access(&intent.public_model_id, intent.model_plan)?;
    let provider_preference = state
        .provider_routing_preference(user.uuid, &intent.public_model_id)
        .await;
    let shadow = state.provider_router.shadow_completion_plan(
        &state.proxy_router,
        &intent,
        provider_preference,
    );
    if let Ok(plan) = &shadow {
        let candidate_health = plan
            .eligible_routes
            .iter()
            .map(|candidate| {
                let route_key = crate::inference::RouteKey {
                    provider: candidate.provider,
                    provider_model_id: candidate.provider_model_id.clone(),
                };
                (
                    candidate.provider.as_str(),
                    candidate.provider_model_id.as_str(),
                    state.provider_router.shadow_health_snapshot(&route_key),
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "Baseline route candidate health before active selection: request_id={}, public_model={}, routing_policy_version={}, candidate_health={:?}",
            intent.request_id,
            intent.public_model_id,
            plan.policy_version,
            candidate_health
        );
    }
    let route = select_prepared_completion_route(
        &state.provider_router,
        &state.proxy_router,
        &intent,
        provider_preference,
        shadow,
    )?;
    ensure_completion_model_access(&route.public_model_id, intent.model_plan)?;

    debug!(
        "Pinned inference route: request_id={}, selection_mode={:?}, model_selection={:?}, auto={}, surface={:?}, workload={:?}, requested_model={}, preferred_public_model={}, public_model={}, provider={}, provider_model={}, bucket={:?}, source={:?}",
        intent.request_id,
        intent.selection_mode,
        route.model_selection_source,
        intent.selection_mode.is_auto(),
        intent.surface,
        intent.workload_class,
        intent.requested_model_id,
        intent.public_model_id,
        route.public_model_id,
        route.provider.as_str(),
        route.provider_model_id,
        route.bucket,
        route.selection_source
    );

    Ok(PinnedCompletionRequest {
        intent,
        route,
        provider_preference,
        finalized_route: Arc::new(OnceLock::new()),
    })
}

#[derive(Debug)]
struct AdmittedProviderTurn {
    route: SelectedProviderRoute,
    permit: RouteTurnPermit,
    probe: Option<ProbeLease>,
}

fn admission_api_error(rejection: AdmissionRejection) -> ApiError {
    let status = if rejection.status_hint() == 429 {
        StatusCode::TOO_MANY_REQUESTS
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    ApiError::InferenceCapacity {
        status,
        retry_after: Some(rejection.retry_after),
        client_replay_safe: false,
    }
}

fn probe_api_error(retry_after: Duration) -> ApiError {
    ApiError::InferenceCapacity {
        status: StatusCode::SERVICE_UNAVAILABLE,
        retry_after: Some(retry_after),
        client_replay_safe: false,
    }
}

fn route_image_token_reservation(route: &SelectedProviderRoute) -> u64 {
    PROVIDER_REGISTRY
        .completion_route(route.provider, &route.provider_model_id)
        .map_or(UNKNOWN_ROUTE_IMAGE_TOKEN_RESERVATION, |route| {
            route.image_token_reservation
        })
}

/// Ensures every accepted request has a provider-visible output bound that
/// matches admission's reservation. This bounds provider extensions such as
/// `ignore_eos` and token allowlists without trying to enumerate every option
/// that can suppress ordinary generation termination.
fn ensure_bounded_completion_generation(
    body: &mut serde_json::Map<String, Value>,
    completion_default_reservation: u64,
) {
    for field in ["max_completion_tokens", "max_tokens"] {
        if body.get(field).is_some_and(Value::is_null) {
            body.remove(field);
        }
    }

    let has_explicit_maximum = ["max_completion_tokens", "max_tokens"]
        .into_iter()
        .any(|field| body.contains_key(field));
    if !has_explicit_maximum {
        let requested_minimum = body.get("min_tokens").and_then(Value::as_u64).unwrap_or(0);
        body.insert(
            "max_tokens".to_string(),
            json!(completion_default_reservation.max(requested_minimum)),
        );
    }
}

fn validate_completion_request_controls(
    body: &serde_json::Map<String, Value>,
    max_completion_choices: u64,
) -> Result<(), ApiError> {
    for field in ["max_completion_tokens", "max_tokens"] {
        let Some(value) = body.get(field) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        if value.as_u64().is_none_or(|tokens| tokens == 0) {
            warn!(field, "Rejected non-canonical completion maximum");
            return Err(ApiError::BadRequest);
        }
    }

    if let Some(value) = body.get("n") {
        if !value.is_null() {
            let Some(count) = value.as_u64().filter(|count| *count > 0) else {
                warn!("Rejected non-canonical completion choice count");
                return Err(ApiError::BadRequest);
            };
            if count > max_completion_choices {
                warn!(
                    count,
                    max_completion_choices, "Rejected excessive completion choice count"
                );
                return Err(ApiError::BadRequest);
            }
        }
    }
    if let Some(value) = body.get("min_tokens") {
        if value.as_u64().is_none() {
            warn!("Rejected non-canonical minimum completion tokens");
            return Err(ApiError::BadRequest);
        }
    }
    if let Some(value) = body.get("stream") {
        if !value.is_null() && !value.is_boolean() {
            warn!("Rejected non-canonical completion stream flag");
            return Err(ApiError::BadRequest);
        }
    }

    validate_completion_names(body)?;
    validate_completion_tool_arguments(body)?;

    Ok(())
}

fn validate_completion_name(value: &Value, field: &'static str) -> Result<(), ApiError> {
    let Some(name) = value.as_str() else {
        warn!(field, "Rejected non-string completion name");
        return Err(ApiError::BadRequest);
    };
    if name.is_empty()
        || name.len() > MAX_COMPLETION_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        warn!(field, "Rejected non-canonical completion name");
        return Err(ApiError::BadRequest);
    }
    Ok(())
}

fn validate_nested_completion_name(
    value: &Value,
    container: &str,
    field: &'static str,
) -> Result<(), ApiError> {
    if let Some(name) = value.get(container).and_then(|nested| nested.get("name")) {
        validate_completion_name(name, field)?;
    }
    Ok(())
}

fn validate_completion_names(body: &serde_json::Map<String, Value>) -> Result<(), ApiError> {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            if let Some(name) = message.get("name") {
                validate_completion_name(name, "messages[].name")?;
            }
            validate_nested_completion_name(
                message,
                "function_call",
                "messages[].function_call.name",
            )?;
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    validate_nested_completion_name(
                        tool_call,
                        "function",
                        "messages[].tool_calls[].function.name",
                    )?;
                }
            }
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            validate_nested_completion_name(tool, "function", "tools[].function.name")?;
            validate_nested_completion_name(tool, "custom", "tools[].custom.name")?;
        }
    }
    if let Some(tool_choice) = body.get("tool_choice") {
        validate_nested_completion_name(tool_choice, "function", "tool_choice.function.name")?;
        validate_nested_completion_name(tool_choice, "custom", "tool_choice.custom.name")?;
    }

    Ok(())
}

fn validate_completion_tool_arguments(
    body: &serde_json::Map<String, Value>,
) -> Result<(), ApiError> {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(());
    };
    for message in messages {
        let calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(message.get("function_call"));
        for call in calls {
            let Some(arguments) = call.get("function").unwrap_or(call).get("arguments") else {
                continue;
            };
            let Some(arguments) = arguments.as_str() else {
                warn!("Rejected non-string historical tool arguments");
                return Err(ApiError::BadRequest);
            };
            if arguments.len() > MAX_COMPLETION_TOOL_ARGUMENT_BYTES
                || serde_json::from_str::<Value>(arguments).is_err()
            {
                warn!("Rejected invalid or oversized historical tool arguments");
                return Err(ApiError::BadRequest);
            }
        }
    }
    Ok(())
}

fn is_unsupported_completion_media_part(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(
                kind,
                "video_url" | "input_video" | "image_pil" | "image_embeds" | "prompt_embeds"
            )
        })
        || ["video_url", "image_pil", "image_embeds", "prompt_embeds"]
            .into_iter()
            .any(|field| object.contains_key(field))
}

fn completion_contains_unsupported_media(body: &serde_json::Map<String, Value>) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                let Some(content) = message.get("content") else {
                    return false;
                };
                content.as_array().map_or_else(
                    || is_unsupported_completion_media_part(content),
                    |parts| parts.iter().any(is_unsupported_completion_media_part),
                )
            })
        })
}

fn strip_provider_internal_request_fields(body: &mut serde_json::Map<String, Value>) {
    if body
        .remove(PROVIDER_MANAGED_KV_TRANSFER_PARAMS_FIELD)
        .is_some()
    {
        debug!("Stripped provider-internal KV transfer parameters");
    }
}

fn estimate_completion_admission(
    body: &serde_json::Map<String, Value>,
    route: &SelectedProviderRoute,
    completion_default_reservation: u64,
) -> AdmissionEstimate {
    let image_token_reservation = route_image_token_reservation(route);
    let prompt_tokens = estimate_json_prompt_tokens(body, image_token_reservation);
    let completion_count = body
        .get("n")
        .and_then(Value::as_u64)
        .filter(|count| *count > 0)
        .unwrap_or(1);
    let requested_maximum = ["max_completion_tokens", "max_tokens"]
        .into_iter()
        .find_map(|field| body.get(field).and_then(Value::as_u64))
        .filter(|tokens| *tokens > 0)
        .unwrap_or(completion_default_reservation);
    let requested_minimum = body.get("min_tokens").and_then(Value::as_u64).unwrap_or(0);
    let completion_tokens = requested_maximum
        .max(requested_minimum)
        .saturating_mul(completion_count);
    AdmissionEstimate::new(prompt_tokens, Some(completion_tokens))
}

fn is_completion_image_content_part(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let explicit_image = object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "image_url" | "input_image"));
    let shorthand_image = object.contains_key("image_url");
    explicit_image || shorthand_image
}

fn sanitize_completion_message_content(value: &Value, image_count: &mut u64) -> Value {
    let sanitize_part = |part: &Value, image_count: &mut u64| {
        if !is_completion_image_content_part(part) {
            return part.clone();
        }

        *image_count = image_count.saturating_add(1);
        let mut sanitized = part.clone();
        if let Some(object) = sanitized.as_object_mut() {
            if object.contains_key("image_url") {
                object.insert("image_url".to_string(), json!("[image]"));
            }
        }
        sanitized
    };

    match value {
        Value::Array(parts) => Value::Array(
            parts
                .iter()
                .map(|part| sanitize_part(part, image_count))
                .collect(),
        ),
        part => sanitize_part(part, image_count),
    }
}

/// Returns messages with only actual content-part images replaced by a small
/// sentinel, plus the number of images removed. Restricting recognition to
/// `messages[*].content[*]` ensures a tool or response JSON schema containing
/// keys such as `type: image_url` is still counted in full.
fn sanitize_completion_messages(value: &Value, image_count: &mut u64) -> Value {
    let Value::Array(messages) = value else {
        return value.clone();
    };
    Value::Array(
        messages
            .iter()
            .map(|message| {
                let Some(message) = message.as_object() else {
                    return message.clone();
                };
                let mut sanitized = message.clone();
                if let Some(content) = sanitized.get_mut("content") {
                    *content = sanitize_completion_message_content(content, image_count);
                }
                Value::Object(sanitized)
            })
            .collect(),
    )
}

fn count_json_object_entries(value: &Value) -> u64 {
    match value {
        Value::Array(values) => values.iter().fold(0_u64, |total, value| {
            total.saturating_add(count_json_object_entries(value))
        }),
        Value::Object(object) => object.values().fold(
            u64::try_from(object.len()).unwrap_or(u64::MAX),
            |total, value| total.saturating_add(count_json_object_entries(value)),
        ),
        _ => 0,
    }
}

fn completion_tool_argument_shape(body: &serde_json::Map<String, Value>) -> (u64, u64, u64) {
    let mut tool_calls = 0_u64;
    let mut argument_entries = 0_u64;
    let mut normalized_argument_bytes = 0_u64;
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return (tool_calls, argument_entries, normalized_argument_bytes);
    };

    for message in messages {
        let calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(message.get("function_call"));
        for call in calls {
            tool_calls = tool_calls.saturating_add(1);
            let Some(arguments) = call.get("function").unwrap_or(call).get("arguments") else {
                continue;
            };
            if let Some(arguments) = arguments.as_str() {
                argument_entries = argument_entries.saturating_add(
                    u64::try_from(arguments.bytes().filter(|byte| *byte == b':').count())
                        .unwrap_or(u64::MAX),
                );
                if let Ok(arguments) = serde_json::from_str::<Value>(arguments) {
                    normalized_argument_bytes = normalized_argument_bytes.saturating_add(
                        serde_json::to_string(&arguments)
                            .map(|arguments| u64::try_from(arguments.len()).unwrap_or(u64::MAX))
                            .unwrap_or(u64::MAX),
                    );
                }
            } else {
                argument_entries =
                    argument_entries.saturating_add(count_json_object_entries(arguments));
            }
        }
    }
    (tool_calls, argument_entries, normalized_argument_bytes)
}

fn count_prompt_line_breaks(value: &Value) -> u64 {
    match value {
        Value::String(text) => text.chars().fold(0_u64, |total, ch| {
            total.saturating_add(u64::from(matches!(
                ch,
                '\n' | '\r' | '\u{000B}' | '\u{000C}' | '\u{0085}' | '\u{2028}' | '\u{2029}'
            )))
        }),
        Value::Array(values) => values.iter().fold(0_u64, |total, value| {
            total.saturating_add(count_prompt_line_breaks(value))
        }),
        Value::Object(object) => object.values().fold(0_u64, |total, value| {
            total.saturating_add(count_prompt_line_breaks(value))
        }),
        _ => 0,
    }
}

fn estimate_json_prompt_tokens(
    body: &serde_json::Map<String, Value>,
    image_token_reservation: u64,
) -> u64 {
    let mut image_count = 0_u64;
    let mut sanitized_body = body.clone();
    if let Some(messages) = sanitized_body.get_mut("messages") {
        *messages = sanitize_completion_messages(messages, &mut image_count);
    }
    let sanitized_body = Value::Object(sanitized_body);
    let prompt_line_break_count = count_prompt_line_breaks(&sanitized_body);
    let serialized = serde_json::to_string(&sanitized_body)
        .expect("serde_json::Value serialization is infallible");
    let message_count = body
        .get("messages")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let tool_count = body
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let attribute_escape_expansion = serialized.bytes().fold(0_u64, |total, byte| {
        total.saturating_add(match byte {
            b'&' => 4, // `&` -> `&amp;`
            b'"' => 5, // `"` -> `&quot;` after JSON decoding
            _ => 0,
        })
    });
    let normalized_json_separator_expansion = u64::try_from(
        serialized
            .bytes()
            .filter(|byte| matches!(byte, b',' | b':'))
            .count(),
    )
    .unwrap_or(u64::MAX);
    let normalized_json_unicode_expansion = serialized.chars().fold(0_u64, |total, ch| {
        if ch.is_ascii() {
            return total;
        }
        let escaped_bytes = if ch.len_utf8() == 4 { 12 } else { 6 };
        total.saturating_add(u64::try_from(escaped_bytes - ch.len_utf8()).unwrap_or(u64::MAX))
    });
    let (tool_call_count, argument_entry_count, normalized_argument_bytes) =
        completion_tool_argument_shape(body);

    // Provider tokenizers are not interchangeable: K3 can split ordinary
    // ASCII nearly eleven times more finely than cl100k. UTF-8 bytes are a
    // tokenizer-independent upper bound for the byte-level provider
    // tokenizers on these routes; explicit structural allowances cover prompt
    // markers and template text that are not present verbatim in the JSON.
    u64::try_from(serialized.len())
        .unwrap_or(u64::MAX)
        .saturating_add(attribute_escape_expansion)
        .saturating_add(normalized_json_separator_expansion)
        .saturating_add(normalized_json_unicode_expansion)
        .saturating_add(normalized_argument_bytes)
        .saturating_add(COMPLETION_PROMPT_BASE_TOKEN_OVERHEAD)
        .saturating_add(
            u64::try_from(message_count)
                .unwrap_or(u64::MAX)
                .saturating_mul(COMPLETION_PROMPT_MESSAGE_TOKEN_OVERHEAD),
        )
        .saturating_add(
            u64::try_from(tool_count)
                .unwrap_or(u64::MAX)
                .saturating_mul(COMPLETION_PROMPT_TOOL_TOKEN_OVERHEAD),
        )
        .saturating_add(tool_call_count.saturating_mul(COMPLETION_PROMPT_TOOL_CALL_TOKEN_OVERHEAD))
        .saturating_add(
            argument_entry_count.saturating_mul(COMPLETION_PROMPT_ARGUMENT_ENTRY_TOKEN_OVERHEAD),
        )
        .saturating_add(
            prompt_line_break_count.saturating_mul(COMPLETION_PROMPT_LINE_PREFIX_TOKEN_OVERHEAD),
        )
        .saturating_add(image_count.saturating_mul(image_token_reservation))
}

fn replan_completion_route_excluding(
    state: &AppState,
    pinned: &PinnedCompletionRequest,
    excluded_routes: &mut HashSet<crate::inference::RouteKey>,
) -> Result<SelectedProviderRoute, ApiError> {
    loop {
        let route = state
            .provider_router
            .select_active_completion_route_excluding(
                &state.proxy_router,
                &pinned.intent,
                pinned.provider_preference,
                excluded_routes,
            )
            .map_err(provider_routing_api_error)?;

        // Model substitution is finalized during request preparation, before
        // Responses builds model-specific context and defaults. Admission may
        // still move a request to another provider for that public model, but
        // it must never change the public model behind an already-built body.
        if route.public_model_id == pinned.route.public_model_id {
            return Ok(route);
        }
        excluded_routes.insert(route.identity().route_key());
    }
}

/// Acquires one fair provider-turn reservation, then atomically claims any
/// expired health/capacity gates. A first turn may replan to another provider
/// for the prepared public model only before its first upstream send. Once
/// finalized, Responses tool turns reacquire admission on the identical route
/// and never consult health or switch provider/model.
async fn admit_completion_turn(
    state: &Arc<AppState>,
    pinned: &PinnedCompletionRequest,
    body: &serde_json::Map<String, Value>,
    allow_replan: bool,
) -> Result<AdmittedProviderTurn, ApiError> {
    let completion_default_reservation = state
        .inference_admission
        .policy()
        .completion_default_reservation();
    if let Some(route) = pinned.finalized_route.get() {
        let estimate = estimate_completion_admission(body, route, completion_default_reservation);
        let permit = state
            .inference_admission
            .acquire_turn(
                &route.identity().route_key(),
                pinned.intent.account_uuid,
                pinned.intent.workload_class,
                estimate,
                None,
            )
            .await
            .map_err(admission_api_error)?;
        return Ok(AdmittedProviderTurn {
            route: route.clone(),
            permit,
            probe: None,
        });
    }

    let admission_deadline = Instant::now()
        .checked_add(match pinned.intent.workload_class {
            WorkloadClass::Interactive => state.inference_admission.policy().interactive_wait(),
            WorkloadClass::Background => state.inference_admission.policy().background_wait(),
        })
        .unwrap_or_else(Instant::now);
    let mut excluded_routes = HashSet::new();
    let mut route = pinned.route.clone();

    loop {
        let route_key = route.identity().route_key();
        let estimate = estimate_completion_admission(body, &route, completion_default_reservation);
        let permit = match state
            .inference_admission
            .acquire_turn(
                &route_key,
                pinned.intent.account_uuid,
                pinned.intent.workload_class,
                estimate,
                Some(admission_deadline),
            )
            .await
        {
            Ok(permit) => permit,
            Err(rejection) => {
                debug!(
                    "Inference route lost local admission before send: request_id={}, provider={}, provider_model={}, kind={:?}",
                    pinned.intent.request_id,
                    route.provider.as_str(),
                    route.provider_model_id,
                    rejection.kind
                );
                let local_error = admission_api_error(rejection);
                if !allow_replan {
                    return Err(local_error);
                }
                excluded_routes.insert(route_key);
                match replan_completion_route_excluding(state, pinned, &mut excluded_routes) {
                    Ok(replanned) => {
                        route = replanned;
                        continue;
                    }
                    Err(_) => return Err(local_error),
                }
            }
        };

        let probe = match state.provider_router.try_claim_probe(&route_key) {
            ProbeClaimResult::Ready(probe) => probe,
            ProbeClaimResult::Rejected {
                reason,
                retry_after,
            } => {
                debug!(
                    "Inference route lost half-open claim before send: request_id={}, provider={}, provider_model={}, reason={:?}",
                    pinned.intent.request_id,
                    route.provider.as_str(),
                    route.provider_model_id,
                    reason
                );
                permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
                let local_error = probe_api_error(retry_after);
                if !allow_replan {
                    return Err(local_error);
                }
                excluded_routes.insert(route_key);
                match replan_completion_route_excluding(state, pinned, &mut excluded_routes) {
                    Ok(replanned) => {
                        route = replanned;
                        continue;
                    }
                    Err(_) => return Err(local_error),
                }
            }
        };

        match pinned.finalize_route(route.clone()) {
            true => {
                return Ok(AdmittedProviderTurn {
                    route,
                    permit,
                    probe,
                });
            }
            false => {
                // Pinned requests are not started concurrently today, but fail
                // safely if that invariant changes: refund before sending and
                // reacquire the already-finalized route on the next iteration.
                drop(probe);
                permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
                let finalized = pinned
                    .finalized_route
                    .get()
                    .expect("failed OnceLock set means another route was finalized")
                    .clone();
                let estimate =
                    estimate_completion_admission(body, &finalized, completion_default_reservation);
                let permit = state
                    .inference_admission
                    .acquire_turn(
                        &finalized.identity().route_key(),
                        pinned.intent.account_uuid,
                        pinned.intent.workload_class,
                        estimate,
                        Some(admission_deadline),
                    )
                    .await
                    .map_err(admission_api_error)?;
                return Ok(AdmittedProviderTurn {
                    route: finalized,
                    permit,
                    probe: None,
                });
            }
        }
    }
}

/// Ensures cancellation or panic cannot make an in-flight attempt disappear
/// from the terminal observation stream. Consumer cancellation is deliberately
/// neutral for route health, but it must still be represented exactly once.
struct AttemptObservationGuard {
    attempt: InferenceAttempt,
    provider_router: Arc<ProviderRouter>,
    stage: AttemptStage,
    turn_permit: Option<RouteTurnPermit>,
    probe: Option<ProbeLease>,
    armed: bool,
}

impl AttemptObservationGuard {
    #[cfg(test)]
    fn new(
        attempt: InferenceAttempt,
        provider_router: Arc<ProviderRouter>,
        stage: AttemptStage,
    ) -> Self {
        Self {
            attempt,
            provider_router,
            stage,
            turn_permit: None,
            probe: None,
            armed: true,
        }
    }

    fn new_admitted(
        attempt: InferenceAttempt,
        provider_router: Arc<ProviderRouter>,
        stage: AttemptStage,
        turn_permit: RouteTurnPermit,
        probe: Option<ProbeLease>,
    ) -> Self {
        Self {
            attempt,
            provider_router,
            stage,
            turn_permit: Some(turn_permit),
            probe,
            armed: true,
        }
    }

    fn attempt(&self) -> &InferenceAttempt {
        &self.attempt
    }

    fn set_stage(&mut self, stage: AttemptStage) {
        self.stage = stage;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn record_terminal(&mut self, terminal: &AttemptTerminal) {
        self.record_terminal_with_usage(terminal, None);
    }

    fn record_terminal_with_usage(
        &mut self,
        terminal: &AttemptTerminal,
        usage: Option<&CompletionUsage>,
    ) {
        debug_assert_eq!(terminal.attempt(), &self.attempt);
        let (actual, disposition) = match terminal {
            AttemptTerminal::Completed { .. } => (
                usage.map(completion_actual_usage),
                TerminalDisposition::Completed,
            ),
            AttemptTerminal::Failed { failure, .. }
                if failure.replay_safety == ReplaySafety::ProvenPreAcceptance =>
            {
                (None, TerminalDisposition::ProvenPreAcceptance)
            }
            AttemptTerminal::Failed { .. } => (None, TerminalDisposition::Ambiguous),
        };
        record_attempt_terminal_with_probe(
            &self.provider_router,
            terminal,
            ShadowObservationMode::Update,
            self.probe.take(),
        );
        // Commit health/probe state before admission can wake another waiter.
        if let Some(permit) = self.turn_permit.take() {
            permit.settle(actual, disposition);
        }
        self.disarm();
    }

    fn cancellation_terminal(&self) -> AttemptTerminal {
        AttemptTerminal::Failed {
            attempt: self.attempt.clone(),
            failure: AttemptFailure::new(
                AttemptFailureKind::ConsumerDropped,
                self.stage,
                ReplaySafety::NotProvenPreAcceptance,
            ),
        }
    }
}

fn completion_actual_usage(usage: &CompletionUsage) -> ActualUsage {
    ActualUsage {
        prompt_tokens: usage
            .prompt_tokens_observed
            .then(|| u64::try_from(usage.prompt_tokens).ok())
            .flatten(),
        completion_tokens: usage
            .completion_tokens_observed
            .then(|| u64::try_from(usage.completion_tokens).ok())
            .flatten(),
        cached_prompt_tokens: usage
            .cached_prompt_tokens
            .and_then(|tokens| u64::try_from(tokens).ok()),
    }
}

impl Drop for AttemptObservationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        let terminal = self.cancellation_terminal();
        record_attempt_terminal_with_probe(
            &self.provider_router,
            &terminal,
            ShadowObservationMode::Update,
            self.probe.take(),
        );
        // Commit the neutral terminal before admission can wake a waiter.
        if let Some(permit) = self.turn_permit.take() {
            permit.settle(None, TerminalDisposition::Ambiguous);
        }
        warn!(
            "Inference attempt abandoned before terminal processing: request_id={}, execution_id={}, attempt_id={}, stage={:?}",
            self.attempt.request_id,
            self.attempt.execution_id,
            self.attempt.attempt_id,
            self.stage
        );
    }
}

async fn await_attempt_result<T>(
    terminal_guard: AttemptObservationGuard,
    future: impl std::future::Future<Output = T>,
) -> (T, AttemptObservationGuard) {
    let result = future.await;
    (result, terminal_guard)
}

/// A provider response whose HTTP success status has been observed, but whose
/// body has not yet been consumed. Responses holds this value only across its
/// persistence boundary so a capacity rejection can still be returned as an
/// ordinary HTTP error without committing conversation state.
pub(crate) struct StartedCompletion {
    response: Option<ProviderResponse>,
    successful_provider: String,
    attempt: InferenceAttempt,
    terminal_guard: Option<AttemptObservationGuard>,
    response_model_id: String,
    public_model_id: String,
    is_streaming: bool,
    non_streaming_body_limit: Option<usize>,
    usage_settlement: Option<UsageSettlement>,
}

/// Starts one model turn against a route pinned for the logical inference
/// request and returns immediately after a successful provider response start.
pub(crate) async fn start_chat_completion_response(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
    pinned: &PinnedCompletionRequest,
) -> Result<StartedCompletion, CompletionExecutionError> {
    get_chat_completion_response_with_options(
        state,
        user,
        body,
        headers,
        billing_context,
        pinned,
        CompletionExecutionOptions::default(),
    )
    .await
}

/// Run an entitled completion only if routing resolves to the server-selected
/// provider and provider model. The constraint is checked before any request is
/// serialized or sent; all ordinary plan, cache-field, and billing behavior is
/// otherwise shared with [`get_chat_completion_response`].
pub(crate) async fn get_chat_completion_response_for_expected_route(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
    model_plan: ModelPlan,
    route: ServerSelectedCompletionRoute<'_>,
) -> Result<CompletionStream, CompletionExecutionError> {
    let public_model_id = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or(CompletionExecutionError::Request(ApiError::BadRequest))?
        .to_string();
    let intent = InferenceIntent::new(
        user.uuid,
        public_model_id.clone(),
        public_model_id,
        model_plan,
        InferenceSurface::Internal,
        WorkloadClass::Interactive,
    );
    let pinned = prepare_completion_request(state, user, intent)
        .await
        .map_err(|error| {
            error!(
                "Failed to prepare server-selected completion route: expected_provider={}, expected_provider_model={}, error={error:?}",
                route.provider_name, route.provider_model_id
            );
            CompletionExecutionError::Request(ApiError::ServiceUnavailable)
        })?;

    let continuum_cache_salt = (route.provider_name == ProviderId::Continuum.as_str())
        .then(|| format!("server-selected-{}", Uuid::new_v4().simple()));
    let started = get_chat_completion_response_with_options(
        state,
        user,
        body,
        headers,
        billing_context,
        &pinned,
        CompletionExecutionOptions {
            exact_route: Some(ExactCompletionRoute {
                provider_name: route.provider_name.to_string(),
                provider_model_id: route.provider_model_id.to_string(),
            }),
            continuum_cache_salt,
            non_streaming_body_limit: Some(MAX_BOUNDED_PROVIDER_RESPONSE_BYTES),
        },
    )
    .await?;
    finish_started_completion(state, user, started).await
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ServerSelectedCompletionRoute<'a> {
    pub provider_name: &'a str,
    pub provider_model_id: &'a str,
}

struct ExactCompletionRoute {
    provider_name: String,
    provider_model_id: String,
}

fn completion_route_matches_exact_constraint(
    route: &SelectedProviderRoute,
    expected: &ExactCompletionRoute,
) -> bool {
    route.provider.as_str() == expected.provider_name
        && route.proxy.provider_name == expected.provider_name
        && route.provider_model_id == expected.provider_model_id
}

#[derive(Default)]
struct CompletionExecutionOptions {
    exact_route: Option<ExactCompletionRoute>,
    continuum_cache_salt: Option<String>,
    non_streaming_body_limit: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
async fn get_chat_completion_response_with_options(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
    pinned: &PinnedCompletionRequest,
    options: CompletionExecutionOptions,
) -> Result<StartedCompletion, CompletionExecutionError> {
    if body.is_null() || body.as_object().is_none_or(|obj| obj.is_empty()) {
        error!("Request body is empty or invalid");
        return Err(ApiError::BadRequest.into());
    }

    let mut modified_body = body
        .as_object()
        .ok_or_else(|| {
            error!("Request body is not a JSON object");
            ApiError::BadRequest
        })?
        .clone();

    validate_completion_request_controls(
        &modified_body,
        state.inference_admission.policy().max_completion_choices(),
    )
    .map_err(CompletionExecutionError::from)?;
    if completion_contains_unsupported_media(&modified_body) {
        warn!("Rejected unsupported media content in completion request");
        return Err(ApiError::BadRequest.into());
    }
    strip_provider_internal_request_fields(&mut modified_body);

    // Check if streaming is requested (default to false if not specified)
    let is_streaming = modified_body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let body_model_name = modified_body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?
        .to_string();

    if body_model_name != pinned.public_model_id() {
        error!(
            "Prepared inference model did not match execution body: request_id={}, preferred_model={}, prepared_model={}, body_model={}",
            pinned.intent.request_id,
            pinned.intent.public_model_id,
            pinned.public_model_id(),
            body_model_name
        );
        return Err(ApiError::InternalServerError.into());
    }

    ensure_completion_model_access(&pinned.route.public_model_id, pinned.intent.model_plan)?;
    if let Some(expected_route) = &options.exact_route {
        if !completion_route_matches_exact_constraint(&pinned.route, expected_route) {
            error!(
                "Completion route did not match the server-selected constraint: request_id={}, public_model={}, expected_provider={}, expected_provider_model={}, selected_provider={}, selected_proxy_provider={}, selected_provider_model={}",
                pinned.intent.request_id,
                pinned.route.public_model_id,
                expected_route.provider_name,
                expected_route.provider_model_id,
                pinned.route.provider.as_str(),
                pinned.route.proxy.provider_name,
                pinned.route.provider_model_id
            );
            return Err(CompletionExecutionError::Request(
                ApiError::ServiceUnavailable,
            ));
        }
    }
    ensure_bounded_completion_generation(
        &mut modified_body,
        state
            .inference_admission
            .policy()
            .completion_default_reservation(),
    );

    let AdmittedProviderTurn {
        route: selected_route,
        permit,
        probe,
    } = admit_completion_turn(state, pinned, &modified_body, options.exact_route.is_none())
        .await
        .map_err(CompletionExecutionError::from)?;
    if selected_route.public_model_id != body_model_name {
        error!(
            "Admission changed the prepared public model: request_id={}, prepared_model={}, admitted_model={}",
            pinned.intent.request_id,
            body_model_name,
            selected_route.public_model_id
        );
        drop(probe);
        permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
        return Err(CompletionExecutionError::Request(
            ApiError::InternalServerError,
        ));
    }
    if let Err(error) =
        ensure_completion_model_access(&selected_route.public_model_id, pinned.intent.model_plan)
    {
        drop(probe);
        permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
        return Err(CompletionExecutionError::Request(error));
    }
    if let Some(expected_route) = &options.exact_route {
        if !completion_route_matches_exact_constraint(&selected_route, expected_route) {
            error!(
                "Admitted completion route did not match the server-selected constraint: request_id={}, public_model={}, expected_provider={}, expected_provider_model={}, selected_provider={}, selected_proxy_provider={}, selected_provider_model={}",
                pinned.intent.request_id,
                selected_route.public_model_id,
                expected_route.provider_name,
                expected_route.provider_model_id,
                selected_route.provider.as_str(),
                selected_route.proxy.provider_name,
                selected_route.provider_model_id
            );
            drop(probe);
            permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
            return Err(CompletionExecutionError::Request(
                ApiError::ServiceUnavailable,
            ));
        }
    }
    let route_identity = selected_route.identity();
    let execution = pinned.begin_execution();
    modified_body.insert(
        "model".to_string(),
        json!(selected_route.provider_model_id.clone()),
    );
    apply_provider_managed_request_fields(
        &mut modified_body,
        &selected_route.proxy.provider_name,
        user.uuid,
    );
    if options.exact_route.is_some() {
        if let Err(error) = apply_server_selected_cache_isolation(
            &mut modified_body,
            &selected_route.proxy.provider_name,
            options.continuum_cache_salt.as_deref(),
        ) {
            drop(probe);
            permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
            return Err(CompletionExecutionError::Request(error));
        }
    }
    // Prepare one logical model-turn execution. The provider transport may
    // report more than one attempt when Tinfoil safely refreshes a stale
    // attested route after a proven pre-connect failure.
    debug!(
        "Sending inference execution: request_id={}, execution_id={}, public_model={}, provider_model={}, provider={}",
        execution.request_id,
        execution.execution_id,
        selected_route.public_model_id,
        selected_route.provider_model_id,
        selected_route.provider.as_str()
    );

    let (response, successful_provider, attempt, terminal_guard) = {
        let mut request_body = modified_body.clone();
        let proxy_config = selected_route.proxy.clone();
        let provider_model_name = selected_route.provider_model_id.clone();

        ensure_stream_usage(&mut request_body);

        let request_body_value = Value::Object(request_body);
        let attempt = execution.begin_attempt(route_identity.clone());
        let mut terminal_guard = AttemptObservationGuard::new_admitted(
            attempt.clone(),
            Arc::clone(&state.provider_router),
            AttemptStage::BeforeSend,
            permit,
            probe,
        );
        let request_body_json = match serde_json::to_string(&request_body_value) {
            Ok(json) => json,
            Err(error) => {
                let failure = AttemptFailure::new(
                    AttemptFailureKind::RequestBuild,
                    AttemptStage::BeforeSend,
                    ReplaySafety::ProvenPreAcceptance,
                );
                error!(
                    "Failed to serialize inference request: request_id={}, execution_id={}, attempt_id={}, error={:?}",
                    attempt.request_id, attempt.execution_id, attempt.attempt_id, error
                );
                let terminal = AttemptTerminal::Failed { attempt, failure };
                terminal_guard.record_terminal(&terminal);
                return Err(CompletionExecutionError::Attempt {
                    terminal,
                    public_error: ApiError::InternalServerError,
                });
            }
        };
        let request_log_metadata =
            CompletionRequestLogMetadata::from_body(&request_body_value, request_body_json.len());

        debug!(
            "Completion request metadata before provider call: request_id={}, execution_id={}, provider={}, model={}, metadata={:?}",
            execution.request_id,
            execution.execution_id,
            proxy_config.provider_name,
            provider_model_name,
            request_log_metadata
        );

        terminal_guard.set_stage(AttemptStage::AwaitingResponse);
        let (
            ProviderSendTrace {
                prior_failures,
                result,
            },
            mut terminal_guard,
        ) = await_attempt_result(
            terminal_guard,
            try_provider(
                &state.provider_client,
                &proxy_config,
                request_body_json,
                headers,
            ),
        )
        .await;

        let _recovered_terminals = terminalize_recovered_provider_failures(
            &state.provider_router,
            execution,
            &route_identity,
            prior_failures,
        );

        let attempt = terminal_guard.attempt().clone();
        match result {
            Ok(response) => {
                record_attempt_outcome(
                    &state.provider_router,
                    &AttemptOutcome::ResponseStarted {
                        attempt: attempt.clone(),
                        status: response.status_code(),
                    },
                    ShadowObservationMode::Update,
                );
                info!(
                    "Inference response started: request_id={}, execution_id={}, attempt_id={}",
                    attempt.request_id, attempt.execution_id, attempt.attempt_id
                );
                terminal_guard.set_stage(AttemptStage::ResponseBody);
                (
                    response,
                    selected_route.provider.as_str().to_string(),
                    attempt,
                    terminal_guard,
                )
            }
            Err(err) => {
                let failure = attempt_failure_from_provider_error(&err);
                error!(
                    "Completion provider attempt failed: request_id={}, execution_id={}, attempt_id={}, kind={:?}, stage={:?}, metadata={:?}",
                    attempt.request_id,
                    attempt.execution_id,
                    attempt.attempt_id,
                    failure.kind,
                    failure.stage,
                    request_log_metadata
                );
                let public_error = public_completion_error(&err, &failure);
                let terminal = AttemptTerminal::Failed { attempt, failure };
                terminal_guard.record_terminal(&terminal);
                return Err(CompletionExecutionError::Attempt {
                    terminal,
                    public_error,
                });
            }
        }
    };

    debug!(
        "Successfully received response from provider: {}",
        successful_provider
    );

    Ok(StartedCompletion {
        response: Some(response),
        successful_provider,
        usage_settlement: Some(UsageSettlement::new(billing_context, &attempt)),
        attempt,
        terminal_guard: Some(terminal_guard),
        response_model_id: selected_route.response_model_id,
        public_model_id: selected_route.public_model_id,
        is_streaming,
        non_streaming_body_limit: options.non_streaming_body_limit,
    })
}

/// Consumes a successfully started provider response. Billing remains internal
/// until the usage-settlement stack separates the recorder.
pub(crate) async fn finish_started_completion(
    state: &Arc<AppState>,
    user: &User,
    mut started: StartedCompletion,
) -> Result<CompletionStream, CompletionExecutionError> {
    let response = started
        .response
        .take()
        .expect("started completion response can only be consumed once");
    let mut terminal_guard = started
        .terminal_guard
        .take()
        .expect("started completion terminal guard can only be consumed once");
    let successful_provider = started.successful_provider.clone();
    let attempt = started.attempt.clone();
    let response_model_id = started.response_model_id.clone();
    let public_model_id = started.public_model_id.clone();
    let is_streaming = started.is_streaming;
    let non_streaming_body_limit = started.non_streaming_body_limit;
    let usage_settlement = started
        .usage_settlement
        .take()
        .expect("started completion usage settlement can only be consumed once");

    // NOW: Process the response internally and handle billing
    if !is_streaming {
        // NON-STREAMING: Simple case
        debug!("Processing non-streaming response with internal billing");
        // The request's streaming fields are forwarded unchanged, but this
        // encrypted endpoint currently buffers one byte-exact response carrier.
        let response_json = match read_non_streaming_completion_response(
            response,
            &response_model_id,
            &attempt,
            non_streaming_body_limit,
            Duration::from_secs(REQUEST_TIMEOUT_SECS),
        )
        .await
        {
            Ok(response_json) => response_json,
            Err(failure) => {
                let terminal = AttemptTerminal::Failed {
                    attempt: attempt.clone(),
                    failure,
                };
                terminal_guard.record_terminal(&terminal);
                return Err(CompletionExecutionError::Attempt {
                    terminal,
                    public_error: ApiError::InternalServerError,
                });
            }
        };

        let usage = extract_usage(&response_json);
        let terminal = AttemptTerminal::Completed {
            attempt: attempt.clone(),
            evidence: CompletionEvidence::NonStreamingResponse,
        };
        terminal_guard.record_terminal_with_usage(&terminal, usage.as_ref());

        // ✅ Handle billing HERE, inside completions API
        if let Some(usage) = usage {
            settle_completion_usage(state, user, usage_settlement, usage).await;
        }

        // Return the full response as a single chunk
        let (tx, rx) = mpsc::channel(2); // Need space for FullResponse + terminal
        let _ = tx.send(CompletionChunk::FullResponse(response_json)).await;
        let _ = tx.send(CompletionChunk::Terminal(terminal)).await;

        return Ok(CompletionStream {
            stream: rx,
            metadata: CompletionMetadata {
                provider_name: successful_provider,
                model_name: public_model_id,
                is_streaming: false,
                attempt,
            },
        });
    }

    // STREAMING: Complex case - spawn internal processor
    debug!("Processing streaming response with internal billing");
    let (tx_consumer, rx_consumer) = mpsc::channel(100);

    // Spawn INTERNAL task that handles billing
    let state_clone = state.clone();
    let user_clone = user.clone();
    let stream_attempt = attempt.clone();
    terminal_guard.set_stage(AttemptStage::Stream);

    tokio::spawn(async move {
        let result = process_completion_stream(
            response,
            &response_model_id,
            stream_attempt,
            &tx_consumer,
            Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS),
        )
        .await;
        let terminal = result.terminal.clone();
        terminal_guard.record_terminal_with_usage(&terminal, result.usage.as_ref());
        publish_stream_usage(
            result.usage,
            result.finalization,
            &state_clone,
            &user_clone,
            usage_settlement,
            &tx_consumer,
        )
        .await;
        let _ = tx_consumer.send(CompletionChunk::Terminal(terminal)).await;
    });

    Ok(CompletionStream {
        stream: rx_consumer,
        metadata: CompletionMetadata {
            provider_name: successful_provider,
            model_name: public_model_id,
            is_streaming: true,
            attempt,
        },
    })
}

/// Executes one complete model turn. Responses uses the split start/finish
/// functions so only the first response headers cross its persistence seam.
pub(crate) async fn get_chat_completion_response(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
    pinned: &PinnedCompletionRequest,
) -> Result<CompletionStream, CompletionExecutionError> {
    let started =
        start_chat_completion_response(state, user, body, headers, billing_context, pinned).await?;
    finish_started_completion(state, user, started).await
}

pub(crate) fn ensure_completion_model_access(
    model_name: &str,
    model_plan: ModelPlan,
) -> Result<(), ApiError> {
    if !model_plan.allows_model(model_name) {
        error!(
            "Paid completion model requested without entitlement: {}",
            model_name
        );
        return Err(ApiError::ModelNotAvailableOnPlan);
    }

    Ok(())
}

// ============================================================================
// Centralized Billing Architecture - Internal Functions
// ============================================================================

/// Helper to extract usage from response JSON
fn extract_usage(json: &Value) -> Option<CompletionUsage> {
    let observed = extract_usage_observation(json)?;
    let prompt_tokens = observed.prompt_tokens.unwrap_or(0);

    Some(CompletionUsage {
        prompt_tokens,
        prompt_tokens_observed: observed.prompt_tokens.is_some(),
        completion_tokens: observed.completion_tokens.unwrap_or(0),
        completion_tokens_observed: observed.completion_tokens.is_some(),
        cached_prompt_tokens: observed.cached_prompt_tokens.map(|cached| {
            observed
                .prompt_tokens
                .map_or(cached, |prompt| cached.min(prompt))
        }),
    })
}

fn extract_usage_observation(json: &Value) -> Option<CompletionUsageObservation> {
    let usage_json = json.get("usage")?.as_object()?;
    let prompt_tokens = usage_json
        .get("prompt_tokens")
        .and_then(|v| v.as_i64())
        .filter(|tokens| *tokens >= 0)
        .map(|tokens| tokens.min(i32::MAX as i64) as i32);

    let cached_prompt_tokens = usage_json
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|v| v.as_i64())
        .filter(|tokens| *tokens >= 0)
        .map(|tokens| tokens.min(i32::MAX as i64) as i32);

    Some(CompletionUsageObservation {
        prompt_tokens,
        completion_tokens: usage_json
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .filter(|tokens| *tokens >= 0)
            .map(|tokens| tokens.min(i32::MAX as i64) as i32),
        cached_prompt_tokens,
    })
}

fn ensure_stream_usage(body: &mut serde_json::Map<String, Value>) {
    let is_streaming = body
        .get("stream")
        .and_then(|stream| stream.as_bool())
        .unwrap_or(false);

    if !is_streaming {
        return;
    }

    match body.entry("stream_options".to_string()) {
        serde_json::map::Entry::Occupied(mut entry) => {
            if let Some(options) = entry.get_mut().as_object_mut() {
                options.insert("include_usage".to_string(), json!(true));
            } else {
                entry.insert(json!({ "include_usage": true }));
            }
        }
        serde_json::map::Entry::Vacant(entry) => {
            entry.insert(json!({ "include_usage": true }));
        }
    }
}

fn tinfoil_user_cache_secret(user_uuid: Uuid) -> String {
    // Keep the cache namespace stable without exposing the raw user identifier.
    hex::encode(Sha256::digest(user_uuid.as_bytes()))
}

fn strip_tinfoil_router_execution_controls(body: &mut serde_json::Map<String, Value>) {
    for field in TINFOIL_ROUTER_EXECUTION_FIELDS {
        if body.remove(*field).is_some() {
            debug!(field, "Stripped Tinfoil router-owned execution field");
        }
    }

    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };
    for tool in tools {
        let Some(tool) = tool.as_object_mut() else {
            continue;
        };
        let mut stripped = tool.remove(TINFOIL_TOOL_AUTO_CONTINUE_FIELD).is_some();
        stripped |= tool
            .get_mut("function")
            .and_then(Value::as_object_mut)
            .is_some_and(|function| function.remove(TINFOIL_TOOL_AUTO_CONTINUE_FIELD).is_some());
        if stripped {
            debug!("Stripped Tinfoil router-owned tool auto-continue flag");
        }
    }
}

fn apply_provider_managed_request_fields(
    body: &mut serde_json::Map<String, Value>,
    provider_name: &str,
    user_uuid: Uuid,
) {
    strip_provider_internal_request_fields(body);
    if body.remove(PROVIDER_MANAGED_CACHE_SALT_FIELD).is_some() {
        debug!("Stripped provider-managed completion request field: cache_salt");
    }

    let replaced_user_cache_secret = body
        .remove(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD)
        .is_some();

    if provider_name == ProviderId::Tinfoil.as_str() {
        // OpenSecret owns tool loops, admission, and model routing. Tinfoil's
        // router-only controls can otherwise fan one admitted request into
        // several hidden upstream generations.
        strip_tinfoil_router_execution_controls(body);
        body.insert(
            PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD.to_string(),
            json!(tinfoil_user_cache_secret(user_uuid)),
        );
        if replaced_user_cache_secret {
            debug!("Replaced provider-managed completion request field: user_cache_secret");
        }
    } else if replaced_user_cache_secret {
        debug!("Stripped provider-managed completion request field: user_cache_secret");
    }
}

fn apply_server_selected_cache_isolation(
    body: &mut serde_json::Map<String, Value>,
    provider_name: &str,
    continuum_cache_salt: Option<&str>,
) -> Result<(), ApiError> {
    if provider_name != ProviderId::Continuum.as_str() {
        return Ok(());
    }

    let cache_salt = continuum_cache_salt
        .filter(|salt| salt.len() >= 32)
        .ok_or_else(|| {
            error!("Missing or invalid server-owned Continuum cache salt");
            ApiError::InternalServerError
        })?;
    body.insert(
        PROVIDER_MANAGED_CACHE_SALT_FIELD.to_string(),
        json!(cache_salt),
    );
    Ok(())
}

/// A finish reason marks model completion, but providers may send a richer
/// usage-only frame after it and before the SSE [DONE] marker.
fn has_finish_reason(json: &Value) -> bool {
    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            if let Some(finish_reason) = choice.get("finish_reason") {
                // finish_reason is present and not null
                if !finish_reason.is_null() {
                    return true;
                }
            }
        }
    }
    false
}

/// Records enough terminal evidence to safely fall back when a provider omits
/// [DONE]. This does not imply that no additional usage frame can follow.
fn has_terminal_stream_signal(json: &Value) -> bool {
    if has_finish_reason(json) {
        return true;
    }

    json.get("choices")
        .and_then(|choices| choices.as_array())
        .is_some_and(|choices| choices.is_empty())
}

fn canonicalize_response_model(json: &mut Value, response_model_id: &str) {
    if let Some(model_value) = json.get_mut("model") {
        if model_value.as_str().is_some() {
            *model_value = json!(response_model_id);
        }
    }
}

/// Returns the joined data payload of the next complete SSE frame.
///
/// Buffering bytes until the frame boundary avoids corrupting a UTF-8 code
/// point when the network splits it across chunks.
fn extract_sse_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    loop {
        let (frame_end, delimiter_len) = find_sse_frame_boundary(buffer)?;
        let frame = buffer[..frame_end].to_vec();
        buffer.drain(..frame_end + delimiter_len);

        if frame.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }

        let mut data = Vec::new();
        for line in frame.split(|byte| *byte == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(value) = line.strip_prefix(b"data:") else {
                continue;
            };
            let value = value.strip_prefix(b" ").unwrap_or(value);
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(value);
        }

        if !data.is_empty() {
            return Some(data);
        }
    }
}

fn find_sse_frame_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");

    match (lf, crlf) {
        (Some(lf), Some(crlf)) if lf < crlf => Some((lf, 2)),
        (Some(_), Some(crlf)) | (None, Some(crlf)) => Some((crlf, 4)),
        (Some(lf), None) => Some((lf, 2)),
        (None, None) => None,
    }
}

/// Internal billing function - NEVER exposed outside this module
/// Settles one completion execution with a stable idempotency key.
async fn settle_completion_usage(
    state: &Arc<AppState>,
    user: &User,
    settlement: UsageSettlement,
    usage: CompletionUsage,
) {
    debug!(
        "Settling inference usage exactly once: request_id={}, execution_id={}, attempt_id={}, requested_model={}, public_model={}, provider={}",
        settlement.request_id,
        settlement.execution_id,
        settlement.attempt_id,
        settlement.requested_model_id,
        settlement.public_model_id,
        settlement.provider_name
    );
    publish_usage_record(
        state,
        user,
        settlement.event_id,
        settlement.auth_method == AuthMethod::ApiKey,
        settlement.provider_name,
        settlement.public_model_id,
        usage,
    )
    .await;
}

/// Legacy settlement seam for modalities that have not migrated to routing-v2.
/// Stack 9 owns moving those callers onto execution-scoped identifiers.
async fn publish_usage_event_internal(
    state: &Arc<AppState>,
    user: &User,
    billing_context: &BillingContext,
    usage: CompletionUsage,
    provider_name: &str,
) {
    publish_usage_record(
        state,
        user,
        Uuid::new_v4(),
        billing_context.auth_method == AuthMethod::ApiKey,
        provider_name.to_string(),
        billing_context.model_name.clone(),
        usage,
    )
    .await;
}

/// Publishes one already-identified usage record to the legacy local table and
/// the authoritative billing queue. Callers own idempotency before this seam.
#[allow(clippy::too_many_arguments)]
async fn publish_usage_record(
    state: &Arc<AppState>,
    user: &User,
    event_id: Uuid,
    is_api_request: bool,
    provider_name: String,
    model_name: String,
    mut usage: CompletionUsage,
) {
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return;
    }

    // Partial provider usage can still carry a raw cached-token observation for
    // conservative admission settlement. Billing requires cached input to be a
    // subset of an observed prompt total, so omit it when that total is absent.
    usage.cached_prompt_tokens = usage
        .prompt_tokens_observed
        .then(|| {
            usage
                .cached_prompt_tokens
                .map(|cached| cached.min(usage.prompt_tokens))
        })
        .flatten();

    // Local token_usage keeps the legacy rough estimate for observability.
    // The billing server recomputes authoritative provider cost from SQS tokens.
    let input_cost =
        BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(usage.prompt_tokens);
    let output_cost =
        BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(usage.completion_tokens);
    let total_cost = input_cost + output_cost;

    info!(
        "Chat completion usage for user {}: model={}, provider={}, prompt_tokens={}, cached_prompt_tokens={}, completion_tokens={}, total_tokens={}, estimated_cost={}",
        user.uuid,
        model_name,
        provider_name,
        usage.prompt_tokens,
        usage.cached_prompt_tokens.unwrap_or(0),
        usage.completion_tokens,
        usage.prompt_tokens + usage.completion_tokens,
        total_cost
    );

    // Spawn background task for DB + SQS
    let state_clone = state.clone();
    let user_id = user.uuid;

    tokio::spawn(async move {
        // Create and store token usage record
        let new_usage = NewTokenUsage::new(
            user_id,
            usage.prompt_tokens,
            usage.completion_tokens,
            total_cost.clone(),
        );

        if let Err(e) = state_clone.db.create_token_usage(new_usage) {
            error!("Failed to save token usage: {:?}", e);
        }

        let cached_input_tokens = usage.cached_prompt_tokens;

        // Post event to SQS if configured
        if let Some(publisher) = &state_clone.sqs_publisher {
            let event = build_usage_event(
                event_id,
                user_id,
                usage,
                total_cost,
                is_api_request,
                provider_name,
                model_name,
            );

            debug!(
                "Prepared SQS usage event: user_uuid={}, provider={}, model={}, input_tokens={}, output_tokens={}, cached_input_tokens={:?}",
                event.user_id,
                event.provider_name,
                event.model_name,
                event.input_tokens,
                event.output_tokens,
                event.cached_input_tokens
            );

            match publisher.publish_event(event).await {
                Ok(_) => debug!("published usage event successfully"),
                Err(e) => error!("error publishing usage event: {e}"),
            }
        } else {
            debug!(
                "SQS publisher not configured; usage event would include cached_input_tokens={:?}",
                cached_input_tokens
            );
        }
    });
}

fn build_usage_event(
    event_id: Uuid,
    user_id: Uuid,
    usage: CompletionUsage,
    estimated_cost: BigDecimal,
    is_api_request: bool,
    provider_name: String,
    model_name: String,
) -> UsageEvent {
    UsageEvent {
        event_id,
        user_id,
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        estimated_cost,
        chat_time: Utc::now(),
        is_api_request,
        provider_name,
        model_name,
        cached_input_tokens: usage.cached_prompt_tokens,
    }
}

/// Helper to encrypt an SSE event
async fn encrypt_sse_event(
    state: &AppState,
    session_id: &Uuid,
    json: &Value,
) -> Result<Event, ApiError> {
    let json_str = json.to_string();
    let encrypted_data = state
        .encrypt_session_data(session_id, json_str.as_bytes())
        .await
        .map_err(|e| {
            error!("Failed to encrypt SSE event data: {:?}", e);
            ApiError::InternalServerError
        })?;

    Ok(sse_event_from_encrypted_data(&encrypted_data))
}

fn completion_error_payload(message: &'static str) -> Value {
    json!({
        "error": {
            "message": message
        }
    })
}

fn sse_event_from_encrypted_data(encrypted_data: &[u8]) -> Event {
    Event::default().data(general_purpose::STANDARD.encode(encrypted_data))
}

async fn proxy_models(
    State(state): State<Arc<AppState>>,
    axum::Extension(session_id): axum::Extension<Uuid>,
    user: Option<axum::Extension<User>>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let _ = user;
    let proxy_config = state.proxy_router.get_completion_proxy();
    let models_response = if proxy_config.provider_name == "tinfoil" {
        openai_models_response()
    } else {
        fetch_provider_models(&state.provider_client, &proxy_config).await?
    };
    encrypt_response(&state, &session_id, &models_response).await
}

async fn fetch_provider_models(
    client: &ProviderClient,
    proxy_config: &ProxyConfig,
) -> Result<Value, ApiError> {
    let res = client
        .send(
            proxy_config,
            ProviderRequest::new(
                Method::GET,
                "/v1/models",
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
            ),
        )
        .await
        .map_err(|e| {
            error!(
                "Failed to fetch models from provider {}: {:?}",
                proxy_config.provider_name, e
            );
            ApiError::from(e)
        })?;

    if !res.is_success() {
        let status = res.status_code();
        let body_bytes = res.bytes().await.ok();
        let error_msg = body_bytes
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|| status.to_string());
        error!(
            "Provider {} returned non-success status for models: {} - {}",
            proxy_config.provider_name, status, error_msg
        );
        return Err(ApiError::InternalServerError);
    }

    let body_bytes = res.bytes().await.map_err(|e| {
        error!("Failed to read models response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    serde_json::from_slice(&body_bytes).map_err(|e| {
        error!("Failed to parse models response: {:?}", e);
        ApiError::InternalServerError
    })
}

async fn proxy_model_catalog(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(_body): axum::Extension<()>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let billing_access = state.chat_billing_access(user.uuid, false).await;
    let model_plan = ModelPlan::from_is_paid(
        billing_access.is_some_and(crate::billing::ChatBillingAccess::is_paid),
    );
    let alias_targets = state.model_alias_targets(user.uuid, model_plan).await;
    let catalog_response = model_catalog_response(alias_targets);
    encrypt_response(&state, &session_id, &catalog_response).await
}

fn transcription_model_for_provider(model_name: &str, provider_name: &str) -> String {
    match (model_name, provider_name) {
        ("whisper-large-v3", "tinfoil") => "whisper-large-v3-turbo".to_string(),
        ("whisper-large-v3-turbo", provider) if provider != "tinfoil" => {
            "whisper-large-v3".to_string()
        }
        _ => model_name.to_string(),
    }
}

/// Helper function to send transcription request with retries to primary and fallback providers
async fn send_transcription_with_retries(
    client: &ProviderClient,
    primary_provider: &ProxyConfig,
    fallback_provider: Option<&ProxyConfig>,
    model_name: &str,
    params: &TranscriptionParams<'_>,
) -> Result<Value, ApiError> {
    let max_cycles = 3;
    let mut last_error = None;

    for cycle in 0..max_cycles {
        if cycle > 0 {
            let delay = cycle as u64;
            debug!("Starting cycle {} after {}s delay", cycle + 1, delay);
            sleep(Duration::from_secs(delay)).await;
        }

        debug!(
            "Cycle {}: Trying primary provider {} for transcription",
            cycle + 1,
            primary_provider.provider_name
        );

        let primary_model =
            transcription_model_for_provider(model_name, &primary_provider.provider_name);

        match send_transcription_request(client, primary_provider, &primary_model, params).await {
            Ok(response) => {
                info!(
                    "Successfully got transcription from primary provider {} on cycle {}",
                    primary_provider.provider_name,
                    cycle + 1
                );
                return Ok(response);
            }
            Err(err) => {
                error!(
                    "Cycle {}: Primary provider {} failed: {}",
                    cycle + 1,
                    primary_provider.provider_name,
                    err
                );
                last_error = Some(err);
            }
        }

        if let Some(fallback_provider) = fallback_provider {
            debug!(
                "Cycle {}: Trying fallback provider {} for transcription",
                cycle + 1,
                fallback_provider.provider_name
            );

            let fallback_model =
                transcription_model_for_provider(model_name, &fallback_provider.provider_name);

            match send_transcription_request(client, fallback_provider, &fallback_model, params)
                .await
            {
                Ok(response) => {
                    info!(
                        "Successfully got transcription from fallback provider {} on cycle {}",
                        fallback_provider.provider_name,
                        cycle + 1
                    );
                    return Ok(response);
                }
                Err(err) => {
                    error!(
                        "Cycle {}: Fallback provider {} failed: {}",
                        cycle + 1,
                        fallback_provider.provider_name,
                        err
                    );
                    last_error = Some(err);
                }
            }
        }
    }

    error!(
        "All transcription providers failed after {} cycles. Last error: {:?}",
        max_cycles, last_error
    );
    Err(last_error.unwrap_or(ApiError::InternalServerError))
}

async fn proxy_transcription(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(transcription_request): axum::Extension<TranscriptionRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for transcription: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use transcription feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use transcription without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
    }

    // Decode base64 audio file
    let file_bytes = general_purpose::STANDARD
        .decode(&transcription_request.file)
        .map_err(|e| {
            error!("Failed to decode base64 audio file: {:?}", e);
            ApiError::BadRequest
        })?;

    // Validate file size (100MB limit as sanity check, CF already limits to 50MB)
    let file_size = file_bytes.len();
    if file_size == 0 {
        error!("Audio file is empty");
        return Err(ApiError::BadRequest);
    }
    if file_size > MAX_AUDIO_SIZE {
        error!(
            "Audio file size {} bytes exceeds maximum allowed size of {} bytes",
            file_size, MAX_AUDIO_SIZE
        );
        return Err(ApiError::BadRequest);
    }
    info!("Audio file size: {} bytes", file_size);

    // Check if we need to split the audio
    let splitter = AudioSplitter::new();

    let default_proxy = state.proxy_router.get_default_proxy();
    let tinfoil_proxy = state.proxy_router.get_tinfoil_proxy();

    let client = state.provider_client.clone();

    // Always split the audio (returns single chunk if no splitting needed)
    let chunks = splitter
        .split_audio(&file_bytes, &transcription_request.content_type)
        .map_err(|e| {
            error!("Failed to split audio: {}", e);
            ApiError::InternalServerError
        })?;

    info!("Processing {} chunk(s)", chunks.len());

    // Process chunks in parallel (even if it's just one)
    let mut futures = Vec::new();

    for chunk in chunks {
        let client = client.clone();
        let model_name = transcription_request.model.clone();
        let filename = transcription_request.filename.clone();
        let content_type = transcription_request.content_type.clone();
        let language = transcription_request.language.clone();
        let prompt = transcription_request.prompt.clone();
        let response_format = transcription_request.response_format.clone();
        let temperature = transcription_request.temperature;
        let default_proxy = default_proxy.clone();
        let tinfoil_proxy = tinfoil_proxy.clone();

        let future = async move {
            let chunk_size = chunk.data.len();
            info!(
                "Processing chunk {} (size: {} bytes)",
                chunk.index, chunk_size
            );

            let mut primary_provider = tinfoil_proxy.clone();
            let mut fallback_provider = Some(default_proxy.clone());

            if chunk_size > TINFOIL_MAX_SIZE && primary_provider.provider_name == "tinfoil" {
                info!(
                    "Chunk {} size {} bytes exceeds Tinfoil's 0.5MB limit, using fallback only",
                    chunk.index, chunk_size
                );

                if let Some(fallback) = fallback_provider.take() {
                    primary_provider = fallback;
                } else {
                    error!(
                        "Chunk {} size {} bytes exceeds Tinfoil's limit and no fallback is available",
                        chunk.index, chunk_size
                    );
                    return Err(ApiError::InternalServerError);
                }
            }

            let params = TranscriptionParams {
                audio_data: &chunk.data,
                filename: &filename,
                content_type: &content_type,
                language: language.as_deref(),
                prompt: prompt.as_deref(),
                response_format: &response_format,
                temperature,
            };

            match send_transcription_with_retries(
                &client,
                &primary_provider,
                fallback_provider.as_ref(),
                &model_name,
                &params,
            )
            .await
            {
                Ok(response) => {
                    info!("Chunk {} transcribed successfully", chunk.index);
                    Ok((chunk.index, response))
                }
                Err(err) => {
                    error!("Chunk {} failed: {}", chunk.index, err);
                    Err(err)
                }
            }
        };

        futures.push(future);
    }

    // Execute all futures in parallel
    let results: Vec<Result<(usize, Value), ApiError>> = futures::future::join_all(futures).await;

    // Check if all chunks succeeded
    let mut successful_results = Vec::new();
    for result in results {
        match result {
            Ok(r) => successful_results.push(r),
            Err(e) => {
                error!("Chunk processing failed: {}", e);
                return Err(e);
            }
        }
    }

    // Get the response (merge if multiple chunks, return as-is if single)
    let response = if successful_results.is_empty() {
        error!("No successful transcription results");
        return Err(ApiError::InternalServerError);
    } else if successful_results.len() == 1 {
        // Single chunk - return the response directly
        successful_results
            .into_iter()
            .next()
            .map(|(_, r)| r)
            .ok_or_else(|| {
                error!("Failed to get single result");
                ApiError::InternalServerError
            })?
    } else {
        // Multiple chunks - merge the results
        let merged = merge_transcriptions(successful_results).map_err(|e| {
            error!("Failed to merge transcriptions: {}", e);
            ApiError::InternalServerError
        })?;

        // Convert merged result to standard response format
        let mut response = json!({
            "text": merged.text,
        });

        if let Some(lang) = merged.language {
            response["language"] = json!(lang);
        }

        if let Some(segments) = merged.segments {
            response["segments"] = json!(segments);
        }

        response
    };

    // TODO: Add SQS-based billing events for transcription usage
    // Should track: audio duration/size, model used, user ID, timestamp, provider

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &response).await
}

/// Sanitize form field values to prevent HTTP header injection attacks
/// Removes or replaces any CRLF sequences that could be used to inject headers
fn sanitize_form_field(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '\r' | '\n'))
        .collect()
}

async fn send_transcription_request(
    client: &ProviderClient,
    provider: &ProxyConfig,
    model: &str,
    params: &TranscriptionParams<'_>,
) -> Result<Value, ApiError> {
    // Build multipart form data
    let boundary = format!("----FormBoundary{}", Uuid::new_v4().simple());
    let mut form_data = Vec::new();

    // Add model field (sanitized to prevent header injection)
    let safe_model = sanitize_form_field(model);
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    form_data.extend_from_slice(safe_model.as_bytes());
    form_data.extend_from_slice(b"\r\n");

    // Add file field with sanitized filename to prevent header injection
    let safe_filename: String = params
        .filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();

    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
            safe_filename
        )
        .as_bytes(),
    );
    let safe_content_type = sanitize_form_field(params.content_type);
    form_data.extend_from_slice(format!("Content-Type: {}\r\n\r\n", safe_content_type).as_bytes());
    form_data.extend_from_slice(params.audio_data);
    form_data.extend_from_slice(b"\r\n");

    // Add optional fields (sanitized to prevent header injection)
    if let Some(lang) = params.language {
        let safe_lang = sanitize_form_field(lang);
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        form_data.extend_from_slice(safe_lang.as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }
    if let Some(p) = params.prompt {
        let safe_prompt = sanitize_form_field(p);
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"prompt\"\r\n\r\n");
        form_data.extend_from_slice(safe_prompt.as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }
    let safe_response_format = sanitize_form_field(params.response_format);
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data
        .extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
    form_data.extend_from_slice(safe_response_format.as_bytes());
    form_data.extend_from_slice(b"\r\n");
    if let Some(temp) = params.temperature {
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data
            .extend_from_slice(b"Content-Disposition: form-data; name=\"temperature\"\r\n\r\n");
        form_data.extend_from_slice(temp.to_string().as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }

    // End boundary
    form_data.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    let content_type = format!("multipart/form-data; boundary={boundary}");
    match client
        .send(
            provider,
            ProviderRequest::new(
                Method::POST,
                "/v1/audio/transcriptions",
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
            )
            .content_type(&content_type)
            .body(form_data),
        )
        .await
    {
        Ok(res) => {
            if res.is_success() {
                let body_bytes = res.bytes().await.map_err(|e| {
                    error!("Failed to read transcription response body: {:?}", e);
                    ApiError::InternalServerError
                })?;

                let response_json: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
                    error!("Failed to parse transcription response: {:?}", e);
                    ApiError::InternalServerError
                })?;

                Ok(response_json)
            } else {
                let status = res.status_code();
                let body_bytes = res.bytes().await.ok();
                let error_msg = body_bytes
                    .map(|b| String::from_utf8_lossy(&b).to_string())
                    .unwrap_or_else(|| status.to_string());

                error!(
                    "Provider {} returned transcription error: {} - {}",
                    provider.provider_name, status, error_msg
                );
                Err(ApiError::InternalServerError)
            }
        }
        Err(e) => {
            error!(
                "Failed to send transcription request to {}: {:?}",
                provider.provider_name, e
            );
            Err(ApiError::from(e))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TTSBillingAccess {
    Allowed,
    FreeOrExhausted,
    Unavailable,
}

fn tts_billing_access_decision(access: TTSBillingAccess) -> Result<(), ApiError> {
    match access {
        TTSBillingAccess::Allowed => Ok(()),
        TTSBillingAccess::FreeOrExhausted => Err(ApiError::UsageLimitReached),
        TTSBillingAccess::Unavailable => Err(ApiError::ServiceUnavailable),
    }
}

async fn ensure_paid_tts_access(state: &AppState, user: &User) -> Result<(), ApiError> {
    let access = if let Some(billing_client) = &state.billing_client {
        match timeout(
            TTS_BILLING_CHECK_TIMEOUT,
            billing_client.can_user_use_paid_features(user.uuid),
        )
        .await
        {
            Ok(Ok(true)) => TTSBillingAccess::Allowed,
            Ok(Ok(false)) => TTSBillingAccess::FreeOrExhausted,
            Ok(Err(_)) => {
                warn!(
                    user_uuid = %user.uuid,
                    "TTS billing entitlement check failed"
                );
                TTSBillingAccess::Unavailable
            }
            Err(_) => {
                warn!(
                    user_uuid = %user.uuid,
                    timeout_seconds = TTS_BILLING_CHECK_TIMEOUT.as_secs(),
                    "TTS billing entitlement check timed out"
                );
                TTSBillingAccess::Unavailable
            }
        }
    } else {
        warn!(
            user_uuid = %user.uuid,
            "TTS requested while the billing client is unavailable"
        );
        TTSBillingAccess::Unavailable
    };

    let result = tts_billing_access_decision(access);
    if result.is_err() {
        warn!(
            user_uuid = %user.uuid,
            access = ?access,
            "Denied paid TTS access"
        );
    }
    result
}

async fn proxy_tts(
    State(state): State<Arc<AppState>>,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(tts_request): axum::Extension<TTSRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let prepared = prepare_tts_request(tts_request).map_err(|validation_error| {
        warn!(
            error = %validation_error,
            "Rejected invalid TTS request"
        );
        match validation_error {
            TTSRequestValidationError::InputTooLong => ApiError::PayloadTooLarge,
            _ => ApiError::BadRequest,
        }
    })?;
    ensure_paid_tts_access(&state, &user).await?;

    let proxy_config = state.proxy_router.get_tinfoil_proxy();
    let request_body = serde_json::to_vec(&prepared.provider_payload).map_err(|e| {
        error!("Failed to serialize TTS request: {:?}", e);
        ApiError::InternalServerError
    })?;

    let (body_bytes, response_content_type) = timeout(TTS_PROVIDER_TIMEOUT, async {
        let res = state
            .provider_client
            .send(
                &proxy_config,
                ProviderRequest::new(Method::POST, "/v1/audio/speech", TTS_PROVIDER_TIMEOUT)
                    .content_type("application/json")
                    .body(request_body),
            )
            .await
            .map_err(|e| {
                error!("Failed to create TTS request: {:?}", e);
                ApiError::from(e)
            })?;

        if !res.is_success() {
            error!(
                model = %prepared.model,
                status = res.status_code(),
                "TTS provider returned a non-success status"
            );
            return Err(ApiError::InternalServerError);
        }

        let content_type = res
            .content_type()
            .unwrap_or("application/octet-stream")
            .to_string();
        let body_bytes = res.bytes().await.map_err(|e| {
            error!("Failed to read TTS response body: {:?}", e);
            ApiError::InternalServerError
        })?;
        Ok((body_bytes, content_type))
    })
    .await
    .map_err(|_| {
        error!(
            model = %prepared.model,
            timeout_seconds = TTS_PROVIDER_TIMEOUT.as_secs(),
            "TTS provider request timed out"
        );
        ApiError::InternalServerError
    })??;

    let (response_payload, is_json_response) =
        build_tts_response_payload(&body_bytes, &response_content_type);
    if is_json_response {
        warn!(
            model = %prepared.model,
            voice = %prepared.voice_for_log,
            response_bytes = body_bytes.len(),
            "TTS provider returned a successful JSON response"
        );
    } else {
        info!(
            model = %prepared.model,
            voice = %prepared.voice_for_log,
            content_type = %response_content_type,
            audio_bytes = body_bytes.len(),
            "TTS synthesis succeeded"
        );
    }

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &response_payload).await
}

async fn proxy_embeddings(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(embedding_request): axum::Extension<EmbeddingRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for embeddings: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use embeddings feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use embeddings without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
    }

    // Validate input is not empty
    let is_empty = match &embedding_request.input {
        Value::String(s) => s.trim().is_empty(),
        Value::Array(arr) => arr.is_empty(),
        _ => true,
    };
    if is_empty {
        error!("Input is empty or invalid");
        return Err(ApiError::BadRequest);
    }

    let proxy_config = state.proxy_router.get_tinfoil_proxy();

    // Build request body
    let request_body = serde_json::to_string(&embedding_request).map_err(|e| {
        error!("Failed to serialize embedding request: {:?}", e);
        ApiError::InternalServerError
    })?;

    let res = state
        .provider_client
        .send(
            &proxy_config,
            ProviderRequest::new(
                Method::POST,
                "/v1/embeddings",
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
            )
            .content_type("application/json")
            .body(request_body.into_bytes()),
        )
        .await
        .map_err(|e| {
            error!("Failed to send embeddings request: {:?}", e);
            ApiError::from(e)
        })?;

    if !res.is_success() {
        let status = res.status_code();
        let body_bytes = res.bytes().await.ok();
        let error_msg = body_bytes
            .map(|b| String::from_utf8_lossy(&b).to_string())
            .unwrap_or_else(|| status.to_string());
        error!(
            "Embeddings proxy returned non-success status: {} - {}",
            status, error_msg
        );
        return Err(ApiError::InternalServerError);
    }

    // Parse response
    let body_bytes = res.bytes().await.map_err(|e| {
        error!("Failed to read embeddings response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    let response_json: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
        error!("Failed to parse embeddings response: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Handle billing - embeddings only have prompt_tokens (no completion_tokens)
    if let Some(usage) = response_json.get("usage") {
        let prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;

        if prompt_tokens > 0 {
            let billing_context =
                BillingContext::new(_auth_method, embedding_request.model.clone());
            let embedding_usage = CompletionUsage {
                prompt_tokens,
                prompt_tokens_observed: true,
                completion_tokens: 0, // Embeddings don't have completion tokens
                completion_tokens_observed: false,
                cached_prompt_tokens: None,
            };
            publish_usage_event_internal(
                &state,
                &user,
                &billing_context,
                embedding_usage,
                &proxy_config.provider_name,
            )
            .await;
        }
    }

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &response_json).await
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum BoundedProviderResponseBodyError {
    #[error("provider response body read failed")]
    Read,
    #[error("provider response body exceeded the {limit_bytes}-byte limit")]
    TooLarge { limit_bytes: usize },
}

async fn collect_bounded_provider_response_body<S>(
    mut body_stream: S,
    limit_bytes: usize,
) -> Result<Vec<u8>, BoundedProviderResponseBodyError>
where
    S: futures::Stream<Item = Result<bytes::Bytes, String>> + Unpin,
{
    let mut body = Vec::new();

    while let Some(chunk) = body_stream.next().await {
        let chunk = chunk.map_err(|_| BoundedProviderResponseBodyError::Read)?;
        if chunk.len() > limit_bytes.saturating_sub(body.len()) {
            return Err(BoundedProviderResponseBodyError::TooLarge { limit_bytes });
        }
        body.extend_from_slice(&chunk);
    }

    Ok(body)
}

/// Helper function to try a provider once
async fn try_provider(
    client: &ProviderClient,
    proxy_config: &ProxyConfig,
    body_json: String,
    headers: &HeaderMap,
) -> ProviderSendTrace {
    debug!("Making request to {}", proxy_config.provider_name);

    let ProviderSendTrace {
        prior_failures,
        result,
    } = client
        .send_traced(
            proxy_config,
            ProviderRequest::new(
                Method::POST,
                "/v1/chat/completions",
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
            )
            .source_headers(headers)
            .content_type("application/json")
            // `Bytes::from(String)` reuses the serialized request allocation;
            // retry templates then clone that buffer by reference.
            .body(body_json),
        )
        .await;

    let result = match result {
        Ok(response) => {
            if response.is_success() {
                Ok(response)
            } else {
                let status = response.status_code();
                let retry_after = parse_retry_after_hint(response.header_str(&header::RETRY_AFTER));
                let upstream_request_id = upstream_request_id(&response);
                error!(
                    "Provider {} returned non-success status: {}",
                    proxy_config.provider_name, status
                );
                // The status and bounded safe headers are the complete routing
                // contract. Never wait for or buffer an untrusted error body.
                drop(response);
                Err(ProviderRequestError::Upstream(UpstreamProviderError {
                    status,
                    retry_after,
                    upstream_request_id,
                }))
            }
        }
        Err(e) => {
            error!(
                "Failed to send request to {}: {:?}",
                proxy_config.provider_name, e
            );
            Err(e)
        }
    };

    ProviderSendTrace {
        prior_failures,
        result,
    }
}

fn parse_retry_after_hint(value: Option<&str>) -> Option<Duration> {
    const MAX_RETRY_AFTER_HINT_SECS: u64 = 60 * 60;

    let seconds = value?.trim().parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds.min(MAX_RETRY_AFTER_HINT_SECS)))
}

fn upstream_request_id(response: &ProviderResponse) -> Option<String> {
    const REQUEST_ID_HEADERS: &[&str] =
        &["x-request-id", "request-id", "x-amzn-requestid", "cf-ray"];

    REQUEST_ID_HEADERS.iter().find_map(|name| {
        let name = HeaderName::from_static(name);
        let value = response.header_str(&name)?.trim();
        (!value.is_empty() && value.len() <= 128 && value.chars().all(is_safe_identifier_char))
            .then(|| value.to_string())
    })
}

fn is_safe_identifier_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | ':')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn bounded_provider_response_body_accepts_exact_limit_across_chunks() {
        let body_stream = futures::stream::iter([
            Ok::<_, String>(bytes::Bytes::from_static(b"abcd")),
            Ok(bytes::Bytes::from_static(b"ef")),
        ]);

        let body = collect_bounded_provider_response_body(body_stream, 6)
            .await
            .expect("body at the limit should be accepted");

        assert_eq!(body, b"abcdef");
    }

    #[tokio::test]
    async fn bounded_provider_response_body_rejects_before_appending_over_limit_chunk() {
        let body_stream = futures::stream::iter([
            Ok::<_, String>(bytes::Bytes::from_static(b"abcd")),
            Ok(bytes::Bytes::from_static(b"efg")),
        ]);

        assert_eq!(
            collect_bounded_provider_response_body(body_stream, 6).await,
            Err(BoundedProviderResponseBodyError::TooLarge { limit_bytes: 6 })
        );
    }

    async fn start_mock_provider(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock provider");
        let address = listener.local_addr().expect("mock provider address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("serve mock provider");
        });
        (format!("http://{address}"), server)
    }

    async fn call_mock_provider(app: Router) -> (ProviderSendTrace, tokio::task::JoinHandle<()>) {
        let (base_url, server) = start_mock_provider(app).await;
        let client = ProviderClient::for_test(base_url.clone()).expect("mock provider client");
        let proxy = ProxyConfig {
            base_url,
            api_key: None,
            provider_name: ProviderId::Tinfoil.as_str().to_string(),
        };
        let trace = try_provider(
            &client,
            &proxy,
            r#"{"model":"kimi-k3","messages":[]}"#.to_string(),
            &HeaderMap::new(),
        )
        .await;
        (trace, server)
    }

    fn static_sse_app(body: &'static str) -> Router {
        Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(axum::body::Body::from(body))
                    .expect("mock SSE response")
            }),
        )
    }

    async fn process_mock_sse(body: &'static str) -> (StreamProcessResult, Vec<Value>) {
        let (trace, server) = call_mock_provider(static_sse_app(body)).await;
        assert!(trace.prior_failures.is_empty());
        let response = match trace.result {
            Ok(response) => response,
            Err(error) => panic!("mock provider failed before stream processing: {error}"),
        };
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let (tx, mut rx) = mpsc::channel(16);
        let result =
            process_completion_stream(response, "kimi-k3", attempt, &tx, Duration::from_secs(1))
                .await;
        drop(tx);
        server.abort();

        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            match chunk {
                CompletionChunk::StreamChunk(json) => chunks.push(json),
                unexpected => panic!("parser emitted unexpected chunk: {unexpected:?}"),
            }
        }
        (result, chunks)
    }

    fn pinned_test_completion() -> PinnedCompletionRequest {
        PinnedCompletionRequest {
            intent: InferenceIntent::new(
                Uuid::nil(),
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
                "kimi-k3",
                ModelPlan::Paid,
                InferenceSurface::Responses,
                WorkloadClass::Interactive,
            ),
            route: SelectedProviderRoute {
                provider: ProviderId::Tinfoil,
                proxy: ProxyConfig {
                    base_url: "http://tinfoil.example.com".to_string(),
                    api_key: None,
                    provider_name: "tinfoil".to_string(),
                },
                public_model_id: "kimi-k3".to_string(),
                provider_model_id: "kimi-k3".to_string(),
                response_model_id: "kimi-k3".to_string(),
                bucket: None,
                selection_source: crate::provider_registry::RouteSelectionSource::StaticSplit,
                model_selection_source: crate::provider_routing::ModelSelectionSource::AutoPrimary,
            },
            provider_preference: None,
            finalized_route: Arc::new(OnceLock::new()),
        }
    }

    #[test]
    fn exact_completion_constraint_checks_typed_and_transport_route() {
        let pinned = pinned_test_completion();
        let matching = ExactCompletionRoute {
            provider_name: "tinfoil".to_string(),
            provider_model_id: "kimi-k3".to_string(),
        };
        assert!(completion_route_matches_exact_constraint(
            &pinned.route,
            &matching
        ));

        let wrong_provider = ExactCompletionRoute {
            provider_name: "continuum".to_string(),
            provider_model_id: "kimi-k3".to_string(),
        };
        assert!(!completion_route_matches_exact_constraint(
            &pinned.route,
            &wrong_provider
        ));

        let wrong_model = ExactCompletionRoute {
            provider_name: "tinfoil".to_string(),
            provider_model_id: "gemma4-31b".to_string(),
        };
        assert!(!completion_route_matches_exact_constraint(
            &pinned.route,
            &wrong_model
        ));
    }

    fn pinned_auto_fallback_completion() -> PinnedCompletionRequest {
        PinnedCompletionRequest {
            intent: InferenceIntent::new(
                Uuid::nil(),
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
                crate::model_config::KIMI_K3_MODEL_ID,
                ModelPlan::Paid,
                InferenceSurface::ChatCompletions,
                WorkloadClass::Interactive,
            ),
            route: SelectedProviderRoute {
                provider: ProviderId::Continuum,
                proxy: ProxyConfig {
                    base_url: "http://continuum.example.com".to_string(),
                    api_key: None,
                    provider_name: "continuum".to_string(),
                },
                public_model_id: crate::model_config::POWERFUL_MODEL_ID.to_string(),
                provider_model_id: "kimi-k2.6".to_string(),
                response_model_id: crate::model_config::POWERFUL_MODEL_ID.to_string(),
                bucket: None,
                selection_source: crate::provider_registry::RouteSelectionSource::DefaultProvider,
                model_selection_source: crate::provider_routing::ModelSelectionSource::AutoFallback,
            },
            provider_preference: None,
            finalized_route: Arc::new(OnceLock::new()),
        }
    }

    #[test]
    fn admission_estimate_sanitizes_images_and_uses_selected_deployment_bounds() {
        let body = json!({
            "model": "kimi-k3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello admission"},
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", "A".repeat(100_000))}},
                    {"type": "input_image", "image_url": "data:image/png;base64,BBBB"},
                    {"image_url": "https://example.com/shorthand.png"},
                    {"type": null, "image_url": "https://example.com/null-shorthand.png"},
                    {"type": "garbage", "image_url": "https://example.com/coerced-shorthand.png"}
                ]
            }],
            "max_completion_tokens": 321,
            "n": 4
        });

        let k3 = pinned_test_completion();
        let k2 = pinned_auto_fallback_completion();
        let k3_estimate =
            estimate_completion_admission(body.as_object().expect("object"), &k3.route, 4_096);
        let k2_estimate =
            estimate_completion_admission(body.as_object().expect("object"), &k2.route, 4_096);

        assert_eq!(k3_estimate.completion_tokens, Some(1_284));
        assert_eq!(k2_estimate.completion_tokens, Some(1_284));
        assert_eq!(
            k3_estimate.prompt_tokens - k2_estimate.prompt_tokens,
            5 * (16_384 - 4_096)
        );
        assert!(k3_estimate.prompt_tokens >= 5 * 16_384);
        assert!(k3_estimate.prompt_tokens < 90_000);

        let compact_body = json!({
            "model": "kimi-k3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello admission"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}},
                    {"type": "input_image", "image_url": "data:image/png;base64,B"},
                    {"image_url": "https://example.com/short.png"},
                    {"type": null, "image_url": "https://example.com/null-short.png"},
                    {"type": false, "image_url": "https://example.com/coerced-short.png"}
                ]
            }],
            "max_completion_tokens": 321,
            "n": 4
        });
        let compact_estimate = estimate_completion_admission(
            compact_body.as_object().expect("object"),
            &k3.route,
            4_096,
        );
        assert!(
            k3_estimate
                .prompt_tokens
                .abs_diff(compact_estimate.prompt_tokens)
                < 16
        );
    }

    #[test]
    fn missing_completion_max_is_bounded_and_reserved_per_choice() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "ignore_eos": true,
            "allowed_token_ids": [1, 2],
            "n": 3
        })
        .as_object()
        .expect("object")
        .clone();
        let pinned = pinned_test_completion();

        ensure_bounded_completion_generation(&mut body, 4_096);
        let estimate = estimate_completion_admission(&body, &pinned.route, 4_096);

        assert_eq!(body["max_tokens"], 4_096);
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(body["allowed_token_ids"], json!([1, 2]));
        assert_eq!(estimate.completion_tokens, Some(12_288));
    }

    #[test]
    fn null_completion_maxima_are_replaced_by_the_bounded_minimum() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": null,
            "max_tokens": null,
            "min_tokens": 8_192
        })
        .as_object()
        .expect("object")
        .clone();

        validate_completion_request_controls(&body, 8).expect("canonical controls");
        ensure_bounded_completion_generation(&mut body, 4_096);

        assert!(!body.contains_key("max_completion_tokens"));
        assert_eq!(body["max_tokens"], 8_192);
    }

    #[test]
    fn completion_controls_reject_provider_coercible_json_types() {
        for (field, value) in [
            ("max_completion_tokens", json!("20000")),
            ("max_tokens", json!(20_000.0)),
            ("n", json!("4")),
            ("min_tokens", json!("20000")),
            ("stream", json!("true")),
            ("stream", json!(1)),
        ] {
            let body = serde_json::Map::from_iter([(field.to_string(), value)]);
            assert!(matches!(
                validate_completion_request_controls(&body, 8),
                Err(ApiError::BadRequest)
            ));
        }

        let canonical = json!({
            "max_completion_tokens": 20_000,
            "max_tokens": null,
            "n": 4,
            "min_tokens": 0,
            "stream": true
        });
        assert!(
            validate_completion_request_controls(canonical.as_object().expect("object"), 8).is_ok()
        );

        let excessive = json!({"n": 9});
        assert!(matches!(
            validate_completion_request_controls(excessive.as_object().expect("object"), 8),
            Err(ApiError::BadRequest)
        ));
    }

    #[test]
    fn completion_names_bound_derived_tool_attribute_expansion() {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "shared-call",
                "type": "function",
                "function": {"name": "&".repeat(65), "arguments": "{}"}
            }]
        })];
        messages.extend(
            (0..128)
                .map(|_| json!({"role": "tool", "tool_call_id": "shared-call", "content": "ok"})),
        );
        let repeated = json!({"messages": messages, "max_tokens": 64});
        assert!(matches!(
            validate_completion_request_controls(repeated.as_object().expect("object"), 8),
            Err(ApiError::BadRequest)
        ));

        for body in [
            json!({"messages": [{"role": "user", "name": "bad&name", "content": "x"}]}),
            json!({"tools": [{"type": "function", "function": {"name": "bad name"}}]}),
            json!({"tool_choice": {"type": "function", "function": {"name": "bad\"name"}}}),
        ] {
            assert!(matches!(
                validate_completion_request_controls(body.as_object().expect("object"), 8),
                Err(ApiError::BadRequest)
            ));
        }

        let canonical = json!({
            "messages": [{
                "role": "assistant",
                "tool_calls": [{"function": {"name": "open_urls-2"}}]
            }],
            "tools": [{"type": "function", "function": {"name": "open_urls-2"}}],
            "tool_choice": {"type": "function", "function": {"name": "open_urls-2"}}
        });
        assert!(
            validate_completion_request_controls(canonical.as_object().expect("object"), 8).is_ok()
        );
    }

    #[test]
    fn historical_tool_arguments_must_be_bounded_valid_json_strings() {
        for arguments in [
            Value::String("{invalid".to_string()),
            Value::String(format!(
                "\"{}\"",
                "x".repeat(MAX_COMPLETION_TOOL_ARGUMENT_BYTES)
            )),
            json!({"not": "a string"}),
        ] {
            let body = json!({
                "messages": [{
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": "bounded_tool", "arguments": arguments}
                    }]
                }]
            });
            assert!(matches!(
                validate_completion_request_controls(body.as_object().expect("object"), 8),
                Err(ApiError::BadRequest)
            ));
        }
    }

    #[test]
    fn admission_estimate_reserves_forwarded_min_tokens_per_choice() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "min_tokens": 20_000,
            "n": 2
        })
        .as_object()
        .expect("object")
        .clone();
        let pinned = pinned_test_completion();

        ensure_bounded_completion_generation(&mut body, 4_096);
        let estimate = estimate_completion_admission(&body, &pinned.route, 4_096);

        assert_eq!(body["max_tokens"], 20_000);
        assert_eq!(estimate.completion_tokens, Some(40_000));
    }

    #[test]
    fn admission_estimate_counts_schema_keys_numbers_and_structure() {
        let mut properties = serde_json::Map::new();
        for index in 0..256 {
            properties.insert(
                format!("property_{index:04}_{}", "x".repeat(48)),
                json!({"type": "integer", "enum": [index, index + 1, index + 2]}),
            );
        }
        let body = json!({
            "messages": [{"role": "user", "content": "return structured data"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "large_schema",
                    "parameters": {"type": "object", "properties": properties}
                }
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "numeric_schema",
                    "schema": {"type": "integer", "enum": (0..512).collect::<Vec<_>>()}
                }
            },
            "max_tokens": 64
        });
        let baseline = json!({
            "messages": [{"role": "user", "content": "return structured data"}],
            "max_tokens": 64
        });
        let pinned = pinned_test_completion();

        let estimate =
            estimate_completion_admission(body.as_object().expect("object"), &pinned.route, 4_096);
        let baseline_estimate = estimate_completion_admission(
            baseline.as_object().expect("object"),
            &pinned.route,
            4_096,
        );

        assert!(estimate.prompt_tokens > baseline_estimate.prompt_tokens + 1_000);
    }

    #[test]
    fn prompt_bound_is_tokenizer_independent_and_preserves_text_with_image_key() {
        let adversarial_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".repeat(4_000);
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": adversarial_text,
                    "image_url": null
                }]
            }],
            "max_tokens": 64
        });
        let pinned = pinned_test_completion();

        let estimate =
            estimate_completion_admission(body.as_object().expect("object"), &pinned.route, 4_096);

        assert!(estimate.prompt_tokens >= u64::try_from(adversarial_text.len()).unwrap() + 16_384);
    }

    #[test]
    fn prompt_bound_accounts_for_provider_attribute_entity_expansion() {
        let attribute = "&\"".repeat(5_000);
        let body = json!({
            "messages": [{"role": "user", "name": attribute, "content": "hello"}],
            "max_tokens": 64
        });
        let pinned = pinned_test_completion();

        let estimate =
            estimate_completion_admission(body.as_object().expect("object"), &pinned.route, 4_096);

        // Each pair expands from two decoded bytes to eleven entity bytes.
        assert!(estimate.prompt_tokens >= 55_000);
    }

    #[test]
    fn prompt_bound_accounts_for_tool_argument_normalization_and_tags() {
        let array_arguments = json!({"a": vec![0; 10_000]}).to_string();
        let array_body = json!({
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-array",
                    "type": "function",
                    "function": {"name": "array_tool", "arguments": array_arguments}
                }]
            }],
            "max_tokens": 64
        });
        let mut object_arguments = serde_json::Map::new();
        for index in 0..1_000 {
            object_arguments.insert(format!("a{index}"), Value::Null);
        }
        let object_body = json!({
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-object",
                    "type": "function",
                    "function": {
                        "name": "object_tool",
                        "arguments": Value::Object(object_arguments).to_string()
                    }
                }]
            }],
            "max_tokens": 64
        });
        let pinned = pinned_test_completion();

        let array_estimate = estimate_completion_admission(
            array_body.as_object().expect("object"),
            &pinned.route,
            4_096,
        );
        let object_estimate = estimate_completion_admission(
            object_body.as_object().expect("object"),
            &pinned.route,
            4_096,
        );

        assert!(array_estimate.prompt_tokens >= 30_000);
        assert!(object_estimate.prompt_tokens >= 17_000);

        let exponent_arguments = format!(
            r#"{{"x":[{}]}}"#,
            std::iter::repeat_n("1e15", 10_000)
                .collect::<Vec<_>>()
                .join(",")
        );
        let exponent_body = json!({
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-exponent",
                    "type": "function",
                    "function": {"name": "exponent_tool", "arguments": exponent_arguments}
                }]
            }],
            "max_tokens": 64
        });
        validate_completion_request_controls(exponent_body.as_object().expect("object"), 8)
            .expect("bounded valid arguments");
        let exponent_estimate = estimate_completion_admission(
            exponent_body.as_object().expect("object"),
            &pinned.route,
            4_096,
        );
        assert!(exponent_estimate.prompt_tokens >= 100_134);
    }

    #[test]
    fn prompt_bound_accounts_for_multiline_tool_schema_formatting() {
        let description = format!("{}x", "x\n".repeat(10_000));
        let body = json!({
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "multiline_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string", "description": description}
                        }
                    }
                }
            }],
            "max_tokens": 64
        });
        let pinned = pinned_auto_fallback_completion();

        let estimate =
            estimate_completion_admission(body.as_object().expect("object"), &pinned.route, 4_096);

        assert!(estimate.prompt_tokens >= 40_044);
    }

    #[test]
    fn message_tool_schema_named_image_url_is_counted_not_sanitized() {
        let large_description = "schema detail ".repeat(2_000);
        let body = json!({
            "messages": [{
                "role": "developer",
                "content": "use the provided schema",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "schema_tool",
                        "parameters": {
                            "type": "image_url",
                            "description": large_description
                        }
                    }
                }]
            }],
            "max_tokens": 64
        });
        let baseline = json!({
            "messages": [{"role": "developer", "content": "use the provided schema"}],
            "max_tokens": 64
        });
        let pinned = pinned_test_completion();

        let estimate =
            estimate_completion_admission(body.as_object().expect("object"), &pinned.route, 4_096);
        let baseline_estimate = estimate_completion_admission(
            baseline.as_object().expect("object"),
            &pinned.route,
            4_096,
        );

        assert!(estimate.prompt_tokens > baseline_estimate.prompt_tokens + 2_000);
    }

    #[test]
    fn unsupported_media_is_rejected_in_explicit_shorthand_and_cached_shapes() {
        for part in [
            json!({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}),
            json!({"video_url": "https://example.com/a.mp4"}),
            json!({"type": null, "video_url": "https://example.com/a.mp4"}),
            json!({"type": "garbage", "video_url": "https://example.com/a.mp4"}),
            json!({"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"}),
            json!({"type": "image_pil", "image_pil": null, "uuid": "cached-image"}),
            json!({"image_embeds": null, "uuid": "cached-image"}),
            json!({"type": "prompt_embeds", "prompt_embeds": null}),
        ] {
            let body = json!({
                "messages": [{"role": "user", "content": [part]}],
                "max_tokens": 64
            });
            assert!(completion_contains_unsupported_media(
                body.as_object().expect("object")
            ));
        }

        let image = json!({
            "messages": [{
                "role": "user",
                "content": [{"image_url": "https://example.com/a.png"}]
            }]
        });
        assert!(!completion_contains_unsupported_media(
            image.as_object().expect("object")
        ));
    }

    #[test]
    fn usage_settlement_uses_one_stable_id_per_execution() {
        let pinned = pinned_test_completion();
        let route = pinned.route.identity();
        let first_execution = pinned.begin_execution();
        let first_attempt = first_execution.begin_attempt(route.clone());
        let recovered_attempt = first_execution.begin_attempt(route.clone());
        let next_attempt = pinned.begin_execution().begin_attempt(route);
        let billing = || BillingContext::new(AuthMethod::Jwt, "auto-powerful".to_string());

        let first = UsageSettlement::new(billing(), &first_attempt);
        let recovered = UsageSettlement::new(billing(), &recovered_attempt);
        let next = UsageSettlement::new(billing(), &next_attempt);

        assert_eq!(first.event_id, recovered.event_id);
        assert_eq!(first.event_id, first_execution.execution_id.as_uuid());
        assert_ne!(first.attempt_id, recovered.attempt_id);
        assert_ne!(first.event_id, next.event_id);
    }

    #[tokio::test]
    async fn attempt_guard_holds_admission_until_authoritative_terminal() {
        let policy =
            crate::inference::admission::AdmissionPolicy::with_deployment_in_flight_for_test(
                &crate::provider_registry::PROVIDER_REGISTRY,
                1,
            )
            .expect("one-slot policy");
        let controller = crate::inference::admission::AdmissionController::new(
            &crate::provider_registry::PROVIDER_REGISTRY,
            policy,
        )
        .expect("controller");
        let pinned = pinned_test_completion();
        let route_key = pinned.route.identity().route_key();
        let permit = controller
            .acquire_turn(
                &route_key,
                Uuid::nil(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(10, Some(5)),
                None,
            )
            .await
            .expect("first permit");
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let mut guard = AttemptObservationGuard::new_admitted(
            attempt.clone(),
            Arc::new(ProviderRouter::default()),
            AttemptStage::ResponseBody,
            permit,
            None,
        );

        let rejection = controller
            .acquire_turn(
                &route_key,
                Uuid::from_u128(1),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .expect_err("response headers must not release the permit");
        assert_eq!(
            rejection.kind,
            crate::inference::admission::AdmissionRejectionKind::DeploymentBusy
        );

        let terminal = AttemptTerminal::Completed {
            attempt,
            evidence: CompletionEvidence::NonStreamingResponse,
        };
        let usage = CompletionUsage {
            prompt_tokens: 8,
            prompt_tokens_observed: true,
            completion_tokens: 4,
            completion_tokens_observed: true,
            cached_prompt_tokens: Some(2),
        };
        guard.record_terminal_with_usage(&terminal, Some(&usage));

        let next = controller
            .acquire_turn(
                &route_key,
                Uuid::from_u128(1),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .expect("terminal releases the deployment slot");
        next.settle(None, TerminalDisposition::ProvenPreAcceptance);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn terminal_health_is_visible_before_a_queued_waiter_is_released() {
        let policy =
            crate::inference::admission::AdmissionPolicy::with_deployment_in_flight_for_test(
                &PROVIDER_REGISTRY,
                1,
            )
            .expect("one-slot policy");
        let controller =
            crate::inference::admission::AdmissionController::new(&PROVIDER_REGISTRY, policy)
                .expect("controller");
        let provider_router = Arc::new(ProviderRouter::default());
        let pinned = pinned_test_completion();
        let route_key = pinned.route.identity().route_key();
        let permit = controller
            .acquire_turn(
                &route_key,
                Uuid::nil(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .expect("first permit");
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let mut guard = AttemptObservationGuard::new_admitted(
            attempt.clone(),
            provider_router.clone(),
            AttemptStage::ResponseBody,
            permit,
            None,
        );

        let waiter_controller = controller.clone();
        let waiter_router = provider_router.clone();
        let waiter_route = route_key.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move {
            let _ = started_tx.send(());
            let permit = waiter_controller
                .acquire_turn(
                    &waiter_route,
                    Uuid::from_u128(1),
                    WorkloadClass::Interactive,
                    AdmissionEstimate::new(1, Some(1)),
                    None,
                )
                .await
                .expect("queued waiter should acquire after release");
            let circuit_is_open = matches!(
                waiter_router.try_claim_probe(&waiter_route),
                ProbeClaimResult::Rejected {
                    reason: crate::inference::health::ProbeRejectionReason::CircuitOpen,
                    ..
                }
            );
            permit.settle(None, TerminalDisposition::ProvenPreAcceptance);
            circuit_is_open
        });
        started_rx.await.expect("waiter started");
        tokio::task::yield_now().await;

        let terminal = AttemptTerminal::Failed {
            attempt,
            failure: AttemptFailure::new(
                AttemptFailureKind::CapacityRejected,
                AttemptStage::ResponseBody,
                ReplaySafety::NotProvenPreAcceptance,
            )
            .with_upstream_response(503, Some(Duration::from_secs(30)), None),
        };
        guard.record_terminal(&terminal);

        assert!(
            timeout(Duration::from_secs(1), waiter)
                .await
                .expect("waiter should wake")
                .expect("waiter task"),
            "the circuit-opening terminal must be visible before admission wakes the waiter"
        );
    }

    #[test]
    fn chat_body_uses_the_pinned_auto_fallback_public_model() {
        let pinned = pinned_auto_fallback_completion();
        let mut body = json!({
            "model": crate::model_config::AUTO_POWERFUL_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}]
        });

        pin_chat_request_model(&mut body, &pinned);

        assert_eq!(body["model"], crate::model_config::POWERFUL_MODEL_ID);
        assert_eq!(
            pinned.intent().public_model_id,
            crate::model_config::KIMI_K3_MODEL_ID
        );
        assert_eq!(
            pinned.public_model_id(),
            crate::model_config::POWERFUL_MODEL_ID
        );
        assert_eq!(
            pinned.intent().requested_model_id,
            crate::model_config::AUTO_POWERFUL_MODEL_ID
        );
    }

    async fn record_terminal_once(
        provider_router: &ProviderRouter,
        terminal: &AttemptTerminal,
    ) -> crate::inference::health::ShadowRouteSnapshot {
        let route_key = terminal.attempt().route.route_key();
        let (terminal_tx, mut terminal_rx) = mpsc::channel(1);
        send_attempt_terminal(provider_router, &terminal_tx, terminal.clone()).await;
        drop(terminal_tx);
        assert!(matches!(
            terminal_rx.recv().await,
            Some(CompletionChunk::Terminal(_))
        ));
        assert!(terminal_rx.recv().await.is_none());
        provider_router
            .shadow_health_snapshot(&route_key)
            .expect("registered terminal route")
    }

    #[tokio::test]
    async fn cancelling_while_awaiting_provider_result_records_one_neutral_terminal() {
        let provider_router = Arc::new(ProviderRouter::default());
        let pinned = pinned_test_completion();
        let route_key = pinned.route.identity().route_key();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let terminal_guard = AttemptObservationGuard::new(
            attempt,
            Arc::clone(&provider_router),
            AttemptStage::AwaitingResponse,
        );

        let result = timeout(
            Duration::from_millis(5),
            await_attempt_result(terminal_guard, futures::future::pending::<()>()),
        )
        .await;

        assert!(result.is_err());
        assert_eq!(provider_router.shadow_observation_count(), 1);
        assert_eq!(
            provider_router
                .shadow_health_snapshot(&route_key)
                .expect("registered K3 route")
                .effective,
            crate::inference::health::ShadowDisposition::Healthy
        );
    }

    #[tokio::test]
    async fn taking_started_response_does_not_disarm_drop_observation() {
        let (trace, server) = call_mock_provider(static_sse_app("data: [DONE]\n\n")).await;
        let response = trace.result.expect("mock provider response start");
        let provider_router = Arc::new(ProviderRouter::default());
        let pinned = pinned_test_completion();
        let route_key = pinned.route.identity().route_key();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let terminal_guard = AttemptObservationGuard::new(
            attempt.clone(),
            Arc::clone(&provider_router),
            AttemptStage::ResponseBody,
        );
        let usage_settlement = UsageSettlement::new(
            BillingContext::new(AuthMethod::Jwt, "kimi-k3".to_string()),
            &attempt,
        );
        let mut started = StartedCompletion {
            response: Some(response),
            successful_provider: ProviderId::Tinfoil.as_str().to_string(),
            attempt,
            terminal_guard: Some(terminal_guard),
            response_model_id: "kimi-k3".to_string(),
            public_model_id: "kimi-k3".to_string(),
            is_streaming: false,
            non_streaming_body_limit: None,
            usage_settlement: Some(usage_settlement),
        };

        drop(started.response.take());
        drop(started);
        server.abort();

        assert_eq!(provider_router.shadow_observation_count(), 1);
        assert_eq!(
            provider_router
                .shadow_health_snapshot(&route_key)
                .expect("registered K3 route")
                .effective,
            crate::inference::health::ShadowDisposition::Healthy
        );
    }

    #[tokio::test]
    async fn mock_provider_streams_produce_one_sanitized_terminal_result() {
        enum ExpectedTerminal {
            Completed(CompletionEvidence),
            Failed(AttemptFailureKind),
        }

        let cases = [
            (
                "provider done",
                concat!(
                    "data: {\"model\":\"upstream-k3\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
                    "data: [DONE]\n\n"
                ),
                ExpectedTerminal::Completed(CompletionEvidence::ProviderDone),
                1,
            ),
            (
                "finish then eof",
                "data: {\"model\":\"upstream-k3\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                ExpectedTerminal::Completed(CompletionEvidence::FinishSignalThenEof),
                1,
            ),
            (
                "bare eof",
                "data: {\"model\":\"upstream-k3\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                ExpectedTerminal::Failed(AttemptFailureKind::UnexpectedEof),
                1,
            ),
            (
                "truncated trailing frame",
                concat!(
                    "data: {\"model\":\"upstream-k3\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: {\"choices\":["
                ),
                ExpectedTerminal::Failed(AttemptFailureKind::UnexpectedEof),
                1,
            ),
            (
                "invalid json",
                "data: not-json\n\n",
                ExpectedTerminal::Failed(AttemptFailureKind::InvalidResponse),
                0,
            ),
            (
                "provider error then done",
                concat!(
                    "data: {\"model\":\"upstream-k3\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
                    "data: {\"error\":{\"message\":\"private upstream detail\",\"code\":\"capacity_error-17\"}}\n\n",
                    "data: [DONE]\n\n"
                ),
                ExpectedTerminal::Failed(AttemptFailureKind::UpstreamStreamError),
                1,
            ),
        ];

        for (name, body, expected, expected_chunks) in cases {
            let (result, chunks) = process_mock_sse(body).await;
            assert_eq!(chunks.len(), expected_chunks, "{name}");
            assert!(
                chunks.iter().all(|chunk| chunk.get("error").is_none()),
                "{name}"
            );
            assert!(
                chunks.iter().all(|chunk| chunk["model"] == "kimi-k3"),
                "{name}"
            );

            let provider_router = ProviderRouter::default();
            let health = record_terminal_once(&provider_router, &result.terminal).await;
            match &result.terminal {
                AttemptTerminal::Completed { .. } => assert_eq!(
                    health.effective,
                    crate::inference::health::ShadowDisposition::Healthy,
                    "{name}"
                ),
                AttemptTerminal::Failed { .. } => assert_eq!(
                    health.route_health,
                    crate::inference::health::ShadowDisposition::Watch {
                        consecutive_failures: 1
                    },
                    "{name}"
                ),
            }

            match (expected, result.terminal) {
                (
                    ExpectedTerminal::Completed(expected_evidence),
                    AttemptTerminal::Completed { evidence, .. },
                ) => assert_eq!(evidence, expected_evidence, "{name}"),
                (
                    ExpectedTerminal::Failed(expected_kind),
                    AttemptTerminal::Failed { failure, .. },
                ) => {
                    assert_eq!(failure.kind, expected_kind, "{name}");
                    assert_eq!(
                        failure.replay_safety,
                        ReplaySafety::NotProvenPreAcceptance,
                        "{name}"
                    );
                    if expected_kind == AttemptFailureKind::UpstreamStreamError {
                        assert_eq!(failure.upstream_code.as_deref(), Some("capacity_error-17"));
                    }
                }
                (_, terminal) => panic!("{name}: unexpected terminal {terminal:?}"),
            }
        }
    }

    #[tokio::test]
    async fn mock_provider_429_preserves_bounded_metadata_without_body_leak() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::TOO_MANY_REQUESTS)
                    .header(header::RETRY_AFTER, "17")
                    .header("x-request-id", "req-429_a.b:c")
                    .body(axum::body::Body::from("private upstream capacity detail"))
                    .expect("mock rate-limit response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        server.abort();
        assert!(trace.prior_failures.is_empty());
        let error = match trace.result {
            Err(error) => error,
            Ok(_) => panic!("mock 429 unexpectedly succeeded"),
        };
        assert!(!error
            .to_string()
            .contains("private upstream capacity detail"));

        let failure = attempt_failure_from_provider_error(&error);
        assert_eq!(failure.kind, AttemptFailureKind::CapacityRejected);
        assert_eq!(failure.stage, AttemptStage::AwaitingResponse);
        assert_eq!(failure.replay_safety, ReplaySafety::ProvenPreAcceptance);
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.retry_after, Some(Duration::from_secs(17)));
        assert_eq!(
            failure.upstream_request_id.as_deref(),
            Some("req-429_a.b:c")
        );

        let provider_router = ProviderRouter::default();
        let pinned = pinned_test_completion();
        let route_key = pinned.route.identity().route_key();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let _ = failed_completion_execution(
            &provider_router,
            attempt,
            failure,
            ApiError::InternalServerError,
        );
        assert!(matches!(
            provider_router
                .shadow_health_snapshot(&route_key)
                .expect("registered K3 route")
                .effective,
            crate::inference::health::ShadowDisposition::WouldOpen { .. }
        ));
    }

    #[tokio::test]
    async fn bounded_provider_response_body_maps_stream_errors_without_content() {
        let body_stream = futures::stream::iter([
            Ok::<_, String>(bytes::Bytes::from_static(b"abcd")),
            Err("sensitive transport detail".to_string()),
        ]);

        assert_eq!(
            collect_bounded_provider_response_body(body_stream, 6).await,
            Err(BoundedProviderResponseBodyError::Read)
        );
    }

    #[tokio::test]
    async fn glm_capacity_failure_switches_only_the_next_request_to_the_mock_alternate() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_bodies = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            let bodies = Arc::clone(&tinfoil_bodies);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let count = Arc::clone(&count);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        bodies.lock().expect("Tinfoil body lock").push(body);
                        axum::http::Response::builder()
                            .status(StatusCode::TOO_MANY_REQUESTS)
                            .header(header::RETRY_AFTER, "60")
                            .body(axum::body::Body::empty())
                            .expect("mock Tinfoil 429")
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_bodies = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            let bodies = Arc::clone(&continuum_bodies);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let count = Arc::clone(&count);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        bodies.lock().expect("Continuum body lock").push(body);
                        axum::http::Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "application/json")
                            .body(axum::body::Body::from(
                                r#"{"model":"glm-5.2","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
                            ))
                            .expect("mock Continuum success")
                    }
                }),
            )
        };

        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let provider_client =
            ProviderClient::for_test(tinfoil_url.clone()).expect("test provider client");
        let proxy_router = crate::proxy_config::ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();
        let intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::GLM_5_2_MODEL_ID,
            crate::model_config::GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );

        let first_baseline = provider_router.shadow_completion_plan(&proxy_router, &intent, None);
        let first_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &intent,
            None,
            first_baseline,
        )
        .expect("initial GLM route");
        assert_eq!(first_route.provider, ProviderId::Tinfoil);
        let first_trace = try_provider(
            &provider_client,
            &first_route.proxy,
            json!({
                "model": first_route.provider_model_id,
                "messages": [{"role": "user", "content": "one"}]
            })
            .to_string(),
            &HeaderMap::new(),
        )
        .await;
        assert!(first_trace.prior_failures.is_empty());
        let first_error = match first_trace.result {
            Err(error) => error,
            Ok(_) => panic!("mock Tinfoil unexpectedly succeeded"),
        };
        let first_failure = attempt_failure_from_provider_error(&first_error);
        let first_attempt = intent
            .begin_execution()
            .begin_attempt(first_route.identity());
        let surfaced = public_completion_error(&first_error, &first_failure);
        let _ =
            failed_completion_execution(&provider_router, first_attempt, first_failure, surfaced);

        // The triggering logical request made exactly one provider call and did
        // not replay or fail over to Continuum.
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 1);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);

        let second_intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::GLM_5_2_MODEL_ID,
            crate::model_config::GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        assert_ne!(second_intent.request_id, intent.request_id);
        let second_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &second_intent, None);
        let second_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &second_intent,
            None,
            second_baseline,
        )
        .expect("next request uses Continuum GLM");
        assert_eq!(second_route.provider, ProviderId::Continuum);
        assert_eq!(second_route.provider_model_id, "glm-5.2");
        assert_eq!(
            second_route.response_model_id,
            crate::model_config::GLM_5_2_MODEL_ID
        );
        let second_trace = try_provider(
            &provider_client,
            &second_route.proxy,
            json!({
                "model": second_route.provider_model_id,
                "messages": [{"role": "user", "content": "two"}]
            })
            .to_string(),
            &HeaderMap::new(),
        )
        .await;
        assert!(second_trace.prior_failures.is_empty());
        let response = second_trace.result.expect("mock Continuum succeeds");
        let second_attempt = second_intent
            .begin_execution()
            .begin_attempt(second_route.identity());
        let canonical_response = read_non_streaming_completion_response(
            response,
            &second_route.response_model_id,
            &second_attempt,
            None,
            Duration::from_secs(1),
        )
        .await
        .expect("canonical Continuum response");
        assert_eq!(
            canonical_response["model"],
            crate::model_config::GLM_5_2_MODEL_ID
        );
        assert_eq!(
            second_attempt.route.public_model_id,
            crate::model_config::GLM_5_2_MODEL_ID
        );
        assert_eq!(second_attempt.route.provider, ProviderId::Continuum);

        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 1);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            tinfoil_bodies.lock().expect("Tinfoil bodies")[0]["model"],
            crate::model_config::GLM_5_2_MODEL_ID
        );
        assert_eq!(
            continuum_bodies.lock().expect("Continuum bodies")[0]["model"],
            "glm-5.2"
        );

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn auto_powerful_capacity_failure_switches_only_a_distinct_request_to_k2_6() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_bodies = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            let bodies = Arc::clone(&tinfoil_bodies);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let count = Arc::clone(&count);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        bodies.lock().expect("Tinfoil body lock").push(body);
                        axum::http::Response::builder()
                            .status(StatusCode::TOO_MANY_REQUESTS)
                            .header(header::RETRY_AFTER, "60")
                            .body(axum::body::Body::empty())
                            .expect("mock K3 429")
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_bodies = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            let bodies = Arc::clone(&continuum_bodies);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let count = Arc::clone(&count);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        bodies.lock().expect("Continuum body lock").push(body);
                        axum::http::Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "application/json")
                            .body(axum::body::Body::from(
                                r#"{"model":"kimi-k2.6","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
                            ))
                            .expect("mock K2.6 success")
                    }
                }),
            )
        };

        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let provider_client =
            ProviderClient::for_test(tinfoil_url.clone()).expect("test provider client");
        let proxy_router = ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();
        let first_intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::AUTO_POWERFUL_MODEL_ID,
            crate::model_config::KIMI_K3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let first_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &first_intent, None);
        let first_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &first_intent,
            None,
            first_baseline,
        )
        .expect("healthy Auto Powerful prefers K3");
        assert_eq!(first_route.provider, ProviderId::Tinfoil);
        assert_eq!(
            first_route.public_model_id,
            crate::model_config::KIMI_K3_MODEL_ID
        );
        assert_eq!(
            first_route.model_selection_source,
            crate::provider_routing::ModelSelectionSource::AutoPrimary
        );

        let user_content = json!("hello");
        let first_trace = try_provider(
            &provider_client,
            &first_route.proxy,
            json!({
                "model": first_route.provider_model_id,
                "messages": [{"role": "user", "content": user_content.clone()}]
            })
            .to_string(),
            &HeaderMap::new(),
        )
        .await;
        assert!(first_trace.prior_failures.is_empty());
        let first_error = match first_trace.result {
            Err(error) => error,
            Ok(_) => panic!("mock K3 unexpectedly succeeded"),
        };
        let first_failure = attempt_failure_from_provider_error(&first_error);
        let first_attempt = first_intent
            .begin_execution()
            .begin_attempt(first_route.identity());
        let surfaced = public_completion_error(&first_error, &first_failure);
        let _ =
            failed_completion_execution(&provider_router, first_attempt, first_failure, surfaced);

        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 1);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);

        let second_intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::AUTO_POWERFUL_MODEL_ID,
            crate::model_config::KIMI_K3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        assert_ne!(second_intent.request_id, first_intent.request_id);
        let second_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &second_intent, None);
        let second_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &second_intent,
            None,
            second_baseline,
        )
        .expect("later Auto Powerful request uses compatible K2.6");
        assert_eq!(second_route.provider, ProviderId::Continuum);
        assert_eq!(
            second_route.public_model_id,
            crate::model_config::POWERFUL_MODEL_ID
        );
        assert_eq!(second_route.provider_model_id, "kimi-k2.6");
        assert_eq!(
            second_route.model_selection_source,
            crate::provider_routing::ModelSelectionSource::AutoFallback
        );
        let second_trace = try_provider(
            &provider_client,
            &second_route.proxy,
            json!({
                "model": second_route.provider_model_id,
                "messages": [{"role": "user", "content": user_content.clone()}]
            })
            .to_string(),
            &HeaderMap::new(),
        )
        .await;
        assert!(second_trace.prior_failures.is_empty());
        let response = second_trace.result.expect("mock K2.6 succeeds");
        let second_attempt = second_intent
            .begin_execution()
            .begin_attempt(second_route.identity());
        let canonical_response = read_non_streaming_completion_response(
            response,
            &second_route.response_model_id,
            &second_attempt,
            None,
            Duration::from_secs(1),
        )
        .await
        .expect("canonical K2.6 response");

        assert_eq!(
            canonical_response["model"],
            crate::model_config::POWERFUL_MODEL_ID
        );
        assert_eq!(
            second_intent.requested_model_id,
            crate::model_config::AUTO_POWERFUL_MODEL_ID
        );
        assert_eq!(
            second_attempt.route.public_model_id,
            crate::model_config::POWERFUL_MODEL_ID
        );
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 1);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            tinfoil_bodies.lock().expect("Tinfoil bodies")[0]["model"],
            crate::model_config::KIMI_K3_MODEL_ID
        );
        assert_eq!(
            continuum_bodies.lock().expect("Continuum bodies")[0]["model"],
            "kimi-k2.6"
        );
        assert_eq!(
            continuum_bodies.lock().expect("Continuum bodies")[0]["messages"][0]["content"],
            user_content
        );

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn glm_tool_turns_keep_the_original_mock_provider_after_health_opens() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(_body): Json<Value>| {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Json(json!({
                            "model": "glm-5-2",
                            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                        }))
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(_body): Json<Value>| {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Json(json!({
                            "model": "glm-5.2",
                            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                        }))
                    }
                }),
            )
        };
        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let provider_client =
            ProviderClient::for_test(tinfoil_url.clone()).expect("test provider client");
        let proxy_router = ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();
        let intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::GLM_5_2_MODEL_ID,
            crate::model_config::GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let baseline = provider_router.shadow_completion_plan(&proxy_router, &intent, None);
        let route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &intent,
            None,
            baseline,
        )
        .expect("initial GLM route");
        let pinned = PinnedCompletionRequest {
            intent,
            route,
            provider_preference: None,
            finalized_route: Arc::new(OnceLock::new()),
        };

        let send_pinned_turn = |content: &'static str| {
            let provider_client = &provider_client;
            let pinned = &pinned;
            async move {
                let trace = try_provider(
                    provider_client,
                    &pinned.route.proxy,
                    json!({
                        "model": pinned.route.provider_model_id,
                        "messages": [{"role": "user", "content": content}]
                    })
                    .to_string(),
                    &HeaderMap::new(),
                )
                .await;
                assert!(trace.prior_failures.is_empty());
                let response = trace.result.expect("pinned mock turn succeeds");
                let attempt = pinned
                    .begin_execution()
                    .begin_attempt(pinned.route.identity());
                let response = read_non_streaming_completion_response(
                    response,
                    &pinned.route.response_model_id,
                    &attempt,
                    None,
                    Duration::from_secs(1),
                )
                .await
                .expect("canonical pinned response");
                assert_eq!(response["model"], crate::model_config::GLM_5_2_MODEL_ID);
                attempt
            }
        };

        let first_attempt = send_pinned_turn("first tool turn").await;
        let external_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(60)),
            upstream_request_id: None,
        });
        let external_failure = attempt_failure_from_provider_error(&external_error);
        let external_intent = InferenceIntent::new(
            Uuid::from_u128(1),
            crate::model_config::GLM_5_2_MODEL_ID,
            crate::model_config::GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let _ = failed_completion_execution(
            &provider_router,
            external_intent
                .begin_execution()
                .begin_attempt(pinned.route.identity()),
            external_failure.clone(),
            public_completion_error(&external_error, &external_failure),
        );

        let second_attempt = send_pinned_turn("second tool turn").await;
        assert_eq!(first_attempt.request_id, second_attempt.request_id);
        assert_ne!(first_attempt.execution_id, second_attempt.execution_id);
        assert_eq!(first_attempt.route, second_attempt.route);
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 2);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);

        let next_intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::GLM_5_2_MODEL_ID,
            crate::model_config::GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let next_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &next_intent, None);
        let next_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &next_intent,
            None,
            next_baseline,
        )
        .expect("next logical request switches providers");
        assert_eq!(next_route.provider, ProviderId::Continuum);
        assert_eq!(next_route.provider_model_id, "glm-5.2");

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn auto_powerful_tool_turns_keep_the_pinned_k3_after_health_opens() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_bodies = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            let bodies = Arc::clone(&tinfoil_bodies);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let count = Arc::clone(&count);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        bodies.lock().expect("Tinfoil body lock").push(body);
                        Json(json!({
                            "model": "kimi-k3",
                            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                        }))
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move |Json(_body): Json<Value>| {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Json(json!({
                            "model": "kimi-k2.6",
                            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                        }))
                    }
                }),
            )
        };
        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let provider_client =
            ProviderClient::for_test(tinfoil_url.clone()).expect("test provider client");
        let proxy_router = ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();
        let intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::AUTO_POWERFUL_MODEL_ID,
            crate::model_config::KIMI_K3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let baseline = provider_router.shadow_completion_plan(&proxy_router, &intent, None);
        let route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &intent,
            None,
            baseline,
        )
        .expect("initial Auto Powerful K3 route");
        assert_eq!(
            route.model_selection_source,
            crate::provider_routing::ModelSelectionSource::AutoPrimary
        );
        let pinned = PinnedCompletionRequest {
            intent,
            route,
            provider_preference: None,
            finalized_route: Arc::new(OnceLock::new()),
        };

        let send_pinned_turn = |content: &'static str| {
            let provider_client = &provider_client;
            let pinned = &pinned;
            async move {
                let trace = try_provider(
                    provider_client,
                    &pinned.route.proxy,
                    json!({
                        "model": pinned.route.provider_model_id,
                        "messages": [{"role": "user", "content": content}]
                    })
                    .to_string(),
                    &HeaderMap::new(),
                )
                .await;
                assert!(trace.prior_failures.is_empty());
                let response = trace.result.expect("pinned K3 turn succeeds");
                let attempt = pinned
                    .begin_execution()
                    .begin_attempt(pinned.route.identity());
                let response = read_non_streaming_completion_response(
                    response,
                    &pinned.route.response_model_id,
                    &attempt,
                    None,
                    Duration::from_secs(1),
                )
                .await
                .expect("canonical pinned K3 response");
                assert_eq!(response["model"], crate::model_config::KIMI_K3_MODEL_ID);
                attempt
            }
        };

        let first_attempt = send_pinned_turn("first tool turn").await;
        let external_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(60)),
            upstream_request_id: None,
        });
        let external_failure = attempt_failure_from_provider_error(&external_error);
        let external_intent = InferenceIntent::new(
            Uuid::from_u128(7),
            crate::model_config::KIMI_K3_MODEL_ID,
            crate::model_config::KIMI_K3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let _ = failed_completion_execution(
            &provider_router,
            external_intent
                .begin_execution()
                .begin_attempt(pinned.route.identity()),
            external_failure.clone(),
            public_completion_error(&external_error, &external_failure),
        );

        let second_attempt = send_pinned_turn("second tool turn").await;
        assert_eq!(first_attempt.request_id, second_attempt.request_id);
        assert_ne!(first_attempt.execution_id, second_attempt.execution_id);
        assert_eq!(first_attempt.route, second_attempt.route);
        assert_eq!(
            second_attempt.route.public_model_id,
            crate::model_config::KIMI_K3_MODEL_ID
        );
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 2);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);
        assert!(tinfoil_bodies
            .lock()
            .expect("Tinfoil bodies")
            .iter()
            .all(|body| body["model"] == crate::model_config::KIMI_K3_MODEL_ID));

        let next_intent = InferenceIntent::new(
            Uuid::nil(),
            crate::model_config::AUTO_POWERFUL_MODEL_ID,
            crate::model_config::KIMI_K3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );
        let next_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &next_intent, None);
        let next_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &next_intent,
            None,
            next_baseline,
        )
        .expect("new Auto request switches to K2.6");
        assert_eq!(next_route.provider, ProviderId::Continuum);
        assert_eq!(next_route.provider_model_id, "kimi-k2.6");
        assert_eq!(
            next_route.public_model_id,
            crate::model_config::POWERFUL_MODEL_ID
        );
        assert_eq!(
            next_route.model_selection_source,
            crate::provider_routing::ModelSelectionSource::AutoFallback
        );

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn mock_provider_429_does_not_wait_for_pending_error_body() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let pending = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::TOO_MANY_REQUESTS)
                    .header(header::RETRY_AFTER, "23")
                    .body(axum::body::Body::from_stream(pending))
                    .expect("mock pending rate-limit response")
            }),
        );

        let (trace, server) = timeout(Duration::from_secs(1), call_mock_provider(app))
            .await
            .expect("429 classification must finish after headers without reading the body");
        server.abort();

        let error = match trace.result {
            Err(error) => error,
            Ok(_) => panic!("mock 429 unexpectedly succeeded"),
        };
        let failure = attempt_failure_from_provider_error(&error);
        assert_eq!(failure.kind, AttemptFailureKind::CapacityRejected);
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.retry_after, Some(Duration::from_secs(23)));
    }

    #[tokio::test]
    async fn mock_provider_capacity_matrix_normalizes_503_and_synthetic_529() {
        for status in [503u16, 529u16] {
            let app = Router::new().route(
                "/v1/chat/completions",
                post(move || async move {
                    axum::http::Response::builder()
                        .status(status)
                        .header(header::RETRY_AFTER, "11")
                        .body(axum::body::Body::from(
                            r#"{"error":{"message":"private capacity detail"}}"#,
                        ))
                        .expect("mock capacity response")
                }),
            );
            let (trace, server) = call_mock_provider(app).await;
            server.abort();
            let error = match trace.result {
                Err(error) => error,
                Ok(_) => panic!("capacity response unexpectedly succeeded"),
            };
            assert!(!error.to_string().contains("private capacity detail"));
            let failure = attempt_failure_from_provider_error(&error);
            assert_eq!(failure.kind, AttemptFailureKind::CapacityRejected);
            assert_eq!(failure.replay_safety, ReplaySafety::ProvenPreAcceptance);
            assert_eq!(failure.status, Some(status));
            assert_eq!(failure.retry_after, Some(Duration::from_secs(11)));
            assert!(matches!(
                public_completion_error(&error, &failure),
                ApiError::InferenceCapacity {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    client_replay_safe: false,
                    ..
                }
            ));

            let provider_router = ProviderRouter::default();
            let pinned = pinned_test_completion();
            let route_key = pinned.route.identity().route_key();
            let attempt = pinned
                .begin_execution()
                .begin_attempt(pinned.route.identity());
            let _ = failed_completion_execution(
                &provider_router,
                attempt,
                failure,
                ApiError::InternalServerError,
            );
            assert!(matches!(
                provider_router
                    .shadow_health_snapshot(&route_key)
                    .expect("registered K3 route")
                    .deployment_capacity,
                crate::inference::health::ShadowDisposition::WouldOpen { .. }
            ));
        }
    }

    #[tokio::test]
    async fn pre_persistence_capacity_failure_marks_client_replay_safe() {
        let provider_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(7)),
            upstream_request_id: None,
        });
        let failure = attempt_failure_from_provider_error(&provider_error);
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let provider_router = ProviderRouter::default();
        let response = failed_completion_execution(
            &provider_router,
            attempt,
            failure.clone(),
            public_completion_error(&provider_error, &failure),
        )
        .into_pre_persistence_api_error()
        .into_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[crate::CLIENT_REPLAY_HEADER], "safe");
        assert_eq!(response.headers()[crate::ERROR_CONTRACT_HEADER], "1");
        assert_eq!(
            response.headers()[crate::ERROR_CODE_HEADER],
            crate::INFERENCE_CAPACITY_ERROR_CODE
        );
        assert_eq!(response.headers()[header::RETRY_AFTER], "7");
    }

    #[tokio::test]
    async fn prepared_all_open_glm_route_sends_zero_requests_and_returns_capacity_contract() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        StatusCode::OK
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        StatusCode::OK
                    }
                }),
            )
        };
        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let proxy_router = ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();

        let new_intent = || {
            InferenceIntent::new(
                Uuid::nil(),
                crate::model_config::GLM_5_2_MODEL_ID,
                crate::model_config::GLM_5_2_MODEL_ID,
                ModelPlan::Paid,
                InferenceSurface::Responses,
                WorkloadClass::Interactive,
            )
        };
        let first_intent = new_intent();
        let first_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &first_intent, None);
        let first_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &first_intent,
            None,
            first_baseline,
        )
        .expect("initial Tinfoil GLM route");
        let first_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(40)),
            upstream_request_id: None,
        });
        let first_failure = attempt_failure_from_provider_error(&first_error);
        let _ = failed_completion_execution(
            &provider_router,
            first_intent
                .begin_execution()
                .begin_attempt(first_route.identity()),
            first_failure.clone(),
            public_completion_error(&first_error, &first_failure),
        );

        let second_intent = new_intent();
        let second_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &second_intent, None);
        let second_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &second_intent,
            None,
            second_baseline,
        )
        .expect("Continuum GLM route while Tinfoil is open");
        assert_eq!(second_route.provider, ProviderId::Continuum);
        let second_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 503,
            retry_after: Some(Duration::from_secs(10)),
            upstream_request_id: None,
        });
        let second_failure = attempt_failure_from_provider_error(&second_error);
        let _ = failed_completion_execution(
            &provider_router,
            second_intent
                .begin_execution()
                .begin_attempt(second_route.identity()),
            second_failure.clone(),
            public_completion_error(&second_error, &second_failure),
        );

        let third_intent = new_intent();
        let third_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &third_intent, None);
        let error = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &third_intent,
            None,
            third_baseline,
        )
        .expect_err("both GLM routes are open");
        let response = error.into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[crate::CLIENT_REPLAY_HEADER], "safe");
        assert_eq!(response.headers()[crate::ERROR_CONTRACT_HEADER], "1");
        assert_eq!(
            response.headers()[crate::ERROR_CODE_HEADER],
            crate::INFERENCE_CAPACITY_ERROR_CODE
        );
        assert_eq!(response.headers()[header::RETRY_AFTER], "30");
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 0);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn prepared_all_open_auto_powerful_models_send_zero_requests_and_return_capacity() {
        let tinfoil_count = Arc::new(AtomicUsize::new(0));
        let tinfoil_app = {
            let count = Arc::clone(&tinfoil_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        StatusCode::OK
                    }
                }),
            )
        };
        let continuum_count = Arc::new(AtomicUsize::new(0));
        let continuum_app = {
            let count = Arc::clone(&continuum_count);
            Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let count = Arc::clone(&count);
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        StatusCode::OK
                    }
                }),
            )
        };
        let (tinfoil_url, tinfoil_server) = start_mock_provider(tinfoil_app).await;
        let (continuum_url, continuum_server) = start_mock_provider(continuum_app).await;
        let proxy_router = ProxyRouter::new(continuum_url, None, tinfoil_url);
        let provider_router = ProviderRouter::default();

        let new_intent = || {
            InferenceIntent::new(
                Uuid::nil(),
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
                crate::model_config::KIMI_K3_MODEL_ID,
                ModelPlan::Paid,
                InferenceSurface::Responses,
                WorkloadClass::Interactive,
            )
        };
        let first_intent = new_intent();
        let first_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &first_intent, None);
        let first_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &first_intent,
            None,
            first_baseline,
        )
        .expect("initial K3 route");
        let first_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(40)),
            upstream_request_id: None,
        });
        let first_failure = attempt_failure_from_provider_error(&first_error);
        let _ = failed_completion_execution(
            &provider_router,
            first_intent
                .begin_execution()
                .begin_attempt(first_route.identity()),
            first_failure.clone(),
            public_completion_error(&first_error, &first_failure),
        );

        let second_intent = new_intent();
        let second_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &second_intent, None);
        let second_route = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &second_intent,
            None,
            second_baseline,
        )
        .expect("K2.6 route while K3 is open");
        assert_eq!(second_route.provider, ProviderId::Continuum);
        assert_eq!(
            second_route.public_model_id,
            crate::model_config::POWERFUL_MODEL_ID
        );
        let second_error = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 503,
            retry_after: Some(Duration::from_secs(10)),
            upstream_request_id: None,
        });
        let second_failure = attempt_failure_from_provider_error(&second_error);
        let _ = failed_completion_execution(
            &provider_router,
            second_intent
                .begin_execution()
                .begin_attempt(second_route.identity()),
            second_failure.clone(),
            public_completion_error(&second_error, &second_failure),
        );

        let third_intent = new_intent();
        let third_baseline =
            provider_router.shadow_completion_plan(&proxy_router, &third_intent, None);
        let error = select_prepared_completion_route(
            &provider_router,
            &proxy_router,
            &third_intent,
            None,
            third_baseline,
        )
        .expect_err("K3 and K2.6 circuits are open");
        let response = error.into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[crate::CLIENT_REPLAY_HEADER], "safe");
        assert_eq!(response.headers()[crate::ERROR_CONTRACT_HEADER], "1");
        assert_eq!(
            response.headers()[crate::ERROR_CODE_HEADER],
            crate::INFERENCE_CAPACITY_ERROR_CODE
        );
        assert_eq!(response.headers()[header::RETRY_AFTER], "30");
        assert_eq!(tinfoil_count.load(Ordering::SeqCst), 0);
        assert_eq!(continuum_count.load(Ordering::SeqCst), 0);

        tinfoil_server.abort();
        continuum_server.abort();
    }

    #[tokio::test]
    async fn mock_provider_non_streaming_error_payload_is_a_sanitized_failure() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"error":{"message":"private upstream detail","code":"capacity_error-17"}}"#,
                    ))
                    .expect("mock non-streaming error response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = match trace.result {
            Ok(response) => response,
            Err(error) => panic!("mock provider failed before response parsing: {error}"),
        };
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let failure = read_non_streaming_completion_response(
            response,
            "kimi-k3",
            &attempt,
            None,
            Duration::from_secs(1),
        )
        .await
        .expect_err("top-level provider error payload must fail the attempt");
        server.abort();

        assert_eq!(failure.kind, AttemptFailureKind::UpstreamResponseError);
        assert_eq!(failure.stage, AttemptStage::ResponseBody);
        assert_eq!(failure.replay_safety, ReplaySafety::NotProvenPreAcceptance);
        assert_eq!(failure.upstream_code.as_deref(), Some("capacity_error-17"));
        assert!(!format!("{failure:?}").contains("private upstream detail"));

        let provider_router = ProviderRouter::default();
        let terminal = AttemptTerminal::Failed { attempt, failure };
        let health = record_terminal_once(&provider_router, &terminal).await;
        assert_eq!(
            health.route_health,
            crate::inference::health::ShadowDisposition::Watch {
                consecutive_failures: 1
            }
        );
    }

    #[tokio::test]
    async fn bounded_non_streaming_response_is_a_typed_invalid_response() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"model":"upstream-k3","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
                    ))
                    .expect("mock bounded response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = match trace.result {
            Ok(response) => response,
            Err(error) => panic!("mock provider failed before bounded read: {error}"),
        };
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let failure = read_non_streaming_completion_response(
            response,
            "kimi-k3",
            &attempt,
            Some(8),
            Duration::from_secs(1),
        )
        .await
        .expect_err("response over the descriptor cap must fail");
        server.abort();

        assert_eq!(failure.kind, AttemptFailureKind::InvalidResponse);
        assert_eq!(failure.stage, AttemptStage::ResponseBody);
        assert_eq!(failure.replay_safety, ReplaySafety::NotProvenPreAcceptance);
    }

    #[tokio::test]
    async fn terminal_error_event_round_trips_through_maple_encrypted_sse_contract() {
        use axum::body::to_bytes;
        use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, Key, KeyInit, Nonce};
        use std::convert::Infallible;

        let payload = completion_error_payload("Inference provider request failed");
        let plaintext = payload.to_string();
        let key_bytes = [0x42; 32];
        let nonce_bytes = [0x24; 12];
        let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_bytes));
        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce_bytes), plaintext.as_bytes())
            .expect("encrypt mock SSE error payload");
        let encrypted = [nonce_bytes.as_slice(), ciphertext.as_slice()].concat();

        let event = sse_event_from_encrypted_data(&encrypted);
        let response =
            Sse::new(futures::stream::iter([Ok::<Event, Infallible>(event)])).into_response();
        let body = to_bytes(response.into_body(), 4096)
            .await
            .expect("serialize SSE event");
        let body = std::str::from_utf8(&body).expect("SSE body is UTF-8");
        let encoded = body
            .strip_prefix("data: ")
            .and_then(|value| value.strip_suffix("\n\n"))
            .expect("one SSE data event");

        assert_ne!(encoded, plaintext);
        let decoded = general_purpose::STANDARD
            .decode(encoded)
            .expect("Maple SDK base64 contract");
        let (nonce, ciphertext) = decoded.split_at(12);
        let nonce: [u8; 12] = nonce.try_into().expect("12-byte nonce");
        let decrypted = crate::web::attestation_routes::SessionState::new(key_bytes)
            .decrypt(ciphertext, &nonce)
            .expect("Maple SDK AEAD contract");
        assert_eq!(decrypted, plaintext.as_bytes());
    }

    #[tokio::test]
    async fn mock_provider_pending_non_streaming_body_times_out() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let pending = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from_stream(pending))
                    .expect("mock pending response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = trace.result.expect("mock provider response start");
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());

        let failure = read_non_streaming_completion_response(
            response,
            "kimi-k3",
            &attempt,
            None,
            Duration::from_millis(20),
        )
        .await
        .expect_err("pending non-streaming body must time out");
        server.abort();

        assert_eq!(failure.kind, AttemptFailureKind::ResponseBody);
        assert_eq!(failure.stage, AttemptStage::ResponseBody);
        assert_eq!(failure.replay_safety, ReplaySafety::NotProvenPreAcceptance);
    }

    #[tokio::test]
    async fn mock_provider_pending_stream_has_one_timeout_terminal_result() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let pending = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(axum::body::Body::from_stream(pending))
                    .expect("mock pending response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = match trace.result {
            Ok(response) => response,
            Err(error) => panic!("mock provider failed before timeout test: {error}"),
        };
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let (tx, mut rx) = mpsc::channel(1);
        let result =
            process_completion_stream(response, "kimi-k3", attempt, &tx, Duration::from_millis(20))
                .await;
        drop(tx);
        server.abort();

        assert!(rx.recv().await.is_none());
        assert!(matches!(
            result.terminal,
            AttemptTerminal::Failed {
                failure: AttemptFailure {
                    kind: AttemptFailureKind::StreamTimeout,
                    ..
                },
                ..
            }
        ));
        let provider_router = ProviderRouter::default();
        let health = record_terminal_once(&provider_router, &result.terminal).await;
        assert_eq!(
            health.route_health,
            crate::inference::health::ShadowDisposition::Watch {
                consecutive_failures: 1
            }
        );
    }

    #[tokio::test]
    async fn slow_stream_consumer_cannot_hold_the_processor_forever() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(axum::body::Body::from(concat!(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"one\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"content\":\"two\"},\"finish_reason\":null}]}\n\n",
                        "data: [DONE]\n\n"
                    )))
                    .expect("mock streaming response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = trace.result.expect("mock provider response start");
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let (tx, _rx) = mpsc::channel(1);

        let result = timeout(
            Duration::from_millis(200),
            process_completion_stream(response, "kimi-k3", attempt, &tx, Duration::from_millis(20)),
        )
        .await
        .expect("slow consumer must be bounded");
        server.abort();

        assert!(matches!(
            result.terminal,
            AttemptTerminal::Failed {
                failure: AttemptFailure {
                    kind: AttemptFailureKind::ConsumerDropped,
                    stage: AttemptStage::Stream,
                    ..
                },
                ..
            }
        ));
    }

    #[tokio::test]
    async fn dropping_completion_consumer_cancels_a_pending_provider_body() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let pending = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
                axum::http::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(axum::body::Body::from_stream(pending))
                    .expect("mock pending response")
            }),
        );
        let (trace, server) = call_mock_provider(app).await;
        let response = trace.result.expect("mock provider response start");
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let (tx, rx) = mpsc::channel(1);
        drop(rx);

        let result = timeout(
            Duration::from_millis(100),
            process_completion_stream(response, "kimi-k3", attempt, &tx, Duration::from_secs(30)),
        )
        .await
        .expect("consumer drop should cancel without waiting for chunk timeout");
        server.abort();

        assert!(matches!(
            result.terminal,
            AttemptTerminal::Failed {
                failure: AttemptFailure {
                    kind: AttemptFailureKind::ConsumerDropped,
                    stage: AttemptStage::Stream,
                    ..
                },
                ..
            }
        ));
        let provider_router = ProviderRouter::default();
        let health = record_terminal_once(&provider_router, &result.terminal).await;
        assert_eq!(
            health.effective,
            crate::inference::health::ShadowDisposition::Healthy
        );
    }

    #[tokio::test]
    async fn mock_provider_truncated_http_body_has_one_transport_terminal_result() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind truncated-body provider");
        let address = listener
            .local_addr()
            .expect("truncated-body provider address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept provider request");
            let mut request = vec![0u8; 4096];
            let _ = socket
                .read(&mut request)
                .await
                .expect("read provider request");

            let body = concat!(
                "data: {\"model\":\"upstream-k3\",",
                "\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"
            );
            let declared_length = body.len() + 128;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n"
            );
            socket
                .write_all(headers.as_bytes())
                .await
                .expect("write response headers");
            socket
                .write_all(body.as_bytes())
                .await
                .expect("write partial response body");
            socket.shutdown().await.expect("close partial response");
        });

        let base_url = format!("http://{address}");
        let client = ProviderClient::for_test(base_url.clone()).expect("mock provider client");
        let proxy = ProxyConfig {
            base_url,
            api_key: None,
            provider_name: ProviderId::Tinfoil.as_str().to_string(),
        };
        let trace = try_provider(
            &client,
            &proxy,
            r#"{"model":"kimi-k3","messages":[]}"#.to_string(),
            &HeaderMap::new(),
        )
        .await;
        let response = match trace.result {
            Ok(response) => response,
            Err(error) => panic!("mock provider failed before body truncation: {error}"),
        };
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let (tx, mut rx) = mpsc::channel(4);
        let result =
            process_completion_stream(response, "kimi-k3", attempt, &tx, Duration::from_secs(1))
                .await;
        drop(tx);
        server.await.expect("truncated-body provider task");

        let chunk = rx.recv().await.expect("partial stream chunk");
        assert!(matches!(chunk, CompletionChunk::StreamChunk(_)));
        assert!(rx.recv().await.is_none());
        assert!(matches!(
            result.terminal,
            AttemptTerminal::Failed {
                failure: AttemptFailure {
                    kind: AttemptFailureKind::Transport,
                    ..
                },
                ..
            }
        ));
        let provider_router = ProviderRouter::default();
        let health = record_terminal_once(&provider_router, &result.terminal).await;
        assert_eq!(
            health.route_health,
            crate::inference::health::ShadowDisposition::Watch {
                consecutive_failures: 1
            }
        );
    }

    #[test]
    fn response_started_is_not_a_health_success_and_cannot_clear_capacity() {
        let provider_router = ProviderRouter::default();
        let pinned = pinned_test_completion();
        let route = pinned.route.identity();
        let route_key = route.route_key();
        let execution = pinned.begin_execution();
        let mut capacity_failure = AttemptFailure::new(
            AttemptFailureKind::CapacityRejected,
            AttemptStage::AwaitingResponse,
            ReplaySafety::ProvenPreAcceptance,
        );
        capacity_failure.status = Some(429);
        capacity_failure.retry_after = Some(Duration::from_secs(60));
        record_attempt_outcome(
            &provider_router,
            &AttemptOutcome::Terminal(AttemptTerminal::Failed {
                attempt: execution.begin_attempt(route.clone()),
                failure: capacity_failure,
            }),
            ShadowObservationMode::Update,
        );
        record_attempt_outcome(
            &provider_router,
            &AttemptOutcome::ResponseStarted {
                attempt: execution.begin_attempt(route),
                status: 200,
            },
            ShadowObservationMode::Update,
        );

        assert!(matches!(
            provider_router
                .shadow_health_snapshot(&route_key)
                .expect("registered K3 route")
                .effective,
            crate::inference::health::ShadowDisposition::WouldOpen { .. }
        ));
    }

    #[test]
    fn pinned_completion_keeps_route_and_request_identity_across_model_turns() {
        let pinned = pinned_test_completion();
        let first = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let second = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());

        assert_eq!(first.request_id, second.request_id);
        assert_ne!(first.execution_id, second.execution_id);
        assert_ne!(first.attempt_id, second.attempt_id);
        assert_eq!(first.route, second.route);
        assert_eq!(first.route.public_model_id, "kimi-k3");
        assert_eq!(first.route.provider_model_id, "kimi-k3");
        assert_eq!(
            pinned.intent().requested_model_id,
            crate::model_config::AUTO_POWERFUL_MODEL_ID
        );
        assert!(pinned.intent().selection_mode.is_auto());
    }

    #[test]
    fn shadow_mismatch_or_error_never_changes_the_active_route() {
        use crate::inference_planning::{CandidateScope, PlanDecision};

        let mut pinned = pinned_test_completion();
        pinned.route.proxy.base_url = "https://active-route.example".to_string();
        pinned.route.proxy.api_key = Some("sentinel-active-api-key".to_string());
        let active = pinned.route.clone();
        let shadow = RoutePlan {
            selected: crate::inference::RouteIdentity::new(
                ProviderId::Continuum,
                "kimi-k3",
                "shadow-provider-model",
                "kimi-k3",
                crate::provider_registry::RouteSelectionSource::Fallback,
                Some(73),
            ),
            eligible_routes: Vec::new(),
            decision: PlanDecision::PreferredProviderUnavailable,
            candidate_scope: CandidateScope::SamePublicModelOnly,
            policy_version: SHADOW_ROUTING_POLICY_VERSION,
        };

        let comparison = compare_shadow_route(&Ok(active.clone()), &Ok(shadow.clone()));
        let comparison_debug = format!("{comparison:?}");
        assert!(!comparison_debug.contains("active-route.example"));
        assert!(!comparison_debug.contains("sentinel-active-api-key"));

        let retained = retain_active_route_after_shadow_observation(
            pinned.intent(),
            Ok(active.clone()),
            Ok(shadow.clone()),
        )
        .expect("shadow mismatch must retain active route");
        assert_eq!(retained.identity(), active.identity());
        assert_eq!(retained.proxy, active.proxy);

        let retained = retain_active_route_after_shadow_observation(
            pinned.intent(),
            Ok(active.clone()),
            Err(RoutePlanningError::NoEligibleRoute("kimi-k3".to_string())),
        )
        .expect("shadow error must retain active route");
        assert_eq!(retained.identity(), active.identity());
        assert_eq!(retained.proxy, active.proxy);

        let active_error = ProviderRoutingError::NoEligibleRoute("kimi-k3".to_string());
        let result = retain_active_route_after_shadow_observation(
            pinned.intent(),
            Err(active_error.clone()),
            Ok(shadow),
        );
        assert_eq!(
            result.expect_err("shadow success cannot rescue active failure"),
            active_error
        );
    }

    #[test]
    fn provider_request_replay_safety_is_conservative() {
        let cases = [
            (
                ProviderRequestError::TinfoilUnavailable,
                AttemptFailureKind::ProviderUnavailable,
                ReplaySafety::ProvenPreAcceptance,
            ),
            (
                ProviderRequestError::Build("invalid request".to_string()),
                AttemptFailureKind::RequestBuild,
                ReplaySafety::ProvenPreAcceptance,
            ),
            (
                ProviderRequestError::Connect("connection refused".to_string()),
                AttemptFailureKind::Connect,
                ReplaySafety::ProvenPreAcceptance,
            ),
            (
                ProviderRequestError::Timeout(Duration::from_secs(30)),
                AttemptFailureKind::ResponseStartTimeout,
                ReplaySafety::NotProvenPreAcceptance,
            ),
            (
                ProviderRequestError::Send("connection reset".to_string()),
                AttemptFailureKind::Transport,
                ReplaySafety::NotProvenPreAcceptance,
            ),
        ];

        for (error, expected_kind, expected_replay_safety) in cases {
            let failure = attempt_failure_from_provider_error(&error);
            assert_eq!(failure.kind, expected_kind);
            assert_eq!(failure.replay_safety, expected_replay_safety);
            assert_eq!(failure.status, None);
        }

        let upstream = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 429,
            retry_after: Some(Duration::from_secs(60)),
            upstream_request_id: Some("request-123".to_string()),
        });
        let failure = attempt_failure_from_provider_error(&upstream);
        assert_eq!(failure.kind, AttemptFailureKind::CapacityRejected);
        assert_eq!(failure.replay_safety, ReplaySafety::ProvenPreAcceptance);
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.retry_after, Some(Duration::from_secs(60)));
        assert_eq!(failure.upstream_request_id.as_deref(), Some("request-123"));
    }

    #[test]
    fn cancelled_in_flight_attempt_has_one_conservative_terminal_shape() {
        let provider_router = Arc::new(ProviderRouter::default());
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());
        let mut guard = AttemptObservationGuard::new(
            attempt.clone(),
            provider_router,
            AttemptStage::AwaitingResponse,
        );

        assert!(matches!(
            guard.cancellation_terminal(),
            AttemptTerminal::Failed {
                attempt: terminal_attempt,
                failure: AttemptFailure {
                    kind: AttemptFailureKind::ConsumerDropped,
                    stage: AttemptStage::AwaitingResponse,
                    replay_safety: ReplaySafety::NotProvenPreAcceptance,
                    ..
                },
            } if terminal_attempt == attempt
        ));
        guard.disarm();
    }

    #[test]
    fn recovered_transport_retry_gets_a_distinct_attempt_in_the_same_execution() {
        let pinned = pinned_test_completion();
        let provider_router = ProviderRouter::default();
        let execution = pinned.begin_execution();
        let route = pinned.route.identity();
        let recovered = terminalize_recovered_provider_failures(
            &provider_router,
            execution,
            &route,
            vec![ProviderRequestError::Connect(
                "attested route changed before connect".to_string(),
            )],
        );
        let final_attempt = execution.begin_attempt(route);

        let recovered_attempt = match recovered.as_slice() {
            [AttemptTerminal::Failed { attempt, failure }] => {
                assert_eq!(failure.kind, AttemptFailureKind::Connect);
                assert_eq!(failure.replay_safety, ReplaySafety::ProvenPreAcceptance);
                attempt
            }
            terminals => panic!("unexpected recovered terminal sequence: {terminals:?}"),
        };
        assert_eq!(recovered_attempt.request_id, final_attempt.request_id);
        assert_eq!(recovered_attempt.execution_id, final_attempt.execution_id);
        assert_ne!(recovered_attempt.attempt_id, final_attempt.attempt_id);
    }

    #[test]
    fn stream_eof_requires_prior_finish_evidence() {
        let pinned = pinned_test_completion();
        let attempt = pinned
            .begin_execution()
            .begin_attempt(pinned.route.identity());

        assert!(matches!(
            stream_end_terminal(attempt.clone(), true, false),
            AttemptTerminal::Completed {
                evidence: CompletionEvidence::FinishSignalThenEof,
                ..
            }
        ));
        assert!(matches!(
            stream_end_terminal(attempt, false, false),
            AttemptTerminal::Failed {
                failure: AttemptFailure {
                    kind: AttemptFailureKind::UnexpectedEof,
                    replay_safety: ReplaySafety::NotProvenPreAcceptance,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn retry_after_seconds_are_bounded_and_invalid_values_are_ignored() {
        assert_eq!(
            parse_retry_after_hint(Some("60")),
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            parse_retry_after_hint(Some("7200")),
            Some(Duration::from_secs(3600))
        );
        assert_eq!(parse_retry_after_hint(Some("not-seconds")), None);
        assert_eq!(parse_retry_after_hint(None), None);
    }

    #[test]
    fn capacity_statuses_are_explicit_rejections_and_529_is_normalized() {
        for (upstream_status, public_status) in [
            (429, StatusCode::TOO_MANY_REQUESTS),
            (503, StatusCode::SERVICE_UNAVAILABLE),
            (529, StatusCode::SERVICE_UNAVAILABLE),
        ] {
            let provider_error = ProviderRequestError::Upstream(UpstreamProviderError {
                status: upstream_status,
                retry_after: Some(Duration::from_secs(7)),
                upstream_request_id: None,
            });
            let failure = attempt_failure_from_provider_error(&provider_error);
            assert_eq!(failure.kind, AttemptFailureKind::CapacityRejected);
            assert_eq!(failure.replay_safety, ReplaySafety::ProvenPreAcceptance);
            assert_eq!(failure.status, Some(upstream_status));
            assert!(matches!(
                public_completion_error(&provider_error, &failure),
                ApiError::InferenceCapacity {
                    status,
                    retry_after: Some(delay),
                    client_replay_safe: false,
                } if status == public_status && delay == Duration::from_secs(7)
            ));
        }

        let generic = ProviderRequestError::Upstream(UpstreamProviderError {
            status: 500,
            retry_after: None,
            upstream_request_id: None,
        });
        let failure = attempt_failure_from_provider_error(&generic);
        assert_eq!(failure.kind, AttemptFailureKind::HttpStatus);
        assert_eq!(failure.replay_safety, ReplaySafety::NotProvenPreAcceptance);
        assert!(matches!(
            public_completion_error(&generic, &failure),
            ApiError::InternalServerError
        ));
    }

    #[test]
    fn completion_model_access_enforces_plan_gates() {
        assert!(matches!(
            ensure_completion_model_access("kimi-k3", ModelPlan::Free),
            Err(ApiError::ModelNotAvailableOnPlan)
        ));
        assert!(ensure_completion_model_access("kimi-k3", ModelPlan::Paid).is_ok());
        assert!(matches!(
            ensure_completion_model_access("deepseek-v4-flash", ModelPlan::Free),
            Err(ApiError::ModelNotAvailableOnPlan)
        ));
        assert!(ensure_completion_model_access("deepseek-v4-flash", ModelPlan::Paid).is_ok());
        assert!(matches!(
            ensure_completion_model_access("glm-5-2", ModelPlan::Free),
            Err(ApiError::ModelNotAvailableOnPlan)
        ));
        assert!(ensure_completion_model_access("glm-5-2", ModelPlan::Paid).is_ok());
        assert!(matches!(
            ensure_completion_model_access(
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
                ModelPlan::Free
            ),
            Err(ApiError::ModelNotAvailableOnPlan)
        ));
        // Plan access does not replace normal model validation. Unknown IDs
        // continue to the existing provider-routing validation path.
        assert!(ensure_completion_model_access("gpt-oss-120b", ModelPlan::Free).is_ok());
        assert!(ensure_completion_model_access("not-a-real-model", ModelPlan::Free).is_ok());
    }

    fn tts_request(payload: Value) -> TTSRequest {
        serde_json::from_value(payload).unwrap()
    }

    #[test]
    fn voxtral_is_the_default_tts_model_with_a_minimal_payload() {
        let request = tts_request(json!({
            "input": "Privacy-safe speech",
        }));

        let prepared = prepare_tts_request(request).unwrap();
        assert_eq!(prepared.model, VOXTRAL_TTS_MODEL);
        assert_eq!(prepared.voice_for_log, DEFAULT_VOXTRAL_TTS_VOICE);
        assert_eq!(
            prepared.provider_payload,
            json!({
                "model": VOXTRAL_TTS_MODEL,
                "voice": DEFAULT_VOXTRAL_TTS_VOICE,
                "input": "Privacy-safe speech",
            })
        );
        assert_eq!(
            prepared.provider_payload.as_object().unwrap().len(),
            3,
            "Voxtral payloads should contain only model, voice, and input"
        );
    }

    #[test]
    fn tts_request_is_an_exact_passthrough_when_defaults_are_present() {
        let payload = json!({
            "input": "Privacy-safe speech",
            "model": VOXTRAL_TTS_MODEL,
            "voice": "neutral_male",
            "speed": 1.2,
            "response_format": "flac",
            "stream": false,
            "stream_format": "audio",
            "task_type": "CustomVoice",
            "max_new_tokens": 2048,
            "seed": 42,
            "language": "fr",
            "instructions": "Speak warmly",
            "ref_audio": "data:audio/wav;base64,UklGRg==",
            "ref_text": "Reference text",
            "ref_audio_2": "data:audio/wav;base64,UklGRg==",
            "ambient_sound": "quiet room",
            "duration_seconds": 4.5,
            "x_vector_only_mode": false,
            "speaker_embedding": [0.1, 0.2],
            "initial_codec_chunk_frames": 4,
            "non_streaming_mode": true,
            "word_timestamps": false,
            "extra_params": {"cfg_alpha": 0.5},
        });
        let prepared = prepare_tts_request(tts_request(payload.clone())).unwrap();

        assert_eq!(prepared.voice_for_log, "neutral_male");
        assert_eq!(prepared.provider_payload, payload);
    }

    #[test]
    fn tts_preserves_unknown_future_parameters_and_provider_validates_voice() {
        let payload = json!({
            "input": "Privacy-safe speech",
            "model": VOXTRAL_TTS_MODEL,
            "voice": "future_provider_voice",
            "future_top_level": {"nested": [1, true, "value"]},
        });
        let prepared = prepare_tts_request(tts_request(payload.clone())).unwrap();

        assert_eq!(prepared.voice_for_log, "future_provider_voice");
        assert_eq!(prepared.provider_payload, payload);
    }

    #[test]
    fn tts_preserves_speed_and_explicit_nulls() {
        let payload = json!({
            "input": "Privacy-safe speech",
            "model": VOXTRAL_TTS_MODEL,
            "voice": null,
            "speed": 1.2,
            "language": null,
            "instructions": null,
            "ref_audio": null,
            "ref_text": null,
            "speaker_embedding": null,
            "max_new_tokens": null,
            "seed": null,
            "extra_params": null,
            "future_optional": null,
        });
        let prepared = prepare_tts_request(tts_request(payload.clone())).unwrap();

        assert_eq!(prepared.voice_for_log, "<null>");
        assert_eq!(prepared.provider_payload, payload);
    }

    #[test]
    fn tts_defaults_voice_only_without_voice_conditioning() {
        for payload in [
            json!({
                "input": "Privacy-safe speech",
                "speaker": "provider-speaker-id",
            }),
            json!({
                "input": "Privacy-safe speech",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
            }),
            json!({
                "input": "Privacy-safe speech",
                "references": [{
                    "audio_path": "data:audio/wav;base64,UklGRg==",
                    "text": "Reference text",
                }],
            }),
            json!({
                "input": "Privacy-safe speech",
                "speaker": null,
            }),
        ] {
            let prepared = prepare_tts_request(tts_request(payload.clone())).unwrap();
            let provider_payload = prepared.provider_payload.as_object().unwrap();

            assert_eq!(
                provider_payload.get("model"),
                Some(&json!(VOXTRAL_TTS_MODEL))
            );
            assert!(!provider_payload.contains_key("voice"));
            for (key, value) in payload.as_object().unwrap() {
                assert_eq!(provider_payload.get(key), Some(value));
            }
        }
    }

    #[test]
    fn tts_stream_fields_are_passed_through_for_the_buffered_encrypted_response() {
        let payload = json!({
            "input": "Privacy-safe speech",
            "model": VOXTRAL_TTS_MODEL,
            "voice": "neutral_female",
            "stream": true,
            "stream_format": "sse",
        });

        let prepared = prepare_tts_request(tts_request(payload.clone())).unwrap();

        assert_eq!(prepared.provider_payload, payload);
    }

    #[test]
    fn tts_validation_rejects_empty_input_and_non_voxtral_models() {
        assert_eq!(
            prepare_tts_request(tts_request(json!({
                "input": " \n\t",
                "model": VOXTRAL_TTS_MODEL,
            }))),
            Err(TTSRequestValidationError::EmptyInput)
        );
        assert_eq!(
            prepare_tts_request(tts_request(json!({
                "input": "Privacy-safe speech",
                "model": "qwen3-tts",
            }))),
            Err(TTSRequestValidationError::UnsupportedModel)
        );
        assert_eq!(
            prepare_tts_request(tts_request(json!({
                "input": "Privacy-safe speech",
                "model": "unknown-tts",
            }))),
            Err(TTSRequestValidationError::UnsupportedModel)
        );
        assert_eq!(
            prepare_tts_request(tts_request(json!({
                "input": "Privacy-safe speech",
                "model": null,
            }))),
            Err(TTSRequestValidationError::UnsupportedModel)
        );
        assert_eq!(
            prepare_tts_request(tts_request(json!({"input": 123}))),
            Err(TTSRequestValidationError::EmptyInput)
        );
        assert_eq!(
            prepare_tts_request(tts_request(json!({"voice": "neutral_female"}))),
            Err(TTSRequestValidationError::EmptyInput)
        );
    }

    #[test]
    fn tts_input_length_has_a_high_sanity_ceiling() {
        let at_limit = tts_request(json!({
            "input": "a".repeat(MAX_TTS_INPUT_CHARS),
        }));
        assert!(prepare_tts_request(at_limit).is_ok());

        let over_limit = tts_request(json!({
            "input": "a".repeat(MAX_TTS_INPUT_CHARS + 1),
        }));
        assert_eq!(
            prepare_tts_request(over_limit),
            Err(TTSRequestValidationError::InputTooLong)
        );
    }

    #[test]
    fn tts_entitlement_decision_is_fail_closed() {
        assert!(tts_billing_access_decision(TTSBillingAccess::Allowed).is_ok());
        assert!(matches!(
            tts_billing_access_decision(TTSBillingAccess::FreeOrExhausted),
            Err(ApiError::UsageLimitReached)
        ));
        assert!(matches!(
            tts_billing_access_decision(TTSBillingAccess::Unavailable),
            Err(ApiError::ServiceUnavailable)
        ));
    }

    #[test]
    fn tts_binary_response_preserves_bytes_and_mime_type() {
        let audio_bytes = [b'R', b'I', b'F', b'F', 0, 0xff, b'W', b'A', b'V', b'E'];
        let (response, is_json_response) = build_tts_response_payload(&audio_bytes, "audio/flac");
        let decoded = general_purpose::STANDARD
            .decode(response["content_base64"].as_str().unwrap())
            .unwrap();

        assert!(!is_json_response);
        assert_eq!(decoded, audio_bytes);
        assert_eq!(response["content_type"], "audio/flac");
    }

    #[test]
    fn tts_json_response_preserves_exact_bytes_and_problem_mime_type() {
        let body = br#"{ "error" : { "message" : "voice generation failed" } }"#;

        let (response, is_json_response) =
            build_tts_response_payload(body, "application/problem+json; charset=utf-8");
        let decoded = general_purpose::STANDARD
            .decode(response["content_base64"].as_str().unwrap())
            .unwrap();

        assert!(is_json_response);
        assert_eq!(decoded, body);
        assert_eq!(
            response["content_type"],
            "application/problem+json; charset=utf-8"
        );
    }

    #[test]
    fn extracts_sse_after_utf8_code_point_is_split_across_chunks() {
        let json = "{\"choices\":[{\"delta\":{\"content\":\"café\"}}]}";
        let event = format!("data: {json}\n\n").into_bytes();
        let split = event
            .windows(2)
            .position(|window| window == "é".as_bytes())
            .unwrap()
            + 1;
        let mut buffer = event[..split].to_vec();

        assert_eq!(extract_sse_frame(&mut buffer), None);

        buffer.extend_from_slice(&event[split..]);
        let frame = extract_sse_frame(&mut buffer).expect("complete SSE frame");

        assert_eq!(
            serde_json::from_slice::<Value>(&frame).unwrap(),
            json!(
                {"choices": [{"delta": {"content": "café"}}]}
            )
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn extracts_crlf_and_multiline_sse_data() {
        let mut buffer = b": keepalive\r\ndata: first\r\ndata: second\r\n\r\n".to_vec();

        assert_eq!(
            extract_sse_frame(&mut buffer),
            Some(b"first\nsecond".to_vec())
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn tinfoil_user_cache_secret_is_stable_and_unique_per_user() {
        let first_user = Uuid::nil();
        let second_user = Uuid::from_u128(1);

        assert_eq!(
            tinfoil_user_cache_secret(first_user),
            "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
        );
        assert_eq!(
            tinfoil_user_cache_secret(first_user),
            tinfoil_user_cache_secret(first_user)
        );
        assert_ne!(
            tinfoil_user_cache_secret(first_user),
            tinfoil_user_cache_secret(second_user)
        );
    }

    #[test]
    fn applies_tinfoil_user_cache_secret_and_overwrites_client_value() {
        let user_uuid = Uuid::from_u128(42);
        let mut body = serde_json::Map::from_iter([
            ("model".to_string(), json!("kimi-k2-6")),
            ("cache_salt".to_string(), json!("user-supplied")),
            ("user_cache_secret".to_string(), json!("client-controlled")),
            ("messages".to_string(), json!([])),
        ]);

        apply_provider_managed_request_fields(&mut body, ProviderId::Tinfoil.as_str(), user_uuid);

        assert_eq!(body.get("model"), Some(&json!("kimi-k2-6")));
        assert_eq!(body.get("messages"), Some(&json!([])));
        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));
        assert_eq!(
            body.get(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD),
            Some(&json!(tinfoil_user_cache_secret(user_uuid)))
        );
    }

    #[test]
    fn strips_tinfoil_router_execution_controls_but_preserves_tool_schema() {
        let mut body = serde_json::Map::from_iter([
            ("web_search_options".to_string(), json!({"enabled": true})),
            (
                "code_execution_options".to_string(),
                json!({"accessToken": "client-controlled"}),
            ),
            ("pii_check_options".to_string(), json!({})),
            (
                "auto_model_options".to_string(),
                json!([{"model": "different-model"}]),
            ),
            (
                "tools".to_string(),
                json!([{
                    "type": "function",
                    "function": {
                        "name": "render_preview",
                        "description": "Render a preview",
                        "parameters": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}}
                        },
                        "x-tinfoil-tool-auto-continue": true
                    }
                }]),
            ),
            (
                "tool_choice".to_string(),
                json!({"type": "function", "function": {"name": "render_preview"}}),
            ),
        ]);

        apply_provider_managed_request_fields(
            &mut body,
            ProviderId::Tinfoil.as_str(),
            Uuid::from_u128(42),
        );

        for field in TINFOIL_ROUTER_EXECUTION_FIELDS {
            assert!(!body.contains_key(*field));
        }
        let function = body["tools"][0]["function"]
            .as_object()
            .expect("ordinary function schema remains");
        assert!(!function.contains_key(TINFOIL_TOOL_AUTO_CONTINUE_FIELD));
        assert_eq!(function["name"], "render_preview");
        assert_eq!(function["description"], "Render a preview");
        assert_eq!(
            function["parameters"]["properties"]["title"]["type"],
            "string"
        );
        assert_eq!(body["tool_choice"]["function"]["name"], "render_preview");
    }

    #[test]
    fn strips_tinfoil_cache_fields_from_non_tinfoil_requests() {
        let mut body = serde_json::Map::from_iter([
            ("cache_salt".to_string(), json!("user-supplied")),
            ("user_cache_secret".to_string(), json!("client-controlled")),
            (
                "kv_transfer_params".to_string(),
                json!({"prompt_token_ids": [1, 2, 3]}),
            ),
        ]);

        apply_provider_managed_request_fields(
            &mut body,
            ProviderId::Continuum.as_str(),
            Uuid::from_u128(42),
        );

        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));
        assert!(!body.contains_key(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD));
        assert!(!body.contains_key(PROVIDER_MANAGED_KV_TRANSFER_PARAMS_FIELD));
    }

    #[test]
    fn server_selected_continuum_route_replaces_caller_salt_with_isolated_salt() {
        let user_uuid = Uuid::from_u128(42);
        let mut body = serde_json::Map::from_iter([
            ("cache_salt".to_string(), json!("caller-controlled")),
            ("messages".to_string(), json!([])),
        ]);

        apply_provider_managed_request_fields(&mut body, ProviderId::Continuum.as_str(), user_uuid);
        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));

        let server_salt = format!("server-selected-{}", Uuid::new_v4().simple());
        apply_server_selected_cache_isolation(
            &mut body,
            ProviderId::Continuum.as_str(),
            Some(&server_salt),
        )
        .expect("server-selected Continuum salt should be accepted");

        assert_eq!(
            body.get(PROVIDER_MANAGED_CACHE_SALT_FIELD),
            Some(&json!(server_salt))
        );
        assert_ne!(
            body.get(PROVIDER_MANAGED_CACHE_SALT_FIELD),
            Some(&json!("caller-controlled"))
        );
    }

    #[test]
    fn extracts_cached_prompt_tokens_from_openai_usage_details() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {
                    "cached_tokens": 42
                }
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");

        assert_eq!(usage.prompt_tokens, 100);
        assert!(usage.prompt_tokens_observed);
        assert_eq!(usage.completion_tokens, 20);
        assert!(usage.completion_tokens_observed);
        assert_eq!(usage.cached_prompt_tokens, Some(42));
    }

    #[test]
    fn prompt_only_usage_preserves_missing_completion_token_signal() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");

        assert_eq!(usage.prompt_tokens, 100);
        assert!(usage.prompt_tokens_observed);
        assert_eq!(usage.completion_tokens, 0);
        assert!(!usage.completion_tokens_observed);
    }

    #[test]
    fn completion_only_usage_preserves_missing_prompt_token_signal() {
        let response = json!({
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens_details": {
                    "cached_tokens": 5
                }
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");
        let actual = completion_actual_usage(&usage);

        assert_eq!(usage.prompt_tokens, 0);
        assert!(!usage.prompt_tokens_observed);
        assert_eq!(usage.completion_tokens, 7);
        assert!(usage.completion_tokens_observed);
        assert_eq!(usage.cached_prompt_tokens, Some(5));
        assert_eq!(actual.prompt_tokens, None);
        assert_eq!(actual.completion_tokens, Some(7));
        assert_eq!(actual.cached_prompt_tokens, Some(5));
    }

    #[test]
    fn negative_completion_usage_is_not_treated_as_observed() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": -1
            }
        });

        let usage = extract_usage(&response).expect("prompt usage should parse");

        assert_eq!(usage.completion_tokens, 0);
        assert!(!usage.completion_tokens_observed);
    }

    #[test]
    fn cached_prompt_tokens_are_optional_in_usage_details() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");

        assert_eq!(usage.cached_prompt_tokens, None);
    }

    #[test]
    fn cached_prompt_tokens_are_clamped_to_prompt_tokens() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {
                    "cached_tokens": 150
                }
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cached_prompt_tokens, Some(100));
    }

    #[test]
    fn completion_tokens_are_clamped_to_i32_range() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": i64::MAX,
                "total_tokens": i64::MAX
            }
        });

        let usage = extract_usage(&response).expect("usage should parse");

        assert_eq!(usage.completion_tokens, i32::MAX);
    }

    #[test]
    fn cached_prompt_tokens_are_mapped_to_sqs_cached_input_tokens() {
        let usage = CompletionUsage {
            prompt_tokens: 100,
            prompt_tokens_observed: true,
            completion_tokens: 20,
            completion_tokens_observed: true,
            cached_prompt_tokens: Some(42),
        };

        let event = build_usage_event(
            Uuid::parse_str("dca25195-ae0a-4c49-aa7a-bd2ba21a7d2b").unwrap(),
            Uuid::parse_str("6142db59-fc0c-413d-8792-579fc1457fe2").unwrap(),
            usage,
            BigDecimal::from_str("0.001").unwrap(),
            true,
            "continuum".to_string(),
            "kimi-k2-6".to_string(),
        );

        assert_eq!(event.input_tokens, 100);
        assert_eq!(event.output_tokens, 20);
        assert_eq!(event.cached_input_tokens, Some(42));
        assert!(event.is_api_request);
        assert_eq!(event.provider_name, "continuum");
        assert_eq!(event.model_name, "kimi-k2-6");
    }

    #[test]
    fn route_identity_is_preserved_in_usage_events() {
        let usage = CompletionUsage {
            prompt_tokens: 100,
            prompt_tokens_observed: true,
            completion_tokens: 20,
            completion_tokens_observed: true,
            cached_prompt_tokens: Some(42),
        };

        for (provider, public_model) in [
            ("tinfoil", "gpt-oss-120b"),
            ("tinfoil", "deepseek-v4-flash"),
            ("tinfoil", "kimi-k3"),
            ("tinfoil", "glm-5-2"),
            ("continuum", "kimi-k2-6"),
            ("continuum", "glm-5-2"),
        ] {
            let event = build_usage_event(
                Uuid::parse_str("dca25195-ae0a-4c49-aa7a-bd2ba21a7d2b").unwrap(),
                Uuid::parse_str("6142db59-fc0c-413d-8792-579fc1457fe2").unwrap(),
                usage.clone(),
                BigDecimal::from_str("0.001").unwrap(),
                false,
                provider.to_string(),
                public_model.to_string(),
            );

            assert_eq!(event.provider_name, provider);
            assert_eq!(event.model_name, public_model);
            assert_eq!(event.input_tokens, 100);
            assert_eq!(event.output_tokens, 20);
            assert_eq!(event.cached_input_tokens, Some(42));
        }
    }

    #[test]
    fn provider_model_ids_are_canonicalized_in_client_responses() {
        for (provider_model, public_model) in [
            ("kimi-k2.6", "kimi-k2-6"),
            ("glm-5.2", "glm-5-2"),
            ("kimi-k3", "kimi-k3"),
        ] {
            let mut response = json!({
                "id": "completion-id",
                "model": provider_model,
                "choices": [],
            });

            canonicalize_response_model(&mut response, public_model);

            assert_eq!(response["model"], public_model);
            assert_eq!(response["id"], "completion-id");
            assert_eq!(response["choices"], json!([]));
        }

        let mut response_without_model = json!({"choices": []});
        canonicalize_response_model(&mut response_without_model, "glm-5-2");
        assert!(response_without_model.get("model").is_none());
    }

    fn stream_usage_chunk(
        prompt_tokens: i32,
        completion_tokens: i32,
        cached_prompt_tokens: Option<i32>,
        finish_reason: Option<&str>,
        usage_only: bool,
    ) -> Value {
        let mut usage = json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        });
        if let Some(cached_prompt_tokens) = cached_prompt_tokens {
            usage["prompt_tokens_details"] = json!({
                "cached_tokens": cached_prompt_tokens,
            });
        }

        let choices = if usage_only {
            json!([])
        } else {
            json!([{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }])
        };

        json!({
            "choices": choices,
            "usage": usage,
        })
    }

    fn assert_usage(
        usage: CompletionUsage,
        prompt_tokens: i32,
        completion_tokens: i32,
        cached_prompt_tokens: Option<i32>,
    ) {
        assert_eq!(usage.prompt_tokens, prompt_tokens);
        assert!(usage.prompt_tokens_observed);
        assert_eq!(usage.completion_tokens, completion_tokens);
        assert!(usage.completion_tokens_observed);
        assert_eq!(usage.cached_prompt_tokens, cached_prompt_tokens);
    }

    #[test]
    fn continuum_cache_details_after_finish_reason_are_preserved() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(20_208, 43, None, None, false));
        accumulator.observe(&stream_usage_chunk(20_208, 44, None, Some("stop"), false));
        accumulator.observe(&stream_usage_chunk(20_208, 44, Some(20_128), None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 20_208, 44, Some(20_128));
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .is_none());
    }

    #[test]
    fn tinfoil_usage_only_frame_without_cache_finalizes_once() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(30_823, 64, None, Some("length"), false));
        accumulator.observe(&stream_usage_chunk(30_823, 64, None, None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 30_823, 64, None);
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::EndOfStream)
            .is_none());
    }

    #[test]
    fn cumulative_delta_usage_is_not_summed_or_finalized_more_than_once() {
        let mut accumulator = StreamUsageAccumulator::default();
        for completion_tokens in 0..=100 {
            accumulator.observe(&stream_usage_chunk(
                500,
                completion_tokens,
                None,
                None,
                false,
            ));
        }
        accumulator.observe(&stream_usage_chunk(500, 100, None, Some("stop"), false));
        accumulator.observe(&stream_usage_chunk(500, 100, Some(480), None, true));
        accumulator.observe(&stream_usage_chunk(500, 100, Some(480), None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 500, 100, Some(480));
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .is_none());
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::ConsumerDropped)
            .is_none());
    }

    #[test]
    fn regressing_cumulative_usage_preserves_highest_totals() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(500, 100, None, None, false));
        accumulator.observe(&stream_usage_chunk(400, 0, None, Some("stop"), true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 500, 100, None);
    }

    #[test]
    fn provider_done_uses_finish_usage_when_usage_only_frame_is_missing() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 20, None, Some("stop"), false));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("[DONE] should use the latest available totals");
        assert_usage(usage, 400, 20, None);
    }

    #[test]
    fn end_of_stream_after_finish_reason_uses_terminal_fallback() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 20, None, Some("stop"), false));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::EndOfStream)
            .expect("terminal EOF should preserve existing billing behavior");
        assert_usage(usage, 400, 20, None);
    }

    #[test]
    fn interruption_before_terminal_signal_does_not_finalize_usage() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 19, None, None, false));

        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::TransportError)
            .is_none());
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::Timeout)
            .is_none());
    }

    #[test]
    fn interruption_after_terminal_signal_finalizes_once() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 20, None, Some("stop"), false));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::TransportError)
            .expect("terminal transport failure should retain usage");
        assert_usage(usage, 400, 20, None);
        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::InvalidData)
            .is_none());
    }

    #[test]
    fn cached_tokens_survive_later_usage_that_omits_details() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(120, 1, Some(96), None, false));
        accumulator.observe(&stream_usage_chunk(120, 2, None, Some("stop"), false));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 120, 2, Some(96));
    }

    #[test]
    fn later_explicit_zero_cached_tokens_is_authoritative() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(120, 1, Some(96), None, false));
        accumulator.observe(&stream_usage_chunk(120, 1, Some(0), None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("provider [DONE] should finalize usage");
        assert_usage(usage, 120, 1, Some(0));
    }

    #[test]
    fn partial_usage_observation_does_not_erase_final_totals() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 20, None, Some("stop"), false));
        accumulator.observe(&json!({
            "choices": [],
            "usage": {
                "prompt_tokens_details": {
                    "cached_tokens": 300,
                }
            }
        }));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("partial final usage should merge with prior totals");
        assert_usage(usage, 400, 20, Some(300));
    }

    #[test]
    fn empty_usage_observation_does_not_erase_final_totals() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(400, 20, None, Some("stop"), false));
        accumulator.observe(&json!({
            "choices": [],
            "usage": {},
        }));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("empty final usage should not erase prior totals");
        assert_usage(usage, 400, 20, None);
    }

    #[test]
    fn prompt_only_usage_is_not_dropped() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(33, 0, None, None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("non-zero prompt usage should be retained");
        assert_usage(usage, 33, 0, None);
    }

    #[test]
    fn authoritative_zero_usage_is_not_dropped() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(0, 0, None, None, true));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("explicit zero totals are authoritative usage");
        assert_usage(usage, 0, 0, None);
    }

    #[test]
    fn stream_prompt_only_usage_keeps_completion_tokens_unobserved() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&json!({
            "choices": [{ "finish_reason": "tool_calls" }],
            "usage": { "prompt_tokens": 33 }
        }));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("non-zero prompt usage should be retained");
        assert_eq!(usage.prompt_tokens, 33);
        assert!(usage.prompt_tokens_observed);
        assert_eq!(usage.completion_tokens, 0);
        assert!(!usage.completion_tokens_observed);
    }

    #[test]
    fn stream_cached_usage_without_prompt_total_keeps_cached_tokens_unobserved() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&json!({
            "choices": [{ "finish_reason": "stop" }],
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens_details": { "cached_tokens": 5 }
            }
        }));

        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("observed completion usage should be retained");
        let actual = completion_actual_usage(&usage);
        assert!(!usage.prompt_tokens_observed);
        assert!(usage.completion_tokens_observed);
        assert_eq!(usage.cached_prompt_tokens, Some(5));
        assert_eq!(actual.prompt_tokens, None);
        assert_eq!(actual.completion_tokens, Some(7));
        assert_eq!(actual.cached_prompt_tokens, Some(5));
    }

    #[test]
    fn provider_done_is_authoritative_without_finish_reason() {
        let mut accumulator = StreamUsageAccumulator::default();
        accumulator.observe(&stream_usage_chunk(80, 3, None, None, false));

        assert!(accumulator
            .take_final_usage(StreamUsageFinalization::EndOfStream)
            .is_none());
        let usage = accumulator
            .take_final_usage(StreamUsageFinalization::ProviderDone)
            .expect("explicit [DONE] should finalize the latest usage");
        assert_usage(usage, 80, 3, None);
    }
}
