use crate::model_config::{
    model_catalog_response, openai_models_response, ModelAliasTargets, ModelPlan,
};
use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::provider_cache::DerivedCacheNamespace;
use crate::provider_client::{
    ProviderClient, ProviderRequest, ProviderRequestError, ProviderResponse,
};
use crate::provider_routing::{ProviderName, ProviderRoutingError};
use crate::proxy_config::ProxyConfig;
use crate::sqs::UsageEvent;
use crate::web::audio_utils::{merge_transcriptions, AudioSplitter, TINFOIL_MAX_SIZE};
use crate::web::encryption_middleware::{
    decrypt_request, encrypt_response, EncryptedResponse, TransportSession,
};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use axum::http::HeaderMap;
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
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

// Maximum audio file size (100MB) - sanity check, CF already limits to 50MB
const MAX_AUDIO_SIZE: usize = 100 * 1024 * 1024;

// Timeout constants for provider requests
const REQUEST_TIMEOUT_SECS: u64 = 120; // Request timeout (generous for large non-streaming responses)
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 120; // Per-chunk timeout for streaming reads
const TTS_BILLING_CHECK_TIMEOUT: Duration = Duration::from_secs(5);
const TTS_PROVIDER_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_TTS_INPUT_CHARS: usize = 100_000;
const MAX_BOUNDED_PROVIDER_RESPONSE_BYTES: usize = 256 * 1024;
const V2_MAX_PROVIDER_RESPONSE_BYTES: usize = 28 * 1024 * 1024;
// TTS wraps provider bytes in padded base64 plus a small JSON object. Keep the
// raw response below three quarters of the logical v2 response ceiling so the
// transport's bounded serializer always has room for that expansion.
const V2_MAX_TTS_PROVIDER_RESPONSE_BYTES: usize = (V2_MAX_PROVIDER_RESPONSE_BYTES - 1024) / 4 * 3;
const V2_MAX_PROVIDER_JSON_DEPTH: usize = 128;
const V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS: usize = 1_048_576;
const V2_MAX_TRANSCRIPTION_CHUNKS: usize = 4;
// The v2 gateway holds a 128 MiB provider working-set reservation. Bound
// retained chunk allocations to 64 MiB, then process them sequentially. With
// one at-most-32 MiB multipart body and the per-chunk share of the 28 MiB
// provider response bound, this leaves allocator and metadata headroom.
const V2_MAX_TRANSCRIPTION_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const V2_MAX_TRANSCRIPTION_MULTIPART_BYTES: usize = 32 * 1024 * 1024;

const PROVIDER_MANAGED_CACHE_SALT_FIELD: &str = "cache_salt";
const PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD: &str = "user_cache_secret";

/// Cache isolation is a required input to every completion dispatch. Keeping
/// the legacy derivation as an explicit variant prevents a new transport from
/// silently inheriting the v1 SHA256(UUID) namespace.
pub(crate) enum CompletionCachePolicy<'a> {
    LegacyV1,
    BoundV2(&'a DerivedCacheNamespace),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderResponseBodyPolicy {
    Unbounded,
    Bounded {
        limit_bytes: usize,
        max_json_structural_tokens: Option<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderErrorBodyPolicy {
    LegacyBody,
    StatusOnly { drain_limit_bytes: usize },
}

impl ProviderResponseBodyPolicy {
    const fn error_body(self) -> ProviderErrorBodyPolicy {
        match self {
            Self::Unbounded => ProviderErrorBodyPolicy::LegacyBody,
            Self::Bounded { .. } => ProviderErrorBodyPolicy::StatusOnly {
                drain_limit_bytes: MAX_BOUNDED_PROVIDER_RESPONSE_BYTES,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderRetryPolicy {
    LegacyV1,
    NoAmbiguousRetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnaryProviderPolicy {
    response_body: ProviderResponseBodyPolicy,
    retry: ProviderRetryPolicy,
}

impl UnaryProviderPolicy {
    const LEGACY_V1: Self = Self {
        response_body: ProviderResponseBodyPolicy::Unbounded,
        retry: ProviderRetryPolicy::LegacyV1,
    };

    const V2: Self = Self {
        response_body: ProviderResponseBodyPolicy::Bounded {
            limit_bytes: V2_MAX_PROVIDER_RESPONSE_BYTES,
            max_json_structural_tokens: Some(V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS),
        },
        retry: ProviderRetryPolicy::NoAmbiguousRetry,
    };

    const V2_TTS: Self = Self {
        response_body: ProviderResponseBodyPolicy::Bounded {
            limit_bytes: V2_MAX_TTS_PROVIDER_RESPONSE_BYTES,
            max_json_structural_tokens: None,
        },
        retry: ProviderRetryPolicy::NoAmbiguousRetry,
    };

    fn divide_bounded_response_across(self, parts: usize) -> Self {
        let response_body = match self.response_body {
            ProviderResponseBodyPolicy::Unbounded => ProviderResponseBodyPolicy::Unbounded,
            ProviderResponseBodyPolicy::Bounded {
                limit_bytes,
                max_json_structural_tokens,
            } => ProviderResponseBodyPolicy::Bounded {
                limit_bytes: limit_bytes / parts.max(1),
                max_json_structural_tokens: max_json_structural_tokens
                    .map(|limit| limit / parts.max(1)),
            },
        };
        Self {
            response_body,
            retry: self.retry,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderJsonShapeError {
    TooLarge,
    Malformed,
}

/// Bound JSON container amplification before serde allocates provider-owned
/// arrays and maps. Strings are skipped without allocation; serde remains the
/// authoritative syntax validator after this v2-only preflight.
fn validate_provider_json_shape(
    bytes: &[u8],
    max_structural_tokens: usize,
) -> Result<(), ProviderJsonShapeError> {
    let mut depth = 0usize;
    let mut structural_tokens = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match *byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth
                    .checked_add(1)
                    .ok_or(ProviderJsonShapeError::TooLarge)?;
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(ProviderJsonShapeError::TooLarge)?;
                if depth > V2_MAX_PROVIDER_JSON_DEPTH || structural_tokens > max_structural_tokens {
                    return Err(ProviderJsonShapeError::TooLarge);
                }
            }
            b'}' | b']' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or(ProviderJsonShapeError::Malformed)?;
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(ProviderJsonShapeError::TooLarge)?;
                if structural_tokens > max_structural_tokens {
                    return Err(ProviderJsonShapeError::TooLarge);
                }
            }
            b',' | b':' => {
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(ProviderJsonShapeError::TooLarge)?;
                if structural_tokens > max_structural_tokens {
                    return Err(ProviderJsonShapeError::TooLarge);
                }
            }
            _ => {}
        }
    }

    if in_string || depth != 0 {
        Err(ProviderJsonShapeError::Malformed)
    } else {
        Ok(())
    }
}

fn preflight_provider_json(
    bytes: &[u8],
    response_body_policy: ProviderResponseBodyPolicy,
) -> Result<(), ApiError> {
    let ProviderResponseBodyPolicy::Bounded {
        max_json_structural_tokens: Some(max_structural_tokens),
        ..
    } = response_body_policy
    else {
        return Ok(());
    };

    match validate_provider_json_shape(bytes, max_structural_tokens) {
        Ok(()) => Ok(()),
        Err(ProviderJsonShapeError::TooLarge) => Err(ApiError::PayloadTooLarge),
        Err(ProviderJsonShapeError::Malformed) => Err(ApiError::InternalServerError),
    }
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
pub(crate) struct TTSRequest {
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

fn build_tts_response_payload_with_policy(
    body_bytes: &[u8],
    content_type: &str,
    avoid_provider_value_allocation: bool,
) -> (Value, bool) {
    let is_json_response = if avoid_provider_value_allocation {
        serde_json::from_slice::<serde::de::IgnoredAny>(body_bytes).is_ok()
    } else {
        // Preserve v1's exact JSON detection behavior.
        serde_json::from_slice::<Value>(body_bytes).is_ok()
    };
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
pub(crate) struct TranscriptionRequest {
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
pub(crate) struct EmbeddingRequest {
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

/// Usage statistics extracted from a completion
#[derive(Debug, Clone)]
pub struct CompletionUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
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
                    "Streaming usage totals regressed: previous_prompt_tokens={}, previous_completion_tokens={}, observed_prompt_tokens={:?}, observed_completion_tokens={:?}; using the latest explicit provider values",
                    previous.prompt_tokens,
                    previous.completion_tokens,
                    observed.prompt_tokens,
                    observed.completion_tokens
                );
            }
        }

        self.latest_usage = Some(CompletionUsage {
            prompt_tokens: observed
                .prompt_tokens
                .or_else(|| self.latest_usage.as_ref().map(|usage| usage.prompt_tokens))
                .unwrap_or(0),
            completion_tokens: observed
                .completion_tokens
                .or_else(|| {
                    self.latest_usage
                        .as_ref()
                        .map(|usage| usage.completion_tokens)
                })
                .unwrap_or(0),
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
        usage.cached_prompt_tokens = usage
            .cached_prompt_tokens
            .map(|cached| cached.min(usage.prompt_tokens));
        (usage.prompt_tokens > 0 || usage.completion_tokens > 0).then_some(usage)
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
    /// Stream finished
    Done,
    /// Stream error occurred
    Error(String),
}

/// Metadata about the completion
#[derive(Clone, Debug)]
pub struct CompletionMetadata {
    pub provider_name: String,
    pub model_name: String,
    pub is_streaming: bool,
}

/// Processed completion stream - billing happens automatically
pub struct CompletionStream {
    /// The actual data stream for consumers
    pub stream: mpsc::Receiver<CompletionChunk>,
    /// Metadata about the completion
    pub metadata: CompletionMetadata,
}

async fn finalize_stream_usage(
    accumulator: &mut StreamUsageAccumulator,
    finalization: StreamUsageFinalization,
    state: &Arc<AppState>,
    user: &User,
    billing_context: &BillingContext,
    provider: &str,
    tx_consumer: &mpsc::Sender<CompletionChunk>,
) {
    let Some(usage) = accumulator.take_final_usage(finalization) else {
        return;
    };

    if !finalization.is_provider_done() {
        warn!(
            "Finalizing streaming usage from terminal fallback: trigger={:?}, provider={}, model={}",
            finalization, provider, billing_context.model_name
        );
    }

    publish_usage_event_internal(state, user, billing_context, usage.clone(), provider).await;

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
    axum::Extension(session_id): axum::Extension<TransportSession>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(auth_method): axum::Extension<AuthMethod>,
    axum::Extension(body): axum::Extension<Value>,
) -> Result<Response, ApiError> {
    let completion = openai_chat_completion_data(
        &state,
        &user,
        auth_method,
        body,
        &headers,
        CompletionCachePolicy::LegacyV1,
        ProviderResponseBodyPolicy::Unbounded,
    )
    .await?;

    openai_completion_v1_response(&state, &session_id, completion).await
}

#[allow(clippy::too_many_arguments)]
async fn openai_chat_completion_data(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    mut body: Value,
    headers: &HeaderMap,
    cache_policy: CompletionCachePolicy<'_>,
    response_body_policy: ProviderResponseBodyPolicy,
) -> Result<CompletionStream, ApiError> {
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

    let alias_targets = ModelAliasTargets::for_plan(model_plan);
    let model_name = alias_targets.resolve(&requested_model_name).to_string();
    if requested_model_name != model_name {
        debug!(
            "Resolved chat model {} to {}",
            requested_model_name, model_name
        );
        body.as_object_mut()
            .expect("model was read from a JSON object")
            .insert("model".to_string(), json!(model_name));
    }

    // Create billing context
    let billing_context = BillingContext::new(auth_method, requested_model_name);

    // Get the completion stream - billing happens automatically inside!
    get_chat_completion_response_with_expected_route(
        state,
        user,
        body,
        headers,
        billing_context,
        model_plan,
        None,
        cache_policy,
        response_body_policy,
    )
    .await
}

/// Execute the non-streaming Chat Completions application contract and return
/// plaintext logical JSON for transport v2. Provider output is collected under
/// the v2 bound before JSON allocation, and the cache namespace must come from
/// the session's already-bound authority.
pub(crate) async fn openai_nonstream_chat_completion_v2_data(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    body: Value,
    headers: &HeaderMap,
    cache_namespace: &DerivedCacheNamespace,
) -> Result<Value, ApiError> {
    ensure_nonstream_chat_completion(&body)?;

    let completion = openai_chat_completion_data(
        state,
        user,
        auth_method,
        body,
        headers,
        CompletionCachePolicy::BoundV2(cache_namespace),
        UnaryProviderPolicy::V2.response_body,
    )
    .await?;

    if completion.metadata.is_streaming {
        error!("Unary Chat Completions unexpectedly produced a stream");
        return Err(ApiError::InternalServerError);
    }

    let mut stream = completion.stream;
    match stream.recv().await {
        Some(CompletionChunk::FullResponse(response)) => Ok(response),
        _ => {
            error!("Unary Chat Completions did not produce a full response");
            Err(ApiError::InternalServerError)
        }
    }
}

fn ensure_nonstream_chat_completion(body: &Value) -> Result<(), ApiError> {
    match body.get("stream") {
        None | Some(Value::Bool(false)) => Ok(()),
        Some(_) => {
            error!("Chat Completions stream must be absent or Boolean false for unary v2");
            Err(ApiError::BadRequest)
        }
    }
}

async fn openai_completion_v1_response(
    state: &Arc<AppState>,
    session_id: &TransportSession,
    completion: CompletionStream,
) -> Result<Response, ApiError> {
    debug!(
        "Received completion from provider: {} (streaming: {})",
        completion.metadata.provider_name, completion.metadata.is_streaming
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
            let encrypted_response = encrypt_response(state, session_id, &response_json).await?;
            return Ok(encrypted_response.into_response());
        } else {
            error!("Expected FullResponse chunk but got something else");
            return Err(ApiError::InternalServerError);
        }
    }

    // For streaming responses, process CompletionChunk stream
    debug!("Handling streaming response");
    let mut rx = completion.stream;
    let stream_state = Arc::clone(state);
    let stream_session = session_id.clone();

    let stream = async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            match chunk {
                CompletionChunk::StreamChunk(json) => {
                    // Pass through full JSON (includes all metadata from upstream)
                    match encrypt_sse_event(&stream_state, &stream_session, &json).await {
                        Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                        Err(e) => {
                            error!("Failed to encrypt event data: {:?}", e);
                            yield Ok(Event::default().data("Error: Encryption failed"));
                            break;
                        }
                    }
                }
                CompletionChunk::Usage(_usage) => {
                    // Billing already handled internally, no need to send to client
                    trace!("Received usage chunk (billing already processed)");
                }
                CompletionChunk::Done => {
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                CompletionChunk::Error(error_msg) => {
                    error!("Stream error from completion API: {}", error_msg);
                    yield Ok(Event::default().data(format!("Error: {}", error_msg)));
                    break;
                }
                CompletionChunk::FullResponse(_) => {
                    // Shouldn't happen in streaming mode
                    error!("Received FullResponse in streaming mode");
                    yield Ok(Event::default().data("Error: Invalid event format"));
                    break;
                }
            }
        }
    };
    Ok(Sse::new(stream).into_response())
}

/// Internal function to get chat completion response with automatic billing
/// This can be used by both the proxy_openai endpoint and the responses API
///
/// Billing happens INTERNALLY within this function - consumers just receive processed chunks
pub async fn get_chat_completion_response(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
    model_plan: ModelPlan,
) -> Result<CompletionStream, ApiError> {
    get_chat_completion_response_with_expected_route(
        state,
        user,
        body,
        headers,
        billing_context,
        model_plan,
        None,
        CompletionCachePolicy::LegacyV1,
        ProviderResponseBodyPolicy::Unbounded,
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
) -> Result<CompletionStream, ApiError> {
    let continuum_cache_salt = (route.provider_name == ProviderName::Continuum.as_str())
        .then(|| format!("server-selected-{}", Uuid::new_v4().simple()));
    get_chat_completion_response_with_expected_route(
        state,
        user,
        body,
        headers,
        billing_context,
        model_plan,
        Some(ExpectedCompletionRoute {
            provider_name: route.provider_name,
            provider_model_id: route.provider_model_id,
            continuum_cache_salt,
        }),
        CompletionCachePolicy::LegacyV1,
        ProviderResponseBodyPolicy::Unbounded,
    )
    .await
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ServerSelectedCompletionRoute<'a> {
    pub provider_name: &'a str,
    pub provider_model_id: &'a str,
}

struct ExpectedCompletionRoute<'a> {
    provider_name: &'a str,
    provider_model_id: &'a str,
    continuum_cache_salt: Option<String>,
}

#[allow(clippy::too_many_arguments)]
async fn get_chat_completion_response_with_expected_route(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    mut billing_context: BillingContext,
    model_plan: ModelPlan,
    expected_route: Option<ExpectedCompletionRoute<'_>>,
    cache_policy: CompletionCachePolicy<'_>,
    response_body_policy: ProviderResponseBodyPolicy,
) -> Result<CompletionStream, ApiError> {
    if body.is_null() || body.as_object().is_none_or(|obj| obj.is_empty()) {
        error!("Request body is empty or invalid");
        return Err(ApiError::BadRequest);
    }

    let mut modified_body = body
        .as_object()
        .ok_or_else(|| {
            error!("Request body is not a JSON object");
            ApiError::BadRequest
        })?
        .clone();

    // Check if streaming is requested (default to false if not specified)
    let is_streaming = modified_body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let requested_model_name = modified_body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?
        .to_string();

    let is_server_selected_route = expected_route.is_some();
    if let Err(error) = ensure_completion_model_access(&requested_model_name, model_plan) {
        if is_server_selected_route {
            error!(
                "Server-selected completion model is not available to the internal paid route: {}",
                requested_model_name
            );
            return Err(ApiError::ServiceUnavailable);
        }
        return Err(error);
    }

    let provider_preference = state
        .provider_routing_preference(user.uuid, &requested_model_name)
        .await;
    let selected_route = state
        .provider_router
        .select_completion_route_with_preference(
            &state.proxy_router,
            user.uuid,
            &requested_model_name,
            provider_preference,
        )
        .map_err(|err| match err {
            ProviderRoutingError::UnsupportedModel(model) => {
                error!("Unsupported completion model requested: {}", model);
                if is_server_selected_route {
                    ApiError::ServiceUnavailable
                } else {
                    ApiError::BadRequest
                }
            }
            ProviderRoutingError::NoEligibleRoute(model) => {
                error!("No eligible provider route for completion model: {}", model);
                if is_server_selected_route {
                    ApiError::ServiceUnavailable
                } else {
                    ApiError::InternalServerError
                }
            }
        })?;

    if let Some(expected_route) = &expected_route {
        if selected_route.proxy.provider_name != expected_route.provider_name
            || selected_route.provider_model_id != expected_route.provider_model_id
        {
            error!(
                "Completion route did not match the server-selected constraint: public_model={}, expected_provider={}, expected_provider_model={}, selected_provider={}, selected_provider_model={}",
                selected_route.public_model_id,
                expected_route.provider_name,
                expected_route.provider_model_id,
                selected_route.proxy.provider_name,
                selected_route.provider_model_id
            );
            return Err(ApiError::ServiceUnavailable);
        }
    }

    if requested_model_name != selected_route.public_model_id
        || selected_route.public_model_id != selected_route.provider_model_id
    {
        debug!(
            "Selected completion route: requested_model={}, public_model={}, provider={}, provider_model={}, bucket={:?}, source={:?}",
            requested_model_name,
            selected_route.public_model_id,
            selected_route.proxy.provider_name,
            selected_route.provider_model_id,
            selected_route.bucket,
            selected_route.selection_source
        );
    }
    modified_body.insert(
        "model".to_string(),
        json!(selected_route.provider_model_id.clone()),
    );
    apply_provider_managed_request_fields(
        &mut modified_body,
        &selected_route.proxy.provider_name,
        user.uuid,
        &cache_policy,
    );
    if let Some(expected_route) = &expected_route {
        apply_server_selected_cache_isolation(
            &mut modified_body,
            &selected_route.proxy.provider_name,
            expected_route.continuum_cache_salt.as_deref(),
        )?;
    }
    billing_context.model_name = selected_route.public_model_id.clone();

    // Prepare the request to proxies
    debug!(
        "Sending request for public model {} as provider model {} via {}",
        selected_route.public_model_id,
        selected_route.provider_model_id,
        selected_route.proxy.provider_name
    );

    let (res, successful_provider) = {
        let mut request_body = modified_body.clone();
        let proxy_config = selected_route.proxy.clone();
        let provider_model_name = selected_route.provider_model_id.clone();

        ensure_stream_usage(&mut request_body);

        let request_body_value = Value::Object(request_body);
        let request_body_json = serde_json::to_string(&request_body_value).map_err(|e| {
            error!("Failed to serialize request body: {:?}", e);
            ApiError::InternalServerError
        })?;
        let request_log_metadata =
            CompletionRequestLogMetadata::from_body(&request_body_value, request_body_json.len());

        debug!(
            "Completion request metadata before provider call: user_uuid={}, provider={}, model={}, metadata={:?}",
            user.uuid, proxy_config.provider_name, provider_model_name, request_log_metadata
        );

        match try_provider(
            &state.provider_client,
            &proxy_config,
            request_body_json,
            headers,
        )
        .await
        {
            Ok(response) => {
                info!(
                    "Successfully got response from provider {} for model {}",
                    proxy_config.provider_name, provider_model_name
                );
                (response, proxy_config.provider_name.clone())
            }
            Err(err) => {
                error!(
                    "Completion request metadata at provider failure: user_uuid={}, provider={}, model={}, error={}, metadata={:?}",
                    user.uuid,
                    proxy_config.provider_name,
                    provider_model_name,
                    err,
                    request_log_metadata
                );
                error!(
                    "Chat completion request failed for provider {} and model {}: {}",
                    proxy_config.provider_name, provider_model_name, err
                );
                return Err(ApiError::from(err));
            }
        }
    };

    debug!(
        "Successfully received response from provider: {}",
        successful_provider
    );

    // NOW: Process the response internally and handle billing
    if !is_streaming {
        // NON-STREAMING: Simple case
        debug!("Processing non-streaming response with internal billing");
        // The request's streaming fields are forwarded unchanged, but this
        // encrypted endpoint currently buffers one byte-exact response carrier.
        let body_bytes = if expected_route.is_some() {
            collect_provider_response_body(
                res,
                ProviderResponseBodyPolicy::Bounded {
                    limit_bytes: MAX_BOUNDED_PROVIDER_RESPONSE_BYTES,
                    max_json_structural_tokens: None,
                },
            )
            .await
            .map_err(|error| {
                error!("Failed to read bounded server-selected response body: {error}");
                ApiError::InternalServerError
            })?
        } else {
            collect_provider_response_body(res, response_body_policy)
                .await
                .map_err(|error| {
                    error!("Failed to read response body: {error}");
                    error
                })?
        };

        preflight_provider_json(&body_bytes, response_body_policy).inspect_err(|_| {
            error!("Completion provider response failed structural JSON preflight");
        })?;

        let mut response_json: Value = serde_json::from_str(&String::from_utf8_lossy(&body_bytes))
            .map_err(|e| {
                error!("Failed to parse response JSON: {:?}", e);
                ApiError::InternalServerError
            })?;

        canonicalize_response_model(&mut response_json, &selected_route.response_model_id);

        // ✅ Handle billing HERE, inside completions API
        if let Some(usage) = extract_usage(&response_json) {
            publish_usage_event_internal(
                state,
                user,
                &billing_context,
                usage,
                &successful_provider,
            )
            .await;
        }

        // Return the full response as a single chunk
        let (tx, rx) = mpsc::channel(2); // Need space for FullResponse + Done
        let _ = tx.send(CompletionChunk::FullResponse(response_json)).await;
        let _ = tx.send(CompletionChunk::Done).await;

        return Ok(CompletionStream {
            stream: rx,
            metadata: CompletionMetadata {
                provider_name: successful_provider,
                model_name: billing_context.model_name.clone(),
                is_streaming: false,
            },
        });
    }

    // STREAMING: Complex case - spawn internal processor
    debug!("Processing streaming response with internal billing");
    let (tx_consumer, rx_consumer) = mpsc::channel(100);

    // Spawn INTERNAL task that handles billing
    let state_clone = state.clone();
    let user_clone = user.clone();
    let billing_ctx = billing_context.clone();
    let provider = successful_provider.clone();
    let response_model_id = selected_route.response_model_id.clone();

    tokio::spawn(async move {
        let mut body_stream = res.bytes_stream();
        let mut buffer = Vec::new();
        let mut usage_accumulator = StreamUsageAccumulator::default();

        loop {
            match timeout(
                Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS),
                body_stream.next(),
            )
            .await
            {
                Ok(Some(chunk_result)) => {
                    match chunk_result {
                        Ok(bytes) => {
                            buffer.extend_from_slice(bytes.as_ref());

                            // Parse SSE frames
                            while let Some(frame) = extract_sse_frame(&mut buffer) {
                                if frame == b"[DONE]" {
                                    finalize_stream_usage(
                                        &mut usage_accumulator,
                                        StreamUsageFinalization::ProviderDone,
                                        &state_clone,
                                        &user_clone,
                                        &billing_ctx,
                                        &provider,
                                        &tx_consumer,
                                    )
                                    .await;
                                    let _ = tx_consumer.send(CompletionChunk::Done).await;
                                    return;
                                }

                                match serde_json::from_slice::<Value>(&frame) {
                                    Ok(mut json) => {
                                        // vLLM can report cumulative usage on every delta, while
                                        // providers may add cache details after finish_reason.
                                        // Accumulate observations and publish only when the stream
                                        // lifecycle confirms completion.
                                        usage_accumulator.observe(&json);

                                        canonicalize_response_model(&mut json, &response_model_id);

                                        // Send full JSON chunk to consumer (preserves all metadata)
                                        if tx_consumer
                                            .send(CompletionChunk::StreamChunk(json))
                                            .await
                                            .is_err()
                                        {
                                            finalize_stream_usage(
                                                &mut usage_accumulator,
                                                StreamUsageFinalization::ConsumerDropped,
                                                &state_clone,
                                                &user_clone,
                                                &billing_ctx,
                                                &provider,
                                                &tx_consumer,
                                            )
                                            .await;
                                            return;
                                        }
                                    }
                                    Err(e) => {
                                        error!("Received non-JSON data event. Error: {:?}", e);
                                        finalize_stream_usage(
                                            &mut usage_accumulator,
                                            StreamUsageFinalization::InvalidData,
                                            &state_clone,
                                            &user_clone,
                                            &billing_ctx,
                                            &provider,
                                            &tx_consumer,
                                        )
                                        .await;
                                        let _ = tx_consumer
                                            .send(CompletionChunk::Error(
                                                "Invalid JSON".to_string(),
                                            ))
                                            .await;
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Stream error: {:?}", e);
                            finalize_stream_usage(
                                &mut usage_accumulator,
                                StreamUsageFinalization::TransportError,
                                &state_clone,
                                &user_clone,
                                &billing_ctx,
                                &provider,
                                &tx_consumer,
                            )
                            .await;
                            let _ = tx_consumer
                                .send(CompletionChunk::Error(e.to_string()))
                                .await;
                            return;
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended without explicit [DONE]
                    finalize_stream_usage(
                        &mut usage_accumulator,
                        StreamUsageFinalization::EndOfStream,
                        &state_clone,
                        &user_clone,
                        &billing_ctx,
                        &provider,
                        &tx_consumer,
                    )
                    .await;
                    let _ = tx_consumer.send(CompletionChunk::Done).await;
                    return;
                }
                Err(_) => {
                    // Timeout waiting for next chunk
                    error!("Stream chunk timeout after {}s", STREAM_CHUNK_TIMEOUT_SECS);
                    finalize_stream_usage(
                        &mut usage_accumulator,
                        StreamUsageFinalization::Timeout,
                        &state_clone,
                        &user_clone,
                        &billing_ctx,
                        &provider,
                        &tx_consumer,
                    )
                    .await;
                    let _ = tx_consumer
                        .send(CompletionChunk::Error("Stream timeout".to_string()))
                        .await;
                    return;
                }
            }
        }
    });

    Ok(CompletionStream {
        stream: rx_consumer,
        metadata: CompletionMetadata {
            provider_name: successful_provider,
            model_name: billing_context.model_name.clone(),
            is_streaming: true,
        },
    })
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
        completion_tokens: observed.completion_tokens.unwrap_or(0),
        cached_prompt_tokens: observed
            .cached_prompt_tokens
            .map(|tokens| tokens.min(prompt_tokens)),
    })
}

fn extract_usage_observation(json: &Value) -> Option<CompletionUsageObservation> {
    let usage_json = json.get("usage")?.as_object()?;
    let prompt_tokens = usage_json
        .get("prompt_tokens")
        .and_then(|v| v.as_i64())
        .map(|tokens| tokens.clamp(0, i32::MAX as i64) as i32);

    let cached_prompt_tokens = usage_json
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|v| v.as_i64())
        .map(|tokens| tokens.clamp(0, i32::MAX as i64) as i32);

    Some(CompletionUsageObservation {
        prompt_tokens,
        completion_tokens: usage_json
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .map(|tokens| tokens.clamp(0, i32::MAX as i64) as i32),
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

fn apply_provider_managed_request_fields(
    body: &mut serde_json::Map<String, Value>,
    provider_name: &str,
    user_uuid: Uuid,
    cache_policy: &CompletionCachePolicy<'_>,
) {
    if body.remove(PROVIDER_MANAGED_CACHE_SALT_FIELD).is_some() {
        debug!("Stripped provider-managed completion request field: cache_salt");
    }

    let replaced_user_cache_secret = body
        .remove(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD)
        .is_some();

    if provider_name == ProviderName::Tinfoil.as_str() {
        let user_cache_secret = match cache_policy {
            CompletionCachePolicy::LegacyV1 => tinfoil_user_cache_secret(user_uuid),
            CompletionCachePolicy::BoundV2(namespace) => namespace.tinfoil_user_cache_secret(),
        };
        body.insert(
            PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD.to_string(),
            json!(user_cache_secret),
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
    if provider_name != ProviderName::Continuum.as_str() {
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
/// This function publishes usage events to both the database and SQS
async fn publish_usage_event_internal(
    state: &Arc<AppState>,
    user: &User,
    billing_context: &BillingContext,
    usage: CompletionUsage,
    provider_name: &str,
) {
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return;
    }

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
        billing_context.model_name,
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
    let is_api_request = billing_context.auth_method == AuthMethod::ApiKey;
    let provider_name = provider_name.to_string();
    let model_name = billing_context.model_name.clone();

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
    user_id: Uuid,
    usage: CompletionUsage,
    estimated_cost: BigDecimal,
    is_api_request: bool,
    provider_name: String,
    model_name: String,
) -> UsageEvent {
    UsageEvent {
        event_id: Uuid::new_v4(),
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
    transport_session: &TransportSession,
    json: &Value,
) -> Result<Event, ApiError> {
    let json_str = json.to_string();
    let encrypted_data = transport_session
        .encrypt_response_bytes(state, json_str.as_bytes())
        .await
        .map_err(|e| {
            error!("Failed to encrypt SSE event data: {:?}", e);
            ApiError::InternalServerError
        })?;

    let base64_encrypted = general_purpose::STANDARD.encode(&encrypted_data);
    Ok(Event::default().data(base64_encrypted))
}

async fn proxy_models(
    State(state): State<Arc<AppState>>,
    axum::Extension(session_id): axum::Extension<TransportSession>,
    user: Option<axum::Extension<User>>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let _ = user;
    let models_response = openai_models_data(&state, UnaryProviderPolicy::LEGACY_V1).await?;
    encrypt_response(&state, &session_id, &models_response).await
}

async fn openai_models_data(
    state: &Arc<AppState>,
    provider_policy: UnaryProviderPolicy,
) -> Result<Value, ApiError> {
    let proxy_config = state.proxy_router.get_completion_proxy();
    if proxy_config.provider_name == "tinfoil" {
        Ok(openai_models_response())
    } else {
        fetch_provider_models(
            &state.provider_client,
            &proxy_config,
            provider_policy.response_body,
        )
        .await
    }
}

/// Fetch the OpenAI-shaped model listing without applying either transport's
/// response encryption. The v2 adapter owns the bounded logical response.
pub(crate) async fn openai_models_v2_data(state: &Arc<AppState>) -> Result<Value, ApiError> {
    openai_models_data(state, UnaryProviderPolicy::V2).await
}

async fn fetch_provider_models(
    client: &ProviderClient,
    proxy_config: &ProxyConfig,
    response_body_policy: ProviderResponseBodyPolicy,
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
        match collect_provider_error_body_for_log(res, response_body_policy, status).await {
            ProviderErrorBodyForLog::Legacy(error_msg) => error!(
                "Provider {} returned non-success status for models: {} - {}",
                proxy_config.provider_name, status, error_msg
            ),
            ProviderErrorBodyForLog::StatusOnly => error!(
                "Provider {} returned non-success status for models: {}",
                proxy_config.provider_name, status
            ),
        }
        return Err(ApiError::InternalServerError);
    }

    let body_bytes = collect_provider_response_body(res, response_body_policy)
        .await
        .map_err(|error| {
            error!("Failed to read models response body: {error}");
            error
        })?;
    preflight_provider_json(&body_bytes, response_body_policy).inspect_err(|_| {
        error!("Models provider response failed structural JSON preflight");
    })?;

    serde_json::from_slice(&body_bytes).map_err(|e| {
        error!("Failed to parse models response: {:?}", e);
        ApiError::InternalServerError
    })
}

async fn proxy_model_catalog(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<TransportSession>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(_body): axum::Extension<()>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let catalog_response = openai_model_catalog_data(&state, &user).await;
    encrypt_response(&state, &session_id, &catalog_response).await
}

/// Build the entitled model catalog independently of the outer transport.
pub(crate) async fn openai_model_catalog_data(state: &Arc<AppState>, user: &User) -> Value {
    let billing_access = state.chat_billing_access(user.uuid, false).await;
    let model_plan = ModelPlan::from_is_paid(
        billing_access.is_some_and(crate::billing::ChatBillingAccess::is_paid),
    );
    let alias_targets = state.model_alias_targets(user.uuid, model_plan).await;
    model_catalog_response(alias_targets)
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
    provider_policy: UnaryProviderPolicy,
) -> Result<Value, ApiError> {
    let max_cycles = match provider_policy.retry {
        ProviderRetryPolicy::LegacyV1 => 3,
        ProviderRetryPolicy::NoAmbiguousRetry => 1,
    };
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

        match send_transcription_request(
            client,
            primary_provider,
            &primary_model,
            params,
            provider_policy.response_body,
        )
        .await
        {
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

        if provider_policy.retry == ProviderRetryPolicy::LegacyV1 {
            let Some(fallback_provider) = fallback_provider else {
                continue;
            };
            debug!(
                "Cycle {}: Trying fallback provider {} for transcription",
                cycle + 1,
                fallback_provider.provider_name
            );

            let fallback_model =
                transcription_model_for_provider(model_name, &fallback_provider.provider_name);

            match send_transcription_request(
                client,
                fallback_provider,
                &fallback_model,
                params,
                provider_policy.response_body,
            )
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
    axum::Extension(session_id): axum::Extension<TransportSession>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(transcription_request): axum::Extension<TranscriptionRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let response = openai_transcription_data(
        &state,
        &user,
        transcription_request,
        UnaryProviderPolicy::LEGACY_V1,
    )
    .await?;
    encrypt_response(&state, &session_id, &response).await
}

async fn openai_transcription_data(
    state: &Arc<AppState>,
    user: &User,
    mut transcription_request: TranscriptionRequest,
    provider_policy: UnaryProviderPolicy,
) -> Result<Value, ApiError> {
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

    let bounded_chunking = provider_policy.retry == ProviderRetryPolicy::NoAmbiguousRetry;

    // V2 moves the encoded input out of the request and drops it immediately
    // after decoding. The v1 path deliberately retains its existing request
    // shape and lifetime.
    let encoded_file = bounded_chunking.then(|| std::mem::take(&mut transcription_request.file));
    let encoded_file_for_decode = encoded_file
        .as_deref()
        .unwrap_or(transcription_request.file.as_str());
    let file_bytes = general_purpose::STANDARD
        .decode(encoded_file_for_decode)
        .map_err(|e| {
            error!("Failed to decode base64 audio file: {:?}", e);
            ApiError::BadRequest
        })?;
    drop(encoded_file);

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
    let chunks = if bounded_chunking {
        splitter.split_audio_bounded(
            file_bytes,
            &transcription_request.content_type,
            V2_MAX_TRANSCRIPTION_CHUNKS,
            V2_MAX_TRANSCRIPTION_CHUNK_BYTES,
        )
    } else {
        splitter.split_audio(&file_bytes, &transcription_request.content_type)
    }
    .map_err(|e| {
        error!("Failed to split audio: {}", e);
        if bounded_chunking
            && matches!(
                e.as_str(),
                "WAV file requires too many bounded audio chunks"
                    | "WAV chunks exceed bounded aggregate chunk bytes"
                    | "Audio exceeds bounded aggregate chunk bytes"
            )
        {
            ApiError::PayloadTooLarge
        } else {
            ApiError::InternalServerError
        }
    })?;
    // V2 admits one bounded logical response, not one full response budget per
    // parallel audio chunk. Dividing only the bounded policy keeps aggregate
    // provider bytes under the same 28 MiB ceiling while v1 stays unbounded.
    let provider_policy = provider_policy.divide_bounded_response_across(chunks.len());

    info!("Processing {} chunk(s)", chunks.len());

    // Keep v1's parallel chunk dispatch unchanged. V2 processes one owned
    // chunk and one multipart body at a time so the aggregate audio working set
    // remains inside the gateway's provider reservation.
    let results: Vec<Result<(usize, Value), ApiError>> = if bounded_chunking {
        let mut results = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let result = transcribe_audio_chunk(
                &client,
                &default_proxy,
                &tinfoil_proxy,
                &transcription_request,
                chunk,
                provider_policy,
            )
            .await;
            let failed = result.is_err();
            results.push(result);
            if failed {
                break;
            }
        }
        results
    } else {
        futures::future::join_all(chunks.into_iter().map(|chunk| {
            transcribe_audio_chunk(
                &client,
                &default_proxy,
                &tinfoil_proxy,
                &transcription_request,
                chunk,
                provider_policy,
            )
        }))
        .await
    };

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

    Ok(response)
}

async fn transcribe_audio_chunk(
    client: &ProviderClient,
    default_proxy: &ProxyConfig,
    tinfoil_proxy: &ProxyConfig,
    transcription_request: &TranscriptionRequest,
    chunk: crate::web::audio_utils::AudioChunk,
    provider_policy: UnaryProviderPolicy,
) -> Result<(usize, Value), ApiError> {
    let chunk_size = chunk.data.len();
    info!(
        "Processing chunk {} (size: {} bytes)",
        chunk.index, chunk_size
    );

    let (primary_provider, fallback_provider) =
        if chunk_size > TINFOIL_MAX_SIZE && tinfoil_proxy.provider_name == "tinfoil" {
            info!(
                "Chunk {} size {} bytes exceeds Tinfoil's 0.5MB limit, using fallback only",
                chunk.index, chunk_size
            );
            (default_proxy, None)
        } else {
            (tinfoil_proxy, Some(default_proxy))
        };

    let params = TranscriptionParams {
        audio_data: &chunk.data,
        filename: &transcription_request.filename,
        content_type: &transcription_request.content_type,
        language: transcription_request.language.as_deref(),
        prompt: transcription_request.prompt.as_deref(),
        response_format: &transcription_request.response_format,
        temperature: transcription_request.temperature,
    };

    match send_transcription_with_retries(
        client,
        primary_provider,
        fallback_provider,
        &transcription_request.model,
        &params,
        provider_policy,
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
}

/// Execute transcription without transport encryption. V2 performs no
/// post-send provider retry or fallback and bounds each provider JSON response
/// before structural preflight and serde allocation.
pub(crate) async fn openai_transcription_v2_data(
    state: &Arc<AppState>,
    user: &User,
    transcription_request: TranscriptionRequest,
) -> Result<Value, ApiError> {
    openai_transcription_data(state, user, transcription_request, UnaryProviderPolicy::V2).await
}

/// Count and append form values after removing CRLF without allocating an
/// intermediate sanitized copy. This matters when an optional transcription
/// prompt consumes most of the bounded logical request.
fn sanitized_form_field_bytes(value: &str) -> usize {
    value
        .as_bytes()
        .iter()
        .filter(|byte| !matches!(byte, b'\r' | b'\n'))
        .count()
}

fn extend_sanitized_form_field(output: &mut Vec<u8>, value: &str) {
    let mut retained_start = 0_usize;
    for (index, byte) in value.as_bytes().iter().enumerate() {
        if matches!(byte, b'\r' | b'\n') {
            output.extend_from_slice(&value.as_bytes()[retained_start..index]);
            retained_start = index + 1;
        }
    }
    output.extend_from_slice(&value.as_bytes()[retained_start..]);
}

fn sanitized_filename_bytes(value: &str) -> usize {
    value.chars().count()
}

fn extend_sanitized_filename(output: &mut Vec<u8>, value: &str) {
    output.extend(value.chars().map(|character| {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            character as u8
        } else {
            b'_'
        }
    }));
}

fn checked_byte_sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .try_fold(0_usize, |total, part| total.checked_add(*part))
}

fn multipart_text_field_bytes(boundary: &str, name: &str, value_bytes: usize) -> Option<usize> {
    checked_byte_sum(&[
        2,
        boundary.len(),
        2,
        b"Content-Disposition: form-data; name=\"".len(),
        name.len(),
        b"\"\r\n\r\n".len(),
        value_bytes,
        2,
    ])
}

struct TranscriptionMultipartLengths {
    model_bytes: usize,
    filename_bytes: usize,
    content_type_bytes: usize,
    language_bytes: Option<usize>,
    prompt_bytes: Option<usize>,
    response_format_bytes: usize,
    temperature_bytes: Option<usize>,
    audio_bytes: usize,
}

fn transcription_multipart_bytes(
    boundary: &str,
    lengths: &TranscriptionMultipartLengths,
) -> Option<usize> {
    let mut total = multipart_text_field_bytes(boundary, "model", lengths.model_bytes)?;
    let file_part = checked_byte_sum(&[
        2,
        boundary.len(),
        2,
        b"Content-Disposition: form-data; name=\"file\"; filename=\"".len(),
        lengths.filename_bytes,
        b"\"\r\n".len(),
        b"Content-Type: ".len(),
        lengths.content_type_bytes,
        b"\r\n\r\n".len(),
        lengths.audio_bytes,
        2,
    ])?;
    total = total.checked_add(file_part)?;
    if let Some(language_bytes) = lengths.language_bytes {
        total = total.checked_add(multipart_text_field_bytes(
            boundary,
            "language",
            language_bytes,
        )?)?;
    }
    if let Some(prompt_bytes) = lengths.prompt_bytes {
        total = total.checked_add(multipart_text_field_bytes(
            boundary,
            "prompt",
            prompt_bytes,
        )?)?;
    }
    total = total.checked_add(multipart_text_field_bytes(
        boundary,
        "response_format",
        lengths.response_format_bytes,
    )?)?;
    if let Some(temperature_bytes) = lengths.temperature_bytes {
        total = total.checked_add(multipart_text_field_bytes(
            boundary,
            "temperature",
            temperature_bytes,
        )?)?;
    }
    total.checked_add(checked_byte_sum(&[2, boundary.len(), 4])?)
}

async fn send_transcription_request(
    client: &ProviderClient,
    provider: &ProxyConfig,
    model: &str,
    params: &TranscriptionParams<'_>,
    response_body_policy: ProviderResponseBodyPolicy,
) -> Result<Value, ApiError> {
    // Build multipart form data
    let boundary = format!("----FormBoundary{}", Uuid::new_v4().simple());
    let temperature = params.temperature.map(|value| value.to_string());

    let bounded_form_bytes = match response_body_policy {
        ProviderResponseBodyPolicy::Unbounded => None,
        ProviderResponseBodyPolicy::Bounded { .. } => {
            let form_bytes = transcription_multipart_bytes(
                &boundary,
                &TranscriptionMultipartLengths {
                    model_bytes: sanitized_form_field_bytes(model),
                    filename_bytes: sanitized_filename_bytes(params.filename),
                    content_type_bytes: sanitized_form_field_bytes(params.content_type),
                    language_bytes: params.language.map(sanitized_form_field_bytes),
                    prompt_bytes: params.prompt.map(sanitized_form_field_bytes),
                    response_format_bytes: sanitized_form_field_bytes(params.response_format),
                    temperature_bytes: temperature.as_deref().map(str::len),
                    audio_bytes: params.audio_data.len(),
                },
            )
            .ok_or(ApiError::PayloadTooLarge)?;
            if form_bytes > V2_MAX_TRANSCRIPTION_MULTIPART_BYTES {
                return Err(ApiError::PayloadTooLarge);
            }
            Some(form_bytes)
        }
    };
    let mut form_data = bounded_form_bytes
        .map(Vec::with_capacity)
        .unwrap_or_default();

    // Add model field (sanitized to prevent header injection)
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    extend_sanitized_form_field(&mut form_data, model);
    form_data.extend_from_slice(b"\r\n");

    // Add file field with sanitized filename to prevent header injection
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"file\"; filename=\"");
    extend_sanitized_filename(&mut form_data, params.filename);
    form_data.extend_from_slice(b"\"\r\nContent-Type: ");
    extend_sanitized_form_field(&mut form_data, params.content_type);
    form_data.extend_from_slice(b"\r\n\r\n");
    form_data.extend_from_slice(params.audio_data);
    form_data.extend_from_slice(b"\r\n");

    // Add optional fields (sanitized to prevent header injection)
    if let Some(language) = params.language {
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        extend_sanitized_form_field(&mut form_data, language);
        form_data.extend_from_slice(b"\r\n");
    }
    if let Some(prompt) = params.prompt {
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"prompt\"\r\n\r\n");
        extend_sanitized_form_field(&mut form_data, prompt);
        form_data.extend_from_slice(b"\r\n");
    }
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data
        .extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
    extend_sanitized_form_field(&mut form_data, params.response_format);
    form_data.extend_from_slice(b"\r\n");
    if let Some(temperature) = temperature.as_deref() {
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data
            .extend_from_slice(b"Content-Disposition: form-data; name=\"temperature\"\r\n\r\n");
        form_data.extend_from_slice(temperature.as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }

    // End boundary
    form_data.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
    debug_assert!(bounded_form_bytes.is_none_or(|expected| form_data.len() == expected));

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
                let body_bytes = collect_provider_response_body(res, response_body_policy)
                    .await
                    .map_err(|error| {
                        error!("Failed to read transcription response body: {error}");
                        error
                    })?;
                preflight_provider_json(&body_bytes, response_body_policy).inspect_err(|_| {
                    error!("Transcription provider response failed structural JSON preflight");
                })?;

                let response_json: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
                    error!("Failed to parse transcription response: {:?}", e);
                    ApiError::InternalServerError
                })?;

                Ok(response_json)
            } else {
                let status = res.status_code();
                match collect_provider_error_body_for_log(res, response_body_policy, status).await {
                    ProviderErrorBodyForLog::Legacy(error_msg) => error!(
                        "Provider {} returned transcription error: {} - {}",
                        provider.provider_name, status, error_msg
                    ),
                    ProviderErrorBodyForLog::StatusOnly => error!(
                        "Provider {} returned transcription error: {}",
                        provider.provider_name, status
                    ),
                }
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
    axum::Extension(session_id): axum::Extension<TransportSession>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(tts_request): axum::Extension<TTSRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let response =
        openai_tts_data(&state, &user, tts_request, UnaryProviderPolicy::LEGACY_V1).await?;
    encrypt_response(&state, &session_id, &response).await
}

async fn openai_tts_data(
    state: &Arc<AppState>,
    user: &User,
    tts_request: TTSRequest,
    provider_policy: UnaryProviderPolicy,
) -> Result<Value, ApiError> {
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
    ensure_paid_tts_access(state, user).await?;

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
        let body_bytes = collect_provider_response_body(res, provider_policy.response_body)
            .await
            .map_err(|error| {
                error!("Failed to read TTS response body: {error}");
                error
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

    let (response_payload, is_json_response) = build_tts_response_payload_with_policy(
        &body_bytes,
        &response_content_type,
        provider_policy.retry == ProviderRetryPolicy::NoAmbiguousRetry,
    );
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

    Ok(response_payload)
}

/// Execute speech synthesis without transport encryption. The raw provider
/// audio is capped so padded base64 plus the JSON wrapper fits the v2 logical
/// response ceiling.
pub(crate) async fn openai_tts_v2_data(
    state: &Arc<AppState>,
    user: &User,
    tts_request: TTSRequest,
) -> Result<Value, ApiError> {
    openai_tts_data(state, user, tts_request, UnaryProviderPolicy::V2_TTS).await
}

async fn proxy_embeddings(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<TransportSession>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(auth_method): axum::Extension<AuthMethod>,
    axum::Extension(embedding_request): axum::Extension<EmbeddingRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    let response = openai_embeddings_data(
        &state,
        &user,
        auth_method,
        embedding_request,
        UnaryProviderPolicy::LEGACY_V1,
    )
    .await?;
    encrypt_response(&state, &session_id, &response).await
}

async fn openai_embeddings_data(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    embedding_request: EmbeddingRequest,
    provider_policy: UnaryProviderPolicy,
) -> Result<Value, ApiError> {
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
        match collect_provider_error_body_for_log(res, provider_policy.response_body, status).await
        {
            ProviderErrorBodyForLog::Legacy(error_msg) => error!(
                "Embeddings proxy returned non-success status: {} - {}",
                status, error_msg
            ),
            ProviderErrorBodyForLog::StatusOnly => {
                error!("Embeddings proxy returned non-success status: {}", status)
            }
        }
        return Err(ApiError::InternalServerError);
    }

    // Parse response
    let body_bytes = collect_provider_response_body(res, provider_policy.response_body)
        .await
        .map_err(|error| {
            error!("Failed to read embeddings response body: {error}");
            error
        })?;
    preflight_provider_json(&body_bytes, provider_policy.response_body).inspect_err(|_| {
        error!("Embeddings provider response failed structural JSON preflight");
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
            let billing_context = BillingContext::new(auth_method, embedding_request.model.clone());
            let embedding_usage = CompletionUsage {
                prompt_tokens,
                completion_tokens: 0, // Embeddings don't have completion tokens
                cached_prompt_tokens: None,
            };
            publish_usage_event_internal(
                state,
                user,
                &billing_context,
                embedding_usage,
                &proxy_config.provider_name,
            )
            .await;
        }
    }

    Ok(response_json)
}

/// Execute embeddings without transport encryption. Provider JSON is bounded
/// and structurally preflighted before serde allocation.
pub(crate) async fn openai_embeddings_v2_data(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    embedding_request: EmbeddingRequest,
) -> Result<Value, ApiError> {
    openai_embeddings_data(
        state,
        user,
        auth_method,
        embedding_request,
        UnaryProviderPolicy::V2,
    )
    .await
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum BoundedProviderResponseBodyError {
    #[error("provider response body read failed")]
    Read,
    #[error("provider response body exceeded the {limit_bytes}-byte limit")]
    TooLarge { limit_bytes: usize },
}

enum ProviderErrorBodyForLog {
    Legacy(String),
    StatusOnly,
}

async fn collect_provider_error_body_for_log(
    response: ProviderResponse,
    response_body_policy: ProviderResponseBodyPolicy,
    status: u16,
) -> ProviderErrorBodyForLog {
    match response_body_policy.error_body() {
        ProviderErrorBodyPolicy::LegacyBody => {
            let error_msg = response
                .bytes()
                .await
                .ok()
                .map(|body| String::from_utf8_lossy(&body).to_string())
                .unwrap_or_else(|| status.to_string());
            ProviderErrorBodyForLog::Legacy(error_msg)
        }
        ProviderErrorBodyPolicy::StatusOnly { drain_limit_bytes } => {
            // V2 provider errors are status-only. Consume at most a small
            // prefix for connection hygiene without copying or retaining it,
            // and never make provider-controlled error text loggable.
            let _ =
                drain_bounded_provider_response_body(response.bytes_stream(), drain_limit_bytes)
                    .await;
            ProviderErrorBodyForLog::StatusOnly
        }
    }
}

async fn collect_provider_response_body(
    response: ProviderResponse,
    policy: ProviderResponseBodyPolicy,
) -> Result<bytes::Bytes, ApiError> {
    match policy {
        ProviderResponseBodyPolicy::Unbounded => response
            .bytes()
            .await
            .map_err(|_| ApiError::InternalServerError),
        ProviderResponseBodyPolicy::Bounded { limit_bytes, .. } => {
            collect_bounded_provider_response_body(response.bytes_stream(), limit_bytes)
                .await
                .map(bytes::Bytes::from)
                .map_err(|error| match error {
                    BoundedProviderResponseBodyError::Read => ApiError::InternalServerError,
                    BoundedProviderResponseBodyError::TooLarge { .. } => ApiError::PayloadTooLarge,
                })
        }
    }
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

async fn drain_bounded_provider_response_body<S>(
    mut body_stream: S,
    limit_bytes: usize,
) -> Result<usize, BoundedProviderResponseBodyError>
where
    S: futures::Stream<Item = Result<bytes::Bytes, String>> + Unpin,
{
    let mut drained_bytes = 0_usize;

    while drained_bytes < limit_bytes {
        let Some(chunk) = body_stream.next().await else {
            break;
        };
        let chunk = chunk.map_err(|_| BoundedProviderResponseBodyError::Read)?;
        let remaining = limit_bytes - drained_bytes;
        drained_bytes += chunk.len().min(remaining);
        if chunk.len() >= remaining {
            break;
        }
    }

    Ok(drained_bytes)
}

/// Helper function to try a provider once
async fn try_provider(
    client: &ProviderClient,
    proxy_config: &ProxyConfig,
    body_json: String,
    headers: &HeaderMap,
) -> Result<ProviderResponse, ProviderRequestError> {
    debug!("Making request to {}", proxy_config.provider_name);

    match client
        .send(
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
        .await
    {
        Ok(response) => {
            if response.is_success() {
                Ok(response)
            } else {
                let status = response.status_code();
                error!(
                    "Provider {} returned non-success status: {}",
                    proxy_config.provider_name, status
                );
                // Drain only a bounded amount and never log provider response
                // bodies because they may echo user content.
                let _ = collect_bounded_provider_response_body(
                    response.bytes_stream(),
                    MAX_BOUNDED_PROVIDER_RESPONSE_BYTES,
                )
                .await;
                Err(ProviderRequestError::Send(format!(
                    "Provider {} returned status {}",
                    proxy_config.provider_name, status
                )))
            }
        }
        Err(e) => {
            error!(
                "Failed to send request to {}: {:?}",
                proxy_config.provider_name, e
            );
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_provider_policies_keep_v1_legacy_and_v2_bounded_nonretrying() {
        assert_eq!(
            UnaryProviderPolicy::LEGACY_V1,
            UnaryProviderPolicy {
                response_body: ProviderResponseBodyPolicy::Unbounded,
                retry: ProviderRetryPolicy::LegacyV1,
            }
        );
        assert_eq!(
            UnaryProviderPolicy::V2,
            UnaryProviderPolicy {
                response_body: ProviderResponseBodyPolicy::Bounded {
                    limit_bytes: V2_MAX_PROVIDER_RESPONSE_BYTES,
                    max_json_structural_tokens: Some(V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS,),
                },
                retry: ProviderRetryPolicy::NoAmbiguousRetry,
            }
        );
        assert_eq!(
            UnaryProviderPolicy::V2_TTS,
            UnaryProviderPolicy {
                response_body: ProviderResponseBodyPolicy::Bounded {
                    limit_bytes: V2_MAX_TTS_PROVIDER_RESPONSE_BYTES,
                    max_json_structural_tokens: None,
                },
                retry: ProviderRetryPolicy::NoAmbiguousRetry,
            }
        );
        assert_eq!(
            UnaryProviderPolicy::LEGACY_V1.response_body.error_body(),
            ProviderErrorBodyPolicy::LegacyBody
        );
        assert_eq!(
            UnaryProviderPolicy::V2.response_body.error_body(),
            ProviderErrorBodyPolicy::StatusOnly {
                drain_limit_bytes: MAX_BOUNDED_PROVIDER_RESPONSE_BYTES,
            }
        );
    }

    #[test]
    fn v2_unary_chat_helper_accepts_only_absent_or_boolean_false_stream() {
        assert!(ensure_nonstream_chat_completion(&json!({"stream": false})).is_ok());
        assert!(ensure_nonstream_chat_completion(&json!({})).is_ok());
        for rejected in [
            json!({"stream": true}),
            json!({"stream": null}),
            json!({"stream": "false"}),
            json!({"stream": 0}),
            json!({"stream": {}}),
            json!({"stream": []}),
        ] {
            assert!(ensure_nonstream_chat_completion(&rejected).is_err());
        }
    }

    #[test]
    fn v2_provider_json_preflight_bounds_depth_and_structure_only_for_v2() {
        let too_deep = format!(
            "{}0{}",
            "[".repeat(V2_MAX_PROVIDER_JSON_DEPTH + 1),
            "]".repeat(V2_MAX_PROVIDER_JSON_DEPTH + 1)
        );
        assert!(matches!(
            preflight_provider_json(too_deep.as_bytes(), UnaryProviderPolicy::V2.response_body),
            Err(ApiError::PayloadTooLarge)
        ));
        assert!(preflight_provider_json(
            too_deep.as_bytes(),
            UnaryProviderPolicy::LEGACY_V1.response_body
        )
        .is_ok());

        let mut too_many_tokens =
            String::with_capacity(V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS.saturating_add(3));
        too_many_tokens.push('[');
        for index in 0..=V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS {
            if index > 0 {
                too_many_tokens.push(',');
            }
            too_many_tokens.push('0');
        }
        too_many_tokens.push(']');
        assert!(matches!(
            preflight_provider_json(
                too_many_tokens.as_bytes(),
                UnaryProviderPolicy::V2.response_body
            ),
            Err(ApiError::PayloadTooLarge)
        ));
    }

    #[test]
    fn bounded_transcription_chunks_share_one_provider_response_budget() {
        let divided = UnaryProviderPolicy::V2.divide_bounded_response_across(4);
        let per_chunk_structural_tokens = V2_MAX_PROVIDER_JSON_STRUCTURAL_TOKENS / 4;
        assert_eq!(
            divided.response_body,
            ProviderResponseBodyPolicy::Bounded {
                limit_bytes: V2_MAX_PROVIDER_RESPONSE_BYTES / 4,
                max_json_structural_tokens: Some(per_chunk_structural_tokens),
            }
        );
        assert_eq!(divided.retry, ProviderRetryPolicy::NoAmbiguousRetry);

        let numeric_array = |elements: usize| {
            let mut json = String::with_capacity(elements.saturating_mul(2).saturating_add(1));
            json.push('[');
            for index in 0..elements {
                if index > 0 {
                    json.push(',');
                }
                json.push('0');
            }
            json.push(']');
            json
        };
        let exact_limit = numeric_array(per_chunk_structural_tokens - 1);
        assert!(preflight_provider_json(exact_limit.as_bytes(), divided.response_body).is_ok());
        let over_limit = numeric_array(per_chunk_structural_tokens);
        assert!(matches!(
            preflight_provider_json(over_limit.as_bytes(), divided.response_body),
            Err(ApiError::PayloadTooLarge)
        ));

        assert_eq!(
            UnaryProviderPolicy::LEGACY_V1
                .divide_bounded_response_across(4)
                .response_body,
            ProviderResponseBodyPolicy::Unbounded
        );
    }

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
    async fn bounded_provider_error_drain_stops_without_polling_past_limit() {
        let body_stream = futures::stream::iter([
            Ok::<_, String>(bytes::Bytes::from_static(b"abcd")),
            Ok(bytes::Bytes::from_static(b"efgh")),
            Err("must not be polled after the bounded prefix".to_string()),
        ]);

        assert_eq!(
            drain_bounded_provider_response_body(body_stream, 6).await,
            Ok(6)
        );
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
        let (response, is_json_response) =
            build_tts_response_payload_with_policy(&audio_bytes, "audio/flac", false);
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

        let (response, is_json_response) = build_tts_response_payload_with_policy(
            body,
            "application/problem+json; charset=utf-8",
            false,
        );
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

        apply_provider_managed_request_fields(
            &mut body,
            ProviderName::Tinfoil.as_str(),
            user_uuid,
            &CompletionCachePolicy::LegacyV1,
        );

        assert_eq!(body.get("model"), Some(&json!("kimi-k2-6")));
        assert_eq!(body.get("messages"), Some(&json!([])));
        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));
        assert_eq!(
            body.get(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD),
            Some(&json!(tinfoil_user_cache_secret(user_uuid)))
        );
    }

    #[test]
    fn bound_v2_cache_namespace_replaces_legacy_and_client_values() {
        let user_uuid = Uuid::from_u128(42);
        let root = crate::provider_cache::CacheNamespaceRoot::from_bytes([0x55; 32]);
        let namespace = crate::provider_cache::derive_tinfoil_cache_namespace(&root, user_uuid);
        let expected = namespace.tinfoil_user_cache_secret();
        let mut body = serde_json::Map::from_iter([
            ("cache_salt".to_string(), json!("user-supplied")),
            ("user_cache_secret".to_string(), json!("client-controlled")),
        ]);

        apply_provider_managed_request_fields(
            &mut body,
            ProviderName::Tinfoil.as_str(),
            user_uuid,
            &CompletionCachePolicy::BoundV2(&namespace),
        );

        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));
        assert_eq!(
            body.get(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD),
            Some(&json!(expected))
        );
        assert_ne!(
            body.get(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD),
            Some(&json!(tinfoil_user_cache_secret(user_uuid)))
        );
    }

    #[test]
    fn strips_tinfoil_cache_fields_from_non_tinfoil_requests() {
        let user_uuid = Uuid::from_u128(42);
        let root = crate::provider_cache::CacheNamespaceRoot::from_bytes([0x55; 32]);
        let namespace = crate::provider_cache::derive_tinfoil_cache_namespace(&root, user_uuid);
        let mut body = serde_json::Map::from_iter([
            ("cache_salt".to_string(), json!("user-supplied")),
            ("user_cache_secret".to_string(), json!("client-controlled")),
        ]);

        apply_provider_managed_request_fields(
            &mut body,
            ProviderName::Continuum.as_str(),
            user_uuid,
            &CompletionCachePolicy::BoundV2(&namespace),
        );

        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));
        assert!(!body.contains_key(PROVIDER_MANAGED_USER_CACHE_SECRET_FIELD));
    }

    #[test]
    fn server_selected_continuum_route_replaces_caller_salt_with_isolated_salt() {
        let user_uuid = Uuid::from_u128(42);
        let mut body = serde_json::Map::from_iter([
            ("cache_salt".to_string(), json!("caller-controlled")),
            ("messages".to_string(), json!([])),
        ]);

        apply_provider_managed_request_fields(
            &mut body,
            ProviderName::Continuum.as_str(),
            user_uuid,
            &CompletionCachePolicy::LegacyV1,
        );
        assert!(!body.contains_key(PROVIDER_MANAGED_CACHE_SALT_FIELD));

        let server_salt = format!("server-selected-{}", Uuid::new_v4().simple());
        apply_server_selected_cache_isolation(
            &mut body,
            ProviderName::Continuum.as_str(),
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
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.cached_prompt_tokens, Some(42));
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
            completion_tokens: 20,
            cached_prompt_tokens: Some(42),
        };

        let event = build_usage_event(
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
        assert_eq!(usage.completion_tokens, completion_tokens);
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
