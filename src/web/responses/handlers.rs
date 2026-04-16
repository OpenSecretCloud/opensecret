//! Responses API implementation with SSE streaming and dual-stream storage.
//! Phases 4 & 5: Always streams to client while concurrently storing to database.

use crate::{
    billing::BillingError,
    db::DBError,
    encrypt::{decrypt_content, decrypt_string, encrypt_with_key},
    models::responses::{NewUserMessage, ResponseStatus, ResponsesError},
    models::users::User,
    tokens::{count_tokens, model_max_ctx},
    web::{
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        openai::get_chat_completion_response,
        responses::{
            build_prompt, build_usage, constants::*, error_mapping, storage_task, tools,
            ContentPartBuilder, DeletedObjectResponse, MessageContent, MessageContentConverter,
            MessageContentPart, OutputItemBuilder, ResponseBuilder, ResponseEvent, SseEventEmitter,
        },
    },
    ApiError, AppState,
};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    middleware::from_fn_with_state,
    response::sse::{Event, Sse},
    routing::{delete, get, post},
    Extension, Json, Router,
};
use base64::Engine;
use chrono::Utc;
use futures::Stream;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

// Default functions for serde
fn default_store() -> bool {
    true
}

fn default_stream() -> bool {
    true
}

fn apply_responses_model_defaults(chat_request: &mut Value, model: &str) {
    if model != "gemma4-31b" {
        return;
    }

    let Some(obj) = chat_request.as_object_mut() else {
        return;
    };

    obj.insert("include_reasoning".to_string(), json!(true));

    let chat_template_kwargs = obj
        .entry("chat_template_kwargs".to_string())
        .or_insert_with(|| json!({}));

    if let Some(kwargs) = chat_template_kwargs.as_object_mut() {
        kwargs.insert("enable_thinking".to_string(), json!(true));
    } else {
        *chat_template_kwargs = json!({
            "enable_thinking": true
        });
    }
}

const MAPLE_SYSTEM_PROMPT: &str = "You are Maple, a friendly, concise, and helpful assistant. Give direct answers, be honest about uncertainty, and never invent tool use, search results, or sources.";
const MAPLE_WEB_SEARCH_PROMPT: &str = "If the web_search tool is available and the user explicitly asks you to search, look something up, verify, confirm, or check the web, call web_search before answering. Also use web_search when the answer depends on current or time-sensitive information. You may use web_search repeatedly across a single response when needed, but only one tool call at a time and never more than 15 tool calls for one user request. After each tool output, decide whether you have enough information to answer or whether another search is still needed. Prefer to stop searching and answer as soon as you have enough information. Never output raw tool call syntax.";

#[derive(Debug, Clone)]
struct ModelToolCall {
    name: String,
    arguments: Value,
}

#[derive(Debug, Clone, Default)]
struct StreamedToolCall {
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Clone)]
enum AssistantTurnOutcome {
    ToolCall(ModelToolCall),
    Final,
}

fn should_enable_web_search_tool(state: &AppState, body: &ResponsesCreateRequest) -> bool {
    is_tool_choice_allowed(&body.tool_choice)
        && is_web_search_enabled(&body.tools)
        && state.brave_client.is_some()
}

fn build_internal_system_prompt_for_now(
    now: chrono::DateTime<Utc>,
    web_search_enabled: bool,
) -> String {
    let current_utc_date = now.format("%A, %Y-%m-%d").to_string();
    let current_date_prompt = format!(
        "Current UTC date: {current_utc_date}. Use this as today's date for any date-sensitive reasoning."
    );

    if web_search_enabled {
        format!("{MAPLE_SYSTEM_PROMPT}\n\n{current_date_prompt}\n\n{MAPLE_WEB_SEARCH_PROMPT}")
    } else {
        format!("{MAPLE_SYSTEM_PROMPT}\n\n{current_date_prompt}")
    }
}

fn build_internal_system_prompt(web_search_enabled: bool) -> String {
    build_internal_system_prompt_for_now(Utc::now(), web_search_enabled)
}

fn build_provider_tools(request_tools: &Option<Value>) -> Vec<Value> {
    let registry = tools::ToolRegistry::new();

    request_tools
        .as_ref()
        .and_then(|tools| tools.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let tool_name = tool.get("type").and_then(|t| t.as_str())?;
                    registry.get_tool_schema(tool_name).map(|schema| {
                        json!({
                            "type": "function",
                            "function": schema,
                        })
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn build_tool_choice_value(tool_choice: &Option<String>) -> Value {
    match tool_choice.as_deref() {
        Some(choice) if !choice.is_empty() => json!(choice),
        _ => json!("auto"),
    }
}

fn build_model_turn_request(
    body: &ResponsesCreateRequest,
    prompt_messages: &[Value],
    tools_enabled: bool,
) -> Value {
    let mut chat_request = json!({
        "model": body.model,
        "messages": prompt_messages,
        "temperature": body.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        "top_p": body.top_p.unwrap_or(DEFAULT_TOP_P),
        "max_tokens": body.max_output_tokens,
        "stream": true,
        "stream_options": { "include_usage": true }
    });

    if tools_enabled {
        let provider_tools = build_provider_tools(&body.tools);
        if !provider_tools.is_empty() {
            chat_request["tools"] = Value::Array(provider_tools);
            chat_request["tool_choice"] = build_tool_choice_value(&body.tool_choice);
            chat_request["parallel_tool_calls"] = json!(false);
        }
    }

    apply_responses_model_defaults(&mut chat_request, &body.model);
    chat_request
}

fn append_streamed_tool_calls(tool_calls: &mut Vec<StreamedToolCall>, tool_call_delta: &Value) {
    let Some(tool_call_entries) = tool_call_delta.as_array() else {
        return;
    };

    if tool_call_entries.len() > 1 {
        warn!(
            "Model streamed {} tool calls in one chunk; only the first call will be executed in v1",
            tool_call_entries.len()
        );
    }

    for tool_call in tool_call_entries {
        let index = tool_call
            .get("index")
            .and_then(|index| index.as_u64())
            .unwrap_or(0) as usize;

        while tool_calls.len() <= index {
            tool_calls.push(StreamedToolCall::default());
        }

        if let Some(function) = tool_call.get("function") {
            if let Some(name) = function.get("name").and_then(|name| name.as_str()) {
                tool_calls[index].name = Some(name.to_string());
            }

            if let Some(arguments) = function
                .get("arguments")
                .and_then(|arguments| arguments.as_str())
            {
                tool_calls[index].arguments.push_str(arguments);
            }
        }
    }
}

fn finalize_first_model_tool_call(tool_calls: &[StreamedToolCall]) -> Option<ModelToolCall> {
    let tool_call = tool_calls.first()?;
    let name = tool_call.name.clone()?;
    let arguments = serde_json::from_str(&tool_call.arguments).unwrap_or_else(|e| {
        warn!(
            "Failed to parse tool arguments for {} as JSON: {:?}. Using empty object.",
            name, e
        );
        json!({})
    });

    Some(ModelToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::{
        append_streamed_tool_calls, apply_responses_model_defaults,
        build_internal_system_prompt_for_now, build_provider_tools, finalize_first_model_tool_call,
        ClientResponseState, StreamedToolCall, MAPLE_WEB_SEARCH_PROMPT,
    };
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn test_apply_responses_model_defaults_enables_gemma_thinking() {
        let mut chat_request = json!({
            "model": "gemma4-31b",
            "chat_template_kwargs": {
                "foo": "bar"
            }
        });

        apply_responses_model_defaults(&mut chat_request, "gemma4-31b");

        assert_eq!(chat_request["include_reasoning"], true);
        assert_eq!(
            chat_request["chat_template_kwargs"]["enable_thinking"],
            true
        );
        assert_eq!(chat_request["chat_template_kwargs"]["foo"], "bar");
    }

    #[test]
    fn test_apply_responses_model_defaults_skips_other_models() {
        let mut chat_request = json!({
            "model": "gpt-oss-120b"
        });

        apply_responses_model_defaults(&mut chat_request, "gpt-oss-120b");

        assert!(chat_request.get("include_reasoning").is_none());
        assert!(chat_request.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn test_append_streamed_tool_calls_reassembles_arguments() {
        let mut tool_calls = Vec::<StreamedToolCall>::new();

        append_streamed_tool_calls(
            &mut tool_calls,
            &json!([{
                "index": 0,
                "function": {
                    "name": "web_search",
                    "arguments": "{\"query\":\"Don"
                }
            }]),
        );
        append_streamed_tool_calls(
            &mut tool_calls,
            &json!([{
                "index": 0,
                "function": {
                    "arguments": "ald Trump birthday\"}"
                }
            }]),
        );

        let tool_call = finalize_first_model_tool_call(&tool_calls).expect("tool call");
        assert_eq!(tool_call.name, "web_search");
        assert_eq!(tool_call.arguments["query"], "Donald Trump birthday");
    }

    #[test]
    fn test_build_provider_tools_filters_unknown_tools() {
        let tools = build_provider_tools(&Some(json!([
            { "type": "web_search" },
            { "type": "unknown_tool" }
        ])));

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "web_search");
    }

    #[test]
    fn test_build_internal_system_prompt_includes_current_utc_date() {
        let now = Utc
            .with_ymd_and_hms(2026, 4, 15, 12, 0, 0)
            .single()
            .expect("valid UTC timestamp");

        let prompt = build_internal_system_prompt_for_now(now, true);

        assert!(prompt.contains("Current UTC date: Wednesday, 2026-04-15."));
        assert!(prompt.contains(MAPLE_WEB_SEARCH_PROMPT));
    }

    #[test]
    fn test_client_response_state_build_output_items_uses_maple_tool_types() {
        let mut state = ClientResponseState::default();
        let tool_call_id = Uuid::new_v4();
        let tool_output_id = Uuid::new_v4();

        state.push_tool_call(
            tool_call_id,
            "web_search".to_string(),
            json!({ "query": "ufc" }),
        );
        state.push_tool_output(
            tool_output_id,
            tool_call_id,
            "Search Results:\n\n1. Example".to_string(),
        );

        let output_items = state.build_output_items();
        let tool_call_id_str = tool_call_id.to_string();

        assert_eq!(output_items.len(), 2);
        assert_eq!(output_items[0].output_type, "tool_call");
        assert_eq!(
            output_items[0].call_id.as_deref(),
            Some(tool_call_id_str.as_str())
        );
        assert_eq!(output_items[1].output_type, "tool_output");
        assert_eq!(
            output_items[1].call_id.as_deref(),
            Some(tool_call_id_str.as_str())
        );
    }
}

/// Conversation parameter - can be a string UUID or an object with id field
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ConversationParam {
    String(Uuid),
    Object { id: Uuid },
}

/// Input message - can be a simple string or an array of message objects
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputMessage {
    String(String),
    Messages(Vec<MessageInput>),
}

impl InputMessage {
    /// Normalize any input format to our standard format: always MessageContent::Parts
    ///
    /// Also validates that unsupported features are not used (e.g., file_id for images)
    pub fn normalize(self) -> Result<Vec<MessageInput>, ApiError> {
        match self {
            InputMessage::String(s) => {
                // Simple string -> user message with input_text content parts
                Ok(vec![MessageInput {
                    role: ROLE_USER.to_string(),
                    content: MessageContent::Parts(vec![MessageContentPart::InputText { text: s }]),
                }])
            }
            InputMessage::Messages(mut messages) => {
                // Ensure all message content is normalized to Parts format and validated
                for msg in &mut messages {
                    MessageContentConverter::validate_content(&msg.content)?;
                    msg.content = MessageContentConverter::normalize_content(msg.content.clone());
                }
                Ok(messages)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInput {
    pub role: String,
    pub content: MessageContent,
}

/// Request payload for creating a new response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesCreateRequest {
    /// Model to use for the response
    pub model: String,

    /// User's input - can be a string or array of messages
    /// Supports both: "hello" or [{"role": "user", "content": "hello"}]
    pub input: InputMessage,

    /// Conversation to associate with (UUID string or {id: UUID} object) - REQUIRED
    pub conversation: ConversationParam,

    /// System instructions for this response (overrides default user instructions)
    #[serde(default)]
    pub instructions: Option<String>,

    /// Temperature for randomness (0-2)
    pub temperature: Option<f32>,

    /// Top-p for nucleus sampling
    pub top_p: Option<f32>,

    /// Maximum tokens for the response
    pub max_output_tokens: Option<i32>,

    /// Tool choice strategy
    #[serde(default)]
    pub tool_choice: Option<String>,

    /// Tools available for the model
    #[serde(default)]
    pub tools: Option<Value>,

    /// Enable parallel tool calls
    #[serde(default)]
    pub parallel_tool_calls: bool,

    /// Whether to store the conversation (defaults to true)
    #[serde(default = "default_store")]
    pub store: bool,

    /// Arbitrary metadata
    #[serde(default)]
    pub metadata: Option<Value>,

    /// Always stream (defaults to true)
    #[serde(default = "default_stream")]
    pub stream: bool,
}

/// Immediate response returned when creating a new response
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesCreateResponse {
    /// Response ID (UUID)
    pub id: Uuid,

    /// Object type (always "response")
    pub object: &'static str,

    /// Unix timestamp of creation
    pub created_at: i64,

    /// Current status (always "in_progress" for immediate response)
    pub status: String,

    /// Whether this is a background response
    pub background: bool,

    /// Error information (null for successful requests)
    pub error: Option<ResponseError>,

    /// Details about why the response is incomplete
    pub incomplete_details: Option<serde_json::Value>,

    /// Instructions for the model
    pub instructions: Option<String>,

    /// Maximum output tokens
    pub max_output_tokens: Option<i32>,

    /// Maximum tool calls
    pub max_tool_calls: Option<i32>,

    /// Model used for the response
    pub model: String,

    /// Output array (empty for in_progress responses)
    pub output: Vec<OutputItem>,

    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: bool,

    /// Previous response ID if continuing a conversation
    pub previous_response_id: Option<Uuid>,

    /// Prompt cache key
    pub prompt_cache_key: Option<String>,

    /// Reasoning information
    pub reasoning: ReasoningInfo,

    /// Safety identifier
    pub safety_identifier: Option<String>,

    /// Whether the response is stored
    pub store: bool,

    /// Temperature setting
    pub temperature: f32,

    /// Text formatting options
    pub text: TextFormat,

    /// Tool choice setting
    pub tool_choice: String,

    /// Available tools
    pub tools: Vec<serde_json::Value>,

    /// Top logprobs
    pub top_logprobs: i32,

    /// Top-p setting
    pub top_p: f32,

    /// Truncation strategy
    pub truncation: &'static str,

    /// Usage statistics (null for in_progress)
    pub usage: Option<ResponseUsage>,

    /// User identifier
    pub user: Option<String>,

    /// Metadata from the request
    pub metadata: Option<Value>,
}

/// Reasoning information
#[derive(Debug, Clone, Serialize)]
pub struct ReasoningInfo {
    /// Reasoning effort
    pub effort: Option<String>,

    /// Reasoning summary
    pub summary: Option<String>,
}

/// Text formatting options
#[derive(Debug, Clone, Serialize)]
pub struct TextFormat {
    /// Format specification
    pub format: TextFormatSpec,
}

/// Text format specification
#[derive(Debug, Clone, Serialize)]
pub struct TextFormatSpec {
    /// Format type (always "text")
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Output item in the response
#[derive(Debug, Clone, Serialize)]
pub struct OutputItem {
    /// Type of output item
    #[serde(rename = "type")]
    pub output_type: String,

    /// ID of the item
    pub id: String,

    /// Status of the item
    pub status: String,

    /// Role (for message type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content array (for message type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentPart>>,

    /// Tool call ID (for tool_call / tool_output types)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,

    /// Tool/function name (for tool_call type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool arguments JSON (for tool_call type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,

    /// Tool output payload (for tool_output type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Response error structure
#[derive(Debug, Clone, Serialize)]
pub struct ResponseError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error message
    pub message: String,
}

/// SSE Event wrapper for response.created
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCreatedEvent {
    /// Event type (always "response.created")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The response payload
    pub response: ResponsesCreateResponse,

    /// Sequence number for ordering
    pub sequence_number: i32,
}

/// SSE Event wrapper for response.output_text.delta
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputTextDeltaEvent {
    /// Event type (always "response.output_text.delta")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The content delta
    pub delta: String,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// Log probabilities (empty array for now)
    pub logprobs: Vec<serde_json::Value>,
}

/// SSE Event wrapper for response.completed
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCompletedEvent {
    /// Event type (always "response.completed")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The final response payload
    pub response: ResponsesCreateResponse,

    /// Sequence number for ordering
    pub sequence_number: i32,
}

/// SSE Event wrapper for response.error
#[derive(Debug, Clone, Serialize)]
pub struct ResponseErrorEvent {
    /// Event type (always "response.error")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The error information
    pub error: ResponseError,
}

/// SSE Event wrapper for response.cancelled
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCancelledEvent {
    /// Event ID
    pub id: String,

    /// Event type (always "response.cancelled")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Unix timestamp when cancelled
    pub created_at: i64,

    /// Event data payload
    pub data: ResponseCancelledData,
}

/// Data payload for response.cancelled event
#[derive(Debug, Clone, Serialize)]
pub struct ResponseCancelledData {
    /// The unique ID of the response
    pub id: Uuid,
}

/// SSE Event wrapper for response.in_progress
#[derive(Debug, Clone, Serialize)]
pub struct ResponseInProgressEvent {
    /// Event type (always "response.in_progress")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The response payload
    pub response: ResponsesCreateResponse,

    /// Sequence number for ordering
    pub sequence_number: i32,
}

/// SSE Event wrapper for response.output_item.added
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputItemAddedEvent {
    /// Event type (always "response.output_item.added")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// Index of the output item
    pub output_index: i32,

    /// The item being added
    pub item: OutputItem,
}

/// SSE Event wrapper for response.content_part.added
#[derive(Debug, Clone, Serialize)]
pub struct ResponseContentPartAddedEvent {
    /// Event type (always "response.content_part.added")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// The content part
    pub part: ContentPart,
}

/// SSE Event wrapper for response.output_text.done
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputTextDoneEvent {
    /// Event type (always "response.output_text.done")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// The complete text
    pub text: String,

    /// Log probabilities
    pub logprobs: Vec<serde_json::Value>,
}

/// SSE Event wrapper for response.reasoning_text.delta
/// Used for thinking/reasoning models (e.g., deepseek-r1) that emit reasoning tokens
/// TODO: Consider adding reasoning to final response output (like OpenAI's reasoning summary)
#[derive(Debug, Clone, Serialize)]
pub struct ResponseReasoningTextDeltaEvent {
    /// Event type (always "response.reasoning_text.delta")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// The reasoning delta
    pub delta: String,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// Sequence number for ordering
    pub sequence_number: i32,
}

/// SSE Event wrapper for response.reasoning_text.done
#[derive(Debug, Clone, Serialize)]
pub struct ResponseReasoningTextDoneEvent {
    /// Event type (always "response.reasoning_text.done")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// The complete reasoning text
    pub text: String,
}

/// SSE Event wrapper for response.content_part.done
#[derive(Debug, Clone, Serialize)]
pub struct ResponseContentPartDoneEvent {
    /// Event type (always "response.content_part.done")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// The ID of the output item
    pub item_id: String,

    /// The index of the output item
    pub output_index: i32,

    /// The index of the content part
    pub content_index: i32,

    /// The content part
    pub part: ContentPart,
}

/// SSE Event wrapper for response.output_item.done
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputItemDoneEvent {
    /// Event type (always "response.output_item.done")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// Index of the output item
    pub output_index: i32,

    /// The item that was completed
    pub item: OutputItem,
}

/// Content part structure
#[derive(Debug, Clone, Serialize)]
pub struct ContentPart {
    /// Type of content part
    #[serde(rename = "type")]
    pub part_type: String,

    /// Annotations
    pub annotations: Vec<serde_json::Value>,

    /// Log probabilities
    pub logprobs: Vec<serde_json::Value>,

    /// Text content
    pub text: String,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct ResponseUsage {
    /// Number of input tokens
    pub input_tokens: i32,

    /// Details about input tokens
    pub input_tokens_details: InputTokenDetails,

    /// Number of output tokens
    pub output_tokens: i32,

    /// Details about output tokens
    pub output_tokens_details: OutputTokenDetails,

    /// Total tokens used
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputTokenDetails {
    pub cached_tokens: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputTokenDetails {
    pub reasoning_tokens: i32,
}

/// Response returned by GET /v1/responses/{id}
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRetrieveResponse {
    pub id: Uuid,
    pub object: &'static str,
    pub created_at: i64,
    pub status: String,
    pub model: String,
    pub usage: Option<ResponseUsage>,
    pub output: Vec<OutputItem>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/responses",
            post(create_response_stream).layer(from_fn_with_state(
                state.clone(),
                decrypt_request::<ResponsesCreateRequest>,
            )),
        )
        .route(
            "/v1/responses/:id",
            get(get_response).layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/responses/:id",
            delete(delete_response).layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/responses/:id/cancel",
            post(cancel_response).layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .with_state(state)
}

/// SSE Event wrapper for tool_call.created
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallCreatedEvent {
    /// Event type (always "tool_call.created")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// Index of the corresponding output item
    pub output_index: i32,

    /// Tool call ID
    pub tool_call_id: Uuid,

    /// Tool name
    pub name: String,

    /// Tool arguments (JSON value)
    pub arguments: Value,
}

/// SSE Event wrapper for tool_output.created
#[derive(Debug, Clone, Serialize)]
pub struct ToolOutputCreatedEvent {
    /// Event type (always "tool_output.created")
    #[serde(rename = "type")]
    pub event_type: &'static str,

    /// Sequence number for ordering
    pub sequence_number: i32,

    /// Index of the corresponding output item
    pub output_index: i32,

    /// Tool output ID
    pub tool_output_id: Uuid,

    /// Tool call ID this output belongs to
    pub tool_call_id: Uuid,

    /// Tool output content
    pub output: String,
}

/// Message types for the storage task
#[derive(Debug, Clone)]
pub enum StorageMessage {
    MessageStarted {
        item_id: Uuid,
    },
    ContentDelta {
        item_id: Uuid,
        delta: String,
    },
    MessageDone {
        item_id: Uuid,
        finish_reason: String,
    },
    ReasoningStarted {
        item_id: Uuid,
    },
    /// Reasoning delta with item_id to ensure SSE and DB use the same UUID
    ReasoningDelta {
        item_id: Uuid,
        delta: String,
    },
    ReasoningDone {
        item_id: Uuid,
    },
    Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
    },
    ResponseDone {
        finish_reason: String,
    },
    Error(String),
    Cancelled,
    /// Tool-related messages
    ToolCall {
        tool_call_id: Uuid,
        name: String,
        arguments: Value,
    },
    ToolOutput {
        tool_output_id: Uuid,
        tool_call_id: Uuid,
        output: String,
    },
}

#[derive(Debug, Clone)]
enum StreamOutputItemRecord {
    Message {
        id: Uuid,
        text: String,
    },
    Reasoning {
        id: Uuid,
        text: String,
    },
    ToolCall {
        id: Uuid,
        call_id: Uuid,
        name: String,
        arguments: String,
    },
    ToolOutput {
        id: Uuid,
        call_id: Uuid,
        output: String,
    },
}

#[derive(Default)]
struct ClientResponseState {
    items: Vec<StreamOutputItemRecord>,
    indices: HashMap<Uuid, usize>,
}

impl ClientResponseState {
    fn push_message(&mut self, item_id: Uuid) -> i32 {
        let output_index = self.items.len();
        self.items.push(StreamOutputItemRecord::Message {
            id: item_id,
            text: String::new(),
        });
        self.indices.insert(item_id, output_index);
        output_index as i32
    }

    fn push_reasoning(&mut self, item_id: Uuid) -> i32 {
        let output_index = self.items.len();
        self.items.push(StreamOutputItemRecord::Reasoning {
            id: item_id,
            text: String::new(),
        });
        self.indices.insert(item_id, output_index);
        output_index as i32
    }

    fn push_tool_call(&mut self, item_id: Uuid, name: String, arguments: Value) -> i32 {
        let output_index = self.items.len();
        let arguments = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());
        self.items.push(StreamOutputItemRecord::ToolCall {
            id: item_id,
            call_id: item_id,
            name,
            arguments,
        });
        self.indices.insert(item_id, output_index);
        output_index as i32
    }

    fn push_tool_output(&mut self, item_id: Uuid, call_id: Uuid, output: String) -> i32 {
        let output_index = self.items.len();
        self.items.push(StreamOutputItemRecord::ToolOutput {
            id: item_id,
            call_id,
            output,
        });
        self.indices.insert(item_id, output_index);
        output_index as i32
    }

    fn message_output_index(&self, item_id: Uuid) -> Option<i32> {
        self.indices.get(&item_id).map(|index| *index as i32)
    }

    fn reasoning_output_index(&self, item_id: Uuid) -> Option<i32> {
        self.indices.get(&item_id).map(|index| *index as i32)
    }

    fn append_message_delta(&mut self, item_id: Uuid, delta: &str) -> Option<i32> {
        let index = *self.indices.get(&item_id)?;
        if let Some(StreamOutputItemRecord::Message { text, .. }) = self.items.get_mut(index) {
            text.push_str(delta);
            Some(index as i32)
        } else {
            None
        }
    }

    fn append_reasoning_delta(&mut self, item_id: Uuid, delta: &str) -> Option<i32> {
        let index = *self.indices.get(&item_id)?;
        if let Some(StreamOutputItemRecord::Reasoning { text, .. }) = self.items.get_mut(index) {
            text.push_str(delta);
            Some(index as i32)
        } else {
            None
        }
    }

    fn message_text(&self, item_id: Uuid) -> Option<&str> {
        let index = *self.indices.get(&item_id)?;
        match self.items.get(index)? {
            StreamOutputItemRecord::Message { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    fn reasoning_text(&self, item_id: Uuid) -> Option<&str> {
        let index = *self.indices.get(&item_id)?;
        match self.items.get(index)? {
            StreamOutputItemRecord::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    fn build_output_items(&self) -> Vec<OutputItem> {
        self.items
            .iter()
            .map(|item| match item {
                StreamOutputItemRecord::Message { id, text } => OutputItem {
                    id: id.to_string(),
                    output_type: OUTPUT_TYPE_MESSAGE.to_string(),
                    status: STATUS_COMPLETED.to_string(),
                    role: Some(ROLE_ASSISTANT.to_string()),
                    content: Some(vec![
                        ContentPartBuilder::new_output_text(text.clone()).build()
                    ]),
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                },
                StreamOutputItemRecord::Reasoning { id, .. } => OutputItem {
                    id: id.to_string(),
                    output_type: "reasoning".to_string(),
                    status: STATUS_COMPLETED.to_string(),
                    role: None,
                    content: Some(vec![]),
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                },
                StreamOutputItemRecord::ToolCall {
                    id,
                    call_id,
                    name,
                    arguments,
                } => OutputItem {
                    id: id.to_string(),
                    output_type: "tool_call".to_string(),
                    status: STATUS_COMPLETED.to_string(),
                    role: None,
                    content: None,
                    call_id: Some(call_id.to_string()),
                    name: Some(name.clone()),
                    arguments: Some(arguments.clone()),
                    output: None,
                },
                StreamOutputItemRecord::ToolOutput {
                    id,
                    call_id,
                    output,
                } => OutputItem {
                    id: id.to_string(),
                    output_type: "tool_output".to_string(),
                    status: STATUS_COMPLETED.to_string(),
                    role: None,
                    content: None,
                    call_id: Some(call_id.to_string()),
                    name: None,
                    arguments: None,
                    output: Some(output.clone()),
                },
            })
            .collect()
    }
}

/// Validated and prepared request data
struct PreparedRequest {
    user_key: SecretKey,
    message_content: MessageContent,
    user_message_tokens: i32,
    content_enc: Vec<u8>,
    assistant_message_id: Uuid,
}

/// Context and conversation data after building prompt
struct BuiltContext {
    conversation: crate::models::responses::Conversation,
    prompt_messages: Arc<Vec<Value>>,
    total_prompt_tokens: usize,
}

/// Persisted database records
struct PersistedData {
    response: crate::models::responses::Response,
    decrypted_metadata: Option<Value>,
    user_message_created_at: chrono::DateTime<chrono::Utc>,
}

/// Spawns a background task to generate a conversation title using AI
///
/// This function runs asynchronously and independently - it will not block the response stream.
/// If it fails, it logs the error but does not affect the ongoing response.
///
/// # Arguments
/// * `state` - Application state for database and API access
/// * `conversation_id` - Database ID of the conversation
/// * `conversation_uuid` - UUID of the conversation
/// * `user` - The user who owns the conversation
/// * `user_key` - User's encryption key for metadata
/// * `user_content` - The user's first message content
async fn spawn_title_generation_task(
    state: Arc<AppState>,
    conversation_id: i64,
    conversation_uuid: Uuid,
    user: User,
    user_key: SecretKey,
    user_content: String,
) {
    tokio::spawn(async move {
        debug!(
            "Starting background title generation for conversation {}",
            conversation_uuid
        );

        // Truncate content to first 500 characters
        let truncated_content: String = user_content.chars().take(500).collect();
        trace!(
            "Truncated content for title generation: {}",
            truncated_content
        );

        // Build the title generation request
        let title_request = json!({
            "model": "llama3-3-70b",
            "messages": [
                {
                    "role": ROLE_SYSTEM,
                    "content": "You are a helpful assistant that generates concise, meaningful titles (3-5 words) for chat conversations based on the user's first message. Return only the title without quotes or explanations."
                },
                {
                    "role": ROLE_USER,
                    "content": format!("Generate a concise, contextual title (3-5 words) for a chat that starts with this message: \"{}\"", truncated_content)
                }
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": 15,
            "stream": false
        });

        // Call the completions API with empty headers (no special headers needed)
        // Responses API always uses JWT auth (not API key)
        let headers = HeaderMap::new();
        let billing_context = crate::web::openai::BillingContext::new(
            crate::web::openai_auth::AuthMethod::Jwt,
            "llama3-3-70b".to_string(),
        );

        debug!("Title generation: about to call get_chat_completion_response");
        match get_chat_completion_response(&state, &user, title_request, &headers, billing_context)
            .await
        {
            Ok(mut completion) => {
                debug!("Title generation: received completion stream from API");
                // Get the FullResponse chunk (title generation is non-streaming)
                match completion.stream.recv().await {
                    Some(crate::web::openai::CompletionChunk::FullResponse(response_json)) => {
                        debug!("Title generation: received FullResponse chunk");
                        // Extract the title from choices[0].message.content
                        if let Some(title) = response_json
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            let title = title.trim();
                            trace!(
                                "Generated title for conversation {}: {}",
                                conversation_uuid,
                                title
                            );

                            // Get current conversation metadata
                            match state
                                .db
                                .get_conversation_by_uuid_and_user(conversation_uuid, user.uuid)
                            {
                                Ok(conversation) => {
                                    // Decrypt existing metadata
                                    match decrypt_content(
                                        &user_key,
                                        conversation.metadata_enc.as_ref(),
                                    ) {
                                        Ok(metadata) => {
                                            // Update the title in metadata
                                            let mut meta_obj =
                                                metadata.unwrap_or_else(|| json!({}));
                                            if let Some(obj) = meta_obj.as_object_mut() {
                                                obj.insert("title".to_string(), json!(title));
                                            }

                                            // Encrypt and update
                                            if let Ok(metadata_json) =
                                                serde_json::to_string(&meta_obj)
                                            {
                                                let metadata_enc = encrypt_with_key(
                                                    &user_key,
                                                    metadata_json.as_bytes(),
                                                )
                                                .await;

                                                if let Err(e) =
                                                    state.db.update_conversation_metadata(
                                                        conversation_id,
                                                        user.uuid,
                                                        metadata_enc,
                                                    )
                                                {
                                                    error!("Failed to update conversation metadata with generated title: {:?}", e);
                                                } else {
                                                    debug!("Successfully updated conversation {} with generated title", conversation_uuid);
                                                }
                                            } else {
                                                error!("Failed to serialize updated metadata");
                                            }
                                        }
                                        Err(e) => {
                                            error!(
                                                "Failed to decrypt conversation metadata: {:?}",
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to get conversation for title update: {:?}", e);
                                }
                            }
                        } else {
                            error!(
                                "Failed to extract title from API response - missing content field"
                            );
                        }
                    }
                    Some(other_chunk) => {
                        error!("Expected FullResponse chunk for title generation but got unexpected chunk type");
                        trace!("Unexpected chunk details: {:?}", other_chunk);
                    }
                    None => {
                        error!("Title generation: stream ended without receiving any chunks");
                    }
                }
            }
            Err(e) => {
                error!("Failed to generate conversation title: {:?}", e);
            }
        }
    });
}

/// Phase 1: Validate and normalize input
///
/// Performs all input validation and normalization without any side effects.
/// Ensures the request is valid before proceeding.
///
/// Operations:
/// - Validates user is not guest
/// - Gets user encryption key
/// - Normalizes message content to Parts format
/// - Validates no unsupported features (file uploads)
/// - Counts tokens and encrypts content
/// - Generates assistant message UUID
async fn validate_and_normalize_input(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
) -> Result<PreparedRequest, ApiError> {
    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for Responses API: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use Responses API: {}",
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
                "Guest user attempted to use Responses API without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
    }

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    // Normalize input to our standard format (validates unsupported features like file_id)
    let normalized_messages = body.input.clone().normalize()?;

    // Check if any message contains file uploads (currently unsupported)
    for msg in &normalized_messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                if matches!(part, MessageContentPart::InputFile { .. }) {
                    error!(
                        "User {} attempted to use unsupported file upload feature",
                        user.uuid
                    );
                    return Err(ApiError::BadRequest);
                }
            }
        }
    }

    // Get the first message's content (for user messages there should only be one)
    let message_content = normalized_messages
        .first()
        .ok_or_else(|| {
            error!("No messages provided in request");
            ApiError::BadRequest
        })?
        .content
        .clone();

    // Count tokens for the user's input message (text only for token counting)
    let input_text_for_tokens =
        MessageContentConverter::extract_text_for_token_counting(&message_content);
    let token_count = count_tokens(&input_text_for_tokens);
    let user_message_tokens = if token_count > i32::MAX as usize {
        warn!(
            "Token count {} exceeds i32::MAX, clamping to i32::MAX",
            token_count
        );
        i32::MAX
    } else {
        token_count as i32
    };

    // Validate that the user message doesn't exceed the context budget
    // Even if we drop everything else, we need to fit at least the user's message
    let max_ctx = model_max_ctx(&body.model);
    let response_reserve = 4096usize;
    let safety = 500usize;
    let ctx_budget = max_ctx.saturating_sub(response_reserve + safety);

    if user_message_tokens as usize >= ctx_budget {
        error!(
            "User message too large for user {}: {} tokens exceeds budget {} for model {}",
            user.uuid, user_message_tokens, ctx_budget, body.model
        );
        return Err(ApiError::MessageExceedsContextLimit);
    }

    // Serialize the MessageContent for storage
    let content_for_storage = serde_json::to_string(&message_content).map_err(|e| {
        error!("Failed to serialize message content: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Encrypt the serialized MessageContent
    let content_enc = encrypt_with_key(&user_key, content_for_storage.as_bytes()).await;

    // Generate the assistant message UUID once, to be used everywhere
    let assistant_message_id = Uuid::new_v4();
    debug!("Generated assistant message UUID: {}", assistant_message_id);

    Ok(PreparedRequest {
        user_key,
        message_content,
        user_message_tokens,
        content_enc,
        assistant_message_id,
    })
}

/// Phase 2: Build context and check billing
///
/// Read-only phase that builds conversation context and validates billing quota
/// before any database writes occur.
///
/// Operations:
/// - Fetches conversation and existing messages
/// - Builds prompt context with new user message (not yet persisted)
/// - Checks billing quota and token limits
async fn build_context_and_check_billing(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    user_key: &SecretKey,
    prepared: &PreparedRequest,
) -> Result<BuiltContext, ApiError> {
    let internal_system_prompt =
        build_internal_system_prompt(should_enable_web_search_tool(state.as_ref(), body));

    // Extract conversation ID from the required conversation parameter
    let conv_uuid = match &body.conversation {
        ConversationParam::String(id) | ConversationParam::Object { id } => *id,
    };

    // Get the conversation
    debug!("Using specified conversation: {}", conv_uuid);
    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conv_uuid, user.uuid)
        .map_err(error_mapping::map_conversation_error)?;

    // Build the conversation context from all persisted messages
    // Pass instructions from request (if provided) to override default user instructions
    let (mut prompt_messages, mut total_prompt_tokens) = build_prompt(
        state.db.as_ref(),
        conversation.id,
        user.uuid,
        user_key,
        &body.model,
        body.instructions.as_deref(),
        Some(&internal_system_prompt),
    )?;

    // Add the NEW user message to the context (not yet persisted)
    // This is needed for: 1) billing check, 2) sending to LLM
    let user_message_for_prompt = json!({
        "role": "user",
        "content": MessageContentConverter::to_openai_format(&prepared.message_content)
    });
    prompt_messages.push(user_message_for_prompt);
    total_prompt_tokens += prepared.user_message_tokens as usize;

    trace!(
        "Built prompt with {} total tokens, {} messages (including new user message)",
        total_prompt_tokens,
        prompt_messages.len()
    );

    // Check billing with token validation (BEFORE any persistence)
    // Only check for free users to save processing
    if let Some(billing_client) = &state.billing_client {
        debug!(
            "Checking billing for user {} with {} input tokens",
            user.uuid, total_prompt_tokens
        );

        // Responses API always uses JWT auth (not API key), so is_api = false
        if let Err(e) = billing_client
            .check_user_chat_with_tokens(user.uuid, false, total_prompt_tokens as i32)
            .await
        {
            match e {
                BillingError::UsageLimitExceeded => {
                    error!("Usage limit exceeded for user: {}", user.uuid);
                    return Err(ApiError::UsageLimitReached);
                }
                BillingError::FreeTokenLimitExceeded => {
                    // This error is only returned for free users
                    error!(
                        "Free tier token limit exceeded for user {} with {} tokens",
                        user.uuid, total_prompt_tokens
                    );
                    return Err(ApiError::FreeTokenLimitExceeded);
                }
                _ => {
                    // Log the error but allow the request for other billing service errors
                    error!("Billing service error, allowing request: {}", e);
                }
            }
        }
        debug!("Billing check passed for user {}", user.uuid);
    }

    Ok(BuiltContext {
        conversation,
        prompt_messages: Arc::new(prompt_messages),
        total_prompt_tokens,
    })
}

/// Phase 3: Persist request data
///
/// Writes to database after all validation and billing checks have passed.
///
/// Database operations:
/// - Creates Response record (status=in_progress)
/// - Creates user message
///
/// Note: Assistant and tool items are NOT created here - they're created later by the
/// storage task as stream events arrive so persisted ordering matches the emitted item order.
async fn persist_request_data(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    prepared: &PreparedRequest,
    conversation: &crate::models::responses::Conversation,
) -> Result<PersistedData, ApiError> {
    use crate::models::responses::{NewResponse, ResponseStatus};

    // Extract internal_message_id from metadata if present
    let message_uuid = if let Some(metadata) = &body.metadata {
        if let Some(internal_id) = metadata.get("internal_message_id") {
            if let Some(id_str) = internal_id.as_str() {
                // Try to parse as UUID, use new UUID if parsing fails
                Uuid::parse_str(id_str).unwrap_or_else(|_| {
                    warn!(
                        "Invalid internal_message_id UUID: {}, generating new one",
                        id_str
                    );
                    Uuid::new_v4()
                })
            } else {
                Uuid::new_v4()
            }
        } else {
            Uuid::new_v4()
        }
    } else {
        Uuid::new_v4()
    };

    // Encrypt metadata if provided
    let metadata_enc = if let Some(metadata) = &body.metadata {
        let metadata_json = serde_json::to_string(metadata).map_err(|e| {
            error!("Failed to serialize metadata: {:?}", e);
            ApiError::InternalServerError
        })?;
        Some(encrypt_with_key(&prepared.user_key, metadata_json.as_bytes()).await)
    } else {
        None
    };

    // Create the Response (job tracker)
    let new_response = NewResponse {
        uuid: Uuid::new_v4(),
        user_id: user.uuid,
        conversation_id: conversation.id,
        status: ResponseStatus::InProgress,
        model: body.model.clone(),
        temperature: body.temperature,
        top_p: body.top_p,
        max_output_tokens: body.max_output_tokens,
        tool_choice: body.tool_choice.clone(),
        parallel_tool_calls: body.parallel_tool_calls,
        store: body.store,
        metadata_enc: metadata_enc.clone(),
    };
    let response = state
        .db
        .create_response(new_response)
        .map_err(error_mapping::map_generic_db_error)?;

    // Create the simplified user message with extracted UUID
    let new_msg = NewUserMessage {
        uuid: message_uuid,
        conversation_id: conversation.id,
        response_id: Some(response.id),
        user_id: user.uuid,
        content_enc: prepared.content_enc.clone(),
        prompt_tokens: prepared.user_message_tokens,
    };
    let user_message = state
        .db
        .create_user_message(new_msg)
        .map_err(error_mapping::map_generic_db_error)?;

    // Capture the user message timestamp so persisted response items can be assigned
    // a single monotonic created_at sequence immediately after the user message.
    let user_message_created_at = user_message.created_at;

    info!(
        "Created response {} for user {} in conversation {}",
        response.uuid, user.uuid, conversation.uuid
    );

    // Decrypt metadata for response
    let decrypted_metadata: Option<Value> =
        decrypt_content(&prepared.user_key, response.metadata_enc.as_ref()).map_err(|e| {
            error!("Failed to decrypt response metadata: {:?}", e);
            ApiError::InternalServerError
        })?;

    Ok(PersistedData {
        response,
        decrypted_metadata,
        user_message_created_at,
    })
}

/// Helper function to check if tool_choice allows tool execution
///
/// Returns false if tool_choice is explicitly set to "none", true otherwise
fn is_tool_choice_allowed(tool_choice: &Option<String>) -> bool {
    tool_choice.as_deref() != Some("none")
}

/// Helper function to check if web_search tool is enabled in the request
///
/// Returns true if the tools array contains an object with type="web_search"
fn is_web_search_enabled(tools: &Option<Value>) -> bool {
    if let Some(tools_value) = tools {
        if let Some(tools_array) = tools_value.as_array() {
            return tools_array.iter().any(|tool| {
                tool.get("type")
                    .and_then(|t| t.as_str())
                    .map(|s| s == "web_search")
                    .unwrap_or(false)
            });
        }
    }
    false
}

/// Phase 5: Let the model request tool use (optional)
/// Persist and emit a single requested tool call, then wait for storage to
/// confirm the tool output is durable before the next model turn is started.
async fn execute_tool_call_and_wait(
    state: &Arc<AppState>,
    persisted: &PersistedData,
    tool_call: ModelToolCall,
    tx_client: &mpsc::Sender<StorageMessage>,
    tx_storage: &mpsc::Sender<StorageMessage>,
    rx_tool_ack: &mut mpsc::Receiver<Result<(), String>>,
) -> Result<(), ApiError> {
    let tool_call_id = Uuid::new_v4();
    let tool_output_id = Uuid::new_v4();

    let tool_call_msg = StorageMessage::ToolCall {
        tool_call_id,
        name: tool_call.name.clone(),
        arguments: tool_call.arguments.clone(),
    };

    if let Err(e) = tx_storage.send(tool_call_msg.clone()).await {
        error!(
            "Critical: Storage channel closed during tool_call for response {} - {:?}",
            persisted.response.uuid, e
        );
        let _ = tx_client
            .send(StorageMessage::Error(
                "Internal storage failure - request aborted".to_string(),
            ))
            .await;
        return Err(ApiError::InternalServerError);
    }
    if tx_client.try_send(tool_call_msg).is_err() {
        warn!("Client channel full or closed, skipping tool_call event to client");
    }

    let tool_output = match tools::execute_tool(
        &tool_call.name,
        &tool_call.arguments,
        state.brave_client.as_ref(),
    )
    .await
    {
        Ok(output) => output,
        Err(e) => {
            warn!("Tool execution failed, including error in output: {:?}", e);
            format!("Error: {}", e)
        }
    };

    let tool_output_msg = StorageMessage::ToolOutput {
        tool_output_id,
        tool_call_id,
        output: tool_output,
    };

    if let Err(e) = tx_storage.send(tool_output_msg.clone()).await {
        error!(
            "Critical: Storage channel closed during tool_output for response {} - {:?}",
            persisted.response.uuid, e
        );
        let _ = tx_client
            .send(StorageMessage::Error(
                "Internal storage failure - request aborted".to_string(),
            ))
            .await;
        return Err(ApiError::InternalServerError);
    }
    if tx_client.try_send(tool_output_msg).is_err() {
        warn!("Client channel full or closed, skipping tool_output event to client");
    }

    info!(
        "Successfully sent tool_call {} and tool_output {} to streams for conversation {}",
        tool_call_id, tool_output_id, persisted.response.conversation_id
    );

    match tokio::time::timeout(std::time::Duration::from_secs(5), rx_tool_ack.recv()).await {
        Ok(Some(Ok(()))) => {
            debug!("Tools persisted successfully to database");
            Ok(())
        }
        Ok(Some(Err(e))) => {
            error!("Failed to persist tools to database: {}", e);
            Err(ApiError::InternalServerError)
        }
        Ok(None) => {
            error!("Storage task dropped before sending tool acknowledgment");
            Err(ApiError::InternalServerError)
        }
        Err(_) => {
            error!("Timeout waiting for tool persistence (5s)");
            Err(ApiError::InternalServerError)
        }
    }
}

async fn send_storage_message(
    tx_storage: &mpsc::Sender<StorageMessage>,
    tx_client: &mpsc::Sender<StorageMessage>,
    msg: StorageMessage,
) -> Result<(), ApiError> {
    if tx_storage.send(msg.clone()).await.is_err() {
        error!("Storage channel closed unexpectedly");
        return Err(ApiError::InternalServerError);
    }
    if tx_client.try_send(msg).is_err() {
        warn!("Client channel full or closed");
    }
    Ok(())
}

fn next_assistant_message_id(next_message_id: &mut Option<Uuid>) -> Uuid {
    next_message_id.take().unwrap_or_else(Uuid::new_v4)
}

#[allow(clippy::too_many_arguments)]
async fn stream_one_assistant_turn(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    headers: &HeaderMap,
    prompt_messages: &[Value],
    tools_enabled: bool,
    tx_client: &mpsc::Sender<StorageMessage>,
    tx_storage: &mpsc::Sender<StorageMessage>,
    next_message_id: &mut Option<Uuid>,
) -> Result<AssistantTurnOutcome, ApiError> {
    let mut chat_request = build_model_turn_request(body, prompt_messages, tools_enabled);

    trace!(
        "Chat completion request to model {}: {}",
        body.model,
        serde_json::to_string_pretty(&chat_request)
            .unwrap_or_else(|_| "failed to serialize".to_string())
    );

    let billing_context = crate::web::openai::BillingContext::new(
        crate::web::openai_auth::AuthMethod::Jwt,
        body.model.clone(),
    );

    let mut completion =
        get_chat_completion_response(state, user, chat_request.take(), headers, billing_context)
            .await?;

    debug!(
        "Received completion from provider: {} (model: {})",
        completion.metadata.provider_name, completion.metadata.model_name
    );

    let mut streamed_tool_calls = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut current_message_id: Option<Uuid> = None;
    let mut message_started = false;
    let mut reasoning_item_id: Option<Uuid> = None;
    let mut reasoning_started = false;
    let mut reasoning_done = false;
    let mut saw_tool_calls = false;

    while let Some(chunk) = completion.stream.recv().await {
        match chunk {
            crate::web::openai::CompletionChunk::StreamChunk(json) => {
                let choice = json.get("choices").and_then(|choices| choices.get(0));
                if let Some(reason) = choice
                    .and_then(|choice| choice.get("finish_reason"))
                    .and_then(|reason| reason.as_str())
                {
                    if !reason.is_empty() {
                        finish_reason = Some(reason.to_string());
                    }
                }

                let delta = choice.and_then(|choice| choice.get("delta"));
                if let Some(d) = delta {
                    trace!("Stream delta: {}", d);
                }

                if let Some(reasoning) = delta.and_then(|d| {
                    d.get("reasoning")
                        .and_then(|c| c.as_str())
                        .or_else(|| d.get("reasoning_content").and_then(|c| c.as_str()))
                }) {
                    if !reasoning.is_empty() {
                        if reasoning_done {
                            warn!(
                                "Ignoring reasoning delta after reasoning item was already closed"
                            );
                        } else {
                            let reasoning_id = *reasoning_item_id.get_or_insert_with(Uuid::new_v4);
                            if !reasoning_started {
                                send_storage_message(
                                    tx_storage,
                                    tx_client,
                                    StorageMessage::ReasoningStarted {
                                        item_id: reasoning_id,
                                    },
                                )
                                .await?;
                                reasoning_started = true;
                            }

                            send_storage_message(
                                tx_storage,
                                tx_client,
                                StorageMessage::ReasoningDelta {
                                    item_id: reasoning_id,
                                    delta: reasoning.to_string(),
                                },
                            )
                            .await?;
                        }
                    }
                }

                if let Some(tool_call_delta) = delta.and_then(|d| d.get("tool_calls")) {
                    if let Some(message_id) = current_message_id.take() {
                        send_storage_message(
                            tx_storage,
                            tx_client,
                            StorageMessage::MessageDone {
                                item_id: message_id,
                                finish_reason: "tool_call".to_string(),
                            },
                        )
                        .await?;
                        message_started = false;
                    }

                    if !reasoning_done {
                        if let Some(reasoning_id) = reasoning_item_id {
                            send_storage_message(
                                tx_storage,
                                tx_client,
                                StorageMessage::ReasoningDone {
                                    item_id: reasoning_id,
                                },
                            )
                            .await?;
                            reasoning_done = true;
                        }
                    }

                    saw_tool_calls = true;
                    append_streamed_tool_calls(&mut streamed_tool_calls, tool_call_delta);
                }

                if let Some(content) = delta
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !content.is_empty() {
                        if saw_tool_calls {
                            error!("Received assistant text after tool call deltas had already started");
                            return Err(ApiError::InternalServerError);
                        }

                        if !reasoning_done {
                            if let Some(reasoning_id) = reasoning_item_id {
                                send_storage_message(
                                    tx_storage,
                                    tx_client,
                                    StorageMessage::ReasoningDone {
                                        item_id: reasoning_id,
                                    },
                                )
                                .await?;
                                reasoning_done = true;
                            }
                        }

                        let message_id = *current_message_id
                            .get_or_insert_with(|| next_assistant_message_id(next_message_id));
                        if !message_started {
                            send_storage_message(
                                tx_storage,
                                tx_client,
                                StorageMessage::MessageStarted {
                                    item_id: message_id,
                                },
                            )
                            .await?;
                            message_started = true;
                        }

                        send_storage_message(
                            tx_storage,
                            tx_client,
                            StorageMessage::ContentDelta {
                                item_id: message_id,
                                delta: content.to_string(),
                            },
                        )
                        .await?;
                    }
                }
            }
            crate::web::openai::CompletionChunk::Usage(usage) => {
                send_storage_message(
                    tx_storage,
                    tx_client,
                    StorageMessage::Usage {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                    },
                )
                .await?;
            }
            crate::web::openai::CompletionChunk::Done => {
                if !reasoning_done {
                    if let Some(reasoning_id) = reasoning_item_id {
                        send_storage_message(
                            tx_storage,
                            tx_client,
                            StorageMessage::ReasoningDone {
                                item_id: reasoning_id,
                            },
                        )
                        .await?;
                    }
                }

                if saw_tool_calls || finish_reason.as_deref() == Some("tool_calls") {
                    let tool_call = finalize_first_model_tool_call(&streamed_tool_calls)
                        .ok_or(ApiError::InternalServerError)?;
                    return Ok(AssistantTurnOutcome::ToolCall(tool_call));
                }

                let message_id = current_message_id
                    .unwrap_or_else(|| next_assistant_message_id(next_message_id));
                if current_message_id.is_none() {
                    send_storage_message(
                        tx_storage,
                        tx_client,
                        StorageMessage::MessageStarted {
                            item_id: message_id,
                        },
                    )
                    .await?;
                }

                let final_finish_reason = finish_reason.unwrap_or_else(|| "stop".to_string());
                send_storage_message(
                    tx_storage,
                    tx_client,
                    StorageMessage::MessageDone {
                        item_id: message_id,
                        finish_reason: final_finish_reason.clone(),
                    },
                )
                .await?;

                send_storage_message(
                    tx_storage,
                    tx_client,
                    StorageMessage::ResponseDone {
                        finish_reason: final_finish_reason,
                    },
                )
                .await?;
                return Ok(AssistantTurnOutcome::Final);
            }
            crate::web::openai::CompletionChunk::Error(error_msg) => {
                error!("Received error from completion stream: {}", error_msg);
                return Err(ApiError::InternalServerError);
            }
            crate::web::openai::CompletionChunk::FullResponse(_) => {
                error!("Received FullResponse in streaming mode");
                return Err(ApiError::InternalServerError);
            }
        }
    }

    error!("Completion stream closed unexpectedly without a terminal signal");
    Err(ApiError::InternalServerError)
}

/// Phase 6: Run the normal assistant/tool loop.
///
/// The selected model receives the full prompt with tool schemas, may call one
/// tool at a time, sees the tool output on the next turn, and eventually emits
/// final assistant text that is streamed to the client.
#[allow(clippy::too_many_arguments)]
async fn setup_completion_processor(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    context: &BuiltContext,
    prepared: &PreparedRequest,
    persisted: &PersistedData,
    headers: &HeaderMap,
    tx_client: mpsc::Sender<StorageMessage>,
    tx_storage: mpsc::Sender<StorageMessage>,
    mut rx_tool_ack: mpsc::Receiver<Result<(), String>>,
) -> Result<crate::models::responses::Response, ApiError> {
    const MAX_TOOL_TURNS: usize = 15;

    let tools_enabled = should_enable_web_search_tool(state.as_ref(), body);
    let internal_system_prompt = build_internal_system_prompt(tools_enabled);
    let mut prompt_messages = Arc::as_ref(&context.prompt_messages).clone();

    let loop_result: Result<(), ApiError> = async {
        let mut next_message_id = Some(prepared.assistant_message_id);
        let mut tool_turn_count = 0usize;
        loop {
            match stream_one_assistant_turn(
                state,
                user,
                body,
                headers,
                &prompt_messages,
                tools_enabled,
                &tx_client,
                &tx_storage,
                &mut next_message_id,
            )
            .await?
            {
                AssistantTurnOutcome::ToolCall(tool_call) => {
                    tool_turn_count += 1;
                    if tool_turn_count > MAX_TOOL_TURNS {
                        error!(
                            "Exceeded max tool turns ({}) for response {}",
                            MAX_TOOL_TURNS, persisted.response.uuid
                        );
                        return Err(ApiError::InternalServerError);
                    }

                    execute_tool_call_and_wait(
                        state,
                        persisted,
                        tool_call,
                        &tx_client,
                        &tx_storage,
                        &mut rx_tool_ack,
                    )
                    .await?;

                    let (rebuilt_messages, _tokens) = build_prompt(
                        state.db.as_ref(),
                        context.conversation.id,
                        user.uuid,
                        &prepared.user_key,
                        &body.model,
                        body.instructions.as_deref(),
                        Some(&internal_system_prompt),
                    )?;
                    prompt_messages = rebuilt_messages;
                }
                AssistantTurnOutcome::Final => return Ok(()),
            }
        }
    }
    .await;

    if let Err(e) = loop_result {
        let msg = StorageMessage::Error(format!("Streaming failed: {:?}", e));
        let _ = tx_storage.send(msg.clone()).await;
        let _ = tx_client.try_send(msg);
        return Err(e);
    }

    Ok(persisted.response.clone())
}

async fn create_response_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<ResponsesCreateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    trace!("=== ENTERING create_response_stream ===");
    trace!("User: {}", user.uuid);
    trace!("Request body: {:?}", body);
    trace!("Stream requested: {}", body.stream);

    // Phase 1: Validate and normalize input
    let prepared = validate_and_normalize_input(&state, &user, &body).await?;

    // Phase 2: Build context and check billing
    let context =
        build_context_and_check_billing(&state, &user, &body, &prepared.user_key, &prepared)
            .await?;

    // Phase 3: Persist request data
    let persisted =
        persist_request_data(&state, &user, &body, &prepared, &context.conversation).await?;

    // Check if first message and spawn title generation task
    let (user_count, assistant_count) =
        context
            .prompt_messages
            .iter()
            .fold((0, 0), |(users, assistants), msg| {
                match msg.get("role").and_then(|r| r.as_str()) {
                    Some(ROLE_USER) => (users + 1, assistants),
                    Some(ROLE_ASSISTANT) => (users, assistants + 1),
                    _ => (users, assistants),
                }
            });

    if user_count == 1 && assistant_count == 0 {
        let user_content =
            MessageContentConverter::extract_text_for_token_counting(&prepared.message_content);
        spawn_title_generation_task(
            state.clone(),
            context.conversation.id,
            context.conversation.uuid,
            user.clone(),
            prepared.user_key,
            user_content,
        )
        .await;
    }

    // Capture variables needed inside the stream
    let response_for_stream = persisted.response.clone();
    let decrypted_metadata = persisted.decrypted_metadata.clone();
    let assistant_message_id = prepared.assistant_message_id;
    let total_prompt_tokens = context.total_prompt_tokens;
    let response_id = persisted.response.id;
    let response_uuid = persisted.response.uuid;
    // Persist all generated response items on a single monotonic timestamp sequence that
    // begins immediately after the user message so retrieval order matches stream order.
    let first_response_item_created_at =
        persisted.user_message_created_at + chrono::Duration::microseconds(1);
    let conversation_id = context.conversation.id;
    let user_id = user.uuid;
    let user_key = prepared.user_key;
    let message_content = prepared.message_content.clone();
    let content_enc = prepared.content_enc.clone();
    let conversation_for_stream = context.conversation.clone();
    let prompt_messages = context.prompt_messages.clone();

    // Phases 4-6 now happen INSIDE the stream to start sending events ASAP
    trace!("Creating SSE event stream for client");
    let event_stream = async_stream::stream! {
        trace!("=== STARTING SSE STREAM ===");

        // Initialize the SSE event emitter
        let mut emitter = SseEventEmitter::new(&state, session_id, 0);

        // Send initial response.created event IMMEDIATELY (before any processing)
        trace!("Building response.created event");
        let created_response = ResponseBuilder::from_response(&response_for_stream)
            .status(STATUS_IN_PROGRESS)
            .metadata(decrypted_metadata.clone())
            .build();

        let created_event = ResponseCreatedEvent {
            event_type: EVENT_RESPONSE_CREATED,
            response: created_response.clone(),
            sequence_number: emitter.sequence_number(),
        };

        yield Ok(ResponseEvent::Created(created_event).to_sse_event(&mut emitter).await);

        // Event 2: response.in_progress
        let in_progress_event = ResponseInProgressEvent {
            event_type: EVENT_RESPONSE_IN_PROGRESS,
            response: created_response,
            sequence_number: emitter.sequence_number(),
        };

        yield Ok(ResponseEvent::InProgress(in_progress_event).to_sse_event(&mut emitter).await);

        // Phase 4: Create dual streams and spawn storage task
        trace!("Phase 4: Creating dual streams and spawning storage task");
        let (tx_storage, rx_storage) = mpsc::channel::<StorageMessage>(STORAGE_CHANNEL_BUFFER);
        let (tx_client, mut rx_client) = mpsc::channel::<StorageMessage>(CLIENT_CHANNEL_BUFFER);

        // Create channel for tool persistence acknowledgments (supports multiple tool loops)
        let (tx_tool_ack, rx_tool_ack) = mpsc::channel::<Result<(), String>>(8);

        let _storage_handle = {
            let db = state.db.clone();

            tokio::spawn(async move {
                storage_task(
                    rx_storage,
                    Some(tx_tool_ack),
                    db,
                    response_id,
                    first_response_item_created_at,
                    conversation_id,
                    user_id,
                    user_key,
                    assistant_message_id,
                )
                .await;
            })
        };

        // Spawn orchestrator task for phases 5-6 (runs in background, sends events to tx_client)
        trace!("Spawning background orchestrator for phases 5-6");
        let orchestrator_tx_client = tx_client.clone();
        let orchestrator_tx_storage = tx_storage.clone();
        let orchestrator_state = state.clone();
        let orchestrator_user = user.clone();
        let orchestrator_body = body.clone();
        let orchestrator_headers = headers.clone();
        let orchestrator_response = response_for_stream.clone();
        let orchestrator_metadata = decrypted_metadata.clone();
        let orchestrator_user_message_created_at = persisted.user_message_created_at;
        let orchestrator_conversation = conversation_for_stream.clone();
        let orchestrator_prompt_messages = prompt_messages.clone();

        tokio::spawn(async move {
            trace!("Orchestrator: Starting phases 5-6 in background");

            // Subscribe to cancellation broadcast
            let mut cancel_rx = orchestrator_state.cancellation_broadcast.subscribe();

            // Run phases 5-6 with cancellation support
            tokio::select! {
                _ = async {
                    // Phase 5-6: Run the normal assistant/tool loop
                    trace!("Orchestrator: Setting up assistant/tool loop");

                    let context_for_completion = BuiltContext {
                        conversation: orchestrator_conversation,
                        prompt_messages: orchestrator_prompt_messages,
                        total_prompt_tokens,
                    };

                    let prepared_for_completion = PreparedRequest {
                        user_key,
                        message_content,
                        user_message_tokens: 0,
                        content_enc,
                        assistant_message_id,
                    };

                    let persisted_for_completion = PersistedData {
                        response: orchestrator_response.clone(),
                        decrypted_metadata: orchestrator_metadata.clone(),
                        user_message_created_at: orchestrator_user_message_created_at,
                    };

                    match setup_completion_processor(
                        &orchestrator_state,
                        &orchestrator_user,
                        &orchestrator_body,
                        &context_for_completion,
                        &prepared_for_completion,
                        &persisted_for_completion,
                        &orchestrator_headers,
                        orchestrator_tx_client.clone(),
                        orchestrator_tx_storage.clone(),
                        rx_tool_ack,
                    )
                    .await
                    {
                        Ok(_) => {
                            trace!("Orchestrator: Assistant/tool loop completed");
                        }
                        Err(e) => {
                            error!("Orchestrator: Assistant/tool loop failed: {:?}", e);
                        }
                    }
                } => {
                    trace!("Orchestrator: Phases 5-6 completed normally");
                }

                Ok(cancelled_id) = cancel_rx.recv() => {
                    if cancelled_id == response_uuid {
                        debug!("Orchestrator: Received cancellation during phases 5-6 for response {}", response_uuid);

                        // Send cancellation to both channels
                        let _ = orchestrator_tx_storage.send(StorageMessage::Cancelled).await;
                        let _ = orchestrator_tx_client.send(StorageMessage::Cancelled).await;

                        trace!("Orchestrator: Cancellation handled, exiting");
                    }
                }
            }
        });

        // NOW immediately start the event loop - it will receive events from orchestrator as they happen
        trace!("Starting event loop to receive messages from background tasks");
        let mut client_state = ClientResponseState::default();
        let mut total_prompt_tokens_used = 0i32;
        let mut total_completion_tokens = 0i32;
        while let Some(msg) = rx_client.recv().await {
            trace!("Client stream received message from upstream processor");
            match msg {
                StorageMessage::MessageStarted { item_id } => {
                    let output_index = client_state.push_message(item_id);
                    let output_item_added_event = ResponseOutputItemAddedEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItemBuilder::new_message(item_id).build(),
                    };
                    yield Ok(ResponseEvent::OutputItemAdded(output_item_added_event).to_sse_event(&mut emitter).await);

                    let content_part_added_event = ResponseContentPartAddedEvent {
                        event_type: EVENT_RESPONSE_CONTENT_PART_ADDED,
                        sequence_number: emitter.sequence_number(),
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        part: ContentPart {
                            part_type: CONTENT_PART_TYPE_OUTPUT_TEXT.to_string(),
                            annotations: vec![],
                            logprobs: vec![],
                            text: String::new(),
                        },
                    };
                    yield Ok(ResponseEvent::ContentPartAdded(content_part_added_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ContentDelta { item_id, delta } => {
                    trace!("Client stream received content delta: {}", delta);
                    let Some(output_index) = client_state.append_message_delta(item_id, &delta) else {
                        warn!("Received content delta for unknown message item {}", item_id);
                        continue;
                    };

                    let delta_event = ResponseOutputTextDeltaEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_TEXT_DELTA,
                        delta,
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        sequence_number: emitter.sequence_number(),
                        logprobs: vec![],
                    };

                    yield Ok(ResponseEvent::OutputTextDelta(delta_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::MessageDone { item_id, .. } => {
                    let Some(output_index) = client_state.message_output_index(item_id) else {
                        warn!("Received message done for unknown item {}", item_id);
                        continue;
                    };
                    let text = client_state.message_text(item_id).unwrap_or("").to_string();

                    let output_text_done_event = ResponseOutputTextDoneEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_TEXT_DONE,
                        sequence_number: emitter.sequence_number(),
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        text: text.clone(),
                        logprobs: vec![],
                    };
                    yield Ok(ResponseEvent::OutputTextDone(output_text_done_event).to_sse_event(&mut emitter).await);

                    let content_part_done_event = ResponseContentPartDoneEvent {
                        event_type: EVENT_RESPONSE_CONTENT_PART_DONE,
                        sequence_number: emitter.sequence_number(),
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        part: ContentPartBuilder::new_output_text(text.clone()).build(),
                    };
                    yield Ok(ResponseEvent::ContentPartDone(content_part_done_event).to_sse_event(&mut emitter).await);

                    let output_item_done_event = ResponseOutputItemDoneEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_DONE,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItemBuilder::new_message(item_id)
                            .status(STATUS_COMPLETED)
                            .content(vec![ContentPartBuilder::new_output_text(text).build()])
                            .build(),
                    };
                    yield Ok(ResponseEvent::OutputItemDone(output_item_done_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ReasoningStarted { item_id } => {
                    let output_index = client_state.push_reasoning(item_id);
                    let reasoning_item_added = ResponseOutputItemAddedEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: item_id.to_string(),
                            output_type: "reasoning".to_string(),
                            status: STATUS_IN_PROGRESS.to_string(),
                            role: None,
                            content: Some(vec![]),
                            call_id: None,
                            name: None,
                            arguments: None,
                            output: None,
                        },
                    };
                    yield Ok(ResponseEvent::OutputItemAdded(reasoning_item_added).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ReasoningDelta { item_id, delta } => {
                    trace!("Client stream received reasoning delta: {}", delta);
                    let Some(output_index) = client_state.append_reasoning_delta(item_id, &delta) else {
                        warn!("Received reasoning delta for unknown item {}", item_id);
                        continue;
                    };

                    let delta_event = ResponseReasoningTextDeltaEvent {
                        event_type: EVENT_RESPONSE_REASONING_TEXT_DELTA,
                        delta,
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        sequence_number: emitter.sequence_number(),
                    };

                    yield Ok(ResponseEvent::ReasoningTextDelta(delta_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ReasoningDone { item_id } => {
                    let Some(output_index) = client_state.reasoning_output_index(item_id) else {
                        warn!("Received reasoning done for unknown item {}", item_id);
                        continue;
                    };
                    let text = client_state.reasoning_text(item_id).unwrap_or("").to_string();

                    let reasoning_done_event = ResponseReasoningTextDoneEvent {
                        event_type: EVENT_RESPONSE_REASONING_TEXT_DONE,
                        sequence_number: emitter.sequence_number(),
                        item_id: item_id.to_string(),
                        output_index,
                        content_index: 0,
                        text,
                    };
                    yield Ok(ResponseEvent::ReasoningTextDone(reasoning_done_event).to_sse_event(&mut emitter).await);

                    let reasoning_item_done = ResponseOutputItemDoneEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_DONE,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: item_id.to_string(),
                            output_type: "reasoning".to_string(),
                            status: STATUS_COMPLETED.to_string(),
                            role: None,
                            content: Some(vec![]),
                            call_id: None,
                            name: None,
                            arguments: None,
                            output: None,
                        },
                    };
                    yield Ok(ResponseEvent::OutputItemDone(reasoning_item_done).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ResponseDone { finish_reason: _finish_reason } => {
                    let usage = build_usage(
                        if total_prompt_tokens_used > 0 {
                            total_prompt_tokens_used
                        } else {
                            total_prompt_tokens as i32
                        },
                        total_completion_tokens,
                    );

                    let done_response = ResponseBuilder::from_response(&response_for_stream)
                        .status(STATUS_COMPLETED)
                        .output(client_state.build_output_items())
                        .usage(usage)
                        .metadata(decrypted_metadata.clone())
                        .build();

                    let completed_event = ResponseCompletedEvent {
                        event_type: EVENT_RESPONSE_COMPLETED,
                        response: done_response,
                        sequence_number: emitter.sequence_number(),
                    };

                    yield Ok(ResponseEvent::Completed(completed_event).to_sse_event(&mut emitter).await);
                    break;
                }
                StorageMessage::Usage { prompt_tokens, completion_tokens } => {
                    trace!("Client stream received usage data");
                    total_prompt_tokens_used += prompt_tokens;
                    total_completion_tokens += completion_tokens;
                }
                StorageMessage::Cancelled => {
                    debug!("Client stream received cancellation signal");
                    // Send response.cancelled event
                    let cancelled_event = ResponseCancelledEvent {
                        id: Uuid::new_v4().to_string(),
                        event_type: EVENT_RESPONSE_CANCELLED,
                        created_at: Utc::now().timestamp(),
                        data: ResponseCancelledData {
                            id: response_uuid,
                        },
                    };

                    yield Ok(ResponseEvent::Cancelled(cancelled_event).to_sse_event(&mut emitter).await);
                    break;
                }
                StorageMessage::Error(error_msg) => {
                    error!("Client stream received error: {}", error_msg);
                    // Send error event to client
                    let error_event = ResponseErrorEvent {
                        event_type: EVENT_RESPONSE_ERROR,
                        error: ResponseError {
                            error_type: "stream_error".to_string(),
                            message: error_msg,
                        },
                    };

                    yield Ok(ResponseEvent::Error(error_event).to_sse_event(&mut emitter).await);
                    break;
                }
                StorageMessage::ToolCall { tool_call_id, name, arguments } => {
                    debug!("Client stream received tool_call event: {} ({})", name, tool_call_id);
                    let tool_name = name.clone();
                    let arguments_json =
                        serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());
                    let output_index =
                        client_state.push_tool_call(tool_call_id, tool_name.clone(), arguments.clone());

                    let output_item_added_event = ResponseOutputItemAddedEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: tool_call_id.to_string(),
                            output_type: "tool_call".to_string(),
                            status: STATUS_IN_PROGRESS.to_string(),
                            role: None,
                            content: None,
                            call_id: Some(tool_call_id.to_string()),
                            name: Some(tool_name.clone()),
                            arguments: Some(arguments_json.clone()),
                            output: None,
                        },
                    };

                    yield Ok(ResponseEvent::OutputItemAdded(output_item_added_event).to_sse_event(&mut emitter).await);

                    // Send tool_call.created event
                    let tool_call_event = ToolCallCreatedEvent {
                        event_type: EVENT_TOOL_CALL_CREATED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        tool_call_id,
                        name,
                        arguments,
                    };

                    yield Ok(ResponseEvent::ToolCallCreated(tool_call_event).to_sse_event(&mut emitter).await);

                    let output_item_done_event = ResponseOutputItemDoneEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_DONE,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: tool_call_id.to_string(),
                            output_type: "tool_call".to_string(),
                            status: STATUS_COMPLETED.to_string(),
                            role: None,
                            content: None,
                            call_id: Some(tool_call_id.to_string()),
                            name: Some(tool_name),
                            arguments: Some(arguments_json),
                            output: None,
                        },
                    };

                    yield Ok(ResponseEvent::OutputItemDone(output_item_done_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ToolOutput { tool_output_id, tool_call_id, output } => {
                    debug!("Client stream received tool_output event: {}", tool_output_id);
                    let output_index = client_state.push_tool_output(
                        tool_output_id,
                        tool_call_id,
                        output.clone(),
                    );
                    let output_item_added_event = ResponseOutputItemAddedEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: tool_output_id.to_string(),
                            output_type: "tool_output".to_string(),
                            status: STATUS_IN_PROGRESS.to_string(),
                            role: None,
                            content: None,
                            call_id: Some(tool_call_id.to_string()),
                            name: None,
                            arguments: None,
                            output: Some(output.clone()),
                        },
                    };

                    yield Ok(ResponseEvent::OutputItemAdded(output_item_added_event).to_sse_event(&mut emitter).await);

                    // Send tool_output.created event
                    let tool_output_event = ToolOutputCreatedEvent {
                        event_type: EVENT_TOOL_OUTPUT_CREATED,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        tool_output_id,
                        tool_call_id,
                        output: output.clone(),
                    };

                    yield Ok(ResponseEvent::ToolOutputCreated(tool_output_event).to_sse_event(&mut emitter).await);

                    let output_item_done_event = ResponseOutputItemDoneEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_DONE,
                        sequence_number: emitter.sequence_number(),
                        output_index,
                        item: OutputItem {
                            id: tool_output_id.to_string(),
                            output_type: "tool_output".to_string(),
                            status: STATUS_COMPLETED.to_string(),
                            role: None,
                            content: None,
                            call_id: Some(tool_call_id.to_string()),
                            name: None,
                            arguments: None,
                            output: Some(output),
                        },
                    };

                    yield Ok(ResponseEvent::OutputItemDone(output_item_done_event).to_sse_event(&mut emitter).await);
                }
            }
        }

        // Client stream is done, but storage and upstream tasks continue independently
        trace!("Client SSE stream ending");
    };

    trace!("Returning SSE stream");
    Ok(Sse::new(event_stream))
}

/// Helper to create encrypted SSE event
pub async fn encrypt_event(
    state: &AppState,
    session_id: &Uuid,
    event_type: &str,
    payload: &Value,
) -> Result<Event, ApiError> {
    trace!("encrypt_event called for event type: {}", event_type);
    let payload_str = payload.to_string();
    let encrypted = state
        .encrypt_session_data(session_id, payload_str.as_bytes())
        .await
        .map_err(|e| {
            error!("Failed to encrypt event data: {:?}", e);
            ApiError::InternalServerError
        })?;

    let base64_encrypted = base64::engine::general_purpose::STANDARD.encode(&encrypted);
    Ok(Event::default().event(event_type).data(base64_encrypted))
}

/// GET /v1/responses/{id} - Retrieve a single response
async fn get_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ResponsesRetrieveResponse>>, ApiError> {
    debug!("Getting response {} for user {}", id, user.uuid);

    // Get the response
    let response = state
        .db
        .get_response_by_uuid_and_user(id, user.uuid)
        .map_err(error_mapping::map_response_error)?;

    // Get all messages associated with this response (user, assistant, tool_call, tool_output)
    let messages = state
        .db
        .get_response_context_messages(response.id)
        .map_err(error_mapping::map_generic_db_error)?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut output_items = Vec::new();

    for msg in &messages {
        let status = msg
            .status
            .clone()
            .unwrap_or_else(|| STATUS_COMPLETED.to_string());

        match msg.message_type.as_str() {
            "assistant" => {
                let text = decrypt_string(&user_key, msg.content_enc.as_ref()).map_err(|e| {
                    error!("Failed to decrypt assistant message content: {:?}", e);
                    error_mapping::map_decryption_error("assistant message content")
                })?;

                let output_item = if let Some(text) = text {
                    OutputItemBuilder::new_message(msg.uuid)
                        .status(&status)
                        .content(vec![ContentPartBuilder::new_output_text(text).build()])
                        .build()
                } else {
                    OutputItemBuilder::new_message(msg.uuid)
                        .status(&status)
                        .build()
                };
                output_items.push(output_item);
            }
            "tool_call" => {
                let arguments =
                    decrypt_string(&user_key, msg.content_enc.as_ref()).map_err(|e| {
                        error!("Failed to decrypt tool call arguments: {:?}", e);
                        error_mapping::map_decryption_error("tool call arguments")
                    })?;

                output_items.push(OutputItem {
                    id: msg.uuid.to_string(),
                    output_type: "tool_call".to_string(),
                    status,
                    role: None,
                    content: None,
                    call_id: Some(msg.tool_call_id.unwrap_or(msg.uuid).to_string()),
                    name: Some(
                        msg.tool_name
                            .clone()
                            .unwrap_or_else(|| "function".to_string()),
                    ),
                    arguments,
                    output: None,
                });
            }
            "tool_output" => {
                let output = decrypt_string(&user_key, msg.content_enc.as_ref()).map_err(|e| {
                    error!("Failed to decrypt tool output: {:?}", e);
                    error_mapping::map_decryption_error("tool output")
                })?;

                output_items.push(OutputItem {
                    id: msg.uuid.to_string(),
                    output_type: "tool_output".to_string(),
                    status,
                    role: None,
                    content: None,
                    call_id: msg.tool_call_id.map(|id| id.to_string()),
                    name: None,
                    arguments: None,
                    output,
                });
            }
            "reasoning" => {
                output_items.push(OutputItem {
                    id: msg.uuid.to_string(),
                    output_type: "reasoning".to_string(),
                    status,
                    role: None,
                    content: Some(vec![]),
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                });
            }
            _ => {}
        }
    }

    // Calculate token counts from individual messages
    let usage = if response.status == ResponseStatus::Completed {
        // Sum up tokens from all messages
        let mut input_tokens = 0i32;
        let mut output_tokens = 0i32;
        let mut reasoning_tokens = 0i32;

        for msg in &messages {
            if let Some(token_count) = msg.token_count {
                match msg.message_type.as_str() {
                    "user" => input_tokens += token_count,
                    "assistant" => output_tokens += token_count,
                    "reasoning" => {
                        output_tokens += token_count;
                        reasoning_tokens += token_count;
                    }
                    "tool_call" => input_tokens += token_count,
                    "tool_output" => input_tokens += token_count,
                    _ => {}
                }
            }
        }

        Some(ResponseUsage {
            input_tokens,
            input_tokens_details: InputTokenDetails { cached_tokens: 0 },
            output_tokens,
            output_tokens_details: OutputTokenDetails { reasoning_tokens },
            total_tokens: input_tokens + output_tokens,
        })
    } else {
        None
    };

    let retrieve_response = ResponsesRetrieveResponse {
        id: response.uuid,
        object: OBJECT_TYPE_RESPONSE,
        created_at: response.created_at.timestamp(),
        status: serde_json::to_value(response.status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "unknown".to_string()),
        model: response.model.clone(),
        usage,
        output: output_items,
    };

    encrypt_response(&state, &session_id, &retrieve_response).await
}

/// POST /v1/responses/{id}/cancel - Cancel an in-progress response
async fn cancel_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ResponsesRetrieveResponse>>, ApiError> {
    debug!("Cancelling response {} for user {}", id, user.uuid);

    // Verify the response exists and belongs to the user, and is in_progress
    let response = state
        .db
        .get_response_by_uuid_and_user(id, user.uuid)
        .map_err(|e| {
            debug!("Response {} not found for user {}: {:?}", id, user.uuid, e);
            match e {
                DBError::ResponsesError(ResponsesError::ResponseNotFound) => ApiError::NotFound,
                DBError::ResponsesError(ResponsesError::Unauthorized) => ApiError::Unauthorized,
                _ => ApiError::InternalServerError,
            }
        })?;

    // Only allow cancelling in_progress responses
    if response.status != ResponseStatus::InProgress {
        debug!(
            "Cannot cancel response {} with status {:?}",
            id, response.status
        );
        return Err(ApiError::BadRequest);
    }

    // Broadcast cancellation signal to all listeners
    debug!("Broadcasting cancellation signal for response {}", id);
    let _ = state.cancellation_broadcast.send(id);

    // Update the response status in the database
    let response = state.db.cancel_response(id, user.uuid).map_err(|e| {
        debug!(
            "Response {} not found for user {} during cancel: {:?}",
            id, user.uuid, e
        );
        match e {
            DBError::ResponsesError(ResponsesError::ResponseNotFound) => ApiError::NotFound,
            DBError::ResponsesError(ResponsesError::Unauthorized) => ApiError::Unauthorized,
            DBError::ResponsesError(ResponsesError::ValidationError) => ApiError::BadRequest,
            _ => ApiError::InternalServerError,
        }
    })?;

    // No usage or output for cancelled responses
    let retrieve_response = ResponsesRetrieveResponse {
        id: response.uuid,
        object: OBJECT_TYPE_RESPONSE,
        created_at: response.created_at.timestamp(),
        status: STATUS_CANCELLED.to_string(),
        model: response.model.clone(),
        usage: None,
        output: vec![],
    };

    encrypt_response(&state, &session_id, &retrieve_response).await
}

/// DELETE /v1/responses/{id} - Hard delete a response
async fn delete_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    debug!("Deleting response {} for user {}", id, user.uuid);

    // Delete the response (cascade will handle related records)
    state.db.delete_response(id, user.uuid).map_err(|e| {
        debug!(
            "Response {} not found for user {} during delete: {:?}",
            id, user.uuid, e
        );
        match e {
            DBError::ResponsesError(ResponsesError::ResponseNotFound) => ApiError::NotFound,
            DBError::ResponsesError(ResponsesError::Unauthorized) => ApiError::Unauthorized,
            _ => ApiError::InternalServerError,
        }
    })?;

    let response = DeletedObjectResponse::response(id);

    encrypt_response(&state, &session_id, &response).await
}
