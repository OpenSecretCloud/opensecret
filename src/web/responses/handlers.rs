//! Responses API implementation with SSE streaming and dual-stream storage.
//! Phases 4 & 5: Always streams to client while concurrently storing to database.

use crate::{
    billing::BillingError,
    db::DBError,
    encrypt::{decrypt_content, decrypt_string, encrypt_with_key},
    models::responses::{NewAssistantMessage, NewUserMessage, ResponseStatus, ResponsesError},
    models::users::User,
    tokens::{count_tokens, model_max_ctx},
    web::{
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        openai::get_chat_completion_response,
        responses::{
            build_prompt, build_usage, constants::*, error_mapping, prompts, storage_task, tools,
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
use std::sync::Arc;
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

    /// Tool choice strategy (ignored in Phase 4/5)
    #[serde(default)]
    pub tool_choice: Option<String>,

    /// Tools available for the model (ignored in Phase 4/5)
    #[serde(default)]
    pub tools: Option<Value>,

    /// Enable parallel tool calls (ignored in Phase 4/5)
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

    /// Available tools (empty array for Phase 4/5)
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
    ContentDelta(String),
    Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
    },
    Done {
        finish_reason: String,
        message_id: Uuid,
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
    /// Signal that assistant message is about to start streaming
    AssistantMessageStarting,
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
            "model": "llama-3.3-70b",
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
            "llama-3.3-70b".to_string(),
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
    // Prevent guest users
    if user.is_guest() {
        error!("Guest user attempted to use Responses API: {}", user.uuid);
        return Err(ApiError::Unauthorized);
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
/// Note: Assistant message is NOT created here - it's created later in Phase 6 (after tools).
/// Originally, the assistant placeholder was created here, but this caused timestamp
/// ordering issues: the assistant message would get created_at=T1 (early), then tools
/// would execute at T2/T3, making the assistant appear BEFORE its tools in queries
/// ordered by created_at. By creating the assistant message in Phase 6 (after tools),
/// we ensure the correct semantic order: user → tool_call → tool_output → assistant.
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
    let _user_message = state
        .db
        .create_user_message(new_msg)
        .map_err(error_mapping::map_generic_db_error)?;

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

/// Phase 5: Classify intent and execute tools (optional)
///
/// Classifies user intent and executes tools if needed. Runs after dual streams
/// are created so tool events can be sent to both client and storage.
///
/// Flow:
/// 1. Classify intent: chat vs web_search
/// 2. If web_search: extract query and execute tool
/// 3. Send ToolCall event to streams
/// 4. Send ToolOutput event to streams (always, even on error)
/// 5. Send persistence command via dedicated channel and wait for acknowledgment
///
/// Tool execution is best-effort and uses fast model (llama-3.3-70b).
async fn classify_and_execute_tools(
    state: &Arc<AppState>,
    user: &User,
    prepared: &PreparedRequest,
    persisted: &PersistedData,
    tx_client: &mpsc::Sender<StorageMessage>,
    tx_storage: &mpsc::Sender<StorageMessage>,
    rx_tool_ack: tokio::sync::oneshot::Receiver<Result<(), String>>,
) -> Result<Option<()>, ApiError> {
    // Extract text from user message for classification
    let user_text =
        MessageContentConverter::extract_text_for_token_counting(&prepared.message_content);

    trace!(
        "Classifying user intent for message: {}",
        user_text.chars().take(100).collect::<String>()
    );
    debug!("Starting intent classification");

    // Step 1: Classify intent using LLM
    let classification_request = prompts::build_intent_classification_request(&user_text);
    let headers = HeaderMap::new();
    let billing_context = crate::web::openai::BillingContext::new(
        crate::web::openai_auth::AuthMethod::Jwt,
        "llama-3.3-70b".to_string(),
    );

    let intent = match get_chat_completion_response(
        state,
        user,
        classification_request,
        &headers,
        billing_context,
    )
    .await
    {
        Ok(mut completion) => {
            match completion.stream.recv().await {
                Some(crate::web::openai::CompletionChunk::FullResponse(response_json)) => {
                    // Extract intent from response
                    if let Some(intent_str) = response_json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        let intent = intent_str.trim().to_lowercase();
                        debug!("Classified intent: {}", intent);
                        intent
                    } else {
                        warn!(
                            "Failed to extract intent from classifier response, defaulting to chat"
                        );
                        "chat".to_string()
                    }
                }
                _ => {
                    warn!("Unexpected classifier response format, defaulting to chat");
                    "chat".to_string()
                }
            }
        }
        Err(e) => {
            // Best effort - if classification fails, default to chat
            warn!("Classification failed (defaulting to chat): {:?}", e);
            "chat".to_string()
        }
    };

    // Step 2: If intent is web_search, execute tool
    if intent == "web_search" {
        debug!("User message classified as web_search, executing tool");

        // Extract search query
        let query_request = prompts::build_query_extraction_request(&user_text);
        let billing_context = crate::web::openai::BillingContext::new(
            crate::web::openai_auth::AuthMethod::Jwt,
            "llama-3.3-70b".to_string(),
        );

        let search_query = match get_chat_completion_response(
            state,
            user,
            query_request,
            &headers,
            billing_context,
        )
        .await
        {
            Ok(mut completion) => match completion.stream.recv().await {
                Some(crate::web::openai::CompletionChunk::FullResponse(response_json)) => {
                    if let Some(query) = response_json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        let query = query.trim().to_string();
                        trace!("Extracted search query: {}", query);
                        debug!("Search query extracted successfully");
                        query
                    } else {
                        warn!("Failed to extract query, using original message");
                        user_text.clone()
                    }
                }
                _ => {
                    warn!("Unexpected query extraction response, using original message");
                    user_text.clone()
                }
            },
            Err(e) => {
                warn!("Query extraction failed, using original message: {:?}", e);
                user_text.clone()
            }
        };

        // Generate UUIDs for tool_call and tool_output
        let tool_call_id = Uuid::new_v4();
        let tool_output_id = Uuid::new_v4();

        // Prepare tool arguments
        let tool_arguments = json!({"query": search_query});

        // Send tool_call event through both streams FIRST (before execution)
        let tool_call_msg = StorageMessage::ToolCall {
            tool_call_id,
            name: "web_search".to_string(),
            arguments: tool_arguments.clone(),
        };
        // Send to storage (critical - must succeed)
        if let Err(e) = tx_storage.send(tool_call_msg.clone()).await {
            error!("Failed to send tool_call to storage channel: {:?}", e);
            return Ok(None);
        }
        // Send to client (best-effort)
        if tx_client.try_send(tool_call_msg).is_err() {
            warn!("Client channel full or closed, skipping tool_call event to client");
        }

        debug!("Sent tool_call {} to streams", tool_call_id);

        // Execute web search tool (or capture error as content)
        let tool_output =
            match tools::execute_tool("web_search", &tool_arguments, state.kagi_client.as_ref())
                .await
            {
                Ok(output) => {
                    debug!(
                        "Tool execution successful, output length: {} chars",
                        output.len()
                    );
                    output
                }
                Err(e) => {
                    warn!("Tool execution failed, including error in output: {:?}", e);
                    // Failure becomes content, not a skip!
                    format!("Error: {}", e)
                }
            };

        // Send tool_output event through both streams (ALWAYS sent, even on failure)
        let tool_output_msg = StorageMessage::ToolOutput {
            tool_output_id,
            tool_call_id,
            output: tool_output.clone(),
        };
        // Send to storage (critical - must succeed)
        if let Err(e) = tx_storage.send(tool_output_msg.clone()).await {
            error!("Failed to send tool_output to storage channel: {:?}", e);
            return Ok(None);
        }
        // Send to client (best-effort)
        if tx_client.try_send(tool_output_msg).is_err() {
            warn!("Client channel full or closed, skipping tool_output event to client");
        }

        info!(
            "Successfully sent tool_call {} and tool_output {} to streams for conversation {}",
            tool_call_id, tool_output_id, persisted.response.conversation_id
        );

        // Wait for storage task to confirm persistence (with timeout)
        match tokio::time::timeout(std::time::Duration::from_secs(5), rx_tool_ack).await {
            Ok(Ok(Ok(()))) => {
                debug!("Tools persisted successfully to database");
                return Ok(Some(()));
            }
            Ok(Ok(Err(e))) => {
                error!("Failed to persist tools to database: {}", e);
                // Continue anyway - best effort
                return Ok(Some(()));
            }
            Ok(Err(_)) => {
                error!("Storage task dropped before sending acknowledgment");
                return Ok(Some(()));
            }
            Err(_) => {
                error!("Timeout waiting for tool persistence (5s)");
                return Ok(Some(()));
            }
        }
    } else {
        debug!("User message classified as chat, skipping tool execution");
    }

    Ok(None)
}

/// Phase 6: Setup completion processor
///
/// Gets completion stream from chat API and spawns processor task.
///
/// Operations:
/// - Creates placeholder assistant message (AFTER tools, so timestamp is ordered correctly)
/// - Rebuilds prompt from DB if tools were executed (automatically includes tools)
/// - Calls chat API with streaming enabled
/// - Spawns processor task that converts CompletionChunks to StorageMessages
/// - Processor feeds into dual streams (storage=critical, client=best-effort)
/// - Listens for cancellation signals
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
    tools_executed: bool,
) -> Result<crate::models::responses::Response, ApiError> {
    // Create placeholder assistant message with status='in_progress' and NULL content
    //
    // TIMING: This happens here in Phase 6 (not earlier in Phase 3) for two reasons:
    // 1. Must happen AFTER tool execution (Phase 5) to get correct timestamp ordering
    // 2. Must happen BEFORE calling completion API (below) so storage task can UPDATE it
    //
    // Previously, this was created in Phase 3 under the assumption it needed to exist early.
    // However, it only needs to exist before the storage task tries to UPDATE it (when
    // streaming completes). By creating it here, we ensure proper message ordering:
    // user → tool_call → tool_output → assistant (this creation) → assistant content (update)
    let placeholder_assistant = NewAssistantMessage {
        uuid: prepared.assistant_message_id,
        conversation_id: context.conversation.id,
        response_id: Some(persisted.response.id),
        user_id: user.uuid,
        content_enc: None,
        completion_tokens: 0,
        status: STATUS_IN_PROGRESS.to_string(),
        finish_reason: None,
    };
    state
        .db
        .create_assistant_message(placeholder_assistant)
        .map_err(|e| {
            error!("Error creating placeholder assistant message: {:?}", e);
            ApiError::InternalServerError
        })?;

    debug!(
        "Created placeholder assistant message {} after tool execution",
        prepared.assistant_message_id
    );

    // If tools were executed, rebuild prompt from DB (will now include persisted tools)
    // Otherwise use the context we built earlier
    let prompt_messages = if tools_executed {
        debug!("Tools were executed - rebuilding prompt from DB to include tool messages");
        let (rebuilt_messages, _tokens) = build_prompt(
            state.db.as_ref(),
            context.conversation.id,
            user.uuid,
            &prepared.user_key,
            &body.model,
            body.instructions.as_deref(),
        )?;
        rebuilt_messages
    } else {
        // Clone out of Arc only when actually needed for the completion request
        Arc::as_ref(&context.prompt_messages).clone()
    };

    // Build chat completion request
    let chat_request = json!({
        "model": body.model,
        "messages": prompt_messages,
        "temperature": body.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        "top_p": body.top_p.unwrap_or(DEFAULT_TOP_P),
        "max_tokens": body.max_output_tokens,
        "stream": true,
        "stream_options": { "include_usage": true }
    });

    // Log the exact request we're sending to the completions API
    trace!(
        "Chat completion request to model {}: {}",
        body.model,
        serde_json::to_string_pretty(&chat_request)
            .unwrap_or_else(|_| "failed to serialize".to_string())
    );

    // Create billing context - Responses API always uses JWT auth (not API key)
    let billing_context = crate::web::openai::BillingContext::new(
        crate::web::openai_auth::AuthMethod::Jwt,
        body.model.clone(),
    );

    // Call the chat API - billing happens automatically inside!
    let completion =
        get_chat_completion_response(state, user, chat_request, headers, billing_context).await?;

    debug!(
        "Received completion from provider: {} (model: {})",
        completion.metadata.provider_name, completion.metadata.model_name
    );

    // Signal that assistant message is about to start streaming
    // CRITICAL: Must send BEFORE spawning processor to guarantee ordering
    // (processor will immediately start sending ContentDelta messages)
    if let Err(e) = tx_client
        .send(StorageMessage::AssistantMessageStarting)
        .await
    {
        error!("Failed to send AssistantMessageStarting signal: {:?}", e);
        // Client channel closed - not critical, continue anyway
    }

    // Spawn stream processor task that converts CompletionChunks to StorageMessages
    // and feeds them into the master stream channels (created in Phase 3.5)
    let _processor_handle = {
        let mut rx_completion = completion.stream;
        let message_id = prepared.assistant_message_id;
        let response_uuid = persisted.response.uuid;
        let mut cancel_rx = state.cancellation_broadcast.subscribe();

        tokio::spawn(async move {
            trace!("Starting completion stream processor task");
            let mut client_alive = true;

            loop {
                tokio::select! {
                    // Check for cancellation
                    Ok(cancelled_id) = cancel_rx.recv() => {
                        if cancelled_id == response_uuid {
                            debug!("Received cancellation signal for response {}", response_uuid);
                            let _ = tx_storage.send(StorageMessage::Cancelled).await;
                            if client_alive && tx_client.try_send(StorageMessage::Cancelled).is_err() {
                                warn!("Client channel full or closed during cancellation, terminating client stream");
                                #[allow(unused_assignments)]
                                { client_alive = false; }
                            }
                            break;
                        }
                    }

                    // Process CompletionChunks from centralized billing API
                    chunk_opt = rx_completion.recv() => {
                        let Some(chunk) = chunk_opt else {
                            error!("Completion stream closed unexpectedly without Done signal");
                            // Explicitly notify storage and client of the failure
                            let msg = StorageMessage::Error("Stream closed unexpectedly".to_string());
                            let _ = tx_storage.send(msg.clone()).await;
                            if client_alive && tx_client.try_send(msg).is_err() {
                                warn!("Client channel full or closed during stream error, terminating client stream");
                                #[allow(unused_assignments)]
                                { client_alive = false; }
                            }
                            break;
                        };

                        match chunk {
                            crate::web::openai::CompletionChunk::StreamChunk(json) => {
                                // Extract content from the full JSON chunk (safe chaining to avoid panics)
                                if let Some(content) = json
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("delta"))
                                    .and_then(|d| d.get("content"))
                                    .and_then(|c| c.as_str())
                                {
                                    let msg = StorageMessage::ContentDelta(content.to_string());
                                    // Must send to storage (critical, can block)
                                    if tx_storage.send(msg.clone()).await.is_err() {
                                        error!("Storage channel closed unexpectedly");
                                        break;
                                    }
                                    // Best-effort send to client (non-blocking, never blocks storage)
                                    if client_alive && tx_client.try_send(msg).is_err() {
                                        warn!("Client channel full or closed, terminating client stream");
                                        client_alive = false;
                                    }
                                }
                            }
                            crate::web::openai::CompletionChunk::Usage(usage) => {
                                // Billing already happened in openai.rs!
                                // Just forward to storage for token counting
                                let msg = StorageMessage::Usage {
                                    prompt_tokens: usage.prompt_tokens,
                                    completion_tokens: usage.completion_tokens,
                                };
                                let _ = tx_storage.send(msg.clone()).await;
                                if client_alive && tx_client.try_send(msg).is_err() {
                                    warn!("Client channel full or closed during usage message, terminating client stream");
                                    client_alive = false;
                                }
                            }
                            crate::web::openai::CompletionChunk::Done => {
                                debug!("Received Done chunk from completion stream");
                                let msg = StorageMessage::Done {
                                    finish_reason: "stop".to_string(),
                                    message_id,
                                };
                                let _ = tx_storage.send(msg.clone()).await;
                                if client_alive && tx_client.try_send(msg).is_err() {
                                    warn!("Client channel full or closed during done message, terminating client stream");
                                    #[allow(unused_assignments)]
                                    { client_alive = false; }
                                }
                                break;
                            }
                            crate::web::openai::CompletionChunk::Error(error_msg) => {
                                error!("Received error from completion stream: {}", error_msg);
                                let msg = StorageMessage::Error(error_msg);
                                let _ = tx_storage.send(msg.clone()).await;
                                if client_alive && tx_client.try_send(msg).is_err() {
                                    warn!("Client channel full or closed during error message, terminating client stream");
                                    #[allow(unused_assignments)]
                                    { client_alive = false; }
                                }
                                break;
                            }
                            crate::web::openai::CompletionChunk::FullResponse(_) => {
                                // Shouldn't happen for streaming
                                error!("Received FullResponse in streaming mode");
                                break;
                            }
                        }
                    }
                }
            }

            debug!("Completion stream processor task completed");
        })
    };

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

        // Create oneshot channel for tool persistence acknowledgment
        let (tx_tool_ack, rx_tool_ack) = tokio::sync::oneshot::channel();

        let _storage_handle = {
            let db = state.db.clone();

            tokio::spawn(async move {
                storage_task(
                    rx_storage,
                    Some(tx_tool_ack),
                    db,
                    response_id,
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
        let orchestrator_conversation = conversation_for_stream.clone();
        let orchestrator_prompt_messages = prompt_messages.clone();

        tokio::spawn(async move {
            trace!("Orchestrator: Starting phases 5-6 in background");

            // Subscribe to cancellation broadcast
            let mut cancel_rx = orchestrator_state.cancellation_broadcast.subscribe();

            // Run phases 5-6 with cancellation support
            tokio::select! {
                _ = async {
                    // Phase 5: Classify intent and execute tools (if tool_choice allows it AND web_search is enabled AND Kagi client available)
                    let tools_executed = if is_tool_choice_allowed(&orchestrator_body.tool_choice)
                        && is_web_search_enabled(&orchestrator_body.tools)
                        && orchestrator_state.kagi_client.is_some() {
                        debug!("Orchestrator: tool_choice allows tools, web search enabled, and Kagi client available, proceeding with classification");

                        let prepared_for_tools = PreparedRequest {
                            user_key,
                            message_content: message_content.clone(),
                            user_message_tokens: 0,
                            content_enc: content_enc.clone(),
                            assistant_message_id,
                        };

                        let persisted_for_tools = PersistedData {
                            response: orchestrator_response.clone(),
                            decrypted_metadata: orchestrator_metadata.clone(),
                        };

                        match classify_and_execute_tools(
                            &orchestrator_state,
                            &orchestrator_user,
                            &prepared_for_tools,
                            &persisted_for_tools,
                            &orchestrator_tx_client,
                            &orchestrator_tx_storage,
                            rx_tool_ack,
                        )
                        .await
                        {
                            Ok(result) => result.is_some(),
                            Err(e) => {
                                warn!("Orchestrator: Tool execution error (continuing): {:?}", e);
                                false
                            }
                        }
                    } else {
                        debug!("Orchestrator: Web search tool not enabled or Kagi client not available, skipping classification");
                        drop(rx_tool_ack);
                        false
                    };

                    // Phase 6: Setup completion processor
                    trace!("Orchestrator: Setting up completion processor");

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
                        tools_executed,
                    )
                    .await
                    {
                        Ok(_) => {
                            trace!("Orchestrator: Completion processor setup complete");
                            // AssistantMessageStarting is now sent from inside setup_completion_processor
                            // to guarantee it arrives before any completion deltas
                        }
                        Err(e) => {
                            error!("Orchestrator: Failed to setup completion processor: {:?}", e);

                            // Update response status to failed
                            if let Err(db_err) = orchestrator_state.db.update_response_status(
                                response_id,
                                ResponseStatus::Failed,
                                Some(Utc::now()),
                            ) {
                                error!("Orchestrator: Failed to update response status: {:?}", db_err);
                            }

                            // Update assistant message to incomplete
                            if let Err(db_err) = orchestrator_state.db.update_assistant_message(
                                assistant_message_id,
                                None,
                                0,
                                STATUS_INCOMPLETE.to_string(),
                                None,
                            ) {
                                error!("Orchestrator: Failed to update assistant message: {:?}", db_err);
                            }

                            // Send error to client via channel (best-effort)
                            let _ = orchestrator_tx_client.try_send(StorageMessage::Error(
                                format!("Failed to setup streaming: {:?}", e)
                            ));
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
        let mut assistant_content = String::new();
        let mut total_completion_tokens = 0i32;
        while let Some(msg) = rx_client.recv().await {
            trace!("Client stream received message from upstream processor");
            match msg {
                StorageMessage::Done { finish_reason, message_id: msg_id } => {
                    trace!("Client stream received Done signal with finish_reason={}, message_id={}", finish_reason, msg_id);
                    // Note: msg_id should match our pre-generated assistant_message_id

                                // Event 7: response.output_text.done
                                let output_text_done_event = ResponseOutputTextDoneEvent {
                                    event_type: EVENT_RESPONSE_OUTPUT_TEXT_DONE,
                                    sequence_number: emitter.sequence_number(),
                                    item_id: assistant_message_id.to_string(),
                                    output_index: 0,
                                    content_index: 0,
                                    text: assistant_content.clone(),
                                    logprobs: vec![],
                                };

                                yield Ok(ResponseEvent::OutputTextDone(output_text_done_event).to_sse_event(&mut emitter).await);

                                // Event 8: response.content_part.done
                                let content_part_done_event = ResponseContentPartDoneEvent {
                                    event_type: EVENT_RESPONSE_CONTENT_PART_DONE,
                                    sequence_number: emitter.sequence_number(),
                                    item_id: assistant_message_id.to_string(),
                                    output_index: 0,
                                    content_index: 0,
                                    part: ContentPart {
                                        part_type: CONTENT_PART_TYPE_OUTPUT_TEXT.to_string(),
                                        annotations: vec![],
                                        logprobs: vec![],
                                        text: assistant_content.clone(),
                                    },
                                };

                                yield Ok(ResponseEvent::ContentPartDone(content_part_done_event).to_sse_event(&mut emitter).await);

                                // Event 9: response.output_item.done
                                let content_part = ContentPart {
                                    part_type: CONTENT_PART_TYPE_OUTPUT_TEXT.to_string(),
                                    annotations: vec![],
                                    logprobs: vec![],
                                    text: assistant_content.clone(),
                                };

                                let output_item_done_event = ResponseOutputItemDoneEvent {
                                    event_type: EVENT_RESPONSE_OUTPUT_ITEM_DONE,
                                    sequence_number: emitter.sequence_number(),
                                    output_index: 0,
                                    item: OutputItem {
                                        id: assistant_message_id.to_string(),
                                        output_type: OUTPUT_TYPE_MESSAGE.to_string(),
                                        status: STATUS_COMPLETED.to_string(),
                                        role: Some(ROLE_ASSISTANT.to_string()),
                                        content: Some(vec![content_part]),
                                    },
                                };

                                yield Ok(ResponseEvent::OutputItemDone(output_item_done_event).to_sse_event(&mut emitter).await);

                                // Event 10: response.completed
                                let content_part = ContentPartBuilder::new_output_text(assistant_content.clone()).build();
                                let output_item = OutputItemBuilder::new_message(assistant_message_id)
                                    .status(STATUS_COMPLETED)
                                    .content(vec![content_part])
                                    .build();
                                let usage = build_usage(total_prompt_tokens as i32, total_completion_tokens);

                                let done_response = ResponseBuilder::from_response(&response_for_stream)
                                    .status(STATUS_COMPLETED)
                                    .output(vec![output_item])
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
                StorageMessage::ContentDelta(content) => {
                    trace!("Client stream received content delta: {}", content);
                    assistant_content.push_str(&content);

                    // Send to client
                    let delta_event = ResponseOutputTextDeltaEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_TEXT_DELTA,
                        delta: content.clone(),
                        item_id: assistant_message_id.to_string(),
                        output_index: 0,
                        content_index: 0,
                        sequence_number: emitter.sequence_number(),
                        logprobs: vec![],
                    };

                    yield Ok(ResponseEvent::OutputTextDelta(delta_event).to_sse_event(&mut emitter).await);
                }

                StorageMessage::Usage { prompt_tokens: _, completion_tokens } => {
                    trace!("Client stream received usage data");
                    total_completion_tokens = completion_tokens;
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
                    // Send tool_call.created event
                    let tool_call_event = ToolCallCreatedEvent {
                        event_type: EVENT_TOOL_CALL_CREATED,
                        sequence_number: emitter.sequence_number(),
                        tool_call_id,
                        name,
                        arguments,
                    };

                    yield Ok(ResponseEvent::ToolCallCreated(tool_call_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::ToolOutput { tool_output_id, tool_call_id, output } => {
                    debug!("Client stream received tool_output event: {}", tool_output_id);
                    // Send tool_output.created event
                    let tool_output_event = ToolOutputCreatedEvent {
                        event_type: EVENT_TOOL_OUTPUT_CREATED,
                        sequence_number: emitter.sequence_number(),
                        tool_output_id,
                        tool_call_id,
                        output,
                    };

                    yield Ok(ResponseEvent::ToolOutputCreated(tool_output_event).to_sse_event(&mut emitter).await);
                }
                StorageMessage::AssistantMessageStarting => {
                    debug!("Client stream received assistant message starting signal");

                    // Event 3: response.output_item.added
                    let output_item_added_event = ResponseOutputItemAddedEvent {
                        event_type: EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
                        sequence_number: emitter.sequence_number(),
                        output_index: 0,
                        item: OutputItem {
                            id: assistant_message_id.to_string(),
                            output_type: OUTPUT_TYPE_MESSAGE.to_string(),
                            status: STATUS_IN_PROGRESS.to_string(),
                            role: Some(ROLE_ASSISTANT.to_string()),
                            content: Some(vec![]),
                        },
                    };

                    yield Ok(ResponseEvent::OutputItemAdded(output_item_added_event).to_sse_event(&mut emitter).await);

                    // Event 4: response.content_part.added
                    let content_part_added_event = ResponseContentPartAddedEvent {
                        event_type: EVENT_RESPONSE_CONTENT_PART_ADDED,
                        sequence_number: emitter.sequence_number(),
                        item_id: assistant_message_id.to_string(),
                        output_index: 0,
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

    // Build output from assistant messages only (user messages are input, not output)
    // TODO: Add tool_call and tool_output items to output array once we implement tool support
    // Need to determine correct OpenAI format for tool items in output array
    let mut output_items = Vec::new();

    for msg in &messages {
        // Only include assistant messages in output
        // TODO: When adding tool support, also handle msg.message_type == "tool_call" and "tool_output"
        if msg.message_type != "assistant" {
            continue;
        }

        // Only include messages that have content
        if let Some(content_enc) = &msg.content_enc {
            if let Some(text) = decrypt_string(&user_key, Some(content_enc)).map_err(|e| {
                error!("Failed to decrypt assistant message content: {:?}", e);
                error_mapping::map_decryption_error("assistant message content")
            })? {
                // Build content part using builder
                let content_part = ContentPartBuilder::new_output_text(text).build();

                // Build output item using builder
                let output_item = OutputItemBuilder::new_message(msg.uuid)
                    .status(
                        &msg.status
                            .clone()
                            .unwrap_or_else(|| STATUS_COMPLETED.to_string()),
                    )
                    .content(vec![content_part])
                    .build();

                output_items.push(output_item);
            }
        }
    }

    // Calculate token counts from individual messages
    let usage = if response.status == ResponseStatus::Completed {
        // Sum up tokens from all messages
        let mut input_tokens = 0i32;
        let mut output_tokens = 0i32;

        for msg in &messages {
            if let Some(token_count) = msg.token_count {
                match msg.message_type.as_str() {
                    "user" => input_tokens += token_count,
                    "assistant" => output_tokens += token_count,
                    // tool_call and tool_output tokens are also considered in context
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
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: 0,
            },
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
