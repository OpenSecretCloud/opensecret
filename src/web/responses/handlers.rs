//! Responses API implementation with SSE streaming and dual-stream storage.
//! Phases 4 & 5: Always streams to client while concurrently storing to database.

use crate::{
    billing::BillingError,
    db::DBError,
    encrypt::{decrypt_content, decrypt_string, encrypt_with_key},
    models::responses::{NewAssistantMessage, NewUserMessage, ResponseStatus, ResponsesError},
    models::users::User,
    tokens::count_tokens,
    web::{
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        openai::get_chat_completion_response,
        responses::{
            build_prompt, build_usage, constants::*, error_mapping, storage_task,
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
    pub fn normalize(self) -> Vec<MessageInput> {
        match self {
            InputMessage::String(s) => {
                // Simple string -> user message with input_text content parts
                vec![MessageInput {
                    role: ROLE_USER.to_string(),
                    content: MessageContent::Parts(vec![MessageContentPart::InputText { text: s }]),
                }]
            }
            InputMessage::Messages(mut messages) => {
                // Ensure all message content is normalized to Parts format
                for msg in &mut messages {
                    msg.content = MessageContentConverter::normalize_content(msg.content.clone());
                }
                messages
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
    pub status: &'static str,

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
    prompt_messages: Vec<Value>,
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

/// Phase 1: Validate input and prepare encrypted content
///
/// This phase performs all input validation and normalization without any side effects
/// (no database writes). It ensures the request is valid before proceeding.
///
/// # Validations
/// - Rejects guest users
/// - Gets user encryption key
/// - Normalizes message content to Parts format
/// - Rejects unsupported features (file uploads)
/// - Counts tokens for billing check
/// - Encrypts content for storage
/// - Generates assistant message UUID
///
/// # Arguments
/// * `state` - Application state
/// * `user` - Authenticated user
/// * `body` - Request body
///
/// # Returns
/// PreparedRequest containing validated and encrypted data
///
/// # Errors
/// Returns ApiError if validation fails or user is unauthorized
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

    // Normalize input to our standard format
    let normalized_messages = body.input.clone().normalize();

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

/// Phase 2: Build conversation context and check billing
///
/// This phase is read-only - it builds the conversation context from existing
/// messages and performs billing checks WITHOUT writing to the database. This
/// ensures we don't persist data if the user is over quota.
///
/// # Operations
/// - Gets conversation from database
/// - Builds context from all existing messages
/// - Adds the NEW user message to context (not yet persisted)
/// - Checks billing quota (only for free users)
/// - Validates token limits
///
/// # Critical Design Note
/// The new user message is added to the context array but NOT yet persisted.
/// This allows accurate billing checks before committing to storage.
///
/// # Arguments
/// * `state` - Application state
/// * `user` - Authenticated user
/// * `body` - Request body
/// * `user_key` - User's encryption key
/// * `prepared` - Validated request data from Phase 1
///
/// # Returns
/// BuiltContext containing conversation, prompt messages, and token count
///
/// # Errors
/// Returns ApiError if conversation not found, billing check fails, or user over quota
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
        prompt_messages,
        total_prompt_tokens,
    })
}

/// Phase 3: Persist request data to database
///
/// This phase writes to the database ONLY after all validation and billing checks
/// have passed. This ensures atomic semantics - either everything is written or
/// nothing is written.
///
/// # Database Operations
/// - Creates Response record (job tracker) with status=in_progress
/// - Creates user message record linked to response
/// - Creates placeholder assistant message (status=in_progress, content=NULL)
/// - Encrypts metadata if provided
/// - Extracts internal_message_id from metadata if present
///
/// # Design Notes
/// - Placeholder assistant message allows clients to see in-progress status
/// - Content is NULL until streaming completes
/// - Response ID is used to link all related records
/// - Metadata is encrypted before storage
///
/// # Arguments
/// * `state` - Application state
/// * `user` - Authenticated user
/// * `body` - Request body
/// * `prepared` - Validated request data from Phase 1
/// * `conversation` - Conversation from Phase 2
///
/// # Returns
/// PersistedData containing created records and decrypted metadata
///
/// # Errors
/// Returns ApiError if database operations fail
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

    // Create placeholder assistant message with status='in_progress' and NULL content
    let placeholder_assistant = NewAssistantMessage {
        uuid: prepared.assistant_message_id,
        conversation_id: conversation.id,
        response_id: Some(response.id),
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

/// Phase 4: Setup streaming pipeline with channels and tasks
///
/// This phase sets up the dual-stream architecture that allows simultaneous
/// streaming to the client and storage to the database. The streaming continues
/// independently even if the client disconnects.
///
/// # Architecture
/// - Creates two channels: storage (critical) and client (best-effort)
/// - Spawns storage task to persist data as it arrives
/// - Spawns upstream processor to parse SSE from chat API
/// - Returns client channel for SSE event generation
///
/// # Key Design Principles
/// 1. **Dual streaming**: Client and storage streams operate independently
/// 2. **Storage priority**: Storage sends must succeed, client sends can fail
/// 3. **Independent lifecycle**: Streaming continues even if client disconnects
/// 4. **Cancellation support**: Listens for cancellation broadcast signals
///
/// # Task Spawning
/// - Storage task: Accumulates content and persists on completion
/// - Upstream processor: Parses SSE frames and broadcasts to both channels
///
/// # Arguments
/// * `state` - Application state
/// * `user` - Authenticated user
/// * `body` - Request body
/// * `context` - Built context from Phase 2
/// * `prepared` - Validated request data from Phase 1
/// * `persisted` - Persisted records from Phase 3
/// * `headers` - Request headers for upstream API call
///
/// # Returns
/// Tuple of (client channel receiver, response record) for SSE stream generation
///
/// # Errors
/// Returns ApiError if chat API call fails or channel creation fails
async fn setup_streaming_pipeline(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    context: &BuiltContext,
    prepared: &PreparedRequest,
    persisted: &PersistedData,
    headers: &HeaderMap,
) -> Result<
    (
        mpsc::Receiver<StorageMessage>,
        crate::models::responses::Response,
    ),
    ApiError,
> {
    // Build chat completion request
    let chat_request = json!({
        "model": body.model,
        "messages": context.prompt_messages,
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

    // Create channels for storage task and client stream
    let (tx_storage, rx_storage) = mpsc::channel::<StorageMessage>(STORAGE_CHANNEL_BUFFER);
    let (tx_client, rx_client) = mpsc::channel::<StorageMessage>(CLIENT_CHANNEL_BUFFER);

    // Spawn storage task (no longer needs sqs_publisher - billing is centralized!)
    let _storage_handle = {
        let db = state.db.clone();
        let response_id = persisted.response.id;
        let user_key = prepared.user_key;
        let message_id = prepared.assistant_message_id;

        tokio::spawn(async move {
            storage_task(rx_storage, db, response_id, user_key, message_id).await;
        })
    };

    // Spawn stream processor task that converts CompletionChunks to StorageMessages
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

    Ok((rx_client, persisted.response.clone()))
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

    // Phase 1: Validate and normalize input (no side effects)
    let prepared = validate_and_normalize_input(&state, &user, &body).await?;

    // Phase 2: Build context and check billing (read-only, no DB writes)
    let context =
        build_context_and_check_billing(&state, &user, &body, &prepared.user_key, &prepared)
            .await?;

    // Check if this is the first user message in the conversation
    // The context.prompt_messages includes the NEW user message (added in build_context_and_check_billing)
    // Count user and assistant messages - if there's exactly 1 user and 0 assistant, it's the first message
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
    let is_first_message = user_count == 1 && assistant_count == 0;

    // Phase 3: Persist to database (only after all checks pass)
    let persisted =
        persist_request_data(&state, &user, &body, &prepared, &context.conversation).await?;

    // If this is the first message, spawn background task to generate conversation title
    if is_first_message {
        debug!(
            "First message detected, spawning title generation task for conversation {}",
            context.conversation.uuid
        );

        // Extract text content for title generation
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

    // Phase 4: Setup streaming pipeline
    let (mut rx_client, response) = match setup_streaming_pipeline(
        &state, &user, &body, &context, &prepared, &persisted, &headers,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            // Streaming setup failed - clean up the database records
            error!(
                "Failed to setup streaming pipeline for response {}: {:?}",
                persisted.response.uuid, e
            );

            // Update response status to failed
            if let Err(db_err) = state.db.update_response_status(
                persisted.response.id,
                ResponseStatus::Failed,
                Some(Utc::now()),
            ) {
                error!(
                    "Failed to update response status after pipeline error: {:?}",
                    db_err
                );
            }

            // Update assistant message to incomplete with no content
            if let Err(db_err) = state.db.update_assistant_message(
                prepared.assistant_message_id,
                None, // No content
                0,    // No tokens
                STATUS_INCOMPLETE.to_string(),
                None, // No finish_reason since we failed before streaming
            ) {
                error!(
                    "Failed to update assistant message after pipeline error: {:?}",
                    db_err
                );
            }

            return Err(e);
        }
    };

    let assistant_message_id = prepared.assistant_message_id;
    let total_prompt_tokens = context.total_prompt_tokens;

    trace!("Creating SSE event stream for client");
    let event_stream = async_stream::stream! {
        trace!("=== STARTING SSE STREAM ===");

        // Initialize the SSE event emitter
        let mut emitter = SseEventEmitter::new(&state, session_id, 0);

        // Send initial response.created event
        trace!("Building response.created event");
        let created_response = ResponseBuilder::from_response(&response)
            .status(STATUS_IN_PROGRESS)
            .metadata(persisted.decrypted_metadata.clone())
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

        // Process messages from upstream processor
        let mut assistant_content = String::new();
        let mut total_completion_tokens = 0i32;

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

        trace!("Starting to process messages from upstream processor");
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

                                let done_response = ResponseBuilder::from_response(&response)
                                    .status(STATUS_COMPLETED)
                                    .output(vec![output_item])
                                    .usage(usage)
                                    .metadata(persisted.decrypted_metadata.clone())
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
                            id: response.uuid,
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
