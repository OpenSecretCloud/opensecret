//! Responses API implementation with SSE streaming and dual-stream storage.
//! Phases 4 & 5: Always streams to client while concurrently storing to database.

use crate::{
    context_builder::build_prompt,
    db::DBError,
    encrypt::{decrypt_with_key, encrypt_with_key},
    models::responses::{NewAssistantMessage, NewUserMessage, ResponseStatus, ResponsesError},
    models::token_usage::NewTokenUsage,
    models::users::User,
    sqs::UsageEvent,
    tokens::count_tokens,
    web::{
        conversations::{MessageContent, MessageContentPart},
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        openai::get_chat_completion_response,
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
use bigdecimal::BigDecimal;
use chrono::Utc;
use futures::{Stream, StreamExt, TryStreamExt};
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;
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
                    role: "user".to_string(),
                    content: MessageContent::Parts(vec![MessageContentPart::InputText { text: s }]),
                }]
            }
            InputMessage::Messages(mut messages) => {
                // Ensure all message content is normalized to Parts format
                for msg in &mut messages {
                    msg.content = match msg.content.clone() {
                        MessageContent::Text(text) => {
                            // Convert plain text to input_text part
                            MessageContent::Parts(vec![MessageContentPart::InputText { text }])
                        }
                        MessageContent::Parts(parts) => {
                            // Already in parts format, pass through
                            MessageContent::Parts(parts)
                        }
                    };
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

    /// Conversation to associate with (UUID string or {id: UUID} object)
    #[serde(default)]
    pub conversation: Option<ConversationParam>,

    /// Previous response ID to continue conversation (deprecated, use conversation instead)
    #[serde(default)]
    pub previous_response_id: Option<Uuid>,

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
    pub output: Option<String>,
}

/// Response returned by DELETE /v1/responses/{id}
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesDeleteResponse {
    pub id: Uuid,
    pub object: &'static str,
    pub deleted: bool,
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
#[derive(Debug)]
enum StorageMessage {
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

    // Prevent guest users
    if user.is_guest() {
        error!("Guest user attempted to use Responses API: {}", user.uuid);
        return Err(ApiError::Unauthorized);
    }

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|e| {
            error!("Failed to get user encryption key: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Normalize input to our standard format
    let normalized_messages = body.input.clone().normalize();

    // Get the first message's content (for user messages there should only be one)
    let message_content = &normalized_messages[0].content;

    // Count tokens for the user's input message (text only for token counting)
    let input_text_for_tokens = message_content.as_text_for_input_token_count_only();
    let user_message_tokens = count_tokens(&input_text_for_tokens) as i32;

    // Serialize the MessageContent for storage
    let content_for_storage = serde_json::to_string(message_content).map_err(|e| {
        error!("Failed to serialize message content: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Encrypt the serialized MessageContent
    let content_enc = encrypt_with_key(&user_key, content_for_storage.as_bytes()).await;

    // Create conversation, response, and user message
    let (conversation, response, _user_message) = persist_initial_message(
        &state,
        &user,
        &body,
        content_enc.clone(),
        user_message_tokens,
        &user_key,
    )
    .await?;

    info!(
        "Created response {} for user {} in conversation {}",
        response.uuid, user.uuid, conversation.uuid
    );

    // Decrypt metadata for response
    let decrypted_metadata = if let Some(metadata_enc) = &response.metadata_enc {
        match decrypt_with_key(&user_key, metadata_enc) {
            Ok(metadata_bytes) => serde_json::from_slice(&metadata_bytes).ok(),
            Err(e) => {
                error!("Failed to decrypt response metadata: {:?}", e);
                None
            }
        }
    } else {
        None
    };

    // Build the conversation context from all persisted messages
    let (prompt_messages, total_prompt_tokens) =
        build_prompt(state.db.as_ref(), conversation.id, &user_key, &body.model)?;

    trace!(
        "Built prompt with {} total tokens, {} messages",
        total_prompt_tokens,
        prompt_messages.len()
    );

    // Note: We already stored the user message's own token count during persist_initial_message.
    // The total_prompt_tokens here includes system prompt + conversation history + user message,
    // which is used for the actual API call but not stored on the individual message.

    // Build chat completion request
    let chat_request = json!({
        "model": body.model,
        "messages": prompt_messages,
        "temperature": body.temperature.unwrap_or(0.7),
        "top_p": body.top_p.unwrap_or(1.0),
        "max_tokens": body.max_output_tokens.unwrap_or(10_000),
        "stream": true,
        // TODO should either happen in the completions response method
        "stream_options": { "include_usage": true }
    });

    // Log the exact request we're sending to the completions API
    trace!(
        "Chat completion request to model {}: {}",
        body.model,
        serde_json::to_string_pretty(&chat_request)
            .unwrap_or_else(|_| "failed to serialize".to_string())
    );

    // Call the chat API
    let upstream_response =
        get_chat_completion_response(&state, &user, chat_request, &headers).await?;

    // Create channels for storage task and client stream
    let (tx_storage, rx_storage) = mpsc::channel::<StorageMessage>(1024);
    let (tx_client, mut rx_client) = mpsc::channel::<StorageMessage>(1024);

    // Spawn storage task
    let _storage_handle = {
        let db = state.db.clone();
        let response_id = response.id;
        let conversation_id = conversation.id;
        let user_uuid = user.uuid;
        let sqs_publisher = state.sqs_publisher.clone();

        tokio::spawn(async move {
            storage_task(
                rx_storage,
                db,
                response_id,
                conversation_id,
                user_key,
                user_uuid,
                sqs_publisher,
            )
            .await;
        })
    };

    // Spawn upstream processor task that runs independently of client connection
    // This ensures storage completes even if client disconnects
    let _upstream_handle = {
        let mut body_stream = upstream_response.into_body().into_stream();
        let tx_storage_clone = tx_storage.clone();
        let tx_client_clone = tx_client.clone();
        let message_id = Uuid::new_v4();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut stream_finish_reason: Option<String> = None;

            trace!("Starting upstream processor task");
            while let Some(chunk_result) = body_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(bytes.as_ref()));

                        // Process complete SSE frames
                        while let Some(double_newline_pos) = buffer.find("\n\n") {
                            let frame = buffer[..double_newline_pos].to_string();
                            buffer = buffer[double_newline_pos + 2..].to_string();

                            // Skip empty frames
                            if frame.trim().is_empty() {
                                continue;
                            }

                            // Extract data from SSE frame
                            if let Some(data) = frame.strip_prefix("data: ") {
                                let data = data.trim();

                                if data == "[DONE]" {
                                    trace!(
                                        "Upstream processor: received [DONE], sending completion"
                                    );

                                    // Send to storage
                                    let _ = tx_storage_clone
                                        .send(StorageMessage::Done {
                                            finish_reason: stream_finish_reason
                                                .clone()
                                                .unwrap_or_else(|| "stop".to_string()),
                                            message_id,
                                        })
                                        .await;

                                    // Send to client (if still connected)
                                    let _ = tx_client_clone
                                        .send(StorageMessage::Done {
                                            finish_reason: stream_finish_reason
                                                .clone()
                                                .unwrap_or_else(|| "stop".to_string()),
                                            message_id,
                                        })
                                        .await;

                                    break;
                                }

                                // Parse JSON data
                                if let Ok(json_data) = serde_json::from_str::<Value>(data) {
                                    // Extract content delta
                                    if let Some(content) =
                                        json_data["choices"][0]["delta"]["content"].as_str()
                                    {
                                        trace!(
                                            "Upstream processor: found content delta: {}",
                                            content
                                        );

                                        // Send to storage
                                        let _ = tx_storage_clone
                                            .send(StorageMessage::ContentDelta(content.to_string()))
                                            .await;

                                        // Send to client (if still connected)
                                        let _ = tx_client_clone
                                            .send(StorageMessage::ContentDelta(content.to_string()))
                                            .await;
                                    }

                                    // Extract usage
                                    if let Some(usage) = json_data.get("usage") {
                                        let provider_prompt_tokens =
                                            usage["prompt_tokens"].as_i64().unwrap_or(0) as i32;
                                        let provider_completion_tokens =
                                            usage["completion_tokens"].as_i64().unwrap_or(0) as i32;

                                        debug!("Upstream processor: found usage - prompt_tokens={}, completion_tokens={}",
                                               provider_prompt_tokens, provider_completion_tokens);

                                        // Send to storage
                                        let _ = tx_storage_clone
                                            .send(StorageMessage::Usage {
                                                prompt_tokens: provider_prompt_tokens,
                                                completion_tokens: provider_completion_tokens,
                                            })
                                            .await;

                                        // Send to client (if still connected)
                                        let _ = tx_client_clone
                                            .send(StorageMessage::Usage {
                                                prompt_tokens: provider_prompt_tokens,
                                                completion_tokens: provider_completion_tokens,
                                            })
                                            .await;
                                    }

                                    // Check for finish reason
                                    if let Some(finish_reason) =
                                        json_data["choices"][0]["finish_reason"].as_str()
                                    {
                                        trace!(
                                            "Upstream processor: found finish_reason: {}",
                                            finish_reason
                                        );
                                        stream_finish_reason = Some(finish_reason.to_string());
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Upstream processor: error reading response: {:?}", e);
                        let _ = tx_storage_clone
                            .send(StorageMessage::Error(e.to_string()))
                            .await;
                        let _ = tx_client_clone
                            .send(StorageMessage::Error(e.to_string()))
                            .await;
                        break;
                    }
                }
            }

            // Clean up
            drop(tx_storage_clone);
            drop(tx_client_clone);
            debug!("Upstream processor task completed");
        })
    };

    trace!("Creating SSE event stream for client");
    let event_stream = async_stream::stream! {
        trace!("=== STARTING SSE STREAM ===");
        let mut sequence_number = 0i32;
        // Send initial response.created event
        trace!("Building response.created event");
        let created_response = ResponsesCreateResponse {
            id: response.uuid,
            object: "response",
            created_at: response.created_at.timestamp(),
            status: "in_progress",
            background: false,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: response.max_output_tokens,
            max_tool_calls: None,
            model: response.model.clone(),
            output: vec![],
            parallel_tool_calls: response.parallel_tool_calls,
            previous_response_id: None, // We no longer track previous_response_id
            prompt_cache_key: None,
            reasoning: ReasoningInfo { effort: None, summary: None },
            safety_identifier: None,
            store: response.store,
            temperature: response.temperature.unwrap_or(1.0),
            text: TextFormat { format: TextFormatSpec { format_type: "text".to_string() } },
            tool_choice: response.tool_choice.clone().unwrap_or_else(|| "auto".to_string()),
            tools: vec![],
            top_logprobs: 0,
            top_p: response.top_p.unwrap_or(1.0),
            truncation: "disabled",
            usage: None,
            user: None,
            metadata: decrypted_metadata.clone(),
        };

        let created_event = ResponseCreatedEvent {
            event_type: "response.created",
            response: created_response.clone(),
            sequence_number,
        };
        sequence_number += 1;

        match serde_json::to_value(&created_event) {
            Ok(created_json) => {
                trace!("Serialized response.created event");
                match encrypt_event(&state, &session_id, "response.created", &created_json).await {
                    Ok(event) => {
                        trace!("Yielding response.created event");
                        yield Ok(event)
                    },
                    Err(e) => {
                        error!("Failed to encrypt response.created event: {:?}", e);
                        yield Ok(Event::default().event("error").data("encryption_failed"));
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize response.created: {:?}", e);
                yield Ok(Event::default().event("error").data("serialization_failed"));
            }
        }

        // Event 2: response.in_progress
        let in_progress_event = ResponseInProgressEvent {
            event_type: "response.in_progress",
            response: created_response,
            sequence_number,
        };
        sequence_number += 1;

        match serde_json::to_value(&in_progress_event) {
            Ok(in_progress_json) => {
                match encrypt_event(&state, &session_id, "response.in_progress", &in_progress_json).await {
                    Ok(event) => {
                        trace!("Yielding response.in_progress event");
                        yield Ok(event)
                    },
                    Err(e) => {
                        error!("Failed to encrypt response.in_progress event: {:?}", e);
                        yield Ok(Event::default().event("error").data("encryption_failed"));
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize response.in_progress: {:?}", e);
                yield Ok(Event::default().event("error").data("serialization_failed"));
            }
        }

        // Process messages from upstream processor
        let mut assistant_content = String::new();
        let mut total_completion_tokens = 0i32;
        let mut message_id = Uuid::new_v4(); // Will be set by upstream processor

        // Event 3: response.output_item.added
        let output_item_added_event = ResponseOutputItemAddedEvent {
            event_type: "response.output_item.added",
            sequence_number,
            output_index: 0,
            item: OutputItem {
                id: message_id.to_string(),
                output_type: "message".to_string(),
                status: "in_progress".to_string(),
                role: Some("assistant".to_string()),
                content: Some(vec![]),
            },
        };
        sequence_number += 1;

        match serde_json::to_value(&output_item_added_event) {
            Ok(output_item_json) => {
                match encrypt_event(&state, &session_id, "response.output_item.added", &output_item_json).await {
                    Ok(event) => {
                        trace!("Yielding response.output_item.added event");
                        yield Ok(event)
                    },
                    Err(e) => {
                        error!("Failed to encrypt response.output_item.added event: {:?}", e);
                        yield Ok(Event::default().event("error").data("encryption_failed"));
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize response.output_item.added: {:?}", e);
                yield Ok(Event::default().event("error").data("serialization_failed"));
            }
        }

        // Event 4: response.content_part.added
        let content_part_added_event = ResponseContentPartAddedEvent {
            event_type: "response.content_part.added",
            sequence_number,
            item_id: message_id.to_string(),
            output_index: 0,
            content_index: 0,
            part: ContentPart {
                part_type: "output_text".to_string(),
                annotations: vec![],
                logprobs: vec![],
                text: String::new(),
            },
        };
        sequence_number += 1;

        match serde_json::to_value(&content_part_added_event) {
            Ok(content_part_json) => {
                match encrypt_event(&state, &session_id, "response.content_part.added", &content_part_json).await {
                    Ok(event) => {
                        trace!("Yielding response.content_part.added event");
                        yield Ok(event)
                    },
                    Err(e) => {
                        error!("Failed to encrypt response.content_part.added event: {:?}", e);
                        yield Ok(Event::default().event("error").data("encryption_failed"));
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize response.content_part.added: {:?}", e);
                yield Ok(Event::default().event("error").data("serialization_failed"));
            }
        }

        trace!("Starting to process messages from upstream processor");
        while let Some(msg) = rx_client.recv().await {
            trace!("Client stream received message from upstream processor");
            match msg {
                StorageMessage::Done { finish_reason, message_id: msg_id } => {
                    trace!("Client stream received Done signal with finish_reason={}", finish_reason);
                    message_id = msg_id;

                                // Event 7: response.output_text.done
                                let output_text_done_event = ResponseOutputTextDoneEvent {
                                    event_type: "response.output_text.done",
                                    sequence_number,
                                    item_id: message_id.to_string(),
                                    output_index: 0,
                                    content_index: 0,
                                    text: assistant_content.clone(),
                                    logprobs: vec![],
                                };
                                sequence_number += 1;

                                match serde_json::to_value(&output_text_done_event) {
                                    Ok(output_text_done_json) => {
                                        match encrypt_event(&state, &session_id, "response.output_text.done", &output_text_done_json).await {
                                            Ok(event) => {
                                                trace!("Yielding response.output_text.done event");
                                                yield Ok(event)
                                            },
                                            Err(e) => {
                                                error!("Failed to encrypt response.output_text.done event: {:?}", e);
                                                yield Ok(Event::default().event("error").data("encryption_failed"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize response.output_text.done: {:?}", e);
                                        yield Ok(Event::default().event("error").data("serialization_failed"));
                                    }
                                }

                                // Event 8: response.content_part.done
                                let content_part_done_event = ResponseContentPartDoneEvent {
                                    event_type: "response.content_part.done",
                                    sequence_number,
                                    item_id: message_id.to_string(),
                                    output_index: 0,
                                    content_index: 0,
                                    part: ContentPart {
                                        part_type: "output_text".to_string(),
                                        annotations: vec![],
                                        logprobs: vec![],
                                        text: assistant_content.clone(),
                                    },
                                };
                                sequence_number += 1;

                                match serde_json::to_value(&content_part_done_event) {
                                    Ok(content_part_done_json) => {
                                        match encrypt_event(&state, &session_id, "response.content_part.done", &content_part_done_json).await {
                                            Ok(event) => {
                                                trace!("Yielding response.content_part.done event");
                                                yield Ok(event)
                                            },
                                            Err(e) => {
                                                error!("Failed to encrypt response.content_part.done event: {:?}", e);
                                                yield Ok(Event::default().event("error").data("encryption_failed"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize response.content_part.done: {:?}", e);
                                        yield Ok(Event::default().event("error").data("serialization_failed"));
                                    }
                                }

                                // Event 9: response.output_item.done
                                let content_part = ContentPart {
                                    part_type: "output_text".to_string(),
                                    annotations: vec![],
                                    logprobs: vec![],
                                    text: assistant_content.clone(),
                                };

                                let output_item_done_event = ResponseOutputItemDoneEvent {
                                    event_type: "response.output_item.done",
                                    sequence_number,
                                    output_index: 0,
                                    item: OutputItem {
                                        id: message_id.to_string(),
                                        output_type: "message".to_string(),
                                        status: "completed".to_string(),
                                        role: Some("assistant".to_string()),
                                        content: Some(vec![content_part]),
                                    },
                                };
                                sequence_number += 1;

                                match serde_json::to_value(&output_item_done_event) {
                                    Ok(output_item_done_json) => {
                                        match encrypt_event(&state, &session_id, "response.output_item.done", &output_item_done_json).await {
                                            Ok(event) => {
                                                trace!("Yielding response.output_item.done event");
                                                yield Ok(event)
                                            },
                                            Err(e) => {
                                                error!("Failed to encrypt response.output_item.done event: {:?}", e);
                                                yield Ok(Event::default().event("error").data("encryption_failed"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize response.output_item.done: {:?}", e);
                                        yield Ok(Event::default().event("error").data("serialization_failed"));
                                    }
                                }

                                // Event 10: response.completed
                                let done_response = ResponsesCreateResponse {
                                    id: response.uuid,
                                    object: "response",
                                    created_at: response.created_at.timestamp(),
                                    status: "completed",
                                    background: false,
                                    error: None,
                                    incomplete_details: None,
                                    instructions: None,
                                    max_output_tokens: response.max_output_tokens,
                                    max_tool_calls: None,
                                    model: response.model.clone(),
                                    output: vec![OutputItem {
                                        id: message_id.to_string(),
                                        output_type: "message".to_string(),
                                        status: "completed".to_string(),
                                        role: Some("assistant".to_string()),
                                        content: Some(vec![ContentPart {
                                            part_type: "output_text".to_string(),
                                            annotations: vec![],
                                            logprobs: vec![],
                                            text: assistant_content.clone(),
                                        }]),
                                    }],
                                    parallel_tool_calls: response.parallel_tool_calls,
                                    previous_response_id: None, // We no longer track previous_response_id
                                    prompt_cache_key: None,
                                    reasoning: ReasoningInfo { effort: None, summary: None },
                                    safety_identifier: None,
                                    store: response.store,
                                    temperature: response.temperature.unwrap_or(1.0),
                                    text: TextFormat { format: TextFormatSpec { format_type: "text".to_string() } },
                                    tool_choice: response.tool_choice.clone().unwrap_or_else(|| "auto".to_string()),
                                    tools: vec![],
                                    top_logprobs: 0,
                                    top_p: response.top_p.unwrap_or(1.0),
                                    truncation: "disabled",
                                    usage: Some(ResponseUsage {
                                        input_tokens: total_prompt_tokens as i32,
                                        input_tokens_details: InputTokenDetails { cached_tokens: 0 },
                                        output_tokens: total_completion_tokens,
                                        output_tokens_details: OutputTokenDetails { reasoning_tokens: 0 },
                                        total_tokens: total_prompt_tokens as i32 + total_completion_tokens,
                                    }),
                                    user: None,
                                    metadata: decrypted_metadata.clone(),
                                };

                                let completed_event = ResponseCompletedEvent {
                                    event_type: "response.completed",
                                    response: done_response,
                                    sequence_number,
                                };

                                match serde_json::to_value(&completed_event) {
                                    Ok(completed_json) => {
                                        match encrypt_event(&state, &session_id, "response.completed", &completed_json).await {
                                            Ok(event) => yield Ok(event),
                                            Err(e) => {
                                                error!("Failed to encrypt response.completed event: {:?}", e);
                                                yield Ok(Event::default().event("error").data("encryption_failed"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize response.completed: {:?}", e);
                                        yield Ok(Event::default().event("error").data("serialization_failed"));
                                    }
                                }
                    break;
                }
                StorageMessage::ContentDelta(content) => {
                    trace!("Client stream received content delta: {}", content);
                    assistant_content.push_str(&content);

                    // Send to client
                    let delta_event = ResponseOutputTextDeltaEvent {
                        event_type: "response.output_text.delta",
                        delta: content.clone(),
                        item_id: message_id.to_string(),
                        output_index: 0,
                        content_index: 0,
                        sequence_number,
                        logprobs: vec![],
                    };
                    sequence_number += 1;

                    trace!("Sending response.output_text.delta event");
                    match serde_json::to_value(&delta_event) {
                        Ok(delta_json) => {
                            match encrypt_event(&state, &session_id, "response.output_text.delta", &delta_json).await {
                                Ok(event) => {
                                    trace!("Yielding response.output_text.delta event");
                                    yield Ok(event)
                                },
                                Err(e) => {
                                    error!("Failed to encrypt response.output_text.delta event: {:?}", e);
                                    yield Ok(Event::default().event("error").data("encryption_failed"));
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to serialize response.output_text.delta: {:?}", e);
                            yield Ok(Event::default().event("error").data("serialization_failed"));
                        }
                    }
                }

                StorageMessage::Usage { prompt_tokens: _, completion_tokens } => {
                    trace!("Client stream received usage data");
                    total_completion_tokens = completion_tokens;
                }
                StorageMessage::Error(error_msg) => {
                    error!("Client stream received error: {}", error_msg);
                    // Send error event to client
                    let error_event = ResponseErrorEvent {
                        event_type: "response.error",
                        error: ResponseError {
                            error_type: "stream_error".to_string(),
                            message: error_msg,
                        },
                    };

                    match serde_json::to_value(&error_event) {
                        Ok(error_json) => {
                            match encrypt_event(&state, &session_id, "response.error", &error_json).await {
                                Ok(event) => yield Ok(event),
                                Err(e) => {
                                    error!("Failed to encrypt error event: {:?}", e);
                                    yield Ok(Event::default().event("error").data("encryption_failed"));
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to serialize response.error: {:?}", e);
                            yield Ok(Event::default().event("error").data("serialization_failed"));
                        }
                    }
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

/// Storage task that accumulates content and writes to database
async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    db: Arc<dyn crate::DBConnection + Send + Sync>,
    response_id: i64,
    conversation_id: i64,
    user_key: secp256k1::SecretKey,
    user_uuid: Uuid,
    sqs_publisher: Option<Arc<crate::sqs::SqsEventPublisher>>,
) {
    let mut content = String::new();
    let mut completion_tokens = 0i32;
    let mut prompt_tokens = 0i32;
    let mut finish_reason = String::new();
    let mut message_id = Uuid::new_v4(); // Default, will be overridden
    let mut error_msg: Option<String> = None;

    // Accumulate content from stream
    while let Some(msg) = rx.recv().await {
        match msg {
            StorageMessage::ContentDelta(delta) => {
                trace!("Storage: received content delta: {} chars", delta.len());
                content.push_str(&delta);
            }
            StorageMessage::Usage {
                prompt_tokens: p_tokens,
                completion_tokens: c_tokens,
            } => {
                debug!(
                    "Storage: received usage - prompt_tokens={}, completion_tokens={}",
                    p_tokens, c_tokens
                );
                prompt_tokens = p_tokens;
                completion_tokens = c_tokens;
            }
            StorageMessage::Done {
                finish_reason: reason,
                message_id: msg_id,
            } => {
                debug!(
                    "Storage: received Done signal with finish_reason={}, message_id={}",
                    reason, msg_id
                );
                finish_reason = reason;
                message_id = msg_id;
                break;
            }
            StorageMessage::Error(e) => {
                error!("Storage: received error: {}", e);
                error_msg = Some(e);
                break;
            }
        }
    }

    // If we exit the loop without Done or Error, it means all senders were dropped
    if error_msg.is_none() && finish_reason.is_empty() {
        warn!("Storage: channel closed before receiving Done signal, NOT persisting incomplete content");
        // Don't save partial content - the upstream processor should complete normally
        // If it doesn't, something went wrong and we shouldn't persist bad data
        if let Err(e) = db.update_response_status(
            response_id,
            ResponseStatus::Failed,
            Some(Utc::now()),
            None,
            None,
        ) {
            error!("Failed to update response status to failed: {:?}", e);
        }
        return;
    }

    // Handle error case
    if let Some(_error) = error_msg {
        if let Err(e) = db.update_response_status(
            response_id,
            ResponseStatus::Failed,
            Some(Utc::now()),
            None,
            None,
        ) {
            error!("Failed to update response status to failed: {:?}", e);
        }
        return;
    }

    // Fallback token counting if not provided
    if completion_tokens == 0 && !content.is_empty() {
        completion_tokens = count_tokens(&content) as i32;
        debug!(
            "No completion tokens from upstream, counted {} tokens from content",
            completion_tokens
        );
    }

    // Encrypt and store assistant message as plain text
    let content_enc = encrypt_with_key(&user_key, content.as_bytes()).await;

    let new_assistant = NewAssistantMessage {
        uuid: message_id,
        conversation_id,
        response_id: Some(response_id),
        user_id: user_uuid,
        content_enc,
        completion_tokens,
        finish_reason: Some(finish_reason),
    };

    match db.create_assistant_message(new_assistant) {
        Ok(_) => {
            debug!("Successfully stored assistant message");
        }
        Err(e) => {
            error!("Failed to create assistant message: {:?}", e);
        }
    }

    // Handle billing in background thread
    if prompt_tokens > 0 || completion_tokens > 0 {
        let db_clone = db.clone();
        let sqs_pub = sqs_publisher.clone();

        tokio::spawn(async move {
            // Calculate estimated cost with correct pricing
            let input_cost =
                BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(prompt_tokens);
            let output_cost =
                BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(completion_tokens);
            let total_cost = input_cost + output_cost;

            info!(
                "Responses API usage for user {}: prompt_tokens={}, completion_tokens={}, total_tokens={}, estimated_cost={}",
                user_uuid, prompt_tokens, completion_tokens,
                prompt_tokens + completion_tokens,
                total_cost
            );

            // Create and store token usage record
            let new_usage = NewTokenUsage::new(
                user_uuid,
                prompt_tokens,
                completion_tokens,
                total_cost.clone(),
            );

            if let Err(e) = db_clone.create_token_usage(new_usage) {
                error!("Failed to save token usage: {:?}", e);
            }

            // Post event to SQS if configured
            if let Some(publisher) = sqs_pub {
                let event = UsageEvent {
                    event_id: Uuid::new_v4(),
                    user_id: user_uuid,
                    input_tokens: prompt_tokens,
                    output_tokens: completion_tokens,
                    estimated_cost: total_cost,
                    chat_time: Utc::now(),
                    is_api_request: false, // TODO: Responses API is not API key based
                    provider_name: String::new(), // TODO: Track provider name from upstream response
                    model_name: String::new(),    // TODO: Track actual model name used
                };

                match publisher.publish_event(event).await {
                    Ok(_) => debug!("published usage event successfully"),
                    Err(e) => error!("error publishing usage event: {e}"),
                }
            }
        });
    }

    // Update response status to completed with token counts
    if let Err(e) = db.update_response_status(
        response_id,
        ResponseStatus::Completed,
        Some(Utc::now()),
        Some(prompt_tokens),
        Some(completion_tokens),
    ) {
        error!("Failed to update response status to completed: {:?}", e);
    }
}

/// Helper to create encrypted SSE event
async fn encrypt_event(
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

/// Persist initial response and user message
async fn persist_initial_message(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    content_enc: Vec<u8>,
    user_message_tokens: i32,
    user_key: &SecretKey,
) -> Result<
    (
        crate::models::responses::Conversation,
        crate::models::responses::Response,
        crate::models::responses::UserMessage,
    ),
    ApiError,
> {
    use crate::models::responses::{NewResponse, ResponseStatus};

    // Determine which conversation to use
    let conversation_id = match &body.conversation {
        Some(ConversationParam::String(id)) | Some(ConversationParam::Object { id }) => Some(*id),
        None => None,
    };

    if let Some(conv_uuid) = conversation_id {
        // Use specified conversation
        debug!("Using specified conversation: {}", conv_uuid);
        let conversation = state
            .db
            .get_conversation_by_uuid_and_user(conv_uuid, user.uuid)
            .map_err(|e| {
                error!("Error fetching conversation: {:?}", e);
                match e {
                    DBError::ResponsesError(ResponsesError::ConversationNotFound) => {
                        ApiError::NotFound
                    }
                    _ => ApiError::InternalServerError,
                }
            })?;

        // Encrypt metadata if provided
        let metadata_enc = if let Some(metadata) = &body.metadata {
            let metadata_json = serde_json::to_string(metadata).map_err(|e| {
                error!("Failed to serialize metadata: {:?}", e);
                ApiError::InternalServerError
            })?;
            Some(encrypt_with_key(user_key, metadata_json.as_bytes()).await)
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
            metadata_enc,
        };
        let response = state.db.create_response(new_response).map_err(|e| {
            error!("Error creating response: {:?}", e);
            ApiError::InternalServerError
        })?;

        // Create the simplified user message
        let new_msg = NewUserMessage {
            uuid: Uuid::new_v4(),
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: user.uuid,
            content_enc: content_enc.clone(),
            prompt_tokens: user_message_tokens,
        };
        let user_message = state.db.create_user_message(new_msg).map_err(|e| {
            error!("Error creating user message: {:?}", e);
            ApiError::InternalServerError
        })?;

        Ok((conversation, response, user_message))
    } else {
        // Create new conversation
        debug!(
            "Creating new conversation with response and message for user: {}",
            user.uuid
        );

        let conversation_uuid = Uuid::new_v4();
        let response_uuid = Uuid::new_v4();

        // Create metadata with default title
        let mut metadata = body.metadata.clone().unwrap_or_else(|| json!({}));
        if metadata.get("title").is_none() {
            metadata["title"] = json!("New Chat");
        }

        // Encrypt the metadata
        let metadata_json = serde_json::to_string(&metadata).map_err(|e| {
            error!("Failed to serialize metadata: {:?}", e);
            ApiError::InternalServerError
        })?;
        let conversation_metadata_enc =
            Some(encrypt_with_key(user_key, metadata_json.as_bytes()).await);

        // Encrypt response metadata if provided
        let response_metadata_enc = if let Some(resp_metadata) = &body.metadata {
            let resp_metadata_json = serde_json::to_string(resp_metadata).map_err(|e| {
                error!("Failed to serialize response metadata: {:?}", e);
                ApiError::InternalServerError
            })?;
            Some(encrypt_with_key(user_key, resp_metadata_json.as_bytes()).await)
        } else {
            None
        };

        // Prepare the Response
        let new_response = NewResponse {
            uuid: response_uuid,
            user_id: user.uuid,
            conversation_id: 0, // Will be set by create_conversation_with_response_and_message
            status: ResponseStatus::InProgress,
            model: body.model.clone(),
            temperature: body.temperature,
            top_p: body.top_p,
            max_output_tokens: body.max_output_tokens,
            tool_choice: body.tool_choice.clone(),
            parallel_tool_calls: body.parallel_tool_calls,
            store: body.store,
            metadata_enc: response_metadata_enc,
        };

        // Use transactional method to create conversation, response, and message atomically
        state
            .db
            .create_conversation_with_response_and_message(
                conversation_uuid,
                user.uuid,
                None, // system_prompt_id
                conversation_metadata_enc,
                Some(new_response),
                content_enc,
                user_message_tokens,
            )
            .map_err(|e| {
                error!(
                    "Error creating conversation with response and message: {:?}",
                    e
                );
                ApiError::InternalServerError
            })
            .map(|(conv, resp, msg)| (conv, resp.unwrap(), msg))
    }
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
        .map_err(|e| {
            debug!("Response {} not found for user {}: {:?}", id, user.uuid, e);
            match e {
                DBError::ResponsesError(ResponsesError::ResponseNotFound) => ApiError::NotFound,
                DBError::ResponsesError(ResponsesError::Unauthorized) => ApiError::Unauthorized,
                _ => ApiError::InternalServerError,
            }
        })?;

    // Get associated assistant messages
    let assistant_messages = state
        .db
        .get_assistant_messages_for_response(response.id)
        .map_err(|e| {
            error!("Failed to get assistant messages: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|e| {
            error!("Failed to get user encryption key: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Build output from assistant messages
    let mut output = String::new();
    for msg in &assistant_messages {
        let decrypted = decrypt_with_key(&user_key, &msg.content_enc).unwrap_or_else(|_| vec![]);
        let text = String::from_utf8_lossy(&decrypted);
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&text);
    }

    // Use the stored token counts from the response
    let usage = if response.status == ResponseStatus::Completed {
        let input_tokens = response.input_tokens.unwrap_or(0);
        let output_tokens = response.output_tokens.unwrap_or(0);

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
        object: "response",
        created_at: response.created_at.timestamp(),
        status: serde_json::to_value(response.status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "unknown".to_string()),
        model: response.model.clone(),
        usage,
        output: if output.is_empty() {
            None
        } else {
            Some(output)
        },
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

    // TODO we actually need to try canceling the stream and also make sure the user and assistant
    // message from it is not persisted.

    // Cancel the response
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
        object: "response",
        created_at: response.created_at.timestamp(),
        status: "cancelled".to_string(),
        model: response.model.clone(),
        usage: None,
        output: None,
    };

    encrypt_response(&state, &session_id, &retrieve_response).await
}

/// DELETE /v1/responses/{id} - Hard delete a response
async fn delete_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ResponsesDeleteResponse>>, ApiError> {
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

    let response = ResponsesDeleteResponse {
        id,
        object: "response.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}
