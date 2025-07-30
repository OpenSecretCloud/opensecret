//! Responses API implementation with SSE streaming and dual-stream storage.
//! Phases 4 & 5: Always streams to client while concurrently storing to database.

use crate::{
    context_builder::build_prompt,
    db::DBError,
    encrypt::encrypt_with_key,
    models::responses::{NewAssistantMessage, NewUserMessage, ResponseStatus, ResponsesError},
    models::token_usage::NewTokenUsage,
    models::users::User,
    sqs::UsageEvent,
    tokens::count_tokens,
    web::{encryption_middleware::decrypt_request, openai::get_chat_completion_response},
    ApiError, AppState,
};
use axum::{
    extract::State,
    http::HeaderMap,
    response::sse::{Event, Sse},
    routing::post,
    Extension, Router,
};
use base64::Engine;
use bigdecimal::BigDecimal;
use chrono::Utc;
use futures::{Stream, StreamExt, TryStreamExt};
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

/// Request payload for creating a new response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesCreateRequest {
    /// Model to use for the response
    pub model: String,

    /// User's input message
    pub input: String,

    /// Previous response ID to continue conversation
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

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/responses", post(create_response_stream))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            decrypt_request::<ResponsesCreateRequest>,
        ))
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

    // Count tokens for the user's input message
    let user_message_tokens = count_tokens(&body.input) as i32;

    // Encrypt user input
    let content_enc = encrypt_with_key(&user_key, body.input.as_bytes()).await;

    // Create thread and user message
    let (thread, user_message) = persist_initial_message(
        &state,
        &user,
        &body,
        content_enc.clone(),
        user_message_tokens,
    )
    .await?;

    info!(
        "Created response {} for user {} in thread {}",
        user_message.uuid, user.uuid, thread.uuid
    );

    // Build the conversation context from all persisted messages
    let (prompt_messages, total_prompt_tokens) =
        build_prompt(state.db.as_ref(), thread.id, &user_key, &body.model)?;

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
        "max_tokens": body.max_output_tokens.unwrap_or(512),
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

    // Call the chat API
    let upstream_response =
        get_chat_completion_response(&state, &user, chat_request, &headers).await?;

    // Create channel for storage task
    let (tx_storage, rx_storage) = mpsc::channel::<StorageMessage>(1024);

    // Spawn storage task
    let storage_handle = {
        let db = state.db.clone();
        let user_message_id = user_message.id;
        let thread_id = thread.id;
        let user_uuid = user.uuid;
        let sqs_publisher = state.sqs_publisher.clone();

        tokio::spawn(async move {
            storage_task(
                rx_storage,
                db,
                user_message_id,
                thread_id,
                user_key,
                user_uuid,
                sqs_publisher,
            )
            .await;
        })
    };

    // Create the SSE stream
    let mut body_stream = upstream_response.into_body().into_stream();
    let tx_storage_clone = tx_storage.clone();

    trace!("Creating SSE event stream");
    let event_stream = async_stream::stream! {
        trace!("=== STARTING SSE STREAM ===");
        let mut sequence_number = 0i32;
        // Send initial response.created event
        trace!("Building response.created event");
        let created_response = ResponsesCreateResponse {
            id: user_message.uuid,
            object: "response",
            created_at: user_message.created_at.timestamp(),
            status: "in_progress",
            background: false,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: user_message.max_output_tokens,
            max_tool_calls: None,
            model: user_message.model.clone(),
            output: vec![],
            parallel_tool_calls: user_message.parallel_tool_calls,
            previous_response_id: user_message.previous_response_id,
            prompt_cache_key: None,
            reasoning: ReasoningInfo { effort: None, summary: None },
            safety_identifier: None,
            store: user_message.store,
            temperature: user_message.temperature.unwrap_or(1.0),
            text: TextFormat { format: TextFormatSpec { format_type: "text".to_string() } },
            tool_choice: user_message.tool_choice.clone().unwrap_or_else(|| "auto".to_string()),
            tools: vec![],
            top_logprobs: 0,
            top_p: user_message.top_p.unwrap_or(1.0),
            truncation: "disabled",
            usage: None,
            user: None,
            metadata: user_message.metadata.clone(),
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

        // Process upstream chunks
        let mut buffer = String::new();
        let mut assistant_content = String::new();
        let mut total_completion_tokens = 0i32;
        let message_id = Uuid::new_v4();
        let mut stream_finish_reason: Option<String> = None; // Track finish_reason from stream

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

        trace!("Starting to process upstream chunks");
        while let Some(chunk_result) = body_stream.next().await {
            trace!("Received chunk from upstream");
            match chunk_result {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

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
                                trace!("Received [DONE] from upstream, sending completion events");

                                // Signal completion to storage
                                let _ = tx_storage_clone.send(StorageMessage::Done {
                                    finish_reason: stream_finish_reason.clone().unwrap_or_else(|| {
                                        error!("No finish_reason received from stream, this should not happen!");
                                        "error".to_string()
                                    }),
                                    message_id,
                                }).await;

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
                                    id: user_message.uuid,
                                    object: "response",
                                    created_at: user_message.created_at.timestamp(),
                                    status: "completed",
                                    background: false,
                                    error: None,
                                    incomplete_details: None,
                                    instructions: None,
                                    max_output_tokens: user_message.max_output_tokens,
                                    max_tool_calls: None,
                                    model: user_message.model.clone(),
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
                                    parallel_tool_calls: user_message.parallel_tool_calls,
                                    previous_response_id: user_message.previous_response_id,
                                    prompt_cache_key: None,
                                    reasoning: ReasoningInfo { effort: None, summary: None },
                                    safety_identifier: None,
                                    store: user_message.store,
                                    temperature: user_message.temperature.unwrap_or(1.0),
                                    text: TextFormat { format: TextFormatSpec { format_type: "text".to_string() } },
                                    tool_choice: user_message.tool_choice.clone().unwrap_or_else(|| "auto".to_string()),
                                    tools: vec![],
                                    top_logprobs: 0,
                                    top_p: user_message.top_p.unwrap_or(1.0),
                                    truncation: "disabled",
                                    usage: Some(ResponseUsage {
                                        input_tokens: total_prompt_tokens as i32,
                                        input_tokens_details: InputTokenDetails { cached_tokens: 0 },
                                        output_tokens: total_completion_tokens,
                                        output_tokens_details: OutputTokenDetails { reasoning_tokens: 0 },
                                        total_tokens: total_prompt_tokens as i32 + total_completion_tokens,
                                    }),
                                    user: None,
                                    metadata: user_message.metadata.clone(),
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

                            // Parse JSON data
                            if let Ok(json_data) = serde_json::from_str::<Value>(data) {
                                trace!("Parsed JSON chunk: has content={}, has usage={}, has finish_reason={}",
                                       json_data["choices"][0]["delta"]["content"].is_string(),
                                       json_data.get("usage").is_some(),
                                       json_data["choices"][0]["finish_reason"].is_string());

                                // Extract content delta
                                if let Some(content) = json_data["choices"][0]["delta"]["content"].as_str() {
                                    trace!("Found content delta: {}", content);
                                    assistant_content.push_str(content);

                                    // Send to storage
                                    let _ = tx_storage_clone.send(StorageMessage::ContentDelta(content.to_string())).await;

                                    // Send to client
                                    let delta_event = ResponseOutputTextDeltaEvent {
                                        event_type: "response.output_text.delta",
                                        delta: content.to_string(),
                                        item_id: message_id.to_string(),
                                        output_index: 0,
                                        content_index: 0,
                                        sequence_number,
                                        logprobs: vec![],
                                    };
                                    sequence_number += 1;

                                    trace!("Sending response.output_text.delta event with content: {}", content);
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

                                // Extract usage (usually in final chunk)
                                if let Some(usage) = json_data.get("usage") {
                                    debug!("Found usage data in chunk: {:?}", usage);

                                    // Get the actual token counts from the provider
                                    let provider_prompt_tokens = usage["prompt_tokens"].as_i64().unwrap_or(0) as i32;
                                    let provider_completion_tokens = usage["completion_tokens"].as_i64().unwrap_or(0) as i32;

                                    total_completion_tokens = provider_completion_tokens;
                                    debug!("Provider reported usage: prompt_tokens={}, completion_tokens={}",
                                           provider_prompt_tokens, provider_completion_tokens);

                                    // IMMEDIATELY send the provider's actual token counts to storage
                                    debug!("Immediately sending provider usage to storage: prompt_tokens={}, completion_tokens={}",
                                           provider_prompt_tokens, provider_completion_tokens);
                                    let _ = tx_storage_clone.send(StorageMessage::Usage {
                                        prompt_tokens: provider_prompt_tokens,
                                        completion_tokens: provider_completion_tokens,
                                    }).await;
                                }

                                // Check for finish reason but don't send Done yet
                                // The Done message will be sent when we see [DONE]
                                if let Some(finish_reason) = json_data["choices"][0]["finish_reason"].as_str() {
                                    trace!("Found finish_reason: {}, but waiting for [DONE] to send Done message", finish_reason);
                                    stream_finish_reason = Some(finish_reason.to_string());
                                }
                            } else {
                                trace!("Failed to parse JSON from upstream: {}", data);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Error reading upstream response: {:?}", e);
                    let _ = tx_storage_clone.send(StorageMessage::Error(e.to_string())).await;

                    // Send error event to client
                    let error_event = ResponseErrorEvent {
                        event_type: "response.error",
                        error: ResponseError {
                            error_type: "stream_error".to_string(),
                            message: "Upstream connection error".to_string(),
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

        // Ensure storage task completes
        drop(tx_storage_clone);
        let _ = storage_handle.await;
    };

    trace!("Returning SSE stream");
    Ok(Sse::new(event_stream))
}

/// Storage task that accumulates content and writes to database
async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    db: Arc<dyn crate::DBConnection + Send + Sync>,
    user_message_id: i64,
    thread_id: i64,
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
        warn!("Storage: channel closed before receiving Done signal, saving partial content");
        finish_reason = "error".to_string(); // No finish reason means something went wrong
    }

    // Handle error case
    if let Some(error) = error_msg {
        if let Err(e) = db.update_user_message_status(
            user_message_id,
            ResponseStatus::Failed,
            Some(error),
            Some(Utc::now()),
        ) {
            error!("Failed to update user message status to failed: {:?}", e);
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

    // Encrypt and store assistant message
    let content_enc = encrypt_with_key(&user_key, content.as_bytes()).await;

    let new_assistant = NewAssistantMessage {
        uuid: message_id,
        thread_id,
        user_message_id,
        content_enc,
        completion_tokens: Some(completion_tokens),
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
                    model_name: String::new(), // TODO: Track actual model name used
                };

                match publisher.publish_event(event).await {
                    Ok(_) => debug!("published usage event successfully"),
                    Err(e) => error!("error publishing usage event: {e}"),
                }
            }
        });
    }

    // Update user message status to completed
    if let Err(e) = db.update_user_message_status(
        user_message_id,
        ResponseStatus::Completed,
        None,
        Some(Utc::now()),
    ) {
        error!("Failed to update user message status to completed: {:?}", e);
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

/// Persist initial user message and possibly create new thread
async fn persist_initial_message(
    state: &Arc<AppState>,
    user: &User,
    body: &ResponsesCreateRequest,
    content_enc: Vec<u8>,
    user_message_tokens: i32,
) -> Result<
    (
        crate::models::responses::ChatThread,
        crate::models::responses::UserMessage,
    ),
    ApiError,
> {
    // Thread resolution/creation with message
    if let Some(prev_id) = body.previous_response_id {
        debug!("Looking up previous response: {}", prev_id);

        // Get the previous message to find the thread
        let prev_msg = state
            .db
            .get_user_message_by_uuid(prev_id, user.uuid)
            .map_err(|e| {
                error!("Error fetching previous response: {:?}", e);
                match e {
                    DBError::ResponsesError(ResponsesError::UserMessageNotFound) => {
                        ApiError::NotFound
                    }
                    DBError::ResponsesError(ResponsesError::Unauthorized) => ApiError::Unauthorized,
                    _ => ApiError::InternalServerError,
                }
            })?;

        // Get the thread
        let thread = state
            .db
            .get_thread_by_id_and_user(prev_msg.thread_id, user.uuid)
            .map_err(|e| {
                error!("Error fetching thread: {:?}", e);
                ApiError::InternalServerError
            })?;

        // Create message for existing thread
        let new_msg = NewUserMessage {
            uuid: Uuid::new_v4(),
            thread_id: thread.id,
            user_id: user.uuid,
            content_enc: content_enc.clone(),
            prompt_tokens: Some(user_message_tokens),
            status: ResponseStatus::InProgress,
            model: body.model.clone(),
            previous_response_id: body.previous_response_id,
            temperature: body.temperature,
            top_p: body.top_p,
            max_output_tokens: body.max_output_tokens,
            tool_choice: body.tool_choice.clone(),
            parallel_tool_calls: body.parallel_tool_calls,
            store: body.store,
            metadata: body.metadata.clone(),
            idempotency_key: None,
            request_hash: None,
            idempotency_expires_at: None,
        };

        let inserted = state.db.create_user_message(new_msg).map_err(|e| {
            error!("Error creating user message: {:?}", e);
            ApiError::InternalServerError
        })?;

        Ok((thread, inserted))
    } else {
        debug!(
            "Creating new thread with first message for user: {}",
            user.uuid
        );

        // Create new thread with UUID = message UUID
        let thread_uuid = Uuid::new_v4();

        // Prepare the first message
        let first_message = NewUserMessage {
            uuid: thread_uuid,
            thread_id: 0, // Will be set by create_thread_with_first_message
            user_id: user.uuid,
            content_enc: content_enc.clone(),
            prompt_tokens: Some(user_message_tokens),
            status: ResponseStatus::InProgress,
            model: body.model.clone(),
            previous_response_id: None,
            temperature: body.temperature,
            top_p: body.top_p,
            max_output_tokens: body.max_output_tokens,
            tool_choice: body.tool_choice.clone(),
            parallel_tool_calls: body.parallel_tool_calls,
            store: body.store,
            metadata: body.metadata.clone(),
            idempotency_key: None,
            request_hash: None,
            idempotency_expires_at: None,
        };

        // Use transactional method to create both thread and message atomically
        state
            .db
            .create_thread_with_first_message(
                thread_uuid,
                user.uuid,
                None, // system_prompt_id - will be added in later phases
                first_message,
            )
            .map_err(|e| {
                error!("Error creating thread with first message: {:?}", e);
                ApiError::InternalServerError
            })
    }
}
