use crate::context_builder::build_prompt;
use crate::db::DBError;
use crate::encrypt::encrypt_with_key;
use crate::models::responses::{
    NewAssistantMessage, NewUserMessage, ResponseStatus, ResponsesError,
};
use crate::models::users::User;
use crate::tokens::count_tokens;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::openai::get_chat_completion_response;
use crate::{ApiError, AppState};
use axum::http::HeaderMap;
use axum::{extract::State, routing::post, Extension, Json, Router};
use chrono::Utc;
use hyper::body::to_bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info};
use uuid::Uuid;

// Constants
const IDEMPOTENCY_HEADER: &str = "idempotency-key";

// Default functions for serde
fn default_store() -> bool {
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

    /// Tool choice strategy (ignored in Phase 2)
    #[serde(default)]
    pub tool_choice: Option<String>,

    /// Tools available for the model (ignored in Phase 2)
    #[serde(default)]
    pub tools: Option<Value>,

    /// Enable parallel tool calls (ignored in Phase 2)
    #[serde(default)]
    pub parallel_tool_calls: bool,

    /// Whether to store the conversation (defaults to true)
    #[serde(default = "default_store")]
    pub store: bool,

    /// Arbitrary metadata
    #[serde(default)]
    pub metadata: Option<Value>,
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

    /// Model used for the response
    pub model: String,

    /// Current status (always "in_progress" for immediate response)
    pub status: &'static str,

    /// Output array (empty for in_progress responses)
    pub output: Vec<OutputItem>,

    /// Error information (null for successful requests)
    pub error: Option<ResponseError>,

    /// Details about why the response is incomplete
    pub incomplete_details: Option<serde_json::Value>,

    /// Usage statistics (null for in_progress)
    pub usage: Option<ResponseUsage>,

    /// Metadata from the request
    pub metadata: Option<Value>,

    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: bool,

    /// Previous response ID if continuing a conversation
    pub previous_response_id: Option<Uuid>,

    /// Whether the response is stored
    pub store: bool,

    /// Temperature setting
    pub temperature: Option<f32>,

    /// Top-p setting
    pub top_p: Option<f32>,

    /// Tool choice setting
    pub tool_choice: Option<String>,

    /// Available tools (empty array for Phase 2)
    pub tools: Vec<serde_json::Value>,
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
    pub content: Option<Vec<serde_json::Value>>,
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
        .route("/v1/responses", post(create_response))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            decrypt_request::<ResponsesCreateRequest>,
        ))
        .with_state(state)
}

async fn create_response(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<ResponsesCreateRequest>,
) -> Result<Json<EncryptedResponse<ResponsesCreateResponse>>, ApiError> {
    debug!("Creating new response for user: {}", user.uuid);

    // Prevent guest users from using the Responses API
    if user.is_guest() {
        error!("Guest user attempted to use Responses API: {}", user.uuid);
        return Err(ApiError::Unauthorized);
    }

    // 1. Handle idempotency if header is present
    let idempotency_key = headers
        .get(IDEMPOTENCY_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if let Some(ref key) = idempotency_key {
        debug!("Checking idempotency key: {}", key);

        // Check for existing request with this key
        match state.db.get_user_message_by_idempotency_key(user.uuid, key) {
            Ok(Some(existing_msg)) => {
                // Verify request hash matches
                let request_json =
                    serde_json::to_string(&body).map_err(|_| ApiError::InternalServerError)?;
                let request_hash = format!("{:x}", sha2::Sha256::digest(request_json.as_bytes()));

                if existing_msg.request_hash.as_deref() == Some(&request_hash) {
                    match existing_msg.status {
                        ResponseStatus::InProgress => {
                            // Return 409 Conflict for in-progress requests
                            error!("Request still in progress for idempotency key: {}", key);
                            return Err(ApiError::Conflict);
                        }
                        _ => {
                            // Return cached response with actual status
                            info!("Returning cached response for idempotency key: {}", key);

                            // Map ResponseStatus enum to string
                            let status_str = match existing_msg.status {
                                ResponseStatus::Queued => "queued",
                                ResponseStatus::InProgress => "in_progress",
                                ResponseStatus::Completed => "completed",
                                ResponseStatus::Failed => "failed",
                                ResponseStatus::Cancelled => "cancelled",
                            };

                            // For completed responses, we should include output and usage
                            // But in Phase 2, we don't have that data yet
                            let output = if existing_msg.status == ResponseStatus::Completed {
                                // TODO: In Phase 3, retrieve assistant message and format as output
                                vec![]
                            } else {
                                vec![]
                            };

                            let usage = if existing_msg.status == ResponseStatus::Completed {
                                // TODO: In Phase 3, calculate actual token usage
                                None
                            } else {
                                None
                            };

                            let response = ResponsesCreateResponse {
                                id: existing_msg.uuid,
                                object: "response",
                                created_at: existing_msg.created_at.timestamp(),
                                model: existing_msg.model,
                                status: status_str,
                                output,
                                error: existing_msg.error.as_ref().map(|e| ResponseError {
                                    error_type: "response_error".to_string(),
                                    message: e.clone(),
                                }),
                                incomplete_details: None, // TODO: Populate in later phases
                                usage,
                                metadata: existing_msg.metadata.clone(),
                                parallel_tool_calls: existing_msg.parallel_tool_calls,
                                previous_response_id: existing_msg.previous_response_id,
                                store: existing_msg.store,
                                temperature: existing_msg.temperature,
                                top_p: existing_msg.top_p,
                                tool_choice: existing_msg.tool_choice.clone(),
                                tools: vec![], // No tools in Phase 2
                            };
                            return encrypt_response(&state, &session_id, &response).await;
                        }
                    }
                } else {
                    // Different request body with same key
                    error!("Idempotency key reused with different request body");
                    return Err(ApiError::UnprocessableEntity);
                }
            }
            Ok(None) => {
                // No existing request, proceed
                debug!("No existing request found for idempotency key");
            }
            Err(e) => {
                error!("Error checking idempotency key: {:?}", e);
                return Err(ApiError::InternalServerError);
            }
        }
    }

    // 2. Prepare idempotency fields if key is present
    let (idempotency_key, request_hash, idempotency_expires_at) = if let Some(key) = idempotency_key
    {
        let request_json =
            serde_json::to_string(&body).map_err(|_| ApiError::InternalServerError)?;
        let hash = format!("{:x}", sha2::Sha256::digest(request_json.as_bytes()));
        let expires_at = Utc::now() + chrono::Duration::hours(24);
        (Some(key), Some(hash), Some(expires_at))
    } else {
        (None, None, None)
    };

    // 3. Get user's encryption key (needed before creating message)
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|e| {
            error!("Failed to get user encryption key: {:?}", e);
            ApiError::InternalServerError
        })?;

    // 4. Encrypt user input
    let content_enc = encrypt_with_key(&user_key, body.input.as_bytes()).await;

    // 5. Thread resolution/creation with message
    let (thread, inserted) = if let Some(prev_id) = body.previous_response_id {
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
            prompt_tokens: None, // Will be calculated in Phase 3
            status: ResponseStatus::InProgress,
            model: body.model.clone(),
            previous_response_id: body.previous_response_id,
            temperature: body.temperature,
            top_p: body.top_p,
            max_output_tokens: body.max_output_tokens,
            tool_choice: body.tool_choice,
            parallel_tool_calls: body.parallel_tool_calls,
            store: body.store,
            metadata: body.metadata,
            idempotency_key,
            request_hash,
            idempotency_expires_at,
        };

        let inserted = state.db.create_user_message(new_msg).map_err(|e| {
            error!("Error creating user message: {:?}", e);
            ApiError::InternalServerError
        })?;

        (thread, inserted)
    } else {
        debug!(
            "Creating new thread with first message for user: {}",
            user.uuid
        );

        // Create new thread with UUID = message UUID (as per spec)
        let thread_uuid = Uuid::new_v4();

        // Prepare the first message
        let first_message = NewUserMessage {
            uuid: thread_uuid, // Message UUID = Thread UUID for first message
            thread_id: 0,      // Will be set by create_with_first_message
            user_id: user.uuid,
            content_enc: content_enc.clone(),
            prompt_tokens: None, // Will be calculated in Phase 3
            status: ResponseStatus::InProgress,
            model: body.model.clone(),
            previous_response_id: None,
            temperature: body.temperature,
            top_p: body.top_p,
            max_output_tokens: body.max_output_tokens,
            tool_choice: body.tool_choice,
            parallel_tool_calls: body.parallel_tool_calls,
            store: body.store,
            metadata: body.metadata,
            idempotency_key,
            request_hash,
            idempotency_expires_at,
        };

        // Use transactional method to create both thread and message atomically
        let (thread, message) = state
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
            })?;

        (thread, message)
    };

    info!(
        "Created response {} for user {} in thread {}",
        inserted.uuid, user.uuid, thread.uuid
    );

    // Phase 3: Build prompt and call chat completion

    // Calculate tokens for just the user's input message
    let user_input_tokens = count_tokens(&body.input) as i32;

    // Update user message with its own token count
    state
        .db
        .update_user_message_prompt_tokens(inserted.id, user_input_tokens)
        .map_err(|e| {
            error!("Failed to update prompt tokens: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Build the conversation context for the chat API
    let (prompt_messages, total_prompt_tokens) = build_prompt(
        state.db.as_ref(),
        thread.id,
        &user_key,
        &body.model,
        &body.input,
        content_enc,
    )?;

    // Build chat completion request
    let chat_request = json!({
        "model": body.model,
        "messages": prompt_messages,
        "temperature": body.temperature.unwrap_or(0.7),
        "top_p": body.top_p.unwrap_or(1.0),
        "max_tokens": body.max_output_tokens.unwrap_or(512),
        "stream": true,  // API only supports streaming
        "stream_options": { "include_usage": true }
    });

    // Call the chat completion API (responses.rs doesn't use auth_method)
    let response = get_chat_completion_response(&state, &user, chat_request, &headers).await?;

    // Process the SSE stream to extract content and usage
    let body_bytes = to_bytes(response.into_body()).await.map_err(|e| {
        error!("Failed to read response body: {:?}", e);
        ApiError::InternalServerError
    })?;
    
    let body_str = String::from_utf8_lossy(&body_bytes);
    let mut assistant_content = String::new();
    let mut completion_tokens = 0i32;
    let mut finish_reason = "stop".to_string();
    
    // Parse SSE events
    for line in body_str.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            
            if let Ok(event_json) = serde_json::from_str::<Value>(data) {
                // Extract content from delta
                if let Some(delta_content) = event_json["choices"][0]["delta"]["content"].as_str() {
                    assistant_content.push_str(delta_content);
                }
                
                // Extract usage from the final event
                if let Some(usage) = event_json.get("usage") {
                    if let Some(tokens) = usage["completion_tokens"].as_i64() {
                        completion_tokens = tokens as i32;
                    }
                }
                
                // Extract finish reason if present
                if let Some(reason) = event_json["choices"][0]["finish_reason"].as_str() {
                    finish_reason = reason.to_string();
                }
            }
        }
    }
    
    // Fallback to counting tokens if not provided
    if completion_tokens == 0 {
        completion_tokens = count_tokens(&assistant_content) as i32;
    }

    // Encrypt and store assistant message
    let assistant_enc = encrypt_with_key(&user_key, assistant_content.as_bytes()).await;

    let new_assistant = NewAssistantMessage {
        uuid: Uuid::new_v4(),
        thread_id: thread.id,
        user_message_id: inserted.id,
        content_enc: assistant_enc,
        completion_tokens: Some(completion_tokens),
        finish_reason: Some(finish_reason),
    };

    let _assistant = state
        .db
        .create_assistant_message(new_assistant)
        .map_err(|e| {
            error!("Failed to create assistant message: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Update user message status to completed
    state
        .db
        .update_user_message_status(
            inserted.id,
            ResponseStatus::Completed,
            None,
            Some(Utc::now()),
        )
        .map_err(|e| {
            error!("Failed to update user message status: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Build final response
    let api_resp = ResponsesCreateResponse {
        id: inserted.uuid,
        object: "response",
        created_at: inserted.created_at.timestamp(),
        model: inserted.model.clone(),
        status: "completed",
        output: vec![OutputItem {
            output_type: "message".to_string(),
            id: Uuid::new_v4().to_string(),
            status: "completed".to_string(),
            role: Some("assistant".to_string()),
            content: Some(vec![json!({
                "type": "text",
                "text": assistant_content
            })]),
        }],
        error: None,
        incomplete_details: None,
        usage: Some(ResponseUsage {
            input_tokens: total_prompt_tokens as i32,
            input_tokens_details: InputTokenDetails { cached_tokens: 0 },
            output_tokens: completion_tokens,
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: 0,
            },
            total_tokens: total_prompt_tokens as i32 + completion_tokens,
        }),
        metadata: inserted.metadata.clone(),
        parallel_tool_calls: inserted.parallel_tool_calls,
        previous_response_id: inserted.previous_response_id,
        store: inserted.store,
        temperature: inserted.temperature,
        top_p: inserted.top_p,
        tool_choice: inserted.tool_choice.clone(),
        tools: vec![], // No tools in Phase 3
    };

    encrypt_response(&state, &session_id, &api_resp).await
}

// For SHA256 hashing
use sha2::Digest;
