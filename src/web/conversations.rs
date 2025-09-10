//! OpenAI Conversations API implementation
//! Provides server-side conversation state management compatible with OpenAI's Conversations API

use crate::{
    db::DBError,
    encrypt::{decrypt_with_key, encrypt_with_key},
    models::responses::{NewConversation, NewUserMessage, ResponsesError},
    models::users::User,
    web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
    ApiError, AppState,
};
use axum::{
    extract::{Path, Query, State},
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, error};
use uuid::Uuid;

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to create a new conversation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateConversationRequest {
    /// Metadata to attach to the conversation (max 16 key-value pairs)
    #[serde(default)]
    pub metadata: Option<Value>,

    /// Initial items to include in the conversation
    #[serde(default)]
    pub items: Option<Vec<ConversationInputItem>>,
}

/// Content part for input messages
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum MessageContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    // TODO: Add support for other content types:
    // - image_url: { url: String, detail?: "low" | "high" | "auto" }
    // - audio: { data: String, format: String }
}

/// Content that can be either a string or array of content parts
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

impl MessageContent {
    /// Extract text content, concatenating multiple parts if necessary
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    MessageContentPart::Text { text } => Some(text.clone()),
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// Input item when creating conversation items
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ConversationInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: MessageContent,
    },
}

/// Request to update a conversation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateConversationRequest {
    /// Updated metadata (required per OpenAI spec)
    pub metadata: Value,
}

/// Request to create items in a conversation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateConversationItemsRequest {
    /// Items to add to the conversation (max 20 at a time)
    pub items: Vec<ConversationInputItem>,
}

/// Response for a conversation object
#[derive(Debug, Clone, Serialize)]
pub struct ConversationResponse {
    /// Conversation ID
    pub id: Uuid,

    /// Object type (always "conversation")
    pub object: &'static str,

    /// Metadata attached to the conversation
    pub metadata: Option<Value>,

    /// Unix timestamp when created
    pub created_at: i64,
}

/// Response for deleted conversation
#[derive(Debug, Clone, Serialize)]
pub struct DeletedConversationResponse {
    pub id: Uuid,
    pub object: &'static str,
    pub deleted: bool,
}

/// A single conversation item (message, tool call, etc.)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ConversationItem {
    #[serde(rename = "message")]
    Message {
        id: Uuid,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        role: String,
        content: Vec<ConversationContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: Uuid,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
    #[serde(rename = "function_tool_call_output")]
    FunctionToolCallOutput {
        id: Uuid,
        tool_call_id: Uuid,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
}

/// Content within a message
/// TODO: Extend to support images and other content types:
/// - image_url: { url: String, detail?: "low" | "high" | "auto" }
/// - audio: { data: String, format: String }
///   This will require updates to the Responses API and database storage
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum ConversationContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "input_text")]
    InputText { text: String },
    // TODO: Add ImageUrl { image_url: ImageUrlContent }
    // TODO: Add Audio { audio: AudioContent }
}

/// Response for listing conversation items
#[derive(Debug, Clone, Serialize)]
pub struct ConversationItemListResponse {
    pub object: &'static str,
    pub data: Vec<ConversationItem>,
    pub has_more: bool,
    pub first_id: Option<Uuid>,
    pub last_id: Option<Uuid>,
}

/// Response for listing conversations (custom OpenSecret extension)
#[derive(Debug, Clone, Serialize)]
pub struct ConversationListResponse {
    pub object: &'static str,
    pub data: Vec<ConversationResponse>,
    pub has_more: bool,
    pub first_id: Option<Uuid>,
    pub last_id: Option<Uuid>,
}

/// Query parameters for listing conversations (custom extension)
#[derive(Debug, Deserialize)]
pub struct ListConversationsParams {
    #[serde(default = "default_limit")]
    pub limit: i64,
    pub after: Option<Uuid>,
    pub before: Option<Uuid>,
    #[serde(default = "default_order")]
    pub order: String,
}

/// Query parameters for listing conversation items
#[derive(Debug, Deserialize)]
pub struct ListItemsParams {
    #[serde(default = "default_limit")]
    pub limit: i64,
    pub after: Option<Uuid>,
    #[serde(default = "default_order")]
    pub order: String,
    /// Additional fields to include in the response
    /// TODO: Not yet implemented - will be used to include additional data like logprobs, sources, etc.
    #[allow(dead_code)]
    pub include: Option<Vec<String>>,
}

fn default_limit() -> i64 {
    20
}

fn default_order() -> String {
    "desc".to_string()
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /v1/conversations - Create a new conversation
async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<CreateConversationRequest>,
) -> Result<Json<EncryptedResponse<ConversationResponse>>, ApiError> {
    debug!("Creating new conversation for user: {}", user.uuid);

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Create the conversation
    let conversation_uuid = Uuid::new_v4();

    // Encrypt title if we want to add a default one
    let title_enc = encrypt_with_key(&user_key, b"New Conversation").await;

    let new_conversation = NewConversation {
        uuid: conversation_uuid,
        user_id: user.uuid,
        system_prompt_id: None,
        title_enc: Some(title_enc),
        metadata: body.metadata.clone(),
    };

    let conversation = state
        .db
        .create_conversation(new_conversation)
        .map_err(|e| {
            error!("Failed to create conversation: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Process initial items if provided
    if let Some(items) = &body.items {
        for item in items {
            match item {
                ConversationInputItem::Message { role, content } => {
                    if role == "user" {
                        // Create user message
                        let content_text = content.to_text();
                        let content_enc =
                            encrypt_with_key(&user_key, content_text.as_bytes()).await;
                        
                        // Count tokens for the user message
                        // TODO: Use proper tokenizer for actual token count
                        let prompt_tokens = content_text.len() as i32 / 4; // rough estimate
                        
                        let new_msg = NewUserMessage {
                            uuid: Uuid::new_v4(),
                            conversation_id: conversation.id,
                            response_id: None, // No response when creating via Conversations API
                            user_id: user.uuid,
                            content_enc,
                            prompt_tokens,
                        };

                        state.db.create_user_message(new_msg).map_err(|e| {
                            error!("Failed to create initial user message: {:?}", e);
                            ApiError::InternalServerError
                        })?;
                    } else if role == "assistant" {
                        // Note: We don't support creating assistant messages in initial items
                        // Just log and skip for now
                        // TODO fix this logic
                        debug!("Skipping assistant message in initial items (not yet supported)");
                    }
                }
            }
        }
    }

    let response = ConversationResponse {
        id: conversation.uuid,
        object: "conversation",
        metadata: conversation.metadata,
        created_at: conversation.created_at.timestamp(),
    };

    encrypt_response(&state, &session_id, &response).await
}

/// GET /v1/conversations/{id} - Retrieve a conversation
async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationResponse>>, ApiError> {
    debug!(
        "Getting conversation {} for user {}",
        conversation_id, user.uuid
    );

    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    let response = ConversationResponse {
        id: conversation.uuid,
        object: "conversation",
        metadata: conversation.metadata,
        created_at: conversation.created_at.timestamp(),
    };

    encrypt_response(&state, &session_id, &response).await
}

/// POST /v1/conversations/{id} - Update a conversation
async fn update_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<UpdateConversationRequest>,
) -> Result<Json<EncryptedResponse<ConversationResponse>>, ApiError> {
    debug!(
        "Updating conversation {} for user {}",
        conversation_id, user.uuid
    );

    // Get the conversation to ensure it exists and user owns it
    let mut conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Update metadata (always provided per spec)
    state
        .db
        .update_conversation_metadata(conversation.id, user.uuid, body.metadata.clone())
        .map_err(|e| {
            error!("Failed to update conversation metadata: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Update the in-memory object to reflect the change
    conversation.metadata = Some(body.metadata.clone());

    let response = ConversationResponse {
        id: conversation.uuid,
        object: "conversation",
        metadata: conversation.metadata,
        created_at: conversation.created_at.timestamp(),
    };

    encrypt_response(&state, &session_id, &response).await
}

/// DELETE /v1/conversations/{id} - Delete a conversation
async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<DeletedConversationResponse>>, ApiError> {
    debug!(
        "Deleting conversation {} for user {}",
        conversation_id, user.uuid
    );

    // Get the conversation to ensure it exists and user owns it
    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Delete the conversation (cascades will delete all associated responses)
    state
        .db
        .delete_conversation(conversation.id, user.uuid)
        .map_err(|e| {
            error!("Failed to delete conversation: {:?}", e);
            ApiError::InternalServerError
        })?;

    let response = DeletedConversationResponse {
        id: conversation.uuid,
        object: "conversation.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}

/// POST /v1/conversations/{id}/items - Add items to a conversation
async fn create_conversation_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<CreateConversationItemsRequest>,
) -> Result<Json<EncryptedResponse<ConversationItemListResponse>>, ApiError> {
    debug!(
        "Adding items to conversation {} for user {}",
        conversation_id, user.uuid
    );

    // Verify conversation exists and user owns it
    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    let mut created_items = Vec::new();

    // TODO this is wrong. we should be appending the items to the conversation and then persisting

    // Process each item
    for item in &body.items {
        match item {
            ConversationInputItem::Message { role, content } => {
                if role == "user" {
                    // Create user message
                    // Extract text from content (handles both string and array formats)
                    let content_text = content.to_text();
                    let content_enc = encrypt_with_key(&user_key, content_text.as_bytes()).await;
                    let msg_uuid = Uuid::new_v4();

                    // Count tokens for the user message
                    // TODO: Use proper tokenizer for actual token count
                    let prompt_tokens = content_text.len() as i32 / 4; // rough estimate
                    
                    let new_msg = NewUserMessage {
                        uuid: msg_uuid,
                        conversation_id: conversation.id,
                        response_id: None, // No response when creating via Conversations API
                        user_id: user.uuid,
                        content_enc,
                        prompt_tokens,
                    };

                    let created = state.db.create_user_message(new_msg).map_err(|e| {
                        error!("Failed to create user message: {:?}", e);
                        ApiError::InternalServerError
                    })?;

                    created_items.push(ConversationItem::Message {
                        id: created.uuid,
                        status: Some("completed".to_string()),
                        role: "user".to_string(),
                        content: vec![ConversationContent::Text {
                            text: content_text.clone(),
                        }],
                        created_at: Some(created.created_at.timestamp()),
                    });
                } else if role == "assistant" {
                    // Note: We don't support creating assistant messages directly
                    debug!("Skipping assistant message (not yet supported)");
                }
            }
        }
    }

    let response = ConversationItemListResponse {
        object: "list",
        data: created_items.clone(),
        has_more: false,
        first_id: created_items.first().map(|item| match item {
            ConversationItem::Message { id, .. } => *id,
            ConversationItem::FunctionToolCall { id, .. } => *id,
            ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        }),
        last_id: created_items.last().map(|item| match item {
            ConversationItem::Message { id, .. } => *id,
            ConversationItem::FunctionToolCall { id, .. } => *id,
            ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        }),
    };

    encrypt_response(&state, &session_id, &response).await
}

/// GET /v1/conversations/{id}/items - List conversation items
async fn list_conversation_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<Uuid>,
    Query(params): Query<ListItemsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItemListResponse>>, ApiError> {
    debug!(
        "Listing items for conversation {} for user {}",
        conversation_id, user.uuid
    );

    // Verify conversation exists and user owns it
    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Get all messages from the conversation
    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation.id)
        .map_err(|e| {
            error!("Failed to get conversation messages: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Convert to conversation items
    let mut items = Vec::new();

    // If we have an after cursor, find where to start
    let start_index = if let Some(after_uuid) = params.after {
        raw_messages
            .iter()
            .position(|msg| msg.uuid == after_uuid)
            .map(|idx| idx + 1) // Start from the message after the cursor
            .unwrap_or(0)
    } else {
        0
    };

    for msg in raw_messages.iter().skip(start_index) {
        // Decrypt content
        let content = decrypt_with_key(&user_key, &msg.content_enc)
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|_| "[Decryption failed]".to_string());

        match msg.message_type.as_str() {
            "user" => {
                items.push(ConversationItem::Message {
                    id: msg.uuid,
                    status: Some("completed".to_string()),
                    role: "user".to_string(),
                    content: vec![ConversationContent::Text { text: content }],
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            "assistant" => {
                items.push(ConversationItem::Message {
                    id: msg.uuid,
                    status: Some("completed".to_string()),
                    role: "assistant".to_string(),
                    content: vec![ConversationContent::Text { text: content }],
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            "tool_call" => {
                // Parse tool call
                items.push(ConversationItem::FunctionToolCall {
                    id: msg.uuid,
                    name: "function".to_string(), // We'd need to store this properly
                    arguments: content,
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            "tool_output" => {
                items.push(ConversationItem::FunctionToolCallOutput {
                    id: msg.uuid,
                    tool_call_id: msg.tool_call_id.unwrap_or(Uuid::nil()),
                    output: content,
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            _ => {}
        }
    }

    // Apply pagination
    let limit = params.limit.min(100) as usize;
    let has_more = items.len() > limit;
    if has_more {
        items.truncate(limit);
    }

    // Reverse if ascending order requested
    if params.order == "asc" {
        items.reverse();
    }

    let response = ConversationItemListResponse {
        object: "list",
        data: items.clone(),
        has_more,
        first_id: items.first().map(|item| match item {
            ConversationItem::Message { id, .. } => *id,
            ConversationItem::FunctionToolCall { id, .. } => *id,
            ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        }),
        last_id: items.last().map(|item| match item {
            ConversationItem::Message { id, .. } => *id,
            ConversationItem::FunctionToolCall { id, .. } => *id,
            ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        }),
    };

    encrypt_response(&state, &session_id, &response).await
}

/// GET /v1/conversations/{conversation_id}/items/{item_id} - Retrieve a single item
async fn get_conversation_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(Uuid, Uuid)>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    debug!(
        "Getting item {} from conversation {} for user {}",
        item_id, conversation_id, user.uuid
    );

    // Verify conversation exists and user owns it
    let conversation = state
        .db
        .get_conversation_by_uuid_and_user(conversation_id, user.uuid)
        .map_err(|e| match e {
            DBError::ResponsesError(ResponsesError::ConversationNotFound) => ApiError::NotFound,
            _ => {
                error!("Failed to get conversation: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Get all messages from the conversation
    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation.id)
        .map_err(|e| {
            error!("Failed to get conversation messages: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Find the specific item
    for msg in raw_messages {
        if msg.uuid == item_id {
            // Decrypt content
            let content = decrypt_with_key(&user_key, &msg.content_enc)
                .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                .unwrap_or_else(|_| "[Decryption failed]".to_string());

            let item = match msg.message_type.as_str() {
                "user" => ConversationItem::Message {
                    id: msg.uuid,
                    status: Some("completed".to_string()),
                    role: "user".to_string(),
                    content: vec![ConversationContent::Text { text: content }],
                    created_at: Some(msg.created_at.timestamp()),
                },
                "assistant" => ConversationItem::Message {
                    id: msg.uuid,
                    status: Some("completed".to_string()),
                    role: "assistant".to_string(),
                    content: vec![ConversationContent::Text { text: content }],
                    created_at: Some(msg.created_at.timestamp()),
                },
                "tool_call" => ConversationItem::FunctionToolCall {
                    id: msg.uuid,
                    name: "function".to_string(),
                    arguments: content,
                    created_at: Some(msg.created_at.timestamp()),
                },
                "tool_output" => ConversationItem::FunctionToolCallOutput {
                    id: msg.uuid,
                    tool_call_id: msg.tool_call_id.unwrap_or(Uuid::nil()),
                    output: content,
                    created_at: Some(msg.created_at.timestamp()),
                },
                _ => {
                    return Err(ApiError::InternalServerError);
                }
            };

            return encrypt_response(&state, &session_id, &item).await;
        }
    }

    // Item not found
    Err(ApiError::NotFound)
}

/// GET /v1/conversations - List all conversations (OpenSecret custom extension)
async fn list_conversations(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListConversationsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationListResponse>>, ApiError> {
    debug!("Listing conversations for user: {}", user.uuid);

    let limit = params.limit.min(100);
    let order = &params.order;

    // Convert UUID cursors to (updated_at, id) tuples
    let after_cursor = if let Some(after_uuid) = params.after {
        state
            .db
            .get_conversation_by_uuid_and_user(after_uuid, user.uuid)
            .ok()
            .map(|conv| (conv.updated_at, conv.id))
    } else {
        None
    };

    let before_cursor = if let Some(before_uuid) = params.before {
        state
            .db
            .get_conversation_by_uuid_and_user(before_uuid, user.uuid)
            .ok()
            .map(|conv| (conv.updated_at, conv.id))
    } else {
        None
    };

    // Get conversations
    let mut conversations = state
        .db
        .list_conversations(user.uuid, limit + 1, after_cursor, before_cursor)
        .map_err(|e| {
            error!("Failed to list conversations: {:?}", e);
            ApiError::InternalServerError
        })?;

    let has_more = conversations.len() > limit as usize;
    if has_more {
        conversations.truncate(limit as usize);
    }

    // If ascending order requested, reverse
    if order == "asc" {
        conversations.reverse();
    }

    // Convert to response format
    let data: Vec<ConversationResponse> = conversations
        .iter()
        .map(|conv| ConversationResponse {
            id: conv.uuid,
            object: "conversation",
            metadata: conv.metadata.clone(),
            created_at: conv.created_at.timestamp(),
        })
        .collect();

    let response = ConversationListResponse {
        object: "list",
        data: data.clone(),
        has_more,
        first_id: conversations.first().map(|c| c.uuid),
        last_id: conversations.last().map(|c| c.uuid),
    };

    encrypt_response(&state, &session_id, &response).await
}

// ============================================================================
// Router Configuration
// ============================================================================

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/conversations",
            post(create_conversation).layer(from_fn_with_state(
                state.clone(),
                decrypt_request::<CreateConversationRequest>,
            )),
        )
        .route(
            "/v1/conversations",
            get(list_conversations).layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/conversations/:id",
            get(get_conversation).layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/conversations/:id",
            post(update_conversation).layer(from_fn_with_state(
                state.clone(),
                decrypt_request::<UpdateConversationRequest>,
            )),
        )
        .route(
            "/v1/conversations/:id",
            delete(delete_conversation)
                .layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/conversations/:id/items",
            post(create_conversation_items).layer(from_fn_with_state(
                state.clone(),
                decrypt_request::<CreateConversationItemsRequest>,
            )),
        )
        .route(
            "/v1/conversations/:id/items",
            get(list_conversation_items)
                .layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/conversations/:id/items/:item_id",
            get(get_conversation_item)
                .layer(from_fn_with_state(state.clone(), decrypt_request::<()>)),
        )
        .with_state(state)
}
