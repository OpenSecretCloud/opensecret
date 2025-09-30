//! OpenAI Conversations API implementation
//! Provides server-side conversation state management compatible with OpenAI's Conversations API

use crate::{
    encrypt::{decrypt_content, decrypt_string, encrypt_with_key},
    models::responses::NewConversation,
    models::users::User,
    web::{
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        responses::{
            constants::{
                DEFAULT_PAGINATION_LIMIT, DEFAULT_PAGINATION_ORDER, DEFAULT_TOOL_FUNCTION_NAME,
                MAX_PAGINATION_LIMIT, OBJECT_TYPE_CONVERSATION_DELETED, OBJECT_TYPE_LIST,
                ROLE_ASSISTANT, ROLE_USER,
            },
            error_mapping, ConversationBuilder, ConversationContent, MessageContent,
            MessageContentConverter,
        },
    },
    ApiError, AppState,
};
use axum::{
    extract::{Path, Query, State},
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, trace};
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
    DEFAULT_PAGINATION_LIMIT
}

fn default_order() -> String {
    DEFAULT_PAGINATION_ORDER.to_string()
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
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    // Create the conversation
    let conversation_uuid = Uuid::new_v4();

    // Create metadata with title
    let mut metadata = body.metadata.clone().unwrap_or_else(|| json!({}));
    if metadata.get("title").is_none() {
        metadata["title"] = json!("New Conversation");
    }

    // Encrypt the entire metadata object
    let metadata_json = serde_json::to_string(&metadata)
        .map_err(|_| error_mapping::map_serialization_error("metadata"))?;
    let metadata_enc = Some(encrypt_with_key(&user_key, metadata_json.as_bytes()).await);

    let new_conversation = NewConversation {
        uuid: conversation_uuid,
        user_id: user.uuid,
        system_prompt_id: None,
        metadata_enc,
    };

    trace!("Creating conversation with: {:?}", new_conversation);

    let conversation = state
        .db
        .create_conversation(new_conversation)
        .map_err(error_mapping::map_generic_db_error)?;

    trace!("Created conversation: {:?}", conversation);

    // Reject initial items - not supported in our simplified flow
    // Users must use POST /v1/responses to add messages to conversations
    if body.items.is_some() && !body.items.as_ref().unwrap().is_empty() {
        error!(
            "Initial items not supported in conversation creation for user: {}",
            user.uuid
        );
        return Err(ApiError::BadRequest);
    }

    // Decrypt metadata for response
    let metadata = decrypt_content(&user_key, conversation.metadata_enc.as_ref())
        .map_err(|_| error_mapping::map_decryption_error("conversation metadata"))?;

    let response = ConversationBuilder::from_conversation(&conversation)
        .metadata(metadata)
        .build();

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
        .map_err(error_mapping::map_conversation_error)?;

    // Get user's encryption key for decrypting metadata
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Decrypt metadata
    let metadata = decrypt_content(&user_key, conversation.metadata_enc.as_ref())
        .map_err(|_| error_mapping::map_decryption_error("conversation metadata"))?;

    let response = ConversationBuilder::from_conversation(&conversation)
        .metadata(metadata)
        .build();

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
        .map_err(error_mapping::map_conversation_error)?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    // Encrypt the updated metadata
    let metadata_json = serde_json::to_string(&body.metadata)
        .map_err(|_| error_mapping::map_serialization_error("metadata"))?;
    let metadata_enc = encrypt_with_key(&user_key, metadata_json.as_bytes()).await;

    // Update metadata in database
    state
        .db
        .update_conversation_metadata(conversation.id, user.uuid, metadata_enc.clone())
        .map_err(error_mapping::map_generic_db_error)?;

    // Update the in-memory object to reflect the changes
    conversation.metadata_enc = Some(metadata_enc);

    // For the response, return the decrypted metadata
    let response_metadata = Some(body.metadata.clone());

    let response = ConversationBuilder::from_conversation(&conversation)
        .metadata(response_metadata)
        .build();

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
        .map_err(error_mapping::map_conversation_error)?;

    // Delete the conversation (cascades will delete all associated responses)
    state
        .db
        .delete_conversation(conversation.id, user.uuid)
        .map_err(error_mapping::map_generic_db_error)?;

    let response = DeletedConversationResponse {
        id: conversation.uuid,
        object: OBJECT_TYPE_CONVERSATION_DELETED,
        deleted: true,
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
        .map_err(error_mapping::map_conversation_error)?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    // Get all messages from the conversation
    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation.id)
        .map_err(error_mapping::map_message_error)?;

    trace!(
        "Retrieved {} raw messages for conversation {}",
        raw_messages.len(),
        conversation_id
    );
    for (i, msg) in raw_messages.iter().enumerate() {
        trace!(
            "Message {}: uuid={}, type={}",
            i,
            msg.uuid,
            msg.message_type
        );
    }

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

    trace!(
        "Starting from index {} (after cursor: {:?})",
        start_index,
        params.after
    );

    for msg in raw_messages.iter().skip(start_index) {
        trace!(
            "Processing message: uuid={}, type={}",
            msg.uuid,
            msg.message_type
        );

        // Decrypt content (handle nullable content_enc)
        let content = decrypt_string(&user_key, msg.content_enc.as_ref())
            .map_err(|_| error_mapping::map_decryption_error("message content"))?
            .unwrap_or_default();

        trace!(
            "Decrypted content for {} (status={:?}): {}",
            msg.message_type,
            msg.status,
            if content.len() > 100 {
                format!("{}...", &content[..100])
            } else {
                content.clone()
            }
        );

        match msg.message_type.as_str() {
            "user" => {
                // User messages MUST be stored as MessageContent
                let message_content: MessageContent = serde_json::from_str(&content)
                    .map_err(|_| error_mapping::map_serialization_error("user message content"))?;

                items.push(ConversationItem::Message {
                    id: msg.uuid,
                    status: msg.status.clone(),
                    role: ROLE_USER.to_string(),
                    content: Vec::<ConversationContent>::from(message_content),
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            "assistant" => {
                // Assistant messages are plain text strings
                // If content is empty (in_progress), return empty content array
                let content_parts = if content.is_empty() {
                    vec![]
                } else {
                    MessageContentConverter::assistant_text_to_content(content)
                };

                items.push(ConversationItem::Message {
                    id: msg.uuid,
                    status: msg.status.clone(),
                    role: ROLE_ASSISTANT.to_string(),
                    content: content_parts,
                    created_at: Some(msg.created_at.timestamp()),
                });
            }
            "tool_call" => {
                // Parse tool call
                items.push(ConversationItem::FunctionToolCall {
                    id: msg.uuid,
                    name: DEFAULT_TOOL_FUNCTION_NAME.to_string(),
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

    trace!("Built {} conversation items before pagination", items.len());
    for (i, item) in items.iter().enumerate() {
        if let ConversationItem::Message { role, content, .. } = item {
            trace!("Item {}: role={}, content_len={}", i, role, content.len());
        }
    }

    // Apply pagination
    let limit = params.limit.min(MAX_PAGINATION_LIMIT) as usize;
    let has_more = items.len() > limit;
    if has_more {
        items.truncate(limit);
    }

    // Reverse if ascending order requested
    if params.order == "asc" {
        items.reverse();
    }

    trace!("Final items count: {}, has_more: {}", items.len(), has_more);

    let response = ConversationItemListResponse {
        object: OBJECT_TYPE_LIST,
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
        .map_err(error_mapping::map_conversation_error)?;

    // Get user's encryption key
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    // Get all messages from the conversation
    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation.id)
        .map_err(error_mapping::map_message_error)?;

    // Find the specific item
    for msg in raw_messages {
        if msg.uuid == item_id {
            // Decrypt content (handle nullable content_enc)
            let content = decrypt_string(&user_key, msg.content_enc.as_ref())
                .map_err(|e| {
                    error!("Failed to decrypt message content: {:?}", e);
                    ApiError::InternalServerError
                })?
                .unwrap_or_default();

            let item = match msg.message_type.as_str() {
                "user" => {
                    // User messages MUST be stored as MessageContent
                    let message_content: MessageContent = serde_json::from_str(&content)
                        .map_err(|e| {
                            error!("Failed to deserialize user message content as MessageContent: {:?}, content: {}", e, content);
                            ApiError::InternalServerError
                        })?;

                    ConversationItem::Message {
                        id: msg.uuid,
                        status: msg.status.clone(),
                        role: ROLE_USER.to_string(),
                        content: Vec::<ConversationContent>::from(message_content),
                        created_at: Some(msg.created_at.timestamp()),
                    }
                }
                "assistant" => {
                    // If content is empty (in_progress), return empty content array
                    let content_parts = if content.is_empty() {
                        vec![]
                    } else {
                        MessageContentConverter::assistant_text_to_content(content)
                    };

                    ConversationItem::Message {
                        id: msg.uuid,
                        status: msg.status.clone(),
                        role: ROLE_ASSISTANT.to_string(),
                        content: content_parts,
                        created_at: Some(msg.created_at.timestamp()),
                    }
                }
                "tool_call" => ConversationItem::FunctionToolCall {
                    id: msg.uuid,
                    name: DEFAULT_TOOL_FUNCTION_NAME.to_string(),
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

    let limit = params.limit.min(MAX_PAGINATION_LIMIT);
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
        .map_err(error_mapping::map_generic_db_error)?;

    let has_more = conversations.len() > limit as usize;
    if has_more {
        conversations.truncate(limit as usize);
    }

    // If ascending order requested, reverse
    if order == "asc" {
        conversations.reverse();
    }

    // Get user's encryption key for decrypting titles
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Convert to response format
    let data: Vec<ConversationResponse> = conversations
        .iter()
        .map(|conv| -> Result<ConversationResponse, ApiError> {
            trace!("Raw conversation object: {:?}", conv);
            trace!("Conv metadata_enc present: {}", conv.metadata_enc.is_some());

            // Decrypt metadata
            let metadata = decrypt_content(&user_key, conv.metadata_enc.as_ref())
                .map_err(|_| error_mapping::map_decryption_error("conversation metadata"))?;

            Ok(ConversationBuilder::from_conversation(conv)
                .metadata(metadata)
                .build())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let response = ConversationListResponse {
        object: OBJECT_TYPE_LIST,
        data: data.clone(),
        has_more,
        first_id: conversations.first().map(|c| c.uuid),
        last_id: conversations.last().map(|c| c.uuid),
    };

    trace!("Final ConversationListResponse: {:?}", response);

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
        /* TODO IGNORE FOR NOW - TOO CONFUSING AND NOT NEEDED
        .route(
            "/v1/conversations/:id/items",
            post(create_conversation_items).layer(from_fn_with_state(
                state.clone(),
                decrypt_request::<CreateConversationItemsRequest>,
            )),
        )
        */
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
