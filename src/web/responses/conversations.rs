//! OpenAI Conversations API implementation
//! Provides server-side conversation state management compatible with OpenAI's Conversations API

use crate::{
    encrypt::{decrypt_content, encrypt_with_key},
    models::responses::NewConversation,
    models::users::User,
    web::{
        encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
        responses::{
            constants::{
                DEFAULT_PAGINATION_LIMIT, DEFAULT_PAGINATION_ORDER, MAX_PAGINATION_LIMIT,
                OBJECT_TYPE_LIST,
            },
            error_mapping, ConversationBuilder, ConversationItem, ConversationItemConverter,
            DeletedObjectResponse, MessageContent, Paginator,
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
// Context Helper
// ============================================================================

/// Conversation context with decryption key
///
/// This helper encapsulates the common pattern of loading a conversation
/// and the user's encryption key together. Used by most conversation handlers.
struct ConversationContext {
    conversation: crate::models::responses::Conversation,
    user_key: secp256k1::SecretKey,
}

impl ConversationContext {
    /// Load conversation and user's encryption key in one operation
    ///
    /// Verifies conversation exists, user owns it, and retrieves encryption key.
    ///
    /// # Arguments
    /// * `state` - Application state
    /// * `conversation_id` - UUID of the conversation to load
    /// * `user_uuid` - UUID of the user requesting access
    ///
    /// # Returns
    /// ConversationContext containing the conversation and encryption key
    ///
    /// # Errors
    /// Returns ApiError if conversation not found, user doesn't own it, or key retrieval fails
    async fn load(
        state: &AppState,
        conversation_id: Uuid,
        user_uuid: Uuid,
    ) -> Result<Self, ApiError> {
        // Get conversation (verifies ownership)
        let conversation = state
            .db
            .get_conversation_by_uuid_and_user(conversation_id, user_uuid)
            .map_err(error_mapping::map_conversation_error)?;

        // Get user's encryption key
        let user_key = state
            .get_user_key(user_uuid, None, None)
            .await
            .map_err(|_| error_mapping::map_key_retrieval_error())?;

        Ok(Self {
            conversation,
            user_key,
        })
    }

    /// Decrypt conversation metadata
    ///
    /// # Returns
    /// Option<Value> containing decrypted metadata, or None if no metadata exists
    ///
    /// # Errors
    /// Returns ApiError if decryption fails
    fn decrypt_metadata(&self) -> Result<Option<Value>, ApiError> {
        decrypt_content(&self.user_key, self.conversation.metadata_enc.as_ref())
            .map_err(|_| error_mapping::map_decryption_error("conversation metadata"))
    }
}

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

    // Reject initial items - not supported in our simplified flow
    // Users must use POST /v1/responses to add messages to conversations
    if body.items.is_some() && !body.items.as_ref().unwrap().is_empty() {
        error!(
            "Initial items not supported in conversation creation for user: {}",
            user.uuid
        );
        return Err(ApiError::BadRequest);
    }

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
    let metadata_json = serde_json::to_string(&metadata).map_err(|e| {
        error!("Failed to serialize metadata: {:?}", e);
        ApiError::InternalServerError
    })?;
    let metadata_enc = Some(encrypt_with_key(&user_key, metadata_json.as_bytes()).await);

    let new_conversation = NewConversation {
        uuid: conversation_uuid,
        user_id: user.uuid,
        metadata_enc,
    };

    trace!("Creating conversation with: {:?}", new_conversation);

    let conversation = state
        .db
        .create_conversation(new_conversation)
        .map_err(error_mapping::map_generic_db_error)?;

    trace!("Created conversation: {:?}", conversation);

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

    let ctx = ConversationContext::load(&state, conversation_id, user.uuid).await?;
    let metadata = ctx.decrypt_metadata()?;

    let response = ConversationBuilder::from_conversation(&ctx.conversation)
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

    let ctx = ConversationContext::load(&state, conversation_id, user.uuid).await?;

    // Encrypt the updated metadata
    let metadata_json = serde_json::to_string(&body.metadata).map_err(|e| {
        error!("Failed to serialize metadata: {:?}", e);
        ApiError::InternalServerError
    })?;
    let metadata_enc = encrypt_with_key(&ctx.user_key, metadata_json.as_bytes()).await;

    // Update metadata in database
    state
        .db
        .update_conversation_metadata(ctx.conversation.id, user.uuid, metadata_enc.clone())
        .map_err(error_mapping::map_generic_db_error)?;

    // For the response, return the decrypted metadata (already have it from body)
    let response_metadata = Some(body.metadata.clone());

    let response = ConversationBuilder::from_conversation(&ctx.conversation)
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
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    debug!(
        "Deleting conversation {} for user {}",
        conversation_id, user.uuid
    );

    let ctx = ConversationContext::load(&state, conversation_id, user.uuid).await?;

    // Delete the conversation (cascades will delete all associated responses)
    state
        .db
        .delete_conversation(ctx.conversation.id, user.uuid)
        .map_err(error_mapping::map_generic_db_error)?;

    let response = DeletedObjectResponse::conversation(ctx.conversation.uuid);

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

    let ctx = ConversationContext::load(&state, conversation_id, user.uuid).await?;

    // Validate limit
    let limit = params.limit.min(MAX_PAGINATION_LIMIT);

    // Fetch messages with database-level pagination
    // We fetch limit + 1 to check if there are more results
    let raw_messages = state
        .db
        .get_conversation_context_messages(
            ctx.conversation.id,
            limit + 1,
            params.after,
            &params.order,
        )
        .map_err(error_mapping::map_message_error)?;

    trace!(
        "Retrieved {} raw messages for conversation {} (limit={}, after={:?}, order={})",
        raw_messages.len(),
        conversation_id,
        limit,
        params.after,
        params.order
    );
    for (i, msg) in raw_messages.iter().enumerate() {
        trace!(
            "Message {}: uuid={}, type={}, created_at={}",
            i,
            msg.uuid,
            msg.message_type,
            msg.created_at
        );
    }

    // Check if there are more results
    let has_more = raw_messages.len() > limit as usize;

    // Take only the requested limit
    let messages_to_return = if has_more {
        &raw_messages[..limit as usize]
    } else {
        &raw_messages[..]
    };

    // Convert messages to items
    let items = ConversationItemConverter::messages_to_items(
        messages_to_return,
        &ctx.user_key,
        0, // Start from beginning since pagination is already applied at DB level
        messages_to_return.len(),
    )?;

    trace!("Final items count: {}, has_more: {}", items.len(), has_more);

    // Extract cursor IDs
    let (first_id, last_id) = Paginator::get_cursor_ids(&items, |item| match item {
        ConversationItem::Message { id, .. } => *id,
        ConversationItem::FunctionToolCall { id, .. } => *id,
        ConversationItem::FunctionToolCallOutput { id, .. } => *id,
    });

    let response = ConversationItemListResponse {
        object: OBJECT_TYPE_LIST,
        data: items.clone(),
        has_more,
        first_id,
        last_id,
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

    let ctx = ConversationContext::load(&state, conversation_id, user.uuid).await?;

    // Get all messages from the conversation
    // For this single item lookup, we fetch all messages (no pagination)
    // TODO: Optimize this to fetch only the specific item
    let raw_messages = state
        .db
        .get_conversation_context_messages(
            ctx.conversation.id,
            i64::MAX, // No limit for single item lookup
            None,     // No cursor
            "asc",    // Default order
        )
        .map_err(error_mapping::map_message_error)?;

    // Find the specific item and convert using centralized converter
    for msg in raw_messages {
        if msg.uuid == item_id {
            let item = ConversationItemConverter::message_to_item(&msg, &ctx.user_key)?;
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

    // Validate limit
    let limit = params.limit.min(MAX_PAGINATION_LIMIT);

    // Fetch conversations with database-level pagination
    // We fetch limit + 1 to check if there are more results
    let conversations = state
        .db
        .list_conversations(user.uuid, limit + 1, params.after, &params.order)
        .map_err(error_mapping::map_generic_db_error)?;

    trace!(
        "Retrieved {} conversations for user {} (limit={}, after={:?}, order={})",
        conversations.len(),
        user.uuid,
        limit,
        params.after,
        params.order
    );

    // Check if there are more results
    let has_more = conversations.len() > limit as usize;

    // Take only the requested limit
    let conversations_to_return = if has_more {
        &conversations[..limit as usize]
    } else {
        &conversations[..]
    };

    // Get user's encryption key for decrypting metadata
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| ApiError::InternalServerError)?;

    // Convert to response format
    let data: Vec<ConversationResponse> = conversations_to_return
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

    // Extract cursor IDs
    let (first_id, last_id) = Paginator::get_cursor_ids(conversations_to_return, |c| c.uuid);

    let response = ConversationListResponse {
        object: OBJECT_TYPE_LIST,
        data: data.clone(),
        has_more,
        first_id,
        last_id,
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
