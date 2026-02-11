use axum::{
    extract::{Path, Query, State},
    middleware::from_fn_with_state,
    routing::{delete, get, post, put},
    Extension, Json, Router,
};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

use crate::encrypt::{decrypt_content, decrypt_string, encrypt_with_key};
use crate::models::agent_config::{AgentConfig, NewAgentConfig};
use crate::models::memory_blocks::{MemoryBlock, NewMemoryBlock, DEFAULT_BLOCK_CHAR_LIMIT};
use crate::models::responses::{Conversation, NewConversation};
use crate::models::schema::user_embeddings;
use crate::models::users::User;
use crate::rag;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::openai_auth::AuthMethod;
use crate::web::responses::constants::{DEFAULT_PAGINATION_LIMIT, MAX_PAGINATION_LIMIT};
use crate::web::responses::conversations::{
    ConversationItemListResponse, ConversationListResponse, ConversationResponse, ListItemsParams,
};
use crate::web::responses::{
    error_mapping, ConversationBuilder, ConversationItem, ConversationItemConverter,
    DeletedObjectResponse, Paginator,
};
use crate::{ApiError, AppMode, AppState};

mod compaction;
mod runtime;
mod signatures;
mod tools;

#[derive(Debug, Clone, Deserialize)]
struct AgentChatRequest {
    input: String,
}

#[derive(Debug, Clone, Serialize)]
struct AgentChatResponse {
    messages: Vec<String>,
    tool_calls: Vec<signatures::AgentToolCall>,
}

#[derive(Debug, Clone, Serialize)]
struct AgentConfigResponse {
    enabled: bool,
    model: String,
    max_context_tokens: i32,
    compaction_threshold: f32,
    system_prompt: Option<String>,
    conversation_id: Option<Uuid>,
}

#[derive(Debug, Clone, Deserialize)]
struct UpdateAgentConfigRequest {
    enabled: Option<bool>,
    model: Option<String>,
    max_context_tokens: Option<i32>,
    compaction_threshold: Option<f32>,
    system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryBlockResponse {
    label: String,
    description: Option<String>,
    value: String,
    char_limit: i32,
    read_only: bool,
    version: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct UpdateMemoryBlockRequest {
    description: Option<String>,
    value: Option<String>,
    char_limit: Option<i32>,
    read_only: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct InsertArchivalRequest {
    text: String,
    metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct InsertArchivalResponse {
    id: Uuid,
    source_type: String,
    embedding_model: String,
    token_count: i32,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Deserialize)]
struct MemorySearchRequest {
    query: String,
    top_k: Option<usize>,
    max_tokens: Option<i32>,
    source_types: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
struct MemorySearchResponse {
    results: Vec<rag::RagSearchResult>,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    // Experimental endpoints: only enabled in Local/Dev.
    if !matches!(app_state.app_mode, AppMode::Local | AppMode::Dev) {
        return Router::new().with_state(app_state);
    }

    Router::new()
        .route(
            "/v1/agent/chat",
            post(chat).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AgentChatRequest>,
            )),
        )
        .route(
            "/v1/agent/config",
            get(get_config).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/config",
            put(update_config).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<UpdateAgentConfigRequest>,
            )),
        )
        .route(
            "/v1/agent/memory/blocks",
            get(list_memory_blocks)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/memory/blocks/:label",
            get(get_memory_block)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/memory/blocks/:label",
            put(update_memory_block).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<UpdateMemoryBlockRequest>,
            )),
        )
        .route(
            "/v1/agent/memory/archival",
            post(insert_archival).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<InsertArchivalRequest>,
            )),
        )
        .route(
            "/v1/agent/memory/archival/:id",
            delete(delete_archival)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/memory/search",
            post(memory_search).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<MemorySearchRequest>,
            )),
        )
        .route(
            "/v1/agent/conversations",
            get(list_agent_conversations)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/conversations/:id/items",
            get(list_agent_conversation_items)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/conversations/:id",
            delete(delete_agent_conversation)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<AgentChatRequest>,
) -> Result<Json<EncryptedResponse<AgentChatResponse>>, ApiError> {
    if body.input.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut runtime = runtime::AgentRuntime::new(state.clone(), user.clone(), user_key).await?;
    let (messages, tool_calls) = runtime.process_message(&body.input).await?;

    let response = AgentChatResponse {
        messages,
        tool_calls,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn get_config(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<AgentConfigResponse>>, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let cfg = get_or_create_agent_config(&mut conn, user.uuid).await?;

    let system_prompt = decrypt_string(&user_key, cfg.system_prompt_enc.as_ref()).map_err(|e| {
        error!("Failed to decrypt agent system prompt: {e:?}");
        ApiError::InternalServerError
    })?;

    let conversation_uuid = if let Some(conversation_id) = cfg.conversation_id {
        match Conversation::get_by_id_and_user(&mut conn, conversation_id, user.uuid) {
            Ok(c) => Some(c.uuid),
            Err(_) => None,
        }
    } else {
        None
    };

    let response = AgentConfigResponse {
        enabled: cfg.enabled,
        model: cfg.model,
        max_context_tokens: cfg.max_context_tokens,
        compaction_threshold: cfg.compaction_threshold,
        system_prompt,
        conversation_id: conversation_uuid,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn update_config(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<UpdateAgentConfigRequest>,
) -> Result<Json<EncryptedResponse<AgentConfigResponse>>, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let mut cfg = get_or_create_agent_config(&mut conn, user.uuid).await?;

    if let Some(enabled) = body.enabled {
        cfg.enabled = enabled;
    }
    if let Some(model) = body.model {
        if model.trim().is_empty() {
            return Err(ApiError::BadRequest);
        }
        cfg.model = model;
    }
    if let Some(max_context_tokens) = body.max_context_tokens {
        if max_context_tokens <= 0 {
            return Err(ApiError::BadRequest);
        }
        cfg.max_context_tokens = max_context_tokens;
    }
    if let Some(threshold) = body.compaction_threshold {
        cfg.compaction_threshold = threshold.clamp(0.5, 0.95);
    }

    let system_prompt_enc = match body.system_prompt {
        Some(p) if !p.trim().is_empty() => Some(encrypt_with_key(&user_key, p.as_bytes()).await),
        Some(_) => None,
        None => cfg.system_prompt_enc.clone(),
    };

    // If enabling and conversation is missing, initialize agent thread + blocks.
    let conversation_id = if cfg.enabled && cfg.conversation_id.is_none() {
        let metadata_enc = Some(
            encrypt_with_key(
                &user_key,
                json!({"type":"agent_main"}).to_string().as_bytes(),
            )
            .await,
        );

        let conversation = NewConversation {
            uuid: Uuid::new_v4(),
            user_id: user.uuid,
            metadata_enc,
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to create agent conversation: {e:?}");
            ApiError::InternalServerError
        })?;

        runtime::AgentRuntime::ensure_default_blocks(&state, &mut conn, &user_key, user.uuid)
            .await?;

        Some(conversation.id)
    } else {
        cfg.conversation_id
    };

    let updated = NewAgentConfig {
        uuid: cfg.uuid,
        user_id: cfg.user_id,
        conversation_id,
        enabled: cfg.enabled,
        model: cfg.model.clone(),
        max_context_tokens: cfg.max_context_tokens,
        compaction_threshold: cfg.compaction_threshold,
        system_prompt_enc,
        preferences_enc: cfg.preferences_enc.clone(),
    }
    .insert_or_update(&mut conn)
    .map_err(|e| {
        error!("Failed to update agent config: {e:?}");
        ApiError::InternalServerError
    })?;

    let system_prompt =
        decrypt_string(&user_key, updated.system_prompt_enc.as_ref()).map_err(|e| {
            error!("Failed to decrypt agent system prompt: {e:?}");
            ApiError::InternalServerError
        })?;

    let conversation_uuid = if let Some(conversation_id) = updated.conversation_id {
        match Conversation::get_by_id_and_user(&mut conn, conversation_id, user.uuid) {
            Ok(c) => Some(c.uuid),
            Err(_) => None,
        }
    } else {
        None
    };

    let response = AgentConfigResponse {
        enabled: updated.enabled,
        model: updated.model,
        max_context_tokens: updated.max_context_tokens,
        compaction_threshold: updated.compaction_threshold,
        system_prompt,
        conversation_id: conversation_uuid,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn list_memory_blocks(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<Vec<MemoryBlockResponse>>>, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let blocks = MemoryBlock::get_all_for_user(&mut conn, user.uuid).map_err(|e| {
        error!("Failed to list memory blocks: {e:?}");
        ApiError::InternalServerError
    })?;

    let mut out: Vec<MemoryBlockResponse> = Vec::with_capacity(blocks.len());
    for b in blocks {
        let value = decrypt_string(&user_key, Some(&b.value_enc))
            .map_err(|e| {
                error!("Failed to decrypt memory block: {e:?}");
                ApiError::InternalServerError
            })?
            .unwrap_or_default();

        out.push(MemoryBlockResponse {
            label: b.label,
            description: b.description,
            value,
            char_limit: b.char_limit,
            read_only: b.read_only,
            version: b.version,
        });
    }

    encrypt_response(&state, &session_id, &out).await
}

async fn get_memory_block(
    State(state): State<Arc<AppState>>,
    Path(label): Path<String>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<MemoryBlockResponse>>, ApiError> {
    if label.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let Some(block) =
        MemoryBlock::get_by_user_and_label(&mut conn, user.uuid, &label).map_err(|e| {
            error!("Failed to load memory block: {e:?}");
            ApiError::InternalServerError
        })?
    else {
        return Err(ApiError::NotFound);
    };

    let value = decrypt_string(&user_key, Some(&block.value_enc))
        .map_err(|_| ApiError::InternalServerError)?
        .unwrap_or_default();

    let response = MemoryBlockResponse {
        label: block.label,
        description: block.description,
        value,
        char_limit: block.char_limit,
        read_only: block.read_only,
        version: block.version,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn update_memory_block(
    State(state): State<Arc<AppState>>,
    Path(label): Path<String>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<UpdateMemoryBlockRequest>,
) -> Result<Json<EncryptedResponse<MemoryBlockResponse>>, ApiError> {
    if label.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let existing =
        MemoryBlock::get_by_user_and_label(&mut conn, user.uuid, &label).map_err(|e| {
            error!("Failed to load memory block: {e:?}");
            ApiError::InternalServerError
        })?;

    if let Some(b) = &existing {
        if b.read_only {
            return Err(ApiError::Unauthorized);
        }
    }

    let char_limit = body
        .char_limit
        .or_else(|| existing.as_ref().map(|b| b.char_limit))
        .unwrap_or(DEFAULT_BLOCK_CHAR_LIMIT);

    if char_limit <= 0 {
        return Err(ApiError::BadRequest);
    }

    if let Some(value) = body.value.as_ref() {
        if value.len() > char_limit as usize {
            return Err(ApiError::BadRequest);
        }
    } else if body.char_limit.is_some() {
        if let Some(b) = &existing {
            let existing_value = decrypt_string(&user_key, Some(&b.value_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default();

            if existing_value.len() > char_limit as usize {
                return Err(ApiError::BadRequest);
            }
        }
    }

    let value_enc = match body.value {
        Some(value) => encrypt_with_key(&user_key, value.as_bytes()).await,
        None => match &existing {
            Some(b) => b.value_enc.clone(),
            None => encrypt_with_key(&user_key, b"").await,
        },
    };

    let new_block = NewMemoryBlock {
        uuid: existing
            .as_ref()
            .map(|b| b.uuid)
            .unwrap_or_else(Uuid::new_v4),
        user_id: user.uuid,
        label: label.clone(),
        description: body
            .description
            .or_else(|| existing.as_ref().and_then(|b| b.description.clone())),
        value_enc,
        char_limit,
        read_only: body
            .read_only
            .or_else(|| existing.as_ref().map(|b| b.read_only))
            .unwrap_or(false),
        version: 1,
    };

    let updated = new_block.insert_or_update(&mut conn).map_err(|e| {
        error!("Failed to upsert memory block: {e:?}");
        ApiError::InternalServerError
    })?;

    let decrypted_value = decrypt_string(&user_key, Some(&updated.value_enc))
        .map_err(|_| ApiError::InternalServerError)?
        .unwrap_or_default();

    let response = MemoryBlockResponse {
        label: updated.label,
        description: updated.description,
        value: decrypted_value,
        char_limit: updated.char_limit,
        read_only: updated.read_only,
        version: updated.version,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn insert_archival(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<InsertArchivalRequest>,
) -> Result<Json<EncryptedResponse<InsertArchivalResponse>>, ApiError> {
    if body.text.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }
    if let Some(m) = &body.metadata {
        if !m.is_object() {
            return Err(ApiError::BadRequest);
        }
    }

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let inserted = rag::insert_archival_embedding(
        &state,
        &user,
        AuthMethod::Jwt,
        &user_key,
        &body.text,
        body.metadata.as_ref(),
    )
    .await?;

    let response = InsertArchivalResponse {
        id: inserted.uuid,
        source_type: inserted.source_type,
        embedding_model: inserted.embedding_model,
        token_count: inserted.token_count,
        created_at: inserted.created_at,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn delete_archival(
    State(state): State<Arc<AppState>>,
    Path(embedding_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    use crate::models::schema::user_embeddings::dsl::*;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let source: Option<String> = user_embeddings
        .filter(user_id.eq(user.uuid))
        .filter(uuid.eq(embedding_uuid))
        .select(source_type)
        .first::<String>(&mut conn)
        .optional()
        .map_err(|_| ApiError::InternalServerError)?;

    match source.as_deref() {
        Some(crate::rag::SOURCE_TYPE_ARCHIVAL) => {}
        Some(_) => return Err(ApiError::NotFound),
        None => return Err(ApiError::NotFound),
    }

    rag::delete_user_embedding_by_uuid(&state, user.uuid, embedding_uuid).await?;

    let response = DeletedObjectResponse {
        id: embedding_uuid,
        object: "archival.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn memory_search(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<MemorySearchRequest>,
) -> Result<Json<EncryptedResponse<MemorySearchResponse>>, ApiError> {
    if body.query.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let source_types = body.source_types.unwrap_or_else(|| {
        vec![
            crate::rag::SOURCE_TYPE_MESSAGE.to_string(),
            crate::rag::SOURCE_TYPE_ARCHIVAL.to_string(),
        ]
    });

    let results = rag::search_user_embeddings(
        &state,
        &user,
        AuthMethod::Jwt,
        &user_key,
        &body.query,
        body.top_k.unwrap_or(5),
        body.max_tokens,
        Some(&source_types),
        None,
    )
    .await?;

    let response = MemorySearchResponse { results };

    encrypt_response(&state, &session_id, &response).await
}

async fn list_agent_conversations(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationListResponse>>, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let cfg = AgentConfig::get_by_user_id(&mut conn, user.uuid)
        .map_err(|_| ApiError::InternalServerError)?;

    let mut data: Vec<ConversationResponse> = Vec::new();

    if let Some(cfg) = cfg {
        if let Some(conversation_id) = cfg.conversation_id {
            if let Ok(conv) =
                Conversation::get_by_id_and_user(&mut conn, conversation_id, user.uuid)
            {
                let metadata = decrypt_content(&user_key, conv.metadata_enc.as_ref())
                    .map_err(|_| ApiError::InternalServerError)?;

                data.push(
                    ConversationBuilder::from_conversation(&conv)
                        .metadata(metadata)
                        .build(),
                );
            }
        }
    }

    let (first_id, last_id) = Paginator::get_cursor_ids(&data, |c| c.id);
    let response = ConversationListResponse {
        object: "list",
        data,
        has_more: false,
        first_id,
        last_id,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn list_agent_conversation_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_uuid): Path<Uuid>,
    Query(params): Query<ListItemsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItemListResponse>>, ApiError> {
    let ctx = load_agent_conversation_context(&state, user.uuid, conversation_uuid).await?;

    let limit = if params.limit <= 0 {
        DEFAULT_PAGINATION_LIMIT
    } else {
        params.limit.min(MAX_PAGINATION_LIMIT)
    };

    let raw_messages = state
        .db
        .get_conversation_context_messages(
            ctx.conversation.id,
            limit + 1,
            params.after,
            &params.order,
        )
        .map_err(error_mapping::map_message_error)?;

    let has_more = raw_messages.len() > limit as usize;

    let messages_to_return = if has_more {
        &raw_messages[..limit as usize]
    } else {
        &raw_messages[..]
    };

    let items = ConversationItemConverter::messages_to_items(
        messages_to_return,
        &ctx.user_key,
        0,
        messages_to_return.len(),
    )?;

    let (first_id, last_id) = Paginator::get_cursor_ids(&items, |item| match item {
        ConversationItem::Message { id, .. } => *id,
        ConversationItem::FunctionToolCall { id, .. } => *id,
        ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        ConversationItem::Reasoning { id, .. } => *id,
    });

    let response = ConversationItemListResponse {
        object: "list",
        data: items,
        has_more,
        first_id,
        last_id,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn delete_agent_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    let ctx = load_agent_conversation_context(&state, user.uuid, conversation_uuid).await?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    conn.transaction(|tx| -> Result<(), diesel::result::Error> {
        use crate::models::schema::{agent_config, conversations};

        diesel::delete(
            user_embeddings::table
                .filter(user_embeddings::user_id.eq(user.uuid))
                .filter(user_embeddings::conversation_id.eq(Some(ctx.conversation.id)))
                .filter(user_embeddings::source_type.eq(crate::rag::SOURCE_TYPE_MESSAGE)),
        )
        .execute(tx)?;

        // Delete the conversation (cascades). We keep the filter on user_id for safety.
        let deleted = diesel::delete(
            conversations::table
                .filter(conversations::id.eq(ctx.conversation.id))
                .filter(conversations::user_id.eq(user.uuid)),
        )
        .execute(tx)?;

        if deleted == 0 {
            return Err(diesel::result::Error::NotFound);
        }

        diesel::update(agent_config::table.filter(agent_config::user_id.eq(user.uuid)))
            .set(agent_config::conversation_id.eq::<Option<i64>>(None))
            .execute(tx)?;

        Ok(())
    })
    .map_err(|_| ApiError::InternalServerError)?;

    state.rag_cache.lock().await.evict_user(user.uuid);

    let response = DeletedObjectResponse::conversation(conversation_uuid);

    encrypt_response(&state, &session_id, &response).await
}

struct AgentConversationContext {
    conversation: Conversation,
    user_key: secp256k1::SecretKey,
}

async fn load_agent_conversation_context(
    state: &AppState,
    user_id: Uuid,
    conversation_uuid: Uuid,
) -> Result<AgentConversationContext, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let cfg = AgentConfig::get_by_user_id(&mut conn, user_id)
        .map_err(|_| ApiError::InternalServerError)?
        .ok_or(ApiError::NotFound)?;

    let Some(agent_conversation_id) = cfg.conversation_id else {
        return Err(ApiError::NotFound);
    };

    let conversation = Conversation::get_by_uuid_and_user(&mut conn, conversation_uuid, user_id)
        .map_err(|_| ApiError::NotFound)?;

    if conversation.id != agent_conversation_id {
        return Err(ApiError::NotFound);
    }

    let user_key = state
        .get_user_key(user_id, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    Ok(AgentConversationContext {
        conversation,
        user_key,
    })
}

async fn get_or_create_agent_config(
    conn: &mut PgConnection,
    user_id: Uuid,
) -> Result<AgentConfig, ApiError> {
    let Some(cfg) =
        AgentConfig::get_by_user_id(conn, user_id).map_err(|_| ApiError::InternalServerError)?
    else {
        let new_cfg = NewAgentConfig {
            uuid: Uuid::new_v4(),
            user_id,
            conversation_id: None,
            enabled: true,
            model: "deepseek-r1-0528".to_string(),
            max_context_tokens: 100_000,
            compaction_threshold: 0.80,
            system_prompt_enc: None,
            preferences_enc: None,
        };

        return new_cfg
            .insert_or_update(conn)
            .map_err(|_| ApiError::InternalServerError);
    };

    Ok(cfg)
}
