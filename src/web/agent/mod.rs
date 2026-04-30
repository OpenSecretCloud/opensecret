use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderName, HeaderValue},
    middleware::from_fn_with_state,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Extension, Json, Router,
};
use futures::Stream;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::encrypt::decrypt_string;
use crate::models::agents::{
    Agent, AGENT_CREATED_BY_AGENT, AGENT_CREATED_BY_USER, AGENT_KIND_SUBAGENT,
};
use crate::models::memory_blocks::MemoryBlock;
use crate::models::responses::{AssistantMessage, Conversation, ResponsesError};
use crate::models::users::User;
use crate::push::{enqueue_agent_message_notification, AgentPushTarget};
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::responses::constants::{
    DEFAULT_PAGINATION_LIMIT, DEFAULT_PAGINATION_ORDER, MAX_PAGINATION_LIMIT, OBJECT_TYPE_LIST,
};
use crate::web::responses::conversations::{ConversationItemListResponse, ListItemsParams};
use crate::web::responses::handlers::encrypt_event;
use crate::web::responses::{
    error_mapping, ConversationItem, ConversationItemConverter, DeletedObjectResponse,
    MessageContent, MessageContentConverter, MessageContentPart, Paginator,
};
use crate::{ApiError, AppMode, AppState};

mod compaction;
mod reactions;
mod runtime;
mod schedules;
mod signatures;
mod tools;
mod vision;

pub(crate) use schedules::start_schedule_worker;

#[derive(Debug, Clone, Deserialize)]
struct AgentChatRequest {
    input: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
struct SetMessageReactionRequest {
    emoji: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct InitMainAgentRequest {
    timezone: Option<String>,
    locale: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CreateSubagentRequest {
    display_name: Option<String>,
    purpose: String,
}

#[derive(Debug, Clone, Serialize)]
struct SubagentResponse {
    id: Uuid,
    object: &'static str,
    kind: &'static str,
    conversation_id: Uuid,
    display_name: String,
    purpose: String,
    created_by: String,
    created_at: i64,
    updated_at: i64,
}

#[derive(Debug, Clone, Serialize)]
struct MainAgentResponse {
    id: Uuid,
    object: &'static str,
    kind: &'static str,
    conversation_id: Uuid,
    display_name: &'static str,
    created_at: i64,
    updated_at: i64,
}

#[derive(Debug, Clone, Serialize)]
struct InitMainAgentResponse {
    #[serde(flatten)]
    agent: MainAgentResponse,
    messages: Vec<ConversationItem>,
}

#[derive(Debug, Clone, Serialize)]
struct SubagentListResponse {
    object: &'static str,
    data: Vec<SubagentResponse>,
    has_more: bool,
    first_id: Option<Uuid>,
    last_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
struct ListSubagentsParams {
    #[serde(default = "default_limit")]
    limit: i64,
    after: Option<Uuid>,
    #[serde(default = "default_order")]
    order: String,
    created_by: Option<String>,
}

struct AgentConversationContext {
    agent: Agent,
    conversation: Conversation,
    user_key: SecretKey,
}

#[derive(Debug, Clone)]
enum ChatTarget {
    Main,
    Subagent(Uuid),
}

fn chat_target_label(target: &ChatTarget) -> &'static str {
    match target {
        ChatTarget::Main => "main",
        ChatTarget::Subagent(_) => "subagent",
    }
}

const MAIN_AGENT_DISPLAY_NAME: &str = "Maple";

fn default_limit() -> i64 {
    DEFAULT_PAGINATION_LIMIT
}

fn default_order() -> String {
    DEFAULT_PAGINATION_ORDER.to_string()
}

fn is_empty_message_content(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(text) => text.trim().is_empty(),
        MessageContent::Parts(parts) => parts.iter().all(|part| match part {
            MessageContentPart::Text { text } | MessageContentPart::InputText { text } => {
                text.trim().is_empty()
            }
            MessageContentPart::InputImage {
                image_url, file_id, ..
            } => {
                let url_empty = image_url
                    .as_deref()
                    .map(|s| s.trim().is_empty())
                    .unwrap_or(true);
                let file_empty = file_id
                    .as_deref()
                    .map(|s| s.trim().is_empty())
                    .unwrap_or(true);
                url_empty && file_empty
            }
            MessageContentPart::InputFile {
                filename,
                file_data,
            } => filename.trim().is_empty() && file_data.trim().is_empty(),
        }),
    }
}

// SSE event types for agent chat (message-level delivery, not token streaming)
const EVENT_AGENT_MESSAGE: &str = "agent.message";
const EVENT_AGENT_REACTION: &str = "agent.reaction";
const EVENT_AGENT_TYPING: &str = "agent.typing";
const EVENT_AGENT_DONE: &str = "agent.done";
const EVENT_AGENT_ERROR: &str = "agent.error";
const AGENT_SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

#[derive(Debug, Clone, Serialize)]
struct AgentMessageEvent {
    message_id: Uuid,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct AgentReactionEvent {
    item_id: Uuid,
    emoji: String,
}

#[derive(Debug, Clone, Serialize)]
struct AgentTypingEvent {}

#[derive(Debug, Clone, Serialize)]
struct AgentDoneEvent {}

#[derive(Debug, Clone, Serialize)]
struct AgentErrorEvent {
    error: String,
}

#[derive(Debug)]
enum AgentClientEvent {
    Typing(AgentTypingEvent),
    Message {
        payload: AgentMessageEvent,
        delivery_ack: oneshot::Sender<()>,
    },
    Reaction(AgentReactionEvent),
    Done(AgentDoneEvent),
    Error(AgentErrorEvent),
}

async fn encrypt_agent_event<T: Serialize>(
    state: &AppState,
    session_id: &Uuid,
    event_type: &str,
    payload: &T,
) -> Result<Event, ApiError> {
    let value = serde_json::to_value(payload).map_err(|e| {
        error!("Failed to serialize agent SSE payload for {event_type}: {e:?}");
        ApiError::InternalServerError
    })?;

    encrypt_event(state, session_id, event_type, &value).await
}

fn agent_sse_response<S>(event_stream: S) -> Response
where
    S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    let sse = Sse::new(event_stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(AGENT_SSE_KEEPALIVE_INTERVAL_SECS))
            .text("keep-alive"),
    );

    (
        [
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            (
                HeaderName::from_static("x-accel-buffering"),
                HeaderValue::from_static("no"),
            ),
        ],
        sse,
    )
        .into_response()
}

fn build_main_agent_response(ctx: &AgentConversationContext) -> MainAgentResponse {
    MainAgentResponse {
        id: ctx.agent.uuid,
        object: "agent.main",
        kind: "main",
        conversation_id: ctx.conversation.uuid,
        display_name: MAIN_AGENT_DISPLAY_NAME,
        created_at: ctx.agent.created_at.timestamp(),
        updated_at: ctx.conversation.updated_at.timestamp(),
    }
}

fn build_init_main_agent_response(
    ctx: &AgentConversationContext,
    onboarding_messages: Vec<runtime::SeededOnboardingMessage>,
) -> InitMainAgentResponse {
    InitMainAgentResponse {
        agent: build_main_agent_response(ctx),
        messages: onboarding_messages
            .into_iter()
            .map(|message| ConversationItem::Message {
                id: message.id,
                status: Some("completed".to_string()),
                role: "assistant".to_string(),
                content: MessageContentConverter::assistant_text_to_content(message.content),
                reaction: None,
                created_at: Some(message.created_at),
            })
            .collect(),
    }
}

fn build_subagent_response(
    agent: &Agent,
    conversation: &Conversation,
    user_key: &SecretKey,
) -> Result<SubagentResponse, ApiError> {
    let display_name = decrypt_string(user_key, agent.display_name_enc.as_ref())
        .map_err(|e| {
            error!("Failed to decrypt subagent display name: {e:?}");
            ApiError::InternalServerError
        })?
        .unwrap_or_else(|| "Subagent".to_string());

    let purpose = decrypt_string(user_key, agent.purpose_enc.as_ref())
        .map_err(|e| {
            error!("Failed to decrypt subagent purpose: {e:?}");
            ApiError::InternalServerError
        })?
        .unwrap_or_default();

    Ok(SubagentResponse {
        id: agent.uuid,
        object: "agent.subagent",
        kind: "subagent",
        conversation_id: conversation.uuid,
        display_name,
        purpose,
        created_by: agent.created_by.clone(),
        created_at: agent.created_at.timestamp(),
        updated_at: conversation.updated_at.timestamp(),
    })
}

fn validate_created_by_filter(created_by: Option<&str>) -> Result<Option<&str>, ApiError> {
    match created_by {
        None => Ok(None),
        Some(AGENT_CREATED_BY_USER) => Ok(Some(AGENT_CREATED_BY_USER)),
        Some(AGENT_CREATED_BY_AGENT) => Ok(Some(AGENT_CREATED_BY_AGENT)),
        Some(_) => Err(ApiError::BadRequest),
    }
}

async fn load_main_agent_context(
    state: &Arc<AppState>,
    user: &User,
) -> Result<AgentConversationContext, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let (agent, conversation) =
        runtime::load_main_agent(&mut conn, user.uuid)?.ok_or(ApiError::NotFound)?;

    Ok(AgentConversationContext {
        agent,
        conversation,
        user_key,
    })
}

async fn load_subagent_context(
    state: &Arc<AppState>,
    user: &User,
    agent_uuid: Uuid,
) -> Result<AgentConversationContext, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let agent = Agent::get_by_uuid_and_user(&mut conn, agent_uuid, user.uuid)
        .map_err(|e| {
            error!("Failed to load subagent: {e:?}");
            ApiError::InternalServerError
        })?
        .ok_or(ApiError::NotFound)?;

    if agent.kind != AGENT_KIND_SUBAGENT {
        return Err(ApiError::NotFound);
    }

    let conversation =
        Conversation::get_by_id_and_user(&mut conn, agent.conversation_id, user.uuid).map_err(
            |e| {
                error!("Failed to load subagent conversation: {e:?}");
                ApiError::InternalServerError
            },
        )?;

    Ok(AgentConversationContext {
        agent,
        conversation,
        user_key,
    })
}

fn list_items_for_conversation(
    state: &Arc<AppState>,
    conversation_id: i64,
    user_key: &SecretKey,
    params: &ListItemsParams,
) -> Result<ConversationItemListResponse, ApiError> {
    let limit = if params.limit <= 0 {
        DEFAULT_PAGINATION_LIMIT
    } else {
        params.limit.min(MAX_PAGINATION_LIMIT)
    };

    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation_id, limit + 1, params.after, &params.order)
        .map_err(error_mapping::map_message_error)?;

    let has_more = raw_messages.len() > limit as usize;
    let messages_to_return = if has_more {
        &raw_messages[..limit as usize]
    } else {
        &raw_messages[..]
    };

    let items = ConversationItemConverter::messages_to_items(
        messages_to_return,
        user_key,
        0,
        messages_to_return.len(),
    )?;

    let (first_id, last_id) = Paginator::get_cursor_ids(&items, |item| match item {
        ConversationItem::Message { id, .. } => *id,
        ConversationItem::FunctionToolCall { id, .. } => *id,
        ConversationItem::FunctionToolCallOutput { id, .. } => *id,
        ConversationItem::Reasoning { id, .. } => *id,
    });

    Ok(ConversationItemListResponse {
        object: OBJECT_TYPE_LIST,
        data: items,
        has_more,
        first_id,
        last_id,
    })
}

fn get_item_from_conversation(
    state: &Arc<AppState>,
    conversation_id: i64,
    user_key: &SecretKey,
    item_id: Uuid,
) -> Result<ConversationItem, ApiError> {
    let raw_messages = state
        .db
        .get_conversation_context_messages(conversation_id, i64::MAX, None, "asc")
        .map_err(error_mapping::map_message_error)?;

    for msg in raw_messages {
        if msg.uuid == item_id {
            return ConversationItemConverter::message_to_item(&msg, user_key);
        }
    }

    Err(ApiError::NotFound)
}

fn set_user_reaction_for_conversation(
    state: &Arc<AppState>,
    conversation_id: i64,
    user_key: &SecretKey,
    user: &User,
    item_id: Uuid,
    reaction: Option<String>,
) -> Result<ConversationItem, ApiError> {
    let item = get_item_from_conversation(state, conversation_id, user_key, item_id)?;

    match item {
        ConversationItem::Message { role, .. } if role == "assistant" => {}
        _ => return Err(ApiError::BadRequest),
    }

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    AssistantMessage::set_user_reaction(&mut conn, item_id, user.uuid, reaction).map_err(|e| {
        error!("Failed to set user reaction on assistant message: {e:?}");
        match e {
            ResponsesError::AssistantMessageNotFound => ApiError::NotFound,
            _ => ApiError::InternalServerError,
        }
    })?;

    get_item_from_conversation(state, conversation_id, user_key, item_id)
}

fn delete_conversation_for_user(
    conn: &mut diesel::PgConnection,
    conversation_id: i64,
    user_id: Uuid,
) -> Result<(), ApiError> {
    Conversation::delete_by_id_and_user(conn, conversation_id, user_id).map_err(|e| {
        error!("Failed to delete agent conversation: {e:?}");
        ApiError::InternalServerError
    })
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    // Experimental endpoints: only enabled in Local/Dev.
    if !matches!(app_state.app_mode, AppMode::Local | AppMode::Dev) {
        return Router::new().with_state(app_state);
    }

    Router::new()
        .route(
            "/v1/agent",
            get(get_main_agent).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent",
            delete(delete_main_agent)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/init",
            post(init_main_agent).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<InitMainAgentRequest>,
            )),
        )
        .route(
            "/v1/agent/items",
            get(list_main_agent_items)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/items/:item_id",
            get(get_main_agent_item)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/items/:item_id/reaction",
            post(set_main_agent_item_reaction)
                .delete(clear_main_agent_item_reaction)
                .layer(from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<SetMessageReactionRequest>,
                )),
        )
        .route(
            "/v1/agent/chat",
            post(chat_main).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AgentChatRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents",
            get(list_subagents).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/subagents",
            post(create_subagent).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<CreateSubagentRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents/:id",
            get(get_subagent).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/subagents/:id/chat",
            post(chat_subagent).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AgentChatRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents/:id/items",
            get(list_subagent_items)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/subagents/:id/items/:item_id",
            get(get_subagent_item)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/agent/subagents/:id/items/:item_id/reaction",
            post(set_subagent_item_reaction)
                .delete(clear_subagent_item_reaction)
                .layer(from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<SetMessageReactionRequest>,
                )),
        )
        .route(
            "/v1/agent/subagents/:id",
            delete(delete_subagent)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
}

async fn get_main_agent(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<MainAgentResponse>>, ApiError> {
    let ctx = load_main_agent_context(&state, &user).await?;
    let response = build_main_agent_response(&ctx);

    encrypt_response(&state, &session_id, &response).await
}

async fn init_main_agent(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<InitMainAgentRequest>,
) -> Result<Json<EncryptedResponse<InitMainAgentResponse>>, ApiError> {
    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let init_result = runtime::init_main_agent(
        &state,
        &mut conn,
        &user_key,
        user.uuid,
        &runtime::MainAgentInitOptions {
            timezone: body.timezone,
            locale: body.locale,
        },
    )
    .await?;

    let response = build_init_main_agent_response(
        &AgentConversationContext {
            agent: init_result.agent,
            conversation: init_result.conversation,
            user_key,
        },
        init_result.onboarding_messages,
    );

    encrypt_response(&state, &session_id, &response).await
}

async fn list_main_agent_items(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListItemsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItemListResponse>>, ApiError> {
    let ctx = load_main_agent_context(&state, &user).await?;
    let response =
        list_items_for_conversation(&state, ctx.conversation.id, &ctx.user_key, &params)?;

    encrypt_response(&state, &session_id, &response).await
}

async fn get_main_agent_item(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let ctx = load_main_agent_context(&state, &user).await?;
    let item = get_item_from_conversation(&state, ctx.conversation.id, &ctx.user_key, item_id)?;

    encrypt_response(&state, &session_id, &item).await
}

async fn set_main_agent_item_reaction(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<SetMessageReactionRequest>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let emoji = reactions::require_valid_reaction(&body.emoji)?;
    let ctx = load_main_agent_context(&state, &user).await?;
    let item = set_user_reaction_for_conversation(
        &state,
        ctx.conversation.id,
        &ctx.user_key,
        &user,
        item_id,
        Some(emoji),
    )?;

    encrypt_response(&state, &session_id, &item).await
}

async fn clear_main_agent_item_reaction(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let ctx = load_main_agent_context(&state, &user).await?;
    let item = set_user_reaction_for_conversation(
        &state,
        ctx.conversation.id,
        &ctx.user_key,
        &user,
        item_id,
        None,
    )?;

    encrypt_response(&state, &session_id, &item).await
}

async fn delete_main_agent(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let main_agent = Agent::get_main_for_user(&mut conn, user.uuid)
        .map_err(|e| {
            error!("Failed to load main agent for deletion: {e:?}");
            ApiError::InternalServerError
        })?
        .ok_or(ApiError::NotFound)?;

    let subagents = Agent::list_subagents_for_user(&mut conn, user.uuid).map_err(|e| {
        error!("Failed to load subagents for main agent deletion: {e:?}");
        ApiError::InternalServerError
    })?;

    for subagent in subagents {
        delete_conversation_for_user(&mut conn, subagent.conversation_id, user.uuid)?;
    }

    delete_conversation_for_user(&mut conn, main_agent.conversation_id, user.uuid)?;

    MemoryBlock::delete_all_for_user(&mut conn, user.uuid).map_err(|e| {
        error!("Failed to clear agent memory blocks during main agent deletion: {e:?}");
        ApiError::InternalServerError
    })?;

    state.rag_cache.lock().await.evict_user(user.uuid);

    let response = DeletedObjectResponse {
        id: main_agent.uuid,
        object: "agent.main.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn list_subagents(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListSubagentsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<SubagentListResponse>>, ApiError> {
    let limit = if params.limit <= 0 {
        DEFAULT_PAGINATION_LIMIT
    } else {
        params.limit.min(MAX_PAGINATION_LIMIT)
    };

    let created_by_filter = validate_created_by_filter(params.created_by.as_deref())?;

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let subagents = Agent::list_subagents_for_user_paginated(
        &mut conn,
        user.uuid,
        limit + 1,
        params.after,
        &params.order,
        created_by_filter,
    )
    .map_err(|e| {
        error!("Failed to list subagents: {e:?}");
        ApiError::InternalServerError
    })?;

    let has_more = subagents.len() > limit as usize;
    let subagents_to_return = if has_more {
        &subagents[..limit as usize]
    } else {
        &subagents[..]
    };

    let data = subagents_to_return
        .iter()
        .map(|(agent, conversation)| build_subagent_response(agent, conversation, &user_key))
        .collect::<Result<Vec<_>, _>>()?;

    let (first_id, last_id) =
        Paginator::get_cursor_ids(subagents_to_return, |(agent, _)| agent.uuid);

    let response = SubagentListResponse {
        object: OBJECT_TYPE_LIST,
        data,
        has_more,
        first_id,
        last_id,
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn get_subagent(
    State(state): State<Arc<AppState>>,
    Path(agent_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<SubagentResponse>>, ApiError> {
    let ctx = load_subagent_context(&state, &user, agent_uuid).await?;
    let response = build_subagent_response(&ctx.agent, &ctx.conversation, &ctx.user_key)?;

    encrypt_response(&state, &session_id, &response).await
}

async fn list_subagent_items(
    State(state): State<Arc<AppState>>,
    Path(agent_uuid): Path<Uuid>,
    Query(params): Query<ListItemsParams>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItemListResponse>>, ApiError> {
    let ctx = load_subagent_context(&state, &user, agent_uuid).await?;
    let response =
        list_items_for_conversation(&state, ctx.conversation.id, &ctx.user_key, &params)?;

    encrypt_response(&state, &session_id, &response).await
}

async fn get_subagent_item(
    State(state): State<Arc<AppState>>,
    Path((agent_uuid, item_id)): Path<(Uuid, Uuid)>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let ctx = load_subagent_context(&state, &user, agent_uuid).await?;
    let item = get_item_from_conversation(&state, ctx.conversation.id, &ctx.user_key, item_id)?;

    encrypt_response(&state, &session_id, &item).await
}

async fn set_subagent_item_reaction(
    State(state): State<Arc<AppState>>,
    Path((agent_uuid, item_id)): Path<(Uuid, Uuid)>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<SetMessageReactionRequest>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let emoji = reactions::require_valid_reaction(&body.emoji)?;
    let ctx = load_subagent_context(&state, &user, agent_uuid).await?;
    let item = set_user_reaction_for_conversation(
        &state,
        ctx.conversation.id,
        &ctx.user_key,
        &user,
        item_id,
        Some(emoji),
    )?;

    encrypt_response(&state, &session_id, &item).await
}

async fn clear_subagent_item_reaction(
    State(state): State<Arc<AppState>>,
    Path((agent_uuid, item_id)): Path<(Uuid, Uuid)>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<ConversationItem>>, ApiError> {
    let ctx = load_subagent_context(&state, &user, agent_uuid).await?;
    let item = set_user_reaction_for_conversation(
        &state,
        ctx.conversation.id,
        &ctx.user_key,
        &user,
        item_id,
        None,
    )?;

    encrypt_response(&state, &session_id, &item).await
}

async fn chat_main(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<AgentChatRequest>,
) -> Result<Response, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    if runtime::load_main_agent(&mut conn, user.uuid)?.is_none() {
        return Err(ApiError::NotFound);
    }

    chat_with_target(state, session_id, user, body, ChatTarget::Main).await
}

async fn chat_subagent(
    State(state): State<Arc<AppState>>,
    Path(agent_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<AgentChatRequest>,
) -> Result<Response, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let agent = Agent::get_by_uuid_and_user(&mut conn, agent_uuid, user.uuid)
        .map_err(|_| ApiError::InternalServerError)?
        .ok_or(ApiError::NotFound)?;

    if agent.kind != AGENT_KIND_SUBAGENT {
        return Err(ApiError::NotFound);
    }

    chat_with_target(
        state,
        session_id,
        user,
        body,
        ChatTarget::Subagent(agent_uuid),
    )
    .await
}

async fn chat_with_target(
    state: Arc<AppState>,
    session_id: Uuid,
    user: User,
    body: AgentChatRequest,
    target: ChatTarget,
) -> Result<Response, ApiError> {
    MessageContentConverter::validate_content(&body.input)?;
    let input_content = MessageContentConverter::normalize_content(body.input.clone());
    if is_empty_message_content(&input_content) {
        return Err(ApiError::BadRequest);
    }

    let (tx, mut rx) = mpsc::channel::<AgentClientEvent>(32);
    let worker_state = state.clone();
    let worker_user = user.clone();
    let worker_target = target.clone();

    tokio::spawn(async move {
        run_agent_chat_task(worker_state, worker_user, input_content, worker_target, tx).await;
    });

    let event_stream = async_stream::stream! {
        while let Some(client_event) = rx.recv().await {
            let encrypted = match client_event {
                AgentClientEvent::Typing(payload) => {
                    encrypt_agent_event(&state, &session_id, EVENT_AGENT_TYPING, &payload).await
                }
                AgentClientEvent::Message { payload, delivery_ack } => {
                    let encrypted = encrypt_agent_event(
                        &state,
                        &session_id,
                        EVENT_AGENT_MESSAGE,
                        &payload,
                    )
                    .await;

                    match encrypted {
                        Ok(event) => {
                            yield Ok(event);
                            let _ = delivery_ack.send(());
                            continue;
                        }
                        Err(error) => Err(error),
                    }
                }
                AgentClientEvent::Reaction(payload) => {
                    encrypt_agent_event(&state, &session_id, EVENT_AGENT_REACTION, &payload).await
                }
                AgentClientEvent::Done(payload) => {
                    encrypt_agent_event(&state, &session_id, EVENT_AGENT_DONE, &payload).await
                }
                AgentClientEvent::Error(payload) => {
                    encrypt_agent_event(&state, &session_id, EVENT_AGENT_ERROR, &payload).await
                }
            };

            match encrypted {
                Ok(event) => yield Ok(event),
                Err(e) => {
                    error!("Failed to encrypt agent SSE event: {e:?}");
                    yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Encryption failed"));
                    break;
                }
            }
        }
    };

    Ok(agent_sse_response(event_stream))
}

async fn run_agent_chat_task(
    state: Arc<AppState>,
    user: User,
    input_content: MessageContent,
    target: ChatTarget,
    tx: mpsc::Sender<AgentClientEvent>,
) {
    let (input_kind, input_part_count) = match &input_content {
        MessageContent::Text(_) => ("text", 1),
        MessageContent::Parts(parts) => ("parts", parts.len()),
    };
    debug!(
        user_uuid = %user.uuid,
        target = chat_target_label(&target),
        input_kind,
        input_part_count,
        "Starting agent chat task"
    );
    let mut total_messages: usize = 0;
    let mut had_reaction = false;
    let mut had_error = false;
    let mut client_connected = true;
    let mut first_undelivered_message: Option<(Uuid, String)> = None;

    client_connected = send_agent_client_event(
        &tx,
        AgentClientEvent::Typing(AgentTypingEvent {}),
        client_connected,
    )
    .await;

    let user_key = match state.get_user_key(user.uuid, None, None).await {
        Ok(key) => key,
        Err(_) => {
            let _ = send_agent_client_event(
                &tx,
                AgentClientEvent::Error(AgentErrorEvent {
                    error: "Failed to initialize agent session.".to_string(),
                }),
                client_connected,
            )
            .await;
            return;
        }
    };

    let runtime = match target.clone() {
        ChatTarget::Main => {
            runtime::AgentRuntime::new_main(state.clone(), user.clone(), user_key).await
        }
        ChatTarget::Subagent(agent_uuid) => {
            runtime::AgentRuntime::new_subagent(state.clone(), user.clone(), user_key, agent_uuid)
                .await
        }
    };

    let mut runtime = match runtime {
        Ok(runtime) => runtime,
        Err(e) => {
            error!("Agent runtime initialization error: {e:?}");
            let error_message = if matches!(e, ApiError::NotFound) {
                "Main agent is not initialized.".to_string()
            } else {
                "Failed to initialize agent runtime.".to_string()
            };
            let _ = send_agent_client_event(
                &tx,
                AgentClientEvent::Error(AgentErrorEvent {
                    error: error_message,
                }),
                client_connected,
            )
            .await;
            return;
        }
    };

    let input_for_agent = match runtime.prepare(&input_content).await {
        Ok(prepared) => prepared,
        Err(e) => {
            error!("Agent prepare() failed: {e:?}");
            let _ = send_agent_client_event(
                &tx,
                AgentClientEvent::Error(AgentErrorEvent {
                    error: "Agent encountered an error preparing your request.".to_string(),
                }),
                client_connected,
            )
            .await;
            return;
        }
    };

    let max_steps = runtime.max_steps();

    'steps: for step_num in 0..max_steps {
        debug!(
            user_uuid = %user.uuid,
            target = chat_target_label(&target),
            step_num,
            total_messages,
            had_reaction,
            "Starting agent chat step"
        );

        match runtime.step(&input_for_agent, step_num == 0).await {
            Ok(result) => {
                let runtime::StepResult {
                    messages,
                    reply_reaction,
                    executed_tools,
                    done,
                    ..
                } = result;

                debug!(
                    user_uuid = %user.uuid,
                    target = chat_target_label(&target),
                    step_num,
                    message_count = messages.len(),
                    executed_tool_count = executed_tools.len(),
                    tool_names = ?executed_tools
                        .iter()
                        .map(|executed| executed.tool_call.name.as_str())
                        .collect::<Vec<_>>(),
                    has_reply_reaction = reply_reaction.is_some(),
                    done,
                    "Agent chat step returned"
                );

                for executed in &executed_tools {
                    if let Err(e) = runtime
                        .insert_tool_call_and_output(&executed.tool_call, &executed.result)
                        .await
                    {
                        error!("Failed to persist tool call: {e:?}");
                    }
                }

                if let Some(reaction) = reply_reaction {
                    match runtime
                        .set_assistant_reaction_for_current_user_message(&reaction)
                        .await
                    {
                        Ok(updated_message) => {
                            had_reaction = true;
                            client_connected = send_agent_client_event(
                                &tx,
                                AgentClientEvent::Reaction(AgentReactionEvent {
                                    item_id: updated_message.uuid,
                                    emoji: reaction,
                                }),
                                client_connected,
                            )
                            .await;
                        }
                        Err(e) => {
                            error!("Failed to persist assistant reaction: {e:?}");
                        }
                    }
                }

                if !messages.is_empty() {
                    for msg in messages {
                        let assistant_message = match runtime.insert_assistant_message(&msg).await {
                            Ok(message) => message,
                            Err(e) => {
                                error!("Failed to persist assistant message: {e:?}");
                                had_error = true;
                                let _ = send_agent_client_event(
                                    &tx,
                                    AgentClientEvent::Error(AgentErrorEvent {
                                        error: "Agent encountered an error saving its response."
                                            .to_string(),
                                    }),
                                    client_connected,
                                )
                                .await;
                                break 'steps;
                            }
                        };

                        total_messages += 1;

                        let delivered = send_agent_message_event(
                            &tx,
                            AgentMessageEvent {
                                message_id: assistant_message.uuid,
                                message: msg.clone(),
                            },
                            client_connected,
                        )
                        .await;

                        if !delivered {
                            client_connected = false;
                            if first_undelivered_message.is_none() {
                                first_undelivered_message =
                                    Some((assistant_message.uuid, msg.clone()));
                            }
                        } else {
                            client_connected = true;
                        }
                    }
                }

                if done {
                    break 'steps;
                }
            }
            Err(e) => {
                error!("Agent step {} error: {e:?}", step_num);
                had_error = true;
                let _ = send_agent_client_event(
                    &tx,
                    AgentClientEvent::Error(AgentErrorEvent {
                        error: "Agent encountered an error processing your message.".to_string(),
                    }),
                    client_connected,
                )
                .await;
                break 'steps;
            }
        }
    }

    if !had_error && total_messages == 0 && !had_reaction {
        warn!(
            user_uuid = %user.uuid,
            target = chat_target_label(&target),
            "Agent produced no messages or reactions; sending fallback response"
        );
        let fallback_message = "I apologize, but I wasn't able to generate a response.".to_string();
        match runtime.insert_assistant_message(&fallback_message).await {
            Ok(message) => {
                total_messages = 1;
                let delivered = send_agent_message_event(
                    &tx,
                    AgentMessageEvent {
                        message_id: message.uuid,
                        message: fallback_message.clone(),
                    },
                    client_connected,
                )
                .await;

                if !delivered {
                    client_connected = false;
                    if first_undelivered_message.is_none() {
                        first_undelivered_message = Some((message.uuid, fallback_message));
                    }
                } else {
                    client_connected = true;
                }
            }
            Err(e) => {
                error!("Failed to persist fallback assistant message: {e:?}");
                had_error = true;
                let _ = send_agent_client_event(
                    &tx,
                    AgentClientEvent::Error(AgentErrorEvent {
                        error: "Agent encountered an error saving its response.".to_string(),
                    }),
                    client_connected,
                )
                .await;
            }
        }
    }

    if let Some((message_id, message_text)) = first_undelivered_message {
        enqueue_agent_push_for_disconnect(&state, &user, &target, message_id, &message_text).await;
    }

    if !had_error {
        let _ = send_agent_client_event(
            &tx,
            AgentClientEvent::Done(AgentDoneEvent {}),
            client_connected,
        )
        .await;
    }

    debug!(
        user_uuid = %user.uuid,
        target = chat_target_label(&target),
        total_messages,
        had_reaction,
        had_error,
        client_connected,
        "Agent chat task finished"
    );
}

async fn send_agent_client_event(
    tx: &mpsc::Sender<AgentClientEvent>,
    event: AgentClientEvent,
    client_connected: bool,
) -> bool {
    if !client_connected {
        return false;
    }

    tx.send(event).await.is_ok()
}

async fn send_agent_message_event(
    tx: &mpsc::Sender<AgentClientEvent>,
    payload: AgentMessageEvent,
    client_connected: bool,
) -> bool {
    if !client_connected {
        return false;
    }

    let (delivery_ack, ack_rx) = oneshot::channel();
    if tx
        .send(AgentClientEvent::Message {
            payload,
            delivery_ack,
        })
        .await
        .is_err()
    {
        return false;
    }

    ack_rx.await.is_ok()
}

async fn enqueue_agent_push_for_disconnect(
    state: &Arc<AppState>,
    user: &User,
    target: &ChatTarget,
    message_id: Uuid,
    message_text: &str,
) {
    let push_target = match target {
        ChatTarget::Main => AgentPushTarget::Main,
        ChatTarget::Subagent(agent_uuid) => AgentPushTarget::Subagent(*agent_uuid),
    };

    if let Err(e) =
        enqueue_agent_message_notification(state, user, push_target, message_id, message_text).await
    {
        error!("Failed to enqueue agent push notification after SSE disconnect: {e:?}");
    }
}

async fn create_subagent(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<CreateSubagentRequest>,
) -> Result<Json<EncryptedResponse<SubagentResponse>>, ApiError> {
    if body.purpose.trim().is_empty() {
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

    let (main_agent, _) =
        runtime::load_main_agent(&mut conn, user.uuid)?.ok_or(ApiError::NotFound)?;
    let (agent, conversation, display_name) = runtime::create_subagent(
        &mut conn,
        &user_key,
        user.uuid,
        &main_agent,
        body.display_name.as_deref(),
        body.purpose.trim(),
        AGENT_CREATED_BY_USER,
    )
    .await?;

    let response = SubagentResponse {
        id: agent.uuid,
        object: "agent.subagent",
        kind: "subagent",
        conversation_id: conversation.uuid,
        display_name,
        purpose: body.purpose.trim().to_string(),
        created_by: agent.created_by,
        created_at: agent.created_at.timestamp(),
        updated_at: conversation.updated_at.timestamp(),
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn delete_subagent(
    State(state): State<Arc<AppState>>,
    Path(agent_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<DeletedObjectResponse>>, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let agent = Agent::get_by_uuid_and_user(&mut conn, agent_uuid, user.uuid)
        .map_err(|_| ApiError::InternalServerError)?
        .ok_or(ApiError::NotFound)?;

    if agent.kind != AGENT_KIND_SUBAGENT {
        return Err(ApiError::NotFound);
    }

    delete_conversation_for_user(&mut conn, agent.conversation_id, user.uuid)?;

    state.rag_cache.lock().await.evict_user(user.uuid);

    let response = DeletedObjectResponse {
        id: agent.uuid,
        object: "agent.subagent.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}
