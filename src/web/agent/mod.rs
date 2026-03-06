use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    response::sse::{Event, Sse},
    routing::{delete, post},
    Extension, Json, Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::time::sleep;
use tracing::{error, warn};
use uuid::Uuid;

use crate::models::agents::{Agent, AGENT_CREATED_BY_USER, AGENT_KIND_SUBAGENT};
use crate::models::users::User;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::responses::handlers::encrypt_event;
use crate::web::responses::{
    error_mapping, DeletedObjectResponse, MessageContent, MessageContentConverter,
    MessageContentPart,
};
use crate::{ApiError, AppMode, AppState};

mod compaction;
mod runtime;
mod signatures;
mod tools;
mod vision;

#[derive(Debug, Clone, Deserialize)]
struct AgentChatRequest {
    input: MessageContent,
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
    conversation_id: Uuid,
    display_name: String,
    purpose: String,
    created_by: String,
    created_at: i64,
}

#[derive(Debug, Clone)]
enum ChatTarget {
    Main,
    Subagent(Uuid),
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
const EVENT_AGENT_TYPING: &str = "agent.typing";
const EVENT_AGENT_DONE: &str = "agent.done";
const EVENT_AGENT_ERROR: &str = "agent.error";

const MESSAGE_STAGGER_DELAY_MS: u64 = 1_500;

#[derive(Debug, Clone, Serialize)]
struct AgentMessageEvent {
    messages: Vec<String>,
    step: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AgentTypingEvent {
    step: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AgentDoneEvent {
    total_steps: usize,
    total_messages: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AgentErrorEvent {
    error: String,
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

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    // Experimental endpoints: only enabled in Local/Dev.
    if !matches!(app_state.app_mode, AppMode::Local | AppMode::Dev) {
        return Router::new().with_state(app_state);
    }

    Router::new()
        .route(
            "/v1/agent/chat",
            post(chat_main).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AgentChatRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents",
            post(create_subagent).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<CreateSubagentRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents/:id/chat",
            post(chat_subagent).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AgentChatRequest>,
            )),
        )
        .route(
            "/v1/agent/subagents/:id",
            delete(delete_subagent)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
}

async fn chat_main(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<AgentChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    chat_with_target(state, session_id, user, body, ChatTarget::Main).await
}

async fn chat_subagent(
    State(state): State<Arc<AppState>>,
    Path(agent_uuid): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<AgentChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
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
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    MessageContentConverter::validate_content(&body.input)?;
    let input_content = MessageContentConverter::normalize_content(body.input.clone());
    if is_empty_message_content(&input_content) {
        return Err(ApiError::BadRequest);
    }

    // NOTE: We intentionally do runtime initialization *inside* the SSE stream so the client can
    // receive typing indicators immediately, even if compaction/key retrieval takes time.
    let event_stream = async_stream::stream! {
        let mut max_steps: usize = 0;
        let mut total_messages: usize = 0;
        let mut had_error = false;
        let mut input_for_agent = String::new();

        let mut runtime_opt: Option<runtime::AgentRuntime> = None;
        'init: {
            // Response starts: immediately emit a typing indicator.
            let initial_typing = AgentTypingEvent { step: 0 };
            match encrypt_agent_event(&state, &session_id, EVENT_AGENT_TYPING, &initial_typing)
                .await
            {
                Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                Err(e) => {
                    error!("Failed to encrypt initial typing event: {e:?}");
                    yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Encryption failed"));
                    had_error = true;
                    break 'init;
                }
            }

            let user_key = match state.get_user_key(user.uuid, None, None).await {
                Ok(k) => k,
                Err(_) => {
                    let err_event = AgentErrorEvent {
                        error: "Failed to initialize agent session.".to_string(),
                    };
                    match encrypt_agent_event(&state, &session_id, EVENT_AGENT_ERROR, &err_event)
                        .await
                    {
                        Ok(event) => yield Ok(event),
                        Err(_) => {
                            yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Error"))
                        }
                    }
                    had_error = true;
                    break 'init;
                }
            };

            let runtime = match target.clone() {
                ChatTarget::Main => {
                    runtime::AgentRuntime::new_main(state.clone(), user.clone(), user_key).await
                }
                ChatTarget::Subagent(agent_uuid) => {
                    runtime::AgentRuntime::new_subagent(
                        state.clone(),
                        user.clone(),
                        user_key,
                        agent_uuid,
                    )
                    .await
                }
            };

            let mut runtime = match runtime {
                Ok(r) => r,
                Err(e) => {
                    error!("Agent runtime initialization error: {e:?}");
                    let err_event = AgentErrorEvent {
                        error: "Failed to initialize agent runtime.".to_string(),
                    };
                    match encrypt_agent_event(&state, &session_id, EVENT_AGENT_ERROR, &err_event)
                        .await
                    {
                        Ok(event) => yield Ok(event),
                        Err(_) => {
                            yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Error"))
                        }
                    }
                    had_error = true;
                    break 'init;
                }
            };

            match runtime.prepare(&input_content).await {
                Ok(prepared) => {
                    input_for_agent = prepared;
                }
                Err(e) => {
                    error!("Agent prepare() failed: {e:?}");
                    let err_event = AgentErrorEvent {
                        error: "Agent encountered an error preparing your request.".to_string(),
                    };
                    match encrypt_agent_event(&state, &session_id, EVENT_AGENT_ERROR, &err_event).await
                    {
                        Ok(event) => yield Ok(event),
                        Err(_) => yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Error")),
                    }
                    had_error = true;
                    break 'init;
                }
            }

            max_steps = runtime.max_steps();
            runtime_opt = Some(runtime);
        }

        if let Some(mut runtime) = runtime_opt {
            'steps: for step_num in 0..max_steps {
            // Immediately indicate that the agent is working.
            let typing_event = AgentTypingEvent { step: step_num };
            match encrypt_agent_event(&state, &session_id, EVENT_AGENT_TYPING, &typing_event).await {
                Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                Err(e) => {
                    error!("Failed to encrypt agent typing event: {e:?}");
                    yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Encryption failed"));
                    had_error = true;
                    break 'steps;
                }
            }

            let step_result = runtime.step(&input_for_agent, step_num == 0).await;

            match step_result {
                Ok(result) => {
                    let runtime::StepResult {
                        messages,
                        executed_tools,
                        done,
                        ..
                    } = result;

                    // Persist assistant messages SYNCHRONOUSLY (so next step sees them).
                    // Embedding updates happen async inside insert_assistant_message.
                    for msg in &messages {
                        if let Err(e) = runtime.insert_assistant_message(msg).await {
                            error!("Failed to persist assistant message: {e:?}");
                        }
                    }

                    // Persist tool calls SYNCHRONOUSLY (so next step sees them in context)
                    for executed in &executed_tools {
                        if let Err(e) = runtime
                            .insert_tool_call_and_output(&executed.tool_call, &executed.result)
                            .await
                        {
                            error!("Failed to persist tool call: {e:?}");
                        }
                    }

                    // Emit messages to client with optional staggered delivery.
                    if !messages.is_empty() {
                        total_messages += messages.len();
                        let message_count = messages.len();

                        for (idx, msg) in messages.into_iter().enumerate() {
                            let event_data = AgentMessageEvent {
                                messages: vec![msg],
                                step: step_num,
                            };

                            match encrypt_agent_event(
                                &state,
                                &session_id,
                                EVENT_AGENT_MESSAGE,
                                &event_data,
                            )
                            .await
                            {
                                Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                                Err(e) => {
                                    error!("Failed to encrypt agent message event: {e:?}");
                                    yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Encryption failed"));
                                    had_error = true;
                                    break 'steps;
                                }
                            }

                            if idx + 1 < message_count {
                                let typing_event = AgentTypingEvent { step: step_num };
                                match encrypt_agent_event(
                                    &state,
                                    &session_id,
                                    EVENT_AGENT_TYPING,
                                    &typing_event,
                                )
                                .await
                                {
                                    Ok(event) => {
                                        yield Ok::<Event, std::convert::Infallible>(event)
                                    }
                                    Err(e) => {
                                        error!("Failed to encrypt agent typing event: {e:?}");
                                        yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Encryption failed"));
                                        had_error = true;
                                        break 'steps;
                                    }
                                }

                                sleep(Duration::from_millis(MESSAGE_STAGGER_DELAY_MS)).await;
                            }
                        }
                    }

                    if done {
                        break;
                    }
                }
                Err(e) => {
                    error!("Agent step {} error: {e:?}", step_num);
                    let err_event = AgentErrorEvent {
                        error: "Agent encountered an error processing your message.".to_string(),
                    };
                    match encrypt_agent_event(&state, &session_id, EVENT_AGENT_ERROR, &err_event)
                        .await
                    {
                        Ok(event) => yield Ok(event),
                        Err(_) => yield Ok(Event::default().event(EVENT_AGENT_ERROR).data("Error")),
                    }
                    had_error = true;
                    break 'steps;
                }
            }
            }
        }

        if !had_error && total_messages == 0 {
            warn!("Agent produced no messages");
            let fallback = AgentMessageEvent {
                messages: vec!["I apologize, but I wasn't able to generate a response.".to_string()],
                step: 0,
            };
            if let Ok(event) =
                encrypt_agent_event(&state, &session_id, EVENT_AGENT_MESSAGE, &fallback).await
            {
                yield Ok(event);
                total_messages = 1;
            }
        }

        // Final done event
        let done_event = AgentDoneEvent {
            total_steps: max_steps,
            total_messages,
        };
        match encrypt_agent_event(&state, &session_id, EVENT_AGENT_DONE, &done_event).await {
            Ok(event) => yield Ok(event),
            Err(_) => yield Ok(Event::default().data("[DONE]")),
        }
    };

    Ok(Sse::new(event_stream))
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
        runtime::ensure_main_agent(&state, &mut conn, &user_key, user.uuid).await?;
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
        conversation_id: conversation.uuid,
        display_name,
        purpose: body.purpose.trim().to_string(),
        created_by: agent.created_by,
        created_at: agent.created_at.timestamp(),
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

    state
        .db
        .delete_conversation(agent.conversation_id, user.uuid)
        .map_err(error_mapping::map_generic_db_error)?;

    state.rag_cache.lock().await.evict_user(user.uuid);

    let response = DeletedObjectResponse {
        id: agent.uuid,
        object: "agent.subagent.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}
