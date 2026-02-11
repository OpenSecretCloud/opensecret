use axum::{
    extract::State, middleware::from_fn_with_state, routing::post, Extension, Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

use crate::encrypt::decrypt_string;
use crate::models::agent_config::AgentConfig;
use crate::models::memory_blocks::MemoryBlock;
use crate::models::users::User;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::responses::error_mapping;
use crate::{ApiError, AppMode, AppState};

mod signatures;

#[derive(Debug, Clone, Deserialize)]
struct AgentChatRequest {
    input: String,
}

#[derive(Debug, Clone, Serialize)]
struct AgentChatResponse {
    messages: Vec<String>,
    tool_calls: Vec<signatures::AgentToolCall>,
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

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let agent_config = AgentConfig::get_by_user_id(&mut conn, user.uuid).map_err(|e| {
        error!("Failed to load agent config: {e:?}");
        ApiError::InternalServerError
    })?;

    let persona =
        MemoryBlock::get_by_user_and_label(&mut conn, user.uuid, "persona").map_err(|e| {
            error!("Failed to load persona memory block: {e:?}");
            ApiError::InternalServerError
        })?;
    let human = MemoryBlock::get_by_user_and_label(&mut conn, user.uuid, "human").map_err(|e| {
        error!("Failed to load human memory block: {e:?}");
        ApiError::InternalServerError
    })?;

    let persona_text = decrypt_string(&user_key, persona.as_ref().map(|b| &b.value_enc))
        .map_err(|e| {
            error!("Failed to decrypt persona memory block: {e:?}");
            ApiError::InternalServerError
        })?
        .unwrap_or_default();

    let human_text = decrypt_string(&user_key, human.as_ref().map(|b| &b.value_enc))
        .map_err(|e| {
            error!("Failed to decrypt human memory block: {e:?}");
            ApiError::InternalServerError
        })?
        .unwrap_or_default();

    let is_first_time_user = persona.is_none() && human.is_none();

    let input = signatures::AgentResponseInput {
        input: body.input,
        current_time: Utc::now().to_rfc3339(),
        persona_block: persona_text,
        human_block: human_text,
        memory_metadata: String::new(),
        previous_context_summary: String::new(),
        recent_conversation: String::new(),
        available_tools: "- done: call when you're finished".to_string(),
        is_first_time_user,
    };

    let model = agent_config
        .as_ref()
        .map(|c| c.model.as_str())
        .unwrap_or("deepseek-r1-0528");
    let output = signatures::run_agent_response_signature(
        &state,
        &user,
        model,
        signatures::DEFAULT_AGENT_SYSTEM_PROMPT,
        &input,
    )
    .await?;

    let response = AgentChatResponse {
        messages: output.messages,
        tool_calls: output.tool_calls,
    };

    encrypt_response(&state, &session_id, &response).await
}
