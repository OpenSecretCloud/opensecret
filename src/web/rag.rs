use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Extension, Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::models::users::User;
use crate::rag;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::responses::error_mapping;
use crate::{ApiError, AppMode, AppState};

#[derive(Debug, Clone, Deserialize)]
struct InsertEmbeddingRequest {
    text: String,
    metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct InsertEmbeddingResponse {
    id: Uuid,
    source_type: String,
    embedding_model: String,
    token_count: i32,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
    max_tokens: Option<i32>,
    source_types: Option<Vec<String>>,
    conversation_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize)]
struct SearchResponse {
    results: Vec<rag::RagSearchResult>,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    // Experimental endpoints: only enabled in Local/Dev.
    if !matches!(app_state.app_mode, AppMode::Local | AppMode::Dev) {
        return Router::new().with_state(app_state);
    }

    Router::new()
        .route(
            "/v1/rag/embeddings",
            post(insert_archival_embedding).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<InsertEmbeddingRequest>,
            )),
        )
        .route(
            "/v1/rag/embeddings",
            delete(delete_all_embeddings)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/rag/embeddings/:id",
            delete(delete_embedding)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/rag/search",
            post(search).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<SearchRequest>,
            )),
        )
        .route(
            "/v1/rag/embeddings/status",
            get(status).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
}

async fn insert_archival_embedding(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<InsertEmbeddingRequest>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
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
        user.uuid,
        &user_key,
        &body.text,
        body.metadata.as_ref(),
    )
    .await?;

    let response = InsertEmbeddingResponse {
        id: inserted.uuid,
        source_type: inserted.source_type,
        embedding_model: inserted.embedding_model,
        token_count: inserted.token_count,
        created_at: inserted.created_at,
    };

    let encrypted = encrypt_response(&state, &session_id, &response).await?;
    Ok((axum::http::StatusCode::CREATED, encrypted))
}

async fn search(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
    Extension(body): Extension<SearchRequest>,
) -> Result<Json<EncryptedResponse<SearchResponse>>, ApiError> {
    if body.query.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let top_k = body.top_k.unwrap_or(5);
    if top_k == 0 || top_k > 20 {
        return Err(ApiError::BadRequest);
    }

    if let Some(max_tokens) = body.max_tokens {
        if max_tokens <= 0 {
            return Err(ApiError::BadRequest);
        }
    }

    if let Some(source_types) = &body.source_types {
        if source_types.is_empty() {
            return Err(ApiError::BadRequest);
        }
    }

    let conversation_internal_id = if let Some(conversation_uuid) = body.conversation_id {
        let conversation = state
            .db
            .get_conversation_by_uuid_and_user(conversation_uuid, user.uuid)
            .map_err(error_mapping::map_conversation_error)?;
        Some(conversation.id)
    } else {
        None
    };

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|_| error_mapping::map_key_retrieval_error())?;

    let results = rag::search_user_embeddings(
        &state,
        user.uuid,
        &user_key,
        &body.query,
        top_k,
        body.max_tokens,
        body.source_types.as_deref(),
        conversation_internal_id,
    )
    .await?;

    let response = SearchResponse { results };
    encrypt_response(&state, &session_id, &response).await
}

async fn delete_all_embeddings(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
) -> Result<axum::http::StatusCode, ApiError> {
    rag::delete_all_user_embeddings(&state, user.uuid).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

async fn delete_embedding(
    State(state): State<Arc<AppState>>,
    Path(embedding_id): Path<Uuid>,
    Extension(user): Extension<User>,
) -> Result<axum::http::StatusCode, ApiError> {
    rag::delete_user_embedding_by_uuid(&state, user.uuid, embedding_id).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

async fn status(
    State(state): State<Arc<AppState>>,
    Extension(session_id): Extension<Uuid>,
    Extension(user): Extension<User>,
) -> Result<Json<EncryptedResponse<rag::RagEmbeddingsStatus>>, ApiError> {
    let response = rag::embeddings_status(&state, user.uuid).await?;
    encrypt_response(&state, &session_id, &response).await
}
