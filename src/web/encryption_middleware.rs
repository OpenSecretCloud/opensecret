use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Method, Request},
    middleware::Next,
    response::Response,
    Json,
};
use base64::Engine;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::{ApiError, AppState};

const MAX_ENCRYPTED_BODY_BYTES: usize = 50 * 1024 * 1024; // 50MB

#[derive(Deserialize)]
pub struct EncryptedRequest {
    pub encrypted: String,
}

#[derive(Serialize)]
pub struct EncryptedResponse<T: Serialize> {
    pub encrypted: String,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Serialize> EncryptedResponse<T> {
    pub fn new(encrypted: String) -> Self {
        Self {
            encrypted,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub async fn decrypt_request<T>(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, ApiError>
where
    T: DeserializeOwned + Send + Sync + Clone + 'static,
{
    tracing::debug!("Entering decrypt_request");
    let session_id = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| Uuid::parse_str(v).ok())
        .ok_or(ApiError::BadRequest)?;

    // Skip body processing for GET, DELETE, or when T is ()
    if request.method() == Method::GET
        || request.method() == Method::DELETE
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<()>()
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<()>() {
            request.extensions_mut().insert(());
        }
        request.extensions_mut().insert(session_id);
        return Ok(next.run(request).await);
    }

    let body = std::mem::replace(request.body_mut(), Body::empty());
    let body_bytes = axum::body::to_bytes(body, MAX_ENCRYPTED_BODY_BYTES)
        .await
        .map_err(|_| ApiError::PayloadTooLarge)?;

    let encrypted_request: EncryptedRequest =
        serde_json::from_slice(&body_bytes).map_err(|_| ApiError::BadRequest)?;

    let decrypted_data = state
        .decrypt_session_data(&session_id, &encrypted_request.encrypted)
        .await
        .map_err(|_| ApiError::BadRequest)?;

    let decrypted: T = serde_json::from_slice(&decrypted_data).map_err(|e| {
        tracing::error!("Failed to deserialize decrypted data: {:?}", e);
        ApiError::BadRequest
    })?;

    request.extensions_mut().insert(decrypted);
    request.extensions_mut().insert(session_id);

    tracing::debug!("Exiting decrypt_request");
    Ok(next.run(request).await)
}

pub async fn encrypt_response<T: Serialize>(
    state: &AppState,
    session_id: &Uuid,
    response: &T,
) -> Result<Json<EncryptedResponse<T>>, ApiError> {
    tracing::debug!("Entering encrypt_response");
    let response_json = serde_json::to_vec(response).map_err(|_| ApiError::InternalServerError)?;
    let encrypted_response = state
        .encrypt_session_data(session_id, &response_json)
        .await?;
    tracing::debug!("Exiting encrypt_response");
    Ok(Json(EncryptedResponse::new(
        base64::engine::general_purpose::STANDARD.encode(encrypted_response),
    )))
}
