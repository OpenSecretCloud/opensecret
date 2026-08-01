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
use std::{future::Future, sync::Arc};
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

fn skips_encrypted_body<T: 'static>(method: &Method) -> bool {
    method == Method::GET
        || method == Method::DELETE
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<()>()
}

async fn forward_bodyless_request<T, SessionCheck, RunNext, RunNextFuture>(
    session_check: SessionCheck,
    session_id: Uuid,
    mut request: Request<Body>,
    run_next: RunNext,
) -> Result<Response, ApiError>
where
    T: 'static,
    SessionCheck: Future<Output = Result<(), ApiError>>,
    RunNext: FnOnce(Request<Body>) -> RunNextFuture,
    RunNextFuture: Future<Output = Response>,
{
    session_check.await?;

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<()>() {
        request.extensions_mut().insert(());
    }
    request.extensions_mut().insert(session_id);
    Ok(run_next(request).await)
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
    let session_id = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| Uuid::parse_str(v).ok())
        .ok_or(ApiError::BadRequest)?;

    // Skip body processing for GET, DELETE, or when T is ().
    if skips_encrypted_body::<T>(request.method()) {
        // Bodyless requests cannot prove possession by decrypting a payload.
        // Reject an unknown session before a handler can perform side effects.
        return forward_bodyless_request::<T, _, _, _>(
            state.require_session(&session_id),
            session_id,
            request,
            move |request| next.run(request),
        )
        .await;
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
    Ok(next.run(request).await)
}

pub async fn encrypt_response<T: Serialize>(
    state: &AppState,
    session_id: &Uuid,
    response: &T,
) -> Result<Json<EncryptedResponse<T>>, ApiError> {
    let response_json = serde_json::to_vec(response).map_err(|_| ApiError::InternalServerError)?;
    let encrypted_response = state
        .encrypt_session_data(session_id, &response_json)
        .await?;
    Ok(Json(EncryptedResponse::new(
        base64::engine::general_purpose::STANDARD.encode(encrypted_response),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use std::future::ready;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn classifies_every_bodyless_disjunct() {
        assert!(skips_encrypted_body::<serde_json::Value>(&Method::GET));
        assert!(skips_encrypted_body::<serde_json::Value>(&Method::DELETE));
        assert!(skips_encrypted_body::<()>(&Method::POST));
        assert!(!skips_encrypted_body::<serde_json::Value>(&Method::POST));
    }

    #[tokio::test]
    async fn failed_session_check_never_calls_next() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = calls.clone();

        let result = forward_bodyless_request::<(), _, _, _>(
            ready(Err(ApiError::BadRequest)),
            Uuid::new_v4(),
            Request::builder()
                .method(Method::POST)
                .body(Body::empty())
                .unwrap(),
            move |_| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                async { Response::new(Body::empty()) }
            },
        )
        .await;

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("unknown session reached next"),
        };
        let response = error.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), br#"{"status":400,"message":"Bad Request"}"#);
    }

    #[tokio::test]
    async fn successful_session_check_calls_next_once_with_extensions() {
        let session_id = Uuid::new_v4();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = calls.clone();

        forward_bodyless_request::<(), _, _, _>(
            ready(Ok(())),
            session_id,
            Request::builder()
                .method(Method::POST)
                .body(Body::empty())
                .unwrap(),
            move |request| {
                assert_eq!(request.extensions().get::<Uuid>(), Some(&session_id));
                assert!(request.extensions().get::<()>().is_some());
                observed_calls.fetch_add(1, Ordering::SeqCst);
                async { Response::new(Body::empty()) }
            },
        )
        .await
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
