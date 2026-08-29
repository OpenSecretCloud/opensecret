use axum::{
    body::{Body, Bytes, HttpBody},
    extract::State,
    http::{HeaderMap, Method, Request},
    middleware::Next,
    response::Response,
    Json,
};
use base64::Engine;
use http_body::{Frame, SizeHint};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use uuid::Uuid;

use crate::{ApiError, AppState};

const MAX_ENCRYPTED_BODY_BYTES: usize = 50 * 1024 * 1024; // 50MB

/// Identifies the transport context that must encrypt a handler response.
///
/// Keeping this distinct from the application's authenticated principal makes
/// the response transport explicit at the handler boundary. V1 contains only
/// the legacy session identifier; future protocol versions can carry their
/// exact response-encryption context without changing every handler again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportSession {
    kind: TransportSessionKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TransportSessionKind {
    V1 { session_id: Uuid },
}

impl TransportSession {
    fn v1(session_id: Uuid) -> Self {
        Self {
            kind: TransportSessionKind::V1 { session_id },
        }
    }

    pub(crate) async fn encrypt_response_bytes(
        &self,
        state: &AppState,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ApiError> {
        match &self.kind {
            TransportSessionKind::V1 { session_id } => {
                state.encrypt_session_data(session_id, plaintext).await
            }
        }
    }
}

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

fn parse_session_id(headers: &HeaderMap) -> Result<Uuid, ApiError> {
    headers
        .get("x-session-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or(ApiError::BadRequest)
}

async fn forward_bodyless_request<T, Resource, SessionCheck, RunNext, RunNextFuture>(
    session_check: SessionCheck,
    session_id: Uuid,
    mut request: Request<Body>,
    run_next: RunNext,
) -> Result<Response, ApiError>
where
    T: 'static,
    Resource: Send + Unpin + 'static,
    SessionCheck: Future<Output = Result<Resource, ApiError>>,
    RunNext: FnOnce(Request<Body>) -> RunNextFuture,
    RunNextFuture: Future<Output = Response>,
{
    let resource = session_check.await?;

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<()>() {
        request.extensions_mut().insert(());
    }
    request
        .extensions_mut()
        .insert(TransportSession::v1(session_id));
    let response = run_next(request).await;
    Ok(hold_resource_through_response_body(response, resource))
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
    let session_id = parse_session_id(&headers)?;

    // Skip body processing for GET, DELETE, or when T is ().
    if skips_encrypted_body::<T>(request.method()) {
        // Bodyless requests cannot prove possession by decrypting a payload.
        // Reject an unknown session before a handler can perform side effects.
        return forward_bodyless_request::<T, _, _, _, _>(
            async {
                let session_lease = state.acquire_request_session(&session_id).await?;
                state.touch_session(&session_id).await?;
                Ok(session_lease)
            },
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

    // Pin only after the request body has arrived. A slow or abandoned
    // upload therefore cannot retain a session indefinitely.
    let session_lease = state.acquire_request_session(&session_id).await?;
    let decrypted_data = state
        .decrypt_session_data(&session_id, &session_lease, &encrypted_request.encrypted)
        .await
        .map_err(|_| ApiError::BadRequest)?;

    let decrypted: T = serde_json::from_slice(&decrypted_data).map_err(|e| {
        tracing::error!("Failed to deserialize decrypted data: {:?}", e);
        ApiError::BadRequest
    })?;

    request.extensions_mut().insert(decrypted);
    request
        .extensions_mut()
        .insert(TransportSession::v1(session_id));
    let response = next.run(request).await;
    Ok(hold_resource_through_response_body(response, session_lease))
}

fn hold_resource_through_response_body<T>(mut response: Response, resource: T) -> Response
where
    T: Send + Unpin + 'static,
{
    let body = std::mem::replace(response.body_mut(), Body::empty());
    *response.body_mut() = Body::new(ResourceBody {
        inner: body,
        _resource: resource,
    });
    response
}

struct ResourceBody<T> {
    inner: Body,
    _resource: T,
}

impl<T> HttpBody for ResourceBody<T>
where
    T: Send + Unpin + 'static,
{
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.get_mut().inner).poll_frame(context)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

pub async fn encrypt_response<T: Serialize>(
    state: &AppState,
    transport_session: &TransportSession,
    response: &T,
) -> Result<Json<EncryptedResponse<T>>, ApiError> {
    let response_json = serde_json::to_vec(response).map_err(|_| ApiError::InternalServerError)?;
    let encrypted_response = transport_session
        .encrypt_response_bytes(state, &response_json)
        .await?;
    Ok(Json(EncryptedResponse::new(
        base64::engine::general_purpose::STANDARD.encode(encrypted_response),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lease_aware_cache::{InsertError, LeaseAwareTtlCache};
    use axum::{
        http::{HeaderMap, HeaderValue},
        response::IntoResponse,
    };
    use std::convert::Infallible;
    use std::future::{poll_fn, ready};
    use std::io;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn classifies_every_bodyless_disjunct() {
        assert!(skips_encrypted_body::<serde_json::Value>(&Method::GET));
        assert!(skips_encrypted_body::<serde_json::Value>(&Method::DELETE));
        assert!(skips_encrypted_body::<()>(&Method::POST));
        assert!(!skips_encrypted_body::<serde_json::Value>(&Method::POST));
    }

    #[test]
    fn missing_and_malformed_session_headers_are_generic_bad_requests() {
        for headers in [HeaderMap::new(), {
            let mut headers = HeaderMap::new();
            headers.insert("x-session-id", HeaderValue::from_static("not-a-uuid"));
            headers
        }] {
            let error = parse_session_id(&headers).unwrap_err();
            let response = error.into_response();
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            assert!(response.headers().get(crate::ERROR_CODE_HEADER).is_none());
        }
    }

    #[tokio::test]
    async fn failed_session_check_never_calls_next() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = calls.clone();

        let result = forward_bodyless_request::<(), _, _, _, _>(
            ready(Err::<(), ApiError>(ApiError::SessionNotFound)),
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
        assert_eq!(
            response.headers().get(crate::ERROR_CODE_HEADER).unwrap(),
            "session_not_found"
        );
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

        forward_bodyless_request::<(), _, _, _, _>(
            ready(Ok::<(), ApiError>(())),
            session_id,
            Request::builder()
                .method(Method::POST)
                .body(Body::empty())
                .unwrap(),
            move |request| {
                assert_eq!(
                    request.extensions().get::<TransportSession>(),
                    Some(&TransportSession::v1(session_id))
                );
                assert!(request.extensions().get::<()>().is_some());
                observed_calls.fetch_add(1, Ordering::SeqCst);
                async { Response::new(Body::empty()) }
            },
        )
        .await
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    struct DropSpy(Arc<AtomicUsize>);

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn unpolled_response_holds_resource_until_body_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("encrypted response"));
        let response = hold_resource_through_response_body(response, DropSpy(drops.clone()));

        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(response);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn streaming_response_lease_blocks_eviction_until_body_drop() {
        let mut cache =
            LeaseAwareTtlCache::new(NonZeroUsize::new(1).unwrap(), Duration::from_secs(60));
        cache.insert_evicting(1, "active stream").unwrap();
        let lease = cache.acquire(&1).unwrap();

        let stream = futures::stream::pending::<Result<Bytes, io::Error>>();
        let response = Response::new(Body::from_stream(stream));
        let response = hold_resource_through_response_body(response, lease);

        assert_eq!(
            cache.insert_evicting(2, "new session"),
            Err(InsertError::AllEntriesLeased)
        );
        drop(response);
        assert!(cache.insert_evicting(2, "new session").is_ok());
    }

    #[tokio::test]
    async fn wrapped_body_preserves_payload_and_exact_size_hint() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("encrypted response"));
        let response = hold_resource_through_response_body(response, DropSpy(drops.clone()));

        assert_eq!(response.body().size_hint().exact(), Some(18));
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"encrypted response");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    struct DataThenTrailersBody {
        next_frame: u8,
    }

    impl HttpBody for DataThenTrailersBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            let frame = match self.next_frame {
                0 => Some(Frame::data(Bytes::from_static(b"x"))),
                1 => {
                    let mut trailers = HeaderMap::new();
                    trailers.insert("x-test-trailer", HeaderValue::from_static("preserved"));
                    Some(Frame::trailers(trailers))
                }
                _ => None,
            };
            self.next_frame += 1;
            Poll::Ready(frame.map(Ok))
        }

        fn is_end_stream(&self) -> bool {
            self.next_frame > 1
        }

        fn size_hint(&self) -> SizeHint {
            SizeHint::with_exact(1)
        }
    }

    #[tokio::test]
    async fn wrapped_body_preserves_trailers_and_lease_through_end_of_stream() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::new(DataThenTrailersBody { next_frame: 0 }));
        let response = hold_resource_through_response_body(response, DropSpy(drops.clone()));
        let mut body = Box::pin(response.into_body());

        let data = poll_fn(|context| body.as_mut().poll_frame(context))
            .await
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert_eq!(&data[..], b"x");

        let trailers = poll_fn(|context| body.as_mut().poll_frame(context))
            .await
            .unwrap()
            .unwrap()
            .into_trailers()
            .unwrap();
        assert_eq!(trailers["x-test-trailer"], "preserved");
        assert!(poll_fn(|context| body.as_mut().poll_frame(context))
            .await
            .is_none());
        assert!(body.is_end_stream());
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(body);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    struct DataThenErrorBody {
        next_frame: u8,
    }

    impl HttpBody for DataThenErrorBody {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            let frame = match self.next_frame {
                0 => Some(Ok(Frame::data(Bytes::from_static(b"x")))),
                1 => Some(Err(io::Error::other("test body failure"))),
                _ => None,
            };
            self.next_frame += 1;
            Poll::Ready(frame)
        }
    }

    #[tokio::test]
    async fn wrapped_body_preserves_errors_and_resource_until_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::new(DataThenErrorBody { next_frame: 0 }));
        let response = hold_resource_through_response_body(response, DropSpy(drops.clone()));
        let mut body = Box::pin(response.into_body());

        let data = poll_fn(|context| body.as_mut().poll_frame(context))
            .await
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert_eq!(&data[..], b"x");

        let error = poll_fn(|context| body.as_mut().poll_frame(context))
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("test body failure"));
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(body);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
