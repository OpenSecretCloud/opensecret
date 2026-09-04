use std::{
    convert::Infallible,
    future::poll_fn,
    io,
    num::NonZeroUsize,
    pin::Pin,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use async_stream::try_stream;
use aws_nitro_enclaves_nsm_api::api::Request as NsmRequest;
use axum::{
    body::{to_bytes, Body, Bytes, HttpBody},
    extract::State,
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::Stream;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use tower::ServiceExt;
use x25519_dalek::{EphemeralSecret, PublicKey};

use crate::{
    web::{
        attestation_routes::generate_attestation_document, encryption_middleware::TransportSession,
    },
    ApiError, AppState,
};

use super::{
    crypto::{
        attestation_user_data, derive_server_session, CryptoError, HandshakeTranscript, SessionId,
        HANDSHAKE_CHALLENGE_BYTES, MIN_REQUEST_RECORD_BYTES, RECORD_TAG_BYTES,
        X25519_PUBLIC_KEY_BYTES,
    },
    envelope::{
        EnvelopeError, LogicalHeader, RequestEnvelope, RequestId, MAX_ENCODED_REQUEST_BYTES,
        REQUEST_ID_BYTES, VERSION,
    },
    framing::{
        frame_ciphertext, FramingError, ResponseRecord, ResponseStart, MAX_RESPONSE_CHUNK_BYTES,
    },
    session::{ReplayBudget, ReplayError, Session, SessionError, SessionStore, SessionStoreError},
};

const SESSION_LIFETIME: Duration = Duration::from_secs(60 * 60);
const SESSION_MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60);
const MAX_SESSION_REQUEST_BYTES: usize = 4 * 1024;
const MAX_SESSIONS: usize = 2_097_152;
const MAX_REPLAY_IDS_PER_SESSION: usize = 1_048_576;
const MAX_REPLAY_IDS_PROCESS_WIDE: usize = 16_777_216;
const MAX_REQUEST_RECORD_BYTES: usize =
    REQUEST_ID_BYTES + MAX_ENCODED_REQUEST_BYTES + RECORD_TAG_BYTES;

const SESSION_ID_HEADER: HeaderName = HeaderName::from_static("x-session-id");
const ROUTING_KEY_HEADER: HeaderName = HeaderName::from_static("x-opensecret-routing-key");
const OUTER_CONTENT_TYPE: HeaderValue = HeaderValue::from_static("application/octet-stream");

#[derive(Clone)]
pub(crate) struct TransportV2Gateway {
    sessions: Arc<SessionStore>,
    session_state: Arc<SessionGatewayState>,
    request_state: Arc<RequestGatewayState>,
}

struct SessionGatewayState {
    app_state: Arc<AppState>,
    sessions: Arc<SessionStore>,
    replay_budget: Arc<ReplayBudget>,
}

struct RequestGatewayState {
    application: Router<()>,
    sessions: Arc<SessionStore>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateSessionRequest {
    version: u8,
    challenge: String,
    client_public_key: String,
}

#[derive(Debug, Serialize)]
struct CreateSessionResponse {
    version: u8,
    session_id: String,
    attestation_document: String,
    expires_in_seconds: u64,
}

#[derive(Clone, Copy, Debug)]
enum OuterError {
    BadRequest,
    PayloadTooLarge,
    UnsupportedMediaType,
    Unavailable,
}

impl IntoResponse for OuterError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::BadRequest => StatusCode::BAD_REQUEST,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        };
        (status, "transport request rejected").into_response()
    }
}

impl TransportV2Gateway {
    pub(crate) fn new(app_state: Arc<AppState>, application: Router<()>) -> Self {
        let replay_budget = Arc::new(ReplayBudget::new(
            NonZeroUsize::new(MAX_REPLAY_IDS_PROCESS_WIDE)
                .expect("transport-v2 replay capacity must be non-zero"),
        ));
        let sessions = Arc::new(SessionStore::new(
            NonZeroUsize::new(MAX_SESSIONS)
                .expect("transport-v2 session capacity must be non-zero"),
        ));
        Self {
            sessions: Arc::clone(&sessions),
            session_state: Arc::new(SessionGatewayState {
                app_state,
                sessions: Arc::clone(&sessions),
                replay_budget,
            }),
            request_state: Arc::new(RequestGatewayState {
                application,
                sessions,
            }),
        }
    }

    pub(crate) fn router(&self) -> Router<()> {
        Router::new()
            .merge(
                Router::new()
                    .route("/v2/session", post(create_session))
                    .with_state(Arc::clone(&self.session_state)),
            )
            .merge(
                Router::new()
                    .route("/v2/request", post(dispatch_request))
                    .with_state(Arc::clone(&self.request_state)),
            )
    }

    pub(crate) async fn run_maintenance(&self) -> Infallible {
        let mut interval = tokio::time::interval(SESSION_MAINTENANCE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            match self.sessions.purge_expired(Instant::now()) {
                Ok(removed) if removed > 0 => {
                    tracing::debug!(removed, "purged expired transport-v2 sessions");
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::error!(?error, "failed to purge expired transport-v2 sessions");
                }
            }
        }
    }
}

async fn create_session(
    State(state): State<Arc<SessionGatewayState>>,
    request: Request<Body>,
) -> Result<Json<CreateSessionResponse>, OuterError> {
    require_content_type(request.headers(), "application/json")?;
    reject_forbidden_outer_headers(request.headers())?;
    if request.uri().query().is_some() {
        return Err(OuterError::BadRequest);
    }
    let routing_key = parse_outer_routing_key(request.headers())?;
    reject_declared_oversize(request.headers(), MAX_SESSION_REQUEST_BYTES)?;
    let body = read_bounded_body(request.into_body(), MAX_SESSION_REQUEST_BYTES).await?;
    let payload: CreateSessionRequest =
        serde_json::from_slice(&body).map_err(|_| OuterError::BadRequest)?;
    if payload.version != VERSION {
        return Err(OuterError::BadRequest);
    }

    let challenge = decode_canonical_fixed::<HANDSHAKE_CHALLENGE_BYTES>(&payload.challenge)?;
    if challenge != routing_key {
        return Err(OuterError::BadRequest);
    }
    let client_public_key =
        decode_canonical_fixed::<X25519_PUBLIC_KEY_BYTES>(&payload.client_public_key)?;

    let server_secret = EphemeralSecret::random_from_rng(OsRng);
    let server_public_key = PublicKey::from(&server_secret).to_bytes();
    let transcript = HandshakeTranscript::new(challenge, client_public_key, server_public_key);
    let secrets =
        derive_server_session(server_secret, &transcript).map_err(|_| OuterError::BadRequest)?;
    let session_id = secrets.session_id();

    let attestation_document = generate_attestation_document(
        Arc::clone(&state.app_state),
        NsmRequest::Attestation {
            user_data: Some(ByteBuf::from(attestation_user_data(&client_public_key))),
            public_key: Some(ByteBuf::from(server_public_key.to_vec())),
            nonce: Some(ByteBuf::from(challenge.to_vec())),
        },
    )
    .await
    .map_err(|_| OuterError::Unavailable)?;

    let session = Arc::new(
        Session::new(
            secrets,
            challenge,
            SESSION_LIFETIME,
            NonZeroUsize::new(MAX_REPLAY_IDS_PER_SESSION)
                .expect("transport-v2 per-session replay capacity must be non-zero"),
            Arc::clone(&state.replay_budget),
        )
        .map_err(|_| OuterError::Unavailable)?,
    );
    state
        .sessions
        .insert(session)
        .map_err(map_session_insert_error)?;

    Ok(Json(CreateSessionResponse {
        version: VERSION,
        session_id: session_id.to_string(),
        attestation_document: STANDARD.encode(attestation_document),
        expires_in_seconds: SESSION_LIFETIME.as_secs(),
    }))
}

async fn dispatch_request(
    State(state): State<Arc<RequestGatewayState>>,
    request: Request<Body>,
) -> Result<Response, OuterError> {
    require_content_type(request.headers(), "application/octet-stream")?;
    reject_forbidden_outer_headers(request.headers())?;
    if request.uri().query().is_some() {
        return Err(OuterError::BadRequest);
    }
    let session_id = parse_outer_session_id(request.headers())?;
    let routing_key = parse_outer_routing_key(request.headers())?;
    reject_declared_oversize(request.headers(), MAX_REQUEST_RECORD_BYTES)?;

    // Reject arbitrary session IDs before accepting a potentially large upload.
    // Drop this short-lived reference immediately so a slow or abandoned body
    // cannot pin the session. Admission still performs an authoritative lookup
    // after the body is collected, since the session may expire or disappear in
    // the meantime.
    drop(get_routed_session(
        &state.sessions,
        session_id,
        &routing_key,
    )?);
    let record = read_bounded_body(request.into_body(), MAX_REQUEST_RECORD_BYTES).await?;
    if record.len() < MIN_REQUEST_RECORD_BYTES {
        return Err(OuterError::BadRequest);
    }

    // The body arrives before the session is retained. A slow or abandoned
    // upload therefore cannot pin session state.
    let session = get_routed_session(&state.sessions, session_id, &routing_key)?;
    let (admitted, plaintext) = session
        .open_and_admit(&record)
        .map_err(map_admission_error)?;
    drop(record);

    let envelope = match RequestEnvelope::decode_owned(admitted.request_id(), plaintext) {
        Ok(envelope) => envelope,
        Err(_) => return encrypted_response(admitted, ApiError::BadRequest.into_response()),
    };
    let request =
        match build_application_request(envelope, admitted.session_id(), admitted.request_id()) {
            Ok(request) => request,
            Err(_) => return encrypted_response(admitted, ApiError::BadRequest.into_response()),
        };

    let response = state
        .application
        .clone()
        .oneshot(request)
        .await
        .expect("Axum Router has an infallible service error");
    encrypted_response(admitted, response)
}

fn build_application_request(
    envelope: RequestEnvelope,
    session_id: SessionId,
    request_id: RequestId,
) -> Result<Request<Body>, EnvelopeError> {
    let parts = envelope.into_parts();
    let method =
        Method::from_bytes(parts.method.as_bytes()).map_err(|_| EnvelopeError::InvalidMethod)?;
    let uri = Uri::from_str(&parts.target).map_err(|_| EnvelopeError::InvalidTarget)?;
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .body(parts.body.map_or_else(Body::empty, Body::from))
        .map_err(|_| EnvelopeError::InvalidTarget)?;
    for logical_header in parts.headers {
        let name = HeaderName::from_bytes(logical_header.name().as_bytes())
            .map_err(|_| EnvelopeError::InvalidHeaderName)?;
        let value = HeaderValue::from_str(logical_header.value())
            .map_err(|_| EnvelopeError::InvalidHeaderValue)?;
        request.headers_mut().append(name, value);
    }
    request
        .extensions_mut()
        .insert(TransportSession::v2(session_id));
    request.extensions_mut().insert(request_id);
    if let Some(credential) = parts.credential {
        request.extensions_mut().insert(credential);
    }
    if let Some(cache_namespace_root) = parts.cache_namespace_root {
        request.extensions_mut().insert(cache_namespace_root);
    }
    Ok(request)
}

fn encrypted_response(
    admitted: super::session::AdmittedRequest,
    mut response: Response,
) -> Result<Response, OuterError> {
    let response_start = ResponseStart::new(
        response.status().as_u16(),
        logical_response_headers(response.headers()),
    );
    let (start, mut body) = match response_start {
        Ok(start) => (start, std::mem::replace(response.body_mut(), Body::empty())),
        Err(_) => {
            // Once a request has authenticated and claimed replay state, every
            // representable application failure stays inside the encrypted
            // channel. An oversized or invalid internal response head therefore
            // becomes a minimal encrypted 500 rather than a plaintext outer 503.
            (
                ResponseStart::new(StatusCode::INTERNAL_SERVER_ERROR.as_u16(), Vec::new())
                    .expect("a headerless 500 is a valid transport response start"),
                Body::empty(),
            )
        }
    };
    let start = ResponseRecord::Start(start);
    let mut writer = admitted
        .begin_response()
        .map_err(|_| OuterError::Unavailable)?;
    let start = seal_framed(&mut writer, start).map_err(|_| OuterError::Unavailable)?;
    let stream = try_stream! {
        yield Bytes::from(start);
        while let Some(frame) = poll_fn(|context| Pin::new(&mut body).poll_frame(context)).await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(_) => {
                    let error = seal_framed(
                        &mut writer,
                        ResponseRecord::Error {
                            code: "application_body_failed".to_string(),
                        },
                    ).map_err(io_error)?;
                    yield Bytes::from(error);
                    return;
                }
            };
            match frame.into_data() {
                Ok(bytes) => {
                    for chunk in bytes.chunks(MAX_RESPONSE_CHUNK_BYTES) {
                        let encrypted = seal_framed(
                            &mut writer,
                            ResponseRecord::Chunk(Bytes::copy_from_slice(chunk)),
                        ).map_err(io_error)?;
                        yield Bytes::from(encrypted);
                    }
                }
                Err(frame) => {
                    let code = if frame.is_trailers() {
                        "application_trailers_unsupported"
                    } else {
                        "application_frame_unsupported"
                    };
                    let error = seal_framed(
                        &mut writer,
                        ResponseRecord::Error {
                            code: code.to_string(),
                        },
                    ).map_err(io_error)?;
                    yield Bytes::from(error);
                    return;
                }
            }
        }
        let end = seal_framed(&mut writer, ResponseRecord::End).map_err(io_error)?;
        yield Bytes::from(end);
    };

    let mut outer = Response::new(stream_body(stream));
    *outer.status_mut() = StatusCode::OK;
    outer
        .headers_mut()
        .insert(header::CONTENT_TYPE, OUTER_CONTENT_TYPE);
    outer
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    outer.headers_mut().insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    Ok(outer)
}

fn seal_framed(
    writer: &mut super::session::ResponseWriter,
    record: ResponseRecord,
) -> Result<Vec<u8>, SealFrameError> {
    let plaintext = record.encode()?;
    let ciphertext = writer.seal_next(&plaintext)?;
    Ok(frame_ciphertext(&ciphertext)?)
}

#[derive(Debug, thiserror::Error)]
enum SealFrameError {
    #[error(transparent)]
    Framing(#[from] FramingError),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
}

fn io_error(error: SealFrameError) -> io::Error {
    io::Error::other(error)
}

fn logical_response_headers(headers: &HeaderMap) -> Vec<LogicalHeader> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let value = value.to_str().ok()?;
            LogicalHeader::new(name.as_str().to_string(), value.to_string()).ok()
        })
        .collect()
}

fn require_content_type(headers: &HeaderMap, expected: &str) -> Result<(), OuterError> {
    let values = headers.get_all(header::CONTENT_TYPE);
    let mut values = values.iter();
    let Some(actual) = values.next() else {
        return Err(OuterError::UnsupportedMediaType);
    };
    if values.next().is_some() || actual.as_bytes() != expected.as_bytes() {
        return Err(OuterError::UnsupportedMediaType);
    }
    Ok(())
}

fn reject_forbidden_outer_headers(headers: &HeaderMap) -> Result<(), OuterError> {
    for name in [
        header::AUTHORIZATION,
        HeaderName::from_static("proxy-authorization"),
        header::COOKIE,
        header::CONTENT_ENCODING,
    ] {
        if headers.contains_key(name) {
            return Err(OuterError::BadRequest);
        }
    }
    Ok(())
}

fn reject_declared_oversize(headers: &HeaderMap, maximum: usize) -> Result<(), OuterError> {
    let values = headers.get_all(header::CONTENT_LENGTH);
    let mut values = values.iter();
    let Some(value) = values.next() else {
        return Ok(());
    };
    if values.next().is_some() {
        return Err(OuterError::BadRequest);
    }
    let declared = value
        .to_str()
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or(OuterError::BadRequest)?;
    if declared > maximum {
        return Err(OuterError::PayloadTooLarge);
    }
    Ok(())
}

async fn read_bounded_body(body: Body, maximum: usize) -> Result<Bytes, OuterError> {
    to_bytes(body, maximum)
        .await
        .map_err(|_| OuterError::PayloadTooLarge)
}

fn parse_outer_session_id(headers: &HeaderMap) -> Result<SessionId, OuterError> {
    let values = headers.get_all(&SESSION_ID_HEADER);
    let mut values = values.iter();
    let Some(value) = values.next() else {
        return Err(OuterError::BadRequest);
    };
    if values.next().is_some() {
        return Err(OuterError::BadRequest);
    }
    value
        .to_str()
        .ok()
        .and_then(|value| SessionId::from_str(value).ok())
        .ok_or(OuterError::BadRequest)
}

fn parse_outer_routing_key(
    headers: &HeaderMap,
) -> Result<[u8; HANDSHAKE_CHALLENGE_BYTES], OuterError> {
    let values = headers.get_all(&ROUTING_KEY_HEADER);
    let mut values = values.iter();
    let value = values.next().ok_or(OuterError::BadRequest)?;
    if values.next().is_some() {
        return Err(OuterError::BadRequest);
    }
    let value = value.to_str().map_err(|_| OuterError::BadRequest)?;
    decode_canonical_fixed(value)
}

fn get_routed_session(
    sessions: &SessionStore,
    session_id: SessionId,
    routing_key: &[u8; HANDSHAKE_CHALLENGE_BYTES],
) -> Result<Arc<Session>, OuterError> {
    let session = sessions.get(session_id).map_err(map_session_lookup_error)?;
    if !session.matches_routing_key(routing_key) {
        return Err(OuterError::BadRequest);
    }
    Ok(session)
}

fn decode_canonical_fixed<const N: usize>(encoded: &str) -> Result<[u8; N], OuterError> {
    let decoded = STANDARD
        .decode(encoded.as_bytes())
        .map_err(|_| OuterError::BadRequest)?;
    let bytes: [u8; N] = decoded.try_into().map_err(|_| OuterError::BadRequest)?;
    if STANDARD.encode(bytes) != encoded {
        return Err(OuterError::BadRequest);
    }
    Ok(bytes)
}

fn map_session_insert_error(error: SessionStoreError) -> OuterError {
    match error {
        SessionStoreError::Collision
        | SessionStoreError::Full
        | SessionStoreError::Unavailable
        | SessionStoreError::Expired
        | SessionStoreError::Missing => OuterError::Unavailable,
    }
}

fn map_session_lookup_error(error: SessionStoreError) -> OuterError {
    match error {
        SessionStoreError::Missing | SessionStoreError::Expired => OuterError::BadRequest,
        SessionStoreError::Collision | SessionStoreError::Full | SessionStoreError::Unavailable => {
            OuterError::Unavailable
        }
    }
}

fn map_admission_error(error: SessionError) -> OuterError {
    match error {
        SessionError::Replay(ReplayError::GlobalCapacity | ReplayError::Unavailable) => {
            OuterError::Unavailable
        }
        SessionError::Replay(ReplayError::Duplicate | ReplayError::SessionCapacity)
        | SessionError::Expired
        | SessionError::Crypto(_) => OuterError::BadRequest,
        SessionError::InvalidLifetime => OuterError::Unavailable,
    }
}

fn stream_body(stream: impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static) -> Body {
    Body::from_stream(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::any;
    use futures::stream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::provider_cache::CacheNamespaceRoot;
    use crate::transport_v2::{
        crypto::{derive_client_session, SessionSecrets},
        envelope::{Credential, CredentialKind},
    };

    struct TestSession {
        client: SessionSecrets,
        server: Arc<Session>,
        routing_key: [u8; HANDSHAKE_CHALLENGE_BYTES],
    }

    fn test_session(marker: u8) -> TestSession {
        let client_secret = EphemeralSecret::random_from_rng(OsRng);
        let client_public_key = PublicKey::from(&client_secret).to_bytes();
        let server_secret = EphemeralSecret::random_from_rng(OsRng);
        let server_public_key = PublicKey::from(&server_secret).to_bytes();
        let transcript =
            HandshakeTranscript::new([marker; 32], client_public_key, server_public_key);
        let client = derive_client_session(client_secret, &transcript).unwrap();
        let server_secrets = derive_server_session(server_secret, &transcript).unwrap();
        let replay_budget = Arc::new(ReplayBudget::new(NonZeroUsize::new(128).unwrap()));
        let routing_key = [marker; HANDSHAKE_CHALLENGE_BYTES];
        let server = Arc::new(
            Session::new(
                server_secrets,
                routing_key,
                Duration::from_secs(60),
                NonZeroUsize::new(64).unwrap(),
                replay_budget,
            )
            .unwrap(),
        );
        TestSession {
            client,
            server,
            routing_key,
        }
    }

    fn request_router(application: Router<()>, sessions: Arc<SessionStore>) -> Router<()> {
        Router::new()
            .route("/v2/request", post(dispatch_request))
            .with_state(Arc::new(RequestGatewayState {
                application,
                sessions,
            }))
    }

    fn seal_request(
        client: &SessionSecrets,
        request_id: RequestId,
        target: &str,
        body: Option<Vec<u8>>,
    ) -> Vec<u8> {
        let envelope = RequestEnvelope::new(
            request_id,
            Some(Credential::new(CredentialKind::Bearer, "v2-token".into()).unwrap()),
            Some(CacheNamespaceRoot::from_bytes([0x42; 32])),
            "POST".into(),
            target.into(),
            vec![LogicalHeader::new("x-logical".into(), "present".into()).unwrap()],
            body,
        )
        .unwrap();
        client
            .encrypt_request(request_id, &envelope.encode().unwrap())
            .unwrap()
    }

    fn outer_request(
        session_id: SessionId,
        routing_key: &[u8; HANDSHAKE_CHALLENGE_BYTES],
        record: Vec<u8>,
    ) -> Request<Body> {
        outer_request_with_body(session_id, routing_key, Body::from(record))
    }

    fn outer_request_with_body(
        session_id: SessionId,
        routing_key: &[u8; HANDSHAKE_CHALLENGE_BYTES],
        body: Body,
    ) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri("/v2/request")
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(&SESSION_ID_HEADER, session_id.to_string())
            .header(&ROUTING_KEY_HEADER, STANDARD.encode(routing_key))
            .body(body)
            .unwrap()
    }

    async fn decrypt_records(
        client: &SessionSecrets,
        request_id: RequestId,
        response: Response,
    ) -> Vec<ResponseRecord> {
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&OUTER_CONTENT_TYPE)
        );
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let mut remaining = bytes.as_ref();
        let mut sequence = 0_u64;
        let mut records = Vec::new();
        while !remaining.is_empty() {
            let length = u32::from_be_bytes(remaining[..4].try_into().unwrap()) as usize;
            let ciphertext = &remaining[4..4 + length];
            let plaintext = client
                .decrypt_response(request_id, sequence, ciphertext)
                .unwrap();
            records.push(ResponseRecord::decode(&plaintext).unwrap());
            remaining = &remaining[4 + length..];
            sequence += 1;
        }
        records
    }

    #[test]
    fn application_header_projection_preserves_repeated_values_in_order() {
        let test = test_session(8);
        let request_id = RequestId::from_bytes([0x81; 16]);
        let envelope = RequestEnvelope::new(
            request_id,
            None,
            None,
            "GET".into(),
            "/health-check".into(),
            vec![
                LogicalHeader::new("x-repeat".into(), "request-first".into()).unwrap(),
                LogicalHeader::new("x-repeat".into(), "request-second".into()).unwrap(),
            ],
            None,
        )
        .unwrap();
        let request = build_application_request(envelope, test.server.id(), request_id).unwrap();
        let request_values = request
            .headers()
            .get_all("x-repeat")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(request_values, ["request-first", "request-second"]);

        let mut response_headers = HeaderMap::new();
        response_headers.append("x-repeat", HeaderValue::from_static("response-first"));
        response_headers.append("x-repeat", HeaderValue::from_static("response-second"));
        let projected = logical_response_headers(&response_headers);
        let response_values = projected
            .iter()
            .filter(|header| header.name() == "x-repeat")
            .map(LogicalHeader::value)
            .collect::<Vec<_>>();
        assert_eq!(response_values, ["response-first", "response-second"]);
    }

    #[tokio::test]
    async fn missing_content_length_is_accepted_and_whole_request_is_projected() {
        let test = test_session(1);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let application = Router::new().route(
            "/echo",
            any(|request: Request<Body>| async move {
                assert_eq!(request.method(), Method::POST);
                assert_eq!(request.uri().query(), Some("answer=42"));
                assert_eq!(request.headers().get("x-logical").unwrap(), "present");
                assert!(request
                    .extensions()
                    .get::<TransportSession>()
                    .is_some_and(TransportSession::is_v2));
                assert_eq!(
                    request.extensions().get::<RequestId>(),
                    Some(&RequestId::from_bytes([0x11; 16]))
                );
                assert_eq!(
                    request.extensions().get::<Credential>().unwrap().value(),
                    "v2-token"
                );
                assert!(request.extensions().get::<CacheNamespaceRoot>().is_some());
                let body = to_bytes(request.into_body(), 64).await.unwrap();
                let mut response = (StatusCode::CREATED, body).into_response();
                response
                    .headers_mut()
                    .insert("x-echo", HeaderValue::from_static("yes"));
                response
            }),
        );
        let router = request_router(application, sessions);
        let request_id = RequestId::from_bytes([0x11; 16]);
        let request = outer_request(
            test.server.id(),
            &test.routing_key,
            seal_request(
                &test.client,
                request_id,
                "/echo?answer=42",
                Some(b"hello".to_vec()),
            ),
        );
        assert!(request.headers().get(header::CONTENT_LENGTH).is_none());

        let records = decrypt_records(
            &test.client,
            request_id,
            router.oneshot(request).await.unwrap(),
        )
        .await;
        assert!(matches!(
            &records[0],
            ResponseRecord::Start(start)
                if start.status() == 201
                    && start.headers().iter().any(|header| header.name() == "x-echo")
        ));
        assert!(matches!(&records[1], ResponseRecord::Chunk(body) if body.as_ref() == b"hello"));
        assert!(matches!(&records[2], ResponseRecord::End));
    }

    #[tokio::test]
    async fn outer_rejection_and_failed_authentication_do_not_claim_replay_state() {
        let test = test_session(2);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let application = Router::new().route(
            "/count",
            any({
                let dispatches = Arc::clone(&dispatches);
                move || {
                    let dispatches = Arc::clone(&dispatches);
                    async move {
                        dispatches.fetch_add(1, Ordering::SeqCst);
                        StatusCode::NO_CONTENT
                    }
                }
            }),
        );
        let router = request_router(application, sessions);
        let request_id = RequestId::from_bytes([0x22; 16]);
        let record = seal_request(&test.client, request_id, "/count", None);

        let mut forbidden = outer_request(test.server.id(), &test.routing_key, record.clone());
        forbidden.headers_mut().insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer stolen"),
        );
        assert_eq!(
            router.clone().oneshot(forbidden).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );

        let mut missing_routing_key =
            outer_request(test.server.id(), &test.routing_key, record.clone());
        missing_routing_key
            .headers_mut()
            .remove(&ROUTING_KEY_HEADER);
        assert_eq!(
            router
                .clone()
                .oneshot(missing_routing_key)
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );

        assert_eq!(
            router
                .clone()
                .oneshot(outer_request(
                    test.server.id(),
                    &[0x93; HANDSHAKE_CHALLENGE_BYTES],
                    record.clone(),
                ))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );

        let mut tampered = record.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            router
                .clone()
                .oneshot(outer_request(test.server.id(), &test.routing_key, tampered))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );

        let response = router
            .clone()
            .oneshot(outer_request(
                test.server.id(),
                &test.routing_key,
                record.clone(),
            ))
            .await
            .unwrap();
        let records = decrypt_records(&test.client, request_id, response).await;
        assert!(matches!(&records[0], ResponseRecord::Start(start) if start.status() == 204));
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        assert_eq!(
            router
                .oneshot(outer_request(test.server.id(), &test.routing_key, record))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unknown_session_is_rejected_without_reading_the_body() {
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        let router = request_router(Router::new(), sessions);
        let body = Body::from_stream(stream::once(async {
            panic!("an unknown session body must not be polled");
            #[allow(unreachable_code)]
            Ok::<_, io::Error>(Bytes::new())
        }));
        let response = router
            .oneshot(outer_request_with_body(
                SessionId::from_bytes([0x91; 16]),
                &[0x91; HANDSHAKE_CHALLENGE_BYTES],
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn session_lookup_after_upload_remains_authoritative() {
        let test = test_session(9);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let request_id = RequestId::from_bytes([0x92; 16]);
        let record = seal_request(&test.client, request_id, "/unused", None);
        let session_id = test.server.id();
        let body = Body::from_stream(stream::once({
            let sessions = Arc::clone(&sessions);
            async move {
                sessions.remove(session_id).unwrap();
                Ok::<_, io::Error>(Bytes::from(record))
            }
        }));
        let router = request_router(Router::new(), sessions);

        let response = router
            .oneshot(outer_request_with_body(session_id, &test.routing_key, body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn authenticated_malformed_envelope_gets_an_encrypted_error() {
        let test = test_session(3);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let router = request_router(Router::new(), sessions);
        let request_id = RequestId::from_bytes([0x33; 16]);
        let record = test
            .client
            .encrypt_request(request_id, b"not an envelope")
            .unwrap();

        let records = decrypt_records(
            &test.client,
            request_id,
            router
                .oneshot(outer_request(test.server.id(), &test.routing_key, record))
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(&records[0], ResponseRecord::Start(start) if start.status() == 400));
        assert!(matches!(records.last(), Some(ResponseRecord::End)));
    }

    #[tokio::test]
    async fn response_records_are_bound_to_the_admitted_session_and_request() {
        let first = test_session(4);
        let second = test_session(5);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(3).unwrap()));
        sessions.insert(Arc::clone(&first.server)).unwrap();
        sessions.insert(Arc::clone(&second.server)).unwrap();
        let router = request_router(
            Router::new().route("/ok", any(|| async { StatusCode::NO_CONTENT })),
            sessions,
        );
        let request_id = RequestId::from_bytes([0x44; 16]);
        let response = router
            .oneshot(outer_request(
                first.server.id(),
                &first.routing_key,
                seal_request(&first.client, request_id, "/ok", None),
            ))
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let length = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        let ciphertext = &bytes[4..4 + length];

        assert!(first
            .client
            .decrypt_response(request_id, 0, ciphertext)
            .is_ok());
        assert!(second
            .client
            .decrypt_response(request_id, 0, ciphertext)
            .is_err());
        assert!(first
            .client
            .decrypt_response(RequestId::from_bytes([0x45; 16]), 0, ciphertext)
            .is_err());
    }

    #[tokio::test]
    async fn streams_end_or_fail_with_an_authenticated_terminal_record() {
        let test = test_session(6);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let application = Router::new()
            .route(
                "/stream",
                any(|| async {
                    Response::new(Body::from_stream(stream::iter([
                        Ok::<_, io::Error>(Bytes::from_static(b"one")),
                        Ok(Bytes::from_static(b"two")),
                    ])))
                }),
            )
            .route(
                "/stream-error",
                any(|| async {
                    Response::new(Body::from_stream(stream::iter([
                        Ok::<_, io::Error>(Bytes::from_static(b"one")),
                        Err(io::Error::other("boom")),
                    ])))
                }),
            );
        let router = request_router(application, sessions);

        let success_id = RequestId::from_bytes([0x61; 16]);
        let success = decrypt_records(
            &test.client,
            success_id,
            router
                .clone()
                .oneshot(outer_request(
                    test.server.id(),
                    &test.routing_key,
                    seal_request(&test.client, success_id, "/stream", None),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(success.last(), Some(ResponseRecord::End)));
        assert!(success.iter().any(
            |record| matches!(record, ResponseRecord::Chunk(body) if body.as_ref() == b"one")
        ));

        let error_id = RequestId::from_bytes([0x62; 16]);
        let failure = decrypt_records(
            &test.client,
            error_id,
            router
                .oneshot(outer_request(
                    test.server.id(),
                    &test.routing_key,
                    seal_request(&test.client, error_id, "/stream-error", None),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(
            failure.last(),
            Some(ResponseRecord::Error { code }) if code == "application_body_failed"
        ));
        assert!(!failure
            .iter()
            .any(|record| matches!(record, ResponseRecord::End)));
    }

    #[tokio::test]
    async fn bounded_body_read_uses_actual_bytes_and_invalid_response_heads_stay_encrypted() {
        assert_eq!(
            read_bounded_body(Body::from(Bytes::from_static(b"four")), 4)
                .await
                .unwrap(),
            Bytes::from_static(b"four")
        );
        assert!(matches!(
            read_bounded_body(Body::from(Bytes::from_static(b"five!")), 4).await,
            Err(OuterError::PayloadTooLarge)
        ));

        let test = test_session(7);
        let sessions = Arc::new(SessionStore::new(NonZeroUsize::new(2).unwrap()));
        sessions.insert(Arc::clone(&test.server)).unwrap();
        let application = Router::new().route(
            "/too-many-headers",
            any(|| async {
                let mut response =
                    Response::new(Body::from(Bytes::from_static(b"must be discarded")));
                for index in 0..=32 {
                    response.headers_mut().insert(
                        HeaderName::from_str(&format!("x-test-{index}")).unwrap(),
                        HeaderValue::from_static("value"),
                    );
                }
                response
            }),
        );
        let router = request_router(application, sessions);
        let request_id = RequestId::from_bytes([0x71; 16]);
        let records = decrypt_records(
            &test.client,
            request_id,
            router
                .oneshot(outer_request(
                    test.server.id(),
                    &test.routing_key,
                    seal_request(&test.client, request_id, "/too-many-headers", None),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(&records[0], ResponseRecord::Start(start) if start.status() == 500));
        assert_eq!(records.len(), 2);
        assert!(matches!(&records[1], ResponseRecord::End));
    }
}
