//! Isolated HTTP gateway for encrypted transport v2.
//!
//! This layer deliberately exposes no application operations yet. It owns the
//! v2-only attestation and session caches, authenticates a complete encrypted
//! request envelope, and returns an authenticated logical 404 through the exact
//! session lease that decrypted the request. Later stack layers add explicit
//! authentication and application-operation projections without re-entering
//! the transport-v1 router.

use std::collections::hash_map::RandomState;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, Request, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;
use x25519_dalek::{EphemeralSecret, PublicKey};
use zeroize::Zeroize;

use crate::encrypt::CustomRng;
use crate::web::attestation_routes;
use crate::{AppState, AsyncRngWrapper};

use super::crypto::{
    encrypt_key_exchange_record, CryptoError, DirectionalKeys, HandshakePayload, SessionMaster,
};
use super::envelope::{
    EncodedBytes, EnvelopeLimits, HeaderField, RequestEnvelope, UnaryResponseEnvelope, Version2,
};
use super::session::{
    GlobalReplayBudget, SessionRecordError, V2SessionState, DEFAULT_ABSOLUTE_SESSION_LIFETIME,
    DEFAULT_GLOBAL_REPLAY_IDS,
};
use super::session_cache::{V2SessionCache, V2SessionInsertError, V2SessionLease};
use super::{MAX_LIVE_SESSIONS, MAX_PENDING_ATTESTATIONS};

const MAX_ATTESTATION_NONCE_BYTES: usize = 512;
const PENDING_ATTESTATION_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_KEY_EXCHANGE_BODY_BYTES: usize = 4 * 1024;
const MAX_OUTER_REQUEST_BODY_BYTES: usize = 50 * 1024 * 1024;
const REQUEST_WORKING_SET_BUDGET_BYTES: usize = 256 * 1024 * 1024;
const REQUEST_WORKING_SET_UNIT_BYTES: usize = 64 * 1024;
const REQUEST_WORKING_SET_MULTIPLIER: usize = 4;
const REQUEST_WORKING_SET_UNITS: usize =
    REQUEST_WORKING_SET_BUDGET_BYTES / REQUEST_WORKING_SET_UNIT_BYTES;
const SESSION_ID_HEADER: &str = "x-session-id";

type PendingAttestationKey = [u8; 32];

struct PendingAttestationEntry {
    secret: Option<EphemeralSecret>,
    expires_at: Instant,
}

/// V2-local one-shot cache whose secret bytes are wiped in their cache slot
/// before any removal and whose detached allocations drop after the mutex.
struct PendingAttestationCache {
    entries: clru::CLruCache<PendingAttestationKey, PendingAttestationEntry, RandomState>,
    ttl: Duration,
}

impl PendingAttestationCache {
    fn new(capacity: NonZeroUsize, ttl: Duration) -> Self {
        Self {
            entries: clru::CLruCache::with_memory(capacity, capacity.get()),
            ttl,
        }
    }

    fn insert_at(
        &mut self,
        key: PendingAttestationKey,
        secret: EphemeralSecret,
        now: Instant,
    ) -> RetiredPendingAttestations {
        let expires_at = now
            .checked_add(self.ttl)
            .expect("transport-v2 pending-attestation TTL must fit in Instant");
        let mut retired = RetiredPendingAttestations::empty();

        if self.entries.peek(&key).is_some() {
            self.entries
                .peek_mut(&key)
                .expect("observed pending-attestation entry must remain present")
                .secret
                .as_mut()
                .expect("cached pending-attestation entry must retain its secret")
                .zeroize();
            retired.entries.push(
                self.entries
                    .pop(&key)
                    .expect("observed pending-attestation entry must remain present"),
            );
        } else if self.entries.is_full() {
            self.entries
                .back_mut()
                .expect("a full pending-attestation cache must have an oldest entry")
                .1
                .secret
                .as_mut()
                .expect("cached pending-attestation entry must retain its secret")
                .zeroize();
            let (_, evicted) = self
                .entries
                .pop_back()
                .expect("a full pending-attestation cache must have an oldest entry");
            retired.entries.push(evicted);
        }

        let previous = self.entries.put(
            key,
            PendingAttestationEntry {
                secret: Some(secret),
                expires_at,
            },
        );
        debug_assert!(previous.is_none());
        retired
    }

    fn take_live_at(
        &mut self,
        key: &PendingAttestationKey,
        now: Instant,
    ) -> (Option<EphemeralSecret>, RetiredPendingAttestations) {
        let Some(expires_at) = self.entries.peek(key).map(|entry| entry.expires_at) else {
            return (None, RetiredPendingAttestations::empty());
        };

        let secret = if expires_at > now {
            // Take the secret in-place before removing the cache entry. The
            // preallocated CLru slot therefore contains `None`, not stale key
            // bytes, when its dense backing storage is reused.
            self.entries
                .peek_mut(key)
                .expect("observed pending-attestation entry must remain present")
                .secret
                .take()
        } else {
            self.entries
                .peek_mut(key)
                .expect("observed pending-attestation entry must remain present")
                .secret
                .as_mut()
                .expect("cached pending-attestation entry must retain its secret")
                .zeroize();
            None
        };

        let removed = self
            .entries
            .pop(key)
            .expect("observed pending-attestation entry must remain present");
        (
            secret,
            RetiredPendingAttestations {
                entries: vec![removed],
            },
        )
    }

    fn cleanup_at(&mut self, now: Instant) -> RetiredPendingAttestations {
        let expired_count = self
            .entries
            .iter()
            .rev()
            .take_while(|(_, entry)| entry.expires_at <= now)
            .count();
        let mut retired = RetiredPendingAttestations {
            entries: Vec::with_capacity(expired_count),
        };
        for _ in 0..expired_count {
            self.entries
                .back_mut()
                .expect("counted pending-attestation entry must remain present")
                .1
                .secret
                .as_mut()
                .expect("cached pending-attestation entry must retain its secret")
                .zeroize();
            let (_, entry) = self
                .entries
                .pop_back()
                .expect("counted pending-attestation entry must remain present");
            retired.entries.push(entry);
        }
        retired
    }
}

#[must_use = "drop retired pending attestations only after releasing the cache lock"]
struct RetiredPendingAttestations {
    entries: Vec<PendingAttestationEntry>,
}

impl RetiredPendingAttestations {
    fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn removed_count(&self) -> usize {
        self.entries.len()
    }
}

/// V2-only pending handshakes, live sessions, and replay accounting.
///
/// None of these resources are shared with transport v1. Values detached from
/// the session cache are kept alive until its mutex is released so key
/// zeroization and replay-registry destruction never occur under that lock.
pub(crate) struct TransportV2State {
    pending_attestations: Mutex<PendingAttestationCache>,
    sessions: Mutex<V2SessionCache>,
    global_replay_budget: Arc<GlobalReplayBudget>,
    request_working_set: Arc<Semaphore>,
}

impl TransportV2State {
    pub(crate) fn new() -> Self {
        let pending_capacity = NonZeroUsize::new(MAX_PENDING_ATTESTATIONS)
            .expect("transport-v2 pending-attestation capacity must be nonzero");
        let session_capacity = NonZeroUsize::new(MAX_LIVE_SESSIONS)
            .expect("transport-v2 live-session capacity must be nonzero");

        Self {
            pending_attestations: Mutex::new(PendingAttestationCache::new(
                pending_capacity,
                PENDING_ATTESTATION_TTL,
            )),
            sessions: Mutex::new(V2SessionCache::new(session_capacity)),
            global_replay_budget: Arc::new(GlobalReplayBudget::new(DEFAULT_GLOBAL_REPLAY_IDS)),
            request_working_set: Arc::new(Semaphore::new(REQUEST_WORKING_SET_UNITS)),
        }
    }

    async fn create_pending_attestation(&self, nonce: &str) -> Result<PublicKey, GatewayError> {
        validate_attestation_nonce(nonce)?;

        // Match the existing enclave randomness source while keeping ownership
        // of the resulting secret wholly inside the v2 cache.
        let mut rng = AsyncRngWrapper::new(CustomRng::new());
        let secret = EphemeralSecret::random_from_rng(&mut rng);
        let public_key = PublicKey::from(&secret);

        let retired = {
            let mut pending = self.pending_attestations.lock().await;
            pending.insert_at(attestation_nonce_key(nonce), secret, Instant::now())
        };
        drop(retired);
        Ok(public_key)
    }

    async fn take_pending_attestation(&self, nonce: &str) -> Result<EphemeralSecret, GatewayError> {
        validate_attestation_nonce(nonce)?;
        let (secret, retired) = {
            let mut pending = self.pending_attestations.lock().await;
            pending.take_live_at(&attestation_nonce_key(nonce), Instant::now())
        };
        drop(retired);
        secret.ok_or(GatewayError::InvalidRequest)
    }

    async fn insert_session(&self, session: Arc<V2SessionState>) -> Result<(), GatewayError> {
        let insertion = {
            let mut sessions = self.sessions.lock().await;
            sessions.insert(session)
        };

        match insertion {
            Ok(inserted) => {
                // The detached Arc remains alive until after the mutex guard is
                // gone, then zeroizes on its final drop here.
                let retired = inserted.into_retired();
                drop(retired);
                Ok(())
            }
            Err(rejected) => {
                let reason = rejected.reason();
                // Likewise, destroy a rejected session only outside the cache
                // lock. UUID collision and full leased capacity are both
                // terminal for this one key exchange.
                drop(rejected.into_session());
                match reason {
                    V2SessionInsertError::DuplicateSession => Err(GatewayError::Internal),
                    V2SessionInsertError::AllSessionsLeased => Err(GatewayError::Unavailable),
                }
            }
        }
    }

    async fn acquire_session(
        &self,
        session_id: &Uuid,
        now: Instant,
    ) -> Result<V2SessionLease, GatewayError> {
        self.sessions
            .lock()
            .await
            .acquire_at(session_id, now)
            .ok_or(GatewayError::InvalidRequest)
    }

    /// Refreshes LRU position only after a future application route passes its
    /// AEAD, structural, route-policy, response-capacity, and replay gates.
    pub(crate) async fn mark_admitted(&self, lease: &V2SessionLease) -> bool {
        self.sessions.lock().await.mark_admitted(lease)
    }

    fn reserve_request_working_set(
        &self,
        headers: &HeaderMap,
    ) -> Result<OwnedSemaphorePermit, GatewayError> {
        let units = request_working_set_units(headers)?;
        // Fail closed instead of queueing. Tokio's fair multi-permit waiters
        // can otherwise let queued maximum-size requests head-of-line block
        // later small requests even while capacity for the latter is free.
        Arc::clone(&self.request_working_set)
            .try_acquire_many_owned(units)
            .map_err(|_| GatewayError::Unavailable)
    }

    /// Removes expired v2 pending handshakes and terminal v2 sessions.
    ///
    /// Both retirement collections outlive their respective lock guards. The
    /// returned count includes both resource types because the maintenance loop
    /// needs only aggregate observability for this isolated cache owner.
    pub(crate) async fn cleanup_expired_at(&self, now: Instant) -> usize {
        let retired_pending = {
            let mut pending = self.pending_attestations.lock().await;
            pending.cleanup_at(now)
        };
        let pending_count = retired_pending.removed_count();
        drop(retired_pending);

        let retired_sessions = {
            let mut sessions = self.sessions.lock().await;
            sessions.cleanup_at(now)
        };
        let session_count = retired_sessions.removed_count();
        drop(retired_sessions);

        pending_count + session_count
    }

    async fn process_encrypted_request(
        &self,
        session_id: Uuid,
        encrypted: &[u8],
        now: Instant,
    ) -> Result<EncryptedOuterResponse, GatewayError> {
        let lease = self.acquire_session(&session_id, now).await?;
        let plaintext = lease
            .state()
            .decrypt_request_record(encrypted)
            .map_err(GatewayError::from_session_record)?;
        let envelope = RequestEnvelope::from_json_slice(&plaintext, &EnvelopeLimits::default())
            .map_err(|_| GatewayError::InvalidRequest)?;
        drop(plaintext);
        let request_id = envelope.request_id;
        drop(envelope);

        // This gateway stack intentionally exposes no application operations.
        // Route validation therefore fails before the replay gate and before
        // LRU promotion. The recovered request ID still permits an authenticated
        // logical error through the exact lease selected above.
        encrypt_unsupported_route(&lease, request_id)
    }
}

impl Default for TransportV2State {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/v2/attestation/:nonce", get(get_attestation))
        .route("/v2/key_exchange", post(key_exchange))
        .route("/v2/request", post(encrypted_request))
        .with_state(app_state)
}

async fn get_attestation(
    State(app_state): State<Arc<AppState>>,
    Path(nonce): Path<String>,
    request: Request<Body>,
) -> Response {
    if request.uri().query().is_some()
        || reject_forbidden_outer_headers(request.headers(), true).is_err()
    {
        return GatewayError::InvalidRequest.into_response();
    }

    let public_key = match app_state
        .transport_v2_state
        .create_pending_attestation(&nonce)
        .await
    {
        Ok(public_key) => public_key,
        Err(error) => return error.into_response(),
    };

    match attestation_routes::attestation_document_for_public_key(
        Arc::clone(&app_state),
        nonce,
        public_key,
    )
    .await
    {
        Ok(response) => response.into_response(),
        Err(error) => error.into_response(),
    }
}

async fn key_exchange(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
) -> Result<Json<KeyExchangeResponse>, GatewayError> {
    validate_fixed_outer_request(request.uri(), request.headers(), true)?;
    validate_json_content_type(request.headers())?;
    let body = read_bounded_body(request.into_body(), MAX_KEY_EXCHANGE_BODY_BYTES).await?;
    let payload = parse_key_exchange_request(&body, MAX_KEY_EXCHANGE_BODY_BYTES)?;
    validate_attestation_nonce(&payload.nonce)?;

    // Do not consume the one-shot secret until every attacker-controlled outer
    // field and key encoding has passed strict validation.
    let client_public_key = parse_client_public_key(&payload.client_public_key)?;
    let ephemeral_secret = app_state
        .transport_v2_state
        .take_pending_attestation(&payload.nonce)
        .await?;
    let prepared = prepare_key_exchange(
        ephemeral_secret,
        &client_public_key,
        Instant::now(),
        SystemTime::now(),
        Arc::clone(&app_state.transport_v2_state.global_replay_budget),
    )?;

    app_state
        .transport_v2_state
        .insert_session(prepared.session)
        .await?;
    Ok(Json(prepared.response))
}

async fn encrypted_request(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
) -> Result<Json<EncryptedOuterResponse>, GatewayError> {
    let session_id = parse_request_session_id(request.uri(), request.headers())?;
    validate_json_content_type(request.headers())?;
    let working_set_permit = app_state
        .transport_v2_state
        .reserve_request_working_set(request.headers())?;

    // Fully bound the upload before acquiring a session lease. A slow or
    // oversized outer request therefore cannot pin secret-bearing session
    // state or prevent cache eviction.
    let body = read_bounded_body(request.into_body(), MAX_OUTER_REQUEST_BODY_BYTES).await?;
    let outer = parse_encrypted_outer_request(&body, MAX_OUTER_REQUEST_BODY_BYTES)?;
    drop(body);
    let encrypted = outer.encrypted.into_bytes();

    let response = app_state
        .transport_v2_state
        .process_encrypted_request(session_id, &encrypted, Instant::now())
        .await?;
    drop(encrypted);
    drop(working_set_permit);
    Ok(Json(response))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyExchangeRequest {
    nonce: String,
    client_public_key: EncodedBytes,
}

#[derive(Debug, Serialize)]
struct KeyExchangeResponse {
    session_id: Uuid,
    encrypted_session_key: EncodedBytes,
}

struct PreparedKeyExchange {
    session: Arc<V2SessionState>,
    response: KeyExchangeResponse,
}

fn prepare_key_exchange(
    ephemeral_secret: EphemeralSecret,
    client_public_key: &PublicKey,
    monotonic_now: Instant,
    wall_now: SystemTime,
    global_replay_budget: Arc<GlobalReplayBudget>,
) -> Result<PreparedKeyExchange, GatewayError> {
    let shared_secret = ephemeral_secret.diffie_hellman(client_public_key);
    if !shared_secret.was_contributory() {
        return Err(GatewayError::InvalidRequest);
    }

    let absolute_expires_at = monotonic_now
        .checked_add(DEFAULT_ABSOLUTE_SESSION_LIFETIME)
        .ok_or(GatewayError::Internal)?;
    let expires_at_unix_seconds = wall_now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| GatewayError::Internal)?
        .as_secs()
        .checked_add(DEFAULT_ABSOLUTE_SESSION_LIFETIME.as_secs())
        .ok_or(GatewayError::Internal)?;

    let session_id = Uuid::new_v4();
    let session_master = SessionMaster::random().map_err(GatewayError::from_crypto)?;
    let directional_keys =
        DirectionalKeys::derive(&session_master).map_err(GatewayError::from_crypto)?;
    let handshake_payload =
        HandshakePayload::new(&session_id, &session_master, expires_at_unix_seconds);
    let encrypted_session_key =
        encrypt_key_exchange_record(shared_secret.as_bytes(), &handshake_payload)
            .map_err(GatewayError::from_crypto)?;

    // Build the complete encrypted response record before making the session
    // reachable from the cache.
    let response = KeyExchangeResponse {
        session_id,
        encrypted_session_key: EncodedBytes::from_bytes(encrypted_session_key),
    };
    let session = Arc::new(V2SessionState::new(
        session_id,
        directional_keys,
        absolute_expires_at,
        global_replay_budget,
    ));

    Ok(PreparedKeyExchange { session, response })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EncryptedOuterRequest {
    encrypted: EncodedBytes,
}

#[derive(Debug, Serialize)]
struct EncryptedOuterResponse {
    encrypted: EncodedBytes,
}

#[derive(Serialize)]
struct LogicalErrorBody<'a> {
    error: LogicalError<'a>,
}

#[derive(Serialize)]
struct LogicalError<'a> {
    code: &'a str,
    message: &'a str,
}

fn encrypt_unsupported_route(
    lease: &V2SessionLease,
    request_id: super::envelope::RequestId,
) -> Result<EncryptedOuterResponse, GatewayError> {
    let body = serde_json::to_vec(&LogicalErrorBody {
        error: LogicalError {
            code: "not_found",
            message: "Not found",
        },
    })
    .map_err(|_| GatewayError::Internal)?;
    let response = UnaryResponseEnvelope {
        version: Version2,
        request_id,
        status: StatusCode::NOT_FOUND.as_u16(),
        headers: vec![HeaderField {
            name: "content-type".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
        }],
        body_base64: Some(EncodedBytes::from_bytes(body)),
    };
    response
        .validate(&EnvelopeLimits::default())
        .map_err(|_| GatewayError::Internal)?;
    let plaintext = serde_json::to_vec(&response).map_err(|_| GatewayError::Internal)?;

    let mut reservation = lease
        .state()
        .begin_unary_response()
        .map_err(GatewayError::from_session_record)?;
    let encrypted = lease
        .state()
        .encrypt_unary_response_record(&mut reservation, &request_id, &plaintext)
        .map_err(GatewayError::from_session_record)?;

    Ok(EncryptedOuterResponse {
        encrypted: EncodedBytes::from_bytes(encrypted),
    })
}

fn parse_key_exchange_request(
    body: &[u8],
    limit: usize,
) -> Result<KeyExchangeRequest, GatewayError> {
    if body.len() > limit {
        return Err(GatewayError::PayloadTooLarge);
    }
    serde_json::from_slice(body).map_err(|_| GatewayError::InvalidRequest)
}

fn parse_encrypted_outer_request(
    body: &[u8],
    limit: usize,
) -> Result<EncryptedOuterRequest, GatewayError> {
    if body.len() > limit {
        return Err(GatewayError::PayloadTooLarge);
    }
    serde_json::from_slice(body).map_err(|_| GatewayError::InvalidRequest)
}

fn parse_client_public_key(encoded: &EncodedBytes) -> Result<PublicKey, GatewayError> {
    let bytes: [u8; 32] = encoded
        .as_slice()
        .try_into()
        .map_err(|_| GatewayError::InvalidRequest)?;
    Ok(PublicKey::from(bytes))
}

fn parse_request_session_id(uri: &Uri, headers: &HeaderMap) -> Result<Uuid, GatewayError> {
    validate_fixed_outer_request(uri, headers, false)?;

    let mut values = headers.get_all(SESSION_ID_HEADER).iter();
    let value = values.next().ok_or(GatewayError::InvalidRequest)?;
    if values.next().is_some() {
        return Err(GatewayError::InvalidRequest);
    }
    let value = value.to_str().map_err(|_| GatewayError::InvalidRequest)?;
    let session_id = Uuid::parse_str(value).map_err(|_| GatewayError::InvalidRequest)?;
    if session_id.hyphenated().to_string() != value {
        return Err(GatewayError::InvalidRequest);
    }
    Ok(session_id)
}

fn validate_fixed_outer_request(
    uri: &Uri,
    headers: &HeaderMap,
    forbid_session_id: bool,
) -> Result<(), GatewayError> {
    if uri.query().is_some() {
        return Err(GatewayError::InvalidRequest);
    }
    reject_forbidden_outer_headers(headers, forbid_session_id)
}

fn reject_forbidden_outer_headers(
    headers: &HeaderMap,
    forbid_session_id: bool,
) -> Result<(), GatewayError> {
    for name in [
        header::AUTHORIZATION.as_str(),
        "proxy-authorization",
        header::COOKIE.as_str(),
        header::CONTENT_ENCODING.as_str(),
    ] {
        if headers.contains_key(name) {
            return Err(GatewayError::InvalidRequest);
        }
    }
    if forbid_session_id && headers.contains_key(SESSION_ID_HEADER) {
        return Err(GatewayError::InvalidRequest);
    }
    Ok(())
}

fn validate_json_content_type(headers: &HeaderMap) -> Result<(), GatewayError> {
    let mut values = headers.get_all(header::CONTENT_TYPE).iter();
    let value = values.next().ok_or(GatewayError::InvalidRequest)?;
    if values.next().is_some() {
        return Err(GatewayError::InvalidRequest);
    }
    let value = value.to_str().map_err(|_| GatewayError::InvalidRequest)?;
    let mut parts = value.split(';');
    let media_type = parts.next().unwrap_or_default().trim();
    if !media_type.eq_ignore_ascii_case("application/json") {
        return Err(GatewayError::InvalidRequest);
    }

    let mut saw_charset = false;
    for parameter in parts {
        let parameter = parameter.trim();
        let Some((name, value)) = parameter.split_once('=') else {
            return Err(GatewayError::InvalidRequest);
        };
        if saw_charset
            || !name.trim().eq_ignore_ascii_case("charset")
            || !value.trim().eq_ignore_ascii_case("utf-8")
        {
            return Err(GatewayError::InvalidRequest);
        }
        saw_charset = true;
    }
    Ok(())
}

fn request_working_set_units(headers: &HeaderMap) -> Result<u32, GatewayError> {
    let mut content_lengths = headers.get_all(header::CONTENT_LENGTH).iter();
    let declared_bytes = match content_lengths.next() {
        Some(value) => {
            if content_lengths.next().is_some() {
                return Err(GatewayError::InvalidRequest);
            }
            let value = value.to_str().map_err(|_| GatewayError::InvalidRequest)?;
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(GatewayError::InvalidRequest);
            }
            value
                .parse::<usize>()
                .map_err(|_| GatewayError::InvalidRequest)?
        }
        // With no trustworthy declared size, reserve the worst permitted
        // working set before reading one byte from the upload.
        None => MAX_OUTER_REQUEST_BODY_BYTES,
    };

    if declared_bytes > MAX_OUTER_REQUEST_BODY_BYTES {
        return Err(GatewayError::PayloadTooLarge);
    }
    let accounted_bytes = declared_bytes
        .checked_mul(REQUEST_WORKING_SET_MULTIPLIER)
        .ok_or(GatewayError::PayloadTooLarge)?;
    let units = accounted_bytes
        .checked_add(REQUEST_WORKING_SET_UNIT_BYTES - 1)
        .ok_or(GatewayError::PayloadTooLarge)?
        / REQUEST_WORKING_SET_UNIT_BYTES;
    let units = units.max(1);
    u32::try_from(units).map_err(|_| GatewayError::PayloadTooLarge)
}

async fn read_bounded_body(body: Body, limit: usize) -> Result<axum::body::Bytes, GatewayError> {
    to_bytes(body, limit)
        .await
        .map_err(|_| GatewayError::PayloadTooLarge)
}

fn validate_attestation_nonce(nonce: &str) -> Result<(), GatewayError> {
    if nonce.len() > MAX_ATTESTATION_NONCE_BYTES {
        Err(GatewayError::InvalidRequest)
    } else {
        Ok(())
    }
}

fn attestation_nonce_key(nonce: &str) -> PendingAttestationKey {
    use sha2::{Digest, Sha256};
    Sha256::digest(nonce.as_bytes()).into()
}

#[derive(Debug, Clone, Copy)]
enum GatewayError {
    InvalidRequest,
    PayloadTooLarge,
    Unavailable,
    Internal,
}

impl GatewayError {
    fn from_crypto(error: CryptoError) -> Self {
        match error {
            CryptoError::NonContributorySharedSecret
            | CryptoError::DecryptionFailed
            | CryptoError::RecordTooShort
            | CryptoError::InvalidBase64
            | CryptoError::NonCanonicalBase64 => Self::InvalidRequest,
            CryptoError::RandomnessUnavailable
            | CryptoError::KeyDerivationFailed
            | CryptoError::EncryptionFailed => Self::Internal,
        }
    }

    fn from_session_record(error: SessionRecordError) -> Self {
        match error {
            SessionRecordError::Crypto(error) => Self::from_crypto(error),
            SessionRecordError::RequestRecordsExhausted
            | SessionRecordError::ResponseRecordsExhausted => Self::Unavailable,
            SessionRecordError::ResponseReservationMismatch
            | SessionRecordError::ResponseReservationConsumed
            | SessionRecordError::StreamNotStarted
            | SessionRecordError::StreamAlreadyStarted
            | SessionRecordError::StreamClosed
            | SessionRecordError::InvalidStreamSequence => Self::Internal,
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid transport request"),
            Self::PayloadTooLarge => (StatusCode::PAYLOAD_TOO_LARGE, "transport request too large"),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "transport unavailable"),
            Self::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "transport failure"),
        };
        (
            status,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            message,
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport_v2::crypto::{decrypt_key_exchange_record, SessionMaster};
    use crate::transport_v2::envelope::{LogicalMethod, LogicalRequest, RequestId, ResponseMode};

    fn test_session(
        session_id: Uuid,
        master_bytes: [u8; 32],
        expires_at: Instant,
        global_budget: Arc<GlobalReplayBudget>,
    ) -> Arc<V2SessionState> {
        let master = SessionMaster::from_bytes(master_bytes);
        let keys = DirectionalKeys::derive(&master).expect("derive test keys");
        Arc::new(V2SessionState::new(
            session_id,
            keys,
            expires_at,
            global_budget,
        ))
    }

    fn request_envelope(request_id: RequestId) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Unary,
            credential: None,
            request: LogicalRequest {
                method: LogicalMethod::Get,
                path: "/v1/protected/private_key".to_owned(),
                query: None,
                headers: Vec::new(),
                body_base64: None,
            },
        }
    }

    #[test]
    fn key_exchange_json_is_strict_bounded_and_canonical() {
        let valid = br#"{"nonce":"fresh","client_public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}"#;
        let parsed = parse_key_exchange_request(valid, valid.len()).expect("valid request");
        assert_eq!(parsed.client_public_key.len(), 32);

        let duplicate = br#"{"nonce":"fresh","nonce":"again","client_public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}"#;
        assert!(parse_key_exchange_request(duplicate, duplicate.len()).is_err());
        let unknown = br#"{"nonce":"fresh","client_public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","extra":true}"#;
        assert!(parse_key_exchange_request(unknown, unknown.len()).is_err());
        let unpadded = br#"{"nonce":"fresh","client_public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}"#;
        assert!(parse_key_exchange_request(unpadded, unpadded.len()).is_err());
        assert!(matches!(
            parse_key_exchange_request(valid, valid.len() - 1),
            Err(GatewayError::PayloadTooLarge)
        ));
    }

    #[test]
    fn public_key_requires_exactly_32_decoded_bytes() {
        let short = EncodedBytes::from_bytes(vec![0; 31]);
        let long = EncodedBytes::from_bytes(vec![0; 33]);
        assert!(parse_client_public_key(&short).is_err());
        assert!(parse_client_public_key(&long).is_err());
        assert!(parse_client_public_key(&EncodedBytes::from_bytes(vec![0; 32])).is_ok());
    }

    #[test]
    fn key_exchange_rejects_non_contributory_public_key() {
        let server_secret = EphemeralSecret::random_from_rng(rand_core::OsRng);
        let low_order = PublicKey::from([0_u8; 32]);
        let result = prepare_key_exchange(
            server_secret,
            &low_order,
            Instant::now(),
            UNIX_EPOCH + Duration::from_secs(1_000),
            Arc::new(GlobalReplayBudget::new(DEFAULT_GLOBAL_REPLAY_IDS)),
        );
        assert!(matches!(result, Err(GatewayError::InvalidRequest)));
    }

    #[test]
    fn key_exchange_payload_and_session_keys_agree() {
        let server_secret = EphemeralSecret::random_from_rng(rand_core::OsRng);
        let server_public = PublicKey::from(&server_secret);
        let client_secret = EphemeralSecret::random_from_rng(rand_core::OsRng);
        let client_public = PublicKey::from(&client_secret);
        let shared_secret = client_secret.diffie_hellman(&server_public);
        let wall_seconds = 1_000_u64;

        let prepared = prepare_key_exchange(
            server_secret,
            &client_public,
            Instant::now(),
            UNIX_EPOCH + Duration::from_secs(wall_seconds),
            Arc::new(GlobalReplayBudget::new(DEFAULT_GLOBAL_REPLAY_IDS)),
        )
        .expect("honest exchange");
        let plaintext = decrypt_key_exchange_record(
            shared_secret.as_bytes(),
            prepared.response.encrypted_session_key.as_slice(),
        )
        .expect("decrypt handshake");

        assert_eq!(plaintext.len(), 57);
        assert_eq!(plaintext[0], 2);
        assert_eq!(&plaintext[1..17], prepared.response.session_id.as_bytes());
        assert_eq!(
            u64::from_be_bytes(plaintext[49..57].try_into().unwrap()),
            wall_seconds + DEFAULT_ABSOLUTE_SESSION_LIFETIME.as_secs()
        );

        let master_bytes: [u8; 32] = plaintext[17..49].try_into().unwrap();
        let client_master = SessionMaster::from_bytes(master_bytes);
        let client_keys = DirectionalKeys::derive(&client_master).unwrap();
        assert_eq!(
            client_keys.request_key_bytes(),
            prepared.session.keys().request_key_bytes()
        );
        assert_eq!(
            client_keys.response_key_bytes(),
            prepared.session.keys().response_key_bytes()
        );
    }

    #[test]
    fn outer_request_metadata_is_strict() {
        let session_id = Uuid::new_v4();
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, session_id.to_string().parse().unwrap());
        assert_eq!(
            parse_request_session_id(&"/v2/request".parse().unwrap(), &headers).unwrap(),
            session_id
        );

        assert!(parse_request_session_id(&"/v2/request?x=1".parse().unwrap(), &headers).is_err());

        let mut duplicate = headers.clone();
        duplicate.append(SESSION_ID_HEADER, session_id.to_string().parse().unwrap());
        assert!(parse_request_session_id(&"/v2/request".parse().unwrap(), &duplicate).is_err());

        let compact = session_id.simple().to_string();
        let mut noncanonical = HeaderMap::new();
        noncanonical.insert(SESSION_ID_HEADER, compact.parse().unwrap());
        assert!(parse_request_session_id(&"/v2/request".parse().unwrap(), &noncanonical).is_err());

        for forbidden in [
            header::AUTHORIZATION.as_str(),
            "proxy-authorization",
            header::COOKIE.as_str(),
            header::CONTENT_ENCODING.as_str(),
        ] {
            let mut rejected = headers.clone();
            rejected.insert(forbidden, "value".parse().unwrap());
            assert!(
                parse_request_session_id(&"/v2/request".parse().unwrap(), &rejected).is_err(),
                "{forbidden}"
            );
        }
    }

    #[test]
    fn post_content_type_requires_one_unambiguous_json_value() {
        let mut headers = HeaderMap::new();
        assert!(validate_json_content_type(&headers).is_err());

        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        assert!(validate_json_content_type(&headers).is_ok());
        headers.insert(
            header::CONTENT_TYPE,
            "Application/JSON; charset=UTF-8".parse().unwrap(),
        );
        assert!(validate_json_content_type(&headers).is_ok());

        for invalid in [
            "text/json",
            "application/json; profile=v2",
            "application/json; charset=latin1",
            "application/json; charset=utf-8; charset=utf-8",
            "application/json;",
        ] {
            headers.insert(header::CONTENT_TYPE, invalid.parse().unwrap());
            assert!(validate_json_content_type(&headers).is_err(), "{invalid}");
        }

        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.append(header::CONTENT_TYPE, "application/json".parse().unwrap());
        assert!(validate_json_content_type(&headers).is_err());
    }

    #[test]
    fn request_working_set_accounting_is_bounded_and_conservative() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            request_working_set_units(&headers).unwrap(),
            u32::try_from(
                MAX_OUTER_REQUEST_BODY_BYTES * REQUEST_WORKING_SET_MULTIPLIER
                    / REQUEST_WORKING_SET_UNIT_BYTES
            )
            .unwrap()
        );

        headers.insert(header::CONTENT_LENGTH, "1".parse().unwrap());
        assert_eq!(request_working_set_units(&headers).unwrap(), 1);
        headers.insert(
            header::CONTENT_LENGTH,
            REQUEST_WORKING_SET_UNIT_BYTES.to_string().parse().unwrap(),
        );
        assert_eq!(request_working_set_units(&headers).unwrap(), 4);
        headers.insert(
            header::CONTENT_LENGTH,
            MAX_OUTER_REQUEST_BODY_BYTES.to_string().parse().unwrap(),
        );
        let maximum_units = request_working_set_units(&headers).unwrap();
        assert!(usize::try_from(maximum_units).unwrap() > REQUEST_WORKING_SET_UNITS / 2);
        assert!(usize::try_from(maximum_units).unwrap() <= REQUEST_WORKING_SET_UNITS);

        headers.insert(
            header::CONTENT_LENGTH,
            (MAX_OUTER_REQUEST_BODY_BYTES + 1)
                .to_string()
                .parse()
                .unwrap(),
        );
        assert!(matches!(
            request_working_set_units(&headers),
            Err(GatewayError::PayloadTooLarge)
        ));

        headers.insert(header::CONTENT_LENGTH, "invalid".parse().unwrap());
        assert!(matches!(
            request_working_set_units(&headers),
            Err(GatewayError::InvalidRequest)
        ));
        headers.insert(header::CONTENT_LENGTH, "+1".parse().unwrap());
        assert!(matches!(
            request_working_set_units(&headers),
            Err(GatewayError::InvalidRequest)
        ));
        headers.insert(header::CONTENT_LENGTH, "1".parse().unwrap());
        headers.append(header::CONTENT_LENGTH, "1".parse().unwrap());
        assert!(matches!(
            request_working_set_units(&headers),
            Err(GatewayError::InvalidRequest)
        ));
    }

    #[test]
    fn one_maximum_request_excludes_another_maximum_working_set() {
        let state = TransportV2State::new();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_LENGTH,
            MAX_OUTER_REQUEST_BODY_BYTES.to_string().parse().unwrap(),
        );
        let first = state
            .reserve_request_working_set(&headers)
            .expect("first maximum request fits");
        let maximum_units = request_working_set_units(&headers).unwrap();
        assert!(Arc::clone(&state.request_working_set)
            .try_acquire_many_owned(maximum_units)
            .is_err());
        drop(first);
        assert!(Arc::clone(&state.request_working_set)
            .try_acquire_many_owned(maximum_units)
            .is_ok());
    }

    #[test]
    fn encrypted_outer_json_is_strict_canonical_and_bounded() {
        let valid = br#"{"encrypted":"YQ=="}"#;
        assert_eq!(
            parse_encrypted_outer_request(valid, valid.len())
                .unwrap()
                .encrypted
                .as_slice(),
            b"a"
        );
        assert!(
            parse_encrypted_outer_request(br#"{"encrypted":"YQ==","extra":true}"#, 64).is_err()
        );
        assert!(parse_encrypted_outer_request(br#"{"encrypted":"YQ"}"#, 64).is_err());
        assert!(matches!(
            parse_encrypted_outer_request(valid, valid.len() - 1),
            Err(GatewayError::PayloadTooLarge)
        ));
    }

    #[tokio::test]
    async fn body_collection_enforces_the_exact_inclusive_limit() {
        assert_eq!(
            read_bounded_body(Body::from(vec![0_u8; 8]), 8)
                .await
                .expect("body at limit")
                .len(),
            8
        );
        assert!(matches!(
            read_bounded_body(Body::from(vec![0_u8; 9]), 8).await,
            Err(GatewayError::PayloadTooLarge)
        ));
    }

    #[tokio::test]
    async fn unsupported_route_returns_authenticated_404_from_exact_session() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let first_master = [0x41; 32];
        let second_master = [0x42; 32];
        state
            .insert_session(test_session(
                first_id,
                first_master,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();
        state
            .insert_session(test_session(
                second_id,
                second_master,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();

        let request_id = RequestId::from_bytes([0x11; 16]);
        let request_plaintext = serde_json::to_vec(&request_envelope(request_id)).unwrap();
        let client_master = SessionMaster::from_bytes(first_master);
        let client_keys = DirectionalKeys::derive(&client_master).unwrap();
        let encrypted_request = client_keys
            .encrypt_request_record_with_nonce(&first_id, &request_plaintext, [0x12; 12])
            .unwrap();

        let response = state
            .process_encrypted_request(first_id, &encrypted_request, now)
            .await
            .expect("authenticated logical error");
        let response_plaintext = client_keys
            .decrypt_unary_response_record(&first_id, &request_id, response.encrypted.as_slice())
            .expect("first session opens its response");
        let response_envelope =
            UnaryResponseEnvelope::from_json_slice(&response_plaintext, &EnvelopeLimits::default())
                .unwrap();
        assert_eq!(response_envelope.status, 404);
        assert_eq!(response_envelope.request_id, request_id);
        assert_eq!(response_envelope.headers[0].name, "content-type");
        assert_eq!(
            response_envelope.body_base64.unwrap().as_slice(),
            br#"{"error":{"code":"not_found","message":"Not found"}}"#
        );

        let wrong_master = SessionMaster::from_bytes(second_master);
        let wrong_keys = DirectionalKeys::derive(&wrong_master).unwrap();
        assert!(wrong_keys
            .decrypt_unary_response_record(&second_id, &request_id, response.encrypted.as_slice())
            .is_err());

        // Unsupported operations do not pass the replay gate or receive LRU
        // admission promotion.
        let first_lease = state.acquire_session(&first_id, now).await.unwrap();
        assert_eq!(first_lease.state().replay_id_count(), 0);
    }

    #[tokio::test]
    async fn cleanup_respects_live_leases_and_retires_after_drop() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let expires_at = now + Duration::from_secs(10);
        let session_id = Uuid::new_v4();
        state
            .insert_session(test_session(
                session_id,
                [0x51; 32],
                expires_at,
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();

        let lease = state.acquire_session(&session_id, now).await.unwrap();
        assert_eq!(state.cleanup_expired_at(expires_at).await, 0);
        assert!(state
            .acquire_session(&session_id, expires_at)
            .await
            .is_err());

        drop(lease);
        assert_eq!(state.cleanup_expired_at(expires_at).await, 1);
        assert!(state.acquire_session(&session_id, now).await.is_err());
    }

    #[tokio::test]
    async fn pending_attestation_is_one_shot_and_expires_independently() {
        let state = TransportV2State::new();
        let nonce = "v2-only-nonce";
        state
            .create_pending_attestation(nonce)
            .await
            .expect("create pending key");
        let _secret = state
            .take_pending_attestation(nonce)
            .await
            .expect("consume once");
        assert!(state.take_pending_attestation(nonce).await.is_err());

        state
            .create_pending_attestation(nonce)
            .await
            .expect("create replacement key");
        let expired_at = Instant::now() + PENDING_ATTESTATION_TTL;
        assert_eq!(state.cleanup_expired_at(expired_at).await, 1);
        assert!(state.take_pending_attestation(nonce).await.is_err());
    }
}
