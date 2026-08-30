//! Isolated HTTP gateway for encrypted transport v2.
//!
//! This layer owns the v2-only attestation and session caches, authenticates a
//! complete encrypted request envelope, and dispatches only the explicitly
//! projected application operations. Unsupported operations receive an
//! authenticated logical 404 through the exact session lease that decrypted
//! the request; transport v1 is never re-entered.

use std::collections::hash_map::RandomState;
use std::future::Future;
use std::io;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, Request, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;
use x25519_dalek::{EphemeralSecret, PublicKey};
use zeroize::Zeroize;

use crate::encrypt::CustomRng;
use crate::web::attestation_routes;
use crate::web::encryption_middleware::hold_resource_through_response_body;
use crate::{AppState, AsyncRngWrapper};

use super::application::{
    begin_authentication_transition, execute_user_operation, prepare_user_operation,
    LogicalApplicationResponse, LogicalUnaryResponse, OperationPreparation, SessionEffect,
};
use super::crypto::{
    encrypt_key_exchange_record, CryptoError, DirectionalKeys, HandshakePayload, SessionMaster,
    RECORD_OVERHEAD_BYTES,
};
use super::envelope::{
    EncodedBytes, EnvelopeLimits, RequestEnvelope, StreamRecord, UnaryResponseEnvelope, Version2,
    MAX_STREAM_CHUNK_BYTES, MAX_STREAM_ERROR_BYTES,
};
use super::session::{
    GlobalReplayBudget, ReplayClaim, SessionRecordError, StreamResponseAdmission,
    StreamResponseReservation, UnaryResponseReservation, V2SessionState,
    DEFAULT_ABSOLUTE_SESSION_LIFETIME, DEFAULT_GLOBAL_REPLAY_IDS,
};
use super::session_cache::{V2SessionCache, V2SessionInsertError, V2SessionLease};
use super::streaming::{
    LogicalStreamFailure, LogicalStreamItem, LogicalStreamResponse, StreamExecutionGuard,
};
use super::{MAX_LIVE_SESSIONS, MAX_PENDING_ATTESTATIONS};

const MAX_ATTESTATION_NONCE_BYTES: usize = 512;
const PENDING_ATTESTATION_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_KEY_EXCHANGE_BODY_BYTES: usize = 4 * 1024;
const MAX_OUTER_REQUEST_BODY_BYTES: usize =
    EnvelopeLimits::REQUEST.envelope_bytes + RECORD_OVERHEAD_BYTES;
const REQUEST_WORKING_SET_BUDGET_BYTES: usize = 320 * 1024 * 1024;
const REQUEST_WORKING_SET_UNIT_BYTES: usize = 64 * 1024;
const REQUEST_WORKING_SET_MULTIPLIER: usize = 4;
const REQUEST_WORKING_SET_UNITS: usize =
    REQUEST_WORKING_SET_BUDGET_BYTES / REQUEST_WORKING_SET_UNIT_BYTES;
const STORED_OUTPUT_WORKING_SET_BYTES: usize = 200 * 1024 * 1024;
const STORED_OUTPUT_WORKING_SET_UNITS: usize =
    STORED_OUTPUT_WORKING_SET_BYTES / REQUEST_WORKING_SET_UNIT_BYTES;
const PROVIDER_OUTPUT_WORKING_SET_BYTES: usize = 128 * 1024 * 1024;
const PROVIDER_OUTPUT_WORKING_SET_UNITS: usize =
    PROVIDER_OUTPUT_WORKING_SET_BYTES / REQUEST_WORKING_SET_UNIT_BYTES;
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

    fn promote_stored_output_working_set(
        &self,
        permit: &mut OwnedSemaphorePermit,
    ) -> Result<(), GatewayError> {
        self.promote_working_set(permit, STORED_OUTPUT_WORKING_SET_UNITS)
    }

    fn promote_provider_output_working_set(
        &self,
        permit: &mut OwnedSemaphorePermit,
    ) -> Result<(), GatewayError> {
        self.promote_working_set(permit, PROVIDER_OUTPUT_WORKING_SET_UNITS)
    }

    fn promote_working_set(
        &self,
        permit: &mut OwnedSemaphorePermit,
        target_units: usize,
    ) -> Result<(), GatewayError> {
        let additional_units = target_units
            .checked_sub(permit.num_permits())
            .unwrap_or_default();
        if additional_units == 0 {
            return Ok(());
        }
        let additional_units =
            u32::try_from(additional_units).map_err(|_| GatewayError::Internal)?;
        let additional = Arc::clone(&self.request_working_set)
            .try_acquire_many_owned(additional_units)
            .map_err(|_| GatewayError::Unavailable)?;
        permit.merge(additional);
        Ok(())
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
        app_state: Arc<AppState>,
        session_id: Uuid,
        encrypted: &[u8],
        mut working_set_permit: OwnedSemaphorePermit,
        now: Instant,
    ) -> Result<Response, GatewayError> {
        let (lease, envelope) = self
            .decrypt_request_envelope(session_id, encrypted, now)
            .await?;
        let request_id = envelope.request_id;

        let operation = match prepare_user_operation(envelope, lease.state().authority()) {
            OperationPreparation::Unsupported => {
                let response = encrypt_new_logical_response(
                    &lease,
                    request_id,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "Not found",
                    ),
                )?;
                return Ok(encrypted_outer_http_response(response, working_set_permit));
            }
            OperationPreparation::Rejected(response) => {
                let response = encrypt_new_logical_response(&lease, request_id, response)?;
                return Ok(encrypted_outer_http_response(response, working_set_permit));
            }
            OperationPreparation::Ready(operation) => operation,
        };

        if operation.is_streaming() {
            return self
                .process_ready_stream_operation(
                    &lease,
                    request_id,
                    operation,
                    working_set_permit,
                    now,
                    move |dispatch_lease, operation, authentication, admitted_at, stream_guard| {
                        execute_user_operation(
                            app_state,
                            dispatch_lease,
                            operation,
                            authentication,
                            admitted_at,
                            Some(stream_guard),
                        )
                    },
                )
                .await;
        }

        let response = self
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                move |dispatch_lease, operation, authentication, admitted_at| {
                    execute_user_operation(
                        app_state,
                        dispatch_lease,
                        operation,
                        authentication,
                        admitted_at,
                        None,
                    )
                },
            )
            .await?;
        Ok(encrypted_outer_http_response(response, working_set_permit))
    }

    async fn process_ready_operation<Dispatch, DispatchFuture>(
        &self,
        lease: &V2SessionLease,
        request_id: super::envelope::RequestId,
        operation: super::application::UserOperation,
        working_set_permit: &mut OwnedSemaphorePermit,
        now: Instant,
        dispatch: Dispatch,
    ) -> Result<EncryptedOuterResponse, GatewayError>
    where
        Dispatch: FnOnce(
            V2SessionLease,
            super::application::UserOperation,
            Option<super::session::AuthenticationReservation>,
            Instant,
        ) -> DispatchFuture,
        DispatchFuture: Future<Output = super::application::ApplicationOutcome>,
    {
        // Reserve a response before consuming replay capacity or dispatching
        // application work. Every admitted request can therefore receive one
        // authenticated terminal result through this exact session lease.
        let reservation = lease
            .state()
            .begin_unary_response()
            .map_err(GatewayError::from_session_record)?;
        if operation.requires_stored_output_reservation()
            && self
                .promote_stored_output_working_set(working_set_permit)
                .is_err()
        {
            return encrypt_reserved_logical_response(
                lease,
                request_id,
                reservation,
                LogicalUnaryResponse::protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "stored_output_unavailable",
                    "Stored response capacity is unavailable",
                ),
            );
        }
        if operation.requires_provider_output_reservation()
            && self
                .promote_provider_output_working_set(working_set_permit)
                .is_err()
        {
            return encrypt_reserved_logical_response(
                lease,
                request_id,
                reservation,
                LogicalUnaryResponse::protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "provider_output_unavailable",
                    "Provider response capacity is unavailable",
                ),
            );
        }
        match lease.state().claim_request_id(request_id) {
            ReplayClaim::Claimed => {}
            ReplayClaim::Duplicate => {
                return encrypt_reserved_logical_response(
                    lease,
                    request_id,
                    reservation,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::CONFLICT,
                        "replay_detected",
                        "Request identifier has already been used",
                    ),
                );
            }
            ReplayClaim::Exhausted => {
                return encrypt_reserved_logical_response(
                    lease,
                    request_id,
                    reservation,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "session_exhausted",
                        "Session request capacity is exhausted",
                    ),
                );
            }
        }

        let authentication = match begin_authentication_transition(&operation, lease, request_id) {
            Ok(authentication) => authentication,
            Err(response) => {
                return encrypt_reserved_logical_response(lease, request_id, reservation, response);
            }
        };

        if !self.mark_admitted(lease).await {
            lease.state().close();
            return encrypt_reserved_logical_response(
                lease,
                request_id,
                reservation,
                LogicalUnaryResponse::protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "session_unavailable",
                    "Session is unavailable",
                ),
            );
        }

        let outcome = dispatch(lease.clone(), operation, authentication, now).await;
        let session_effect = outcome.session_effect;
        if session_effect == SessionEffect::Close {
            // Stop new admission before producing the terminal response. The
            // exact held lease and pre-dispatch reservation intentionally stay
            // usable while Closing so this admitted response can still finish.
            lease.state().close();
        }
        let LogicalApplicationResponse::Unary(response) = outcome.response else {
            lease.state().close();
            return Err(GatewayError::Internal);
        };
        let encrypted = encrypt_reserved_logical_response(lease, request_id, reservation, response);
        if encrypted.is_err() && session_effect == SessionEffect::NewlyBound {
            // Never leave a newly authenticated session reachable when its
            // only binding response could not be authenticated to the client.
            lease.state().close();
        }
        encrypted
    }

    async fn process_ready_stream_operation<Dispatch, DispatchFuture>(
        &self,
        lease: &V2SessionLease,
        request_id: super::envelope::RequestId,
        operation: super::application::UserOperation,
        mut working_set_permit: OwnedSemaphorePermit,
        now: Instant,
        dispatch: Dispatch,
    ) -> Result<Response, GatewayError>
    where
        Dispatch: FnOnce(
            V2SessionLease,
            super::application::UserOperation,
            Option<super::session::AuthenticationReservation>,
            Instant,
            StreamExecutionGuard,
        ) -> DispatchFuture,
        DispatchFuture: Future<Output = super::application::ApplicationOutcome>,
    {
        let reservation = match lease
            .state()
            .begin_stream_response_or_exhaustion(request_id)
        {
            Ok(StreamResponseAdmission::Stream(reservation)) => reservation,
            Ok(StreamResponseAdmission::ExhaustionResponse(unary)) => {
                let response = encrypt_reserved_logical_response(
                    lease,
                    request_id,
                    unary,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "session_exhausted",
                        "Session response capacity is exhausted",
                    ),
                )?;
                return Ok(encrypted_outer_http_response(response, working_set_permit));
            }
            Err(error) => return Err(GatewayError::from_session_record(error)),
        };

        if self
            .promote_provider_output_working_set(&mut working_set_permit)
            .is_err()
        {
            return encrypt_pre_start_unary_http_response(
                lease,
                request_id,
                reservation,
                LogicalUnaryResponse::protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "provider_output_unavailable",
                    "Provider response capacity is unavailable",
                ),
                working_set_permit,
            );
        }

        match lease.state().claim_request_id(request_id) {
            ReplayClaim::Claimed => {}
            ReplayClaim::Duplicate => {
                return encrypt_pre_start_unary_http_response(
                    lease,
                    request_id,
                    reservation,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::CONFLICT,
                        "replay_detected",
                        "Request identifier has already been used",
                    ),
                    working_set_permit,
                );
            }
            ReplayClaim::Exhausted => {
                return encrypt_pre_start_unary_http_response(
                    lease,
                    request_id,
                    reservation,
                    LogicalUnaryResponse::protocol_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "session_exhausted",
                        "Session request capacity is exhausted",
                    ),
                    working_set_permit,
                );
            }
        }

        let authentication = match begin_authentication_transition(&operation, lease, request_id) {
            Ok(authentication) => authentication,
            Err(response) => {
                return encrypt_pre_start_unary_http_response(
                    lease,
                    request_id,
                    reservation,
                    response,
                    working_set_permit,
                );
            }
        };

        if !self.mark_admitted(lease).await {
            lease.state().close();
            return encrypt_pre_start_unary_http_response(
                lease,
                request_id,
                reservation,
                LogicalUnaryResponse::protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "session_unavailable",
                    "Session is unavailable",
                ),
                working_set_permit,
            );
        }

        let stream_guard = StreamExecutionGuard::new(working_set_permit);
        let outcome = dispatch(
            lease.clone(),
            operation,
            authentication,
            now,
            stream_guard.clone(),
        )
        .await;
        let session_effect = outcome.session_effect;
        if session_effect == SessionEffect::Close {
            lease.state().close();
        }

        match outcome.response {
            LogicalApplicationResponse::Unary(response) => {
                let unary = reservation
                    .into_unary_before_start()
                    .map_err(GatewayError::from_session_record)?;
                let encrypted =
                    encrypt_reserved_logical_response(lease, request_id, unary, response);
                if encrypted.is_err() && session_effect == SessionEffect::NewlyBound {
                    lease.state().close();
                }
                Ok(encrypted_outer_http_response(encrypted?, stream_guard))
            }
            LogicalApplicationResponse::Stream(response) => {
                let response = encrypted_stream_http_response(
                    lease.clone(),
                    reservation,
                    response,
                    stream_guard,
                );
                if response.is_err() && session_effect == SessionEffect::NewlyBound {
                    lease.state().close();
                }
                response
            }
        }
    }

    async fn decrypt_request_envelope(
        &self,
        session_id: Uuid,
        encrypted: &[u8],
        now: Instant,
    ) -> Result<(V2SessionLease, RequestEnvelope), GatewayError> {
        let lease = self.acquire_session(&session_id, now).await?;
        let mut plaintext = lease
            .state()
            .decrypt_request_record(encrypted)
            .map_err(GatewayError::from_session_record)?;
        let parsed = RequestEnvelope::from_json_slice(&plaintext, &EnvelopeLimits::REQUEST);
        plaintext.zeroize();
        let envelope = parsed.map_err(|_| GatewayError::InvalidRequest)?;
        Ok((lease, envelope))
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
) -> Result<Response, GatewayError> {
    let session_id = parse_request_session_id(request.uri(), request.headers())?;
    validate_octet_stream_content_type(request.headers())?;
    let working_set_permit = app_state
        .transport_v2_state
        .reserve_request_working_set(request.headers())?;

    // Fully bound the upload before acquiring a session lease. A slow or
    // oversized outer request therefore cannot pin secret-bearing session
    // state or prevent cache eviction.
    let body = read_bounded_body(request.into_body(), MAX_OUTER_REQUEST_BODY_BYTES).await?;
    validate_encrypted_outer_request(&body)?;

    let response = app_state
        .transport_v2_state
        .process_encrypted_request(
            Arc::clone(&app_state),
            session_id,
            &body,
            working_set_permit,
            Instant::now(),
        )
        .await?;
    drop(body);
    Ok(response)
}

fn encrypted_outer_http_response<T>(
    response: EncryptedOuterResponse,
    working_set_resource: T,
) -> Response
where
    T: Send + Unpin + 'static,
{
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CACHE_CONTROL, "no-store, no-transform")
        .body(Body::from(response.encrypted))
        .expect("static v2 unary response metadata must be valid");
    hold_resource_through_response_body(response, working_set_resource)
}

fn encrypt_pre_start_unary_http_response<T>(
    lease: &V2SessionLease,
    request_id: super::envelope::RequestId,
    reservation: StreamResponseReservation,
    response: LogicalUnaryResponse,
    working_set_resource: T,
) -> Result<Response, GatewayError>
where
    T: Send + Unpin + 'static,
{
    let unary = reservation
        .into_unary_before_start()
        .map_err(GatewayError::from_session_record)?;
    let encrypted = encrypt_reserved_logical_response(lease, request_id, unary, response)?;
    Ok(encrypted_outer_http_response(
        encrypted,
        working_set_resource,
    ))
}

fn encrypted_stream_http_response(
    lease: V2SessionLease,
    mut reservation: StreamResponseReservation,
    response: LogicalStreamResponse,
    stream_guard: StreamExecutionGuard,
) -> Result<Response, GatewayError> {
    let start = encrypt_stream_start(&lease, &mut reservation, response.status, response.headers)?;
    let start_frame = encrypted_outer_sse_frame(start);
    let mut application_stream = response.stream;

    let encrypted_stream = async_stream::stream! {
        // Keeping these resources inside the body guarantees that an unpolled
        // response retains both the exact session lease and its promoted
        // working-set admission until the response itself is dropped.
        let _stream_guard = stream_guard;
        let _lease_guard = &lease;
        yield Ok::<bytes::Bytes, io::Error>(start_frame);

        let mut terminal_emitted = false;
        'application: while let Some(item) = application_stream.next().await {
            match item {
                LogicalStreamItem::Bytes(bytes) => {
                    for chunk in bytes.chunks(MAX_STREAM_CHUNK_BYTES) {
                        if chunk.is_empty() {
                            continue;
                        }
                        match encrypt_stream_chunk(&lease, &mut reservation, chunk) {
                            Ok(encrypted) => {
                                yield Ok(encrypted_outer_sse_frame(encrypted));
                            }
                            Err(GatewayError::Unavailable) => {
                                let failure = LogicalStreamFailure::protocol(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "session_exhausted",
                                    "Session response capacity is exhausted",
                                );
                                match encrypt_stream_error(&lease, &mut reservation, failure) {
                                    Ok(encrypted) => {
                                        yield Ok(encrypted_outer_sse_frame(encrypted));
                                    }
                                    Err(_) => {
                                        yield Err(stream_transport_io_error());
                                    }
                                }
                                terminal_emitted = true;
                                break 'application;
                            }
                            Err(_) => {
                                yield Err(stream_transport_io_error());
                                terminal_emitted = true;
                                break 'application;
                            }
                        }
                    }
                }
                LogicalStreamItem::Complete => {
                    match encrypt_stream_end(&lease, &mut reservation) {
                        Ok(encrypted) => yield Ok(encrypted_outer_sse_frame(encrypted)),
                        Err(_) => yield Err(stream_transport_io_error()),
                    }
                    terminal_emitted = true;
                    break;
                }
                LogicalStreamItem::Failure(failure) => {
                    match encrypt_stream_error(&lease, &mut reservation, failure) {
                        Ok(encrypted) => yield Ok(encrypted_outer_sse_frame(encrypted)),
                        Err(_) => yield Err(stream_transport_io_error()),
                    }
                    terminal_emitted = true;
                    break;
                }
            }
        }

        if !terminal_emitted {
            match encrypt_stream_error(
                &lease,
                &mut reservation,
                LogicalStreamFailure::internal(),
            ) {
                Ok(encrypted) => yield Ok(encrypted_outer_sse_frame(encrypted)),
                Err(_) => yield Err(stream_transport_io_error()),
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(encrypted_stream))
        .map_err(|_| GatewayError::Internal)
}

fn encrypt_stream_start(
    lease: &V2SessionLease,
    reservation: &mut StreamResponseReservation,
    status: StatusCode,
    headers: Vec<super::envelope::HeaderField>,
) -> Result<Vec<u8>, GatewayError> {
    let record = StreamRecord::Start {
        version: Version2,
        request_id: reservation.request_id(),
        sequence: 0,
        status: status.as_u16(),
        headers,
    };
    let mut plaintext = serialize_stream_record(record)?;
    let encrypted = lease
        .state()
        .encrypt_stream_start_record(reservation, &plaintext)
        .map_err(GatewayError::from_session_record);
    plaintext.zeroize();
    encrypted
}

fn encrypt_stream_chunk(
    lease: &V2SessionLease,
    reservation: &mut StreamResponseReservation,
    body: &[u8],
) -> Result<Vec<u8>, GatewayError> {
    let sequence = reservation
        .next_sequence()
        .map_err(GatewayError::from_session_record)?;
    let record = StreamRecord::Chunk {
        version: Version2,
        request_id: reservation.request_id(),
        sequence,
        body_base64: EncodedBytes::from_bytes(body.to_vec()),
    };
    let mut plaintext = serialize_stream_record(record)?;
    let encrypted = lease
        .state()
        .encrypt_stream_chunk_record(reservation, &plaintext)
        .map_err(GatewayError::from_session_record);
    plaintext.zeroize();
    encrypted
}

fn encrypt_stream_end(
    lease: &V2SessionLease,
    reservation: &mut StreamResponseReservation,
) -> Result<Vec<u8>, GatewayError> {
    let sequence = reservation
        .next_sequence()
        .map_err(GatewayError::from_session_record)?;
    let record = StreamRecord::End {
        version: Version2,
        request_id: reservation.request_id(),
        sequence,
    };
    let mut plaintext = serialize_stream_record(record)?;
    let encrypted = lease
        .state()
        .encrypt_stream_terminal_record(reservation, &plaintext)
        .map_err(GatewayError::from_session_record);
    plaintext.zeroize();
    encrypted
}

fn encrypt_stream_error(
    lease: &V2SessionLease,
    reservation: &mut StreamResponseReservation,
    mut failure: LogicalStreamFailure,
) -> Result<Vec<u8>, GatewayError> {
    if !failure.status.is_client_error() && !failure.status.is_server_error() {
        failure = LogicalStreamFailure::internal();
    }
    if failure.body.len() > MAX_STREAM_ERROR_BYTES {
        failure = LogicalStreamFailure::internal();
    }
    let sequence = reservation
        .next_sequence()
        .map_err(GatewayError::from_session_record)?;
    let record = StreamRecord::Error {
        version: Version2,
        request_id: reservation.request_id(),
        sequence,
        status: failure.status.as_u16(),
        body_base64: EncodedBytes::from_bytes(std::mem::take(&mut *failure.body)),
    };
    let mut plaintext = serialize_stream_record(record)?;
    let encrypted = lease
        .state()
        .encrypt_stream_terminal_record(reservation, &plaintext)
        .map_err(GatewayError::from_session_record);
    plaintext.zeroize();
    encrypted
}

fn serialize_stream_record(record: StreamRecord) -> Result<Vec<u8>, GatewayError> {
    record
        .validate(&EnvelopeLimits::default())
        .map_err(|_| GatewayError::Internal)?;
    serde_json::to_vec(&record).map_err(|_| GatewayError::Internal)
}

fn encrypted_outer_sse_frame(mut encrypted: Vec<u8>) -> bytes::Bytes {
    let encoded = STANDARD.encode(&encrypted);
    encrypted.zeroize();
    let mut frame = Vec::with_capacity("data: ".len() + encoded.len() + 2);
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(encoded.as_bytes());
    frame.extend_from_slice(b"\n\n");
    bytes::Bytes::from(frame)
}

fn stream_transport_io_error() -> io::Error {
    io::Error::other("encrypted stream terminated")
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

#[derive(Debug)]
struct EncryptedOuterResponse {
    encrypted: Vec<u8>,
}

fn encrypt_new_logical_response(
    lease: &V2SessionLease,
    request_id: super::envelope::RequestId,
    response: LogicalUnaryResponse,
) -> Result<EncryptedOuterResponse, GatewayError> {
    let reservation = lease
        .state()
        .begin_unary_response()
        .map_err(GatewayError::from_session_record)?;
    encrypt_reserved_logical_response(lease, request_id, reservation, response)
}

fn encrypt_reserved_logical_response(
    lease: &V2SessionLease,
    request_id: super::envelope::RequestId,
    mut reservation: UnaryResponseReservation,
    response: LogicalUnaryResponse,
) -> Result<EncryptedOuterResponse, GatewayError> {
    let mut response = UnaryResponseEnvelope {
        version: Version2,
        request_id,
        status: response.status.as_u16(),
        headers: response.headers,
        body_base64: response
            .body
            .map(|mut body| EncodedBytes::from_bytes(std::mem::take(&mut *body))),
    };
    if response.validate(&EnvelopeLimits::default()).is_err() {
        zeroize_response_body(&mut response);
        return Err(GatewayError::Internal);
    }
    let mut plaintext = match serde_json::to_vec(&response) {
        Ok(plaintext) => plaintext,
        Err(_) => {
            zeroize_response_body(&mut response);
            return Err(GatewayError::Internal);
        }
    };
    zeroize_response_body(&mut response);

    let result = lease
        .state()
        .encrypt_unary_response_record(&mut reservation, &request_id, &plaintext)
        .map_err(GatewayError::from_session_record);
    plaintext.zeroize();
    let encrypted = result?;

    Ok(EncryptedOuterResponse { encrypted })
}

fn zeroize_response_body(response: &mut UnaryResponseEnvelope) {
    if let Some(body) = response.body_base64.take() {
        let mut bytes = body.into_bytes();
        bytes.zeroize();
    }
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

fn validate_encrypted_outer_request(body: &[u8]) -> Result<(), GatewayError> {
    validate_encrypted_outer_request_len(body.len())
}

fn validate_encrypted_outer_request_len(body_len: usize) -> Result<(), GatewayError> {
    if body_len > MAX_OUTER_REQUEST_BODY_BYTES {
        return Err(GatewayError::PayloadTooLarge);
    }
    if body_len < RECORD_OVERHEAD_BYTES {
        return Err(GatewayError::InvalidRequest);
    }
    Ok(())
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
        header::TRANSFER_ENCODING.as_str(),
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

fn validate_octet_stream_content_type(headers: &HeaderMap) -> Result<(), GatewayError> {
    let mut values = headers.get_all(header::CONTENT_TYPE).iter();
    let value = values.next().ok_or(GatewayError::InvalidRequest)?;
    if values.next().is_some() {
        return Err(GatewayError::InvalidRequest);
    }
    let value = value.to_str().map_err(|_| GatewayError::InvalidRequest)?;
    if value.eq_ignore_ascii_case("application/octet-stream") {
        Ok(())
    } else {
        Err(GatewayError::InvalidRequest)
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::jwt::{
        issue_transport_v2_native_handoff_grant_for_test,
        validate_transport_v2_native_handoff_grant_claims_for_test, AuthContext, AuthMethod,
        JwtKeys,
    };
    use crate::provider_cache::{derive_tinfoil_cache_namespace, CacheNamespaceRoot};
    use crate::transport_v2::application::{ApplicationOutcome, UserOperation};
    use crate::transport_v2::crypto::{decrypt_key_exchange_record, SessionMaster};
    use crate::transport_v2::envelope::{
        HeaderField, LogicalMethod, LogicalRequest, RequestId, ResponseMode,
    };
    use crate::transport_v2::session::{
        AuthorityState, BoundAuthority, BoundPrincipal, SessionLimits,
    };
    use crate::ApiError;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn test_session(
        session_id: Uuid,
        master_bytes: [u8; 32],
        expires_at: Instant,
        global_budget: Arc<GlobalReplayBudget>,
    ) -> Arc<V2SessionState> {
        test_session_with_limits(
            session_id,
            master_bytes,
            expires_at,
            global_budget,
            SessionLimits::default(),
        )
    }

    fn test_session_with_limits(
        session_id: Uuid,
        master_bytes: [u8; 32],
        expires_at: Instant,
        global_budget: Arc<GlobalReplayBudget>,
        limits: SessionLimits,
    ) -> Arc<V2SessionState> {
        let master = SessionMaster::from_bytes(master_bytes);
        let keys = DirectionalKeys::derive(&master).expect("derive test keys");
        Arc::new(V2SessionState::new_with_limits(
            session_id,
            keys,
            expires_at,
            global_budget,
            limits,
        ))
    }

    fn request_envelope(request_id: RequestId) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Unary,
            credential: None,
            cache_namespace_root_base64: None,
            request: LogicalRequest {
                method: LogicalMethod::Get,
                path: "/v1/protected/private_key".to_owned(),
                query: None,
                headers: Vec::new(),
                body_base64: None,
            },
        }
    }

    fn login_envelope(request_id: RequestId) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Unary,
            credential: None,
            cache_namespace_root_base64: Some(
                crate::provider_cache::CacheNamespaceRoot::from_bytes([0x42; 32]),
            ),
            request: LogicalRequest {
                method: LogicalMethod::Post,
                path: "/login".to_owned(),
                query: None,
                headers: vec![HeaderField {
                    name: "content-type".to_owned(),
                    value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
                }],
                body_base64: Some(EncodedBytes::from_bytes(b"{}".to_vec())),
            },
        }
    }

    fn get_user_envelope(request_id: RequestId, body: Option<Vec<u8>>) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Unary,
            credential: None,
            cache_namespace_root_base64: None,
            request: LogicalRequest {
                method: LogicalMethod::Get,
                path: "/protected/user".to_owned(),
                query: None,
                headers: Vec::new(),
                body_base64: body.map(EncodedBytes::from_bytes),
            },
        }
    }

    fn stream_chat_envelope(request_id: RequestId) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Stream,
            credential: None,
            cache_namespace_root_base64: None,
            request: LogicalRequest {
                method: LogicalMethod::Post,
                path: "/v1/chat/completions".to_owned(),
                query: None,
                headers: vec![HeaderField {
                    name: "content-type".to_owned(),
                    value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
                }],
                body_base64: Some(EncodedBytes::from_bytes(
                    br#"{"model":"model","messages":[],"stream":true}"#.to_vec(),
                )),
            },
        }
    }

    fn native_handoff_redeem_envelope(
        request_id: RequestId,
        grant: &str,
        native_attempt_id: Uuid,
    ) -> RequestEnvelope {
        RequestEnvelope {
            version: Version2,
            request_id,
            response_mode: ResponseMode::Unary,
            credential: None,
            cache_namespace_root_base64: Some(CacheNamespaceRoot::from_bytes([0xc1; 32])),
            request: LogicalRequest {
                method: LogicalMethod::Post,
                path: "/auth/native-handoff/redeem".to_owned(),
                query: None,
                headers: vec![HeaderField {
                    name: "content-type".to_owned(),
                    value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
                }],
                body_base64: Some(EncodedBytes::from_bytes(
                    serde_json::to_vec(&serde_json::json!({
                        "grant": grant,
                        "native_attempt_id": native_attempt_id,
                    }))
                    .expect("serialize native handoff redemption"),
                )),
            },
        }
    }

    struct NativeHandoffTestRedemption<'a> {
        session_id: Uuid,
        master_bytes: [u8; 32],
        request_id: RequestId,
        grant: &'a str,
        native_attempt_id: Uuid,
        now: Instant,
    }

    async fn process_native_handoff_test_redemption(
        state: &TransportV2State,
        jwt_keys: &JwtKeys,
        redemption: NativeHandoffTestRedemption<'_>,
    ) -> (V2SessionLease, EncryptedOuterResponse) {
        let NativeHandoffTestRedemption {
            session_id,
            master_bytes,
            request_id,
            grant,
            native_attempt_id,
            now,
        } = redemption;
        let plaintext = serde_json::to_vec(&native_handoff_redeem_envelope(
            request_id,
            grant,
            native_attempt_id,
        ))
        .expect("serialize native handoff envelope");
        let client_keys = DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes))
            .expect("derive native handoff client keys");
        let encrypted = client_keys
            .encrypt_request_record(&session_id, &plaintext)
            .expect("encrypt native handoff request");
        let (lease, envelope) = state
            .decrypt_request_envelope(session_id, &encrypted, now)
            .await
            .expect("decrypt native handoff request");
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("anonymous native handoff redemption must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .expect("acquire native handoff working set");
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_dispatch_lease, operation, authentication, admitted_at| async move {
                    let UserOperation::RedeemNativeHandoff {
                        body,
                        cache_namespace_root,
                    } = operation
                    else {
                        panic!("prepared operation must remain native handoff redemption");
                    };
                    let request: serde_json::Value =
                        serde_json::from_slice(&body).expect("parse prepared native handoff body");
                    let request_grant = request["grant"]
                        .as_str()
                        .expect("prepared native handoff grant");
                    let request_attempt = request["native_attempt_id"]
                        .as_str()
                        .and_then(|value| Uuid::parse_str(value).ok())
                        .expect("prepared native handoff attempt");
                    let verified = validate_transport_v2_native_handoff_grant_claims_for_test(
                        request_grant,
                        session_id,
                        request_attempt,
                        jwt_keys,
                    );
                    let (user_id, auth_context) = match verified {
                        Ok(verified) => verified,
                        Err(_) => {
                            drop(authentication);
                            return ApplicationOutcome {
                                response: LogicalApplicationResponse::Unary(
                                    LogicalUnaryResponse::api_error(ApiError::InvalidJwt),
                                ),
                                session_effect: SessionEffect::Retain,
                            };
                        }
                    };

                    authentication
                        .expect("native handoff must reserve authentication")
                        .commit_at(
                            BoundAuthority::user(
                                user_id,
                                auth_context.project_id,
                                &auth_context,
                                admitted_at + Duration::from_secs(30),
                                derive_tinfoil_cache_namespace(&cache_namespace_root, user_id),
                            ),
                            admitted_at,
                        )
                        .expect("commit native handoff binding");
                    ApplicationOutcome {
                        response: LogicalApplicationResponse::Unary(
                            LogicalUnaryResponse::json(
                                StatusCode::OK,
                                &serde_json::json!({ "id": user_id }),
                            )
                            .expect("serialize native handoff success"),
                        ),
                        session_effect: SessionEffect::NewlyBound,
                    }
                },
            )
            .await
            .expect("encrypt native handoff response");
        (lease, response)
    }

    fn bind_test_user(session: &Arc<V2SessionState>, now: Instant, user_byte: u8) {
        let user_id = Uuid::from_bytes([user_byte; 16]);
        session
            .begin_authentication(RequestId::from_bytes([user_byte.wrapping_add(1); 16]))
            .expect("reserve test authentication")
            .commit_at(
                BoundAuthority::user(
                    user_id,
                    7,
                    &AuthContext::new(AuthMethod::Password, 7, [user_byte.wrapping_add(2); 32]),
                    now + Duration::from_secs(30),
                    crate::provider_cache::derive_tinfoil_cache_namespace(
                        &crate::provider_cache::CacheNamespaceRoot::from_bytes(
                            [user_byte.wrapping_add(3); 32],
                        ),
                        user_id,
                    ),
                ),
                now,
            )
            .expect("bind test user");
    }

    fn successful_test_outcome() -> super::super::application::ApplicationOutcome {
        super::super::application::ApplicationOutcome {
            response: LogicalApplicationResponse::Unary(LogicalUnaryResponse {
                status: StatusCode::OK,
                headers: Vec::new(),
                body: Some(zeroize::Zeroizing::new(br#"{"ok":true}"#.to_vec())),
            }),
            session_effect: SessionEffect::Retain,
        }
    }

    fn response_status(
        keys: &DirectionalKeys,
        session_id: &Uuid,
        request_id: &RequestId,
        response: &EncryptedOuterResponse,
    ) -> u16 {
        let plaintext = keys
            .decrypt_unary_response_record(session_id, request_id, response.encrypted.as_slice())
            .expect("decrypt authenticated unary response");
        UnaryResponseEnvelope::from_json_slice(&plaintext, &EnvelopeLimits::default())
            .expect("parse authenticated unary response")
            .status
    }

    async fn unary_http_response_status(
        keys: &DirectionalKeys,
        session_id: &Uuid,
        request_id: &RequestId,
        response: Response,
    ) -> u16 {
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/octet-stream"
        );
        let body = to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read encrypted unary carrier");
        let plaintext = keys
            .decrypt_unary_response_record(session_id, request_id, &body)
            .expect("decrypt authenticated unary response");
        UnaryResponseEnvelope::from_json_slice(&plaintext, &EnvelopeLimits::default())
            .expect("parse authenticated unary response")
            .status
    }

    fn decrypt_outer_stream_records(
        keys: &DirectionalKeys,
        session_id: &Uuid,
        request_id: &RequestId,
        carrier: &[u8],
    ) -> Vec<StreamRecord> {
        let carrier = std::str::from_utf8(carrier).expect("outer SSE is UTF-8");
        carrier
            .split("\n\n")
            .filter(|frame| !frame.is_empty())
            .enumerate()
            .map(|(sequence, frame)| {
                let encoded = frame
                    .strip_prefix("data: ")
                    .expect("one canonical outer data field");
                let encrypted = STANDARD.decode(encoded).expect("canonical outer base64");
                let plaintext = keys
                    .decrypt_stream_response_record(
                        session_id,
                        request_id,
                        sequence as u64,
                        &encrypted,
                    )
                    .expect("decrypt authenticated stream record");
                StreamRecord::from_json_slice(&plaintext, &EnvelopeLimits::default())
                    .expect("parse authenticated stream record")
            })
            .collect()
    }

    #[tokio::test]
    async fn stream_gateway_splits_bytes_and_emits_explicit_end() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let request_id = RequestId::from_bytes([0x72; 16]);
        let master_bytes = [0x73; 32];
        state
            .insert_session(test_session(
                session_id,
                master_bytes,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let reservation = lease
            .state()
            .begin_stream_response(request_id)
            .expect("reserve start and terminal");
        let permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let guard = StreamExecutionGuard::new(permit);
        let application_bytes = bytes::Bytes::from(vec![0x41; MAX_STREAM_CHUNK_BYTES + 1]);
        let logical = LogicalStreamResponse::sse(Box::pin(futures::stream::iter([
            LogicalStreamItem::Bytes(application_bytes),
            LogicalStreamItem::Complete,
        ])));

        let response = encrypted_stream_http_response(lease, reservation, logical, guard)
            .expect("build authenticated stream response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let carrier = to_bytes(response.into_body(), 512 * 1024)
            .await
            .expect("read encrypted carrier");

        let master = SessionMaster::from_bytes(master_bytes);
        let keys = DirectionalKeys::derive(&master).unwrap();
        let records = decrypt_outer_stream_records(&keys, &session_id, &request_id, &carrier);
        assert_eq!(records.len(), 4);
        assert!(matches!(
            &records[0],
            StreamRecord::Start {
                sequence: 0,
                status: 200,
                ..
            }
        ));
        assert!(matches!(
            &records[1],
            StreamRecord::Chunk {
                sequence: 1,
                body_base64,
                ..
            } if body_base64.len() == MAX_STREAM_CHUNK_BYTES
        ));
        assert!(matches!(
            &records[2],
            StreamRecord::Chunk {
                sequence: 2,
                body_base64,
                ..
            } if body_base64.len() == 1
        ));
        assert!(matches!(&records[3], StreamRecord::End { sequence: 3, .. }));
    }

    #[tokio::test]
    async fn stream_gateway_turns_unexpected_source_eof_into_authenticated_error() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let request_id = RequestId::from_bytes([0x74; 16]);
        let master_bytes = [0x75; 32];
        state
            .insert_session(test_session(
                session_id,
                master_bytes,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let reservation = lease
            .state()
            .begin_stream_response(request_id)
            .expect("reserve start and terminal");
        let guard = StreamExecutionGuard::new(
            Arc::clone(&state.request_working_set)
                .try_acquire_owned()
                .unwrap(),
        );
        let logical = LogicalStreamResponse::sse(Box::pin(futures::stream::empty()));

        let response = encrypted_stream_http_response(lease, reservation, logical, guard)
            .expect("build authenticated stream response");
        let carrier = to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read encrypted carrier");
        let master = SessionMaster::from_bytes(master_bytes);
        let keys = DirectionalKeys::derive(&master).unwrap();
        let records = decrypt_outer_stream_records(&keys, &session_id, &request_id, &carrier);
        assert!(matches!(
            records.as_slice(),
            [
                StreamRecord::Start { sequence: 0, .. },
                StreamRecord::Error {
                    sequence: 1,
                    status: 500,
                    ..
                }
            ]
        ));
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
            header::TRANSFER_ENCODING.as_str(),
        ] {
            let mut rejected = headers.clone();
            rejected.insert(forbidden, "value".parse().unwrap());
            assert!(
                parse_request_session_id(&"/v2/request".parse().unwrap(), &rejected).is_err(),
                "{forbidden}"
            );
        }
    }

    #[tokio::test]
    async fn content_length_before_chunked_transfer_is_rejected_at_outer_boundary() {
        let handler_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&handler_calls);
        let app = Router::new().route(
            "/v2/request",
            post(move |request: Request<Body>| {
                let calls = Arc::clone(&calls);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    match validate_fixed_outer_request(request.uri(), request.headers(), false) {
                        Ok(()) => StatusCode::OK,
                        Err(_) => StatusCode::BAD_REQUEST,
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("read test address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("serve raw HTTP regression");
        });

        let session_id = Uuid::new_v4();
        let request = format!(
            "POST /v2/request HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nx-session-id: {session_id}\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n1\r\nx\r\n0\r\n\r\n"
        );
        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect raw client");
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write raw request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read raw response");
        server.abort();

        let response = String::from_utf8(response).expect("HTTP response is ASCII");
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "{response}"
        );
        assert_eq!(handler_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn key_exchange_content_type_requires_one_unambiguous_json_value() {
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
    fn encrypted_request_content_type_is_exact_octet_stream() {
        let mut headers = HeaderMap::new();
        assert!(validate_octet_stream_content_type(&headers).is_err());

        headers.insert(
            header::CONTENT_TYPE,
            "application/octet-stream".parse().unwrap(),
        );
        assert!(validate_octet_stream_content_type(&headers).is_ok());
        headers.insert(
            header::CONTENT_TYPE,
            "Application/Octet-Stream".parse().unwrap(),
        );
        assert!(validate_octet_stream_content_type(&headers).is_ok());

        for invalid in [
            "application/json",
            "application/octet-stream; charset=binary",
            "application/octet-stream; profile=v2",
            " application/octet-stream",
        ] {
            headers.insert(header::CONTENT_TYPE, invalid.parse().unwrap());
            assert!(
                validate_octet_stream_content_type(&headers).is_err(),
                "{invalid}"
            );
        }

        headers.insert(
            header::CONTENT_TYPE,
            "application/octet-stream".parse().unwrap(),
        );
        headers.append(
            header::CONTENT_TYPE,
            "application/octet-stream".parse().unwrap(),
        );
        assert!(validate_octet_stream_content_type(&headers).is_err());
    }

    #[test]
    fn request_working_set_accounting_is_bounded_and_conservative() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            request_working_set_units(&headers).unwrap(),
            u32::try_from(
                (MAX_OUTER_REQUEST_BODY_BYTES * REQUEST_WORKING_SET_MULTIPLIER)
                    .div_ceil(REQUEST_WORKING_SET_UNIT_BYTES)
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
        assert_eq!(maximum_units, 4_289);
        assert_eq!(REQUEST_WORKING_SET_UNITS, 5_120);

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
    fn stored_output_promotion_reaches_the_conservative_response_reservation() {
        let state = TransportV2State::new();
        let mut permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        state
            .promote_stored_output_working_set(&mut permit)
            .unwrap();
        assert_eq!(permit.num_permits(), STORED_OUTPUT_WORKING_SET_UNITS);
        assert_eq!(
            state.request_working_set.available_permits(),
            REQUEST_WORKING_SET_UNITS - STORED_OUTPUT_WORKING_SET_UNITS
        );

        state
            .promote_stored_output_working_set(&mut permit)
            .unwrap();
        assert_eq!(permit.num_permits(), STORED_OUTPUT_WORKING_SET_UNITS);
    }

    #[tokio::test]
    async fn encrypted_outer_response_holds_working_set_through_body_lifetime() {
        let working_set = Arc::new(Semaphore::new(1));
        let response_value = || EncryptedOuterResponse {
            encrypted: vec![1, 2, 3],
        };

        let permit = Arc::clone(&working_set).try_acquire_owned().unwrap();
        let response = encrypted_outer_http_response(response_value(), permit);
        assert_eq!(working_set.available_permits(), 0);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/octet-stream"
        );
        assert_eq!(
            response.headers()[header::CACHE_CONTROL],
            "no-store, no-transform"
        );
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(&body[..], &[1, 2, 3]);
        assert_eq!(working_set.available_permits(), 1);

        let permit = Arc::clone(&working_set).try_acquire_owned().unwrap();
        let response = encrypted_outer_http_response(response_value(), permit);
        assert_eq!(working_set.available_permits(), 0);
        drop(response);
        assert_eq!(working_set.available_permits(), 1);
    }

    #[test]
    fn encrypted_outer_record_is_raw_exact_and_bounded() {
        assert!(validate_encrypted_outer_request_len(RECORD_OVERHEAD_BYTES).is_ok());
        assert!(validate_encrypted_outer_request_len(MAX_OUTER_REQUEST_BODY_BYTES).is_ok());
        assert!(matches!(
            validate_encrypted_outer_request_len(RECORD_OVERHEAD_BYTES - 1),
            Err(GatewayError::InvalidRequest)
        ));
        assert!(matches!(
            validate_encrypted_outer_request_len(MAX_OUTER_REQUEST_BODY_BYTES + 1),
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
            .encrypt_request_record(&first_id, &request_plaintext)
            .unwrap();

        let (lease, envelope) = state
            .decrypt_request_envelope(first_id, &encrypted_request, now)
            .await
            .expect("authenticated logical request");
        let request_id = envelope.request_id;
        assert!(matches!(
            prepare_user_operation(envelope, lease.state().authority()),
            OperationPreparation::Unsupported
        ));
        let response = encrypt_new_logical_response(
            &lease,
            request_id,
            LogicalUnaryResponse::protocol_error(StatusCode::NOT_FOUND, "not_found", "Not found"),
        )
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
    async fn native_handoff_grant_binds_only_its_exact_session_and_attempt() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let target_session_id = Uuid::from_bytes([0xc2; 16]);
        let other_session_id = Uuid::from_bytes([0xc3; 16]);
        let target_master = [0xc4; 32];
        let other_master = [0xc5; 32];
        for (session_id, master) in [
            (target_session_id, target_master),
            (other_session_id, other_master),
        ] {
            state
                .insert_session(test_session(
                    session_id,
                    master,
                    now + Duration::from_secs(60),
                    Arc::clone(&state.global_replay_budget),
                ))
                .await
                .expect("insert native handoff test session");
        }

        let jwt_keys = JwtKeys::new(vec![0xc6; 32]).expect("native handoff signing key");
        let user_id = Uuid::from_bytes([0xc7; 16]);
        let auth_context = AuthContext::new(AuthMethod::OAuth, 7, [0xc8; 32]);
        let native_attempt_id = Uuid::from_bytes([0xc9; 16]);
        let issued = issue_transport_v2_native_handoff_grant_for_test(
            user_id,
            &auth_context,
            target_session_id,
            native_attempt_id,
            &jwt_keys,
        )
        .expect("issue native handoff grant");

        let wrong_session_request_id = RequestId::from_bytes([0xca; 16]);
        let (wrong_session_lease, wrong_session_response) = process_native_handoff_test_redemption(
            &state,
            &jwt_keys,
            NativeHandoffTestRedemption {
                session_id: other_session_id,
                master_bytes: other_master,
                request_id: wrong_session_request_id,
                grant: &issued.grant,
                native_attempt_id,
                now,
            },
        )
        .await;
        let other_client_keys =
            DirectionalKeys::derive(&SessionMaster::from_bytes(other_master)).unwrap();
        assert_eq!(
            response_status(
                &other_client_keys,
                &other_session_id,
                &wrong_session_request_id,
                &wrong_session_response,
            ),
            StatusCode::UNAUTHORIZED.as_u16()
        );
        assert!(matches!(
            wrong_session_lease.state().authority(),
            AuthorityState::Anonymous
        ));

        let wrong_attempt_request_id = RequestId::from_bytes([0xcc; 16]);
        let (wrong_attempt_lease, wrong_attempt_response) = process_native_handoff_test_redemption(
            &state,
            &jwt_keys,
            NativeHandoffTestRedemption {
                session_id: target_session_id,
                master_bytes: target_master,
                request_id: wrong_attempt_request_id,
                grant: &issued.grant,
                native_attempt_id: Uuid::from_bytes([0xce; 16]),
                now,
            },
        )
        .await;
        let target_client_keys =
            DirectionalKeys::derive(&SessionMaster::from_bytes(target_master)).unwrap();
        assert_eq!(
            response_status(
                &target_client_keys,
                &target_session_id,
                &wrong_attempt_request_id,
                &wrong_attempt_response,
            ),
            StatusCode::UNAUTHORIZED.as_u16()
        );
        assert!(matches!(
            wrong_attempt_lease.state().authority(),
            AuthorityState::Anonymous
        ));

        let success_request_id = RequestId::from_bytes([0xcf; 16]);
        let (target_lease, success_response) = process_native_handoff_test_redemption(
            &state,
            &jwt_keys,
            NativeHandoffTestRedemption {
                session_id: target_session_id,
                master_bytes: target_master,
                request_id: success_request_id,
                grant: &issued.grant,
                native_attempt_id,
                now,
            },
        )
        .await;
        assert_eq!(
            response_status(
                &target_client_keys,
                &target_session_id,
                &success_request_id,
                &success_response,
            ),
            StatusCode::OK.as_u16()
        );
        assert!(other_client_keys
            .decrypt_unary_response_record(
                &target_session_id,
                &success_request_id,
                success_response.encrypted.as_slice(),
            )
            .is_err());

        let expected_cache_namespace =
            derive_tinfoil_cache_namespace(&CacheNamespaceRoot::from_bytes([0xc1; 32]), user_id);
        let AuthorityState::Bound(bound) = target_lease.state().authority() else {
            panic!("exact native handoff redemption must bind the target session");
        };
        let BoundPrincipal::User {
            user_id: bound_user_id,
            project_id: bound_project_id,
            auth_context: bound_auth_context,
            cache_namespace: bound_cache_namespace,
        } = bound.principal()
        else {
            panic!("native handoff must bind a user principal");
        };
        assert_eq!(*bound_user_id, user_id);
        assert_eq!(*bound_project_id, auth_context.project_id);
        assert_eq!(bound_auth_context, &auth_context);
        assert_eq!(bound_cache_namespace, &expected_cache_namespace);

        let second_request_id = RequestId::from_bytes([0xd1; 16]);
        let second_plaintext = serde_json::to_vec(&native_handoff_redeem_envelope(
            second_request_id,
            &issued.grant,
            native_attempt_id,
        ))
        .unwrap();
        let second_encrypted = target_client_keys
            .encrypt_request_record(&target_session_id, &second_plaintext)
            .unwrap();
        let (second_lease, second_envelope) = state
            .decrypt_request_envelope(target_session_id, &second_encrypted, now)
            .await
            .unwrap();
        let OperationPreparation::Rejected(second_rejection) =
            prepare_user_operation(second_envelope, second_lease.state().authority())
        else {
            panic!("a bound target session must reject a second native handoff redemption");
        };
        let second_response =
            encrypt_new_logical_response(&second_lease, second_request_id, second_rejection)
                .unwrap();
        assert_eq!(
            response_status(
                &target_client_keys,
                &target_session_id,
                &second_request_id,
                &second_response,
            ),
            StatusCode::CONFLICT.as_u16()
        );
        assert!(matches!(
            second_lease.state().authority(),
            AuthorityState::Bound(_)
        ));
    }

    #[tokio::test]
    async fn duplicate_request_id_dispatches_exactly_once() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x51; 32];
        state
            .insert_session(test_session(
                session_id,
                master_bytes,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();

        let request_id = RequestId::from_bytes([0x52; 16]);
        let plaintext = serde_json::to_vec(&login_envelope(request_id)).unwrap();
        let master = SessionMaster::from_bytes(master_bytes);
        let client_keys = DirectionalKeys::derive(&master).unwrap();
        let encrypted = client_keys
            .encrypt_request_record(&session_id, &plaintext)
            .unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));

        let (lease, envelope) = state
            .decrypt_request_envelope(session_id, &encrypted, now)
            .await
            .unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("valid login operation must be ready");
        };
        let first_dispatches = Arc::clone(&dispatches);
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let first = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                move |_lease, _operation, authentication, _| async move {
                    first_dispatches.fetch_add(1, Ordering::SeqCst);
                    drop(authentication);
                    successful_test_outcome()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &first),
            200
        );

        let (lease, envelope) = state
            .decrypt_request_envelope(session_id, &encrypted, now)
            .await
            .unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("failed authentication reservation must restore anonymous authority");
        };
        let second_dispatches = Arc::clone(&dispatches);
        let duplicate = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                move |_lease, _operation, authentication, _| async move {
                    second_dispatches.fetch_add(1, Ordering::SeqCst);
                    drop(authentication);
                    successful_test_outcome()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &duplicate),
            409
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(lease.state().replay_id_count(), 1);
        assert!(matches!(
            lease.state().authority(),
            AuthorityState::Anonymous
        ));
    }

    #[tokio::test]
    async fn duplicate_stream_request_id_returns_authenticated_unary_conflict_without_dispatch() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x57; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        bind_test_user(&session, now, 0x58);
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x59; 16]);
        let dispatches = Arc::new(AtomicUsize::new(0));
        let keys = DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes)).unwrap();

        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(stream_chat_envelope(request_id), lease.state().authority())
        else {
            panic!("bound streaming Chat operation must be ready");
        };
        let first_dispatches = Arc::clone(&dispatches);
        let first_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let first = state
            .process_ready_stream_operation(
                &lease,
                request_id,
                operation,
                first_permit,
                now,
                move |_lease, _operation, authentication, _, _guard| async move {
                    assert!(authentication.is_none());
                    first_dispatches.fetch_add(1, Ordering::SeqCst);
                    super::super::application::ApplicationOutcome {
                        response: LogicalApplicationResponse::Stream(LogicalStreamResponse::sse(
                            Box::pin(futures::stream::iter([LogicalStreamItem::Complete])),
                        )),
                        session_effect: SessionEffect::Retain,
                    }
                },
            )
            .await
            .expect("first stream request succeeds");
        let carrier = to_bytes(first.into_body(), 64 * 1024)
            .await
            .expect("read first encrypted stream");
        let records = decrypt_outer_stream_records(&keys, &session_id, &request_id, &carrier);
        assert!(matches!(
            records.as_slice(),
            [
                StreamRecord::Start { sequence: 0, .. },
                StreamRecord::End { sequence: 1, .. }
            ]
        ));

        let duplicate_lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(duplicate_operation) = prepare_user_operation(
            stream_chat_envelope(request_id),
            duplicate_lease.state().authority(),
        ) else {
            panic!("duplicate streaming Chat operation must still classify");
        };
        let duplicate_dispatches = Arc::clone(&dispatches);
        let duplicate_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let duplicate = state
            .process_ready_stream_operation(
                &duplicate_lease,
                request_id,
                duplicate_operation,
                duplicate_permit,
                now,
                move |_lease, _operation, _authentication, _, _guard| async move {
                    duplicate_dispatches.fetch_add(1, Ordering::SeqCst);
                    successful_test_outcome()
                },
            )
            .await
            .expect("duplicate receives an authenticated response");

        assert_eq!(
            unary_http_response_status(&keys, &session_id, &request_id, duplicate).await,
            409
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(duplicate_lease.state().replay_id_count(), 1);
    }

    #[tokio::test]
    async fn stream_gateway_uses_final_response_slot_for_authenticated_exhaustion() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x91; 32];
        let session = test_session_with_limits(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
            SessionLimits::new(4, 4, 2),
        );
        bind_test_user(&session, now, 0x92);
        let mut spent = session
            .begin_unary_response()
            .expect("reserve prior response slot");
        session
            .encrypt_unary_response_record(
                &mut spent,
                &RequestId::from_bytes([0x93; 16]),
                b"prior response",
            )
            .expect("consume prior response slot");
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x94; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(stream_chat_envelope(request_id), lease.state().authority())
        else {
            panic!("bound streaming Chat operation must be ready");
        };
        let dispatches = Arc::new(AtomicUsize::new(0));
        let counted_dispatches = Arc::clone(&dispatches);
        let permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_stream_operation(
                &lease,
                request_id,
                operation,
                permit,
                now,
                move |_lease, _operation, _authentication, _, _guard| async move {
                    counted_dispatches.fetch_add(1, Ordering::SeqCst);
                    successful_test_outcome()
                },
            )
            .await
            .expect("final response slot authenticates exhaustion");

        let keys = DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes)).unwrap();
        assert_eq!(
            unary_http_response_status(&keys, &session_id, &request_id, response).await,
            503
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(lease.state().replay_id_count(), 0);
        assert_eq!(lease.state().response_record_count(), 2);
    }

    #[tokio::test]
    async fn stream_gateway_fails_outer_transport_when_no_response_slot_remains() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x95; 32];
        let session = test_session_with_limits(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
            SessionLimits::new(4, 4, 1),
        );
        bind_test_user(&session, now, 0x96);
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x97; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(stream_chat_envelope(request_id), lease.state().authority())
        else {
            panic!("bound streaming Chat operation must be ready");
        };
        let mut spent = lease
            .state()
            .begin_unary_response()
            .expect("reserve final response slot");
        lease
            .state()
            .encrypt_unary_response_record(
                &mut spent,
                &RequestId::from_bytes([0x98; 16]),
                b"prior response",
            )
            .expect("consume final response slot");
        let dispatches = Arc::new(AtomicUsize::new(0));
        let counted_dispatches = Arc::clone(&dispatches);
        let permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_stream_operation(
                &lease,
                request_id,
                operation,
                permit,
                now,
                move |_lease, _operation, _authentication, _, _guard| async move {
                    counted_dispatches.fetch_add(1, Ordering::SeqCst);
                    successful_test_outcome()
                },
            )
            .await;

        assert!(matches!(response, Err(GatewayError::Unavailable)));
        assert_eq!(dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(lease.state().replay_id_count(), 0);
        assert_eq!(lease.state().response_record_count(), 1);
    }

    #[tokio::test]
    async fn pre_start_stream_failure_converts_reserved_capacity_to_exact_bound_unary_response() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x5a; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        bind_test_user(&session, now, 0x5b);
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x5c; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(stream_chat_envelope(request_id), lease.state().authority())
        else {
            panic!("bound streaming Chat operation must be ready");
        };
        let permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_stream_operation(
                &lease,
                request_id,
                operation,
                permit,
                now,
                |_lease, _operation, authentication, _, _guard| async move {
                    assert!(authentication.is_none());
                    super::super::application::ApplicationOutcome {
                        response: LogicalApplicationResponse::Unary(
                            LogicalUnaryResponse::protocol_error(
                                StatusCode::BAD_REQUEST,
                                "setup_failed",
                                "Stream setup failed",
                            ),
                        ),
                        session_effect: SessionEffect::Retain,
                    }
                },
            )
            .await
            .expect("pre-Start failure remains authentically reportable");

        let keys = DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes)).unwrap();
        assert_eq!(
            unary_http_response_status(&keys, &session_id, &request_id, response).await,
            400
        );
        assert_eq!(lease.state().replay_id_count(), 1);
        assert_eq!(lease.state().response_record_count(), 1);
    }

    #[tokio::test]
    async fn invalid_bodyless_shape_does_not_consume_replay_id() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x61; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        let auth_context = AuthContext::new(AuthMethod::Password, 7, [0x62; 32]);
        session
            .begin_authentication(RequestId::from_bytes([0x63; 16]))
            .unwrap()
            .commit_at(
                BoundAuthority::user(
                    Uuid::from_bytes([0x64; 16]),
                    7,
                    &auth_context,
                    now + Duration::from_secs(30),
                    crate::provider_cache::derive_tinfoil_cache_namespace(
                        &crate::provider_cache::CacheNamespaceRoot::from_bytes([0x66; 32]),
                        Uuid::from_bytes([0x64; 16]),
                    ),
                ),
                now,
            )
            .unwrap();
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x65; 16]);
        let master = SessionMaster::from_bytes(master_bytes);
        let client_keys = DirectionalKeys::derive(&master).unwrap();
        let invalid_plaintext =
            serde_json::to_vec(&get_user_envelope(request_id, Some(Vec::new()))).unwrap();
        let invalid_encrypted = client_keys
            .encrypt_request_record(&session_id, &invalid_plaintext)
            .unwrap();
        let (lease, envelope) = state
            .decrypt_request_envelope(session_id, &invalid_encrypted, now)
            .await
            .unwrap();
        let OperationPreparation::Rejected(rejection) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("an explicit empty body is not a bodyless request");
        };
        let response = encrypt_new_logical_response(&lease, request_id, rejection).unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            400
        );
        assert_eq!(lease.state().replay_id_count(), 0);

        let valid_plaintext = serde_json::to_vec(&get_user_envelope(request_id, None)).unwrap();
        let valid_encrypted = client_keys
            .encrypt_request_record(&session_id, &valid_plaintext)
            .unwrap();
        let (lease, envelope) = state
            .decrypt_request_envelope(session_id, &valid_encrypted, now)
            .await
            .unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("a null body is a valid bodyless protected request");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_lease, _operation, authentication, _| async move {
                    assert!(authentication.is_none());
                    successful_test_outcome()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            200
        );
        assert_eq!(lease.state().replay_id_count(), 1);
    }

    #[tokio::test]
    async fn terminal_effect_closes_before_encrypting_the_admitted_response() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x81; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        session
            .begin_authentication(RequestId::from_bytes([0x82; 16]))
            .unwrap()
            .commit_at(
                BoundAuthority::user(
                    Uuid::from_bytes([0x83; 16]),
                    7,
                    &AuthContext::new(AuthMethod::Password, 7, [0x84; 32]),
                    now + Duration::from_secs(30),
                    crate::provider_cache::derive_tinfoil_cache_namespace(
                        &crate::provider_cache::CacheNamespaceRoot::from_bytes([0x86; 32]),
                        Uuid::from_bytes([0x83; 16]),
                    ),
                ),
                now,
            )
            .unwrap();
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x85; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) = prepare_user_operation(
            get_user_envelope(request_id, None),
            lease.state().authority(),
        ) else {
            panic!("bound user operation must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_lease, _operation, authentication, _| async move {
                    assert!(authentication.is_none());
                    let mut outcome = successful_test_outcome();
                    outcome.session_effect = SessionEffect::Close;
                    outcome
                },
            )
            .await
            .expect("held terminal response must still encrypt");

        let client_keys =
            DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes)).unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            200
        );
        assert!(lease.state().is_closing());
        assert!(state.acquire_session(&session_id, now).await.is_err());
        assert_eq!(state.cleanup_expired_at(now).await, 0);
        drop(lease);
        assert_eq!(state.cleanup_expired_at(now).await, 1);
    }

    #[tokio::test]
    async fn terminal_effect_stays_closed_when_response_encryption_fails() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let session = test_session(
            session_id,
            [0x91; 32],
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        session
            .begin_authentication(RequestId::from_bytes([0x92; 16]))
            .unwrap()
            .commit_at(
                BoundAuthority::user(
                    Uuid::from_bytes([0x93; 16]),
                    7,
                    &AuthContext::new(AuthMethod::Password, 7, [0x94; 32]),
                    now + Duration::from_secs(30),
                    crate::provider_cache::derive_tinfoil_cache_namespace(
                        &crate::provider_cache::CacheNamespaceRoot::from_bytes([0x96; 32]),
                        Uuid::from_bytes([0x93; 16]),
                    ),
                ),
                now,
            )
            .unwrap();
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x95; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) = prepare_user_operation(
            get_user_envelope(request_id, None),
            lease.state().authority(),
        ) else {
            panic!("bound user operation must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let result = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_lease, _operation, authentication, _| async move {
                    assert!(authentication.is_none());
                    super::super::application::ApplicationOutcome {
                        response: LogicalApplicationResponse::Unary(LogicalUnaryResponse {
                            status: StatusCode::OK,
                            headers: vec![HeaderField {
                                name: "invalid header name".to_owned(),
                                value_base64: EncodedBytes::from_bytes(b"x".to_vec()),
                            }],
                            body: None,
                        }),
                        session_effect: SessionEffect::Close,
                    }
                },
            )
            .await;

        assert!(matches!(result, Err(GatewayError::Internal)));
        assert!(lease.state().is_closing());
        assert!(state.acquire_session(&session_id, now).await.is_err());
        assert_eq!(state.cleanup_expired_at(now).await, 0);
        drop(lease);
        assert_eq!(state.cleanup_expired_at(now).await, 1);
    }

    #[tokio::test]
    async fn newly_bound_effect_keeps_a_successfully_delivered_session_open() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0xa1; 32];
        state
            .insert_session(test_session(
                session_id,
                master_bytes,
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();

        let request_id = RequestId::from_bytes([0xa2; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(login_envelope(request_id), lease.state().authority())
        else {
            panic!("anonymous login operation must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_lease, _operation, authentication, _| async move {
                    authentication
                        .expect("login must reserve authentication")
                        .commit_at(
                            BoundAuthority::user(
                                Uuid::from_bytes([0xa3; 16]),
                                7,
                                &AuthContext::new(AuthMethod::Password, 7, [0xa4; 32]),
                                now + Duration::from_secs(30),
                                crate::provider_cache::derive_tinfoil_cache_namespace(
                                    &crate::provider_cache::CacheNamespaceRoot::from_bytes(
                                        [0xa6; 32],
                                    ),
                                    Uuid::from_bytes([0xa3; 16]),
                                ),
                            ),
                            now,
                        )
                        .unwrap();
                    let mut outcome = successful_test_outcome();
                    outcome.session_effect = SessionEffect::NewlyBound;
                    outcome
                },
            )
            .await
            .expect("binding response must encrypt");

        let client_keys =
            DirectionalKeys::derive(&SessionMaster::from_bytes(master_bytes)).unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            200
        );
        assert!(matches!(
            lease.state().authority(),
            AuthorityState::Bound(_)
        ));
        assert!(!lease.state().is_closing());
        assert!(state.acquire_session(&session_id, now).await.is_ok());
    }

    #[tokio::test]
    async fn newly_bound_effect_closes_when_its_binding_response_cannot_encrypt() {
        let state = TransportV2State::new();
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        state
            .insert_session(test_session(
                session_id,
                [0xb1; 32],
                now + Duration::from_secs(60),
                Arc::clone(&state.global_replay_budget),
            ))
            .await
            .unwrap();

        let request_id = RequestId::from_bytes([0xb2; 16]);
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(login_envelope(request_id), lease.state().authority())
        else {
            panic!("anonymous login operation must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let result = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                |_lease, _operation, authentication, _| async move {
                    authentication
                        .expect("login must reserve authentication")
                        .commit_at(
                            BoundAuthority::user(
                                Uuid::from_bytes([0xb3; 16]),
                                7,
                                &AuthContext::new(AuthMethod::Password, 7, [0xb4; 32]),
                                now + Duration::from_secs(30),
                                crate::provider_cache::derive_tinfoil_cache_namespace(
                                    &crate::provider_cache::CacheNamespaceRoot::from_bytes(
                                        [0xb6; 32],
                                    ),
                                    Uuid::from_bytes([0xb3; 16]),
                                ),
                            ),
                            now,
                        )
                        .unwrap();
                    super::super::application::ApplicationOutcome {
                        response: LogicalApplicationResponse::Unary(LogicalUnaryResponse {
                            status: StatusCode::OK,
                            headers: vec![HeaderField {
                                name: "invalid header name".to_owned(),
                                value_base64: EncodedBytes::from_bytes(b"x".to_vec()),
                            }],
                            body: None,
                        }),
                        session_effect: SessionEffect::NewlyBound,
                    }
                },
            )
            .await;

        assert!(matches!(result, Err(GatewayError::Internal)));
        assert!(lease.state().is_closing());
        assert!(state.acquire_session(&session_id, now).await.is_err());
    }

    #[tokio::test]
    async fn stored_output_contention_is_authenticated_before_replay_or_dispatch() {
        let mut state = TransportV2State::new();
        state.request_working_set = Arc::new(Semaphore::new(1));
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0x71; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        let auth_context = AuthContext::new(AuthMethod::Password, 7, [0x72; 32]);
        session
            .begin_authentication(RequestId::from_bytes([0x73; 16]))
            .unwrap()
            .commit_at(
                BoundAuthority::user(
                    Uuid::from_bytes([0x74; 16]),
                    7,
                    &auth_context,
                    now + Duration::from_secs(30),
                    crate::provider_cache::derive_tinfoil_cache_namespace(
                        &crate::provider_cache::CacheNamespaceRoot::from_bytes([0x76; 32]),
                        Uuid::from_bytes([0x74; 16]),
                    ),
                ),
                now,
            )
            .unwrap();
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0x75; 16]);
        let mut envelope = get_user_envelope(request_id, None);
        envelope.request.path = "/protected/kv".to_owned();
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("valid KV list operation must be ready");
        };
        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let observed_dispatches = Arc::clone(&dispatches);
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                move |_lease, _operation, _authentication, _| async move {
                    observed_dispatches.fetch_add(1, Ordering::SeqCst);
                    successful_test_outcome()
                },
            )
            .await
            .unwrap();

        let master = SessionMaster::from_bytes(master_bytes);
        let client_keys = DirectionalKeys::derive(&master).unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            503
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(lease.state().replay_id_count(), 0);
        assert_eq!(working_set_permit.num_permits(), 1);
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
        let nonce = Uuid::new_v4().to_string();
        state
            .create_pending_attestation(&nonce)
            .await
            .expect("create pending key");
        let _secret = state
            .take_pending_attestation(&nonce)
            .await
            .expect("consume once");
        assert!(state.take_pending_attestation(&nonce).await.is_err());

        state
            .create_pending_attestation(&nonce)
            .await
            .expect("create replacement key");
        let expired_at = Instant::now() + PENDING_ATTESTATION_TTL;
        assert_eq!(state.cleanup_expired_at(expired_at).await, 1);
        assert!(state.take_pending_attestation(&nonce).await.is_err());
    }

    #[test]
    fn provider_output_promotion_reaches_the_bounded_response_reservation() {
        let state = TransportV2State::new();
        let mut permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        state
            .promote_provider_output_working_set(&mut permit)
            .unwrap();
        assert_eq!(permit.num_permits(), PROVIDER_OUTPUT_WORKING_SET_UNITS);
        assert_eq!(
            state.request_working_set.available_permits(),
            REQUEST_WORKING_SET_UNITS - PROVIDER_OUTPUT_WORKING_SET_UNITS
        );

        state
            .promote_provider_output_working_set(&mut permit)
            .unwrap();
        assert_eq!(permit.num_permits(), PROVIDER_OUTPUT_WORKING_SET_UNITS);
    }

    #[tokio::test]
    async fn provider_output_contention_is_authenticated_before_replay_or_dispatch() {
        let mut state = TransportV2State::new();
        state.request_working_set = Arc::new(Semaphore::new(1));
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let master_bytes = [0xc1; 32];
        let session = test_session(
            session_id,
            master_bytes,
            now + Duration::from_secs(60),
            Arc::clone(&state.global_replay_budget),
        );
        state.insert_session(session).await.unwrap();

        let request_id = RequestId::from_bytes([0xc2; 16]);
        let mut envelope = request_envelope(request_id);
        envelope.request.path = "/v1/models".to_owned();
        let lease = state.acquire_session(&session_id, now).await.unwrap();
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(envelope, lease.state().authority())
        else {
            panic!("public models operation must be ready");
        };
        assert!(operation.requires_provider_output_reservation());

        let mut working_set_permit = Arc::clone(&state.request_working_set)
            .try_acquire_owned()
            .unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let observed_dispatches = Arc::clone(&dispatches);
        let response = state
            .process_ready_operation(
                &lease,
                request_id,
                operation,
                &mut working_set_permit,
                now,
                move |_lease, _operation, _authentication, _| async move {
                    observed_dispatches.fetch_add(1, Ordering::SeqCst);
                    successful_test_outcome()
                },
            )
            .await
            .unwrap();

        let master = SessionMaster::from_bytes(master_bytes);
        let client_keys = DirectionalKeys::derive(&master).unwrap();
        assert_eq!(
            response_status(&client_keys, &session_id, &request_id, &response),
            503
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(lease.state().replay_id_count(), 0);
        assert_eq!(working_set_permit.num_permits(), 1);
    }
}
