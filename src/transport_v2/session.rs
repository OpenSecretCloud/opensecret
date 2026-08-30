use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use uuid::Uuid;

use super::crypto::{CryptoError, DirectionalKeys};
use super::envelope::RequestId;

pub(crate) const DEFAULT_ABSOLUTE_SESSION_LIFETIME: Duration = Duration::from_secs(3_900);
pub(crate) const DEFAULT_REPLAY_IDS_PER_SESSION: usize = 65_536;
pub(crate) const DEFAULT_GLOBAL_REPLAY_IDS: usize = 2_097_152;
pub(crate) const DEFAULT_REQUEST_RECORDS_PER_SESSION: usize = 65_536;
pub(crate) const DEFAULT_RESPONSE_RECORDS_PER_SESSION: usize = 65_536;

#[derive(Clone, Copy)]
pub(crate) struct SessionLimits {
    replay_ids: usize,
    request_records: usize,
    response_records: usize,
}

impl SessionLimits {
    #[cfg(test)]
    pub(super) const fn new(
        replay_ids: usize,
        request_records: usize,
        response_records: usize,
    ) -> Self {
        Self {
            replay_ids,
            request_records,
            response_records,
        }
    }
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            replay_ids: DEFAULT_REPLAY_IDS_PER_SESSION,
            request_records: DEFAULT_REQUEST_RECORDS_PER_SESSION,
            response_records: DEFAULT_RESPONSE_RECORDS_PER_SESSION,
        }
    }
}

/// Shared accounting for exact replay identifiers retained across all live v2
/// sessions.
///
/// A successful reservation belongs to one [`ReplayRegistry`] until that
/// registry is dropped. A registry which once observes global exhaustion is
/// permanently exhausted even if another session later releases capacity.
pub(crate) struct GlobalReplayBudget {
    limit: usize,
    used: AtomicUsize,
}

impl GlobalReplayBudget {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    fn try_reserve_one(&self) -> bool {
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            if current >= self.limit {
                return false;
            }

            match self.used.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, count: usize) {
        if count == 0 {
            return;
        }

        let released = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(count)
            });
        debug_assert!(released.is_ok(), "global replay accounting underflow");
    }

    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    pub(crate) const fn limit(&self) -> usize {
        self.limit
    }
}

impl Default for GlobalReplayBudget {
    fn default() -> Self {
        Self::new(DEFAULT_GLOBAL_REPLAY_IDS)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplayClaim {
    Claimed,
    Duplicate,
    Exhausted,
}

/// The exact, unordered request identifiers claimed by one session/key epoch.
///
/// The set is allocated only when the first identifier is accepted. Individual
/// identifiers are never evicted. Capacity failure permanently exhausts this
/// registry, while duplicates consume neither local nor global capacity.
pub(crate) struct ReplayRegistry {
    ids: Mutex<Option<HashSet<RequestId>>>,
    local_limit: usize,
    global_budget: Arc<GlobalReplayBudget>,
    exhausted: AtomicBool,
}

impl ReplayRegistry {
    pub(crate) fn new(local_limit: usize, global_budget: Arc<GlobalReplayBudget>) -> Self {
        Self {
            ids: Mutex::new(None),
            local_limit,
            global_budget,
            exhausted: AtomicBool::new(false),
        }
    }

    pub(crate) fn claim(&self, request_id: RequestId) -> ReplayClaim {
        if self.exhausted.load(Ordering::Acquire) {
            return ReplayClaim::Exhausted;
        }

        let mut ids = lock_unpoisoned(&self.ids);
        if self.exhausted.load(Ordering::Acquire) {
            return ReplayClaim::Exhausted;
        }

        if ids
            .as_ref()
            .is_some_and(|claimed| claimed.contains(&request_id))
        {
            return ReplayClaim::Duplicate;
        }

        let local_count = ids.as_ref().map_or(0, HashSet::len);
        if local_count >= self.local_limit {
            self.exhausted.store(true, Ordering::Release);
            return ReplayClaim::Exhausted;
        }

        if !self.global_budget.try_reserve_one() {
            self.exhausted.store(true, Ordering::Release);
            return ReplayClaim::Exhausted;
        }

        let inserted = ids.get_or_insert_with(HashSet::new).insert(request_id);
        if !inserted {
            self.global_budget.release(1);
            return ReplayClaim::Duplicate;
        }

        ReplayClaim::Claimed
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }

    pub(crate) fn len(&self) -> usize {
        lock_unpoisoned(&self.ids).as_ref().map_or(0, HashSet::len)
    }
}

impl Drop for ReplayRegistry {
    fn drop(&mut self) {
        let retained = lock_unpoisoned(&self.ids).as_ref().map_or(0, HashSet::len);
        self.global_budget.release(retained);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) enum BoundPrincipal {
    User { user_id: Uuid, project_id: i32 },
    Platform { platform_user_id: Uuid },
    ApiKey { api_key_id: i32, user_id: Uuid },
}

/// Stable authority identity retained by the transport session.
///
/// Credential-derived authorization context remains an application concern in
/// the binding PR. API keys have no independent token expiry, so their
/// authority remains bounded by the session's absolute expiry and live
/// database checks.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct BoundAuthority {
    principal: BoundPrincipal,
    authentication_expires_at: Option<Instant>,
}

impl BoundAuthority {
    pub(crate) const fn user(
        user_id: Uuid,
        project_id: i32,
        authentication_expires_at: Instant,
    ) -> Self {
        Self {
            principal: BoundPrincipal::User {
                user_id,
                project_id,
            },
            authentication_expires_at: Some(authentication_expires_at),
        }
    }

    pub(crate) const fn platform(
        platform_user_id: Uuid,
        authentication_expires_at: Instant,
    ) -> Self {
        Self {
            principal: BoundPrincipal::Platform { platform_user_id },
            authentication_expires_at: Some(authentication_expires_at),
        }
    }

    pub(crate) const fn api_key(api_key_id: i32, user_id: Uuid) -> Self {
        Self {
            principal: BoundPrincipal::ApiKey {
                api_key_id,
                user_id,
            },
            authentication_expires_at: None,
        }
    }

    pub(crate) const fn principal(&self) -> &BoundPrincipal {
        &self.principal
    }

    pub(crate) const fn authentication_expires_at(&self) -> Option<Instant> {
        self.authentication_expires_at
    }

    fn is_expired_at(&self, now: Instant) -> bool {
        self.authentication_expires_at
            .is_some_and(|expires_at| now >= expires_at)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) enum AuthorityState {
    Anonymous,
    Authenticating(RequestId),
    Bound(BoundAuthority),
    Closing,
}

struct AuthorityCell {
    state: Mutex<AuthorityState>,
}

impl AuthorityCell {
    fn new() -> Self {
        Self {
            state: Mutex::new(AuthorityState::Anonymous),
        }
    }

    fn snapshot(&self) -> AuthorityState {
        lock_unpoisoned(&self.state).clone()
    }

    fn begin(
        self: &Arc<Self>,
        request_id: RequestId,
    ) -> Result<AuthenticationReservation, AuthenticationStartError> {
        let mut state = lock_unpoisoned(&self.state);
        match &*state {
            AuthorityState::Anonymous => {
                *state = AuthorityState::Authenticating(request_id);
                drop(state);
                Ok(AuthenticationReservation {
                    authority: Arc::clone(self),
                    request_id,
                    finalized: false,
                })
            }
            AuthorityState::Authenticating(_) => {
                Err(AuthenticationStartError::AuthenticationInProgress)
            }
            AuthorityState::Bound(_) => Err(AuthenticationStartError::AlreadyBound),
            AuthorityState::Closing => Err(AuthenticationStartError::Closing),
        }
    }

    fn close(&self) {
        *lock_unpoisoned(&self.state) = AuthorityState::Closing;
    }

    fn expire_bound_authority_at(&self, now: Instant) -> bool {
        let mut state = lock_unpoisoned(&self.state);
        let expired = matches!(&*state, AuthorityState::Bound(bound) if bound.is_expired_at(now));
        if expired {
            *state = AuthorityState::Closing;
        }
        expired
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AuthenticationStartError {
    AuthenticationInProgress,
    AlreadyBound,
    Closing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AuthenticationCommitError {
    AuthenticationExpired,
    ReservationLost,
}

/// Cancellation-safe ownership of one `Anonymous -> Authenticating` state
/// transition.
///
/// Dropping or explicitly cancelling the reservation restores `Anonymous` only
/// if the cell still contains this exact request identifier. Committing is a
/// single locked compare-and-transition and cannot rebind an existing session.
pub(crate) struct AuthenticationReservation {
    authority: Arc<AuthorityCell>,
    request_id: RequestId,
    finalized: bool,
}

impl AuthenticationReservation {
    pub(crate) const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub(crate) fn commit(self, bound: BoundAuthority) -> Result<(), AuthenticationCommitError> {
        self.commit_at(bound, Instant::now())
    }

    pub(crate) fn commit_at(
        mut self,
        bound: BoundAuthority,
        now: Instant,
    ) -> Result<(), AuthenticationCommitError> {
        if bound.is_expired_at(now) {
            self.rollback_if_matching();
            self.finalized = true;
            return Err(AuthenticationCommitError::AuthenticationExpired);
        }

        let mut state = lock_unpoisoned(&self.authority.state);
        if matches!(&*state, AuthorityState::Authenticating(active) if *active == self.request_id) {
            *state = AuthorityState::Bound(bound);
            self.finalized = true;
            Ok(())
        } else {
            Err(AuthenticationCommitError::ReservationLost)
        }
    }

    pub(crate) fn cancel(mut self) {
        self.rollback_if_matching();
        self.finalized = true;
    }

    fn rollback_if_matching(&self) {
        let mut state = lock_unpoisoned(&self.authority.state);
        if matches!(&*state, AuthorityState::Authenticating(active) if *active == self.request_id) {
            *state = AuthorityState::Anonymous;
        }
    }
}

impl Drop for AuthenticationReservation {
    fn drop(&mut self) {
        if !self.finalized {
            self.rollback_if_matching();
        }
    }
}

struct RecordBudget {
    limit: usize,
    used: AtomicUsize,
}

impl RecordBudget {
    const fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    fn try_reserve(&self, count: usize) -> bool {
        if count == 0 {
            return true;
        }

        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(count) else {
                return false;
            };
            if next > self.limit {
                return false;
            }

            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn try_consume(&self) -> bool {
        self.try_reserve(1)
    }

    fn is_full(&self) -> bool {
        self.used.load(Ordering::Acquire) >= self.limit
    }

    fn release(&self, count: usize) {
        if count == 0 {
            return;
        }
        let released = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(count)
            });
        debug_assert!(released.is_ok(), "response record accounting underflow");
    }

    fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    AbsoluteSessionExpired,
    AuthenticationExpired,
    AuthenticationInProgress,
    Closing,
    Exhausted,
    RequestRecordBudgetExhausted,
    ResponseRecordBudgetExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecordBudgetError {
    RequestRecordsExhausted,
    ResponseRecordsExhausted,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SessionRecordError {
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("request record budget exhausted")]
    RequestRecordsExhausted,
    #[error("response record budget exhausted")]
    ResponseRecordsExhausted,
    #[error("response reservation belongs to another session")]
    ResponseReservationMismatch,
    #[error("response reservation was already consumed")]
    ResponseReservationConsumed,
    #[error("stream start has not been encrypted")]
    StreamNotStarted,
    #[error("stream start was already encrypted")]
    StreamAlreadyStarted,
    #[error("stream response is already closed")]
    StreamClosed,
    #[error("invalid stream response sequence")]
    InvalidStreamSequence,
}

/// One response-record slot reserved before a unary request may dispatch.
pub(crate) struct UnaryResponseReservation {
    response_records: Arc<RecordBudget>,
    reserved: bool,
}

impl UnaryResponseReservation {
    fn belongs_to(&self, response_records: &Arc<RecordBudget>) -> bool {
        Arc::ptr_eq(&self.response_records, response_records)
    }

    fn consume(&mut self) {
        self.reserved = false;
    }
}

impl Drop for UnaryResponseReservation {
    fn drop(&mut self) {
        if self.reserved {
            self.response_records.release(1);
        }
    }
}

enum StreamResponsePhase {
    BeforeStart,
    Started { next_sequence: u64 },
    Failed,
    Finished,
}

/// Start and terminal response-record slots reserved atomically before a
/// streaming request may dispatch.
///
/// Later chunks charge capacity dynamically, but cannot consume either
/// reserved slot. Dropping before an encryption attempt returns unused slots;
/// a slot remains charged once its encryption attempt begins, including when
/// cryptographic encryption fails.
pub(crate) struct StreamResponseReservation {
    response_records: Arc<RecordBudget>,
    start_reserved: bool,
    terminal_reserved: bool,
    phase: StreamResponsePhase,
}

impl StreamResponseReservation {
    fn belongs_to(&self, response_records: &Arc<RecordBudget>) -> bool {
        Arc::ptr_eq(&self.response_records, response_records)
    }

    fn consume_start(&mut self) {
        self.start_reserved = false;
    }

    fn consume_terminal(&mut self) {
        self.terminal_reserved = false;
    }
}

impl Drop for StreamResponseReservation {
    fn drop(&mut self) {
        let unused = usize::from(self.start_reserved) + usize::from(self.terminal_reserved);
        self.response_records.release(unused);
    }
}

/// Secret-bearing state for one transport-v2 session/key epoch.
///
/// New admissions are checked against monotonic absolute and authentication
/// expiry. Its budget-coupled encryption methods remain accessible to an
/// already held `Arc<V2SessionState>` so an admitted unary response or stream
/// can finish after expiry. Final `Arc` drop zeroizes the directional keys.
pub(crate) struct V2SessionState {
    session_id: Uuid,
    keys: DirectionalKeys,
    absolute_expires_at: Instant,
    authority: Arc<AuthorityCell>,
    replay: ReplayRegistry,
    request_records: RecordBudget,
    response_records: Arc<RecordBudget>,
    exhausted: AtomicBool,
}

impl V2SessionState {
    pub(crate) fn new(
        session_id: Uuid,
        keys: DirectionalKeys,
        absolute_expires_at: Instant,
        global_replay_budget: Arc<GlobalReplayBudget>,
    ) -> Self {
        Self::new_with_limits(
            session_id,
            keys,
            absolute_expires_at,
            global_replay_budget,
            SessionLimits::default(),
        )
    }

    pub(crate) fn new_with_limits(
        session_id: Uuid,
        keys: DirectionalKeys,
        absolute_expires_at: Instant,
        global_replay_budget: Arc<GlobalReplayBudget>,
        limits: SessionLimits,
    ) -> Self {
        Self {
            session_id,
            keys,
            absolute_expires_at,
            authority: Arc::new(AuthorityCell::new()),
            replay: ReplayRegistry::new(limits.replay_ids, global_replay_budget),
            request_records: RecordBudget::new(limits.request_records),
            response_records: Arc::new(RecordBudget::new(limits.response_records)),
            exhausted: AtomicBool::new(false),
        }
    }

    pub(crate) const fn session_id(&self) -> Uuid {
        self.session_id
    }

    #[cfg(test)]
    pub(crate) const fn keys(&self) -> &DirectionalKeys {
        &self.keys
    }

    pub(crate) const fn absolute_expires_at(&self) -> Instant {
        self.absolute_expires_at
    }

    pub(crate) fn authority(&self) -> AuthorityState {
        self.authority.snapshot()
    }

    pub(crate) fn begin_authentication(
        &self,
        request_id: RequestId,
    ) -> Result<AuthenticationReservation, AuthenticationStartError> {
        self.authority.begin(request_id)
    }

    pub(crate) fn check_new_admission_at(&self, now: Instant) -> Result<(), AdmissionError> {
        if now >= self.absolute_expires_at {
            self.authority.close();
            return Err(AdmissionError::AbsoluteSessionExpired);
        }

        if self.exhausted.load(Ordering::Acquire) || self.replay.is_exhausted() {
            return Err(AdmissionError::Exhausted);
        }
        if self.request_records.is_full() {
            self.exhausted.store(true, Ordering::Release);
            return Err(AdmissionError::RequestRecordBudgetExhausted);
        }
        if self.response_records.is_full() {
            return Err(AdmissionError::ResponseRecordBudgetExhausted);
        }

        let mut authority = lock_unpoisoned(&self.authority.state);
        match &*authority {
            AuthorityState::Anonymous | AuthorityState::Bound(_) => {
                let auth_expired = matches!(
                    &*authority,
                    AuthorityState::Bound(bound) if bound.is_expired_at(now)
                );
                if auth_expired {
                    *authority = AuthorityState::Closing;
                    Err(AdmissionError::AuthenticationExpired)
                } else {
                    Ok(())
                }
            }
            AuthorityState::Authenticating(_) => Err(AdmissionError::AuthenticationInProgress),
            AuthorityState::Closing => Err(AdmissionError::Closing),
        }
    }

    pub(crate) fn accepts_new_requests_at(&self, now: Instant) -> bool {
        self.check_new_admission_at(now).is_ok()
    }

    pub(crate) fn should_retire_at(&self, now: Instant) -> bool {
        if now >= self.absolute_expires_at || self.is_exhausted() {
            self.authority.close();
            return true;
        }
        if self.authority.expire_bound_authority_at(now) {
            return true;
        }
        self.is_closing()
    }

    pub(crate) fn close(&self) {
        self.authority.close();
    }

    pub(crate) fn is_closing(&self) -> bool {
        matches!(self.authority.snapshot(), AuthorityState::Closing)
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
            || self.replay.is_exhausted()
            || self.request_records.is_full()
            || self.response_records.is_full()
    }

    /// Authenticate a request record and charge its record slot only after AEAD
    /// succeeds. Keeping the directional keys private makes the charge
    /// inseparable from production request decryption.
    pub(crate) fn decrypt_request_record(
        &self,
        record: &[u8],
    ) -> Result<Vec<u8>, SessionRecordError> {
        let plaintext = self
            .keys
            .decrypt_request_record(&self.session_id, record)
            .map_err(SessionRecordError::Crypto)?;
        self.record_authenticated_request()
            .map_err(|_| SessionRecordError::RequestRecordsExhausted)?;
        Ok(plaintext)
    }

    fn record_authenticated_request(&self) -> Result<(), RecordBudgetError> {
        if self.request_records.try_consume() {
            Ok(())
        } else {
            self.exhausted.store(true, Ordering::Release);
            Err(RecordBudgetError::RequestRecordsExhausted)
        }
    }

    fn reserve_response_records(&self, count: usize) -> Result<(), RecordBudgetError> {
        if self.response_records.try_reserve(count) {
            Ok(())
        } else {
            Err(RecordBudgetError::ResponseRecordsExhausted)
        }
    }

    /// Reserve response capacity before a unary request may dispatch.
    pub(crate) fn begin_unary_response(
        &self,
    ) -> Result<UnaryResponseReservation, SessionRecordError> {
        self.reserve_response_records(1)
            .map_err(|_| SessionRecordError::ResponseRecordsExhausted)?;
        Ok(UnaryResponseReservation {
            response_records: Arc::clone(&self.response_records),
            reserved: true,
        })
    }

    /// Consume a pre-dispatch unary reservation and encrypt its response.
    ///
    /// This intentionally does not re-check expiry or closing state. A held
    /// response lease may finish after new admissions have stopped.
    pub(crate) fn encrypt_unary_response_record(
        &self,
        reservation: &mut UnaryResponseReservation,
        request_id: &RequestId,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SessionRecordError> {
        if !reservation.belongs_to(&self.response_records) {
            return Err(SessionRecordError::ResponseReservationMismatch);
        }
        if !reservation.reserved {
            return Err(SessionRecordError::ResponseReservationConsumed);
        }
        reservation.consume();
        self.keys
            .encrypt_unary_response_record(&self.session_id, request_id, plaintext)
            .map_err(SessionRecordError::Crypto)
    }

    /// Atomically reserve start and terminal records before a streaming request
    /// may dispatch.
    pub(crate) fn begin_stream_response(
        &self,
    ) -> Result<StreamResponseReservation, SessionRecordError> {
        self.reserve_response_records(2)
            .map_err(|_| SessionRecordError::ResponseRecordsExhausted)?;
        Ok(StreamResponseReservation {
            response_records: Arc::clone(&self.response_records),
            start_reserved: true,
            terminal_reserved: true,
            phase: StreamResponsePhase::BeforeStart,
        })
    }

    /// Consume the stream's reserved start slot. Sequence zero is fixed by this
    /// API and cannot be caller-substituted.
    pub(crate) fn encrypt_stream_start_record(
        &self,
        reservation: &mut StreamResponseReservation,
        request_id: &RequestId,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SessionRecordError> {
        if !reservation.belongs_to(&self.response_records) {
            return Err(SessionRecordError::ResponseReservationMismatch);
        }
        match reservation.phase {
            StreamResponsePhase::BeforeStart => {}
            StreamResponsePhase::Started { .. } => {
                return Err(SessionRecordError::StreamAlreadyStarted);
            }
            StreamResponsePhase::Failed | StreamResponsePhase::Finished => {
                return Err(SessionRecordError::StreamClosed);
            }
        }
        if !reservation.start_reserved {
            return Err(SessionRecordError::ResponseReservationConsumed);
        }

        reservation.consume_start();
        let encrypted = self
            .keys
            .encrypt_stream_response_record(&self.session_id, request_id, 0, plaintext)
            .map_err(SessionRecordError::Crypto);
        reservation.phase = if encrypted.is_ok() {
            StreamResponsePhase::Started { next_sequence: 1 }
        } else {
            StreamResponsePhase::Failed
        };
        encrypted
    }

    /// Dynamically charge and encrypt one chunk after a successful start while
    /// preserving the pre-reserved terminal slot.
    pub(crate) fn encrypt_stream_chunk_record(
        &self,
        reservation: &mut StreamResponseReservation,
        request_id: &RequestId,
        sequence: u64,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SessionRecordError> {
        if !reservation.belongs_to(&self.response_records) {
            return Err(SessionRecordError::ResponseReservationMismatch);
        }
        let next_sequence = match reservation.phase {
            StreamResponsePhase::BeforeStart => return Err(SessionRecordError::StreamNotStarted),
            StreamResponsePhase::Started { next_sequence } => next_sequence,
            StreamResponsePhase::Failed | StreamResponsePhase::Finished => {
                return Err(SessionRecordError::StreamClosed);
            }
        };
        if sequence != next_sequence || sequence == u64::MAX {
            return Err(SessionRecordError::InvalidStreamSequence);
        }

        self.reserve_response_records(1)
            .map_err(|_| SessionRecordError::ResponseRecordsExhausted)?;
        let encrypted = self
            .keys
            .encrypt_stream_response_record(&self.session_id, request_id, sequence, plaintext)
            .map_err(SessionRecordError::Crypto);
        reservation.phase = if encrypted.is_ok() {
            StreamResponsePhase::Started {
                next_sequence: sequence + 1,
            }
        } else {
            StreamResponsePhase::Failed
        };
        encrypted
    }

    /// Consume the stream's pre-reserved terminal slot and encrypt its terminal
    /// record. The slot remains charged after this call, including on a crypto
    /// failure, so a failing RNG cannot reopen budget for another response.
    pub(crate) fn encrypt_stream_terminal_record(
        &self,
        reservation: &mut StreamResponseReservation,
        request_id: &RequestId,
        sequence: u64,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SessionRecordError> {
        if !reservation.belongs_to(&self.response_records) {
            return Err(SessionRecordError::ResponseReservationMismatch);
        }
        let next_sequence = match reservation.phase {
            StreamResponsePhase::BeforeStart => return Err(SessionRecordError::StreamNotStarted),
            StreamResponsePhase::Started { next_sequence } => next_sequence,
            StreamResponsePhase::Failed | StreamResponsePhase::Finished => {
                return Err(SessionRecordError::StreamClosed);
            }
        };
        if sequence != next_sequence {
            return Err(SessionRecordError::InvalidStreamSequence);
        }
        if !reservation.terminal_reserved {
            return Err(SessionRecordError::ResponseReservationConsumed);
        }
        reservation.consume_terminal();
        reservation.phase = StreamResponsePhase::Finished;
        self.keys
            .encrypt_stream_response_record(&self.session_id, request_id, sequence, plaintext)
            .map_err(SessionRecordError::Crypto)
    }

    pub(crate) fn claim_request_id(&self, request_id: RequestId) -> ReplayClaim {
        let claim = self.replay.claim(request_id);
        if claim == ReplayClaim::Exhausted {
            self.exhausted.store(true, Ordering::Release);
        }
        claim
    }

    pub(crate) fn request_record_count(&self) -> usize {
        self.request_records.used()
    }

    pub(crate) fn response_record_count(&self) -> usize {
        self.response_records.used()
    }

    pub(crate) fn replay_id_count(&self) -> usize {
        self.replay.len()
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::thread;

    use super::super::crypto::SessionMaster;
    use super::*;

    fn request_id(value: u128) -> RequestId {
        RequestId::from_bytes(value.to_be_bytes())
    }

    fn session(
        absolute_expires_at: Instant,
        limits: SessionLimits,
        global: Arc<GlobalReplayBudget>,
    ) -> V2SessionState {
        let master = SessionMaster::from_bytes([0x5a; 32]);
        let keys = DirectionalKeys::derive(&master).expect("derive test session keys");
        V2SessionState::new_with_limits(
            Uuid::from_bytes([0x11; 16]),
            keys,
            absolute_expires_at,
            global,
            limits,
        )
    }

    #[test]
    fn default_limits_match_the_protocol_contract() {
        assert_eq!(
            DEFAULT_ABSOLUTE_SESSION_LIFETIME,
            Duration::from_secs(3_900)
        );
        assert_eq!(DEFAULT_REPLAY_IDS_PER_SESSION, 65_536);
        assert_eq!(DEFAULT_GLOBAL_REPLAY_IDS, 2_097_152);
        assert_eq!(DEFAULT_REQUEST_RECORDS_PER_SESSION, 65_536);
        assert_eq!(DEFAULT_RESPONSE_RECORDS_PER_SESSION, 65_536);
    }

    #[test]
    fn replay_registry_is_exact_unordered_and_duplicates_are_free() {
        let global = Arc::new(GlobalReplayBudget::new(8));
        let registry = ReplayRegistry::new(3, Arc::clone(&global));

        assert_eq!(registry.claim(request_id(9)), ReplayClaim::Claimed);
        assert_eq!(registry.claim(request_id(1)), ReplayClaim::Claimed);
        assert_eq!(registry.claim(request_id(9)), ReplayClaim::Duplicate);
        assert_eq!(registry.claim(request_id(5)), ReplayClaim::Claimed);
        assert_eq!(registry.len(), 3);
        assert_eq!(global.used(), 3);

        assert_eq!(registry.claim(request_id(2)), ReplayClaim::Exhausted);
        assert_eq!(registry.claim(request_id(9)), ReplayClaim::Exhausted);
        assert_eq!(registry.len(), 3);
        assert_eq!(global.used(), 3);
    }

    #[test]
    fn global_exhaustion_is_permanent_for_the_observing_session() {
        let global = Arc::new(GlobalReplayBudget::new(2));
        let first = ReplayRegistry::new(2, Arc::clone(&global));
        let second = ReplayRegistry::new(2, Arc::clone(&global));
        let exhausted = ReplayRegistry::new(2, Arc::clone(&global));

        assert_eq!(first.claim(request_id(1)), ReplayClaim::Claimed);
        assert_eq!(second.claim(request_id(2)), ReplayClaim::Claimed);
        assert_eq!(exhausted.claim(request_id(3)), ReplayClaim::Exhausted);
        assert_eq!(global.used(), 2);

        drop(first);
        assert_eq!(global.used(), 1);
        assert_eq!(exhausted.claim(request_id(3)), ReplayClaim::Exhausted);

        let replacement = ReplayRegistry::new(2, Arc::clone(&global));
        assert_eq!(replacement.claim(request_id(4)), ReplayClaim::Claimed);
        assert_eq!(global.used(), 2);

        drop(second);
        drop(exhausted);
        drop(replacement);
        assert_eq!(global.used(), 0);
    }

    #[test]
    fn concurrent_duplicate_has_exactly_one_winner() {
        const WORKERS: usize = 32;
        let global = Arc::new(GlobalReplayBudget::new(WORKERS));
        let registry = Arc::new(ReplayRegistry::new(WORKERS, Arc::clone(&global)));
        let start = Arc::new(Barrier::new(WORKERS));

        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    registry.claim(request_id(7))
                })
            })
            .collect();

        let claims: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("replay worker panicked"))
            .collect();
        assert_eq!(
            claims
                .iter()
                .filter(|claim| **claim == ReplayClaim::Claimed)
                .count(),
            1
        );
        assert_eq!(
            claims
                .iter()
                .filter(|claim| **claim == ReplayClaim::Duplicate)
                .count(),
            WORKERS - 1
        );
        assert_eq!(global.used(), 1);
    }

    #[test]
    fn authentication_reservation_drop_and_cancel_restore_anonymous() {
        let authority = Arc::new(AuthorityCell::new());

        {
            let reservation = authority.begin(request_id(1)).expect("reserve auth");
            assert_eq!(reservation.request_id(), request_id(1));
            assert!(matches!(
                authority.snapshot(),
                AuthorityState::Authenticating(active) if active == request_id(1)
            ));
        }
        assert!(matches!(authority.snapshot(), AuthorityState::Anonymous));

        authority
            .begin(request_id(2))
            .expect("reserve second auth")
            .cancel();
        assert!(matches!(authority.snapshot(), AuthorityState::Anonymous));
    }

    #[test]
    fn authentication_commit_is_request_owned_and_cannot_rebind() {
        let authority = Arc::new(AuthorityCell::new());
        let now = Instant::now();
        let user = Uuid::new_v4();

        authority
            .begin(request_id(1))
            .expect("reserve auth")
            .commit_at(
                BoundAuthority::user(user, 17, now + Duration::from_secs(30)),
                now,
            )
            .expect("commit auth");

        match authority.snapshot() {
            AuthorityState::Bound(bound) => {
                assert!(matches!(
                    bound.principal(),
                    BoundPrincipal::User { user_id, project_id }
                        if *user_id == user && *project_id == 17
                ));
            }
            _ => panic!("authority was not bound"),
        }
        assert_eq!(
            authority.begin(request_id(2)).err(),
            Some(AuthenticationStartError::AlreadyBound)
        );
    }

    #[test]
    fn authentication_commit_rejects_exact_expiry_and_rolls_back() {
        let authority = Arc::new(AuthorityCell::new());
        let now = Instant::now();

        let error = authority
            .begin(request_id(1))
            .expect("reserve auth")
            .commit_at(BoundAuthority::platform(Uuid::new_v4(), now), now)
            .expect_err("expired auth must fail");
        assert_eq!(error, AuthenticationCommitError::AuthenticationExpired);
        assert!(matches!(authority.snapshot(), AuthorityState::Anonymous));
    }

    #[test]
    fn authentication_commit_only_changes_its_matching_request() {
        let authority = Arc::new(AuthorityCell::new());
        let now = Instant::now();
        let reservation = authority.begin(request_id(1)).expect("reserve auth");

        *lock_unpoisoned(&authority.state) = AuthorityState::Authenticating(request_id(2));
        let error = reservation
            .commit_at(
                BoundAuthority::user(Uuid::new_v4(), 7, now + Duration::from_secs(30)),
                now,
            )
            .expect_err("stale reservation must not commit");

        assert_eq!(error, AuthenticationCommitError::ReservationLost);
        assert!(matches!(
            authority.snapshot(),
            AuthorityState::Authenticating(active) if active == request_id(2)
        ));
    }

    #[test]
    fn only_one_concurrent_authentication_reservation_wins() {
        const WORKERS: usize = 16;
        let authority = Arc::new(AuthorityCell::new());
        let start = Arc::new(Barrier::new(WORKERS));
        let reserved = Arc::new(Barrier::new(WORKERS));

        let handles: Vec<_> = (0..WORKERS)
            .map(|worker| {
                let authority = Arc::clone(&authority);
                let start = Arc::clone(&start);
                let reserved = Arc::clone(&reserved);
                thread::spawn(move || {
                    start.wait();
                    let reservation = authority.begin(request_id(worker as u128));
                    reserved.wait();
                    reservation.is_ok()
                })
            })
            .collect();

        let winners = handles
            .into_iter()
            .map(|handle| handle.join().expect("authentication worker panicked"))
            .filter(|won| *won)
            .count();
        assert_eq!(winners, 1);
        assert!(matches!(authority.snapshot(), AuthorityState::Anonymous));
    }

    #[test]
    fn record_budget_is_atomic_at_the_exact_boundary() {
        const WORKERS: usize = 32;
        const LIMIT: usize = 11;
        let budget = Arc::new(RecordBudget::new(LIMIT));
        let start = Arc::new(Barrier::new(WORKERS));

        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let budget = Arc::clone(&budget);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    budget.try_consume()
                })
            })
            .collect();

        let admitted = handles
            .into_iter()
            .map(|handle| handle.join().expect("record worker panicked"))
            .filter(|admitted| *admitted)
            .count();
        assert_eq!(admitted, LIMIT);
        assert_eq!(budget.used(), LIMIT);
        assert!(budget.is_full());
    }

    #[test]
    fn new_admission_rejects_exact_absolute_expiry_but_held_response_can_finish() {
        let expiry = Instant::now() + Duration::from_secs(30);
        let state = session(
            expiry,
            SessionLimits::new(4, 4, 2),
            Arc::new(GlobalReplayBudget::new(4)),
        );

        assert!(state.accepts_new_requests_at(expiry - Duration::from_nanos(1)));
        let mut response = state
            .begin_unary_response()
            .expect("reserve response before dispatch");
        assert_eq!(
            state.check_new_admission_at(expiry),
            Err(AdmissionError::AbsoluteSessionExpired)
        );
        assert!(state.is_closing());

        state
            .encrypt_unary_response_record(&mut response, &request_id(1), b"late but admitted")
            .expect("held response context remains usable after expiry");
    }

    #[test]
    fn bound_authentication_expiry_is_exact_and_closes_fail_closed() {
        let now = Instant::now();
        let auth_expiry = now + Duration::from_secs(10);
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 4),
            Arc::new(GlobalReplayBudget::new(4)),
        );

        state
            .begin_authentication(request_id(1))
            .expect("reserve auth")
            .commit_at(BoundAuthority::user(Uuid::new_v4(), 3, auth_expiry), now)
            .expect("bind authority");

        assert!(state.accepts_new_requests_at(auth_expiry - Duration::from_nanos(1)));
        let mut response = state
            .begin_unary_response()
            .expect("reserve response before dispatch");
        assert_eq!(
            state.check_new_admission_at(auth_expiry),
            Err(AdmissionError::AuthenticationExpired)
        );
        assert!(state.is_closing());
        state
            .encrypt_unary_response_record(&mut response, &request_id(1), b"admitted response")
            .expect("held response remains usable after auth expiry");
    }

    #[test]
    fn authenticating_blocks_new_admission_and_drop_reopens_anonymous_session() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 4),
            Arc::new(GlobalReplayBudget::new(4)),
        );
        let reservation = state
            .begin_authentication(request_id(1))
            .expect("reserve auth");

        assert_eq!(
            state.check_new_admission_at(now),
            Err(AdmissionError::AuthenticationInProgress)
        );
        assert!(!state.should_retire_at(now));
        drop(reservation);
        assert!(state.accepts_new_requests_at(now));
    }

    #[test]
    fn request_records_count_only_after_successful_aead_and_stop_at_limit() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 2, 4),
            Arc::new(GlobalReplayBudget::new(4)),
        );

        let valid_record = state
            .keys()
            .encrypt_request_record(&state.session_id(), b"authenticated request")
            .expect("encrypt test request");
        assert!(state.decrypt_request_record(&[0u8; 28]).is_err());
        assert_eq!(state.request_record_count(), 0);

        state
            .decrypt_request_record(&valid_record)
            .expect("first authenticated record");
        state
            .decrypt_request_record(&valid_record)
            .expect("second authenticated record");
        assert_eq!(state.request_record_count(), 2);
        assert!(matches!(
            state.decrypt_request_record(&valid_record),
            Err(SessionRecordError::RequestRecordsExhausted)
        ));
        assert!(state.is_exhausted());
        assert!(!state.accepts_new_requests_at(now));
    }

    #[test]
    fn response_record_budget_is_exact_and_fail_closed() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 2),
            Arc::new(GlobalReplayBudget::new(4)),
        );

        let mut first = state
            .begin_unary_response()
            .expect("reserve first response before dispatch");
        let mut second = state
            .begin_unary_response()
            .expect("reserve second response before dispatch");
        assert!(matches!(
            state.begin_unary_response(),
            Err(SessionRecordError::ResponseRecordsExhausted)
        ));
        assert_eq!(state.response_record_count(), 2);
        state
            .encrypt_unary_response_record(&mut first, &request_id(1), b"first")
            .expect("encrypt first reserved response");
        state
            .encrypt_unary_response_record(&mut second, &request_id(2), b"second")
            .expect("encrypt second reserved response");
        assert!(state.is_exhausted());
    }

    #[test]
    fn concurrent_unary_capacity_is_won_before_dispatch() {
        const WORKERS: usize = 16;
        const LIMIT: usize = 3;
        let now = Instant::now();
        let state = Arc::new(session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, LIMIT),
            Arc::new(GlobalReplayBudget::new(4)),
        ));
        let start = Arc::new(Barrier::new(WORKERS));
        let reserved = Arc::new(Barrier::new(WORKERS));

        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let state = Arc::clone(&state);
                let start = Arc::clone(&start);
                let reserved = Arc::clone(&reserved);
                thread::spawn(move || {
                    start.wait();
                    let reservation = state.begin_unary_response();
                    reserved.wait();
                    reservation.is_ok()
                })
            })
            .collect();

        let winners = handles
            .into_iter()
            .map(|handle| handle.join().expect("response reservation worker panicked"))
            .filter(|won| *won)
            .count();
        assert_eq!(winners, LIMIT);
        assert_eq!(state.response_record_count(), 0);
        assert!(state.accepts_new_requests_at(now));
    }

    #[test]
    fn stream_reserves_start_and_terminal_before_dispatch() {
        let now = Instant::now();
        let too_small = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 1),
            Arc::new(GlobalReplayBudget::new(4)),
        );
        assert!(matches!(
            too_small.begin_stream_response(),
            Err(SessionRecordError::ResponseRecordsExhausted)
        ));
        assert_eq!(too_small.response_record_count(), 0);

        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 2),
            Arc::new(GlobalReplayBudget::new(4)),
        );
        let mut reservation = state
            .begin_stream_response()
            .expect("atomically reserve start and terminal records");
        assert_eq!(state.response_record_count(), 2);

        state
            .encrypt_stream_start_record(&mut reservation, &request_id(1), b"start")
            .expect("pre-reserved start remains available");
        assert!(state.is_exhausted());

        state
            .encrypt_stream_terminal_record(&mut reservation, &request_id(1), 1, b"end")
            .expect("pre-reserved terminal remains available at the limit");
        assert_eq!(state.response_record_count(), 2);
    }

    #[test]
    fn dropping_unstarted_stream_returns_both_reservations() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 2),
            Arc::new(GlobalReplayBudget::new(4)),
        );

        let reservation = state
            .begin_stream_response()
            .expect("reserve start and terminal response records");
        assert_eq!(state.response_record_count(), 2);
        drop(reservation);
        assert_eq!(state.response_record_count(), 0);
        assert!(state.accepts_new_requests_at(now));
    }

    #[test]
    fn stream_requires_matching_reservation_start_and_exact_sequence() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 3),
            Arc::new(GlobalReplayBudget::new(4)),
        );
        let other = session(
            now + Duration::from_secs(30),
            SessionLimits::new(4, 4, 3),
            Arc::new(GlobalReplayBudget::new(4)),
        );
        let mut reservation = state
            .begin_stream_response()
            .expect("reserve stream response");

        assert!(matches!(
            state.encrypt_stream_chunk_record(&mut reservation, &request_id(1), 1, b"chunk"),
            Err(SessionRecordError::StreamNotStarted)
        ));
        assert!(matches!(
            state.encrypt_stream_terminal_record(&mut reservation, &request_id(1), 1, b"end"),
            Err(SessionRecordError::StreamNotStarted)
        ));
        assert!(matches!(
            other.encrypt_stream_start_record(&mut reservation, &request_id(1), b"start"),
            Err(SessionRecordError::ResponseReservationMismatch)
        ));

        state
            .encrypt_stream_start_record(&mut reservation, &request_id(1), b"start")
            .expect("encrypt start");
        assert!(matches!(
            state.encrypt_stream_chunk_record(&mut reservation, &request_id(1), 2, b"out of order"),
            Err(SessionRecordError::InvalidStreamSequence)
        ));
        state
            .encrypt_stream_chunk_record(&mut reservation, &request_id(1), 1, b"chunk")
            .expect("encrypt next chunk");
        assert!(matches!(
            state.encrypt_stream_terminal_record(
                &mut reservation,
                &request_id(1),
                3,
                b"out of order end"
            ),
            Err(SessionRecordError::InvalidStreamSequence)
        ));
        state
            .encrypt_stream_terminal_record(&mut reservation, &request_id(1), 2, b"end")
            .expect("encrypt exact next terminal");
        assert_eq!(state.response_record_count(), 3);
    }

    #[test]
    fn replay_exhaustion_closes_only_new_admission_not_an_admitted_response() {
        let now = Instant::now();
        let state = session(
            now + Duration::from_secs(30),
            SessionLimits::new(1, 4, 2),
            Arc::new(GlobalReplayBudget::new(1)),
        );

        assert_eq!(state.claim_request_id(request_id(1)), ReplayClaim::Claimed);
        assert_eq!(
            state.claim_request_id(request_id(2)),
            ReplayClaim::Exhausted
        );
        assert_eq!(state.replay_id_count(), 1);
        let mut response = state
            .begin_unary_response()
            .expect("reserve response for admitted request");
        assert!(!state.accepts_new_requests_at(now));
        state
            .encrypt_unary_response_record(&mut response, &request_id(1), b"admitted response")
            .expect("already-admitted response remains possible");
    }
}
