use super::session::V2SessionState;
use clru::CLruCache;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// How a successfully inserted v2 session affected cache residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum V2SessionInsertOutcome {
    Inserted,
    EvictedLeastRecentlyUsed,
}

/// Why a v2 session could not be inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum V2SessionInsertError {
    DuplicateSession,
    AllSessionsLeased,
}

/// A rejected insertion that retains the caller's session allocation.
///
/// Keeping the rejected `Arc` in the return value prevents key destruction or
/// a large replay-registry drop while a future outer cache mutex is held.
#[must_use = "release the rejected session after releasing the cache lock"]
pub(crate) struct RejectedV2Session {
    reason: V2SessionInsertError,
    session: Arc<V2SessionState>,
}

impl fmt::Debug for RejectedV2Session {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedV2Session")
            .field("reason", &self.reason)
            .field("session", &"[REDACTED]")
            .finish()
    }
}

impl RejectedV2Session {
    pub(crate) fn reason(&self) -> V2SessionInsertError {
        self.reason
    }

    pub(crate) fn into_session(self) -> Arc<V2SessionState> {
        self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport_v2::crypto::{DirectionalKeys, SessionMaster};
    use crate::transport_v2::envelope::RequestId;
    use crate::transport_v2::session::{
        GlobalReplayBudget, ReplayClaim, DEFAULT_GLOBAL_REPLAY_IDS,
    };
    use std::time::Duration;

    fn cache(capacity: usize) -> V2SessionCache {
        V2SessionCache::new(NonZeroUsize::new(capacity).expect("test capacity must be nonzero"))
    }

    fn session(session_id: Uuid, absolute_expires_at: Instant) -> Arc<V2SessionState> {
        session_with_global_budget(
            session_id,
            absolute_expires_at,
            Arc::new(GlobalReplayBudget::new(DEFAULT_GLOBAL_REPLAY_IDS)),
        )
    }

    fn session_with_global_budget(
        session_id: Uuid,
        absolute_expires_at: Instant,
        global_replay_budget: Arc<GlobalReplayBudget>,
    ) -> Arc<V2SessionState> {
        let master = SessionMaster::from_bytes([session_id.as_bytes()[15]; 32]);
        let keys = DirectionalKeys::derive(&master).expect("derive test keys");
        Arc::new(V2SessionState::new(
            session_id,
            keys,
            absolute_expires_at,
            global_replay_budget,
        ))
    }

    fn insert_new(cache: &mut V2SessionCache, session: Arc<V2SessionState>) {
        let insertion = cache.insert(session).expect("insert test session");
        assert_eq!(insertion.outcome(), V2SessionInsertOutcome::Inserted);
        assert!(insertion.retired.is_empty());
    }

    #[test]
    fn exact_absolute_expiry_boundary_rejects_and_cleans_up() {
        let start = Instant::now();
        let expiry = start + Duration::from_secs(10);
        let session_id = Uuid::new_v4();
        let mut cache = cache(1);
        insert_new(&mut cache, session(session_id, expiry));

        let before_expiry = expiry
            .checked_sub(Duration::from_nanos(1))
            .expect("test instant supports subtraction");
        assert!(cache.acquire_at(&session_id, before_expiry).is_some());
        assert!(cache.acquire_at(&session_id, expiry).is_none());

        let retired = cache.cleanup_at(expiry);
        assert_eq!(retired.removed_count(), 1);
        assert!(!cache.contains(&session_id));
    }

    #[test]
    fn held_old_lease_blocks_removal_but_not_expiry_rejection() {
        let start = Instant::now();
        let expiry = start + Duration::from_secs(10);
        let session_id = Uuid::new_v4();
        let mut cache = cache(1);
        insert_new(&mut cache, session(session_id, expiry));

        let lease = cache
            .acquire_at(&session_id, start)
            .expect("pre-expiry lease");
        assert!(cache.acquire_at(&session_id, expiry).is_none());

        let retired = cache.cleanup_at(expiry);
        assert!(retired.is_empty());
        assert!(cache.contains(&session_id));

        // Expiry stops new admission; it does not invalidate response keys on
        // the already-admitted exact session.
        assert_eq!(lease.state().session_id(), session_id);
        let _response_keys = lease.state().keys();

        drop(lease);
        let retired = cache.cleanup_at(expiry);
        assert_eq!(retired.removed_count(), 1);
        assert!(!cache.contains(&session_id));
    }

    #[test]
    fn acquire_does_not_promote_but_mark_admitted_does() {
        let expiry = Instant::now() + Duration::from_secs(60);
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let third_id = Uuid::new_v4();

        let mut unadmitted = cache(2);
        insert_new(&mut unadmitted, session(first_id, expiry));
        insert_new(&mut unadmitted, session(second_id, expiry));
        let lease = unadmitted
            .acquire_at(&first_id, Instant::now())
            .expect("lease first session");
        drop(lease);
        let insertion = unadmitted
            .insert(session(third_id, expiry))
            .expect("insert third session");
        assert_eq!(
            insertion.outcome(),
            V2SessionInsertOutcome::EvictedLeastRecentlyUsed
        );
        assert_eq!(insertion.retired.sessions[0].session_id(), first_id);

        let mut admitted = cache(2);
        insert_new(&mut admitted, session(first_id, expiry));
        insert_new(&mut admitted, session(second_id, expiry));
        let lease = admitted
            .acquire_at(&first_id, Instant::now())
            .expect("lease first session");
        assert!(admitted.mark_admitted(&lease));
        drop(lease);
        let insertion = admitted
            .insert(session(third_id, expiry))
            .expect("insert third session");
        assert_eq!(
            insertion.outcome(),
            V2SessionInsertOutcome::EvictedLeastRecentlyUsed
        );
        assert_eq!(insertion.retired.sessions[0].session_id(), second_id);
    }

    #[test]
    fn full_cache_with_only_leased_sessions_reports_overload() {
        let now = Instant::now();
        let expiry = now + Duration::from_secs(60);
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let rejected_id = Uuid::new_v4();
        let mut cache = cache(2);
        insert_new(&mut cache, session(first_id, expiry));
        insert_new(&mut cache, session(second_id, expiry));

        let first_lease = cache.acquire_at(&first_id, now).expect("first lease");
        let second_lease = cache.acquire_at(&second_id, now).expect("second lease");
        let rejected = match cache.insert(session(rejected_id, expiry)) {
            Ok(_) => panic!("all leased cache must reject insertion"),
            Err(rejected) => rejected,
        };

        assert_eq!(rejected.reason(), V2SessionInsertError::AllSessionsLeased);
        assert_eq!(rejected.into_session().session_id(), rejected_id);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&first_id));
        assert!(cache.contains(&second_id));

        drop(first_lease);
        drop(second_lease);
    }

    #[test]
    fn leased_lru_is_preserved_and_retired_arc_drops_later() {
        let now = Instant::now();
        let expiry = now + Duration::from_secs(60);
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let third_id = Uuid::new_v4();
        let mut cache = cache(2);
        insert_new(&mut cache, session(first_id, expiry));

        let second = session(second_id, expiry);
        let evicted_lifetime = Arc::downgrade(&second);
        insert_new(&mut cache, second);

        let first_lease = cache.acquire_at(&first_id, now).expect("first lease");
        let insertion = cache
            .insert(session(third_id, expiry))
            .expect("one unleased session leaves capacity");

        assert_eq!(
            insertion.outcome(),
            V2SessionInsertOutcome::EvictedLeastRecentlyUsed
        );
        assert_eq!(insertion.retired.sessions[0].session_id(), second_id);
        assert!(cache.contains(&first_id));
        assert!(cache.contains(&third_id));
        assert!(evicted_lifetime.upgrade().is_some());

        let retired = insertion.into_retired();
        assert!(evicted_lifetime.upgrade().is_some());
        drop(retired);
        assert!(evicted_lifetime.upgrade().is_none());
        drop(first_lease);
    }

    #[test]
    fn duplicate_insert_never_replaces_the_original_arc() {
        let now = Instant::now();
        let expiry = now + Duration::from_secs(60);
        let session_id = Uuid::new_v4();
        let original = session(session_id, expiry);
        let original_pointer = Arc::as_ptr(&original);
        let mut cache = cache(1);
        insert_new(&mut cache, original);

        let replacement = session(session_id, expiry);
        let replacement_pointer = Arc::as_ptr(&replacement);
        let rejected = match cache.insert(replacement) {
            Ok(_) => panic!("duplicate session must not replace the original"),
            Err(rejected) => rejected,
        };
        assert_eq!(rejected.reason(), V2SessionInsertError::DuplicateSession);
        assert_eq!(Arc::as_ptr(&rejected.into_session()), replacement_pointer);

        let lease = cache.acquire_at(&session_id, now).expect("original lease");
        assert_eq!(lease.state() as *const V2SessionState, original_pointer);
    }

    #[test]
    fn cleanup_removes_expired_closed_and_exhausted_unleased_sessions() {
        let now = Instant::now();
        let cutoff = now + Duration::from_secs(10);
        let expired_id = Uuid::new_v4();
        let closed_id = Uuid::new_v4();
        let exhausted_id = Uuid::new_v4();
        let live_id = Uuid::new_v4();
        let mut cache = cache(4);

        insert_new(&mut cache, session(expired_id, cutoff));

        let closed = session(closed_id, cutoff + Duration::from_secs(60));
        closed.close();
        insert_new(&mut cache, closed);

        let exhausted = session_with_global_budget(
            exhausted_id,
            cutoff + Duration::from_secs(60),
            Arc::new(GlobalReplayBudget::new(0)),
        );
        assert_eq!(
            exhausted.claim_request_id(RequestId::from_bytes([1; 16])),
            ReplayClaim::Exhausted
        );
        assert!(exhausted.is_exhausted());
        insert_new(&mut cache, exhausted);

        insert_new(
            &mut cache,
            session(live_id, cutoff + Duration::from_secs(60)),
        );

        let retired = cache.cleanup_at(cutoff);
        assert_eq!(retired.removed_count(), 3);
        let retired_ids = retired
            .sessions
            .iter()
            .map(|state| state.session_id())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            retired_ids,
            std::collections::HashSet::from([expired_id, closed_id, exhausted_id])
        );
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&live_id));
    }

    #[test]
    fn lease_and_clone_preserve_exact_arc_for_response_continuity() {
        let now = Instant::now();
        let session_id = Uuid::new_v4();
        let state = session(session_id, now + Duration::from_secs(60));
        let expected = Arc::as_ptr(&state);
        let mut cache = cache(1);
        insert_new(&mut cache, state);

        let lease = cache.acquire_at(&session_id, now).expect("lease session");
        let cloned = lease.clone();
        assert_eq!(lease.state() as *const V2SessionState, expected);
        assert_eq!(cloned.state() as *const V2SessionState, expected);
        assert!(cache.mark_admitted(&lease));
    }
}

/// Sessions detached from the cache for destruction outside its future mutex.
#[must_use = "keep retired sessions alive until after releasing the cache lock"]
pub(crate) struct RetiredV2Sessions {
    sessions: Vec<Arc<V2SessionState>>,
}

impl RetiredV2Sessions {
    fn empty() -> Self {
        Self {
            sessions: Vec::new(),
        }
    }

    pub(crate) fn removed_count(&self) -> usize {
        self.sessions.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

/// The result of inserting a session, including any cache-owned state that was
/// detached and must be dropped after the caller releases its outer lock.
#[must_use = "release retired sessions after releasing the cache lock"]
pub(crate) struct InsertedV2Session {
    outcome: V2SessionInsertOutcome,
    retired: RetiredV2Sessions,
}

impl InsertedV2Session {
    pub(crate) fn outcome(&self) -> V2SessionInsertOutcome {
        self.outcome
    }

    pub(crate) fn into_retired(self) -> RetiredV2Sessions {
        self.retired
    }
}

/// A request-lifetime reference to the exact cryptographic session selected at
/// admission. Cloning the lease keeps that same state alive for response bodies
/// and streams without a second lookup by attacker-visible session UUID.
#[derive(Clone)]
pub(crate) struct V2SessionLease {
    session_id: Uuid,
    state: Arc<V2SessionState>,
}

impl V2SessionLease {
    pub(crate) fn state(&self) -> &V2SessionState {
        &self.state
    }
}

/// A fixed-capacity v2-only session cache.
///
/// The cache deliberately has no global instance and exposes no resizing. It
/// is compiled route-free in this stack layer so allocating its production
/// capacity cannot change startup memory or transport-v1 behavior.
pub(crate) struct V2SessionCache {
    entries: CLruCache<Uuid, Arc<V2SessionState>, RandomState>,
}

impl V2SessionCache {
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            entries: CLruCache::with_memory(capacity, capacity.get()),
        }
    }

    /// Inserts without replacing an existing session ID.
    ///
    /// At capacity, the least-recently-used session without any external `Arc`
    /// is detached. A cache containing only leased sessions rejects admission
    /// rather than invalidating in-flight response encryption.
    pub(crate) fn insert(
        &mut self,
        session: Arc<V2SessionState>,
    ) -> Result<InsertedV2Session, RejectedV2Session> {
        let session_id = session.session_id();
        if self.entries.peek(&session_id).is_some() {
            return Err(RejectedV2Session {
                reason: V2SessionInsertError::DuplicateSession,
                session,
            });
        }

        let mut retired = RetiredV2Sessions::empty();
        let outcome = if self.entries.is_full() {
            // Scan from LRU to MRU without altering recency. Any external Arc
            // may be an active request/response lease, so only the cache's sole
            // Arc is safe to detach.
            let eviction_id = self
                .entries
                .iter()
                .rev()
                .find(|(_, state)| Arc::strong_count(state) == 1)
                .map(|(candidate_id, _)| *candidate_id);

            let Some(eviction_id) = eviction_id else {
                return Err(RejectedV2Session {
                    reason: V2SessionInsertError::AllSessionsLeased,
                    session,
                });
            };

            let evicted = self
                .entries
                .pop(&eviction_id)
                .expect("selected unleased v2 session must remain present");
            debug_assert_eq!(Arc::strong_count(&evicted), 1);
            retired.sessions.push(evicted);
            V2SessionInsertOutcome::EvictedLeastRecentlyUsed
        } else {
            V2SessionInsertOutcome::Inserted
        };

        let previous = self.entries.put(session_id, session);
        debug_assert!(previous.is_none());
        debug_assert!(self.entries.len() <= self.entries.capacity());

        Ok(InsertedV2Session { outcome, retired })
    }

    /// Leases the exact session state without refreshing its LRU position.
    ///
    /// Admission is checked on every call, even when an older request still
    /// holds a lease. Thus `now >= absolute_expiry`, authentication expiry,
    /// exhaustion, or closing rejects new work while the older lease remains
    /// available to finish its already-admitted response.
    pub(crate) fn acquire_at(&self, session_id: &Uuid, now: Instant) -> Option<V2SessionLease> {
        let state = self.entries.peek(session_id)?;
        if !state.accepts_new_requests_at(now) {
            return None;
        }

        Some(V2SessionLease {
            session_id: *session_id,
            state: Arc::clone(state),
        })
    }

    /// Marks an exact leased session as recently used only after the request
    /// has passed AEAD, structural validation, and the replay gate.
    ///
    /// Pointer equality prevents a stale lease from promoting a different
    /// session that happens to reuse the same UUID.
    pub(crate) fn mark_admitted(&mut self, lease: &V2SessionLease) -> bool {
        let same_session = self
            .entries
            .peek(&lease.session_id)
            .is_some_and(|cached| Arc::ptr_eq(cached, &lease.state));
        if !same_session {
            return false;
        }

        self.entries.get(&lease.session_id).is_some()
    }

    /// Detaches terminal or expired sessions that have no active lease.
    ///
    /// Session state can become closing or exhausted independently of LRU
    /// order, so cleanup examines every v2 entry. It never touches the
    /// transport-v1 cache.
    pub(crate) fn cleanup_at(&mut self, now: Instant) -> RetiredV2Sessions {
        let removable_ids = self
            .entries
            .iter()
            .filter(|(_, state)| Arc::strong_count(state) == 1 && state.should_retire_at(now))
            .map(|(session_id, _)| *session_id)
            .collect::<Vec<_>>();

        let mut retired = RetiredV2Sessions {
            sessions: Vec::with_capacity(removable_ids.len()),
        };
        for session_id in removable_ids {
            let session = self
                .entries
                .pop(&session_id)
                .expect("selected removable v2 session must remain present");
            debug_assert_eq!(Arc::strong_count(&session), 1);
            retired.sessions.push(session);
        }
        retired
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn contains(&self, session_id: &Uuid) -> bool {
        self.entries.peek(session_id).is_some()
    }
}
