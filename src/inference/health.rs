//! Enclave-local health classification for inference routes.
//!
//! Stack 5 introduced this state observationally. Stack 6 consumes one coherent
//! snapshot only for the explicit GLM 5.3 canary; every other route remains
//! observational until its own rollout boundary.

use super::{AttemptFailureKind, AttemptTerminal, RouteKey};
use crate::provider_registry::{ProviderId, ProviderRegistry, RateLimitScope, PROVIDER_REGISTRY};
use std::collections::{HashMap, VecDeque};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub(crate) const SHADOW_HEALTH_POLICY_VERSION: &str = "routing-v2-health-shadow-v1";

// These are observation-only v1 hypotheses, not active product policy. Stack 5
// intentionally versions and exercises them against real incidents before a
// later stack may use any disposition during route selection.
const ROUTE_FAILURE_WINDOW: Duration = Duration::from_secs(60);
const ROUTE_FAILURES_TO_OPEN: u8 = 3;
const ROUTE_OPEN_COOLDOWN: Duration = Duration::from_secs(30);
// A missing or zero upstream hint must not turn an active circuit into a hot
// retry loop. Thirty seconds matches the route-health cooldown while remaining
// bounded well below the one-hour maximum accepted from providers.
pub(crate) const MIN_CAPACITY_COOLDOWN: Duration = Duration::from_secs(30);
const MAX_CAPACITY_COOLDOWN: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadowObservationMode {
    Update,
    TelemetryOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CapacityPoolKey {
    ProviderModel(RouteKey),
    ProviderAccount(ProviderId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadowDisposition {
    Healthy,
    Watch { consecutive_failures: u8 },
    WouldOpen { remaining: Duration },
    WouldProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShadowRouteSnapshot {
    pub(crate) route_health: ShadowDisposition,
    pub(crate) deployment_capacity: ShadowDisposition,
    pub(crate) rate_limit_capacity: ShadowDisposition,
    pub(crate) effective: ShadowDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadowSignal {
    Completed,
    CapacityRejected {
        status: u16,
    },
    RouteFailure {
        kind: AttemptFailureKind,
        status: Option<u16>,
    },
    Neutral {
        kind: AttemptFailureKind,
        status: Option<u16>,
    },
    UnknownRoute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShadowObservationReport {
    pub(crate) policy_version: &'static str,
    pub(crate) mode: ShadowObservationMode,
    pub(crate) route: RouteKey,
    pub(crate) signal: ShadowSignal,
    pub(crate) capacity_pool: Option<CapacityPoolKey>,
    pub(crate) snapshot: Option<ShadowRouteSnapshot>,
    pub(crate) mutated: bool,
}

#[derive(Debug, Clone, Copy)]
struct ShadowHealthPolicy {
    route_failure_window: Duration,
    route_failures_to_open: u8,
    route_open_cooldown: Duration,
    minimum_capacity_cooldown: Duration,
    max_capacity_cooldown: Duration,
}

impl Default for ShadowHealthPolicy {
    fn default() -> Self {
        Self {
            route_failure_window: ROUTE_FAILURE_WINDOW,
            route_failures_to_open: ROUTE_FAILURES_TO_OPEN,
            route_open_cooldown: ROUTE_OPEN_COOLDOWN,
            minimum_capacity_cooldown: MIN_CAPACITY_COOLDOWN,
            max_capacity_cooldown: MAX_CAPACITY_COOLDOWN,
        }
    }
}

#[derive(Debug, Default)]
struct RouteHealthState {
    failure_times: VecDeque<Instant>,
    open_until: Option<Instant>,
}

#[derive(Debug, Default)]
struct CapacityState {
    open_until: Option<Instant>,
}

#[derive(Debug)]
struct ShadowHealthInner {
    route_health: HashMap<RouteKey, RouteHealthState>,
    capacity: HashMap<CapacityPoolKey, CapacityState>,
}

/// Fixed-topology, enclave-local shadow state.
///
/// Every key is pre-populated from the static provider registry. Observations
/// for unknown routes are reported but can never grow the maps.
#[derive(Debug)]
pub(crate) struct ShadowHealthState {
    policy: ShadowHealthPolicy,
    rate_limit_pools: HashMap<RouteKey, CapacityPoolKey>,
    inner: Mutex<ShadowHealthInner>,
    #[cfg(test)]
    observation_count: AtomicUsize,
}

impl Default for ShadowHealthState {
    fn default() -> Self {
        Self::new(&PROVIDER_REGISTRY)
    }
}

impl ShadowHealthState {
    pub(crate) fn new(registry: &'static ProviderRegistry) -> Self {
        Self::new_with_policy(registry, ShadowHealthPolicy::default())
    }

    fn new_with_policy(registry: &'static ProviderRegistry, policy: ShadowHealthPolicy) -> Self {
        let mut route_health = HashMap::new();
        let mut capacity = HashMap::new();
        let mut rate_limit_pools = HashMap::new();

        for model in registry.completion_models() {
            for route in model.routes {
                let route_key = RouteKey {
                    provider: route.provider,
                    provider_model_id: route.provider_model_id.to_string(),
                };
                let deployment_pool = CapacityPoolKey::ProviderModel(route_key.clone());
                let rate_limit_pool = match route.rate_limit_scope {
                    RateLimitScope::ProviderModel => deployment_pool.clone(),
                    RateLimitScope::ProviderAccount => {
                        CapacityPoolKey::ProviderAccount(route.provider)
                    }
                };

                route_health.entry(route_key.clone()).or_default();
                capacity.entry(deployment_pool).or_default();
                capacity.entry(rate_limit_pool.clone()).or_default();
                rate_limit_pools.entry(route_key).or_insert(rate_limit_pool);
            }
        }

        Self {
            policy,
            rate_limit_pools,
            inner: Mutex::new(ShadowHealthInner {
                route_health,
                capacity,
            }),
            #[cfg(test)]
            observation_count: AtomicUsize::new(0),
        }
    }

    pub(crate) fn observe_terminal(
        &self,
        terminal: &AttemptTerminal,
        mode: ShadowObservationMode,
    ) -> ShadowObservationReport {
        let mut inner = self.lock();
        let now = Instant::now();
        self.observe_terminal_locked(&mut inner, terminal, mode, now)
    }

    pub(crate) fn snapshot(&self, route: &RouteKey) -> Option<ShadowRouteSnapshot> {
        let inner = self.lock();
        self.snapshot_locked(&inner, route, Instant::now())
    }

    /// Returns a point-in-time snapshot for every requested route while holding
    /// the state lock once. Active selection must not combine observations from
    /// different instants when deciding whether all canary routes are open.
    pub(crate) fn snapshot_routes(&self, routes: &[RouteKey]) -> Option<Vec<ShadowRouteSnapshot>> {
        let inner = self.lock();
        let now = Instant::now();
        routes
            .iter()
            .map(|route| self.snapshot_locked(&inner, route, now))
            .collect()
    }

    fn observe_terminal_locked(
        &self,
        inner: &mut ShadowHealthInner,
        terminal: &AttemptTerminal,
        mode: ShadowObservationMode,
        now: Instant,
    ) -> ShadowObservationReport {
        #[cfg(test)]
        self.observation_count.fetch_add(1, Ordering::Relaxed);

        let route = terminal.attempt().route.route_key();
        let Some(rate_limit_pool) = self.rate_limit_pools.get(&route).cloned() else {
            return ShadowObservationReport {
                policy_version: SHADOW_HEALTH_POLICY_VERSION,
                mode,
                route,
                signal: ShadowSignal::UnknownRoute,
                capacity_pool: None,
                snapshot: None,
                mutated: false,
            };
        };

        let mut signal = ShadowSignal::Completed;
        let mut capacity_pool = None;
        let mut mutation = Mutation::Completed;

        if let AttemptTerminal::Failed { failure, .. } = terminal {
            let status = failure.status;
            if failure.kind == AttemptFailureKind::CapacityRejected {
                match status {
                    Some(429) => {
                        signal = ShadowSignal::CapacityRejected { status: 429 };
                        capacity_pool = Some(rate_limit_pool.clone());
                        mutation = Mutation::Capacity {
                            pool: rate_limit_pool.clone(),
                            retry_after: failure.retry_after,
                        };
                    }
                    Some(status @ (503 | 529)) => {
                        let pool = CapacityPoolKey::ProviderModel(route.clone());
                        signal = ShadowSignal::CapacityRejected { status };
                        capacity_pool = Some(pool.clone());
                        mutation = Mutation::Capacity {
                            pool,
                            retry_after: failure.retry_after,
                        };
                    }
                    _ => {
                        signal = ShadowSignal::Neutral {
                            kind: failure.kind,
                            status,
                        };
                        mutation = Mutation::None;
                    }
                }
            } else if failure.kind == AttemptFailureKind::HttpStatus {
                if status.is_some_and(|status| (500..=599).contains(&status)) {
                    signal = ShadowSignal::RouteFailure {
                        kind: failure.kind,
                        status,
                    };
                    mutation = Mutation::RouteFailure;
                } else {
                    signal = ShadowSignal::Neutral {
                        kind: failure.kind,
                        status,
                    };
                    mutation = Mutation::None;
                }
            } else if is_route_health_failure(failure.kind) {
                signal = ShadowSignal::RouteFailure {
                    kind: failure.kind,
                    status,
                };
                mutation = Mutation::RouteFailure;
            } else {
                signal = ShadowSignal::Neutral {
                    kind: failure.kind,
                    status,
                };
                mutation = Mutation::None;
            }
        }

        let mutated = mode == ShadowObservationMode::Update && mutation != Mutation::None;
        if mutated {
            self.apply_mutation(inner, &route, &rate_limit_pool, mutation, now);
        }

        ShadowObservationReport {
            policy_version: SHADOW_HEALTH_POLICY_VERSION,
            mode,
            route: route.clone(),
            signal,
            capacity_pool,
            snapshot: self.snapshot_locked(inner, &route, now),
            mutated,
        }
    }

    fn apply_mutation(
        &self,
        inner: &mut ShadowHealthInner,
        route: &RouteKey,
        rate_limit_pool: &CapacityPoolKey,
        mutation: Mutation,
        now: Instant,
    ) {
        match mutation {
            Mutation::Completed => {
                if let Some(state) = inner.route_health.get_mut(route) {
                    apply_route_success(state, now);
                }

                let deployment_pool = CapacityPoolKey::ProviderModel(route.clone());
                if let Some(state) = inner.capacity.get_mut(&deployment_pool) {
                    apply_capacity_success(state, now);
                }
                if rate_limit_pool != &deployment_pool {
                    if let Some(state) = inner.capacity.get_mut(rate_limit_pool) {
                        apply_capacity_success(state, now);
                    }
                }
            }
            Mutation::Capacity { pool, retry_after } => {
                if let Some(state) = inner.capacity.get_mut(&pool) {
                    let cooldown = retry_after
                        .unwrap_or(self.policy.minimum_capacity_cooldown)
                        .max(self.policy.minimum_capacity_cooldown)
                        .min(self.policy.max_capacity_cooldown);
                    extend_open_until(&mut state.open_until, now, cooldown);
                }
            }
            Mutation::RouteFailure => {
                if let Some(state) = inner.route_health.get_mut(route) {
                    apply_route_failure(state, self.policy, now);
                }
            }
            Mutation::None => {}
        }
    }

    fn snapshot_locked(
        &self,
        inner: &ShadowHealthInner,
        route: &RouteKey,
        now: Instant,
    ) -> Option<ShadowRouteSnapshot> {
        let route_health = route_health_disposition(
            inner.route_health.get(route)?,
            self.policy.route_failure_window,
            now,
        );
        let deployment_pool = CapacityPoolKey::ProviderModel(route.clone());
        let deployment_capacity = capacity_disposition(inner.capacity.get(&deployment_pool)?, now);
        let rate_limit_pool = self.rate_limit_pools.get(route)?;
        let rate_limit_capacity = capacity_disposition(inner.capacity.get(rate_limit_pool)?, now);
        let effective =
            strongest_disposition([route_health, deployment_capacity, rate_limit_capacity]);

        Some(ShadowRouteSnapshot {
            route_health,
            deployment_capacity,
            rate_limit_capacity,
            effective,
        })
    }

    fn lock(&self) -> MutexGuard<'_, ShadowHealthInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    fn with_policy(policy: ShadowHealthPolicy) -> Self {
        Self::new_with_policy(&PROVIDER_REGISTRY, policy)
    }

    #[cfg(test)]
    fn observe_terminal_at(
        &self,
        terminal: &AttemptTerminal,
        mode: ShadowObservationMode,
        now: Instant,
    ) -> ShadowObservationReport {
        let mut inner = self.lock();
        self.observe_terminal_locked(&mut inner, terminal, mode, now)
    }

    #[cfg(test)]
    fn snapshot_at(&self, route: &RouteKey, now: Instant) -> Option<ShadowRouteSnapshot> {
        let inner = self.lock();
        self.snapshot_locked(&inner, route, now)
    }

    #[cfg(test)]
    fn cardinality(&self) -> (usize, usize) {
        let inner = self.lock();
        (inner.route_health.len(), inner.capacity.len())
    }

    #[cfg(test)]
    pub(crate) fn observation_count(&self) -> usize {
        self.observation_count.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mutation {
    Completed,
    Capacity {
        pool: CapacityPoolKey,
        retry_after: Option<Duration>,
    },
    RouteFailure,
    None,
}

fn is_route_health_failure(kind: AttemptFailureKind) -> bool {
    match kind {
        AttemptFailureKind::Connect
        | AttemptFailureKind::Transport
        | AttemptFailureKind::ResponseStartTimeout
        | AttemptFailureKind::ResponseBody
        | AttemptFailureKind::InvalidResponse
        | AttemptFailureKind::UpstreamResponseError
        | AttemptFailureKind::UpstreamStreamError
        | AttemptFailureKind::StreamTimeout
        | AttemptFailureKind::UnexpectedEof => true,
        AttemptFailureKind::ProviderUnavailable
        | AttemptFailureKind::RequestBuild
        | AttemptFailureKind::HttpStatus
        | AttemptFailureKind::CapacityRejected
        | AttemptFailureKind::ConsumerDropped => false,
    }
}

fn apply_route_failure(state: &mut RouteHealthState, policy: ShadowHealthPolicy, now: Instant) {
    if state.open_until.is_some_and(|until| now < until) {
        extend_open_until(&mut state.open_until, now, policy.route_open_cooldown);
        return;
    }

    state.failure_times.retain(|failure_at| {
        now.checked_duration_since(*failure_at)
            .is_none_or(|age| age <= policy.route_failure_window)
    });
    state.failure_times.push_back(now);

    // The queue never needs more entries than the threshold. Keeping only the
    // bounded recent tail makes the fixed-topology state safe even if a future
    // shadow policy raises its observation volume.
    let threshold = usize::from(policy.route_failures_to_open.max(1));
    while state.failure_times.len() > threshold {
        state.failure_times.pop_front();
    }

    if state.failure_times.len() >= threshold {
        extend_open_until(&mut state.open_until, now, policy.route_open_cooldown);
    } else {
        state.open_until = None;
    }
}

fn apply_route_success(state: &mut RouteHealthState, now: Instant) {
    if state.open_until.is_some_and(|until| now < until) {
        return;
    }
    state.failure_times.clear();
    state.open_until = None;
}

fn apply_capacity_success(state: &mut CapacityState, now: Instant) {
    if state.open_until.is_some_and(|until| now >= until) {
        state.open_until = None;
    }
}

fn extend_open_until(open_until: &mut Option<Instant>, now: Instant, cooldown: Duration) {
    let candidate = now.checked_add(cooldown).unwrap_or(now);
    *open_until = Some(open_until.map_or(candidate, |current| current.max(candidate)));
}

fn route_health_disposition(
    state: &RouteHealthState,
    failure_window: Duration,
    now: Instant,
) -> ShadowDisposition {
    if let Some(until) = state.open_until {
        return if now < until {
            ShadowDisposition::WouldOpen {
                remaining: until.duration_since(now),
            }
        } else {
            ShadowDisposition::WouldProbe
        };
    }

    let current_failures = state
        .failure_times
        .iter()
        .filter(|failure_at| {
            now.checked_duration_since(**failure_at)
                .is_none_or(|age| age <= failure_window)
        })
        .count();
    if current_failures > 0 {
        ShadowDisposition::Watch {
            consecutive_failures: u8::try_from(current_failures).unwrap_or(u8::MAX),
        }
    } else {
        ShadowDisposition::Healthy
    }
}

fn capacity_disposition(state: &CapacityState, now: Instant) -> ShadowDisposition {
    match state.open_until {
        Some(until) if now < until => ShadowDisposition::WouldOpen {
            remaining: until.duration_since(now),
        },
        Some(_) => ShadowDisposition::WouldProbe,
        None => ShadowDisposition::Healthy,
    }
}

fn strongest_disposition(
    dispositions: impl IntoIterator<Item = ShadowDisposition>,
) -> ShadowDisposition {
    dispositions
        .into_iter()
        .fold(ShadowDisposition::Healthy, stronger_disposition)
}

fn stronger_disposition(left: ShadowDisposition, right: ShadowDisposition) -> ShadowDisposition {
    match (left, right) {
        (
            ShadowDisposition::WouldOpen {
                remaining: left_remaining,
            },
            ShadowDisposition::WouldOpen {
                remaining: right_remaining,
            },
        ) => ShadowDisposition::WouldOpen {
            remaining: left_remaining.max(right_remaining),
        },
        (left @ ShadowDisposition::WouldOpen { .. }, _) => left,
        (_, right @ ShadowDisposition::WouldOpen { .. }) => right,
        (ShadowDisposition::WouldProbe, _) | (_, ShadowDisposition::WouldProbe) => {
            ShadowDisposition::WouldProbe
        }
        (
            ShadowDisposition::Watch {
                consecutive_failures: left_failures,
            },
            ShadowDisposition::Watch {
                consecutive_failures: right_failures,
            },
        ) => ShadowDisposition::Watch {
            consecutive_failures: left_failures.max(right_failures),
        },
        (left @ ShadowDisposition::Watch { .. }, _) => left,
        (_, right @ ShadowDisposition::Watch { .. }) => right,
        (ShadowDisposition::Healthy, ShadowDisposition::Healthy) => ShadowDisposition::Healthy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        AttemptFailure, AttemptStage, CompletionEvidence, InferenceExecution, ReplaySafety,
        RouteIdentity,
    };
    use crate::provider_registry::RouteSelectionSource;
    use std::sync::Arc;
    use std::thread;

    fn route(provider: ProviderId, public_model: &str, provider_model: &str) -> RouteIdentity {
        RouteIdentity::new(
            provider,
            public_model,
            provider_model,
            public_model,
            RouteSelectionSource::StaticSplit,
            None,
        )
    }

    fn completed(route: RouteIdentity) -> AttemptTerminal {
        AttemptTerminal::Completed {
            attempt: InferenceExecution {
                request_id: super::super::InferenceRequestId::new(),
                execution_id: super::super::InferenceExecutionId::new(),
            }
            .begin_attempt(route),
            evidence: CompletionEvidence::ProviderDone,
        }
    }

    fn failed(
        route: RouteIdentity,
        kind: AttemptFailureKind,
        status: Option<u16>,
        retry_after: Option<Duration>,
    ) -> AttemptTerminal {
        let mut failure = AttemptFailure::new(
            kind,
            AttemptStage::AwaitingResponse,
            ReplaySafety::NotProvenPreAcceptance,
        );
        failure.status = status;
        failure.retry_after = retry_after;
        AttemptTerminal::Failed {
            attempt: InferenceExecution {
                request_id: super::super::InferenceRequestId::new(),
                execution_id: super::super::InferenceExecutionId::new(),
            }
            .begin_attempt(route),
            failure,
        }
    }

    fn test_policy() -> ShadowHealthPolicy {
        ShadowHealthPolicy {
            route_failure_window: Duration::from_secs(10),
            route_failures_to_open: 3,
            route_open_cooldown: Duration::from_secs(5),
            minimum_capacity_cooldown: Duration::from_secs(4),
            max_capacity_cooldown: Duration::from_secs(10),
        }
    }

    #[test]
    fn classifier_is_exhaustive_and_only_provider_faults_penalize_health() {
        let state = ShadowHealthState::with_policy(test_policy());
        let now = Instant::now();
        let kimi = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");

        let route_failures = [
            AttemptFailureKind::Connect,
            AttemptFailureKind::Transport,
            AttemptFailureKind::ResponseStartTimeout,
            AttemptFailureKind::ResponseBody,
            AttemptFailureKind::InvalidResponse,
            AttemptFailureKind::UpstreamResponseError,
            AttemptFailureKind::UpstreamStreamError,
            AttemptFailureKind::StreamTimeout,
            AttemptFailureKind::UnexpectedEof,
        ];
        for kind in route_failures {
            let report = state.observe_terminal_at(
                &failed(kimi.clone(), kind, None, None),
                ShadowObservationMode::TelemetryOnly,
                now,
            );
            assert!(matches!(
                report.signal,
                ShadowSignal::RouteFailure {
                    kind: observed,
                    status: None
                } if observed == kind
            ));
            assert!(!report.mutated);
        }

        for kind in [
            AttemptFailureKind::ProviderUnavailable,
            AttemptFailureKind::RequestBuild,
            AttemptFailureKind::ConsumerDropped,
        ] {
            let report = state.observe_terminal_at(
                &failed(kimi.clone(), kind, None, None),
                ShadowObservationMode::Update,
                now,
            );
            assert!(matches!(report.signal, ShadowSignal::Neutral { .. }));
            assert!(!report.mutated);
        }

        let server = state.observe_terminal_at(
            &failed(
                kimi.clone(),
                AttemptFailureKind::HttpStatus,
                Some(500),
                None,
            ),
            ShadowObservationMode::TelemetryOnly,
            now,
        );
        assert!(matches!(server.signal, ShadowSignal::RouteFailure { .. }));

        let caller = state.observe_terminal_at(
            &failed(
                kimi.clone(),
                AttemptFailureKind::HttpStatus,
                Some(400),
                None,
            ),
            ShadowObservationMode::Update,
            now,
        );
        assert!(matches!(caller.signal, ShadowSignal::Neutral { .. }));
        assert!(!caller.mutated);

        let malformed_capacity = state.observe_terminal_at(
            &failed(kimi, AttemptFailureKind::CapacityRejected, Some(500), None),
            ShadowObservationMode::Update,
            now,
        );
        assert!(matches!(
            malformed_capacity.signal,
            ShadowSignal::Neutral { .. }
        ));
        assert!(!malformed_capacity.mutated);
    }

    #[test]
    fn provider_specific_rate_limit_scopes_and_deployment_capacity_are_isolated() {
        let state = ShadowHealthState::with_policy(test_policy());
        let now = Instant::now();
        let k3 = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");
        let quick = route(ProviderId::Tinfoil, "gpt-oss-120b", "gpt-oss-120b");
        let k2 = route(ProviderId::Continuum, "kimi-k2-6", "kimi-k2.6");
        let glm_continuum = route(ProviderId::Continuum, "glm-5-3", "glm-5.3");
        let glm_tinfoil = route(ProviderId::Tinfoil, "glm-5-3", "glm-5-3");
        let glm_flash = route(ProviderId::Tinfoil, "glm-5-3-flash", "glm-5-3-flash");

        let tinfoil = state.observe_terminal_at(
            &failed(
                k3.clone(),
                AttemptFailureKind::CapacityRejected,
                Some(429),
                Some(Duration::from_secs(5)),
            ),
            ShadowObservationMode::Update,
            now,
        );
        assert_eq!(
            tinfoil.capacity_pool,
            Some(CapacityPoolKey::ProviderModel(k3.route_key()))
        );
        assert!(matches!(
            state.snapshot_at(&k3.route_key(), now).unwrap().effective,
            ShadowDisposition::WouldOpen { .. }
        ));
        assert_eq!(
            state
                .snapshot_at(&quick.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );

        let continuum = state.observe_terminal_at(
            &failed(
                k2.clone(),
                AttemptFailureKind::CapacityRejected,
                Some(429),
                Some(Duration::from_secs(5)),
            ),
            ShadowObservationMode::Update,
            now,
        );
        assert_eq!(
            continuum.capacity_pool,
            Some(CapacityPoolKey::ProviderAccount(ProviderId::Continuum))
        );
        assert!(matches!(
            state
                .snapshot_at(&glm_continuum.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::WouldOpen { .. }
        ));
        assert_eq!(
            state
                .snapshot_at(&glm_tinfoil.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );
        assert_eq!(
            state
                .snapshot_at(&glm_flash.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );

        let tinfoil_glm = state.observe_terminal_at(
            &failed(
                glm_tinfoil.clone(),
                AttemptFailureKind::CapacityRejected,
                Some(429),
                Some(Duration::from_secs(5)),
            ),
            ShadowObservationMode::Update,
            now,
        );
        assert_eq!(
            tinfoil_glm.capacity_pool,
            Some(CapacityPoolKey::ProviderModel(glm_tinfoil.route_key()))
        );
        assert!(matches!(
            state
                .snapshot_at(&glm_tinfoil.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::WouldOpen { .. }
        ));
        assert_eq!(
            state
                .snapshot_at(&glm_flash.route_key(), now)
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );

        for status in [503, 529] {
            let deployment_state = ShadowHealthState::with_policy(test_policy());
            deployment_state.observe_terminal_at(
                &failed(
                    k2.clone(),
                    AttemptFailureKind::CapacityRejected,
                    Some(status),
                    Some(Duration::from_secs(5)),
                ),
                ShadowObservationMode::Update,
                now,
            );
            assert!(matches!(
                deployment_state
                    .snapshot_at(&k2.route_key(), now)
                    .unwrap()
                    .effective,
                ShadowDisposition::WouldOpen { .. }
            ));
            assert_eq!(
                deployment_state
                    .snapshot_at(&glm_continuum.route_key(), now)
                    .unwrap()
                    .effective,
                ShadowDisposition::Healthy
            );

            deployment_state.observe_terminal_at(
                &failed(
                    glm_tinfoil.clone(),
                    AttemptFailureKind::CapacityRejected,
                    Some(status),
                    Some(Duration::from_secs(5)),
                ),
                ShadowObservationMode::Update,
                now,
            );
            assert!(matches!(
                deployment_state
                    .snapshot_at(&glm_tinfoil.route_key(), now)
                    .unwrap()
                    .effective,
                ShadowDisposition::WouldOpen { .. }
            ));
            assert_eq!(
                deployment_state
                    .snapshot_at(&glm_flash.route_key(), now)
                    .unwrap()
                    .effective,
                ShadowDisposition::Healthy
            );
        }
    }

    #[test]
    fn capacity_cooldowns_are_bounded_and_success_cannot_clear_them_early() {
        let state = ShadowHealthState::with_policy(test_policy());
        let start = Instant::now();
        let k3 = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");
        let key = k3.route_key();

        state.observe_terminal_at(
            &failed(
                k3.clone(),
                AttemptFailureKind::CapacityRejected,
                Some(429),
                None,
            ),
            ShadowObservationMode::Update,
            start,
        );
        assert_eq!(
            state.snapshot_at(&key, start).unwrap().effective,
            ShadowDisposition::WouldOpen {
                remaining: Duration::from_secs(4)
            }
        );

        state.observe_terminal_at(
            &completed(k3.clone()),
            ShadowObservationMode::Update,
            start + Duration::from_secs(1),
        );
        assert!(matches!(
            state
                .snapshot_at(&key, start + Duration::from_secs(1))
                .unwrap()
                .effective,
            ShadowDisposition::WouldOpen { .. }
        ));
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(4))
                .unwrap()
                .effective,
            ShadowDisposition::WouldProbe
        );

        state.observe_terminal_at(
            &completed(k3.clone()),
            ShadowObservationMode::Update,
            start + Duration::from_secs(4),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(4))
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );

        state.observe_terminal_at(
            &failed(
                k3.clone(),
                AttemptFailureKind::CapacityRejected,
                Some(429),
                Some(Duration::ZERO),
            ),
            ShadowObservationMode::Update,
            start + Duration::from_secs(10),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(10))
                .unwrap()
                .effective,
            ShadowDisposition::WouldOpen {
                remaining: Duration::from_secs(4)
            }
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(14))
                .unwrap()
                .effective,
            ShadowDisposition::WouldProbe
        );

        state.observe_terminal_at(
            &failed(
                k3,
                AttemptFailureKind::CapacityRejected,
                Some(429),
                Some(Duration::from_secs(100)),
            ),
            ShadowObservationMode::Update,
            start + Duration::from_secs(20),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(20))
                .unwrap()
                .effective,
            ShadowDisposition::WouldOpen {
                remaining: Duration::from_secs(10)
            }
        );
    }

    #[test]
    fn route_snapshots_share_one_point_in_time_and_fail_closed_for_unknown_routes() {
        let state = ShadowHealthState::with_policy(test_policy());
        let glm_tinfoil = route(ProviderId::Tinfoil, "glm-5-3", "glm-5-3").route_key();
        let glm_continuum = route(ProviderId::Continuum, "glm-5-3", "glm-5.3").route_key();

        let snapshots = state
            .snapshot_routes(&[glm_tinfoil.clone(), glm_continuum.clone()])
            .expect("registered GLM routes");
        assert_eq!(snapshots.len(), 2);
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.effective == ShadowDisposition::Healthy));

        let unknown = RouteKey {
            provider: ProviderId::Tinfoil,
            provider_model_id: "not-registered".to_string(),
        };
        assert!(state
            .snapshot_routes(&[glm_tinfoil, unknown, glm_continuum])
            .is_none());
    }

    #[test]
    fn route_health_uses_a_window_threshold_and_terminal_success_reset() {
        let state = ShadowHealthState::with_policy(test_policy());
        let start = Instant::now();
        let k3 = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");
        let key = k3.route_key();
        let fault = || failed(k3.clone(), AttemptFailureKind::StreamTimeout, None, None);

        state.observe_terminal_at(&fault(), ShadowObservationMode::Update, start);
        assert_eq!(
            state.snapshot_at(&key, start).unwrap().route_health,
            ShadowDisposition::Watch {
                consecutive_failures: 1
            }
        );

        state.observe_terminal_at(
            &completed(k3.clone()),
            ShadowObservationMode::Update,
            start + Duration::from_secs(1),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(1))
                .unwrap()
                .route_health,
            ShadowDisposition::Healthy
        );

        for offset in [2, 3, 4] {
            state.observe_terminal_at(
                &fault(),
                ShadowObservationMode::Update,
                start + Duration::from_secs(offset),
            );
        }
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(4))
                .unwrap()
                .route_health,
            ShadowDisposition::WouldOpen {
                remaining: Duration::from_secs(5)
            }
        );

        state.observe_terminal_at(
            &completed(k3.clone()),
            ShadowObservationMode::Update,
            start + Duration::from_secs(5),
        );
        assert!(matches!(
            state
                .snapshot_at(&key, start + Duration::from_secs(5))
                .unwrap()
                .route_health,
            ShadowDisposition::WouldOpen { .. }
        ));
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(9))
                .unwrap()
                .route_health,
            ShadowDisposition::WouldProbe
        );

        state.observe_terminal_at(
            &completed(k3.clone()),
            ShadowObservationMode::Update,
            start + Duration::from_secs(9),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(9))
                .unwrap()
                .route_health,
            ShadowDisposition::Healthy
        );

        state.observe_terminal_at(
            &fault(),
            ShadowObservationMode::Update,
            start + Duration::from_secs(30),
        );
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(30))
                .unwrap()
                .route_health,
            ShadowDisposition::Watch {
                consecutive_failures: 1
            }
        );
    }

    #[test]
    fn route_health_threshold_uses_one_bounded_rolling_window() {
        let state = ShadowHealthState::with_policy(test_policy());
        let start = Instant::now();
        let k3 = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");
        let key = k3.route_key();
        let fault = || failed(k3.clone(), AttemptFailureKind::StreamTimeout, None, None);

        for offset in [0, 9, 18] {
            state.observe_terminal_at(
                &fault(),
                ShadowObservationMode::Update,
                start + Duration::from_secs(offset),
            );
        }
        assert_eq!(
            state
                .snapshot_at(&key, start + Duration::from_secs(18))
                .unwrap()
                .route_health,
            ShadowDisposition::Watch {
                consecutive_failures: 2
            }
        );

        state.observe_terminal_at(
            &fault(),
            ShadowObservationMode::Update,
            start + Duration::from_secs(19),
        );
        assert!(matches!(
            state
                .snapshot_at(&key, start + Duration::from_secs(19))
                .unwrap()
                .route_health,
            ShadowDisposition::WouldOpen { .. }
        ));
    }

    #[test]
    fn recovered_transport_attempts_are_telemetry_only() {
        let state = ShadowHealthState::with_policy(test_policy());
        let now = Instant::now();
        let k3 = route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3");
        let key = k3.route_key();

        let recovered = state.observe_terminal_at(
            &failed(k3.clone(), AttemptFailureKind::Connect, None, None),
            ShadowObservationMode::TelemetryOnly,
            now,
        );
        assert!(matches!(
            recovered.signal,
            ShadowSignal::RouteFailure {
                kind: AttemptFailureKind::Connect,
                ..
            }
        ));
        assert!(!recovered.mutated);

        state.observe_terminal_at(
            &completed(k3),
            ShadowObservationMode::Update,
            now + Duration::from_millis(1),
        );
        assert_eq!(
            state
                .snapshot_at(&key, now + Duration::from_millis(1))
                .unwrap()
                .effective,
            ShadowDisposition::Healthy
        );
    }

    #[test]
    fn registry_topology_bounds_state_under_concurrent_observation() {
        let state = Arc::new(ShadowHealthState::with_policy(test_policy()));
        let cardinality = state.cardinality();
        let terminal = completed(route(ProviderId::Tinfoil, "kimi-k3", "kimi-k3"));

        let threads = (0..32)
            .map(|_| {
                let state = Arc::clone(&state);
                let terminal = terminal.clone();
                thread::spawn(move || {
                    state.observe_terminal(&terminal, ShadowObservationMode::Update)
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            let report = thread.join().expect("observation thread");
            assert!(report.mutated);
        }
        assert_eq!(state.cardinality(), cardinality);

        let unknown = completed(route(ProviderId::Tinfoil, "unknown", "unknown"));
        let report = state.observe_terminal(&unknown, ShadowObservationMode::Update);
        assert_eq!(report.signal, ShadowSignal::UnknownRoute);
        assert!(!report.mutated);
        assert_eq!(state.cardinality(), cardinality);
    }
}
