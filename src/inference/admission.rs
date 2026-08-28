//! Enclave-local admission control for completion inference.
//!
//! The controller deliberately has no distributed coordination. It uses the
//! credential-free provider registry as a fixed topology, bounds every waiting
//! queue, and admits a turn only when its quota-pool reservation and deployment
//! concurrency slot can be acquired atomically. Logical tickets provide a
//! separate per-account bound for a complete response/tool loop.

use super::health::CapacityPoolKey;
use super::{RouteKey, WorkloadClass};
use crate::provider_registry::{ProviderRegistry, RateLimitScope, PROVIDER_REGISTRY};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::oneshot;
use uuid::Uuid;

// The first release deliberately owns its policy in code. Changing these
// values requires review, a new enclave build, and a restart. AdmissionPolicy
// remains the typed boundary that a future authenticated admin control plane
// can populate without changing scheduler internals.
const BASELINE_DEPLOYMENT_IN_FLIGHT: usize = 4;
const BASELINE_PER_ACCOUNT_IN_FLIGHT: usize = 2;
const BASELINE_POOL_QUEUE: usize = 16;
const BASELINE_PER_ACCOUNT_QUEUE: usize = 2;
const BASELINE_INTERACTIVE_WAIT: Duration = Duration::from_secs(5);
const BASELINE_BACKGROUND_WAIT: Duration = Duration::ZERO;
const BASELINE_LOCAL_RETRY: Duration = Duration::from_secs(1);
const BASELINE_ROLLING_WINDOW: Duration = Duration::from_secs(60);
const BASELINE_COMPLETION_RESERVATION: u64 = 4_096;
const BASELINE_MAX_COMPLETION_CHOICES: u64 = 8;
const MAX_POLICY_COMPLETION_CHOICES: u64 = 128;

// Individual accounting dimensions are added with saturating arithmetic, but
// rejecting values above i64::MAX avoids accepting an accidental overflow
// sentinel as a real provider limit and keeps a future persisted policy
// round-trippable through common control-plane stores.
const MAX_POLICY_BUDGET: u64 = i64::MAX as u64;

/// Typed, enclave-local admission policy.
///
/// The initial policy is compiled into the enclave. No provider quota is
/// guessed: absent RPM/TPM dimensions remain unmetered until a reviewed source
/// policy supplies a confirmed limit.
#[derive(Debug, Clone)]
pub(crate) struct AdmissionPolicy {
    deployment_in_flight: usize,
    per_account_in_flight: usize,
    pool_queue: usize,
    per_account_queue: usize,
    interactive_wait: Duration,
    background_wait: Duration,
    local_retry: Duration,
    rolling_window: Duration,
    completion_default_reservation: u64,
    max_completion_choices: u64,
    quota_budgets: HashMap<CapacityPoolKey, QuotaBudget>,
    deployment_overrides: HashMap<RouteKey, usize>,
}

impl AdmissionPolicy {
    fn baseline() -> Self {
        Self {
            deployment_in_flight: BASELINE_DEPLOYMENT_IN_FLIGHT,
            per_account_in_flight: BASELINE_PER_ACCOUNT_IN_FLIGHT,
            pool_queue: BASELINE_POOL_QUEUE,
            per_account_queue: BASELINE_PER_ACCOUNT_QUEUE,
            interactive_wait: BASELINE_INTERACTIVE_WAIT,
            background_wait: BASELINE_BACKGROUND_WAIT,
            local_retry: BASELINE_LOCAL_RETRY,
            rolling_window: BASELINE_ROLLING_WINDOW,
            completion_default_reservation: BASELINE_COMPLETION_RESERVATION,
            max_completion_choices: BASELINE_MAX_COMPLETION_CHOICES,
            // Provider entitlements can drift and may differ by environment.
            // Do not guess them in the compiled baseline.
            quota_budgets: HashMap::new(),
            deployment_overrides: HashMap::new(),
        }
    }

    pub(crate) fn baseline_for(
        registry: &'static ProviderRegistry,
    ) -> Result<Self, AdmissionConfigError> {
        let policy = Self::baseline();
        policy.validate_for(registry)?;
        Ok(policy)
    }

    #[cfg(test)]
    pub(crate) fn with_deployment_in_flight_for_test(
        registry: &'static ProviderRegistry,
        deployment_in_flight: usize,
    ) -> Result<Self, AdmissionConfigError> {
        let mut policy = Self::baseline();
        policy.deployment_in_flight = deployment_in_flight;
        policy.validate_for(registry)?;
        Ok(policy)
    }

    #[cfg(test)]
    pub(crate) fn deployment_in_flight(&self) -> usize {
        self.deployment_in_flight
    }

    pub(crate) fn per_account_in_flight(&self) -> usize {
        self.per_account_in_flight
    }

    #[cfg(test)]
    pub(crate) fn pool_queue(&self) -> usize {
        self.pool_queue
    }

    #[cfg(test)]
    pub(crate) fn per_account_queue(&self) -> usize {
        self.per_account_queue
    }

    pub(crate) fn interactive_wait(&self) -> Duration {
        self.interactive_wait
    }

    pub(crate) fn background_wait(&self) -> Duration {
        self.background_wait
    }

    #[cfg(test)]
    pub(crate) fn local_retry(&self) -> Duration {
        self.local_retry
    }

    #[cfg(test)]
    pub(crate) fn rolling_window(&self) -> Duration {
        self.rolling_window
    }

    pub(crate) fn completion_default_reservation(&self) -> u64 {
        self.completion_default_reservation
    }

    pub(crate) fn max_completion_choices(&self) -> u64 {
        self.max_completion_choices
    }

    fn validate_for(
        &self,
        registry: &'static ProviderRegistry,
    ) -> Result<(), AdmissionConfigError> {
        if self.deployment_in_flight == 0 {
            return Err(AdmissionConfigError::InvalidValue {
                field: "deployment_in_flight",
                reason: "must be greater than zero",
            });
        }
        if self.per_account_in_flight == 0 {
            return Err(AdmissionConfigError::InvalidValue {
                field: "per_account_in_flight",
                reason: "must be greater than zero",
            });
        }
        if self.pool_queue == 0 {
            return Err(AdmissionConfigError::InvalidValue {
                field: "pool_queue",
                reason: "must be greater than zero",
            });
        }
        if self.per_account_queue == 0 {
            return Err(AdmissionConfigError::InvalidValue {
                field: "per_account_queue",
                reason: "must be greater than zero",
            });
        }
        if self.interactive_wait.is_zero() {
            return Err(AdmissionConfigError::InvalidValue {
                field: "interactive_wait_ms",
                reason: "must be greater than zero",
            });
        }
        if !self.background_wait.is_zero() {
            return Err(AdmissionConfigError::InvalidValue {
                field: "background_wait_ms",
                reason: "background work is immediate-only and must use zero",
            });
        }
        if self.local_retry.is_zero() {
            return Err(AdmissionConfigError::InvalidValue {
                field: "local_retry_ms",
                reason: "must be greater than zero",
            });
        }
        if self.rolling_window.is_zero() {
            return Err(AdmissionConfigError::InvalidValue {
                field: "rolling_window_seconds",
                reason: "must be greater than zero",
            });
        }
        if self.completion_default_reservation == 0
            || self.completion_default_reservation > MAX_POLICY_BUDGET
        {
            return Err(AdmissionConfigError::InvalidValue {
                field: "completion_default_reservation",
                reason: "must be between 1 and i64::MAX",
            });
        }
        if self.max_completion_choices == 0
            || self.max_completion_choices > MAX_POLICY_COMPLETION_CHOICES
        {
            return Err(AdmissionConfigError::InvalidValue {
                field: "max_completion_choices",
                reason: "must be between 1 and 128",
            });
        }
        validate_instant_duration("interactive_wait_ms", self.interactive_wait)?;
        validate_instant_duration("background_wait_ms", self.background_wait)?;
        validate_instant_duration("local_retry_ms", self.local_retry)?;
        validate_instant_duration("rolling_window_seconds", self.rolling_window)?;

        let topology = Topology::from_registry(registry)?;
        for (pool, budget) in &self.quota_budgets {
            if !topology.quota_pools.contains(pool) {
                return Err(AdmissionConfigError::UnknownPoolId(capacity_pool_id(pool)));
            }
            budget.validate()?;
        }
        for (route, limit) in &self.deployment_overrides {
            if !topology.routes.contains(route) {
                return Err(AdmissionConfigError::UnknownDeploymentId(
                    deployment_pool_id(route),
                ));
            }
            if *limit == 0 {
                return Err(AdmissionConfigError::InvalidValue {
                    field: "deployments[].in_flight",
                    reason: "must be greater than zero",
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum AdmissionConfigError {
    #[error("invalid admission policy field {field}: {reason}")]
    InvalidValue {
        field: &'static str,
        reason: &'static str,
    },
    #[error("unknown admission quota pool ID: {0}")]
    UnknownPoolId(String),
    #[error("unknown admission deployment ID: {0}")]
    UnknownDeploymentId(String),
    #[error("provider registry maps route {route} to conflicting quota scopes")]
    ConflictingRouteScope { route: String },
}

fn validate_instant_duration(
    field: &'static str,
    duration: Duration,
) -> Result<(), AdmissionConfigError> {
    if Instant::now().checked_add(duration).is_none() {
        return Err(AdmissionConfigError::InvalidValue {
            field,
            reason: "duration overflows a monotonic deadline",
        });
    }
    Ok(())
}

/// A conservative reservation estimate. The prompt estimate is reserved
/// against both prompt and cached-token budgets because cached usage is not
/// reliably knowable before the upstream accepts the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionEstimate {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: Option<u64>,
}

impl AdmissionEstimate {
    pub(crate) const fn new(prompt_tokens: u64, completion_tokens: Option<u64>) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
        }
    }
}

/// Usage known after a provider turn. Missing dimensions conservatively keep
/// their original reservations during reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ActualUsage {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) cached_prompt_tokens: Option<u64>,
}

impl ActualUsage {
    #[cfg(test)]
    pub(crate) const fn complete(
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_prompt_tokens: Option<u64>,
    ) -> Self {
        Self {
            prompt_tokens: Some(prompt_tokens),
            completion_tokens: Some(completion_tokens),
            cached_prompt_tokens,
        }
    }
}

/// Whether an upstream terminal permits the reservation to be changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalDisposition {
    /// The turn completed and reported usage may replace reserved dimensions.
    Completed,
    /// The request is proven not to have been accepted; refund it completely.
    ProvenPreAcceptance,
    /// Acceptance or usage is ambiguous; retain the full reservation.
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionRejectionKind {
    RateBudget,
    DeploymentBusy,
    AccountBusy,
    QueueFull,
    WaitTimeout,
    BackgroundShed,
    UnknownRoute,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("inference admission rejected: {kind:?}")]
pub(crate) struct AdmissionRejection {
    pub(crate) kind: AdmissionRejectionKind,
    pub(crate) retry_after: Duration,
}

impl AdmissionRejection {
    pub(crate) const fn status_hint(&self) -> u16 {
        match self.kind {
            AdmissionRejectionKind::RateBudget => 429,
            AdmissionRejectionKind::DeploymentBusy
            | AdmissionRejectionKind::AccountBusy
            | AdmissionRejectionKind::QueueFull
            | AdmissionRejectionKind::WaitTimeout
            | AdmissionRejectionKind::BackgroundShed
            | AdmissionRejectionKind::UnknownRoute => 503,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UsageAmount {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
}

impl UsageAmount {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            requests: self.requests.saturating_add(other.requests),
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
            cached_tokens: self.cached_tokens.saturating_add(other.cached_tokens),
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(other.requests),
            prompt_tokens: self.prompt_tokens.saturating_sub(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_sub(other.completion_tokens),
            cached_tokens: self.cached_tokens.saturating_sub(other.cached_tokens),
        }
    }

    fn is_zero(self) -> bool {
        self == Self::default()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct QuotaBudget {
    requests: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cached_tokens: Option<u64>,
}

impl QuotaBudget {
    fn validate(self) -> Result<(), AdmissionConfigError> {
        for (field, value) in [
            ("rpm", self.requests),
            ("prompt_tpm", self.prompt_tokens),
            ("completion_tpm", self.completion_tokens),
            ("cached_tpm", self.cached_tokens),
        ] {
            if value.is_some_and(|value| value == 0 || value > MAX_POLICY_BUDGET) {
                return Err(AdmissionConfigError::InvalidValue {
                    field,
                    reason: "must be between 1 and i64::MAX",
                });
            }
        }
        Ok(())
    }

    fn charged(self, usage: UsageAmount) -> UsageAmount {
        UsageAmount {
            requests: self.requests.map_or(0, |_| usage.requests),
            prompt_tokens: self.prompt_tokens.map_or(0, |_| usage.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .map_or(0, |_| usage.completion_tokens),
            cached_tokens: self.cached_tokens.map_or(0, |_| usage.cached_tokens),
        }
    }

    fn fits(self, current: UsageAmount, requested: UsageAmount) -> bool {
        fn dimension_fits(current: u64, requested: u64, limit: Option<u64>) -> bool {
            limit.is_none_or(|limit| current.saturating_add(requested) <= limit)
        }

        dimension_fits(current.requests, requested.requests, self.requests)
            && dimension_fits(
                current.prompt_tokens,
                requested.prompt_tokens,
                self.prompt_tokens,
            )
            && dimension_fits(
                current.completion_tokens,
                requested.completion_tokens,
                self.completion_tokens,
            )
            && dimension_fits(
                current.cached_tokens,
                requested.cached_tokens,
                self.cached_tokens,
            )
    }

    fn request_can_ever_fit(self, requested: UsageAmount) -> bool {
        self.fits(UsageAmount::default(), requested)
    }
}

#[derive(Debug, Clone, Copy)]
struct TimedUsage {
    started_at: Instant,
    amount: UsageAmount,
}

#[derive(Debug, Default)]
struct QuotaLedger {
    completed: VecDeque<TimedUsage>,
    active: HashMap<u64, TimedUsage>,
}

impl QuotaLedger {
    fn purge(&mut self, now: Instant, window: Duration) {
        while self.completed.front().is_some_and(|entry| {
            entry
                .started_at
                .checked_add(window)
                .is_none_or(|expires| expires <= now)
        }) {
            self.completed.pop_front();
        }
    }

    fn total(&self) -> UsageAmount {
        self.completed
            .iter()
            .chain(self.active.values())
            .fold(UsageAmount::default(), |total, entry| {
                total.saturating_add(entry.amount)
            })
    }

    fn reserve(&mut self, id: u64, started_at: Instant, amount: UsageAmount) {
        let replaced = self.active.insert(id, TimedUsage { started_at, amount });
        debug_assert!(replaced.is_none(), "admission reservation IDs are unique");
    }

    fn cancel_pending(&mut self, id: u64) -> bool {
        self.active.remove(&id).is_some()
    }

    fn finish(
        &mut self,
        id: u64,
        budget: QuotaBudget,
        usage: Option<ActualUsage>,
        disposition: TerminalDisposition,
        now: Instant,
        window: Duration,
    ) -> bool {
        let Some(reserved) = self.active.remove(&id) else {
            return false;
        };

        let retained = match disposition {
            TerminalDisposition::ProvenPreAcceptance => None,
            TerminalDisposition::Ambiguous => Some(reserved.amount),
            TerminalDisposition::Completed => usage.map_or(Some(reserved.amount), |usage| {
                let completion_tokens = usage
                    .completion_tokens
                    .unwrap_or(reserved.amount.completion_tokens);
                let (prompt_tokens, cached_tokens) =
                    match (usage.prompt_tokens, usage.cached_prompt_tokens) {
                        // Both dimensions are authoritative. Cached prompt
                        // tokens are a subset of total prompt tokens.
                        (Some(total), Some(cached)) => {
                            let cached = cached.min(total);
                            (total.saturating_sub(cached), cached)
                        }
                        // The total is authoritative but its cached split is
                        // missing. Charge all of it as uncached and preserve a
                        // cached charge at least as large as both known total
                        // prompt usage and the original reservation.
                        (Some(total), None) => (total, reserved.amount.cached_tokens.max(total)),
                        // A cached count without a total cannot safely reduce
                        // the uncached reservation. The explicit cached count
                        // is nevertheless authoritative and must not be
                        // clamped to a synthesized total.
                        (None, Some(cached)) => (reserved.amount.prompt_tokens, cached),
                        // Neither prompt dimension is known, so retain both
                        // original reservations.
                        (None, None) => {
                            (reserved.amount.prompt_tokens, reserved.amount.cached_tokens)
                        }
                    };
                Some(UsageAmount {
                    requests: reserved.amount.requests,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                })
            }),
        };

        if let Some(amount) = retained.map(|amount| budget.charged(amount)) {
            let still_in_window = reserved
                .started_at
                .checked_add(window)
                .is_some_and(|expires| expires > now);
            if still_in_window && !amount.is_zero() {
                let entry = TimedUsage {
                    started_at: reserved.started_at,
                    amount,
                };
                let insertion = self
                    .completed
                    .iter()
                    .position(|existing| existing.started_at > entry.started_at)
                    .unwrap_or(self.completed.len());
                self.completed.insert(insertion, entry);
            }
        }
        self.purge(now, window);
        true
    }

    fn earliest_expiry(&self, window: Duration, now: Instant) -> Option<Instant> {
        self.completed.iter().find_map(|entry| {
            entry
                .started_at
                .checked_add(window)
                .filter(|expires| *expires > now)
        })
    }

    fn earliest_budget_fit(
        &self,
        budget: QuotaBudget,
        requested: UsageAmount,
        window: Duration,
        now: Instant,
    ) -> Option<Instant> {
        let mut remaining = self.total();
        for entry in &self.completed {
            let Some(expiry) = entry.started_at.checked_add(window) else {
                continue;
            };
            if expiry <= now {
                continue;
            }
            remaining = remaining.saturating_sub(entry.amount);
            if budget.fits(remaining, requested) {
                return Some(expiry);
            }
        }
        None
    }
}

#[derive(Debug)]
struct LogicalWaiter {
    id: u64,
    deadline: Instant,
    sender: oneshot::Sender<Result<(), AdmissionRejection>>,
}

#[derive(Debug, Default)]
struct LogicalAccountState {
    in_flight: usize,
    waiters: VecDeque<LogicalWaiter>,
}

#[derive(Debug)]
struct TurnWaiter {
    id: u64,
    account: Uuid,
    route: RouteKey,
    reservation: UsageAmount,
    deadline: Instant,
    sender: oneshot::Sender<Result<(), AdmissionRejection>>,
}

#[derive(Debug, Default)]
struct FairQueue {
    by_account: HashMap<Uuid, VecDeque<TurnWaiter>>,
    round_robin: VecDeque<Uuid>,
    len: usize,
}

impl FairQueue {
    fn push(&mut self, waiter: TurnWaiter) {
        let account = waiter.account;
        let queue = self.by_account.entry(account).or_default();
        if queue.is_empty() {
            self.round_robin.push_back(account);
        }
        queue.push_back(waiter);
        self.len += 1;
    }

    fn account_len(&self, account: Uuid) -> usize {
        self.by_account.get(&account).map_or(0, VecDeque::len)
    }

    fn pop_round_robin_account(&mut self) -> Option<Uuid> {
        self.round_robin.pop_front()
    }

    fn take_front(&mut self, account: Uuid) -> Option<TurnWaiter> {
        let waiter = self.by_account.get_mut(&account)?.pop_front()?;
        self.len -= 1;
        Some(waiter)
    }

    fn restore_front(&mut self, account: Uuid, waiter: TurnWaiter) {
        self.by_account
            .entry(account)
            .or_default()
            .push_front(waiter);
        self.len += 1;
    }

    fn account_has_more(&mut self, account: Uuid) -> bool {
        let empty = self.by_account.get(&account).is_none_or(VecDeque::is_empty);
        if empty {
            self.by_account.remove(&account);
        }
        !empty
    }

    fn remove(&mut self, account: Uuid, id: u64) -> bool {
        let mut removed = false;
        if let Some(queue) = self.by_account.get_mut(&account) {
            let before = queue.len();
            queue.retain(|waiter| waiter.id != id);
            removed = queue.len() != before;
            if removed {
                self.len -= 1;
            }
            if queue.is_empty() {
                self.by_account.remove(&account);
                self.round_robin.retain(|queued| *queued != account);
            }
        }
        removed
    }
}

#[derive(Debug)]
struct DeploymentState {
    in_flight: usize,
    limit: usize,
}

#[derive(Debug)]
struct QuotaPoolState {
    budget: QuotaBudget,
    ledger: QuotaLedger,
    queue: FairQueue,
    wake_at: Option<Instant>,
}

#[derive(Debug, Clone)]
struct PendingTurnGrant {
    route: RouteKey,
    quota_pool: CapacityPoolKey,
}

#[derive(Debug)]
struct AdmissionInner {
    next_id: u64,
    deployments: HashMap<RouteKey, DeploymentState>,
    quota_pools: HashMap<CapacityPoolKey, QuotaPoolState>,
    logical_accounts: HashMap<Uuid, LogicalAccountState>,
    pending_logical: HashMap<u64, Uuid>,
    pending_turns: HashMap<u64, PendingTurnGrant>,
}

impl AdmissionInner {
    fn allocate_id(&mut self) -> u64 {
        loop {
            self.next_id = self.next_id.wrapping_add(1);
            if self.next_id != 0
                && !self.pending_logical.contains_key(&self.next_id)
                && !self.pending_turns.contains_key(&self.next_id)
                && self
                    .quota_pools
                    .values()
                    .all(|pool| !pool.ledger.active.contains_key(&self.next_id))
            {
                return self.next_id;
            }
        }
    }
}

#[derive(Debug)]
struct Topology {
    routes: HashSet<RouteKey>,
    quota_pools: HashSet<CapacityPoolKey>,
    route_to_quota: HashMap<RouteKey, CapacityPoolKey>,
}

impl Topology {
    fn from_registry(registry: &'static ProviderRegistry) -> Result<Self, AdmissionConfigError> {
        let mut routes = HashSet::new();
        let mut quota_pools = HashSet::new();
        let mut route_to_quota = HashMap::new();

        for model in registry.completion_models() {
            for route in model.routes {
                let route_key = RouteKey {
                    provider: route.provider,
                    provider_model_id: route.provider_model_id.to_string(),
                };
                let quota_pool = match route.rate_limit_scope {
                    RateLimitScope::ProviderModel => {
                        CapacityPoolKey::ProviderModel(route_key.clone())
                    }
                    RateLimitScope::ProviderAccount => {
                        CapacityPoolKey::ProviderAccount(route.provider)
                    }
                };

                if let Some(existing) = route_to_quota.insert(route_key.clone(), quota_pool.clone())
                {
                    if existing != quota_pool {
                        return Err(AdmissionConfigError::ConflictingRouteScope {
                            route: deployment_pool_id(&route_key),
                        });
                    }
                }
                routes.insert(route_key.clone());
                quota_pools.insert(quota_pool.clone());
            }
        }

        Ok(Self {
            routes,
            quota_pools,
            route_to_quota,
        })
    }
}

pub(crate) fn capacity_pool_id(pool: &CapacityPoolKey) -> String {
    match pool {
        CapacityPoolKey::ProviderAccount(provider) => {
            format!("{}:account", provider.as_str())
        }
        CapacityPoolKey::ProviderModel(route) => deployment_pool_id(route),
    }
}

pub(crate) fn deployment_pool_id(route: &RouteKey) -> String {
    format!(
        "{}:model:{}",
        route.provider.as_str(),
        route.provider_model_id
    )
}

#[derive(Debug)]
struct AdmissionShared {
    policy: AdmissionPolicy,
    route_to_quota: HashMap<RouteKey, CapacityPoolKey>,
    inner: Mutex<AdmissionInner>,
}

/// Cloneable handle to one enclave's admission state.
#[derive(Debug, Clone)]
pub(crate) struct AdmissionController {
    shared: Arc<AdmissionShared>,
}

impl Default for AdmissionController {
    fn default() -> Self {
        Self::new(&PROVIDER_REGISTRY, AdmissionPolicy::baseline())
            .expect("static provider registry has a valid baseline admission topology")
    }
}

impl AdmissionController {
    pub(crate) fn new(
        registry: &'static ProviderRegistry,
        policy: AdmissionPolicy,
    ) -> Result<Self, AdmissionConfigError> {
        policy.validate_for(registry)?;
        let topology = Topology::from_registry(registry)?;

        let deployments = topology
            .routes
            .iter()
            .map(|route| {
                let limit = policy
                    .deployment_overrides
                    .get(route)
                    .copied()
                    .unwrap_or(policy.deployment_in_flight);
                (
                    route.clone(),
                    DeploymentState {
                        in_flight: 0,
                        limit,
                    },
                )
            })
            .collect();
        let quota_pools = topology
            .quota_pools
            .iter()
            .map(|pool| {
                (
                    pool.clone(),
                    QuotaPoolState {
                        budget: policy.quota_budgets.get(pool).copied().unwrap_or_default(),
                        ledger: QuotaLedger::default(),
                        queue: FairQueue::default(),
                        wake_at: None,
                    },
                )
            })
            .collect();

        Ok(Self {
            shared: Arc::new(AdmissionShared {
                policy,
                route_to_quota: topology.route_to_quota,
                inner: Mutex::new(AdmissionInner {
                    next_id: 0,
                    deployments,
                    quota_pools,
                    logical_accounts: HashMap::new(),
                    pending_logical: HashMap::new(),
                    pending_turns: HashMap::new(),
                }),
            }),
        })
    }

    pub(crate) fn policy(&self) -> &AdmissionPolicy {
        &self.shared.policy
    }

    /// Acquires the per-account logical-request allowance. Hold the returned
    /// ticket for the complete response, including any tool loop.
    pub(crate) async fn acquire_logical(
        &self,
        account: Uuid,
        workload: WorkloadClass,
    ) -> Result<LogicalAdmissionTicket, AdmissionRejection> {
        let now = Instant::now();
        let wait = self.wait_for(workload);
        let deadline = checked_deadline(now, wait);
        let (receiver, id) = {
            let mut inner = self.lock();
            let can_admit = inner.logical_accounts.get(&account).is_none_or(|state| {
                state.in_flight < self.shared.policy.per_account_in_flight
                    && state.waiters.is_empty()
            });
            if can_admit {
                inner.logical_accounts.entry(account).or_default().in_flight += 1;
                return Ok(LogicalAdmissionTicket {
                    controller: self.clone(),
                    account,
                    released: false,
                });
            }

            if matches!(workload, WorkloadClass::Background) && wait.is_zero() {
                return Err(self.rejection(AdmissionRejectionKind::BackgroundShed));
            }
            if wait.is_zero() || deadline <= now {
                return Err(self.rejection(AdmissionRejectionKind::AccountBusy));
            }
            let queued = inner
                .logical_accounts
                .get(&account)
                .map_or(0, |state| state.waiters.len());
            if queued >= self.shared.policy.per_account_queue {
                return Err(self.rejection(AdmissionRejectionKind::QueueFull));
            }

            let id = inner.allocate_id();
            let (sender, receiver) = oneshot::channel();
            inner
                .logical_accounts
                .entry(account)
                .or_default()
                .waiters
                .push_back(LogicalWaiter {
                    id,
                    deadline,
                    sender,
                });
            self.pump_logical_locked(&mut inner, account, now);
            (receiver, id)
        };

        let mut cancellation = CancellationGuard::logical(self.clone(), id, account);
        match tokio::time::timeout_at(deadline.into(), receiver).await {
            Ok(Ok(Ok(()))) if self.activate_logical(id, account) => {
                cancellation.disarm();
                Ok(LogicalAdmissionTicket {
                    controller: self.clone(),
                    account,
                    released: false,
                })
            }
            Ok(Ok(Err(rejection))) => Err(rejection),
            Ok(Err(_)) | Err(_) => Err(self.rejection(AdmissionRejectionKind::WaitTimeout)),
            Ok(Ok(Ok(()))) => Err(self.rejection(AdmissionRejectionKind::WaitTimeout)),
        }
    }

    /// Atomically acquires a provider quota reservation and a deployment slot.
    /// The route must come from the fixed provider registry.
    pub(crate) async fn acquire_turn(
        &self,
        route: &RouteKey,
        account: Uuid,
        workload: WorkloadClass,
        estimate: AdmissionEstimate,
        absolute_deadline: Option<Instant>,
    ) -> Result<RouteTurnPermit, AdmissionRejection> {
        let Some(quota_pool) = self.shared.route_to_quota.get(route).cloned() else {
            return Err(self.rejection(AdmissionRejectionKind::UnknownRoute));
        };
        let now = Instant::now();
        let workload_deadline = checked_deadline(now, self.wait_for(workload));
        let deadline = absolute_deadline
            .map(|deadline| deadline.min(workload_deadline))
            .unwrap_or(workload_deadline);
        let reservation = UsageAmount {
            requests: 1,
            prompt_tokens: estimate.prompt_tokens,
            completion_tokens: estimate
                .completion_tokens
                .unwrap_or(self.shared.policy.completion_default_reservation),
            cached_tokens: estimate.prompt_tokens,
        };

        let (receiver, id) = {
            let mut inner = self.lock();
            let id = inner.allocate_id();
            let queue_empty = inner
                .quota_pools
                .get(&quota_pool)
                .is_some_and(|pool| pool.queue.len == 0);
            if queue_empty {
                match self.try_reserve_turn_locked(
                    &mut inner,
                    TurnReservation {
                        id,
                        route: route.clone(),
                        quota_pool: quota_pool.clone(),
                        reservation,
                        now,
                        pending: false,
                    },
                ) {
                    Ok(()) => {
                        return Ok(RouteTurnPermit::new(
                            self.clone(),
                            id,
                            route.clone(),
                            quota_pool,
                        ));
                    }
                    Err(blocked)
                        if matches!(workload, WorkloadClass::Background) || deadline <= now =>
                    {
                        return Err(AdmissionRejection {
                            kind: blocked.kind,
                            retry_after: blocked.retry_after,
                        });
                    }
                    Err(_) => {}
                }
            } else if matches!(workload, WorkloadClass::Background) {
                // A background request never jumps an interactive queue.
                return Err(self.rejection(AdmissionRejectionKind::BackgroundShed));
            }

            if matches!(workload, WorkloadClass::Background) || deadline <= now {
                return Err(
                    self.rejection(if matches!(workload, WorkloadClass::Background) {
                        AdmissionRejectionKind::BackgroundShed
                    } else {
                        AdmissionRejectionKind::WaitTimeout
                    }),
                );
            }

            let pool = inner
                .quota_pools
                .get(&quota_pool)
                .expect("fixed admission topology contains every route quota pool");
            if !pool
                .budget
                .request_can_ever_fit(pool.budget.charged(reservation))
            {
                return Err(AdmissionRejection {
                    kind: AdmissionRejectionKind::RateBudget,
                    retry_after: self.shared.policy.rolling_window,
                });
            }
            if pool.queue.len >= self.shared.policy.pool_queue
                || pool.queue.account_len(account) >= self.shared.policy.per_account_queue
            {
                return Err(self.rejection(AdmissionRejectionKind::QueueFull));
            }

            let (sender, receiver) = oneshot::channel();
            inner
                .quota_pools
                .get_mut(&quota_pool)
                .expect("fixed admission topology contains every route quota pool")
                .queue
                .push(TurnWaiter {
                    id,
                    account,
                    route: route.clone(),
                    reservation,
                    deadline,
                    sender,
                });
            self.pump_turn_pool_locked(&mut inner, &quota_pool, now);
            (receiver, id)
        };
        self.schedule_pool_wake(quota_pool.clone());

        let mut cancellation =
            CancellationGuard::turn(self.clone(), id, account, quota_pool.clone());
        match tokio::time::timeout_at(deadline.into(), receiver).await {
            Ok(Ok(Ok(()))) => {
                if let Some(grant) = self.activate_turn(id) {
                    cancellation.disarm();
                    Ok(RouteTurnPermit::new(
                        self.clone(),
                        id,
                        grant.route,
                        grant.quota_pool,
                    ))
                } else {
                    Err(self.rejection(AdmissionRejectionKind::WaitTimeout))
                }
            }
            Ok(Ok(Err(rejection))) => {
                cancellation.disarm();
                Err(rejection)
            }
            Ok(Err(_)) | Err(_) => {
                let rejection = self.timeout_turn(id, account, &quota_pool, reservation);
                cancellation.disarm();
                Err(rejection)
            }
        }
    }

    fn wait_for(&self, workload: WorkloadClass) -> Duration {
        match workload {
            WorkloadClass::Interactive => self.shared.policy.interactive_wait,
            WorkloadClass::Background => self.shared.policy.background_wait,
        }
    }

    fn rejection(&self, kind: AdmissionRejectionKind) -> AdmissionRejection {
        AdmissionRejection {
            kind,
            retry_after: self.shared.policy.local_retry,
        }
    }

    fn lock(&self) -> MutexGuard<'_, AdmissionInner> {
        self.shared
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn activate_logical(&self, id: u64, account: Uuid) -> bool {
        self.lock().pending_logical.remove(&id) == Some(account)
    }

    fn activate_turn(&self, id: u64) -> Option<PendingTurnGrant> {
        self.lock().pending_turns.remove(&id)
    }

    fn rate_budget_blocked_locked(
        &self,
        pool: &mut QuotaPoolState,
        reservation: UsageAmount,
        now: Instant,
    ) -> Option<Blocked> {
        pool.ledger.purge(now, self.shared.policy.rolling_window);
        let charged = pool.budget.charged(reservation);
        if !pool.budget.request_can_ever_fit(charged) {
            return Some(Blocked {
                kind: AdmissionRejectionKind::RateBudget,
                retry_after: self.shared.policy.rolling_window,
            });
        }
        if pool.budget.fits(pool.ledger.total(), charged) {
            return None;
        }

        let retry_after = pool
            .ledger
            .earliest_budget_fit(pool.budget, charged, self.shared.policy.rolling_window, now)
            .and_then(|expiry| expiry.checked_duration_since(now))
            .unwrap_or(self.shared.policy.rolling_window);
        Some(Blocked {
            kind: AdmissionRejectionKind::RateBudget,
            retry_after,
        })
    }

    fn turn_wait_timeout_rejection_locked(
        &self,
        inner: &mut AdmissionInner,
        quota_pool: &CapacityPoolKey,
        reservation: UsageAmount,
        now: Instant,
    ) -> AdmissionRejection {
        let rate_block = inner
            .quota_pools
            .get_mut(quota_pool)
            .and_then(|pool| self.rate_budget_blocked_locked(pool, reservation, now));
        rate_block.map_or_else(
            || self.rejection(AdmissionRejectionKind::WaitTimeout),
            |blocked| AdmissionRejection {
                kind: blocked.kind,
                retry_after: blocked.retry_after,
            },
        )
    }

    fn try_reserve_turn_locked(
        &self,
        inner: &mut AdmissionInner,
        request: TurnReservation,
    ) -> Result<(), Blocked> {
        let TurnReservation {
            id,
            route,
            quota_pool,
            reservation,
            now,
            pending,
        } = request;
        let pool = inner
            .quota_pools
            .get_mut(&quota_pool)
            .expect("fixed admission topology contains every route quota pool");
        if let Some(blocked) = self.rate_budget_blocked_locked(pool, reservation, now) {
            return Err(blocked);
        }
        let charged = pool.budget.charged(reservation);

        let deployment = inner
            .deployments
            .get_mut(&route)
            .expect("fixed admission topology contains every route deployment");
        if deployment.in_flight >= deployment.limit {
            return Err(Blocked {
                kind: AdmissionRejectionKind::DeploymentBusy,
                retry_after: self.shared.policy.local_retry,
            });
        }

        deployment.in_flight += 1;
        pool.ledger.reserve(id, now, charged);
        if pending {
            let replaced = inner
                .pending_turns
                .insert(id, PendingTurnGrant { route, quota_pool });
            debug_assert!(replaced.is_none());
        }
        Ok(())
    }

    fn pump_logical_locked(&self, inner: &mut AdmissionInner, account: Uuid, now: Instant) {
        loop {
            let Some(state) = inner.logical_accounts.get_mut(&account) else {
                return;
            };
            if state.in_flight >= self.shared.policy.per_account_in_flight {
                return;
            }
            let Some(waiter) = state.waiters.pop_front() else {
                return;
            };
            if waiter.deadline <= now {
                let _ = waiter
                    .sender
                    .send(Err(self.rejection(AdmissionRejectionKind::WaitTimeout)));
                continue;
            }

            state.in_flight += 1;
            inner.pending_logical.insert(waiter.id, account);
            if waiter.sender.send(Ok(())).is_err() {
                inner.pending_logical.remove(&waiter.id);
                state.in_flight -= 1;
                continue;
            }
        }
    }

    fn pump_turn_pool_locked(
        &self,
        inner: &mut AdmissionInner,
        quota_pool: &CapacityPoolKey,
        now: Instant,
    ) {
        loop {
            let accounts_this_round = inner
                .quota_pools
                .get(quota_pool)
                .map_or(0, |pool| pool.queue.round_robin.len());
            if accounts_this_round == 0 {
                return;
            }
            let mut made_progress_this_round = false;
            // Accounts which did not receive service stay ahead of accounts
            // which did. Keeping these lists outside the queue prevents a
            // blocked account from rotating behind a just-serviced account
            // merely because capacity filled part-way through the round.
            let mut unserved_accounts = Vec::with_capacity(accounts_this_round);
            let mut served_accounts = Vec::with_capacity(accounts_this_round);

            for _ in 0..accounts_this_round {
                let account = inner
                    .quota_pools
                    .get_mut(quota_pool)
                    .and_then(|pool| pool.queue.pop_round_robin_account())
                    .expect("round-robin cardinality was captured under the same lock");
                let waiter = inner
                    .quota_pools
                    .get_mut(quota_pool)
                    .and_then(|pool| pool.queue.take_front(account))
                    .expect("round-robin accounts always have a waiter");

                if waiter.deadline <= now {
                    made_progress_this_round = true;
                    let rejection = self.turn_wait_timeout_rejection_locked(
                        inner,
                        quota_pool,
                        waiter.reservation,
                        now,
                    );
                    let has_more = inner
                        .quota_pools
                        .get_mut(quota_pool)
                        .expect("fixed quota pool")
                        .queue
                        .account_has_more(account);
                    if has_more {
                        unserved_accounts.push(account);
                    }
                    let _ = waiter.sender.send(Err(rejection));
                    continue;
                }

                match self.try_reserve_turn_locked(
                    inner,
                    TurnReservation {
                        id: waiter.id,
                        route: waiter.route.clone(),
                        quota_pool: quota_pool.clone(),
                        reservation: waiter.reservation,
                        now,
                        pending: true,
                    },
                ) {
                    Ok(()) => {
                        let has_more = inner
                            .quota_pools
                            .get_mut(quota_pool)
                            .expect("fixed quota pool")
                            .queue
                            .account_has_more(account);
                        if waiter.sender.send(Ok(())).is_err() {
                            made_progress_this_round = true;
                            self.cancel_pending_turn_locked(inner, waiter.id);
                            if has_more {
                                unserved_accounts.push(account);
                            }
                        } else {
                            made_progress_this_round = true;
                            if has_more {
                                served_accounts.push(account);
                            }
                        }
                    }
                    Err(_) => {
                        inner
                            .quota_pools
                            .get_mut(quota_pool)
                            .expect("fixed quota pool")
                            .queue
                            .restore_front(account, waiter);
                        unserved_accounts.push(account);
                    }
                }
            }

            let queue = &mut inner
                .quota_pools
                .get_mut(quota_pool)
                .expect("fixed quota pool")
                .queue;
            queue.round_robin.extend(unserved_accounts);
            queue.round_robin.extend(served_accounts);

            if !made_progress_this_round {
                return;
            }
        }
    }

    fn release_logical(&self, account: Uuid) {
        let now = Instant::now();
        let mut inner = self.lock();
        if let Some(state) = inner.logical_accounts.get_mut(&account) {
            state.in_flight = state.in_flight.saturating_sub(1);
        }
        self.pump_logical_locked(&mut inner, account, now);
        self.prune_logical_locked(&mut inner, account);
    }

    fn cancel_logical(&self, id: u64, account: Uuid) {
        let now = Instant::now();
        let mut inner = self.lock();
        let removed_waiter = inner
            .logical_accounts
            .get_mut(&account)
            .is_some_and(|state| {
                let before = state.waiters.len();
                state.waiters.retain(|waiter| waiter.id != id);
                state.waiters.len() != before
            });
        if !removed_waiter && inner.pending_logical.remove(&id) == Some(account) {
            if let Some(state) = inner.logical_accounts.get_mut(&account) {
                state.in_flight = state.in_flight.saturating_sub(1);
            }
        }
        self.pump_logical_locked(&mut inner, account, now);
        self.prune_logical_locked(&mut inner, account);
    }

    fn prune_logical_locked(&self, inner: &mut AdmissionInner, account: Uuid) {
        let can_prune = inner
            .logical_accounts
            .get(&account)
            .is_some_and(|state| state.in_flight == 0 && state.waiters.is_empty())
            && !inner
                .pending_logical
                .values()
                .any(|pending| *pending == account);
        if can_prune {
            inner.logical_accounts.remove(&account);
        }
    }

    fn cancel_turn(&self, id: u64, account: Uuid, quota_pool: &CapacityPoolKey) {
        let now = Instant::now();
        let mut inner = self.lock();
        let removed_waiter = inner
            .quota_pools
            .get_mut(quota_pool)
            .is_some_and(|pool| pool.queue.remove(account, id));
        if !removed_waiter {
            self.cancel_pending_turn_locked(&mut inner, id);
        }
        self.pump_turn_pool_locked(&mut inner, quota_pool, now);
        drop(inner);
        self.schedule_pool_wake(quota_pool.clone());
    }

    fn timeout_turn(
        &self,
        id: u64,
        account: Uuid,
        quota_pool: &CapacityPoolKey,
        reservation: UsageAmount,
    ) -> AdmissionRejection {
        self.timeout_turn_at(id, account, quota_pool, reservation, Instant::now())
    }

    fn timeout_turn_at(
        &self,
        id: u64,
        account: Uuid,
        quota_pool: &CapacityPoolKey,
        reservation: UsageAmount,
        now: Instant,
    ) -> AdmissionRejection {
        let mut inner = self.lock();
        let removed_waiter = inner
            .quota_pools
            .get_mut(quota_pool)
            .is_some_and(|pool| pool.queue.remove(account, id));
        let rejection = if removed_waiter {
            self.turn_wait_timeout_rejection_locked(&mut inner, quota_pool, reservation, now)
        } else {
            self.rejection(AdmissionRejectionKind::WaitTimeout)
        };
        if !removed_waiter {
            self.cancel_pending_turn_locked(&mut inner, id);
        }
        self.pump_turn_pool_locked(&mut inner, quota_pool, now);
        drop(inner);
        self.schedule_pool_wake(quota_pool.clone());
        rejection
    }

    fn cancel_pending_turn_locked(&self, inner: &mut AdmissionInner, id: u64) {
        let Some(grant) = inner.pending_turns.remove(&id) else {
            return;
        };
        let removed = inner
            .quota_pools
            .get_mut(&grant.quota_pool)
            .is_some_and(|pool| pool.ledger.cancel_pending(id));
        if removed {
            if let Some(deployment) = inner.deployments.get_mut(&grant.route) {
                deployment.in_flight = deployment.in_flight.saturating_sub(1);
            }
        }
    }

    fn finish_turn(
        &self,
        id: u64,
        route: &RouteKey,
        quota_pool: &CapacityPoolKey,
        usage: Option<ActualUsage>,
        disposition: TerminalDisposition,
    ) {
        let now = Instant::now();
        let mut inner = self.lock();
        let finished = inner.quota_pools.get_mut(quota_pool).is_some_and(|pool| {
            pool.ledger.finish(
                id,
                pool.budget,
                usage,
                disposition,
                now,
                self.shared.policy.rolling_window,
            )
        });
        if finished {
            if let Some(deployment) = inner.deployments.get_mut(route) {
                deployment.in_flight = deployment.in_flight.saturating_sub(1);
            }
        }
        self.pump_turn_pool_locked(&mut inner, quota_pool, now);
        drop(inner);
        self.schedule_pool_wake(quota_pool.clone());
    }

    fn schedule_pool_wake(&self, quota_pool: CapacityPoolKey) {
        let now = Instant::now();
        let wake_at = {
            let mut inner = self.lock();
            let Some(pool) = inner.quota_pools.get_mut(&quota_pool) else {
                return;
            };
            if pool.queue.len == 0 {
                pool.wake_at = None;
                return;
            }
            pool.ledger.purge(now, self.shared.policy.rolling_window);
            let Some(candidate) = pool
                .ledger
                .earliest_expiry(self.shared.policy.rolling_window, now)
            else {
                return;
            };
            if pool.wake_at.is_some_and(|scheduled| scheduled <= candidate) {
                return;
            }
            pool.wake_at = Some(candidate);
            candidate
        };

        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let controller = self.clone();
        runtime.spawn(async move {
            tokio::time::sleep_until(wake_at.into()).await;
            controller.on_pool_wake(quota_pool, wake_at);
        });
    }

    fn on_pool_wake(&self, quota_pool: CapacityPoolKey, wake_at: Instant) {
        let now = Instant::now();
        let mut inner = self.lock();
        let Some(pool) = inner.quota_pools.get_mut(&quota_pool) else {
            return;
        };
        if pool.wake_at != Some(wake_at) {
            return;
        }
        pool.wake_at = None;
        pool.ledger.purge(now, self.shared.policy.rolling_window);
        self.pump_turn_pool_locked(&mut inner, &quota_pool, now);
        drop(inner);
        self.schedule_pool_wake(quota_pool);
    }

    #[cfg(test)]
    fn debug_counts(&self) -> DebugCounts {
        let inner = self.lock();
        DebugCounts {
            deployment_pools: inner.deployments.len(),
            quota_pools: inner.quota_pools.len(),
            logical_accounts: inner.logical_accounts.len(),
            queued_turns: inner.quota_pools.values().map(|pool| pool.queue.len).sum(),
            active_turns: inner
                .deployments
                .values()
                .map(|deployment| deployment.in_flight)
                .sum(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Blocked {
    kind: AdmissionRejectionKind,
    retry_after: Duration,
}

#[derive(Debug)]
struct TurnReservation {
    id: u64,
    route: RouteKey,
    quota_pool: CapacityPoolKey,
    reservation: UsageAmount,
    now: Instant,
    pending: bool,
}

fn checked_deadline(now: Instant, wait: Duration) -> Instant {
    now.checked_add(wait).unwrap_or(now)
}

/// RAII guard held for a complete logical response/tool loop.
#[derive(Debug)]
pub(crate) struct LogicalAdmissionTicket {
    controller: AdmissionController,
    account: Uuid,
    released: bool,
}

impl Drop for LogicalAdmissionTicket {
    fn drop(&mut self) {
        if !self.released {
            self.released = true;
            self.controller.release_logical(self.account);
        }
    }
}

/// Non-cloneable RAII guard for one provider turn.
#[derive(Debug)]
pub(crate) struct RouteTurnPermit {
    controller: AdmissionController,
    reservation_id: u64,
    route: RouteKey,
    quota_pool: CapacityPoolKey,
    released: bool,
}

impl RouteTurnPermit {
    fn new(
        controller: AdmissionController,
        reservation_id: u64,
        route: RouteKey,
        quota_pool: CapacityPoolKey,
    ) -> Self {
        Self {
            controller,
            reservation_id,
            route,
            quota_pool,
            released: false,
        }
    }

    /// Reconciles the reservation exactly once and releases concurrency. The
    /// consuming receiver prevents accidental double settlement.
    pub(crate) fn settle(mut self, actual: Option<ActualUsage>, disposition: TerminalDisposition) {
        self.controller.finish_turn(
            self.reservation_id,
            &self.route,
            &self.quota_pool,
            actual,
            disposition,
        );
        self.released = true;
    }
}

impl Drop for RouteTurnPermit {
    fn drop(&mut self) {
        if !self.released {
            self.controller.finish_turn(
                self.reservation_id,
                &self.route,
                &self.quota_pool,
                None,
                TerminalDisposition::Ambiguous,
            );
            self.released = true;
        }
    }
}

#[derive(Debug)]
enum CancellationKind {
    Logical {
        account: Uuid,
    },
    Turn {
        account: Uuid,
        quota_pool: CapacityPoolKey,
    },
}

#[derive(Debug)]
struct CancellationGuard {
    controller: AdmissionController,
    id: u64,
    kind: CancellationKind,
    armed: bool,
}

impl CancellationGuard {
    fn logical(controller: AdmissionController, id: u64, account: Uuid) -> Self {
        Self {
            controller,
            id,
            kind: CancellationKind::Logical { account },
            armed: true,
        }
    }

    fn turn(
        controller: AdmissionController,
        id: u64,
        account: Uuid,
        quota_pool: CapacityPoolKey,
    ) -> Self {
        Self {
            controller,
            id,
            kind: CancellationKind::Turn {
                account,
                quota_pool,
            },
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        match &self.kind {
            CancellationKind::Logical { account } => {
                self.controller.cancel_logical(self.id, *account);
            }
            CancellationKind::Turn {
                account,
                quota_pool,
            } => self.controller.cancel_turn(self.id, *account, quota_pool),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DebugCounts {
    deployment_pools: usize,
    quota_pools: usize,
    logical_accounts: usize,
    queued_turns: usize,
    active_turns: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_registry::ProviderId;
    use tokio::task::JoinHandle;

    fn route(provider: ProviderId, model: &str) -> RouteKey {
        RouteKey {
            provider,
            provider_model_id: model.to_string(),
        }
    }

    fn k3() -> RouteKey {
        route(ProviderId::Tinfoil, "kimi-k3")
    }

    fn continuum_k2() -> RouteKey {
        route(ProviderId::Continuum, "kimi-k2.6")
    }

    fn continuum_glm() -> RouteKey {
        route(ProviderId::Continuum, "glm-5.2")
    }

    fn test_policy(deployment_limit: usize) -> AdmissionPolicy {
        AdmissionPolicy {
            deployment_in_flight: deployment_limit,
            per_account_in_flight: 1,
            pool_queue: 4,
            per_account_queue: 2,
            interactive_wait: Duration::from_secs(1),
            background_wait: Duration::ZERO,
            local_retry: Duration::from_millis(1),
            rolling_window: Duration::from_millis(30),
            completion_default_reservation: 10,
            max_completion_choices: BASELINE_MAX_COMPLETION_CHOICES,
            quota_budgets: HashMap::new(),
            deployment_overrides: HashMap::new(),
        }
    }

    fn controller(policy: AdmissionPolicy) -> AdmissionController {
        AdmissionController::new(&PROVIDER_REGISTRY, policy).unwrap()
    }

    fn spawn_turn(
        controller: AdmissionController,
        route: RouteKey,
        account: Uuid,
    ) -> JoinHandle<Result<RouteTurnPermit, AdmissionRejection>> {
        tokio::spawn(async move {
            controller
                .acquire_turn(
                    &route,
                    account,
                    WorkloadClass::Interactive,
                    AdmissionEstimate::new(1, Some(1)),
                    None,
                )
                .await
        })
    }

    #[test]
    fn compiled_baseline_is_bounded_and_does_not_guess_provider_quotas() {
        let policy = AdmissionPolicy::baseline_for(&PROVIDER_REGISTRY).unwrap();
        assert_eq!(policy.deployment_in_flight(), 4);
        assert_eq!(policy.per_account_in_flight(), 2);
        assert_eq!(policy.pool_queue(), 16);
        assert_eq!(policy.per_account_queue(), 2);
        assert_eq!(policy.interactive_wait(), Duration::from_secs(5));
        assert_eq!(policy.background_wait(), Duration::ZERO);
        assert_eq!(policy.local_retry(), Duration::from_secs(1));
        assert_eq!(policy.rolling_window(), Duration::from_secs(60));
        assert_eq!(policy.completion_default_reservation(), 4_096);
        assert_eq!(policy.max_completion_choices(), 8);
        assert!(policy.quota_budgets.is_empty());
        assert!(policy.deployment_overrides.is_empty());
    }

    #[test]
    fn test_only_policy_override_still_passes_the_production_validator() {
        let policy =
            AdmissionPolicy::with_deployment_in_flight_for_test(&PROVIDER_REGISTRY, 3).unwrap();
        assert_eq!(policy.deployment_in_flight(), 3);
        assert!(
            AdmissionPolicy::with_deployment_in_flight_for_test(&PROVIDER_REGISTRY, 0).is_err()
        );
    }

    #[tokio::test]
    async fn logical_account_bound_queues_fifo_and_prunes_account_state() {
        let controller = controller(test_policy(2));
        let account = Uuid::new_v4();
        let first = controller
            .acquire_logical(account, WorkloadClass::Interactive)
            .await
            .unwrap();
        let waiting = {
            let controller = controller.clone();
            tokio::spawn(async move {
                controller
                    .acquire_logical(account, WorkloadClass::Interactive)
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(controller.debug_counts().logical_accounts, 1);
        drop(first);
        let second = waiting.await.unwrap().unwrap();
        drop(second);
        assert_eq!(controller.debug_counts().logical_accounts, 0);
    }

    #[tokio::test]
    async fn fair_queue_round_robins_accounts_without_intra_account_reordering() {
        let controller = controller(test_policy(1));
        let route = k3();
        let blocker = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let a1 = spawn_turn(controller.clone(), route.clone(), a);
        tokio::task::yield_now().await;
        let a2 = spawn_turn(controller.clone(), route.clone(), a);
        tokio::task::yield_now().await;
        let b1 = spawn_turn(controller.clone(), route.clone(), b);
        tokio::task::yield_now().await;

        drop(blocker);
        let permit_a1 = a1.await.unwrap().unwrap();
        assert!(!a2.is_finished());
        assert!(!b1.is_finished());
        drop(permit_a1);
        let permit_b1 = b1.await.unwrap().unwrap();
        assert!(!a2.is_finished());
        drop(permit_b1);
        drop(a2.await.unwrap().unwrap());
        assert_eq!(controller.debug_counts().queued_turns, 0);
    }

    #[test]
    fn expired_head_waiter_does_not_strand_live_same_account_waiter() {
        let controller = controller(test_policy(1));
        let route = k3();
        let quota_pool = controller
            .shared
            .route_to_quota
            .get(&route)
            .expect("registered K3 quota pool")
            .clone();
        let account = Uuid::new_v4();
        let now = Instant::now();
        let reservation = UsageAmount {
            requests: 1,
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: 1,
        };
        let (expired_sender, mut expired_receiver) = oneshot::channel();
        let (live_sender, mut live_receiver) = oneshot::channel();

        let live_id = {
            let mut inner = controller.lock();
            let expired_id = inner.allocate_id();
            let live_id = inner.allocate_id();
            let queue = &mut inner
                .quota_pools
                .get_mut(&quota_pool)
                .expect("fixed K3 quota pool")
                .queue;
            queue.push(TurnWaiter {
                id: expired_id,
                account,
                route: route.clone(),
                reservation,
                deadline: now
                    .checked_sub(Duration::from_millis(1))
                    .expect("test instant supports one millisecond of history"),
                sender: expired_sender,
            });
            queue.push(TurnWaiter {
                id: live_id,
                account,
                route: route.clone(),
                reservation,
                deadline: now
                    .checked_add(Duration::from_secs(1))
                    .expect("test instant supports one second of future"),
                sender: live_sender,
            });

            controller.pump_turn_pool_locked(&mut inner, &quota_pool, now);
            live_id
        };

        assert!(matches!(
            expired_receiver.try_recv(),
            Ok(Err(AdmissionRejection {
                kind: AdmissionRejectionKind::WaitTimeout,
                ..
            }))
        ));
        assert!(matches!(live_receiver.try_recv(), Ok(Ok(()))));
        assert_eq!(controller.debug_counts().queued_turns, 0);
        assert_eq!(controller.debug_counts().active_turns, 1);

        let grant = controller
            .activate_turn(live_id)
            .expect("live waiter received a pending grant");
        RouteTurnPermit::new(controller.clone(), live_id, grant.route, grant.quota_pool)
            .settle(None, TerminalDisposition::ProvenPreAcceptance);
        assert_eq!(controller.debug_counts().active_turns, 0);
    }

    #[tokio::test]
    async fn arrivals_never_bypass_existing_waiters() {
        let controller = controller(test_policy(1));
        let route = k3();
        let blocker = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        let old = spawn_turn(controller.clone(), route.clone(), Uuid::new_v4());
        tokio::task::yield_now().await;
        drop(blocker);
        let newcomer = spawn_turn(controller.clone(), route, Uuid::new_v4());
        let old_permit = old.await.unwrap().unwrap();
        assert!(!newcomer.is_finished());
        drop(old_permit);
        drop(newcomer.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn queue_bounds_and_background_shedding_are_enforced() {
        let mut policy = test_policy(1);
        policy.pool_queue = 2;
        policy.per_account_queue = 1;
        let controller = controller(policy);
        let route = k3();
        let blocker = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        let account = Uuid::new_v4();
        let queued = spawn_turn(controller.clone(), route.clone(), account);
        tokio::task::yield_now().await;
        let duplicate = controller
            .acquire_turn(
                &route,
                account,
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(duplicate.kind, AdmissionRejectionKind::QueueFull);

        let second_queued = spawn_turn(controller.clone(), route.clone(), Uuid::new_v4());
        tokio::task::yield_now().await;
        let pool_full = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(pool_full.kind, AdmissionRejectionKind::QueueFull);

        let background = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(background.kind, AdmissionRejectionKind::BackgroundShed);
        assert_eq!(background.status_hint(), 503);
        drop(blocker);
        drop(queued.await.unwrap().unwrap());
        drop(second_queued.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn cancelled_and_timed_out_waiters_are_removed() {
        let mut policy = test_policy(1);
        policy.interactive_wait = Duration::from_millis(15);
        let controller = controller(policy);
        let route = k3();
        let blocker = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();

        let cancelled = spawn_turn(controller.clone(), route.clone(), Uuid::new_v4());
        tokio::task::yield_now().await;
        cancelled.abort();
        let _ = cancelled.await;
        tokio::task::yield_now().await;
        assert_eq!(controller.debug_counts().queued_turns, 0);

        let timeout = controller
            .acquire_turn(
                &route,
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                Some(Instant::now() + Duration::from_millis(5)),
            )
            .await
            .unwrap_err();
        assert_eq!(timeout.kind, AdmissionRejectionKind::WaitTimeout);
        assert_eq!(controller.debug_counts().queued_turns, 0);
        drop(blocker);
    }

    #[test]
    fn rpm_and_tpm_wait_timeouts_report_remaining_budget_window_as_429() {
        let cases = [
            (
                QuotaBudget {
                    requests: Some(1),
                    ..QuotaBudget::default()
                },
                UsageAmount {
                    requests: 1,
                    ..UsageAmount::default()
                },
                UsageAmount {
                    requests: 1,
                    ..UsageAmount::default()
                },
            ),
            (
                QuotaBudget {
                    prompt_tokens: Some(10),
                    ..QuotaBudget::default()
                },
                UsageAmount {
                    prompt_tokens: 10,
                    ..UsageAmount::default()
                },
                UsageAmount {
                    requests: 1,
                    prompt_tokens: 1,
                    ..UsageAmount::default()
                },
            ),
        ];

        for (budget, completed, reservation) in cases {
            let mut policy = test_policy(1);
            policy.rolling_window = Duration::from_secs(60);
            let quota_pool = CapacityPoolKey::ProviderModel(k3());
            policy.quota_budgets.insert(quota_pool.clone(), budget);
            let controller = controller(policy);
            let started_at = Instant::now();
            let timeout_at = started_at + Duration::from_secs(5);
            let account = Uuid::new_v4();
            let id = {
                let mut inner = controller.lock();
                inner
                    .quota_pools
                    .get_mut(&quota_pool)
                    .unwrap()
                    .ledger
                    .completed
                    .push_back(TimedUsage {
                        started_at,
                        amount: completed,
                    });
                let id = inner.allocate_id();
                let (sender, _receiver) = oneshot::channel();
                inner
                    .quota_pools
                    .get_mut(&quota_pool)
                    .unwrap()
                    .queue
                    .push(TurnWaiter {
                        id,
                        account,
                        route: k3(),
                        reservation,
                        deadline: timeout_at,
                        sender,
                    });
                id
            };

            let rejection =
                controller.timeout_turn_at(id, account, &quota_pool, reservation, timeout_at);
            assert_eq!(rejection.kind, AdmissionRejectionKind::RateBudget);
            assert_eq!(rejection.status_hint(), 429);
            assert_eq!(rejection.retry_after, Duration::from_secs(55));
            assert_eq!(controller.debug_counts().queued_turns, 0);
        }
    }

    #[test]
    fn retry_after_waits_until_enough_completed_usage_expires() {
        let budget = QuotaBudget {
            prompt_tokens: Some(100),
            ..QuotaBudget::default()
        };
        let base = Instant::now();
        let now = base + Duration::from_secs(40);
        let mut ledger = QuotaLedger::default();
        for offset in [0, 10, 20, 30] {
            ledger.completed.push_back(TimedUsage {
                started_at: base + Duration::from_secs(offset),
                amount: UsageAmount {
                    prompt_tokens: 30,
                    ..UsageAmount::default()
                },
            });
        }

        let expiry = ledger
            .earliest_budget_fit(
                budget,
                UsageAmount {
                    prompt_tokens: 50,
                    ..UsageAmount::default()
                },
                Duration::from_secs(100),
                now,
            )
            .unwrap();
        assert_eq!(expiry - now, Duration::from_secs(80));
    }

    #[test]
    fn deployment_wait_timeout_remains_503_with_local_retry() {
        let policy = test_policy(1);
        let local_retry = policy.local_retry;
        let controller = controller(policy);
        let quota_pool = CapacityPoolKey::ProviderModel(k3());
        let account = Uuid::new_v4();
        let reservation = UsageAmount {
            requests: 1,
            ..UsageAmount::default()
        };
        let timeout_at = Instant::now();
        let id = {
            let mut inner = controller.lock();
            inner.deployments.get_mut(&k3()).unwrap().in_flight = 1;
            let id = inner.allocate_id();
            let (sender, _receiver) = oneshot::channel();
            inner
                .quota_pools
                .get_mut(&quota_pool)
                .unwrap()
                .queue
                .push(TurnWaiter {
                    id,
                    account,
                    route: k3(),
                    reservation,
                    deadline: timeout_at,
                    sender,
                });
            id
        };

        let rejection =
            controller.timeout_turn_at(id, account, &quota_pool, reservation, timeout_at);
        assert_eq!(rejection.kind, AdmissionRejectionKind::WaitTimeout);
        assert_eq!(rejection.status_hint(), 503);
        assert_eq!(rejection.retry_after, local_retry);
        assert_eq!(controller.debug_counts().queued_turns, 0);
    }

    #[tokio::test]
    async fn provider_account_quota_is_shared_but_deployment_capacity_is_not() {
        let mut policy = test_policy(1);
        policy.quota_budgets.insert(
            CapacityPoolKey::ProviderAccount(ProviderId::Continuum),
            QuotaBudget {
                requests: Some(1),
                ..QuotaBudget::default()
            },
        );
        let admission = controller(policy);
        let first = admission
            .acquire_turn(
                &continuum_k2(),
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        first.settle(
            Some(ActualUsage::complete(1, 1, None)),
            TerminalDisposition::Completed,
        );

        let rejection = admission
            .acquire_turn(
                &continuum_glm(),
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(rejection.kind, AdmissionRejectionKind::RateBudget);
        assert_eq!(rejection.status_hint(), 429);

        let no_quota = controller(test_policy(1));
        let k2 = no_quota
            .acquire_turn(
                &continuum_k2(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        let glm = no_quota
            .acquire_turn(
                &continuum_glm(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        drop((k2, glm));
    }

    #[tokio::test]
    async fn quota_and_deployment_reservations_are_atomic() {
        let mut policy = test_policy(1);
        policy.quota_budgets.insert(
            CapacityPoolKey::ProviderModel(k3()),
            QuotaBudget {
                requests: Some(1),
                ..QuotaBudget::default()
            },
        );
        let controller = controller(policy);
        let first = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        let blocked = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(blocked.kind, AdmissionRejectionKind::RateBudget);
        assert_eq!(controller.debug_counts().active_turns, 1);
        first.settle(None, TerminalDisposition::ProvenPreAcceptance);
        let second = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, Some(1)),
                None,
            )
            .await
            .unwrap();
        drop(second);
    }

    #[tokio::test]
    async fn reconciliation_refunds_only_proven_pre_acceptance_and_releases_once() {
        let mut policy = test_policy(1);
        policy.quota_budgets.insert(
            CapacityPoolKey::ProviderModel(k3()),
            QuotaBudget {
                completion_tokens: Some(10),
                ..QuotaBudget::default()
            },
        );
        let controller = controller(policy);

        let refunded = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(0, Some(10)),
                None,
            )
            .await
            .unwrap();
        refunded.settle(None, TerminalDisposition::ProvenPreAcceptance);
        assert_eq!(controller.debug_counts().active_turns, 0);

        let reconciled = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(0, Some(10)),
                None,
            )
            .await
            .unwrap();
        reconciled.settle(
            Some(ActualUsage::complete(0, 4, None)),
            TerminalDisposition::Completed,
        );
        let remaining = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(0, Some(6)),
                None,
            )
            .await
            .unwrap();
        drop(remaining); // Ambiguous drop retains six: the budget is now full.
        let rejected = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(0, Some(1)),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(rejected.kind, AdmissionRejectionKind::RateBudget);
        assert_eq!(controller.debug_counts().active_turns, 0);
    }

    #[test]
    fn prompt_and_cached_usage_reconcile_as_independent_observations() {
        fn reconciled(usage: ActualUsage) -> UsageAmount {
            let budget = QuotaBudget {
                prompt_tokens: Some(100),
                cached_tokens: Some(100),
                ..QuotaBudget::default()
            };
            let reserved = budget.charged(UsageAmount {
                requests: 1,
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_tokens: 10,
            });
            let now = Instant::now();
            let mut ledger = QuotaLedger::default();
            ledger.reserve(1, now, reserved);
            assert!(ledger.finish(
                1,
                budget,
                Some(usage),
                TerminalDisposition::Completed,
                now,
                Duration::from_secs(60),
            ));
            ledger.completed.front().unwrap().amount
        }

        let both = reconciled(ActualUsage {
            prompt_tokens: Some(20),
            completion_tokens: None,
            cached_prompt_tokens: Some(7),
        });
        assert_eq!(both.prompt_tokens, 13);
        assert_eq!(both.cached_tokens, 7);

        let prompt_only = reconciled(ActualUsage {
            prompt_tokens: Some(20),
            completion_tokens: None,
            cached_prompt_tokens: None,
        });
        assert_eq!(prompt_only.prompt_tokens, 20);
        assert_eq!(prompt_only.cached_tokens, 20);

        let cached_only = reconciled(ActualUsage {
            prompt_tokens: None,
            completion_tokens: None,
            cached_prompt_tokens: Some(25),
        });
        assert_eq!(cached_only.prompt_tokens, 10);
        assert_eq!(cached_only.cached_tokens, 25);

        let neither = reconciled(ActualUsage::default());
        assert_eq!(neither.prompt_tokens, 10);
        assert_eq!(neither.cached_tokens, 10);
    }

    #[tokio::test]
    async fn authoritative_cached_usage_reconciles_prompt_to_uncached_tokens() {
        let mut policy = test_policy(1);
        policy.quota_budgets.insert(
            CapacityPoolKey::ProviderModel(k3()),
            QuotaBudget {
                prompt_tokens: Some(10),
                cached_tokens: Some(10),
                ..QuotaBudget::default()
            },
        );
        let controller = controller(policy);

        let first = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(10, Some(0)),
                None,
            )
            .await
            .unwrap();
        first.settle(
            Some(ActualUsage::complete(10, 0, Some(3))),
            TerminalDisposition::Completed,
        );

        // The authoritative split charges seven uncached prompt tokens and
        // three cached tokens, so a conservative three-token reservation fits
        // both ten-token dimensions. Charging the full prompt plus cached
        // tokens would reject this request.
        let second = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Background,
                AdmissionEstimate::new(3, Some(0)),
                None,
            )
            .await
            .unwrap();
        second.settle(None, TerminalDisposition::ProvenPreAcceptance);
    }

    #[tokio::test]
    async fn rolling_budget_expiry_wakes_a_waiter_without_a_new_arrival() {
        let mut policy = test_policy(1);
        policy.rolling_window = Duration::from_millis(20);
        policy.quota_budgets.insert(
            CapacityPoolKey::ProviderModel(k3()),
            QuotaBudget {
                requests: Some(1),
                ..QuotaBudget::default()
            },
        );
        let controller = controller(policy);

        let first = controller
            .acquire_turn(
                &k3(),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(0, Some(0)),
                None,
            )
            .await
            .unwrap();
        first.settle(
            Some(ActualUsage::complete(0, 0, Some(0))),
            TerminalDisposition::Completed,
        );

        let waiting = spawn_turn(controller.clone(), k3(), Uuid::new_v4());
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        let permit = tokio::time::timeout(Duration::from_millis(250), waiting)
            .await
            .expect("rolling-window timer should wake the queue")
            .unwrap()
            .unwrap();
        drop(permit);
    }

    #[tokio::test]
    async fn unknown_routes_fail_closed_without_growing_fixed_topology() {
        let controller = controller(test_policy(1));
        let before = controller.debug_counts();
        let rejection = controller
            .acquire_turn(
                &route(ProviderId::Tinfoil, "not-in-the-registry"),
                Uuid::new_v4(),
                WorkloadClass::Interactive,
                AdmissionEstimate::new(1, None),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(rejection.kind, AdmissionRejectionKind::UnknownRoute);
        assert_eq!(rejection.status_hint(), 503);
        assert_eq!(controller.debug_counts(), before);
    }

    #[test]
    fn fixed_pool_cardinality_matches_registry_topology() {
        let topology = Topology::from_registry(&PROVIDER_REGISTRY).unwrap();
        let controller = controller(test_policy(4));
        let counts = controller.debug_counts();
        assert_eq!(counts.deployment_pools, topology.routes.len());
        assert_eq!(counts.quota_pools, topology.quota_pools.len());
        assert_eq!(counts.logical_accounts, 0);
        assert_eq!(counts.queued_turns, 0);
        assert_eq!(counts.active_turns, 0);
    }
}
