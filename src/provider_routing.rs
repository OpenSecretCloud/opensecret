use crate::inference::health::{
    ShadowDisposition, ShadowHealthState, ShadowObservationMode, ShadowObservationReport,
    ShadowRouteSnapshot, MIN_CAPACITY_COOLDOWN,
};
use crate::inference::{
    AttemptTerminal, InferenceIntent, ModelSelectionMode, RouteIdentity, RouteKey,
};
use crate::inference_planning::{
    plan_completion_route, ConfiguredProviders, ProviderPreference, RoutePlan, RoutePlanningError,
    RoutePlanningInput,
};
use crate::model_config::{
    resolve_completion_model_id, resolve_public_model_id, GLM_5_2_MODEL_ID, GLM_5_3_MODEL_ID,
};
use crate::os_flags::GLM_5_3_TINFOIL_FLAG_KEY;
use crate::provider_registry::{
    ProviderId, ProviderRegistry, RouteSelectionSource, PROVIDER_REGISTRY,
};
use crate::proxy_config::{canonicalize_tinfoil_model, ProxyConfig, ProxyRouter};
use std::time::Duration;
use uuid::Uuid;

/// Selects the completion-routing implementation once at an authenticated
/// inference entrypoint. The choice is carried through the complete logical
/// request so feature-flag changes cannot switch routers between provider
/// turns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum InferenceRoutingMode {
    #[default]
    Legacy,
    V2,
}

impl InferenceRoutingMode {
    pub(crate) const fn from_router_v2_flag(value: Option<bool>) -> Self {
        match value {
            Some(true) => Self::V2,
            Some(false) | None => Self::Legacy,
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct ProviderConfig {
    provider: ProviderId,
    weight: u16,
    enabled: bool,
}

#[derive(Debug, Clone, Copy)]
struct ModelProviderRoute {
    provider: ProviderId,
    provider_model_id: &'static str,
    weight: u16,
    enabled: bool,
    requires_explicit_preference: bool,
}

#[derive(Debug, Clone, Copy)]
struct ModelRoutingConfig {
    public_model_id: &'static str,
    routes: &'static [ModelProviderRoute],
    provider_flag: Option<ProviderRoutingFlag>,
    default_provider: Option<ProviderId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderRoutingFlag {
    key: &'static str,
    enabled_provider: ProviderId,
    disabled_provider: ProviderId,
}

impl ProviderRoutingFlag {
    pub(crate) const fn key(self) -> &'static str {
        self.key
    }

    pub(crate) const fn preference_for(self, enabled: bool) -> ProviderPreference {
        ProviderPreference::feature_flag(if enabled {
            self.enabled_provider
        } else {
            self.disabled_provider
        })
    }
}

#[derive(Debug)]
struct ProviderRoutingConfig {
    providers: &'static [ProviderConfig],
    models: &'static [ModelRoutingConfig],
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedProviderRoute {
    pub(crate) provider: ProviderId,
    pub(crate) proxy: ProxyConfig,
    pub(crate) public_model_id: String,
    pub(crate) provider_model_id: String,
    pub(crate) response_model_id: String,
    pub(crate) bucket: Option<u8>,
    pub(crate) selection_source: RouteSelectionSource,
}

impl SelectedProviderRoute {
    pub(crate) fn identity(&self) -> RouteIdentity {
        RouteIdentity::new(
            self.provider,
            self.public_model_id.clone(),
            self.provider_model_id.clone(),
            self.response_model_id.clone(),
            self.selection_source,
            self.bucket,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProviderRoutingError {
    UnsupportedModel(String),
    NoEligibleRoute(String),
    CapacityUnavailable {
        model: String,
        retry_after: Duration,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CredentialFreeRouteOutcome {
    Selected(RouteIdentity),
    UnsupportedModel,
    NoEligibleRoute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShadowRouteComparison {
    Match {
        outcome: CredentialFreeRouteOutcome,
        decision: Option<crate::inference_planning::PlanDecision>,
        candidate_count: usize,
    },
    Mismatch {
        active: CredentialFreeRouteOutcome,
        shadow: CredentialFreeRouteOutcome,
        decision: Option<crate::inference_planning::PlanDecision>,
        candidate_count: usize,
    },
}

#[derive(Debug)]
pub(crate) struct ProviderRouter {
    config: &'static ProviderRoutingConfig,
    registry: &'static ProviderRegistry,
    shadow_health: ShadowHealthState,
}

#[derive(Debug, Clone)]
struct EligibleRoute {
    provider: ProviderId,
    proxy: ProxyConfig,
    provider_model_id: &'static str,
    effective_weight: u32,
}

const PROVIDERS: &[ProviderConfig] = &[
    ProviderConfig {
        provider: ProviderId::Tinfoil,
        weight: 70,
        enabled: true,
    },
    ProviderConfig {
        provider: ProviderId::Continuum,
        weight: 30,
        enabled: true,
    },
];

const KIMI_K2_6_ROUTES: &[ModelProviderRoute] = &[ModelProviderRoute {
    provider: ProviderId::Continuum,
    provider_model_id: "kimi-k2.6",
    weight: 100,
    enabled: true,
    requires_explicit_preference: false,
}];

const GLM_5_2_ROUTES: &[ModelProviderRoute] = &[ModelProviderRoute {
    provider: ProviderId::Tinfoil,
    provider_model_id: GLM_5_2_MODEL_ID,
    weight: 100,
    enabled: true,
    requires_explicit_preference: false,
}];

const GLM_5_3_ROUTES: &[ModelProviderRoute] = &[
    ModelProviderRoute {
        provider: ProviderId::Continuum,
        provider_model_id: "glm-5.3",
        weight: 100,
        enabled: true,
        requires_explicit_preference: false,
    },
    ModelProviderRoute {
        // Preserve the current Router v1 rollout fence. Router v2's separate
        // registry treats the now-GA Tinfoil route as a normal same-model
        // candidate without changing behavior for the feature-flag-off cohort.
        provider: ProviderId::Tinfoil,
        provider_model_id: GLM_5_3_MODEL_ID,
        weight: 100,
        enabled: true,
        requires_explicit_preference: true,
    },
];

const MODEL_ROUTES: &[ModelRoutingConfig] = &[
    ModelRoutingConfig {
        public_model_id: "kimi-k2-6",
        routes: KIMI_K2_6_ROUTES,
        provider_flag: None,
        default_provider: Some(ProviderId::Continuum),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_2_MODEL_ID,
        routes: GLM_5_2_ROUTES,
        provider_flag: None,
        default_provider: Some(ProviderId::Tinfoil),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_3_MODEL_ID,
        routes: GLM_5_3_ROUTES,
        provider_flag: Some(ProviderRoutingFlag {
            key: GLM_5_3_TINFOIL_FLAG_KEY,
            enabled_provider: ProviderId::Tinfoil,
            disabled_provider: ProviderId::Continuum,
        }),
        default_provider: Some(ProviderId::Continuum),
    },
];

static DEFAULT_PROVIDER_ROUTING_CONFIG: ProviderRoutingConfig = ProviderRoutingConfig {
    providers: PROVIDERS,
    models: MODEL_ROUTES,
};

impl Default for ProviderRouter {
    fn default() -> Self {
        Self {
            config: &DEFAULT_PROVIDER_ROUTING_CONFIG,
            registry: &PROVIDER_REGISTRY,
            shadow_health: ShadowHealthState::new(&PROVIDER_REGISTRY),
        }
    }
}

impl ProviderRouter {
    pub(crate) fn observe_attempt_terminal(
        &self,
        terminal: &AttemptTerminal,
        mode: ShadowObservationMode,
    ) -> ShadowObservationReport {
        self.shadow_health.observe_terminal(terminal, mode)
    }

    pub(crate) fn shadow_health_snapshot(&self, route: &RouteKey) -> Option<ShadowRouteSnapshot> {
        self.shadow_health.snapshot(route)
    }

    #[cfg(test)]
    pub(crate) fn shadow_observation_count(&self) -> usize {
        self.shadow_health.observation_count()
    }

    #[cfg(test)]
    pub(crate) fn select_completion_route(
        &self,
        proxy_router: &ProxyRouter,
        account_uuid: Uuid,
        requested_model: &str,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        self.select_completion_route_with_preference(
            proxy_router,
            account_uuid,
            requested_model,
            None,
        )
    }

    pub(crate) fn select_completion_route_with_preference(
        &self,
        proxy_router: &ProxyRouter,
        account_uuid: Uuid,
        requested_model: &str,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        if let Some(public_model_id) = resolve_public_model_id(requested_model) {
            if let Some(model_config) = self.model_config(public_model_id) {
                return self.select_configured_route(
                    proxy_router,
                    account_uuid,
                    model_config,
                    provider_preference,
                );
            }
        }

        self.fallback_completion_route(proxy_router, requested_model)
    }

    /// Dispatch completion routing through the request-scoped implementation
    /// selected at the public inference entrypoint. The legacy implementation
    /// stays intact while Router v2 may evolve independently.
    pub(crate) fn select_completion_route_for_mode(
        &self,
        proxy_router: &ProxyRouter,
        intent: &InferenceIntent,
        provider_preference: Option<ProviderPreference>,
        routing_mode: InferenceRoutingMode,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        match routing_mode {
            InferenceRoutingMode::Legacy => self.select_completion_route_with_preference(
                proxy_router,
                intent.account_uuid,
                &intent.public_model_id,
                provider_preference,
            ),
            InferenceRoutingMode::V2 => {
                self.select_active_completion_route(proxy_router, intent, provider_preference)
            }
        }
    }

    /// Selects the route used by a newly prepared logical request.
    ///
    /// Stack 6 activates health filtering only for explicit GLM 5.3 requests.
    /// Auto aliases and every other public model deliberately retain the legacy
    /// route decision until their own rollout stack.
    pub(crate) fn select_active_completion_route(
        &self,
        proxy_router: &ProxyRouter,
        intent: &InferenceIntent,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        if intent.selection_mode == ModelSelectionMode::Explicit
            && intent.public_model_id == GLM_5_3_MODEL_ID
        {
            return self.select_glm_canary_route(proxy_router, intent, provider_preference);
        }

        self.select_completion_route_with_preference(
            proxy_router,
            intent.account_uuid,
            &intent.public_model_id,
            provider_preference,
        )
    }

    pub(crate) fn shadow_completion_plan(
        &self,
        proxy_router: &ProxyRouter,
        intent: &InferenceIntent,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<RoutePlan, RoutePlanningError> {
        let configured_providers = self.registry.providers().iter().fold(
            ConfiguredProviders::none(),
            |configured, provider| {
                if proxy_for_provider(proxy_router, provider.id).is_some() {
                    configured.with_provider(provider.id)
                } else {
                    configured
                }
            },
        );

        plan_completion_route(
            self.registry,
            RoutePlanningInput {
                intent,
                configured_providers,
                provider_preference,
            },
        )
    }

    fn select_glm_canary_route(
        &self,
        proxy_router: &ProxyRouter,
        intent: &InferenceIntent,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        let model = self
            .registry
            .completion_model(&intent.public_model_id)
            .ok_or_else(|| {
                ProviderRoutingError::UnsupportedModel(intent.public_model_id.clone())
            })?;

        let configured_routes = model
            .routes
            .iter()
            .filter_map(|route| {
                let provider = self.registry.provider(route.provider)?;
                if !route.enabled
                    || route.weight == 0
                    || !provider.enabled
                    || provider.weight == 0
                    || proxy_for_provider(proxy_router, route.provider).is_none()
                {
                    return None;
                }

                Some((
                    route.provider,
                    RouteKey {
                        provider: route.provider,
                        provider_model_id: route.provider_model_id.to_string(),
                    },
                ))
            })
            .collect::<Vec<_>>();

        if configured_routes.is_empty() {
            return Err(ProviderRoutingError::NoEligibleRoute(
                intent.public_model_id.clone(),
            ));
        }

        let route_keys = configured_routes
            .iter()
            .map(|(_, route)| route.clone())
            .collect::<Vec<_>>();
        let snapshots = self
            .shadow_health
            .snapshot_routes(&route_keys)
            .ok_or_else(|| ProviderRoutingError::CapacityUnavailable {
                model: intent.public_model_id.clone(),
                retry_after: MIN_CAPACITY_COOLDOWN,
            })?;

        let mut available_providers = ConfiguredProviders::none();
        let mut earliest_recovery = None;
        for ((provider, _), snapshot) in configured_routes.iter().zip(snapshots) {
            match snapshot.effective {
                ShadowDisposition::WouldOpen { remaining } => {
                    let remaining = ceil_retry_after(remaining);
                    earliest_recovery = Some(
                        earliest_recovery
                            .map_or(remaining, |current: Duration| current.min(remaining)),
                    );
                }
                disposition if route_is_available_for_new_request(disposition) => {
                    available_providers = available_providers.with_provider(*provider);
                }
                _ => unreachable!("WouldOpen is handled above"),
            }
        }

        if available_providers == ConfiguredProviders::none() {
            return Err(ProviderRoutingError::CapacityUnavailable {
                model: intent.public_model_id.clone(),
                retry_after: earliest_recovery.unwrap_or(MIN_CAPACITY_COOLDOWN),
            });
        }

        let plan = plan_completion_route(
            self.registry,
            RoutePlanningInput {
                intent,
                configured_providers: available_providers,
                provider_preference,
            },
        )
        .map_err(provider_routing_error_from_plan)?;

        let selected = plan.selected;
        let proxy = proxy_for_provider(proxy_router, selected.provider)
            .ok_or_else(|| ProviderRoutingError::NoEligibleRoute(intent.public_model_id.clone()))?;
        Ok(SelectedProviderRoute {
            provider: selected.provider,
            proxy,
            public_model_id: selected.public_model_id,
            provider_model_id: selected.provider_model_id,
            response_model_id: selected.response_model_id,
            bucket: selected.bucket,
            selection_source: selected.selection_source,
        })
    }

    pub(crate) fn provider_routing_flag_for_completion_model(
        &self,
        requested_model: &str,
    ) -> Option<ProviderRoutingFlag> {
        let public_model_id = resolve_public_model_id(requested_model)?;
        self.model_config(public_model_id)?.provider_flag
    }

    fn select_configured_route(
        &self,
        proxy_router: &ProxyRouter,
        account_uuid: Uuid,
        model_config: &ModelRoutingConfig,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        let mut eligible_routes = Vec::new();

        for route in model_config.routes {
            if !route.enabled || route.weight == 0 {
                continue;
            }
            if route.requires_explicit_preference
                && provider_preference
                    .is_none_or(|preference| preference.provider() != route.provider)
            {
                continue;
            }

            let Some(provider_config) = self.provider_config(route.provider) else {
                continue;
            };
            if !provider_config.enabled || provider_config.weight == 0 {
                continue;
            }

            let Some(proxy) = proxy_for_provider(proxy_router, route.provider) else {
                continue;
            };

            eligible_routes.push(EligibleRoute {
                provider: route.provider,
                proxy,
                provider_model_id: route.provider_model_id,
                effective_weight: u32::from(provider_config.weight) * u32::from(route.weight),
            });
        }

        if eligible_routes.is_empty() {
            return Err(ProviderRoutingError::NoEligibleRoute(
                model_config.public_model_id.into(),
            ));
        }

        let default_preference = model_config
            .default_provider
            .map(ProviderPreference::default_provider);

        let provider_preference_route = provider_preference.and_then(|preference| {
            eligible_routes
                .iter()
                .find(|route| route.provider == preference.provider())
                .map(|route| (route, preference.source()))
        });
        let default_preference_route = default_preference.and_then(|preference| {
            eligible_routes
                .iter()
                .find(|route| route.provider == preference.provider())
                .map(|route| {
                    let source = if provider_preference.is_some() {
                        RouteSelectionSource::Fallback
                    } else {
                        preference.source()
                    };
                    (route, source)
                })
        });

        let preferred_route = provider_preference_route.or(default_preference_route);

        let (selected, bucket, selection_source) = if let Some((route, source)) = preferred_route {
            (route, None, source)
        } else {
            let selected =
                select_weighted_route(account_uuid, &eligible_routes).ok_or_else(|| {
                    ProviderRoutingError::NoEligibleRoute(model_config.public_model_id.into())
                })?;
            (
                selected.route,
                Some(selected.bucket),
                if provider_preference.is_some() || default_preference.is_some() {
                    RouteSelectionSource::Fallback
                } else {
                    RouteSelectionSource::StaticSplit
                },
            )
        };

        Ok(SelectedProviderRoute {
            provider: selected.provider,
            proxy: selected.proxy.clone(),
            public_model_id: model_config.public_model_id.to_string(),
            provider_model_id: selected.provider_model_id.to_string(),
            response_model_id: model_config.public_model_id.to_string(),
            bucket,
            selection_source,
        })
    }

    fn fallback_completion_route(
        &self,
        proxy_router: &ProxyRouter,
        requested_model: &str,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        let proxy = proxy_router.get_completion_proxy();
        let resolved_public_model_id =
            resolve_public_model_id(requested_model).map(ToOwned::to_owned);
        let provider_model_id = if proxy.provider_name == ProviderId::Tinfoil.as_str() {
            resolve_completion_model_id(requested_model)
                .ok_or_else(|| ProviderRoutingError::UnsupportedModel(requested_model.into()))?
                .to_string()
        } else {
            resolved_public_model_id
                .clone()
                .unwrap_or_else(|| requested_model.to_string())
        };

        let public_model_id = resolved_public_model_id.unwrap_or_else(|| provider_model_id.clone());

        let response_model_id = if proxy.provider_name == ProviderId::Tinfoil.as_str() {
            canonicalize_tinfoil_model(&provider_model_id)
        } else {
            public_model_id.clone()
        };

        Ok(SelectedProviderRoute {
            provider: ProviderId::Tinfoil,
            proxy,
            public_model_id,
            provider_model_id,
            response_model_id,
            bucket: None,
            selection_source: RouteSelectionSource::StaticSplit,
        })
    }

    fn provider_config(&self, provider: ProviderId) -> Option<&ProviderConfig> {
        self.config
            .providers
            .iter()
            .find(|config| config.provider == provider)
    }

    fn model_config(&self, public_model_id: &str) -> Option<&ModelRoutingConfig> {
        self.config
            .models
            .iter()
            .find(|config| config.public_model_id == public_model_id)
    }
}

pub(crate) fn compare_shadow_route(
    active: &Result<SelectedProviderRoute, ProviderRoutingError>,
    shadow: &Result<RoutePlan, RoutePlanningError>,
) -> ShadowRouteComparison {
    let active_outcome = match active {
        Ok(route) => CredentialFreeRouteOutcome::Selected(route.identity()),
        Err(ProviderRoutingError::UnsupportedModel(_)) => {
            CredentialFreeRouteOutcome::UnsupportedModel
        }
        Err(ProviderRoutingError::NoEligibleRoute(_)) => {
            CredentialFreeRouteOutcome::NoEligibleRoute
        }
        Err(ProviderRoutingError::CapacityUnavailable { .. }) => {
            CredentialFreeRouteOutcome::NoEligibleRoute
        }
    };
    let (shadow_outcome, decision, candidate_count) = match shadow {
        Ok(plan) => (
            CredentialFreeRouteOutcome::Selected(plan.selected.clone()),
            Some(plan.decision),
            plan.eligible_routes.len(),
        ),
        Err(RoutePlanningError::UnsupportedModel(_)) => {
            (CredentialFreeRouteOutcome::UnsupportedModel, None, 0)
        }
        Err(RoutePlanningError::NoEligibleRoute(_)) => {
            (CredentialFreeRouteOutcome::NoEligibleRoute, None, 0)
        }
    };

    if active_outcome == shadow_outcome {
        ShadowRouteComparison::Match {
            outcome: active_outcome,
            decision,
            candidate_count,
        }
    } else {
        ShadowRouteComparison::Mismatch {
            active: active_outcome,
            shadow: shadow_outcome,
            decision,
            candidate_count,
        }
    }
}

#[derive(Debug, Clone)]
struct WeightedSelection<'a> {
    route: &'a EligibleRoute,
    bucket: u8,
}

fn select_weighted_route<'a>(
    account_uuid: Uuid,
    routes: &'a [EligibleRoute],
) -> Option<WeightedSelection<'a>> {
    if routes.is_empty() {
        return None;
    }

    let total_weight = routes
        .iter()
        .map(|route| route.effective_weight)
        .sum::<u32>();
    if total_weight == 0 {
        return None;
    }

    let bucket = stable_account_bucket(account_uuid);
    let mut cumulative = 0u32;

    for (index, route) in routes.iter().enumerate() {
        let bucket_span = if index == routes.len() - 1 {
            100u32.saturating_sub(cumulative)
        } else {
            (route.effective_weight * 100) / total_weight
        };
        cumulative = cumulative.saturating_add(bucket_span);

        if u32::from(bucket) < cumulative || index == routes.len() - 1 {
            return Some(WeightedSelection { route, bucket });
        }
    }

    None
}

fn stable_account_bucket(account_uuid: Uuid) -> u8 {
    (u128::from_be_bytes(*account_uuid.as_bytes()) % 100) as u8
}

fn provider_routing_error_from_plan(error: RoutePlanningError) -> ProviderRoutingError {
    match error {
        RoutePlanningError::UnsupportedModel(model) => {
            ProviderRoutingError::UnsupportedModel(model)
        }
        RoutePlanningError::NoEligibleRoute(model) => ProviderRoutingError::NoEligibleRoute(model),
    }
}

fn ceil_retry_after(duration: Duration) -> Duration {
    let seconds = duration
        .as_secs()
        .saturating_add(u64::from(duration.subsec_nanos() > 0))
        .max(1);
    Duration::from_secs(seconds)
}

fn route_is_available_for_new_request(disposition: ShadowDisposition) -> bool {
    !matches!(disposition, ShadowDisposition::WouldOpen { .. })
}

fn proxy_for_provider(proxy_router: &ProxyRouter, provider: ProviderId) -> Option<ProxyConfig> {
    match provider {
        ProviderId::Tinfoil => Some(proxy_router.get_tinfoil_proxy()),
        ProviderId::Continuum => {
            let proxy = proxy_router.get_default_proxy();
            (proxy.provider_name == ProviderId::Continuum.as_str()).then_some(proxy)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::health::{ShadowDisposition, ShadowObservationMode};
    use crate::inference::{
        AttemptFailure, AttemptFailureKind, AttemptStage, AttemptTerminal, InferenceSurface,
        ReplaySafety, WorkloadClass,
    };
    use crate::model_config::{
        ModelAliasTargets, ModelPlan, PaidModelAliasOverrides, AUTO_POWERFUL_MODEL_ID,
        AUTO_QUICK_MODEL_ID, DEEPSEEK_V4_FLASH_MODEL_ID, GLM_5_2_MODEL_ID, GLM_5_3_FLASH_MODEL_ID,
        GLM_5_3_MODEL_ID, KIMI_K2_6_MODEL_ID, KIMI_K3_MODEL_ID, QUICK_MODEL_ID,
    };
    use crate::os_flags::PAID_POWERFUL_GLM_5_3_ALIAS_FLAG_KEY;
    use std::collections::HashMap;
    use std::time::Duration;

    fn proxy_router_with_both_providers() -> ProxyRouter {
        ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        )
    }

    fn uuid_for_bucket(bucket: u8) -> Uuid {
        Uuid::from_u128(u128::from(bucket))
    }

    fn intent(requested_model: &str, public_model: &str) -> InferenceIntent {
        InferenceIntent::new(
            uuid_for_bucket(73),
            requested_model,
            public_model,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        )
    }

    fn capacity_terminal(
        provider: ProviderId,
        public_model: &str,
        provider_model: &str,
        status: u16,
        retry_after: Duration,
    ) -> AttemptTerminal {
        let mut failure = AttemptFailure::new(
            AttemptFailureKind::CapacityRejected,
            AttemptStage::AwaitingResponse,
            ReplaySafety::ProvenPreAcceptance,
        );
        failure.status = Some(status);
        failure.retry_after = Some(retry_after);
        let route = RouteIdentity::new(
            provider,
            public_model,
            provider_model,
            public_model,
            RouteSelectionSource::DefaultProvider,
            None,
        );
        AttemptTerminal::Failed {
            attempt: intent(public_model, public_model)
                .begin_execution()
                .begin_attempt(route),
            failure,
        }
    }

    fn route_failure_terminal(
        provider: ProviderId,
        public_model: &str,
        provider_model: &str,
    ) -> AttemptTerminal {
        let failure = AttemptFailure::new(
            AttemptFailureKind::StreamTimeout,
            AttemptStage::Stream,
            ReplaySafety::NotProvenPreAcceptance,
        );
        let route = RouteIdentity::new(
            provider,
            public_model,
            provider_model,
            public_model,
            RouteSelectionSource::DefaultProvider,
            None,
        );
        AttemptTerminal::Failed {
            attempt: intent(public_model, public_model)
                .begin_execution()
                .begin_attempt(route),
            failure,
        }
    }

    #[test]
    fn test_stable_account_bucket_uses_uuid_mod_100() {
        assert_eq!(stable_account_bucket(uuid_for_bucket(0)), 0);
        assert_eq!(stable_account_bucket(uuid_for_bucket(49)), 49);
        assert_eq!(stable_account_bucket(uuid_for_bucket(50)), 50);
        assert_eq!(stable_account_bucket(uuid_for_bucket(99)), 99);
    }

    #[test]
    fn router_v2_flag_is_strictly_opt_in() {
        assert_eq!(
            InferenceRoutingMode::from_router_v2_flag(Some(true)),
            InferenceRoutingMode::V2
        );
        assert_eq!(
            InferenceRoutingMode::from_router_v2_flag(Some(false)),
            InferenceRoutingMode::Legacy
        );
        assert_eq!(
            InferenceRoutingMode::from_router_v2_flag(None),
            InferenceRoutingMode::Legacy
        );
    }

    #[test]
    fn healthy_router_v2_seam_is_route_identical_to_legacy() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let cases = [
            (GLM_5_2_MODEL_ID, None),
            (
                GLM_5_2_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            ),
            (GLM_5_3_MODEL_ID, None),
            (
                GLM_5_3_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
            ),
            (GLM_5_3_FLASH_MODEL_ID, None),
            ("kimi-k2-6", None),
            ("gpt-oss-120b", None),
        ];

        for (model, preference) in cases {
            let intent = intent(model, model);
            let legacy = router
                .select_completion_route_for_mode(
                    &proxy_router,
                    &intent,
                    preference,
                    InferenceRoutingMode::Legacy,
                )
                .expect("legacy route");
            let v2 = router
                .select_completion_route_for_mode(
                    &proxy_router,
                    &intent,
                    preference,
                    InferenceRoutingMode::V2,
                )
                .expect("v2 route");

            assert_eq!(
                legacy.proxy.provider_name, v2.proxy.provider_name,
                "{model}"
            );
            assert_eq!(legacy.proxy.base_url, v2.proxy.base_url, "{model}");
            assert_eq!(legacy.public_model_id, v2.public_model_id, "{model}");
            assert_eq!(legacy.provider_model_id, v2.provider_model_id, "{model}");
            assert_eq!(legacy.response_model_id, v2.response_model_id, "{model}");
            assert_eq!(legacy.bucket, v2.bucket, "{model}");
            assert_eq!(legacy.selection_source, v2.selection_source, "{model}");
        }
    }

    #[test]
    fn router_v2_gate_is_the_only_glm_health_activation_boundary() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);

        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                GLM_5_3_MODEL_ID,
                "glm-5.3",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let legacy = router
            .select_completion_route_for_mode(
                &proxy_router,
                &intent,
                None,
                InferenceRoutingMode::Legacy,
            )
            .expect("legacy GLM route ignores Router v2 health");
        let v2 = router
            .select_completion_route_for_mode(
                &proxy_router,
                &intent,
                None,
                InferenceRoutingMode::V2,
            )
            .expect("Router v2 selects the healthy GLM route");

        assert_eq!(legacy.provider, ProviderId::Continuum);
        assert_eq!(v2.provider, ProviderId::Tinfoil);
        assert_eq!(legacy.public_model_id, v2.public_model_id);
    }

    #[test]
    fn test_golden_completion_route_matrix_by_selector_plan_and_provider_preference() {
        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            selector: &'static str,
            plan: ModelPlan,
            provider_preference: Option<ProviderPreference>,
            continuum_available: bool,
            expected_access: bool,
            expected_public_model: &'static str,
            expected_provider: &'static str,
            expected_provider_model: &'static str,
            expected_source: RouteSelectionSource,
        }

        let cases = [
            Case {
                name: "free auto quick",
                selector: AUTO_QUICK_MODEL_ID,
                plan: ModelPlan::Free,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: QUICK_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: QUICK_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "free auto powerful remains unavailable",
                selector: AUTO_POWERFUL_MODEL_ID,
                plan: ModelPlan::Free,
                provider_preference: None,
                continuum_available: true,
                expected_access: false,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "paid auto quick",
                selector: AUTO_QUICK_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: DEEPSEEK_V4_FLASH_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: DEEPSEEK_V4_FLASH_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "paid auto powerful uses GLM default",
                selector: AUTO_POWERFUL_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "explicit K3 is independent of the auto target",
                selector: KIMI_K3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: KIMI_K3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: KIMI_K3_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "explicit GLM default",
                selector: GLM_5_2_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "explicit GLM 5.3 Tinfoil preference",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_3_MODEL_ID,
                expected_source: RouteSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM 5.3 Continuum preference",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "continuum",
                expected_provider_model: "glm-5.3",
                expected_source: RouteSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM 5.3 Tinfoil preference without a Continuum proxy",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
                continuum_available: false,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_3_MODEL_ID,
                expected_source: RouteSelectionSource::FeatureFlag,
            },
        ];

        let router = ProviderRouter::default();
        for case in cases {
            let alias_targets = ModelAliasTargets::for_plan(case.plan);
            let resolved_model = alias_targets.resolve(case.selector);
            let access = case.plan.allows_model(resolved_model);

            assert_eq!(access, case.expected_access, "{}", case.name);
            if !case.expected_access {
                continue;
            }

            let proxy_router = if case.continuum_available {
                proxy_router_with_both_providers()
            } else {
                ProxyRouter::new(
                    "https://api.openai.com".to_string(),
                    None,
                    "http://tinfoil.example.com".to_string(),
                )
            };
            let selected = router
                .select_completion_route_with_preference(
                    &proxy_router,
                    uuid_for_bucket(73),
                    resolved_model,
                    case.provider_preference,
                )
                .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));

            assert_eq!(
                selected.public_model_id, case.expected_public_model,
                "{}",
                case.name
            );
            assert_eq!(
                selected.provider_model_id, case.expected_provider_model,
                "{}",
                case.name
            );
            assert_eq!(
                selected.response_model_id, case.expected_public_model,
                "{}",
                case.name
            );
            assert_eq!(
                selected.proxy.provider_name, case.expected_provider,
                "{}",
                case.name
            );
            assert_eq!(
                selected.selection_source, case.expected_source,
                "{}",
                case.name
            );
            assert_eq!(selected.bucket, None, "{}", case.name);

            let intent = InferenceIntent::new(
                uuid_for_bucket(73),
                case.selector,
                resolved_model,
                case.plan,
                InferenceSurface::ChatCompletions,
                WorkloadClass::Interactive,
            );
            let active = Ok(selected.clone());
            let shadow =
                router.shadow_completion_plan(&proxy_router, &intent, case.provider_preference);
            assert!(
                matches!(
                    compare_shadow_route(&active, &shadow),
                    ShadowRouteComparison::Match { .. }
                ),
                "{}: active={active:?}, shadow={shadow:?}",
                case.name
            );
        }
    }

    #[test]
    fn shadow_planner_matches_legacy_except_for_the_router_v2_glm_5_3_ga_fallback() {
        let router = ProviderRouter::default();
        let both = proxy_router_with_both_providers();
        let tinfoil_only = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        for proxy_router in [&both, &tinfoil_only] {
            for model in PROVIDER_REGISTRY.completion_models() {
                for bucket in [0, 29, 30, 69, 70, 99] {
                    let account_uuid = uuid_for_bucket(bucket);
                    let intent = InferenceIntent::new(
                        account_uuid,
                        model.public_model_id,
                        model.public_model_id,
                        ModelPlan::Paid,
                        InferenceSurface::Responses,
                        WorkloadClass::Interactive,
                    );
                    let active = router.select_completion_route_with_preference(
                        proxy_router,
                        account_uuid,
                        model.public_model_id,
                        None,
                    );
                    let shadow = router.shadow_completion_plan(proxy_router, &intent, None);
                    let comparison = compare_shadow_route(&active, &shadow);
                    let is_router_v2_only_glm_fallback = model.public_model_id == GLM_5_3_MODEL_ID
                        && proxy_for_provider(proxy_router, ProviderId::Continuum).is_none();

                    if is_router_v2_only_glm_fallback {
                        assert!(
                            matches!(
                                comparison,
                                ShadowRouteComparison::Mismatch {
                                    active: CredentialFreeRouteOutcome::NoEligibleRoute,
                                    shadow: CredentialFreeRouteOutcome::Selected(RouteIdentity {
                                        provider: ProviderId::Tinfoil,
                                        ..
                                    }),
                                    ..
                                }
                            ),
                            "model={}, bucket={bucket}, active={active:?}, shadow={shadow:?}",
                            model.public_model_id
                        );
                    } else {
                        assert!(
                            matches!(comparison, ShadowRouteComparison::Match { .. }),
                            "model={}, bucket={bucket}, active={active:?}, shadow={shadow:?}",
                            model.public_model_id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shadow_planner_matches_legacy_error_classes_for_unknown_models() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for model in [
            "unknown-model",
            "kimi-k-3",
            "kimi-k3-latest",
            "deepseek-v4-flash-0731",
        ] {
            let account_uuid = uuid_for_bucket(50);
            let intent = InferenceIntent::new(
                account_uuid,
                model,
                model,
                ModelPlan::Paid,
                InferenceSurface::ChatCompletions,
                WorkloadClass::Interactive,
            );
            let active = router.select_completion_route_with_preference(
                &proxy_router,
                account_uuid,
                model,
                None,
            );
            let shadow = router.shadow_completion_plan(&proxy_router, &intent, None);

            assert!(matches!(
                compare_shadow_route(&active, &shadow),
                ShadowRouteComparison::Match {
                    outcome: CredentialFreeRouteOutcome::UnsupportedModel,
                    ..
                }
            ));
        }
    }

    #[test]
    fn legacy_selector_and_baseline_planner_remain_health_independent() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let account_uuid = uuid_for_bucket(73);
        let provider_preference = Some(ProviderPreference::feature_flag(ProviderId::Tinfoil));
        let intent = InferenceIntent::new(
            account_uuid,
            GLM_5_3_MODEL_ID,
            GLM_5_3_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );

        let active_before = router
            .select_completion_route_with_preference(
                &proxy_router,
                account_uuid,
                GLM_5_3_MODEL_ID,
                provider_preference,
            )
            .expect("active route before shadow health");
        let shadow_before = router
            .shadow_completion_plan(&proxy_router, &intent, provider_preference)
            .expect("shadow route before shadow health");

        let mut failure = AttemptFailure::new(
            AttemptFailureKind::CapacityRejected,
            AttemptStage::AwaitingResponse,
            ReplaySafety::ProvenPreAcceptance,
        );
        failure.status = Some(429);
        failure.retry_after = Some(Duration::from_secs(60));
        let terminal = AttemptTerminal::Failed {
            attempt: intent
                .begin_execution()
                .begin_attempt(active_before.identity()),
            failure,
        };
        let report = router.observe_attempt_terminal(&terminal, ShadowObservationMode::Update);
        assert!(matches!(
            report.snapshot.expect("known route").effective,
            ShadowDisposition::WouldOpen { .. }
        ));

        let active_after = router
            .select_completion_route_with_preference(
                &proxy_router,
                account_uuid,
                GLM_5_3_MODEL_ID,
                provider_preference,
            )
            .expect("active route after shadow health");
        let shadow_after = router
            .shadow_completion_plan(&proxy_router, &intent, provider_preference)
            .expect("shadow route after shadow health");

        assert_eq!(active_after.identity(), active_before.identity());
        assert_eq!(shadow_after, shadow_before);
        assert!(matches!(
            compare_shadow_route(&Ok(active_after), &Ok(shadow_after)),
            ShadowRouteComparison::Match { .. }
        ));
    }

    #[test]
    fn active_glm_canary_preserves_healthy_preferences_and_canonical_identity() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);

        let cases = [
            (
                None,
                ProviderId::Continuum,
                "glm-5.3",
                RouteSelectionSource::DefaultProvider,
            ),
            (
                Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
                ProviderId::Tinfoil,
                GLM_5_3_MODEL_ID,
                RouteSelectionSource::FeatureFlag,
            ),
            (
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
                ProviderId::Continuum,
                "glm-5.3",
                RouteSelectionSource::FeatureFlag,
            ),
        ];

        for (preference, provider, provider_model, source) in cases {
            let selected = router
                .select_active_completion_route(&proxy_router, &intent, preference)
                .expect("healthy GLM route");
            assert_eq!(selected.provider, provider);
            assert_eq!(selected.provider_model_id, provider_model);
            assert_eq!(selected.public_model_id, GLM_5_3_MODEL_ID);
            assert_eq!(selected.response_model_id, GLM_5_3_MODEL_ID);
            assert_eq!(selected.selection_source, source);
        }
    }

    #[test]
    fn active_glm_canary_switches_only_new_requests_to_same_model_alternate() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);

        let pinned_before_failure = router
            .select_active_completion_route(&proxy_router, &intent, None)
            .expect("initial Continuum route");
        assert_eq!(pinned_before_failure.provider, ProviderId::Continuum);

        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                GLM_5_3_MODEL_ID,
                "glm-5.3",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        // The previously returned pin is immutable; only a fresh preparation
        // observes the newly opened circuit.
        assert_eq!(pinned_before_failure.provider, ProviderId::Continuum);
        assert_eq!(pinned_before_failure.provider_model_id, "glm-5.3");

        let selected_after_failure = router
            .select_active_completion_route(&proxy_router, &intent, None)
            .expect("Tinfoil GLM fallback");
        assert_eq!(selected_after_failure.provider, ProviderId::Tinfoil);
        assert_eq!(selected_after_failure.provider_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected_after_failure.public_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(
            selected_after_failure.selection_source,
            RouteSelectionSource::Fallback
        );
    }

    #[test]
    fn active_glm_canary_bypasses_an_open_feature_flag_preference() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);

        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Tinfoil,
                GLM_5_3_MODEL_ID,
                GLM_5_3_MODEL_ID,
                503,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let selected = router
            .select_active_completion_route(
                &proxy_router,
                &intent,
                Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
            )
            .expect("Continuum GLM fallback");
        assert_eq!(selected.provider, ProviderId::Continuum);
        assert_eq!(selected.provider_model_id, "glm-5.3");
        assert_eq!(selected.selection_source, RouteSelectionSource::Fallback);
    }

    #[test]
    fn active_glm_canary_returns_typed_capacity_when_every_configured_route_is_open() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);

        for terminal in [
            capacity_terminal(
                ProviderId::Tinfoil,
                GLM_5_3_MODEL_ID,
                GLM_5_3_MODEL_ID,
                429,
                Duration::from_secs(40),
            ),
            capacity_terminal(
                ProviderId::Continuum,
                GLM_5_3_MODEL_ID,
                "glm-5.3",
                529,
                Duration::from_secs(10),
            ),
        ] {
            router.observe_attempt_terminal(&terminal, ShadowObservationMode::Update);
        }

        let error = router
            .select_active_completion_route(&proxy_router, &intent, None)
            .expect_err("both GLM routes are open");
        match error {
            ProviderRoutingError::CapacityUnavailable { model, retry_after } => {
                assert_eq!(model, GLM_5_3_MODEL_ID);
                assert_eq!(retry_after, Duration::from_secs(30));
            }
            other => panic!("unexpected route error: {other:?}"),
        }
    }

    #[test]
    fn active_glm_canary_uses_route_failure_threshold_but_not_watch_state() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID);
        let failure = || route_failure_terminal(ProviderId::Continuum, GLM_5_3_MODEL_ID, "glm-5.3");

        for observed_failures in 1..=3 {
            router.observe_attempt_terminal(&failure(), ShadowObservationMode::Update);
            let selected = router
                .select_active_completion_route(&proxy_router, &intent, None)
                .expect("GLM route");
            let expected = if observed_failures < 3 {
                ProviderId::Continuum
            } else {
                ProviderId::Tinfoil
            };
            assert_eq!(selected.provider, expected, "failure {observed_failures}");
        }
    }

    #[test]
    fn active_glm_canary_consumes_continuum_account_429_without_changing_kimi_routing() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                KIMI_K2_6_MODEL_ID,
                "kimi-k2.6",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let glm = router
            .select_active_completion_route(
                &proxy_router,
                &intent(GLM_5_3_MODEL_ID, GLM_5_3_MODEL_ID),
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            )
            .expect("Tinfoil GLM after Continuum account limit");
        assert_eq!(glm.provider, ProviderId::Tinfoil);

        let kimi = router
            .select_active_completion_route(
                &proxy_router,
                &intent(KIMI_K2_6_MODEL_ID, KIMI_K2_6_MODEL_ID),
                None,
            )
            .expect("Kimi remains on its legacy route in Stack 6");
        assert_eq!(kimi.provider, ProviderId::Continuum);
        assert_eq!(kimi.provider_model_id, "kimi-k2.6");
    }

    #[test]
    fn active_glm_health_filter_is_inert_for_auto_and_other_models() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                GLM_5_3_MODEL_ID,
                "glm-5.3",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let synthetic_auto_glm = intent(AUTO_POWERFUL_MODEL_ID, GLM_5_3_MODEL_ID);
        let auto_route = router
            .select_active_completion_route(&proxy_router, &synthetic_auto_glm, None)
            .expect("Auto stays on legacy selection until Stack 7");
        assert_eq!(auto_route.provider, ProviderId::Continuum);

        for (requested, public, expected_provider) in [
            (GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID, ProviderId::Tinfoil),
            (
                GLM_5_3_FLASH_MODEL_ID,
                GLM_5_3_FLASH_MODEL_ID,
                ProviderId::Tinfoil,
            ),
            (KIMI_K3_MODEL_ID, KIMI_K3_MODEL_ID, ProviderId::Tinfoil),
            (
                KIMI_K2_6_MODEL_ID,
                KIMI_K2_6_MODEL_ID,
                ProviderId::Continuum,
            ),
            (QUICK_MODEL_ID, QUICK_MODEL_ID, ProviderId::Tinfoil),
        ] {
            let selected = router
                .select_active_completion_route(&proxy_router, &intent(requested, public), None)
                .expect("non-canary route");
            assert_eq!(selected.provider, expected_provider, "{requested}");
        }
    }

    #[test]
    fn non_canary_glm_models_ignore_their_own_open_health_in_stack_6() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for model in [GLM_5_2_MODEL_ID, GLM_5_3_FLASH_MODEL_ID] {
            router.observe_attempt_terminal(
                &capacity_terminal(
                    ProviderId::Tinfoil,
                    model,
                    model,
                    429,
                    Duration::from_secs(60),
                ),
                ShadowObservationMode::Update,
            );

            let selected = router
                .select_active_completion_route(&proxy_router, &intent(model, model), None)
                .expect("non-canary GLM route remains on legacy selection");
            assert_eq!(selected.provider, ProviderId::Tinfoil, "{model}");
            assert_eq!(selected.public_model_id, model, "{model}");
            assert_eq!(selected.provider_model_id, model, "{model}");
        }
    }

    #[test]
    fn only_would_open_blocks_a_new_canary_request() {
        assert!(route_is_available_for_new_request(
            ShadowDisposition::Healthy
        ));
        assert!(route_is_available_for_new_request(
            ShadowDisposition::Watch {
                consecutive_failures: 2
            }
        ));
        assert!(route_is_available_for_new_request(
            ShadowDisposition::WouldProbe
        ));
        assert!(!route_is_available_for_new_request(
            ShadowDisposition::WouldOpen {
                remaining: Duration::from_secs(1)
            }
        ));
    }

    #[test]
    fn retry_after_rounds_up_and_never_returns_zero() {
        assert_eq!(ceil_retry_after(Duration::ZERO), Duration::from_secs(1));
        assert_eq!(
            ceil_retry_after(Duration::from_nanos(1)),
            Duration::from_secs(1)
        );
        assert_eq!(
            ceil_retry_after(Duration::from_millis(1_001)),
            Duration::from_secs(2)
        );
    }

    #[test]
    fn test_glm_5_2_always_uses_tinfoil() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for bucket in [0, 29, 30, 69, 70, 99] {
            let selected = router
                .select_completion_route(&proxy_router, uuid_for_bucket(bucket), GLM_5_2_MODEL_ID)
                .expect("route");

            assert_eq!(selected.proxy.provider_name, "tinfoil");
            assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
            assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
            assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
            assert_eq!(selected.bucket, None);
            assert_eq!(
                selected.selection_source,
                RouteSelectionSource::DefaultProvider
            );
        }
    }

    #[test]
    fn test_glm_5_2_and_alias_have_no_provider_routing_flag() {
        let router = ProviderRouter::default();

        assert_eq!(
            router.provider_routing_flag_for_completion_model(GLM_5_2_MODEL_ID),
            None
        );
        assert_eq!(
            router.provider_routing_flag_for_completion_model(
                crate::model_config::AUTO_POWERFUL_MODEL_ID
            ),
            None
        );
        assert_eq!(
            router.provider_routing_flag_for_completion_model("kimi-k2-6"),
            None
        );
        assert_eq!(
            router.provider_routing_flag_for_completion_model("gpt-oss-120b"),
            None
        );
    }

    #[test]
    fn test_glm_5_3_tinfoil_flag_maps_true_and_false_to_separate_providers() {
        let router = ProviderRouter::default();
        let flag = router
            .provider_routing_flag_for_completion_model(GLM_5_3_MODEL_ID)
            .expect("GLM 5.3 provider flag");

        assert_eq!(flag.key(), GLM_5_3_TINFOIL_FLAG_KEY);
        assert_eq!(flag.preference_for(true).provider(), ProviderId::Tinfoil);
        assert_eq!(flag.preference_for(false).provider(), ProviderId::Continuum);
        assert_eq!(
            flag.preference_for(true).source(),
            RouteSelectionSource::FeatureFlag
        );
    }

    #[test]
    fn test_glm_5_2_rejects_continuum_preference_and_stays_on_tinfoil() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(1),
                GLM_5_2_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, RouteSelectionSource::Fallback);
    }

    #[test]
    fn test_generic_tinfoil_preference_selects_glm_5_2_route() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(99),
                GLM_5_2_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, RouteSelectionSource::FeatureFlag);
    }

    #[test]
    fn test_glm_continuum_preference_falls_back_to_tinfoil_when_unavailable() {
        let router = ProviderRouter::default();
        let tinfoil_only = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let selected = router
            .select_completion_route_with_preference(
                &tinfoil_only,
                uuid_for_bucket(70),
                GLM_5_2_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, RouteSelectionSource::Fallback);
    }

    #[test]
    fn test_kimi_always_uses_continuum() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for bucket in [0, 29, 30, 69, 70, 99] {
            let selected = router
                .select_completion_route(&proxy_router, uuid_for_bucket(bucket), "kimi-k2-6")
                .expect("route");

            assert_eq!(selected.proxy.provider_name, "continuum");
            assert_eq!(selected.public_model_id, "kimi-k2-6");
            assert_eq!(selected.provider_model_id, "kimi-k2.6");
            assert_eq!(selected.response_model_id, "kimi-k2-6");
            assert_eq!(selected.bucket, None);
            assert_eq!(
                selected.selection_source,
                RouteSelectionSource::DefaultProvider
            );
        }
    }

    #[test]
    fn test_glm_5_3_always_uses_continuum_and_canonicalizes_the_response() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), GLM_5_3_MODEL_ID)
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.public_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.provider_model_id, "glm-5.3");
        assert_eq!(selected.response_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            RouteSelectionSource::DefaultProvider
        );
    }

    #[test]
    fn test_glm_5_3_tinfoil_route_requires_explicit_enabled_preference() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let flag = router
            .provider_routing_flag_for_completion_model(GLM_5_3_MODEL_ID)
            .expect("GLM 5.3 provider flag");

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(50),
                GLM_5_3_MODEL_ID,
                Some(flag.preference_for(true)),
            )
            .expect("Tinfoil route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.provider_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.response_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, RouteSelectionSource::FeatureFlag);

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(50),
                GLM_5_3_MODEL_ID,
                Some(flag.preference_for(false)),
            )
            .expect("PrivateMode route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.provider_model_id, "glm-5.3");
        assert_eq!(selected.selection_source, RouteSelectionSource::FeatureFlag);
    }

    #[test]
    fn test_auto_powerful_uses_glm_route_table() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(
                &proxy_router,
                uuid_for_bucket(70),
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            RouteSelectionSource::DefaultProvider
        );
    }

    #[test]
    fn test_paid_powerful_glm_5_3_override_uses_continuum_route() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let flags = HashMap::from([(PAID_POWERFUL_GLM_5_3_ALIAS_FLAG_KEY.to_string(), true)]);
        let targets = ModelAliasTargets::for_plan_with_overrides(
            ModelPlan::Paid,
            PaidModelAliasOverrides::from_flag_values(&flags),
        );

        let selected = router
            .select_completion_route(
                &proxy_router,
                uuid_for_bucket(70),
                targets.resolve(crate::model_config::AUTO_POWERFUL_MODEL_ID),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.public_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.provider_model_id, "glm-5.3");
        assert_eq!(selected.response_model_id, GLM_5_3_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            RouteSelectionSource::DefaultProvider
        );
    }

    #[test]
    fn test_kimi_has_no_tinfoil_fallback_when_continuum_proxy_is_missing() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let error = router
            .select_completion_route(&proxy_router, uuid_for_bucket(1), "kimi-k2-6")
            .expect_err("no eligible Kimi route");

        assert_eq!(
            error,
            ProviderRoutingError::NoEligibleRoute("kimi-k2-6".to_string())
        );
    }

    #[test]
    fn test_non_configured_model_preserves_existing_tinfoil_completion_route() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), "gpt-oss-120b")
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, "gpt-oss-120b");
        assert_eq!(selected.provider_model_id, "gpt-oss-120b");
        assert_eq!(selected.response_model_id, "gpt-oss-120b");
        assert_eq!(selected.bucket, None);
    }

    #[test]
    fn test_new_tinfoil_models_preserve_canonical_ids() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for model_id in ["kimi-k3", "deepseek-v4-flash", "glm-5-3-flash"] {
            let selected = router
                .select_completion_route(&proxy_router, uuid_for_bucket(50), model_id)
                .expect("canonical Tinfoil model should route");

            assert_eq!(selected.proxy.provider_name, "tinfoil");
            assert_eq!(selected.public_model_id, model_id);
            assert_eq!(selected.provider_model_id, model_id);
            assert_eq!(selected.response_model_id, model_id);
            assert_eq!(selected.bucket, None);
        }
    }

    #[test]
    fn test_new_tinfoil_models_reject_near_spellings() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        for model_id in [
            "kimi-k-3",
            "kimi-k3-latest",
            "deepseek-v4-flash-0731",
            "deepseek-v4flash",
            "glm-5.3-flash",
            "glm-5-3-flash-latest",
        ] {
            let error = router
                .select_completion_route(&proxy_router, uuid_for_bucket(50), model_id)
                .expect_err("non-canonical model spelling should be rejected");

            assert_eq!(
                error,
                ProviderRoutingError::UnsupportedModel(model_id.to_string())
            );
        }
    }

    #[test]
    fn test_tinfoil_fallback_resolves_known_alias_before_provider_request() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let selected = router
            .select_completion_route(
                &proxy_router,
                uuid_for_bucket(50),
                crate::model_config::AUTO_QUICK_MODEL_ID,
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(
            selected.public_model_id,
            crate::model_config::QUICK_MODEL_ID
        );
        assert_eq!(
            selected.provider_model_id,
            crate::model_config::QUICK_MODEL_ID
        );
        assert_eq!(
            selected.response_model_id,
            crate::model_config::QUICK_MODEL_ID
        );
        assert_eq!(selected.bucket, None);
    }

    #[test]
    fn test_tinfoil_fallback_rejects_unknown_model_passthrough() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let error = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), "provider-native-model")
            .expect_err("unsupported model");

        assert_eq!(
            error,
            ProviderRoutingError::UnsupportedModel("provider-native-model".to_string())
        );
    }

    #[test]
    fn test_tinfoil_fallback_rejects_unknown_models() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let error = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), "unknown-model")
            .expect_err("unsupported model");

        assert_eq!(
            error,
            ProviderRoutingError::UnsupportedModel("unknown-model".to_string())
        );
    }

    #[test]
    fn test_configured_model_errors_when_continuum_is_missing() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let error = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), "kimi-k2-6")
            .expect_err("no eligible Kimi route");

        assert_eq!(
            error,
            ProviderRoutingError::NoEligibleRoute("kimi-k2-6".to_string())
        );

        let error = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), GLM_5_3_MODEL_ID)
            .expect_err("no eligible GLM 5.3 route");

        assert_eq!(
            error,
            ProviderRoutingError::NoEligibleRoute(GLM_5_3_MODEL_ID.to_string())
        );
    }
}
