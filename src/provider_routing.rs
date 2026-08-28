use crate::inference::health::{
    ShadowDisposition, ShadowHealthState, ShadowObservationMode, ShadowObservationReport,
    ShadowRouteSnapshot, MIN_CAPACITY_COOLDOWN,
};
use crate::inference::{AttemptTerminal, InferenceIntent, RouteIdentity, RouteKey};
use crate::inference_planning::{
    plan_completion_model_candidates, plan_completion_route, ConfiguredProviders,
    ProviderPreference, RoutePlan, RoutePlanningError, RoutePlanningInput,
};
#[cfg(test)]
use crate::model_config::resolve_completion_model_id;
use crate::model_config::{resolve_public_model_id, GLM_5_2_MODEL_ID};
use crate::os_flags::GLM_5_2_CONTINUUM_FLAG_KEY;
use crate::provider_registry::{
    ProviderId, ProviderRegistry, RouteSelectionSource, PROVIDER_REGISTRY,
};
#[cfg(test)]
use crate::proxy_config::canonicalize_tinfoil_model;
use crate::proxy_config::{ProxyConfig, ProxyRouter};
use std::time::Duration;
#[cfg(test)]
use uuid::Uuid;

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct ProviderConfig {
    provider: ProviderId,
    weight: u16,
    enabled: bool,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct ModelProviderRoute {
    provider: ProviderId,
    provider_model_id: &'static str,
    weight: u16,
    enabled: bool,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct ModelRoutingConfig {
    public_model_id: &'static str,
    routes: &'static [ModelProviderRoute],
    default_provider: Option<ProviderId>,
}

#[cfg(test)]
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
    pub(crate) model_selection_source: ModelSelectionSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSelectionSource {
    Explicit,
    AutoPrimary,
    AutoFallback,
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
    #[cfg(test)]
    config: &'static ProviderRoutingConfig,
    registry: &'static ProviderRegistry,
    shadow_health: ShadowHealthState,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct EligibleRoute {
    provider: ProviderId,
    proxy: ProxyConfig,
    provider_model_id: &'static str,
    effective_weight: u32,
}

#[cfg(test)]
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

#[cfg(test)]
const KIMI_K2_6_ROUTES: &[ModelProviderRoute] = &[ModelProviderRoute {
    provider: ProviderId::Continuum,
    provider_model_id: "kimi-k2.6",
    weight: 100,
    enabled: true,
}];

#[cfg(test)]
const GLM_5_2_ROUTES: &[ModelProviderRoute] = &[
    ModelProviderRoute {
        provider: ProviderId::Tinfoil,
        provider_model_id: GLM_5_2_MODEL_ID,
        weight: 100,
        enabled: true,
    },
    ModelProviderRoute {
        provider: ProviderId::Continuum,
        provider_model_id: "glm-5.2",
        weight: 100,
        enabled: true,
    },
];

#[cfg(test)]
const MODEL_ROUTES: &[ModelRoutingConfig] = &[
    ModelRoutingConfig {
        public_model_id: "kimi-k2-6",
        routes: KIMI_K2_6_ROUTES,
        default_provider: Some(ProviderId::Continuum),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_2_MODEL_ID,
        routes: GLM_5_2_ROUTES,
        default_provider: Some(ProviderId::Tinfoil),
    },
];

#[cfg(test)]
static DEFAULT_PROVIDER_ROUTING_CONFIG: ProviderRoutingConfig = ProviderRoutingConfig {
    providers: PROVIDERS,
    models: MODEL_ROUTES,
};

impl Default for ProviderRouter {
    fn default() -> Self {
        Self {
            #[cfg(test)]
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

    #[cfg(test)]
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

    /// Selects the route used by a newly prepared logical request.
    ///
    /// Stack 6 activated the first explicit GLM canary. Stack 7 applies the
    /// same future-request-only circuit filtering to every completion model and
    /// composes it with the narrow Auto Powerful K3/K2.6 candidate policy.
    /// Explicit requests and Auto Quick still contain exactly one public model.
    pub(crate) fn select_active_completion_route(
        &self,
        proxy_router: &ProxyRouter,
        intent: &InferenceIntent,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        let model_plan = plan_completion_model_candidates(intent);
        let candidates = model_plan
            .public_model_ids
            .iter()
            .map(|public_model_id| intent.for_candidate_public_model(public_model_id.clone()))
            .collect::<Vec<_>>();
        self.select_health_aware_model_candidates(
            proxy_router,
            intent,
            &candidates,
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

    fn select_health_aware_model_candidates(
        &self,
        proxy_router: &ProxyRouter,
        preferred_intent: &InferenceIntent,
        candidates: &[InferenceIntent],
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        if candidates.is_empty() {
            return Err(ProviderRoutingError::NoEligibleRoute(
                preferred_intent.public_model_id.clone(),
            ));
        }

        let mut configured_routes = Vec::new();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let model = self
                .registry
                .completion_model(&candidate.public_model_id)
                .ok_or_else(|| {
                    ProviderRoutingError::UnsupportedModel(candidate.public_model_id.clone())
                })?;
            for route in model.routes {
                let Some(provider) = self.registry.provider(route.provider) else {
                    continue;
                };
                if !route.enabled
                    || route.weight == 0
                    || !provider.enabled
                    || provider.weight == 0
                    || proxy_for_provider(proxy_router, route.provider).is_none()
                {
                    continue;
                }
                configured_routes.push((
                    candidate_index,
                    route.provider,
                    RouteKey {
                        provider: route.provider,
                        provider_model_id: route.provider_model_id.to_string(),
                    },
                ));
            }
        }

        if configured_routes.is_empty() {
            return Err(ProviderRoutingError::NoEligibleRoute(
                preferred_intent.public_model_id.clone(),
            ));
        }

        let route_keys = configured_routes
            .iter()
            .map(|(_, _, route)| route.clone())
            .collect::<Vec<_>>();
        let snapshots = self
            .shadow_health
            .snapshot_routes(&route_keys)
            .ok_or_else(|| ProviderRoutingError::CapacityUnavailable {
                model: preferred_intent.public_model_id.clone(),
                retry_after: MIN_CAPACITY_COOLDOWN,
            })?;

        let mut available_providers = vec![ConfiguredProviders::none(); candidates.len()];
        let mut earliest_recovery = vec![None; candidates.len()];
        for ((candidate_index, provider, _), snapshot) in configured_routes.iter().zip(snapshots) {
            match snapshot.effective {
                ShadowDisposition::WouldOpen { remaining } => {
                    let remaining = ceil_retry_after(remaining);
                    earliest_recovery[*candidate_index] = Some(
                        earliest_recovery[*candidate_index]
                            .map_or(remaining, |current: Duration| current.min(remaining)),
                    );
                }
                disposition if route_is_available_for_new_request(disposition) => {
                    available_providers[*candidate_index] =
                        available_providers[*candidate_index].with_provider(*provider);
                }
                _ => unreachable!("WouldOpen is handled above"),
            }
        }

        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let configured = available_providers[candidate_index];
            if configured == ConfiguredProviders::none() {
                continue;
            }

            let candidate_preference = (candidate_index == 0)
                .then_some(provider_preference)
                .flatten();
            let plan = plan_completion_route(
                self.registry,
                RoutePlanningInput {
                    intent: candidate,
                    configured_providers: configured,
                    provider_preference: candidate_preference,
                },
            )
            .map_err(provider_routing_error_from_plan)?;

            let selected = plan.selected;
            let proxy = proxy_for_provider(proxy_router, selected.provider).ok_or_else(|| {
                ProviderRoutingError::NoEligibleRoute(candidate.public_model_id.clone())
            })?;
            return Ok(SelectedProviderRoute {
                provider: selected.provider,
                proxy,
                public_model_id: selected.public_model_id,
                provider_model_id: selected.provider_model_id,
                response_model_id: selected.response_model_id,
                bucket: selected.bucket,
                // Provider-route provenance remains independent from model
                // substitution provenance. AutoFallback below is the only
                // signal that a different public model was selected.
                selection_source: selected.selection_source,
                model_selection_source: match (
                    preferred_intent.selection_mode.is_auto(),
                    candidate_index,
                ) {
                    (false, _) => ModelSelectionSource::Explicit,
                    (true, 0) => ModelSelectionSource::AutoPrimary,
                    (true, _) => ModelSelectionSource::AutoFallback,
                },
            });
        }

        let retry_after = earliest_recovery
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(MIN_CAPACITY_COOLDOWN);
        Err(ProviderRoutingError::CapacityUnavailable {
            model: preferred_intent.public_model_id.clone(),
            retry_after,
        })
    }

    pub(crate) fn continuum_flag_key_for_completion_model(
        &self,
        requested_model: &str,
    ) -> Option<&'static str> {
        let public_model_id = resolve_public_model_id(requested_model)?;
        (public_model_id == GLM_5_2_MODEL_ID).then_some(GLM_5_2_CONTINUUM_FLAG_KEY)
    }

    #[cfg(test)]
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
            model_selection_source: ModelSelectionSource::Explicit,
        })
    }

    #[cfg(test)]
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
            model_selection_source: ModelSelectionSource::Explicit,
        })
    }

    #[cfg(test)]
    fn provider_config(&self, provider: ProviderId) -> Option<&ProviderConfig> {
        self.config
            .providers
            .iter()
            .find(|config| config.provider == provider)
    }

    #[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone)]
struct WeightedSelection<'a> {
    route: &'a EligibleRoute,
    bucket: u8,
}

#[cfg(test)]
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

#[cfg(test)]
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
        AUTO_QUICK_MODEL_ID, DEEPSEEK_V4_FLASH_MODEL_ID, KIMI_K3_MODEL_ID, POWERFUL_MODEL_ID,
        QUICK_MODEL_ID,
    };
    use crate::os_flags::PAID_POWERFUL_KIMI_K3_ALIAS_FLAG_KEY;
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
    fn test_golden_completion_route_matrix_by_selector_plan_and_provider_preference() {
        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            selector: &'static str,
            plan: ModelPlan,
            powerful_kimi_k3: bool,
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
                powerful_kimi_k3: false,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: QUICK_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: QUICK_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "free auto powerful remains unavailable when the paid flag is on",
                selector: AUTO_POWERFUL_MODEL_ID,
                plan: ModelPlan::Free,
                powerful_kimi_k3: true,
                provider_preference: None,
                continuum_available: true,
                expected_access: false,
                expected_public_model: POWERFUL_MODEL_ID,
                expected_provider: "continuum",
                expected_provider_model: "kimi-k2.6",
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "paid auto quick",
                selector: AUTO_QUICK_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: DEEPSEEK_V4_FLASH_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: DEEPSEEK_V4_FLASH_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "paid auto powerful, flag off",
                selector: AUTO_POWERFUL_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: POWERFUL_MODEL_ID,
                expected_provider: "continuum",
                expected_provider_model: "kimi-k2.6",
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "paid auto powerful, flag on",
                selector: AUTO_POWERFUL_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: true,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: KIMI_K3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: KIMI_K3_MODEL_ID,
                expected_source: RouteSelectionSource::StaticSplit,
            },
            Case {
                name: "explicit K3 ignores the paid auto flag",
                selector: KIMI_K3_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
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
                powerful_kimi_k3: false,
                provider_preference: None,
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::DefaultProvider,
            },
            Case {
                name: "explicit GLM Tinfoil preference",
                selector: GLM_5_2_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM Continuum preference",
                selector: GLM_5_2_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "continuum",
                expected_provider_model: "glm-5.2",
                expected_source: RouteSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM Continuum preference without a Continuum proxy",
                selector: GLM_5_2_MODEL_ID,
                plan: ModelPlan::Paid,
                powerful_kimi_k3: false,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
                continuum_available: false,
                expected_access: true,
                expected_public_model: GLM_5_2_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_2_MODEL_ID,
                expected_source: RouteSelectionSource::Fallback,
            },
        ];

        let router = ProviderRouter::default();
        for case in cases {
            let flags = HashMap::from([(
                PAID_POWERFUL_KIMI_K3_ALIAS_FLAG_KEY.to_string(),
                case.powerful_kimi_k3,
            )]);
            let alias_targets = ModelAliasTargets::for_plan_with_overrides(
                case.plan,
                PaidModelAliasOverrides::from_flag_values(&flags),
            );
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
    fn shadow_planner_matches_legacy_for_every_registered_model_and_topology() {
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
                    assert!(
                        matches!(
                            compare_shadow_route(&active, &shadow),
                            ShadowRouteComparison::Match { .. }
                        ),
                        "model={}, bucket={bucket}, active={active:?}, shadow={shadow:?}",
                        model.public_model_id
                    );
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
        let provider_preference = Some(ProviderPreference::feature_flag(ProviderId::Continuum));
        let intent = InferenceIntent::new(
            account_uuid,
            GLM_5_2_MODEL_ID,
            GLM_5_2_MODEL_ID,
            ModelPlan::Paid,
            InferenceSurface::Responses,
            WorkloadClass::Interactive,
        );

        let active_before = router
            .select_completion_route_with_preference(
                &proxy_router,
                account_uuid,
                GLM_5_2_MODEL_ID,
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
                GLM_5_2_MODEL_ID,
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
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);

        let cases = [
            (
                None,
                ProviderId::Tinfoil,
                GLM_5_2_MODEL_ID,
                RouteSelectionSource::DefaultProvider,
            ),
            (
                Some(ProviderPreference::feature_flag(ProviderId::Tinfoil)),
                ProviderId::Tinfoil,
                GLM_5_2_MODEL_ID,
                RouteSelectionSource::FeatureFlag,
            ),
            (
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
                ProviderId::Continuum,
                "glm-5.2",
                RouteSelectionSource::FeatureFlag,
            ),
        ];

        for (preference, provider, provider_model, source) in cases {
            let selected = router
                .select_active_completion_route(&proxy_router, &intent, preference)
                .expect("healthy GLM route");
            assert_eq!(selected.provider, provider);
            assert_eq!(selected.provider_model_id, provider_model);
            assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
            assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
            assert_eq!(selected.selection_source, source);
        }
    }

    #[test]
    fn active_glm_canary_switches_only_new_requests_to_same_model_alternate() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);

        let pinned_before_failure = router
            .select_active_completion_route(&proxy_router, &intent, None)
            .expect("initial Tinfoil route");
        assert_eq!(pinned_before_failure.provider, ProviderId::Tinfoil);

        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Tinfoil,
                GLM_5_2_MODEL_ID,
                GLM_5_2_MODEL_ID,
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        // The previously returned pin is immutable; only a fresh preparation
        // observes the newly opened circuit.
        assert_eq!(pinned_before_failure.provider, ProviderId::Tinfoil);
        assert_eq!(pinned_before_failure.provider_model_id, GLM_5_2_MODEL_ID);

        let selected_after_failure = router
            .select_active_completion_route(&proxy_router, &intent, None)
            .expect("Continuum GLM fallback");
        assert_eq!(selected_after_failure.provider, ProviderId::Continuum);
        assert_eq!(selected_after_failure.provider_model_id, "glm-5.2");
        assert_eq!(selected_after_failure.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(
            selected_after_failure.selection_source,
            RouteSelectionSource::Fallback
        );
    }

    #[test]
    fn active_glm_canary_bypasses_an_open_feature_flag_preference() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);

        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                GLM_5_2_MODEL_ID,
                "glm-5.2",
                503,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let selected = router
            .select_active_completion_route(
                &proxy_router,
                &intent,
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            )
            .expect("Tinfoil GLM fallback");
        assert_eq!(selected.provider, ProviderId::Tinfoil);
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.selection_source, RouteSelectionSource::Fallback);
    }

    #[test]
    fn active_glm_canary_returns_typed_capacity_when_every_configured_route_is_open() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);

        for terminal in [
            capacity_terminal(
                ProviderId::Tinfoil,
                GLM_5_2_MODEL_ID,
                GLM_5_2_MODEL_ID,
                429,
                Duration::from_secs(40),
            ),
            capacity_terminal(
                ProviderId::Continuum,
                GLM_5_2_MODEL_ID,
                "glm-5.2",
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
                assert_eq!(model, GLM_5_2_MODEL_ID);
                assert_eq!(retry_after, Duration::from_secs(30));
            }
            other => panic!("unexpected route error: {other:?}"),
        }
    }

    #[test]
    fn active_glm_canary_uses_route_failure_threshold_but_not_watch_state() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);
        let failure =
            || route_failure_terminal(ProviderId::Tinfoil, GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);

        for observed_failures in 1..=3 {
            router.observe_attempt_terminal(&failure(), ShadowObservationMode::Update);
            let selected = router
                .select_active_completion_route(&proxy_router, &intent, None)
                .expect("GLM route");
            let expected = if observed_failures < 3 {
                ProviderId::Tinfoil
            } else {
                ProviderId::Continuum
            };
            assert_eq!(selected.provider, expected, "failure {observed_failures}");
        }
    }

    #[test]
    fn continuum_account_429_blocks_explicit_kimi_and_auto_falls_back_to_k3() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                POWERFUL_MODEL_ID,
                "kimi-k2.6",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let glm = router
            .select_active_completion_route(
                &proxy_router,
                &intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID),
                Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            )
            .expect("Tinfoil GLM after Continuum account limit");
        assert_eq!(glm.provider, ProviderId::Tinfoil);

        let explicit_kimi_error = router
            .select_active_completion_route(
                &proxy_router,
                &intent(POWERFUL_MODEL_ID, POWERFUL_MODEL_ID),
                None,
            )
            .expect_err("explicit Kimi never changes public model");
        assert!(matches!(
            explicit_kimi_error,
            ProviderRoutingError::CapacityUnavailable { model, .. }
                if model == POWERFUL_MODEL_ID
        ));

        let auto_kimi = router
            .select_active_completion_route(
                &proxy_router,
                &intent(AUTO_POWERFUL_MODEL_ID, POWERFUL_MODEL_ID),
                None,
            )
            .expect("Auto Powerful may use the compatible K3 model");
        assert_eq!(auto_kimi.provider, ProviderId::Tinfoil);
        assert_eq!(auto_kimi.public_model_id, KIMI_K3_MODEL_ID);
        assert_eq!(auto_kimi.provider_model_id, KIMI_K3_MODEL_ID);
        assert_eq!(
            auto_kimi.model_selection_source,
            ModelSelectionSource::AutoFallback
        );
    }

    #[test]
    fn singleton_model_plans_circuit_break_without_crossing_public_identity() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Tinfoil,
                KIMI_K3_MODEL_ID,
                KIMI_K3_MODEL_ID,
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );

        let explicit_error = router
            .select_active_completion_route(
                &proxy_router,
                &intent(KIMI_K3_MODEL_ID, KIMI_K3_MODEL_ID),
                None,
            )
            .expect_err("explicit K3 circuit is open");
        assert!(matches!(
            explicit_error,
            ProviderRoutingError::CapacityUnavailable { model, .. }
                if model == KIMI_K3_MODEL_ID
        ));

        // A healthy compatible model cannot broaden an explicit request.
        let explicit_k2 = router
            .select_active_completion_route(
                &proxy_router,
                &intent(POWERFUL_MODEL_ID, POWERFUL_MODEL_ID),
                None,
            )
            .expect("explicit K2.6 remains K2.6");
        assert_eq!(explicit_k2.public_model_id, POWERFUL_MODEL_ID);
        assert_eq!(
            explicit_k2.model_selection_source,
            ModelSelectionSource::Explicit
        );

        // Auto Quick is also a singleton model plan in Stack 7.
        router.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Tinfoil,
                QUICK_MODEL_ID,
                QUICK_MODEL_ID,
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );
        let auto_quick_error = router
            .select_active_completion_route(
                &proxy_router,
                &intent(AUTO_QUICK_MODEL_ID, QUICK_MODEL_ID),
                None,
            )
            .expect_err("Auto Quick has no alternate public model");
        assert!(matches!(
            auto_quick_error,
            ProviderRoutingError::CapacityUnavailable { model, .. }
                if model == QUICK_MODEL_ID
        ));
    }

    #[test]
    fn auto_powerful_falls_back_in_both_flag_selected_directions() {
        let proxy_router = proxy_router_with_both_providers();

        let k3_preferred = ProviderRouter::default();
        k3_preferred.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Tinfoil,
                KIMI_K3_MODEL_ID,
                KIMI_K3_MODEL_ID,
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );
        let selected = k3_preferred
            .select_active_completion_route(
                &proxy_router,
                &intent(AUTO_POWERFUL_MODEL_ID, KIMI_K3_MODEL_ID),
                None,
            )
            .expect("flag-on Auto Powerful falls back to K2.6");
        assert_eq!(selected.public_model_id, POWERFUL_MODEL_ID);
        assert_eq!(selected.provider, ProviderId::Continuum);
        assert_eq!(selected.provider_model_id, "kimi-k2.6");
        assert_eq!(
            selected.selection_source,
            RouteSelectionSource::DefaultProvider
        );
        assert_eq!(
            selected.model_selection_source,
            ModelSelectionSource::AutoFallback
        );

        let k2_preferred = ProviderRouter::default();
        k2_preferred.observe_attempt_terminal(
            &capacity_terminal(
                ProviderId::Continuum,
                POWERFUL_MODEL_ID,
                "kimi-k2.6",
                429,
                Duration::from_secs(60),
            ),
            ShadowObservationMode::Update,
        );
        let selected = k2_preferred
            .select_active_completion_route(
                &proxy_router,
                &intent(AUTO_POWERFUL_MODEL_ID, POWERFUL_MODEL_ID),
                None,
            )
            .expect("flag-off Auto Powerful falls back to K3");
        assert_eq!(selected.public_model_id, KIMI_K3_MODEL_ID);
        assert_eq!(selected.provider, ProviderId::Tinfoil);
        assert_eq!(selected.provider_model_id, KIMI_K3_MODEL_ID);
        assert_eq!(selected.selection_source, RouteSelectionSource::StaticSplit);
        assert_eq!(
            selected.model_selection_source,
            ModelSelectionSource::AutoFallback
        );
    }

    #[test]
    fn auto_powerful_returns_earliest_capacity_when_both_kimi_candidates_are_open() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        for terminal in [
            capacity_terminal(
                ProviderId::Tinfoil,
                KIMI_K3_MODEL_ID,
                KIMI_K3_MODEL_ID,
                429,
                Duration::from_secs(50),
            ),
            capacity_terminal(
                ProviderId::Continuum,
                POWERFUL_MODEL_ID,
                "kimi-k2.6",
                429,
                Duration::from_secs(10),
            ),
        ] {
            router.observe_attempt_terminal(&terminal, ShadowObservationMode::Update);
        }

        let error = router
            .select_active_completion_route(
                &proxy_router,
                &intent(AUTO_POWERFUL_MODEL_ID, KIMI_K3_MODEL_ID),
                None,
            )
            .expect_err("both compatible Auto Powerful models are open");
        match error {
            ProviderRoutingError::CapacityUnavailable { model, retry_after } => {
                assert_eq!(model, KIMI_K3_MODEL_ID);
                assert_eq!(retry_after, Duration::from_secs(30));
            }
            other => panic!("unexpected route error: {other:?}"),
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
    fn test_glm_defaults_to_tinfoil_without_feature_flag_preference() {
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
    fn test_glm_routing_flag_key_does_not_apply_to_kimi_or_other_models() {
        let router = ProviderRouter::default();

        assert_eq!(
            router.continuum_flag_key_for_completion_model(GLM_5_2_MODEL_ID),
            Some(GLM_5_2_CONTINUUM_FLAG_KEY)
        );
        assert_eq!(
            router.continuum_flag_key_for_completion_model("kimi-k2-6"),
            None
        );
        assert_eq!(
            router.continuum_flag_key_for_completion_model("gpt-oss-120b"),
            None
        );
    }

    #[test]
    fn test_glm_true_feature_flag_selects_continuum_model_id() {
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

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.provider_model_id, "glm-5.2");
        assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, RouteSelectionSource::FeatureFlag);
    }

    #[test]
    fn test_glm_false_feature_flag_selects_tinfoil_model_id() {
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
    fn test_auto_powerful_uses_kimi_route_table() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(
                &proxy_router,
                uuid_for_bucket(70),
                crate::model_config::AUTO_POWERFUL_MODEL_ID,
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.public_model_id, "kimi-k2-6");
        assert_eq!(selected.provider_model_id, "kimi-k2.6");
        assert_eq!(selected.response_model_id, "kimi-k2-6");
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

        for model_id in ["kimi-k3", "deepseek-v4-flash"] {
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
    }
}
