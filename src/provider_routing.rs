use crate::inference::health::{
    ShadowHealthState, ShadowObservationMode, ShadowObservationReport, ShadowRouteSnapshot,
};
use crate::inference::{AttemptTerminal, InferenceIntent, RouteIdentity, RouteKey};
use crate::inference_planning::{
    plan_completion_route, ConfiguredProviders, ProviderPreference, RoutePlan, RoutePlanningError,
    RoutePlanningInput,
};
use crate::model_config::{resolve_completion_model_id, resolve_public_model_id, GLM_5_2_MODEL_ID};
use crate::os_flags::GLM_5_2_CONTINUUM_FLAG_KEY;
use crate::provider_registry::{
    ProviderId, ProviderRegistry, RouteSelectionSource, PROVIDER_REGISTRY,
};
use crate::proxy_config::{canonicalize_tinfoil_model, ProxyConfig, ProxyRouter};
use uuid::Uuid;

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
}

#[derive(Debug, Clone, Copy)]
struct ModelRoutingConfig {
    public_model_id: &'static str,
    routes: &'static [ModelProviderRoute],
    continuum_flag_key: Option<&'static str>,
    default_provider: Option<ProviderId>,
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
}];

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

const MODEL_ROUTES: &[ModelRoutingConfig] = &[
    ModelRoutingConfig {
        public_model_id: "kimi-k2-6",
        routes: KIMI_K2_6_ROUTES,
        continuum_flag_key: None,
        default_provider: Some(ProviderId::Continuum),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_2_MODEL_ID,
        routes: GLM_5_2_ROUTES,
        continuum_flag_key: Some(GLM_5_2_CONTINUUM_FLAG_KEY),
        default_provider: Some(ProviderId::Tinfoil),
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

    pub(crate) fn continuum_flag_key_for_completion_model(
        &self,
        requested_model: &str,
    ) -> Option<&'static str> {
        let public_model_id = resolve_public_model_id(requested_model)?;
        self.model_config(public_model_id)?.continuum_flag_key
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
    fn hypothetical_open_capacity_never_changes_active_or_shadow_route_selection() {
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
