use crate::model_config::{
    resolve_completion_model_id, resolve_public_model_id, GLM_5_2_MODEL_ID, GLM_5_3_MODEL_ID,
};
use crate::os_flags::GLM_5_3_TINFOIL_FLAG_KEY;
use crate::proxy_config::{canonicalize_tinfoil_model, ProxyConfig, ProxyRouter};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderName {
    Tinfoil,
    Continuum,
}

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

impl ProviderName {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Tinfoil => "tinfoil",
            Self::Continuum => "continuum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderSelectionSource {
    StaticSplit,
    FeatureFlag,
    DefaultProvider,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderPreference {
    provider: ProviderName,
    source: ProviderSelectionSource,
}

impl ProviderPreference {
    pub(crate) const fn feature_flag(provider: ProviderName) -> Self {
        Self {
            provider,
            source: ProviderSelectionSource::FeatureFlag,
        }
    }

    const fn default_provider(provider: ProviderName) -> Self {
        Self {
            provider,
            source: ProviderSelectionSource::DefaultProvider,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProviderConfig {
    provider: ProviderName,
    weight: u16,
    enabled: bool,
}

#[derive(Debug, Clone, Copy)]
struct ModelProviderRoute {
    provider: ProviderName,
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
    default_provider: Option<ProviderName>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderRoutingFlag {
    key: &'static str,
    enabled_provider: ProviderName,
    disabled_provider: ProviderName,
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
    pub(crate) proxy: ProxyConfig,
    pub(crate) public_model_id: String,
    pub(crate) provider_model_id: String,
    pub(crate) response_model_id: String,
    pub(crate) bucket: Option<u8>,
    pub(crate) selection_source: ProviderSelectionSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProviderRoutingError {
    UnsupportedModel(String),
    NoEligibleRoute(String),
}

#[derive(Debug)]
pub(crate) struct ProviderRouter {
    config: &'static ProviderRoutingConfig,
}

#[derive(Debug, Clone)]
struct EligibleRoute {
    provider: ProviderName,
    proxy: ProxyConfig,
    provider_model_id: &'static str,
    effective_weight: u32,
}

const PROVIDERS: &[ProviderConfig] = &[
    ProviderConfig {
        provider: ProviderName::Tinfoil,
        weight: 70,
        enabled: true,
    },
    ProviderConfig {
        provider: ProviderName::Continuum,
        weight: 30,
        enabled: true,
    },
];

const KIMI_K2_6_ROUTES: &[ModelProviderRoute] = &[ModelProviderRoute {
    provider: ProviderName::Continuum,
    provider_model_id: "kimi-k2.6",
    weight: 100,
    enabled: true,
    requires_explicit_preference: false,
}];

const GLM_5_2_ROUTES: &[ModelProviderRoute] = &[ModelProviderRoute {
    provider: ProviderName::Tinfoil,
    provider_model_id: GLM_5_2_MODEL_ID,
    weight: 100,
    enabled: true,
    requires_explicit_preference: false,
}];

const GLM_5_3_ROUTES: &[ModelProviderRoute] = &[
    ModelProviderRoute {
        provider: ProviderName::Continuum,
        provider_model_id: "glm-5.3",
        weight: 100,
        enabled: true,
        requires_explicit_preference: false,
    },
    ModelProviderRoute {
        // Staged for Tinfoil's in-progress launch. Keep this route out of
        // automatic fallback until the live catalog publishes the model and
        // billing rates have been added.
        provider: ProviderName::Tinfoil,
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
        default_provider: Some(ProviderName::Continuum),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_2_MODEL_ID,
        routes: GLM_5_2_ROUTES,
        provider_flag: None,
        default_provider: Some(ProviderName::Tinfoil),
    },
    ModelRoutingConfig {
        public_model_id: GLM_5_3_MODEL_ID,
        routes: GLM_5_3_ROUTES,
        provider_flag: Some(ProviderRoutingFlag {
            key: GLM_5_3_TINFOIL_FLAG_KEY,
            enabled_provider: ProviderName::Tinfoil,
            disabled_provider: ProviderName::Continuum,
        }),
        default_provider: Some(ProviderName::Continuum),
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
        }
    }
}

impl ProviderRouter {
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
    /// selected at the public inference entrypoint. Router v2 intentionally
    /// delegates to the legacy implementation in this foundation stack; later
    /// stacks may evolve only the v2 branch while the legacy path stays intact.
    pub(crate) fn select_completion_route_for_mode(
        &self,
        proxy_router: &ProxyRouter,
        account_uuid: Uuid,
        requested_model: &str,
        provider_preference: Option<ProviderPreference>,
        routing_mode: InferenceRoutingMode,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        match routing_mode {
            InferenceRoutingMode::Legacy => self.select_completion_route_with_preference(
                proxy_router,
                account_uuid,
                requested_model,
                provider_preference,
            ),
            InferenceRoutingMode::V2 => self.select_completion_route_v2(
                proxy_router,
                account_uuid,
                requested_model,
                provider_preference,
            ),
        }
    }

    fn select_completion_route_v2(
        &self,
        proxy_router: &ProxyRouter,
        account_uuid: Uuid,
        requested_model: &str,
        provider_preference: Option<ProviderPreference>,
    ) -> Result<SelectedProviderRoute, ProviderRoutingError> {
        self.select_completion_route_with_preference(
            proxy_router,
            account_uuid,
            requested_model,
            provider_preference,
        )
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
                    .is_none_or(|preference| preference.provider != route.provider)
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
                .find(|route| route.provider == preference.provider)
                .map(|route| (route, preference.source))
        });
        let default_preference_route = default_preference.and_then(|preference| {
            eligible_routes
                .iter()
                .find(|route| route.provider == preference.provider)
                .map(|route| {
                    let source = if provider_preference.is_some() {
                        ProviderSelectionSource::Fallback
                    } else {
                        preference.source
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
                    ProviderSelectionSource::Fallback
                } else {
                    ProviderSelectionSource::StaticSplit
                },
            )
        };

        Ok(SelectedProviderRoute {
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
        let provider_model_id = if proxy.provider_name == ProviderName::Tinfoil.as_str() {
            resolve_completion_model_id(requested_model)
                .ok_or_else(|| ProviderRoutingError::UnsupportedModel(requested_model.into()))?
                .to_string()
        } else {
            resolved_public_model_id
                .clone()
                .unwrap_or_else(|| requested_model.to_string())
        };

        let public_model_id = resolved_public_model_id.unwrap_or_else(|| provider_model_id.clone());

        let response_model_id = if proxy.provider_name == ProviderName::Tinfoil.as_str() {
            canonicalize_tinfoil_model(&provider_model_id)
        } else {
            public_model_id.clone()
        };

        Ok(SelectedProviderRoute {
            proxy,
            public_model_id,
            provider_model_id,
            response_model_id,
            bucket: None,
            selection_source: ProviderSelectionSource::StaticSplit,
        })
    }

    fn provider_config(&self, provider: ProviderName) -> Option<&ProviderConfig> {
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

fn proxy_for_provider(proxy_router: &ProxyRouter, provider: ProviderName) -> Option<ProxyConfig> {
    match provider {
        ProviderName::Tinfoil => Some(proxy_router.get_tinfoil_proxy()),
        ProviderName::Continuum => {
            let proxy = proxy_router.get_default_proxy();
            (proxy.provider_name == ProviderName::Continuum.as_str()).then_some(proxy)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{
        ModelAliasTargets, ModelPlan, PaidModelAliasOverrides, AUTO_POWERFUL_MODEL_ID,
        AUTO_QUICK_MODEL_ID, DEEPSEEK_V4_FLASH_MODEL_ID, GLM_5_2_MODEL_ID, GLM_5_3_FLASH_MODEL_ID,
        GLM_5_3_MODEL_ID, KIMI_K3_MODEL_ID, QUICK_MODEL_ID,
    };
    use crate::os_flags::PAID_POWERFUL_GLM_5_3_ALIAS_FLAG_KEY;
    use std::collections::HashMap;

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
    fn initial_router_v2_seam_is_route_identical_to_legacy() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();
        let cases = [
            (GLM_5_2_MODEL_ID, None),
            (
                GLM_5_2_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            ),
            (GLM_5_3_MODEL_ID, None),
            (
                GLM_5_3_MODEL_ID,
                Some(ProviderPreference::feature_flag(ProviderName::Tinfoil)),
            ),
            (GLM_5_3_FLASH_MODEL_ID, None),
            ("kimi-k2-6", None),
            ("gpt-oss-120b", None),
        ];

        for (model, preference) in cases {
            let legacy = router
                .select_completion_route_for_mode(
                    &proxy_router,
                    uuid_for_bucket(73),
                    model,
                    preference,
                    InferenceRoutingMode::Legacy,
                )
                .expect("legacy route");
            let v2 = router
                .select_completion_route_for_mode(
                    &proxy_router,
                    uuid_for_bucket(73),
                    model,
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
            expected_source: ProviderSelectionSource,
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
                expected_source: ProviderSelectionSource::StaticSplit,
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
                expected_source: ProviderSelectionSource::DefaultProvider,
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
                expected_source: ProviderSelectionSource::StaticSplit,
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
                expected_source: ProviderSelectionSource::DefaultProvider,
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
                expected_source: ProviderSelectionSource::StaticSplit,
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
                expected_source: ProviderSelectionSource::DefaultProvider,
            },
            Case {
                name: "explicit GLM 5.3 Tinfoil preference",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderName::Tinfoil)),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_3_MODEL_ID,
                expected_source: ProviderSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM 5.3 Continuum preference",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(
                    ProviderName::Continuum,
                )),
                continuum_available: true,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "continuum",
                expected_provider_model: "glm-5.3",
                expected_source: ProviderSelectionSource::FeatureFlag,
            },
            Case {
                name: "explicit GLM 5.3 Tinfoil preference without a Continuum proxy",
                selector: GLM_5_3_MODEL_ID,
                plan: ModelPlan::Paid,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderName::Tinfoil)),
                continuum_available: false,
                expected_access: true,
                expected_public_model: GLM_5_3_MODEL_ID,
                expected_provider: "tinfoil",
                expected_provider_model: GLM_5_3_MODEL_ID,
                expected_source: ProviderSelectionSource::FeatureFlag,
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
        }
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
                ProviderSelectionSource::DefaultProvider
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
        assert_eq!(flag.preference_for(true).provider, ProviderName::Tinfoil);
        assert_eq!(flag.preference_for(false).provider, ProviderName::Continuum);
        assert_eq!(
            flag.preference_for(true).source,
            ProviderSelectionSource::FeatureFlag
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
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.response_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, ProviderSelectionSource::Fallback);
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
                Some(ProviderPreference::feature_flag(ProviderName::Tinfoil)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );
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
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, ProviderSelectionSource::Fallback);
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
                ProviderSelectionSource::DefaultProvider
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
            ProviderSelectionSource::DefaultProvider
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
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );

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
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );
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
            ProviderSelectionSource::DefaultProvider
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
            ProviderSelectionSource::DefaultProvider
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
