use crate::model_config::{resolve_completion_model_id, resolve_public_model_id};
use crate::os_flags::KIMI_K2_6_CONTINUUM_FLAG_KEY;
use crate::proxy_config::{canonicalize_tinfoil_model, ProxyConfig, ProxyRouter};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderName {
    Tinfoil,
    Continuum,
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
}

#[derive(Debug, Clone, Copy)]
struct ModelRoutingConfig {
    public_model_id: &'static str,
    routes: &'static [ModelProviderRoute],
    continuum_flag_key: Option<&'static str>,
    default_provider: Option<ProviderName>,
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

const KIMI_K2_6_ROUTES: &[ModelProviderRoute] = &[
    ModelProviderRoute {
        provider: ProviderName::Tinfoil,
        provider_model_id: "kimi-k2-6",
        weight: 100,
        enabled: true,
    },
    ModelProviderRoute {
        provider: ProviderName::Continuum,
        provider_model_id: "kimi-k2.6",
        weight: 100,
        enabled: true,
    },
];

const MODEL_ROUTES: &[ModelRoutingConfig] = &[ModelRoutingConfig {
    public_model_id: "kimi-k2-6",
    routes: KIMI_K2_6_ROUTES,
    continuum_flag_key: Some(KIMI_K2_6_CONTINUUM_FLAG_KEY),
    default_provider: Some(ProviderName::Tinfoil),
}];

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
    fn test_kimi_defaults_to_tinfoil_without_feature_flag_preference() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(99), "kimi-k2-6")
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, "kimi-k2-6");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(selected.response_model_id, "kimi-k2-6");
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::DefaultProvider
        );
    }

    #[test]
    fn test_kimi_does_not_use_static_30_percent_rollout_without_feature_flag() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(70), "kimi-k2-6")
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, "kimi-k2-6");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(selected.response_model_id, "kimi-k2-6");
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::DefaultProvider
        );
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

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.public_model_id, "kimi-k2-6");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(selected.response_model_id, "kimi-k2-6");
    }

    #[test]
    fn test_kimi_routing_flag_key_resolves_aliases() {
        let router = ProviderRouter::default();

        assert_eq!(
            router.continuum_flag_key_for_completion_model("kimi-k2-6"),
            Some(KIMI_K2_6_CONTINUUM_FLAG_KEY)
        );
        assert_eq!(
            router.continuum_flag_key_for_completion_model(
                crate::model_config::AUTO_POWERFUL_MODEL_ID
            ),
            Some(KIMI_K2_6_CONTINUUM_FLAG_KEY)
        );
        assert_eq!(
            router
                .continuum_flag_key_for_completion_model(crate::model_config::AUTO_QUICK_MODEL_ID),
            None
        );
    }

    #[test]
    fn test_feature_flag_preference_can_select_continuum_outside_static_bucket() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(1),
                "kimi-k2-6",
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(selected.provider_model_id, "kimi-k2.6");
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );
    }

    #[test]
    fn test_feature_flag_preference_can_select_tinfoil_inside_static_continuum_bucket() {
        let router = ProviderRouter::default();
        let proxy_router = proxy_router_with_both_providers();

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(99),
                "kimi-k2-6",
                Some(ProviderPreference::feature_flag(ProviderName::Tinfoil)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );
    }

    #[test]
    fn test_preferred_provider_falls_back_to_default_provider_when_unavailable() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let selected = router
            .select_completion_route_with_preference(
                &proxy_router,
                uuid_for_bucket(70),
                "kimi-k2-6",
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "continuum");
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::FeatureFlag
        );

        let tinfoil_only = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );
        let selected = router
            .select_completion_route_with_preference(
                &tinfoil_only,
                uuid_for_bucket(70),
                "kimi-k2-6",
                Some(ProviderPreference::feature_flag(ProviderName::Continuum)),
            )
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.bucket, None);
        assert_eq!(selected.selection_source, ProviderSelectionSource::Fallback);
    }

    #[test]
    fn test_kimi_uses_tinfoil_when_continuum_proxy_is_missing() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(1), "kimi-k2-6")
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(selected.bucket, None);
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::DefaultProvider
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
    fn test_configured_model_uses_tinfoil_when_continuum_is_missing() {
        let router = ProviderRouter::default();
        let proxy_router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            None,
            "http://tinfoil.example.com".to_string(),
        );

        let selected = router
            .select_completion_route(&proxy_router, uuid_for_bucket(50), "kimi-k2-6")
            .expect("route");

        assert_eq!(selected.proxy.provider_name, "tinfoil");
        assert_eq!(selected.provider_model_id, "kimi-k2-6");
        assert_eq!(
            selected.selection_source,
            ProviderSelectionSource::DefaultProvider
        );
    }
}
