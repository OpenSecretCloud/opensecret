//! Deterministic, credential-free completion route planning.
//!
//! The plan describes route identity and ordered candidates without naming or
//! invoking an executable endpoint. The same-public-model planner remains the
//! deterministic inner decision; Stack 7 composes it with the narrow ordered
//! Auto Powerful K3/K2.6 model policy after one health snapshot.

use crate::inference::{InferenceIntent, ModelSelectionMode, RouteIdentity};
use crate::model_config::{
    models_are_auto_substitution_compatible, KIMI_K3_MODEL_ID, POWERFUL_MODEL_ID,
};
use crate::provider_registry::{
    ProviderId, ProviderRegistry, RouteSelectionSource, SHADOW_ROUTING_POLICY_VERSION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderPreference {
    provider: ProviderId,
    source: RouteSelectionSource,
}

impl ProviderPreference {
    pub(crate) const fn feature_flag(provider: ProviderId) -> Self {
        Self {
            provider,
            source: RouteSelectionSource::FeatureFlag,
        }
    }

    #[cfg(test)]
    pub(crate) const fn default_provider(provider: ProviderId) -> Self {
        Self {
            provider,
            source: RouteSelectionSource::DefaultProvider,
        }
    }

    pub(crate) const fn provider(self) -> ProviderId {
        self.provider
    }

    pub(crate) const fn source(self) -> RouteSelectionSource {
        self.source
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ConfiguredProviders {
    tinfoil: bool,
    continuum: bool,
}

impl ConfiguredProviders {
    pub(crate) const fn none() -> Self {
        Self {
            tinfoil: false,
            continuum: false,
        }
    }

    #[cfg(test)]
    pub(crate) const fn all() -> Self {
        Self {
            tinfoil: true,
            continuum: true,
        }
    }

    pub(crate) fn with_provider(mut self, provider: ProviderId) -> Self {
        match provider {
            ProviderId::Tinfoil => self.tinfoil = true,
            ProviderId::Continuum => self.continuum = true,
        }
        self
    }

    pub(crate) const fn contains(self, provider: ProviderId) -> bool {
        match provider {
            ProviderId::Tinfoil => self.tinfoil,
            ProviderId::Continuum => self.continuum,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RoutePlanningInput<'a> {
    pub(crate) intent: &'a InferenceIntent,
    pub(crate) configured_providers: ConfiguredProviders,
    pub(crate) provider_preference: Option<ProviderPreference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteCandidate {
    pub(crate) provider: ProviderId,
    pub(crate) public_model_id: String,
    pub(crate) provider_model_id: String,
    pub(crate) response_model_id: String,
    pub(crate) effective_weight: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanDecision {
    FixedRoute,
    StaticBucket,
    DefaultProvider,
    FeatureFlagPreference,
    PreferredProviderUnavailable,
    DefaultProviderUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateScope {
    SamePublicModelOnly,
}

pub(crate) const AUTO_MODEL_ROUTING_POLICY_VERSION: &str = "routing-v2-auto-model-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelCandidateScope {
    PreferredPublicModelOnly,
    CompatibleAutoModels,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelCandidatePlan {
    pub(crate) public_model_ids: Vec<String>,
    pub(crate) candidate_scope: ModelCandidateScope,
    pub(crate) policy_version: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutePlan {
    pub(crate) selected: RouteIdentity,
    pub(crate) eligible_routes: Vec<RouteCandidate>,
    pub(crate) decision: PlanDecision,
    pub(crate) candidate_scope: CandidateScope,
    pub(crate) policy_version: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RoutePlanningError {
    UnsupportedModel(String),
    NoEligibleRoute(String),
}

#[derive(Debug, Clone)]
struct EligibleRoute {
    candidate: RouteCandidate,
}

/// Plans the ordered public-model candidates for one logical completion.
///
/// Only the original Auto selector can authorize a different public model.
/// Stack 7 intentionally contains one compatible pair: paid Auto Powerful may
/// move between Kimi K3 and Kimi K2.6. Auto Quick and every explicit request
/// remain single-model plans.
pub(crate) fn plan_completion_model_candidates(intent: &InferenceIntent) -> ModelCandidatePlan {
    let mut public_model_ids = vec![intent.public_model_id.clone()];

    if intent.selection_mode == ModelSelectionMode::AutoPowerful {
        let alternate = match intent.public_model_id.as_str() {
            KIMI_K3_MODEL_ID => Some(POWERFUL_MODEL_ID),
            POWERFUL_MODEL_ID => Some(KIMI_K3_MODEL_ID),
            _ => None,
        };
        if let Some(alternate) = alternate {
            if intent.model_plan.allows_model(alternate)
                && models_are_auto_substitution_compatible(&intent.public_model_id, alternate)
            {
                public_model_ids.push(alternate.to_string());
            }
        }
    }

    ModelCandidatePlan {
        candidate_scope: if public_model_ids.len() > 1 {
            ModelCandidateScope::CompatibleAutoModels
        } else {
            ModelCandidateScope::PreferredPublicModelOnly
        },
        public_model_ids,
        policy_version: AUTO_MODEL_ROUTING_POLICY_VERSION,
    }
}

pub(crate) fn plan_completion_route(
    registry: &ProviderRegistry,
    input: RoutePlanningInput<'_>,
) -> Result<RoutePlan, RoutePlanningError> {
    let public_model_id = input.intent.public_model_id.as_str();
    let model = registry
        .completion_model(public_model_id)
        .ok_or_else(|| RoutePlanningError::UnsupportedModel(public_model_id.to_string()))?;

    let eligible_routes = model
        .routes
        .iter()
        .filter_map(|route| {
            if !route.enabled || route.weight == 0 {
                return None;
            }
            let provider = registry.provider(route.provider)?;
            if !provider.enabled
                || provider.weight == 0
                || !input.configured_providers.contains(route.provider)
            {
                return None;
            }

            Some(EligibleRoute {
                candidate: RouteCandidate {
                    provider: route.provider,
                    public_model_id: model.public_model_id.to_string(),
                    provider_model_id: route.provider_model_id.to_string(),
                    response_model_id: route.response_model_id.to_string(),
                    effective_weight: u32::from(provider.weight) * u32::from(route.weight),
                },
            })
        })
        .collect::<Vec<_>>();

    if eligible_routes.is_empty() {
        return Err(RoutePlanningError::NoEligibleRoute(
            model.public_model_id.to_string(),
        ));
    }

    if let Some(preference) = input.provider_preference {
        if let Some(route) = eligible_routes
            .iter()
            .find(|route| route.candidate.provider == preference.provider())
        {
            return Ok(build_plan(
                route,
                &eligible_routes,
                preference.source(),
                None,
                PlanDecision::FeatureFlagPreference,
            ));
        }
    }

    if let Some(default_provider) = model.default_provider {
        if let Some(route) = eligible_routes
            .iter()
            .find(|route| route.candidate.provider == default_provider)
        {
            let (source, decision) = if input.provider_preference.is_some() {
                (
                    RouteSelectionSource::Fallback,
                    PlanDecision::PreferredProviderUnavailable,
                )
            } else {
                (
                    RouteSelectionSource::DefaultProvider,
                    PlanDecision::DefaultProvider,
                )
            };
            return Ok(build_plan(route, &eligible_routes, source, None, decision));
        }
    }

    let fallback_source = if input.provider_preference.is_some() || model.default_provider.is_some()
    {
        RouteSelectionSource::Fallback
    } else {
        RouteSelectionSource::StaticSplit
    };
    let missing_preference_decision = if input.provider_preference.is_some() {
        Some(PlanDecision::PreferredProviderUnavailable)
    } else if model.default_provider.is_some() {
        Some(PlanDecision::DefaultProviderUnavailable)
    } else {
        None
    };

    if eligible_routes.len() == 1 {
        return Ok(build_plan(
            &eligible_routes[0],
            &eligible_routes,
            fallback_source,
            None,
            missing_preference_decision.unwrap_or(PlanDecision::FixedRoute),
        ));
    }

    let bucket = stable_account_bucket(input.intent.account_uuid);
    let selected = select_weighted_route(bucket, &eligible_routes)
        .ok_or_else(|| RoutePlanningError::NoEligibleRoute(model.public_model_id.to_string()))?;
    Ok(build_plan(
        selected,
        &eligible_routes,
        fallback_source,
        Some(bucket),
        missing_preference_decision.unwrap_or(PlanDecision::StaticBucket),
    ))
}

fn build_plan(
    selected: &EligibleRoute,
    eligible_routes: &[EligibleRoute],
    selection_source: RouteSelectionSource,
    bucket: Option<u8>,
    decision: PlanDecision,
) -> RoutePlan {
    let selected = &selected.candidate;
    RoutePlan {
        selected: RouteIdentity::new(
            selected.provider,
            selected.public_model_id.clone(),
            selected.provider_model_id.clone(),
            selected.response_model_id.clone(),
            selection_source,
            bucket,
        ),
        eligible_routes: eligible_routes
            .iter()
            .map(|route| route.candidate.clone())
            .collect(),
        decision,
        candidate_scope: CandidateScope::SamePublicModelOnly,
        policy_version: SHADOW_ROUTING_POLICY_VERSION,
    }
}

fn select_weighted_route(bucket: u8, routes: &[EligibleRoute]) -> Option<&EligibleRoute> {
    let total_weight = routes
        .iter()
        .map(|route| route.candidate.effective_weight)
        .sum::<u32>();
    if total_weight == 0 {
        return None;
    }

    let mut cumulative = 0u32;
    for (index, route) in routes.iter().enumerate() {
        let bucket_span = if index == routes.len() - 1 {
            100u32.saturating_sub(cumulative)
        } else {
            (route.candidate.effective_weight * 100) / total_weight
        };
        cumulative = cumulative.saturating_add(bucket_span);

        if u32::from(bucket) < cumulative || index == routes.len() - 1 {
            return Some(route);
        }
    }
    None
}

fn stable_account_bucket(account_uuid: uuid::Uuid) -> u8 {
    (u128::from_be_bytes(*account_uuid.as_bytes()) % 100) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{
        ModelPlan, AUTO_POWERFUL_MODEL_ID, AUTO_QUICK_MODEL_ID, DEEPSEEK_V4_FLASH_MODEL_ID,
        GLM_5_2_MODEL_ID, KIMI_K3_MODEL_ID, POWERFUL_MODEL_ID, QUICK_MODEL_ID,
    };
    use crate::provider_registry::{ProviderId, PROVIDER_REGISTRY};
    use uuid::Uuid;

    fn intent(requested_model: &str, public_model: &str) -> InferenceIntent {
        InferenceIntent::new(
            Uuid::from_u128(73),
            requested_model,
            public_model,
            ModelPlan::Paid,
            crate::inference::InferenceSurface::Responses,
            crate::inference::WorkloadClass::Interactive,
        )
    }

    fn intent_for_plan(
        requested_model: &str,
        public_model: &str,
        model_plan: ModelPlan,
    ) -> InferenceIntent {
        InferenceIntent::new(
            Uuid::from_u128(73),
            requested_model,
            public_model,
            model_plan,
            crate::inference::InferenceSurface::Responses,
            crate::inference::WorkloadClass::Interactive,
        )
    }

    #[test]
    fn auto_model_policy_changes_only_paid_powerful_between_the_agreed_kimi_pair() {
        let cases = [
            (
                intent(AUTO_POWERFUL_MODEL_ID, KIMI_K3_MODEL_ID),
                vec![KIMI_K3_MODEL_ID, POWERFUL_MODEL_ID],
                ModelCandidateScope::CompatibleAutoModels,
            ),
            (
                intent(AUTO_POWERFUL_MODEL_ID, POWERFUL_MODEL_ID),
                vec![POWERFUL_MODEL_ID, KIMI_K3_MODEL_ID],
                ModelCandidateScope::CompatibleAutoModels,
            ),
            (
                intent(AUTO_QUICK_MODEL_ID, DEEPSEEK_V4_FLASH_MODEL_ID),
                vec![DEEPSEEK_V4_FLASH_MODEL_ID],
                ModelCandidateScope::PreferredPublicModelOnly,
            ),
            (
                intent_for_plan(AUTO_QUICK_MODEL_ID, QUICK_MODEL_ID, ModelPlan::Free),
                vec![QUICK_MODEL_ID],
                ModelCandidateScope::PreferredPublicModelOnly,
            ),
            (
                intent(KIMI_K3_MODEL_ID, KIMI_K3_MODEL_ID),
                vec![KIMI_K3_MODEL_ID],
                ModelCandidateScope::PreferredPublicModelOnly,
            ),
        ];

        for (intent, expected, scope) in cases {
            let plan = plan_completion_model_candidates(&intent);
            assert_eq!(plan.public_model_ids, expected);
            assert_eq!(plan.candidate_scope, scope);
            assert_eq!(plan.policy_version, AUTO_MODEL_ROUTING_POLICY_VERSION);
        }
    }

    #[test]
    fn free_powerful_policy_never_adds_a_paid_alternate() {
        let intent = intent_for_plan(AUTO_POWERFUL_MODEL_ID, POWERFUL_MODEL_ID, ModelPlan::Free);
        let plan = plan_completion_model_candidates(&intent);

        assert_eq!(plan.public_model_ids, vec![POWERFUL_MODEL_ID]);
        assert_eq!(
            plan.candidate_scope,
            ModelCandidateScope::PreferredPublicModelOnly
        );
    }

    #[test]
    fn planner_uses_resolved_public_model_without_resolving_auto_again() {
        let intent = intent(AUTO_POWERFUL_MODEL_ID, "kimi-k3");
        let plan = plan_completion_route(
            &PROVIDER_REGISTRY,
            RoutePlanningInput {
                intent: &intent,
                configured_providers: ConfiguredProviders::all(),
                provider_preference: None,
            },
        )
        .expect("shadow route");

        assert_eq!(intent.requested_model_id, AUTO_POWERFUL_MODEL_ID);
        assert_eq!(plan.selected.public_model_id, "kimi-k3");
        assert_eq!(plan.selected.provider_model_id, "kimi-k3");
        assert_eq!(plan.selected.provider, ProviderId::Tinfoil);
        assert_eq!(plan.selected.bucket, None);
        assert_eq!(plan.decision, PlanDecision::FixedRoute);
    }

    #[test]
    fn planner_is_deterministic_and_candidates_never_cross_public_models() {
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);
        let input = RoutePlanningInput {
            intent: &intent,
            configured_providers: ConfiguredProviders::all(),
            provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
        };

        let first = plan_completion_route(&PROVIDER_REGISTRY, input).expect("first plan");
        let second = plan_completion_route(&PROVIDER_REGISTRY, input).expect("second plan");
        assert_eq!(first, second);
        assert_eq!(first.candidate_scope, CandidateScope::SamePublicModelOnly);
        assert!(first
            .eligible_routes
            .iter()
            .all(|route| route.public_model_id == intent.public_model_id));
        assert_eq!(
            first
                .eligible_routes
                .iter()
                .map(|route| (route.provider, route.provider_model_id.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (ProviderId::Tinfoil, GLM_5_2_MODEL_ID),
                (ProviderId::Continuum, "glm-5.2"),
            ]
        );
    }

    #[test]
    fn unavailable_preference_falls_back_without_authorizing_another_model() {
        let intent = intent(GLM_5_2_MODEL_ID, GLM_5_2_MODEL_ID);
        let configured = ConfiguredProviders::none().with_provider(ProviderId::Tinfoil);
        let plan = plan_completion_route(
            &PROVIDER_REGISTRY,
            RoutePlanningInput {
                intent: &intent,
                configured_providers: configured,
                provider_preference: Some(ProviderPreference::feature_flag(ProviderId::Continuum)),
            },
        )
        .expect("Tinfoil fallback plan");

        assert_eq!(plan.selected.provider, ProviderId::Tinfoil);
        assert_eq!(plan.selected.public_model_id, GLM_5_2_MODEL_ID);
        assert_eq!(
            plan.selected.selection_source,
            RouteSelectionSource::Fallback
        );
        assert_eq!(plan.decision, PlanDecision::PreferredProviderUnavailable);
        assert_eq!(plan.eligible_routes.len(), 1);
    }

    #[test]
    fn planner_fails_closed_for_unknown_and_unconfigured_models() {
        let unknown = intent("unknown-model", "unknown-model");
        assert_eq!(
            plan_completion_route(
                &PROVIDER_REGISTRY,
                RoutePlanningInput {
                    intent: &unknown,
                    configured_providers: ConfiguredProviders::all(),
                    provider_preference: None,
                },
            ),
            Err(RoutePlanningError::UnsupportedModel(
                "unknown-model".to_string()
            ))
        );

        let kimi = intent("kimi-k2-6", "kimi-k2-6");
        let tinfoil_only = ConfiguredProviders::none().with_provider(ProviderId::Tinfoil);
        assert_eq!(
            plan_completion_route(
                &PROVIDER_REGISTRY,
                RoutePlanningInput {
                    intent: &kimi,
                    configured_providers: tinfoil_only,
                    provider_preference: None,
                },
            ),
            Err(RoutePlanningError::NoEligibleRoute("kimi-k2-6".to_string()))
        );
    }
}
