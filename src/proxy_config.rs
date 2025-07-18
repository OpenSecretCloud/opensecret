use chrono::{DateTime, Utc};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use lazy_static::lazy_static;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Model name constants
const CONTINUUM_LLAMA_33_70B: &str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4";
const TINFOIL_LLAMA_33_70B: &str = "llama3-3-70b";
const CANONICAL_LLAMA_33_70B: &str = "llama-3.3-70b";

lazy_static! {
    /// Known model equivalencies across providers
    /// This maps a canonical model identifier to provider-specific names
    static ref MODEL_EQUIVALENCIES: HashMap<&'static str, HashMap<&'static str, &'static str>> = {
        let mut equivalencies = HashMap::new();

        // Llama 3.3 70B
        let mut llama_33_70b = HashMap::new();
        llama_33_70b.insert("continuum", CONTINUUM_LLAMA_33_70B);
        llama_33_70b.insert("tinfoil", TINFOIL_LLAMA_33_70B);
        equivalencies.insert(CANONICAL_LLAMA_33_70B, llama_33_70b);

        equivalencies
    };
}

/// Model routing configuration
#[derive(Debug, Clone)]
pub struct ModelRoute {
    /// Primary provider configuration
    pub primary: ProxyConfig,
    /// Optional fallback providers in order of preference
    pub fallbacks: Vec<ProxyConfig>,
}

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    /// Provider name for logging
    pub provider_name: String,
}

#[derive(Debug)]
struct ModelsCache {
    // Cached models response for user-facing API
    models_response: Option<Value>,
    // Map from model name to proxy configuration for internal routing
    model_to_proxy: HashMap<String, ProxyConfig>,
    // Model routing configurations with fallback support
    model_routes: HashMap<String, ModelRoute>,
    // When the cache expires
    expires_at: DateTime<Utc>,
}

impl ModelsCache {
    fn new_with_default() -> Self {
        // Start with empty cache
        let models_response = serde_json::json!({
            "object": "list",
            "data": []
        });

        Self {
            models_response: Some(models_response),
            model_to_proxy: HashMap::new(),
            model_routes: HashMap::new(),
            expires_at: Utc::now(),
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    fn update(
        &mut self,
        models_response: Value,
        model_to_proxy: HashMap<String, ProxyConfig>,
        model_routes: HashMap<String, ModelRoute>,
    ) {
        self.models_response = Some(models_response);
        self.model_to_proxy = model_to_proxy;
        self.model_routes = model_routes;
        self.expires_at = Utc::now() + chrono::Duration::minutes(5);
    }
}

#[derive(Debug, Clone)]
pub struct ProxyRouter {
    // Unified cache for both models response and routing map
    cache: Arc<RwLock<ModelsCache>>,
    // Default proxy if model not found
    default_proxy: ProxyConfig,
    // Tinfoil proxy configuration if configured
    tinfoil_proxy: Option<ProxyConfig>,
}

impl ProxyRouter {
    /// Get the provider-specific model name for a given canonical model
    pub fn get_model_name_for_provider(
        &self,
        canonical_model: &str,
        provider_name: &str,
    ) -> String {
        // Direct lookup: check if input is a canonical name with a mapping
        if let Some(provider_map) = MODEL_EQUIVALENCIES.get(canonical_model) {
            if let Some(provider_specific_name) = provider_map.get(provider_name) {
                return provider_specific_name.to_string();
            }
        }

        // Reverse lookup: check if input is already a provider-specific name
        for (canonical_name, provider_map) in &*MODEL_EQUIVALENCIES {
            // If this model name exists in any provider's mapping
            if provider_map.values().any(|name| *name == canonical_model) {
                // Return the name for the requested provider
                if let Some(provider_specific_name) = provider_map.get(provider_name) {
                    return provider_specific_name.to_string();
                }
                // If the requested provider doesn't have this model, return canonical
                return canonical_name.to_string();
            }
        }

        // No translation needed - return as-is
        canonical_model.to_string()
    }

    pub fn new(
        openai_base: String,
        openai_key: Option<String>,
        tinfoil_base: Option<String>,
    ) -> Self {
        let cache = Arc::new(RwLock::new(ModelsCache::new_with_default()));

        // Default OpenAI/Continuum proxy config
        let default_proxy = ProxyConfig {
            base_url: openai_base.clone(),
            api_key: if openai_base.contains("api.openai.com") {
                openai_key.clone()
            } else {
                None // Continuum proxy doesn't need API key
            },
            provider_name: if openai_base.contains("api.openai.com") {
                "openai".to_string()
            } else {
                "continuum".to_string()
            },
        };

        // Tinfoil proxy configuration
        let tinfoil_proxy = tinfoil_base.map(|base| ProxyConfig {
            base_url: base.clone(),
            api_key: None, // Tinfoil proxy doesn't need API key
            provider_name: "tinfoil".to_string(),
        });

        ProxyRouter {
            cache,
            default_proxy,
            tinfoil_proxy,
        }
    }

    /// Get the model route configuration for a given model
    /// Always returns a route - all models have routes after initialization
    pub async fn get_model_route(&self, model_name: &str) -> ModelRoute {
        // Ensure cache is fresh
        self.refresh_cache_if_needed().await;

        let cache = self.cache.read().await;

        // All models should have routes now
        cache
            .model_routes
            .get(model_name)
            .cloned()
            .unwrap_or_else(|| {
                // Fallback for unknown models - use default proxy with no fallbacks
                warn!(
                    "Unknown model '{}' requested, using default proxy",
                    model_name
                );
                ModelRoute {
                    primary: self.default_proxy.clone(),
                    fallbacks: vec![],
                }
            })
    }

    /// Refresh the cache if it's expired
    async fn refresh_cache_if_needed(&self) {
        let should_refresh = {
            let cache = self.cache.read().await;
            cache.is_expired()
        };

        if should_refresh {
            if let Err(e) = self.refresh_cache().await {
                error!("Failed to refresh models cache: {:?}", e);
            }
        }
    }

    /// Force refresh the cache with latest data from all proxies
    async fn refresh_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Refreshing models cache from all configured proxies");

        // Create HTTP client
        let https = HttpsConnector::new();
        let client = Client::builder()
            .pool_idle_timeout(Duration::from_secs(15))
            .build::<_, Body>(https);

        let mut all_models = Vec::new();
        let mut model_to_proxy = HashMap::new();
        let mut available_models_by_provider: HashMap<String, Vec<String>> = HashMap::new();
        let mut tinfoil_models = HashMap::new();

        // First, fetch from Tinfoil if configured - these will be primary
        if let Some(ref tinfoil_proxy) = self.tinfoil_proxy {
            match self.fetch_models_from_proxy(&client, tinfoil_proxy).await {
                Ok(models) => {
                    let mut provider_models = Vec::new();

                    for model in &models {
                        if let Some(model_id) = model.get("id").and_then(|v| v.as_str()) {
                            provider_models.push(model_id.to_string());
                            tinfoil_models.insert(model_id.to_string(), model.clone());

                            debug!("Tinfoil model '{}' will be primary", model_id);
                            model_to_proxy.insert(model_id.to_string(), tinfoil_proxy.clone());
                        }
                    }

                    all_models.extend(models);
                    available_models_by_provider.insert("tinfoil".to_string(), provider_models);
                    info!("Fetched {} models from Tinfoil", tinfoil_models.len());
                }
                Err(e) => {
                    warn!("Failed to fetch models from Tinfoil: {:?}", e);
                    available_models_by_provider.insert("tinfoil".to_string(), Vec::new());
                }
            }
        }

        // Then fetch from Continuum - only primary if not available from Tinfoil
        match self
            .fetch_models_from_proxy(&client, &self.default_proxy)
            .await
        {
            Ok(models) => {
                let mut provider_models = Vec::new();

                for model in &models {
                    if let Some(model_id) = model.get("id").and_then(|v| v.as_str()) {
                        provider_models.push(model_id.to_string());

                        // Check if this is equivalent to any Tinfoil model
                        let mut is_equivalent_to_tinfoil = false;

                        for provider_names in MODEL_EQUIVALENCIES.values() {
                            if let (Some(continuum_name), Some(tinfoil_name)) = (
                                provider_names.get("continuum"),
                                provider_names.get("tinfoil"),
                            ) {
                                if *continuum_name == model_id
                                    && tinfoil_models.contains_key(*tinfoil_name)
                                {
                                    is_equivalent_to_tinfoil = true;
                                    debug!("Continuum model '{}' is equivalent to Tinfoil model '{}' - will be fallback", 
                                           model_id, tinfoil_name);
                                    break;
                                }
                            }
                        }

                        // Only add to all_models if not equivalent to a Tinfoil model
                        if !is_equivalent_to_tinfoil {
                            all_models.push(model.clone());
                            debug!(
                                "Continuum model '{}' will be primary (no Tinfoil equivalent)",
                                model_id
                            );
                        }
                    }
                }

                let model_count = provider_models.len();
                available_models_by_provider.insert("continuum".to_string(), provider_models);
                info!("Fetched {} models from Continuum", model_count);
            }
            Err(e) => {
                warn!("Failed to fetch models from Continuum: {:?}", e);
                available_models_by_provider.insert("continuum".to_string(), Vec::new());
            }
        }

        // Note: If adding a third provider in the future, consider deduplicating models
        // by ID to prevent duplicates in the all_models list. Currently not needed as:
        // 1. We control Tinfoil proxy and ensure no duplicates
        // 2. Continuum models are filtered for known equivalencies
        // 3. Only two providers exist currently
        let models_response = serde_json::json!({
            "object": "list",
            "data": all_models
        });

        // Build model routes - create a route for EVERY model
        let mut model_routes = HashMap::new();

        // First, create routes for all Tinfoil models
        if let Some(ref tinfoil_proxy) = self.tinfoil_proxy {
            if let Some(tinfoil_model_list) = available_models_by_provider.get("tinfoil") {
                for model_name in tinfoil_model_list {
                    // Check if this model has an equivalent in Continuum
                    let mut has_continuum_fallback = false;

                    for provider_names in MODEL_EQUIVALENCIES.values() {
                        if let (Some(tinfoil_equiv), Some(continuum_equiv)) = (
                            provider_names.get("tinfoil"),
                            provider_names.get("continuum"),
                        ) {
                            if *tinfoil_equiv == model_name {
                                // Check if Continuum has the equivalent model
                                if let Some(continuum_models) =
                                    available_models_by_provider.get("continuum")
                                {
                                    if continuum_models.contains(&continuum_equiv.to_string()) {
                                        // This model has a Continuum fallback
                                        let route = ModelRoute {
                                            primary: tinfoil_proxy.clone(),
                                            fallbacks: vec![self.default_proxy.clone()],
                                        };
                                        model_routes.insert(model_name.clone(), route.clone());
                                        // Also map the Continuum name to the same route
                                        model_routes.insert(continuum_equiv.to_string(), route);
                                        has_continuum_fallback = true;
                                        info!("Model available from both providers - Tinfoil ({}) primary, Continuum ({}) fallback", 
                                              model_name, continuum_equiv);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // If no fallback found, create route with just Tinfoil
                    if !has_continuum_fallback {
                        let route = ModelRoute {
                            primary: tinfoil_proxy.clone(),
                            fallbacks: vec![],
                        };
                        model_routes.insert(model_name.clone(), route);
                        debug!("Tinfoil-only model '{}' has no fallbacks", model_name);
                    }
                }
            }
        }

        // Then, create routes for Continuum models that aren't already mapped
        if let Some(continuum_model_list) = available_models_by_provider.get("continuum") {
            for model_name in continuum_model_list {
                if !model_routes.contains_key(model_name) {
                    // This is a Continuum-only model
                    let route = ModelRoute {
                        primary: self.default_proxy.clone(),
                        fallbacks: vec![],
                    };
                    model_routes.insert(model_name.clone(), route);
                    debug!("Continuum-only model '{}' has no fallbacks", model_name);
                }
            }
        }

        // Update the cache
        let mut cache = self.cache.write().await;
        cache.update(models_response, model_to_proxy, model_routes);

        info!(
            "Models cache refreshed. Total models: {}, Additional proxy models: {}",
            all_models.len(),
            cache.model_to_proxy.len()
        );

        Ok(())
    }

    /// Get all available models from all configured proxies
    pub async fn get_all_models(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        // Ensure cache is fresh
        self.refresh_cache_if_needed().await;

        // Return the cached response
        let cache = self.cache.read().await;
        cache
            .models_response
            .clone()
            .ok_or_else(|| "No models available".into())
    }

    async fn fetch_models_from_proxy(
        &self,
        client: &Client<HttpsConnector<hyper::client::HttpConnector>>,
        proxy_config: &ProxyConfig,
    ) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        debug!("Fetching models from proxy: {}", proxy_config.base_url);

        let mut req = Request::builder()
            .method("GET")
            .uri(format!("{}/v1/models", proxy_config.base_url));

        if let Some(api_key) = &proxy_config.api_key {
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }
        }

        let req = req.body(Body::empty())?;
        let res = client.request(req).await?;

        if !res.status().is_success() {
            let status = res.status();
            let body_bytes = hyper::body::to_bytes(res.into_body()).await?;
            let body_str = String::from_utf8_lossy(&body_bytes);
            return Err(format!("Failed to fetch models: {} - {}", status, body_str).into());
        }

        let body_bytes = hyper::body::to_bytes(res.into_body()).await?;
        let models_response: Value = serde_json::from_slice(&body_bytes)?;

        // Extract the model list
        if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
            Ok(data.clone())
        } else {
            Ok(vec![])
        }
    }

    /// Get the Tinfoil proxy base URL if configured
    pub fn get_tinfoil_base_url(&self) -> Option<String> {
        self.tinfoil_proxy.as_ref().map(|p| p.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_get_model_equivalencies() {
        // Should have at least one canonical model
        assert!(!MODEL_EQUIVALENCIES.is_empty());

        // Check Llama 3.3 70B mapping
        let llama_mapping = MODEL_EQUIVALENCIES.get(CANONICAL_LLAMA_33_70B).unwrap();
        assert_eq!(
            llama_mapping.get("continuum"),
            Some(&CONTINUUM_LLAMA_33_70B)
        );
        assert_eq!(llama_mapping.get("tinfoil"), Some(&TINFOIL_LLAMA_33_70B));
    }

    #[test]
    fn test_models_cache_new_with_default() {
        let cache = ModelsCache::new_with_default();

        // Should start with empty models
        assert!(cache.models_response.is_some());
        let response = cache.models_response.as_ref().unwrap();
        assert_eq!(response["object"], "list");
        assert_eq!(response["data"].as_array().unwrap().len(), 0);

        // Should be immediately expired
        assert!(cache.is_expired());
        assert!(cache.model_to_proxy.is_empty());
        assert!(cache.model_routes.is_empty());
    }

    #[test]
    fn test_models_cache_update_and_expiry() {
        let mut cache = ModelsCache::new_with_default();

        // Update cache
        let test_models = serde_json::json!({
            "object": "list",
            "data": [{"id": "test-model"}]
        });
        let mut model_to_proxy = HashMap::new();
        model_to_proxy.insert(
            "test-model".to_string(),
            ProxyConfig {
                base_url: "http://test".to_string(),
                api_key: None,
                provider_name: "test".to_string(),
            },
        );
        let model_routes = HashMap::new();

        cache.update(
            test_models.clone(),
            model_to_proxy.clone(),
            model_routes.clone(),
        );

        // Should not be expired immediately after update
        assert!(!cache.is_expired());
        assert_eq!(cache.models_response, Some(test_models));
        assert_eq!(cache.model_to_proxy.len(), 1);

        // Test that expiry is set to 5 minutes in the future
        let expected_expiry = Utc::now() + Duration::minutes(5);
        let time_diff = cache
            .expires_at
            .signed_duration_since(expected_expiry)
            .num_seconds()
            .abs();
        assert!(time_diff <= 1); // Allow 1 second tolerance
    }

    #[test]
    fn test_proxy_router_new_configurations() {
        // Test with OpenAI configuration
        let router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            Some("test-key".to_string()),
            None,
        );
        assert_eq!(router.default_proxy.provider_name, "openai");
        assert_eq!(router.default_proxy.api_key, Some("test-key".to_string()));
        assert!(router.tinfoil_proxy.is_none());

        // Test with Continuum configuration
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            Some("test-key".to_string()),
            None,
        );
        assert_eq!(router.default_proxy.provider_name, "continuum");
        assert_eq!(router.default_proxy.api_key, None); // Continuum doesn't use API key

        // Test with Tinfoil configuration
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );
        assert!(router.tinfoil_proxy.is_some());
        let tinfoil = router.tinfoil_proxy.as_ref().unwrap();
        assert_eq!(tinfoil.provider_name, "tinfoil");
        assert_eq!(tinfoil.base_url, "http://tinfoil.example.com");
        assert_eq!(tinfoil.api_key, None);
    }

    #[test]
    fn test_get_tinfoil_base_url() {
        // Without Tinfoil
        let router = ProxyRouter::new("http://continuum.example.com".to_string(), None, None);
        assert_eq!(router.get_tinfoil_base_url(), None);

        // With Tinfoil
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );
        assert_eq!(
            router.get_tinfoil_base_url(),
            Some("http://tinfoil.example.com".to_string())
        );
    }

    #[tokio::test]
    async fn test_proxy_router_default() {
        let router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            Some("test-key".to_string()),
            None,
        );

        let route = router.get_model_route("gpt-4").await;
        assert_eq!(route.primary.base_url, "https://api.openai.com");
        assert!(route.primary.api_key.is_some());
    }

    #[tokio::test]
    async fn test_proxy_router_with_tinfoil() {
        let router = ProxyRouter::new(
            "http://127.0.0.1:8092".to_string(),
            None,
            Some("http://127.0.0.1:8093".to_string()),
        );

        // Since model discovery is async, we can't test specific model mapping
        // without mocking the HTTP client. Test the default proxy behavior instead.
        let route = router.get_model_route("gpt-4").await;
        assert_eq!(route.primary.base_url, "http://127.0.0.1:8092");
        assert!(route.primary.api_key.is_none());

        // Verify Tinfoil proxy was configured
        assert!(router.tinfoil_proxy.is_some());
        assert_eq!(
            router.get_tinfoil_base_url(),
            Some("http://127.0.0.1:8093".to_string())
        );
    }

    #[test]
    fn test_model_name_translation() {
        let router = ProxyRouter::new(
            "http://127.0.0.1:8092".to_string(),
            None,
            Some("http://127.0.0.1:8093".to_string()),
        );

        // Test canonical name -> provider specific
        assert_eq!(
            router.get_model_name_for_provider("llama-3.3-70b", "tinfoil"),
            "llama3-3-70b"
        );
        assert_eq!(
            router.get_model_name_for_provider("llama-3.3-70b", "continuum"),
            "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
        );

        // Test provider specific -> same provider (should return as-is)
        assert_eq!(
            router.get_model_name_for_provider("llama3-3-70b", "tinfoil"),
            "llama3-3-70b"
        );

        // Test provider specific -> different provider (should translate)
        assert_eq!(
            router.get_model_name_for_provider("llama3-3-70b", "continuum"),
            "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
        );
        assert_eq!(
            router.get_model_name_for_provider(
                "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
                "tinfoil"
            ),
            "llama3-3-70b"
        );

        // Test unknown model (should return as-is)
        assert_eq!(
            router.get_model_name_for_provider("gpt-4", "tinfoil"),
            "gpt-4"
        );
    }

    #[test]
    fn test_model_route_structure() {
        // Test that ModelRoute can be created and cloned
        let primary = ProxyConfig {
            base_url: "http://primary.com".to_string(),
            api_key: Some("key".to_string()),
            provider_name: "primary".to_string(),
        };
        let fallback = ProxyConfig {
            base_url: "http://fallback.com".to_string(),
            api_key: None,
            provider_name: "fallback".to_string(),
        };

        let route = ModelRoute {
            primary: primary.clone(),
            fallbacks: vec![fallback.clone()],
        };

        // Test clone
        let cloned_route = route.clone();
        assert_eq!(cloned_route.primary.provider_name, "primary");
        assert_eq!(cloned_route.fallbacks.len(), 1);
        assert_eq!(cloned_route.fallbacks[0].provider_name, "fallback");
    }

    #[test]
    fn test_proxy_config_debug_trait() {
        // Test that ProxyConfig implements Debug properly
        let config = ProxyConfig {
            base_url: "http://test.com".to_string(),
            api_key: Some("secret".to_string()),
            provider_name: "test".to_string(),
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("test.com"));
        assert!(debug_str.contains("test"));
        // Should contain api_key but we're not testing the exact format
        assert!(debug_str.contains("api_key"));
    }

    #[tokio::test]
    async fn test_get_model_route_with_cache() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );

        // Before cache is populated, should return default proxy route
        let route = router.get_model_route("unknown-model").await;
        assert_eq!(route.primary.provider_name, "continuum");
        assert!(route.fallbacks.is_empty());

        // Test that cache is checked (this is implementation detail but good to verify)
        // The actual cache population would happen via refresh_cache which requires HTTP mocking
    }

    #[test]
    fn test_model_route_with_empty_fallbacks() {
        // Test edge case where primary provider has no fallbacks
        let primary = ProxyConfig {
            base_url: "http://primary.com".to_string(),
            api_key: None,
            provider_name: "primary".to_string(),
        };

        let route = ModelRoute {
            primary: primary.clone(),
            fallbacks: vec![], // No fallbacks available
        };

        // Should still be able to access primary
        assert_eq!(route.primary.provider_name, "primary");
        assert!(route.fallbacks.is_empty());
    }

    #[test]
    fn test_cache_expiry_edge_case() {
        let mut cache = ModelsCache::new_with_default();

        // Set expiry to past time to test immediate expiration
        cache.expires_at = Utc::now() - chrono::Duration::seconds(1);
        assert!(cache.is_expired());

        // Update and verify not expired
        cache.update(
            serde_json::json!({"object": "list", "data": []}),
            HashMap::new(),
            HashMap::new(),
        );
        assert!(!cache.is_expired());
    }

    #[tokio::test]
    async fn test_get_all_models_with_empty_cache() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            None, // No Tinfoil configured
        );

        // This will attempt to refresh cache but fail due to no actual HTTP client
        // In real usage, this would return an error when providers are unavailable
        let result = router.get_all_models().await;

        // The implementation will return cached empty response on refresh failure
        assert!(result.is_ok());
        let models = result.unwrap();
        assert_eq!(models["object"], "list");
    }
}
