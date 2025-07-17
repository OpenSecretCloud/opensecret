use chrono::{DateTime, Utc};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
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

/// Known model equivalencies across providers
/// This maps a canonical model identifier to provider-specific names
fn get_model_equivalencies() -> HashMap<&'static str, HashMap<&'static str, &'static str>> {
    let mut equivalencies = HashMap::new();

    // Llama 3.3 70B
    let mut llama_33_70b = HashMap::new();
    llama_33_70b.insert("continuum", CONTINUUM_LLAMA_33_70B);
    llama_33_70b.insert("tinfoil", TINFOIL_LLAMA_33_70B);
    equivalencies.insert(CANONICAL_LLAMA_33_70B, llama_33_70b);

    equivalencies
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
        let equivalencies = get_model_equivalencies();

        // First check if this is already a provider-specific name
        for provider_names in equivalencies.values() {
            if let Some((_, model_name)) = provider_names
                .iter()
                .find(|(p, m)| **p == provider_name && **m == canonical_model)
            {
                return model_name.to_string();
            }
        }

        // Then check if we need to translate
        for provider_names in equivalencies.values() {
            // Check if the canonical model matches any known model
            if provider_names.values().any(|m| *m == canonical_model) {
                // Found it, now get the name for the requested provider
                if let Some(name) = provider_names.get(provider_name) {
                    return name.to_string();
                }
            }
        }

        // No translation needed
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
    pub async fn get_model_route(&self, model_name: &str) -> Option<ModelRoute> {
        // Ensure cache is fresh
        self.refresh_cache_if_needed().await;

        let cache = self.cache.read().await;
        cache.model_routes.get(model_name).cloned()
    }

    pub async fn get_proxy_for_model(&self, model_name: &str) -> ProxyConfig {
        // Ensure cache is fresh
        self.refresh_cache_if_needed().await;

        let cache = self.cache.read().await;

        // Check if there's a special route for this model
        if let Some(route) = cache.model_routes.get(model_name) {
            return route.primary.clone();
        }

        cache
            .model_to_proxy
            .get(model_name)
            .cloned()
            .unwrap_or_else(|| self.default_proxy.clone())
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
                        let equivalencies = get_model_equivalencies();

                        for provider_names in equivalencies.values() {
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

        let models_response = serde_json::json!({
            "object": "list",
            "data": all_models
        });

        // Build model routes - simpler approach based on our new fetching order
        let mut model_routes = HashMap::new();
        let model_equivalencies = get_model_equivalencies();

        // For each known equivalency, check if we have both providers
        for provider_names in model_equivalencies.values() {
            let tinfoil_model = provider_names.get("tinfoil");
            let continuum_model = provider_names.get("continuum");

            if let (Some(tinfoil_name), Some(continuum_name)) = (tinfoil_model, continuum_model) {
                // Check if both providers have their respective models
                let tinfoil_has_it = available_models_by_provider
                    .get("tinfoil")
                    .map(|models| models.contains(&tinfoil_name.to_string()))
                    .unwrap_or(false);

                let continuum_has_it = available_models_by_provider
                    .get("continuum")
                    .map(|models| models.contains(&continuum_name.to_string()))
                    .unwrap_or(false);

                if tinfoil_has_it && continuum_has_it {
                    // Both have it - Tinfoil primary, Continuum fallback
                    if let Some(ref tinfoil_proxy) = self.tinfoil_proxy {
                        info!("Model available from both providers - Tinfoil ({}) primary, Continuum ({}) fallback", 
                              tinfoil_name, continuum_name);

                        let route = ModelRoute {
                            primary: tinfoil_proxy.clone(),
                            fallbacks: vec![self.default_proxy.clone()],
                        };

                        // Map both names to this route
                        model_routes.insert(tinfoil_name.to_string(), route.clone());
                        model_routes.insert(continuum_name.to_string(), route);
                    }
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

    #[tokio::test]
    async fn test_proxy_router_default() {
        let router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            Some("test-key".to_string()),
            None,
        );

        let proxy = router.get_proxy_for_model("gpt-4").await;
        assert_eq!(proxy.base_url, "https://api.openai.com");
        assert!(proxy.api_key.is_some());
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
        let proxy = router.get_proxy_for_model("gpt-4").await;
        assert_eq!(proxy.base_url, "http://127.0.0.1:8092");
        assert!(proxy.api_key.is_none());

        // Verify Tinfoil proxy was configured
        assert!(router.tinfoil_proxy.is_some());
        assert_eq!(
            router.get_tinfoil_base_url(),
            Some("http://127.0.0.1:8093".to_string())
        );
    }
}
