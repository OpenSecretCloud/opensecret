use chrono::{DateTime, Utc};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

const DEFAULT_CONTINUUM_MODEL: &str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4";

/// Known model equivalencies across providers
/// This maps a canonical model identifier to provider-specific names
fn get_model_equivalencies() -> HashMap<&'static str, HashMap<&'static str, &'static str>> {
    let mut equivalencies = HashMap::new();
    
    // Llama 3.3 70B
    let mut llama_33_70b = HashMap::new();
    llama_33_70b.insert("continuum", DEFAULT_CONTINUUM_MODEL);
    llama_33_70b.insert("tinfoil", "llama3-3-70b");
    equivalencies.insert("llama-3.3-70b", llama_33_70b);
    
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
        // Default Continuum model that should always be available
        let default_model = serde_json::json!({
            "id": DEFAULT_CONTINUUM_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "continuum"
        });

        let models_response = serde_json::json!({
            "object": "list",
            "data": [default_model]
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
    // Additional proxy configurations (e.g., tinfoil)
    additional_proxies: Vec<ProxyConfig>,
    // Tinfoil proxy configuration if configured
    tinfoil_proxy: Option<ProxyConfig>,
}

impl ProxyRouter {
    /// Get the provider-specific model name for a given canonical model
    pub fn get_model_name_for_provider(&self, canonical_model: &str, provider_name: &str) -> String {
        let equivalencies = get_model_equivalencies();
        
        // First check if this is already a provider-specific name
        for provider_names in equivalencies.values() {
            if let Some((_, model_name)) = provider_names.iter().find(|(p, m)| **p == provider_name && **m == canonical_model) {
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

        // Collect additional proxies
        let mut additional_proxies = Vec::new();
        if let Some(ref tp) = tinfoil_proxy {
            additional_proxies.push(tp.clone());
        }

        ProxyRouter {
            cache,
            default_proxy,
            additional_proxies,
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
        let mut fetched_proxies = HashMap::new();
        let mut available_models_by_provider: HashMap<String, Vec<String>> = HashMap::new();

        // Always include the default Continuum model
        let default_continuum_model = serde_json::json!({
            "id": DEFAULT_CONTINUUM_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "continuum"
        });
        all_models.push(default_continuum_model);

        // First, fetch from the default proxy
        match self
            .fetch_models_from_proxy(&client, &self.default_proxy)
            .await
        {
            Ok(mut models) => {
                let mut provider_models = Vec::new();

                // Remove duplicate of default model if it exists
                models.retain(|m| {
                    if let Some(model_id) = m.get("id").and_then(|v| v.as_str()) {
                        provider_models.push(model_id.to_string());
                        model_id != DEFAULT_CONTINUUM_MODEL
                    } else {
                        true
                    }
                });

                // Track continuum models
                available_models_by_provider
                    .insert(self.default_proxy.provider_name.clone(), provider_models);

                all_models.extend(models);
                fetched_proxies.insert(self.default_proxy.base_url.clone(), true);
                // Note: We don't add default proxy models to model_to_proxy map
                // because they use the default proxy by default
            }
            Err(e) => {
                warn!("Failed to fetch models from default proxy: {:?}", e);
                available_models_by_provider
                    .insert(self.default_proxy.provider_name.clone(), Vec::new());
            }
        }

        // Then fetch from any unique additional proxies (like tinfoil)
        for proxy_config in &self.additional_proxies {
            if !fetched_proxies.contains_key(&proxy_config.base_url) {
                match self.fetch_models_from_proxy(&client, proxy_config).await {
                    Ok(models) => {
                        let mut provider_models = Vec::new();

                        all_models.extend(models.clone());
                        fetched_proxies.insert(proxy_config.base_url.clone(), true);

                        // Add these models to the routing map
                        for model in &models {
                            if let Some(model_id) = model.get("id").and_then(|v| v.as_str()) {
                                provider_models.push(model_id.to_string());
                                debug!(
                                    "Mapped model '{}' to proxy {}",
                                    model_id, proxy_config.base_url
                                );
                                model_to_proxy.insert(model_id.to_string(), proxy_config.clone());
                            }
                        }

                        available_models_by_provider
                            .insert(proxy_config.provider_name.clone(), provider_models);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to fetch models from proxy {}: {:?}",
                            proxy_config.base_url, e
                        );
                        available_models_by_provider
                            .insert(proxy_config.provider_name.clone(), Vec::new());
                    }
                }
            }
        }

        let models_response = serde_json::json!({
            "object": "list",
            "data": all_models
        });

        // Build model routes dynamically based on what's available
        let mut model_routes = HashMap::new();
        let model_equivalencies = get_model_equivalencies();

        // For each known model equivalency, check which providers have it
        for (canonical_name, provider_names) in &model_equivalencies {
            let mut providers_with_model = Vec::new();
            
            // Check each provider
            for (provider, model_name) in provider_names {
                if let Some(models) = available_models_by_provider.get(*provider) {
                    if models.contains(&model_name.to_string()) {
                        providers_with_model.push((*provider, *model_name));
                    }
                }
            }
            
            // If multiple providers have this model, set up routing with fallback
            if providers_with_model.len() > 1 {
                // For now, prioritize tinfoil over continuum for shared models
                let primary_provider = if providers_with_model.iter().any(|(p, _)| *p == "tinfoil") {
                    "tinfoil"
                } else {
                    providers_with_model.get(0).map(|(p, _)| *p).unwrap_or("continuum")
                };
                
                let fallback_provider = if primary_provider == "tinfoil" {
                    "continuum"
                } else {
                    "tinfoil"
                };
                
                // Get proxy configs
                let primary_proxy = if primary_provider == "tinfoil" {
                    self.tinfoil_proxy.as_ref()
                } else {
                    Some(&self.default_proxy)
                };
                
                let fallback_proxy = if fallback_provider == "tinfoil" {
                    self.tinfoil_proxy.as_ref()
                } else {
                    Some(&self.default_proxy)
                };
                
                if let (Some(primary), Some(fallback)) = (primary_proxy, fallback_proxy) {
                    info!(
                        "Model {} available from multiple providers - {} (primary) and {} (fallback)",
                        canonical_name, primary_provider, fallback_provider
                    );
                    
                    // Build list of providers in order (tinfoil first if available)
                    let mut ordered_providers = Vec::new();
                    if primary_provider == "tinfoil" {
                        ordered_providers.push(primary.clone());
                        ordered_providers.push(fallback.clone());
                    } else {
                        ordered_providers.push(fallback.clone());
                        ordered_providers.push(primary.clone());
                    }
                    
                    let route = ModelRoute {
                        primary: ordered_providers[0].clone(),
                        fallbacks: vec![ordered_providers[1].clone()],
                    };
                    
                    // Map all provider-specific names to this route
                    for (_, model_name) in &providers_with_model {
                        model_routes.insert(model_name.to_string(), route.clone());
                    }
                }
            } else if providers_with_model.len() == 1 {
                // Single provider - no special routing needed
                if let Some((provider, _)) = providers_with_model.get(0) {
                    info!("Model {} only available from {}", canonical_name, provider);
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

        // Verify providers were configured
        assert_eq!(router.providers.len(), 2);
        assert!(router.providers.contains_key("continuum"));
        assert!(router.providers.contains_key("tinfoil"));
    }
}
