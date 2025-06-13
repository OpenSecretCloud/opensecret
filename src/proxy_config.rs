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

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub base_url: String,
    pub api_key: Option<String>,
}

#[derive(Debug)]
struct ModelsCache {
    // Cached models response for user-facing API
    models_response: Option<Value>,
    // Map from model name to proxy configuration for internal routing
    model_to_proxy: HashMap<String, ProxyConfig>,
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
            expires_at: Utc::now(),
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    fn update(&mut self, models_response: Value, model_to_proxy: HashMap<String, ProxyConfig>) {
        self.models_response = Some(models_response);
        self.model_to_proxy = model_to_proxy;
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
}

impl ProxyRouter {
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
        };

        // Collect additional proxies
        let mut additional_proxies = Vec::new();
        if let Some(base) = tinfoil_base {
            additional_proxies.push(ProxyConfig {
                base_url: base,
                api_key: None, // Tinfoil proxy doesn't need API key
            });
        }

        ProxyRouter {
            cache,
            default_proxy,
            additional_proxies,
        }
    }

    /// Get the Tinfoil proxy base URL if configured
    pub fn get_tinfoil_base_url(&self) -> Option<String> {
        // The first additional proxy is the Tinfoil proxy if configured
        self.additional_proxies.first().map(|p| p.base_url.clone())
    }

    pub async fn get_proxy_for_model(&self, model_name: &str) -> ProxyConfig {
        // Ensure cache is fresh
        self.refresh_cache_if_needed().await;

        let cache = self.cache.read().await;
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
                // Remove duplicate of default model if it exists
                models.retain(|m| {
                    m.get("id").and_then(|v| v.as_str()) != Some(DEFAULT_CONTINUUM_MODEL)
                });
                all_models.extend(models);
                fetched_proxies.insert(self.default_proxy.base_url.clone(), true);
                // Note: We don't add default proxy models to model_to_proxy map
                // because they use the default proxy by default
            }
            Err(e) => {
                warn!("Failed to fetch models from default proxy: {:?}", e);
            }
        }

        // Then fetch from any unique additional proxies (like tinfoil)
        for proxy_config in &self.additional_proxies {
            if !fetched_proxies.contains_key(&proxy_config.base_url) {
                match self.fetch_models_from_proxy(&client, proxy_config).await {
                    Ok(models) => {
                        all_models.extend(models.clone());
                        fetched_proxies.insert(proxy_config.base_url.clone(), true);

                        // Add these models to the routing map
                        for model in &models {
                            if let Some(model_id) = model.get("id").and_then(|v| v.as_str()) {
                                debug!(
                                    "Mapped model '{}' to proxy {}",
                                    model_id, proxy_config.base_url
                                );
                                model_to_proxy.insert(model_id.to_string(), proxy_config.clone());
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to fetch models from proxy {}: {:?}",
                            proxy_config.base_url, e
                        );
                    }
                }
            }
        }

        let models_response = serde_json::json!({
            "object": "list",
            "data": all_models
        });

        // Update the cache
        let mut cache = self.cache.write().await;
        cache.update(models_response, model_to_proxy);

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

        // Verify additional proxies were configured
        assert_eq!(router.additional_proxies.len(), 1);
        assert_eq!(
            router.additional_proxies[0].base_url,
            "http://127.0.0.1:8093"
        );
    }
}
