use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, error};

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub base_url: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProxyRouter {
    // Map from model name to proxy configuration
    model_to_proxy: HashMap<String, ProxyConfig>,
    // Default proxy if model not found
    default_proxy: ProxyConfig,
}

impl ProxyRouter {
    pub fn new(
        openai_base: String,
        openai_key: Option<String>,
        tinfoil_base: Option<String>,
    ) -> Self {
        let mut model_to_proxy = HashMap::new();

        // Default OpenAI/Continuum proxy config
        let default_proxy = ProxyConfig {
            base_url: openai_base.clone(),
            api_key: if openai_base.contains("api.openai.com") {
                openai_key.clone()
            } else {
                None // Continuum proxy doesn't need API key
            },
        };

        // If tinfoil is configured, add its models
        if let Some(base) = tinfoil_base {
            let tinfoil_config = ProxyConfig {
                base_url: base,
                api_key: None, // Tinfoil proxy doesn't need API key
            };

            // Register tinfoil models
            model_to_proxy.insert("deepseek-r1-70b".to_string(), tinfoil_config.clone());
            model_to_proxy.insert("llama3-3-70b".to_string(), tinfoil_config.clone());
            model_to_proxy.insert("nomic-embed-text".to_string(), tinfoil_config.clone());
        }

        ProxyRouter {
            model_to_proxy,
            default_proxy,
        }
    }

    pub fn get_proxy_for_model(&self, model_name: &str) -> &ProxyConfig {
        self.model_to_proxy
            .get(model_name)
            .unwrap_or(&self.default_proxy)
    }

    /// Get all available models from all configured proxies
    pub async fn get_all_models(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut all_models = Vec::new();
        let mut fetched_proxies = HashMap::new();

        // Create HTTP client
        let https = HttpsConnector::new();
        let client = Client::builder()
            .pool_idle_timeout(Duration::from_secs(15))
            .build::<_, Body>(https);

        // First, fetch from the default proxy
        match self
            .fetch_models_from_proxy(&client, &self.default_proxy)
            .await
        {
            Ok(models) => {
                all_models.extend(models);
                fetched_proxies.insert(self.default_proxy.base_url.clone(), true);
            }
            Err(e) => {
                error!("Failed to fetch models from default proxy: {:?}", e);
            }
        }

        // Then fetch from any unique additional proxies (like tinfoil)
        for config in self.model_to_proxy.values() {
            if !fetched_proxies.contains_key(&config.base_url) {
                match self.fetch_models_from_proxy(&client, config).await {
                    Ok(models) => {
                        all_models.extend(models);
                        fetched_proxies.insert(config.base_url.clone(), true);
                    }
                    Err(e) => {
                        error!(
                            "Failed to fetch models from proxy {}: {:?}",
                            config.base_url, e
                        );
                    }
                }
            }
        }

        Ok(serde_json::json!({
            "object": "list",
            "data": all_models
        }))
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
            return Err(format!("Failed to fetch models: {}", res.status()).into());
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

    #[test]
    fn test_proxy_router_default() {
        let router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            Some("test-key".to_string()),
            None,
        );

        let proxy = router.get_proxy_for_model("gpt-4");
        assert_eq!(proxy.base_url, "https://api.openai.com");
        assert!(proxy.api_key.is_some());
    }

    #[test]
    fn test_proxy_router_with_tinfoil() {
        let router = ProxyRouter::new(
            "http://127.0.0.1:8092".to_string(),
            None,
            Some("http://127.0.0.1:8093".to_string()),
        );

        let proxy = router.get_proxy_for_model("deepseek-r1-70b");
        assert_eq!(proxy.base_url, "http://127.0.0.1:8093");
        assert!(proxy.api_key.is_none());

        let proxy = router.get_proxy_for_model("gpt-4");
        assert_eq!(proxy.base_url, "http://127.0.0.1:8092");
        assert!(proxy.api_key.is_none());
    }
}
