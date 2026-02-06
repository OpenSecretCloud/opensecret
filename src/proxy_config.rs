use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::warn;

// Canonical model names - exported for use across the codebase
#[allow(dead_code)]
pub const MODEL_LLAMA_33_70B: &str = "llama-3.3-70b";
#[allow(dead_code)]
pub const MODEL_GEMMA_3_27B: &str = "gemma-3-27b";
#[allow(dead_code)]
pub const MODEL_GPT_OSS_120B: &str = "gpt-oss-120b";
#[allow(dead_code)]
pub const MODEL_WHISPER_LARGE_V3: &str = "whisper-large-v3";

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

#[derive(Debug, Clone)]
pub struct ProxyRouter {
    // Static routing table built at initialization
    model_routes: HashMap<String, ModelRoute>,
    // Static models response for user-facing API
    models_response: Value,
    // Default proxy if model not found
    default_proxy: ProxyConfig,
    // Tinfoil proxy configuration if configured
    tinfoil_proxy: Option<ProxyConfig>,
}

impl ProxyRouter {
    /// Get the default proxy configuration (for health checks)
    pub fn get_default_proxy(&self) -> ProxyConfig {
        self.default_proxy.clone()
    }

    /// Get the provider-specific model name for a given canonical model
    pub fn get_model_name_for_provider(&self, model: &str, provider_name: &str) -> String {
        // Simple hardcoded translations for models that have different names
        match (model, provider_name) {
            // Llama 3.3 70B translations
            ("llama-3.3-70b", "tinfoil") => "llama3-3-70b".to_string(),
            ("llama3-3-70b", "continuum") => "llama-3.3-70b".to_string(),

            // Whisper translations
            ("whisper-large-v3", "tinfoil") => "whisper-large-v3-turbo".to_string(),
            ("whisper-large-v3-turbo", "continuum") => "whisper-large-v3".to_string(),

            // All other models use the same name on both providers
            _ => model.to_string(),
        }
    }

    pub fn new(
        openai_base: String,
        openai_key: Option<String>,
        tinfoil_base: Option<String>,
    ) -> Self {
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

        // Build static routing table
        let model_routes = Self::build_static_routes(&default_proxy, &tinfoil_proxy);

        // Build static models response
        let models_response = Self::build_models_response(&tinfoil_proxy);

        ProxyRouter {
            model_routes,
            models_response,
            default_proxy,
            tinfoil_proxy,
        }
    }

    /// Build static routing table at initialization
    fn build_static_routes(
        continuum_proxy: &ProxyConfig,
        tinfoil_proxy: &Option<ProxyConfig>,
    ) -> HashMap<String, ModelRoute> {
        let mut routes = HashMap::new();

        if let Some(tinfoil) = tinfoil_proxy {
            // Models on both providers: Tinfoil primary, Continuum fallback
            let both_route = ModelRoute {
                primary: tinfoil.clone(),
                fallbacks: vec![continuum_proxy.clone()],
            };

            // GPT-OSS 120B (on both, same name)
            routes.insert("gpt-oss-120b".to_string(), both_route.clone());

            // Whisper Large V3 (on both)
            routes.insert("whisper-large-v3".to_string(), both_route.clone());
            routes.insert("whisper-large-v3-turbo".to_string(), both_route.clone());

            // Tinfoil-only models
            let tinfoil_route = ModelRoute {
                primary: tinfoil.clone(),
                fallbacks: vec![],
            };
            routes.insert("llama-3.3-70b".to_string(), tinfoil_route.clone());
            routes.insert("llama3-3-70b".to_string(), tinfoil_route.clone());
            routes.insert("deepseek-r1-0528".to_string(), tinfoil_route.clone());
            routes.insert("qwen3-vl-30b".to_string(), tinfoil_route.clone());
            routes.insert("kimi-k2-5".to_string(), tinfoil_route.clone());
            routes.insert("nomic-embed-text".to_string(), tinfoil_route.clone());

            // Continuum-only models
            let continuum_route = ModelRoute {
                primary: continuum_proxy.clone(),
                fallbacks: vec![],
            };
            routes.insert("gemma-3-27b".to_string(), continuum_route);
        } else {
            // No Tinfoil: only Continuum models (llama-3.3-70b not available without Tinfoil)
            let continuum_route = ModelRoute {
                primary: continuum_proxy.clone(),
                fallbacks: vec![],
            };
            routes.insert("gemma-3-27b".to_string(), continuum_route.clone());
            routes.insert("gpt-oss-120b".to_string(), continuum_route.clone());
            routes.insert("whisper-large-v3".to_string(), continuum_route);
        }

        routes
    }

    /// Build static models response for user-facing API
    fn build_models_response(tinfoil_proxy: &Option<ProxyConfig>) -> Value {
        let created_timestamp = 1700000000i64;

        let models = if tinfoil_proxy.is_some() {
            // With Tinfoil: show all available models using canonical names
            vec![
                // Models available on both (use canonical names)
                "gpt-oss-120b",
                "whisper-large-v3",
                // Continuum-only
                "gemma-3-27b",
                // Tinfoil-only
                "llama-3.3-70b",
                "deepseek-r1-0528",
                "qwen3-vl-30b",
                "kimi-k2-5",
                "nomic-embed-text",
            ]
        } else {
            // Without Tinfoil: only Continuum models (llama-3.3-70b not available)
            vec!["gemma-3-27b", "gpt-oss-120b", "whisper-large-v3"]
        };

        let model_objects: Vec<Value> = models
            .iter()
            .map(|id| {
                json!({
                    "id": id,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": "system"
                })
            })
            .collect();

        json!({
            "object": "list",
            "data": model_objects
        })
    }

    /// Get the model route configuration for a given model
    /// Returns None if the model is not found
    pub fn get_model_route(&self, model_name: &str) -> Option<ModelRoute> {
        let route = self.model_routes.get(model_name).cloned();

        if route.is_none() {
            warn!("Unknown model '{}' requested - no route found", model_name);
        }

        route
    }

    /// Get all available models (static list built at initialization)
    pub fn get_all_models(&self) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(self.models_response.clone())
    }

    /// Get the Tinfoil proxy base URL if configured
    pub fn get_tinfoil_base_url(&self) -> Option<String> {
        self.tinfoil_proxy.as_ref().map(|p| p.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_translation() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );

        // Llama translations
        assert_eq!(
            router.get_model_name_for_provider("llama-3.3-70b", "tinfoil"),
            "llama3-3-70b"
        );
        assert_eq!(
            router.get_model_name_for_provider("llama3-3-70b", "continuum"),
            "llama-3.3-70b"
        );

        // Whisper translations
        assert_eq!(
            router.get_model_name_for_provider("whisper-large-v3", "tinfoil"),
            "whisper-large-v3-turbo"
        );
        assert_eq!(
            router.get_model_name_for_provider("whisper-large-v3-turbo", "continuum"),
            "whisper-large-v3"
        );

        // Models with same name on both providers
        assert_eq!(
            router.get_model_name_for_provider("gpt-oss-120b", "tinfoil"),
            "gpt-oss-120b"
        );
        assert_eq!(
            router.get_model_name_for_provider("gemma-3-27b", "continuum"),
            "gemma-3-27b"
        );
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

    #[test]
    fn test_proxy_router_static_routing() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );

        // Test llama-3.3-70b is Tinfoil-only (no Continuum fallback)
        let llama_route = router.get_model_route("llama-3.3-70b");
        assert!(llama_route.is_some());
        let route = llama_route.unwrap();
        assert_eq!(route.primary.provider_name, "tinfoil");
        assert!(route.fallbacks.is_empty()); // No Continuum fallback

        // Test provider-specific names work too
        let tinfoil_llama_route = router.get_model_route("llama3-3-70b");
        assert!(tinfoil_llama_route.is_some());

        // Test gpt-oss-120b has both providers (Tinfoil primary, Continuum fallback)
        let gpt_route = router.get_model_route("gpt-oss-120b");
        assert!(gpt_route.is_some());
        let route = gpt_route.unwrap();
        assert_eq!(route.primary.provider_name, "tinfoil");
        assert_eq!(route.fallbacks.len(), 1);
        assert_eq!(route.fallbacks[0].provider_name, "continuum");

        // Unknown model should return None
        let unknown_route = router.get_model_route("gpt-4");
        assert!(unknown_route.is_none());
    }

    #[test]
    fn test_proxy_router_without_tinfoil() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            None, // No Tinfoil
        );

        // llama-3.3-70b should NOT be available without Tinfoil
        let llama_route = router.get_model_route("llama-3.3-70b");
        assert!(llama_route.is_none());

        // But gemma and gpt-oss should be available on Continuum
        let gemma_route = router.get_model_route("gemma-3-27b");
        assert!(gemma_route.is_some());
        let route = gemma_route.unwrap();
        assert_eq!(route.primary.provider_name, "continuum");
        assert!(route.fallbacks.is_empty());

        // Verify Tinfoil proxy not configured
        assert!(router.tinfoil_proxy.is_none());
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
    fn test_get_all_models_static() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );

        // Should return static models list immediately
        let result = router.get_all_models();
        assert!(result.is_ok());

        let models = result.unwrap();
        assert_eq!(models["object"], "list");

        // Should have models in the data array
        let data = models["data"].as_array().unwrap();
        assert!(!data.is_empty());

        // Check that canonical names are used
        let model_ids: Vec<String> = data
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        assert!(model_ids.contains(&"llama-3.3-70b".to_string()));
        assert!(model_ids.contains(&"gpt-oss-120b".to_string()));
    }
}
