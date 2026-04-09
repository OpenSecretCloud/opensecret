#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub provider_name: String,
}

#[derive(Debug, Clone)]
pub struct ProxyRouter {
    default_proxy: ProxyConfig,
    tinfoil_proxy: Option<ProxyConfig>,
}

pub fn canonicalize_tinfoil_model(model: &str) -> String {
    match model {
        "whisper-large-v3-turbo" => "whisper-large-v3".to_string(),
        _ => model.to_string(),
    }
}

impl ProxyRouter {
    pub fn new(
        openai_base: String,
        openai_key: Option<String>,
        tinfoil_base: Option<String>,
    ) -> Self {
        let default_proxy = ProxyConfig {
            base_url: openai_base.clone(),
            api_key: if openai_base.contains("api.openai.com") {
                openai_key
            } else {
                None
            },
            provider_name: if openai_base.contains("api.openai.com") {
                "openai".to_string()
            } else {
                "continuum".to_string()
            },
        };

        let tinfoil_proxy = tinfoil_base
            .filter(|base| !base.is_empty())
            .map(|base_url| ProxyConfig {
                base_url,
                api_key: None,
                provider_name: "tinfoil".to_string(),
            });

        Self {
            default_proxy,
            tinfoil_proxy,
        }
    }

    pub fn get_default_proxy(&self) -> ProxyConfig {
        self.default_proxy.clone()
    }

    pub fn get_completion_proxy(&self) -> ProxyConfig {
        self.tinfoil_proxy
            .clone()
            .unwrap_or_else(|| self.default_proxy.clone())
    }

    pub fn get_tinfoil_proxy(&self) -> Option<ProxyConfig> {
        self.tinfoil_proxy.clone()
    }

    pub fn get_tinfoil_base_url(&self) -> Option<String> {
        self.tinfoil_proxy
            .as_ref()
            .map(|proxy| proxy.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_router_uses_openai_api_key_only_for_openai() {
        let router = ProxyRouter::new(
            "https://api.openai.com".to_string(),
            Some("test-key".to_string()),
            None,
        );

        assert_eq!(router.get_default_proxy().provider_name, "openai");
        assert_eq!(
            router.get_default_proxy().api_key,
            Some("test-key".to_string())
        );

        let continuum_router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            Some("ignored".to_string()),
            None,
        );

        assert_eq!(
            continuum_router.get_default_proxy().provider_name,
            "continuum"
        );
        assert_eq!(continuum_router.get_default_proxy().api_key, None);
    }

    #[test]
    fn test_completion_proxy_prefers_tinfoil_when_configured() {
        let router = ProxyRouter::new(
            "http://continuum.example.com".to_string(),
            None,
            Some("http://tinfoil.example.com".to_string()),
        );

        assert_eq!(router.get_completion_proxy().provider_name, "tinfoil");
        assert_eq!(
            router.get_tinfoil_base_url(),
            Some("http://tinfoil.example.com".to_string())
        );
    }

    #[test]
    fn test_completion_proxy_falls_back_to_default_without_tinfoil() {
        let router = ProxyRouter::new("http://continuum.example.com".to_string(), None, None);

        assert_eq!(router.get_completion_proxy().provider_name, "continuum");
        assert_eq!(router.get_tinfoil_proxy(), None);
    }

    #[test]
    fn test_tinfoil_model_canonicalization() {
        assert_eq!(canonicalize_tinfoil_model("llama3-3-70b"), "llama3-3-70b");
        assert_eq!(
            canonicalize_tinfoil_model("whisper-large-v3-turbo"),
            "whisper-large-v3"
        );
    }
}
