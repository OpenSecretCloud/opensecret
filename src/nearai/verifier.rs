use crate::nearai::attestation::{
    verify_compose_manifest, verify_report_data_binding, verify_signing_pubkey_matches_address,
    verify_tdx_quote,
};
use crate::nearai::error::NearAiError;
use crate::nearai::models::{AttestationBaseInfo, AttestationReport};
use crate::nearai::nras::{fetch_jwks, verify_gpu_attestation, NrasJwks, NRAS_ISSUER};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

const VERIFICATION_TTL: Duration = Duration::from_secs(10 * 60);
const PERIODIC_REVERIFY_INTERVAL: Duration = Duration::from_secs(10 * 60);
const JWKS_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone)]
pub struct VerifiedModelNode {
    pub signing_public_key: String,
}

#[derive(Debug, Clone)]
struct VerifiedModel {
    nodes: Vec<VerifiedModelNode>,
    verified_at: Instant,
}

#[derive(Debug, Clone)]
struct CachedJwks {
    jwks: NrasJwks,
    fetched_at: Instant,
}

pub struct NearAiVerifier {
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    cache: RwLock<HashMap<String, VerifiedModel>>,
    verify_lock: Mutex<()>,
    jwks_cache: RwLock<Option<CachedJwks>>,
    rr_counter: AtomicUsize,
}

impl NearAiVerifier {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client should build");

        Self {
            base_url,
            api_key,
            http,
            cache: RwLock::new(HashMap::new()),
            verify_lock: Mutex::new(()),
            jwks_cache: RwLock::new(None),
            rr_counter: AtomicUsize::new(0),
        }
    }

    pub fn is_configured(&self) -> bool {
        self.api_key.as_ref().is_some_and(|k| !k.trim().is_empty())
    }

    pub fn spawn_periodic_verification(self: Arc<Self>, models: Vec<String>) {
        if !self.is_configured() {
            warn!("Near.AI verifier not configured (missing API key); skipping preflight verification");
            return;
        }

        tokio::spawn(async move {
            info!("Near.AI preflight verification starting");
            for model in &models {
                if let Err(e) = self.verify_and_cache_model(model).await {
                    warn!(
                        "Near.AI preflight verification failed for model {}: {}",
                        model, e
                    );
                } else {
                    info!(
                        "Near.AI preflight verification succeeded for model {}",
                        model
                    );
                }
            }

            loop {
                tokio::time::sleep(PERIODIC_REVERIFY_INTERVAL).await;
                for model in &models {
                    if let Err(e) = self.verify_and_cache_model(model).await {
                        warn!(
                            "Near.AI periodic re-verification failed for model {}: {}",
                            model, e
                        );
                    }
                }
            }
        });
    }

    pub async fn get_verified_model_node(
        &self,
        model: &str,
    ) -> Result<VerifiedModelNode, NearAiError> {
        if !self.is_configured() {
            return Err(NearAiError::Unconfigured);
        }

        // Fast path: fresh cached model.
        if let Some(node) = self.try_get_fresh_cached_node(model).await {
            return Ok(node);
        }

        // Slow path: ensure only one verification runs at a time.
        let _guard = self.verify_lock.lock().await;

        // Re-check after acquiring lock.
        if let Some(node) = self.try_get_fresh_cached_node(model).await {
            return Ok(node);
        }

        self.verify_and_cache_model(model).await?;

        self.try_get_fresh_cached_node(model).await.ok_or_else(|| {
            NearAiError::Attestation("no verified model nodes available".to_string())
        })
    }

    async fn try_get_fresh_cached_node(&self, model: &str) -> Option<VerifiedModelNode> {
        let cache = self.cache.read().await;
        let verified = cache.get(model)?;
        if verified.verified_at.elapsed() > VERIFICATION_TTL {
            return None;
        }

        if verified.nodes.is_empty() {
            return None;
        }

        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % verified.nodes.len();
        Some(verified.nodes[idx].clone())
    }

    async fn verify_and_cache_model(&self, model: &str) -> Result<(), NearAiError> {
        let verified = self.verify_model(model).await?;
        let mut cache = self.cache.write().await;
        cache.insert(model.to_string(), verified);
        Ok(())
    }

    async fn verify_model(&self, model: &str) -> Result<VerifiedModel, NearAiError> {
        let api_key = self.api_key.as_ref().ok_or(NearAiError::Unconfigured)?;

        let nonce_bytes = generate_nonce_bytes()?;
        let nonce_hex = hex::encode(nonce_bytes);

        let report = self
            .fetch_attestation_report(api_key, model, &nonce_hex)
            .await?;

        // Verify gateway attestation (no GPU verification).
        self.verify_single_attestation(
            &report.gateway_attestation,
            &nonce_bytes,
            &nonce_hex,
            false,
        )
        .await?;

        let mut nodes = Vec::new();
        for model_att in &report.model_attestations {
            match self
                .verify_single_attestation(model_att, &nonce_bytes, &nonce_hex, true)
                .await
            {
                Ok(Some(node)) => nodes.push(node),
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        "Near.AI model node attestation failed (model={}, signing_address={}): {}",
                        model, model_att.signing_address, e
                    );
                }
            }
        }

        if nodes.is_empty() {
            return Err(NearAiError::Attestation(
                "no verified model nodes available".to_string(),
            ));
        }

        Ok(VerifiedModel {
            nodes,
            verified_at: Instant::now(),
        })
    }

    async fn fetch_attestation_report(
        &self,
        api_key: &str,
        model: &str,
        nonce_hex: &str,
    ) -> Result<AttestationReport, NearAiError> {
        let url = format!(
            "{}/v1/attestation/report",
            self.base_url.trim_end_matches('/')
        );

        let mut req = self.http.get(&url).query(&[
            ("model", model),
            ("nonce", nonce_hex),
            ("signing_algo", "ecdsa"),
        ]);

        // Attestation report may be public, but include auth when present.
        if !api_key.trim().is_empty() {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let res = req.send().await?;
        let status = res.status();
        let text = res.text().await?;
        if !status.is_success() {
            return Err(NearAiError::HttpStatus {
                url,
                status,
                body: text,
            });
        }

        Ok(serde_json::from_str(&text)?)
    }

    async fn verify_single_attestation(
        &self,
        att: &AttestationBaseInfo,
        nonce_bytes: &[u8; 32],
        nonce_hex: &str,
        verify_gpu: bool,
    ) -> Result<Option<VerifiedModelNode>, NearAiError> {
        let signing_algo = att
            .signing_algo
            .as_deref()
            .unwrap_or("ecdsa")
            .to_lowercase();
        if signing_algo != "ecdsa" {
            return Err(NearAiError::Attestation(format!(
                "unsupported signing_algo: {}",
                signing_algo
            )));
        }

        let quote = verify_tdx_quote(&att.intel_quote).await?;
        verify_report_data_binding(&quote.report_data, &att.signing_address, nonce_bytes)?;
        verify_compose_manifest(&quote.mr_config_id, &att.info)?;

        if verify_gpu {
            let payload_str = att
                .nvidia_payload
                .as_deref()
                .ok_or_else(|| NearAiError::Attestation("missing nvidia_payload".to_string()))?;
            let payload_json: Value = serde_json::from_str(payload_str)?;
            let jwks = self.get_or_fetch_jwks().await?;
            if let Err(e) =
                verify_gpu_attestation(&self.http, &jwks, &payload_json, nonce_hex).await
            {
                match e {
                    NearAiError::JwtKidNotFound(_) => {
                        let jwks = self.refresh_jwks().await?;
                        verify_gpu_attestation(&self.http, &jwks, &payload_json, nonce_hex).await?;
                    }
                    _ => return Err(e),
                }
            }
        }

        let Some(pubkey_hex) = att.signing_public_key.as_deref() else {
            // Gateway attestations do not always include a signing_public_key.
            return Ok(None);
        };

        let normalized_pubkey =
            verify_signing_pubkey_matches_address(pubkey_hex, &att.signing_address)?;

        Ok(Some(VerifiedModelNode {
            signing_public_key: normalized_pubkey,
        }))
    }

    async fn get_or_fetch_jwks(&self) -> Result<NrasJwks, NearAiError> {
        {
            let cache = self.jwks_cache.read().await;
            if let Some(cached) = cache.as_ref() {
                if cached.fetched_at.elapsed() <= JWKS_TTL {
                    return Ok(cached.jwks.clone());
                }
            }
        }

        self.refresh_jwks().await
    }

    async fn refresh_jwks(&self) -> Result<NrasJwks, NearAiError> {
        let jwks = fetch_jwks(&self.http).await?;

        // Sanity check: NRAS issuer is as expected.
        if jwks.keys.is_empty() {
            return Err(NearAiError::Jwt(format!(
                "JWKS from {} contained no keys",
                NRAS_ISSUER
            )));
        }

        let mut cache = self.jwks_cache.write().await;
        *cache = Some(CachedJwks {
            jwks: jwks.clone(),
            fetched_at: Instant::now(),
        });
        Ok(jwks)
    }
}

fn generate_nonce_bytes() -> Result<[u8; 32], NearAiError> {
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce)
        .map_err(|e| NearAiError::Crypto(format!("nonce generation failed: {e}")))?;
    Ok(nonce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nearai::e2ee::{decrypt_chat_completion_json_in_place, prepare_e2ee_request};
    use serde_json::json;

    #[tokio::test]
    async fn test_unconfigured_fails_fast() {
        let v = NearAiVerifier::new("https://cloud-api.near.ai".to_string(), None);
        let err = v
            .get_verified_model_node("zai-org/GLM-5-FP8")
            .await
            .unwrap_err();
        match err {
            NearAiError::Unconfigured => {}
            other => panic!("expected Unconfigured, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_cached_node_round_robins_when_fresh() {
        let v = NearAiVerifier::new(
            "https://cloud-api.near.ai".to_string(),
            Some("test".to_string()),
        );

        {
            let mut cache = v.cache.write().await;
            cache.insert(
                "zai-org/GLM-5-FP8".to_string(),
                VerifiedModel {
                    nodes: vec![
                        VerifiedModelNode {
                            signing_public_key: "pk1".to_string(),
                        },
                        VerifiedModelNode {
                            signing_public_key: "pk2".to_string(),
                        },
                    ],
                    verified_at: Instant::now(),
                },
            );
        }

        let n1 = v
            .get_verified_model_node("zai-org/GLM-5-FP8")
            .await
            .unwrap();
        let n2 = v
            .get_verified_model_node("zai-org/GLM-5-FP8")
            .await
            .unwrap();
        assert_ne!(n1.signing_public_key, n2.signing_public_key);
    }

    #[tokio::test]
    async fn test_try_get_fresh_cached_node_returns_none_when_stale() {
        let v = NearAiVerifier::new(
            "https://cloud-api.near.ai".to_string(),
            Some("test".to_string()),
        );

        {
            let mut cache = v.cache.write().await;
            cache.insert(
                "zai-org/GLM-5-FP8".to_string(),
                VerifiedModel {
                    nodes: vec![VerifiedModelNode {
                        signing_public_key: "pk1".to_string(),
                    }],
                    verified_at: Instant::now() - (VERIFICATION_TTL + Duration::from_secs(1)),
                },
            );
        }

        assert!(v
            .try_get_fresh_cached_node("zai-org/GLM-5-FP8")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_get_or_fetch_jwks_uses_cache_when_fresh() {
        let v = NearAiVerifier::new(
            "https://cloud-api.near.ai".to_string(),
            Some("test".to_string()),
        );

        let jwks = NrasJwks {
            keys: vec![crate::nearai::nras::NrasJwk {
                kid: Some("kid".to_string()),
                x: Some("x".to_string()),
                y: Some("y".to_string()),
            }],
        };

        {
            let mut cache = v.jwks_cache.write().await;
            *cache = Some(CachedJwks {
                jwks: jwks.clone(),
                fetched_at: Instant::now(),
            });
        }

        let fetched = v.get_or_fetch_jwks().await.unwrap();
        assert_eq!(fetched.keys.len(), 1);
        assert_eq!(fetched.keys[0].kid.as_deref(), Some("kid"));
    }

    #[tokio::test]
    #[ignore]
    async fn live_nearai_attestation_glm5() {
        let api_key = match std::env::var("NEAR_API_KEY") {
            Ok(v) if !v.trim().is_empty() => v,
            _ => {
                eprintln!("NEAR_API_KEY not set; skipping");
                return;
            }
        };

        let base_url = std::env::var("NEARAI_API_BASE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "https://cloud-api.near.ai".to_string());

        let v = NearAiVerifier::new(base_url, Some(api_key));
        let node = v
            .get_verified_model_node("zai-org/GLM-5-FP8")
            .await
            .expect("attestation verification should succeed");

        assert!(!node.signing_public_key.trim().is_empty());
    }

    #[tokio::test]
    #[ignore]
    async fn live_nearai_chat_completion_glm5() {
        let api_key = match std::env::var("NEAR_API_KEY") {
            Ok(v) if !v.trim().is_empty() => v,
            _ => {
                eprintln!("NEAR_API_KEY not set; skipping");
                return;
            }
        };

        let base_url = std::env::var("NEARAI_API_BASE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "https://cloud-api.near.ai".to_string());

        let v = NearAiVerifier::new(base_url.clone(), Some(api_key.clone()));
        let node = v
            .get_verified_model_node("zai-org/GLM-5-FP8")
            .await
            .expect("attestation verification should succeed");

        let mut body = json!({
            "model": "zai-org/GLM-5-FP8",
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.0,
            "max_tokens": 16
        });

        let crypto = prepare_e2ee_request(&mut body, &node.signing_public_key)
            .expect("request encryption should succeed");

        let sent = body["messages"][0]["content"].as_str().unwrap();
        assert_ne!(sent, "ping");

        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let res = client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("x-signing-algo", "ecdsa")
            .header("x-client-pub-key", &crypto.client_public_key_hex)
            .header("x-model-pub-key", &node.signing_public_key)
            .json(&body)
            .send()
            .await
            .expect("Near.AI request should send");

        assert!(res.status().is_success(), "status={}", res.status());

        let mut resp: serde_json::Value =
            res.json().await.expect("Near.AI response should be JSON");
        decrypt_chat_completion_json_in_place(&mut resp, &crypto)
            .expect("Near.AI response should decrypt");

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.trim().is_empty());
    }
}
