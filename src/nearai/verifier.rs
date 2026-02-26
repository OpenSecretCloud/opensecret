use crate::nearai::attestation::{
    verify_compose_manifest, verify_report_data_binding, verify_signing_pubkey_matches_address,
    verify_tdx_quote,
};
use crate::nearai::error::NearAiError;
use crate::nearai::models::{AttestationBaseInfo, AttestationReport};
use crate::nearai::nras::{fetch_jwks, verify_gpu_attestation, NrasJwks, NRAS_JWKS_URL};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, trace, warn};

const VERIFICATION_TTL: Duration = Duration::from_secs(10 * 60);
const PERIODIC_REVERIFY_INTERVAL: Duration = Duration::from_secs(10 * 60);
const JWKS_TTL: Duration = Duration::from_secs(60 * 60);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const HTTP_RETRY_ATTEMPTS: usize = 3;
const HTTP_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);

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
}

impl NearAiVerifier {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let base_url = normalize_base_url(&base_url);
        let http = reqwest::Client::builder()
            .timeout(HTTP_REQUEST_TIMEOUT)
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .pool_idle_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(100)
            .user_agent("opensecret/nearai-verifier")
            .build()
            .expect("reqwest client should build");

        Self {
            base_url,
            api_key,
            http,
            cache: RwLock::new(HashMap::new()),
            verify_lock: Mutex::new(()),
            jwks_cache: RwLock::new(None),
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

            let mut last_state: HashMap<String, bool> = HashMap::new();

            for model in &models {
                match self.verify_and_cache_model(model).await {
                    Ok(_) => {
                        debug!(
                            "Near.AI preflight verification succeeded for model {}",
                            model
                        );

                        let prev = last_state.insert(model.clone(), true);
                        if prev != Some(true) {
                            info!(
                                "Near.AI verification state changed for model {}: {} -> verified",
                                model,
                                prev.map(|v| if v { "verified" } else { "failed" })
                                    .unwrap_or("unknown")
                            );
                        }
                    }
                    Err(e) => {
                        error!(
                            "Near.AI preflight verification failed for model {}: {}",
                            model, e
                        );

                        let prev = last_state.insert(model.clone(), false);
                        if prev != Some(false) {
                            info!(
                                "Near.AI verification state changed for model {}: {} -> failed",
                                model,
                                prev.map(|v| if v { "verified" } else { "failed" })
                                    .unwrap_or("unknown")
                            );
                        }
                    }
                }
            }

            loop {
                tokio::time::sleep(PERIODIC_REVERIFY_INTERVAL).await;
                for model in &models {
                    match self.verify_and_cache_model(model).await {
                        Ok(_) => {
                            debug!(
                                "Near.AI periodic re-verification succeeded for model {}",
                                model
                            );

                            let prev = last_state.insert(model.clone(), true);
                            if prev == Some(false) {
                                info!(
                                    "Near.AI verification state changed for model {}: failed -> verified",
                                    model
                                );
                            }
                        }
                        Err(e) => {
                            error!(
                                "Near.AI periodic re-verification failed for model {}: {}",
                                model, e
                            );

                            let prev = last_state.insert(model.clone(), false);
                            if prev == Some(true) {
                                info!(
                                    "Near.AI verification state changed for model {}: verified -> failed",
                                    model
                                );
                            }
                        }
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
            trace!(
                "Near.AI verifier: cache hit for model={}, node_pubkey={}...",
                model,
                &node.signing_public_key[..node.signing_public_key.len().min(16)]
            );
            return Ok(node);
        }

        debug!(
            "Near.AI verifier: cache miss for model={}, acquiring verify lock",
            model
        );

        // Slow path: ensure only one verification runs at a time.
        let _guard = self.verify_lock.lock().await;

        // Re-check after acquiring lock.
        if let Some(node) = self.try_get_fresh_cached_node(model).await {
            trace!(
                "Near.AI verifier: cache hit after lock for model={}, node_pubkey={}...",
                model,
                &node.signing_public_key[..node.signing_public_key.len().min(16)]
            );
            return Ok(node);
        }

        info!("Near.AI verifier: verifying model={}", model);
        self.verify_and_cache_model(model).await?;

        self.try_get_fresh_cached_node(model).await.ok_or_else(|| {
            NearAiError::Attestation("no verified model nodes available".to_string())
        })
    }

    async fn try_get_fresh_cached_node(&self, model: &str) -> Option<VerifiedModelNode> {
        let cache = self.cache.read().await;
        let verified = cache.get(model)?;
        let age = verified.verified_at.elapsed();
        if age > VERIFICATION_TTL {
            trace!(
                "Near.AI verifier: cached model={} is stale (age={:.0}s, ttl={:.0}s)",
                model,
                age.as_secs_f64(),
                VERIFICATION_TTL.as_secs_f64()
            );
            return None;
        }

        if verified.nodes.is_empty() {
            trace!("Near.AI verifier: cached model={} has no nodes", model);
            return None;
        }

        let idx = random_node_index(verified.nodes.len());
        let node = &verified.nodes[idx];
        trace!(
            "Near.AI verifier: selecting node {}/{} for model={}, pubkey={}..., cache_age={:.0}s",
            idx + 1,
            verified.nodes.len(),
            model,
            &node.signing_public_key[..node.signing_public_key.len().min(16)],
            age.as_secs_f64()
        );
        Some(verified.nodes[idx].clone())
    }

    pub async fn invalidate_model(&self, model: &str) {
        let mut cache = self.cache.write().await;
        cache.remove(model);
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
        let mut total_nodes = 0usize;
        let mut ok_missing_pubkey = 0usize;
        let mut last_error: Option<String> = None;
        for model_att in &report.model_attestations {
            total_nodes += 1;
            match self
                .verify_single_attestation(model_att, &nonce_bytes, &nonce_hex, true)
                .await
            {
                Ok(Some(node)) => nodes.push(node),
                Ok(None) => {
                    ok_missing_pubkey += 1;
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    warn!(
                        "Near.AI model node attestation failed (model={}, signing_address={}): {}",
                        model, model_att.signing_address, e
                    );
                }
            }
        }

        if nodes.is_empty() {
            let mut msg = format!(
                "no verified model nodes available (total={}, ok_missing_pubkey={})",
                total_nodes, ok_missing_pubkey
            );
            if let Some(e) = last_error {
                msg.push_str(&format!(", last_error={e}"));
            }
            return Err(NearAiError::Attestation(msg));
        }

        debug!(
            "Near.AI verifier: model={} verified, {} nodes (total_attestations={}, ok_missing_pubkey={})",
            model, nodes.len(), total_nodes, ok_missing_pubkey
        );
        for (i, node) in nodes.iter().enumerate() {
            debug!(
                "Near.AI verifier: model={} node[{}] pubkey={}...",
                model,
                i,
                &node.signing_public_key[..node.signing_public_key.len().min(16)]
            );
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

        for attempt in 1..=HTTP_RETRY_ATTEMPTS {
            let mut req = self.http.get(&url).query(&[
                ("model", model),
                ("nonce", nonce_hex),
                ("signing_algo", "ecdsa"),
            ]);

            // Attestation report may be public, but include auth when present.
            if !api_key.trim().is_empty() {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }

            let res = match req.send().await {
                Ok(res) => res,
                Err(e) => {
                    if attempt < HTTP_RETRY_ATTEMPTS && (e.is_timeout() || e.is_connect()) {
                        warn!(
                            "Near.AI attestation report request failed (attempt {}/{}): {}; retrying",
                            attempt, HTTP_RETRY_ATTEMPTS, e
                        );
                        tokio::time::sleep(HTTP_RETRY_BASE_DELAY * attempt as u32).await;
                        continue;
                    }
                    return Err(NearAiError::Http(e));
                }
            };

            let status = res.status();
            let text = match res.text().await {
                Ok(t) => t,
                Err(e) => {
                    if attempt < HTTP_RETRY_ATTEMPTS && (e.is_timeout() || e.is_connect()) {
                        warn!(
                            "Near.AI attestation report read failed (attempt {}/{}): {}; retrying",
                            attempt, HTTP_RETRY_ATTEMPTS, e
                        );
                        tokio::time::sleep(HTTP_RETRY_BASE_DELAY * attempt as u32).await;
                        continue;
                    }
                    return Err(NearAiError::Http(e));
                }
            };

            if status.is_server_error() && attempt < HTTP_RETRY_ATTEMPTS {
                warn!(
                    "Near.AI attestation report returned {} (attempt {}/{}); retrying",
                    status, attempt, HTTP_RETRY_ATTEMPTS
                );
                tokio::time::sleep(HTTP_RETRY_BASE_DELAY * attempt as u32).await;
                continue;
            }

            if !status.is_success() {
                return Err(NearAiError::HttpStatus {
                    url,
                    status,
                    body: text,
                });
            }

            return Ok(serde_json::from_str(&text)?);
        }

        unreachable!("attestation report retry loop exhausted without returning")
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

        let quote = retry_verify_tdx_quote(&att.intel_quote).await?;
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
                retry_verify_gpu_attestation(&self.http, &jwks, &payload_json, nonce_hex).await
            {
                match e {
                    NearAiError::JwtKidNotFound(_) => {
                        let jwks = self.refresh_jwks().await?;
                        retry_verify_gpu_attestation(&self.http, &jwks, &payload_json, nonce_hex)
                            .await?;
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
                NRAS_JWKS_URL
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

fn is_retryable_error(e: &NearAiError) -> bool {
    match e {
        NearAiError::Http(re) => re.is_timeout() || re.is_connect(),
        NearAiError::Tdx(msg) | NearAiError::Nras(msg) => {
            let m = msg.to_lowercase();
            m.contains("timeout")
                || m.contains("connect")
                || m.contains("error sending request")
                || m.contains("connection")
        }
        _ => false,
    }
}

async fn retry_verify_tdx_quote(
    intel_quote_hex: &str,
) -> Result<crate::nearai::attestation::VerifiedTdxQuote, NearAiError> {
    for attempt in 1..=HTTP_RETRY_ATTEMPTS {
        match verify_tdx_quote(intel_quote_hex).await {
            Ok(v) => return Ok(v),
            Err(e) => {
                if attempt < HTTP_RETRY_ATTEMPTS && is_retryable_error(&e) {
                    warn!(
                        "TDX quote verification failed (attempt {}/{}): {}; retrying",
                        attempt, HTTP_RETRY_ATTEMPTS, e
                    );
                    tokio::time::sleep(HTTP_RETRY_BASE_DELAY * attempt as u32).await;
                    continue;
                }
                return Err(e);
            }
        }
    }
    unreachable!("retry loop exhausted without returning")
}

async fn retry_verify_gpu_attestation(
    client: &reqwest::Client,
    jwks: &NrasJwks,
    nvidia_payload: &Value,
    nonce_hex: &str,
) -> Result<(), NearAiError> {
    for attempt in 1..=HTTP_RETRY_ATTEMPTS {
        match verify_gpu_attestation(client, jwks, nvidia_payload, nonce_hex).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                if attempt < HTTP_RETRY_ATTEMPTS && is_retryable_error(&e) {
                    warn!(
                        "GPU attestation verification failed (attempt {}/{}): {}; retrying",
                        attempt, HTTP_RETRY_ATTEMPTS, e
                    );
                    tokio::time::sleep(HTTP_RETRY_BASE_DELAY * attempt as u32).await;
                    continue;
                }
                return Err(e);
            }
        }
    }
    unreachable!("retry loop exhausted without returning")
}

fn generate_nonce_bytes() -> Result<[u8; 32], NearAiError> {
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce)
        .map_err(|e| NearAiError::Crypto(format!("nonce generation failed: {e}")))?;
    Ok(nonce)
}

fn normalize_base_url(input: &str) -> String {
    let trimmed = input.trim_end_matches('/');
    let trimmed = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
    trimmed.to_string()
}

fn random_node_index(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }

    let mut bytes = [0u8; 8];
    if getrandom::getrandom(&mut bytes).is_ok() {
        (u64::from_le_bytes(bytes) as usize) % len
    } else {
        // If randomness is unavailable, fall back to the first verified node.
        0
    }
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
    async fn test_cached_node_selects_from_verified_nodes() {
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
        assert!(
            n1.signing_public_key == "pk1" || n1.signing_public_key == "pk2",
            "unexpected node pubkey: {}",
            n1.signing_public_key
        );
        assert!(
            n2.signing_public_key == "pk1" || n2.signing_public_key == "pk2",
            "unexpected node pubkey: {}",
            n2.signing_public_key
        );
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

        let message = &resp["choices"][0]["message"];

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let reasoning_content = message
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let reasoning = message
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if content.trim().is_empty()
            && reasoning_content.trim().is_empty()
            && reasoning.trim().is_empty()
        {
            eprintln!("Near.AI decrypted response had no textual fields: {resp}");
        }

        assert!(
            !content.trim().is_empty()
                || !reasoning_content.trim().is_empty()
                || !reasoning.trim().is_empty()
        );
    }

    /// Test that replicates the exact app flow: serde_json::to_string serialization,
    /// pre-serialized body (like hyper does), extra headers appended the same way try_provider does.
    #[tokio::test]
    #[ignore]
    async fn live_nearai_chat_completion_hyper_flow() {
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

        // Build body exactly how the app does: start with user body, replace model name,
        // add stream_options
        let mut body_map = serde_json::Map::new();
        body_map.insert("model".to_string(), json!("zai-org/GLM-5-FP8"));
        body_map.insert(
            "messages".to_string(),
            json!([{"role": "user", "content": "ping"}]),
        );
        body_map.insert("max_tokens".to_string(), json!(16));
        body_map.insert("temperature".to_string(), json!(0.0));
        body_map.insert("stream".to_string(), json!(false));
        // App adds stream_options when streaming or tinfoil
        body_map.insert("stream_options".to_string(), json!({"include_usage": true}));

        let mut body_value = serde_json::Value::Object(body_map);

        let crypto = prepare_e2ee_request(&mut body_value, &node.signing_public_key)
            .expect("request encryption should succeed");

        // App uses serde_json::to_string, NOT reqwest .json()
        let body_json = serde_json::to_string(&body_value).unwrap();

        eprintln!("body_json len={}", body_json.len());
        eprintln!("body_json={}", body_json);
        eprintln!("x-model-pub-key={}", node.signing_public_key);
        eprintln!("x-client-pub-key={}", crypto.client_public_key_hex);

        // Use reqwest but send pre-serialized body (like hyper does)
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("x-signing-algo", "ecdsa")
            .header("x-client-pub-key", &crypto.client_public_key_hex)
            .header("x-model-pub-key", &node.signing_public_key)
            .body(body_json.clone())
            .send()
            .await
            .expect("request should send");

        let status = res.status();
        let resp_text = res.text().await.unwrap();
        eprintln!("status={} body={}", status, resp_text);

        assert!(status.is_success(), "status={} body={}", status, resp_text);

        let mut resp: serde_json::Value =
            serde_json::from_str(&resp_text).expect("response should be JSON");
        decrypt_chat_completion_json_in_place(&mut resp, &crypto).expect("response should decrypt");

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        let reasoning_content = resp["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap_or("");
        let reasoning = resp["choices"][0]["message"]["reasoning"]
            .as_str()
            .unwrap_or("");

        assert!(
            !content.trim().is_empty()
                || !reasoning_content.trim().is_empty()
                || !reasoning.trim().is_empty(),
            "decrypted response should have some text"
        );
    }

    /// Test with streaming=true (how the responses API actually calls it)
    #[tokio::test]
    #[ignore]
    async fn live_nearai_chat_completion_streaming() {
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

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".to_string(), json!("zai-org/GLM-5-FP8"));
        body_map.insert(
            "messages".to_string(),
            json!([{"role": "user", "content": "Say hello"}]),
        );
        body_map.insert("max_tokens".to_string(), json!(32));
        body_map.insert("temperature".to_string(), json!(0.0));
        body_map.insert("stream".to_string(), json!(true));
        body_map.insert("stream_options".to_string(), json!({"include_usage": true}));

        let mut body_value = serde_json::Value::Object(body_map);

        let crypto = prepare_e2ee_request(&mut body_value, &node.signing_public_key)
            .expect("request encryption should succeed");

        let body_json = serde_json::to_string(&body_value).unwrap();
        eprintln!("streaming body_json len={}", body_json.len());

        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("x-signing-algo", "ecdsa")
            .header("x-client-pub-key", &crypto.client_public_key_hex)
            .header("x-model-pub-key", &node.signing_public_key)
            .body(body_json)
            .send()
            .await
            .expect("request should send");

        let status = res.status();
        let resp_text = res.text().await.unwrap();
        eprintln!("streaming status={}", status);

        if !status.is_success() {
            eprintln!("streaming response body={}", resp_text);
            panic!(
                "streaming request failed: status={} body={}",
                status, resp_text
            );
        }

        // Parse SSE chunks
        let mut got_content = false;
        for line in resp_text.lines() {
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line["data: ".len()..];
            if data == "[DONE]" {
                break;
            }
            let mut chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("failed to parse SSE chunk: {} -- data: {}", e, data);
                    continue;
                }
            };

            if let Err(e) = decrypt_chat_completion_json_in_place(&mut chunk, &crypto) {
                eprintln!("failed to decrypt SSE chunk: {} -- data: {}", e, data);
                panic!("stream chunk decryption failed");
            }

            let delta = &chunk["choices"][0]["delta"];
            let delta_content = delta.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let delta_reasoning = delta
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let delta_reasoning2 = delta
                .get("reasoning")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !delta_content.is_empty()
                || !delta_reasoning.is_empty()
                || !delta_reasoning2.is_empty()
            {
                got_content = true;
            }
        }

        assert!(
            got_content,
            "streaming response should have produced some decrypted content"
        );
    }

    /// Test with multi-turn conversation (system + user messages, like the app sends)
    #[tokio::test]
    #[ignore]
    async fn live_nearai_chat_completion_multi_turn() {
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
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"}
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": false,
            "stream_options": {"include_usage": true}
        });

        let crypto = prepare_e2ee_request(&mut body, &node.signing_public_key)
            .expect("request encryption should succeed");

        let body_json = serde_json::to_string(&body).unwrap();
        eprintln!("multi-turn body_json len={}", body_json.len());

        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("x-signing-algo", "ecdsa")
            .header("x-client-pub-key", &crypto.client_public_key_hex)
            .header("x-model-pub-key", &node.signing_public_key)
            .body(body_json)
            .send()
            .await
            .expect("request should send");

        let status = res.status();
        let resp_text = res.text().await.unwrap();
        eprintln!("multi-turn status={} body={}", status, resp_text);

        assert!(
            status.is_success(),
            "multi-turn failed: status={} body={}",
            status,
            resp_text
        );

        let mut resp: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        decrypt_chat_completion_json_in_place(&mut resp, &crypto)
            .expect("multi-turn response should decrypt");
    }

    /// Repeated requests to catch intermittent failures
    #[tokio::test]
    #[ignore]
    async fn live_nearai_chat_completion_repeated() {
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

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

        for i in 0..10 {
            // Re-fetch verified node each iteration (like the app would on separate requests)
            let node = v
                .get_verified_model_node("zai-org/GLM-5-FP8")
                .await
                .expect("attestation verification should succeed");

            let mut body = json!({
                "model": "zai-org/GLM-5-FP8",
                "messages": [{"role": "user", "content": format!("Say the number {}", i)}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": false,
                "stream_options": {"include_usage": true}
            });

            let crypto = prepare_e2ee_request(&mut body, &node.signing_public_key)
                .expect("request encryption should succeed");

            let body_json = serde_json::to_string(&body).unwrap();

            let res = client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("x-signing-algo", "ecdsa")
                .header("x-client-pub-key", &crypto.client_public_key_hex)
                .header("x-model-pub-key", &node.signing_public_key)
                .body(body_json)
                .send()
                .await
                .expect("request should send");

            let status = res.status();
            let resp_text = res.text().await.unwrap();
            eprintln!(
                "iteration {} status={} node_pubkey={}... body_preview={}",
                i,
                status,
                &node.signing_public_key[..16],
                &resp_text[..resp_text.len().min(120)]
            );

            assert!(
                status.is_success(),
                "iteration {} failed: status={} body={}",
                i,
                status,
                resp_text
            );

            let mut resp: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
            decrypt_chat_completion_json_in_place(&mut resp, &crypto)
                .expect("response should decrypt");
        }
    }
}
