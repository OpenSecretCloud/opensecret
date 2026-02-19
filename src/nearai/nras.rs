use crate::nearai::error::NearAiError;
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use serde_json::Value;

pub const NRAS_GPU_VERIFIER_URL: &str = "https://nras.attestation.nvidia.com/v3/attest/gpu";
pub const NRAS_JWKS_URL: &str = "https://nras.attestation.nvidia.com/.well-known/jwks.json";
pub const NRAS_ISSUER: &str = "nras.attestation.nvidia.com";

#[derive(Debug, Clone, Deserialize)]
pub struct NrasJwks {
    pub keys: Vec<NrasJwk>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NrasJwk {
    #[serde(default)]
    pub kid: Option<String>,

    #[serde(default)]
    pub x: Option<String>,

    #[serde(default)]
    pub y: Option<String>,
}

pub async fn fetch_jwks(client: &reqwest::Client) -> Result<NrasJwks, NearAiError> {
    let res = client.get(NRAS_JWKS_URL).send().await?;
    let status = res.status();
    let text = res.text().await?;
    if !status.is_success() {
        return Err(NearAiError::HttpStatus {
            url: NRAS_JWKS_URL.to_string(),
            status,
            body: text,
        });
    }

    Ok(serde_json::from_str(&text)?)
}

pub async fn verify_gpu_attestation(
    client: &reqwest::Client,
    jwks: &NrasJwks,
    nvidia_payload: &Value,
    request_nonce_hex: &str,
) -> Result<(), NearAiError> {
    let payload_nonce = nvidia_payload
        .get("nonce")
        .and_then(|v| v.as_str())
        .ok_or_else(|| NearAiError::Nras("missing nvidia_payload.nonce".to_string()))?;

    if payload_nonce.to_lowercase() != request_nonce_hex.to_lowercase() {
        return Err(NearAiError::Nras(
            "GPU payload nonce does not match request_nonce".to_string(),
        ));
    }

    let res = client
        .post(NRAS_GPU_VERIFIER_URL)
        .header("Accept", "application/json")
        .json(nvidia_payload)
        .send()
        .await?;

    let status = res.status();
    let text = res.text().await?;
    if !status.is_success() {
        return Err(NearAiError::HttpStatus {
            url: NRAS_GPU_VERIFIER_URL.to_string(),
            status,
            body: text,
        });
    }

    let body: Value = serde_json::from_str(&text)?;
    let jwt = extract_jwt_from_nras_response(&body)
        .ok_or_else(|| NearAiError::Nras("unexpected NRAS response format".to_string()))?;

    verify_nras_jwt(&jwt, jwks, request_nonce_hex)
}

fn extract_jwt_from_nras_response(body: &Value) -> Option<String> {
    let arr = body.as_array()?;

    for item in arr {
        let Some(pair) = item.as_array() else {
            continue;
        };
        if pair.len() < 2 {
            continue;
        }
        if let Some(jwt) = pair[1].as_str() {
            return Some(jwt.to_string());
        }
    }
    None
}

pub fn verify_nras_jwt(
    jwt: &str,
    jwks: &NrasJwks,
    request_nonce_hex: &str,
) -> Result<(), NearAiError> {
    let header = decode_header(jwt)?;
    let kid = header
        .kid
        .ok_or_else(|| NearAiError::Jwt("NRAS JWT missing kid".to_string()))?;

    let jwk = jwks
        .keys
        .iter()
        .find(|k| k.kid.as_deref() == Some(kid.as_str()))
        .ok_or_else(|| NearAiError::JwtKidNotFound(kid.clone()))?;

    let x = jwk
        .x
        .as_ref()
        .ok_or_else(|| NearAiError::Jwt("NRAS JWK missing x".to_string()))?;
    let y = jwk
        .y
        .as_ref()
        .ok_or_else(|| NearAiError::Jwt("NRAS JWK missing y".to_string()))?;

    let key = DecodingKey::from_ec_components(x, y)?;

    let mut validation = Validation::new(Algorithm::ES384);
    validation.set_issuer(&[NRAS_ISSUER]);
    validation.validate_aud = false;
    validation.leeway = 60;

    let token = decode::<Value>(jwt, &key, &validation)?;
    let claims = token.claims;

    let verdict = claims
        .get("x-nvidia-overall-att-result")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !verdict {
        return Err(NearAiError::Nras(
            "NRAS attestation verdict is false".to_string(),
        ));
    }

    let nonce_ok = match claims.get("eat_nonce") {
        Some(Value::String(s)) => s.to_lowercase() == request_nonce_hex.to_lowercase(),
        Some(Value::Array(arr)) => arr.iter().any(|v| {
            v.as_str()
                .is_some_and(|s| s.to_lowercase() == request_nonce_hex.to_lowercase())
        }),
        _ => false,
    };

    if !nonce_ok {
        return Err(NearAiError::Nras(
            "NRAS eat_nonce does not match request_nonce".to_string(),
        ));
    }

    Ok(())
}
