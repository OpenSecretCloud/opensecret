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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use p384::ecdsa::SigningKey;
    use p384::pkcs8::EncodePrivateKey;
    use rand_core::OsRng;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_epoch_seconds() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_secs()
    }

    fn build_jwks_and_jwt(kid: &str, claims: &serde_json::Value) -> (NrasJwks, String) {
        let signing_key = SigningKey::random(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let pkcs8 = signing_key
            .to_pkcs8_der()
            .expect("pkcs8 encoding should succeed");
        let encoding_key = EncodingKey::from_ec_der(pkcs8.as_bytes());

        let mut header = Header::new(Algorithm::ES384);
        header.kid = Some(kid.to_string());

        let token = encode(&header, claims, &encoding_key).expect("JWT encode should succeed");

        let point = verifying_key.to_encoded_point(false);
        let bytes = point.as_bytes();
        assert_eq!(bytes.len(), 97);
        assert_eq!(bytes[0], 0x04);

        let x = &bytes[1..49];
        let y = &bytes[49..97];
        let x_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(x);
        let y_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(y);

        let jwks = NrasJwks {
            keys: vec![NrasJwk {
                kid: Some(kid.to_string()),
                x: Some(x_b64),
                y: Some(y_b64),
            }],
        };

        (jwks, token)
    }

    #[test]
    fn test_extract_jwt_from_nras_response() {
        let body = json!([[0, "jwt1"], [1, "jwt2"]]);
        assert_eq!(extract_jwt_from_nras_response(&body).unwrap(), "jwt1");

        let body = json!([[], [0, 1], [0, "jwt3"]]);
        assert_eq!(extract_jwt_from_nras_response(&body).unwrap(), "jwt3");

        let body = json!({"not": "an array"});
        assert!(extract_jwt_from_nras_response(&body).is_none());
    }

    #[test]
    fn test_verify_nras_jwt_kid_not_found() {
        let header = json!({"alg": "ES384", "kid": "missing"});
        let payload = json!({"iss": NRAS_ISSUER, "exp": now_epoch_seconds() + 60});
        let token = format!(
            "{}.{}.sig",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(header.to_string()),
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.to_string())
        );

        let jwks = NrasJwks { keys: vec![] };
        let err = verify_nras_jwt(&token, &jwks, "nonce").unwrap_err();
        match err {
            NearAiError::JwtKidNotFound(k) => assert_eq!(k, "missing"),
            other => panic!("expected JwtKidNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_verify_nras_jwt_valid_string_nonce() {
        let nonce = "abcd1234";
        let now = now_epoch_seconds();
        let claims = json!({
            "iss": NRAS_ISSUER,
            "exp": now + 60,
            "iat": now,
            "x-nvidia-overall-att-result": true,
            "eat_nonce": nonce
        });

        let (jwks, token) = build_jwks_and_jwt("kid1", &claims);
        verify_nras_jwt(&token, &jwks, nonce).unwrap();
    }

    #[test]
    fn test_verify_nras_jwt_valid_array_nonce() {
        let nonce = "abcd1234";
        let now = now_epoch_seconds();
        let claims = json!({
            "iss": NRAS_ISSUER,
            "exp": now + 60,
            "iat": now,
            "x-nvidia-overall-att-result": true,
            "eat_nonce": ["zzz", nonce]
        });

        let (jwks, token) = build_jwks_and_jwt("kid1", &claims);
        verify_nras_jwt(&token, &jwks, nonce).unwrap();
    }

    #[test]
    fn test_verify_nras_jwt_nonce_mismatch_fails() {
        let now = now_epoch_seconds();
        let claims = json!({
            "iss": NRAS_ISSUER,
            "exp": now + 60,
            "iat": now,
            "x-nvidia-overall-att-result": true,
            "eat_nonce": "abcd1234"
        });

        let (jwks, token) = build_jwks_and_jwt("kid1", &claims);
        assert!(verify_nras_jwt(&token, &jwks, "different").is_err());
    }

    #[test]
    fn test_verify_nras_jwt_verdict_false_fails() {
        let nonce = "abcd1234";
        let now = now_epoch_seconds();
        let claims = json!({
            "iss": NRAS_ISSUER,
            "exp": now + 60,
            "iat": now,
            "x-nvidia-overall-att-result": false,
            "eat_nonce": nonce
        });

        let (jwks, token) = build_jwks_and_jwt("kid1", &claims);
        assert!(verify_nras_jwt(&token, &jwks, nonce).is_err());
    }
}
