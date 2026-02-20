use crate::nearai::attestation::secp256k1_pubkey_from_64hex;
use crate::nearai::error::NearAiError;
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use hkdf::Hkdf;
use secp256k1::ecdh::shared_secret_point;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use serde_json::Value;
use sha2::Sha256;
use tracing::{debug, trace, warn};

const HKDF_INFO: &[u8] = b"ecdsa_encryption";
const MIN_CIPHERTEXT_BYTES: usize = 65 + 12 + 16;

#[derive(Clone)]
pub struct NearAiResponseCrypto {
    pub client_public_key_hex: String,
    client_secret_key: SecretKey,
}

impl NearAiResponseCrypto {
    fn decrypt_field(&self, encrypted_hex: &str) -> Result<String, NearAiError> {
        let decrypted = decrypt_ecies_hex(encrypted_hex, &self.client_secret_key)?;
        String::from_utf8(decrypted)
            .map_err(|e| NearAiError::Crypto(format!("invalid UTF-8 plaintext: {e}")))
    }
}

pub fn prepare_e2ee_request(
    body: &mut Value,
    model_public_key_hex: &str,
) -> Result<NearAiResponseCrypto, NearAiError> {
    trace!(
        "Near.AI E2EE prepare: model_pubkey_hex len={} first16={}",
        model_public_key_hex.len(),
        &model_public_key_hex[..model_public_key_hex.len().min(16)]
    );
    let model_pubkey = secp256k1_pubkey_from_64hex(model_public_key_hex)?;

    let secp = Secp256k1::new();
    let client_secret_key = generate_secp256k1_secret_key()?;
    let client_pubkey = PublicKey::from_secret_key(&secp, &client_secret_key);
    let client_pubkey_uncompressed = client_pubkey.serialize_uncompressed();
    let client_public_key_hex = hex::encode(&client_pubkey_uncompressed[1..]); // strip 0x04

    trace!(
        "Near.AI E2EE prepare: client_pubkey_hex len={} first16={}",
        client_public_key_hex.len(),
        &client_public_key_hex[..client_public_key_hex.len().min(16)]
    );

    // Encrypt messages[].content (string only). Non-string content is forwarded plaintext.
    let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return Err(NearAiError::Crypto("missing messages array".to_string()));
    };

    let mut encrypted_count = 0usize;
    let mut skipped_count = 0usize;
    for (i, msg) in messages.iter_mut().enumerate() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        let role = obj
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let Some(content_val) = obj.get_mut("content") else {
            continue;
        };

        // Flatten text-only array content to a plain string before encryption.
        // The responses API wraps text in [{"type":"text","text":"..."}] format,
        // but Near.AI E2EE expects content to be a string.
        if let Some(flattened) = try_flatten_text_content_array(content_val) {
            trace!(
                "Near.AI E2EE: flattened text-array content to string for messages[{}] role={}",
                i,
                role
            );
            *content_val = Value::String(flattened);
        }

        match content_val {
            Value::String(s) => {
                let plaintext_len = s.len();
                let plaintext = s.clone();
                let encrypted_hex = encrypt_ecies_hex(plaintext.as_bytes(), &model_pubkey)?;
                trace!(
                    "Near.AI E2EE: encrypted messages[{}] role={} plaintext_len={} ciphertext_hex_len={}",
                    i,
                    role,
                    plaintext_len,
                    encrypted_hex.len()
                );
                *content_val = Value::String(encrypted_hex);
                encrypted_count += 1;
            }
            Value::Null => {
                skipped_count += 1;
            }
            _ => {
                // Genuinely multimodal content (images, etc.) -- cannot flatten to string.
                warn!(
                    "Near.AI E2EE: leaving non-string messages[{}].content (role={}) in plaintext (v1 limitation)",
                    i, role
                );
                skipped_count += 1;
            }
        }
    }

    debug!(
        "Near.AI E2EE prepare: encrypted={} skipped={} total={}",
        encrypted_count,
        skipped_count,
        encrypted_count + skipped_count
    );

    if body.get("tools").is_some()
        || body.get("tool_choice").is_some()
        || body.get("functions").is_some()
        || body.get("function_call").is_some()
    {
        // v1 limitation: tool schemas and tool-call arguments are forwarded plaintext.
        warn!("Near.AI E2EE: leaving tool-related fields in plaintext (v1 limitation)");
    }

    Ok(NearAiResponseCrypto {
        client_public_key_hex,
        client_secret_key,
    })
}

pub fn decrypt_chat_completion_json_in_place(
    json: &mut Value,
    crypto: &NearAiResponseCrypto,
) -> Result<(), NearAiError> {
    let Some(choices) = json.get_mut("choices").and_then(|v| v.as_array_mut()) else {
        trace!("Near.AI E2EE decrypt: no choices array in response, skipping");
        return Ok(());
    };

    trace!("Near.AI E2EE decrypt: processing {} choices", choices.len());

    for (i, choice) in choices.iter_mut().enumerate() {
        let Some(choice_obj) = choice.as_object_mut() else {
            continue;
        };

        if let Some(msg) = choice_obj.get_mut("message") {
            trace!("Near.AI E2EE decrypt: decrypting choices[{}].message", i);
            decrypt_message_like_in_place(msg, crypto)?;
        }

        if let Some(delta) = choice_obj.get_mut("delta") {
            trace!("Near.AI E2EE decrypt: decrypting choices[{}].delta", i);
            decrypt_message_like_in_place(delta, crypto)?;
        }
    }

    Ok(())
}

fn decrypt_message_like_in_place(
    val: &mut Value,
    crypto: &NearAiResponseCrypto,
) -> Result<(), NearAiError> {
    let Some(obj) = val.as_object_mut() else {
        return Ok(());
    };

    for field in ["content", "reasoning_content", "reasoning"] {
        if let Some(v) = obj.get_mut(field) {
            decrypt_string_field_in_place(v, crypto)?;
        }
    }

    Ok(())
}

fn decrypt_string_field_in_place(
    val: &mut Value,
    crypto: &NearAiResponseCrypto,
) -> Result<(), NearAiError> {
    match val {
        Value::Null => Ok(()),
        Value::String(s) => {
            if s.is_empty() {
                return Ok(());
            }
            // Strict: must be valid hex ciphertext and must decrypt.
            if !looks_like_hex_ciphertext(s) {
                debug!(
                    "Near.AI E2EE decrypt: field is not hex ciphertext, len={} first32='{}'",
                    s.len(),
                    &s[..s.len().min(32)]
                );
                return Err(NearAiError::Crypto(
                    "Near.AI response field did not look like hex ciphertext".to_string(),
                ));
            }
            trace!(
                "Near.AI E2EE decrypt: decrypting field, ciphertext_hex_len={}",
                s.len()
            );
            let plaintext = crypto.decrypt_field(s)?;
            trace!(
                "Near.AI E2EE decrypt: decrypted OK, plaintext_len={}",
                plaintext.len()
            );
            *val = Value::String(plaintext);
            Ok(())
        }
        _ => {
            trace!("Near.AI E2EE decrypt: unexpected non-string field type");
            Err(NearAiError::Crypto(
                "Near.AI response field had unexpected non-string type".to_string(),
            ))
        }
    }
}

/// If content is an array where every element is `{"type":"text","text":"..."}`,
/// concatenate the text parts into a single string. Returns None if any element
/// is not a text part (e.g. images), meaning the content is genuinely multimodal.
fn try_flatten_text_content_array(val: &Value) -> Option<String> {
    let arr = val.as_array()?;
    if arr.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    for item in arr {
        let obj = item.as_object()?;
        let typ = obj.get("type").and_then(|v| v.as_str())?;
        if typ != "text" && typ != "input_text" {
            return None;
        }
        let text = obj.get("text").and_then(|v| v.as_str())?;
        parts.push(text);
    }

    Some(parts.join("\n"))
}

fn looks_like_hex_ciphertext(s: &str) -> bool {
    if s.len() < (MIN_CIPHERTEXT_BYTES * 2) {
        return false;
    }
    if !s.len().is_multiple_of(2) {
        return false;
    }
    s.as_bytes().iter().all(|b| b.is_ascii_hexdigit())
}

fn encrypt_ecies_hex(
    plaintext: &[u8],
    recipient_pubkey: &PublicKey,
) -> Result<String, NearAiError> {
    let secp = Secp256k1::new();
    let ephemeral_secret_key = generate_secp256k1_secret_key()?;
    let ephemeral_pubkey = PublicKey::from_secret_key(&secp, &ephemeral_secret_key);
    let ephemeral_pubkey_uncompressed = ephemeral_pubkey.serialize_uncompressed();

    let shared_point = shared_secret_point(recipient_pubkey, &ephemeral_secret_key);
    let shared_x = &shared_point[..32];

    let hk = Hkdf::<Sha256>::new(None, shared_x);
    let mut aes_key = [0u8; 32];
    hk.expand(HKDF_INFO, &mut aes_key)?;

    let cipher = Aes256Gcm::new_from_slice(&aes_key)?;
    let mut nonce_bytes = [0u8; 12];
    getrandom::getrandom(&mut nonce_bytes)
        .map_err(|e| NearAiError::Crypto(format!("nonce generation failed: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext)?;

    let mut out = Vec::with_capacity(
        ephemeral_pubkey_uncompressed.len() + nonce_bytes.len() + ciphertext.len(),
    );
    out.extend_from_slice(&ephemeral_pubkey_uncompressed);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(hex::encode(out))
}

fn decrypt_ecies_hex(
    ciphertext_hex: &str,
    recipient_secret_key: &SecretKey,
) -> Result<Vec<u8>, NearAiError> {
    let ciphertext = hex::decode(ciphertext_hex.trim())?;
    if ciphertext.len() < MIN_CIPHERTEXT_BYTES {
        return Err(NearAiError::Crypto("ciphertext too short".to_string()));
    }

    let ephemeral_pubkey = PublicKey::from_slice(&ciphertext[..65])?;
    let nonce_bytes = &ciphertext[65..77];
    let body = &ciphertext[77..];

    let shared_point = shared_secret_point(&ephemeral_pubkey, recipient_secret_key);
    let shared_x = &shared_point[..32];

    let hk = Hkdf::<Sha256>::new(None, shared_x);
    let mut aes_key = [0u8; 32];
    hk.expand(HKDF_INFO, &mut aes_key)?;

    let cipher = Aes256Gcm::new_from_slice(&aes_key)?;
    let nonce = Nonce::from_slice(nonce_bytes);
    Ok(cipher.decrypt(nonce, body)?)
}

fn generate_secp256k1_secret_key() -> Result<SecretKey, NearAiError> {
    let mut sk_bytes = [0u8; 32];
    loop {
        getrandom::getrandom(&mut sk_bytes)
            .map_err(|e| NearAiError::Crypto(format!("secret key generation failed: {e}")))?;
        if let Ok(sk) = SecretKey::from_slice(&sk_bytes) {
            return Ok(sk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixed_secret_key(last_byte: u8) -> SecretKey {
        let mut sk = [0u8; 32];
        sk[31] = last_byte;
        SecretKey::from_slice(&sk).unwrap()
    }

    #[test]
    fn test_ecies_hex_roundtrip() {
        let secp = Secp256k1::new();
        let recipient_sk = fixed_secret_key(2);
        let recipient_pk = PublicKey::from_secret_key(&secp, &recipient_sk);

        let plaintext = b"hello world";
        let ciphertext_hex = encrypt_ecies_hex(plaintext, &recipient_pk).unwrap();
        assert!(looks_like_hex_ciphertext(&ciphertext_hex));

        let decrypted = decrypt_ecies_hex(&ciphertext_hex, &recipient_sk).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_prepare_e2ee_request_encrypts_string_content_only() {
        let secp = Secp256k1::new();
        let model_sk = fixed_secret_key(3);
        let model_pk = PublicKey::from_secret_key(&secp, &model_sk);
        let model_pk_uncompressed = model_pk.serialize_uncompressed();
        let model_pubkey_hex = hex::encode(&model_pk_uncompressed[1..]);

        // Genuinely multimodal: has an image part, so cannot be flattened
        let multimodal = json!([
            {"type": "input_text", "text": "this is plaintext"},
            {"type": "input_image", "image_url": "https://example.com/foo.png"}
        ]);

        let mut body = json!({
            "model": "zai-org/GLM-5-FP8",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": multimodal.clone()},
                {"role": "user", "content": ""}
            ],
            "tools": [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
        });

        let crypto = prepare_e2ee_request(&mut body, &model_pubkey_hex).unwrap();

        let c0 = body["messages"][0]["content"].as_str().unwrap();
        assert_ne!(c0, "hello");
        assert!(looks_like_hex_ciphertext(c0));

        // Genuinely multimodal content stays as array (can't flatten)
        assert_eq!(body["messages"][1]["content"], multimodal);
        let c2 = body["messages"][2]["content"].as_str().unwrap();
        assert_ne!(c2, "");
        assert!(looks_like_hex_ciphertext(c2));

        assert_eq!(crypto.client_public_key_hex.len(), 128);
        assert!(crypto
            .client_public_key_hex
            .as_bytes()
            .iter()
            .all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn test_prepare_e2ee_request_flattens_text_only_array_content() {
        let secp = Secp256k1::new();
        let model_sk = fixed_secret_key(3);
        let model_pk = PublicKey::from_secret_key(&secp, &model_sk);
        let model_pk_uncompressed = model_pk.serialize_uncompressed();
        let model_pubkey_hex = hex::encode(&model_pk_uncompressed[1..]);

        // This is the format the responses API sends: text wrapped in array
        let mut body = json!({
            "model": "zai-org/GLM-5-FP8",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hey"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "also text"}]}
            ]
        });

        prepare_e2ee_request(&mut body, &model_pubkey_hex).unwrap();

        // Both should have been flattened to strings and then encrypted
        let c0 = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            looks_like_hex_ciphertext(c0),
            "text-array content should be flattened and encrypted, got: {}",
            &c0[..c0.len().min(40)]
        );

        let c1 = body["messages"][1]["content"].as_str().unwrap();
        assert!(
            looks_like_hex_ciphertext(c1),
            "input_text-array content should be flattened and encrypted, got: {}",
            &c1[..c1.len().min(40)]
        );
    }

    #[test]
    fn test_try_flatten_text_content_array() {
        // Single text part
        let val = json!([{"type": "text", "text": "hello"}]);
        assert_eq!(
            try_flatten_text_content_array(&val),
            Some("hello".to_string())
        );

        // Multiple text parts
        let val = json!([
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"}
        ]);
        assert_eq!(
            try_flatten_text_content_array(&val),
            Some("hello\nworld".to_string())
        );

        // input_text type also works
        let val = json!([{"type": "input_text", "text": "hey"}]);
        assert_eq!(
            try_flatten_text_content_array(&val),
            Some("hey".to_string())
        );

        // Mixed with image -- not flattenable
        let val = json!([
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]);
        assert_eq!(try_flatten_text_content_array(&val), None);

        // String content -- not an array
        let val = json!("hello");
        assert_eq!(try_flatten_text_content_array(&val), None);

        // Empty array
        let val = json!([]);
        assert_eq!(try_flatten_text_content_array(&val), None);

        // Null
        let val = json!(null);
        assert_eq!(try_flatten_text_content_array(&val), None);
    }

    #[test]
    fn test_decrypt_chat_completion_json_in_place_decrypts_fields() {
        let secp = Secp256k1::new();
        let model_sk = fixed_secret_key(4);
        let model_pk = PublicKey::from_secret_key(&secp, &model_sk);
        let model_pk_uncompressed = model_pk.serialize_uncompressed();
        let model_pubkey_hex = hex::encode(&model_pk_uncompressed[1..]);

        let mut req = json!({
            "model": "zai-org/GLM-5-FP8",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let crypto = prepare_e2ee_request(&mut req, &model_pubkey_hex).unwrap();

        let client_pk = PublicKey::from_secret_key(&secp, &crypto.client_secret_key);
        let enc_content = encrypt_ecies_hex(b"hi", &client_pk).unwrap();
        let enc_reasoning = encrypt_ecies_hex(b"because", &client_pk).unwrap();

        let mut resp = json!({
            "choices": [{
                "message": {"content": enc_content.clone(), "reasoning_content": enc_reasoning},
                "delta": {"content": enc_content}
            }]
        });

        decrypt_chat_completion_json_in_place(&mut resp, &crypto).unwrap();

        assert_eq!(resp["choices"][0]["message"]["content"], "hi");
        assert_eq!(
            resp["choices"][0]["message"]["reasoning_content"],
            "because"
        );
        assert_eq!(resp["choices"][0]["delta"]["content"], "hi");
    }

    #[test]
    fn test_decrypt_rejects_non_ciphertext_strings() {
        let secp = Secp256k1::new();
        let model_sk = fixed_secret_key(5);
        let model_pk = PublicKey::from_secret_key(&secp, &model_sk);
        let model_pk_uncompressed = model_pk.serialize_uncompressed();
        let model_pubkey_hex = hex::encode(&model_pk_uncompressed[1..]);

        let mut req = json!({
            "model": "zai-org/GLM-5-FP8",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let crypto = prepare_e2ee_request(&mut req, &model_pubkey_hex).unwrap();

        let mut resp = json!({
            "choices": [{"message": {"content": "plaintext"}}]
        });

        assert!(decrypt_chat_completion_json_in_place(&mut resp, &crypto).is_err());
    }
}
