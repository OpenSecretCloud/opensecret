use crate::encrypt::generate_random;
use crate::models::push_devices::PushDevice;
use crate::push::{NotificationPreviewPayload, PushError};
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::{engine::general_purpose, Engine};
use hkdf::Hkdf;
use p256::ecdh::EphemeralSecret;
use p256::elliptic_curve::sec1::ToEncodedPoint;
use p256::pkcs8::DecodePublicKey;
use p256::PublicKey;
use serde::{Deserialize, Serialize};
use sha2::Sha256;

pub const PUSH_PREVIEW_INFO: &[u8] = b"opensecret-push-preview-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedPreviewEnvelope {
    #[serde(alias = "v")]
    pub enc_v: i32,
    pub alg: String,
    pub kid: String,
    pub epk: String,
    pub salt: String,
    pub nonce: String,
    pub ciphertext: String,
}

pub fn encrypt_preview_payload(
    device: &PushDevice,
    payload: &NotificationPreviewPayload,
) -> Result<EncryptedPreviewEnvelope, PushError> {
    let recipient_key = PublicKey::from_public_key_der(&device.notification_public_key)
        .map_err(|e| PushError::CryptoError(e.to_string()))?;
    let ephemeral_secret = EphemeralSecret::random(&mut p256::elliptic_curve::rand_core::OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);
    let shared_secret = ephemeral_secret.diffie_hellman(&recipient_key);

    let salt = generate_random::<32>();
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), shared_secret.raw_secret_bytes().as_slice());
    let mut key = [0_u8; 32];
    hkdf.expand(PUSH_PREVIEW_INFO, &mut key)
        .map_err(|_| PushError::CryptoError("failed to derive push preview key".to_string()))?;

    let nonce_bytes = generate_random::<12>();
    let cipher =
        Aes256Gcm::new_from_slice(&key).map_err(|e| PushError::CryptoError(e.to_string()))?;
    let plaintext = serde_json::to_vec(payload)?;
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce_bytes), plaintext.as_ref())
        .map_err(|e| PushError::CryptoError(e.to_string()))?;

    Ok(EncryptedPreviewEnvelope {
        enc_v: 1,
        alg: "p256-hkdf-sha256-aes256gcm".to_string(),
        kid: device.uuid.to_string(),
        epk: general_purpose::STANDARD.encode(ephemeral_public.to_encoded_point(false).as_bytes()),
        salt: general_purpose::STANDARD.encode(salt),
        nonce: general_purpose::STANDARD.encode(nonce_bytes),
        ciphertext: general_purpose::STANDARD.encode(ciphertext),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::push_devices::{
        PushDevice, PUSH_ENV_PROD, PUSH_KEY_ALGORITHM_P256_ECDH_V1, PUSH_PLATFORM_IOS,
        PUSH_PROVIDER_APNS,
    };
    use aes_gcm::aead::{Aead, KeyInit};
    use aes_gcm::{Aes256Gcm, Nonce};
    use base64::{engine::general_purpose, Engine};
    use chrono::Utc;
    use hkdf::Hkdf;
    use p256::elliptic_curve::rand_core::OsRng;
    use p256::pkcs8::EncodePublicKey;
    use p256::SecretKey;
    use uuid::Uuid;

    #[test]
    fn encrypt_preview_payload_round_trips() {
        let recipient_secret = SecretKey::random(&mut OsRng);
        let recipient_public = recipient_secret.public_key();
        let device = PushDevice {
            id: 1,
            uuid: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            installation_id: Uuid::new_v4(),
            platform: PUSH_PLATFORM_IOS.to_string(),
            provider: PUSH_PROVIDER_APNS.to_string(),
            environment: PUSH_ENV_PROD.to_string(),
            app_id: "ai.trymaple.ios".to_string(),
            push_token_enc: vec![1, 2, 3],
            push_token_hash: vec![4, 5, 6],
            notification_public_key: recipient_public
                .to_public_key_der()
                .unwrap()
                .as_bytes()
                .to_vec(),
            key_algorithm: PUSH_KEY_ALGORITHM_P256_ECDH_V1.to_string(),
            supports_encrypted_preview: true,
            supports_background_processing: true,
            last_seen_at: Utc::now(),
            revoked_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let payload = NotificationPreviewPayload {
            v: 1,
            notification_id: Uuid::new_v4(),
            message_id: Uuid::new_v4(),
            kind: "agent.message".to_string(),
            title: "Maple".to_string(),
            body: "Hello from Maple".to_string(),
            deep_link: "opensecret://agent".to_string(),
            thread_id: "agent:main".to_string(),
            sent_at: Utc::now().timestamp(),
        };

        let envelope = encrypt_preview_payload(&device, &payload).unwrap();
        let epk_bytes = general_purpose::STANDARD.decode(&envelope.epk).unwrap();
        let salt = general_purpose::STANDARD.decode(&envelope.salt).unwrap();
        let nonce = general_purpose::STANDARD.decode(&envelope.nonce).unwrap();
        let ciphertext = general_purpose::STANDARD
            .decode(&envelope.ciphertext)
            .unwrap();

        let ephemeral_public = PublicKey::from_sec1_bytes(&epk_bytes).unwrap();
        let shared_secret = p256::ecdh::diffie_hellman(
            recipient_secret.to_nonzero_scalar(),
            ephemeral_public.as_affine(),
        );
        let hkdf = Hkdf::<Sha256>::new(Some(&salt), shared_secret.raw_secret_bytes().as_slice());
        let mut key = [0_u8; 32];
        hkdf.expand(PUSH_PREVIEW_INFO, &mut key).unwrap();

        let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
        let plaintext = cipher
            .decrypt(Nonce::from_slice(&nonce), ciphertext.as_ref())
            .unwrap();
        let decoded: NotificationPreviewPayload = serde_json::from_slice(&plaintext).unwrap();

        assert_eq!(decoded.notification_id, payload.notification_id);
        assert_eq!(decoded.body, payload.body);
        assert_eq!(decoded.deep_link, payload.deep_link);
        assert_eq!(envelope.enc_v, 1);
        assert_eq!(envelope.alg, "p256-hkdf-sha256-aes256gcm");
    }
}
