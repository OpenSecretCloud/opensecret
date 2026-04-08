use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::agent_background::{decrypt_background_grant_v1, AgentBackgroundGrantPlaintextV1};
use crate::encrypt::{decrypt_aead_v1, derive_key, encrypt_aead_v1, CanonicalBytes, EncryptError};
use crate::models::agent_background_grants::AgentBackgroundGrant;
use crate::models::notification_events::NotificationEvent;
use crate::models::push_devices::PushDevice;

const PUSH_DEVICE_AEAD_INFO: &[u8] = b"os.push-device-capability-aead.v1";
const PUSH_DEVICE_AAD_DOMAIN: &str = "os.push-device-capability.v1";
const PUSH_EVENT_PAYLOAD_AEAD_INFO: &[u8] = b"os.push-event-payload-aead.v1";
const PUSH_EVENT_PAYLOAD_AAD_DOMAIN: &str = "os.push-event-payload.v1";

pub const PUSH_CAPABILITY_VERSION_V1: i16 = 1;
pub const PUSH_EVENT_PAYLOAD_VERSION_V1: i16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PushDeviceCapabilityPlaintextV1 {
    pub v: i16,
    pub push_device_uuid: Uuid,
    pub user_uuid: Uuid,
    pub project_id: i32,
    pub installation_id: Uuid,
    pub platform: String,
    pub provider: String,
    pub environment: String,
    pub app_id: String,
    pub key_algorithm: String,
    pub push_token: String,
    pub push_token_hash: Vec<u8>,
    pub notification_public_key: Vec<u8>,
    pub notification_public_key_hash: Vec<u8>,
    pub supports_encrypted_preview: bool,
    pub supports_background_processing: bool,
    pub registered_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PushDeviceCapabilityInput {
    pub push_device_uuid: Uuid,
    pub user_uuid: Uuid,
    pub project_id: i32,
    pub installation_id: Uuid,
    pub platform: String,
    pub provider: String,
    pub environment: String,
    pub app_id: String,
    pub key_algorithm: String,
    pub push_token: String,
    pub notification_public_key: Vec<u8>,
    pub supports_encrypted_preview: bool,
    pub supports_background_processing: bool,
}

impl PushDeviceCapabilityPlaintextV1 {
    pub fn new(input: PushDeviceCapabilityInput) -> Self {
        let push_token_hash = hash_bytes(input.push_token.as_bytes());
        let notification_public_key_hash = hash_bytes(&input.notification_public_key);
        Self {
            v: PUSH_CAPABILITY_VERSION_V1,
            push_device_uuid: input.push_device_uuid,
            user_uuid: input.user_uuid,
            project_id: input.project_id,
            installation_id: input.installation_id,
            platform: input.platform,
            provider: input.provider,
            environment: input.environment,
            app_id: input.app_id,
            key_algorithm: input.key_algorithm,
            push_token: input.push_token,
            push_token_hash,
            notification_public_key: input.notification_public_key,
            notification_public_key_hash,
            supports_encrypted_preview: input.supports_encrypted_preview,
            supports_background_processing: input.supports_background_processing,
            registered_at: Utc::now(),
        }
    }

    pub fn matches_device_row(&self, device: &PushDevice) -> bool {
        self.v == PUSH_CAPABILITY_VERSION_V1
            && self.push_device_uuid == device.uuid
            && self.user_uuid == device.user_id
            && self.project_id == device.project_id
            && self.installation_id == device.installation_id
            && self.platform == device.platform
            && self.provider == device.provider
            && self.environment == device.environment
            && self.app_id == device.app_id
            && self.key_algorithm == device.key_algorithm
            && self.push_token_hash == device.push_token_hash
            && self.notification_public_key_hash == device.notification_public_key_hash
            && self.supports_encrypted_preview == device.supports_encrypted_preview
            && self.supports_background_processing == device.supports_background_processing
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NotificationPreviewPayloadV1 {
    pub v: i16,
    pub notification_id: Uuid,
    pub user_uuid: Uuid,
    pub project_id: i32,
    pub source_kind: String,
    pub source_request_id: Option<Uuid>,
    pub background_grant_uuid: Option<Uuid>,
    pub agent_uuid: Option<Uuid>,
    pub schedule_uuid: Option<Uuid>,
    pub delivery_mode: String,
    pub message_id: Uuid,
    pub kind: String,
    pub title: String,
    pub body: String,
    pub deep_link: String,
    pub thread_id: String,
    pub sent_at: i64,
}

impl NotificationPreviewPayloadV1 {
    pub fn matches_event(
        &self,
        event: &NotificationEvent,
        background_grant_uuid: Option<Uuid>,
    ) -> bool {
        self.v == PUSH_EVENT_PAYLOAD_VERSION_V1
            && self.notification_id == event.uuid
            && self.user_uuid == event.user_id
            && self.project_id == event.project_id
            && self.source_kind == event.source_kind
            && self.source_request_id == event.source_request_id
            && self.background_grant_uuid == background_grant_uuid
            && self.delivery_mode == event.delivery_mode
            && self.kind == event.kind
    }
}

pub fn hash_bytes(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

pub fn encrypt_push_device_capability_v1(
    root_key: &[u8],
    plaintext: &PushDeviceCapabilityPlaintextV1,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(root_key, PUSH_DEVICE_AEAD_INFO)?;
    let aad = push_device_aad_v1(PushDeviceAad {
        push_device_uuid: plaintext.push_device_uuid,
        user_uuid: plaintext.user_uuid,
        project_id: plaintext.project_id,
        installation_id: plaintext.installation_id,
        platform: &plaintext.platform,
        provider: &plaintext.provider,
        environment: &plaintext.environment,
        app_id: &plaintext.app_id,
        version: plaintext.v,
    });
    let bytes = serde_json::to_vec(plaintext).map_err(|_| EncryptError::BadData)?;
    encrypt_aead_v1(&key, &bytes, &aad)
}

pub fn decrypt_push_device_capability_v1(
    root_key: &[u8],
    encrypted: &[u8],
    device: &PushDevice,
) -> Result<PushDeviceCapabilityPlaintextV1, EncryptError> {
    let key = derive_key(root_key, PUSH_DEVICE_AEAD_INFO)?;
    let aad = push_device_aad_v1(PushDeviceAad {
        push_device_uuid: device.uuid,
        user_uuid: device.user_id,
        project_id: device.project_id,
        installation_id: device.installation_id,
        platform: &device.platform,
        provider: &device.provider,
        environment: &device.environment,
        app_id: &device.app_id,
        version: PUSH_CAPABILITY_VERSION_V1,
    });
    let bytes = decrypt_aead_v1(&key, encrypted, &aad)?;
    serde_json::from_slice(&bytes).map_err(|_| EncryptError::BadData)
}

pub fn encrypt_notification_preview_payload_v1(
    root_key: &[u8],
    plaintext: &NotificationPreviewPayloadV1,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(root_key, PUSH_EVENT_PAYLOAD_AEAD_INFO)?;
    let aad = notification_payload_aad_v1(NotificationPayloadAad {
        notification_uuid: plaintext.notification_id,
        user_uuid: plaintext.user_uuid,
        project_id: plaintext.project_id,
        source_kind: &plaintext.source_kind,
        background_grant_uuid: plaintext.background_grant_uuid,
        kind: &plaintext.kind,
        delivery_mode: &plaintext.delivery_mode,
        version: plaintext.v,
    });
    let bytes = serde_json::to_vec(plaintext).map_err(|_| EncryptError::BadData)?;
    encrypt_aead_v1(&key, &bytes, &aad)
}

pub fn decrypt_notification_preview_payload_v1(
    root_key: &[u8],
    encrypted: &[u8],
    event: &NotificationEvent,
    background_grant_uuid: Option<Uuid>,
) -> Result<NotificationPreviewPayloadV1, EncryptError> {
    let key = derive_key(root_key, PUSH_EVENT_PAYLOAD_AEAD_INFO)?;
    let aad = notification_payload_aad_v1(NotificationPayloadAad {
        notification_uuid: event.uuid,
        user_uuid: event.user_id,
        project_id: event.project_id,
        source_kind: &event.source_kind,
        background_grant_uuid,
        kind: &event.kind,
        delivery_mode: &event.delivery_mode,
        version: PUSH_EVENT_PAYLOAD_VERSION_V1,
    });
    let bytes = decrypt_aead_v1(&key, encrypted, &aad)?;
    serde_json::from_slice(&bytes).map_err(|_| EncryptError::BadData)
}

pub fn decrypt_background_grant_for_push(
    root_key: &[u8],
    grant: &AgentBackgroundGrant,
    payload: &NotificationPreviewPayloadV1,
) -> Result<AgentBackgroundGrantPlaintextV1, EncryptError> {
    decrypt_background_grant_v1(
        root_key,
        &grant.grant_enc,
        grant.uuid,
        payload.user_uuid,
        payload.project_id,
        payload.agent_uuid.ok_or(EncryptError::BadData)?,
        payload.schedule_uuid.ok_or(EncryptError::BadData)?,
    )
}

struct PushDeviceAad<'a> {
    push_device_uuid: Uuid,
    user_uuid: Uuid,
    project_id: i32,
    installation_id: Uuid,
    platform: &'a str,
    provider: &'a str,
    environment: &'a str,
    app_id: &'a str,
    version: i16,
}

fn push_device_aad_v1(input: PushDeviceAad<'_>) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(PUSH_DEVICE_AAD_DOMAIN);
    aad.append_uuid(input.push_device_uuid)
        .append_uuid(input.user_uuid)
        .append_i32(input.project_id)
        .append_uuid(input.installation_id)
        .append_str(input.platform)
        .append_str(input.provider)
        .append_str(input.environment)
        .append_str(input.app_id)
        .append_i16(input.version);
    aad.into_bytes()
}

struct NotificationPayloadAad<'a> {
    notification_uuid: Uuid,
    user_uuid: Uuid,
    project_id: i32,
    source_kind: &'a str,
    background_grant_uuid: Option<Uuid>,
    kind: &'a str,
    delivery_mode: &'a str,
    version: i16,
}

fn notification_payload_aad_v1(input: NotificationPayloadAad<'_>) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(PUSH_EVENT_PAYLOAD_AAD_DOMAIN);
    aad.append_uuid(input.notification_uuid)
        .append_uuid(input.user_uuid)
        .append_i32(input.project_id)
        .append_str(input.source_kind)
        .append_str(
            &input
                .background_grant_uuid
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        )
        .append_str(input.kind)
        .append_str(input.delivery_mode)
        .append_i16(input.version);
    aad.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::push_devices::{
        PushDevice, PUSH_ENV_PROD, PUSH_KEY_ALGORITHM_P256_ECDH_V1, PUSH_PLATFORM_IOS,
        PUSH_PROVIDER_APNS,
    };

    fn test_device() -> PushDevice {
        let public_key = vec![8, 9, 10];
        PushDevice {
            id: 1,
            uuid: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            project_id: 42,
            installation_id: Uuid::new_v4(),
            platform: PUSH_PLATFORM_IOS.to_string(),
            provider: PUSH_PROVIDER_APNS.to_string(),
            environment: PUSH_ENV_PROD.to_string(),
            app_id: "ai.trymaple.ios".to_string(),
            push_token_hash: hash_bytes(b"token"),
            capability_enc: vec![],
            notification_public_key_hash: hash_bytes(&public_key),
            key_algorithm: PUSH_KEY_ALGORITHM_P256_ECDH_V1.to_string(),
            supports_encrypted_preview: true,
            supports_background_processing: true,
            last_seen_at: Utc::now(),
            revoked_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn push_device_capability_aad_binds_user_and_device() {
        let root = [4_u8; 32];
        let mut device = test_device();
        let plaintext = PushDeviceCapabilityPlaintextV1::new(PushDeviceCapabilityInput {
            push_device_uuid: device.uuid,
            user_uuid: device.user_id,
            project_id: device.project_id,
            installation_id: device.installation_id,
            platform: device.platform.clone(),
            provider: device.provider.clone(),
            environment: device.environment.clone(),
            app_id: device.app_id.clone(),
            key_algorithm: device.key_algorithm.clone(),
            push_token: "token".to_string(),
            notification_public_key: vec![8, 9, 10],
            supports_encrypted_preview: true,
            supports_background_processing: true,
        });
        device.capability_enc = encrypt_push_device_capability_v1(&root, &plaintext).unwrap();

        let decrypted =
            decrypt_push_device_capability_v1(&root, &device.capability_enc, &device).unwrap();
        assert!(decrypted.matches_device_row(&device));

        let mut attacker_row = device.clone();
        attacker_row.user_id = Uuid::new_v4();
        assert!(
            decrypt_push_device_capability_v1(&root, &device.capability_enc, &attacker_row,)
                .is_err()
        );
    }
}
