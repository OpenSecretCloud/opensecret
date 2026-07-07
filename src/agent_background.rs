use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::encrypt::{
    decrypt_aead_v1, derive_key, encrypt_aead_v1, generate_random, CanonicalBytes, EncryptError,
};

const BACKGROUND_GRANT_AEAD_INFO: &[u8] = b"os.agent-background-grant-aead.v1";
const BACKGROUND_GRANT_AAD_DOMAIN: &str = "os.agent-background-grant.v1";

pub const AGENT_BACKGROUND_GRANT_VERSION_V1: i16 = 1;
pub const AGENT_BACKGROUND_CAPABILITY_FULL_SCHEDULED_AGENT: &str = "full_scheduled_agent";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentBackgroundGrantPlaintextV1 {
    pub v: i16,
    pub grant_uuid: Uuid,
    pub user_uuid: Uuid,
    pub project_id: i32,
    pub agent_uuid: Uuid,
    pub schedule_uuid: Uuid,
    pub instruction_hash: Vec<u8>,
    pub may_decrypt_user_content: bool,
    pub may_send_push: bool,
    pub capability_class: String,
    pub created_from_auth_method: String,
    pub created_at: DateTime<Utc>,
    pub background_secret: [u8; 32],
}

impl AgentBackgroundGrantPlaintextV1 {
    pub fn new_scheduled(
        grant_uuid: Uuid,
        user_uuid: Uuid,
        project_id: i32,
        agent_uuid: Uuid,
        schedule_uuid: Uuid,
        instruction_hash: Vec<u8>,
        created_from_auth_method: impl Into<String>,
    ) -> Self {
        Self {
            v: AGENT_BACKGROUND_GRANT_VERSION_V1,
            grant_uuid,
            user_uuid,
            project_id,
            agent_uuid,
            schedule_uuid,
            instruction_hash,
            may_decrypt_user_content: true,
            may_send_push: true,
            capability_class: AGENT_BACKGROUND_CAPABILITY_FULL_SCHEDULED_AGENT.to_string(),
            created_from_auth_method: created_from_auth_method.into(),
            created_at: Utc::now(),
            background_secret: generate_random::<32>(),
        }
    }

    pub fn verify_scheduled_policy(
        &self,
        user_uuid: Uuid,
        project_id: i32,
        agent_uuid: Uuid,
        schedule_uuid: Uuid,
        instruction_hash: &[u8],
    ) -> bool {
        self.v == AGENT_BACKGROUND_GRANT_VERSION_V1
            && self.user_uuid == user_uuid
            && self.project_id == project_id
            && self.agent_uuid == agent_uuid
            && self.schedule_uuid == schedule_uuid
            && self.instruction_hash == instruction_hash
            && self.may_decrypt_user_content
            && self.capability_class == AGENT_BACKGROUND_CAPABILITY_FULL_SCHEDULED_AGENT
    }
}

pub fn instruction_hash(instruction: &str) -> Vec<u8> {
    Sha256::digest(instruction.as_bytes()).to_vec()
}

pub fn encrypt_background_grant_v1(
    root_key: &[u8],
    plaintext: &AgentBackgroundGrantPlaintextV1,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(root_key, BACKGROUND_GRANT_AEAD_INFO)?;
    let aad = background_grant_aad_v1(
        plaintext.grant_uuid,
        plaintext.user_uuid,
        plaintext.project_id,
        plaintext.agent_uuid,
        plaintext.schedule_uuid,
        plaintext.v,
    );
    let bytes = serde_json::to_vec(plaintext).map_err(|_| EncryptError::BadData)?;
    encrypt_aead_v1(&key, &bytes, &aad)
}

pub fn decrypt_background_grant_v1(
    root_key: &[u8],
    encrypted: &[u8],
    grant_uuid: Uuid,
    user_uuid: Uuid,
    project_id: i32,
    agent_uuid: Uuid,
    schedule_uuid: Uuid,
) -> Result<AgentBackgroundGrantPlaintextV1, EncryptError> {
    let key = derive_key(root_key, BACKGROUND_GRANT_AEAD_INFO)?;
    let aad = background_grant_aad_v1(
        grant_uuid,
        user_uuid,
        project_id,
        agent_uuid,
        schedule_uuid,
        AGENT_BACKGROUND_GRANT_VERSION_V1,
    );
    let bytes = decrypt_aead_v1(&key, encrypted, &aad)?;
    serde_json::from_slice(&bytes).map_err(|_| EncryptError::BadData)
}

fn background_grant_aad_v1(
    grant_uuid: Uuid,
    user_uuid: Uuid,
    project_id: i32,
    agent_uuid: Uuid,
    schedule_uuid: Uuid,
    version: i16,
) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(BACKGROUND_GRANT_AAD_DOMAIN);
    aad.append_uuid(grant_uuid)
        .append_uuid(user_uuid)
        .append_i32(project_id)
        .append_uuid(agent_uuid)
        .append_uuid(schedule_uuid)
        .append_i16(version);
    aad.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_grant_aead_binds_principal_context() {
        let root = [9_u8; 32];
        let user = Uuid::new_v4();
        let agent = Uuid::new_v4();
        let schedule = Uuid::new_v4();
        let grant = Uuid::new_v4();
        let plaintext = AgentBackgroundGrantPlaintextV1::new_scheduled(
            grant,
            user,
            17,
            agent,
            schedule,
            instruction_hash("check the weather"),
            "password",
        );

        let encrypted = encrypt_background_grant_v1(&root, &plaintext).unwrap();
        let decrypted =
            decrypt_background_grant_v1(&root, &encrypted, grant, user, 17, agent, schedule)
                .unwrap();
        assert_eq!(decrypted.user_uuid, user);

        assert!(
            decrypt_background_grant_v1(
                &root,
                &encrypted,
                grant,
                Uuid::new_v4(),
                17,
                agent,
                schedule,
            )
            .is_err(),
            "grant ciphertext must not decrypt under a different user principal",
        );
    }
}
