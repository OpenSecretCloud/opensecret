use aes_gcm::{
    aead::{Aead as GcmAead, KeyInit as GcmKeyInit},
    Aes256Gcm, Nonce as GcmNonce,
};
use aes_siv::{Aes256SivAead, Nonce as SivNonce};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use generic_array::typenum;
use generic_array::GenericArray;
use rand_core::RngCore;
use secp256k1::rand::rngs::OsRng;
use secp256k1::SecretKey;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha512};
use std::{process::Command, sync::Arc};
use tokio::sync::Mutex;
use tracing::error;

use crate::aws_credentials::AwsCredentialManager;

#[derive(Debug, thiserror::Error)]
pub enum EncryptError {
    #[error("Failed to decrypt")]
    FailedToDecrypt,
    #[error("Bad data")]
    BadData,
    #[error("KMS encryption failed: {0}")]
    KmsError(String),
    #[error("No content to decrypt")]
    NoContent,
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
}

pub async fn encrypt_with_key(encryption_key: &SecretKey, bytes: &[u8]) -> Vec<u8> {
    tracing::debug!("Entering encrypt_with_key");
    let cipher = Aes256Gcm::new_from_slice(&encryption_key.secret_bytes()).expect("should convert");

    // Generate a random 96-bit nonce
    let nonce: [u8; 12] = generate_random::<12>();

    let nonce = GcmNonce::from_slice(&nonce);

    let ciphertext = cipher.encrypt(nonce, bytes).expect("should encrypt");

    // Combine nonce and ciphertext
    let mut encrypted = nonce.to_vec();
    encrypted.extend(ciphertext);

    tracing::debug!("Exiting encrypt_with_key");
    encrypted
}

pub fn decrypt_with_key(encryption_key: &SecretKey, bytes: &[u8]) -> Result<Vec<u8>, EncryptError> {
    tracing::trace!("Entering decrypt_with_key");

    if bytes.len() < 12 {
        tracing::error!(
            "Decrypt failed: Input too short (length {}), minimum 12 bytes required",
            bytes.len()
        );
        return Err(EncryptError::BadData);
    }

    // The first 12 bytes are the nonce
    let nonce = GcmNonce::from_slice(&bytes[..12]);

    // The rest is the ciphertext
    let ciphertext = &bytes[12..];

    let cipher = Aes256Gcm::new_from_slice(&encryption_key.secret_bytes()).map_err(|e| {
        tracing::error!("Failed to create cipher from key: {e}");
        EncryptError::FailedToDecrypt
    })?;

    let result = cipher.decrypt(nonce, ciphertext).map_err(|e| {
        // Log detailed error without revealing any sensitive data
        tracing::error!(
            "AES-GCM decryption failed: {:?}. Input length: {}, nonce length: 12, ciphertext length: {}",
            e, bytes.len(), ciphertext.len()
        );
        EncryptError::FailedToDecrypt
    })?;

    tracing::trace!("Exiting decrypt_with_key");
    Ok(result)
}

// ============================================================================
// High-Level Decryption Helpers
//
// The functions above (encrypt_with_key, decrypt_with_key) are LOW-LEVEL
// primitives that work with raw bytes.
//
// The functions below are HIGH-LEVEL helpers that add:
// - Automatic JSON deserialization
// - Type safety with generics
// - Proper error handling (never silent failures!)
//
// Use these helpers when decrypting structured data (JSON) from the database.
// Use the low-level functions when working with raw bytes or custom formats.
// ============================================================================

/// Decrypt and deserialize encrypted content with type safety
///
/// This is a high-level helper that decrypts content and automatically
/// deserializes it into the target type T using serde_json.
///
/// **Important**: If `encrypted` is `None`, this returns `Ok(None)` (not an error).
/// However, if decryption or deserialization FAILS, this returns `Err` (never silently fails!).
///
/// # Arguments
/// * `key` - The encryption key to use for decryption
/// * `encrypted` - Optional encrypted content bytes
///
/// # Returns
/// - `Ok(Some(T))` if data exists and decryption succeeds
/// - `Ok(None)` if `encrypted` is `None` (no data present)
/// - `Err(EncryptError)` if decryption or deserialization fails
///
/// # Example
/// ```ignore
/// let content: Option<MessageContent> = decrypt_content(&user_key, msg.content_enc.as_ref())
///     .map_err(|e| {
///         error!("Failed to decrypt message: {:?}", e);
///         ApiError::InternalServerError
///     })?;
/// ```
pub fn decrypt_content<T>(
    key: &SecretKey,
    encrypted: Option<&Vec<u8>>,
) -> Result<Option<T>, EncryptError>
where
    T: DeserializeOwned,
{
    let Some(encrypted) = encrypted else {
        return Ok(None);
    };

    let decrypted_bytes = decrypt_with_key(key, encrypted)?;

    let value = serde_json::from_slice(&decrypted_bytes)
        .map_err(|e| EncryptError::DeserializationFailed(e.to_string()))?;

    Ok(Some(value))
}

/// Decrypt encrypted content as a plain UTF-8 string
///
/// This is a high-level helper for decrypting plain text content (not JSON).
///
/// **Important**: If `encrypted` is `None`, this returns `Ok(None)` (not an error).
/// However, if decryption FAILS, this returns `Err` (never silently fails!).
///
/// # Arguments
/// * `key` - The encryption key to use for decryption
/// * `encrypted` - Optional encrypted content bytes
///
/// # Returns
/// - `Ok(Some(String))` if data exists and decryption succeeds
/// - `Ok(None)` if `encrypted` is `None` (no data present)
/// - `Err(EncryptError)` if decryption fails
///
/// # Example
/// ```ignore
/// let text: Option<String> = decrypt_string(&user_key, msg.content_enc.as_ref())
///     .map_err(|e| {
///         error!("Failed to decrypt message: {:?}", e);
///         ApiError::InternalServerError
///     })?;
/// ```
pub fn decrypt_string(
    key: &SecretKey,
    encrypted: Option<&Vec<u8>>,
) -> Result<Option<String>, EncryptError> {
    let Some(encrypted) = encrypted else {
        return Ok(None);
    };

    let bytes = decrypt_with_key(key, encrypted)?;
    Ok(Some(String::from_utf8_lossy(&bytes).to_string()))
}

pub fn encrypt_key_deterministic(encryption_key: &SecretKey, key: &[u8]) -> Vec<u8> {
    let key_bytes: [u8; 32] = encryption_key.secret_bytes();
    let extended_key = extend_key(&key_bytes);
    let cipher = Aes256SivAead::new(&extended_key);
    let nonce = SivNonce::default();
    cipher.encrypt(&nonce, key).expect("encryption failure!")
}

pub fn decrypt_key_deterministic(
    encryption_key: &SecretKey,
    encrypted: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key_bytes: [u8; 32] = encryption_key.secret_bytes();
    let extended_key = extend_key(&key_bytes);
    let cipher = Aes256SivAead::new(&extended_key);
    let nonce = SivNonce::default();
    cipher
        .decrypt(&nonce, encrypted)
        .map_err(|_| EncryptError::FailedToDecrypt)
}

fn extend_key(key: &[u8; 32]) -> GenericArray<u8, typenum::U64> {
    let mut hasher = Sha512::new();
    hasher.update(key);
    GenericArray::clone_from_slice(&hasher.finalize())
}

pub fn decrypt_with_kms(
    aws_region: &str,
    aws_key_id: &str,
    aws_secret_key: &str,
    aws_session_token: &str,
    ciphertext: &str,
) -> Result<Vec<u8>, EncryptError> {
    tracing::debug!("Attempting KMS decryption");
    let output = Command::new("/bin/kmstool_enclave_cli")
        .arg("decrypt")
        .arg("--region")
        .arg(aws_region)
        .arg("--proxy-port")
        .arg("8000")
        .arg("--aws-access-key-id")
        .arg(aws_key_id)
        .arg("--aws-secret-access-key")
        .arg(aws_secret_key)
        .arg("--aws-session-token")
        .arg(aws_session_token)
        .arg("--ciphertext")
        .arg(ciphertext)
        .output()
        .map_err(|e| {
            tracing::error!(
                "Failed to execute kmstool_enclave_cli for decryption: {}",
                e
            );
            EncryptError::KmsError(e.to_string())
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!("kmstool_enclave_cli decryption failed: {}", stderr);
        return Err(EncryptError::KmsError(stderr.to_string()));
    }

    let output_str =
        String::from_utf8(output.stdout).map_err(|e| EncryptError::KmsError(e.to_string()))?;

    let plaintext_b64 = output_str
        .strip_prefix("PLAINTEXT: ")
        .ok_or_else(|| EncryptError::KmsError("Failed to parse plaintext".to_string()))?
        .trim();

    STANDARD
        .decode(plaintext_b64)
        .map_err(|e| EncryptError::KmsError(format!("Failed to decode base64: {}", e)))
}

#[derive(Debug)]
pub struct GenKeyResult {
    pub key: Vec<u8>,
    pub encrypted_key: Vec<u8>,
}

pub fn create_new_encryption_key(
    aws_region: &str,
    aws_key_id: &str,
    aws_secret_key: &str,
    aws_session_token: &str,
    aws_kms_key_id: &str,
) -> Result<GenKeyResult, EncryptError> {
    tracing::info!("Creating new encryption key");
    tracing::debug!("Attempting to run kmstool_enclave_cli");
    let output = Command::new("/bin/kmstool_enclave_cli")
        .arg("genkey")
        .arg("--region")
        .arg(aws_region)
        .arg("--proxy-port")
        .arg("8000")
        .arg("--aws-access-key-id")
        .arg(aws_key_id)
        .arg("--aws-secret-access-key")
        .arg(aws_secret_key)
        .arg("--aws-session-token")
        .arg(aws_session_token)
        .arg("--key-id")
        .arg(aws_kms_key_id)
        .arg("--key-spec")
        .arg("AES-256")
        .output()
        .map_err(|e| {
            tracing::error!("Failed to execute kmstool_enclave_cli: {}", e);
            EncryptError::KmsError(e.to_string())
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!("kmstool_enclave_cli failed: {}", stderr);
        return Err(EncryptError::KmsError(stderr.to_string()));
    }

    let output_str =
        String::from_utf8(output.stdout).map_err(|e| EncryptError::KmsError(e.to_string()))?;
    let lines: Vec<&str> = output_str.lines().collect();

    let encrypted_key_b64 = lines[0]
        .split(": ")
        .nth(1)
        .ok_or_else(|| EncryptError::KmsError("Failed to parse encrypted key".to_string()))?;
    let plaintext_key_b64 = lines[1]
        .split(": ")
        .nth(1)
        .ok_or_else(|| EncryptError::KmsError("Failed to parse plaintext key".to_string()))?;

    let encrypted_key = STANDARD
        .decode(encrypted_key_b64)
        .map_err(|e| EncryptError::KmsError(format!("Failed to decode encrypted key: {}", e)))?;

    let plaintext_key = STANDARD
        .decode(plaintext_key_b64)
        .map_err(|e| EncryptError::KmsError(e.to_string()))?;

    Ok(GenKeyResult {
        encrypted_key,
        key: plaintext_key,
    })
}

pub fn generate_random<const LENGTH: usize>() -> [u8; LENGTH] {
    let mut buffer = [0u8; LENGTH];
    getrandom::getrandom(&mut buffer).expect("Failed to generate random bytes");
    buffer
}

pub async fn generate_random_enclave<const LENGTH: usize>(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
) -> [u8; LENGTH] {
    let nonce = if let Some(cred_manager) = aws_credential_manager.read().await.as_ref().cloned() {
        let aws_creds = cred_manager
            .get_credentials()
            .await
            .expect("should have creds");

        generate_random_bytes_from_enclave(
            &aws_creds.region,
            &aws_creds.access_key_id,
            &aws_creds.secret_access_key,
            &aws_creds.token,
            LENGTH,
        )
        .await
        .expect("should generate random bytes")
    } else {
        // Use OS random if aws_credential_manager is None
        let mut nonce = [0u8; LENGTH];
        OsRng.fill_bytes(&mut nonce);
        nonce.to_vec()
    };
    nonce.try_into().expect("Length mismatch")
}

pub async fn generate_random_bytes_from_enclave(
    aws_region: &str,
    aws_key_id: &str,
    aws_secret_key: &str,
    aws_session_token: &str,
    length: usize,
) -> Result<Vec<u8>, EncryptError> {
    tracing::debug!("Attempting to run kmstool_enclave_cli for random byte generation");
    let output = Command::new("/bin/kmstool_enclave_cli")
        .arg("genrandom")
        .arg("--region")
        .arg(aws_region)
        .arg("--proxy-port")
        .arg("8000")
        .arg("--aws-access-key-id")
        .arg(aws_key_id)
        .arg("--aws-secret-access-key")
        .arg(aws_secret_key)
        .arg("--aws-session-token")
        .arg(aws_session_token)
        .arg("--length")
        .arg(length.to_string())
        .output()
        .map_err(|e| {
            tracing::error!(
                "Failed to execute kmstool_enclave_cli for random byte generation: {}",
                e
            );
            EncryptError::KmsError(e.to_string())
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(
            "kmstool_enclave_cli random byte generation failed: {}",
            stderr
        );
        return Err(EncryptError::KmsError(stderr.to_string()));
    }

    let output_str =
        String::from_utf8(output.stdout).map_err(|e| EncryptError::KmsError(e.to_string()))?;

    let plaintext_b64 = output_str
        .strip_prefix("PLAINTEXT: ")
        .ok_or_else(|| EncryptError::KmsError("Failed to parse plaintext".to_string()))?
        .trim();

    STANDARD
        .decode(plaintext_b64)
        .map_err(|e| EncryptError::KmsError(format!("Failed to decode base64: {}", e)))
}

pub struct CustomRng {
    buffer: Mutex<Vec<u8>>,
}

impl CustomRng {
    pub fn new() -> Self {
        CustomRng {
            buffer: Mutex::new(Vec::new()),
        }
    }

    async fn fill_buffer(&self) {
        let bytes: [u8; 1024] = generate_random();
        let mut buffer = self.buffer.lock().await;
        buffer.extend_from_slice(&bytes);
    }

    pub async fn fill_bytes(&self, dest: &mut [u8]) {
        let mut buffer = self.buffer.lock().await;
        while buffer.len() < dest.len() {
            drop(buffer); // Release the lock before filling the buffer
            self.fill_buffer().await;
            buffer = self.buffer.lock().await;
        }

        let n = dest.len();
        dest.copy_from_slice(&buffer[..n]);
        *buffer = buffer[n..].to_vec();
    }

    pub async fn next_u32(&self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes).await;
        u32::from_le_bytes(bytes)
    }

    pub async fn next_u64(&self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes).await;
        u64::from_le_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption_with_key() {
        let key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let content = [6u8; 32].to_vec();

        let encrypted = encrypt_with_key(&key, &content).await;

        let decrypted = decrypt_with_key(&key, &encrypted).unwrap();
        assert_eq!(content, decrypted);
    }

    #[test]
    fn test_deterministic_encryption() {
        let key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let content = b"test_key";

        let encrypted = encrypt_key_deterministic(&key, content);
        let decrypted = decrypt_key_deterministic(&key, &encrypted).unwrap();
        assert_eq!(content.to_vec(), decrypted);
    }
}
