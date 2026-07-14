use aes_gcm::{
    aead::{Aead as GcmAead, KeyInit as GcmKeyInit, Payload},
    Aes256Gcm, Nonce as GcmNonce,
};
use aes_siv::{Aes256SivAead, Nonce as SivNonce};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use generic_array::typenum;
use generic_array::GenericArray;
use hkdf::Hkdf;
use rand_core::RngCore;
use secp256k1::rand::rngs::OsRng;
use secp256k1::SecretKey;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256, Sha512};
use std::{process::Command, sync::Arc};
use tokio::sync::Mutex;
use tracing::error;
use uuid::Uuid;

use crate::aws_credentials::AwsCredentialManager;

#[derive(Debug, thiserror::Error)]
pub enum EncryptError {
    #[error("Failed to decrypt")]
    FailedToDecrypt,
    #[error("Bad data")]
    BadData,
    #[error("KMS operation failed: {0}")]
    KmsError(String),
    #[error("Random generation failed: {0}")]
    RandomGenerationFailed(String),
    #[error("No content to decrypt")]
    NoContent,
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
    #[error("Key derivation failed")]
    KeyDerivationFailed,
}

pub async fn encrypt_with_key(encryption_key: &SecretKey, bytes: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(&encryption_key.secret_bytes()).expect("should convert");

    // Generate a random 96-bit nonce
    let nonce: [u8; 12] = generate_random::<12>();

    let nonce = GcmNonce::from_slice(&nonce);

    let ciphertext = cipher.encrypt(nonce, bytes).expect("should encrypt");

    // Combine nonce and ciphertext
    let mut encrypted = nonce.to_vec();
    encrypted.extend(ciphertext);
    encrypted
}

pub fn decrypt_with_key(encryption_key: &SecretKey, bytes: &[u8]) -> Result<Vec<u8>, EncryptError> {
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
    Ok(result)
}

pub type AeadKey = [u8; 32];

pub fn derive_key(root_key: &[u8], info: &[u8]) -> Result<AeadKey, EncryptError> {
    derive_key_with_optional_salt(root_key, None, info)
}

pub fn derive_key_with_salt(
    input_key: &[u8],
    salt: &[u8],
    info: &[u8],
) -> Result<AeadKey, EncryptError> {
    derive_key_with_optional_salt(input_key, Some(salt), info)
}

fn derive_key_with_optional_salt(
    input_key: &[u8],
    salt: Option<&[u8]>,
    info: &[u8],
) -> Result<AeadKey, EncryptError> {
    let hkdf = Hkdf::<Sha256>::new(salt, input_key);
    let mut output = [0u8; 32];
    hkdf.expand(info, &mut output)
        .map_err(|_| EncryptError::KeyDerivationFailed)?;
    Ok(output)
}

pub fn encrypt_aead_v1(
    encryption_key: &AeadKey,
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let cipher =
        Aes256Gcm::new_from_slice(encryption_key).map_err(|_| EncryptError::FailedToDecrypt)?;
    let nonce: [u8; 12] = generate_random::<12>();
    let nonce = GcmNonce::from_slice(&nonce);

    let ciphertext = cipher
        .encrypt(
            nonce,
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| EncryptError::FailedToDecrypt)?;

    let mut encrypted = nonce.to_vec();
    encrypted.extend(ciphertext);
    Ok(encrypted)
}

pub fn decrypt_aead_v1(
    encryption_key: &AeadKey,
    bytes: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    if bytes.len() < 12 {
        return Err(EncryptError::BadData);
    }

    let nonce = GcmNonce::from_slice(&bytes[..12]);
    let ciphertext = &bytes[12..];
    let cipher =
        Aes256Gcm::new_from_slice(encryption_key).map_err(|_| EncryptError::FailedToDecrypt)?;

    cipher
        .decrypt(
            nonce,
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| EncryptError::FailedToDecrypt)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CanonicalBytes {
    bytes: Vec<u8>,
}

impl CanonicalBytes {
    pub fn new(domain: &str) -> Self {
        let mut canonical = Self::default();
        canonical.append_str(domain);
        canonical
    }

    pub fn append_str(&mut self, value: &str) -> &mut Self {
        self.append_field(b's', value.as_bytes())
    }

    pub fn append_bytes(&mut self, value: &[u8]) -> &mut Self {
        self.append_field(b'b', value)
    }

    pub fn append_i16(&mut self, value: i16) -> &mut Self {
        self.append_field(b'i', &value.to_be_bytes())
    }

    pub fn append_i32(&mut self, value: i32) -> &mut Self {
        self.append_field(b'I', &value.to_be_bytes())
    }

    pub fn append_uuid(&mut self, value: Uuid) -> &mut Self {
        self.append_field(b'u', value.as_bytes())
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn append_field(&mut self, tag: u8, value: &[u8]) -> &mut Self {
        let len: u32 = value
            .len()
            .try_into()
            .expect("canonical field length should fit in u32");
        self.bytes.push(tag);
        self.bytes.extend_from_slice(&len.to_be_bytes());
        self.bytes.extend_from_slice(value);
        self
    }
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
            let error = kms_execution_error("decrypt", &e);
            tracing::error!("{error}");
            error
        })?;

    if !output.status.success() {
        let error = kms_command_error("decrypt", &output.stderr);
        tracing::error!("{error}");
        return Err(error);
    }

    parse_kms_plaintext_output(&output.stdout, "decrypt")
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
            let error = kms_execution_error("genkey", &e);
            tracing::error!("{error}");
            error
        })?;

    if !output.status.success() {
        let error = kms_command_error("genkey", &output.stderr);
        tracing::error!("{error}");
        return Err(error);
    }

    parse_kms_genkey_output(&output.stdout)
}

fn kms_execution_error(operation: &str, error: &std::io::Error) -> EncryptError {
    EncryptError::KmsError(format!(
        "kmstool_enclave_cli {operation} could not be executed: {error}"
    ))
}

fn kms_command_error(operation: &str, stderr: &[u8]) -> EncryptError {
    let stderr = String::from_utf8_lossy(stderr);
    let stderr = stderr.trim();
    let detail = if stderr.is_empty() {
        "no error output".to_string()
    } else {
        stderr.to_string()
    };

    EncryptError::KmsError(format!(
        "kmstool_enclave_cli {operation} exited unsuccessfully: {detail}"
    ))
}

fn parse_kms_plaintext_output(stdout: &[u8], operation: &str) -> Result<Vec<u8>, EncryptError> {
    let output = parse_kms_stdout(stdout, operation)?;
    decode_kms_output_field(output, "PLAINTEXT", operation)
}

fn parse_kms_genkey_output(stdout: &[u8]) -> Result<GenKeyResult, EncryptError> {
    const OPERATION: &str = "genkey";

    let output = parse_kms_stdout(stdout, OPERATION)?;
    let encrypted_key = decode_kms_output_field(output, "CIPHERTEXT", OPERATION)?;
    let key = decode_kms_output_field(output, "PLAINTEXT", OPERATION)?;

    Ok(GenKeyResult { key, encrypted_key })
}

fn parse_kms_random_output(stdout: &[u8], expected_length: usize) -> Result<Vec<u8>, EncryptError> {
    let bytes = parse_kms_plaintext_output(stdout, "genrandom")?;
    if bytes.len() != expected_length {
        return Err(EncryptError::KmsError(format!(
            "kmstool_enclave_cli genrandom returned {} bytes; expected {expected_length}",
            bytes.len()
        )));
    }

    Ok(bytes)
}

fn parse_kms_stdout<'a>(stdout: &'a [u8], operation: &str) -> Result<&'a str, EncryptError> {
    std::str::from_utf8(stdout).map_err(|error| {
        EncryptError::KmsError(format!(
            "kmstool_enclave_cli {operation} returned non-UTF-8 output: {error}"
        ))
    })
}

fn decode_kms_output_field(
    output: &str,
    field: &str,
    operation: &str,
) -> Result<Vec<u8>, EncryptError> {
    let prefix = format!("{field}:");
    let encoded = output
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).map(str::trim))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            EncryptError::KmsError(format!(
                "kmstool_enclave_cli {operation} output is missing {field}"
            ))
        })?;

    STANDARD.decode(encoded).map_err(|error| {
        EncryptError::KmsError(format!(
            "kmstool_enclave_cli {operation} returned invalid base64 for {field}: {error}"
        ))
    })
}

pub fn generate_random<const LENGTH: usize>() -> [u8; LENGTH] {
    let mut buffer = [0u8; LENGTH];
    getrandom::getrandom(&mut buffer).expect("Failed to generate random bytes");
    buffer
}

pub async fn generate_random_enclave<const LENGTH: usize>(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
) -> Result<[u8; LENGTH], EncryptError> {
    if let Some(cred_manager) = aws_credential_manager.read().await.as_ref().cloned() {
        let aws_creds = cred_manager.get_credentials().await.ok_or_else(|| {
            EncryptError::KmsError(
                "AWS credentials are unavailable for kmstool_enclave_cli genrandom".to_string(),
            )
        })?;

        let bytes = generate_random_bytes_from_enclave(
            &aws_creds.region,
            &aws_creds.access_key_id,
            &aws_creds.secret_access_key,
            &aws_creds.token,
            LENGTH,
        )
        .await?;

        bytes.try_into().map_err(|bytes: Vec<u8>| {
            EncryptError::KmsError(format!(
                "kmstool_enclave_cli genrandom returned {} bytes; expected {LENGTH}",
                bytes.len()
            ))
        })
    } else {
        // Use OS random if aws_credential_manager is None
        let mut nonce = [0u8; LENGTH];
        OsRng.try_fill_bytes(&mut nonce).map_err(|error| {
            EncryptError::RandomGenerationFailed(format!(
                "OS entropy source could not fill {LENGTH} bytes: {error}"
            ))
        })?;
        Ok(nonce)
    }
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
            let error = kms_execution_error("genrandom", &e);
            tracing::error!("{error}");
            error
        })?;

    if !output.status.success() {
        let error = kms_command_error("genrandom", &output.stderr);
        tracing::error!("{error}");
        return Err(error);
    }

    parse_kms_random_output(&output.stdout, length)
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

    #[tokio::test]
    async fn legacy_aes_gcm_ciphertext_has_no_owner_context_binding() {
        let root_key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let wrong_root_key = SecretKey::from_slice(&[2u8; 32]).unwrap();
        let victim_seed = b"victim seed mnemonic bytes";

        let encrypted_seed = encrypt_with_key(&root_key, victim_seed).await;

        assert!(
            decrypt_with_key(&wrong_root_key, &encrypted_seed).is_err(),
            "legacy AES-GCM still authenticates the root key"
        );

        let attacker_row_decrypt = decrypt_with_key(&root_key, &encrypted_seed).unwrap();
        assert_eq!(
            victim_seed.to_vec(),
            attacker_row_decrypt,
            "legacy ciphertext has no AAD parameter for user, credential, table, or row ownership"
        );
    }

    #[test]
    fn canonical_bytes_are_length_delimited() {
        let mut first = CanonicalBytes::new("domain");
        first.append_str("ab").append_str("c");

        let mut second = CanonicalBytes::new("domain");
        second.append_str("a").append_str("bc");

        assert_ne!(first.into_bytes(), second.into_bytes());
    }

    #[test]
    fn aead_v1_authenticates_aad() {
        let key = derive_key(&[1u8; 32], b"test.aead-v1").unwrap();
        let content = b"plaintext";
        let encrypted = encrypt_aead_v1(&key, content, b"aad:victim").unwrap();

        let decrypted = decrypt_aead_v1(&key, &encrypted, b"aad:victim").unwrap();
        assert_eq!(content.to_vec(), decrypted);

        assert!(decrypt_aead_v1(&key, &encrypted, b"aad:attacker").is_err());
    }

    #[test]
    fn test_deterministic_encryption() {
        let key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let content = b"test_key";

        let encrypted = encrypt_key_deterministic(&key, content);
        let decrypted = decrypt_key_deterministic(&key, &encrypted).unwrap();
        assert_eq!(content.to_vec(), decrypted);
    }

    #[test]
    fn parses_labeled_genkey_output_without_assuming_line_order() {
        let output = b"PLAINTEXT: cGxhaW50ZXh0\nCIPHERTEXT: Y2lwaGVydGV4dA==\n";

        let result = parse_kms_genkey_output(output).unwrap();

        assert_eq!(result.key, b"plaintext");
        assert_eq!(result.encrypted_key, b"ciphertext");
    }

    #[test]
    fn truncated_genkey_output_returns_contextual_errors() {
        let missing_ciphertext = parse_kms_genkey_output(b"PLAINTEXT: cGxhaW50ZXh0\n")
            .unwrap_err()
            .to_string();
        assert!(missing_ciphertext.contains("genkey output is missing CIPHERTEXT"));

        let missing_plaintext = parse_kms_genkey_output(b"CIPHERTEXT: Y2lwaGVydGV4dA==\n")
            .unwrap_err()
            .to_string();
        assert!(missing_plaintext.contains("genkey output is missing PLAINTEXT"));
    }

    #[test]
    fn non_utf8_kms_output_returns_contextual_error() {
        let error = parse_kms_plaintext_output(&[0xff], "decrypt")
            .unwrap_err()
            .to_string();

        assert!(error.contains("kmstool_enclave_cli decrypt returned non-UTF-8 output"));
    }

    #[test]
    fn malformed_base64_returns_field_and_operation_context() {
        let error = parse_kms_plaintext_output(b"PLAINTEXT: not-base64!\n", "genrandom")
            .unwrap_err()
            .to_string();

        assert!(error.contains("genrandom returned invalid base64 for PLAINTEXT"));
    }

    #[test]
    fn unexpected_random_output_length_returns_contextual_error() {
        let error = parse_kms_random_output(b"PLAINTEXT: AQI=\n", 16)
            .unwrap_err()
            .to_string();

        assert!(error.contains("genrandom returned 2 bytes; expected 16"));
    }

    #[tokio::test]
    async fn missing_aws_credentials_are_propagated() {
        let credential_manager =
            Arc::new(tokio::sync::RwLock::new(Some(AwsCredentialManager::new())));

        let error = generate_random_enclave::<16>(credential_manager)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("AWS credentials are unavailable"));
        assert!(error.contains("genrandom"));
    }

    #[tokio::test]
    async fn local_random_fallback_returns_requested_length() {
        let no_credential_manager = Arc::new(tokio::sync::RwLock::new(None));

        let random = generate_random_enclave::<16>(no_credential_manager)
            .await
            .unwrap();

        assert_eq!(random.len(), 16);
    }

    #[test]
    fn unsuccessful_command_errors_include_operation_and_lossy_stderr() {
        let error = kms_command_error("genkey", &[b'f', 0x80]).to_string();
        assert!(error.contains("kmstool_enclave_cli genkey exited unsuccessfully"));
        assert!(error.contains('f'));

        let empty_error = kms_command_error("decrypt", b" \n").to_string();
        assert!(empty_error.contains("decrypt exited unsuccessfully: no error output"));
    }
}
