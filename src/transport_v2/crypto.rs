//! Cryptographic primitives and byte encodings for transport v2.
//!
//! This module deliberately exposes operation-specific record methods rather
//! than raw keys. That keeps key direction and associated-data construction at
//! the same boundary as ChaCha20-Poly1305 authentication.

use std::fmt;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use chacha20poly1305::{
    aead::{Aead, Payload},
    ChaCha20Poly1305, KeyInit, Nonce,
};
use hkdf::Hkdf;
use sha2::Sha256;
use subtle::ConstantTimeEq;
use thiserror::Error;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::envelope::RequestId;

const KEY_LEN: usize = 32;
const RECORD_NONCE_LEN: usize = 12;
const RECORD_TAG_LEN: usize = 16;
pub(crate) const RECORD_OVERHEAD_BYTES: usize = RECORD_NONCE_LEN + RECORD_TAG_LEN;
const MIN_RECORD_LEN: usize = RECORD_OVERHEAD_BYTES;

const HANDSHAKE_PAYLOAD_VERSION: u8 = 2;
const HANDSHAKE_PAYLOAD_LEN: usize = 1 + 16 + KEY_LEN + 8;

const HANDSHAKE_KEY_INFO: &[u8] = b"opensecret/transport-v2/handshake-key";
const REQUEST_KEY_INFO: &[u8] = b"opensecret/transport-v2/client-request";
const RESPONSE_KEY_INFO: &[u8] = b"opensecret/transport-v2/enclave-response";

const KEY_EXCHANGE_AAD: &[u8] = b"opensecret/transport-v2/key-exchange";
const REQUEST_RECORD_AAD: &[u8] = b"opensecret/transport-v2/request-record";
const UNARY_RESPONSE_RECORD_AAD: &[u8] = b"opensecret/transport-v2/unary-response-record";
const STREAM_RESPONSE_RECORD_AAD: &[u8] = b"opensecret/transport-v2/stream-response-record";

/// Stable, non-sensitive failures from the transport-v2 cryptographic layer.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum CryptoError {
    #[error("secure randomness is unavailable")]
    RandomnessUnavailable,
    #[error("key derivation failed")]
    KeyDerivationFailed,
    #[error("non-contributory key exchange")]
    NonContributorySharedSecret,
    #[error("record encryption failed")]
    EncryptionFailed,
    #[error("record authentication failed")]
    DecryptionFailed,
    #[error("encrypted record is too short")]
    RecordTooShort,
    #[error("invalid standard base64")]
    InvalidBase64,
    #[error("non-canonical standard base64")]
    NonCanonicalBase64,
}

/// Fresh random secret from which one directional session-key pair is derived.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct SessionMaster([u8; KEY_LEN]);

impl SessionMaster {
    pub(crate) fn random() -> Result<Self, CryptoError> {
        let mut master = Self([0; KEY_LEN]);
        fill_random(&mut master.0)?;
        Ok(master)
    }

    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(bytes)
    }

    fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl fmt::Debug for SessionMaster {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SessionMaster([REDACTED])")
    }
}

/// The fixed plaintext protected by the handshake wrapping key.
///
/// The payload contains the session master, so it is itself secret-bearing and
/// is zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct HandshakePayload([u8; HANDSHAKE_PAYLOAD_LEN]);

impl HandshakePayload {
    pub(crate) fn new(
        session_id: &Uuid,
        session_master: &SessionMaster,
        expires_at_unix_seconds: u64,
    ) -> Self {
        let mut bytes = [0; HANDSHAKE_PAYLOAD_LEN];
        bytes[0] = HANDSHAKE_PAYLOAD_VERSION;
        bytes[1..17].copy_from_slice(session_id.as_bytes());
        bytes[17..49].copy_from_slice(session_master.as_bytes());
        bytes[49..57].copy_from_slice(&expires_at_unix_seconds.to_be_bytes());
        Self(bytes)
    }

    pub(crate) fn as_bytes(&self) -> &[u8; HANDSHAKE_PAYLOAD_LEN] {
        &self.0
    }
}

impl fmt::Debug for HandshakePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HandshakePayload([REDACTED])")
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
struct RecordKey([u8; KEY_LEN]);

impl RecordKey {
    fn derive(input_key_material: &[u8], info: &[u8]) -> Result<Self, CryptoError> {
        let hkdf = Hkdf::<Sha256>::new(None, input_key_material);
        let mut key = Self([0; KEY_LEN]);
        hkdf.expand(info, &mut key.0)
            .map_err(|_| CryptoError::KeyDerivationFailed)?;
        Ok(key)
    }

    fn encrypt(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let mut nonce = [0; RECORD_NONCE_LEN];
        fill_random(&mut nonce)?;
        self.encrypt_with_nonce(plaintext, aad, nonce)
    }

    fn encrypt_with_nonce(
        &self,
        plaintext: &[u8],
        aad: &[u8],
        nonce: [u8; RECORD_NONCE_LEN],
    ) -> Result<Vec<u8>, CryptoError> {
        let cipher =
            ChaCha20Poly1305::new_from_slice(&self.0).map_err(|_| CryptoError::EncryptionFailed)?;
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| CryptoError::EncryptionFailed)?;

        let mut record = Vec::with_capacity(RECORD_NONCE_LEN + ciphertext.len());
        record.extend_from_slice(&nonce);
        record.extend_from_slice(&ciphertext);
        Ok(record)
    }

    fn decrypt(&self, record: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if record.len() < MIN_RECORD_LEN {
            return Err(CryptoError::RecordTooShort);
        }

        let (nonce, ciphertext) = record.split_at(RECORD_NONCE_LEN);
        let cipher =
            ChaCha20Poly1305::new_from_slice(&self.0).map_err(|_| CryptoError::DecryptionFailed)?;
        cipher
            .decrypt(
                Nonce::from_slice(nonce),
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| CryptoError::DecryptionFailed)
    }
}

/// Direction-separated request and response keys for one v2 session.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct DirectionalKeys {
    request: RecordKey,
    response: RecordKey,
}

impl DirectionalKeys {
    pub(crate) fn derive(session_master: &SessionMaster) -> Result<Self, CryptoError> {
        Ok(Self {
            request: RecordKey::derive(session_master.as_bytes(), REQUEST_KEY_INFO)?,
            response: RecordKey::derive(session_master.as_bytes(), RESPONSE_KEY_INFO)?,
        })
    }

    pub(crate) fn decrypt_request_record(
        &self,
        session_id: &Uuid,
        record: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.request
            .decrypt(record, &request_record_aad(session_id))
    }

    pub(crate) fn encrypt_unary_response_record(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response.encrypt(
            plaintext,
            &unary_response_record_aad(session_id, request_id),
        )
    }

    pub(crate) fn encrypt_stream_response_record(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        sequence: u64,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response.encrypt(
            plaintext,
            &stream_response_record_aad(session_id, request_id, sequence),
        )
    }

    #[cfg(test)]
    pub(crate) fn encrypt_request_record(
        &self,
        session_id: &Uuid,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.request
            .encrypt(plaintext, &request_record_aad(session_id))
    }

    #[cfg(test)]
    pub(crate) fn encrypt_request_record_with_nonce(
        &self,
        session_id: &Uuid,
        plaintext: &[u8],
        nonce: [u8; RECORD_NONCE_LEN],
    ) -> Result<Vec<u8>, CryptoError> {
        self.request
            .encrypt_with_nonce(plaintext, &request_record_aad(session_id), nonce)
    }

    #[cfg(test)]
    pub(crate) fn encrypt_unary_response_record_with_nonce(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        plaintext: &[u8],
        nonce: [u8; RECORD_NONCE_LEN],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response.encrypt_with_nonce(
            plaintext,
            &unary_response_record_aad(session_id, request_id),
            nonce,
        )
    }

    #[cfg(test)]
    pub(crate) fn encrypt_stream_response_record_with_nonce(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        sequence: u64,
        plaintext: &[u8],
        nonce: [u8; RECORD_NONCE_LEN],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response.encrypt_with_nonce(
            plaintext,
            &stream_response_record_aad(session_id, request_id, sequence),
            nonce,
        )
    }

    #[cfg(test)]
    pub(crate) fn decrypt_unary_response_record(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        record: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response
            .decrypt(record, &unary_response_record_aad(session_id, request_id))
    }

    #[cfg(test)]
    pub(crate) fn decrypt_stream_response_record(
        &self,
        session_id: &Uuid,
        request_id: &RequestId,
        sequence: u64,
        record: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        self.response.decrypt(
            record,
            &stream_response_record_aad(session_id, request_id, sequence),
        )
    }

    #[cfg(test)]
    pub(crate) fn request_key_bytes(&self) -> &[u8; KEY_LEN] {
        &self.request.0
    }

    #[cfg(test)]
    pub(crate) fn response_key_bytes(&self) -> &[u8; KEY_LEN] {
        &self.response.0
    }
}

impl fmt::Debug for DirectionalKeys {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DirectionalKeys([REDACTED])")
    }
}

/// Encrypts the fixed handshake payload under the HKDF-derived wrapping key.
pub(crate) fn encrypt_key_exchange_record(
    x25519_shared_secret: &[u8; KEY_LEN],
    payload: &HandshakePayload,
) -> Result<Vec<u8>, CryptoError> {
    let key = derive_handshake_key(x25519_shared_secret)?;
    key.encrypt(payload.as_bytes(), KEY_EXCHANGE_AAD)
}

#[cfg(test)]
pub(crate) fn encrypt_key_exchange_record_with_nonce(
    x25519_shared_secret: &[u8; KEY_LEN],
    payload: &HandshakePayload,
    nonce: [u8; RECORD_NONCE_LEN],
) -> Result<Vec<u8>, CryptoError> {
    let key = derive_handshake_key(x25519_shared_secret)?;
    key.encrypt_with_nonce(payload.as_bytes(), KEY_EXCHANGE_AAD, nonce)
}

#[cfg(test)]
pub(crate) fn decrypt_key_exchange_record(
    x25519_shared_secret: &[u8; KEY_LEN],
    record: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let key = derive_handshake_key(x25519_shared_secret)?;
    key.decrypt(record, KEY_EXCHANGE_AAD)
}

fn derive_handshake_key(x25519_shared_secret: &[u8; KEY_LEN]) -> Result<RecordKey, CryptoError> {
    if bool::from(x25519_shared_secret.ct_eq(&[0; KEY_LEN])) {
        return Err(CryptoError::NonContributorySharedSecret);
    }
    RecordKey::derive(x25519_shared_secret, HANDSHAKE_KEY_INFO)
}

pub(crate) fn encode_canonical_base64(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

pub(crate) fn decode_canonical_base64(encoded: &str) -> Result<Vec<u8>, CryptoError> {
    let decoded = STANDARD
        .decode(encoded)
        .map_err(|_| CryptoError::InvalidBase64)?;
    if STANDARD.encode(&decoded) != encoded {
        return Err(CryptoError::NonCanonicalBase64);
    }
    Ok(decoded)
}

pub(crate) fn request_record_aad(session_id: &Uuid) -> Vec<u8> {
    let mut aad = Vec::with_capacity(REQUEST_RECORD_AAD.len() + 1 + 16);
    aad.extend_from_slice(REQUEST_RECORD_AAD);
    aad.push(0);
    aad.extend_from_slice(session_id.as_bytes());
    aad
}

pub(crate) fn unary_response_record_aad(session_id: &Uuid, request_id: &RequestId) -> Vec<u8> {
    let mut aad = Vec::with_capacity(UNARY_RESPONSE_RECORD_AAD.len() + 1 + 16 + 16);
    aad.extend_from_slice(UNARY_RESPONSE_RECORD_AAD);
    aad.push(0);
    aad.extend_from_slice(session_id.as_bytes());
    aad.extend_from_slice(request_id.as_bytes());
    aad
}

pub(crate) fn stream_response_record_aad(
    session_id: &Uuid,
    request_id: &RequestId,
    sequence: u64,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(STREAM_RESPONSE_RECORD_AAD.len() + 1 + 16 + 16 + 8);
    aad.extend_from_slice(STREAM_RESPONSE_RECORD_AAD);
    aad.push(0);
    aad.extend_from_slice(session_id.as_bytes());
    aad.extend_from_slice(request_id.as_bytes());
    aad.extend_from_slice(&sequence.to_be_bytes());
    aad
}

fn fill_random(destination: &mut [u8]) -> Result<(), CryptoError> {
    getrandom::getrandom(destination).map_err(|_| CryptoError::RandomnessUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_payload_has_frozen_binary_layout() {
        let session_id = Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap();
        let master = SessionMaster::from_bytes([0x5a; KEY_LEN]);
        let expiry = 0x0102_0304_0506_0708;
        let payload = HandshakePayload::new(&session_id, &master, expiry);

        assert_eq!(payload.as_bytes().len(), 57);
        assert_eq!(payload.as_bytes()[0], 2);
        assert_eq!(&payload.as_bytes()[1..17], session_id.as_bytes());
        assert_eq!(&payload.as_bytes()[17..49], &[0x5a; KEY_LEN]);
        assert_eq!(
            &payload.as_bytes()[49..57],
            &0x0102_0304_0506_0708_u64.to_be_bytes()
        );
    }

    #[test]
    fn canonical_base64_rejects_alternate_text() {
        let bytes = b"transport-v2!";
        let canonical = encode_canonical_base64(bytes);
        assert_eq!(decode_canonical_base64(&canonical).unwrap(), bytes);

        let unpadded = canonical.trim_end_matches('=');
        assert_ne!(unpadded, canonical);
        assert!(matches!(
            decode_canonical_base64(unpadded),
            Err(CryptoError::InvalidBase64 | CryptoError::NonCanonicalBase64)
        ));
    }

    #[test]
    fn records_require_nonce_and_tag() {
        let master = SessionMaster::from_bytes([0x42; KEY_LEN]);
        let keys = DirectionalKeys::derive(&master).unwrap();
        let session_id = Uuid::nil();

        assert_eq!(
            keys.decrypt_request_record(&session_id, &[0; MIN_RECORD_LEN - 1]),
            Err(CryptoError::RecordTooShort)
        );
    }

    #[test]
    fn request_record_authenticates_session_and_direction() {
        let master = SessionMaster::from_bytes([0x24; KEY_LEN]);
        let keys = DirectionalKeys::derive(&master).unwrap();
        let session_id = Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap();
        let other_session = Uuid::parse_str("10112233-4455-6677-8899-aabbccddeeff").unwrap();
        let plaintext = br#"{"version":2}"#;
        let record = keys.encrypt_request_record(&session_id, plaintext).unwrap();

        assert_eq!(
            keys.decrypt_request_record(&session_id, &record).unwrap(),
            plaintext
        );
        assert_eq!(
            keys.decrypt_request_record(&other_session, &record),
            Err(CryptoError::DecryptionFailed)
        );
    }

    #[test]
    fn key_exchange_uses_nonce_record_shape_and_aad() {
        let shared_secret = [0x11; KEY_LEN];
        let session_id = Uuid::nil();
        let master = SessionMaster::from_bytes([0x22; KEY_LEN]);
        let payload = HandshakePayload::new(&session_id, &master, 1234);

        let record = encrypt_key_exchange_record(&shared_secret, &payload).unwrap();
        assert_eq!(record.len(), RECORD_NONCE_LEN + 57 + RECORD_TAG_LEN);
        assert_eq!(
            decrypt_key_exchange_record(&shared_secret, &record).unwrap(),
            payload.as_bytes()
        );

        let mut tampered = record;
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            decrypt_key_exchange_record(&shared_secret, &tampered),
            Err(CryptoError::DecryptionFailed)
        );
    }

    #[test]
    fn key_exchange_rejects_non_contributory_shared_secret() {
        let master = SessionMaster::from_bytes([0x22; KEY_LEN]);
        let payload = HandshakePayload::new(&Uuid::nil(), &master, 1234);

        assert_eq!(
            encrypt_key_exchange_record(&[0; KEY_LEN], &payload),
            Err(CryptoError::NonContributorySharedSecret)
        );
    }

    #[test]
    fn secret_types_have_redacted_debug_output() {
        let master = SessionMaster::from_bytes([0xaa; KEY_LEN]);
        let payload = HandshakePayload::new(&Uuid::nil(), &master, 0);
        let keys = DirectionalKeys::derive(&master).unwrap();

        assert_eq!(format!("{master:?}"), "SessionMaster([REDACTED])");
        assert_eq!(format!("{payload:?}"), "HandshakePayload([REDACTED])");
        assert_eq!(format!("{keys:?}"), "DirectionalKeys([REDACTED])");
    }
}
