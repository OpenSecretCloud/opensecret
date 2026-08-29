//! Cross-module transport-v2 vectors and invariants.

use serde::Deserialize;
use uuid::Uuid;

use super::{
    crypto::{
        decode_canonical_base64, decrypt_key_exchange_record, encode_canonical_base64,
        encrypt_key_exchange_record_with_nonce, request_record_aad, stream_response_record_aad,
        unary_response_record_aad, CryptoError, DirectionalKeys, HandshakePayload, SessionMaster,
    },
    envelope::{EnvelopeLimits, LogicalMethod, RequestEnvelope, RequestId, ResponseMode},
    maximum_accounted_memory_bytes, V2_MEMORY_BUDGET_BYTES,
};

#[derive(Deserialize)]
struct GoldenVectors {
    shared_secret_hex: String,
    session_master_hex: String,
    session_id: String,
    expires_at_unix_seconds: u64,
    request_id_hex: String,
    stream_sequence: u64,
    handshake: HandshakeVector,
    request: DirectionalVector,
    unary_response: DirectionalVector,
    stream_response: RecordVector,
    request_without_body_json: String,
    request_with_empty_body_json: String,
}

#[derive(Deserialize)]
struct HandshakeVector {
    aad_hex: String,
    nonce_hex: String,
    plaintext_hex: String,
    record_hex: String,
    record_base64: String,
}

#[derive(Deserialize)]
struct DirectionalVector {
    derived_key_hex: String,
    aad_hex: String,
    nonce_hex: String,
    plaintext_utf8: String,
    plaintext_hex: String,
    record_hex: String,
    record_base64: String,
}

#[derive(Deserialize)]
struct RecordVector {
    aad_hex: String,
    nonce_hex: String,
    plaintext_utf8: String,
    plaintext_hex: String,
    record_hex: String,
    record_base64: String,
}

fn vectors() -> GoldenVectors {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/transport-v2-golden-vectors.json"
    )))
    .unwrap()
}

fn fixed_hex<const N: usize>(encoded: &str) -> [u8; N] {
    hex::decode(encoded).unwrap().try_into().unwrap()
}

fn assert_record(record: &[u8], expected_hex: &str, expected_base64: &str) {
    assert_eq!(record, hex::decode(expected_hex).unwrap());
    assert_eq!(encode_canonical_base64(record), expected_base64);
    assert_eq!(decode_canonical_base64(expected_base64).unwrap(), record);
}

#[test]
fn golden_vectors_fix_key_derivation_aad_and_record_bytes() {
    let fixture = vectors();
    let shared_secret = fixed_hex::<32>(&fixture.shared_secret_hex);
    let master = SessionMaster::from_bytes(fixed_hex(&fixture.session_master_hex));
    let session_id = Uuid::parse_str(&fixture.session_id).unwrap();
    let request_id = RequestId::from_bytes(fixed_hex(&fixture.request_id_hex));

    let handshake = HandshakePayload::new(&session_id, &master, fixture.expires_at_unix_seconds);
    assert_eq!(
        handshake.as_bytes().as_slice(),
        hex::decode(&fixture.handshake.plaintext_hex).unwrap()
    );
    assert_eq!(
        b"opensecret/transport-v2/key-exchange",
        hex::decode(&fixture.handshake.aad_hex).unwrap().as_slice()
    );
    let handshake_record = encrypt_key_exchange_record_with_nonce(
        &shared_secret,
        &handshake,
        fixed_hex(&fixture.handshake.nonce_hex),
    )
    .unwrap();
    assert_record(
        &handshake_record,
        &fixture.handshake.record_hex,
        &fixture.handshake.record_base64,
    );
    assert_eq!(
        decrypt_key_exchange_record(&shared_secret, &handshake_record).unwrap(),
        handshake.as_bytes()
    );

    let keys = DirectionalKeys::derive(&master).unwrap();
    assert_eq!(
        keys.request_key_bytes(),
        &fixed_hex::<32>(&fixture.request.derived_key_hex)
    );
    assert_eq!(
        keys.response_key_bytes(),
        &fixed_hex::<32>(&fixture.unary_response.derived_key_hex)
    );

    assert_eq!(
        request_record_aad(&session_id),
        hex::decode(&fixture.request.aad_hex).unwrap()
    );
    let request_plaintext = fixture.request.plaintext_utf8.as_bytes();
    assert_eq!(
        request_plaintext,
        hex::decode(&fixture.request.plaintext_hex).unwrap()
    );
    let request_record = keys
        .encrypt_request_record_with_nonce(
            &session_id,
            request_plaintext,
            fixed_hex(&fixture.request.nonce_hex),
        )
        .unwrap();
    assert_record(
        &request_record,
        &fixture.request.record_hex,
        &fixture.request.record_base64,
    );
    assert_eq!(
        keys.decrypt_request_record(&session_id, &request_record)
            .unwrap(),
        request_plaintext
    );

    assert_eq!(
        unary_response_record_aad(&session_id, &request_id),
        hex::decode(&fixture.unary_response.aad_hex).unwrap()
    );
    let unary_plaintext = fixture.unary_response.plaintext_utf8.as_bytes();
    assert_eq!(
        unary_plaintext,
        hex::decode(&fixture.unary_response.plaintext_hex).unwrap()
    );
    let unary_record = keys
        .encrypt_unary_response_record_with_nonce(
            &session_id,
            &request_id,
            unary_plaintext,
            fixed_hex(&fixture.unary_response.nonce_hex),
        )
        .unwrap();
    assert_record(
        &unary_record,
        &fixture.unary_response.record_hex,
        &fixture.unary_response.record_base64,
    );
    assert_eq!(
        keys.decrypt_unary_response_record(&session_id, &request_id, &unary_record)
            .unwrap(),
        unary_plaintext
    );

    assert_eq!(
        stream_response_record_aad(&session_id, &request_id, fixture.stream_sequence),
        hex::decode(&fixture.stream_response.aad_hex).unwrap()
    );
    let stream_plaintext = fixture.stream_response.plaintext_utf8.as_bytes();
    assert_eq!(
        stream_plaintext,
        hex::decode(&fixture.stream_response.plaintext_hex).unwrap()
    );
    let stream_record = keys
        .encrypt_stream_response_record_with_nonce(
            &session_id,
            &request_id,
            fixture.stream_sequence,
            stream_plaintext,
            fixed_hex(&fixture.stream_response.nonce_hex),
        )
        .unwrap();
    assert_record(
        &stream_record,
        &fixture.stream_response.record_hex,
        &fixture.stream_response.record_base64,
    );
    assert_eq!(
        keys.decrypt_stream_response_record(
            &session_id,
            &request_id,
            fixture.stream_sequence,
            &stream_record,
        )
        .unwrap(),
        stream_plaintext
    );
}

#[test]
fn records_fail_closed_when_direction_or_binding_changes() {
    let fixture = vectors();
    let master = SessionMaster::from_bytes(fixed_hex(&fixture.session_master_hex));
    let keys = DirectionalKeys::derive(&master).unwrap();
    let session_id = Uuid::parse_str(&fixture.session_id).unwrap();
    let other_session_id = Uuid::from_bytes([0x55; 16]);
    let request_id = RequestId::from_bytes(fixed_hex(&fixture.request_id_hex));
    let other_request_id = RequestId::from_bytes([0x66; 16]);

    let request_record = hex::decode(&fixture.request.record_hex).unwrap();
    assert_eq!(
        keys.decrypt_request_record(&other_session_id, &request_record),
        Err(CryptoError::DecryptionFailed)
    );
    assert_eq!(
        keys.decrypt_unary_response_record(&session_id, &request_id, &request_record),
        Err(CryptoError::DecryptionFailed)
    );

    let unary_record = hex::decode(&fixture.unary_response.record_hex).unwrap();
    assert_eq!(
        keys.decrypt_request_record(&session_id, &unary_record),
        Err(CryptoError::DecryptionFailed)
    );
    assert_eq!(
        keys.decrypt_unary_response_record(&other_session_id, &request_id, &unary_record),
        Err(CryptoError::DecryptionFailed)
    );
    assert_eq!(
        keys.decrypt_unary_response_record(&session_id, &other_request_id, &unary_record),
        Err(CryptoError::DecryptionFailed)
    );

    let stream_record = hex::decode(&fixture.stream_response.record_hex).unwrap();
    assert_eq!(
        keys.decrypt_stream_response_record(
            &session_id,
            &request_id,
            fixture.stream_sequence + 1,
            &stream_record,
        ),
        Err(CryptoError::DecryptionFailed)
    );

    let mut tampered_request = request_record;
    *tampered_request.last_mut().unwrap() ^= 1;
    assert_eq!(
        keys.decrypt_request_record(&session_id, &tampered_request),
        Err(CryptoError::DecryptionFailed)
    );

    let mut tampered_stream = stream_record;
    *tampered_stream.last_mut().unwrap() ^= 1;
    assert_eq!(
        keys.decrypt_stream_response_record(
            &session_id,
            &request_id,
            fixture.stream_sequence,
            &tampered_stream,
        ),
        Err(CryptoError::DecryptionFailed)
    );
}

#[test]
fn envelope_preserves_no_body_distinct_from_an_explicit_empty_body() {
    let fixture = vectors();
    let limits = EnvelopeLimits::DEFAULT;

    let without_body =
        RequestEnvelope::from_json_slice(fixture.request_without_body_json.as_bytes(), &limits)
            .unwrap();
    assert_eq!(without_body.response_mode, ResponseMode::Unary);
    assert_eq!(without_body.request.method, LogicalMethod::Get);
    assert!(without_body.request.body_base64.is_none());

    let with_empty_body =
        RequestEnvelope::from_json_slice(fixture.request_with_empty_body_json.as_bytes(), &limits)
            .unwrap();
    assert_eq!(with_empty_body.response_mode, ResponseMode::Unary);
    assert_eq!(with_empty_body.request.method, LogicalMethod::Post);
    assert_eq!(
        with_empty_body
            .request
            .body_base64
            .as_ref()
            .unwrap()
            .as_slice(),
        b""
    );
}

#[test]
fn maximum_accounted_core_state_stays_within_the_fixed_memory_budget() {
    const MIB: usize = 1024 * 1024;

    assert_eq!(maximum_accounted_memory_bytes(), 172 * MIB);
    assert!(maximum_accounted_memory_bytes() <= V2_MEMORY_BUDGET_BYTES);
}
