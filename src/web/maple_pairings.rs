//! Enclave-mediated, one-way Maple installation pairing.
//!
//! The service authenticates and signs the controller-to-host authorization,
//! while the host remains the final authority: approval only moves a request
//! to `awaiting_host_commit`. The host must durably install the authorization
//! in its local allowlist before it signs `confirm`; only that second mutation
//! makes the controller-visible service state `active`.

use axum::{
    extract::State, middleware::from_fn_with_state, routing::post, Extension, Json, Router,
};
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine as _,
};
use chrono::{DateTime, TimeZone, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use hmac::{Hmac, Mac};
use serde::{de::DeserializeOwned, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use subtle::ConstantTimeEq;
use uuid::Uuid;

use crate::{
    db::DBError,
    encrypt::{decrypt_aead_v1, derive_key, encrypt_aead_v1, CanonicalBytes, EncryptError},
    jwt::{AuthContext, AuthMethod},
    models::{
        maple_devices::{MapleDevice, MapleDeviceListAuthorization},
        maple_pairing_db::{
            MaplePairing, MaplePairingApproval, MaplePairingAuthorization,
            MaplePairingConfirmation, MaplePairingCreateDeviceContext, MaplePairingCreateMaterial,
            MaplePairingCreateMaterializationContext, MaplePairingCursor,
            MaplePairingMaterializationError, MaplePairingOperationKind, MaplePairingRevocation,
            MaplePairingRevocationAck, MaplePairingRevocationContext,
            MaplePairingRevocationMaterial, MaplePairingRole as DbPairingRole,
            MaplePairingState as DbPairingState, NewMaplePairingRequest,
            StoredMaplePairingPayloadV1, MAPLE_PAIRING_PAYLOAD_VERSION_V1,
            MAPLE_PAIRING_RECEIPT_VERSION_V1,
        },
        maple_pairings::{
            AckMaplePairingRevocationRequest, ApproveMaplePairingRequest,
            ConfirmMaplePairingRequest, CreateMaplePairingRequest,
            ListMaplePairingRevocationsRequest, ListMaplePairingRevocationsResponse,
            ListMaplePairingsRequest, ListMaplePairingsResponse, MapleDeviceClaimV1,
            MaplePairAuthorizationV1, MaplePairRequestTicketV1, MaplePairRevocationV1,
            MaplePairingDirection, MaplePairingIdentityAlgorithm, MaplePairingIssuerKeySetV1,
            MaplePairingMutationResponse, MaplePairingRole, MaplePairingState,
            MaplePairingStatusRequest, MaplePairingStatusResponse, MaplePairingStatusV1,
            MapleRevocationStreamCheckpointV1, MapleRevocationStreamEventV1, MapleRevocationSyncV1,
            RevokeMaplePairingRequest, MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS,
        },
        users::User,
    },
    web::{
        encryption_middleware::{decrypt_request_bounded, encrypt_response, EncryptedResponse},
        maple_devices::{decrypt_device_response, MapleDeviceResponse},
    },
    ApiError, AppState,
};

type HmacSha256 = Hmac<Sha256>;

const PROTOCOL_VERSION_V1: u16 = 1;
const MAX_PAIRING_REQUEST_PLAINTEXT_BYTES: usize = 16 * 1024;
const MAX_PAIRING_REQUEST_ENCRYPTED_BYTES: usize = 32 * 1024;

const PAIR_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-pair-record-key.v1";
const PAIR_PAYLOAD_DOMAIN: &str = "os.maple-pair-record.v1";
const PAIR_OPERATION_MAC_KEY_INFO: &[u8] = b"os.maple-pair-operation-mac-key.v1";
const PAIR_OPERATION_MAC_DOMAIN: &str = "os.maple-pair-operation.v1";
const PAIR_RECEIPT_KEY_INFO: &[u8] = b"os.maple-pair-receipt-key.v1";
const PAIR_RECEIPT_DOMAIN: &str = "os.maple-pair-receipt.v1";
const PAIR_CURSOR_KEY_INFO: &[u8] = b"os.maple-pair-cursor-mac-key.v1";
const PAIR_CURSOR_DOMAIN: &str = "os.maple-pair-cursor.v1";
const REVOCATION_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-pair-revocation-record-key.v1";
const REVOCATION_PAYLOAD_DOMAIN: &str = "os.maple-pair-revocation-record.v1";

#[derive(Clone, Copy)]
struct PairPayloadContext {
    account_id: Uuid,
    project_id: i32,
    pairing_request_id: Uuid,
    pair_id: Uuid,
    pairing_incarnation: u64,
    revocation_stream_id: Option<Uuid>,
    revocation_stream_generation: Option<u64>,
    payload_version: i16,
}

#[derive(Clone, Copy)]
struct ReceiptContext {
    account_id: Uuid,
    project_id: i32,
    actor_registration_id: Uuid,
    operation_id: Uuid,
    operation_kind: i16,
    pair_id: Uuid,
    pairing_revision: i64,
    receipt_version: i16,
}

#[derive(Clone, Copy)]
struct RevocationPayloadContext {
    account_id: Uuid,
    project_id: i32,
    host_registration_id: Uuid,
    revocation_stream_id: Uuid,
    revocation_stream_generation: u64,
    event_id: Uuid,
    issuer_sequence: u64,
    pair_id: Uuid,
    pairing_incarnation: u64,
    payload_version: i16,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/protected/maple/pairings/request",
            post(request_pairing).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    CreateMaplePairingRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/list",
            post(list_pairings).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    ListMaplePairingsRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/status",
            post(pairing_status).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    MaplePairingStatusRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/approve",
            post(approve_pairing).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    ApproveMaplePairingRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/confirm",
            post(confirm_pairing).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    ConfirmMaplePairingRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/revoke",
            post(revoke_pairing).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    RevokeMaplePairingRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/revocations/list",
            post(list_revocations).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    ListMaplePairingRevocationsRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/pairings/revocations/ack",
            post(ack_revocation).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    AckMaplePairingRevocationRequest,
                    MAX_PAIRING_REQUEST_ENCRYPTED_BYTES,
                    MAX_PAIRING_REQUEST_PLAINTEXT_BYTES,
                >,
            )),
        )
        .with_state(app_state)
}

fn pairing_authorization(
    state: &AppState,
    user: &User,
    auth_context: &AuthContext,
) -> MaplePairingAuthorization {
    MaplePairingAuthorization {
        user_id: user.uuid,
        project_id: user.project_id,
        auth_credential_kind: match auth_context.method {
            AuthMethod::Password => "password",
            AuthMethod::OAuth => "oauth",
        }
        .to_owned(),
        auth_binding: auth_context.auth_binding,
        enclave_key: state.enclave_key.clone(),
    }
}

fn device_list_authorization(
    state: &AppState,
    user: &User,
    auth_context: &AuthContext,
) -> MapleDeviceListAuthorization {
    MapleDeviceListAuthorization {
        user_id: user.uuid,
        project_id: user.project_id,
        auth_credential_kind: match auth_context.method {
            AuthMethod::Password => "password",
            AuthMethod::OAuth => "oauth",
        }
        .to_owned(),
        auth_binding: auth_context.auth_binding,
        enclave_key: state.enclave_key.clone(),
    }
}

fn load_devices(
    state: &AppState,
    user: &User,
    auth_context: &AuthContext,
) -> Result<Vec<(MapleDevice, MapleDeviceResponse)>, ApiError> {
    let page = state
        .db
        .list_maple_devices(
            device_list_authorization(state, user, auth_context),
            33,
            None,
        )
        .map_err(map_pairing_db_error)?;
    page.devices
        .into_iter()
        .map(|row| {
            let response = decrypt_device_response(
                &state.enclave_key,
                user.uuid,
                user.project_id,
                row.clone(),
            )?;
            Ok((row, response))
        })
        .collect()
}

fn require_device(
    devices: &[(MapleDevice, MapleDeviceResponse)],
    registration_id: Uuid,
) -> Result<&(MapleDevice, MapleDeviceResponse), ApiError> {
    devices
        .iter()
        .find(|(_, device)| device.registration_id == registration_id)
        .ok_or(ApiError::NotFound)
}

fn pairing_participants<'a>(
    devices: &'a [(MapleDevice, MapleDeviceResponse)],
    row: &MaplePairing,
) -> Result<(&'a MapleDeviceResponse, &'a MapleDeviceResponse), ApiError> {
    let controller = devices
        .iter()
        .find(|(device, _)| device.id == row.controller_maple_device_id)
        .map(|(_, response)| response)
        .ok_or(ApiError::InternalServerError)?;
    let host = devices
        .iter()
        .find(|(device, _)| device.id == row.host_maple_device_id)
        .map(|(_, response)| response)
        .ok_or(ApiError::InternalServerError)?;
    Ok((controller, host))
}

fn device_claim(device: &MapleDeviceResponse) -> MapleDeviceClaimV1 {
    MapleDeviceClaimV1 {
        registration_id: device.registration_id,
        device_id: device.device_id,
        installation_id: device.installation_id,
        identity_algorithm: MaplePairingIdentityAlgorithm::Ed25519,
        identity_public_key: device.identity_public_key.clone(),
        endpoint_id: device.iroh_endpoint_id.clone(),
        endpoint_epoch: device.endpoint_epoch,
    }
}

fn create_materialization_device_response(
    enclave_key: &[u8],
    account_id: Uuid,
    internal_project_id: i32,
    device: MaplePairingCreateDeviceContext,
    observed_at: DateTime<Utc>,
) -> Result<MapleDeviceResponse, MaplePairingMaterializationError> {
    let endpoint_epoch = device
        .endpoint_epoch
        .try_into()
        .map_err(|_| MaplePairingMaterializationError)?;
    decrypt_device_response(
        enclave_key,
        account_id,
        internal_project_id,
        MapleDevice {
            id: 1,
            uuid: device.registration_id,
            user_id: account_id,
            project_id: internal_project_id,
            device_id: device.device_id,
            installation_id: device.installation_id,
            identity_mac: device.identity_mac,
            endpoint_epoch,
            payload_version: device.payload_version,
            payload_enc: device.payload_enc,
            record_mac: device.record_mac,
            revision: device.device_revision,
            registered_at: observed_at,
            updated_at: observed_at,
        },
    )
    .map_err(|_| MaplePairingMaterializationError)
}

fn verifying_key(device: &MapleDeviceResponse) -> Result<VerifyingKey, ApiError> {
    let bytes = STANDARD
        .decode(&device.identity_public_key)
        .map_err(|_| ApiError::InternalServerError)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| ApiError::InternalServerError)?;
    VerifyingKey::from_bytes(&bytes).map_err(|_| ApiError::InternalServerError)
}

fn verify_device_signature(
    device: &MapleDeviceResponse,
    transcript: &[u8],
    signature: &str,
) -> Result<(), ApiError> {
    let bytes = STANDARD
        .decode(signature)
        .map_err(|_| ApiError::BadRequest)?;
    if STANDARD.encode(&bytes) != signature {
        return Err(ApiError::BadRequest);
    }
    let bytes: [u8; 64] = bytes.try_into().map_err(|_| ApiError::BadRequest)?;
    verifying_key(device)?
        .verify_strict(transcript, &Signature::from_bytes(&bytes))
        .map_err(|_| ApiError::BadRequest)
}

fn hmac_sha256(enclave_key: &[u8], key_info: &[u8], body: &[u8]) -> Result<[u8; 32], EncryptError> {
    let key = derive_key(enclave_key, key_info)?;
    let mut mac =
        HmacSha256::new_from_slice(&key).map_err(|_| EncryptError::KeyDerivationFailed)?;
    mac.update(body);
    Ok(mac.finalize().into_bytes().into())
}

pub(crate) fn request_operation_mac(
    enclave_key: &[u8],
    transcript: &[u8],
    signature: &str,
) -> Result<[u8; 32], ApiError> {
    let signature = STANDARD
        .decode(signature)
        .map_err(|_| ApiError::BadRequest)?;
    let mut body = CanonicalBytes::new(PAIR_OPERATION_MAC_DOMAIN);
    body.append_bytes(transcript).append_bytes(&signature);
    hmac_sha256(enclave_key, PAIR_OPERATION_MAC_KEY_INFO, &body.into_bytes())
        .map_err(|_| ApiError::InternalServerError)
}

fn pair_payload_aad(context: PairPayloadContext) -> Result<Vec<u8>, EncryptError> {
    let incarnation: i64 = context
        .pairing_incarnation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    let mut aad = CanonicalBytes::new(PAIR_PAYLOAD_DOMAIN);
    aad.append_uuid(context.account_id)
        .append_i32(context.project_id)
        .append_uuid(context.pairing_request_id)
        .append_uuid(context.pair_id)
        .append_i64(incarnation)
        .append_bool(context.revocation_stream_id.is_some());
    if let Some(stream_id) = context.revocation_stream_id {
        if stream_id.is_nil() {
            return Err(EncryptError::BadData);
        }
        aad.append_uuid(stream_id);
    }
    aad.append_bool(context.revocation_stream_generation.is_some());
    if let Some(generation) = context.revocation_stream_generation {
        let generation: i64 = generation.try_into().map_err(|_| EncryptError::BadData)?;
        if generation <= 0 {
            return Err(EncryptError::BadData);
        }
        aad.append_i64(generation);
    }
    if context.revocation_stream_id.is_some() != context.revocation_stream_generation.is_some() {
        return Err(EncryptError::BadData);
    }
    aad.append_i16(context.payload_version);
    Ok(aad.into_bytes())
}

fn encrypt_pair_payload(
    enclave_key: &[u8],
    payload: &StoredMaplePairingPayloadV1,
    context: PairPayloadContext,
) -> Result<Vec<u8>, ApiError> {
    let plaintext = serde_json::to_vec(payload).map_err(|_| ApiError::InternalServerError)?;
    let key = derive_key(enclave_key, PAIR_PAYLOAD_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    encrypt_aead_v1(
        &key,
        &plaintext,
        &pair_payload_aad(context).map_err(|_| ApiError::InternalServerError)?,
    )
    .map_err(|_| ApiError::InternalServerError)
}

fn decrypt_pair_payload(
    enclave_key: &[u8],
    encrypted: &[u8],
    context: PairPayloadContext,
) -> Result<StoredMaplePairingPayloadV1, ApiError> {
    let key = derive_key(enclave_key, PAIR_PAYLOAD_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    let plaintext = decrypt_aead_v1(
        &key,
        encrypted,
        &pair_payload_aad(context).map_err(|_| ApiError::InternalServerError)?,
    )
    .map_err(|_| ApiError::InternalServerError)?;
    serde_json::from_slice(&plaintext).map_err(|_| ApiError::InternalServerError)
}

#[cfg(test)]
pub(crate) fn decrypt_pair_payload_for_test(
    enclave_key: &[u8],
    row: &MaplePairing,
) -> Result<StoredMaplePairingPayloadV1, ApiError> {
    decrypt_pair_payload(
        enclave_key,
        &row.payload_enc,
        PairPayloadContext {
            account_id: row.user_id,
            project_id: row.project_id,
            pairing_request_id: row.pairing_request_id,
            pair_id: row.uuid,
            pairing_incarnation: u64::try_from(row.pairing_incarnation)
                .map_err(|_| ApiError::InternalServerError)?,
            revocation_stream_id: row.revocation_stream_id,
            revocation_stream_generation: row
                .revocation_stream_generation
                .map(u64::try_from)
                .transpose()
                .map_err(|_| ApiError::InternalServerError)?,
            payload_version: row.payload_version,
        },
    )
}

#[cfg(test)]
pub(crate) fn encrypt_pair_payload_for_test(
    enclave_key: &[u8],
    row: &MaplePairing,
    revocation_stream_id: Uuid,
    revocation_stream_generation: u64,
    payload: &StoredMaplePairingPayloadV1,
) -> Result<Vec<u8>, ApiError> {
    encrypt_pair_payload(
        enclave_key,
        payload,
        PairPayloadContext {
            account_id: row.user_id,
            project_id: row.project_id,
            pairing_request_id: row.pairing_request_id,
            pair_id: row.uuid,
            pairing_incarnation: u64::try_from(row.pairing_incarnation)
                .map_err(|_| ApiError::InternalServerError)?,
            revocation_stream_id: Some(revocation_stream_id),
            revocation_stream_generation: Some(revocation_stream_generation),
            payload_version: row.payload_version,
        },
    )
}

fn receipt_aad(context: ReceiptContext) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(PAIR_RECEIPT_DOMAIN);
    aad.append_uuid(context.account_id)
        .append_i32(context.project_id)
        .append_uuid(context.actor_registration_id)
        .append_uuid(context.operation_id)
        .append_i16(context.operation_kind)
        .append_uuid(context.pair_id)
        .append_i64(context.pairing_revision)
        .append_i16(context.receipt_version);
    aad.into_bytes()
}

fn encrypt_receipt<T: Serialize>(
    enclave_key: &[u8],
    receipt: &T,
    context: ReceiptContext,
) -> Result<Vec<u8>, ApiError> {
    let plaintext = serde_json::to_vec(receipt).map_err(|_| ApiError::InternalServerError)?;
    let key = derive_key(enclave_key, PAIR_RECEIPT_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    encrypt_aead_v1(&key, &plaintext, &receipt_aad(context))
        .map_err(|_| ApiError::InternalServerError)
}

fn decrypt_receipt<T: DeserializeOwned>(
    enclave_key: &[u8],
    encrypted: &[u8],
    context: ReceiptContext,
) -> Result<T, ApiError> {
    let key = derive_key(enclave_key, PAIR_RECEIPT_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    let plaintext = decrypt_aead_v1(&key, encrypted, &receipt_aad(context))
        .map_err(|_| ApiError::InternalServerError)?;
    serde_json::from_slice(&plaintext).map_err(|_| ApiError::InternalServerError)
}

fn revocation_payload_aad(context: RevocationPayloadContext) -> Result<Vec<u8>, EncryptError> {
    let sequence: i64 = context
        .issuer_sequence
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    let incarnation: i64 = context
        .pairing_incarnation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    let stream_generation: i64 = context
        .revocation_stream_generation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    if context.revocation_stream_id.is_nil() || stream_generation <= 0 {
        return Err(EncryptError::BadData);
    }
    let mut aad = CanonicalBytes::new(REVOCATION_PAYLOAD_DOMAIN);
    aad.append_uuid(context.account_id)
        .append_i32(context.project_id)
        .append_uuid(context.host_registration_id)
        .append_uuid(context.revocation_stream_id)
        .append_i64(stream_generation)
        .append_uuid(context.event_id)
        .append_i64(sequence)
        .append_uuid(context.pair_id)
        .append_i64(incarnation)
        .append_i16(context.payload_version);
    Ok(aad.into_bytes())
}

fn decrypt_revocation_payload(
    enclave_key: &[u8],
    encrypted: &[u8],
    context: RevocationPayloadContext,
) -> Result<MaplePairRevocationV1, ApiError> {
    let key = derive_key(enclave_key, REVOCATION_PAYLOAD_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    let plaintext = decrypt_aead_v1(
        &key,
        encrypted,
        &revocation_payload_aad(context).map_err(|_| ApiError::InternalServerError)?,
    )
    .map_err(|_| ApiError::InternalServerError)?;
    serde_json::from_slice(&plaintext).map_err(|_| ApiError::InternalServerError)
}

fn encode_pair_cursor(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    actor_registration_id: Uuid,
    role: &str,
    states: &[&str],
    pair_id: Uuid,
) -> Result<String, ApiError> {
    let body = pair_cursor_body(
        account_id,
        project_id,
        actor_registration_id,
        role,
        states,
        pair_id,
    );
    let mac = hmac_sha256(enclave_key, PAIR_CURSOR_KEY_INFO, &body)
        .map_err(|_| ApiError::InternalServerError)?;
    let mut bytes = Vec::with_capacity(48);
    bytes.extend_from_slice(pair_id.as_bytes());
    bytes.extend_from_slice(&mac);
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn decode_pair_cursor(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    actor_registration_id: Uuid,
    role: &str,
    states: &[&str],
    cursor: &str,
) -> Result<Uuid, ApiError> {
    if cursor.len() > 512 || cursor.len() != 64 || !cursor.is_ascii() {
        return Err(ApiError::BadRequest);
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| ApiError::BadRequest)?;
    if URL_SAFE_NO_PAD.encode(&bytes) != cursor || bytes.len() != 48 {
        return Err(ApiError::BadRequest);
    }
    let pair_id = Uuid::from_slice(&bytes[..16]).map_err(|_| ApiError::BadRequest)?;
    if pair_id.is_nil() {
        return Err(ApiError::BadRequest);
    }
    let body = pair_cursor_body(
        account_id,
        project_id,
        actor_registration_id,
        role,
        states,
        pair_id,
    );
    let expected = hmac_sha256(enclave_key, PAIR_CURSOR_KEY_INFO, &body)
        .map_err(|_| ApiError::InternalServerError)?;
    if !bool::from(expected.as_slice().ct_eq(&bytes[16..])) {
        return Err(ApiError::BadRequest);
    }
    Ok(pair_id)
}

fn pair_cursor_body(
    account_id: Uuid,
    project_id: i32,
    actor_registration_id: Uuid,
    role: &str,
    states: &[&str],
    pair_id: Uuid,
) -> Vec<u8> {
    let mut body = CanonicalBytes::new(PAIR_CURSOR_DOMAIN);
    body.append_uuid(account_id)
        .append_i32(project_id)
        .append_uuid(actor_registration_id)
        .append_str(role)
        .append_u16(states.len() as u16);
    for state in states {
        body.append_str(state);
    }
    body.append_uuid(pair_id);
    body.into_bytes()
}

fn canonical_b64_32(value: &str) -> Result<[u8; 32], ApiError> {
    let bytes = STANDARD.decode(value).map_err(|_| ApiError::BadRequest)?;
    if STANDARD.encode(&bytes) != value {
        return Err(ApiError::BadRequest);
    }
    bytes.try_into().map_err(|_| ApiError::BadRequest)
}

fn unix_ms(timestamp: DateTime<Utc>) -> i64 {
    timestamp.timestamp_millis()
}

fn from_unix_ms(timestamp: i64) -> Result<DateTime<Utc>, ApiError> {
    Utc.timestamp_millis_opt(timestamp)
        .single()
        .ok_or(ApiError::InternalServerError)
}

fn now_millis() -> Result<DateTime<Utc>, ApiError> {
    from_unix_ms(Utc::now().timestamp_millis())
}

fn monotonic_millis(
    observed_unix_ms: i64,
    predecessor_unix_ms: i64,
) -> Result<DateTime<Utc>, ApiError> {
    from_unix_ms(observed_unix_ms.max(predecessor_unix_ms))
}

/// Return a millisecond-serializable timestamp that is never earlier than a
/// database lifecycle predecessor. PostgreSQL retains microseconds, so a
/// direct `timestamp_millis()` truncation occasionally needs one millisecond
/// of ceiling before the value is sent back for a transactional comparison.
fn monotonic_millis_at_or_after(
    observed_unix_ms: i64,
    predecessor: DateTime<Utc>,
) -> Result<DateTime<Utc>, ApiError> {
    let truncated = predecessor.timestamp_millis();
    let predecessor_unix_ms = if from_unix_ms(truncated)? < predecessor {
        truncated
            .checked_add(1)
            .ok_or(ApiError::InternalServerError)?
    } else {
        truncated
    };
    from_unix_ms(observed_unix_ms.max(predecessor_unix_ms))
}

fn wire_input<T>(
    result: Result<T, crate::models::maple_pairings::MaplePairingWireError>,
) -> Result<T, ApiError> {
    result.map_err(|_| ApiError::BadRequest)
}

fn stored_wire<T>(
    result: Result<T, crate::models::maple_pairings::MaplePairingWireError>,
) -> Result<T, ApiError> {
    result.map_err(|_| ApiError::InternalServerError)
}

fn db_role(role: MaplePairingRole) -> DbPairingRole {
    match role {
        MaplePairingRole::Controller => DbPairingRole::Controller,
        MaplePairingRole::Host => DbPairingRole::Host,
    }
}

fn db_state(state: MaplePairingState) -> DbPairingState {
    match state {
        MaplePairingState::Pending => DbPairingState::Pending,
        MaplePairingState::AwaitingHostCommit => DbPairingState::AwaitingHostCommit,
        MaplePairingState::Active => DbPairingState::Active,
        MaplePairingState::Expired => DbPairingState::Expired,
        MaplePairingState::Revoked => DbPairingState::Revoked,
    }
}

fn wire_state(state: i16) -> Result<MaplePairingState, ApiError> {
    match DbPairingState::try_from(state).map_err(|_| ApiError::InternalServerError)? {
        DbPairingState::Pending => Ok(MaplePairingState::Pending),
        DbPairingState::AwaitingHostCommit => Ok(MaplePairingState::AwaitingHostCommit),
        DbPairingState::Active => Ok(MaplePairingState::Active),
        DbPairingState::Expired => Ok(MaplePairingState::Expired),
        DbPairingState::Revoked => Ok(MaplePairingState::Revoked),
    }
}

fn role_name(role: MaplePairingRole) -> &'static str {
    match role {
        MaplePairingRole::Controller => "controller",
        MaplePairingRole::Host => "host",
    }
}

fn state_name(state: MaplePairingState) -> &'static str {
    match state {
        MaplePairingState::Pending => "pending",
        MaplePairingState::AwaitingHostCommit => "awaiting_host_commit",
        MaplePairingState::Active => "active",
        MaplePairingState::Expired => "expired",
        MaplePairingState::Revoked => "revoked",
    }
}

fn pair_context(user: &User, row: &MaplePairing) -> Result<PairPayloadContext, ApiError> {
    Ok(PairPayloadContext {
        account_id: user.uuid,
        project_id: user.project_id,
        pairing_request_id: row.pairing_request_id,
        pair_id: row.uuid,
        pairing_incarnation: row
            .pairing_incarnation
            .try_into()
            .map_err(|_| ApiError::InternalServerError)?,
        revocation_stream_id: row.revocation_stream_id,
        revocation_stream_generation: row
            .revocation_stream_generation
            .map(u64::try_from)
            .transpose()
            .map_err(|_| ApiError::InternalServerError)?,
        payload_version: row.payload_version,
    })
}

fn verify_ticket_signature(
    ticket: &MaplePairRequestTicketV1,
    keyset: &MaplePairingIssuerKeySetV1,
) -> Result<(), ApiError> {
    stored_wire(ticket.validate())?;
    stored_wire(keyset.verify(
        &ticket.issuer_key_id,
        &stored_wire(ticket.transcript())?,
        &ticket.issuer_signature,
    ))
}

fn validate_device_claim_binding(
    claim: &MapleDeviceClaimV1,
    device: &MapleDeviceResponse,
) -> Result<(), ApiError> {
    let current = device_claim(device);
    // The signed artifact deliberately freezes the endpoint generation used
    // at pairing time. A later monotonic device-record refresh must not make a
    // durable authorization unreadable, while rollback or identity changes
    // remain corruption.
    if claim.registration_id != current.registration_id
        || claim.device_id != current.device_id
        || claim.installation_id != current.installation_id
        || claim.identity_algorithm != current.identity_algorithm
        || claim.identity_public_key != current.identity_public_key
        || claim.endpoint_id != current.endpoint_id
        || claim.endpoint_epoch > current.endpoint_epoch
    {
        return Err(ApiError::InternalServerError);
    }
    Ok(())
}

fn pairing_authorization_is_visible(
    state: MaplePairingState,
    viewer_role: MaplePairingRole,
    activated: bool,
) -> bool {
    !matches!(
        state,
        MaplePairingState::Pending | MaplePairingState::Expired
    ) && !(viewer_role == MaplePairingRole::Controller
        && (state == MaplePairingState::AwaitingHostCommit
            || (state == MaplePairingState::Revoked && !activated)))
}

#[allow(clippy::too_many_arguments)] // Explicit trust inputs prevent ambient viewer/scope authority.
fn pairing_status_from_row(
    enclave_key: &[u8],
    user: &User,
    project_client_id: Uuid,
    row: &MaplePairing,
    viewer_role: MaplePairingRole,
    controller: &MapleDeviceResponse,
    host: &MapleDeviceResponse,
    keyset: &MaplePairingIssuerKeySetV1,
    trusted_now_unix_ms: i64,
) -> Result<MaplePairingStatusV1, ApiError> {
    let payload = decrypt_pair_payload(enclave_key, &row.payload_enc, pair_context(user, row)?)?;
    let ticket = &payload.request_ticket;
    verify_ticket_signature(ticket, keyset)?;
    let incarnation: u64 = row
        .pairing_incarnation
        .try_into()
        .map_err(|_| ApiError::InternalServerError)?;
    if row.user_id != user.uuid
        || row.project_id != user.project_id
        || row.uuid != ticket.pair_id
        || row.pairing_request_id != ticket.pairing_request_id
        || ticket.subject_account_id != user.uuid
        || ticket.subject_project_id != project_client_id
        || ticket.direction != MaplePairingDirection::ControllerToHost
        || ticket.execution_target_id != host.registration_id
        || ticket.pairing_incarnation != incarnation
        || ticket.created_at_unix_ms != unix_ms(row.created_at)
        || ticket.expires_at_unix_ms != unix_ms(row.expires_at)
        || ticket.issuer_key_id != row.ticket_issuer_key_id
    {
        return Err(ApiError::InternalServerError);
    }
    validate_device_claim_binding(&ticket.controller, controller)?;
    validate_device_claim_binding(&ticket.host, host)?;

    let wire_state = wire_state(row.state)?;
    let authorization = payload.pair_authorization.as_ref();
    let revocation = payload.revocation.as_ref();
    match wire_state {
        MaplePairingState::Pending => {
            stored_wire(ticket.verify_unexpired(
                keyset,
                trusted_now_unix_ms,
                MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS,
            ))?;
            if row.revision != 1
                || row.approved_at.is_some()
                || row.activated_at.is_some()
                || row.revoked_at.is_some()
                || authorization.is_some()
                || revocation.is_some()
            {
                return Err(ApiError::InternalServerError);
            }
        }
        MaplePairingState::Expired => {
            if row.revision != 2
                || row.approved_at.is_some()
                || row.activated_at.is_some()
                || row.revoked_at.is_some()
                || authorization.is_some()
                || revocation.is_some()
            {
                return Err(ApiError::InternalServerError);
            }
        }
        MaplePairingState::AwaitingHostCommit
        | MaplePairingState::Active
        | MaplePairingState::Revoked => {
            let authorization = authorization.ok_or(ApiError::InternalServerError)?;
            stored_wire(authorization.verify(keyset))?;
            if authorization.subject_account_id != user.uuid
                || authorization.subject_project_id != project_client_id
                || authorization.pair_id != row.uuid
                || authorization.pairing_request_id != row.pairing_request_id
                || authorization.pairing_incarnation != incarnation
                || row.revocation_stream_id != Some(authorization.revocation_stream_id)
                || row.revocation_stream_generation
                    != i64::try_from(authorization.revocation_stream_generation).ok()
                || row.approved_at.map(unix_ms) != Some(authorization.approved_at_unix_ms)
                || row.authorization_issuer_key_id.as_deref()
                    != Some(authorization.issuer_key_id.as_str())
            {
                return Err(ApiError::InternalServerError);
            }
            // Ticket currentness is intentionally not required after approval;
            // the durable issuer authorization outlives its short-lived ticket.
            let verified_at_approval = stored_wire(ticket.verify_unexpired(
                keyset,
                authorization.approved_at_unix_ms,
                MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS,
            ))?;
            stored_wire(authorization.verify_against_ticket(keyset, &verified_at_approval))?;

            match wire_state {
                MaplePairingState::AwaitingHostCommit => {
                    if row.revision != 2
                        || row.activated_at.is_some()
                        || row.revoked_at.is_some()
                        || revocation.is_some()
                    {
                        return Err(ApiError::InternalServerError);
                    }
                }
                MaplePairingState::Active => {
                    if row.revision != 3
                        || row.activated_at.is_none_or(|activated_at| {
                            row.approved_at
                                .is_none_or(|approved_at| activated_at < approved_at)
                        })
                        || row.revoked_at.is_some()
                        || revocation.is_some()
                    {
                        return Err(ApiError::InternalServerError);
                    }
                }
                MaplePairingState::Revoked => {
                    let revocation = revocation.ok_or(ApiError::InternalServerError)?;
                    stored_wire(revocation.verify_against_authorization(keyset, authorization))?;
                    let expected_revision = if row.activated_at.is_some() { 4 } else { 3 };
                    if row.revision != expected_revision
                        || row.revoked_at.map(unix_ms) != Some(revocation.revoked_at_unix_ms)
                        || row.revocation_issuer_key_id.as_deref()
                            != Some(revocation.issuer_key_id.as_str())
                        || row.revoked_at.is_none_or(|revoked_at| {
                            row.approved_at
                                .is_none_or(|approved_at| revoked_at < approved_at)
                                || row
                                    .activated_at
                                    .is_some_and(|activated_at| revoked_at < activated_at)
                        })
                    {
                        return Err(ApiError::InternalServerError);
                    }
                }
                MaplePairingState::Pending | MaplePairingState::Expired => unreachable!(),
            }
        }
    }

    let visible_authorization =
        if pairing_authorization_is_visible(wire_state, viewer_role, row.activated_at.is_some()) {
            payload.pair_authorization
        } else {
            None
        };
    let visible_revocation = if wire_state == MaplePairingState::Revoked {
        payload.revocation
    } else {
        None
    };
    let status = MaplePairingStatusV1 {
        pairing_request_id: row.pairing_request_id,
        pair_id: row.uuid,
        state: wire_state,
        revision: row.revision,
        pairing_incarnation: incarnation,
        revocation_stream_id: row.revocation_stream_id,
        revocation_stream_generation: row
            .revocation_stream_generation
            .map(u64::try_from)
            .transpose()
            .map_err(|_| ApiError::InternalServerError)?,
        direction: MaplePairingDirection::ControllerToHost,
        execution_target_id: host.registration_id,
        controller_registration_id: controller.registration_id,
        host_registration_id: host.registration_id,
        created_at_unix_ms: unix_ms(row.created_at),
        expires_at_unix_ms: unix_ms(row.expires_at),
        approved_at_unix_ms: row.approved_at.map(unix_ms),
        activated_at_unix_ms: row.activated_at.map(unix_ms),
        revoked_at_unix_ms: row.revoked_at.map(unix_ms),
        request_ticket: Some(payload.request_ticket),
        pair_authorization: visible_authorization,
        revocation: visible_revocation,
    };
    stored_wire(status.validate_revocation_stream_shape())?;
    Ok(status)
}

fn validate_common_assertions(
    protocol_version: u16,
    asserted_account_id: Uuid,
    asserted_project_id: Uuid,
    user: &User,
    project_client_id: Uuid,
) -> Result<(), ApiError> {
    if protocol_version != PROTOCOL_VERSION_V1
        || asserted_account_id != user.uuid
        || asserted_project_id != project_client_id
    {
        return Err(ApiError::BadRequest);
    }
    Ok(())
}

fn require_pairing_keyset(state: &AppState) -> Result<&Arc<MaplePairingIssuerKeySetV1>, ApiError> {
    let keyset = state
        .maple_pairing_issuer_keyset
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable)?;
    keyset
        .validate()
        .map_err(|_| ApiError::ServiceUnavailable)?;
    Ok(keyset)
}

fn require_pairing_crypto(
    state: &AppState,
) -> Result<
    (
        &Arc<dyn crate::models::maple_pairings::MaplePairingIssuer>,
        &Arc<MaplePairingIssuerKeySetV1>,
    ),
    ApiError,
> {
    let keyset = require_pairing_keyset(state)?;
    let issuer = state
        .maple_pairing_issuer
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable)?;
    let issuer_is_trusted = keyset
        .contains_issuer(issuer.as_ref())
        .map_err(|_| ApiError::ServiceUnavailable)?;
    if !issuer_is_trusted {
        return Err(ApiError::ServiceUnavailable);
    }
    Ok((issuer, keyset))
}

fn map_pairing_db_error(error: DBError) -> ApiError {
    match error {
        DBError::MaplePairingAuthorityBusy
        | DBError::MaplePairingAuthorityCapacityExceeded
        | DBError::MaplePairingAuthorityDeletionBlocked
        | DBError::MaplePairingAuthorityCorrupt => ApiError::from(error),
        DBError::MaplePairingConflict
        | DBError::MaplePairingLimitExceeded
        | DBError::MaplePairingOperationLimitExceeded => ApiError::Conflict,
        DBError::MaplePairingResetClearRequired => ApiError::MaplePairingResetClearRequired,
        DBError::MaplePairingNotFound => ApiError::NotFound,
        DBError::StaleCredentialState => ApiError::InvalidJwt,
        DBError::MaplePairingCorrupt => ApiError::InternalServerError,
        _ => {
            tracing::error!("Maple pairing database operation failed");
            ApiError::InternalServerError
        }
    }
}

fn replay_mutation_if_present<T: DeserializeOwned>(
    state: &AppState,
    user: &User,
    auth_context: &AuthContext,
    actor_registration_id: Uuid,
    operation_id: Uuid,
    operation_kind: MaplePairingOperationKind,
    request_mac: &[u8; 32],
) -> Result<Option<T>, ApiError> {
    let Some(receipt) = state
        .db
        .replay_maple_pairing_operation(
            pairing_authorization(state, user, auth_context),
            actor_registration_id,
            operation_id,
            operation_kind,
            request_mac.to_vec(),
        )
        .map_err(map_pairing_db_error)?
    else {
        return Ok(None);
    };
    decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: operation_kind.as_db(),
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )
    .map(Some)
}

fn replay_reset_clear_ack_if_present(
    state: &AppState,
    user: &User,
    auth_context: &AuthContext,
    request: &AckMaplePairingRevocationRequest,
    request_mac: &[u8; 32],
) -> Result<Option<crate::models::maple_pairings::AckMaplePairingRevocationResponse>, ApiError> {
    let Some(receipt) = state
        .db
        .replay_maple_reset_clear_ack(
            pairing_authorization(state, user, auth_context),
            request.host_registration_id,
            request.operation_id,
            request_mac.to_vec(),
        )
        .map_err(map_pairing_db_error)?
    else {
        return Ok(None);
    };
    let response: crate::models::maple_pairings::AckMaplePairingRevocationResponse =
        decrypt_receipt(
            &state.enclave_key,
            &receipt.receipt_enc,
            ReceiptContext {
                account_id: user.uuid,
                project_id: user.project_id,
                actor_registration_id: request.host_registration_id,
                operation_id: receipt.operation_id,
                operation_kind: MaplePairingOperationKind::Ack.as_db(),
                pair_id: receipt.pair_id,
                pairing_revision: receipt.pairing_revision,
                receipt_version: receipt.receipt_version,
            },
        )?;
    let keyset = require_pairing_keyset(state)?;
    stored_wire(response.verify_against_request(request, keyset))?;
    Ok(Some(response))
}

// Handler bodies follow below; each successful response remains encrypted by
// the attested session and each mutation is exact-operation idempotent in DB.

pub(crate) fn materialize_maple_pairing_create(
    enclave_key: &[u8],
    issuer: &dyn crate::models::maple_pairings::MaplePairingIssuer,
    internal_project_id: i32,
    context: MaplePairingCreateMaterializationContext,
) -> Result<MaplePairingCreateMaterial, MaplePairingMaterializationError> {
    let expected_request_mac = request_operation_mac(
        enclave_key,
        &context
            .create_request
            .transcript()
            .map_err(|_| MaplePairingMaterializationError)?,
        &context.create_request.signature,
    )
    .map_err(|_| MaplePairingMaterializationError)?;
    if !bool::from(expected_request_mac.ct_eq(&context.request_mac)) {
        return Err(MaplePairingMaterializationError);
    }
    context
        .create_request
        .verify_signature()
        .map_err(|_| MaplePairingMaterializationError)?;
    let controller = create_materialization_device_response(
        enclave_key,
        context.account_id,
        internal_project_id,
        context.controller,
        context.created_at,
    )?;
    let host = create_materialization_device_response(
        enclave_key,
        context.account_id,
        internal_project_id,
        context.host,
        context.created_at,
    )?;
    let controller = device_claim(&controller);
    let host = device_claim(&host);
    canonical_b64_32(&context.create_request.pairing_request_nonce)
        .map_err(|_| MaplePairingMaterializationError)?;
    let pairing_request_id = Uuid::new_v4();
    let pair_id = Uuid::new_v4();
    let ticket = crate::models::maple_pairings::sign_pair_request_ticket(
        MaplePairRequestTicketV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: context.account_id,
            subject_project_id: context.subject_project_id,
            pairing_request_id,
            pair_id,
            direction: context.create_request.direction,
            execution_target_id: context.create_request.execution_target_id,
            controller: controller.clone(),
            host: host.clone(),
            pairing_request_nonce: context.create_request.pairing_request_nonce.clone(),
            controller_request_operation_id: context.operation_id,
            controller_request_digest: STANDARD.encode(
                context
                    .create_request
                    .digest()
                    .map_err(|_| MaplePairingMaterializationError)?,
            ),
            controller_request_signature: context.create_request.signature.clone(),
            pairing_incarnation: context.pairing_incarnation,
            protocol_min: context.create_request.protocol_min,
            protocol_max: context.create_request.protocol_max,
            created_at_unix_ms: unix_ms(context.created_at),
            expires_at_unix_ms: unix_ms(context.expires_at),
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        issuer,
    )
    .map_err(|_| MaplePairingMaterializationError)?;
    let response = MaplePairingMutationResponse {
        protocol_version: PROTOCOL_VERSION_V1,
        operation_id: context.operation_id,
        pairing: MaplePairingStatusV1 {
            pairing_request_id,
            pair_id,
            state: MaplePairingState::Pending,
            revision: 1,
            pairing_incarnation: context.pairing_incarnation,
            revocation_stream_id: None,
            revocation_stream_generation: None,
            direction: context.create_request.direction,
            execution_target_id: context.create_request.execution_target_id,
            controller_registration_id: controller.registration_id,
            host_registration_id: host.registration_id,
            created_at_unix_ms: unix_ms(context.created_at),
            expires_at_unix_ms: unix_ms(context.expires_at),
            approved_at_unix_ms: None,
            activated_at_unix_ms: None,
            revoked_at_unix_ms: None,
            request_ticket: Some(ticket.clone()),
            pair_authorization: None,
            revocation: None,
        },
    };
    Ok(MaplePairingCreateMaterial {
        request_ticket: ticket,
        response,
    })
}

async fn request_pairing(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<CreateMaplePairingRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<MaplePairingMutationResponse>>, ApiError> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    wire_input(request.verify_signature())?;
    let transcript = wire_input(request.transcript())?;
    let request_mac = request_operation_mac(&state.enclave_key, &transcript, &request.signature)?;
    let (issuer, keyset) = require_pairing_crypto(&state)?;
    let enclave_key = state.enclave_key.clone();
    let internal_project_id = user.project_id;
    let issuer = Arc::clone(issuer);
    let materialize = move |context: MaplePairingCreateMaterializationContext| {
        materialize_maple_pairing_create(
            &enclave_key,
            issuer.as_ref(),
            internal_project_id,
            context,
        )
    };
    let receipt = state
        .db
        .create_maple_pairing(
            NewMaplePairingRequest {
                authorization: pairing_authorization(&state, &user, &auth_context),
                subject_project_id: project.client_id,
                operation_id: request.operation_id,
                request_mac: request_mac.to_vec(),
                create_request: request.clone(),
                controller_registration_id: request.controller_registration_id,
                expected_controller_endpoint_epoch: request.controller_endpoint_epoch,
                host_registration_id: request.host_registration_id,
                expected_host_endpoint_epoch: request.host_endpoint_epoch,
            },
            keyset.as_ref(),
            &materialize,
        )
        .map_err(map_pairing_db_error)?;
    let response = decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.controller_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: 1,
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )?;
    encrypt_response(&state, &session_id, &response).await
}

async fn list_pairings(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<ListMaplePairingsRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<
    Json<EncryptedResponse<crate::models::maple_pairings::ListMaplePairingsResponse>>,
    ApiError,
> {
    let keyset = require_pairing_keyset(&state)?;
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (_, actor) = require_device(&devices, request.actor_registration_id)?;
    verify_device_signature(
        actor,
        &wire_input(request.transcript())?,
        &request.signature,
    )?;
    let state_names: Vec<&str> = request.states.iter().copied().map(state_name).collect();
    let after = request
        .cursor
        .as_deref()
        .map(|cursor| {
            decode_pair_cursor(
                &state.enclave_key,
                user.uuid,
                user.project_id,
                request.actor_registration_id,
                role_name(request.role),
                &state_names,
                cursor,
            )
            .map(|pair_id| MaplePairingCursor { pair_id })
        })
        .transpose()?;
    let limit = wire_input(request.effective_limit())?;
    let mut rows = state
        .db
        .list_maple_pairings(
            pairing_authorization(&state, &user, &auth_context),
            request.actor_registration_id,
            db_role(request.role),
            request.states.iter().copied().map(db_state).collect(),
            i64::from(limit) + 1,
            after,
        )
        .map_err(map_pairing_db_error)?;
    let has_more = rows.len() > usize::from(limit);
    if has_more {
        rows.truncate(usize::from(limit));
    }
    let next_cursor = if has_more {
        rows.last()
            .map(|row| {
                encode_pair_cursor(
                    &state.enclave_key,
                    user.uuid,
                    user.project_id,
                    request.actor_registration_id,
                    role_name(request.role),
                    &state_names,
                    row.uuid,
                )
            })
            .transpose()?
    } else {
        None
    };
    let trusted_now_unix_ms = Utc::now().timestamp_millis();
    let pairings = rows
        .iter()
        .map(|row| {
            let (controller, host) = pairing_participants(&devices, row)?;
            pairing_status_from_row(
                &state.enclave_key,
                &user,
                project.client_id,
                row,
                request.role,
                controller,
                host,
                keyset,
                trusted_now_unix_ms,
            )
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    encrypt_response(
        &state,
        &session_id,
        &ListMaplePairingsResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            query_id: request.query_id,
            role: request.role,
            pairings,
            next_cursor,
            has_more,
        },
    )
    .await
}

async fn pairing_status(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<MaplePairingStatusRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<
    Json<EncryptedResponse<crate::models::maple_pairings::MaplePairingStatusResponse>>,
    ApiError,
> {
    let keyset = require_pairing_keyset(&state)?;
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (actor_row, actor) = require_device(&devices, request.actor_registration_id)?;
    verify_device_signature(
        actor,
        &wire_input(request.transcript())?,
        &request.signature,
    )?;
    let row = state
        .db
        .get_maple_pairing(
            pairing_authorization(&state, &user, &auth_context),
            request.actor_registration_id,
            request.pair_id,
        )
        .map_err(map_pairing_db_error)?
        .ok_or(ApiError::NotFound)?;
    let viewer_role = if row.controller_maple_device_id == actor_row.id {
        MaplePairingRole::Controller
    } else if row.host_maple_device_id == actor_row.id {
        MaplePairingRole::Host
    } else {
        return Err(ApiError::NotFound);
    };
    let (controller, host) = pairing_participants(&devices, &row)?;
    let pairing = pairing_status_from_row(
        &state.enclave_key,
        &user,
        project.client_id,
        &row,
        viewer_role,
        controller,
        host,
        keyset,
        Utc::now().timestamp_millis(),
    )?;
    encrypt_response(
        &state,
        &session_id,
        &MaplePairingStatusResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            query_id: request.query_id,
            pairing,
        },
    )
    .await
}

async fn approve_pairing(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<ApproveMaplePairingRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<MaplePairingMutationResponse>>, ApiError> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (host_row, host) = require_device(&devices, request.host_registration_id)?;
    let transcript = wire_input(request.transcript())?;
    verify_device_signature(host, &transcript, &request.signature)?;
    let request_mac = request_operation_mac(&state.enclave_key, &transcript, &request.signature)?;
    if let Some(response) = replay_mutation_if_present::<MaplePairingMutationResponse>(
        &state,
        &user,
        &auth_context,
        request.host_registration_id,
        request.operation_id,
        MaplePairingOperationKind::Approve,
        &request_mac,
    )? {
        return encrypt_response(&state, &session_id, &response).await;
    }
    let (issuer, keyset) = require_pairing_crypto(&state)?;
    let row = state
        .db
        .get_maple_pairing(
            pairing_authorization(&state, &user, &auth_context),
            request.host_registration_id,
            request.pair_id,
        )
        .map_err(map_pairing_db_error)?
        .ok_or(ApiError::NotFound)?;
    if row.host_maple_device_id != host_row.id
        || row.pairing_request_id != request.pairing_request_id
        || u64::try_from(row.pairing_incarnation).ok() != Some(request.pairing_incarnation)
    {
        return Err(ApiError::NotFound);
    }
    let authorization = pairing_authorization(&state, &user, &auth_context);
    let (
        payload_version,
        payload_enc,
        receipt_enc,
        approved_at,
        authorization_issuer_key_id,
        pair_authorization_digest,
    ) = if wire_state(row.state)? == MaplePairingState::Pending
        && row.revision == request.expected_pairing_revision
    {
        let (controller, bound_host) = pairing_participants(&devices, &row)?;
        let pending = pairing_status_from_row(
            &state.enclave_key,
            &user,
            project.client_id,
            &row,
            MaplePairingRole::Host,
            controller,
            bound_host,
            keyset,
            Utc::now().timestamp_millis(),
        )?;
        let ticket = pending
            .request_ticket
            .ok_or(ApiError::InternalServerError)?;
        // Keep the signed lifecycle monotonic even if the application
        // wall clock steps backward after ticket issuance. The DB still
        // independently bounds this instant against its trusted clock.
        let approved_at =
            monotonic_millis(Utc::now().timestamp_millis(), ticket.created_at_unix_ms)?;
        let verified_ticket = wire_input(ticket.verify_unexpired(
            keyset,
            unix_ms(approved_at),
            MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS,
        ))?;
        if STANDARD.encode(stored_wire(ticket.digest())?) != request.request_ticket_digest
            || request.approved_protocol_min != ticket.protocol_min
            || request.approved_protocol_max != ticket.protocol_max
        {
            return Err(ApiError::BadRequest);
        }
        let pair_authorization = crate::models::maple_pairings::sign_pair_authorization(
            MaplePairAuthorizationV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                subject_account_id: ticket.subject_account_id,
                subject_project_id: ticket.subject_project_id,
                pairing_request_id: ticket.pairing_request_id,
                pair_id: ticket.pair_id,
                direction: ticket.direction,
                execution_target_id: ticket.execution_target_id,
                controller: ticket.controller.clone(),
                host: ticket.host.clone(),
                pairing_request_nonce: ticket.pairing_request_nonce.clone(),
                controller_request_operation_id: ticket.controller_request_operation_id,
                controller_request_digest: ticket.controller_request_digest.clone(),
                controller_request_signature: ticket.controller_request_signature.clone(),
                request_ticket_digest: request.request_ticket_digest.clone(),
                host_approval_operation_id: request.operation_id,
                host_approval_expected_pairing_revision: request.expected_pairing_revision,
                host_approval_nonce: request.host_approval_nonce.clone(),
                host_approval_digest: STANDARD.encode(wire_input(request.digest())?),
                host_approval_signature: request.signature.clone(),
                pairing_incarnation: ticket.pairing_incarnation,
                revocation_stream_id: request.revocation_stream_id,
                revocation_stream_generation: request.revocation_stream_generation,
                protocol_min: request.approved_protocol_min,
                protocol_max: request.approved_protocol_max,
                approved_at_unix_ms: unix_ms(approved_at),
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            issuer.as_ref(),
        )
        .map_err(|_| ApiError::ServiceUnavailable)?;
        stored_wire(pair_authorization.verify_against_ticket(keyset, &verified_ticket))?;
        let payload = StoredMaplePairingPayloadV1 {
            request_ticket: ticket.clone(),
            pair_authorization: Some(pair_authorization.clone()),
            revocation: None,
        };
        let payload_enc = encrypt_pair_payload(
            &state.enclave_key,
            &payload,
            PairPayloadContext {
                account_id: user.uuid,
                project_id: user.project_id,
                pairing_request_id: row.pairing_request_id,
                pair_id: row.uuid,
                pairing_incarnation: request.pairing_incarnation,
                revocation_stream_id: Some(request.revocation_stream_id),
                revocation_stream_generation: Some(request.revocation_stream_generation),
                payload_version: MAPLE_PAIRING_PAYLOAD_VERSION_V1,
            },
        )?;
        let response = MaplePairingMutationResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            operation_id: request.operation_id,
            pairing: MaplePairingStatusV1 {
                pairing_request_id: row.pairing_request_id,
                pair_id: row.uuid,
                state: MaplePairingState::AwaitingHostCommit,
                revision: 2,
                pairing_incarnation: request.pairing_incarnation,
                revocation_stream_id: Some(request.revocation_stream_id),
                revocation_stream_generation: Some(request.revocation_stream_generation),
                direction: ticket.direction,
                execution_target_id: ticket.execution_target_id,
                controller_registration_id: ticket.controller.registration_id,
                host_registration_id: ticket.host.registration_id,
                created_at_unix_ms: ticket.created_at_unix_ms,
                expires_at_unix_ms: ticket.expires_at_unix_ms,
                approved_at_unix_ms: Some(unix_ms(approved_at)),
                activated_at_unix_ms: None,
                revoked_at_unix_ms: None,
                request_ticket: Some(ticket),
                pair_authorization: Some(pair_authorization.clone()),
                revocation: None,
            },
        };
        let receipt_enc = encrypt_receipt(
            &state.enclave_key,
            &response,
            ReceiptContext {
                account_id: user.uuid,
                project_id: user.project_id,
                actor_registration_id: request.host_registration_id,
                operation_id: request.operation_id,
                operation_kind: 2,
                pair_id: row.uuid,
                pairing_revision: 2,
                receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            },
        )?;
        (
            MAPLE_PAIRING_PAYLOAD_VERSION_V1,
            payload_enc,
            receipt_enc,
            approved_at,
            pair_authorization.issuer_key_id.clone(),
            stored_wire(pair_authorization.digest())?.to_vec(),
        )
    } else {
        // Exact-operation retries are resolved before state/CAS checks by
        // the DB. These placeholders are never persisted for a replay.
        (
            row.payload_version,
            row.payload_enc.clone(),
            Vec::new(),
            now_millis()?,
            String::new(),
            Vec::new(),
        )
    };
    let receipt = state
        .db
        .approve_maple_pairing(MaplePairingApproval {
            authorization,
            operation_id: request.operation_id,
            request_mac: request_mac.to_vec(),
            host_registration_id: request.host_registration_id,
            pairing_request_id: request.pairing_request_id,
            pair_id: request.pair_id,
            expected_pairing_revision: request.expected_pairing_revision,
            pairing_incarnation: request.pairing_incarnation,
            expected_revocation_stream_id: request.revocation_stream_id,
            expected_revocation_stream_generation: request.revocation_stream_generation,
            authorization_issuer_key_id,
            pair_authorization_digest,
            payload_version,
            payload_enc,
            receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            receipt_enc,
            approved_at,
        })
        .map_err(map_pairing_db_error)?;
    let response = decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.host_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: 2,
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )?;
    encrypt_response(&state, &session_id, &response).await
}

async fn confirm_pairing(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<ConfirmMaplePairingRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<MaplePairingMutationResponse>>, ApiError> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (host_row, host) = require_device(&devices, request.host_registration_id)?;
    let transcript = wire_input(request.transcript())?;
    verify_device_signature(host, &transcript, &request.signature)?;
    let request_mac = request_operation_mac(&state.enclave_key, &transcript, &request.signature)?;
    if let Some(response) = replay_mutation_if_present::<MaplePairingMutationResponse>(
        &state,
        &user,
        &auth_context,
        request.host_registration_id,
        request.operation_id,
        MaplePairingOperationKind::Confirm,
        &request_mac,
    )? {
        return encrypt_response(&state, &session_id, &response).await;
    }
    let keyset = require_pairing_keyset(&state)?;
    let row = state
        .db
        .get_maple_pairing(
            pairing_authorization(&state, &user, &auth_context),
            request.host_registration_id,
            request.pair_id,
        )
        .map_err(map_pairing_db_error)?
        .ok_or(ApiError::NotFound)?;
    if row.host_maple_device_id != host_row.id
        || row.pairing_request_id != request.pairing_request_id
        || u64::try_from(row.pairing_incarnation).ok() != Some(request.pairing_incarnation)
    {
        return Err(ApiError::NotFound);
    }
    let (payload_version, payload_enc, receipt_enc, activated_at) = if wire_state(row.state)?
        == MaplePairingState::AwaitingHostCommit
        && row.revision == request.expected_pairing_revision
    {
        let (controller, bound_host) = pairing_participants(&devices, &row)?;
        let awaiting = pairing_status_from_row(
            &state.enclave_key,
            &user,
            project.client_id,
            &row,
            MaplePairingRole::Host,
            controller,
            bound_host,
            keyset,
            Utc::now().timestamp_millis(),
        )?;
        let ticket = awaiting
            .request_ticket
            .ok_or(ApiError::InternalServerError)?;
        let pair_authorization = awaiting
            .pair_authorization
            .ok_or(ApiError::InternalServerError)?;
        if STANDARD.encode(stored_wire(pair_authorization.digest())?)
            != request.pair_authorization_digest
        {
            return Err(ApiError::BadRequest);
        }
        // Protocol precondition: Maple signs and sends this confirmation
        // only after its local allowlist CAS has committed durably. A
        // service receipt proves service activation, not local persistence.
        // Activation follows the issuer-signed approval timestamp. Clamp
        // rather than making a harmless wall-clock regression strand a host
        // after it has already durably committed its local allowlist.
        let activated_at = monotonic_millis(
            Utc::now().timestamp_millis(),
            pair_authorization.approved_at_unix_ms,
        )?;
        let payload = StoredMaplePairingPayloadV1 {
            request_ticket: ticket.clone(),
            pair_authorization: Some(pair_authorization.clone()),
            revocation: None,
        };
        let payload_enc = encrypt_pair_payload(
            &state.enclave_key,
            &payload,
            PairPayloadContext {
                account_id: user.uuid,
                project_id: user.project_id,
                pairing_request_id: row.pairing_request_id,
                pair_id: row.uuid,
                pairing_incarnation: request.pairing_incarnation,
                revocation_stream_id: row.revocation_stream_id,
                revocation_stream_generation: row
                    .revocation_stream_generation
                    .map(u64::try_from)
                    .transpose()
                    .map_err(|_| ApiError::InternalServerError)?,
                payload_version: MAPLE_PAIRING_PAYLOAD_VERSION_V1,
            },
        )?;
        let response = MaplePairingMutationResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            operation_id: request.operation_id,
            pairing: MaplePairingStatusV1 {
                pairing_request_id: row.pairing_request_id,
                pair_id: row.uuid,
                state: MaplePairingState::Active,
                revision: 3,
                pairing_incarnation: request.pairing_incarnation,
                revocation_stream_id: Some(pair_authorization.revocation_stream_id),
                revocation_stream_generation: Some(pair_authorization.revocation_stream_generation),
                direction: ticket.direction,
                execution_target_id: ticket.execution_target_id,
                controller_registration_id: ticket.controller.registration_id,
                host_registration_id: ticket.host.registration_id,
                created_at_unix_ms: ticket.created_at_unix_ms,
                expires_at_unix_ms: ticket.expires_at_unix_ms,
                approved_at_unix_ms: Some(pair_authorization.approved_at_unix_ms),
                activated_at_unix_ms: Some(unix_ms(activated_at)),
                revoked_at_unix_ms: None,
                request_ticket: Some(ticket),
                pair_authorization: Some(pair_authorization),
                revocation: None,
            },
        };
        let receipt_enc = encrypt_receipt(
            &state.enclave_key,
            &response,
            ReceiptContext {
                account_id: user.uuid,
                project_id: user.project_id,
                actor_registration_id: request.host_registration_id,
                operation_id: request.operation_id,
                operation_kind: 3,
                pair_id: row.uuid,
                pairing_revision: 3,
                receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            },
        )?;
        (
            MAPLE_PAIRING_PAYLOAD_VERSION_V1,
            payload_enc,
            receipt_enc,
            activated_at,
        )
    } else {
        (
            row.payload_version,
            row.payload_enc.clone(),
            Vec::new(),
            now_millis()?,
        )
    };
    let receipt = state
        .db
        .confirm_maple_pairing(MaplePairingConfirmation {
            authorization: pairing_authorization(&state, &user, &auth_context),
            operation_id: request.operation_id,
            request_mac: request_mac.to_vec(),
            host_registration_id: request.host_registration_id,
            pairing_request_id: request.pairing_request_id,
            pair_id: request.pair_id,
            expected_pairing_revision: request.expected_pairing_revision,
            pairing_incarnation: request.pairing_incarnation,
            payload_version,
            payload_enc,
            receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            receipt_enc,
            activated_at,
        })
        .map_err(map_pairing_db_error)?;
    let response = decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.host_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: 3,
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )?;
    encrypt_response(&state, &session_id, &response).await
}

async fn revoke_pairing(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<RevokeMaplePairingRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<MaplePairingMutationResponse>>, ApiError> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (actor_row, actor) = require_device(&devices, request.actor_registration_id)?;
    let transcript = wire_input(request.transcript())?;
    verify_device_signature(actor, &transcript, &request.signature)?;
    let request_mac = request_operation_mac(&state.enclave_key, &transcript, &request.signature)?;
    if let Some(response) = replay_mutation_if_present::<MaplePairingMutationResponse>(
        &state,
        &user,
        &auth_context,
        request.actor_registration_id,
        request.operation_id,
        MaplePairingOperationKind::Revoke,
        &request_mac,
    )? {
        return encrypt_response(&state, &session_id, &response).await;
    }
    let (issuer, keyset) = require_pairing_crypto(&state)?;
    let row = state
        .db
        .get_maple_pairing(
            pairing_authorization(&state, &user, &auth_context),
            request.actor_registration_id,
            request.pair_id,
        )
        .map_err(map_pairing_db_error)?
        .ok_or(ApiError::NotFound)?;
    let actor_matches = match request.actor_role {
        MaplePairingRole::Controller => row.controller_maple_device_id == actor_row.id,
        MaplePairingRole::Host => row.host_maple_device_id == actor_row.id,
    };
    if !actor_matches
        || row.pairing_request_id != request.pairing_request_id
        || u64::try_from(row.pairing_incarnation).ok() != Some(request.pairing_incarnation)
        || row.revocation_stream_id != Some(request.revocation_stream_id)
        || row.revocation_stream_generation
            != i64::try_from(request.revocation_stream_generation).ok()
    {
        return Err(ApiError::NotFound);
    }
    let (controller, host) = pairing_participants(&devices, &row)?;
    let current = pairing_status_from_row(
        &state.enclave_key,
        &user,
        project.client_id,
        &row,
        MaplePairingRole::Host,
        controller,
        host,
        keyset,
        Utc::now().timestamp_millis(),
    )?;
    let ticket = current
        .request_ticket
        .clone()
        .ok_or(ApiError::InternalServerError)?;
    let pair_authorization = current.pair_authorization.clone().ok_or_else(|| {
        if current.state == MaplePairingState::Pending
            || current.state == MaplePairingState::Expired
        {
            ApiError::Conflict
        } else {
            ApiError::InternalServerError
        }
    })?;
    let issuer = Arc::clone(issuer);
    let keyset_for_material = Arc::clone(keyset);
    let request_for_material = request.clone();
    let ticket_for_material = ticket.clone();
    let authorization_for_material = pair_authorization.clone();
    let activated_at_unix_ms = current.activated_at_unix_ms;
    let materialize = move |context: MaplePairingRevocationContext| {
        let event_id = Uuid::new_v4();
        let revocation = crate::models::maple_pairings::sign_pair_revocation(
            MaplePairRevocationV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                event_id,
                subject_account_id: authorization_for_material.subject_account_id,
                subject_project_id: authorization_for_material.subject_project_id,
                recipient_host_registration_id: authorization_for_material.host.registration_id,
                issuer_sequence: context.issuer_sequence,
                revocation_stream_id: context.revocation_stream_id,
                revocation_stream_generation: context.revocation_stream_generation,
                pairing_request_id: context.pairing_request_id,
                pair_id: context.pair_id,
                direction: authorization_for_material.direction,
                execution_target_id: authorization_for_material.execution_target_id,
                controller: authorization_for_material.controller.clone(),
                host: authorization_for_material.host.clone(),
                pairing_incarnation: context.pairing_incarnation,
                pair_authorization_digest: STANDARD.encode(
                    authorization_for_material
                        .digest()
                        .map_err(|_| MaplePairingMaterializationError)?,
                ),
                revoked_by_registration_id: request_for_material.actor_registration_id,
                revoked_by_role: request_for_material.actor_role,
                reason_code: request_for_material.reason_code.clone(),
                revoked_at_unix_ms: unix_ms(context.revoked_at),
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            issuer.as_ref(),
        )
        .map_err(|_| MaplePairingMaterializationError)?;
        revocation
            .verify_against_authorization(&keyset_for_material, &authorization_for_material)
            .map_err(|_| MaplePairingMaterializationError)?;
        let response = MaplePairingMutationResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            operation_id: request_for_material.operation_id,
            pairing: MaplePairingStatusV1 {
                pairing_request_id: context.pairing_request_id,
                pair_id: context.pair_id,
                state: MaplePairingState::Revoked,
                revision: context.target_revision,
                pairing_incarnation: context.pairing_incarnation,
                revocation_stream_id: Some(context.revocation_stream_id),
                revocation_stream_generation: Some(context.revocation_stream_generation),
                direction: authorization_for_material.direction,
                execution_target_id: authorization_for_material.execution_target_id,
                controller_registration_id: authorization_for_material.controller.registration_id,
                host_registration_id: authorization_for_material.host.registration_id,
                created_at_unix_ms: ticket_for_material.created_at_unix_ms,
                expires_at_unix_ms: ticket_for_material.expires_at_unix_ms,
                approved_at_unix_ms: Some(authorization_for_material.approved_at_unix_ms),
                activated_at_unix_ms,
                revoked_at_unix_ms: Some(unix_ms(context.revoked_at)),
                request_ticket: Some(ticket_for_material.clone()),
                // The controller does not receive usable authority unless the
                // host's durable-install confirmation reached Active. A host
                // revoking its own precommit authorization still sees it so
                // it can reconcile and remove the local allowlist entry.
                pair_authorization: if request_for_material.actor_role
                    == MaplePairingRole::Controller
                    && activated_at_unix_ms.is_none()
                {
                    None
                } else {
                    Some(authorization_for_material.clone())
                },
                revocation: Some(revocation.clone()),
            },
        };
        Ok(MaplePairingRevocationMaterial {
            request_ticket: ticket_for_material.clone(),
            pair_authorization: authorization_for_material.clone(),
            revocation,
            response,
        })
    };
    let receipt = state
        .db
        .revoke_maple_pairing(
            MaplePairingRevocation {
                authorization: pairing_authorization(&state, &user, &auth_context),
                revoke_request: request.clone(),
                operation_id: request.operation_id,
                request_mac: request_mac.to_vec(),
                actor_registration_id: request.actor_registration_id,
                actor_role: db_role(request.actor_role),
                pairing_request_id: request.pairing_request_id,
                pair_id: request.pair_id,
                expected_pairing_revision: request.expected_pairing_revision,
                pairing_incarnation: request.pairing_incarnation,
                expected_revocation_stream_id: request.revocation_stream_id,
                expected_revocation_stream_generation: request.revocation_stream_generation,
            },
            keyset.as_ref(),
            &materialize,
        )
        .map_err(map_pairing_db_error)?;
    let response = decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.actor_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: 4,
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )?;
    encrypt_response(&state, &session_id, &response).await
}

async fn list_revocations(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<ListMaplePairingRevocationsRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<
    Json<EncryptedResponse<crate::models::maple_pairings::ListMaplePairingRevocationsResponse>>,
    ApiError,
> {
    let (issuer, keyset) = require_pairing_crypto(&state)?;
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let devices = load_devices(&state, &user, &auth_context)?;
    let (_, host) = require_device(&devices, request.host_registration_id)?;
    verify_device_signature(host, &wire_input(request.transcript())?, &request.signature)?;
    let limit = wire_input(request.effective_limit())?;
    let mut page = state
        .db
        .list_maple_pairing_revocations(
            pairing_authorization(&state, &user, &auth_context),
            request.host_registration_id,
            request.revocation_stream_id,
            request.revocation_stream_generation,
            request.after_issuer_sequence,
            i64::from(limit) + 1,
        )
        .map_err(map_pairing_db_error)?;
    if page.reset_clear_sync_payload.is_some() != page.reset_clear_lifecycle_floor.is_some() {
        return Err(ApiError::InternalServerError);
    }
    let discovery = request.revocation_stream_id.is_nil()
        && request.revocation_stream_generation == 0
        && request.after_issuer_sequence == 0;
    if page.revocation_stream_id.is_nil()
        || page.revocation_stream_generation == 0
        || (!discovery
            && (page.revocation_stream_id != request.revocation_stream_id
                || page.revocation_stream_generation != request.revocation_stream_generation))
    {
        return Err(ApiError::InternalServerError);
    }
    let has_more = page.events.len() > usize::from(limit);
    if has_more {
        page.events.truncate(usize::from(limit));
    }
    let mut events =
        Vec::<MapleRevocationStreamEventV1>::with_capacity(page.events.len().saturating_add(1));
    let mut expected_sequence = request
        .after_issuer_sequence
        .checked_add(1)
        .ok_or(ApiError::BadRequest)?;
    let trusted_now_unix_ms = Utc::now().timestamp_millis();
    for entry in &page.events {
        let event = &entry.event;
        let pairing = &entry.pairing;
        let issuer_sequence: u64 = event
            .issuer_sequence
            .try_into()
            .map_err(|_| ApiError::InternalServerError)?;
        let pairing_incarnation: u64 = event
            .pairing_incarnation
            .try_into()
            .map_err(|_| ApiError::InternalServerError)?;
        if issuer_sequence != expected_sequence {
            return Err(ApiError::InternalServerError);
        }
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or(ApiError::InternalServerError)?;
        let revocation = decrypt_revocation_payload(
            &state.enclave_key,
            &event.payload_enc,
            RevocationPayloadContext {
                account_id: user.uuid,
                project_id: user.project_id,
                host_registration_id: request.host_registration_id,
                revocation_stream_id: event.revocation_stream_id,
                revocation_stream_generation: event
                    .revocation_stream_generation
                    .try_into()
                    .map_err(|_| ApiError::InternalServerError)?,
                event_id: event.uuid,
                issuer_sequence,
                pair_id: pairing.uuid,
                pairing_incarnation,
                payload_version: event.payload_version,
            },
        )?;
        stored_wire(revocation.verify(keyset))?;
        let (controller, bound_host) = pairing_participants(&devices, pairing)?;
        let status = pairing_status_from_row(
            &state.enclave_key,
            &user,
            project.client_id,
            pairing,
            MaplePairingRole::Host,
            controller,
            bound_host,
            keyset,
            trusted_now_unix_ms,
        )?;
        let status_revocation = status.revocation.ok_or(ApiError::InternalServerError)?;
        if revocation != status_revocation
            || revocation.event_id != event.uuid
            || revocation.issuer_sequence != issuer_sequence
            || revocation.revocation_stream_id != page.revocation_stream_id
            || revocation.revocation_stream_generation != page.revocation_stream_generation
            || revocation.recipient_host_registration_id != request.host_registration_id
            || revocation.issuer_key_id != event.issuer_key_id
            || event.event_digest != stored_wire(revocation.digest())?.to_vec()
        {
            return Err(ApiError::InternalServerError);
        }
        events.push(MapleRevocationStreamEventV1::PairRevocation(revocation));
    }
    let revocation_sync = if let Some(sync_payload) = page.reset_clear_sync_payload.as_ref() {
        if !events.is_empty() || has_more || request.after_issuer_sequence != 0 {
            return Err(ApiError::InternalServerError);
        }
        let sync: MapleRevocationSyncV1 =
            serde_json::from_slice(sync_payload).map_err(|_| ApiError::InternalServerError)?;
        sync.verify_against_registration(
            user.uuid,
            project.client_id,
            host.registration_id,
            page.security_epoch,
            keyset,
        )
        .map_err(|_| ApiError::InternalServerError)?;
        let checkpoint = &sync.stream_checkpoint;
        if checkpoint.host != device_claim(host)
            || checkpoint.revocation_stream_id != page.revocation_stream_id
            || checkpoint.revocation_stream_generation != page.revocation_stream_generation
            || checkpoint.last_issued_issuer_sequence != page.last_issued_revocation_sequence
            || checkpoint.last_acked_issuer_sequence != page.last_acked_revocation_sequence
        {
            return Err(ApiError::InternalServerError);
        }
        let instruction = sync
            .reset_clear_instruction
            .clone()
            .ok_or(ApiError::InternalServerError)?;
        events.push(MapleRevocationStreamEventV1::ResetClearRequired(
            instruction,
        ));
        sync
    } else {
        let stream_checkpoint = crate::models::maple_pairings::sign_revocation_stream_checkpoint(
            MapleRevocationStreamCheckpointV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                subject_account_id: user.uuid,
                subject_project_id: project.client_id,
                host: device_claim(host),
                security_epoch: page.security_epoch,
                revocation_stream_id: page.revocation_stream_id,
                revocation_stream_generation: page.revocation_stream_generation,
                last_issued_issuer_sequence: page.last_issued_revocation_sequence,
                last_acked_issuer_sequence: page.last_acked_revocation_sequence,
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            issuer.as_ref(),
        )
        .map_err(|_| ApiError::ServiceUnavailable)?;
        stored_wire(stream_checkpoint.verify(keyset))?;
        MapleRevocationSyncV1::status_for_checkpoint(page.security_epoch, stream_checkpoint, None)
            .map_err(|_| ApiError::InternalServerError)?
    };
    let next_after_issuer_sequence = page.events.last().map_or_else(
        || {
            if page.reset_clear_sync_payload.is_some() {
                1
            } else {
                request.after_issuer_sequence
            }
        },
        |entry| u64::try_from(entry.event.issuer_sequence).unwrap_or(u64::MAX),
    );
    let has_more = has_more || next_after_issuer_sequence < page.last_issued_revocation_sequence;
    if !has_more && next_after_issuer_sequence != page.last_issued_revocation_sequence {
        return Err(ApiError::InternalServerError);
    }
    let response = ListMaplePairingRevocationsResponse {
        protocol_version: PROTOCOL_VERSION_V1,
        query_id: request.query_id,
        revocation_sync,
        events,
        next_after_issuer_sequence,
        has_more,
    };
    stored_wire(response.verify_against_request(&request, keyset))?;
    encrypt_response(&state, &session_id, &response).await
}

async fn ack_revocation(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<AckMaplePairingRevocationRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<
    Json<EncryptedResponse<crate::models::maple_pairings::AckMaplePairingRevocationResponse>>,
    ApiError,
> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    wire_input(request.validate())?;
    validate_common_assertions(
        request.protocol_version,
        request.asserted_account_id,
        request.asserted_project_id,
        &user,
        project.client_id,
    )?;
    let transcript = wire_input(request.transcript())?;
    let request_mac = request_operation_mac(&state.enclave_key, &transcript, &request.signature)?;
    if let Some(response) =
        replay_reset_clear_ack_if_present(&state, &user, &auth_context, &request, &request_mac)?
    {
        return encrypt_response(&state, &session_id, &response).await;
    }
    let devices = load_devices(&state, &user, &auth_context)?;
    let (_, host) = match require_device(&devices, request.host_registration_id) {
        Ok(host) => host,
        Err(ApiError::NotFound) => {
            // Close the gap between the read-only replay preflight and live
            // device discovery. The mutation path performs the same check
            // again under its write transaction before consuming the host.
            if let Some(response) = replay_reset_clear_ack_if_present(
                &state,
                &user,
                &auth_context,
                &request,
                &request_mac,
            )? {
                return encrypt_response(&state, &session_id, &response).await;
            }
            return Err(ApiError::NotFound);
        }
        Err(error) => return Err(error),
    };
    verify_device_signature(host, &transcript, &request.signature)?;
    let (issuer, keyset) = require_pairing_crypto(&state)?;
    let page_result = state.db.list_maple_pairing_revocations(
        pairing_authorization(&state, &user, &auth_context),
        request.host_registration_id,
        request.revocation_stream_id,
        request.revocation_stream_generation,
        request.expected_previous_issuer_sequence,
        1,
    );
    let page = match page_result {
        Ok(page) => page,
        Err(DBError::MaplePairingNotFound) => {
            // The host may have been retired after device discovery. Resolve
            // the now-durable ACK before surfacing the transient missing host.
            if let Some(response) = replay_reset_clear_ack_if_present(
                &state,
                &user,
                &auth_context,
                &request,
                &request_mac,
            )? {
                return encrypt_response(&state, &session_id, &response).await;
            }
            return Err(ApiError::NotFound);
        }
        Err(error) => return Err(map_pairing_db_error(error)),
    };
    if page.reset_clear_sync_payload.is_some() != page.reset_clear_lifecycle_floor.is_some() {
        return Err(ApiError::InternalServerError);
    }
    let event_digest = canonical_b64_32(&request.event_digest)?;
    if page.events.is_empty() {
        let sync_payload = page
            .reset_clear_sync_payload
            .as_deref()
            .ok_or(ApiError::Conflict)?;
        let lifecycle_floor = page
            .reset_clear_lifecycle_floor
            .ok_or(ApiError::InternalServerError)?;
        let sync: MapleRevocationSyncV1 =
            serde_json::from_slice(sync_payload).map_err(|_| ApiError::InternalServerError)?;
        stored_wire(sync.verify_against_registration(
            user.uuid,
            project.client_id,
            host.registration_id,
            page.security_epoch,
            keyset,
        ))?;
        let instruction = sync
            .reset_clear_instruction
            .as_ref()
            .ok_or(ApiError::InternalServerError)?;
        if request.issuer_sequence != 1
            || request.expected_previous_issuer_sequence != 0
            || page.last_issued_revocation_sequence != 1
            || page.last_acked_revocation_sequence != 0
            || page.revocation_stream_id != request.revocation_stream_id
            || page.revocation_stream_generation != request.revocation_stream_generation
            || sync.stream_checkpoint.host != device_claim(host)
            || sync.stream_checkpoint.revocation_stream_id != page.revocation_stream_id
            || sync.stream_checkpoint.revocation_stream_generation
                != page.revocation_stream_generation
            || instruction.event_id != request.event_id
            || instruction.issuer_sequence != request.issuer_sequence
            || stored_wire(instruction.event_digest())? != event_digest
        {
            return Err(ApiError::Conflict);
        }
        let accepted_at =
            monotonic_millis_at_or_after(Utc::now().timestamp_millis(), lifecycle_floor)?;
        let stream_checkpoint = crate::models::maple_pairings::sign_revocation_stream_checkpoint(
            MapleRevocationStreamCheckpointV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                subject_account_id: user.uuid,
                subject_project_id: project.client_id,
                host: device_claim(host),
                security_epoch: page.security_epoch,
                revocation_stream_id: page.revocation_stream_id,
                revocation_stream_generation: page.revocation_stream_generation,
                last_issued_issuer_sequence: 1,
                last_acked_issuer_sequence: 1,
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            issuer.as_ref(),
        )
        .map_err(|_| ApiError::ServiceUnavailable)?;
        stored_wire(stream_checkpoint.verify(keyset))?;
        let response = crate::models::maple_pairings::AckMaplePairingRevocationResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            operation_id: request.operation_id,
            host_registration_id: request.host_registration_id,
            stream_checkpoint,
            event_id: request.event_id,
            issuer_sequence: request.issuer_sequence,
            last_acked_issuer_sequence: request.issuer_sequence,
            accepted_at_unix_ms: accepted_at.timestamp_millis(),
        };
        stored_wire(response.verify_against_request(&request, keyset))?;
        let receipt_enc = encrypt_receipt(
            &state.enclave_key,
            &response,
            ReceiptContext {
                account_id: user.uuid,
                project_id: user.project_id,
                actor_registration_id: request.host_registration_id,
                operation_id: request.operation_id,
                operation_kind: MaplePairingOperationKind::Ack.as_db(),
                pair_id: request.event_id,
                pairing_revision: 3,
                receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            },
        )?;
        let receipt = state
            .db
            .ack_maple_pairing_revocation(MaplePairingRevocationAck {
                authorization: pairing_authorization(&state, &user, &auth_context),
                operation_id: request.operation_id,
                request_mac: request_mac.to_vec(),
                host_registration_id: request.host_registration_id,
                revocation_stream_id: request.revocation_stream_id,
                revocation_stream_generation: request.revocation_stream_generation,
                event_id: request.event_id,
                issuer_sequence: request.issuer_sequence,
                event_digest: event_digest.to_vec(),
                expected_previous_issuer_sequence: request.expected_previous_issuer_sequence,
                checkpoint_issuer_key_id: response.stream_checkpoint.issuer_key_id.clone(),
                receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
                receipt_enc,
                accepted_at,
            })
            .map_err(map_pairing_db_error)?;
        let response = decrypt_receipt(
            &state.enclave_key,
            &receipt.receipt_enc,
            ReceiptContext {
                account_id: user.uuid,
                project_id: user.project_id,
                actor_registration_id: request.host_registration_id,
                operation_id: receipt.operation_id,
                operation_kind: MaplePairingOperationKind::Ack.as_db(),
                pair_id: receipt.pair_id,
                pairing_revision: receipt.pairing_revision,
                receipt_version: receipt.receipt_version,
            },
        )?;
        return encrypt_response(&state, &session_id, &response).await;
    }
    if let Some(response) = replay_mutation_if_present::<
        crate::models::maple_pairings::AckMaplePairingRevocationResponse,
    >(
        &state,
        &user,
        &auth_context,
        request.host_registration_id,
        request.operation_id,
        MaplePairingOperationKind::Ack,
        &request_mac,
    )? {
        return encrypt_response(&state, &session_id, &response).await;
    }
    let entry = page.events.first().ok_or(ApiError::NotFound)?;
    if entry.event.uuid != request.event_id
        || u64::try_from(entry.event.issuer_sequence).ok() != Some(request.issuer_sequence)
        || entry.event.event_digest != event_digest.to_vec()
    {
        return Err(ApiError::Conflict);
    }
    let pairing_incarnation: u64 = entry
        .event
        .pairing_incarnation
        .try_into()
        .map_err(|_| ApiError::InternalServerError)?;
    let revocation = decrypt_revocation_payload(
        &state.enclave_key,
        &entry.event.payload_enc,
        RevocationPayloadContext {
            account_id: user.uuid,
            project_id: user.project_id,
            host_registration_id: request.host_registration_id,
            revocation_stream_id: entry.event.revocation_stream_id,
            revocation_stream_generation: entry
                .event
                .revocation_stream_generation
                .try_into()
                .map_err(|_| ApiError::InternalServerError)?,
            event_id: entry.event.uuid,
            issuer_sequence: request.issuer_sequence,
            pair_id: entry.pairing.uuid,
            pairing_incarnation,
            payload_version: entry.event.payload_version,
        },
    )?;
    stored_wire(revocation.verify(keyset))?;
    let (controller, bound_host) = pairing_participants(&devices, &entry.pairing)?;
    let status = pairing_status_from_row(
        &state.enclave_key,
        &user,
        project.client_id,
        &entry.pairing,
        MaplePairingRole::Host,
        controller,
        bound_host,
        keyset,
        Utc::now().timestamp_millis(),
    )?;
    if status.revocation.as_ref() != Some(&revocation)
        || revocation.issuer_key_id != entry.event.issuer_key_id
        || revocation.revocation_stream_id != request.revocation_stream_id
        || revocation.revocation_stream_generation != request.revocation_stream_generation
        || page.revocation_stream_id != request.revocation_stream_id
        || page.revocation_stream_generation != request.revocation_stream_generation
        || stored_wire(revocation.digest())? != event_digest
    {
        return Err(ApiError::InternalServerError);
    }
    // A revocation timestamp is monotonically clamped by the DB to the pair's
    // lifecycle predecessor. ACK must use the same monotonic rule so an
    // urgent revocation remains acknowledgeable across a wall-clock step.
    let accepted_at =
        monotonic_millis(Utc::now().timestamp_millis(), revocation.revoked_at_unix_ms)?;
    let stream_checkpoint = crate::models::maple_pairings::sign_revocation_stream_checkpoint(
        MapleRevocationStreamCheckpointV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: user.uuid,
            subject_project_id: project.client_id,
            host: device_claim(host),
            security_epoch: page.security_epoch,
            revocation_stream_id: request.revocation_stream_id,
            revocation_stream_generation: request.revocation_stream_generation,
            last_issued_issuer_sequence: page.last_issued_revocation_sequence,
            last_acked_issuer_sequence: request.issuer_sequence,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        issuer.as_ref(),
    )
    .map_err(|_| ApiError::ServiceUnavailable)?;
    stored_wire(stream_checkpoint.verify(keyset))?;
    let _ = stored_wire(stream_checkpoint.digest())?;
    let response = crate::models::maple_pairings::AckMaplePairingRevocationResponse {
        protocol_version: PROTOCOL_VERSION_V1,
        operation_id: request.operation_id,
        host_registration_id: request.host_registration_id,
        stream_checkpoint,
        event_id: request.event_id,
        issuer_sequence: request.issuer_sequence,
        last_acked_issuer_sequence: request.issuer_sequence,
        accepted_at_unix_ms: unix_ms(accepted_at),
    };
    stored_wire(response.verify_against_request(&request, keyset))?;
    let receipt_enc = encrypt_receipt(
        &state.enclave_key,
        &response,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.host_registration_id,
            operation_id: request.operation_id,
            operation_kind: 5,
            pair_id: entry.pairing.uuid,
            pairing_revision: entry.pairing.revision,
            receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
        },
    )?;
    let receipt = state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: pairing_authorization(&state, &user, &auth_context),
            operation_id: request.operation_id,
            request_mac: request_mac.to_vec(),
            host_registration_id: request.host_registration_id,
            revocation_stream_id: request.revocation_stream_id,
            revocation_stream_generation: request.revocation_stream_generation,
            event_id: request.event_id,
            issuer_sequence: request.issuer_sequence,
            event_digest: event_digest.to_vec(),
            expected_previous_issuer_sequence: request.expected_previous_issuer_sequence,
            checkpoint_issuer_key_id: response.stream_checkpoint.issuer_key_id.clone(),
            receipt_version: MAPLE_PAIRING_RECEIPT_VERSION_V1,
            receipt_enc,
            accepted_at,
        })
        .map_err(map_pairing_db_error)?;
    let response = decrypt_receipt(
        &state.enclave_key,
        &receipt.receipt_enc,
        ReceiptContext {
            account_id: user.uuid,
            project_id: user.project_id,
            actor_registration_id: request.host_registration_id,
            operation_id: receipt.operation_id,
            operation_kind: 5,
            pair_id: receipt.pair_id,
            pairing_revision: receipt.pairing_revision,
            receipt_version: receipt.receipt_version,
        },
    )?;
    encrypt_response(&state, &session_id, &response).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct TestReceipt {
        accepted: bool,
    }

    fn test_uuid(value: &str) -> Uuid {
        Uuid::parse_str(value).expect("valid test UUID")
    }

    #[test]
    fn unresolved_reset_clear_maps_to_the_sanitized_typed_api_error() {
        assert!(matches!(
            map_pairing_db_error(DBError::MaplePairingResetClearRequired),
            ApiError::MaplePairingResetClearRequired
        ));
    }

    #[test]
    fn controller_never_sees_authorization_before_host_commit() {
        assert!(!pairing_authorization_is_visible(
            MaplePairingState::AwaitingHostCommit,
            MaplePairingRole::Controller,
            false,
        ));
        assert!(!pairing_authorization_is_visible(
            MaplePairingState::Revoked,
            MaplePairingRole::Controller,
            false,
        ));
        assert!(pairing_authorization_is_visible(
            MaplePairingState::Revoked,
            MaplePairingRole::Host,
            false,
        ));
        assert!(pairing_authorization_is_visible(
            MaplePairingState::Active,
            MaplePairingRole::Controller,
            true,
        ));
        assert!(pairing_authorization_is_visible(
            MaplePairingState::Revoked,
            MaplePairingRole::Controller,
            true,
        ));
    }

    #[test]
    fn signed_lifecycle_time_never_regresses_behind_its_predecessor() {
        let predecessor = 1_786_579_260_000_i64;
        assert_eq!(
            unix_ms(monotonic_millis(predecessor - 5_000, predecessor).unwrap()),
            predecessor,
            "a backward wall-clock step must clamp to the committed predecessor"
        );
        assert_eq!(
            unix_ms(monotonic_millis(predecessor + 5_000, predecessor).unwrap()),
            predecessor + 5_000,
            "a forward wall clock remains observable"
        );
    }

    #[test]
    fn pair_cursor_is_bound_to_the_account_actor_and_query() {
        let enclave_key = [0x42; 32];
        let account_id = test_uuid("11111111-1111-4111-8111-111111111111");
        let actor_id = test_uuid("22222222-2222-4222-8222-222222222222");
        let pair_id = test_uuid("33333333-3333-4333-8333-333333333333");
        let states = ["active", "revoked"];
        let cursor = encode_pair_cursor(
            &enclave_key,
            account_id,
            7,
            actor_id,
            "controller",
            &states,
            pair_id,
        )
        .expect("encode cursor");

        assert_eq!(
            decode_pair_cursor(
                &enclave_key,
                account_id,
                7,
                actor_id,
                "controller",
                &states,
                &cursor,
            )
            .expect("decode cursor"),
            pair_id
        );
        assert!(decode_pair_cursor(
            &enclave_key,
            test_uuid("44444444-4444-4444-8444-444444444444"),
            7,
            actor_id,
            "controller",
            &states,
            &cursor,
        )
        .is_err());
        assert!(decode_pair_cursor(
            &enclave_key,
            account_id,
            8,
            actor_id,
            "controller",
            &states,
            &cursor,
        )
        .is_err());
        assert!(decode_pair_cursor(
            &enclave_key,
            account_id,
            7,
            test_uuid("55555555-5555-4555-8555-555555555555"),
            "controller",
            &states,
            &cursor,
        )
        .is_err());
        assert!(decode_pair_cursor(
            &enclave_key,
            account_id,
            7,
            actor_id,
            "host",
            &states,
            &cursor,
        )
        .is_err());
        assert!(decode_pair_cursor(
            &enclave_key,
            account_id,
            7,
            actor_id,
            "controller",
            &["revoked", "active"],
            &cursor,
        )
        .is_err());

        let mut tampered = cursor.into_bytes();
        tampered[63] = if tampered[63] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(tampered).expect("ASCII cursor");
        assert!(decode_pair_cursor(
            &enclave_key,
            account_id,
            7,
            actor_id,
            "controller",
            &states,
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn receipt_aead_rejects_context_substitution() {
        let enclave_key = [0x24; 32];
        let context = ReceiptContext {
            account_id: test_uuid("11111111-1111-4111-8111-111111111111"),
            project_id: 7,
            actor_registration_id: test_uuid("22222222-2222-4222-8222-222222222222"),
            operation_id: test_uuid("33333333-3333-4333-8333-333333333333"),
            operation_kind: 3,
            pair_id: test_uuid("44444444-4444-4444-8444-444444444444"),
            pairing_revision: 9,
            receipt_version: 1,
        };
        let receipt = TestReceipt { accepted: true };
        let encrypted = encrypt_receipt(&enclave_key, &receipt, context).expect("encrypt receipt");
        assert_eq!(
            decrypt_receipt::<TestReceipt>(&enclave_key, &encrypted, context)
                .expect("decrypt receipt"),
            receipt
        );

        let substitutions = [
            ReceiptContext {
                account_id: test_uuid("55555555-5555-4555-8555-555555555555"),
                ..context
            },
            ReceiptContext {
                actor_registration_id: test_uuid("66666666-6666-4666-8666-666666666666"),
                ..context
            },
            ReceiptContext {
                operation_id: test_uuid("77777777-7777-4777-8777-777777777777"),
                ..context
            },
            ReceiptContext {
                operation_kind: 4,
                ..context
            },
            ReceiptContext {
                pair_id: test_uuid("88888888-8888-4888-8888-888888888888"),
                ..context
            },
            ReceiptContext {
                pairing_revision: 10,
                ..context
            },
        ];
        for substitution in substitutions {
            assert!(
                decrypt_receipt::<TestReceipt>(&enclave_key, &encrypted, substitution).is_err()
            );
        }
    }

    #[test]
    fn encrypted_record_aad_binds_pair_and_revocation_authority() {
        let pair = PairPayloadContext {
            account_id: test_uuid("11111111-1111-4111-8111-111111111111"),
            project_id: 7,
            pairing_request_id: test_uuid("22222222-2222-4222-8222-222222222222"),
            pair_id: test_uuid("33333333-3333-4333-8333-333333333333"),
            pairing_incarnation: 9,
            revocation_stream_id: Some(test_uuid("44444444-4444-4444-8444-444444444444")),
            revocation_stream_generation: Some(3),
            payload_version: 1,
        };
        let pair_aad = pair_payload_aad(pair).expect("pair AAD");
        assert_ne!(
            pair_aad,
            pair_payload_aad(PairPayloadContext {
                pair_id: test_uuid("44444444-4444-4444-8444-444444444444"),
                ..pair
            })
            .expect("substituted pair AAD")
        );
        assert_ne!(
            pair_aad,
            pair_payload_aad(PairPayloadContext {
                pairing_incarnation: 10,
                ..pair
            })
            .expect("substituted incarnation AAD")
        );
        assert_ne!(
            pair_aad,
            pair_payload_aad(PairPayloadContext {
                revocation_stream_generation: Some(4),
                ..pair
            })
            .expect("substituted stream generation AAD")
        );

        let revocation = RevocationPayloadContext {
            account_id: pair.account_id,
            project_id: pair.project_id,
            host_registration_id: test_uuid("55555555-5555-4555-8555-555555555555"),
            revocation_stream_id: pair.revocation_stream_id.expect("stream id"),
            revocation_stream_generation: pair
                .revocation_stream_generation
                .expect("stream generation"),
            event_id: test_uuid("66666666-6666-4666-8666-666666666666"),
            issuer_sequence: 11,
            pair_id: pair.pair_id,
            pairing_incarnation: pair.pairing_incarnation,
            payload_version: 1,
        };
        let revocation_aad = revocation_payload_aad(revocation).expect("revocation AAD");
        assert_ne!(
            revocation_aad,
            revocation_payload_aad(RevocationPayloadContext {
                host_registration_id: test_uuid("77777777-7777-4777-8777-777777777777"),
                ..revocation
            })
            .expect("substituted host AAD")
        );
        assert_ne!(
            revocation_aad,
            revocation_payload_aad(RevocationPayloadContext {
                issuer_sequence: 12,
                ..revocation
            })
            .expect("substituted sequence AAD")
        );
        assert_ne!(
            revocation_aad,
            revocation_payload_aad(RevocationPayloadContext {
                revocation_stream_id: test_uuid("99999999-9999-4999-8999-999999999999"),
                ..revocation
            })
            .expect("substituted stream AAD")
        );
        assert_ne!(
            revocation_aad,
            revocation_payload_aad(RevocationPayloadContext {
                event_id: test_uuid("88888888-8888-4888-8888-888888888888"),
                ..revocation
            })
            .expect("substituted event AAD")
        );
    }
}
