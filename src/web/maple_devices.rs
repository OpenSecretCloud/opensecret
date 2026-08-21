//! Encrypted, account-scoped Maple device registration.
//!
//! A registration proves possession of the durable Iroh / Ed25519 device key.
//! The authenticated user and project are authoritative; the matching request
//! fields are signed preconditions and never select the storage namespace.

use axum::{
    extract::{Query, State},
    middleware::from_fn_with_state,
    routing::{get, post},
    Extension, Json, Router,
};
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine as _,
};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::{collections::HashSet, net::SocketAddr, sync::Arc};
use url::Url;
use uuid::Uuid;

use crate::{
    db::DBError,
    encrypt::{decrypt_aead_v1, derive_key, encrypt_aead_v1, CanonicalBytes, EncryptError},
    jwt::{AuthContext, AuthMethod},
    models::{
        maple_devices::{
            MapleDevice, MapleDeviceListAuthorization, MapleDeviceListCursor,
            NewMapleDeviceRegistration,
        },
        maple_pairing_db::{
            MapleDeviceRegistrationSyncMaterial, MapleDeviceRegistrationSyncMaterializationContext,
            MaplePairingCreateDeviceContext, MaplePairingMaterializationError,
            MapleResetClearSource, MapleResetClearUnsignedMaterial,
            MapleResetClearUnsignedMaterializationContext,
        },
        maple_pairings::{
            reset_clear_admission_set_digest, reset_clear_chain_transcript,
            reset_clear_instruction_material_transcript, sha256_digest, sign_reset_clear_required,
            sign_revocation_stream_checkpoint, MapleDeviceClaimV1, MaplePairingIdentityAlgorithm,
            MaplePairingIssuer, MaplePairingIssuerKeySetV1, MapleResetClearAdmissionLeafV1,
            MapleResetClearRequiredV1, MapleResetClearScopeV1, MapleRevocationStreamCheckpointV1,
            MapleRevocationSyncV1, MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        },
        users::User,
    },
    web::encryption_middleware::{
        decrypt_request, decrypt_request_bounded, encrypt_response, EncryptedResponse,
    },
    ApiError, AppState,
};

type HmacSha256 = Hmac<Sha256>;

const PROTOCOL_VERSION_V1: u16 = 1;
const TRANSCRIPT_VERSION_V1: u16 = 1;
const PAYLOAD_VERSION_V1: i16 = 1;
const REGISTRATION_TRANSCRIPT_DOMAIN: &str = "os.maple-device-registration.v1";
const REGISTRATION_OPERATION_DOMAIN: &str = "os.maple-device-registration-operation.v1";
const DEVICE_PAYLOAD_DOMAIN: &str = "os.maple-device-record.v1";
const DEVICE_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-device-record-key.v1";
const OPERATION_MAC_KEY_INFO: &[u8] = b"os.maple-device-operation-mac-key.v1";
const DEVICE_IDENTITY_MAC_DOMAIN: &str = "os.maple-device-identity.v1";
const DEVICE_IDENTITY_MAC_KEY_INFO: &[u8] = b"os.maple-device-identity-mac-key.v1";
const DEVICE_REGISTRATION_ID_DOMAIN: &str = "os.maple-device-registration-id.v1";
const DEVICE_REGISTRATION_ID_KEY_INFO: &[u8] = b"os.maple-device-registration-id-key.v1";
const DEVICE_CURSOR_DOMAIN: &str = "os.maple-device-cursor.v1";
const DEVICE_CURSOR_MAC_KEY_INFO: &[u8] = b"os.maple-device-cursor-mac-key.v1";
const IDENTITY_ALGORITHM: &str = "ed25519";
const DEFAULT_PAGE_SIZE: u16 = 25;
const MAX_PAGE_SIZE: u16 = 100;
const MAX_DISPLAY_NAME_CHARS: usize = 80;
const MAX_CAPABILITIES: usize = 32;
const MAX_CAPABILITY_LEN: usize = 64;
const MAX_RELAY_URLS: usize = 4;
const MAX_DIRECT_ADDRESSES: usize = 16;
const MAX_RELAY_URL_LEN: usize = 512;
const MAX_DIRECT_ADDRESS_LEN: usize = 64;
// The bounded v1 DTO is below 8 KiB even at every per-field maximum. Leave
// headroom for JSON/base64/AEAD overhead without inheriting the generic 50 MiB
// encrypted request envelope used by attachment-capable routes.
const MAX_REGISTRATION_PLAINTEXT_BYTES: usize = 8 * 1024;
const MAX_REGISTRATION_ENCRYPTED_BYTES: usize = 16 * 1024;
const RESET_CLEAR_PAYLOAD_VERSION_V1: i16 = 1;

/// Bounded, non-secret routing information for an Iroh endpoint. Direct routes
/// may be public, private, or loopback unicast addresses. This deliberately
/// omits private keys and custom/opaque transport addresses; the entire record
/// remains encrypted at the account boundary.
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MapleIrohEndpointAddr {
    pub relay_urls: Vec<String>,
    pub direct_addresses: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMapleDeviceRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    /// Null creates a new registration; updates must sign the current revision.
    pub expected_revision: Option<i64>,
    /// Server-controlled account authority epoch learned from the encrypted
    /// protected device-list bootstrap. It must exactly match current state.
    pub known_security_epoch: u64,
    pub asserted_account_id: Uuid,
    /// Public `org_projects.client_id`, not the internal integer primary key.
    pub asserted_project_id: Uuid,
    pub identity_algorithm: String,
    /// Standard-base64 Ed25519 public key (exactly 32 decoded bytes).
    pub identity_public_key: String,
    /// Canonical lowercase Iroh endpoint hex (exactly 64 characters).
    pub iroh_endpoint_id: String,
    /// Monotonic endpoint lifecycle epoch. Address churn keeps this stable and
    /// advances the signed `expected_revision` CAS instead; v1 may retain or
    /// advance the epoch for the same immutable identity but never decrease it.
    pub endpoint_epoch: u64,
    pub iroh_endpoint_addr: MapleIrohEndpointAddr,
    pub platform: String,
    pub display_name: String,
    pub capabilities: Vec<String>,
    /// Standard-base64 Ed25519 signature over the server-defined transcript.
    pub signature: String,
}

impl std::fmt::Debug for RegisterMapleDeviceRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RegisterMapleDeviceRequest")
            .field("protocol_version", &self.protocol_version)
            .field("transcript_version", &self.transcript_version)
            .field("operation_id", &self.operation_id)
            .field("device_id", &self.device_id)
            .field("installation_id", &self.installation_id)
            .field("expected_revision", &self.expected_revision)
            .field("known_security_epoch", &self.known_security_epoch)
            .field("asserted_account_id", &self.asserted_account_id)
            .field("asserted_project_id", &self.asserted_project_id)
            .field("identity_algorithm", &self.identity_algorithm)
            .field("identity_public_key", &"<redacted>")
            .field("iroh_endpoint_id", &"<redacted>")
            .field("endpoint_epoch", &self.endpoint_epoch)
            .field("relay_url_count", &self.iroh_endpoint_addr.relay_urls.len())
            .field(
                "direct_address_count",
                &self.iroh_endpoint_addr.direct_addresses.len(),
            )
            .field("platform", &self.platform)
            .field("display_name", &"<redacted>")
            .field("capability_count", &self.capabilities.len())
            .field("signature", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterMapleDeviceResponse {
    pub protocol_version: u16,
    pub operation_id: Uuid,
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub revision: i64,
    pub accepted_at: DateTime<Utc>,
    pub security_epoch: u64,
    pub revocation_sync: MapleRevocationSyncV1,
}

impl std::fmt::Debug for RegisterMapleDeviceResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RegisterMapleDeviceResponse")
            .field("protocol_version", &self.protocol_version)
            .field("revision", &self.revision)
            .field("security_epoch", &self.security_epoch)
            .field("sync_status", &self.revocation_sync.status)
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MapleDeviceResponse {
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub identity_algorithm: String,
    pub identity_public_key: String,
    pub iroh_endpoint_id: String,
    pub endpoint_epoch: u64,
    pub iroh_endpoint_addr: MapleIrohEndpointAddr,
    pub platform: String,
    pub display_name: String,
    pub capabilities: Vec<String>,
    pub revision: i64,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListMapleDevicesResponse {
    pub protocol_version: u16,
    pub security_epoch: u64,
    pub devices: Vec<MapleDeviceResponse>,
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

#[derive(Clone, Deserialize, Default)]
struct ListMapleDevicesQuery {
    cursor: Option<String>,
    limit: Option<u16>,
}

#[derive(Serialize, Deserialize)]
struct StoredMapleDevicePayloadV1 {
    registration_id: Uuid,
    revision: i64,
    identity_algorithm: String,
    identity_public_key: Vec<u8>,
    iroh_endpoint_id: String,
    endpoint_epoch: u64,
    iroh_endpoint_addr: MapleIrohEndpointAddr,
    platform: String,
    display_name: String,
    capabilities: Vec<String>,
}

struct ValidatedRegistration {
    identity_public_key: [u8; 32],
    signature: [u8; 64],
    iroh_endpoint_addr: MapleIrohEndpointAddr,
    capabilities: Vec<String>,
}

#[derive(Clone, Copy)]
struct MapleDeviceRecordContext {
    account_id: Uuid,
    project_id: i32,
    registration_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    revision: i64,
    payload_version: i16,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/protected/maple/devices/register",
            post(register_device).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request_bounded::<
                    RegisterMapleDeviceRequest,
                    MAX_REGISTRATION_ENCRYPTED_BYTES,
                    MAX_REGISTRATION_PLAINTEXT_BYTES,
                >,
            )),
        )
        .route(
            "/protected/maple/devices",
            get(list_devices).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
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
        &Arc<dyn MaplePairingIssuer>,
        &Arc<MaplePairingIssuerKeySetV1>,
    ),
    ApiError,
> {
    let keyset = require_pairing_keyset(state)?;
    let issuer = state
        .maple_pairing_issuer
        .as_ref()
        .ok_or(ApiError::ServiceUnavailable)?;
    if !keyset
        .contains_issuer(issuer.as_ref())
        .map_err(|_| ApiError::ServiceUnavailable)?
    {
        return Err(ApiError::ServiceUnavailable);
    }
    Ok((issuer, keyset))
}

fn registration_device_claim(
    enclave_key: &[u8],
    account_id: Uuid,
    internal_project_id: i32,
    device: &MaplePairingCreateDeviceContext,
) -> Result<MapleDeviceClaimV1, MaplePairingMaterializationError> {
    let plaintext = decrypt_maple_device_payload(
        enclave_key,
        &device.payload_enc,
        MapleDeviceRecordContext {
            account_id,
            project_id: internal_project_id,
            registration_id: device.registration_id,
            device_id: device.device_id,
            installation_id: device.installation_id,
            revision: device.device_revision,
            payload_version: device.payload_version,
        },
    )
    .map_err(|_| MaplePairingMaterializationError)?;
    let payload: StoredMapleDevicePayloadV1 =
        serde_json::from_slice(&plaintext).map_err(|_| MaplePairingMaterializationError)?;
    let epoch = DateTime::<Utc>::from_timestamp(0, 0).ok_or(MaplePairingMaterializationError)?;
    let response = reset_source_device_response(
        enclave_key,
        account_id,
        internal_project_id,
        device.registration_id,
        device.device_id,
        device.installation_id,
        device.device_revision,
        &device.identity_mac,
        device.payload_version,
        &device.payload_enc,
        &device.record_mac,
        epoch,
        payload,
    )?;
    if response.endpoint_epoch != device.endpoint_epoch {
        return Err(MaplePairingMaterializationError);
    }
    Ok(maple_device_claim(&response))
}

pub(crate) fn materialize_maple_device_registration_sync(
    enclave_key: &[u8],
    issuer: &dyn MaplePairingIssuer,
    issuer_keyset: &MaplePairingIssuerKeySetV1,
    context: MapleDeviceRegistrationSyncMaterializationContext,
) -> Result<MapleDeviceRegistrationSyncMaterial, MaplePairingMaterializationError> {
    match context {
        MapleDeviceRegistrationSyncMaterializationContext::Ordinary(context) => {
            let host = registration_device_claim(
                enclave_key,
                context.account_id,
                context.internal_project_id,
                &context.current_device,
            )?;
            let checkpoint = sign_revocation_stream_checkpoint(
                MapleRevocationStreamCheckpointV1 {
                    artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                    subject_account_id: context.account_id,
                    subject_project_id: context.subject_project_id,
                    host,
                    security_epoch: context.security_epoch,
                    revocation_stream_id: context.revocation_stream_id,
                    revocation_stream_generation: context.revocation_stream_generation,
                    last_issued_issuer_sequence: context.last_issued_issuer_sequence,
                    last_acked_issuer_sequence: context.last_acked_issuer_sequence,
                    issuer_key_id: String::new(),
                    issuer_signature: String::new(),
                },
                issuer,
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            let sync = MapleRevocationSyncV1::status_for_checkpoint(
                context.security_epoch,
                checkpoint,
                None,
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            if sync.status != context.status {
                return Err(MaplePairingMaterializationError);
            }
            sync.verify(issuer_keyset)
                .map_err(|_| MaplePairingMaterializationError)?;
            let sync_payload =
                serde_json::to_vec(&sync).map_err(|_| MaplePairingMaterializationError)?;
            Ok(MapleDeviceRegistrationSyncMaterial::Ordinary {
                sync,
                sync_payload_version: PAYLOAD_VERSION_V1,
                sync_payload,
            })
        }
        MapleDeviceRegistrationSyncMaterializationContext::ResetClearRequired(context) => {
            if context.host_claim_payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1
                || context.instruction_payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1
            {
                return Err(MaplePairingMaterializationError);
            }
            let current_host = registration_device_claim(
                enclave_key,
                context.account_id,
                context.internal_project_id,
                &context.current_device,
            )?;
            let retained_host: MapleDeviceClaimV1 =
                serde_json::from_slice(&context.host_claim_payload)
                    .map_err(|_| MaplePairingMaterializationError)?;
            retained_host
                .validate()
                .map_err(|_| MaplePairingMaterializationError)?;
            let canonical_retained_host =
                serde_json::to_vec(&retained_host).map_err(|_| MaplePairingMaterializationError)?;
            if retained_host != current_host
                || canonical_retained_host != context.host_claim_payload
            {
                return Err(MaplePairingMaterializationError);
            }
            let unsigned_instruction: MapleResetClearRequiredV1 =
                serde_json::from_slice(&context.instruction_payload)
                    .map_err(|_| MaplePairingMaterializationError)?;
            if serde_json::to_vec(&unsigned_instruction)
                .map_err(|_| MaplePairingMaterializationError)?
                != context.instruction_payload
            {
                return Err(MaplePairingMaterializationError);
            }
            let instruction = sign_reset_clear_required(unsigned_instruction, issuer)
                .map_err(|_| MaplePairingMaterializationError)?;
            let checkpoint = sign_revocation_stream_checkpoint(
                MapleRevocationStreamCheckpointV1 {
                    artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                    subject_account_id: context.account_id,
                    subject_project_id: context.subject_project_id,
                    host: current_host,
                    security_epoch: context.security_epoch,
                    revocation_stream_id: context.revocation_stream_id,
                    revocation_stream_generation: context.revocation_stream_generation,
                    last_issued_issuer_sequence: context.issuer_sequence,
                    last_acked_issuer_sequence: 0,
                    issuer_key_id: String::new(),
                    issuer_signature: String::new(),
                },
                issuer,
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            let sync = MapleRevocationSyncV1::status_for_checkpoint(
                context.security_epoch,
                checkpoint,
                Some(instruction.clone()),
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            sync.verify(issuer_keyset)
                .map_err(|_| MaplePairingMaterializationError)?;
            let signed_instruction_payload =
                serde_json::to_vec(&instruction).map_err(|_| MaplePairingMaterializationError)?;
            let sync_payload =
                serde_json::to_vec(&sync).map_err(|_| MaplePairingMaterializationError)?;
            Ok(MapleDeviceRegistrationSyncMaterial::ResetClearRequired {
                sync,
                signed_instruction_payload_version: PAYLOAD_VERSION_V1,
                signed_instruction_payload,
                sync_payload_version: PAYLOAD_VERSION_V1,
                sync_payload,
            })
        }
    }
}

async fn register_device(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(request): Extension<RegisterMapleDeviceRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<RegisterMapleDeviceResponse>>, ApiError> {
    let project = state.db.get_org_project_by_id(user.project_id)?;
    let issuer_keyset = require_pairing_keyset(&state)?;
    let validated = validate_registration(&request, user.uuid, project.client_id)?;
    let transcript = registration_transcript(
        &request,
        &validated.identity_public_key,
        &validated.iroh_endpoint_addr,
        &validated.capabilities,
    );

    let identity_key = VerifyingKey::from_bytes(&validated.identity_public_key)
        .map_err(|_| ApiError::BadRequest)?;
    let signature = Signature::from_bytes(&validated.signature);
    identity_key
        .verify_strict(&transcript, &signature)
        .map_err(|_| ApiError::BadRequest)?;

    let registration_id = maple_device_registration_id(
        &state.enclave_key,
        user.uuid,
        user.project_id,
        request.device_id,
        request.installation_id,
    )
    .map_err(|_| ApiError::InternalServerError)?;
    let revision = request
        .expected_revision
        .map_or(Some(1), |revision| revision.checked_add(1))
        .ok_or(ApiError::BadRequest)?;

    let payload = StoredMapleDevicePayloadV1 {
        registration_id,
        revision,
        identity_algorithm: IDENTITY_ALGORITHM.to_string(),
        identity_public_key: validated.identity_public_key.to_vec(),
        iroh_endpoint_id: request.iroh_endpoint_id.clone(),
        endpoint_epoch: request.endpoint_epoch,
        iroh_endpoint_addr: validated.iroh_endpoint_addr,
        platform: request.platform.clone(),
        display_name: request.display_name.clone(),
        capabilities: validated.capabilities,
    };
    let payload_plaintext =
        serde_json::to_vec(&payload).map_err(|_| ApiError::InternalServerError)?;
    let payload_enc = encrypt_maple_device_payload(
        &state.enclave_key,
        &payload_plaintext,
        MapleDeviceRecordContext {
            account_id: user.uuid,
            project_id: user.project_id,
            registration_id,
            device_id: request.device_id,
            installation_id: request.installation_id,
            revision,
            payload_version: PAYLOAD_VERSION_V1,
        },
    )
    .map_err(|_| ApiError::InternalServerError)?;

    let request_mac = registration_operation_mac(
        &state.enclave_key,
        &transcript,
        &request.iroh_endpoint_addr,
        &request.capabilities,
        &validated.signature,
    )
    .map_err(|_| ApiError::InternalServerError)?;
    let identity_mac = maple_device_identity_mac(
        &state.enclave_key,
        user.uuid,
        user.project_id,
        &validated.identity_public_key,
    )
    .map_err(|_| ApiError::InternalServerError)?;

    let receipt = state
        .db
        .register_maple_device(
            NewMapleDeviceRegistration {
                user_id: user.uuid,
                subject_project_id: project.client_id,
                project_id: user.project_id,
                operation_id: request.operation_id,
                request_mac: request_mac.to_vec(),
                auth_credential_kind: match auth_context.method {
                    AuthMethod::Password => "password",
                    AuthMethod::OAuth => "oauth",
                }
                .to_string(),
                auth_binding: auth_context.auth_binding,
                enclave_key: state.enclave_key.clone(),
                registration_id,
                device_id: request.device_id,
                installation_id: request.installation_id,
                identity_mac: identity_mac.to_vec(),
                endpoint_epoch: request
                    .endpoint_epoch
                    .try_into()
                    .map_err(|_| ApiError::BadRequest)?,
                expected_revision: request.expected_revision,
                known_security_epoch: request
                    .known_security_epoch
                    .try_into()
                    .map_err(|_| ApiError::BadRequest)?,
                payload_version: PAYLOAD_VERSION_V1,
                payload_enc,
                revision,
            },
            issuer_keyset,
            &|context| {
                // Historical replay and terminal-retirement checks require only
                // retained verification keys. Demand an active signer lazily,
                // after the database has admitted a genuinely fresh lineage.
                let (issuer, materializer_keyset) =
                    require_pairing_crypto(&state).map_err(|_| MaplePairingMaterializationError)?;
                materialize_maple_device_registration_sync(
                    &state.enclave_key,
                    issuer.as_ref(),
                    materializer_keyset,
                    context,
                )
            },
        )
        .map_err(map_maple_registration_db_error)?;

    let revocation_sync: MapleRevocationSyncV1 =
        serde_json::from_slice(&receipt.sync_payload).map_err(|_| ApiError::InternalServerError)?;
    revocation_sync
        .verify_against_registration(
            user.uuid,
            project.client_id,
            receipt.registration_id,
            receipt
                .security_epoch
                .try_into()
                .map_err(|_| ApiError::InternalServerError)?,
            issuer_keyset,
        )
        .map_err(|_| ApiError::InternalServerError)?;

    let response = RegisterMapleDeviceResponse {
        protocol_version: PROTOCOL_VERSION_V1,
        operation_id: receipt.operation_id,
        registration_id: receipt.registration_id,
        device_id: receipt.device_id,
        revision: receipt.revision,
        accepted_at: receipt.accepted_at,
        security_epoch: receipt
            .security_epoch
            .try_into()
            .map_err(|_| ApiError::InternalServerError)?,
        revocation_sync,
    };
    encrypt_response(&state, &session_id, &response).await
}

fn map_maple_registration_db_error(error: DBError) -> ApiError {
    match error {
        DBError::MaplePairingAuthorityBusy
        | DBError::MaplePairingAuthorityCapacityExceeded
        | DBError::MaplePairingAuthorityDeletionBlocked
        | DBError::MaplePairingAuthorityCorrupt => ApiError::from(error),
        DBError::MapleDeviceRegistrationConflict
        | DBError::MapleDeviceLimitExceeded
        | DBError::MapleDeviceOperationLimitExceeded => ApiError::Conflict,
        DBError::MapleInstallationRetired => ApiError::MapleInstallationRetired,
        DBError::MapleDeviceSecurityEpochStale => ApiError::MapleSecurityEpochStale,
        DBError::MaplePairingMaterializationFailed => ApiError::ServiceUnavailable,
        DBError::StaleCredentialState => ApiError::InvalidJwt,
        _ => {
            tracing::error!("Maple device registration database operation failed");
            ApiError::InternalServerError
        }
    }
}

async fn list_devices(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListMapleDevicesQuery>,
    Extension(user): Extension<User>,
    Extension(auth_context): Extension<AuthContext>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ListMapleDevicesResponse>>, ApiError> {
    let limit = query.limit.unwrap_or(DEFAULT_PAGE_SIZE);
    if limit == 0 || limit > MAX_PAGE_SIZE {
        return Err(ApiError::BadRequest);
    }
    let cursor = query
        .cursor
        .as_deref()
        .map(|cursor| decode_cursor(&state.enclave_key, user.uuid, user.project_id, cursor))
        .transpose()?;

    let page = state
        .db
        .list_maple_devices(
            MapleDeviceListAuthorization {
                user_id: user.uuid,
                project_id: user.project_id,
                auth_credential_kind: match auth_context.method {
                    AuthMethod::Password => "password",
                    AuthMethod::OAuth => "oauth",
                }
                .to_string(),
                auth_binding: auth_context.auth_binding,
                enclave_key: state.enclave_key.clone(),
            },
            i64::from(limit) + 1,
            cursor,
        )
        .map_err(|error| match error {
            DBError::MaplePairingAuthorityBusy
            | DBError::MaplePairingAuthorityCapacityExceeded
            | DBError::MaplePairingAuthorityDeletionBlocked
            | DBError::MaplePairingAuthorityCorrupt => ApiError::from(error),
            DBError::StaleCredentialState => ApiError::InvalidJwt,
            _ => {
                tracing::error!("Maple device list database operation failed");
                ApiError::InternalServerError
            }
        })?;
    let security_epoch = page.security_epoch;
    let mut rows = page.devices;

    let has_more = rows.len() > usize::from(limit);
    if has_more {
        rows.truncate(usize::from(limit));
    }
    let next_cursor = if has_more {
        rows.last()
            .map(|row| {
                encode_cursor(
                    &state.enclave_key,
                    user.uuid,
                    user.project_id,
                    MapleDeviceListCursor {
                        registration_id: row.uuid,
                    },
                )
            })
            .transpose()?
    } else {
        None
    };

    let devices = rows
        .into_iter()
        .map(|row| decrypt_device_response(&state.enclave_key, user.uuid, user.project_id, row))
        .collect::<Result<Vec<_>, _>>()?;

    encrypt_response(
        &state,
        &session_id,
        &ListMapleDevicesResponse {
            protocol_version: PROTOCOL_VERSION_V1,
            security_epoch,
            devices,
            next_cursor,
            has_more,
        },
    )
    .await
}

fn validate_registration(
    request: &RegisterMapleDeviceRequest,
    authenticated_account_id: Uuid,
    authenticated_project_id: Uuid,
) -> Result<ValidatedRegistration, ApiError> {
    if request.protocol_version != PROTOCOL_VERSION_V1
        || request.transcript_version != TRANSCRIPT_VERSION_V1
        || request.asserted_account_id != authenticated_account_id
        || request.asserted_project_id != authenticated_project_id
        || request.identity_algorithm != IDENTITY_ALGORITHM
        || request.operation_id.is_nil()
        || request.device_id.is_nil()
        || request.installation_id.is_nil()
        || matches!(request.expected_revision, Some(revision) if revision <= 0 || revision == i64::MAX)
        || request.known_security_epoch == 0
        || request.known_security_epoch > i64::MAX as u64
        || request.endpoint_epoch > i64::MAX as u64
        || !valid_platform(&request.platform)
        || request.display_name.trim() != request.display_name
        || request.display_name.is_empty()
        || request.display_name.chars().count() > MAX_DISPLAY_NAME_CHARS
        || request.display_name.chars().any(char::is_control)
        || request.identity_public_key.len() != 44
        || !request.identity_public_key.is_ascii()
        || request.iroh_endpoint_id.len() != 64
        || !request.iroh_endpoint_id.is_ascii()
        || request.signature.len() != 88
        || !request.signature.is_ascii()
    {
        return Err(ApiError::BadRequest);
    }

    let public_key = STANDARD
        .decode(&request.identity_public_key)
        .map_err(|_| ApiError::BadRequest)?;
    if STANDARD.encode(&public_key) != request.identity_public_key {
        return Err(ApiError::BadRequest);
    }
    let public_key: [u8; 32] = public_key.try_into().map_err(|_| ApiError::BadRequest)?;
    VerifyingKey::from_bytes(&public_key).map_err(|_| ApiError::BadRequest)?;
    let endpoint_bytes =
        hex::decode(&request.iroh_endpoint_id).map_err(|_| ApiError::BadRequest)?;
    if endpoint_bytes.as_slice() != public_key
        || hex::encode(public_key) != request.iroh_endpoint_id
    {
        return Err(ApiError::BadRequest);
    }

    let signature = STANDARD
        .decode(&request.signature)
        .map_err(|_| ApiError::BadRequest)?;
    if STANDARD.encode(&signature) != request.signature {
        return Err(ApiError::BadRequest);
    }
    let signature: [u8; 64] = signature.try_into().map_err(|_| ApiError::BadRequest)?;

    if request.capabilities.len() > MAX_CAPABILITIES {
        return Err(ApiError::BadRequest);
    }
    let mut seen = HashSet::with_capacity(request.capabilities.len());
    for capability in &request.capabilities {
        if capability.is_empty()
            || capability.len() > MAX_CAPABILITY_LEN
            || !capability.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'_' | b':' | b'-')
            })
            || !seen.insert(capability.as_str())
        {
            return Err(ApiError::BadRequest);
        }
    }
    let iroh_endpoint_addr = validate_iroh_endpoint_addr(&request.iroh_endpoint_addr)?;

    let mut capabilities = request.capabilities.clone();
    capabilities.sort_unstable();

    Ok(ValidatedRegistration {
        identity_public_key: public_key,
        signature,
        iroh_endpoint_addr,
        capabilities,
    })
}

fn valid_platform(platform: &str) -> bool {
    matches!(platform, "macos" | "windows" | "linux" | "ios" | "android")
}

fn validate_iroh_endpoint_addr(
    endpoint_addr: &MapleIrohEndpointAddr,
) -> Result<MapleIrohEndpointAddr, ApiError> {
    if endpoint_addr.relay_urls.len() > MAX_RELAY_URLS
        || endpoint_addr.direct_addresses.len() > MAX_DIRECT_ADDRESSES
        || (endpoint_addr.relay_urls.is_empty() && endpoint_addr.direct_addresses.is_empty())
    {
        return Err(ApiError::BadRequest);
    }

    let mut seen_relays = HashSet::with_capacity(endpoint_addr.relay_urls.len());
    for relay_url in &endpoint_addr.relay_urls {
        if relay_url.is_empty() || relay_url.len() > MAX_RELAY_URL_LEN {
            return Err(ApiError::BadRequest);
        }
        let parsed = Url::parse(relay_url).map_err(|_| ApiError::BadRequest)?;
        if parsed.as_str() != relay_url
            || parsed.scheme() != "https"
            || parsed.host().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.port() == Some(0)
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || !seen_relays.insert(relay_url.as_str())
        {
            return Err(ApiError::BadRequest);
        }
    }

    let mut seen_direct = HashSet::with_capacity(endpoint_addr.direct_addresses.len());
    for direct_address in &endpoint_addr.direct_addresses {
        if direct_address.is_empty() || direct_address.len() > MAX_DIRECT_ADDRESS_LEN {
            return Err(ApiError::BadRequest);
        }
        let parsed = direct_address
            .parse::<SocketAddr>()
            .map_err(|_| ApiError::BadRequest)?;
        if parsed.to_string() != *direct_address
            || parsed.port() == 0
            || parsed.ip().is_unspecified()
            || parsed.ip().is_multicast()
            || parsed.ip() == std::net::IpAddr::V4(std::net::Ipv4Addr::BROADCAST)
            || !seen_direct.insert(direct_address.as_str())
        {
            return Err(ApiError::BadRequest);
        }
    }

    let mut canonical = endpoint_addr.clone();
    canonical.relay_urls.sort_unstable();
    canonical.direct_addresses.sort_unstable();
    Ok(canonical)
}

pub(crate) fn registration_transcript(
    request: &RegisterMapleDeviceRequest,
    identity_public_key: &[u8; 32],
    validated_endpoint_addr: &MapleIrohEndpointAddr,
    sorted_capabilities: &[String],
) -> Vec<u8> {
    let mut transcript = CanonicalBytes::new(REGISTRATION_TRANSCRIPT_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_u64(request.known_security_epoch)
        .append_uuid(request.operation_id)
        .append_uuid(request.device_id)
        .append_uuid(request.installation_id)
        .append_bool(request.expected_revision.is_some());
    if let Some(expected_revision) = request.expected_revision {
        transcript.append_i64(expected_revision);
    }
    transcript
        .append_str(&request.identity_algorithm)
        .append_bytes(identity_public_key)
        .append_bytes(identity_public_key)
        .append_u64(request.endpoint_epoch)
        .append_u16(validated_endpoint_addr.relay_urls.len() as u16);
    for relay_url in &validated_endpoint_addr.relay_urls {
        transcript.append_str(relay_url);
    }
    transcript.append_u16(validated_endpoint_addr.direct_addresses.len() as u16);
    for direct_address in &validated_endpoint_addr.direct_addresses {
        transcript.append_str(direct_address);
    }
    transcript
        .append_str(&request.platform)
        .append_str(&request.display_name)
        .append_u16(sorted_capabilities.len() as u16);
    for capability in sorted_capabilities {
        transcript.append_str(capability);
    }
    transcript.into_bytes()
}

fn registration_operation_mac(
    enclave_key: &[u8],
    transcript: &[u8],
    request_endpoint_addr: &MapleIrohEndpointAddr,
    request_capabilities: &[String],
    signature: &[u8; 64],
) -> Result<[u8; 32], EncryptError> {
    let key = derive_key(enclave_key, OPERATION_MAC_KEY_INFO)?;
    let mut body = CanonicalBytes::new(REGISTRATION_OPERATION_DOMAIN);
    body.append_bytes(transcript)
        .append_u16(request_endpoint_addr.relay_urls.len() as u16);
    // Preserve caller order in the operation fingerprint even though the
    // possession transcript canonicalizes each routing set.
    for relay_url in &request_endpoint_addr.relay_urls {
        body.append_str(relay_url);
    }
    body.append_u16(request_endpoint_addr.direct_addresses.len() as u16);
    for direct_address in &request_endpoint_addr.direct_addresses {
        body.append_str(direct_address);
    }
    body.append_u16(request_capabilities.len() as u16);
    // Preserve caller order in the operation fingerprint even though the
    // possession transcript canonicalizes the capability set. An operation ID
    // therefore cannot be reused with a changed structured request body.
    for capability in request_capabilities {
        body.append_str(capability);
    }
    body.append_bytes(signature);
    let mut mac =
        HmacSha256::new_from_slice(&key).map_err(|_| EncryptError::KeyDerivationFailed)?;
    mac.update(&body.into_bytes());
    let bytes = mac.finalize().into_bytes();
    let mut output = [0u8; 32];
    output.copy_from_slice(&bytes);
    Ok(output)
}

fn maple_device_identity_mac(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    identity_public_key: &[u8; 32],
) -> Result<[u8; 32], EncryptError> {
    let key = derive_key(enclave_key, DEVICE_IDENTITY_MAC_KEY_INFO)?;
    let mut body = CanonicalBytes::new(DEVICE_IDENTITY_MAC_DOMAIN);
    body.append_uuid(account_id)
        .append_i32(project_id)
        .append_bytes(identity_public_key);
    hmac_sha256(&key, &body.into_bytes())
}

fn maple_device_registration_id(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    device_id: Uuid,
    installation_id: Uuid,
) -> Result<Uuid, EncryptError> {
    let key = derive_key(enclave_key, DEVICE_REGISTRATION_ID_KEY_INFO)?;
    let mut body = CanonicalBytes::new(DEVICE_REGISTRATION_ID_DOMAIN);
    body.append_uuid(account_id)
        .append_i32(project_id)
        .append_uuid(device_id)
        .append_uuid(installation_id);
    let mac = hmac_sha256(&key, &body.into_bytes())?;
    let mut uuid_bytes = [0u8; 16];
    uuid_bytes.copy_from_slice(&mac[..16]);
    // Mark this as an RFC 9562-compatible custom/name-derived UUID while the
    // full keyed preimage binding remains server-private.
    uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x80;
    uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;
    Ok(Uuid::from_bytes(uuid_bytes))
}

fn hmac_sha256(key: &[u8], body: &[u8]) -> Result<[u8; 32], EncryptError> {
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| EncryptError::KeyDerivationFailed)?;
    mac.update(body);
    let bytes = mac.finalize().into_bytes();
    let mut output = [0u8; 32];
    output.copy_from_slice(&bytes);
    Ok(output)
}

fn encrypt_maple_device_payload(
    enclave_key: &[u8],
    plaintext: &[u8],
    context: MapleDeviceRecordContext,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, DEVICE_PAYLOAD_KEY_INFO)?;
    let aad = maple_device_payload_aad(context);
    encrypt_aead_v1(&key, plaintext, &aad)
}

#[cfg(test)]
// Explicit fields reproduce the production AEAD trust context in codec tests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_test_maple_device_payload(
    enclave_key: &[u8],
    account_id: Uuid,
    internal_project_id: i32,
    registration_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    revision: i64,
    endpoint_epoch: u64,
    payload_version: i16,
    identity_public_key: [u8; 32],
) -> Result<(Vec<u8>, Vec<u8>), EncryptError> {
    let payload = StoredMapleDevicePayloadV1 {
        registration_id,
        revision,
        identity_algorithm: IDENTITY_ALGORITHM.to_string(),
        identity_public_key: identity_public_key.to_vec(),
        iroh_endpoint_id: hex::encode(identity_public_key),
        endpoint_epoch,
        iroh_endpoint_addr: MapleIrohEndpointAddr {
            relay_urls: Vec::new(),
            direct_addresses: vec!["127.0.0.1:4433".to_string()],
        },
        platform: "macos".to_string(),
        display_name: "Test Device".to_string(),
        capabilities: vec!["agent.v1".to_string()],
    };
    let plaintext = serde_json::to_vec(&payload)
        .map_err(|error| EncryptError::DeserializationFailed(error.to_string()))?;
    let payload_enc = encrypt_maple_device_payload(
        enclave_key,
        &plaintext,
        MapleDeviceRecordContext {
            account_id,
            project_id: internal_project_id,
            registration_id,
            device_id,
            installation_id,
            revision,
            payload_version,
        },
    )?;
    let identity_mac = maple_device_identity_mac(
        enclave_key,
        account_id,
        internal_project_id,
        &identity_public_key,
    )?;
    Ok((identity_mac.to_vec(), payload_enc))
}

fn decrypt_maple_device_payload(
    enclave_key: &[u8],
    encrypted: &[u8],
    context: MapleDeviceRecordContext,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, DEVICE_PAYLOAD_KEY_INFO)?;
    let aad = maple_device_payload_aad(context);
    decrypt_aead_v1(&key, encrypted, &aad)
}

fn maple_device_payload_aad(context: MapleDeviceRecordContext) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(DEVICE_PAYLOAD_DOMAIN);
    aad.append_uuid(context.account_id)
        .append_i32(context.project_id)
        .append_uuid(context.registration_id)
        .append_uuid(context.device_id)
        .append_uuid(context.installation_id)
        .append_i64(context.revision)
        .append_i16(context.payload_version);
    aad.into_bytes()
}

pub(crate) fn decrypt_device_response(
    enclave_key: &[u8],
    expected_account_id: Uuid,
    expected_project_id: i32,
    row: MapleDevice,
) -> Result<MapleDeviceResponse, ApiError> {
    // Query filters are not an authority boundary under the hostile-database
    // model. Reject foreign-scope or MAC-invalid rows before deriving AAD from
    // any returned cleartext fields.
    if row.id <= 0
        || row.user_id != expected_account_id
        || row.project_id != expected_project_id
        || row.payload_version != PAYLOAD_VERSION_V1
        || !crate::db::maple_device_record_mac_is_valid(enclave_key, &row)
            .map_err(|_| ApiError::InternalServerError)?
    {
        return Err(ApiError::InternalServerError);
    }
    let plaintext = decrypt_maple_device_payload(
        enclave_key,
        &row.payload_enc,
        MapleDeviceRecordContext {
            account_id: row.user_id,
            project_id: row.project_id,
            registration_id: row.uuid,
            device_id: row.device_id,
            installation_id: row.installation_id,
            revision: row.revision,
            payload_version: row.payload_version,
        },
    )
    .map_err(|_| ApiError::InternalServerError)?;
    let payload: StoredMapleDevicePayloadV1 =
        serde_json::from_slice(&plaintext).map_err(|_| ApiError::InternalServerError)?;
    let identity_public_key: [u8; 32] = payload
        .identity_public_key
        .as_slice()
        .try_into()
        .map_err(|_| ApiError::InternalServerError)?;
    VerifyingKey::from_bytes(&identity_public_key).map_err(|_| ApiError::InternalServerError)?;
    let identity_mac = maple_device_identity_mac(
        enclave_key,
        row.user_id,
        row.project_id,
        &identity_public_key,
    )
    .map_err(|_| ApiError::InternalServerError)?;
    use subtle::ConstantTimeEq;
    if payload.registration_id != row.uuid
        || payload.revision != row.revision
        || payload.identity_algorithm != IDENTITY_ALGORITHM
        || payload.iroh_endpoint_id != hex::encode(identity_public_key)
        || payload.endpoint_epoch != row.endpoint_epoch as u64
        || validate_iroh_endpoint_addr(&payload.iroh_endpoint_addr)
            .map_or(true, |canonical| canonical != payload.iroh_endpoint_addr)
        || !valid_platform(&payload.platform)
        || payload.display_name.trim() != payload.display_name
        || payload.display_name.is_empty()
        || payload.display_name.chars().count() > MAX_DISPLAY_NAME_CHARS
        || payload.display_name.chars().any(char::is_control)
        || !bool::from(identity_mac.as_slice().ct_eq(row.identity_mac.as_slice()))
        || payload.capabilities.len() > MAX_CAPABILITIES
        || payload
            .capabilities
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || payload.capabilities.iter().any(|capability| {
            capability.is_empty()
                || capability.len() > MAX_CAPABILITY_LEN
                || !capability.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'.' | b'_' | b':' | b'-')
                })
        })
    {
        return Err(ApiError::InternalServerError);
    }

    Ok(MapleDeviceResponse {
        registration_id: row.uuid,
        device_id: row.device_id,
        installation_id: row.installation_id,
        identity_algorithm: payload.identity_algorithm,
        identity_public_key: STANDARD.encode(identity_public_key),
        iroh_endpoint_id: payload.iroh_endpoint_id,
        endpoint_epoch: payload.endpoint_epoch,
        iroh_endpoint_addr: payload.iroh_endpoint_addr,
        platform: payload.platform,
        display_name: payload.display_name,
        capabilities: payload.capabilities,
        revision: row.revision,
    })
}

pub(crate) fn build_reset_clear_material(
    enclave_key: &[u8],
    context: MapleResetClearUnsignedMaterializationContext,
) -> Result<MapleResetClearUnsignedMaterial, MaplePairingMaterializationError> {
    let (host, host_identity_mac) = match &context.source {
        MapleResetClearSource::LiveDevice {
            registration_id,
            device_id,
            installation_id,
            revision,
            endpoint_epoch,
            payload_version,
            payload_enc,
            identity_mac,
            record_mac,
        } => {
            let plaintext = decrypt_maple_device_payload(
                enclave_key,
                payload_enc,
                MapleDeviceRecordContext {
                    account_id: context.account_id,
                    project_id: context.internal_project_id,
                    registration_id: *registration_id,
                    device_id: *device_id,
                    installation_id: *installation_id,
                    revision: *revision,
                    payload_version: *payload_version,
                },
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            let payload: StoredMapleDevicePayloadV1 =
                serde_json::from_slice(&plaintext).map_err(|_| MaplePairingMaterializationError)?;
            let response = reset_source_device_response(
                enclave_key,
                context.account_id,
                context.internal_project_id,
                *registration_id,
                *device_id,
                *installation_id,
                *revision,
                identity_mac,
                *payload_version,
                payload_enc,
                record_mac,
                context.reset_at,
                payload,
            )?;
            if response.endpoint_epoch
                != u64::try_from(*endpoint_epoch).map_err(|_| MaplePairingMaterializationError)?
            {
                return Err(MaplePairingMaterializationError);
            }
            (maple_device_claim(&response), identity_mac.clone())
        }
        MapleResetClearSource::RetainedHostClaim {
            prior_event_id,
            payload_version,
            payload,
            payload_digest,
            identity_mac,
            prior_target_revocation_stream_id,
            prior_target_revocation_stream_generation,
            prior_target_security_epoch,
        } => {
            if *payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1
                || *prior_event_id
                    != context
                        .previous_event_id
                        .ok_or(MaplePairingMaterializationError)?
                || *prior_target_revocation_stream_id != context.source_revocation_stream_id
                || *prior_target_revocation_stream_generation
                    != context.source_revocation_stream_generation
                || *prior_target_security_epoch != context.source_security_epoch
            {
                return Err(MaplePairingMaterializationError);
            }
            use subtle::ConstantTimeEq;
            if !bool::from(
                payload_digest
                    .as_slice()
                    .ct_eq(sha256_digest(payload).as_slice()),
            ) {
                return Err(MaplePairingMaterializationError);
            }
            let claim: MapleDeviceClaimV1 =
                serde_json::from_slice(payload).map_err(|_| MaplePairingMaterializationError)?;
            claim
                .validate()
                .map_err(|_| MaplePairingMaterializationError)?;
            let expected_identity_mac = maple_device_identity_mac(
                enclave_key,
                context.account_id,
                context.internal_project_id,
                &claim
                    .verifying_key_bytes()
                    .map_err(|_| MaplePairingMaterializationError)?,
            )
            .map_err(|_| MaplePairingMaterializationError)?;
            if !bool::from(
                expected_identity_mac
                    .as_slice()
                    .ct_eq(identity_mac.as_slice()),
            ) {
                return Err(MaplePairingMaterializationError);
            }
            (claim, identity_mac.clone())
        }
    };
    if host_identity_mac.len() != 32
        || context.admission_leaves.len()
            > usize::from(crate::models::maple_pairings::MAPLE_RESET_CLEAR_MAX_ADMISSIONS)
    {
        return Err(MaplePairingMaterializationError);
    }
    let leaves = context
        .admission_leaves
        .iter()
        .map(|leaf| MapleResetClearAdmissionLeafV1 {
            pair_id: leaf.pair_id,
            pairing_incarnation: leaf.pairing_incarnation,
            pair_authorization_digest: leaf.pair_authorization_digest,
        })
        .collect::<Vec<_>>();
    let admission_count: u16 = leaves
        .len()
        .try_into()
        .map_err(|_| MaplePairingMaterializationError)?;
    let admission_set_digest =
        reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, &leaves)
            .map_err(|_| MaplePairingMaterializationError)?;
    let host_claim_payload =
        serde_json::to_vec(&host).map_err(|_| MaplePairingMaterializationError)?;
    let host_claim_digest = sha256_digest(&host_claim_payload);
    let mut instruction = MapleResetClearRequiredV1 {
        artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        event_id: context.event_id,
        reset_id: context.reset_id,
        reset_generation: context.reset_generation,
        cumulative_reset_count: context.cumulative_reset_count,
        source_security_epoch: context.source_security_epoch,
        security_epoch: context.security_epoch,
        subject_account_id: context.account_id,
        subject_project_id: context.subject_project_id,
        recipient_host_registration_id: host.registration_id,
        host,
        issuer_sequence: context.issuer_sequence,
        source_revocation_stream_id: context.source_revocation_stream_id,
        source_revocation_stream_generation: context.source_revocation_stream_generation,
        revocation_stream_id: context.revocation_stream_id,
        revocation_stream_generation: context.revocation_stream_generation,
        clear_scope: MapleResetClearScopeV1::AllPairAuthorizationsForAccountProjectHostInstallation,
        admission_count,
        admission_set_digest: STANDARD.encode(admission_set_digest),
        previous_reset_clear_event_id: context.previous_event_id,
        previous_instruction_material_digest: context
            .previous_instruction_material_digest
            .map(|digest| STANDARD.encode(digest)),
        previous_chain_digest: context
            .previous_chain_digest
            .map(|digest| STANDARD.encode(digest)),
        reset_at_unix_ms: context.reset_at.timestamp_millis(),
        instruction_material_digest: String::new(),
        chain_digest: String::new(),
        issuer_key_id: String::new(),
        issuer_signature: String::new(),
    };
    let instruction_material_transcript = reset_clear_instruction_material_transcript(&instruction)
        .map_err(|_| MaplePairingMaterializationError)?;
    let instruction_material_digest = sha256_digest(&instruction_material_transcript);
    instruction.instruction_material_digest = STANDARD.encode(instruction_material_digest);
    instruction.chain_digest = STANDARD.encode(sha256_digest(
        &reset_clear_chain_transcript(&instruction)
            .map_err(|_| MaplePairingMaterializationError)?,
    ));
    let chain_digest = decode_fixed_base64::<32>(&instruction.chain_digest)
        .map_err(|_| MaplePairingMaterializationError)?;
    let instruction_payload =
        serde_json::to_vec(&instruction).map_err(|_| MaplePairingMaterializationError)?;
    Ok(MapleResetClearUnsignedMaterial {
        host_identity_mac,
        host_claim_payload_version: RESET_CLEAR_PAYLOAD_VERSION_V1,
        host_claim_payload,
        host_claim_digest,
        instruction_payload_version: RESET_CLEAR_PAYLOAD_VERSION_V1,
        instruction_payload,
        instruction_material_transcript,
        instruction_material_digest,
        chain_digest,
    })
}

#[allow(clippy::too_many_arguments)]
fn reset_source_device_response(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    registration_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    revision: i64,
    identity_mac: &[u8],
    payload_version: i16,
    payload_enc: &[u8],
    record_mac: &[u8],
    source_timestamp: DateTime<Utc>,
    payload: StoredMapleDevicePayloadV1,
) -> Result<MapleDeviceResponse, MaplePairingMaterializationError> {
    let identity_public_key: [u8; 32] = payload
        .identity_public_key
        .as_slice()
        .try_into()
        .map_err(|_| MaplePairingMaterializationError)?;
    VerifyingKey::from_bytes(&identity_public_key).map_err(|_| MaplePairingMaterializationError)?;
    let expected_identity_mac =
        maple_device_identity_mac(enclave_key, account_id, project_id, &identity_public_key)
            .map_err(|_| MaplePairingMaterializationError)?;
    use subtle::ConstantTimeEq;
    let synthetic_row = MapleDevice {
        id: 1,
        uuid: registration_id,
        user_id: account_id,
        project_id,
        device_id,
        installation_id,
        identity_mac: identity_mac.to_vec(),
        endpoint_epoch: payload
            .endpoint_epoch
            .try_into()
            .map_err(|_| MaplePairingMaterializationError)?,
        payload_version,
        payload_enc: payload_enc.to_vec(),
        record_mac: record_mac.to_vec(),
        revision,
        registered_at: source_timestamp,
        updated_at: source_timestamp,
    };
    if !crate::db::maple_device_record_mac_is_valid(enclave_key, &synthetic_row)
        .map_err(|_| MaplePairingMaterializationError)?
        || payload.registration_id != registration_id
        || payload.revision != revision
        || payload.identity_algorithm != IDENTITY_ALGORITHM
        || payload.iroh_endpoint_id != hex::encode(identity_public_key)
        || validate_iroh_endpoint_addr(&payload.iroh_endpoint_addr)
            .map_or(true, |canonical| canonical != payload.iroh_endpoint_addr)
        || !valid_platform(&payload.platform)
        || payload.display_name.trim() != payload.display_name
        || payload.display_name.is_empty()
        || payload.display_name.chars().count() > MAX_DISPLAY_NAME_CHARS
        || payload.display_name.chars().any(char::is_control)
        || !bool::from(expected_identity_mac.as_slice().ct_eq(identity_mac))
        || payload.capabilities.len() > MAX_CAPABILITIES
        || payload
            .capabilities
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || payload.capabilities.iter().any(|capability| {
            capability.is_empty()
                || capability.len() > MAX_CAPABILITY_LEN
                || !capability.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'.' | b'_' | b':' | b'-')
                })
        })
    {
        return Err(MaplePairingMaterializationError);
    }
    Ok(MapleDeviceResponse {
        registration_id,
        device_id,
        installation_id,
        identity_algorithm: payload.identity_algorithm,
        identity_public_key: STANDARD.encode(identity_public_key),
        iroh_endpoint_id: payload.iroh_endpoint_id,
        endpoint_epoch: payload.endpoint_epoch,
        iroh_endpoint_addr: payload.iroh_endpoint_addr,
        platform: payload.platform,
        display_name: payload.display_name,
        capabilities: payload.capabilities,
        revision,
    })
}

fn maple_device_claim(device: &MapleDeviceResponse) -> MapleDeviceClaimV1 {
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

fn decode_fixed_base64<const N: usize>(value: &str) -> Result<[u8; N], ()> {
    let decoded = STANDARD.decode(value).map_err(|_| ())?;
    if STANDARD.encode(&decoded) != value {
        return Err(());
    }
    decoded.try_into().map_err(|_| ())
}

fn encode_cursor(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    cursor: MapleDeviceListCursor,
) -> Result<String, ApiError> {
    let body = maple_device_cursor_body(account_id, project_id, cursor);
    let key = derive_key(enclave_key, DEVICE_CURSOR_MAC_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    let mac = hmac_sha256(&key, &body).map_err(|_| ApiError::InternalServerError)?;
    let mut encoded = Vec::with_capacity(48);
    encoded.extend_from_slice(cursor.registration_id.as_bytes());
    encoded.extend_from_slice(&mac);
    Ok(URL_SAFE_NO_PAD.encode(encoded))
}

fn decode_cursor(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    cursor: &str,
) -> Result<MapleDeviceListCursor, ApiError> {
    if cursor.len() != 64 || !cursor.is_ascii() {
        return Err(ApiError::BadRequest);
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| ApiError::BadRequest)?;
    if URL_SAFE_NO_PAD.encode(&bytes) != cursor {
        return Err(ApiError::BadRequest);
    }
    let bytes: [u8; 48] = bytes.try_into().map_err(|_| ApiError::BadRequest)?;
    let registration_id =
        Uuid::from_bytes(bytes[0..16].try_into().map_err(|_| ApiError::BadRequest)?);
    if registration_id.is_nil() {
        return Err(ApiError::BadRequest);
    }
    let decoded = MapleDeviceListCursor { registration_id };
    let body = maple_device_cursor_body(account_id, project_id, decoded);
    let key = derive_key(enclave_key, DEVICE_CURSOR_MAC_KEY_INFO)
        .map_err(|_| ApiError::InternalServerError)?;
    let expected = hmac_sha256(&key, &body).map_err(|_| ApiError::InternalServerError)?;
    use subtle::ConstantTimeEq;
    if !bool::from(expected.as_slice().ct_eq(&bytes[16..])) {
        return Err(ApiError::BadRequest);
    }
    Ok(decoded)
}

fn maple_device_cursor_body(
    account_id: Uuid,
    project_id: i32,
    cursor: MapleDeviceListCursor,
) -> Vec<u8> {
    let mut body = CanonicalBytes::new(DEVICE_CURSOR_DOMAIN);
    body.append_uuid(account_id)
        .append_i32(project_id)
        .append_uuid(cursor.registration_id);
    body.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn sample_iroh_addr() -> MapleIrohEndpointAddr {
        validate_iroh_endpoint_addr(&MapleIrohEndpointAddr {
            relay_urls: vec![
                "https://use1-1.relay.n0.iroh.link./".to_string(),
                "https://euw1-1.relay.n0.iroh.link./".to_string(),
            ],
            direct_addresses: vec![
                "[2001:db8::1]:4433".to_string(),
                "203.0.113.7:4433".to_string(),
            ],
        })
        .unwrap()
    }

    fn signed_request(secret: &SigningKey) -> RegisterMapleDeviceRequest {
        let public = secret.verifying_key();
        let mut request = RegisterMapleDeviceRequest {
            protocol_version: PROTOCOL_VERSION_V1,
            transcript_version: TRANSCRIPT_VERSION_V1,
            operation_id: Uuid::from_u128(1),
            device_id: Uuid::from_u128(2),
            installation_id: Uuid::from_u128(3),
            expected_revision: None,
            known_security_epoch: 1,
            asserted_account_id: Uuid::from_u128(4),
            asserted_project_id: Uuid::from_u128(5),
            identity_algorithm: IDENTITY_ALGORITHM.to_string(),
            identity_public_key: STANDARD.encode(public.as_bytes()),
            iroh_endpoint_id: hex::encode(public.as_bytes()),
            endpoint_epoch: 7,
            iroh_endpoint_addr: sample_iroh_addr(),
            platform: "macos".to_string(),
            display_name: "MacBook Pro".to_string(),
            capabilities: vec!["maple.remote.host".to_string(), "agent.v1".to_string()],
            signature: String::new(),
        };
        let validated = validate_registration(
            &RegisterMapleDeviceRequest {
                signature: STANDARD.encode([0u8; 64]),
                ..request.clone()
            },
            request.asserted_account_id,
            request.asserted_project_id,
        )
        .unwrap();
        let transcript = registration_transcript(
            &request,
            public.as_bytes(),
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        request.signature = STANDARD.encode(secret.sign(&transcript).to_bytes());
        request
    }

    #[test]
    fn possession_proof_binds_account_project_and_every_registration_field() {
        let secret = SigningKey::from_bytes(&[17u8; 32]);
        let request = signed_request(&secret);
        let validated = validate_registration(
            &request,
            request.asserted_account_id,
            request.asserted_project_id,
        )
        .unwrap();
        let transcript = registration_transcript(
            &request,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        let signature = Signature::from_bytes(&validated.signature);
        secret
            .verifying_key()
            .verify_strict(&transcript, &signature)
            .unwrap();

        let mut changed = request.clone();
        changed.asserted_account_id = Uuid::from_u128(99);
        let changed_transcript = registration_transcript(
            &changed,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        assert!(secret
            .verifying_key()
            .verify_strict(&changed_transcript, &signature)
            .is_err());

        let mut changed = request.clone();
        changed.iroh_endpoint_addr.direct_addresses[0] = "203.0.113.8:4433".to_string();
        let changed_endpoint_addr =
            validate_iroh_endpoint_addr(&changed.iroh_endpoint_addr).unwrap();
        let changed_transcript = registration_transcript(
            &changed,
            &validated.identity_public_key,
            &changed_endpoint_addr,
            &validated.capabilities,
        );
        assert!(secret
            .verifying_key()
            .verify_strict(&changed_transcript, &signature)
            .is_err());

        let mut changed = request.clone();
        changed.endpoint_epoch += 1;
        let changed_transcript = registration_transcript(
            &changed,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        assert!(secret
            .verifying_key()
            .verify_strict(&changed_transcript, &signature)
            .is_err());

        let mut changed = request.clone();
        changed.expected_revision = Some(1);
        let changed_transcript = registration_transcript(
            &changed,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        assert!(secret
            .verifying_key()
            .verify_strict(&changed_transcript, &signature)
            .is_err());
    }

    #[test]
    fn canonical_transcript_matches_sdk_v1_test_vector() {
        let key = SigningKey::from_bytes(&[17u8; 32])
            .verifying_key()
            .to_bytes();
        let request = RegisterMapleDeviceRequest {
            protocol_version: 1,
            transcript_version: 1,
            operation_id: "550e8400-e29b-41d4-a716-446655440000".parse().unwrap(),
            asserted_account_id: "550e8400-e29b-41d4-a716-446655440001".parse().unwrap(),
            asserted_project_id: "550e8400-e29b-41d4-a716-446655440002".parse().unwrap(),
            device_id: "550e8400-e29b-41d4-a716-446655440003".parse().unwrap(),
            installation_id: "550e8400-e29b-41d4-a716-446655440004".parse().unwrap(),
            expected_revision: None,
            known_security_epoch: 1,
            identity_algorithm: "ed25519".to_string(),
            identity_public_key: STANDARD.encode(key),
            iroh_endpoint_id: hex::encode(key),
            endpoint_epoch: 7,
            iroh_endpoint_addr: sample_iroh_addr(),
            platform: "macos".to_string(),
            display_name: "MacBook Pro".to_string(),
            capabilities: vec!["agent.host".to_string(), "agent.control".to_string()],
            signature: STANDARD.encode([0u8; 64]),
        };
        let mut capabilities = request.capabilities.clone();
        capabilities.sort_unstable();
        let transcript =
            registration_transcript(&request, &key, &request.iroh_endpoint_addr, &capabilities);
        assert_eq!(
            hex::encode(transcript),
            "730000001f6f732e6d61706c652d6465766963652d726567697374726174696f6e2e76316a0000000200016a0000000200017500000010550e8400e29b41d4a7164466554400017500000010550e8400e29b41d4a7164466554400024c0000000800000000000000017500000010550e8400e29b41d4a7164466554400007500000010550e8400e29b41d4a7164466554400037500000010550e8400e29b41d4a7164466554400043f00000001007300000007656432353531396200000020d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787376200000020d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787374c0000000800000000000000076a000000020002730000002368747470733a2f2f657577312d312e72656c61792e6e302e69726f682e6c696e6b2e2f730000002368747470733a2f2f757365312d312e72656c61792e6e302e69726f682e6c696e6b2e2f6a00000002000273000000103230332e302e3131332e373a3434333373000000125b323030313a6462383a3a315d3a3434333373000000056d61636f73730000000b4d6163426f6f6b2050726f6a000000020002730000000d6167656e742e636f6e74726f6c730000000a6167656e742e686f7374"
        );

        let mut update_request = request;
        update_request.expected_revision = Some(7);
        let update_transcript = registration_transcript(
            &update_request,
            &key,
            &update_request.iroh_endpoint_addr,
            &capabilities,
        );
        assert_eq!(
            hex::encode(&update_transcript),
            "730000001f6f732e6d61706c652d6465766963652d726567697374726174696f6e2e76316a0000000200016a0000000200017500000010550e8400e29b41d4a7164466554400017500000010550e8400e29b41d4a7164466554400024c0000000800000000000000017500000010550e8400e29b41d4a7164466554400007500000010550e8400e29b41d4a7164466554400037500000010550e8400e29b41d4a7164466554400043f00000001016c0000000800000000000000077300000007656432353531396200000020d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787376200000020d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787374c0000000800000000000000076a000000020002730000002368747470733a2f2f657577312d312e72656c61792e6e302e69726f682e6c696e6b2e2f730000002368747470733a2f2f757365312d312e72656c61792e6e302e69726f682e6c696e6b2e2f6a00000002000273000000103230332e302e3131332e373a3434333373000000125b323030313a6462383a3a315d3a3434333373000000056d61636f73730000000b4d6163426f6f6b2050726f6a000000020002730000000d6167656e742e636f6e74726f6c730000000a6167656e742e686f7374"
        );
        let signing_key = SigningKey::from_bytes(&[17u8; 32]);
        assert_eq!(
            STANDARD.encode(signing_key.sign(&update_transcript).to_bytes()),
            "kBzvA/AavRvBLVDX4vhZuPlgsrwlOpEDZPQvwH5Frn484xCHVu+EH/Tn4pnGyOrecnH5p/A2LLNsekqFUzBuDw=="
        );
    }

    #[test]
    fn frozen_security_epoch_registration_vectors_match_wire_contract() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/maple_pairing_v1_vectors.json"
        ))
        .unwrap();
        for (request_key, transcript_key, digest_key) in [
            (
                "register_device_request_epoch_1",
                "register_device_request_epoch_1_transcript_hex",
                "register_device_request_epoch_1_digest",
            ),
            (
                "register_device_request_epoch_4",
                "register_device_request_epoch_4_transcript_hex",
                "register_device_request_epoch_4_digest",
            ),
        ] {
            let request: RegisterMapleDeviceRequest =
                serde_json::from_value(vectors[request_key].clone()).unwrap();
            let validated = validate_registration(
                &request,
                request.asserted_account_id,
                request.asserted_project_id,
            )
            .unwrap();
            let transcript = registration_transcript(
                &request,
                &validated.identity_public_key,
                &validated.iroh_endpoint_addr,
                &validated.capabilities,
            );
            assert_eq!(
                hex::encode(&transcript),
                vectors[transcript_key].as_str().unwrap()
            );
            assert_eq!(
                sha256_digest(&transcript),
                decode_fixed_base64::<32>(vectors[digest_key].as_str().unwrap()).unwrap()
            );
            VerifyingKey::from_bytes(&validated.identity_public_key)
                .unwrap()
                .verify_strict(&transcript, &Signature::from_bytes(&validated.signature))
                .unwrap();

            let json = serde_json::to_string(&request).unwrap();
            assert!(json
                .starts_with("{\"protocol_version\":1,\"transcript_version\":1,\"operation_id\":"));
            assert!(json.contains("\"expected_revision\":5,\"known_security_epoch\":"));
            assert!(
                json.contains("\"known_security_epoch\":1,\"asserted_account_id\":")
                    || json.contains("\"known_security_epoch\":4,\"asserted_account_id\":")
            );
        }

        let list_response: ListMapleDevicesResponse =
            serde_json::from_value(vectors["list_devices_response_security_epoch"].clone())
                .unwrap();
        assert_eq!(list_response.security_epoch, 4);
        assert_eq!(
            serde_json::to_value(list_response).unwrap(),
            vectors["list_devices_response_security_epoch"]
        );
    }

    #[test]
    fn authentication_facts_are_preconditions_not_authority_selectors() {
        let secret = SigningKey::from_bytes(&[23u8; 32]);
        let request = signed_request(&secret);
        assert!(
            validate_registration(&request, Uuid::from_u128(999), request.asserted_project_id)
                .is_err()
        );
        assert!(
            validate_registration(&request, request.asserted_account_id, Uuid::from_u128(999))
                .is_err()
        );
    }

    #[test]
    fn request_debug_redacts_device_identity_endpoint_signature_and_name() {
        let request = signed_request(&SigningKey::from_bytes(&[29u8; 32]));
        let debug = format!("{request:?}");
        assert!(!debug.contains(&request.identity_public_key));
        assert!(!debug.contains(&request.iroh_endpoint_id));
        assert!(!debug.contains(&request.signature));
        assert!(!debug.contains(&request.display_name));
        assert!(!request
            .iroh_endpoint_addr
            .relay_urls
            .iter()
            .any(|relay_url| debug.contains(relay_url)));
        assert!(!request
            .iroh_endpoint_addr
            .direct_addresses
            .iter()
            .any(|address| debug.contains(address)));
    }

    #[test]
    fn registration_response_debug_redacts_all_authority_material() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/maple_pairing_v1_vectors.json"
        ))
        .unwrap();
        let response: RegisterMapleDeviceResponse = serde_json::from_value(
            vectors["register_device_response_reset_clear_required"].clone(),
        )
        .unwrap();
        let debug = format!("{response:?}");
        assert!(debug.contains("[redacted]"));
        for secret in [
            response.operation_id.to_string(),
            response.registration_id.to_string(),
            response.device_id.to_string(),
            response
                .revocation_sync
                .stream_checkpoint
                .revocation_stream_id
                .to_string(),
            response
                .revocation_sync
                .stream_checkpoint
                .issuer_key_id
                .clone(),
            response
                .revocation_sync
                .stream_checkpoint
                .issuer_signature
                .clone(),
        ] {
            assert!(!debug.contains(&secret), "Debug leaked authority material");
        }
    }

    #[test]
    fn keyed_registration_and_identity_lookups_are_tenant_scoped() {
        let root_key = [37u8; 32];
        let account = Uuid::from_u128(100);
        let device = Uuid::from_u128(101);
        let installation = Uuid::from_u128(102);
        let identity = SigningKey::from_bytes(&[38u8; 32])
            .verifying_key()
            .to_bytes();
        let registration =
            maple_device_registration_id(&root_key, account, 9, device, installation).unwrap();
        assert_eq!(
            registration,
            maple_device_registration_id(&root_key, account, 9, device, installation).unwrap()
        );
        assert_ne!(
            registration,
            maple_device_registration_id(&root_key, Uuid::from_u128(999), 9, device, installation)
                .unwrap()
        );
        assert_ne!(
            maple_device_identity_mac(&root_key, account, 9, &identity).unwrap(),
            maple_device_identity_mac(&root_key, account, 10, &identity).unwrap()
        );
    }

    #[test]
    fn iroh_endpoint_must_be_the_same_key_in_canonical_form() {
        let secret = SigningKey::from_bytes(&[31u8; 32]);
        let mut request = signed_request(&secret);
        request.iroh_endpoint_id = hex::encode(
            SigningKey::from_bytes(&[32u8; 32])
                .verifying_key()
                .as_bytes(),
        );
        assert!(validate_registration(
            &request,
            request.asserted_account_id,
            request.asserted_project_id
        )
        .is_err());

        request.iroh_endpoint_id =
            hex::encode(secret.verifying_key().as_bytes()).to_ascii_uppercase();
        assert!(validate_registration(
            &request,
            request.asserted_account_id,
            request.asserted_project_id
        )
        .is_err());
    }

    #[test]
    fn encoded_identity_signature_and_cursor_require_exact_preparse_lengths() {
        let secret = SigningKey::from_bytes(&[32u8; 32]);
        let request = signed_request(&secret);
        for changed in [
            RegisterMapleDeviceRequest {
                identity_public_key: "A".repeat(45),
                ..request.clone()
            },
            RegisterMapleDeviceRequest {
                iroh_endpoint_id: "a".repeat(65),
                ..request.clone()
            },
            RegisterMapleDeviceRequest {
                signature: "A".repeat(89),
                ..request.clone()
            },
        ] {
            assert!(validate_registration(
                &changed,
                changed.asserted_account_id,
                changed.asserted_project_id,
            )
            .is_err());
        }

        assert!(decode_cursor(&[9u8; 32], Uuid::from_u128(1), 2, &"A".repeat(65)).is_err());
    }

    #[test]
    fn iroh_endpoint_address_is_bounded_canonical_and_has_no_private_field() {
        let endpoint_addr = sample_iroh_addr();
        assert_eq!(
            serde_json::to_value(&endpoint_addr).unwrap(),
            serde_json::json!({
                "relay_urls": [
                    "https://euw1-1.relay.n0.iroh.link./",
                    "https://use1-1.relay.n0.iroh.link./"
                ],
                "direct_addresses": ["203.0.113.7:4433", "[2001:db8::1]:4433"]
            })
        );

        for invalid in [
            MapleIrohEndpointAddr {
                relay_urls: vec![],
                direct_addresses: vec![],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec!["http://relay.example/".to_string()],
                direct_addresses: vec![],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec!["https://user@relay.example/".to_string()],
                direct_addresses: vec![],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec!["A".repeat(MAX_RELAY_URL_LEN + 1)],
                direct_addresses: vec![],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec![],
                direct_addresses: vec!["0.0.0.0:4433".to_string()],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec![],
                direct_addresses: vec!["255.255.255.255:4433".to_string()],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec![],
                direct_addresses: vec!["1".repeat(MAX_DIRECT_ADDRESS_LEN + 1)],
            },
            MapleIrohEndpointAddr {
                relay_urls: vec![],
                direct_addresses: vec!["2001:db8::1:4433".to_string()],
            },
        ] {
            assert!(validate_iroh_endpoint_addr(&invalid).is_err());
        }

        assert!(
            serde_json::from_value::<MapleIrohEndpointAddr>(serde_json::json!({
                "relay_urls": ["https://relay.example/"],
                "direct_addresses": [],
                "private_key": "must-not-fit-this-schema"
            }))
            .is_err()
        );
    }

    #[test]
    fn device_response_literal_json_preserves_endpoint_address_contract() {
        let literal = serde_json::json!({
            "registration_id": "550e8400-e29b-41d4-a716-446655440005",
            "device_id": "550e8400-e29b-41d4-a716-446655440003",
            "installation_id": "550e8400-e29b-41d4-a716-446655440004",
            "identity_algorithm": "ed25519",
            "identity_public_key": "0EqyMnQrtKs6E2i9RhXk5tAiSrcaAWuvhSCjMsl3hzc=",
            "iroh_endpoint_id": "d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c9778737",
            "endpoint_epoch": 7,
            "iroh_endpoint_addr": {
                "relay_urls": ["https://relay.example/"],
                "direct_addresses": ["192.0.2.7:7777"]
            },
            "platform": "macos",
            "display_name": "MacBook Pro",
            "capabilities": ["agent.host"],
            "revision": 2
        });
        let response: MapleDeviceResponse = serde_json::from_value(literal.clone()).unwrap();
        assert_eq!(serde_json::to_value(response).unwrap(), literal);
    }

    #[test]
    fn address_refresh_keeps_identity_epoch_and_uses_revision_cas() {
        let secret = SigningKey::from_bytes(&[33u8; 32]);
        let original = signed_request(&secret);
        let mut refresh = original.clone();
        refresh.operation_id = Uuid::from_u128(44);
        refresh.expected_revision = Some(1);
        refresh.iroh_endpoint_addr = validate_iroh_endpoint_addr(&MapleIrohEndpointAddr {
            relay_urls: vec!["https://relay.example/".to_string()],
            direct_addresses: vec![],
        })
        .unwrap();
        refresh.signature = STANDARD.encode([0u8; 64]);
        let validated = validate_registration(
            &refresh,
            refresh.asserted_account_id,
            refresh.asserted_project_id,
        )
        .unwrap();
        let transcript = registration_transcript(
            &refresh,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );
        refresh.signature = STANDARD.encode(secret.sign(&transcript).to_bytes());

        assert_eq!(refresh.endpoint_epoch, original.endpoint_epoch);
        assert_eq!(refresh.expected_revision, Some(1));
        validate_registration(
            &refresh,
            refresh.asserted_account_id,
            refresh.asserted_project_id,
        )
        .unwrap();
    }

    #[test]
    fn operation_fingerprint_rejects_reordered_route_body_with_same_transcript() {
        let secret = SigningKey::from_bytes(&[35u8; 32]);
        let request = signed_request(&secret);
        let validated = validate_registration(
            &request,
            request.asserted_account_id,
            request.asserted_project_id,
        )
        .unwrap();
        let transcript = registration_transcript(
            &request,
            &validated.identity_public_key,
            &validated.iroh_endpoint_addr,
            &validated.capabilities,
        );

        let mut reordered = request.clone();
        reordered.iroh_endpoint_addr.relay_urls.reverse();
        reordered.iroh_endpoint_addr.direct_addresses.reverse();
        let reordered_validated = validate_registration(
            &reordered,
            reordered.asserted_account_id,
            reordered.asserted_project_id,
        )
        .unwrap();
        let reordered_transcript = registration_transcript(
            &reordered,
            &reordered_validated.identity_public_key,
            &reordered_validated.iroh_endpoint_addr,
            &reordered_validated.capabilities,
        );
        assert_eq!(transcript, reordered_transcript);

        let root_key = [36u8; 32];
        let request_mac = registration_operation_mac(
            &root_key,
            &transcript,
            &request.iroh_endpoint_addr,
            &request.capabilities,
            &validated.signature,
        )
        .unwrap();
        let reordered_mac = registration_operation_mac(
            &root_key,
            &reordered_transcript,
            &reordered.iroh_endpoint_addr,
            &reordered.capabilities,
            &reordered_validated.signature,
        )
        .unwrap();
        assert_ne!(request_mac, reordered_mac);
    }

    #[test]
    fn encrypted_device_payload_rejects_cross_account_and_row_substitution() {
        let root_key = [41u8; 32];
        let account = Uuid::from_u128(10);
        let registration = Uuid::from_u128(20);
        let device = Uuid::from_u128(11);
        let installation = Uuid::from_u128(12);
        let context = MapleDeviceRecordContext {
            account_id: account,
            project_id: 13,
            registration_id: registration,
            device_id: device,
            installation_id: installation,
            revision: 1,
            payload_version: PAYLOAD_VERSION_V1,
        };
        let encrypted =
            encrypt_maple_device_payload(&root_key, b"device payload", context).unwrap();
        assert_eq!(
            decrypt_maple_device_payload(&root_key, &encrypted, context).unwrap(),
            b"device payload"
        );
        assert!(decrypt_maple_device_payload(
            &root_key,
            &encrypted,
            MapleDeviceRecordContext {
                account_id: Uuid::from_u128(99),
                ..context
            },
        )
        .is_err());
        assert!(decrypt_maple_device_payload(
            &root_key,
            &encrypted,
            MapleDeviceRecordContext {
                registration_id: Uuid::from_u128(999),
                ..context
            },
        )
        .is_err());
        assert!(decrypt_maple_device_payload(
            &root_key,
            &encrypted,
            MapleDeviceRecordContext {
                installation_id: Uuid::from_u128(99),
                ..context
            },
        )
        .is_err());
        assert!(decrypt_maple_device_payload(
            &root_key,
            &encrypted,
            MapleDeviceRecordContext {
                revision: 2,
                ..context
            },
        )
        .is_err());
    }

    #[test]
    fn list_conversion_rejects_foreign_scope_before_decrypting_valid_foreign_row() {
        let root_key = [42u8; 32];
        let foreign_account = Uuid::from_u128(500);
        let foreign_project = 9;
        let registration_id = Uuid::from_u128(501);
        let device_id = Uuid::from_u128(502);
        let installation_id = Uuid::from_u128(503);
        let identity = SigningKey::from_bytes(&[43u8; 32])
            .verifying_key()
            .to_bytes();
        let identity_mac =
            maple_device_identity_mac(&root_key, foreign_account, foreign_project, &identity)
                .unwrap();
        let payload = StoredMapleDevicePayloadV1 {
            registration_id,
            revision: 1,
            identity_algorithm: IDENTITY_ALGORITHM.to_string(),
            identity_public_key: identity.to_vec(),
            iroh_endpoint_id: hex::encode(identity),
            endpoint_epoch: 1,
            iroh_endpoint_addr: sample_iroh_addr(),
            platform: "macos".to_string(),
            display_name: "Foreign host".to_string(),
            capabilities: vec!["agent.host".to_string()],
        };
        let payload_enc = encrypt_maple_device_payload(
            &root_key,
            &serde_json::to_vec(&payload).unwrap(),
            MapleDeviceRecordContext {
                account_id: foreign_account,
                project_id: foreign_project,
                registration_id,
                device_id,
                installation_id,
                revision: 1,
                payload_version: PAYLOAD_VERSION_V1,
            },
        )
        .unwrap();
        let now = Utc::now();
        let mut row = MapleDevice {
            id: 10,
            uuid: registration_id,
            user_id: foreign_account,
            project_id: foreign_project,
            device_id,
            installation_id,
            identity_mac: identity_mac.to_vec(),
            endpoint_epoch: 1,
            payload_version: PAYLOAD_VERSION_V1,
            payload_enc,
            record_mac: Vec::new(),
            revision: 1,
            registered_at: now,
            updated_at: now,
        };
        row.record_mac = crate::db::maple_device_record_mac_for_test(&root_key, &row).unwrap();

        let mut oversized = row.clone();
        oversized.payload_enc = vec![0u8; crate::db::MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES + 1];
        assert!(
            decrypt_device_response(&root_key, foreign_account, foreign_project, oversized,)
                .is_err()
        );

        assert!(
            decrypt_device_response(&root_key, Uuid::from_u128(999), foreign_project, row,)
                .is_err()
        );
    }

    #[test]
    fn reset_clear_materializer_handles_live_and_retained_host_sources() {
        use crate::models::maple_pairing_db::MapleResetClearAdmissionMaterial;
        use chrono::TimeZone;

        let root_key = [44u8; 32];
        let account_id = Uuid::from_u128(600);
        let subject_project_id = Uuid::from_u128(601);
        let internal_project_id = 17;
        let registration_id = Uuid::from_u128(602);
        let device_id = Uuid::from_u128(603);
        let installation_id = Uuid::from_u128(604);
        let identity = SigningKey::from_bytes(&[45u8; 32])
            .verifying_key()
            .to_bytes();
        let identity_mac =
            maple_device_identity_mac(&root_key, account_id, internal_project_id, &identity)
                .unwrap();
        let payload = StoredMapleDevicePayloadV1 {
            registration_id,
            revision: 3,
            identity_algorithm: IDENTITY_ALGORITHM.to_string(),
            identity_public_key: identity.to_vec(),
            iroh_endpoint_id: hex::encode(identity),
            endpoint_epoch: 7,
            iroh_endpoint_addr: sample_iroh_addr(),
            platform: "macos".to_string(),
            display_name: "Reset material host".to_string(),
            capabilities: vec!["agent.host".to_string(), "maple.remote.host".to_string()],
        };
        let payload_enc = encrypt_maple_device_payload(
            &root_key,
            &serde_json::to_vec(&payload).unwrap(),
            MapleDeviceRecordContext {
                account_id,
                project_id: internal_project_id,
                registration_id,
                device_id,
                installation_id,
                revision: payload.revision,
                payload_version: PAYLOAD_VERSION_V1,
            },
        )
        .unwrap();
        let reset_at = Utc.timestamp_millis_opt(1_786_579_700_000).unwrap();
        let mut row = MapleDevice {
            id: 91,
            uuid: registration_id,
            user_id: account_id,
            project_id: internal_project_id,
            device_id,
            installation_id,
            identity_mac: identity_mac.to_vec(),
            endpoint_epoch: payload.endpoint_epoch as i64,
            payload_version: PAYLOAD_VERSION_V1,
            payload_enc: payload_enc.clone(),
            record_mac: Vec::new(),
            revision: payload.revision,
            registered_at: reset_at,
            updated_at: reset_at,
        };
        row.record_mac = crate::db::maple_device_record_mac_for_test(&root_key, &row).unwrap();
        let first_event_id = Uuid::from_u128(605);
        let first_target_stream = Uuid::from_u128(607);
        let first = build_reset_clear_material(
            &root_key,
            MapleResetClearUnsignedMaterializationContext {
                account_id,
                subject_project_id,
                internal_project_id,
                source: MapleResetClearSource::LiveDevice {
                    registration_id,
                    device_id,
                    installation_id,
                    revision: row.revision,
                    endpoint_epoch: row.endpoint_epoch,
                    payload_version: row.payload_version,
                    payload_enc: row.payload_enc.clone(),
                    identity_mac: row.identity_mac.clone(),
                    record_mac: row.record_mac.clone(),
                },
                event_id: first_event_id,
                reset_id: Uuid::from_u128(606),
                reset_generation: 1,
                cumulative_reset_count: 1,
                source_security_epoch: 1,
                security_epoch: 2,
                source_revocation_stream_id: Uuid::from_u128(608),
                source_revocation_stream_generation: 5,
                source_last_issued_revocation_sequence: 9,
                revocation_stream_id: first_target_stream,
                revocation_stream_generation: 6,
                issuer_sequence: 1,
                previous_event_id: None,
                previous_instruction_material_digest: None,
                previous_chain_digest: None,
                admission_leaves: vec![MapleResetClearAdmissionMaterial {
                    pair_id: Uuid::from_u128(609),
                    pairing_incarnation: 3,
                    pair_authorization_digest: [46u8; 32],
                }],
                reset_at,
            },
        )
        .unwrap();
        assert_eq!(
            first.host_claim_payload_version,
            RESET_CLEAR_PAYLOAD_VERSION_V1
        );
        assert_eq!(
            first.instruction_payload_version,
            RESET_CLEAR_PAYLOAD_VERSION_V1
        );
        assert_eq!(
            sha256_digest(&first.host_claim_payload),
            first.host_claim_digest
        );
        assert_eq!(
            sha256_digest(&first.instruction_material_transcript),
            first.instruction_material_digest
        );
        let first_host: MapleDeviceClaimV1 =
            serde_json::from_slice(&first.host_claim_payload).unwrap();
        let first_instruction: MapleResetClearRequiredV1 =
            serde_json::from_slice(&first.instruction_payload).unwrap();
        assert_eq!(first_host, first_instruction.host);
        assert_eq!(first_instruction.admission_count, 1);
        assert!(first_instruction.issuer_key_id.is_empty());
        assert!(first_instruction.issuer_signature.is_empty());
        assert_eq!(
            decode_fixed_base64::<32>(&first_instruction.instruction_material_digest).unwrap(),
            first.instruction_material_digest
        );
        assert_eq!(
            decode_fixed_base64::<32>(&first_instruction.chain_digest).unwrap(),
            first.chain_digest
        );

        let second_target_stream = Uuid::from_u128(610);
        let second_context = MapleResetClearUnsignedMaterializationContext {
            account_id,
            subject_project_id,
            internal_project_id,
            source: MapleResetClearSource::RetainedHostClaim {
                prior_event_id: first_event_id,
                payload_version: first.host_claim_payload_version,
                payload: first.host_claim_payload.clone(),
                payload_digest: first.host_claim_digest.to_vec(),
                identity_mac: first.host_identity_mac.clone(),
                prior_target_revocation_stream_id: first_target_stream,
                prior_target_revocation_stream_generation: 6,
                prior_target_security_epoch: 2,
            },
            event_id: Uuid::from_u128(611),
            reset_id: Uuid::from_u128(612),
            reset_generation: 2,
            cumulative_reset_count: 2,
            source_security_epoch: 2,
            security_epoch: 3,
            source_revocation_stream_id: first_target_stream,
            source_revocation_stream_generation: 6,
            source_last_issued_revocation_sequence: 1,
            revocation_stream_id: second_target_stream,
            revocation_stream_generation: 7,
            issuer_sequence: 1,
            previous_event_id: Some(first_event_id),
            previous_instruction_material_digest: Some(first.instruction_material_digest),
            previous_chain_digest: Some(first.chain_digest),
            admission_leaves: Vec::new(),
            reset_at: reset_at + chrono::Duration::milliseconds(1),
        };
        let second = build_reset_clear_material(&root_key, second_context.clone()).unwrap();
        let second_instruction: MapleResetClearRequiredV1 =
            serde_json::from_slice(&second.instruction_payload).unwrap();
        assert_eq!(second_instruction.host, first_host);
        assert_eq!(
            second_instruction.previous_reset_clear_event_id,
            Some(first_event_id)
        );
        assert_eq!(second_instruction.reset_generation, 2);
        assert_eq!(second_instruction.security_epoch, 3);

        let mut tampered = second_context;
        match &mut tampered.source {
            MapleResetClearSource::RetainedHostClaim { payload_digest, .. } => {
                payload_digest[0] ^= 1
            }
            MapleResetClearSource::LiveDevice { .. } => unreachable!(),
        }
        assert!(build_reset_clear_material(&root_key, tampered).is_err());
    }

    #[test]
    fn cursor_is_canonical_and_rejects_malformed_input() {
        let root_key = [43u8; 32];
        let account = Uuid::from_u128(1234);
        let position = MapleDeviceListCursor {
            registration_id: Uuid::from_u128(77),
        };
        let cursor = encode_cursor(&root_key, account, 9, position).unwrap();
        assert_eq!(
            decode_cursor(&root_key, account, 9, &cursor).unwrap(),
            position
        );
        assert!(decode_cursor(&root_key, Uuid::from_u128(999), 9, &cursor).is_err());
        assert!(decode_cursor(&root_key, account, 10, &cursor).is_err());
        let mut tampered = URL_SAFE_NO_PAD.decode(&cursor).unwrap();
        tampered[0] ^= 1;
        assert!(decode_cursor(&root_key, account, 9, &URL_SAFE_NO_PAD.encode(tampered)).is_err());
        assert!(decode_cursor(&root_key, account, 9, "not a cursor").is_err());
        assert!(decode_cursor(&root_key, account, 9, &URL_SAFE_NO_PAD.encode([0u8; 47])).is_err());
    }
}

#[cfg(test)]
#[path = "maple_pairing_vectors.rs"]
mod maple_pairing_vectors;
