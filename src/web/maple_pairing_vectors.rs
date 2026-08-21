//! Deterministic, backend-owned Maple pairing wire-vector generation.
//!
//! This module is a child of `web::maple_devices` so it can exercise the
//! registration transcript without widening any production visibility. The
//! normal test is read-only and compares exact bytes. Updating the checked-in
//! backend fixture is deliberately ignored and requires an explicit opt-in;
//! the SDK fixture remains a separate reviewed byte-copy.

use super::{
    registration_transcript, ListMapleDevicesResponse, MapleIrohEndpointAddr,
    RegisterMapleDeviceRequest, RegisterMapleDeviceResponse,
};
use crate::models::maple_pairings::*;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{TimeZone, Utc};
use ed25519_dalek::{Signer, SigningKey};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

const FIXTURE_SCHEMA_VERSION: u16 = 1;
const PINNED_FIXTURE_PREIMAGE_SHA256: &str =
    "78c7c22e6d0b86a5c6aa0796a6a516f6ebd8de92b995c173becf071f10dfe9d6";
const FIXTURE_UPDATE_TEMP_FILE_NAME: &str = ".maple_pairing_v1_vectors.json.update.tmp";
const CREATED_AT_UNIX_MS: i64 = 1_786_579_200_000;
const EXPIRES_AT_UNIX_MS: i64 = CREATED_AT_UNIX_MS + 600_000;
const APPROVED_AT_UNIX_MS: i64 = CREATED_AT_UNIX_MS + 60_000;
const ACTIVATED_AT_UNIX_MS: i64 = CREATED_AT_UNIX_MS + 120_000;
const REVOKED_AT_UNIX_MS: i64 = CREATED_AT_UNIX_MS + 300_000;
const ISSUER_KEY_ID: &str = "maple-test-issuer-2026-08-13";
const NEXT_ISSUER_KEY_ID: &str = "maple-test-issuer-2026-08-14";

const CONTROLLER_SEED: [u8; 32] = [0x11; 32];
const HOST_SEED: [u8; 32] = [0x17; 32];
const SECOND_HOST_SEED: [u8; 32] = [0x37; 32];
const FRESH_INSTALLATION_SEED: [u8; 32] = [0x29; 32];
const ISSUER_SEED: [u8; 32] = [0x5b; 32];
const NEXT_ISSUER_SEED: [u8; 32] = [0x6d; 32];
const REMAPPED_ISSUER_SEED: [u8; 32] = [0x7f; 32];

#[derive(Serialize)]
struct FixtureVectors {
    description: &'static str,
    fixture_schema_version: u16,
    issuer_keyset: MaplePairingIssuerKeySetV1,
    test_private_seeds_hex: TestPrivateSeeds,

    create_request: CreateMaplePairingRequest,
    create_request_transcript_hex: String,
    create_request_digest: String,
    request_ticket: MaplePairRequestTicketV1,
    request_ticket_transcript_hex: String,
    request_ticket_digest: String,
    approval_request: ApproveMaplePairingRequest,
    approval_request_transcript_hex: String,
    approval_request_digest: String,
    pair_authorization: MaplePairAuthorizationV1,
    pair_authorization_transcript_hex: String,
    pair_authorization_digest: String,
    confirm_request: ConfirmMaplePairingRequest,
    confirm_request_transcript_hex: String,
    confirm_request_digest: String,
    active_receipt: MaplePairingMutationResponse,
    typed_materialization_vectors: TypedMaterializationVectors,

    list_pairings_request: ListMaplePairingsRequest,
    list_pairings_request_transcript_hex: String,
    list_pairings_request_digest: String,
    pairing_status_request: MaplePairingStatusRequest,
    pairing_status_request_transcript_hex: String,
    pairing_status_request_digest: String,
    revoke_request: RevokeMaplePairingRequest,
    revoke_request_transcript_hex: String,
    revoke_request_digest: String,
    list_revocations_request: ListMaplePairingRevocationsRequest,
    list_revocations_request_transcript_hex: String,
    list_revocations_request_digest: String,
    ack_revocation_request: AckMaplePairingRevocationRequest,
    ack_revocation_request_transcript_hex: String,
    ack_revocation_request_digest: String,
    pair_revocation: MaplePairRevocationV1,
    pair_revocation_transcript_hex: String,
    pair_revocation_digest: String,

    list_pairings_response: ListMaplePairingsResponse,
    pairing_status_response: MaplePairingStatusResponse,
    list_revocations_response: ListMaplePairingRevocationsResponse,
    revoke_receipt: MaplePairingMutationResponse,
    ack_revocation_response: AckMaplePairingRevocationResponse,
    revocation_stream_checkpoint: MapleRevocationStreamCheckpointV1,
    revocation_stream_checkpoint_transcript_hex: String,
    revocation_stream_checkpoint_digest: String,
    ack_revocation_stream_checkpoint: MapleRevocationStreamCheckpointV1,
    ack_revocation_stream_checkpoint_transcript_hex: String,
    ack_revocation_stream_checkpoint_digest: String,

    discovery_list_revocations_request: ListMaplePairingRevocationsRequest,
    discovery_list_revocations_request_transcript_hex: String,
    discovery_list_revocations_request_digest: String,
    discovery_revocation_stream_checkpoint: MapleRevocationStreamCheckpointV1,
    discovery_revocation_stream_checkpoint_transcript_hex: String,
    discovery_revocation_stream_checkpoint_digest: String,
    discovery_list_revocations_response: ListMaplePairingRevocationsResponse,

    reset_clear_admission_set_vectors: ResetClearAdmissionSetVectors,
    reset_clear_three_reset_chain: Vec<ResetClearChainVector>,
    reset_clear_successor_vectors: ResetClearSuccessorVectors,
    two_host_ack_namespace_vectors: TwoHostAckNamespaceVectors,
    issuer_rotation_vectors: IssuerRotationVectors,
    reset_clear_pending_checkpoint: MapleRevocationStreamCheckpointV1,
    reset_clear_pending_checkpoint_transcript_hex: String,
    reset_clear_pending_checkpoint_digest: String,
    reset_clear_pending_sync: MapleRevocationSyncV1,
    reset_clear_list_revocations_request: ListMaplePairingRevocationsRequest,
    reset_clear_list_revocations_request_transcript_hex: String,
    reset_clear_list_revocations_request_digest: String,
    reset_clear_list_revocations_response: ListMaplePairingRevocationsResponse,
    reset_clear_ack_request: AckMaplePairingRevocationRequest,
    reset_clear_ack_request_transcript_hex: String,
    reset_clear_ack_request_digest: String,
    reset_clear_acked_checkpoint: MapleRevocationStreamCheckpointV1,
    reset_clear_acked_checkpoint_transcript_hex: String,
    reset_clear_acked_checkpoint_digest: String,
    reset_clear_ack_response: AckMaplePairingRevocationResponse,
    reset_clear_historical_acked_response: ListMaplePairingRevocationsResponse,
    reset_clear_later_pair_revocation: MaplePairRevocationV1,
    reset_clear_later_pair_revocation_transcript_hex: String,
    reset_clear_later_pair_revocation_digest: String,
    reset_clear_later_checkpoint: MapleRevocationStreamCheckpointV1,
    reset_clear_later_checkpoint_transcript_hex: String,
    reset_clear_later_checkpoint_digest: String,
    reset_clear_historical_later_request: ListMaplePairingRevocationsRequest,
    reset_clear_historical_later_request_transcript_hex: String,
    reset_clear_historical_later_request_digest: String,
    reset_clear_historical_later_response: ListMaplePairingRevocationsResponse,

    register_device_request_epoch_1: RegisterMapleDeviceRequest,
    register_device_request_epoch_1_transcript_hex: String,
    register_device_request_epoch_1_digest: String,
    register_device_request_epoch_4: RegisterMapleDeviceRequest,
    register_device_request_epoch_4_transcript_hex: String,
    register_device_request_epoch_4_digest: String,
    register_device_response_ready: RegisterMapleDeviceResponse,
    register_device_response_revocations_pending: RegisterMapleDeviceResponse,
    register_device_response_reset_clear_required: RegisterMapleDeviceResponse,
    list_devices_response_security_epoch: ListMapleDevicesResponse,
    post_ack_registration_outcome_vectors: PostAckRegistrationOutcomeVectors,

    maple_security_epoch_stale_error: PublicErrorVector,
    maple_pairing_reset_clear_required_error: PublicErrorVector,
    maple_installation_retired_error: PublicErrorVector,
    security_epoch_outcome_vectors: Vec<SecurityEpochOutcomeVector>,
    registration_operation_tombstone_vectors: RegistrationOperationTombstoneVectors,
    wire_structural_assertions: WireStructuralAssertions,
}

#[derive(Serialize)]
struct TestPrivateSeeds {
    controller: String,
    host: String,
    second_host: String,
    fresh_installation: String,
    issuer: String,
    issuer_next: String,
    issuer_remap: String,
}

#[derive(Serialize)]
struct TypedMaterializationVectors {
    create: TypedCreateMaterial,
    revoke: TypedRevokeMaterial,
}

#[derive(Serialize)]
struct TypedCreateMaterial {
    request: CreateMaplePairingRequest,
    request_ticket: MaplePairRequestTicketV1,
    response: MaplePairingMutationResponse,
}

#[derive(Serialize)]
struct TypedRevokeMaterial {
    request: RevokeMaplePairingRequest,
    request_ticket: MaplePairRequestTicketV1,
    pair_authorization: MaplePairAuthorizationV1,
    revocation: MaplePairRevocationV1,
    response: MaplePairingMutationResponse,
}

#[derive(Serialize)]
struct ResetClearAdmissionSetVectors {
    empty: ResetClearAdmissionSetVector,
    max_128: ResetClearAdmissionSetVector,
}

#[derive(Serialize)]
struct ResetClearAdmissionSetVector {
    artifact_version: u16,
    leaves: Vec<ResetClearAdmissionLeafVector>,
    transcript_hex: String,
    digest: String,
}

#[derive(Clone, Serialize)]
struct ResetClearAdmissionLeafVector {
    pair_id: Uuid,
    pairing_incarnation: u64,
    pair_authorization_digest: String,
}

#[derive(Serialize)]
struct ResetClearChainVector {
    instruction: MapleResetClearRequiredV1,
    instruction_material_transcript_hex: String,
    instruction_material_digest: String,
    chain_transcript_hex: String,
    chain_digest: String,
    signed_transcript_hex: String,
    event_digest: String,
}

#[derive(Serialize)]
struct ResetClearSuccessorVectors {
    exact_full_host_claim_fields: [&'static str; 7],
    predecessor: MapleResetClearRequiredV1,
    accepted: SuccessorOutcome,
    changed_host: SuccessorOutcome,
}

#[derive(Serialize)]
struct SuccessorOutcome {
    instruction: MapleResetClearRequiredV1,
    checkpoint: MapleRevocationStreamCheckpointV1,
    expected: &'static str,
}

#[derive(Serialize)]
struct TwoHostAckNamespaceVectors {
    namespace_fields: [&'static str; 4],
    shared_operation_id: Uuid,
    host_a: HostAckVector,
    host_b: HostAckVector,
    same_host_replay: &'static str,
    cross_host_request_response_binding: &'static str,
}

#[derive(Serialize)]
struct HostAckVector {
    request: AckMaplePairingRevocationRequest,
    request_transcript_hex: String,
    request_digest: String,
    response: AckMaplePairingRevocationResponse,
}

#[derive(Serialize)]
struct IssuerRotationVectors {
    artifact_signed_by_initial: MaplePairRequestTicketV1,
    initial: IssuerKeySetOutcome,
    rotated_retaining_previous: IssuerKeySetOutcome,
    rotated_without_previous: IssuerKeySetOutcome,
    remapped_previous_key_id: IssuerKeySetOutcome,
    retained_registry_rule: &'static str,
}

#[derive(Serialize)]
struct IssuerKeySetOutcome {
    keyset: MaplePairingIssuerKeySetV1,
    wire_verification_expected: &'static str,
    registry_reconciliation_expected: &'static str,
}

#[derive(Serialize)]
struct RegistrationOutcomeVector {
    request: RegisterMapleDeviceRequest,
    response: Option<RegisterMapleDeviceResponse>,
    expected: &'static str,
}

#[derive(Serialize)]
struct PostAckRegistrationOutcomeVectors {
    exact_pre_reset_replay: RegistrationOutcomeVector,
    changed_pre_reset_same_operation: RegistrationOutcomeVector,
    exact_pending_reset_replay: RegistrationOutcomeVector,
    changed_pending_reset_same_operation: RegistrationOutcomeVector,
    fresh_operation_retired_installation: RegistrationOutcomeVector,
    fresh_installation_current_epoch: RegistrationOutcomeVector,
    operation_replay_precedes_retirement_and_epoch_gates: bool,
}

#[derive(Serialize)]
struct PublicErrorVector {
    status: u16,
    message: &'static str,
    code: &'static str,
}

#[derive(Serialize)]
struct SecurityEpochOutcomeVector {
    name: &'static str,
    known_security_epoch: u64,
    current_security_epoch: u64,
    pending_reset_clear: bool,
    registration_operation_accepted: bool,
    outcome: &'static str,
}

#[derive(Serialize)]
struct RegistrationOperationTombstoneVectors {
    storage: RegistrationTombstoneStorage,
    exact_old_request: &'static str,
    changed_request_same_operation_lookup: &'static str,
    exact_pending_reset_request_after_ack: &'static str,
    changed_pending_reset_request_after_ack: &'static str,
    fresh_operation_retired_installation: &'static str,
    fresh_installation_current_epoch: &'static str,
}

#[derive(Serialize)]
struct RegistrationTombstoneStorage {
    operation_lookup_digest: &'static str,
    raw_operation_id_stored: bool,
    request_binding: &'static str,
    frozen_response_retained: bool,
}

#[derive(Serialize)]
struct WireStructuralAssertions {
    list_devices_security_epoch_source: &'static str,
    list_devices_response_signed: bool,
    reset_event_json_key_order: [&'static str; 2],
    pending_reset_checkpoint: PendingResetCheckpointShape,
    historical_reset_event_allowed_after_ack: bool,
    historical_reset_event_allowed_when_later_events_exist: bool,
    reset_admission_public_shape: [&'static str; 2],
    retained_admission_leaves_cross_wire: bool,
    fixture_pretty_indent_spaces: u8,
    fixture_trailing_newline_count: u8,
    sdk_fixture_copy: &'static str,
}

#[derive(Serialize)]
struct PendingResetCheckpointShape {
    last_issued_issuer_sequence: u64,
    last_acked_issuer_sequence: u64,
}

fn uuid(value: &str) -> Uuid {
    Uuid::parse_str(value).expect("fixed fixture UUID")
}

fn digest_base64(transcript: &[u8]) -> String {
    STANDARD.encode(sha256_digest(transcript))
}

fn transcript_pair(transcript: &[u8]) -> (String, String) {
    (hex::encode(transcript), digest_base64(transcript))
}

fn empty_signature() -> String {
    STANDARD.encode([0u8; 64])
}

fn sign_transcript(signing_key: &SigningKey, transcript: &[u8]) -> String {
    STANDARD.encode(signing_key.sign(transcript).to_bytes())
}

fn endpoint_id(signing_key: &SigningKey) -> String {
    hex::encode(signing_key.verifying_key().as_bytes())
}

fn identity_public_key(signing_key: &SigningKey) -> String {
    STANDARD.encode(signing_key.verifying_key().as_bytes())
}

fn make_claim(
    registration_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    signing_key: &SigningKey,
    endpoint_epoch: u64,
) -> MapleDeviceClaimV1 {
    MapleDeviceClaimV1 {
        registration_id,
        device_id,
        installation_id,
        identity_algorithm: MaplePairingIdentityAlgorithm::Ed25519,
        identity_public_key: identity_public_key(signing_key),
        endpoint_id: endpoint_id(signing_key),
        endpoint_epoch,
    }
}

fn sign_create_request(
    mut request: CreateMaplePairingRequest,
    signing_key: &SigningKey,
) -> CreateMaplePairingRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_approval_request(
    mut request: ApproveMaplePairingRequest,
    signing_key: &SigningKey,
) -> ApproveMaplePairingRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_confirm_request(
    mut request: ConfirmMaplePairingRequest,
    signing_key: &SigningKey,
) -> ConfirmMaplePairingRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_revoke_request(
    mut request: RevokeMaplePairingRequest,
    signing_key: &SigningKey,
) -> RevokeMaplePairingRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_list_pairings_request(
    mut request: ListMaplePairingsRequest,
    signing_key: &SigningKey,
) -> ListMaplePairingsRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_status_request(
    mut request: MaplePairingStatusRequest,
    signing_key: &SigningKey,
) -> MaplePairingStatusRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_list_revocations_request(
    mut request: ListMaplePairingRevocationsRequest,
    signing_key: &SigningKey,
) -> ListMaplePairingRevocationsRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_ack_request(
    mut request: AckMaplePairingRevocationRequest,
    signing_key: &SigningKey,
) -> AckMaplePairingRevocationRequest {
    request.signature = empty_signature();
    request.signature = sign_transcript(signing_key, &request.transcript().unwrap());
    request
}

fn sign_registration_request(
    mut request: RegisterMapleDeviceRequest,
    signing_key: &SigningKey,
) -> RegisterMapleDeviceRequest {
    request.signature = empty_signature();
    let mut capabilities = request.capabilities.clone();
    capabilities.sort_unstable();
    let transcript = registration_transcript(
        &request,
        signing_key.verifying_key().as_bytes(),
        &request.iroh_endpoint_addr,
        &capabilities,
    );
    request.signature = sign_transcript(signing_key, &transcript);
    request
}

fn registration_transcript_pair(
    request: &RegisterMapleDeviceRequest,
    signing_key: &SigningKey,
) -> (String, String) {
    let mut capabilities = request.capabilities.clone();
    capabilities.sort_unstable();
    let transcript = registration_transcript(
        request,
        signing_key.verifying_key().as_bytes(),
        &request.iroh_endpoint_addr,
        &capabilities,
    );
    transcript_pair(&transcript)
}

fn pairing_status(
    state: MaplePairingState,
    revision: i64,
    ticket: &MaplePairRequestTicketV1,
    authorization: Option<&MaplePairAuthorizationV1>,
    revocation: Option<&MaplePairRevocationV1>,
) -> MaplePairingStatusV1 {
    let has_stream = !matches!(
        state,
        MaplePairingState::Pending | MaplePairingState::Expired
    );
    MaplePairingStatusV1 {
        pairing_request_id: ticket.pairing_request_id,
        pair_id: ticket.pair_id,
        state,
        revision,
        pairing_incarnation: ticket.pairing_incarnation,
        revocation_stream_id: has_stream.then_some(uuid("15151515-1515-4515-8515-151515151515")),
        revocation_stream_generation: has_stream.then_some(5),
        direction: ticket.direction,
        execution_target_id: ticket.execution_target_id,
        controller_registration_id: ticket.controller.registration_id,
        host_registration_id: ticket.host.registration_id,
        created_at_unix_ms: ticket.created_at_unix_ms,
        expires_at_unix_ms: ticket.expires_at_unix_ms,
        approved_at_unix_ms: authorization.map(|value| value.approved_at_unix_ms),
        activated_at_unix_ms: matches!(
            state,
            MaplePairingState::Active | MaplePairingState::Revoked
        )
        .then_some(ACTIVATED_AT_UNIX_MS),
        revoked_at_unix_ms: revocation.map(|value| value.revoked_at_unix_ms),
        request_ticket: Some(ticket.clone()),
        pair_authorization: authorization.cloned(),
        revocation: revocation.cloned(),
    }
}

fn signed_checkpoint(
    issuer: &dyn MaplePairingIssuer,
    host: MapleDeviceClaimV1,
    security_epoch: u64,
    stream_id: Uuid,
    stream_generation: u64,
    last_issued: u64,
    last_acked: u64,
) -> MapleRevocationStreamCheckpointV1 {
    sign_revocation_stream_checkpoint(
        MapleRevocationStreamCheckpointV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: uuid("11111111-1111-4111-8111-111111111111"),
            subject_project_id: uuid("22222222-2222-4222-8222-222222222222"),
            host,
            security_epoch,
            revocation_stream_id: stream_id,
            revocation_stream_generation: stream_generation,
            last_issued_issuer_sequence: last_issued,
            last_acked_issuer_sequence: last_acked,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        issuer,
    )
    .unwrap()
}

fn admission_leaves(count: usize) -> Vec<MapleResetClearAdmissionLeafV1> {
    (0..count)
        .map(|index| MapleResetClearAdmissionLeafV1 {
            pair_id: Uuid::from_u128(0xa100_0000_0000_4000_8000_0000_0000_0000 + index as u128),
            pairing_incarnation: index as u64 + 1,
            pair_authorization_digest: sha256_digest(format!("admission-{index:03}").as_bytes()),
        })
        .collect()
}

fn admission_vector(leaves: &[MapleResetClearAdmissionLeafV1]) -> ResetClearAdmissionSetVector {
    let transcript =
        reset_clear_admission_set_transcript(MAPLE_PAIRING_ARTIFACT_VERSION_V1, leaves).unwrap();
    ResetClearAdmissionSetVector {
        artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        leaves: leaves
            .iter()
            .map(|leaf| ResetClearAdmissionLeafVector {
                pair_id: leaf.pair_id,
                pairing_incarnation: leaf.pairing_incarnation,
                pair_authorization_digest: STANDARD.encode(leaf.pair_authorization_digest),
            })
            .collect(),
        transcript_hex: hex::encode(&transcript),
        digest: digest_base64(&transcript),
    }
}

#[allow(clippy::too_many_arguments)]
fn signed_reset_instruction(
    issuer: &dyn MaplePairingIssuer,
    host: MapleDeviceClaimV1,
    event_id: Uuid,
    reset_id: Uuid,
    reset_generation: u64,
    source_security_epoch: u64,
    source_stream_id: Uuid,
    source_stream_generation: u64,
    target_stream_id: Uuid,
    leaves: &[MapleResetClearAdmissionLeafV1],
    predecessor: Option<&MapleResetClearRequiredV1>,
    reset_at_unix_ms: i64,
) -> MapleResetClearRequiredV1 {
    sign_reset_clear_required(
        MapleResetClearRequiredV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            event_id,
            reset_id,
            reset_generation,
            cumulative_reset_count: reset_generation,
            source_security_epoch,
            security_epoch: source_security_epoch + 1,
            subject_account_id: uuid("11111111-1111-4111-8111-111111111111"),
            subject_project_id: uuid("22222222-2222-4222-8222-222222222222"),
            recipient_host_registration_id: host.registration_id,
            host,
            issuer_sequence: 1,
            source_revocation_stream_id: source_stream_id,
            source_revocation_stream_generation: source_stream_generation,
            revocation_stream_id: target_stream_id,
            revocation_stream_generation: source_stream_generation + 1,
            clear_scope:
                MapleResetClearScopeV1::AllPairAuthorizationsForAccountProjectHostInstallation,
            admission_count: u16::try_from(leaves.len()).unwrap(),
            admission_set_digest: STANDARD.encode(
                reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, leaves)
                    .unwrap(),
            ),
            previous_reset_clear_event_id: predecessor.map(|value| value.event_id),
            previous_instruction_material_digest: predecessor
                .map(|value| value.instruction_material_digest.clone()),
            previous_chain_digest: predecessor.map(|value| value.chain_digest.clone()),
            reset_at_unix_ms,
            instruction_material_digest: String::new(),
            chain_digest: String::new(),
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        issuer,
    )
    .unwrap()
}

fn reset_chain_vector(instruction: MapleResetClearRequiredV1) -> ResetClearChainVector {
    let material = reset_clear_instruction_material_transcript(&instruction).unwrap();
    let chain = reset_clear_chain_transcript(&instruction).unwrap();
    let signed = instruction.transcript().unwrap();
    ResetClearChainVector {
        instruction_material_transcript_hex: hex::encode(&material),
        instruction_material_digest: STANDARD.encode(sha256_digest(&material)),
        chain_transcript_hex: hex::encode(&chain),
        chain_digest: STANDARD.encode(sha256_digest(&chain)),
        signed_transcript_hex: hex::encode(&signed),
        event_digest: STANDARD.encode(sha256_digest(&signed)),
        instruction,
    }
}

struct RegistrationRequestHostFixture<'a> {
    device_id: Uuid,
    installation_id: Uuid,
    endpoint_epoch: u64,
    signing_key: &'a SigningKey,
    display_name: &'a str,
}

fn registration_request(
    operation_id: Uuid,
    expected_revision: Option<i64>,
    known_security_epoch: u64,
    host: RegistrationRequestHostFixture<'_>,
) -> RegisterMapleDeviceRequest {
    sign_registration_request(
        RegisterMapleDeviceRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id,
            device_id: host.device_id,
            installation_id: host.installation_id,
            expected_revision,
            known_security_epoch,
            asserted_account_id: uuid("11111111-1111-4111-8111-111111111111"),
            asserted_project_id: uuid("22222222-2222-4222-8222-222222222222"),
            identity_algorithm: "ed25519".to_string(),
            identity_public_key: identity_public_key(host.signing_key),
            iroh_endpoint_id: endpoint_id(host.signing_key),
            endpoint_epoch: host.endpoint_epoch,
            iroh_endpoint_addr: MapleIrohEndpointAddr {
                relay_urls: vec!["https://relay.example.test/".to_string()],
                direct_addresses: vec!["127.0.0.1:7777".to_string()],
            },
            platform: "macos".to_string(),
            display_name: host.display_name.to_string(),
            capabilities: vec!["pairing".to_string(), "agent.remote".to_string()],
            signature: String::new(),
        },
        host.signing_key,
    )
}

fn registration_response(
    request: &RegisterMapleDeviceRequest,
    registration_id: Uuid,
    revision: i64,
    accepted_at_unix_ms: i64,
    revocation_sync: MapleRevocationSyncV1,
) -> RegisterMapleDeviceResponse {
    RegisterMapleDeviceResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: request.operation_id,
        registration_id,
        device_id: request.device_id,
        revision,
        accepted_at: Utc
            .timestamp_millis_opt(accepted_at_unix_ms)
            .single()
            .unwrap(),
        security_epoch: request.known_security_epoch,
        revocation_sync,
    }
}

fn build_vectors() -> FixtureVectors {
    let controller_key = SigningKey::from_bytes(&CONTROLLER_SEED);
    let host_key = SigningKey::from_bytes(&HOST_SEED);
    let second_host_key = SigningKey::from_bytes(&SECOND_HOST_SEED);
    let fresh_installation_key = SigningKey::from_bytes(&FRESH_INSTALLATION_SEED);
    let issuer = Ed25519MaplePairingIssuer::new(
        ISSUER_KEY_ID.to_string(),
        SigningKey::from_bytes(&ISSUER_SEED),
    )
    .unwrap();
    let next_issuer = Ed25519MaplePairingIssuer::new(
        NEXT_ISSUER_KEY_ID.to_string(),
        SigningKey::from_bytes(&NEXT_ISSUER_SEED),
    )
    .unwrap();
    let remapped_issuer = Ed25519MaplePairingIssuer::new(
        ISSUER_KEY_ID.to_string(),
        SigningKey::from_bytes(&REMAPPED_ISSUER_SEED),
    )
    .unwrap();

    let account_id = uuid("11111111-1111-4111-8111-111111111111");
    let project_id = uuid("22222222-2222-4222-8222-222222222222");
    let controller_registration_id = uuid("44444444-4444-4444-8444-444444444444");
    let controller_device_id = uuid("55555555-5555-4555-8555-555555555555");
    let controller_installation_id = uuid("66666666-6666-4666-8666-666666666666");
    let host_registration_id = uuid("77777777-7777-4777-8777-777777777777");
    let host_device_id = uuid("88888888-8888-4888-8888-888888888888");
    let host_installation_id = uuid("99999999-9999-4999-8999-999999999999");
    let pairing_request_id = uuid("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb");
    let pair_id = uuid("cccccccc-cccc-4ccc-8ccc-cccccccccccc");
    let base_stream_id = uuid("15151515-1515-4515-8515-151515151515");

    let controller_claim = make_claim(
        controller_registration_id,
        controller_device_id,
        controller_installation_id,
        &controller_key,
        7,
    );
    let host_claim_at_pairing = make_claim(
        host_registration_id,
        host_device_id,
        host_installation_id,
        &host_key,
        4,
    );
    let host_claim_current = make_claim(
        host_registration_id,
        host_device_id,
        host_installation_id,
        &host_key,
        5,
    );

    let create_request = sign_create_request(
        CreateMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("33333333-3333-4333-8333-333333333333"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            controller_registration_id,
            controller_device_id,
            controller_installation_id,
            controller_endpoint_id: controller_claim.endpoint_id.clone(),
            controller_endpoint_epoch: controller_claim.endpoint_epoch,
            host_registration_id,
            host_device_id,
            host_installation_id,
            host_endpoint_id: host_claim_at_pairing.endpoint_id.clone(),
            host_endpoint_epoch: host_claim_at_pairing.endpoint_epoch,
            direction: MaplePairingDirection::ControllerToHost,
            execution_target_id: host_registration_id,
            pairing_request_nonce: STANDARD.encode([0x21; 32]),
            protocol_min: 1,
            protocol_max: 1,
            signature: String::new(),
        },
        &controller_key,
    );
    let create_transcript = create_request.transcript().unwrap();
    let (create_request_transcript_hex, create_request_digest) =
        transcript_pair(&create_transcript);

    let request_ticket = sign_pair_request_ticket(
        MaplePairRequestTicketV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: account_id,
            subject_project_id: project_id,
            pairing_request_id,
            pair_id,
            direction: MaplePairingDirection::ControllerToHost,
            execution_target_id: host_registration_id,
            controller: controller_claim.clone(),
            host: host_claim_at_pairing.clone(),
            pairing_request_nonce: create_request.pairing_request_nonce.clone(),
            controller_request_operation_id: create_request.operation_id,
            controller_request_digest: digest_base64(&create_transcript),
            controller_request_signature: create_request.signature.clone(),
            pairing_incarnation: 3,
            protocol_min: 1,
            protocol_max: 1,
            created_at_unix_ms: CREATED_AT_UNIX_MS,
            expires_at_unix_ms: EXPIRES_AT_UNIX_MS,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        &issuer,
    )
    .unwrap();
    let request_ticket_transcript = request_ticket.transcript().unwrap();
    let (request_ticket_transcript_hex, request_ticket_digest) =
        transcript_pair(&request_ticket_transcript);

    let approval_request = sign_approval_request(
        ApproveMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            pairing_request_id,
            pair_id,
            expected_pairing_revision: 1,
            pairing_incarnation: 3,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            request_ticket_digest: digest_base64(&request_ticket_transcript),
            host_approval_nonce: STANDARD.encode([0x2c; 32]),
            approved_protocol_min: 1,
            approved_protocol_max: 1,
            signature: String::new(),
        },
        &host_key,
    );
    let approval_transcript = approval_request.transcript().unwrap();
    let (approval_request_transcript_hex, approval_request_digest) =
        transcript_pair(&approval_transcript);

    let pair_authorization = sign_pair_authorization(
        MaplePairAuthorizationV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: account_id,
            subject_project_id: project_id,
            pairing_request_id,
            pair_id,
            direction: MaplePairingDirection::ControllerToHost,
            execution_target_id: host_registration_id,
            controller: controller_claim.clone(),
            host: host_claim_at_pairing.clone(),
            pairing_request_nonce: create_request.pairing_request_nonce.clone(),
            controller_request_operation_id: create_request.operation_id,
            controller_request_digest: digest_base64(&create_transcript),
            controller_request_signature: create_request.signature.clone(),
            request_ticket_digest: digest_base64(&request_ticket_transcript),
            host_approval_operation_id: approval_request.operation_id,
            host_approval_expected_pairing_revision: approval_request.expected_pairing_revision,
            host_approval_nonce: approval_request.host_approval_nonce.clone(),
            host_approval_digest: digest_base64(&approval_transcript),
            host_approval_signature: approval_request.signature.clone(),
            pairing_incarnation: 3,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            protocol_min: 1,
            protocol_max: 1,
            approved_at_unix_ms: APPROVED_AT_UNIX_MS,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        &issuer,
    )
    .unwrap();
    let pair_authorization_transcript = pair_authorization.transcript().unwrap();
    let (pair_authorization_transcript_hex, pair_authorization_digest) =
        transcript_pair(&pair_authorization_transcript);

    let confirm_request = sign_confirm_request(
        ConfirmMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            pairing_request_id,
            pair_id,
            expected_pairing_revision: 2,
            pairing_incarnation: 3,
            pair_authorization_digest: digest_base64(&pair_authorization_transcript),
            signature: String::new(),
        },
        &host_key,
    );
    let confirm_transcript = confirm_request.transcript().unwrap();
    let (confirm_request_transcript_hex, confirm_request_digest) =
        transcript_pair(&confirm_transcript);

    let revoke_request = sign_revoke_request(
        RevokeMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("abababab-abab-4bab-8bab-abababababab"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            actor_registration_id: controller_registration_id,
            actor_role: MaplePairingRole::Controller,
            pairing_request_id,
            pair_id,
            expected_pairing_revision: 3,
            pairing_incarnation: 3,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            reason_code: "user_removed_access".to_string(),
            signature: String::new(),
        },
        &controller_key,
    );
    let revoke_transcript = revoke_request.transcript().unwrap();
    let (revoke_request_transcript_hex, revoke_request_digest) =
        transcript_pair(&revoke_transcript);

    let pair_revocation = sign_pair_revocation(
        MaplePairRevocationV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            event_id: uuid("ffffffff-ffff-4fff-8fff-ffffffffffff"),
            subject_account_id: account_id,
            subject_project_id: project_id,
            recipient_host_registration_id: host_registration_id,
            issuer_sequence: 13,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            pairing_request_id,
            pair_id,
            direction: MaplePairingDirection::ControllerToHost,
            execution_target_id: host_registration_id,
            controller: controller_claim.clone(),
            host: host_claim_at_pairing.clone(),
            pairing_incarnation: 3,
            pair_authorization_digest: digest_base64(&pair_authorization_transcript),
            revoked_by_registration_id: controller_registration_id,
            revoked_by_role: MaplePairingRole::Controller,
            reason_code: revoke_request.reason_code.clone(),
            revoked_at_unix_ms: REVOKED_AT_UNIX_MS,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        &issuer,
    )
    .unwrap();
    let pair_revocation_transcript = pair_revocation.transcript().unwrap();
    let (pair_revocation_transcript_hex, pair_revocation_digest) =
        transcript_pair(&pair_revocation_transcript);

    let pending_status = pairing_status(MaplePairingState::Pending, 1, &request_ticket, None, None);
    let active_status = pairing_status(
        MaplePairingState::Active,
        3,
        &request_ticket,
        Some(&pair_authorization),
        None,
    );
    let revoked_status = pairing_status(
        MaplePairingState::Revoked,
        4,
        &request_ticket,
        Some(&pair_authorization),
        Some(&pair_revocation),
    );
    let pending_receipt = MaplePairingMutationResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: create_request.operation_id,
        pairing: pending_status.clone(),
    };
    let active_receipt = MaplePairingMutationResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: confirm_request.operation_id,
        pairing: active_status.clone(),
    };
    let revoke_receipt = MaplePairingMutationResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: revoke_request.operation_id,
        pairing: revoked_status,
    };

    let typed_materialization_vectors = TypedMaterializationVectors {
        create: TypedCreateMaterial {
            request: create_request.clone(),
            request_ticket: request_ticket.clone(),
            response: pending_receipt,
        },
        revoke: TypedRevokeMaterial {
            request: revoke_request.clone(),
            request_ticket: request_ticket.clone(),
            pair_authorization: pair_authorization.clone(),
            revocation: pair_revocation.clone(),
            response: revoke_receipt.clone(),
        },
    };

    let list_pairings_request = sign_list_pairings_request(
        ListMaplePairingsRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            query_id: uuid("10101010-1010-4010-8010-101010101010"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            actor_registration_id: controller_registration_id,
            role: MaplePairingRole::Controller,
            states: vec![
                MaplePairingState::Pending,
                MaplePairingState::AwaitingHostCommit,
                MaplePairingState::Active,
                MaplePairingState::Expired,
                MaplePairingState::Revoked,
            ],
            cursor: Some("dGVzdC1vcGFxdWUtcGFpcmVkLWN1cnNvcg".to_string()),
            limit: Some(2),
            signature: String::new(),
        },
        &controller_key,
    );
    let list_pairings_transcript = list_pairings_request.transcript().unwrap();
    let (list_pairings_request_transcript_hex, list_pairings_request_digest) =
        transcript_pair(&list_pairings_transcript);
    let list_pairings_response = ListMaplePairingsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: list_pairings_request.query_id,
        role: list_pairings_request.role,
        pairings: vec![pending_status],
        next_cursor: Some("dGVzdC1uZXh0LWN1cnNvcg".to_string()),
        has_more: true,
    };

    let pairing_status_request = sign_status_request(
        MaplePairingStatusRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            query_id: uuid("12121212-1212-4212-8212-121212121212"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            actor_registration_id: controller_registration_id,
            pair_id,
            signature: String::new(),
        },
        &controller_key,
    );
    let status_transcript = pairing_status_request.transcript().unwrap();
    let (pairing_status_request_transcript_hex, pairing_status_request_digest) =
        transcript_pair(&status_transcript);
    let pairing_status_response = MaplePairingStatusResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: pairing_status_request.query_id,
        pairing: active_status,
    };

    let revocation_stream_checkpoint = signed_checkpoint(
        &issuer,
        host_claim_current.clone(),
        1,
        base_stream_id,
        5,
        13,
        12,
    );
    let revocation_checkpoint_transcript = revocation_stream_checkpoint.transcript().unwrap();
    let (revocation_stream_checkpoint_transcript_hex, revocation_stream_checkpoint_digest) =
        transcript_pair(&revocation_checkpoint_transcript);
    let revocations_pending_sync =
        MapleRevocationSyncV1::status_for_checkpoint(1, revocation_stream_checkpoint.clone(), None)
            .unwrap();

    let list_revocations_request = sign_list_revocations_request(
        ListMaplePairingRevocationsRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            query_id: uuid("13131313-1313-4313-8313-131313131313"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            after_issuer_sequence: 12,
            limit: Some(1),
            signature: String::new(),
        },
        &host_key,
    );
    let list_revocations_transcript = list_revocations_request.transcript().unwrap();
    let (list_revocations_request_transcript_hex, list_revocations_request_digest) =
        transcript_pair(&list_revocations_transcript);
    let list_revocations_response = ListMaplePairingRevocationsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: list_revocations_request.query_id,
        revocation_sync: revocations_pending_sync.clone(),
        events: vec![MapleRevocationStreamEventV1::PairRevocation(
            pair_revocation.clone(),
        )],
        next_after_issuer_sequence: 13,
        has_more: false,
    };

    let ack_revocation_request = sign_ack_request(
        AckMaplePairingRevocationRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("14141414-1414-4414-8414-141414141414"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            revocation_stream_id: base_stream_id,
            revocation_stream_generation: 5,
            event_id: pair_revocation.event_id,
            issuer_sequence: 13,
            event_digest: digest_base64(&pair_revocation_transcript),
            expected_previous_issuer_sequence: 12,
            signature: String::new(),
        },
        &host_key,
    );
    let ack_revocation_transcript = ack_revocation_request.transcript().unwrap();
    let (ack_revocation_request_transcript_hex, ack_revocation_request_digest) =
        transcript_pair(&ack_revocation_transcript);
    let ack_revocation_stream_checkpoint = signed_checkpoint(
        &issuer,
        host_claim_current.clone(),
        1,
        base_stream_id,
        5,
        13,
        13,
    );
    let ack_checkpoint_transcript = ack_revocation_stream_checkpoint.transcript().unwrap();
    let (ack_revocation_stream_checkpoint_transcript_hex, ack_revocation_stream_checkpoint_digest) =
        transcript_pair(&ack_checkpoint_transcript);
    let ack_revocation_response = AckMaplePairingRevocationResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: ack_revocation_request.operation_id,
        host_registration_id,
        stream_checkpoint: ack_revocation_stream_checkpoint.clone(),
        event_id: pair_revocation.event_id,
        issuer_sequence: 13,
        last_acked_issuer_sequence: 13,
        accepted_at_unix_ms: REVOKED_AT_UNIX_MS + 1_000,
    };

    let discovery_list_revocations_request = sign_list_revocations_request(
        ListMaplePairingRevocationsRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            query_id: uuid("18181818-1818-4818-8818-181818181818"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            revocation_stream_id: Uuid::nil(),
            revocation_stream_generation: 0,
            after_issuer_sequence: 0,
            limit: Some(1),
            signature: String::new(),
        },
        &host_key,
    );
    let discovery_request_transcript = discovery_list_revocations_request.transcript().unwrap();
    let (
        discovery_list_revocations_request_transcript_hex,
        discovery_list_revocations_request_digest,
    ) = transcript_pair(&discovery_request_transcript);
    let discovery_revocation_stream_checkpoint = signed_checkpoint(
        &issuer,
        host_claim_current.clone(),
        1,
        uuid("16161616-1616-4616-8616-161616161616"),
        6,
        0,
        0,
    );
    let discovery_checkpoint_transcript =
        discovery_revocation_stream_checkpoint.transcript().unwrap();
    let (
        discovery_revocation_stream_checkpoint_transcript_hex,
        discovery_revocation_stream_checkpoint_digest,
    ) = transcript_pair(&discovery_checkpoint_transcript);
    let discovery_ready_sync = MapleRevocationSyncV1::status_for_checkpoint(
        1,
        discovery_revocation_stream_checkpoint.clone(),
        None,
    )
    .unwrap();
    let discovery_list_revocations_response = ListMaplePairingRevocationsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: discovery_list_revocations_request.query_id,
        revocation_sync: discovery_ready_sync.clone(),
        events: Vec::new(),
        next_after_issuer_sequence: 0,
        has_more: false,
    };

    let empty_admissions = admission_leaves(0);
    let one_admission = admission_leaves(1);
    let maximum_admissions = admission_leaves(128);
    let reset_clear_admission_set_vectors = ResetClearAdmissionSetVectors {
        empty: admission_vector(&empty_admissions),
        max_128: admission_vector(&maximum_admissions),
    };

    let first_reset = signed_reset_instruction(
        &issuer,
        host_claim_current.clone(),
        uuid("81000000-0000-4000-8000-000000000001"),
        uuid("91000000-0000-4000-8000-000000000001"),
        1,
        1,
        base_stream_id,
        5,
        uuid("26262626-2626-4626-8626-262626262626"),
        &empty_admissions,
        None,
        CREATED_AT_UNIX_MS + 500_000,
    );
    let second_reset = signed_reset_instruction(
        &issuer,
        host_claim_current.clone(),
        uuid("82000000-0000-4000-8000-000000000002"),
        uuid("92000000-0000-4000-8000-000000000002"),
        2,
        2,
        first_reset.revocation_stream_id,
        first_reset.revocation_stream_generation,
        uuid("27272727-2727-4727-8727-272727272727"),
        &one_admission,
        Some(&first_reset),
        CREATED_AT_UNIX_MS + 510_000,
    );
    let third_reset = signed_reset_instruction(
        &issuer,
        host_claim_current.clone(),
        uuid("83000000-0000-4000-8000-000000000003"),
        uuid("93000000-0000-4000-8000-000000000003"),
        3,
        3,
        second_reset.revocation_stream_id,
        second_reset.revocation_stream_generation,
        uuid("28282828-2828-4828-8828-282828282828"),
        &maximum_admissions,
        Some(&second_reset),
        CREATED_AT_UNIX_MS + 520_000,
    );
    let reset_clear_three_reset_chain = vec![
        reset_chain_vector(first_reset.clone()),
        reset_chain_vector(second_reset.clone()),
        reset_chain_vector(third_reset.clone()),
    ];

    let accepted_successor_checkpoint = signed_checkpoint(
        &issuer,
        second_reset.host.clone(),
        second_reset.security_epoch,
        second_reset.revocation_stream_id,
        second_reset.revocation_stream_generation,
        1,
        0,
    );
    let mut changed_host_successor = second_reset.clone();
    changed_host_successor.host.endpoint_epoch += 1;
    let changed_host_successor =
        sign_reset_clear_required(changed_host_successor, &issuer).unwrap();
    let changed_host_successor_checkpoint = signed_checkpoint(
        &issuer,
        changed_host_successor.host.clone(),
        changed_host_successor.security_epoch,
        changed_host_successor.revocation_stream_id,
        changed_host_successor.revocation_stream_generation,
        1,
        0,
    );
    let reset_clear_successor_vectors = ResetClearSuccessorVectors {
        exact_full_host_claim_fields: [
            "registration_id",
            "device_id",
            "installation_id",
            "identity_algorithm",
            "identity_public_key",
            "endpoint_id",
            "endpoint_epoch",
        ],
        predecessor: first_reset.clone(),
        accepted: SuccessorOutcome {
            instruction: second_reset.clone(),
            checkpoint: accepted_successor_checkpoint,
            expected: "accepted",
        },
        changed_host: SuccessorOutcome {
            instruction: changed_host_successor,
            checkpoint: changed_host_successor_checkpoint,
            expected: "reset_clear_successor_binding_rejected",
        },
    };

    let second_host_claim = make_claim(
        uuid("a7777777-7777-4777-8777-777777777777"),
        uuid("a8888888-8888-4888-8888-888888888888"),
        uuid("a9999999-9999-4999-8999-999999999999"),
        &second_host_key,
        2,
    );
    let shared_ack_operation_id = uuid("45454545-4545-4545-8545-454545454545");
    let shared_reset_id = uuid("a3000000-0000-4000-8000-000000000001");
    let host_a_reset = signed_reset_instruction(
        &issuer,
        host_claim_current.clone(),
        uuid("a1000000-0000-4000-8000-000000000001"),
        shared_reset_id,
        1,
        3,
        uuid("a4000000-0000-4000-8000-000000000001"),
        7,
        uuid("a5000000-0000-4000-8000-000000000001"),
        &empty_admissions,
        None,
        CREATED_AT_UNIX_MS + 530_000,
    );
    let host_b_reset = signed_reset_instruction(
        &issuer,
        second_host_claim.clone(),
        uuid("b1000000-0000-4000-8000-000000000001"),
        shared_reset_id,
        1,
        3,
        uuid("b4000000-0000-4000-8000-000000000001"),
        7,
        uuid("b5000000-0000-4000-8000-000000000001"),
        &empty_admissions,
        None,
        CREATED_AT_UNIX_MS + 530_000,
    );
    let host_a_ack_request = sign_ack_request(
        AckMaplePairingRevocationRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: shared_ack_operation_id,
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id: host_a_reset.host.registration_id,
            revocation_stream_id: host_a_reset.revocation_stream_id,
            revocation_stream_generation: host_a_reset.revocation_stream_generation,
            event_id: host_a_reset.event_id,
            issuer_sequence: 1,
            event_digest: STANDARD.encode(host_a_reset.event_digest().unwrap()),
            expected_previous_issuer_sequence: 0,
            signature: String::new(),
        },
        &host_key,
    );
    let host_b_ack_request = sign_ack_request(
        AckMaplePairingRevocationRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: shared_ack_operation_id,
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id: host_b_reset.host.registration_id,
            revocation_stream_id: host_b_reset.revocation_stream_id,
            revocation_stream_generation: host_b_reset.revocation_stream_generation,
            event_id: host_b_reset.event_id,
            issuer_sequence: 1,
            event_digest: STANDARD.encode(host_b_reset.event_digest().unwrap()),
            expected_previous_issuer_sequence: 0,
            signature: String::new(),
        },
        &second_host_key,
    );
    let host_a_ack_transcript = host_a_ack_request.transcript().unwrap();
    let host_b_ack_transcript = host_b_ack_request.transcript().unwrap();
    let host_a_acked_checkpoint = signed_checkpoint(
        &issuer,
        host_a_reset.host.clone(),
        4,
        host_a_reset.revocation_stream_id,
        host_a_reset.revocation_stream_generation,
        1,
        1,
    );
    let host_b_acked_checkpoint = signed_checkpoint(
        &issuer,
        host_b_reset.host.clone(),
        4,
        host_b_reset.revocation_stream_id,
        host_b_reset.revocation_stream_generation,
        1,
        1,
    );
    let two_host_ack_namespace_vectors = TwoHostAckNamespaceVectors {
        namespace_fields: [
            "asserted_account_id",
            "asserted_project_id",
            "host_registration_id",
            "operation_id",
        ],
        shared_operation_id: shared_ack_operation_id,
        host_a: HostAckVector {
            request: host_a_ack_request.clone(),
            request_transcript_hex: hex::encode(&host_a_ack_transcript),
            request_digest: digest_base64(&host_a_ack_transcript),
            response: AckMaplePairingRevocationResponse {
                protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
                operation_id: shared_ack_operation_id,
                host_registration_id: host_a_ack_request.host_registration_id,
                stream_checkpoint: host_a_acked_checkpoint,
                event_id: host_a_ack_request.event_id,
                issuer_sequence: 1,
                last_acked_issuer_sequence: 1,
                accepted_at_unix_ms: CREATED_AT_UNIX_MS + 531_000,
            },
        },
        host_b: HostAckVector {
            request: host_b_ack_request.clone(),
            request_transcript_hex: hex::encode(&host_b_ack_transcript),
            request_digest: digest_base64(&host_b_ack_transcript),
            response: AckMaplePairingRevocationResponse {
                protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
                operation_id: shared_ack_operation_id,
                host_registration_id: host_b_ack_request.host_registration_id,
                stream_checkpoint: host_b_acked_checkpoint,
                event_id: host_b_ack_request.event_id,
                issuer_sequence: 1,
                last_acked_issuer_sequence: 1,
                accepted_at_unix_ms: CREATED_AT_UNIX_MS + 531_000,
            },
        },
        same_host_replay: "byte_identical",
        cross_host_request_response_binding: "rejected",
    };

    let issuer_keyset = MaplePairingIssuerKeySetV1 {
        version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        keys: vec![issuer.public_key_entry()],
    };
    let rotated_retaining_previous = MaplePairingIssuerKeySetV1 {
        version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        keys: vec![issuer.public_key_entry(), next_issuer.public_key_entry()],
    };
    let rotated_without_previous = MaplePairingIssuerKeySetV1 {
        version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        keys: vec![next_issuer.public_key_entry()],
    };
    let remapped_previous_key_id = MaplePairingIssuerKeySetV1 {
        version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        keys: vec![
            remapped_issuer.public_key_entry(),
            next_issuer.public_key_entry(),
        ],
    };
    let issuer_rotation_vectors = IssuerRotationVectors {
        artifact_signed_by_initial: request_ticket.clone(),
        initial: IssuerKeySetOutcome {
            keyset: issuer_keyset.clone(),
            wire_verification_expected: "accepted",
            registry_reconciliation_expected: "accepted_initial",
        },
        rotated_retaining_previous: IssuerKeySetOutcome {
            keyset: rotated_retaining_previous,
            wire_verification_expected: "accepted",
            registry_reconciliation_expected: "append_accepted",
        },
        rotated_without_previous: IssuerKeySetOutcome {
            keyset: rotated_without_previous,
            wire_verification_expected: "unknown_issuer",
            registry_reconciliation_expected: "MaplePairingIssuerConfigurationConflict",
        },
        remapped_previous_key_id: IssuerKeySetOutcome {
            keyset: remapped_previous_key_id,
            wire_verification_expected: "invalid_signature",
            registry_reconciliation_expected: "MaplePairingIssuerConfigurationConflict",
        },
        retained_registry_rule: "existing_key_id_must_retain_exact_public_key",
    };

    let reset_clear_pending_checkpoint = signed_checkpoint(
        &issuer,
        third_reset.host.clone(),
        4,
        third_reset.revocation_stream_id,
        third_reset.revocation_stream_generation,
        1,
        0,
    );
    let reset_pending_checkpoint_transcript = reset_clear_pending_checkpoint.transcript().unwrap();
    let (reset_clear_pending_checkpoint_transcript_hex, reset_clear_pending_checkpoint_digest) =
        transcript_pair(&reset_pending_checkpoint_transcript);
    let reset_clear_pending_sync = MapleRevocationSyncV1::status_for_checkpoint(
        4,
        reset_clear_pending_checkpoint.clone(),
        Some(third_reset.clone()),
    )
    .unwrap();

    let reset_clear_list_revocations_request = sign_list_revocations_request(
        ListMaplePairingRevocationsRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            query_id: uuid("31313131-3131-4131-8131-313131313131"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            revocation_stream_id: third_reset.revocation_stream_id,
            revocation_stream_generation: third_reset.revocation_stream_generation,
            after_issuer_sequence: 0,
            limit: Some(1),
            signature: String::new(),
        },
        &host_key,
    );
    let reset_list_transcript = reset_clear_list_revocations_request.transcript().unwrap();
    let (
        reset_clear_list_revocations_request_transcript_hex,
        reset_clear_list_revocations_request_digest,
    ) = transcript_pair(&reset_list_transcript);
    let reset_clear_list_revocations_response = ListMaplePairingRevocationsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: reset_clear_list_revocations_request.query_id,
        revocation_sync: reset_clear_pending_sync.clone(),
        events: vec![MapleRevocationStreamEventV1::ResetClearRequired(
            third_reset.clone(),
        )],
        next_after_issuer_sequence: 1,
        has_more: false,
    };

    let reset_clear_ack_request = sign_ack_request(
        AckMaplePairingRevocationRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: uuid("32323232-3232-4232-8232-323232323232"),
            asserted_account_id: account_id,
            asserted_project_id: project_id,
            host_registration_id,
            revocation_stream_id: third_reset.revocation_stream_id,
            revocation_stream_generation: third_reset.revocation_stream_generation,
            event_id: third_reset.event_id,
            issuer_sequence: 1,
            event_digest: STANDARD.encode(third_reset.event_digest().unwrap()),
            expected_previous_issuer_sequence: 0,
            signature: String::new(),
        },
        &host_key,
    );
    let reset_ack_transcript = reset_clear_ack_request.transcript().unwrap();
    let (reset_clear_ack_request_transcript_hex, reset_clear_ack_request_digest) =
        transcript_pair(&reset_ack_transcript);
    let reset_clear_acked_checkpoint = signed_checkpoint(
        &issuer,
        third_reset.host.clone(),
        4,
        third_reset.revocation_stream_id,
        third_reset.revocation_stream_generation,
        1,
        1,
    );
    let reset_acked_checkpoint_transcript = reset_clear_acked_checkpoint.transcript().unwrap();
    let (reset_clear_acked_checkpoint_transcript_hex, reset_clear_acked_checkpoint_digest) =
        transcript_pair(&reset_acked_checkpoint_transcript);
    let reset_clear_ack_response = AckMaplePairingRevocationResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        operation_id: reset_clear_ack_request.operation_id,
        host_registration_id,
        stream_checkpoint: reset_clear_acked_checkpoint.clone(),
        event_id: third_reset.event_id,
        issuer_sequence: 1,
        last_acked_issuer_sequence: 1,
        accepted_at_unix_ms: third_reset.reset_at_unix_ms + 1_000,
    };
    let reset_acked_sync =
        MapleRevocationSyncV1::status_for_checkpoint(4, reset_clear_acked_checkpoint.clone(), None)
            .unwrap();
    let reset_clear_historical_acked_response = ListMaplePairingRevocationsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: reset_clear_list_revocations_request.query_id,
        revocation_sync: reset_acked_sync,
        events: vec![MapleRevocationStreamEventV1::ResetClearRequired(
            third_reset.clone(),
        )],
        next_after_issuer_sequence: 1,
        has_more: false,
    };

    let reset_clear_later_pair_revocation = sign_pair_revocation(
        MaplePairRevocationV1 {
            event_id: uuid("33333333-0000-4333-8333-333333333333"),
            issuer_sequence: 2,
            revocation_stream_id: third_reset.revocation_stream_id,
            revocation_stream_generation: third_reset.revocation_stream_generation,
            revoked_at_unix_ms: third_reset.reset_at_unix_ms + 2_000,
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
            ..pair_revocation.clone()
        },
        &issuer,
    )
    .unwrap();
    let reset_later_revocation_transcript = reset_clear_later_pair_revocation.transcript().unwrap();
    let (
        reset_clear_later_pair_revocation_transcript_hex,
        reset_clear_later_pair_revocation_digest,
    ) = transcript_pair(&reset_later_revocation_transcript);
    let reset_clear_later_checkpoint = signed_checkpoint(
        &issuer,
        third_reset.host.clone(),
        4,
        third_reset.revocation_stream_id,
        third_reset.revocation_stream_generation,
        2,
        1,
    );
    let reset_later_checkpoint_transcript = reset_clear_later_checkpoint.transcript().unwrap();
    let (reset_clear_later_checkpoint_transcript_hex, reset_clear_later_checkpoint_digest) =
        transcript_pair(&reset_later_checkpoint_transcript);
    let reset_later_sync =
        MapleRevocationSyncV1::status_for_checkpoint(4, reset_clear_later_checkpoint.clone(), None)
            .unwrap();
    let reset_clear_historical_later_request = sign_list_revocations_request(
        ListMaplePairingRevocationsRequest {
            query_id: uuid("34343434-3434-4434-8434-343434343434"),
            limit: Some(2),
            signature: String::new(),
            ..reset_clear_list_revocations_request.clone()
        },
        &host_key,
    );
    let reset_later_request_transcript = reset_clear_historical_later_request.transcript().unwrap();
    let (
        reset_clear_historical_later_request_transcript_hex,
        reset_clear_historical_later_request_digest,
    ) = transcript_pair(&reset_later_request_transcript);
    let reset_clear_historical_later_response = ListMaplePairingRevocationsResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        query_id: reset_clear_historical_later_request.query_id,
        revocation_sync: reset_later_sync,
        events: vec![
            MapleRevocationStreamEventV1::ResetClearRequired(third_reset.clone()),
            MapleRevocationStreamEventV1::PairRevocation(reset_clear_later_pair_revocation.clone()),
        ],
        next_after_issuer_sequence: 2,
        has_more: false,
    };

    let register_device_request_epoch_1 = registration_request(
        uuid("35353535-3535-4535-8535-353535353535"),
        Some(5),
        1,
        RegistrationRequestHostFixture {
            device_id: host_device_id,
            installation_id: host_installation_id,
            endpoint_epoch: 5,
            signing_key: &host_key,
            display_name: "Maple fixture host",
        },
    );
    let (register_device_request_epoch_1_transcript_hex, register_device_request_epoch_1_digest) =
        registration_transcript_pair(&register_device_request_epoch_1, &host_key);
    let register_device_request_epoch_4 = registration_request(
        uuid("36363636-3636-4636-8636-363636363636"),
        Some(5),
        4,
        RegistrationRequestHostFixture {
            device_id: host_device_id,
            installation_id: host_installation_id,
            endpoint_epoch: 5,
            signing_key: &host_key,
            display_name: "Maple fixture host",
        },
    );
    let (register_device_request_epoch_4_transcript_hex, register_device_request_epoch_4_digest) =
        registration_transcript_pair(&register_device_request_epoch_4, &host_key);

    let pre_reset_ready_checkpoint = signed_checkpoint(
        &issuer,
        host_claim_current.clone(),
        1,
        base_stream_id,
        5,
        12,
        12,
    );
    let pre_reset_ready_sync =
        MapleRevocationSyncV1::status_for_checkpoint(1, pre_reset_ready_checkpoint, None).unwrap();
    let register_device_response_ready = registration_response(
        &register_device_request_epoch_1,
        host_registration_id,
        6,
        CREATED_AT_UNIX_MS,
        pre_reset_ready_sync,
    );
    let registration_pending_checkpoint = signed_checkpoint(
        &issuer,
        host_claim_current.clone(),
        1,
        base_stream_id,
        5,
        13,
        12,
    );
    let registration_pending_sync =
        MapleRevocationSyncV1::status_for_checkpoint(1, registration_pending_checkpoint, None)
            .unwrap();
    let register_device_response_revocations_pending = registration_response(
        &register_device_request_epoch_1,
        host_registration_id,
        6,
        CREATED_AT_UNIX_MS,
        registration_pending_sync,
    );
    let register_device_response_reset_clear_required = registration_response(
        &register_device_request_epoch_4,
        host_registration_id,
        6,
        third_reset.reset_at_unix_ms + 500,
        reset_clear_pending_sync.clone(),
    );
    let list_devices_response_security_epoch = ListMapleDevicesResponse {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        security_epoch: 4,
        devices: Vec::new(),
        next_cursor: None,
        has_more: false,
    };

    let mut changed_pre_reset_request = register_device_request_epoch_1.clone();
    changed_pre_reset_request.display_name = "changed pre-reset request".to_string();
    let changed_pre_reset_request = sign_registration_request(changed_pre_reset_request, &host_key);
    let mut changed_pending_reset_request = register_device_request_epoch_4.clone();
    changed_pending_reset_request.display_name = "changed pending-reset request".to_string();
    let changed_pending_reset_request =
        sign_registration_request(changed_pending_reset_request, &host_key);
    let mut fresh_retired_operation = register_device_request_epoch_4.clone();
    fresh_retired_operation.operation_id = uuid("37373737-3737-4737-8737-373737373737");
    let fresh_retired_operation = sign_registration_request(fresh_retired_operation, &host_key);

    let fresh_device_id = uuid("58585858-5858-4858-8858-585858585858");
    let fresh_installation_id = uuid("59595959-5959-4959-8959-595959595959");
    let fresh_registration_id = uuid("57575757-5757-4757-8757-575757575757");
    let fresh_installation_request = registration_request(
        uuid("56565656-5656-4656-8656-565656565656"),
        None,
        4,
        RegistrationRequestHostFixture {
            device_id: fresh_device_id,
            installation_id: fresh_installation_id,
            endpoint_epoch: 1,
            signing_key: &fresh_installation_key,
            display_name: "Fresh Maple fixture host",
        },
    );
    let fresh_host_claim = make_claim(
        fresh_registration_id,
        fresh_device_id,
        fresh_installation_id,
        &fresh_installation_key,
        1,
    );
    let fresh_checkpoint = signed_checkpoint(
        &issuer,
        fresh_host_claim,
        4,
        uuid("59590000-0000-4000-8000-000000000001"),
        1,
        0,
        0,
    );
    let fresh_ready_sync =
        MapleRevocationSyncV1::status_for_checkpoint(4, fresh_checkpoint, None).unwrap();
    let fresh_installation_response = registration_response(
        &fresh_installation_request,
        fresh_registration_id,
        1,
        reset_clear_ack_response.accepted_at_unix_ms + 1_000,
        fresh_ready_sync,
    );
    let post_ack_registration_outcome_vectors = PostAckRegistrationOutcomeVectors {
        exact_pre_reset_replay: RegistrationOutcomeVector {
            request: register_device_request_epoch_1.clone(),
            response: Some(register_device_response_ready.clone()),
            expected: "frozen_ready_replay",
        },
        changed_pre_reset_same_operation: RegistrationOutcomeVector {
            request: changed_pre_reset_request,
            response: None,
            expected: "conflict",
        },
        exact_pending_reset_replay: RegistrationOutcomeVector {
            request: register_device_request_epoch_4.clone(),
            response: Some(register_device_response_reset_clear_required.clone()),
            expected: "reset_clear_required_replay",
        },
        changed_pending_reset_same_operation: RegistrationOutcomeVector {
            request: changed_pending_reset_request,
            response: None,
            expected: "conflict",
        },
        fresh_operation_retired_installation: RegistrationOutcomeVector {
            request: fresh_retired_operation,
            response: None,
            expected: "MapleInstallationRetired",
        },
        fresh_installation_current_epoch: RegistrationOutcomeVector {
            request: fresh_installation_request,
            response: Some(fresh_installation_response),
            expected: "ready",
        },
        operation_replay_precedes_retirement_and_epoch_gates: true,
    };

    let security_epoch_outcome_vectors = vec![
        SecurityEpochOutcomeVector {
            name: "known_epoch_stale",
            known_security_epoch: 1,
            current_security_epoch: 4,
            pending_reset_clear: true,
            registration_operation_accepted: false,
            outcome: "MapleSecurityEpochStale",
        },
        SecurityEpochOutcomeVector {
            name: "known_epoch_ahead",
            known_security_epoch: 5,
            current_security_epoch: 4,
            pending_reset_clear: true,
            registration_operation_accepted: false,
            outcome: "conflict",
        },
        SecurityEpochOutcomeVector {
            name: "current_epoch_pending_reset_dominates",
            known_security_epoch: 4,
            current_security_epoch: 4,
            pending_reset_clear: true,
            registration_operation_accepted: true,
            outcome: "reset_clear_required",
        },
        SecurityEpochOutcomeVector {
            name: "fresh_current_epoch_after_ack",
            known_security_epoch: 4,
            current_security_epoch: 4,
            pending_reset_clear: false,
            registration_operation_accepted: true,
            outcome: "ready",
        },
    ];

    let registration_operation_tombstone_vectors = RegistrationOperationTombstoneVectors {
        storage: RegistrationTombstoneStorage {
            operation_lookup_digest: "hmac_only",
            raw_operation_id_stored: false,
            request_binding: "request_mac",
            frozen_response_retained: true,
        },
        exact_old_request: "frozen_ready_replay",
        changed_request_same_operation_lookup: "conflict",
        exact_pending_reset_request_after_ack: "reset_clear_required_replay",
        changed_pending_reset_request_after_ack: "conflict",
        fresh_operation_retired_installation: "MapleInstallationRetired",
        fresh_installation_current_epoch: "ready",
    };

    let wire_structural_assertions = WireStructuralAssertions {
        list_devices_security_epoch_source: "authenticated_encrypted_account_authority_head",
        list_devices_response_signed: false,
        reset_event_json_key_order: ["event_type", "event"],
        pending_reset_checkpoint: PendingResetCheckpointShape {
            last_issued_issuer_sequence: 1,
            last_acked_issuer_sequence: 0,
        },
        historical_reset_event_allowed_after_ack: true,
        historical_reset_event_allowed_when_later_events_exist: true,
        reset_admission_public_shape: ["admission_count", "admission_set_digest"],
        retained_admission_leaves_cross_wire: false,
        fixture_pretty_indent_spaces: 2,
        fixture_trailing_newline_count: 1,
        sdk_fixture_copy: "explicit_reviewed_byte_copy",
    };

    FixtureVectors {
        description: "Authoritative deterministic Maple pairing v1 wire vectors, including typed CREATE/REVOKE material bundles, security-epoch checkpoints, typed revocation events, aggregate reset-clear instructions, recursive chaining, registration sync and tombstone outcomes, host-scoped ACK namespaces, exact-host successor binding, retained issuer rotation, paging, and sanitized conflicts.",
        fixture_schema_version: FIXTURE_SCHEMA_VERSION,
        issuer_keyset,
        test_private_seeds_hex: TestPrivateSeeds {
            controller: hex::encode(CONTROLLER_SEED),
            host: hex::encode(HOST_SEED),
            second_host: hex::encode(SECOND_HOST_SEED),
            fresh_installation: hex::encode(FRESH_INSTALLATION_SEED),
            issuer: hex::encode(ISSUER_SEED),
            issuer_next: hex::encode(NEXT_ISSUER_SEED),
            issuer_remap: hex::encode(REMAPPED_ISSUER_SEED),
        },

        create_request,
        create_request_transcript_hex,
        create_request_digest,
        request_ticket,
        request_ticket_transcript_hex,
        request_ticket_digest,
        approval_request,
        approval_request_transcript_hex,
        approval_request_digest,
        pair_authorization,
        pair_authorization_transcript_hex,
        pair_authorization_digest,
        confirm_request,
        confirm_request_transcript_hex,
        confirm_request_digest,
        active_receipt,
        typed_materialization_vectors,

        list_pairings_request,
        list_pairings_request_transcript_hex,
        list_pairings_request_digest,
        pairing_status_request,
        pairing_status_request_transcript_hex,
        pairing_status_request_digest,
        revoke_request,
        revoke_request_transcript_hex,
        revoke_request_digest,
        list_revocations_request,
        list_revocations_request_transcript_hex,
        list_revocations_request_digest,
        ack_revocation_request,
        ack_revocation_request_transcript_hex,
        ack_revocation_request_digest,
        pair_revocation,
        pair_revocation_transcript_hex,
        pair_revocation_digest,

        list_pairings_response,
        pairing_status_response,
        list_revocations_response,
        revoke_receipt,
        ack_revocation_response,
        revocation_stream_checkpoint,
        revocation_stream_checkpoint_transcript_hex,
        revocation_stream_checkpoint_digest,
        ack_revocation_stream_checkpoint,
        ack_revocation_stream_checkpoint_transcript_hex,
        ack_revocation_stream_checkpoint_digest,

        discovery_list_revocations_request,
        discovery_list_revocations_request_transcript_hex,
        discovery_list_revocations_request_digest,
        discovery_revocation_stream_checkpoint,
        discovery_revocation_stream_checkpoint_transcript_hex,
        discovery_revocation_stream_checkpoint_digest,
        discovery_list_revocations_response,

        reset_clear_admission_set_vectors,
        reset_clear_three_reset_chain,
        reset_clear_successor_vectors,
        two_host_ack_namespace_vectors,
        issuer_rotation_vectors,
        reset_clear_pending_checkpoint,
        reset_clear_pending_checkpoint_transcript_hex,
        reset_clear_pending_checkpoint_digest,
        reset_clear_pending_sync,
        reset_clear_list_revocations_request,
        reset_clear_list_revocations_request_transcript_hex,
        reset_clear_list_revocations_request_digest,
        reset_clear_list_revocations_response,
        reset_clear_ack_request,
        reset_clear_ack_request_transcript_hex,
        reset_clear_ack_request_digest,
        reset_clear_acked_checkpoint,
        reset_clear_acked_checkpoint_transcript_hex,
        reset_clear_acked_checkpoint_digest,
        reset_clear_ack_response,
        reset_clear_historical_acked_response,
        reset_clear_later_pair_revocation,
        reset_clear_later_pair_revocation_transcript_hex,
        reset_clear_later_pair_revocation_digest,
        reset_clear_later_checkpoint,
        reset_clear_later_checkpoint_transcript_hex,
        reset_clear_later_checkpoint_digest,
        reset_clear_historical_later_request,
        reset_clear_historical_later_request_transcript_hex,
        reset_clear_historical_later_request_digest,
        reset_clear_historical_later_response,

        register_device_request_epoch_1,
        register_device_request_epoch_1_transcript_hex,
        register_device_request_epoch_1_digest,
        register_device_request_epoch_4,
        register_device_request_epoch_4_transcript_hex,
        register_device_request_epoch_4_digest,
        register_device_response_ready,
        register_device_response_revocations_pending,
        register_device_response_reset_clear_required,
        list_devices_response_security_epoch,
        post_ack_registration_outcome_vectors,

        maple_security_epoch_stale_error: PublicErrorVector {
            status: 409,
            message: "Maple device security epoch is stale; refresh device state and retry.",
            code: "MapleSecurityEpochStale",
        },
        maple_pairing_reset_clear_required_error: PublicErrorVector {
            status: 409,
            message: "Maple remote access must be cleared on the host before this operation can continue.",
            code: "MaplePairingResetClearRequired",
        },
        maple_installation_retired_error: PublicErrorVector {
            status: 409,
            message: "This Maple installation enrollment is retired; reset Remote access on this device and enroll it again.",
            code: "MapleInstallationRetired",
        },
        security_epoch_outcome_vectors,
        registration_operation_tombstone_vectors,
        wire_structural_assertions,
    }
}

fn render_vectors() -> String {
    let mut rendered = serde_json::to_string_pretty(&build_vectors()).unwrap();
    rendered.push('\n');
    rendered
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/maple_pairing_v1_vectors.json")
}

fn invalid_fixture_update(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn require_pinned_fixture_preimage(path: &Path) -> io::Result<()> {
    let path_metadata = fs::symlink_metadata(path)?;
    if path_metadata.file_type().is_symlink() {
        return Err(invalid_fixture_update(format!(
            "refusing to update symlinked fixture {}",
            path.display()
        )));
    }
    if !path_metadata.is_file() {
        return Err(invalid_fixture_update(format!(
            "refusing to update non-regular fixture {}",
            path.display()
        )));
    }

    let mut file = File::open(path)?;
    let opened_metadata = file.metadata()?;
    if !opened_metadata.is_file() {
        return Err(invalid_fixture_update(format!(
            "refusing to update non-regular opened fixture {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if path_metadata.dev() != opened_metadata.dev()
            || path_metadata.ino() != opened_metadata.ino()
        {
            return Err(invalid_fixture_update(format!(
                "fixture changed while opening {}",
                path.display()
            )));
        }
    }

    let mut current = Vec::new();
    file.read_to_end(&mut current)?;
    let actual_sha256 = hex::encode(Sha256::digest(&current));
    if actual_sha256 != PINNED_FIXTURE_PREIMAGE_SHA256 {
        return Err(invalid_fixture_update(format!(
            "refusing to replace fixture {} with unexpected SHA-256 {}; expected {}",
            path.display(),
            actual_sha256,
            PINNED_FIXTURE_PREIMAGE_SHA256
        )));
    }
    Ok(())
}

fn sync_fixture_directory(parent: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
        Ok(())
    }
}

fn replace_fixture_atomically(path: &Path, rendered: &[u8]) -> io::Result<()> {
    require_pinned_fixture_preimage(path)?;
    let parent = path.parent().ok_or_else(|| {
        invalid_fixture_update(format!("fixture path {} has no parent", path.display()))
    })?;
    let temp_path = parent.join(FIXTURE_UPDATE_TEMP_FILE_NAME);
    let mut temp_created = false;
    let result = (|| {
        let mut temp = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        temp_created = true;
        temp.write_all(rendered)?;
        temp.sync_all()?;
        drop(temp);

        require_pinned_fixture_preimage(path)?;
        fs::rename(&temp_path, path)?;
        sync_fixture_directory(parent)
    })();
    if result.is_err() && temp_created {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[test]
fn generated_maple_pairing_vectors_match_the_checked_in_fixture_exactly() {
    let generated = render_vectors();
    let checked_in = include_str!("../../tests/fixtures/maple_pairing_v1_vectors.json");
    assert_eq!(
        generated.as_bytes(),
        checked_in.as_bytes(),
        "run the ignored, env-gated updater intentionally and review the complete fixture diff"
    );
    assert_eq!(
        build_vectors().fixture_schema_version,
        FIXTURE_SCHEMA_VERSION
    );
    assert!(generated.ends_with('\n'));
    assert!(!generated.ends_with("\n\n"));
    assert!(!generated.contains('\r'));
}

#[test]
fn generated_extension_vectors_exercise_the_frozen_authority_contract() {
    let vectors = build_vectors();
    let issuers = &vectors.issuer_keyset;

    assert_eq!(
        vectors.typed_materialization_vectors.create.request,
        vectors.create_request
    );
    assert_eq!(
        vectors
            .typed_materialization_vectors
            .create
            .response
            .pairing
            .state,
        MaplePairingState::Pending
    );
    assert_eq!(
        vectors.typed_materialization_vectors.revoke.request,
        vectors.revoke_request
    );
    assert_eq!(
        vectors.typed_materialization_vectors.revoke.response,
        vectors.revoke_receipt
    );

    let successor = &vectors.reset_clear_successor_vectors;
    successor
        .accepted
        .instruction
        .verify_direct_successor(
            &successor.predecessor,
            &successor.accepted.checkpoint,
            issuers,
        )
        .unwrap();
    assert!(successor
        .changed_host
        .instruction
        .verify_direct_successor(
            &successor.predecessor,
            &successor.changed_host.checkpoint,
            issuers,
        )
        .is_err());

    let two_host = &vectors.two_host_ack_namespace_vectors;
    assert_eq!(
        two_host.host_a.request.operation_id,
        two_host.host_b.request.operation_id
    );
    assert_ne!(
        two_host.host_a.request.host_registration_id,
        two_host.host_b.request.host_registration_id
    );
    two_host
        .host_a
        .response
        .verify_against_request(&two_host.host_a.request, issuers)
        .unwrap();
    two_host
        .host_b
        .response
        .verify_against_request(&two_host.host_b.request, issuers)
        .unwrap();
    assert!(two_host
        .host_a
        .response
        .verify_against_request(&two_host.host_b.request, issuers)
        .is_err());
    assert!(two_host
        .host_b
        .response
        .verify_against_request(&two_host.host_a.request, issuers)
        .is_err());

    let rotation = &vectors.issuer_rotation_vectors;
    for keyset in [
        &rotation.initial.keyset,
        &rotation.rotated_retaining_previous.keyset,
    ] {
        rotation
            .artifact_signed_by_initial
            .verify_unexpired(
                keyset,
                rotation.artifact_signed_by_initial.created_at_unix_ms,
                0,
            )
            .unwrap();
    }
    assert!(rotation
        .artifact_signed_by_initial
        .verify_unexpired(
            &rotation.rotated_without_previous.keyset,
            rotation.artifact_signed_by_initial.created_at_unix_ms,
            0,
        )
        .is_err());
    assert!(rotation
        .artifact_signed_by_initial
        .verify_unexpired(
            &rotation.remapped_previous_key_id.keyset,
            rotation.artifact_signed_by_initial.created_at_unix_ms,
            0,
        )
        .is_err());
    assert_eq!(
        rotation
            .rotated_retaining_previous
            .registry_reconciliation_expected,
        "append_accepted"
    );
    assert_eq!(
        rotation
            .rotated_without_previous
            .registry_reconciliation_expected,
        "MaplePairingIssuerConfigurationConflict"
    );
    assert_eq!(
        rotation
            .remapped_previous_key_id
            .registry_reconciliation_expected,
        "MaplePairingIssuerConfigurationConflict"
    );

    let post_ack = &vectors.post_ack_registration_outcome_vectors;
    assert_eq!(
        post_ack.exact_pre_reset_replay.request.operation_id,
        post_ack
            .changed_pre_reset_same_operation
            .request
            .operation_id
    );
    assert_eq!(
        post_ack.exact_pending_reset_replay.request.operation_id,
        post_ack
            .changed_pending_reset_same_operation
            .request
            .operation_id
    );
    assert_eq!(
        post_ack.exact_pre_reset_replay.response,
        Some(vectors.register_device_response_ready.clone())
    );
    assert_eq!(
        post_ack.exact_pre_reset_replay.expected,
        "frozen_ready_replay"
    );
    assert_eq!(
        post_ack.changed_pre_reset_same_operation.expected,
        "conflict"
    );
    let frozen_ready_checkpoint = &vectors
        .register_device_response_ready
        .revocation_sync
        .stream_checkpoint;
    let first_reset = &vectors.reset_clear_three_reset_chain[0].instruction;
    assert_eq!(frozen_ready_checkpoint.host, first_reset.host);
    assert_eq!(
        frozen_ready_checkpoint.security_epoch,
        first_reset.source_security_epoch
    );
    assert_eq!(
        frozen_ready_checkpoint.revocation_stream_id,
        first_reset.source_revocation_stream_id
    );
    assert_eq!(
        frozen_ready_checkpoint.revocation_stream_generation,
        first_reset.source_revocation_stream_generation
    );
    assert_eq!(
        post_ack.exact_pending_reset_replay.response,
        Some(
            vectors
                .register_device_response_reset_clear_required
                .clone()
        )
    );
    assert_eq!(
        post_ack.exact_pending_reset_replay.expected,
        "reset_clear_required_replay"
    );
    assert_eq!(
        post_ack.changed_pending_reset_same_operation.expected,
        "conflict"
    );
    assert_eq!(
        post_ack.fresh_operation_retired_installation.expected,
        "MapleInstallationRetired"
    );
    assert_eq!(post_ack.fresh_installation_current_epoch.expected, "ready");
    assert!(post_ack.operation_replay_precedes_retirement_and_epoch_gates);
    let reset_recovery_accepted_at = vectors
        .register_device_response_reset_clear_required
        .accepted_at
        .timestamp_millis();
    assert!(
        reset_recovery_accepted_at
            >= vectors.reset_clear_three_reset_chain[2]
                .instruction
                .reset_at_unix_ms
    );
    assert!(reset_recovery_accepted_at < vectors.reset_clear_ack_response.accepted_at_unix_ms);
    assert!(
        post_ack
            .fresh_installation_current_epoch
            .response
            .as_ref()
            .unwrap()
            .accepted_at
            .timestamp_millis()
            > vectors.reset_clear_ack_response.accepted_at_unix_ms
    );
    assert_eq!(vectors.maple_installation_retired_error.status, 409);
    assert_eq!(
        vectors.maple_installation_retired_error.code,
        "MapleInstallationRetired"
    );
}

#[test]
#[ignore = "explicit fixture update; set MAPLE_PAIRING_UPDATE_VECTORS=1"]
fn update_maple_pairing_vectors_fixture() {
    assert_eq!(
        env::var("MAPLE_PAIRING_UPDATE_VECTORS").as_deref(),
        Ok("1"),
        "refusing to update without MAPLE_PAIRING_UPDATE_VECTORS=1"
    );
    let rendered = render_vectors();
    replace_fixture_atomically(&fixture_path(), rendered.as_bytes())
        .expect("atomically replace the pinned backend pairing fixture");
}
