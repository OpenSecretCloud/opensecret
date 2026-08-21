//! Maple one-way pairing v1 wire contract and cryptographic transcripts.
//!
//! The authenticated account/project at the HTTP boundary remain authoritative.
//! The asserted identifiers in these DTOs are signed preconditions, never an
//! authorization source. V1 deliberately uses the same durable Ed25519 public
//! key for the Maple installation identity and its Iroh endpoint identity.

use crate::encrypt::CanonicalBytes;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;
use uuid::Uuid;

pub const MAPLE_PAIRING_PROTOCOL_VERSION_V1: u16 = 1;
pub const MAPLE_PAIRING_TRANSCRIPT_VERSION_V1: u16 = 1;
pub const MAPLE_PAIRING_ARTIFACT_VERSION_V1: u16 = 1;
pub const MAPLE_PAIRING_DEFAULT_PAGE_SIZE: u16 = 25;
pub const MAPLE_PAIRING_MAX_PAGE_SIZE: u16 = 100;
pub const MAPLE_RESET_CLEAR_MAX_ADMISSIONS: u16 = 128;
pub const MAPLE_PAIRING_MAX_CURSOR_BYTES: usize = 512;
pub const MAPLE_PAIRING_MAX_ISSUER_KEYS: usize = 1024;
pub const MAPLE_PAIR_REQUEST_MAX_TTL_MS: i64 = 600_000;
pub const MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS: i64 = 30_000;

const CREATE_REQUEST_DOMAIN: &str = "os.maple-pair-request.v1";
const LIST_PAIRINGS_DOMAIN: &str = "os.maple-pair-list.v1";
const PAIRING_STATUS_DOMAIN: &str = "os.maple-pair-status.v1";
const APPROVE_PAIRING_DOMAIN: &str = "os.maple-pair-approval.v1";
const CONFIRM_PAIRING_DOMAIN: &str = "os.maple-pair-host-commit.v1";
const REVOKE_PAIRING_DOMAIN: &str = "os.maple-pair-revocation-request.v1";
const LIST_REVOCATIONS_DOMAIN: &str = "os.maple-pair-revocation-list.v1";
const ACK_REVOCATION_DOMAIN: &str = "os.maple-pair-revocation-ack.v1";
const REQUEST_TICKET_DOMAIN: &str = "os.maple-pair-request-ticket.v1";
const PAIR_AUTHORIZATION_DOMAIN: &str = "os.maple-pair-authorization.v1";
const PAIR_REVOCATION_DOMAIN: &str = "os.maple-pair-revocation.v1";
const REVOCATION_STREAM_CHECKPOINT_DOMAIN: &str = "os.maple-revocation-stream-checkpoint.v1";
const RESET_CLEAR_ADMISSION_SET_DOMAIN: &str = "os.maple-reset-clear-admission-set.v1";
const RESET_CLEAR_INSTRUCTION_MATERIAL_DOMAIN: &str =
    "os.maple-reset-clear-instruction-material.v1";
const RESET_CLEAR_CHAIN_DOMAIN: &str = "os.maple-reset-clear-chain.v1";
const RESET_CLEAR_REQUIRED_DOMAIN: &str = "os.maple-reset-clear-required.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MaplePairingWireError {
    #[error("unsupported Maple pairing version")]
    UnsupportedVersion,
    #[error("invalid Maple pairing field: {0}")]
    InvalidField(&'static str),
    #[error("non-canonical Maple pairing encoding: {0}")]
    NonCanonicalEncoding(&'static str),
    #[error("invalid Maple pairing signature")]
    InvalidSignature,
    #[error("unknown Maple pairing issuer")]
    UnknownIssuer,
    #[error("invalid Maple pairing issuer keyset")]
    InvalidIssuerKeySet,
    #[error("Maple pairing issuer signing failed")]
    IssuerSigningFailed,
    #[error("pair request ticket lifetime exceeds the v1 bound")]
    TicketLifetimeTooLong,
    #[error("pair request ticket is not valid at the trusted time")]
    TicketNotCurrentlyValid,
    #[error("requested clock skew exceeds the v1 bound")]
    ClockSkewOutOfRange,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaplePairingDirection {
    ControllerToHost,
}

impl MaplePairingDirection {
    fn as_wire(self) -> &'static str {
        match self {
            Self::ControllerToHost => "controller_to_host",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MaplePairingState {
    Pending,
    AwaitingHostCommit,
    Active,
    Expired,
    Revoked,
}

impl MaplePairingState {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::AwaitingHostCommit => "awaiting_host_commit",
            Self::Active => "active",
            Self::Expired => "expired",
            Self::Revoked => "revoked",
        }
    }

    fn canonical_rank(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::AwaitingHostCommit => 1,
            Self::Active => 2,
            Self::Expired => 3,
            Self::Revoked => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaplePairingRole {
    Controller,
    Host,
}

impl MaplePairingRole {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Controller => "controller",
            Self::Host => "host",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaplePairingIdentityAlgorithm {
    Ed25519,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MapleResetClearScopeV1 {
    AllPairAuthorizationsForAccountProjectHostInstallation,
}

impl MapleResetClearScopeV1 {
    fn as_wire(self) -> &'static str {
        match self {
            Self::AllPairAuthorizationsForAccountProjectHostInstallation => {
                "all_pair_authorizations_for_account_project_host_installation"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MapleRevocationSyncStatusV1 {
    Ready,
    RevocationsPending,
    ResetClearRequired,
}

impl MaplePairingIdentityAlgorithm {
    pub(crate) fn as_wire(self) -> &'static str {
        match self {
            Self::Ed25519 => "ed25519",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CreateMaplePairingRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub controller_registration_id: Uuid,
    pub controller_device_id: Uuid,
    pub controller_installation_id: Uuid,
    pub controller_endpoint_id: String,
    pub controller_endpoint_epoch: u64,
    pub host_registration_id: Uuid,
    pub host_device_id: Uuid,
    pub host_installation_id: Uuid,
    pub host_endpoint_id: String,
    pub host_endpoint_epoch: u64,
    pub direction: MaplePairingDirection,
    pub execution_target_id: Uuid,
    pub pairing_request_nonce: String,
    pub protocol_min: u16,
    pub protocol_max: u16,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ListMaplePairingsRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub query_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub actor_registration_id: Uuid,
    pub role: MaplePairingRole,
    pub states: Vec<MaplePairingState>,
    pub cursor: Option<String>,
    pub limit: Option<u16>,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingStatusRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub query_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub actor_registration_id: Uuid,
    pub pair_id: Uuid,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ApproveMaplePairingRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub host_registration_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub request_ticket_digest: String,
    pub host_approval_nonce: String,
    pub approved_protocol_min: u16,
    pub approved_protocol_max: u16,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConfirmMaplePairingRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub host_registration_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub pair_authorization_digest: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RevokeMaplePairingRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub actor_registration_id: Uuid,
    pub actor_role: MaplePairingRole,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub reason_code: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ListMaplePairingRevocationsRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub query_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub host_registration_id: Uuid,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub after_issuer_sequence: u64,
    pub limit: Option<u16>,
    pub signature: String,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AckMaplePairingRevocationRequest {
    pub protocol_version: u16,
    pub transcript_version: u16,
    pub operation_id: Uuid,
    pub asserted_account_id: Uuid,
    pub asserted_project_id: Uuid,
    pub host_registration_id: Uuid,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub event_id: Uuid,
    pub issuer_sequence: u64,
    pub event_digest: String,
    pub expected_previous_issuer_sequence: u64,
    pub signature: String,
}

impl std::fmt::Debug for AckMaplePairingRevocationRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AckMaplePairingRevocationRequest")
            .field("protocol_version", &self.protocol_version)
            .field("transcript_version", &self.transcript_version)
            .field("issuer_sequence", &self.issuer_sequence)
            .field(
                "expected_previous_issuer_sequence",
                &self.expected_previous_issuer_sequence,
            )
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MapleDeviceClaimV1 {
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub identity_algorithm: MaplePairingIdentityAlgorithm,
    pub identity_public_key: String,
    pub endpoint_id: String,
    pub endpoint_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairRequestTicketV1 {
    pub artifact_version: u16,
    pub subject_account_id: Uuid,
    pub subject_project_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub direction: MaplePairingDirection,
    pub execution_target_id: Uuid,
    pub controller: MapleDeviceClaimV1,
    pub host: MapleDeviceClaimV1,
    pub pairing_request_nonce: String,
    pub controller_request_operation_id: Uuid,
    pub controller_request_digest: String,
    pub controller_request_signature: String,
    pub pairing_incarnation: u64,
    pub protocol_min: u16,
    pub protocol_max: u16,
    pub created_at_unix_ms: i64,
    pub expires_at_unix_ms: i64,
    pub issuer_key_id: String,
    pub issuer_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairAuthorizationV1 {
    pub artifact_version: u16,
    pub subject_account_id: Uuid,
    pub subject_project_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub direction: MaplePairingDirection,
    pub execution_target_id: Uuid,
    pub controller: MapleDeviceClaimV1,
    pub host: MapleDeviceClaimV1,
    pub pairing_request_nonce: String,
    pub controller_request_operation_id: Uuid,
    pub controller_request_digest: String,
    pub controller_request_signature: String,
    pub request_ticket_digest: String,
    pub host_approval_operation_id: Uuid,
    pub host_approval_expected_pairing_revision: i64,
    pub host_approval_nonce: String,
    pub host_approval_digest: String,
    pub host_approval_signature: String,
    pub pairing_incarnation: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub protocol_min: u16,
    pub protocol_max: u16,
    pub approved_at_unix_ms: i64,
    pub issuer_key_id: String,
    pub issuer_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairRevocationV1 {
    pub artifact_version: u16,
    pub event_id: Uuid,
    pub subject_account_id: Uuid,
    pub subject_project_id: Uuid,
    pub recipient_host_registration_id: Uuid,
    pub issuer_sequence: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub direction: MaplePairingDirection,
    pub execution_target_id: Uuid,
    pub controller: MapleDeviceClaimV1,
    pub host: MapleDeviceClaimV1,
    pub pairing_incarnation: u64,
    pub pair_authorization_digest: String,
    pub revoked_by_registration_id: Uuid,
    pub revoked_by_role: MaplePairingRole,
    pub reason_code: String,
    pub revoked_at_unix_ms: i64,
    pub issuer_key_id: String,
    pub issuer_signature: String,
}

/// One issuer-signed command to clear all locally admitted remote authority for
/// one account/project/host-installation scope after a destructive reset.
///
/// `event_id` is unique per host obligation; `reset_id` is shared by every host
/// obligation created by the same account reset. Retained admission leaves are
/// intentionally absent from the public artifact: only their bounded count and
/// canonical set digest cross the wire.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MapleResetClearRequiredV1 {
    pub artifact_version: u16,
    pub event_id: Uuid,
    pub reset_id: Uuid,
    pub reset_generation: u64,
    pub cumulative_reset_count: u64,
    pub source_security_epoch: u64,
    pub security_epoch: u64,
    pub subject_account_id: Uuid,
    pub subject_project_id: Uuid,
    pub recipient_host_registration_id: Uuid,
    pub host: MapleDeviceClaimV1,
    pub issuer_sequence: u64,
    pub source_revocation_stream_id: Uuid,
    pub source_revocation_stream_generation: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub clear_scope: MapleResetClearScopeV1,
    pub admission_count: u16,
    pub admission_set_digest: String,
    pub previous_reset_clear_event_id: Option<Uuid>,
    pub previous_instruction_material_digest: Option<String>,
    pub previous_chain_digest: Option<String>,
    pub reset_at_unix_ms: i64,
    pub instruction_material_digest: String,
    pub chain_digest: String,
    pub issuer_key_id: String,
    pub issuer_signature: String,
}

impl std::fmt::Debug for MapleResetClearRequiredV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MapleResetClearRequiredV1")
            .field("artifact_version", &self.artifact_version)
            .field("reset_generation", &self.reset_generation)
            .field("cumulative_reset_count", &self.cumulative_reset_count)
            .field("source_security_epoch", &self.source_security_epoch)
            .field("security_epoch", &self.security_epoch)
            .field("admission_count", &self.admission_count)
            .field(
                "has_previous_reset",
                &self.previous_reset_clear_event_id.is_some(),
            )
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event_type", content = "event")]
pub enum MapleRevocationStreamEventV1 {
    #[serde(rename = "pair_revocation")]
    PairRevocation(MaplePairRevocationV1),
    #[serde(rename = "reset_clear_required")]
    ResetClearRequired(MapleResetClearRequiredV1),
}

impl std::fmt::Debug for MapleRevocationStreamEventV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let event_type = match self {
            Self::PairRevocation(_) => "pair_revocation",
            Self::ResetClearRequired(_) => "reset_clear_required",
        };
        formatter
            .debug_struct("MapleRevocationStreamEventV1")
            .field("event_type", &event_type)
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingStatusV1 {
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub state: MaplePairingState,
    pub revision: i64,
    pub pairing_incarnation: u64,
    pub revocation_stream_id: Option<Uuid>,
    pub revocation_stream_generation: Option<u64>,
    pub direction: MaplePairingDirection,
    pub execution_target_id: Uuid,
    pub controller_registration_id: Uuid,
    pub host_registration_id: Uuid,
    pub created_at_unix_ms: i64,
    pub expires_at_unix_ms: i64,
    pub approved_at_unix_ms: Option<i64>,
    pub activated_at_unix_ms: Option<i64>,
    pub revoked_at_unix_ms: Option<i64>,
    pub request_ticket: Option<MaplePairRequestTicketV1>,
    pub pair_authorization: Option<MaplePairAuthorizationV1>,
    pub revocation: Option<MaplePairRevocationV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingMutationResponse {
    pub protocol_version: u16,
    pub operation_id: Uuid,
    pub pairing: MaplePairingStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ListMaplePairingsResponse {
    pub protocol_version: u16,
    pub query_id: Uuid,
    pub role: MaplePairingRole,
    pub pairings: Vec<MaplePairingStatusV1>,
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingStatusResponse {
    pub protocol_version: u16,
    pub query_id: Uuid,
    pub pairing: MaplePairingStatusV1,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ListMaplePairingRevocationsResponse {
    pub protocol_version: u16,
    pub query_id: Uuid,
    pub revocation_sync: MapleRevocationSyncV1,
    pub events: Vec<MapleRevocationStreamEventV1>,
    pub next_after_issuer_sequence: u64,
    pub has_more: bool,
}

impl std::fmt::Debug for ListMaplePairingRevocationsResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ListMaplePairingRevocationsResponse")
            .field("protocol_version", &self.protocol_version)
            .field("event_count", &self.events.len())
            .field("has_more", &self.has_more)
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AckMaplePairingRevocationResponse {
    pub protocol_version: u16,
    pub operation_id: Uuid,
    pub host_registration_id: Uuid,
    pub stream_checkpoint: MapleRevocationStreamCheckpointV1,
    pub event_id: Uuid,
    pub issuer_sequence: u64,
    pub last_acked_issuer_sequence: u64,
    pub accepted_at_unix_ms: i64,
}

impl std::fmt::Debug for AckMaplePairingRevocationResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AckMaplePairingRevocationResponse")
            .field("protocol_version", &self.protocol_version)
            .field("issuer_sequence", &self.issuer_sequence)
            .field(
                "last_acked_issuer_sequence",
                &self.last_acked_issuer_sequence,
            )
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MapleRevocationStreamCheckpointV1 {
    pub artifact_version: u16,
    pub subject_account_id: Uuid,
    pub subject_project_id: Uuid,
    pub host: MapleDeviceClaimV1,
    pub security_epoch: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub last_issued_issuer_sequence: u64,
    pub last_acked_issuer_sequence: u64,
    pub issuer_key_id: String,
    pub issuer_signature: String,
}

impl std::fmt::Debug for MapleRevocationStreamCheckpointV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MapleRevocationStreamCheckpointV1")
            .field("artifact_version", &self.artifact_version)
            .field("security_epoch", &self.security_epoch)
            .field(
                "last_issued_issuer_sequence",
                &self.last_issued_issuer_sequence,
            )
            .field(
                "last_acked_issuer_sequence",
                &self.last_acked_issuer_sequence,
            )
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MapleRevocationSyncV1 {
    pub security_epoch: u64,
    pub status: MapleRevocationSyncStatusV1,
    pub stream_checkpoint: MapleRevocationStreamCheckpointV1,
    pub reset_clear_instruction: Option<MapleResetClearRequiredV1>,
}

impl std::fmt::Debug for MapleRevocationSyncV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MapleRevocationSyncV1")
            .field("security_epoch", &self.security_epoch)
            .field("status", &self.status)
            .field(
                "has_reset_clear_instruction",
                &self.reset_clear_instruction.is_some(),
            )
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

/// Private reset inventory leaf shape used to reproduce the public aggregate
/// digest. The leaves are never members of a response DTO.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MapleResetClearAdmissionLeafV1 {
    pub pair_id: Uuid,
    pub pairing_incarnation: u64,
    pub pair_authorization_digest: [u8; 32],
}

impl std::fmt::Debug for MapleResetClearAdmissionLeafV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MapleResetClearAdmissionLeafV1")
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingIssuerKeyV1 {
    pub key_id: String,
    pub algorithm: MaplePairingIdentityAlgorithm,
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaplePairingIssuerKeySetV1 {
    pub version: u16,
    pub keys: Vec<MaplePairingIssuerKeyV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaplePairingIssuerKeyFingerprintV1 {
    pub key_id: String,
    pub algorithm: MaplePairingIdentityAlgorithm,
    pub public_key_digest: [u8; 32],
}

impl MaplePairingStatusV1 {
    pub fn validate_revocation_stream_shape(&self) -> Result<(), MaplePairingWireError> {
        let namespace = match (self.revocation_stream_id, self.revocation_stream_generation) {
            (Some(stream_id), Some(generation)) => {
                validate_revocation_stream(stream_id, generation)?;
                Some((stream_id, generation))
            }
            (None, None) => None,
            _ => {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_stream_shape",
                ));
            }
        };
        let namespace_is_required = matches!(
            self.state,
            MaplePairingState::AwaitingHostCommit
                | MaplePairingState::Active
                | MaplePairingState::Revoked
        );
        if namespace.is_some() != namespace_is_required {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_state",
            ));
        }
        if let (Some((stream_id, generation)), Some(authorization)) =
            (namespace, self.pair_authorization.as_ref())
        {
            if authorization.revocation_stream_id != stream_id
                || authorization.revocation_stream_generation != generation
            {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_stream_authorization_binding",
                ));
            }
        }
        if let (Some((stream_id, generation)), Some(revocation)) =
            (namespace, self.revocation.as_ref())
        {
            if revocation.revocation_stream_id != stream_id
                || revocation.revocation_stream_generation != generation
            {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_stream_event_binding",
                ));
            }
        }
        Ok(())
    }
}

fn validate_security_epoch(epoch: u64, field: &'static str) -> Result<(), MaplePairingWireError> {
    if epoch == 0 || epoch > i64::MAX as u64 {
        return Err(MaplePairingWireError::InvalidField(field));
    }
    Ok(())
}

pub fn reset_clear_admission_set_transcript(
    artifact_version: u16,
    leaves: &[MapleResetClearAdmissionLeafV1],
) -> Result<Vec<u8>, MaplePairingWireError> {
    if artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
        return Err(MaplePairingWireError::UnsupportedVersion);
    }
    let count: u16 = leaves
        .len()
        .try_into()
        .map_err(|_| MaplePairingWireError::InvalidField("admission_count"))?;
    if count > MAPLE_RESET_CLEAR_MAX_ADMISSIONS {
        return Err(MaplePairingWireError::InvalidField("admission_count"));
    }
    let mut canonical = leaves.to_vec();
    canonical.sort_unstable();
    if canonical.windows(2).any(|window| window[0] == window[1]) {
        return Err(MaplePairingWireError::InvalidField(
            "reset_clear_admission_duplicate",
        ));
    }
    let mut transcript = CanonicalBytes::new(RESET_CLEAR_ADMISSION_SET_DOMAIN);
    transcript.append_u16(artifact_version).append_u16(count);
    for leaf in canonical {
        validate_uuid(leaf.pair_id, "pair_id")?;
        validate_incarnation(leaf.pairing_incarnation)?;
        transcript
            .append_uuid(leaf.pair_id)
            .append_u64(leaf.pairing_incarnation)
            .append_bytes(&leaf.pair_authorization_digest);
    }
    Ok(transcript.into_bytes())
}

pub fn reset_clear_admission_set_digest(
    artifact_version: u16,
    leaves: &[MapleResetClearAdmissionLeafV1],
) -> Result<[u8; 32], MaplePairingWireError> {
    Ok(sha256_digest(&reset_clear_admission_set_transcript(
        artifact_version,
        leaves,
    )?))
}

#[derive(Clone, Copy)]
struct ResetClearPredecessor {
    event_id: Uuid,
    instruction_material_digest: [u8; 32],
    chain_digest: [u8; 32],
}

fn reset_clear_predecessor(
    instruction: &MapleResetClearRequiredV1,
) -> Result<Option<ResetClearPredecessor>, MaplePairingWireError> {
    match (
        instruction.previous_reset_clear_event_id,
        instruction.previous_instruction_material_digest.as_deref(),
        instruction.previous_chain_digest.as_deref(),
    ) {
        (None, None, None) => Ok(None),
        (Some(event_id), Some(material_digest), Some(chain_digest)) => {
            validate_uuid(event_id, "previous_reset_clear_event_id")?;
            Ok(Some(ResetClearPredecessor {
                event_id,
                instruction_material_digest: decode_standard_base64::<32>(
                    material_digest,
                    "previous_instruction_material_digest",
                )?,
                chain_digest: decode_standard_base64::<32>(chain_digest, "previous_chain_digest")?,
            }))
        }
        _ => Err(MaplePairingWireError::InvalidField(
            "reset_clear_predecessor_shape",
        )),
    }
}

fn validate_reset_clear_material_shape(
    instruction: &MapleResetClearRequiredV1,
) -> Result<(), MaplePairingWireError> {
    if instruction.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
        return Err(MaplePairingWireError::UnsupportedVersion);
    }
    validate_uuid(instruction.event_id, "event_id")?;
    validate_uuid(instruction.reset_id, "reset_id")?;
    if instruction.reset_generation == 0
        || instruction.reset_generation > i64::MAX as u64
        || instruction.cumulative_reset_count != instruction.reset_generation
    {
        return Err(MaplePairingWireError::InvalidField("reset_generation"));
    }
    validate_security_epoch(instruction.source_security_epoch, "source_security_epoch")?;
    validate_security_epoch(instruction.security_epoch, "security_epoch")?;
    if instruction.source_security_epoch.checked_add(1) != Some(instruction.security_epoch) {
        return Err(MaplePairingWireError::InvalidField("reset_security_epoch"));
    }
    validate_common_scope(
        instruction.subject_account_id,
        instruction.subject_project_id,
    )?;
    validate_uuid(
        instruction.recipient_host_registration_id,
        "recipient_host_registration_id",
    )?;
    instruction.host.validate()?;
    if instruction.recipient_host_registration_id != instruction.host.registration_id {
        return Err(MaplePairingWireError::InvalidField("reset_clear_recipient"));
    }
    if instruction.issuer_sequence != 1 {
        return Err(MaplePairingWireError::InvalidField("issuer_sequence"));
    }
    validate_revocation_stream(
        instruction.source_revocation_stream_id,
        instruction.source_revocation_stream_generation,
    )?;
    validate_revocation_stream(
        instruction.revocation_stream_id,
        instruction.revocation_stream_generation,
    )?;
    if instruction.source_revocation_stream_id == instruction.revocation_stream_id
        || instruction
            .source_revocation_stream_generation
            .checked_add(1)
            != Some(instruction.revocation_stream_generation)
    {
        return Err(MaplePairingWireError::InvalidField(
            "reset_revocation_stream",
        ));
    }
    if instruction.admission_count > MAPLE_RESET_CLEAR_MAX_ADMISSIONS {
        return Err(MaplePairingWireError::InvalidField("admission_count"));
    }
    decode_standard_base64::<32>(&instruction.admission_set_digest, "admission_set_digest")?;
    let predecessor = reset_clear_predecessor(instruction)?;
    if (instruction.reset_generation == 1) != predecessor.is_none() {
        return Err(MaplePairingWireError::InvalidField(
            "reset_clear_predecessor_generation",
        ));
    }
    if instruction.reset_at_unix_ms < 0 {
        return Err(MaplePairingWireError::InvalidField("reset_at_unix_ms"));
    }
    Ok(())
}

pub fn reset_clear_instruction_material_transcript(
    instruction: &MapleResetClearRequiredV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    validate_reset_clear_material_shape(instruction)?;
    let admission_set_digest =
        decode_standard_base64::<32>(&instruction.admission_set_digest, "admission_set_digest")?;
    let predecessor = reset_clear_predecessor(instruction)?;
    let mut transcript = CanonicalBytes::new(RESET_CLEAR_INSTRUCTION_MATERIAL_DOMAIN);
    transcript
        .append_u16(instruction.artifact_version)
        .append_uuid(instruction.event_id)
        .append_uuid(instruction.reset_id)
        .append_u64(instruction.reset_generation)
        .append_u64(instruction.cumulative_reset_count)
        .append_u64(instruction.source_security_epoch)
        .append_u64(instruction.security_epoch)
        .append_uuid(instruction.subject_account_id)
        .append_uuid(instruction.subject_project_id)
        .append_uuid(instruction.recipient_host_registration_id);
    append_device_claim(&mut transcript, &instruction.host)?;
    transcript
        .append_u64(instruction.issuer_sequence)
        .append_uuid(instruction.source_revocation_stream_id)
        .append_u64(instruction.source_revocation_stream_generation)
        .append_uuid(instruction.revocation_stream_id)
        .append_u64(instruction.revocation_stream_generation)
        .append_str(instruction.clear_scope.as_wire())
        .append_u16(instruction.admission_count)
        .append_bytes(&admission_set_digest)
        .append_bool(predecessor.is_some());
    if let Some(ResetClearPredecessor {
        event_id,
        instruction_material_digest: material_digest,
        chain_digest,
    }) = predecessor
    {
        transcript
            .append_uuid(event_id)
            .append_bytes(&material_digest)
            .append_bytes(&chain_digest);
    }
    transcript.append_i64(instruction.reset_at_unix_ms);
    Ok(transcript.into_bytes())
}

pub fn reset_clear_chain_transcript(
    instruction: &MapleResetClearRequiredV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    validate_reset_clear_material_shape(instruction)?;
    let material_digest = decode_standard_base64::<32>(
        &instruction.instruction_material_digest,
        "instruction_material_digest",
    )?;
    let predecessor = reset_clear_predecessor(instruction)?;
    let mut transcript = CanonicalBytes::new(RESET_CLEAR_CHAIN_DOMAIN);
    transcript
        .append_u16(instruction.artifact_version)
        .append_bool(predecessor.is_some());
    if let Some(ResetClearPredecessor {
        event_id,
        instruction_material_digest: previous_material_digest,
        chain_digest: previous_chain_digest,
    }) = predecessor
    {
        transcript
            .append_bytes(&previous_chain_digest)
            .append_uuid(event_id)
            .append_bytes(&previous_material_digest);
    }
    transcript
        .append_uuid(instruction.reset_id)
        .append_uuid(instruction.event_id)
        .append_u64(instruction.reset_generation)
        .append_bytes(&material_digest)
        .append_u64(instruction.cumulative_reset_count);
    Ok(transcript.into_bytes())
}

pub fn reset_clear_required_transcript(
    instruction: &MapleResetClearRequiredV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    instruction.validate_unsigned()?;
    let material_digest = decode_standard_base64::<32>(
        &instruction.instruction_material_digest,
        "instruction_material_digest",
    )?;
    let chain_digest = decode_standard_base64::<32>(&instruction.chain_digest, "chain_digest")?;
    // CanonicalBytes deliberately has no splice operation, so the signed
    // transcript replays the frozen material fields without their domain.
    let admission_set_digest =
        decode_standard_base64::<32>(&instruction.admission_set_digest, "admission_set_digest")?;
    let predecessor = reset_clear_predecessor(instruction)?;
    let mut transcript = CanonicalBytes::new(RESET_CLEAR_REQUIRED_DOMAIN);
    transcript
        .append_u16(instruction.artifact_version)
        .append_uuid(instruction.event_id)
        .append_uuid(instruction.reset_id)
        .append_u64(instruction.reset_generation)
        .append_u64(instruction.cumulative_reset_count)
        .append_u64(instruction.source_security_epoch)
        .append_u64(instruction.security_epoch)
        .append_uuid(instruction.subject_account_id)
        .append_uuid(instruction.subject_project_id)
        .append_uuid(instruction.recipient_host_registration_id);
    append_device_claim(&mut transcript, &instruction.host)?;
    transcript
        .append_u64(instruction.issuer_sequence)
        .append_uuid(instruction.source_revocation_stream_id)
        .append_u64(instruction.source_revocation_stream_generation)
        .append_uuid(instruction.revocation_stream_id)
        .append_u64(instruction.revocation_stream_generation)
        .append_str(instruction.clear_scope.as_wire())
        .append_u16(instruction.admission_count)
        .append_bytes(&admission_set_digest)
        .append_bool(predecessor.is_some());
    if let Some(ResetClearPredecessor {
        event_id,
        instruction_material_digest: previous_material_digest,
        chain_digest: previous_chain_digest,
    }) = predecessor
    {
        transcript
            .append_uuid(event_id)
            .append_bytes(&previous_material_digest)
            .append_bytes(&previous_chain_digest);
    }
    transcript
        .append_i64(instruction.reset_at_unix_ms)
        .append_bytes(&material_digest)
        .append_u64(instruction.cumulative_reset_count)
        .append_bytes(&chain_digest)
        .append_str(&instruction.issuer_key_id);
    Ok(transcript.into_bytes())
}

impl MapleResetClearRequiredV1 {
    fn validate_unsigned(&self) -> Result<(), MaplePairingWireError> {
        validate_reset_clear_material_shape(self)?;
        decode_standard_base64::<32>(
            &self.instruction_material_digest,
            "instruction_material_digest",
        )?;
        decode_standard_base64::<32>(&self.chain_digest, "chain_digest")?;
        validate_token(&self.issuer_key_id, "issuer_key_id")?;
        let expected_material = sha256_digest(&reset_clear_instruction_material_transcript(self)?);
        let material = decode_standard_base64::<32>(
            &self.instruction_material_digest,
            "instruction_material_digest",
        )?;
        if material != expected_material {
            return Err(MaplePairingWireError::InvalidField(
                "instruction_material_digest",
            ));
        }
        let expected_chain = sha256_digest(&reset_clear_chain_transcript(self)?);
        let chain = decode_standard_base64::<32>(&self.chain_digest, "chain_digest")?;
        if chain != expected_chain {
            return Err(MaplePairingWireError::InvalidField("chain_digest"));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        self.validate_unsigned()?;
        decode_standard_base64::<64>(&self.issuer_signature, "issuer_signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        reset_clear_required_transcript(self)
    }

    pub fn event_digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        keyset.verify(
            &self.issuer_key_id,
            &self.transcript()?,
            &self.issuer_signature,
        )
    }

    pub fn verify_against_checkpoint(
        &self,
        checkpoint: &MapleRevocationStreamCheckpointV1,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        self.verify(keyset)?;
        checkpoint.verify(keyset)?;
        if self.subject_account_id != checkpoint.subject_account_id
            || self.subject_project_id != checkpoint.subject_project_id
            || self.recipient_host_registration_id != checkpoint.host.registration_id
            || !device_claim_is_same_identity_at_or_before(&self.host, &checkpoint.host)
            || self.security_epoch != checkpoint.security_epoch
            || self.revocation_stream_id != checkpoint.revocation_stream_id
            || self.revocation_stream_generation != checkpoint.revocation_stream_generation
            || self.issuer_sequence != 1
            || checkpoint.last_issued_issuer_sequence != 1
            || checkpoint.last_acked_issuer_sequence != 0
        {
            return Err(MaplePairingWireError::InvalidField(
                "reset_clear_checkpoint_binding",
            ));
        }
        Ok(())
    }

    /// Verify a freshly discovered latest head. The issuer signature and
    /// recursive chain digest authenticate the cumulative proof; callers that
    /// possess the immediate predecessor should additionally call
    /// [`Self::verify_direct_successor`].
    pub fn verify_discovered_head_against_checkpoint(
        &self,
        checkpoint: &MapleRevocationStreamCheckpointV1,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        self.verify_against_checkpoint(checkpoint, keyset)
    }

    pub fn verify_direct_successor(
        &self,
        predecessor: &MapleResetClearRequiredV1,
        checkpoint: &MapleRevocationStreamCheckpointV1,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        self.verify_against_checkpoint(checkpoint, keyset)?;
        predecessor.verify(keyset)?;
        let predecessor_material = decode_standard_base64::<32>(
            &predecessor.instruction_material_digest,
            "instruction_material_digest",
        )?;
        let predecessor_chain =
            decode_standard_base64::<32>(&predecessor.chain_digest, "chain_digest")?;
        let linked_material = self
            .previous_instruction_material_digest
            .as_deref()
            .map(|value| {
                decode_standard_base64::<32>(value, "previous_instruction_material_digest")
            })
            .transpose()?;
        let linked_chain = self
            .previous_chain_digest
            .as_deref()
            .map(|value| decode_standard_base64::<32>(value, "previous_chain_digest"))
            .transpose()?;
        if self.previous_reset_clear_event_id != Some(predecessor.event_id)
            || linked_material != Some(predecessor_material)
            || linked_chain != Some(predecessor_chain)
            || predecessor.reset_generation.checked_add(1) != Some(self.reset_generation)
            || predecessor.cumulative_reset_count.checked_add(1)
                != Some(self.cumulative_reset_count)
            || predecessor.security_epoch != self.source_security_epoch
            || predecessor.subject_account_id != self.subject_account_id
            || predecessor.subject_project_id != self.subject_project_id
            || predecessor.recipient_host_registration_id != self.recipient_host_registration_id
            || predecessor.host != self.host
            || predecessor.revocation_stream_id != self.source_revocation_stream_id
            || predecessor.revocation_stream_generation != self.source_revocation_stream_generation
            || predecessor.reset_at_unix_ms > self.reset_at_unix_ms
            || predecessor.reset_id == self.reset_id
        {
            return Err(MaplePairingWireError::InvalidField(
                "reset_clear_successor_binding",
            ));
        }
        Ok(())
    }
}

impl MapleRevocationStreamEventV1 {
    pub fn issuer_sequence(&self) -> u64 {
        match self {
            Self::PairRevocation(event) => event.issuer_sequence,
            Self::ResetClearRequired(event) => event.issuer_sequence,
        }
    }

    pub fn event_digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        match self {
            Self::PairRevocation(event) => event.digest(),
            Self::ResetClearRequired(event) => event.event_digest(),
        }
    }

    pub fn verify_against_checkpoint(
        &self,
        checkpoint: &MapleRevocationStreamCheckpointV1,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        match self {
            Self::PairRevocation(event) => {
                event.verify(keyset)?;
                if event.subject_account_id != checkpoint.subject_account_id
                    || event.subject_project_id != checkpoint.subject_project_id
                    || !device_claim_is_same_identity_at_or_before(&event.host, &checkpoint.host)
                    || event.recipient_host_registration_id != checkpoint.host.registration_id
                    || event.revocation_stream_id != checkpoint.revocation_stream_id
                    || event.revocation_stream_generation != checkpoint.revocation_stream_generation
                {
                    return Err(MaplePairingWireError::InvalidField(
                        "revocation_stream_event_checkpoint_binding",
                    ));
                }
                Ok(())
            }
            Self::ResetClearRequired(event) => {
                event.verify(keyset)?;
                if event.subject_account_id != checkpoint.subject_account_id
                    || event.subject_project_id != checkpoint.subject_project_id
                    || event.recipient_host_registration_id != checkpoint.host.registration_id
                    || !device_claim_is_same_identity_at_or_before(&event.host, &checkpoint.host)
                    || event.security_epoch != checkpoint.security_epoch
                    || event.revocation_stream_id != checkpoint.revocation_stream_id
                    || event.revocation_stream_generation != checkpoint.revocation_stream_generation
                    || event.issuer_sequence != 1
                    || event.issuer_sequence > checkpoint.last_issued_issuer_sequence
                {
                    return Err(MaplePairingWireError::InvalidField(
                        "revocation_stream_event_checkpoint_binding",
                    ));
                }
                Ok(())
            }
        }
    }

    fn account_id(&self) -> Uuid {
        match self {
            Self::PairRevocation(event) => event.subject_account_id,
            Self::ResetClearRequired(event) => event.subject_account_id,
        }
    }

    fn project_id(&self) -> Uuid {
        match self {
            Self::PairRevocation(event) => event.subject_project_id,
            Self::ResetClearRequired(event) => event.subject_project_id,
        }
    }

    fn recipient_registration_id(&self) -> Uuid {
        match self {
            Self::PairRevocation(event) => event.recipient_host_registration_id,
            Self::ResetClearRequired(event) => event.recipient_host_registration_id,
        }
    }

    fn host(&self) -> &MapleDeviceClaimV1 {
        match self {
            Self::PairRevocation(event) => &event.host,
            Self::ResetClearRequired(event) => &event.host,
        }
    }

    fn stream_id(&self) -> Uuid {
        match self {
            Self::PairRevocation(event) => event.revocation_stream_id,
            Self::ResetClearRequired(event) => event.revocation_stream_id,
        }
    }

    fn stream_generation(&self) -> u64 {
        match self {
            Self::PairRevocation(event) => event.revocation_stream_generation,
            Self::ResetClearRequired(event) => event.revocation_stream_generation,
        }
    }
}

impl MapleRevocationSyncV1 {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_security_epoch(self.security_epoch, "security_epoch")?;
        self.stream_checkpoint.validate()?;
        if self.security_epoch != self.stream_checkpoint.security_epoch {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_sync_security_epoch",
            ));
        }
        match (self.status, self.reset_clear_instruction.as_ref()) {
            (MapleRevocationSyncStatusV1::Ready, None)
                if self.stream_checkpoint.last_issued_issuer_sequence
                    == self.stream_checkpoint.last_acked_issuer_sequence => {}
            (MapleRevocationSyncStatusV1::RevocationsPending, None)
                if self.stream_checkpoint.last_issued_issuer_sequence
                    > self.stream_checkpoint.last_acked_issuer_sequence => {}
            (MapleRevocationSyncStatusV1::ResetClearRequired, Some(instruction))
                if self.stream_checkpoint.last_issued_issuer_sequence == 1
                    && self.stream_checkpoint.last_acked_issuer_sequence == 0
                    && instruction.security_epoch == self.security_epoch =>
            {
                instruction.validate()?;
            }
            _ => {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_sync_status",
                ));
            }
        }
        Ok(())
    }

    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        self.stream_checkpoint.verify(keyset)?;
        if let Some(instruction) = self.reset_clear_instruction.as_ref() {
            instruction.verify_against_checkpoint(&self.stream_checkpoint, keyset)?;
        }
        Ok(())
    }

    pub fn verify_against_registration(
        &self,
        account_id: Uuid,
        project_id: Uuid,
        registration_id: Uuid,
        security_epoch: u64,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        self.verify(keyset)?;
        if self.security_epoch != security_epoch
            || self.stream_checkpoint.subject_account_id != account_id
            || self.stream_checkpoint.subject_project_id != project_id
            || self.stream_checkpoint.host.registration_id != registration_id
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_sync_registration_binding",
            ));
        }
        Ok(())
    }

    pub fn status_for_checkpoint(
        security_epoch: u64,
        stream_checkpoint: MapleRevocationStreamCheckpointV1,
        reset_clear_instruction: Option<MapleResetClearRequiredV1>,
    ) -> Result<Self, MaplePairingWireError> {
        let status = if reset_clear_instruction.is_some() {
            MapleRevocationSyncStatusV1::ResetClearRequired
        } else if stream_checkpoint.last_issued_issuer_sequence
            == stream_checkpoint.last_acked_issuer_sequence
        {
            MapleRevocationSyncStatusV1::Ready
        } else {
            MapleRevocationSyncStatusV1::RevocationsPending
        };
        let sync = Self {
            security_epoch,
            status,
            stream_checkpoint,
            reset_clear_instruction,
        };
        sync.validate()?;
        Ok(sync)
    }
}

impl MapleRevocationStreamCheckpointV1 {
    fn validate_unsigned(&self) -> Result<(), MaplePairingWireError> {
        if self.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_common_scope(self.subject_account_id, self.subject_project_id)?;
        self.host.validate()?;
        validate_security_epoch(self.security_epoch, "security_epoch")?;
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        if self.last_issued_issuer_sequence > i64::MAX as u64
            || self.last_acked_issuer_sequence > self.last_issued_issuer_sequence
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_checkpoint_sequence",
            ));
        }
        validate_token(&self.issuer_key_id, "issuer_key_id")?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        self.validate_unsigned()?;
        decode_standard_base64::<64>(&self.issuer_signature, "issuer_signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        revocation_stream_checkpoint_transcript(self)
    }

    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        keyset.verify(
            &self.issuer_key_id,
            &self.transcript()?,
            &self.issuer_signature,
        )
    }
}

pub fn revocation_stream_checkpoint_transcript(
    checkpoint: &MapleRevocationStreamCheckpointV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    checkpoint.validate_unsigned()?;
    let mut transcript = CanonicalBytes::new(REVOCATION_STREAM_CHECKPOINT_DOMAIN);
    transcript
        .append_u16(checkpoint.artifact_version)
        .append_uuid(checkpoint.subject_account_id)
        .append_uuid(checkpoint.subject_project_id);
    append_device_claim(&mut transcript, &checkpoint.host)?;
    transcript
        .append_u64(checkpoint.security_epoch)
        .append_uuid(checkpoint.revocation_stream_id)
        .append_u64(checkpoint.revocation_stream_generation)
        .append_u64(checkpoint.last_issued_issuer_sequence)
        .append_u64(checkpoint.last_acked_issuer_sequence)
        .append_str(&checkpoint.issuer_key_id);
    Ok(transcript.into_bytes())
}

impl ListMaplePairingRevocationsResponse {
    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        if self.protocol_version != MAPLE_PAIRING_PROTOCOL_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_uuid(self.query_id, "query_id")?;
        self.revocation_sync.verify(keyset)?;
        let checkpoint = &self.revocation_sync.stream_checkpoint;
        if self.next_after_issuer_sequence > checkpoint.last_issued_issuer_sequence
            || self.has_more
                != (self.next_after_issuer_sequence < checkpoint.last_issued_issuer_sequence)
        {
            return Err(MaplePairingWireError::InvalidField(
                "next_after_issuer_sequence",
            ));
        }
        let mut previous_sequence = None;
        for event in &self.events {
            event.verify_against_checkpoint(checkpoint, keyset)?;
            if event.account_id() != checkpoint.subject_account_id
                || event.project_id() != checkpoint.subject_project_id
                || !device_claim_is_same_identity_at_or_before(event.host(), &checkpoint.host)
                || event.recipient_registration_id() != checkpoint.host.registration_id
                || event.stream_id() != checkpoint.revocation_stream_id
                || event.stream_generation() != checkpoint.revocation_stream_generation
                || previous_sequence.is_some_and(|previous| event.issuer_sequence() != previous + 1)
            {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_stream_page_binding",
                ));
            }
            previous_sequence = Some(event.issuer_sequence());
        }
        match self.revocation_sync.reset_clear_instruction.as_ref() {
            Some(instruction)
                if matches!(
                    self.events.first(),
                    Some(MapleRevocationStreamEventV1::ResetClearRequired(event))
                        if event == instruction
                ) => {}
            Some(_) => {
                return Err(MaplePairingWireError::InvalidField(
                    "reset_clear_stream_event",
                ));
            }
            None if self.events.iter().all(|event| {
                !matches!(event, MapleRevocationStreamEventV1::ResetClearRequired(_))
                    || event.issuer_sequence() <= checkpoint.last_acked_issuer_sequence
            }) => {}
            None => {
                return Err(MaplePairingWireError::InvalidField(
                    "reset_clear_stream_event",
                ));
            }
        }
        if let Some(instruction) = self.revocation_sync.reset_clear_instruction.as_ref() {
            if instruction.event_digest()?
                != self
                    .events
                    .first()
                    .ok_or(MaplePairingWireError::InvalidField(
                        "reset_clear_stream_event",
                    ))?
                    .event_digest()?
            {
                return Err(MaplePairingWireError::InvalidField(
                    "reset_clear_stream_event",
                ));
            }
        }
        if previous_sequence.is_some_and(|sequence| sequence != self.next_after_issuer_sequence) {
            return Err(MaplePairingWireError::InvalidField(
                "next_after_issuer_sequence",
            ));
        }
        Ok(())
    }

    pub fn verify_against_request(
        &self,
        request: &ListMaplePairingRevocationsRequest,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        request.validate()?;
        self.verify(keyset)?;
        if self.events.len() > usize::from(request.effective_limit()?) {
            return Err(MaplePairingWireError::InvalidField("events"));
        }
        if self.query_id != request.query_id
            || self.revocation_sync.stream_checkpoint.subject_account_id
                != request.asserted_account_id
            || self.revocation_sync.stream_checkpoint.subject_project_id
                != request.asserted_project_id
            || self.revocation_sync.stream_checkpoint.host.registration_id
                != request.host_registration_id
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_response_request_binding",
            ));
        }
        let discovery = request.revocation_stream_id.is_nil()
            && request.revocation_stream_generation == 0
            && request.after_issuer_sequence == 0;
        if !discovery
            && (self.revocation_sync.stream_checkpoint.revocation_stream_id
                != request.revocation_stream_id
                || self
                    .revocation_sync
                    .stream_checkpoint
                    .revocation_stream_generation
                    != request.revocation_stream_generation)
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_response_namespace",
            ));
        }
        if let Some(first) = self.events.first() {
            if request.after_issuer_sequence.checked_add(1) != Some(first.issuer_sequence()) {
                return Err(MaplePairingWireError::InvalidField(
                    "revocation_stream_response_cursor",
                ));
            }
        } else if self.next_after_issuer_sequence != request.after_issuer_sequence {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_response_cursor",
            ));
        }
        if self.events.is_empty()
            && request.after_issuer_sequence
                != self
                    .revocation_sync
                    .stream_checkpoint
                    .last_issued_issuer_sequence
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_response_cursor",
            ));
        }
        if request.after_issuer_sequence
            > self
                .revocation_sync
                .stream_checkpoint
                .last_acked_issuer_sequence
        {
            return Err(MaplePairingWireError::InvalidField("after_issuer_sequence"));
        }
        Ok(())
    }
}

impl AckMaplePairingRevocationResponse {
    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        if self.protocol_version != MAPLE_PAIRING_PROTOCOL_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        validate_uuid(self.event_id, "event_id")?;
        self.stream_checkpoint.verify(keyset)?;
        if self.issuer_sequence == 0
            || self.issuer_sequence > i64::MAX as u64
            || self.last_acked_issuer_sequence != self.issuer_sequence
            || self.stream_checkpoint.host.registration_id != self.host_registration_id
            || self.stream_checkpoint.last_acked_issuer_sequence != self.issuer_sequence
            || self.accepted_at_unix_ms < 0
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_ack_response_binding",
            ));
        }
        Ok(())
    }

    pub fn verify_against_request(
        &self,
        request: &AckMaplePairingRevocationRequest,
        keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<(), MaplePairingWireError> {
        request.validate()?;
        self.verify(keyset)?;
        if self.operation_id != request.operation_id
            || self.host_registration_id != request.host_registration_id
            || self.event_id != request.event_id
            || self.issuer_sequence != request.issuer_sequence
            || self.stream_checkpoint.subject_account_id != request.asserted_account_id
            || self.stream_checkpoint.subject_project_id != request.asserted_project_id
            || self.stream_checkpoint.revocation_stream_id != request.revocation_stream_id
            || self.stream_checkpoint.revocation_stream_generation
                != request.revocation_stream_generation
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_ack_response_request_binding",
            ));
        }
        Ok(())
    }
}

fn device_claim_is_same_identity_at_or_before(
    event_claim: &MapleDeviceClaimV1,
    current_claim: &MapleDeviceClaimV1,
) -> bool {
    event_claim.registration_id == current_claim.registration_id
        && event_claim.device_id == current_claim.device_id
        && event_claim.installation_id == current_claim.installation_id
        && event_claim.identity_algorithm == current_claim.identity_algorithm
        && event_claim.identity_public_key == current_claim.identity_public_key
        && event_claim.endpoint_id == current_claim.endpoint_id
        && event_claim.endpoint_epoch <= current_claim.endpoint_epoch
}

pub trait MaplePairingIssuer: Send + Sync {
    fn key_id(&self) -> &str;
    fn public_key_bytes(&self) -> [u8; 32];
    fn sign(&self, transcript: &[u8]) -> Result<[u8; 64], MaplePairingWireError>;
}

#[allow(dead_code)] // Concrete in-process POC/test signer; production injects an attested signer.
pub struct Ed25519MaplePairingIssuer {
    key_id: String,
    signing_key: SigningKey,
}

#[allow(dead_code)]
impl Ed25519MaplePairingIssuer {
    pub fn new(key_id: String, signing_key: SigningKey) -> Result<Self, MaplePairingWireError> {
        validate_token(&key_id, "issuer_key_id")?;
        Ok(Self {
            key_id,
            signing_key,
        })
    }

    pub fn public_key_entry(&self) -> MaplePairingIssuerKeyV1 {
        MaplePairingIssuerKeyV1 {
            key_id: self.key_id.clone(),
            algorithm: MaplePairingIdentityAlgorithm::Ed25519,
            public_key: STANDARD.encode(self.signing_key.verifying_key().as_bytes()),
        }
    }
}

impl MaplePairingIssuer for Ed25519MaplePairingIssuer {
    fn key_id(&self) -> &str {
        &self.key_id
    }

    fn public_key_bytes(&self) -> [u8; 32] {
        *self.signing_key.verifying_key().as_bytes()
    }

    fn sign(&self, transcript: &[u8]) -> Result<[u8; 64], MaplePairingWireError> {
        Ok(self.signing_key.sign(transcript).to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedUnexpiredPairRequestTicket {
    ticket: MaplePairRequestTicketV1,
    verified_at_unix_ms: i64,
}

impl VerifiedUnexpiredPairRequestTicket {
    pub fn as_ticket(&self) -> &MaplePairRequestTicketV1 {
        &self.ticket
    }

    #[allow(dead_code)] // Useful at SDK ownership boundaries; backend borrows the wrapper.
    pub fn into_ticket(self) -> MaplePairRequestTicketV1 {
        self.ticket
    }
}

fn validate_versions(
    protocol_version: u16,
    transcript_version: u16,
) -> Result<(), MaplePairingWireError> {
    if protocol_version != MAPLE_PAIRING_PROTOCOL_VERSION_V1
        || transcript_version != MAPLE_PAIRING_TRANSCRIPT_VERSION_V1
    {
        return Err(MaplePairingWireError::UnsupportedVersion);
    }
    Ok(())
}

fn validate_uuid(value: Uuid, field: &'static str) -> Result<(), MaplePairingWireError> {
    if value.is_nil() {
        return Err(MaplePairingWireError::InvalidField(field));
    }
    Ok(())
}

fn validate_protocol_range(minimum: u16, maximum: u16) -> Result<(), MaplePairingWireError> {
    if minimum == 0 || minimum > maximum {
        return Err(MaplePairingWireError::InvalidField("protocol_range"));
    }
    Ok(())
}

fn validate_positive_revision(revision: i64) -> Result<(), MaplePairingWireError> {
    if revision <= 0 || revision == i64::MAX {
        return Err(MaplePairingWireError::InvalidField(
            "expected_pairing_revision",
        ));
    }
    Ok(())
}

fn validate_incarnation(incarnation: u64) -> Result<(), MaplePairingWireError> {
    if incarnation == 0 || incarnation > i64::MAX as u64 {
        return Err(MaplePairingWireError::InvalidField("pairing_incarnation"));
    }
    Ok(())
}

fn validate_revocation_stream(
    stream_id: Uuid,
    generation: u64,
) -> Result<(), MaplePairingWireError> {
    if stream_id.is_nil() || generation == 0 || generation > i64::MAX as u64 {
        return Err(MaplePairingWireError::InvalidField("revocation_stream"));
    }
    Ok(())
}

fn validate_token(value: &str, field: &'static str) -> Result<(), MaplePairingWireError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(MaplePairingWireError::InvalidField(field));
    }
    Ok(())
}

fn decode_standard_base64<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], MaplePairingWireError> {
    if !value.is_ascii() {
        return Err(MaplePairingWireError::NonCanonicalEncoding(field));
    }
    let decoded = STANDARD
        .decode(value)
        .map_err(|_| MaplePairingWireError::NonCanonicalEncoding(field))?;
    if STANDARD.encode(&decoded) != value {
        return Err(MaplePairingWireError::NonCanonicalEncoding(field));
    }
    decoded
        .try_into()
        .map_err(|_| MaplePairingWireError::InvalidField(field))
}

fn decode_endpoint_id(value: &str, field: &'static str) -> Result<[u8; 32], MaplePairingWireError> {
    if value.len() != 64 || !value.is_ascii() {
        return Err(MaplePairingWireError::NonCanonicalEncoding(field));
    }
    let decoded =
        hex::decode(value).map_err(|_| MaplePairingWireError::NonCanonicalEncoding(field))?;
    if hex::encode(&decoded) != value {
        return Err(MaplePairingWireError::NonCanonicalEncoding(field));
    }
    let decoded: [u8; 32] = decoded
        .try_into()
        .map_err(|_| MaplePairingWireError::InvalidField(field))?;
    VerifyingKey::from_bytes(&decoded).map_err(|_| MaplePairingWireError::InvalidField(field))?;
    Ok(decoded)
}

fn validate_common_scope(account_id: Uuid, project_id: Uuid) -> Result<(), MaplePairingWireError> {
    validate_uuid(account_id, "asserted_account_id")?;
    validate_uuid(project_id, "asserted_project_id")
}

fn append_device_claim(
    transcript: &mut CanonicalBytes,
    claim: &MapleDeviceClaimV1,
) -> Result<(), MaplePairingWireError> {
    claim.validate()?;
    let public_key =
        decode_standard_base64::<32>(&claim.identity_public_key, "identity_public_key")?;
    let endpoint_id = decode_endpoint_id(&claim.endpoint_id, "endpoint_id")?;
    transcript
        .append_uuid(claim.registration_id)
        .append_uuid(claim.device_id)
        .append_uuid(claim.installation_id)
        .append_str(claim.identity_algorithm.as_wire())
        .append_bytes(&public_key)
        .append_bytes(&endpoint_id)
        .append_u64(claim.endpoint_epoch);
    Ok(())
}

impl MapleDeviceClaimV1 {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_uuid(self.registration_id, "registration_id")?;
        validate_uuid(self.device_id, "device_id")?;
        validate_uuid(self.installation_id, "installation_id")?;
        let public_key =
            decode_standard_base64::<32>(&self.identity_public_key, "identity_public_key")?;
        VerifyingKey::from_bytes(&public_key)
            .map_err(|_| MaplePairingWireError::InvalidField("identity_public_key"))?;
        let endpoint_id = decode_endpoint_id(&self.endpoint_id, "endpoint_id")?;
        if public_key != endpoint_id || self.endpoint_epoch > i64::MAX as u64 {
            return Err(MaplePairingWireError::InvalidField(
                "identity_public_key_endpoint_id_mismatch",
            ));
        }
        Ok(())
    }

    pub fn verifying_key_bytes(&self) -> Result<[u8; 32], MaplePairingWireError> {
        self.validate()?;
        decode_standard_base64::<32>(&self.identity_public_key, "identity_public_key")
    }
}

pub fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn verify_ed25519_signature(
    transcript: &[u8],
    signature_base64: &str,
    public_key: &[u8; 32],
) -> Result<(), MaplePairingWireError> {
    let signature = decode_standard_base64::<64>(signature_base64, "signature")?;
    let verifying_key = VerifyingKey::from_bytes(public_key)
        .map_err(|_| MaplePairingWireError::InvalidField("identity_public_key"))?;
    verifying_key
        .verify_strict(transcript, &Signature::from_bytes(&signature))
        .map_err(|_| MaplePairingWireError::InvalidSignature)
}

impl CreateMaplePairingRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(
            self.controller_registration_id,
            "controller_registration_id",
        )?;
        validate_uuid(self.controller_device_id, "controller_device_id")?;
        validate_uuid(
            self.controller_installation_id,
            "controller_installation_id",
        )?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        validate_uuid(self.host_device_id, "host_device_id")?;
        validate_uuid(self.host_installation_id, "host_installation_id")?;
        validate_uuid(self.execution_target_id, "execution_target_id")?;
        if self.controller_registration_id == self.host_registration_id
            || self.controller_device_id == self.host_device_id
            || self.controller_installation_id == self.host_installation_id
            || self.execution_target_id != self.host_registration_id
        {
            return Err(MaplePairingWireError::InvalidField("directed_pair"));
        }
        let controller_endpoint =
            decode_endpoint_id(&self.controller_endpoint_id, "controller_endpoint_id")?;
        let host_endpoint = decode_endpoint_id(&self.host_endpoint_id, "host_endpoint_id")?;
        if controller_endpoint == host_endpoint
            || self.controller_endpoint_epoch > i64::MAX as u64
            || self.host_endpoint_epoch > i64::MAX as u64
        {
            return Err(MaplePairingWireError::InvalidField("directed_pair"));
        }
        decode_standard_base64::<32>(&self.pairing_request_nonce, "pairing_request_nonce")?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        validate_protocol_range(self.protocol_min, self.protocol_max)
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        create_pairing_request_transcript(self)
    }

    #[allow(dead_code)]
    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn verify_signature(&self) -> Result<(), MaplePairingWireError> {
        let key = self.controller_identity_key_bytes()?;
        verify_ed25519_signature(&self.transcript()?, &self.signature, &key)
    }

    pub fn controller_identity_key_bytes(&self) -> Result<[u8; 32], MaplePairingWireError> {
        decode_endpoint_id(&self.controller_endpoint_id, "controller_endpoint_id")
    }

    pub fn host_identity_key_bytes(&self) -> Result<[u8; 32], MaplePairingWireError> {
        decode_endpoint_id(&self.host_endpoint_id, "host_endpoint_id")
    }
}

pub fn create_pairing_request_transcript(
    request: &CreateMaplePairingRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let controller_endpoint =
        decode_endpoint_id(&request.controller_endpoint_id, "controller_endpoint_id")?;
    let host_endpoint = decode_endpoint_id(&request.host_endpoint_id, "host_endpoint_id")?;
    let nonce =
        decode_standard_base64::<32>(&request.pairing_request_nonce, "pairing_request_nonce")?;
    let mut transcript = CanonicalBytes::new(CREATE_REQUEST_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.operation_id)
        .append_uuid(request.controller_registration_id)
        .append_uuid(request.controller_device_id)
        .append_uuid(request.controller_installation_id)
        .append_bytes(&controller_endpoint)
        .append_u64(request.controller_endpoint_epoch)
        .append_uuid(request.host_registration_id)
        .append_uuid(request.host_device_id)
        .append_uuid(request.host_installation_id)
        .append_bytes(&host_endpoint)
        .append_u64(request.host_endpoint_epoch)
        .append_str(request.direction.as_wire())
        .append_uuid(request.execution_target_id)
        .append_bytes(&nonce)
        .append_u16(request.protocol_min)
        .append_u16(request.protocol_max);
    Ok(transcript.into_bytes())
}

impl ListMaplePairingsRequest {
    pub fn effective_limit(&self) -> Result<u16, MaplePairingWireError> {
        let limit = self.limit.unwrap_or(MAPLE_PAIRING_DEFAULT_PAGE_SIZE);
        if limit == 0 || limit > MAPLE_PAIRING_MAX_PAGE_SIZE {
            return Err(MaplePairingWireError::InvalidField("limit"));
        }
        Ok(limit)
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.query_id, "query_id")?;
        validate_uuid(self.actor_registration_id, "actor_registration_id")?;
        if self.states.is_empty() || self.states.len() > 5 {
            return Err(MaplePairingWireError::InvalidField("states"));
        }
        let mut prior = None;
        for state in &self.states {
            let rank = state.canonical_rank();
            if prior.is_some_and(|previous| previous >= rank) {
                return Err(MaplePairingWireError::InvalidField("states"));
            }
            prior = Some(rank);
        }
        if self.cursor.as_ref().is_some_and(|cursor| {
            cursor.is_empty() || cursor.len() > MAPLE_PAIRING_MAX_CURSOR_BYTES || !cursor.is_ascii()
        }) {
            return Err(MaplePairingWireError::InvalidField("cursor"));
        }
        self.effective_limit()?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        list_pairings_request_transcript(self)
    }
}

pub fn list_pairings_request_transcript(
    request: &ListMaplePairingsRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let mut transcript = CanonicalBytes::new(LIST_PAIRINGS_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.query_id)
        .append_uuid(request.actor_registration_id)
        .append_str(request.role.as_wire())
        .append_u16(request.states.len() as u16);
    for state in &request.states {
        transcript.append_str(state.as_wire());
    }
    transcript.append_bool(request.cursor.is_some());
    if let Some(cursor) = &request.cursor {
        transcript.append_str(cursor);
    }
    transcript.append_u16(request.effective_limit()?);
    Ok(transcript.into_bytes())
}

impl MaplePairingStatusRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.query_id, "query_id")?;
        validate_uuid(self.actor_registration_id, "actor_registration_id")?;
        validate_uuid(self.pair_id, "pair_id")?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        pairing_status_request_transcript(self)
    }
}

pub fn pairing_status_request_transcript(
    request: &MaplePairingStatusRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let mut transcript = CanonicalBytes::new(PAIRING_STATUS_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.query_id)
        .append_uuid(request.actor_registration_id)
        .append_uuid(request.pair_id);
    Ok(transcript.into_bytes())
}

impl ConfirmMaplePairingRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        validate_uuid(self.pairing_request_id, "pairing_request_id")?;
        validate_uuid(self.pair_id, "pair_id")?;
        validate_positive_revision(self.expected_pairing_revision)?;
        validate_incarnation(self.pairing_incarnation)?;
        decode_standard_base64::<32>(&self.pair_authorization_digest, "pair_authorization_digest")?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        confirm_pairing_request_transcript(self)
    }

    #[allow(dead_code)]
    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    #[allow(dead_code)]
    pub fn verify_signature(&self, host_key: &[u8; 32]) -> Result<(), MaplePairingWireError> {
        verify_ed25519_signature(&self.transcript()?, &self.signature, host_key)
    }
}

pub fn confirm_pairing_request_transcript(
    request: &ConfirmMaplePairingRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let authorization_digest = decode_standard_base64::<32>(
        &request.pair_authorization_digest,
        "pair_authorization_digest",
    )?;
    let mut transcript = CanonicalBytes::new(CONFIRM_PAIRING_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.operation_id)
        .append_uuid(request.host_registration_id)
        .append_uuid(request.pairing_request_id)
        .append_uuid(request.pair_id)
        .append_i64(request.expected_pairing_revision)
        .append_u64(request.pairing_incarnation)
        .append_bytes(&authorization_digest);
    Ok(transcript.into_bytes())
}

impl ApproveMaplePairingRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        validate_uuid(self.pairing_request_id, "pairing_request_id")?;
        validate_uuid(self.pair_id, "pair_id")?;
        validate_positive_revision(self.expected_pairing_revision)?;
        validate_incarnation(self.pairing_incarnation)?;
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        decode_standard_base64::<32>(&self.request_ticket_digest, "request_ticket_digest")?;
        decode_standard_base64::<32>(&self.host_approval_nonce, "host_approval_nonce")?;
        validate_protocol_range(self.approved_protocol_min, self.approved_protocol_max)?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        approve_pairing_request_transcript(self)
    }

    #[allow(dead_code)]
    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn verify_signature(&self, host_key: &[u8; 32]) -> Result<(), MaplePairingWireError> {
        verify_ed25519_signature(&self.transcript()?, &self.signature, host_key)
    }
}

pub fn approve_pairing_request_transcript(
    request: &ApproveMaplePairingRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let ticket_digest =
        decode_standard_base64::<32>(&request.request_ticket_digest, "request_ticket_digest")?;
    let approval_nonce =
        decode_standard_base64::<32>(&request.host_approval_nonce, "host_approval_nonce")?;
    let mut transcript = CanonicalBytes::new(APPROVE_PAIRING_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.operation_id)
        .append_uuid(request.host_registration_id)
        .append_uuid(request.pairing_request_id)
        .append_uuid(request.pair_id)
        .append_i64(request.expected_pairing_revision)
        .append_u64(request.pairing_incarnation)
        .append_uuid(request.revocation_stream_id)
        .append_u64(request.revocation_stream_generation)
        .append_bytes(&ticket_digest)
        .append_bytes(&approval_nonce)
        .append_u16(request.approved_protocol_min)
        .append_u16(request.approved_protocol_max);
    Ok(transcript.into_bytes())
}

impl RevokeMaplePairingRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(self.actor_registration_id, "actor_registration_id")?;
        validate_uuid(self.pairing_request_id, "pairing_request_id")?;
        validate_uuid(self.pair_id, "pair_id")?;
        validate_positive_revision(self.expected_pairing_revision)?;
        validate_incarnation(self.pairing_incarnation)?;
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        validate_token(&self.reason_code, "reason_code")?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        revoke_pairing_request_transcript(self)
    }

    #[allow(dead_code)]
    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    #[allow(dead_code)]
    pub fn verify_signature(&self, actor_key: &[u8; 32]) -> Result<(), MaplePairingWireError> {
        verify_ed25519_signature(&self.transcript()?, &self.signature, actor_key)
    }
}

pub fn revoke_pairing_request_transcript(
    request: &RevokeMaplePairingRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let mut transcript = CanonicalBytes::new(REVOKE_PAIRING_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.operation_id)
        .append_uuid(request.actor_registration_id)
        .append_str(request.actor_role.as_wire())
        .append_uuid(request.pairing_request_id)
        .append_uuid(request.pair_id)
        .append_i64(request.expected_pairing_revision)
        .append_u64(request.pairing_incarnation)
        .append_uuid(request.revocation_stream_id)
        .append_u64(request.revocation_stream_generation)
        .append_str(&request.reason_code);
    Ok(transcript.into_bytes())
}

impl ListMaplePairingRevocationsRequest {
    pub fn effective_limit(&self) -> Result<u16, MaplePairingWireError> {
        let limit = self.limit.unwrap_or(MAPLE_PAIRING_DEFAULT_PAGE_SIZE);
        if limit == 0 || limit > MAPLE_PAIRING_MAX_PAGE_SIZE {
            return Err(MaplePairingWireError::InvalidField("limit"));
        }
        Ok(limit)
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.query_id, "query_id")?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        let discovery = self.revocation_stream_id.is_nil()
            && self.revocation_stream_generation == 0
            && self.after_issuer_sequence == 0;
        if !discovery {
            validate_revocation_stream(
                self.revocation_stream_id,
                self.revocation_stream_generation,
            )?;
        }
        if (self.revocation_stream_id.is_nil() || self.revocation_stream_generation == 0)
            && !discovery
        {
            return Err(MaplePairingWireError::InvalidField(
                "revocation_stream_discovery",
            ));
        }
        if self.after_issuer_sequence > i64::MAX as u64 {
            return Err(MaplePairingWireError::InvalidField("after_issuer_sequence"));
        }
        self.effective_limit()?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        list_pairing_revocations_request_transcript(self)
    }
}

pub fn list_pairing_revocations_request_transcript(
    request: &ListMaplePairingRevocationsRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let mut transcript = CanonicalBytes::new(LIST_REVOCATIONS_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.query_id)
        .append_uuid(request.host_registration_id)
        .append_uuid(request.revocation_stream_id)
        .append_u64(request.revocation_stream_generation)
        .append_u64(request.after_issuer_sequence)
        .append_u16(request.effective_limit()?);
    Ok(transcript.into_bytes())
}

impl AckMaplePairingRevocationRequest {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        validate_versions(self.protocol_version, self.transcript_version)?;
        validate_common_scope(self.asserted_account_id, self.asserted_project_id)?;
        validate_uuid(self.operation_id, "operation_id")?;
        validate_uuid(self.host_registration_id, "host_registration_id")?;
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        validate_uuid(self.event_id, "event_id")?;
        if self.issuer_sequence == 0
            || self.issuer_sequence > i64::MAX as u64
            || self.expected_previous_issuer_sequence > i64::MAX as u64
            || self.expected_previous_issuer_sequence.checked_add(1) != Some(self.issuer_sequence)
        {
            return Err(MaplePairingWireError::InvalidField("issuer_sequence"));
        }
        decode_standard_base64::<32>(&self.event_digest, "event_digest")?;
        decode_standard_base64::<64>(&self.signature, "signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        ack_pairing_revocation_request_transcript(self)
    }
}

pub fn ack_pairing_revocation_request_transcript(
    request: &AckMaplePairingRevocationRequest,
) -> Result<Vec<u8>, MaplePairingWireError> {
    request.validate()?;
    let event_digest = decode_standard_base64::<32>(&request.event_digest, "event_digest")?;
    let mut transcript = CanonicalBytes::new(ACK_REVOCATION_DOMAIN);
    transcript
        .append_u16(request.protocol_version)
        .append_u16(request.transcript_version)
        .append_uuid(request.asserted_account_id)
        .append_uuid(request.asserted_project_id)
        .append_uuid(request.operation_id)
        .append_uuid(request.host_registration_id)
        .append_uuid(request.revocation_stream_id)
        .append_u64(request.revocation_stream_generation)
        .append_uuid(request.event_id)
        .append_u64(request.issuer_sequence)
        .append_bytes(&event_digest)
        .append_u64(request.expected_previous_issuer_sequence);
    Ok(transcript.into_bytes())
}

#[allow(clippy::too_many_arguments)]
fn validate_artifact_pair(
    account_id: Uuid,
    project_id: Uuid,
    pairing_request_id: Uuid,
    pair_id: Uuid,
    execution_target_id: Uuid,
    controller: &MapleDeviceClaimV1,
    host: &MapleDeviceClaimV1,
    pairing_incarnation: u64,
) -> Result<(), MaplePairingWireError> {
    validate_common_scope(account_id, project_id)?;
    validate_uuid(pairing_request_id, "pairing_request_id")?;
    validate_uuid(pair_id, "pair_id")?;
    validate_uuid(execution_target_id, "execution_target_id")?;
    controller.validate()?;
    host.validate()?;
    validate_incarnation(pairing_incarnation)?;
    if controller.registration_id == host.registration_id
        || controller.device_id == host.device_id
        || controller.installation_id == host.installation_id
        || controller.identity_public_key == host.identity_public_key
        || controller.endpoint_id == host.endpoint_id
        || execution_target_id != host.registration_id
    {
        return Err(MaplePairingWireError::InvalidField("directed_pair"));
    }
    Ok(())
}

impl MaplePairRequestTicketV1 {
    fn validate_unsigned(&self) -> Result<(), MaplePairingWireError> {
        if self.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_artifact_pair(
            self.subject_account_id,
            self.subject_project_id,
            self.pairing_request_id,
            self.pair_id,
            self.execution_target_id,
            &self.controller,
            &self.host,
            self.pairing_incarnation,
        )?;
        decode_standard_base64::<32>(&self.pairing_request_nonce, "pairing_request_nonce")?;
        validate_uuid(
            self.controller_request_operation_id,
            "controller_request_operation_id",
        )?;
        decode_standard_base64::<32>(&self.controller_request_digest, "controller_request_digest")?;
        decode_standard_base64::<64>(
            &self.controller_request_signature,
            "controller_request_signature",
        )?;
        validate_protocol_range(self.protocol_min, self.protocol_max)?;
        if self.created_at_unix_ms < 0 || self.expires_at_unix_ms <= self.created_at_unix_ms {
            return Err(MaplePairingWireError::InvalidField("ticket_lifetime"));
        }
        if self.expires_at_unix_ms - self.created_at_unix_ms > MAPLE_PAIR_REQUEST_MAX_TTL_MS {
            return Err(MaplePairingWireError::TicketLifetimeTooLong);
        }
        validate_token(&self.issuer_key_id, "issuer_key_id")?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        self.validate_unsigned()?;
        decode_standard_base64::<64>(&self.issuer_signature, "issuer_signature")?;
        self.verify_controller_request()
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        pair_request_ticket_transcript(self)
    }

    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn controller_request(&self) -> CreateMaplePairingRequest {
        CreateMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: self.controller_request_operation_id,
            asserted_account_id: self.subject_account_id,
            asserted_project_id: self.subject_project_id,
            controller_registration_id: self.controller.registration_id,
            controller_device_id: self.controller.device_id,
            controller_installation_id: self.controller.installation_id,
            controller_endpoint_id: self.controller.endpoint_id.clone(),
            controller_endpoint_epoch: self.controller.endpoint_epoch,
            host_registration_id: self.host.registration_id,
            host_device_id: self.host.device_id,
            host_installation_id: self.host.installation_id,
            host_endpoint_id: self.host.endpoint_id.clone(),
            host_endpoint_epoch: self.host.endpoint_epoch,
            direction: self.direction,
            execution_target_id: self.execution_target_id,
            pairing_request_nonce: self.pairing_request_nonce.clone(),
            protocol_min: self.protocol_min,
            protocol_max: self.protocol_max,
            signature: self.controller_request_signature.clone(),
        }
    }

    pub fn verify_controller_request(&self) -> Result<(), MaplePairingWireError> {
        let request = self.controller_request();
        let digest = decode_standard_base64::<32>(
            &self.controller_request_digest,
            "controller_request_digest",
        )?;
        if request.digest()? != digest {
            return Err(MaplePairingWireError::InvalidField(
                "controller_request_digest",
            ));
        }
        request.verify_signature()
    }

    pub fn verify_unexpired(
        &self,
        keyset: &MaplePairingIssuerKeySetV1,
        trusted_now_unix_ms: i64,
        allowed_clock_skew_ms: i64,
    ) -> Result<VerifiedUnexpiredPairRequestTicket, MaplePairingWireError> {
        if !(0..=MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS).contains(&allowed_clock_skew_ms) {
            return Err(MaplePairingWireError::ClockSkewOutOfRange);
        }
        self.validate()?;
        keyset.verify(
            &self.issuer_key_id,
            &self.transcript()?,
            &self.issuer_signature,
        )?;
        let latest_allowed_creation = trusted_now_unix_ms
            .checked_add(allowed_clock_skew_ms)
            .ok_or(MaplePairingWireError::TicketNotCurrentlyValid)?;
        let earliest_allowed_expiry = trusted_now_unix_ms
            .checked_sub(allowed_clock_skew_ms)
            .ok_or(MaplePairingWireError::TicketNotCurrentlyValid)?;
        if self.created_at_unix_ms > latest_allowed_creation
            || self.expires_at_unix_ms <= earliest_allowed_expiry
        {
            return Err(MaplePairingWireError::TicketNotCurrentlyValid);
        }
        Ok(VerifiedUnexpiredPairRequestTicket {
            ticket: self.clone(),
            verified_at_unix_ms: trusted_now_unix_ms,
        })
    }
}

pub fn pair_request_ticket_transcript(
    ticket: &MaplePairRequestTicketV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    ticket.validate_unsigned()?;
    let request_nonce =
        decode_standard_base64::<32>(&ticket.pairing_request_nonce, "pairing_request_nonce")?;
    let request_digest = decode_standard_base64::<32>(
        &ticket.controller_request_digest,
        "controller_request_digest",
    )?;
    let request_signature = decode_standard_base64::<64>(
        &ticket.controller_request_signature,
        "controller_request_signature",
    )?;
    let mut transcript = CanonicalBytes::new(REQUEST_TICKET_DOMAIN);
    transcript
        .append_u16(ticket.artifact_version)
        .append_uuid(ticket.subject_account_id)
        .append_uuid(ticket.subject_project_id)
        .append_uuid(ticket.pairing_request_id)
        .append_uuid(ticket.pair_id)
        .append_str(ticket.direction.as_wire())
        .append_uuid(ticket.execution_target_id);
    append_device_claim(&mut transcript, &ticket.controller)?;
    append_device_claim(&mut transcript, &ticket.host)?;
    transcript
        .append_bytes(&request_nonce)
        .append_uuid(ticket.controller_request_operation_id)
        .append_bytes(&request_digest)
        .append_bytes(&request_signature)
        .append_u64(ticket.pairing_incarnation)
        .append_u16(ticket.protocol_min)
        .append_u16(ticket.protocol_max)
        .append_i64(ticket.created_at_unix_ms)
        .append_i64(ticket.expires_at_unix_ms)
        .append_str(&ticket.issuer_key_id);
    Ok(transcript.into_bytes())
}

impl MaplePairAuthorizationV1 {
    fn validate_unsigned(&self) -> Result<(), MaplePairingWireError> {
        if self.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_artifact_pair(
            self.subject_account_id,
            self.subject_project_id,
            self.pairing_request_id,
            self.pair_id,
            self.execution_target_id,
            &self.controller,
            &self.host,
            self.pairing_incarnation,
        )?;
        decode_standard_base64::<32>(&self.pairing_request_nonce, "pairing_request_nonce")?;
        validate_uuid(
            self.controller_request_operation_id,
            "controller_request_operation_id",
        )?;
        decode_standard_base64::<32>(&self.controller_request_digest, "controller_request_digest")?;
        decode_standard_base64::<64>(
            &self.controller_request_signature,
            "controller_request_signature",
        )?;
        decode_standard_base64::<32>(&self.request_ticket_digest, "request_ticket_digest")?;
        validate_uuid(
            self.host_approval_operation_id,
            "host_approval_operation_id",
        )?;
        validate_positive_revision(self.host_approval_expected_pairing_revision)?;
        if self.host_approval_expected_pairing_revision != 1 {
            return Err(MaplePairingWireError::InvalidField(
                "host_approval_expected_pairing_revision",
            ));
        }
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        decode_standard_base64::<32>(&self.host_approval_nonce, "host_approval_nonce")?;
        decode_standard_base64::<32>(&self.host_approval_digest, "host_approval_digest")?;
        decode_standard_base64::<64>(&self.host_approval_signature, "host_approval_signature")?;
        validate_protocol_range(self.protocol_min, self.protocol_max)?;
        if self.approved_at_unix_ms < 0 {
            return Err(MaplePairingWireError::InvalidField("approved_at_unix_ms"));
        }
        validate_token(&self.issuer_key_id, "issuer_key_id")?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        self.validate_unsigned()?;
        decode_standard_base64::<64>(&self.issuer_signature, "issuer_signature")?;
        self.verify_embedded_requests()
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        pair_authorization_transcript(self)
    }

    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    fn controller_request(&self) -> CreateMaplePairingRequest {
        CreateMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: self.controller_request_operation_id,
            asserted_account_id: self.subject_account_id,
            asserted_project_id: self.subject_project_id,
            controller_registration_id: self.controller.registration_id,
            controller_device_id: self.controller.device_id,
            controller_installation_id: self.controller.installation_id,
            controller_endpoint_id: self.controller.endpoint_id.clone(),
            controller_endpoint_epoch: self.controller.endpoint_epoch,
            host_registration_id: self.host.registration_id,
            host_device_id: self.host.device_id,
            host_installation_id: self.host.installation_id,
            host_endpoint_id: self.host.endpoint_id.clone(),
            host_endpoint_epoch: self.host.endpoint_epoch,
            direction: self.direction,
            execution_target_id: self.execution_target_id,
            pairing_request_nonce: self.pairing_request_nonce.clone(),
            protocol_min: self.protocol_min,
            protocol_max: self.protocol_max,
            signature: self.controller_request_signature.clone(),
        }
    }

    pub fn host_approval_request(&self) -> ApproveMaplePairingRequest {
        ApproveMaplePairingRequest {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
            operation_id: self.host_approval_operation_id,
            asserted_account_id: self.subject_account_id,
            asserted_project_id: self.subject_project_id,
            host_registration_id: self.host.registration_id,
            pairing_request_id: self.pairing_request_id,
            pair_id: self.pair_id,
            expected_pairing_revision: self.host_approval_expected_pairing_revision,
            pairing_incarnation: self.pairing_incarnation,
            revocation_stream_id: self.revocation_stream_id,
            revocation_stream_generation: self.revocation_stream_generation,
            request_ticket_digest: self.request_ticket_digest.clone(),
            host_approval_nonce: self.host_approval_nonce.clone(),
            approved_protocol_min: self.protocol_min,
            approved_protocol_max: self.protocol_max,
            signature: self.host_approval_signature.clone(),
        }
    }

    pub fn verify_embedded_requests(&self) -> Result<(), MaplePairingWireError> {
        let controller_request = self.controller_request();
        let controller_digest = decode_standard_base64::<32>(
            &self.controller_request_digest,
            "controller_request_digest",
        )?;
        if controller_request.digest()? != controller_digest {
            return Err(MaplePairingWireError::InvalidField(
                "controller_request_digest",
            ));
        }
        controller_request.verify_signature()?;

        let approval = self.host_approval_request();
        let approval_digest =
            decode_standard_base64::<32>(&self.host_approval_digest, "host_approval_digest")?;
        if approval.digest()? != approval_digest {
            return Err(MaplePairingWireError::InvalidField("host_approval_digest"));
        }
        approval.verify_signature(&self.host.verifying_key_bytes()?)
    }

    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        keyset.verify(
            &self.issuer_key_id,
            &self.transcript()?,
            &self.issuer_signature,
        )
    }

    /// Binds an authorization to the exact issuer-signed request ticket. V1
    /// does not permit protocol-range narrowing during host approval.
    pub fn verify_against_ticket(
        &self,
        keyset: &MaplePairingIssuerKeySetV1,
        ticket: &VerifiedUnexpiredPairRequestTicket,
    ) -> Result<(), MaplePairingWireError> {
        self.verify(keyset)?;
        let verified_at_unix_ms = ticket.verified_at_unix_ms;
        let ticket = ticket.as_ticket();
        let expected_ticket_digest =
            decode_standard_base64::<32>(&self.request_ticket_digest, "request_ticket_digest")?;
        if ticket.digest()? != expected_ticket_digest
            || self.subject_account_id != ticket.subject_account_id
            || self.subject_project_id != ticket.subject_project_id
            || self.pairing_request_id != ticket.pairing_request_id
            || self.pair_id != ticket.pair_id
            || self.direction != ticket.direction
            || self.execution_target_id != ticket.execution_target_id
            || self.controller != ticket.controller
            || self.host != ticket.host
            || self.pairing_request_nonce != ticket.pairing_request_nonce
            || self.controller_request_operation_id != ticket.controller_request_operation_id
            || self.controller_request_digest != ticket.controller_request_digest
            || self.controller_request_signature != ticket.controller_request_signature
            || self.pairing_incarnation != ticket.pairing_incarnation
            || self.protocol_min != ticket.protocol_min
            || self.protocol_max != ticket.protocol_max
            || self.approved_at_unix_ms < ticket.created_at_unix_ms
            || self.approved_at_unix_ms != verified_at_unix_ms
        {
            return Err(MaplePairingWireError::InvalidField(
                "pair_authorization_ticket_binding",
            ));
        }
        Ok(())
    }
}

pub fn pair_authorization_transcript(
    authorization: &MaplePairAuthorizationV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    authorization.validate_unsigned()?;
    let request_nonce = decode_standard_base64::<32>(
        &authorization.pairing_request_nonce,
        "pairing_request_nonce",
    )?;
    let controller_digest = decode_standard_base64::<32>(
        &authorization.controller_request_digest,
        "controller_request_digest",
    )?;
    let controller_signature = decode_standard_base64::<64>(
        &authorization.controller_request_signature,
        "controller_request_signature",
    )?;
    let ticket_digest = decode_standard_base64::<32>(
        &authorization.request_ticket_digest,
        "request_ticket_digest",
    )?;
    let approval_nonce =
        decode_standard_base64::<32>(&authorization.host_approval_nonce, "host_approval_nonce")?;
    let approval_digest =
        decode_standard_base64::<32>(&authorization.host_approval_digest, "host_approval_digest")?;
    let approval_signature = decode_standard_base64::<64>(
        &authorization.host_approval_signature,
        "host_approval_signature",
    )?;
    let mut transcript = CanonicalBytes::new(PAIR_AUTHORIZATION_DOMAIN);
    transcript
        .append_u16(authorization.artifact_version)
        .append_uuid(authorization.subject_account_id)
        .append_uuid(authorization.subject_project_id)
        .append_uuid(authorization.pairing_request_id)
        .append_uuid(authorization.pair_id)
        .append_str(authorization.direction.as_wire())
        .append_uuid(authorization.execution_target_id);
    append_device_claim(&mut transcript, &authorization.controller)?;
    append_device_claim(&mut transcript, &authorization.host)?;
    transcript
        .append_bytes(&request_nonce)
        .append_uuid(authorization.controller_request_operation_id)
        .append_bytes(&controller_digest)
        .append_bytes(&controller_signature)
        .append_bytes(&ticket_digest)
        .append_uuid(authorization.host_approval_operation_id)
        .append_i64(authorization.host_approval_expected_pairing_revision)
        .append_bytes(&approval_nonce)
        .append_bytes(&approval_digest)
        .append_bytes(&approval_signature)
        .append_u64(authorization.pairing_incarnation)
        .append_uuid(authorization.revocation_stream_id)
        .append_u64(authorization.revocation_stream_generation)
        .append_u16(authorization.protocol_min)
        .append_u16(authorization.protocol_max)
        .append_i64(authorization.approved_at_unix_ms)
        .append_str(&authorization.issuer_key_id);
    Ok(transcript.into_bytes())
}

impl MaplePairRevocationV1 {
    fn validate_unsigned(&self) -> Result<(), MaplePairingWireError> {
        if self.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1 {
            return Err(MaplePairingWireError::UnsupportedVersion);
        }
        validate_artifact_pair(
            self.subject_account_id,
            self.subject_project_id,
            self.pairing_request_id,
            self.pair_id,
            self.execution_target_id,
            &self.controller,
            &self.host,
            self.pairing_incarnation,
        )?;
        validate_uuid(self.event_id, "event_id")?;
        validate_uuid(
            self.recipient_host_registration_id,
            "recipient_host_registration_id",
        )?;
        if self.issuer_sequence == 0
            || self.issuer_sequence > i64::MAX as u64
            || self.recipient_host_registration_id != self.host.registration_id
        {
            return Err(MaplePairingWireError::InvalidField("revocation_recipient"));
        }
        validate_revocation_stream(self.revocation_stream_id, self.revocation_stream_generation)?;
        decode_standard_base64::<32>(&self.pair_authorization_digest, "pair_authorization_digest")?;
        validate_uuid(
            self.revoked_by_registration_id,
            "revoked_by_registration_id",
        )?;
        let expected_revoker = match self.revoked_by_role {
            MaplePairingRole::Controller => self.controller.registration_id,
            MaplePairingRole::Host => self.host.registration_id,
        };
        if self.revoked_by_registration_id != expected_revoker {
            return Err(MaplePairingWireError::InvalidField("revoked_by_role"));
        }
        validate_token(&self.reason_code, "reason_code")?;
        if self.revoked_at_unix_ms < 0 {
            return Err(MaplePairingWireError::InvalidField("revoked_at_unix_ms"));
        }
        validate_token(&self.issuer_key_id, "issuer_key_id")?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        self.validate_unsigned()?;
        decode_standard_base64::<64>(&self.issuer_signature, "issuer_signature")?;
        Ok(())
    }

    pub fn transcript(&self) -> Result<Vec<u8>, MaplePairingWireError> {
        pair_revocation_transcript(self)
    }

    pub fn digest(&self) -> Result<[u8; 32], MaplePairingWireError> {
        Ok(sha256_digest(&self.transcript()?))
    }

    pub fn verify(&self, keyset: &MaplePairingIssuerKeySetV1) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        keyset.verify(
            &self.issuer_key_id,
            &self.transcript()?,
            &self.issuer_signature,
        )
    }

    pub fn verify_against_authorization(
        &self,
        keyset: &MaplePairingIssuerKeySetV1,
        authorization: &MaplePairAuthorizationV1,
    ) -> Result<(), MaplePairingWireError> {
        self.verify(keyset)?;
        authorization.verify(keyset)?;
        let expected_authorization_digest = decode_standard_base64::<32>(
            &self.pair_authorization_digest,
            "pair_authorization_digest",
        )?;
        if authorization.digest()? != expected_authorization_digest
            || self.subject_account_id != authorization.subject_account_id
            || self.subject_project_id != authorization.subject_project_id
            || self.pairing_request_id != authorization.pairing_request_id
            || self.pair_id != authorization.pair_id
            || self.direction != authorization.direction
            || self.execution_target_id != authorization.execution_target_id
            || self.controller != authorization.controller
            || self.host != authorization.host
            || self.pairing_incarnation != authorization.pairing_incarnation
            || self.revocation_stream_id != authorization.revocation_stream_id
            || self.revocation_stream_generation != authorization.revocation_stream_generation
            || self.revoked_at_unix_ms < authorization.approved_at_unix_ms
        {
            return Err(MaplePairingWireError::InvalidField(
                "pair_revocation_authorization_binding",
            ));
        }
        Ok(())
    }
}

pub fn pair_revocation_transcript(
    revocation: &MaplePairRevocationV1,
) -> Result<Vec<u8>, MaplePairingWireError> {
    revocation.validate_unsigned()?;
    let authorization_digest = decode_standard_base64::<32>(
        &revocation.pair_authorization_digest,
        "pair_authorization_digest",
    )?;
    let mut transcript = CanonicalBytes::new(PAIR_REVOCATION_DOMAIN);
    transcript
        .append_u16(revocation.artifact_version)
        .append_uuid(revocation.event_id)
        .append_uuid(revocation.subject_account_id)
        .append_uuid(revocation.subject_project_id)
        .append_uuid(revocation.recipient_host_registration_id)
        .append_u64(revocation.issuer_sequence)
        .append_uuid(revocation.revocation_stream_id)
        .append_u64(revocation.revocation_stream_generation)
        .append_uuid(revocation.pairing_request_id)
        .append_uuid(revocation.pair_id)
        .append_str(revocation.direction.as_wire())
        .append_uuid(revocation.execution_target_id);
    append_device_claim(&mut transcript, &revocation.controller)?;
    append_device_claim(&mut transcript, &revocation.host)?;
    transcript
        .append_u64(revocation.pairing_incarnation)
        .append_bytes(&authorization_digest)
        .append_uuid(revocation.revoked_by_registration_id)
        .append_str(revocation.revoked_by_role.as_wire())
        .append_str(&revocation.reason_code)
        .append_i64(revocation.revoked_at_unix_ms)
        .append_str(&revocation.issuer_key_id);
    Ok(transcript.into_bytes())
}

impl MaplePairingIssuerKeySetV1 {
    pub fn validate(&self) -> Result<(), MaplePairingWireError> {
        if self.version != MAPLE_PAIRING_ARTIFACT_VERSION_V1
            || self.keys.is_empty()
            || self.keys.len() > MAPLE_PAIRING_MAX_ISSUER_KEYS
        {
            return Err(MaplePairingWireError::InvalidIssuerKeySet);
        }
        let mut previous_key_id: Option<&str> = None;
        let mut seen_public_keys = HashSet::with_capacity(self.keys.len());
        for key in &self.keys {
            validate_token(&key.key_id, "issuer_key_id")
                .map_err(|_| MaplePairingWireError::InvalidIssuerKeySet)?;
            if previous_key_id.is_some_and(|previous| previous >= key.key_id.as_str()) {
                return Err(MaplePairingWireError::InvalidIssuerKeySet);
            }
            previous_key_id = Some(&key.key_id);
            let public_key = decode_standard_base64::<32>(&key.public_key, "issuer_public_key")
                .map_err(|_| MaplePairingWireError::InvalidIssuerKeySet)?;
            VerifyingKey::from_bytes(&public_key)
                .map_err(|_| MaplePairingWireError::InvalidIssuerKeySet)?;
            if !seen_public_keys.insert(public_key) {
                return Err(MaplePairingWireError::InvalidIssuerKeySet);
            }
        }
        Ok(())
    }

    pub fn fingerprints(
        &self,
    ) -> Result<Vec<MaplePairingIssuerKeyFingerprintV1>, MaplePairingWireError> {
        self.validate()?;
        self.keys
            .iter()
            .map(|key| {
                let public_key =
                    decode_standard_base64::<32>(&key.public_key, "issuer_public_key")?;
                Ok(MaplePairingIssuerKeyFingerprintV1 {
                    key_id: key.key_id.clone(),
                    algorithm: key.algorithm,
                    public_key_digest: Sha256::digest(public_key).into(),
                })
            })
            .collect()
    }

    pub fn verify(
        &self,
        key_id: &str,
        transcript: &[u8],
        signature_base64: &str,
    ) -> Result<(), MaplePairingWireError> {
        self.validate()?;
        let key = self
            .keys
            .binary_search_by(|candidate| candidate.key_id.as_str().cmp(key_id))
            .ok()
            .map(|index| &self.keys[index])
            .ok_or(MaplePairingWireError::UnknownIssuer)?;
        let public_key = decode_standard_base64::<32>(&key.public_key, "issuer_public_key")?;
        verify_ed25519_signature(transcript, signature_base64, &public_key)
    }

    pub fn contains_key_id(&self, key_id: &str) -> Result<bool, MaplePairingWireError> {
        self.validate()?;
        validate_token(key_id, "issuer_key_id")?;
        Ok(self
            .keys
            .binary_search_by(|candidate| candidate.key_id.as_str().cmp(key_id))
            .is_ok())
    }

    pub fn contains_issuer(
        &self,
        issuer: &dyn MaplePairingIssuer,
    ) -> Result<bool, MaplePairingWireError> {
        self.validate()?;
        let Some(key) = self.keys.iter().find(|key| key.key_id == issuer.key_id()) else {
            return Ok(false);
        };
        let public_key = decode_standard_base64::<32>(&key.public_key, "issuer_public_key")?;
        Ok(public_key == issuer.public_key_bytes())
    }
}

pub fn sign_pair_request_ticket(
    mut ticket: MaplePairRequestTicketV1,
    issuer: &dyn MaplePairingIssuer,
) -> Result<MaplePairRequestTicketV1, MaplePairingWireError> {
    ticket.issuer_key_id = issuer.key_id().to_string();
    ticket.issuer_signature.clear();
    let signature = issuer.sign(&ticket.transcript()?)?;
    ticket.issuer_signature = STANDARD.encode(signature);
    ticket.validate()?;
    Ok(ticket)
}

pub fn sign_pair_authorization(
    mut authorization: MaplePairAuthorizationV1,
    issuer: &dyn MaplePairingIssuer,
) -> Result<MaplePairAuthorizationV1, MaplePairingWireError> {
    authorization.issuer_key_id = issuer.key_id().to_string();
    authorization.issuer_signature.clear();
    let signature = issuer.sign(&authorization.transcript()?)?;
    authorization.issuer_signature = STANDARD.encode(signature);
    authorization.validate()?;
    Ok(authorization)
}

pub fn sign_pair_revocation(
    mut revocation: MaplePairRevocationV1,
    issuer: &dyn MaplePairingIssuer,
) -> Result<MaplePairRevocationV1, MaplePairingWireError> {
    revocation.issuer_key_id = issuer.key_id().to_string();
    revocation.issuer_signature.clear();
    let signature = issuer.sign(&revocation.transcript()?)?;
    revocation.issuer_signature = STANDARD.encode(signature);
    revocation.validate()?;
    Ok(revocation)
}

pub fn sign_reset_clear_required(
    mut instruction: MapleResetClearRequiredV1,
    issuer: &dyn MaplePairingIssuer,
) -> Result<MapleResetClearRequiredV1, MaplePairingWireError> {
    instruction.instruction_material_digest.clear();
    instruction.chain_digest.clear();
    instruction.issuer_key_id.clear();
    instruction.issuer_signature.clear();

    let material_digest =
        sha256_digest(&reset_clear_instruction_material_transcript(&instruction)?);
    instruction.instruction_material_digest = STANDARD.encode(material_digest);
    let chain_digest = sha256_digest(&reset_clear_chain_transcript(&instruction)?);
    instruction.chain_digest = STANDARD.encode(chain_digest);
    instruction.issuer_key_id = issuer.key_id().to_string();
    let signature = issuer.sign(&instruction.transcript()?)?;
    instruction.issuer_signature = STANDARD.encode(signature);
    instruction.validate()?;
    Ok(instruction)
}

pub fn sign_revocation_stream_checkpoint(
    mut checkpoint: MapleRevocationStreamCheckpointV1,
    issuer: &dyn MaplePairingIssuer,
) -> Result<MapleRevocationStreamCheckpointV1, MaplePairingWireError> {
    checkpoint.issuer_key_id = issuer.key_id().to_string();
    checkpoint.issuer_signature.clear();
    let signature = issuer.sign(&checkpoint.transcript()?)?;
    checkpoint.issuer_signature = STANDARD.encode(signature);
    checkpoint.validate()?;
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn vectors() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/maple_pairing_v1_vectors.json"
        ))
        .expect("fixture is valid JSON")
    }

    fn decode_digest(value: &Value, key: &str) -> [u8; 32] {
        decode_standard_base64::<32>(
            value[key].as_str().expect("digest fixture string"),
            "fixture_digest",
        )
        .expect("canonical digest")
    }

    fn fixture_keyset(value: &Value) -> MaplePairingIssuerKeySetV1 {
        serde_json::from_value(value["issuer_keyset"].clone()).expect("keyset fixture")
    }

    fn fixture_issuer(value: &Value) -> Ed25519MaplePairingIssuer {
        let seed_bytes =
            hex::decode(value["test_private_seeds_hex"]["issuer"].as_str().unwrap()).unwrap();
        let seed: [u8; 32] = seed_bytes.try_into().unwrap();
        Ed25519MaplePairingIssuer::new(
            "maple-test-issuer-2026-08-13".to_string(),
            SigningKey::from_bytes(&seed),
        )
        .unwrap()
    }

    fn signed_genesis_reset_and_checkpoint(
        value: &Value,
        last_issued_issuer_sequence: u64,
        last_acked_issuer_sequence: u64,
    ) -> (MapleResetClearRequiredV1, MapleRevocationStreamCheckpointV1) {
        let ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        let authorization: MaplePairAuthorizationV1 =
            serde_json::from_value(value["pair_authorization"].clone()).unwrap();
        let issuer = fixture_issuer(value);
        let target_stream_id = Uuid::parse_str("87000000-0000-4000-8000-000000000001").unwrap();
        let instruction = sign_reset_clear_required(
            MapleResetClearRequiredV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                event_id: Uuid::parse_str("87000000-0000-4000-8000-000000000002").unwrap(),
                reset_id: Uuid::parse_str("87000000-0000-4000-8000-000000000003").unwrap(),
                reset_generation: 1,
                cumulative_reset_count: 1,
                source_security_epoch: 1,
                security_epoch: 2,
                subject_account_id: ticket.subject_account_id,
                subject_project_id: ticket.subject_project_id,
                recipient_host_registration_id: ticket.host.registration_id,
                host: ticket.host.clone(),
                issuer_sequence: 1,
                source_revocation_stream_id: authorization.revocation_stream_id,
                source_revocation_stream_generation: authorization.revocation_stream_generation,
                revocation_stream_id: target_stream_id,
                revocation_stream_generation: authorization.revocation_stream_generation + 1,
                clear_scope:
                    MapleResetClearScopeV1::AllPairAuthorizationsForAccountProjectHostInstallation,
                admission_count: 0,
                admission_set_digest: STANDARD.encode(
                    reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, &[])
                        .unwrap(),
                ),
                previous_reset_clear_event_id: None,
                previous_instruction_material_digest: None,
                previous_chain_digest: None,
                reset_at_unix_ms: authorization.approved_at_unix_ms + 1,
                instruction_material_digest: String::new(),
                chain_digest: String::new(),
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            &issuer,
        )
        .unwrap();
        let checkpoint = sign_revocation_stream_checkpoint(
            MapleRevocationStreamCheckpointV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                subject_account_id: instruction.subject_account_id,
                subject_project_id: instruction.subject_project_id,
                host: instruction.host.clone(),
                security_epoch: instruction.security_epoch,
                revocation_stream_id: instruction.revocation_stream_id,
                revocation_stream_generation: instruction.revocation_stream_generation,
                last_issued_issuer_sequence,
                last_acked_issuer_sequence,
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            &issuer,
        )
        .unwrap();
        (instruction, checkpoint)
    }

    #[test]
    fn database_backed_unsigned_values_stop_at_signed_bigint_max() {
        assert!(validate_incarnation(i64::MAX as u64).is_ok());
        assert!(validate_incarnation(i64::MAX as u64 + 1).is_err());

        let value = vectors();
        let mut claim: MapleDeviceClaimV1 =
            serde_json::from_value(value["request_ticket"]["controller"].clone())
                .expect("controller claim fixture");
        claim.endpoint_epoch = i64::MAX as u64 + 1;
        assert!(claim.validate().is_err());

        let mut list: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["list_revocations_request"].clone())
                .expect("list fixture");
        list.after_issuer_sequence = i64::MAX as u64 + 1;
        assert!(list.validate().is_err());

        let mut authorization: MaplePairAuthorizationV1 =
            serde_json::from_value(value["pair_authorization"].clone())
                .expect("authorization fixture");
        authorization.revocation_stream_generation = i64::MAX as u64 + 1;
        assert!(authorization.validate().is_err());

        let mut checkpoint: MapleRevocationStreamCheckpointV1 =
            serde_json::from_value(value["revocation_stream_checkpoint"].clone())
                .expect("checkpoint fixture");
        checkpoint.last_issued_issuer_sequence = i64::MAX as u64 + 1;
        assert!(checkpoint.validate().is_err());
    }

    #[test]
    fn frozen_v1_transcripts_digests_and_signatures_match_fixture() {
        let value = vectors();
        let create: CreateMaplePairingRequest =
            serde_json::from_value(value["create_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(create.transcript().unwrap()),
            value["create_request_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            create.digest().unwrap(),
            decode_digest(&value, "create_request_digest")
        );
        create.verify_signature().unwrap();

        let ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        assert_eq!(
            hex::encode(ticket.transcript().unwrap()),
            value["request_ticket_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            ticket.digest().unwrap(),
            decode_digest(&value, "request_ticket_digest")
        );

        let approval: ApproveMaplePairingRequest =
            serde_json::from_value(value["approval_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(approval.transcript().unwrap()),
            value["approval_request_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            approval.digest().unwrap(),
            decode_digest(&value, "approval_request_digest")
        );
        approval
            .verify_signature(&ticket.host.verifying_key_bytes().unwrap())
            .unwrap();

        let authorization: MaplePairAuthorizationV1 =
            serde_json::from_value(value["pair_authorization"].clone()).unwrap();
        assert_eq!(
            hex::encode(authorization.transcript().unwrap()),
            value["pair_authorization_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            authorization.digest().unwrap(),
            decode_digest(&value, "pair_authorization_digest")
        );

        let confirm: ConfirmMaplePairingRequest =
            serde_json::from_value(value["confirm_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(confirm.transcript().unwrap()),
            value["confirm_request_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            confirm.digest().unwrap(),
            decode_digest(&value, "confirm_request_digest")
        );
        verify_ed25519_signature(
            &confirm.transcript().unwrap(),
            &confirm.signature,
            &ticket.host.verifying_key_bytes().unwrap(),
        )
        .unwrap();

        let revocation: MaplePairRevocationV1 =
            serde_json::from_value(value["pair_revocation"].clone()).unwrap();
        assert_eq!(
            hex::encode(revocation.transcript().unwrap()),
            value["pair_revocation_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            revocation.digest().unwrap(),
            decode_digest(&value, "pair_revocation_digest")
        );

        let keyset = fixture_keyset(&value);
        let verified_ticket = ticket
            .verify_unexpired(&keyset, 1_786_579_260_000, 30_000)
            .unwrap();
        authorization
            .verify_against_ticket(&keyset, &verified_ticket)
            .unwrap();
        revocation
            .verify_against_authorization(&keyset, &authorization)
            .unwrap();

        let controller_key = ticket.controller.verifying_key_bytes().unwrap();
        let host_key = ticket.host.verifying_key_bytes().unwrap();
        let list_request: ListMaplePairingsRequest =
            serde_json::from_value(value["list_pairings_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(list_request.transcript().unwrap()),
            value["list_pairings_request_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            sha256_digest(&list_request.transcript().unwrap()),
            decode_digest(&value, "list_pairings_request_digest")
        );
        verify_ed25519_signature(
            &list_request.transcript().unwrap(),
            &list_request.signature,
            &controller_key,
        )
        .unwrap();

        let status_request: MaplePairingStatusRequest =
            serde_json::from_value(value["pairing_status_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(status_request.transcript().unwrap()),
            value["pairing_status_request_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            sha256_digest(&status_request.transcript().unwrap()),
            decode_digest(&value, "pairing_status_request_digest")
        );
        verify_ed25519_signature(
            &status_request.transcript().unwrap(),
            &status_request.signature,
            &controller_key,
        )
        .unwrap();

        let revoke_request: RevokeMaplePairingRequest =
            serde_json::from_value(value["revoke_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(revoke_request.transcript().unwrap()),
            value["revoke_request_transcript_hex"].as_str().unwrap()
        );
        assert_eq!(
            revoke_request.digest().unwrap(),
            decode_digest(&value, "revoke_request_digest")
        );
        revoke_request.verify_signature(&controller_key).unwrap();

        let revocation_list_request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["list_revocations_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(revocation_list_request.transcript().unwrap()),
            value["list_revocations_request_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            sha256_digest(&revocation_list_request.transcript().unwrap()),
            decode_digest(&value, "list_revocations_request_digest")
        );
        verify_ed25519_signature(
            &revocation_list_request.transcript().unwrap(),
            &revocation_list_request.signature,
            &host_key,
        )
        .unwrap();

        let ack_request: AckMaplePairingRevocationRequest =
            serde_json::from_value(value["ack_revocation_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(ack_request.transcript().unwrap()),
            value["ack_revocation_request_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            sha256_digest(&ack_request.transcript().unwrap()),
            decode_digest(&value, "ack_revocation_request_digest")
        );
        verify_ed25519_signature(
            &ack_request.transcript().unwrap(),
            &ack_request.signature,
            &host_key,
        )
        .unwrap();

        let checkpoint: MapleRevocationStreamCheckpointV1 =
            serde_json::from_value(value["revocation_stream_checkpoint"].clone()).unwrap();
        assert_eq!(
            hex::encode(checkpoint.transcript().unwrap()),
            value["revocation_stream_checkpoint_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            checkpoint.digest().unwrap(),
            decode_digest(&value, "revocation_stream_checkpoint_digest")
        );
        checkpoint.verify(&keyset).unwrap();

        let discovery_request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["discovery_list_revocations_request"].clone()).unwrap();
        assert_eq!(
            hex::encode(discovery_request.transcript().unwrap()),
            value["discovery_list_revocations_request_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            sha256_digest(&discovery_request.transcript().unwrap()),
            decode_digest(&value, "discovery_list_revocations_request_digest")
        );
        verify_ed25519_signature(
            &discovery_request.transcript().unwrap(),
            &discovery_request.signature,
            &host_key,
        )
        .unwrap();

        let discovery_checkpoint: MapleRevocationStreamCheckpointV1 =
            serde_json::from_value(value["discovery_revocation_stream_checkpoint"].clone())
                .unwrap();
        assert_eq!(
            hex::encode(discovery_checkpoint.transcript().unwrap()),
            value["discovery_revocation_stream_checkpoint_transcript_hex"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            discovery_checkpoint.digest().unwrap(),
            decode_digest(&value, "discovery_revocation_stream_checkpoint_digest")
        );
        discovery_checkpoint.verify(&keyset).unwrap();
    }

    #[test]
    fn v1_device_claim_requires_identity_key_to_equal_iroh_endpoint_id() {
        let value = vectors();
        let mut ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        ticket.controller.endpoint_id = ticket.host.endpoint_id.clone();
        assert_eq!(
            ticket.controller.validate(),
            Err(MaplePairingWireError::InvalidField(
                "identity_public_key_endpoint_id_mismatch"
            ))
        );
        assert!(ticket.transcript().is_err());

        let mut aliased_devices: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        aliased_devices.host.identity_public_key =
            aliased_devices.controller.identity_public_key.clone();
        aliased_devices.host.endpoint_id = aliased_devices.controller.endpoint_id.clone();
        assert_eq!(
            aliased_devices.transcript(),
            Err(MaplePairingWireError::InvalidField("directed_pair"))
        );
    }

    #[test]
    fn ticket_verification_is_time_bounded_and_returns_only_verified_wrapper() {
        let value = vectors();
        let ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        let keyset = fixture_keyset(&value);
        assert!(ticket
            .verify_unexpired(&keyset, ticket.created_at_unix_ms, 30_000)
            .is_ok());
        assert_eq!(
            ticket.verify_unexpired(&keyset, ticket.expires_at_unix_ms + 30_000, 30_000),
            Err(MaplePairingWireError::TicketNotCurrentlyValid)
        );
        assert_eq!(
            ticket.verify_unexpired(&keyset, ticket.created_at_unix_ms, 30_001),
            Err(MaplePairingWireError::ClockSkewOutOfRange)
        );

        let mut overlong = ticket;
        overlong.expires_at_unix_ms += 1;
        assert_eq!(
            overlong.verify_unexpired(&keyset, overlong.created_at_unix_ms, 30_000),
            Err(MaplePairingWireError::TicketLifetimeTooLong)
        );
    }

    #[test]
    fn keyset_is_sorted_unique_and_unknown_issuers_fail_closed() {
        let value = vectors();
        let ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        let mut keyset = fixture_keyset(&value);
        keyset.keys.push(keyset.keys[0].clone());
        assert_eq!(
            keyset.validate(),
            Err(MaplePairingWireError::InvalidIssuerKeySet)
        );

        let keyset = fixture_keyset(&value);
        assert!(keyset.contains_key_id(&ticket.issuer_key_id).unwrap());
        assert!(!keyset.contains_key_id("not-a-real-issuer").unwrap());
        assert_eq!(
            keyset.contains_key_id("INVALID KEY ID"),
            Err(MaplePairingWireError::InvalidField("issuer_key_id"))
        );
        assert_eq!(
            keyset.verify(
                "not-a-real-issuer",
                &ticket.transcript().unwrap(),
                &ticket.issuer_signature
            ),
            Err(MaplePairingWireError::UnknownIssuer)
        );
    }

    #[test]
    fn authorization_cannot_narrow_the_ticket_protocol_range() {
        let value = vectors();
        let ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        let keyset = fixture_keyset(&value);
        let verified_ticket = ticket
            .verify_unexpired(&keyset, 1_786_579_260_000, 30_000)
            .unwrap();
        let mut authorization: MaplePairAuthorizationV1 =
            serde_json::from_value(value["pair_authorization"].clone()).unwrap();
        authorization.protocol_max = 2;
        assert!(authorization
            .verify_against_ticket(&keyset, &verified_ticket)
            .is_err());
    }

    #[test]
    fn active_receipt_has_the_frozen_two_phase_state() {
        let value = vectors();
        let response: MaplePairingMutationResponse =
            serde_json::from_value(value["active_receipt"].clone()).unwrap();
        assert_eq!(response.pairing.state, MaplePairingState::Active);
        assert_eq!(response.pairing.revision, 3);
        assert!(response.pairing.request_ticket.is_some());
        assert!(response.pairing.pair_authorization.is_some());
        assert!(response.pairing.revocation.is_none());
        response.pairing.validate_revocation_stream_shape().unwrap();
    }

    #[test]
    fn paged_and_revocation_responses_have_frozen_progress_boundaries() {
        let value = vectors();
        let list: ListMaplePairingsResponse =
            serde_json::from_value(value["list_pairings_response"].clone()).unwrap();
        assert_eq!(list.role, MaplePairingRole::Controller);
        assert_eq!(list.pairings.len(), 1);
        assert_eq!(list.pairings[0].state, MaplePairingState::Pending);
        assert!(list.pairings[0].request_ticket.is_some());
        assert!(list.pairings[0].pair_authorization.is_none());
        assert!(list.has_more && list.next_cursor.is_some());
        list.pairings[0].validate_revocation_stream_shape().unwrap();

        let status: MaplePairingStatusResponse =
            serde_json::from_value(value["pairing_status_response"].clone()).unwrap();
        assert_eq!(status.pairing.pair_id, list.pairings[0].pair_id);
        assert_eq!(status.pairing.state, MaplePairingState::Active);
        assert!(status.pairing.pair_authorization.is_some());
        status.pairing.validate_revocation_stream_shape().unwrap();

        let keyset = fixture_keyset(&value);
        let request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["list_revocations_request"].clone()).unwrap();
        let revocations: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["list_revocations_response"].clone()).unwrap();
        assert_eq!(revocations.events.len(), 1);
        assert_eq!(revocations.events[0].issuer_sequence(), 13);
        assert_eq!(revocations.next_after_issuer_sequence, 13);
        assert!(!revocations.has_more);
        assert_eq!(
            revocations
                .revocation_sync
                .stream_checkpoint
                .host
                .endpoint_epoch,
            5
        );
        assert_eq!(revocations.events[0].host().endpoint_epoch, 4);
        revocations
            .verify_against_request(&request, &keyset)
            .unwrap();

        let ack_request: AckMaplePairingRevocationRequest =
            serde_json::from_value(value["ack_revocation_request"].clone()).unwrap();
        let ack: AckMaplePairingRevocationResponse =
            serde_json::from_value(value["ack_revocation_response"].clone()).unwrap();
        assert_eq!(ack.issuer_sequence, 13);
        assert_eq!(ack.last_acked_issuer_sequence, 13);
        ack.verify_against_request(&ack_request, &keyset).unwrap();

        let discovery_request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["discovery_list_revocations_request"].clone()).unwrap();
        let discovery: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["discovery_list_revocations_response"].clone()).unwrap();
        discovery
            .verify_against_request(&discovery_request, &keyset)
            .unwrap();

        let revoke: MaplePairingMutationResponse =
            serde_json::from_value(value["revoke_receipt"].clone()).unwrap();
        assert_eq!(revoke.pairing.state, MaplePairingState::Revoked);
        assert_eq!(revoke.pairing.revision, 4);
        revoke.pairing.validate_revocation_stream_shape().unwrap();
        assert_eq!(revoke.pairing.revocation.unwrap().issuer_sequence, 13);
    }

    #[test]
    fn revocation_stream_namespace_and_cursor_bindings_fail_closed() {
        let value = vectors();
        let keyset = fixture_keyset(&value);
        let request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["list_revocations_request"].clone()).unwrap();
        let response: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["list_revocations_response"].clone()).unwrap();

        let mut wrong_stream = response.clone();
        match &mut wrong_stream.events[0] {
            MapleRevocationStreamEventV1::PairRevocation(event) => {
                event.revocation_stream_id = Uuid::new_v4()
            }
            MapleRevocationStreamEventV1::ResetClearRequired(event) => {
                event.revocation_stream_id = Uuid::new_v4()
            }
        }
        assert!(wrong_stream.verify(&keyset).is_err());

        let mut wrong_generation = response.clone();
        match &mut wrong_generation.events[0] {
            MapleRevocationStreamEventV1::PairRevocation(event) => {
                event.revocation_stream_generation += 1
            }
            MapleRevocationStreamEventV1::ResetClearRequired(event) => {
                event.revocation_stream_generation += 1
            }
        }
        assert!(wrong_generation.verify(&keyset).is_err());

        let mut wrong_cursor = request.clone();
        wrong_cursor.after_issuer_sequence -= 1;
        assert!(response
            .verify_against_request(&wrong_cursor, &keyset)
            .is_err());

        let mut wrong_progress = response.clone();
        wrong_progress.has_more = true;
        assert!(wrong_progress.verify(&keyset).is_err());

        let discovery: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["discovery_list_revocations_request"].clone()).unwrap();
        assert!(discovery.validate().is_ok());
        let mut partial_sentinel = discovery.clone();
        partial_sentinel.after_issuer_sequence = 1;
        assert!(partial_sentinel.validate().is_err());
        let mut partial_sentinel = discovery;
        partial_sentinel.revocation_stream_generation = 1;
        assert!(partial_sentinel.validate().is_err());
    }

    #[test]
    fn deterministic_issuer_signer_reproduces_every_artifact_signature() {
        let value = vectors();
        let issuer = fixture_issuer(&value);
        assert_eq!(issuer.public_key_entry(), fixture_keyset(&value).keys[0]);
        assert!(fixture_keyset(&value).contains_issuer(&issuer).unwrap());

        let expected_ticket: MaplePairRequestTicketV1 =
            serde_json::from_value(value["request_ticket"].clone()).unwrap();
        let mut unsigned_ticket = expected_ticket.clone();
        unsigned_ticket.issuer_signature.clear();
        assert_eq!(
            sign_pair_request_ticket(unsigned_ticket, &issuer).unwrap(),
            expected_ticket
        );

        let expected_authorization: MaplePairAuthorizationV1 =
            serde_json::from_value(value["pair_authorization"].clone()).unwrap();
        let mut unsigned_authorization = expected_authorization.clone();
        unsigned_authorization.issuer_signature.clear();
        assert_eq!(
            sign_pair_authorization(unsigned_authorization, &issuer).unwrap(),
            expected_authorization
        );

        let expected_revocation: MaplePairRevocationV1 =
            serde_json::from_value(value["pair_revocation"].clone()).unwrap();
        let mut unsigned_revocation = expected_revocation.clone();
        unsigned_revocation.issuer_signature.clear();
        assert_eq!(
            sign_pair_revocation(unsigned_revocation, &issuer).unwrap(),
            expected_revocation
        );

        for fixture_key in [
            "revocation_stream_checkpoint",
            "ack_revocation_stream_checkpoint",
            "discovery_revocation_stream_checkpoint",
        ] {
            let expected: MapleRevocationStreamCheckpointV1 =
                serde_json::from_value(value[fixture_key].clone()).unwrap();
            let mut unsigned = expected.clone();
            unsigned.issuer_signature.clear();
            assert_eq!(
                sign_revocation_stream_checkpoint(unsigned, &issuer).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn historical_reset_event_verifies_after_ack_and_after_later_events() {
        let value = vectors();
        let keyset = fixture_keyset(&value);

        let (instruction, acked_checkpoint) = signed_genesis_reset_and_checkpoint(&value, 1, 1);
        assert!(instruction
            .verify_against_checkpoint(&acked_checkpoint, &keyset)
            .is_err());
        let historical = MapleRevocationStreamEventV1::ResetClearRequired(instruction.clone());
        historical
            .verify_against_checkpoint(&acked_checkpoint, &keyset)
            .unwrap();
        ListMaplePairingRevocationsResponse {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            query_id: Uuid::parse_str("87000000-0000-4000-8000-000000000004").unwrap(),
            revocation_sync: MapleRevocationSyncV1::status_for_checkpoint(
                instruction.security_epoch,
                acked_checkpoint,
                None,
            )
            .unwrap(),
            events: vec![historical.clone()],
            next_after_issuer_sequence: 1,
            has_more: false,
        }
        .verify(&keyset)
        .unwrap();

        let (_, later_checkpoint) = signed_genesis_reset_and_checkpoint(&value, 2, 1);
        ListMaplePairingRevocationsResponse {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            query_id: Uuid::parse_str("87000000-0000-4000-8000-000000000005").unwrap(),
            revocation_sync: MapleRevocationSyncV1::status_for_checkpoint(
                instruction.security_epoch,
                later_checkpoint,
                None,
            )
            .unwrap(),
            events: vec![historical],
            next_after_issuer_sequence: 1,
            has_more: true,
        }
        .verify(&keyset)
        .unwrap();
    }

    #[test]
    fn unacked_reset_event_requires_exact_pending_sync_instruction() {
        let value = vectors();
        let keyset = fixture_keyset(&value);
        let (instruction, checkpoint) = signed_genesis_reset_and_checkpoint(&value, 2, 0);
        let response = ListMaplePairingRevocationsResponse {
            protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            query_id: Uuid::parse_str("87000000-0000-4000-8000-000000000006").unwrap(),
            revocation_sync: MapleRevocationSyncV1::status_for_checkpoint(
                instruction.security_epoch,
                checkpoint,
                None,
            )
            .unwrap(),
            events: vec![MapleRevocationStreamEventV1::ResetClearRequired(
                instruction,
            )],
            next_after_issuer_sequence: 1,
            has_more: true,
        };
        assert_eq!(
            response.verify(&keyset),
            Err(MaplePairingWireError::InvalidField(
                "reset_clear_stream_event"
            ))
        );
    }

    #[test]
    fn revocation_stream_event_json_starts_with_frozen_tag_then_content() {
        let value = vectors();
        let (instruction, _) = signed_genesis_reset_and_checkpoint(&value, 1, 0);
        let reset_json = serde_json::to_string(&MapleRevocationStreamEventV1::ResetClearRequired(
            instruction,
        ))
        .unwrap();
        assert!(reset_json.starts_with("{\"event_type\":\"reset_clear_required\",\"event\":{"));

        let revocation: MaplePairRevocationV1 =
            serde_json::from_value(value["pair_revocation"].clone()).unwrap();
        let revocation_json =
            serde_json::to_string(&MapleRevocationStreamEventV1::PairRevocation(revocation))
                .unwrap();
        assert!(revocation_json.starts_with("{\"event_type\":\"pair_revocation\",\"event\":{"));
    }

    #[test]
    fn frozen_reset_clear_vectors_cover_bounds_chain_paging_and_ack() {
        let value = vectors();
        let keyset = fixture_keyset(&value);
        let issuer = fixture_issuer(&value);

        for name in ["empty", "max_128"] {
            let vector = &value["reset_clear_admission_set_vectors"][name];
            let leaves = vector["leaves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|leaf| MapleResetClearAdmissionLeafV1 {
                    pair_id: Uuid::parse_str(leaf["pair_id"].as_str().unwrap()).unwrap(),
                    pairing_incarnation: leaf["pairing_incarnation"].as_u64().unwrap(),
                    pair_authorization_digest: decode_standard_base64::<32>(
                        leaf["pair_authorization_digest"].as_str().unwrap(),
                        "pair_authorization_digest",
                    )
                    .unwrap(),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                hex::encode(
                    reset_clear_admission_set_transcript(
                        MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                        &leaves,
                    )
                    .unwrap()
                ),
                vector["transcript_hex"].as_str().unwrap()
            );
            assert_eq!(
                reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, &leaves,)
                    .unwrap(),
                decode_standard_base64::<32>(
                    vector["digest"].as_str().unwrap(),
                    "admission_set_digest",
                )
                .unwrap()
            );
        }

        let chain = value["reset_clear_three_reset_chain"].as_array().unwrap();
        assert_eq!(chain.len(), 3);
        let mut instructions = Vec::with_capacity(chain.len());
        for (index, vector) in chain.iter().enumerate() {
            let instruction: MapleResetClearRequiredV1 =
                serde_json::from_value(vector["instruction"].clone()).unwrap();
            assert_eq!(instruction.reset_generation, index as u64 + 1);
            assert_eq!(
                hex::encode(reset_clear_instruction_material_transcript(&instruction).unwrap()),
                vector["instruction_material_transcript_hex"]
                    .as_str()
                    .unwrap()
            );
            assert_eq!(
                hex::encode(reset_clear_chain_transcript(&instruction).unwrap()),
                vector["chain_transcript_hex"].as_str().unwrap()
            );
            assert_eq!(
                hex::encode(instruction.transcript().unwrap()),
                vector["signed_transcript_hex"].as_str().unwrap()
            );
            assert_eq!(
                instruction.event_digest().unwrap(),
                decode_standard_base64::<32>(
                    vector["event_digest"].as_str().unwrap(),
                    "event_digest",
                )
                .unwrap()
            );
            instruction.verify(&keyset).unwrap();

            let checkpoint = sign_revocation_stream_checkpoint(
                MapleRevocationStreamCheckpointV1 {
                    artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                    subject_account_id: instruction.subject_account_id,
                    subject_project_id: instruction.subject_project_id,
                    host: instruction.host.clone(),
                    security_epoch: instruction.security_epoch,
                    revocation_stream_id: instruction.revocation_stream_id,
                    revocation_stream_generation: instruction.revocation_stream_generation,
                    last_issued_issuer_sequence: 1,
                    last_acked_issuer_sequence: 0,
                    issuer_key_id: String::new(),
                    issuer_signature: String::new(),
                },
                &issuer,
            )
            .unwrap();
            instruction
                .verify_discovered_head_against_checkpoint(&checkpoint, &keyset)
                .unwrap();
            if let Some(predecessor) = instructions.last() {
                instruction
                    .verify_direct_successor(predecessor, &checkpoint, &keyset)
                    .unwrap();
            }
            instructions.push(instruction);
        }
        assert_eq!(
            instructions[2].admission_count,
            MAPLE_RESET_CLEAR_MAX_ADMISSIONS
        );

        // A missed-reset successor carries the exact retained host claim. It
        // may not refresh even an otherwise-valid endpoint epoch before the
        // prior clear instruction has been durably acknowledged.
        let predecessor = &instructions[0];
        let mut changed_host = instructions[1].clone();
        changed_host.host.endpoint_epoch = changed_host.host.endpoint_epoch.checked_add(1).unwrap();
        let changed_host = sign_reset_clear_required(changed_host, &issuer).unwrap();
        let changed_checkpoint = sign_revocation_stream_checkpoint(
            MapleRevocationStreamCheckpointV1 {
                artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                subject_account_id: changed_host.subject_account_id,
                subject_project_id: changed_host.subject_project_id,
                host: changed_host.host.clone(),
                security_epoch: changed_host.security_epoch,
                revocation_stream_id: changed_host.revocation_stream_id,
                revocation_stream_generation: changed_host.revocation_stream_generation,
                last_issued_issuer_sequence: 1,
                last_acked_issuer_sequence: 0,
                issuer_key_id: String::new(),
                issuer_signature: String::new(),
            },
            &issuer,
        )
        .unwrap();
        assert!(changed_host
            .verify_direct_successor(predecessor, &changed_checkpoint, &keyset)
            .is_err());

        let pending_checkpoint: MapleRevocationStreamCheckpointV1 =
            serde_json::from_value(value["reset_clear_pending_checkpoint"].clone()).unwrap();
        assert_eq!(pending_checkpoint.last_issued_issuer_sequence, 1);
        assert_eq!(pending_checkpoint.last_acked_issuer_sequence, 0);
        instructions[2]
            .verify_against_checkpoint(&pending_checkpoint, &keyset)
            .unwrap();

        let list_request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["reset_clear_list_revocations_request"].clone()).unwrap();
        let list_response: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["reset_clear_list_revocations_response"].clone()).unwrap();
        list_response
            .verify_against_request(&list_request, &keyset)
            .unwrap();

        let ack_request: AckMaplePairingRevocationRequest =
            serde_json::from_value(value["reset_clear_ack_request"].clone()).unwrap();
        let ack_response: AckMaplePairingRevocationResponse =
            serde_json::from_value(value["reset_clear_ack_response"].clone()).unwrap();
        ack_response
            .verify_against_request(&ack_request, &keyset)
            .unwrap();

        let historical_acked: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["reset_clear_historical_acked_response"].clone()).unwrap();
        historical_acked
            .verify_against_request(&list_request, &keyset)
            .unwrap();

        let later_request: ListMaplePairingRevocationsRequest =
            serde_json::from_value(value["reset_clear_historical_later_request"].clone()).unwrap();
        let later_response: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["reset_clear_historical_later_response"].clone()).unwrap();
        later_response
            .verify_against_request(&later_request, &keyset)
            .unwrap();
    }

    #[test]
    fn v6_authority_debug_is_fully_redacted() {
        let value = vectors();
        let instruction: MapleResetClearRequiredV1 = serde_json::from_value(
            value["reset_clear_three_reset_chain"][2]["instruction"].clone(),
        )
        .unwrap();
        let checkpoint: MapleRevocationStreamCheckpointV1 =
            serde_json::from_value(value["reset_clear_pending_checkpoint"].clone()).unwrap();
        let sync: MapleRevocationSyncV1 =
            serde_json::from_value(value["reset_clear_pending_sync"].clone()).unwrap();
        let event = MapleRevocationStreamEventV1::ResetClearRequired(instruction.clone());
        let list: ListMaplePairingRevocationsResponse =
            serde_json::from_value(value["reset_clear_list_revocations_response"].clone()).unwrap();
        let ack_request: AckMaplePairingRevocationRequest =
            serde_json::from_value(value["reset_clear_ack_request"].clone()).unwrap();
        let ack_response: AckMaplePairingRevocationResponse =
            serde_json::from_value(value["reset_clear_ack_response"].clone()).unwrap();

        let forbidden = [
            instruction.event_id.to_string(),
            instruction.reset_id.to_string(),
            instruction.subject_account_id.to_string(),
            instruction.subject_project_id.to_string(),
            instruction.recipient_host_registration_id.to_string(),
            instruction.revocation_stream_id.to_string(),
            instruction.admission_set_digest.clone(),
            instruction.instruction_material_digest.clone(),
            instruction.chain_digest.clone(),
            instruction.issuer_key_id.clone(),
            instruction.issuer_signature.clone(),
            ack_request.operation_id.to_string(),
            list.query_id.to_string(),
        ];
        for debug in [
            format!("{instruction:?}"),
            format!("{checkpoint:?}"),
            format!("{sync:?}"),
            format!("{event:?}"),
            format!("{list:?}"),
            format!("{ack_request:?}"),
            format!("{ack_response:?}"),
        ] {
            assert!(debug.contains("[redacted]"));
            for secret in &forbidden {
                assert!(!debug.contains(secret), "Debug leaked authority material");
            }
        }
    }
}
