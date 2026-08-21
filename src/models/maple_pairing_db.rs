use crate::models::maple_pairings::{
    CreateMaplePairingRequest, MaplePairAuthorizationV1, MaplePairRequestTicketV1,
    MaplePairRevocationV1, MaplePairingMutationResponse, MapleRevocationSyncStatusV1,
    MapleRevocationSyncV1, RevokeMaplePairingRequest,
};
use crate::models::schema::{
    maple_pairing_authority_account_heads, maple_pairing_authority_global_heads,
    maple_pairing_authority_org_heads, maple_pairing_authority_project_heads,
    maple_pairing_host_states, maple_pairing_issuer_keys, maple_pairing_lineages,
    maple_pairing_operations, maple_pairing_reset_clear_admissions,
    maple_pairing_reset_clear_obligations, maple_pairing_revocation_events,
    maple_pairing_revocation_highwaters, maple_pairings,
};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST: i16 = 1;
pub const MAPLE_PAIRING_PAYLOAD_VERSION_V1: i16 = 1;
pub const MAPLE_PAIRING_RECEIPT_VERSION_V1: i16 = 1;
pub const MAPLE_REGISTRATION_SYNC_READY: i16 = 1;
pub const MAPLE_REGISTRATION_SYNC_REVOCATIONS_PENDING: i16 = 2;
pub const MAPLE_REGISTRATION_SYNC_RESET_CLEAR_REQUIRED: i16 = 3;

pub const MAPLE_PAIRING_AUTHORITY_PENDING: i16 = 1;
pub const MAPLE_PAIRING_AUTHORITY_ACTIVE: i16 = 2;

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_authority_global_heads)]
#[diesel(primary_key(singleton))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingAuthorityGlobalHead {
    pub singleton: bool,
    pub activation_state: i16,
    pub org_inventory_digest: Vec<u8>,
    pub org_count: i64,
    pub issuer_key_inventory_digest: Vec<u8>,
    pub issuer_key_count: i64,
    pub revision: i64,
    pub record_mac: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_issuer_keys)]
#[diesel(primary_key(key_id))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingIssuerKey {
    pub key_id: String,
    pub global_singleton: bool,
    pub algorithm: String,
    pub public_key_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_issuer_keys)]
pub(crate) struct NewMaplePairingIssuerKey {
    pub key_id: String,
    pub global_singleton: bool,
    pub algorithm: String,
    pub public_key_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_authority_org_heads)]
pub(crate) struct NewMaplePairingAuthorityOrgHead {
    pub org_id: i32,
    pub global_singleton: bool,
    pub project_inventory_digest: Vec<u8>,
    pub project_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_authority_org_heads)]
#[diesel(primary_key(org_id))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingAuthorityOrgHead {
    pub org_id: i32,
    pub global_singleton: bool,
    pub project_inventory_digest: Vec<u8>,
    pub project_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_authority_project_heads)]
pub(crate) struct NewMaplePairingAuthorityProjectHead {
    pub project_id: i32,
    pub org_id: i32,
    pub project_uuid: Uuid,
    pub subject_project_id: Uuid,
    pub account_inventory_digest: Vec<u8>,
    pub account_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_authority_project_heads)]
#[diesel(primary_key(project_id))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingAuthorityProjectHead {
    pub project_id: i32,
    pub org_id: i32,
    pub project_uuid: Uuid,
    pub subject_project_id: Uuid,
    pub account_inventory_digest: Vec<u8>,
    pub account_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_authority_account_heads)]
pub(crate) struct NewMaplePairingAuthorityAccountHead {
    pub user_id: Uuid,
    pub project_id: i32,
    pub org_id: i32,
    pub security_epoch: i64,
    pub authority_scope_digest: Vec<u8>,
    pub authority_inventory_digest: Vec<u8>,
    pub authority_row_count: i64,
    pub device_count: i64,
    pub device_operation_count: i64,
    pub lineage_count: i64,
    pub pairing_count: i64,
    pub pairing_operation_count: i64,
    pub host_state_count: i64,
    pub revocation_event_count: i64,
    pub highwater_installation_group_count: i64,
    pub highwater_generation_count: i64,
    pub registration_operation_tombstone_count: i64,
    pub installation_retirement_count: i64,
    pub reset_clear_obligation_count: i64,
    pub reset_clear_admission_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_authority_account_heads)]
#[diesel(primary_key(user_id))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingAuthorityAccountHead {
    pub user_id: Uuid,
    pub project_id: i32,
    pub org_id: i32,
    pub security_epoch: i64,
    pub authority_scope_digest: Vec<u8>,
    pub authority_inventory_digest: Vec<u8>,
    pub authority_row_count: i64,
    pub device_count: i64,
    pub device_operation_count: i64,
    pub lineage_count: i64,
    pub pairing_count: i64,
    pub pairing_operation_count: i64,
    pub host_state_count: i64,
    pub revocation_event_count: i64,
    pub highwater_installation_group_count: i64,
    pub highwater_generation_count: i64,
    pub registration_operation_tombstone_count: i64,
    pub installation_retirement_count: i64,
    pub reset_clear_obligation_count: i64,
    pub reset_clear_admission_count: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i16)]
pub enum MaplePairingState {
    Pending = 1,
    AwaitingHostCommit = 2,
    Active = 3,
    Expired = 4,
    Revoked = 5,
}

impl MaplePairingState {
    pub const fn as_db(self) -> i16 {
        self as i16
    }
}

impl TryFrom<i16> for MaplePairingState {
    type Error = ();

    fn try_from(value: i16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Pending),
            2 => Ok(Self::AwaitingHostCommit),
            3 => Ok(Self::Active),
            4 => Ok(Self::Expired),
            5 => Ok(Self::Revoked),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaplePairingRole {
    Controller,
    Host,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i16)]
pub enum MaplePairingOperationKind {
    Create = 1,
    Approve = 2,
    Confirm = 3,
    Revoke = 4,
    Ack = 5,
}

impl MaplePairingOperationKind {
    pub const fn as_db(self) -> i16 {
        self as i16
    }
}

#[derive(Clone)]
pub struct MaplePairingAuthorization {
    pub user_id: Uuid,
    pub project_id: i32,
    pub auth_credential_kind: String,
    pub auth_binding: [u8; 32],
    pub enclave_key: Vec<u8>,
}

impl std::fmt::Debug for MaplePairingAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaplePairingAuthorization")
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("auth_credential_kind", &self.auth_credential_kind)
            .field("auth_binding", &"[redacted]")
            .field("enclave_key", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaplePairingCursor {
    pub pair_id: Uuid,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_lineages)]
#[diesel(check_for_backend(diesel::pg::Pg))]
#[allow(dead_code)]
pub(crate) struct MaplePairingLineage {
    pub id: i64,
    pub user_id: Uuid,
    pub project_id: i32,
    pub controller_maple_device_id: i64,
    pub host_maple_device_id: i64,
    pub direction: i16,
    pub last_pairing_incarnation: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_lineages)]
pub(crate) struct NewMaplePairingLineage {
    pub user_id: Uuid,
    pub project_id: i32,
    pub controller_maple_device_id: i64,
    pub host_maple_device_id: i64,
    pub direction: i16,
    pub last_pairing_incarnation: i64,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairings)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct MaplePairing {
    pub id: i64,
    pub uuid: Uuid,
    pub pairing_request_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub lineage_id: i64,
    pub controller_maple_device_id: i64,
    pub host_maple_device_id: i64,
    pub direction: i16,
    pub pairing_incarnation: i64,
    pub state: i16,
    pub revision: i64,
    pub request_nonce_mac: Vec<u8>,
    pub revocation_stream_id: Option<Uuid>,
    pub revocation_stream_generation: Option<i64>,
    pub pair_authorization_digest: Option<Vec<u8>>,
    pub ticket_issuer_key_id: String,
    pub authorization_issuer_key_id: Option<String>,
    pub revocation_issuer_key_id: Option<String>,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub approved_at: Option<DateTime<Utc>>,
    pub activated_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

impl std::fmt::Debug for MaplePairing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaplePairing")
            .field("id", &self.id)
            .field("uuid", &self.uuid)
            .field("pairing_request_id", &self.pairing_request_id)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("lineage_id", &self.lineage_id)
            .field(
                "controller_maple_device_id",
                &self.controller_maple_device_id,
            )
            .field("host_maple_device_id", &self.host_maple_device_id)
            .field("direction", &self.direction)
            .field("pairing_incarnation", &self.pairing_incarnation)
            .field("state", &self.state)
            .field("revision", &self.revision)
            .field("request_nonce_mac", &"[redacted]")
            .field("revocation_stream_id", &self.revocation_stream_id)
            .field(
                "revocation_stream_generation",
                &self.revocation_stream_generation,
            )
            .field("pair_authorization_digest", &"[redacted]")
            .field("ticket_issuer_key_id", &"[redacted]")
            .field("authorization_issuer_key_id", &"[redacted]")
            .field("revocation_issuer_key_id", &"[redacted]")
            .field("payload_version", &self.payload_version)
            .field("payload_enc", &"[redacted]")
            .field("record_mac", &"[redacted]")
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("approved_at", &self.approved_at)
            .field("activated_at", &self.activated_at)
            .field("revoked_at", &self.revoked_at)
            .field("updated_at", &self.updated_at)
            .finish()
    }
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairings)]
pub(crate) struct NewMaplePairing {
    pub uuid: Uuid,
    pub pairing_request_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub lineage_id: i64,
    pub controller_maple_device_id: i64,
    pub host_maple_device_id: i64,
    pub direction: i16,
    pub pairing_incarnation: i64,
    pub state: i16,
    pub revision: i64,
    pub request_nonce_mac: Vec<u8>,
    pub revocation_stream_id: Option<Uuid>,
    pub revocation_stream_generation: Option<i64>,
    pub pair_authorization_digest: Option<Vec<u8>>,
    pub ticket_issuer_key_id: String,
    pub authorization_issuer_key_id: Option<String>,
    pub revocation_issuer_key_id: Option<String>,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub approved_at: Option<DateTime<Utc>>,
    pub activated_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_operations)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingOperation {
    pub id: i64,
    pub operation_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub actor_maple_device_id: i64,
    pub operation_kind: i16,
    pub request_mac: Vec<u8>,
    pub maple_pairing_id: i64,
    pub pairing_revision: i64,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub receipt_issuer_key_id: Option<String>,
    pub receipt_mac: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_operations)]
pub(crate) struct NewMaplePairingOperation {
    pub operation_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub actor_maple_device_id: i64,
    pub operation_kind: i16,
    pub request_mac: Vec<u8>,
    pub maple_pairing_id: i64,
    pub pairing_revision: i64,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub receipt_issuer_key_id: Option<String>,
    pub receipt_mac: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_host_states)]
#[diesel(check_for_backend(diesel::pg::Pg))]
#[allow(dead_code)]
pub(crate) struct MaplePairingHostState {
    pub id: i64,
    pub user_id: Uuid,
    pub project_id: i32,
    pub host_maple_device_id: i64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub last_issued_revocation_sequence: i64,
    pub last_acked_revocation_sequence: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_host_states)]
pub(crate) struct NewMaplePairingHostState {
    pub user_id: Uuid,
    pub project_id: i32,
    pub host_maple_device_id: i64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub last_issued_revocation_sequence: i64,
    pub last_acked_revocation_sequence: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
}

/// One append-only generation of the pseudonymous, deletion-surviving
/// allocation fence for a stable host installation. `lookup_digest` is a
/// keyed digest of the account, internal project, and installation identifiers;
/// none of those raw identifiers are retained. Retired generations remain to
/// reserve every previously issued stream UUID and authenticate history.
#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_revocation_highwaters)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingRevocationHighwater {
    pub id: i64,
    pub lookup_digest: Vec<u8>,
    pub authority_scope_digest: Vec<u8>,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub security_epoch: i64,
    pub last_issued_revocation_sequence: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_revocation_highwaters)]
pub(crate) struct NewMaplePairingRevocationHighwater {
    pub lookup_digest: Vec<u8>,
    pub authority_scope_digest: Vec<u8>,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub security_epoch: i64,
    pub last_issued_revocation_sequence: i64,
    pub revision: i64,
    pub record_mac: Vec<u8>,
}

/// One append-only reset instruction for a stable host installation. The row
/// intentionally carries only pseudonymous account/installation selectors;
/// public claim material remains AEAD-encrypted and every field is MAC-bound.
#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_reset_clear_obligations)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingResetClearObligation {
    pub id: i64,
    pub uuid: Uuid,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub host_identity_mac: Vec<u8>,
    pub reset_id: Uuid,
    pub reset_generation: i64,
    pub cumulative_reset_count: i64,
    pub previous_event_id: Option<Uuid>,
    pub previous_instruction_digest: Option<Vec<u8>>,
    pub previous_chain_digest: Option<Vec<u8>>,
    pub old_revocation_stream_id: Uuid,
    pub old_revocation_stream_generation: i64,
    pub source_security_epoch: i64,
    pub source_last_issued_revocation_sequence: i64,
    pub target_revocation_stream_id: Uuid,
    pub target_revocation_stream_generation: i64,
    pub target_security_epoch: i64,
    pub target_instruction_sequence: i64,
    pub clear_scope: i16,
    pub admission_set_digest: Vec<u8>,
    pub admission_count: i16,
    pub host_claim_payload_version: i16,
    pub host_claim_payload_enc: Vec<u8>,
    pub host_claim_digest: Vec<u8>,
    pub instruction_payload_version: i16,
    pub instruction_payload_enc: Vec<u8>,
    pub instruction_digest: Vec<u8>,
    pub chain_digest: Vec<u8>,
    pub reset_at: DateTime<Utc>,
    pub signed_instruction_payload_version: Option<i16>,
    pub signed_instruction_payload_enc: Option<Vec<u8>>,
    pub signed_instruction_issuer_key_id: Option<String>,
    pub signed_instruction_digest: Option<Vec<u8>>,
    pub sync_payload_version: Option<i16>,
    pub sync_payload_enc: Option<Vec<u8>>,
    pub sync_issuer_key_id: Option<String>,
    pub sync_digest: Option<Vec<u8>>,
    pub state: i16,
    pub revision: i64,
    pub acked_by_head_event_id: Option<Uuid>,
    pub acked_at: Option<DateTime<Utc>>,
    pub ack_operation_id: Option<Uuid>,
    pub ack_host_registration_lookup_digest: Option<Vec<u8>>,
    pub ack_request_mac: Option<Vec<u8>>,
    pub ack_receipt_version: Option<i16>,
    pub ack_receipt_enc: Option<Vec<u8>>,
    pub ack_receipt_issuer_key_id: Option<String>,
    pub ack_receipt_digest: Option<Vec<u8>>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_reset_clear_obligations)]
pub(crate) struct NewMaplePairingResetClearObligation {
    pub uuid: Uuid,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub host_identity_mac: Vec<u8>,
    pub reset_id: Uuid,
    pub reset_generation: i64,
    pub cumulative_reset_count: i64,
    pub previous_event_id: Option<Uuid>,
    pub previous_instruction_digest: Option<Vec<u8>>,
    pub previous_chain_digest: Option<Vec<u8>>,
    pub old_revocation_stream_id: Uuid,
    pub old_revocation_stream_generation: i64,
    pub source_security_epoch: i64,
    pub source_last_issued_revocation_sequence: i64,
    pub target_revocation_stream_id: Uuid,
    pub target_revocation_stream_generation: i64,
    pub target_security_epoch: i64,
    pub target_instruction_sequence: i64,
    pub clear_scope: i16,
    pub admission_set_digest: Vec<u8>,
    pub admission_count: i16,
    pub host_claim_payload_version: i16,
    pub host_claim_payload_enc: Vec<u8>,
    pub host_claim_digest: Vec<u8>,
    pub instruction_payload_version: i16,
    pub instruction_payload_enc: Vec<u8>,
    pub instruction_digest: Vec<u8>,
    pub chain_digest: Vec<u8>,
    pub reset_at: DateTime<Utc>,
    pub state: i16,
    pub revision: i64,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_reset_clear_admissions)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingResetClearAdmission {
    pub id: i64,
    pub obligation_uuid: Uuid,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub pair_id: Uuid,
    pub pairing_incarnation: i64,
    pub pair_authorization_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_reset_clear_admissions)]
pub(crate) struct NewMaplePairingResetClearAdmission {
    pub obligation_uuid: Uuid,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub pair_id: Uuid,
    pub pairing_incarnation: i64,
    pub pair_authorization_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

/// Authenticated, DB-linearized public facts passed to the pure local signer.
/// The callback performs no network access; its exact outputs are persisted in
/// the same serializable transaction before they can be returned.
#[derive(Clone)]
pub struct MapleResetClearSyncMaterializationContext {
    pub account_id: Uuid,
    /// Public project identifier used by the signed reset checkpoint.
    pub subject_project_id: Uuid,
    pub internal_project_id: i32,
    pub event_id: Uuid,
    pub reset_id: Uuid,
    pub reset_generation: u64,
    pub cumulative_reset_count: u64,
    pub source_security_epoch: u64,
    pub security_epoch: u64,
    pub source_revocation_stream_id: Uuid,
    pub source_revocation_stream_generation: u64,
    pub source_last_issued_revocation_sequence: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub issuer_sequence: u64,
    pub previous_event_id: Option<Uuid>,
    pub previous_instruction_material_digest: Option<[u8; 32]>,
    pub previous_chain_digest: Option<[u8; 32]>,
    pub admission_count: u16,
    pub admission_set_digest: [u8; 32],
    pub host_claim_payload_version: i16,
    /// Reset-time host claim retained by the unsigned obligation.
    pub host_claim_payload: Vec<u8>,
    /// Current authenticated encrypted device row. The pure application
    /// callback decrypts it with the web-owned device codec and must return a
    /// checkpoint whose host claim matches this exact row.
    pub current_device: MaplePairingCreateDeviceContext,
    pub instruction_payload_version: i16,
    pub instruction_payload: Vec<u8>,
    pub instruction_material_digest: [u8; 32],
    pub chain_digest: [u8; 32],
    pub reset_at: DateTime<Utc>,
}

/// DB-linearized ordinary registration checkpoint facts. The encrypted device
/// row is supplied so the pure application callback can decrypt and validate
/// the current public host claim without DB code interpreting web ciphertext.
#[derive(Clone)]
pub struct MapleDeviceRegistrationOrdinarySyncContext {
    pub account_id: Uuid,
    pub subject_project_id: Uuid,
    pub internal_project_id: i32,
    pub current_device: MaplePairingCreateDeviceContext,
    pub security_epoch: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub last_issued_issuer_sequence: u64,
    pub last_acked_issuer_sequence: u64,
    pub status: MapleRevocationSyncStatusV1,
}

/// Exactly one sync shape is selected by authenticated DB state. Callbacks do
/// not choose between ordinary readiness and a pending reset-clear command.
#[derive(Clone)]
// Ordinary is the allocation-free hot path; rare reset-clear material is explicitly boxed.
#[allow(clippy::large_enum_variant)]
pub enum MapleDeviceRegistrationSyncMaterializationContext {
    Ordinary(MapleDeviceRegistrationOrdinarySyncContext),
    ResetClearRequired(Box<MapleResetClearSyncMaterializationContext>),
}

/// Pure callback output. Clear typed objects let DB code verify signatures and
/// every locked field before it encrypts the exact serialized bytes at rest.
#[derive(Clone)]
pub enum MapleDeviceRegistrationSyncMaterial {
    Ordinary {
        sync: MapleRevocationSyncV1,
        sync_payload_version: i16,
        sync_payload: Vec<u8>,
    },
    ResetClearRequired {
        sync: MapleRevocationSyncV1,
        signed_instruction_payload_version: i16,
        signed_instruction_payload: Vec<u8>,
        sync_payload_version: i16,
        sync_payload: Vec<u8>,
    },
}

impl std::fmt::Debug for MapleDeviceRegistrationSyncMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Ordinary { .. } => "ordinary",
            Self::ResetClearRequired { .. } => "reset_clear_required",
        };
        formatter
            .debug_struct("MapleDeviceRegistrationSyncMaterial")
            .field("kind", &kind)
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

pub type MaterializeMapleDeviceRegistrationSync<'a> = dyn Fn(
        MapleDeviceRegistrationSyncMaterializationContext,
    ) -> Result<MapleDeviceRegistrationSyncMaterial, MaplePairingMaterializationError>
    + 'a;

#[derive(Clone, PartialEq, Eq)]
pub struct MapleResetClearAdmissionMaterial {
    pub pair_id: Uuid,
    pub pairing_incarnation: u64,
    pub pair_authorization_digest: [u8; 32],
}

/// Reset-time facts captured while the account authority graph is locked.
/// A pure local callback decrypts the web-owned device payload and produces
/// canonical public material; DB code verifies its digests and encrypts it at
/// rest before any live authority row can be deleted.
#[derive(Clone)]
pub struct MapleResetClearUnsignedMaterializationContext {
    pub account_id: Uuid,
    /// Public `org_projects.client_id` bound into public reset transcripts.
    pub subject_project_id: Uuid,
    pub internal_project_id: i32,
    pub source: MapleResetClearSource,
    pub event_id: Uuid,
    pub reset_id: Uuid,
    pub reset_generation: u64,
    pub cumulative_reset_count: u64,
    pub source_security_epoch: u64,
    pub security_epoch: u64,
    pub source_revocation_stream_id: Uuid,
    pub source_revocation_stream_generation: u64,
    pub source_last_issued_revocation_sequence: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub issuer_sequence: u64,
    pub previous_event_id: Option<Uuid>,
    pub previous_instruction_material_digest: Option<[u8; 32]>,
    pub previous_chain_digest: Option<[u8; 32]>,
    pub admission_leaves: Vec<MapleResetClearAdmissionMaterial>,
    pub reset_at: DateTime<Utc>,
}

/// A reset may rotate either a currently registered host or an unresolved
/// prior obligation after the live graph has already been deleted. Both
/// variants carry only authenticated encrypted source material; the pure web
/// callback decrypts and validates it into the same stable public host claim.
#[derive(Clone)]
pub enum MapleResetClearSource {
    LiveDevice {
        registration_id: Uuid,
        device_id: Uuid,
        installation_id: Uuid,
        revision: i64,
        endpoint_epoch: i64,
        payload_version: i16,
        payload_enc: Vec<u8>,
        identity_mac: Vec<u8>,
        record_mac: Vec<u8>,
    },
    RetainedHostClaim {
        prior_event_id: Uuid,
        payload_version: i16,
        /// Authenticated plaintext produced by DB-owned AEAD decryption after
        /// the prior obligation MAC and chain have verified under lock.
        payload: Vec<u8>,
        payload_digest: Vec<u8>,
        identity_mac: Vec<u8>,
        prior_target_revocation_stream_id: Uuid,
        prior_target_revocation_stream_generation: u64,
        prior_target_security_epoch: u64,
    },
}

#[derive(Clone)]
pub struct MapleResetClearUnsignedMaterial {
    pub host_identity_mac: Vec<u8>,
    pub host_claim_payload_version: i16,
    pub host_claim_payload: Vec<u8>,
    pub host_claim_digest: [u8; 32],
    pub instruction_payload_version: i16,
    pub instruction_payload: Vec<u8>,
    pub instruction_material_transcript: Vec<u8>,
    pub instruction_material_digest: [u8; 32],
    pub chain_digest: [u8; 32],
}

/// The DB supplies the encrypted source row plus locked canonical authority
/// facts. The application callback decrypts and validates that row using its
/// web-owned codec, then returns deterministic prepared public material. The
/// DB never interprets the web ciphertext and independently verifies every
/// digest before encrypting the prepared bytes into the retained obligation.
pub type BuildResetClearMaterial<'a> = dyn Fn(
        MapleResetClearUnsignedMaterializationContext,
    ) -> Result<MapleResetClearUnsignedMaterial, MaplePairingMaterializationError>
    + 'a;

#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_revocation_events)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct MaplePairingRevocationEvent {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub recipient_host_maple_device_id: i64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub issuer_sequence: i64,
    pub maple_pairing_id: i64,
    pub pairing_incarnation: i64,
    pub issuer_key_id: String,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub event_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub acked_at: Option<DateTime<Utc>>,
}

impl std::fmt::Debug for MaplePairingRevocationEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaplePairingRevocationEvent")
            .field("id", &self.id)
            .field("uuid", &self.uuid)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field(
                "recipient_host_maple_device_id",
                &self.recipient_host_maple_device_id,
            )
            .field("revocation_stream_id", &self.revocation_stream_id)
            .field(
                "revocation_stream_generation",
                &self.revocation_stream_generation,
            )
            .field("issuer_sequence", &self.issuer_sequence)
            .field("maple_pairing_id", &self.maple_pairing_id)
            .field("pairing_incarnation", &self.pairing_incarnation)
            .field("issuer_key_id", &"[redacted]")
            .field("payload_version", &self.payload_version)
            .field("payload_enc", &"[redacted]")
            .field("event_digest", &"[redacted]")
            .field("record_mac", &"[redacted]")
            .field("created_at", &self.created_at)
            .field("acked_at", &self.acked_at)
            .finish()
    }
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_revocation_events)]
pub(crate) struct NewMaplePairingRevocationEvent {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub recipient_host_maple_device_id: i64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: i64,
    pub issuer_sequence: i64,
    pub maple_pairing_id: i64,
    pub pairing_incarnation: i64,
    pub issuer_key_id: String,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub event_digest: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub acked_at: Option<DateTime<Utc>>,
}

/// Immutable request facts authenticated before the database allocates a
/// pairing incarnation or invokes the local signer/materializer.
#[derive(Clone)]
pub struct NewMaplePairingRequest {
    pub authorization: MaplePairingAuthorization,
    pub subject_project_id: Uuid,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub create_request: CreateMaplePairingRequest,
    pub controller_registration_id: Uuid,
    pub expected_controller_endpoint_epoch: u64,
    pub host_registration_id: Uuid,
    pub expected_host_endpoint_epoch: u64,
}

#[derive(Clone)]
pub struct MaplePairingCreateDeviceContext {
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub endpoint_epoch: u64,
    pub device_revision: i64,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub identity_mac: Vec<u8>,
    pub record_mac: Vec<u8>,
}

/// DB-linearized facts passed to a pure local materializer only after replay,
/// participant readiness, endpoint, epoch, and quota checks have succeeded.
#[derive(Clone)]
pub struct MaplePairingCreateMaterializationContext {
    pub account_id: Uuid,
    pub subject_project_id: Uuid,
    pub operation_id: Uuid,
    pub request_mac: [u8; 32],
    pub create_request: CreateMaplePairingRequest,
    pub controller: MaplePairingCreateDeviceContext,
    pub host: MaplePairingCreateDeviceContext,
    pub pairing_incarnation: u64,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// Typed clear artifacts returned by the local issuer callback. The database
/// independently validates them and owns their durable serialization/AEAD.
#[derive(Clone)]
pub struct MaplePairingCreateMaterial {
    pub request_ticket: MaplePairRequestTicketV1,
    pub response: MaplePairingMutationResponse,
}

impl std::fmt::Debug for MaplePairingCreateMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MaplePairingCreateMaterial")
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

pub type MaterializeMaplePairingCreate<'a> = dyn Fn(
        MaplePairingCreateMaterializationContext,
    ) -> Result<MaplePairingCreateMaterial, MaplePairingMaterializationError>
    + 'a;

#[derive(Clone)]
pub struct MaplePairingApproval {
    pub authorization: MaplePairingAuthorization,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub host_registration_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub expected_revocation_stream_id: Uuid,
    pub expected_revocation_stream_generation: u64,
    pub authorization_issuer_key_id: String,
    /// SHA-256 digest of the exact issuer-signed PairAuthorization artifact.
    pub pair_authorization_digest: Vec<u8>,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub approved_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MaplePairingConfirmation {
    pub authorization: MaplePairingAuthorization,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub host_registration_id: Uuid,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub activated_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MaplePairingRevocation {
    pub authorization: MaplePairingAuthorization,
    /// Exact actor-signed wire request. The database recomputes its operation
    /// MAC and verifies its signature against the locked participant identity.
    pub revoke_request: RevokeMaplePairingRequest,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub actor_registration_id: Uuid,
    pub actor_role: MaplePairingRole,
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub expected_pairing_revision: i64,
    pub pairing_incarnation: u64,
    pub expected_revocation_stream_id: Uuid,
    pub expected_revocation_stream_generation: u64,
}

#[derive(Clone, Copy)]
pub struct MaplePairingRevocationContext {
    pub pairing_request_id: Uuid,
    pub pair_id: Uuid,
    pub pairing_incarnation: u64,
    pub target_revision: i64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub issuer_sequence: u64,
    pub revoked_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MaplePairingRevocationMaterial {
    pub request_ticket: MaplePairRequestTicketV1,
    pub pair_authorization: MaplePairAuthorizationV1,
    pub revocation: MaplePairRevocationV1,
    pub response: MaplePairingMutationResponse,
}

impl std::fmt::Debug for MaplePairingRevocationMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MaplePairingRevocationMaterial")
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

/// Exact typed plaintext stored in a Maple pairing row. Mutation callbacks
/// return typed signed artifacts only; the database owns serialization and
/// AEAD for the durable representation.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredMaplePairingPayloadV1 {
    pub request_ticket: MaplePairRequestTicketV1,
    pub pair_authorization: Option<MaplePairAuthorizationV1>,
    pub revocation: Option<MaplePairRevocationV1>,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("Maple pairing materialization failed")]
pub struct MaplePairingMaterializationError;

#[derive(Clone)]
pub struct MaplePairingRevocationAck {
    pub authorization: MaplePairingAuthorization,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub host_registration_id: Uuid,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub event_id: Uuid,
    pub issuer_sequence: u64,
    pub event_digest: Vec<u8>,
    pub expected_previous_issuer_sequence: u64,
    pub checkpoint_issuer_key_id: String,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MaplePairingOperationReceipt {
    pub operation_id: Uuid,
    pub pair_id: Uuid,
    pub pairing_revision: i64,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MaplePairingRevocationPageEntry {
    pub event: MaplePairingRevocationEvent,
    pub pairing: MaplePairing,
}

#[derive(Clone)]
pub struct MaplePairingRevocationPage {
    pub events: Vec<MaplePairingRevocationPageEntry>,
    pub reset_clear_sync_payload: Option<Vec<u8>>,
    pub reset_clear_lifecycle_floor: Option<DateTime<Utc>>,
    pub security_epoch: u64,
    pub revocation_stream_id: Uuid,
    pub revocation_stream_generation: u64,
    pub last_issued_revocation_sequence: u64,
    pub last_acked_revocation_sequence: u64,
}
