use crate::models::schema::{
    maple_device_registration_operations, maple_devices, maple_pairing_installation_retirements,
    maple_pairing_registration_operation_tombstones,
};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use uuid::Uuid;

/// The current encrypted registration state for one account-scoped Maple device.
#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_devices)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct MapleDevice {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub identity_mac: Vec<u8>,
    pub endpoint_epoch: i64,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub revision: i64,
    pub registered_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl std::fmt::Debug for MapleDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapleDevice")
            .field("id", &self.id)
            .field("uuid", &self.uuid)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("device_id", &self.device_id)
            .field("installation_id", &self.installation_id)
            .field("identity_mac", &"[redacted]")
            .field("endpoint_epoch", &self.endpoint_epoch)
            .field("payload_version", &self.payload_version)
            .field("payload_enc", &"[redacted]")
            .field("record_mac", &"[redacted]")
            .field("revision", &self.revision)
            .field("registered_at", &self.registered_at)
            .field("updated_at", &self.updated_at)
            .finish()
    }
}

/// Values used when inserting a new current-state device row.
#[derive(Insertable, Clone)]
#[diesel(table_name = maple_devices)]
pub(crate) struct NewMapleDevice {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub identity_mac: Vec<u8>,
    pub endpoint_epoch: i64,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub record_mac: Vec<u8>,
    pub revision: i64,
}

impl std::fmt::Debug for NewMapleDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewMapleDevice")
            .field("uuid", &self.uuid)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("device_id", &self.device_id)
            .field("installation_id", &self.installation_id)
            .field("identity_mac", &"[redacted]")
            .field("endpoint_epoch", &self.endpoint_epoch)
            .field("payload_version", &self.payload_version)
            .field("payload_enc", &"[redacted]")
            .field("record_mac", &"[redacted]")
            .field("revision", &self.revision)
            .finish()
    }
}

/// An accepted idempotent registration operation.
#[derive(Queryable, Selectable, Identifiable, Associations, Clone)]
#[diesel(table_name = maple_device_registration_operations)]
#[diesel(belongs_to(MapleDevice, foreign_key = maple_device_id))]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MapleDeviceRegistrationOperation {
    pub id: i64,
    pub operation_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub request_mac: Vec<u8>,
    pub maple_device_id: i64,
    pub device_revision: i64,
    pub receipt_mac: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub operation_lookup_digest: Vec<u8>,
    pub known_security_epoch: i64,
    pub accepted_security_epoch: i64,
    pub response_kind: i16,
    pub sync_payload_version: i16,
    pub sync_payload_enc: Vec<u8>,
    pub sync_issuer_key_id: String,
    pub sync_digest: Vec<u8>,
}

impl std::fmt::Debug for MapleDeviceRegistrationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapleDeviceRegistrationOperation")
            .field("id", &self.id)
            .field("operation_id", &self.operation_id)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("request_mac", &"[redacted]")
            .field("maple_device_id", &self.maple_device_id)
            .field("device_revision", &self.device_revision)
            .field("receipt_mac", &"[redacted]")
            .field("accepted_at", &self.accepted_at)
            .field("authority_scope_digest", &"[redacted]")
            .field("lookup_digest", &"[redacted]")
            .field("operation_lookup_digest", &"[redacted]")
            .field("known_security_epoch", &self.known_security_epoch)
            .field("accepted_security_epoch", &self.accepted_security_epoch)
            .field("response_kind", &self.response_kind)
            .field("sync_payload_version", &self.sync_payload_version)
            .field("sync_payload_enc", &"[redacted]")
            .field("sync_issuer_key_id", &"[redacted]")
            .field("sync_digest", &"[redacted]")
            .finish()
    }
}

/// Values used after a device row has been inserted or updated in the same transaction.
#[derive(Insertable, Clone)]
#[diesel(table_name = maple_device_registration_operations)]
pub(crate) struct NewMapleDeviceRegistrationOperation {
    pub operation_id: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub request_mac: Vec<u8>,
    pub maple_device_id: i64,
    pub device_revision: i64,
    pub receipt_mac: Vec<u8>,
    pub accepted_at: DateTime<Utc>,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub operation_lookup_digest: Vec<u8>,
    pub known_security_epoch: i64,
    pub accepted_security_epoch: i64,
    pub response_kind: i16,
    pub sync_payload_version: i16,
    pub sync_payload_enc: Vec<u8>,
    pub sync_issuer_key_id: String,
    pub sync_digest: Vec<u8>,
}

impl std::fmt::Debug for NewMapleDeviceRegistrationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewMapleDeviceRegistrationOperation")
            .field("operation_id", &self.operation_id)
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("request_mac", &"[redacted]")
            .field("maple_device_id", &self.maple_device_id)
            .field("device_revision", &self.device_revision)
            .field("receipt_mac", &"[redacted]")
            .field("accepted_at", &self.accepted_at)
            .field("authority_scope_digest", &"[redacted]")
            .field("lookup_digest", &"[redacted]")
            .field("operation_lookup_digest", &"[redacted]")
            .field("known_security_epoch", &self.known_security_epoch)
            .field("accepted_security_epoch", &self.accepted_security_epoch)
            .field("response_kind", &self.response_kind)
            .field("sync_payload_version", &self.sync_payload_version)
            .field("sync_payload_enc", &"[redacted]")
            .field("sync_issuer_key_id", &"[redacted]")
            .field("sync_digest", &"[redacted]")
            .finish()
    }
}

/// Validated registration input consumed by the database transaction.
#[derive(Clone)]
pub struct NewMapleDeviceRegistration {
    pub user_id: Uuid,
    pub subject_project_id: Uuid,
    pub project_id: i32,
    pub operation_id: Uuid,
    pub request_mac: Vec<u8>,
    pub auth_credential_kind: String,
    pub auth_binding: [u8; 32],
    pub enclave_key: Vec<u8>,
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub installation_id: Uuid,
    pub identity_mac: Vec<u8>,
    pub endpoint_epoch: i64,
    pub expected_revision: Option<i64>,
    pub known_security_epoch: i64,
    pub payload_version: i16,
    pub payload_enc: Vec<u8>,
    pub revision: i64,
}

impl std::fmt::Debug for NewMapleDeviceRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewMapleDeviceRegistration")
            .field("user_id", &self.user_id)
            .field("subject_project_id", &self.subject_project_id)
            .field("project_id", &self.project_id)
            .field("operation_id", &self.operation_id)
            .field("request_mac", &"[redacted]")
            .field("auth_credential_kind", &self.auth_credential_kind)
            .field("auth_binding", &"[redacted]")
            .field("enclave_key", &"[redacted]")
            .field("registration_id", &self.registration_id)
            .field("device_id", &self.device_id)
            .field("installation_id", &self.installation_id)
            .field("identity_mac", &"[redacted]")
            .field("endpoint_epoch", &self.endpoint_epoch)
            .field("expected_revision", &self.expected_revision)
            .field("known_security_epoch", &self.known_security_epoch)
            .field("payload_version", &self.payload_version)
            .field("payload_enc", &"[redacted]")
            .field("revision", &self.revision)
            .finish()
    }
}

/// Immutable tuple captured by a self-authenticating keyset cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapleDeviceListCursor {
    pub registration_id: Uuid,
}

/// Current credential facts that a list transaction must revalidate while the
/// account row is locked. The authenticated user/project remain authoritative.
#[derive(Clone)]
pub struct MapleDeviceListAuthorization {
    pub user_id: Uuid,
    pub project_id: i32,
    pub auth_credential_kind: String,
    pub auth_binding: [u8; 32],
    pub enclave_key: Vec<u8>,
}

impl std::fmt::Debug for MapleDeviceListAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapleDeviceListAuthorization")
            .field("user_id", &self.user_id)
            .field("project_id", &self.project_id)
            .field("auth_credential_kind", &self.auth_credential_kind)
            .field("auth_binding", &"[redacted]")
            .field("enclave_key", &"[redacted]")
            .finish()
    }
}

/// Stable result returned for both a newly accepted and an idempotently replayed request.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapleDeviceRegistrationReceipt {
    pub operation_id: Uuid,
    pub registration_id: Uuid,
    pub device_id: Uuid,
    pub revision: i64,
    pub accepted_at: DateTime<Utc>,
    pub security_epoch: i64,
    pub response_kind: i16,
    /// Exact signed sync JSON, decrypted only after its durable operation row
    /// has authenticated successfully. Callers deserialize but never re-sign
    /// it, so exact operation replay survives issuer rotation.
    pub sync_payload_version: i16,
    pub sync_payload: Vec<u8>,
}

impl std::fmt::Debug for MapleDeviceRegistrationReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapleDeviceRegistrationReceipt")
            .field("revision", &self.revision)
            .field("security_epoch", &self.security_epoch)
            .field("response_kind", &self.response_kind)
            .field("sync_payload_version", &self.sync_payload_version)
            .field("authority_material", &"[redacted]")
            .finish()
    }
}

/// One authenticated device-list snapshot. Epoch and rows are returned from
/// the same serializable authority transaction so reset cannot race bootstrap.
#[derive(Clone)]
pub struct MapleDeviceListPage {
    pub security_epoch: u64,
    pub devices: Vec<MapleDevice>,
}

/// Pseudonymous, account-rooted idempotency fence retained after a reset has
/// removed the corresponding live device and operation graph.
#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_registration_operation_tombstones)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingRegistrationOperationTombstone {
    pub id: i64,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub operation_lookup_digest: Vec<u8>,
    pub retired_security_epoch: i64,
    pub request_mac: Vec<u8>,
    pub outcome_kind: i16,
    pub outcome_digest: Vec<u8>,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub receipt_digest: Vec<u8>,
    pub referenced_issuer_key_ids: Vec<String>,
    pub accepted_at: DateTime<Utc>,
    pub record_mac: Vec<u8>,
    pub retired_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_registration_operation_tombstones)]
pub(crate) struct NewMaplePairingRegistrationOperationTombstone {
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub operation_lookup_digest: Vec<u8>,
    pub retired_security_epoch: i64,
    pub request_mac: Vec<u8>,
    pub outcome_kind: i16,
    pub outcome_digest: Vec<u8>,
    pub receipt_version: i16,
    pub receipt_enc: Vec<u8>,
    pub receipt_digest: Vec<u8>,
    pub referenced_issuer_key_ids: Vec<String>,
    pub accepted_at: DateTime<Utc>,
    pub record_mac: Vec<u8>,
    pub retired_at: DateTime<Utc>,
}

/// Permanent, pseudonymous terminal fence for one acknowledged installation
/// lineage. This row is distinct from operation-id tombstones: it rejects both
/// installation-instance reuse and retained endpoint-identity reuse.
#[derive(Queryable, Selectable, Identifiable, Clone)]
#[diesel(table_name = maple_pairing_installation_retirements)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub(crate) struct MaplePairingInstallationRetirement {
    pub id: i64,
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub host_identity_mac: Vec<u8>,
    pub retired_security_epoch: i64,
    pub final_obligation_event_id: Uuid,
    pub final_instruction_digest: Vec<u8>,
    pub final_chain_digest: Vec<u8>,
    pub ack_host_registration_lookup_digest: Vec<u8>,
    pub ack_operation_lookup_digest: Vec<u8>,
    pub ack_request_mac: Vec<u8>,
    pub ack_receipt_version: i16,
    pub ack_receipt_issuer_key_id: String,
    pub ack_receipt_digest: Vec<u8>,
    pub retired_at: DateTime<Utc>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = maple_pairing_installation_retirements)]
pub(crate) struct NewMaplePairingInstallationRetirement {
    pub authority_scope_digest: Vec<u8>,
    pub lookup_digest: Vec<u8>,
    pub host_identity_mac: Vec<u8>,
    pub retired_security_epoch: i64,
    pub final_obligation_event_id: Uuid,
    pub final_instruction_digest: Vec<u8>,
    pub final_chain_digest: Vec<u8>,
    pub ack_host_registration_lookup_digest: Vec<u8>,
    pub ack_operation_lookup_digest: Vec<u8>,
    pub ack_request_mac: Vec<u8>,
    pub ack_receipt_version: i16,
    pub ack_receipt_issuer_key_id: String,
    pub ack_receipt_digest: Vec<u8>,
    pub retired_at: DateTime<Utc>,
    pub record_mac: Vec<u8>,
    pub created_at: DateTime<Utc>,
}
