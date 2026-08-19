use crate::encrypt::{decrypt_aead_v1, derive_key, encrypt_aead_v1, CanonicalBytes, EncryptError};
use crate::models::account_deletion::{
    AccountDeletionError, AccountDeletionRequest, NewAccountDeletionRequest,
};
use crate::models::app_data_migrations::{
    AppDataMigration, AppDataMigrationError, NewAppDataMigration,
};
use crate::models::enclave_secrets::{EnclaveSecret, EnclaveSecretError, NewEnclaveSecret};
use crate::models::invite_codes::{InviteCode, InviteCodeError, NewInviteCode};
use crate::models::maple_devices::{
    MapleDevice, MapleDeviceListAuthorization, MapleDeviceListCursor, MapleDeviceListPage,
    MapleDeviceRegistrationOperation, MapleDeviceRegistrationReceipt,
    MaplePairingInstallationRetirement, MaplePairingRegistrationOperationTombstone, NewMapleDevice,
    NewMapleDeviceRegistration, NewMapleDeviceRegistrationOperation,
    NewMaplePairingInstallationRetirement, NewMaplePairingRegistrationOperationTombstone,
};
use crate::models::maple_pairing_db::{
    BuildResetClearMaterial, MapleDeviceRegistrationOrdinarySyncContext,
    MapleDeviceRegistrationSyncMaterial, MapleDeviceRegistrationSyncMaterializationContext,
    MaplePairing, MaplePairingApproval, MaplePairingAuthorityAccountHead,
    MaplePairingAuthorityGlobalHead, MaplePairingAuthorityOrgHead,
    MaplePairingAuthorityProjectHead, MaplePairingAuthorization, MaplePairingConfirmation,
    MaplePairingCreateDeviceContext, MaplePairingCreateMaterial,
    MaplePairingCreateMaterializationContext, MaplePairingCursor, MaplePairingHostState,
    MaplePairingIssuerKey, MaplePairingLineage, MaplePairingMaterializationError,
    MaplePairingOperation, MaplePairingOperationKind, MaplePairingOperationReceipt,
    MaplePairingResetClearAdmission, MaplePairingResetClearObligation, MaplePairingRevocation,
    MaplePairingRevocationAck, MaplePairingRevocationContext, MaplePairingRevocationEvent,
    MaplePairingRevocationHighwater, MaplePairingRevocationMaterial, MaplePairingRevocationPage,
    MaplePairingRole, MaplePairingState, MapleResetClearAdmissionMaterial, MapleResetClearSource,
    MapleResetClearSyncMaterializationContext, MapleResetClearUnsignedMaterial,
    MapleResetClearUnsignedMaterializationContext, MaterializeMapleDeviceRegistrationSync,
    MaterializeMaplePairingCreate, NewMaplePairingAuthorityAccountHead,
    NewMaplePairingAuthorityOrgHead, NewMaplePairingAuthorityProjectHead, NewMaplePairingIssuerKey,
    NewMaplePairingRequest, NewMaplePairingResetClearAdmission,
    NewMaplePairingResetClearObligation, StoredMaplePairingPayloadV1,
    MAPLE_PAIRING_AUTHORITY_ACTIVE, MAPLE_PAIRING_AUTHORITY_PENDING,
    MAPLE_PAIRING_PAYLOAD_VERSION_V1, MAPLE_PAIRING_RECEIPT_VERSION_V1,
    MAPLE_REGISTRATION_SYNC_READY, MAPLE_REGISTRATION_SYNC_RESET_CLEAR_REQUIRED,
    MAPLE_REGISTRATION_SYNC_REVOCATIONS_PENDING,
};
use crate::models::maple_pairings::{
    reset_clear_admission_set_digest, reset_clear_chain_transcript,
    reset_clear_instruction_material_transcript, sha256_digest, MapleDeviceClaimV1,
    MaplePairRevocationV1, MaplePairingDirection, MaplePairingIssuerKeyFingerprintV1,
    MaplePairingIssuerKeySetV1, MaplePairingMutationResponse,
    MaplePairingRole as WireMaplePairingRole, MaplePairingState as WireMaplePairingState,
    MaplePairingStatusV1, MapleResetClearAdmissionLeafV1, MapleResetClearRequiredV1,
    MapleResetClearScopeV1, MapleRevocationSyncStatusV1, MapleRevocationSyncV1,
    MAPLE_PAIRING_ARTIFACT_VERSION_V1, MAPLE_PAIRING_MAX_ISSUER_KEYS,
    MAPLE_PAIRING_PROTOCOL_VERSION_V1,
};
use crate::models::oauth::{
    NewOAuthProvider, NewUserOAuthConnection, OAuthError, OAuthProvider, UserOAuthConnection,
};
use crate::models::org_memberships::NewOrgMembership;
use crate::models::org_memberships::{OrgMembership, OrgMembershipError, OrgMembershipWithUser};
use crate::models::org_project_secrets::{
    NewOrgProjectSecret, OrgProjectSecret, OrgProjectSecretError,
};
use crate::models::org_projects::{NewOrgProject, OrgProject, OrgProjectError};
use crate::models::orgs::{NewOrg, Org, OrgError};
use crate::models::password_reset::{
    NewPasswordResetRequest, PasswordResetError, PasswordResetRequest,
};
use crate::models::platform_email_verification::{
    NewPlatformEmailVerification, PlatformEmailVerification, PlatformEmailVerificationError,
};
use crate::models::platform_invite_codes::{PlatformInviteCode, PlatformInviteCodeError};
use crate::models::platform_password_reset::{
    NewPlatformPasswordResetRequest, PlatformPasswordResetError, PlatformPasswordResetRequest,
};
use crate::models::platform_users::{NewPlatformUser, PlatformUser, PlatformUserError};
use crate::models::project_settings::OAuthSettings;
use crate::models::project_settings::{
    EmailSettings, NewProjectSetting, ProjectSetting, ProjectSettingError, SettingCategory,
};
use crate::models::responses::{
    validate_conversation_project_limit, AssistantMessage, Conversation, ConversationProject,
    ConversationProjectFilter, NewAssistantMessage, NewConversation, NewConversationProject,
    NewReasoningItem, NewResponse, NewToolCall, NewToolOutput, NewUserInstruction, NewUserMessage,
    ProjectInstructionUpdate, RawThreadMessage, RawThreadMessageMetadata, ReasoningItem, Response,
    ResponseStatus, ResponsesError, ToolCall, ToolOutput, UserInstruction, UserMessage,
};
use crate::models::schema::users;
use crate::models::token_usage::{NewTokenUsage, TokenUsage, TokenUsageError};
use crate::models::user_api_keys::{NewUserApiKey, UserApiKey, UserApiKeyError};
use crate::models::user_seed_wrappings::{
    NewUserSeedWrapping, UserSeedWrapping, UserSeedWrappingError,
};
use crate::models::users::{NewUser, User, UserError};
use crate::models::{
    email_verification::{EmailVerification, EmailVerificationError, NewEmailVerification},
    org_memberships::OrgRole,
};
use crate::seed_wrapping::CredentialKind;
use crate::seed_wrapping::{decrypt_seed_v1, AuthBinding, SEED_WRAP_VERSION_V1};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{DateTime, Utc};
use diesel::{
    pg::PgConnection,
    r2d2::{ConnectionManager, Pool},
};
use diesel::{BoolExpressionMethods, Connection, ExpressionMethods, QueryDsl, RunQueryDsl};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

const MAPLE_DEVICE_RECORD_MAC_KEY_INFO: &[u8] = b"os.maple-device-db-record-mac-key.v1";
const MAPLE_DEVICE_RECORD_MAC_DOMAIN: &str = "os.maple-device-db-record.v1";
const MAPLE_DEVICE_RECEIPT_MAC_KEY_INFO: &[u8] = b"os.maple-device-receipt-mac-key.v1";
const MAPLE_DEVICE_RECEIPT_MAC_DOMAIN: &str = "os.maple-device-receipt.v1";
const MAPLE_DEVICE_OPERATION_LOOKUP_KEY_INFO: &[u8] = b"os.maple-device-operation-lookup-key.v1";
const MAPLE_DEVICE_OPERATION_LOOKUP_DOMAIN: &str = "os.maple-device-operation-lookup.v1";
const MAPLE_DEVICE_REGISTRATION_TOMBSTONE_MAC_KEY_INFO: &[u8] =
    b"os.maple-device-registration-tombstone-key.v1";
const MAPLE_DEVICE_REGISTRATION_TOMBSTONE_MAC_DOMAIN: &str =
    "os.maple-device-registration-tombstone.v1";
const MAPLE_DEVICE_REGISTRATION_TOMBSTONE_RECEIPT_KEY_INFO: &[u8] =
    b"os.maple-device-registration-tombstone-receipt-key.v1";
const MAPLE_DEVICE_REGISTRATION_TOMBSTONE_RECEIPT_DOMAIN: &str =
    "os.maple-device-registration-tombstone-receipt.v1";
const MAPLE_INSTALLATION_RETIREMENT_MAC_KEY_INFO: &[u8] =
    b"os.maple-installation-retirement-key.v1";
const MAPLE_INSTALLATION_RETIREMENT_MAC_DOMAIN: &str = "os.maple-installation-retirement.v1";
const MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_KEY_INFO: &[u8] =
    b"os.maple-reset-clear-ack-host-lookup-key.v1";
const MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_DOMAIN: &str = "os.maple-reset-clear-ack-host-lookup.v1";
const MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_KEY_INFO: &[u8] =
    b"os.maple-reset-clear-ack-operation-lookup-key.v1";
const MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_DOMAIN: &str =
    "os.maple-reset-clear-ack-operation-lookup.v1";
const MAPLE_DEVICE_SYNC_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-device-sync-payload-key.v1";
const MAPLE_DEVICE_SYNC_PAYLOAD_DOMAIN: &str = "os.maple-device-sync-payload.v1";
const MAPLE_DEVICE_IDENTITY_MAC_KEY_INFO: &[u8] = b"os.maple-device-identity-mac-key.v1";
const MAPLE_DEVICE_IDENTITY_MAC_DOMAIN: &str = "os.maple-device-identity.v1";

const MAPLE_PAIRING_RECORD_MAC_KEY_INFO: &[u8] = b"os.maple-pair-db-record-mac-key.v1";
const MAPLE_PAIRING_RECORD_MAC_DOMAIN: &str = "os.maple-pair-db-record.v1";
const MAPLE_PAIRING_RECEIPT_MAC_KEY_INFO: &[u8] = b"os.maple-pair-db-receipt-mac-key.v1";
const MAPLE_PAIRING_RECEIPT_MAC_DOMAIN: &str = "os.maple-pair-db-receipt.v1";
const MAPLE_PAIRING_REQUEST_NONCE_MAC_KEY_INFO: &[u8] = b"os.maple-pair-request-nonce-mac-key.v1";
const MAPLE_PAIRING_REQUEST_NONCE_MAC_DOMAIN: &str = "os.maple-pair-request-nonce.v1";
const MAPLE_PAIRING_OPERATION_MAC_KEY_INFO: &[u8] = b"os.maple-pair-operation-mac-key.v1";
const MAPLE_PAIRING_OPERATION_MAC_DOMAIN: &str = "os.maple-pair-operation.v1";
const MAPLE_PAIRING_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-pair-record-key.v1";
const MAPLE_PAIRING_PAYLOAD_DOMAIN: &str = "os.maple-pair-record.v1";
const MAPLE_PAIRING_RECEIPT_KEY_INFO: &[u8] = b"os.maple-pair-receipt-key.v1";
const MAPLE_PAIRING_RECEIPT_DOMAIN: &str = "os.maple-pair-receipt.v1";
const MAPLE_PAIRING_REVOCATION_PAYLOAD_KEY_INFO: &[u8] = b"os.maple-pair-revocation-record-key.v1";
const MAPLE_PAIRING_REVOCATION_PAYLOAD_DOMAIN: &str = "os.maple-pair-revocation-record.v1";
const MAPLE_PAIRING_REVOCATION_RECORD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-revocation-db-record-mac-key.v1";
const MAPLE_PAIRING_REVOCATION_RECORD_MAC_DOMAIN: &str = "os.maple-pair-revocation-db-record.v1";
const MAPLE_PAIRING_HOST_STATE_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-host-state-db-record-mac-key.v1";
const MAPLE_PAIRING_HOST_STATE_MAC_DOMAIN: &str = "os.maple-pair-host-state-db-record.v1";
const MAPLE_PAIRING_REVOCATION_HIGHWATER_LOOKUP_KEY_INFO: &[u8] =
    b"os.maple-pair-revocation-highwater-lookup-key.v1";
const MAPLE_PAIRING_REVOCATION_HIGHWATER_LOOKUP_DOMAIN: &str =
    "os.maple-pair-revocation-highwater-lookup.v1";
const MAPLE_PAIRING_REVOCATION_HIGHWATER_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-revocation-highwater-db-record-mac-key.v1";
const MAPLE_PAIRING_REVOCATION_HIGHWATER_MAC_DOMAIN: &str =
    "os.maple-pair-revocation-highwater-db-record.v1";
const MAPLE_PAIRING_RESET_CLEAR_OBLIGATION_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-reset-clear-obligation-key.v1";
const MAPLE_PAIRING_RESET_CLEAR_OBLIGATION_MAC_DOMAIN: &str =
    "os.maple-pair-reset-clear-obligation.v1";
const MAPLE_PAIRING_RESET_CLEAR_ADMISSION_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-reset-clear-admission-key.v1";
const MAPLE_PAIRING_RESET_CLEAR_ADMISSION_MAC_DOMAIN: &str =
    "os.maple-pair-reset-clear-admission.v1";
const MAPLE_PAIRING_RESET_CLEAR_PAYLOAD_KEY_INFO: &[u8] =
    b"os.maple-pair-reset-clear-payload-key.v1";
const MAPLE_PAIRING_RESET_CLEAR_PAYLOAD_DOMAIN: &str = "os.maple-pair-reset-clear-payload.v1";
const MAPLE_PAIRING_AUTHORITY_SCOPE_KEY_INFO: &[u8] = b"os.maple-pair-authority-scope-key.v1";
const MAPLE_PAIRING_AUTHORITY_SCOPE_DOMAIN: &str = "os.maple-pair-authority-scope.v1";
const MAPLE_PAIRING_AUTHORITY_ACCOUNT_HEAD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-authority-account-head-key.v1";
const MAPLE_PAIRING_AUTHORITY_PROJECT_HEAD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-authority-project-head-key.v1";
const MAPLE_PAIRING_AUTHORITY_ORG_HEAD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-authority-org-head-key.v1";
const MAPLE_PAIRING_AUTHORITY_GLOBAL_HEAD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-authority-global-head-key.v1";
const MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_KEY_INFO: &[u8] =
    b"os.maple-pair-issuer-key-record-key.v1";
const MAPLE_PAIRING_AUTHORITY_ACCOUNT_HEAD_MAC_DOMAIN: &str =
    "os.maple-pair-authority-account-head.v1";
const MAPLE_PAIRING_AUTHORITY_PROJECT_HEAD_MAC_DOMAIN: &str =
    "os.maple-pair-authority-project-head.v1";
const MAPLE_PAIRING_AUTHORITY_ORG_HEAD_MAC_DOMAIN: &str = "os.maple-pair-authority-org-head.v1";
const MAPLE_PAIRING_AUTHORITY_GLOBAL_HEAD_MAC_DOMAIN: &str =
    "os.maple-pair-authority-global-head.v1";
const MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_DOMAIN: &str = "os.maple-pair-issuer-key-record.v1";
const MAPLE_PAIRING_AUTHORITY_ACCOUNT_INVENTORY_DOMAIN: &str =
    "os.maple-pair-authority-account-inventory.v1";
const MAPLE_PAIRING_AUTHORITY_PROJECT_INVENTORY_DOMAIN: &str =
    "os.maple-pair-authority-project-inventory.v1";
const MAPLE_PAIRING_AUTHORITY_ORG_INVENTORY_DOMAIN: &str =
    "os.maple-pair-authority-org-inventory.v1";
const MAPLE_PAIRING_AUTHORITY_GLOBAL_INVENTORY_DOMAIN: &str =
    "os.maple-pair-authority-global-inventory.v1";
const MAPLE_PAIRING_ISSUER_KEY_INVENTORY_DOMAIN: &str = "os.maple-pair-issuer-key-inventory.v1";
const MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER: &str = "maple_pairing_authority_v1_activated";
const MAPLE_PAIRING_AUTHORITY_LOCK_KEY_1: i32 = 0x4d41_504c; // MAPL
const MAPLE_PAIRING_AUTHORITY_LOCK_KEY_2: i32 = 0x4155_5448; // AUTH
const MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(10);
const MAPLE_PAIRING_AUTHORITY_STATEMENT_TIMEOUT: &str = "30s";
const MAPLE_PAIRING_AUTHORITY_PAGE_SIZE: i64 = 256;
const MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE: i64 = 64;
const MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT: i64 = 1024;
const MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT: i64 = 4096;
const MAPLE_PAIRING_AUTHORITY_INSTALLATION_RETIREMENT_LIMIT: i64 = 1024;
const MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_OBLIGATION_LIMIT: i64 = 4096;
const MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_ADMISSION_LIMIT: i64 = 524_288;
const MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION: i64 = 128;

const MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT: i64 = 128;
// Each V1 transition has one durable operation kind and the schema enforces
// UNIQUE(pair, kind), so a pairing can hold at most create/approve/confirm/
// revoke/ack receipts.
const MAPLE_PAIRING_OPERATION_LIMIT_PER_PAIRING: i64 = 5;
const MAPLE_PAIRING_LIST_QUERY_LIMIT: i64 = 101;
const MAPLE_PAIRING_REVOCATION_QUERY_LIMIT: i64 = 101;
const MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES: usize = 64 * 1024;
const MAPLE_PAIRING_MAX_ENCRYPTED_REVOCATION_BYTES: usize = 32 * 1024;
const MAPLE_PAIRING_ISSUER_KEY_ID_MAX_BYTES: usize = 64;

const MAPLE_PAIRING_OPERATION_CREATE: i16 = 1;
const MAPLE_PAIRING_OPERATION_APPROVE: i16 = 2;
const MAPLE_PAIRING_OPERATION_CONFIRM: i16 = 3;
const MAPLE_PAIRING_OPERATION_REVOKE: i16 = 4;
const MAPLE_PAIRING_OPERATION_ACK: i16 = 5;

#[cfg(test)]
static FAIL_MAPLE_PAIRING_CREATE_AFTER_STAGED_MUTATIONS: std::sync::Mutex<Option<Uuid>> =
    std::sync::Mutex::new(None);

/// Arm a one-shot failure after CREATE has staged the pair, operation, and
/// cascading authority-head writes but before its transaction may commit.
/// This proves that a route-side reserved/signed candidate is never published
/// when those uncommitted writes roll back.
#[cfg(test)]
pub(crate) fn fail_next_maple_pairing_create_before_commit_for_test(operation_id: Uuid) {
    let mut armed = FAIL_MAPLE_PAIRING_CREATE_AFTER_STAGED_MUTATIONS
        .lock()
        .expect("Maple pairing create test failpoint mutex must not be poisoned");
    *armed = Some(operation_id);
}

#[cfg(test)]
fn take_maple_pairing_create_before_commit_failure_for_test(operation_id: Uuid) -> bool {
    let mut armed = FAIL_MAPLE_PAIRING_CREATE_AFTER_STAGED_MUTATIONS
        .lock()
        .expect("Maple pairing create test failpoint mutex must not be poisoned");
    if *armed == Some(operation_id) {
        armed.take();
        true
    } else {
        false
    }
}

#[cfg(test)]
struct MapleDeviceRegistrationCommitPause {
    operation_id: Uuid,
    reached: std::sync::mpsc::SyncSender<()>,
    resume: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
static PAUSE_MAPLE_DEVICE_REGISTRATION_BEFORE_COMMIT: std::sync::Mutex<
    Option<MapleDeviceRegistrationCommitPause>,
> = std::sync::Mutex::new(None);

/// Pause one fresh device registration after its authenticated head writes are
/// staged but before commit. The returned receiver observes that point; the
/// sender releases the transaction. This test seam makes advisory-lock waiter
/// races deterministic without changing production scheduling.
#[cfg(test)]
pub(crate) fn pause_next_maple_device_registration_before_commit_for_test(
    operation_id: Uuid,
) -> (
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::SyncSender<()>,
) {
    let (reached, reached_rx) = std::sync::mpsc::sync_channel(1);
    let (resume, resume_rx) = std::sync::mpsc::sync_channel(0);
    let mut armed = PAUSE_MAPLE_DEVICE_REGISTRATION_BEFORE_COMMIT
        .lock()
        .expect("Maple device registration pause mutex must not be poisoned");
    assert!(armed.is_none(), "only one registration pause may be armed");
    *armed = Some(MapleDeviceRegistrationCommitPause {
        operation_id,
        reached,
        resume: resume_rx,
    });
    (reached_rx, resume)
}

#[cfg(test)]
fn pause_maple_device_registration_before_commit_if_armed_for_test(operation_id: Uuid) {
    let pause = {
        let mut armed = PAUSE_MAPLE_DEVICE_REGISTRATION_BEFORE_COMMIT
            .lock()
            .expect("Maple device registration pause mutex must not be poisoned");
        if armed
            .as_ref()
            .is_some_and(|pause| pause.operation_id == operation_id)
        {
            armed.take()
        } else {
            None
        }
    };
    if let Some(pause) = pause {
        pause
            .reached
            .send(())
            .expect("registration pause observer must remain available");
        pause
            .resume
            .recv()
            .expect("registration pause must be explicitly released");
    }
}

#[cfg(test)]
static OBSERVE_MAPLE_PAIRING_AUTHORITY_LOCK_CONTENTION: std::sync::Mutex<
    Option<std::sync::mpsc::SyncSender<()>>,
> = std::sync::Mutex::new(None);

/// Observe the next failed `pg_try_advisory_xact_lock` attempt. Tests use this
/// to prove that a waiter fixed its SERIALIZABLE snapshot before releasing the
/// transaction currently holding the authority lock.
#[cfg(test)]
pub(crate) fn observe_next_maple_pairing_authority_lock_contention_for_test(
) -> std::sync::mpsc::Receiver<()> {
    let (observed, observed_rx) = std::sync::mpsc::sync_channel(1);
    let mut observer = OBSERVE_MAPLE_PAIRING_AUTHORITY_LOCK_CONTENTION
        .lock()
        .expect("Maple pairing authority contention observer must not be poisoned");
    assert!(
        observer.is_none(),
        "only one contention observer may be armed"
    );
    *observer = Some(observed);
    observed_rx
}

#[cfg(test)]
fn observe_maple_pairing_authority_lock_contention_if_armed_for_test() {
    if let Some(observer) = OBSERVE_MAPLE_PAIRING_AUTHORITY_LOCK_CONTENTION
        .lock()
        .expect("Maple pairing authority contention observer must not be poisoned")
        .take()
    {
        let _ = observer.send(());
    }
}

#[cfg(test)]
static OBSERVE_MAPLE_PAIRING_AUTHORITY_SCOPED_ACCESS: std::sync::Mutex<
    Option<(Uuid, Arc<std::sync::atomic::AtomicUsize>)>,
> = std::sync::Mutex::new(None);

/// Count any attempt to read an account scope after the global snapshot fence.
/// A corrupted global root must fail before this counter advances.
#[cfg(test)]
pub(crate) fn observe_maple_pairing_authority_scoped_access_for_test(
    user_id: Uuid,
) -> Arc<std::sync::atomic::AtomicUsize> {
    let observations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut observer = OBSERVE_MAPLE_PAIRING_AUTHORITY_SCOPED_ACCESS
        .lock()
        .expect("Maple pairing authority scoped-access observer must not be poisoned");
    assert!(
        observer.is_none(),
        "only one scoped-access observer may be armed"
    );
    *observer = Some((user_id, Arc::clone(&observations)));
    observations
}

#[cfg(test)]
pub(crate) fn clear_maple_pairing_authority_scoped_access_observer_for_test(user_id: Uuid) {
    let mut observer = OBSERVE_MAPLE_PAIRING_AUTHORITY_SCOPED_ACCESS
        .lock()
        .expect("Maple pairing authority scoped-access observer must not be poisoned");
    if observer
        .as_ref()
        .is_some_and(|(observed_user_id, _)| *observed_user_id == user_id)
    {
        observer.take();
    }
}

#[cfg(test)]
fn observe_maple_pairing_authority_scoped_access_if_armed_for_test(user_id: Uuid) {
    if let Some((_, observations)) = OBSERVE_MAPLE_PAIRING_AUTHORITY_SCOPED_ACCESS
        .lock()
        .expect("Maple pairing authority scoped-access observer must not be poisoned")
        .as_ref()
        .filter(|(observed_user_id, _)| *observed_user_id == user_id)
    {
        observations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

const MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS: i64 =
    crate::models::maple_pairings::MAPLE_PAIR_REQUEST_MAX_CLOCK_SKEW_MS;

/// V1 is deliberately fail-closed at these bounds. There is no automatic
/// eviction: deleting idempotency tombstones could resurrect an old operation,
/// so capacity recovery requires an explicit future lifecycle API or one of the
/// existing account security reset/deletion flows.
pub(crate) const MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT: i64 = 32;
pub(crate) const MAPLE_DEVICE_OPERATION_LIMIT_PER_DEVICE: i64 = 1024;
const MAPLE_DEVICE_LIST_QUERY_LIMIT: i64 = 101;
pub(crate) const MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES: usize = 16 * 1024;

fn map_maple_device_write_error(error: diesel::result::Error) -> DBError {
    match error {
        diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::UniqueViolation,
            _,
        ) => DBError::MapleDeviceRegistrationConflict,
        other => DBError::QueryError(other),
    }
}

fn maple_device_hmac(
    enclave_key: &[u8],
    key_info: &[u8],
    body: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, key_info)?;
    let mut mac =
        HmacSha256::new_from_slice(&key).map_err(|_| EncryptError::KeyDerivationFailed)?;
    mac.update(body);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn maple_device_identity_mac_from_claim(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    identity_public_key: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_IDENTITY_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_bytes(identity_public_key);
    maple_device_hmac(
        enclave_key,
        MAPLE_DEVICE_IDENTITY_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_device_sync_payload_aad(
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    registration_id: Uuid,
    device_revision: i64,
    security_epoch: i64,
    response_kind: i16,
    payload_version: i16,
    issuer_key_id: &str,
    payload_digest: &[u8],
) -> Vec<u8> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_SYNC_PAYLOAD_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(operation_id)
        .append_uuid(registration_id)
        .append_i64(device_revision)
        .append_i64(security_epoch)
        .append_i16(response_kind)
        .append_i16(payload_version)
        .append_str(issuer_key_id)
        .append_bytes(payload_digest);
    body.into_bytes()
}

#[allow(clippy::too_many_arguments)]
fn encrypt_maple_device_sync_payload(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    registration_id: Uuid,
    device_revision: i64,
    security_epoch: i64,
    response_kind: i16,
    payload_version: i16,
    issuer_key_id: &str,
    payload_digest: &[u8],
    payload: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, MAPLE_DEVICE_SYNC_PAYLOAD_KEY_INFO)?;
    encrypt_aead_v1(
        &key,
        payload,
        &maple_device_sync_payload_aad(
            user_id,
            project_id,
            operation_id,
            registration_id,
            device_revision,
            security_epoch,
            response_kind,
            payload_version,
            issuer_key_id,
            payload_digest,
        ),
    )
}

fn decrypt_maple_device_sync_payload(
    enclave_key: &[u8],
    operation: &MapleDeviceRegistrationOperation,
    device: &MapleDevice,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, MAPLE_DEVICE_SYNC_PAYLOAD_KEY_INFO)?;
    decrypt_aead_v1(
        &key,
        &operation.sync_payload_enc,
        &maple_device_sync_payload_aad(
            operation.user_id,
            operation.project_id,
            operation.operation_id,
            device.uuid,
            operation.device_revision,
            operation.accepted_security_epoch,
            operation.response_kind,
            operation.sync_payload_version,
            &operation.sync_issuer_key_id,
            &operation.sync_digest,
        ),
    )
}

fn maple_device_record_mac_for_registration(
    registration: &NewMapleDeviceRegistration,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_RECORD_MAC_DOMAIN);
    body.append_uuid(registration.user_id)
        .append_i32(registration.project_id)
        .append_uuid(registration.registration_id)
        .append_uuid(registration.device_id)
        .append_uuid(registration.installation_id)
        .append_bytes(&registration.identity_mac)
        .append_i64(registration.endpoint_epoch)
        .append_i16(registration.payload_version)
        .append_bytes(&registration.payload_enc)
        .append_i64(registration.revision);
    maple_device_hmac(
        &registration.enclave_key,
        MAPLE_DEVICE_RECORD_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_device_record_mac_for_row(
    enclave_key: &[u8],
    row: &MapleDevice,
) -> Result<Vec<u8>, EncryptError> {
    if row.payload_enc.len() > MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES {
        return Err(EncryptError::BadData);
    }
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_RECORD_MAC_DOMAIN);
    body.append_uuid(row.user_id)
        .append_i32(row.project_id)
        .append_uuid(row.uuid)
        .append_uuid(row.device_id)
        .append_uuid(row.installation_id)
        .append_bytes(&row.identity_mac)
        .append_i64(row.endpoint_epoch)
        .append_i16(row.payload_version)
        .append_bytes(&row.payload_enc)
        .append_i64(row.revision);
    maple_device_hmac(
        enclave_key,
        MAPLE_DEVICE_RECORD_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

pub(crate) fn maple_device_record_mac_is_valid(
    enclave_key: &[u8],
    row: &MapleDevice,
) -> Result<bool, EncryptError> {
    use subtle::ConstantTimeEq;

    if row.payload_enc.len() > MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES {
        return Ok(false);
    }
    let expected = maple_device_record_mac_for_row(enclave_key, row)?;
    Ok(bool::from(
        expected.as_slice().ct_eq(row.record_mac.as_slice()),
    ))
}

#[cfg(test)]
pub(crate) fn maple_device_record_mac_for_test(
    enclave_key: &[u8],
    row: &MapleDevice,
) -> Result<Vec<u8>, EncryptError> {
    maple_device_record_mac_for_row(enclave_key, row)
}

fn maple_device_returned_row_matches(
    registration: &NewMapleDeviceRegistration,
    row: &MapleDevice,
    expected_record_mac: &[u8],
    expected_existing_row_id: Option<i64>,
) -> bool {
    use subtle::ConstantTimeEq;

    let identity_matches: bool = row
        .identity_mac
        .as_slice()
        .ct_eq(registration.identity_mac.as_slice())
        .into();
    let payload_matches: bool = row
        .payload_enc
        .as_slice()
        .ct_eq(registration.payload_enc.as_slice())
        .into();
    let record_mac_matches: bool = row.record_mac.as_slice().ct_eq(expected_record_mac).into();
    identity_matches
        && payload_matches
        && record_mac_matches
        && row.id > 0
        && expected_existing_row_id.is_none_or(|expected| row.id == expected)
        && row.user_id == registration.user_id
        && row.project_id == registration.project_id
        && row.uuid == registration.registration_id
        && row.device_id == registration.device_id
        && row.installation_id == registration.installation_id
        && row.endpoint_epoch == registration.endpoint_epoch
        && row.payload_version == registration.payload_version
        && row.revision == registration.revision
}

fn lock_maple_user_and_validate_credential(
    conn: &mut PgConnection,
    authorization: &MapleDeviceListAuthorization,
    exclusive: bool,
) -> Result<(), DBError> {
    use crate::models::schema::{user_seed_wrappings, users};

    let user_query = users::table
        .filter(users::uuid.eq(authorization.user_id))
        .filter(users::project_id.eq(authorization.project_id))
        .select(users::uuid);
    let locked_user = if exclusive {
        user_query.for_update().first::<Uuid>(conn)
    } else {
        // Readers may coexist, but credential update/delete/reset paths take a
        // conflicting row lock so revocation still linearizes with the read.
        user_query.for_share().first::<Uuid>(conn)
    };
    locked_user.map_err(|error| match error {
        diesel::result::Error::NotFound => DBError::UserNotFound,
        other => DBError::QueryError(other),
    })?;

    let credential_kind = match authorization.auth_credential_kind.as_str() {
        "password" => CredentialKind::Password,
        "oauth" => CredentialKind::OAuth,
        _ => return Err(DBError::StaleCredentialState),
    };
    let auth_binding = AuthBinding::from_bytes(authorization.auth_binding);
    let active_wrappings = user_seed_wrappings::table
        .filter(user_seed_wrappings::user_id.eq(authorization.user_id))
        .filter(
            user_seed_wrappings::credential_kind.eq(authorization.auth_credential_kind.as_str()),
        )
        .filter(user_seed_wrappings::wrapping_version.eq(SEED_WRAP_VERSION_V1))
        .select(user_seed_wrappings::seed_enc)
        .load::<Vec<u8>>(conn)?;
    let credential_is_current = active_wrappings.iter().any(|seed_enc| {
        decrypt_seed_v1(
            &authorization.enclave_key,
            seed_enc,
            authorization.user_id,
            authorization.project_id,
            credential_kind,
            &auth_binding,
        )
        .is_ok()
    });
    if !credential_is_current {
        return Err(DBError::StaleCredentialState);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maple_device_registration_operation_receipt_mac_for_parts(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    request_mac: &[u8],
    device: &MaplePairingAuthorityDeviceSummary,
    device_revision: i64,
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    operation_lookup_digest: &[u8],
    known_security_epoch: i64,
    accepted_security_epoch: i64,
    response_kind: i16,
    sync_payload_version: i16,
    sync_payload_enc: &[u8],
    sync_issuer_key_id: &str,
    sync_digest: &[u8],
    accepted_at: DateTime<Utc>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_RECEIPT_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(operation_id)
        .append_bytes(request_mac)
        .append_i64(device.id)
        .append_uuid(device.uuid)
        .append_uuid(device.device_id)
        .append_i64(device_revision)
        .append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_bytes(operation_lookup_digest)
        .append_i64(known_security_epoch)
        .append_i64(accepted_security_epoch)
        .append_i16(response_kind)
        .append_i16(sync_payload_version)
        .append_bytes(sync_payload_enc)
        .append_str(sync_issuer_key_id)
        .append_bytes(sync_digest)
        .append_i64(accepted_at.timestamp_micros());
    maple_device_hmac(
        enclave_key,
        MAPLE_DEVICE_RECEIPT_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_device_registration_operation_receipt_mac(
    enclave_key: &[u8],
    operation: &MapleDeviceRegistrationOperation,
    device: &MaplePairingAuthorityDeviceSummary,
) -> Result<Vec<u8>, EncryptError> {
    maple_device_registration_operation_receipt_mac_for_parts(
        enclave_key,
        operation.user_id,
        operation.project_id,
        operation.operation_id,
        &operation.request_mac,
        device,
        operation.device_revision,
        &operation.authority_scope_digest,
        &operation.lookup_digest,
        &operation.operation_lookup_digest,
        operation.known_security_epoch,
        operation.accepted_security_epoch,
        operation.response_kind,
        operation.sync_payload_version,
        &operation.sync_payload_enc,
        &operation.sync_issuer_key_id,
        &operation.sync_digest,
        operation.accepted_at,
    )
}

fn validate_maple_device_registration_operation(
    enclave_key: &[u8],
    operation: &MapleDeviceRegistrationOperation,
    device: &MaplePairingAuthorityDeviceSummary,
    user_id: Uuid,
    project_id: i32,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_device_registration_operation_receipt_mac(enclave_key, operation, device)?;
    let expected_scope = maple_pairing_authority_scope_digest(enclave_key, user_id, project_id)?;
    let expected_lookup = maple_pairing_revocation_highwater_lookup_digest(
        enclave_key,
        user_id,
        project_id,
        device.installation_id,
    )?;
    let expected_operation_lookup = maple_device_registration_operation_lookup_digest(
        enclave_key,
        &expected_scope,
        operation.operation_id,
    )?;
    if operation.id <= 0
        || operation.operation_id.is_nil()
        || operation.user_id != user_id
        || operation.project_id != project_id
        || operation.request_mac.len() != 32
        || operation.maple_device_id != device.id
        || operation.device_revision <= 0
        || operation.device_revision > device.revision
        || operation.authority_scope_digest.len() != 32
        || operation.lookup_digest.len() != 32
        || operation.operation_lookup_digest.len() != 32
        || operation.known_security_epoch <= 0
        || operation.accepted_security_epoch != operation.known_security_epoch
        || !matches!(
            operation.response_kind,
            MAPLE_REGISTRATION_SYNC_READY
                | MAPLE_REGISTRATION_SYNC_REVOCATIONS_PENDING
                | MAPLE_REGISTRATION_SYNC_RESET_CLEAR_REQUIRED
        )
        || operation.sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || operation.sync_payload_enc.is_empty()
        || operation.sync_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || !maple_pairing_issuer_key_id_is_valid(&operation.sync_issuer_key_id)
        || operation.sync_digest.len() != 32
        || !bool::from(
            expected_scope
                .as_slice()
                .ct_eq(operation.authority_scope_digest.as_slice()),
        )
        || !bool::from(
            expected_lookup
                .as_slice()
                .ct_eq(operation.lookup_digest.as_slice()),
        )
        || !bool::from(
            expected_operation_lookup
                .as_slice()
                .ct_eq(operation.operation_lookup_digest.as_slice()),
        )
        || operation.receipt_mac.len() != 32
        || !bool::from(expected.as_slice().ct_eq(operation.receipt_mac.as_slice()))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

// Explicit inputs keep independent replay, MAC, and signature bindings auditable.
#[allow(clippy::too_many_arguments)]
fn replay_live_maple_device_registration_operation(
    enclave_key: &[u8],
    issuer_keyset: &MaplePairingIssuerKeySetV1,
    operation: &MapleDeviceRegistrationOperation,
    device: &MapleDevice,
    user_id: Uuid,
    subject_project_id: Uuid,
    project_id: i32,
    request_mac: &[u8],
) -> Result<MapleDeviceRegistrationReceipt, DBError> {
    use subtle::ConstantTimeEq;

    if !bool::from(operation.request_mac.as_slice().ct_eq(request_mac)) {
        return Err(DBError::MapleDeviceRegistrationConflict);
    }
    validate_maple_device_registration_operation(
        enclave_key,
        operation,
        &MaplePairingAuthorityDeviceSummary::from(device),
        user_id,
        project_id,
    )?;
    if !maple_device_record_mac_is_valid(enclave_key, device)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let sync_payload = decrypt_maple_device_sync_payload(enclave_key, operation, device)?;
    if !bool::from(
        sha256_digest(&sync_payload)
            .as_slice()
            .ct_eq(operation.sync_digest.as_slice()),
    ) {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let sync: MapleRevocationSyncV1 =
        serde_json::from_slice(&sync_payload).map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    sync.verify_against_registration(
        user_id,
        subject_project_id,
        device.uuid,
        pairing_u64_from_i64(operation.accepted_security_epoch)?,
        issuer_keyset,
    )
    .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    let claim = &sync.stream_checkpoint.host;
    let expected_identity_mac = maple_device_identity_mac_from_claim(
        enclave_key,
        user_id,
        project_id,
        &claim
            .verifying_key_bytes()
            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
    )?;
    if maple_registration_response_kind(sync.status) != operation.response_kind
        || sync.stream_checkpoint.issuer_key_id != operation.sync_issuer_key_id
        || claim.device_id != device.device_id
        || claim.installation_id != device.installation_id
        || claim.endpoint_epoch != pairing_u64_from_i64(device.endpoint_epoch)?
        || !bool::from(
            expected_identity_mac
                .as_slice()
                .ct_eq(device.identity_mac.as_slice()),
        )
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(MapleDeviceRegistrationReceipt {
        operation_id: operation.operation_id,
        registration_id: device.uuid,
        device_id: device.device_id,
        revision: operation.device_revision,
        accepted_at: operation.accepted_at,
        security_epoch: operation.accepted_security_epoch,
        response_kind: operation.response_kind,
        sync_payload_version: operation.sync_payload_version,
        sync_payload,
    })
}

fn pairing_authorization_as_device(
    authorization: &MaplePairingAuthorization,
) -> MapleDeviceListAuthorization {
    MapleDeviceListAuthorization {
        user_id: authorization.user_id,
        project_id: authorization.project_id,
        auth_credential_kind: authorization.auth_credential_kind.clone(),
        auth_binding: authorization.auth_binding,
        enclave_key: authorization.enclave_key.clone(),
    }
}

fn normalize_db_time(time: DateTime<Utc>) -> Result<DateTime<Utc>, DBError> {
    DateTime::from_timestamp_micros(time.timestamp_micros()).ok_or(DBError::MaplePairingConflict)
}

fn maple_pairing_trusted_db_now(conn: &mut PgConnection) -> Result<DateTime<Utc>, DBError> {
    normalize_db_time(
        diesel::select(diesel::dsl::sql::<diesel::sql_types::Timestamptz>(
            "CURRENT_TIMESTAMP",
        ))
        .get_result(conn)?,
    )
}

fn maple_pairing_time_is_near_trusted_now(
    candidate: DateTime<Utc>,
    trusted_now: DateTime<Utc>,
) -> bool {
    let grace = chrono::Duration::milliseconds(MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS);
    trusted_now
        .checked_sub_signed(grace)
        .is_some_and(|lower| candidate >= lower)
        && trusted_now
            .checked_add_signed(grace)
            .is_some_and(|upper| candidate <= upper)
}

fn maple_pairing_pending_is_expired(expires_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
    expires_at
        <= now
            .checked_sub_signed(chrono::Duration::milliseconds(
                MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
            ))
            .unwrap_or(DateTime::<Utc>::MIN_UTC)
}

fn maple_pairing_approval_is_timely(expires_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
    now < expires_at
        .checked_add_signed(chrono::Duration::milliseconds(
            MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
        ))
        .unwrap_or(DateTime::<Utc>::MAX_UTC)
}

fn pairing_incarnation_to_i64(value: u64) -> Result<i64, DBError> {
    i64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or(DBError::MaplePairingConflict)
}

fn pairing_u64_from_i64(value: i64) -> Result<u64, DBError> {
    u64::try_from(value).map_err(|_| DBError::MaplePairingCorrupt)
}

fn maple_pairing_hmac(
    enclave_key: &[u8],
    key_info: &[u8],
    body: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, key_info)?;
    let mut mac =
        HmacSha256::new_from_slice(&key).map_err(|_| EncryptError::KeyDerivationFailed)?;
    mac.update(body);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn maple_pairing_request_nonce_mac(
    enclave_key: &[u8],
    account_id: Uuid,
    project_id: i32,
    controller_registration_id: Uuid,
    nonce: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_REQUEST_NONCE_MAC_DOMAIN);
    body.append_uuid(account_id)
        .append_i32(project_id)
        .append_uuid(controller_registration_id)
        .append_bytes(nonce);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_REQUEST_NONCE_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_pairing_request_operation_mac(
    enclave_key: &[u8],
    transcript: &[u8],
    signature_base64: &str,
) -> Result<Vec<u8>, EncryptError> {
    let signature = STANDARD
        .decode(signature_base64)
        .map_err(|_| EncryptError::BadData)?;
    if STANDARD.encode(&signature) != signature_base64 || signature.len() != 64 {
        return Err(EncryptError::BadData);
    }
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_OPERATION_MAC_DOMAIN);
    body.append_bytes(transcript).append_bytes(&signature);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_OPERATION_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

#[derive(Clone, Copy)]
struct MaplePairingPayloadCryptoContext {
    account_id: Uuid,
    project_id: i32,
    pairing_request_id: Uuid,
    pair_id: Uuid,
    pairing_incarnation: u64,
    revocation_stream_id: Option<Uuid>,
    revocation_stream_generation: Option<u64>,
    payload_version: i16,
}

fn maple_pairing_payload_aad(
    context: MaplePairingPayloadCryptoContext,
) -> Result<Vec<u8>, EncryptError> {
    let incarnation: i64 = context
        .pairing_incarnation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    if incarnation <= 0
        || context.revocation_stream_id.is_some() != context.revocation_stream_generation.is_some()
    {
        return Err(EncryptError::BadData);
    }
    let mut aad = CanonicalBytes::new(MAPLE_PAIRING_PAYLOAD_DOMAIN);
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
    aad.append_i16(context.payload_version);
    Ok(aad.into_bytes())
}

fn encrypt_maple_pairing_payload(
    enclave_key: &[u8],
    payload: &StoredMaplePairingPayloadV1,
    context: MaplePairingPayloadCryptoContext,
) -> Result<Vec<u8>, EncryptError> {
    let plaintext = serde_json::to_vec(payload).map_err(|_| EncryptError::BadData)?;
    let key = derive_key(enclave_key, MAPLE_PAIRING_PAYLOAD_KEY_INFO)?;
    encrypt_aead_v1(&key, &plaintext, &maple_pairing_payload_aad(context)?)
}

fn decrypt_maple_pairing_payload(
    enclave_key: &[u8],
    encrypted: &[u8],
    context: MaplePairingPayloadCryptoContext,
) -> Result<StoredMaplePairingPayloadV1, EncryptError> {
    let key = derive_key(enclave_key, MAPLE_PAIRING_PAYLOAD_KEY_INFO)?;
    let plaintext = decrypt_aead_v1(&key, encrypted, &maple_pairing_payload_aad(context)?)?;
    serde_json::from_slice(&plaintext).map_err(|_| EncryptError::BadData)
}

#[derive(Clone, Copy)]
struct MaplePairingReceiptCryptoContext {
    account_id: Uuid,
    project_id: i32,
    actor_registration_id: Uuid,
    operation_id: Uuid,
    operation_kind: i16,
    pair_id: Uuid,
    pairing_revision: i64,
    receipt_version: i16,
}

fn maple_pairing_receipt_aad(context: MaplePairingReceiptCryptoContext) -> Vec<u8> {
    let mut aad = CanonicalBytes::new(MAPLE_PAIRING_RECEIPT_DOMAIN);
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

fn encrypt_maple_pairing_receipt(
    enclave_key: &[u8],
    response: &MaplePairingMutationResponse,
    context: MaplePairingReceiptCryptoContext,
) -> Result<Vec<u8>, EncryptError> {
    let plaintext = serde_json::to_vec(response).map_err(|_| EncryptError::BadData)?;
    let key = derive_key(enclave_key, MAPLE_PAIRING_RECEIPT_KEY_INFO)?;
    encrypt_aead_v1(&key, &plaintext, &maple_pairing_receipt_aad(context))
}

#[derive(Clone, Copy)]
struct MaplePairingRevocationPayloadCryptoContext {
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

fn maple_pairing_revocation_payload_aad(
    context: MaplePairingRevocationPayloadCryptoContext,
) -> Result<Vec<u8>, EncryptError> {
    let stream_generation: i64 = context
        .revocation_stream_generation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    let issuer_sequence: i64 = context
        .issuer_sequence
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    let incarnation: i64 = context
        .pairing_incarnation
        .try_into()
        .map_err(|_| EncryptError::BadData)?;
    if context.revocation_stream_id.is_nil()
        || context.event_id.is_nil()
        || stream_generation <= 0
        || issuer_sequence <= 0
        || incarnation <= 0
    {
        return Err(EncryptError::BadData);
    }
    let mut aad = CanonicalBytes::new(MAPLE_PAIRING_REVOCATION_PAYLOAD_DOMAIN);
    aad.append_uuid(context.account_id)
        .append_i32(context.project_id)
        .append_uuid(context.host_registration_id)
        .append_uuid(context.revocation_stream_id)
        .append_i64(stream_generation)
        .append_uuid(context.event_id)
        .append_i64(issuer_sequence)
        .append_uuid(context.pair_id)
        .append_i64(incarnation)
        .append_i16(context.payload_version);
    Ok(aad.into_bytes())
}

fn encrypt_maple_pairing_revocation_payload(
    enclave_key: &[u8],
    revocation: &MaplePairRevocationV1,
    context: MaplePairingRevocationPayloadCryptoContext,
) -> Result<Vec<u8>, EncryptError> {
    let plaintext = serde_json::to_vec(revocation).map_err(|_| EncryptError::BadData)?;
    let key = derive_key(enclave_key, MAPLE_PAIRING_REVOCATION_PAYLOAD_KEY_INFO)?;
    encrypt_aead_v1(
        &key,
        &plaintext,
        &maple_pairing_revocation_payload_aad(context)?,
    )
}

fn maple_pairing_issuer_key_id_is_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAPLE_PAIRING_ISSUER_KEY_ID_MAX_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
}

// The explicit field list is the canonical authenticated record contract; a
// catch-all context would make additions easier to omit at call sites.
#[allow(clippy::too_many_arguments)]
fn maple_pairing_record_mac_for_parts(
    enclave_key: &[u8],
    pair_id: Uuid,
    pairing_request_id: Uuid,
    user_id: Uuid,
    project_id: i32,
    lineage_id: i64,
    controller_maple_device_id: i64,
    host_maple_device_id: i64,
    direction: i16,
    pairing_incarnation: i64,
    state: i16,
    revision: i64,
    request_nonce_mac: &[u8],
    revocation_stream_id: Option<Uuid>,
    revocation_stream_generation: Option<i64>,
    pair_authorization_digest: Option<&[u8]>,
    ticket_issuer_key_id: &str,
    authorization_issuer_key_id: Option<&str>,
    revocation_issuer_key_id: Option<&str>,
    payload_version: i16,
    payload_enc: &[u8],
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    approved_at: Option<DateTime<Utc>>,
    activated_at: Option<DateTime<Utc>>,
    revoked_at: Option<DateTime<Utc>>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_RECORD_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(pair_id)
        .append_uuid(pairing_request_id)
        .append_i64(lineage_id)
        .append_i64(controller_maple_device_id)
        .append_i64(host_maple_device_id)
        .append_i16(direction)
        .append_i64(pairing_incarnation)
        .append_i16(state)
        .append_i64(revision)
        .append_bytes(request_nonce_mac)
        .append_bool(revocation_stream_id.is_some());
    if let Some(stream_id) = revocation_stream_id {
        body.append_uuid(stream_id);
    }
    body.append_bool(revocation_stream_generation.is_some());
    if let Some(generation) = revocation_stream_generation {
        body.append_i64(generation);
    }
    body.append_bool(pair_authorization_digest.is_some());
    if let Some(digest) = pair_authorization_digest {
        body.append_bytes(digest);
    }
    body.append_str(ticket_issuer_key_id)
        .append_bool(authorization_issuer_key_id.is_some());
    if let Some(key_id) = authorization_issuer_key_id {
        body.append_str(key_id);
    }
    body.append_bool(revocation_issuer_key_id.is_some());
    if let Some(key_id) = revocation_issuer_key_id {
        body.append_str(key_id);
    }
    body.append_i16(payload_version)
        .append_bytes(payload_enc)
        .append_i64(created_at.timestamp_micros())
        .append_i64(expires_at.timestamp_micros())
        .append_bool(approved_at.is_some());
    if let Some(time) = approved_at {
        body.append_i64(time.timestamp_micros());
    }
    body.append_bool(activated_at.is_some());
    if let Some(time) = activated_at {
        body.append_i64(time.timestamp_micros());
    }
    body.append_bool(revoked_at.is_some());
    if let Some(time) = revoked_at {
        body.append_i64(time.timestamp_micros());
    }
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_RECORD_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_pairing_record_mac_for_row(
    enclave_key: &[u8],
    row: &MaplePairing,
) -> Result<Vec<u8>, EncryptError> {
    if row.payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES {
        return Err(EncryptError::BadData);
    }
    maple_pairing_record_mac_for_parts(
        enclave_key,
        row.uuid,
        row.pairing_request_id,
        row.user_id,
        row.project_id,
        row.lineage_id,
        row.controller_maple_device_id,
        row.host_maple_device_id,
        row.direction,
        row.pairing_incarnation,
        row.state,
        row.revision,
        &row.request_nonce_mac,
        row.revocation_stream_id,
        row.revocation_stream_generation,
        row.pair_authorization_digest.as_deref(),
        &row.ticket_issuer_key_id,
        row.authorization_issuer_key_id.as_deref(),
        row.revocation_issuer_key_id.as_deref(),
        row.payload_version,
        &row.payload_enc,
        row.created_at,
        row.expires_at,
        row.approved_at,
        row.activated_at,
        row.revoked_at,
    )
}

fn maple_pairing_lifecycle_timestamps_are_ordered(row: &MaplePairing) -> bool {
    let created_at = row.created_at;
    match MaplePairingState::try_from(row.state) {
        Ok(MaplePairingState::Pending | MaplePairingState::Expired) => {
            row.approved_at.is_none() && row.activated_at.is_none() && row.revoked_at.is_none()
        }
        Ok(MaplePairingState::AwaitingHostCommit) => row.approved_at.is_some_and(|approved_at| {
            created_at <= approved_at && row.activated_at.is_none() && row.revoked_at.is_none()
        }),
        Ok(MaplePairingState::Active) => {
            row.approved_at
                .zip(row.activated_at)
                .is_some_and(|(approved_at, activated_at)| {
                    created_at <= approved_at
                        && approved_at <= activated_at
                        && row.revoked_at.is_none()
                })
        }
        Ok(MaplePairingState::Revoked) => match (row.revision, row.activated_at) {
            (3, None) => {
                row.approved_at
                    .zip(row.revoked_at)
                    .is_some_and(|(approved_at, revoked_at)| {
                        created_at <= approved_at && approved_at <= revoked_at
                    })
            }
            (4, Some(activated_at)) => {
                row.approved_at
                    .zip(row.revoked_at)
                    .is_some_and(|(approved_at, revoked_at)| {
                        created_at <= approved_at
                            && approved_at <= activated_at
                            && activated_at <= revoked_at
                    })
            }
            _ => false,
        },
        Err(()) => false,
    }
}

fn validate_maple_pairing_record(enclave_key: &[u8], row: &MaplePairing) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_record_mac_for_row(enclave_key, row)?;
    let revocation_stream_is_unset =
        row.revocation_stream_id.is_none() && row.revocation_stream_generation.is_none();
    let revocation_stream_is_set = row
        .revocation_stream_id
        .is_some_and(|stream_id| !stream_id.is_nil())
        && row
            .revocation_stream_generation
            .is_some_and(|generation| generation > 0);
    let state_shape_is_valid = match MaplePairingState::try_from(row.state) {
        Ok(MaplePairingState::Pending) => {
            row.revision == 1
                && revocation_stream_is_unset
                && row.pair_authorization_digest.is_none()
                && row.authorization_issuer_key_id.is_none()
                && row.revocation_issuer_key_id.is_none()
                && row.approved_at.is_none()
                && row.activated_at.is_none()
                && row.revoked_at.is_none()
        }
        Ok(MaplePairingState::AwaitingHostCommit) => {
            row.revision == 2
                && revocation_stream_is_set
                && row
                    .pair_authorization_digest
                    .as_ref()
                    .is_some_and(|v| v.len() == 32)
                && row.authorization_issuer_key_id.is_some()
                && row.revocation_issuer_key_id.is_none()
                && row.approved_at.is_some()
                && row.activated_at.is_none()
                && row.revoked_at.is_none()
        }
        Ok(MaplePairingState::Active) => {
            row.revision == 3
                && revocation_stream_is_set
                && row
                    .pair_authorization_digest
                    .as_ref()
                    .is_some_and(|v| v.len() == 32)
                && row.authorization_issuer_key_id.is_some()
                && row.revocation_issuer_key_id.is_none()
                && row.approved_at.is_some()
                && row.activated_at.is_some()
                && row.revoked_at.is_none()
        }
        Ok(MaplePairingState::Expired) => {
            row.revision == 2
                && revocation_stream_is_unset
                && row.pair_authorization_digest.is_none()
                && row.authorization_issuer_key_id.is_none()
                && row.revocation_issuer_key_id.is_none()
                && row.approved_at.is_none()
                && row.activated_at.is_none()
                && row.revoked_at.is_none()
        }
        Ok(MaplePairingState::Revoked) => match row.revision {
            3 => {
                revocation_stream_is_set
                    && row
                        .pair_authorization_digest
                        .as_ref()
                        .is_some_and(|v| v.len() == 32)
                    && row.authorization_issuer_key_id.is_some()
                    && row.revocation_issuer_key_id.is_some()
                    && row.approved_at.is_some()
                    && row.activated_at.is_none()
                    && row.revoked_at.is_some()
            }
            4 => {
                revocation_stream_is_set
                    && row
                        .pair_authorization_digest
                        .as_ref()
                        .is_some_and(|v| v.len() == 32)
                    && row.authorization_issuer_key_id.is_some()
                    && row.revocation_issuer_key_id.is_some()
                    && row.approved_at.is_some()
                    && row.activated_at.is_some()
                    && row.revoked_at.is_some()
            }
            _ => false,
        },
        Err(()) => false,
    };
    // Wall-clock adjustments may move a later mutation backwards relative to
    // trusted wall time, but a committed lifecycle can never move backwards
    // relative to its own prior authenticated timestamps.
    let timestamps_are_ordered =
        row.expires_at > row.created_at && maple_pairing_lifecycle_timestamps_are_ordered(row);
    if row.id <= 0
        || row.uuid.is_nil()
        || row.pairing_request_id.is_nil()
        || row.controller_maple_device_id <= 0
        || row.host_maple_device_id <= 0
        || row.controller_maple_device_id == row.host_maple_device_id
        || row.direction != 1
        || row.pairing_incarnation <= 0
        || row.revision <= 0
        || !state_shape_is_valid
        || !timestamps_are_ordered
        || row.request_nonce_mac.len() != 32
        || !maple_pairing_issuer_key_id_is_valid(&row.ticket_issuer_key_id)
        || row
            .authorization_issuer_key_id
            .as_deref()
            .is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
        || row
            .revocation_issuer_key_id
            .as_deref()
            .is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
        || row.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || row.payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingCorrupt);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maple_pairing_receipt_mac(
    enclave_key: &[u8],
    operation_id: Uuid,
    user_id: Uuid,
    project_id: i32,
    actor_maple_device_id: i64,
    operation_kind: i16,
    request_mac: &[u8],
    maple_pairing_id: i64,
    pairing_revision: i64,
    receipt_version: i16,
    receipt_enc: &[u8],
    receipt_issuer_key_id: Option<&str>,
    accepted_at: DateTime<Utc>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_RECEIPT_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(operation_id)
        .append_i64(actor_maple_device_id)
        .append_i16(operation_kind)
        .append_bytes(request_mac)
        .append_i64(maple_pairing_id)
        .append_i64(pairing_revision)
        .append_i16(receipt_version)
        .append_bytes(receipt_enc)
        .append_bool(receipt_issuer_key_id.is_some());
    if let Some(key_id) = receipt_issuer_key_id {
        body.append_str(key_id);
    }
    body.append_i64(accepted_at.timestamp_micros());
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_RECEIPT_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_pairing_revocation_record_mac(
    enclave_key: &[u8],
    event_id: Uuid,
    user_id: Uuid,
    project_id: i32,
    host_maple_device_id: i64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: i64,
    issuer_sequence: i64,
    maple_pairing_id: i64,
    pairing_incarnation: i64,
    issuer_key_id: &str,
    payload_version: i16,
    payload_enc: &[u8],
    event_digest: &[u8],
    created_at: DateTime<Utc>,
    acked_at: Option<DateTime<Utc>>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_REVOCATION_RECORD_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(event_id)
        .append_i64(host_maple_device_id)
        .append_uuid(revocation_stream_id)
        .append_i64(revocation_stream_generation)
        .append_i64(issuer_sequence)
        .append_i64(maple_pairing_id)
        .append_i64(pairing_incarnation)
        .append_str(issuer_key_id)
        .append_i16(payload_version)
        .append_bytes(payload_enc)
        .append_bytes(event_digest)
        .append_i64(created_at.timestamp_micros())
        .append_bool(acked_at.is_some());
    if let Some(time) = acked_at {
        body.append_i64(time.timestamp_micros());
    }
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_REVOCATION_RECORD_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

#[allow(clippy::too_many_arguments)] // All fence fields are explicit authenticated inputs.
fn maple_pairing_host_state_mac(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    host_maple_device_id: i64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: i64,
    last_issued_revocation_sequence: i64,
    last_acked_revocation_sequence: i64,
    revision: i64,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_HOST_STATE_MAC_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_i64(host_maple_device_id)
        .append_uuid(revocation_stream_id)
        .append_i64(revocation_stream_generation)
        .append_i64(last_issued_revocation_sequence)
        .append_i64(last_acked_revocation_sequence)
        .append_i64(revision);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_HOST_STATE_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn validate_maple_pairing_host_state(
    enclave_key: &[u8],
    state: &crate::models::maple_pairing_db::MaplePairingHostState,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_host_state_mac(
        enclave_key,
        state.user_id,
        state.project_id,
        state.host_maple_device_id,
        state.revocation_stream_id,
        state.revocation_stream_generation,
        state.last_issued_revocation_sequence,
        state.last_acked_revocation_sequence,
        state.revision,
    )?;
    if state.id <= 0
        || state.host_maple_device_id <= 0
        || state.revocation_stream_id.is_nil()
        || state.revocation_stream_generation <= 0
        || state.last_issued_revocation_sequence < 0
        || state.last_acked_revocation_sequence < 0
        || state.revision <= 0
        || !bool::from(expected.as_slice().ct_eq(state.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingCorrupt);
    }
    Ok(())
}

/// Derive the only stable lookup key retained for a host's revocation
/// allocation fence. Rotating the enclave root key requires a forward
/// migration of these digests (or a coordinated clear of local admission
/// state); otherwise the same raw tuple would enter a fresh namespace.
fn maple_pairing_revocation_highwater_lookup_digest(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    host_installation_id: Uuid,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_REVOCATION_HIGHWATER_LOOKUP_DOMAIN);
    body.append_uuid(user_id)
        .append_i32(project_id)
        .append_uuid(host_installation_id);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_REVOCATION_HIGHWATER_LOOKUP_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_pairing_authority_scope_digest(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_SCOPE_DOMAIN);
    body.append_uuid(user_id).append_i32(project_id);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_AUTHORITY_SCOPE_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_device_registration_operation_lookup_digest(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    operation_id: Uuid,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_OPERATION_LOOKUP_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_uuid(operation_id);
    maple_device_hmac(
        enclave_key,
        MAPLE_DEVICE_OPERATION_LOOKUP_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_reset_clear_ack_host_registration_lookup_digest(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    host_registration_id: Uuid,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_uuid(host_registration_id);
    maple_device_hmac(
        enclave_key,
        MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_reset_clear_ack_operation_lookup_digest(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    host_registration_lookup_digest: &[u8],
    operation_id: Uuid,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_bytes(host_registration_lookup_digest)
        .append_uuid(operation_id);
    maple_device_hmac(
        enclave_key,
        MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_KEY_INFO,
        &body.into_bytes(),
    )
}

fn maple_device_registration_tombstone_record_mac(
    enclave_key: &[u8],
    row: &MaplePairingRegistrationOperationTombstone,
) -> Result<Vec<u8>, EncryptError> {
    maple_device_registration_tombstone_record_mac_for_parts(
        enclave_key,
        &row.authority_scope_digest,
        &row.lookup_digest,
        &row.operation_lookup_digest,
        row.retired_security_epoch,
        &row.request_mac,
        row.outcome_kind,
        &row.outcome_digest,
        row.receipt_version,
        &row.receipt_enc,
        &row.receipt_digest,
        &row.referenced_issuer_key_ids,
        row.accepted_at,
        row.retired_at,
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_device_registration_tombstone_record_mac_for_parts(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    operation_lookup_digest: &[u8],
    retired_security_epoch: i64,
    request_mac: &[u8],
    outcome_kind: i16,
    outcome_digest: &[u8],
    receipt_version: i16,
    receipt_enc: &[u8],
    receipt_digest: &[u8],
    referenced_issuer_key_ids: &[String],
    accepted_at: DateTime<Utc>,
    retired_at: DateTime<Utc>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_REGISTRATION_TOMBSTONE_MAC_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_bytes(operation_lookup_digest)
        .append_i64(retired_security_epoch)
        .append_bytes(request_mac)
        .append_i16(outcome_kind)
        .append_bytes(outcome_digest)
        .append_i16(receipt_version)
        .append_bytes(receipt_enc)
        .append_bytes(receipt_digest)
        .append_u16(
            referenced_issuer_key_ids
                .len()
                .try_into()
                .unwrap_or(u16::MAX),
        );
    for key_id in referenced_issuer_key_ids {
        body.append_str(key_id);
    }
    body.append_i64(accepted_at.timestamp_micros())
        .append_i64(retired_at.timestamp_micros());
    maple_device_hmac(
        enclave_key,
        MAPLE_DEVICE_REGISTRATION_TOMBSTONE_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_device_registration_tombstone_receipt_aad(
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    operation_lookup_digest: &[u8],
    retired_security_epoch: i64,
    outcome_kind: i16,
    receipt_version: i16,
    receipt_digest: &[u8],
) -> Vec<u8> {
    let mut body = CanonicalBytes::new(MAPLE_DEVICE_REGISTRATION_TOMBSTONE_RECEIPT_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_bytes(operation_lookup_digest)
        .append_i64(retired_security_epoch)
        .append_i16(outcome_kind)
        .append_i16(receipt_version)
        .append_bytes(receipt_digest);
    body.into_bytes()
}

#[allow(clippy::too_many_arguments)]
fn encrypt_maple_device_registration_tombstone_receipt(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    operation_lookup_digest: &[u8],
    retired_security_epoch: i64,
    outcome_kind: i16,
    receipt_version: i16,
    receipt_digest: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(
        enclave_key,
        MAPLE_DEVICE_REGISTRATION_TOMBSTONE_RECEIPT_KEY_INFO,
    )?;
    encrypt_aead_v1(
        &key,
        plaintext,
        &maple_device_registration_tombstone_receipt_aad(
            authority_scope_digest,
            lookup_digest,
            operation_lookup_digest,
            retired_security_epoch,
            outcome_kind,
            receipt_version,
            receipt_digest,
        ),
    )
}

fn decrypt_maple_device_registration_tombstone_receipt(
    enclave_key: &[u8],
    row: &MaplePairingRegistrationOperationTombstone,
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(
        enclave_key,
        MAPLE_DEVICE_REGISTRATION_TOMBSTONE_RECEIPT_KEY_INFO,
    )?;
    decrypt_aead_v1(
        &key,
        &row.receipt_enc,
        &maple_device_registration_tombstone_receipt_aad(
            &row.authority_scope_digest,
            &row.lookup_digest,
            &row.operation_lookup_digest,
            row.retired_security_epoch,
            row.outcome_kind,
            row.receipt_version,
            &row.receipt_digest,
        ),
    )
}

fn maple_device_registration_operation_outcome_digest(
    operation: &MapleDeviceRegistrationOperation,
) -> Vec<u8> {
    let mut body = CanonicalBytes::new("os.maple-device-registration-retired-outcome.v1");
    body.append_bytes(&operation.lookup_digest)
        .append_bytes(&operation.operation_lookup_digest)
        .append_i64(operation.accepted_security_epoch)
        .append_i16(operation.response_kind)
        .append_i16(operation.sync_payload_version)
        .append_bytes(&operation.sync_digest)
        .append_bytes(&operation.receipt_mac)
        .append_i64(operation.accepted_at.timestamp_micros());
    Sha256::digest(body.into_bytes()).to_vec()
}

fn validate_maple_device_registration_tombstone(
    enclave_key: &[u8],
    row: &MaplePairingRegistrationOperationTombstone,
    authority_scope_digest: &[u8],
    current_security_epoch: i64,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_device_registration_tombstone_record_mac(enclave_key, row)?;
    if row.id <= 0
        || row.authority_scope_digest.len() != 32
        || !bool::from(
            row.authority_scope_digest
                .as_slice()
                .ct_eq(authority_scope_digest),
        )
        || row.lookup_digest.len() != 32
        || row.operation_lookup_digest.len() != 32
        || row.retired_security_epoch <= 0
        || row.retired_security_epoch > current_security_epoch
        || row.request_mac.len() != 32
        || !matches!(
            row.outcome_kind,
            MAPLE_REGISTRATION_SYNC_READY
                | MAPLE_REGISTRATION_SYNC_REVOCATIONS_PENDING
                | MAPLE_REGISTRATION_SYNC_RESET_CLEAR_REQUIRED
        )
        || row.outcome_digest.len() != 32
        || row.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
        || row.receipt_enc.is_empty()
        || row.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || row.receipt_digest.len() != 32
        || !maple_pairing_issuer_key_ids_are_canonical(&row.referenced_issuer_key_ids, 4)
        || row.accepted_at > row.retired_at
        || row.record_mac.len() != 32
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let receipt = decrypt_maple_device_registration_tombstone_receipt(enclave_key, row)?;
    if !bool::from(
        sha256_digest(&receipt)
            .as_slice()
            .ct_eq(row.receipt_digest.as_slice()),
    ) {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let receipt: MapleDeviceRegistrationReceipt =
        serde_json::from_slice(&receipt).map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    let expected_operation_lookup = maple_device_registration_operation_lookup_digest(
        enclave_key,
        authority_scope_digest,
        receipt.operation_id,
    )?;
    let sync: MapleRevocationSyncV1 = serde_json::from_slice(&receipt.sync_payload)
        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    let mut referenced_issuer_key_ids = BTreeSet::new();
    referenced_issuer_key_ids.insert(sync.stream_checkpoint.issuer_key_id.clone());
    if let Some(instruction) = sync.reset_clear_instruction.as_ref() {
        referenced_issuer_key_ids.insert(instruction.issuer_key_id.clone());
    }
    let referenced_issuer_key_ids = referenced_issuer_key_ids.into_iter().collect::<Vec<_>>();
    if receipt.operation_id.is_nil()
        || receipt.registration_id.is_nil()
        || receipt.device_id.is_nil()
        || receipt.revision <= 0
        || receipt.security_epoch != row.retired_security_epoch
        || receipt.response_kind != row.outcome_kind
        || receipt.accepted_at != row.accepted_at
        || receipt.sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || receipt.sync_payload.is_empty()
        || receipt.sync_payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || maple_registration_response_kind(sync.status) != row.outcome_kind
        || !bool::from(
            expected_operation_lookup
                .as_slice()
                .ct_eq(row.operation_lookup_digest.as_slice()),
        )
        || referenced_issuer_key_ids != row.referenced_issuer_key_ids
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

// Explicit inputs bind the request, account, project, lookup, epoch, and issuer.
#[allow(clippy::too_many_arguments)]
fn replay_maple_device_registration_tombstone(
    enclave_key: &[u8],
    issuer_keyset: &MaplePairingIssuerKeySetV1,
    row: &MaplePairingRegistrationOperationTombstone,
    user_id: Uuid,
    subject_project_id: Uuid,
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    current_security_epoch: i64,
    request_mac: &[u8],
) -> Result<MapleDeviceRegistrationReceipt, DBError> {
    use subtle::ConstantTimeEq;

    validate_maple_device_registration_tombstone(
        enclave_key,
        row,
        authority_scope_digest,
        current_security_epoch,
    )?;
    if !bool::from(row.request_mac.as_slice().ct_eq(request_mac)) {
        return Err(DBError::MapleDeviceRegistrationConflict);
    }
    if !bool::from(row.lookup_digest.as_slice().ct_eq(lookup_digest)) {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let plaintext = decrypt_maple_device_registration_tombstone_receipt(enclave_key, row)?;
    let receipt: MapleDeviceRegistrationReceipt =
        serde_json::from_slice(&plaintext).map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    let sync: MapleRevocationSyncV1 = serde_json::from_slice(&receipt.sync_payload)
        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    sync.verify_against_registration(
        user_id,
        subject_project_id,
        receipt.registration_id,
        receipt
            .security_epoch
            .try_into()
            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
        issuer_keyset,
    )
    .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
    Ok(receipt)
}

fn maple_pairing_issuer_key_ids_are_canonical(key_ids: &[String], maximum: usize) -> bool {
    !key_ids.is_empty()
        && key_ids.len() <= maximum
        && key_ids
            .iter()
            .all(|key_id| maple_pairing_issuer_key_id_is_valid(key_id))
        && key_ids.windows(2).all(|pair| pair[0] < pair[1])
}

fn maple_installation_retirement_record_mac(
    enclave_key: &[u8],
    row: &MaplePairingInstallationRetirement,
) -> Result<Vec<u8>, EncryptError> {
    maple_installation_retirement_record_mac_for_parts(
        enclave_key,
        &row.authority_scope_digest,
        &row.lookup_digest,
        &row.host_identity_mac,
        row.retired_security_epoch,
        row.final_obligation_event_id,
        &row.final_instruction_digest,
        &row.final_chain_digest,
        &row.ack_host_registration_lookup_digest,
        &row.ack_operation_lookup_digest,
        &row.ack_request_mac,
        row.ack_receipt_version,
        &row.ack_receipt_issuer_key_id,
        &row.ack_receipt_digest,
        row.retired_at,
        row.created_at,
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_installation_retirement_record_mac_for_parts(
    enclave_key: &[u8],
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    host_identity_mac: &[u8],
    retired_security_epoch: i64,
    final_obligation_event_id: Uuid,
    final_instruction_digest: &[u8],
    final_chain_digest: &[u8],
    ack_host_registration_lookup_digest: &[u8],
    ack_operation_lookup_digest: &[u8],
    ack_request_mac: &[u8],
    ack_receipt_version: i16,
    ack_receipt_issuer_key_id: &str,
    ack_receipt_digest: &[u8],
    retired_at: DateTime<Utc>,
    created_at: DateTime<Utc>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_INSTALLATION_RETIREMENT_MAC_DOMAIN);
    body.append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_bytes(host_identity_mac)
        .append_i64(retired_security_epoch)
        .append_uuid(final_obligation_event_id)
        .append_bytes(final_instruction_digest)
        .append_bytes(final_chain_digest)
        .append_bytes(ack_host_registration_lookup_digest)
        .append_bytes(ack_operation_lookup_digest)
        .append_bytes(ack_request_mac)
        .append_i16(ack_receipt_version)
        .append_str(ack_receipt_issuer_key_id)
        .append_bytes(ack_receipt_digest)
        .append_i64(retired_at.timestamp_micros())
        .append_i64(created_at.timestamp_micros());
    maple_device_hmac(
        enclave_key,
        MAPLE_INSTALLATION_RETIREMENT_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn validate_maple_installation_retirement(
    enclave_key: &[u8],
    row: &MaplePairingInstallationRetirement,
    authority_scope_digest: &[u8],
    current_security_epoch: i64,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_installation_retirement_record_mac(enclave_key, row)?;
    if row.id <= 0
        || row.authority_scope_digest.len() != 32
        || !bool::from(
            row.authority_scope_digest
                .as_slice()
                .ct_eq(authority_scope_digest),
        )
        || row.lookup_digest.len() != 32
        || row.host_identity_mac.len() != 32
        || row.retired_security_epoch <= 0
        || row.retired_security_epoch > current_security_epoch
        || row.final_obligation_event_id.is_nil()
        || row.final_instruction_digest.len() != 32
        || row.final_chain_digest.len() != 32
        || row.ack_host_registration_lookup_digest.len() != 32
        || row.ack_operation_lookup_digest.len() != 32
        || row.ack_request_mac.len() != 32
        || row.ack_receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
        || !maple_pairing_issuer_key_id_is_valid(&row.ack_receipt_issuer_key_id)
        || row.ack_receipt_digest.len() != 32
        || row.retired_at != row.created_at
        || row.record_mac.len() != 32
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn replay_maple_reset_clear_ack_in_transaction(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    head: &MaplePairingAuthorityAccountHead,
    host_registration_id: Uuid,
    operation_id: Uuid,
    request_mac: &[u8],
) -> Result<Option<MaplePairingOperationReceipt>, DBError> {
    use crate::models::schema::{
        maple_pairing_installation_retirements, maple_pairing_reset_clear_obligations,
    };
    use diesel::OptionalExtension;
    use subtle::ConstantTimeEq;

    if host_registration_id.is_nil() || operation_id.is_nil() || request_mac.len() != 32 {
        return Err(DBError::MaplePairingConflict);
    }
    let host_registration_lookup_digest = maple_reset_clear_ack_host_registration_lookup_digest(
        &authorization.enclave_key,
        &head.authority_scope_digest,
        host_registration_id,
    )?;
    let operation_lookup_digest = maple_reset_clear_ack_operation_lookup_digest(
        &authorization.enclave_key,
        &head.authority_scope_digest,
        &host_registration_lookup_digest,
        operation_id,
    )?;
    let retirement = maple_pairing_installation_retirements::table
        .filter(
            maple_pairing_installation_retirements::authority_scope_digest
                .eq(&head.authority_scope_digest),
        )
        .filter(
            maple_pairing_installation_retirements::ack_host_registration_lookup_digest
                .eq(&host_registration_lookup_digest),
        )
        .filter(
            maple_pairing_installation_retirements::ack_operation_lookup_digest
                .eq(&operation_lookup_digest),
        )
        .for_share()
        .first::<MaplePairingInstallationRetirement>(conn)
        .optional()?;
    let Some(retirement) = retirement else {
        return Ok(None);
    };
    validate_maple_installation_retirement(
        &authorization.enclave_key,
        &retirement,
        &head.authority_scope_digest,
        head.security_epoch,
    )?;
    if !bool::from(
        retirement
            .ack_host_registration_lookup_digest
            .as_slice()
            .ct_eq(host_registration_lookup_digest.as_slice()),
    ) || !bool::from(retirement.ack_request_mac.as_slice().ct_eq(request_mac))
    {
        return Err(DBError::MaplePairingConflict);
    }
    let obligation = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&head.authority_scope_digest),
        )
        .filter(
            maple_pairing_reset_clear_obligations::uuid.eq(retirement.final_obligation_event_id),
        )
        .for_share()
        .first::<MaplePairingResetClearObligation>(conn)?;
    validate_maple_pairing_reset_clear_obligation(
        &authorization.enclave_key,
        &obligation,
        &head.authority_scope_digest,
    )?;
    let receipt_enc = obligation
        .ack_receipt_enc
        .clone()
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    if obligation.state != 2
        || obligation.revision != 3
        || obligation.acked_by_head_event_id != Some(obligation.uuid)
        || obligation.ack_operation_id != Some(operation_id)
        || obligation.ack_host_registration_lookup_digest.as_deref()
            != Some(retirement.ack_host_registration_lookup_digest.as_slice())
        || obligation.ack_request_mac.as_deref() != Some(request_mac)
        || obligation.ack_receipt_version != Some(retirement.ack_receipt_version)
        || obligation.ack_receipt_issuer_key_id.as_deref()
            != Some(retirement.ack_receipt_issuer_key_id.as_str())
        || obligation.ack_receipt_digest.as_deref()
            != Some(retirement.ack_receipt_digest.as_slice())
        || !bool::from(
            sha256_digest(&receipt_enc)
                .as_slice()
                .ct_eq(retirement.ack_receipt_digest.as_slice()),
        )
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(Some(MaplePairingOperationReceipt {
        operation_id,
        pair_id: obligation.uuid,
        pairing_revision: obligation.revision,
        receipt_version: retirement.ack_receipt_version,
        receipt_enc,
        accepted_at: obligation
            .acked_at
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
    }))
}

// Explicit inputs retain retirement scope, device, epoch, and time at the boundary.
#[allow(clippy::too_many_arguments)]
fn tombstone_maple_registration_operations_for_retirement(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    authority_scope_digest: &[u8],
    device: &MapleDevice,
    retired_security_epoch: i64,
    retired_at: DateTime<Utc>,
) -> Result<(), DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_pairing_registration_operation_tombstones,
    };
    use subtle::ConstantTimeEq;

    let mut cursor = 0_i64;
    loop {
        let page = maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(user_id))
            .filter(maple_device_registration_operations::project_id.eq(project_id))
            .filter(maple_device_registration_operations::maple_device_id.eq(device.id))
            .filter(maple_device_registration_operations::id.gt(cursor))
            .order(maple_device_registration_operations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .for_update()
            .load::<MapleDeviceRegistrationOperation>(conn)?;
        if page.is_empty() {
            break;
        }
        for operation in page {
            cursor = operation.id;
            validate_maple_device_registration_operation(
                enclave_key,
                &operation,
                &MaplePairingAuthorityDeviceSummary::from(device),
                user_id,
                project_id,
            )?;
            if operation.accepted_security_epoch != retired_security_epoch
                || operation.accepted_at > retired_at
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let sync_payload = decrypt_maple_device_sync_payload(enclave_key, &operation, device)?;
            if !bool::from(
                sha256_digest(&sync_payload)
                    .as_slice()
                    .ct_eq(operation.sync_digest.as_slice()),
            ) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let sync: MapleRevocationSyncV1 = serde_json::from_slice(&sync_payload)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if maple_registration_response_kind(sync.status) != operation.response_kind
                || sync.stream_checkpoint.issuer_key_id != operation.sync_issuer_key_id
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut referenced_issuer_key_ids = BTreeSet::new();
            referenced_issuer_key_ids.insert(sync.stream_checkpoint.issuer_key_id.clone());
            if let Some(instruction) = sync.reset_clear_instruction.as_ref() {
                referenced_issuer_key_ids.insert(instruction.issuer_key_id.clone());
            }
            let referenced_issuer_key_ids =
                referenced_issuer_key_ids.into_iter().collect::<Vec<_>>();
            if !maple_pairing_issuer_key_ids_are_canonical(&referenced_issuer_key_ids, 4) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let receipt = MapleDeviceRegistrationReceipt {
                operation_id: operation.operation_id,
                registration_id: device.uuid,
                device_id: device.device_id,
                revision: operation.device_revision,
                accepted_at: operation.accepted_at,
                security_epoch: operation.accepted_security_epoch,
                response_kind: operation.response_kind,
                sync_payload_version: operation.sync_payload_version,
                sync_payload,
            };
            let plaintext =
                serde_json::to_vec(&receipt).map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1;
            let receipt_digest = sha256_digest(&plaintext).to_vec();
            let receipt_enc = encrypt_maple_device_registration_tombstone_receipt(
                enclave_key,
                authority_scope_digest,
                &operation.lookup_digest,
                &operation.operation_lookup_digest,
                retired_security_epoch,
                operation.response_kind,
                receipt_version,
                &receipt_digest,
                &plaintext,
            )?;
            let outcome_digest = maple_device_registration_operation_outcome_digest(&operation);
            let record_mac = maple_device_registration_tombstone_record_mac_for_parts(
                enclave_key,
                authority_scope_digest,
                &operation.lookup_digest,
                &operation.operation_lookup_digest,
                retired_security_epoch,
                &operation.request_mac,
                operation.response_kind,
                &outcome_digest,
                receipt_version,
                &receipt_enc,
                &receipt_digest,
                &referenced_issuer_key_ids,
                operation.accepted_at,
                retired_at,
            )?;
            diesel::insert_into(maple_pairing_registration_operation_tombstones::table)
                .values(NewMaplePairingRegistrationOperationTombstone {
                    authority_scope_digest: authority_scope_digest.to_vec(),
                    lookup_digest: operation.lookup_digest,
                    operation_lookup_digest: operation.operation_lookup_digest,
                    retired_security_epoch,
                    request_mac: operation.request_mac,
                    outcome_kind: operation.response_kind,
                    outcome_digest,
                    receipt_version,
                    receipt_enc,
                    receipt_digest,
                    referenced_issuer_key_ids,
                    accepted_at: operation.accepted_at,
                    record_mac,
                    retired_at,
                })
                .execute(conn)
                .map_err(map_maple_device_write_error)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn acknowledge_pending_maple_reset_clear(
    conn: &mut PgConnection,
    ack: &MaplePairingRevocationAck,
    head: &MaplePairingAuthorityAccountHead,
    host: &MapleDevice,
    highwater: &MaplePairingRevocationHighwater,
    state: &MaplePairingHostState,
    pending: &MaplePairingResetClearObligation,
) -> Result<MaplePairingOperationReceipt, DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_host_states,
        maple_pairing_installation_retirements, maple_pairing_reset_clear_obligations,
        maple_pairings,
    };
    use subtle::ConstantTimeEq;

    let authorization = &ack.authorization;
    let expected_generation = i64::try_from(ack.revocation_stream_generation)
        .map_err(|_| DBError::MaplePairingConflict)?;
    let signed_event_digest = pending
        .signed_instruction_digest
        .as_deref()
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    if pending.state != 1
        || pending.revision != 2
        || pending.uuid != ack.event_id
        || pending.target_security_epoch != head.security_epoch
        || pending.target_revocation_stream_id != ack.revocation_stream_id
        || pending.target_revocation_stream_generation != expected_generation
        || pending.target_instruction_sequence != 1
        || ack.issuer_sequence != 1
        || ack.expected_previous_issuer_sequence != 0
        || ack.host_registration_id != host.uuid
        || ack.request_mac.len() != 32
        || ack.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
        || ack.receipt_enc.is_empty()
        || ack.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || !maple_pairing_issuer_key_id_is_valid(&ack.checkpoint_issuer_key_id)
        || !bool::from(signed_event_digest.ct_eq(ack.event_digest.as_slice()))
        || !bool::from(
            pending
                .host_identity_mac
                .as_slice()
                .ct_eq(host.identity_mac.as_slice()),
        )
        || highwater.revocation_stream_id != pending.target_revocation_stream_id
        || highwater.revocation_stream_generation != pending.target_revocation_stream_generation
        || highwater.security_epoch != pending.target_security_epoch
        || highwater.last_issued_revocation_sequence != 1
        || state.revocation_stream_id != highwater.revocation_stream_id
        || state.revocation_stream_generation != highwater.revocation_stream_generation
        || state.last_issued_revocation_sequence != 1
        || state.last_acked_revocation_sequence != 0
    {
        return Err(DBError::MaplePairingConflict);
    }

    let accepted_at = normalize_db_time(ack.accepted_at)?;
    let trusted_now = maple_pairing_trusted_db_now(conn)?;
    let max_registration_acceptance = maple_device_registration_operations::table
        .filter(maple_device_registration_operations::maple_device_id.eq(host.id))
        .select(diesel::dsl::max(
            maple_device_registration_operations::accepted_at,
        ))
        .first::<Option<DateTime<Utc>>>(conn)?;
    let lifecycle_floor = max_registration_acceptance
        .map_or(pending.reset_at, |accepted| accepted.max(pending.reset_at));
    if accepted_at < lifecycle_floor
        || (accepted_at != lifecycle_floor
            && !maple_pairing_time_is_near_trusted_now(accepted_at, trusted_now))
    {
        return Err(DBError::MaplePairingConflict);
    }
    let receipt_digest = sha256_digest(&ack.receipt_enc).to_vec();
    let host_registration_lookup_digest = maple_reset_clear_ack_host_registration_lookup_digest(
        &authorization.enclave_key,
        &head.authority_scope_digest,
        host.uuid,
    )?;

    let chain = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&head.authority_scope_digest),
        )
        .filter(maple_pairing_reset_clear_obligations::lookup_digest.eq(&pending.lookup_digest))
        .order(maple_pairing_reset_clear_obligations::reset_generation.asc())
        .for_update()
        .load::<MaplePairingResetClearObligation>(conn)?;
    if chain.is_empty()
        || chain.last().map(|row| row.uuid) != Some(pending.uuid)
        || chain.len() as i64 != pending.reset_generation
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    for mut obligation in chain {
        validate_maple_pairing_reset_clear_obligation(
            &authorization.enclave_key,
            &obligation,
            &head.authority_scope_digest,
        )?;
        if obligation.state != 1 {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let prior_revision = obligation.revision;
        let is_head = obligation.uuid == pending.uuid;
        obligation.state = 2;
        obligation.revision = if obligation.signed_instruction_payload_version.is_some() {
            3
        } else {
            2
        };
        obligation.acked_by_head_event_id = Some(pending.uuid);
        obligation.acked_at = Some(accepted_at);
        if is_head {
            obligation.ack_operation_id = Some(ack.operation_id);
            obligation.ack_host_registration_lookup_digest =
                Some(host_registration_lookup_digest.clone());
            obligation.ack_request_mac = Some(ack.request_mac.clone());
            obligation.ack_receipt_version = Some(ack.receipt_version);
            obligation.ack_receipt_enc = Some(ack.receipt_enc.clone());
            obligation.ack_receipt_issuer_key_id = Some(ack.checkpoint_issuer_key_id.clone());
            obligation.ack_receipt_digest = Some(receipt_digest.clone());
        }
        obligation.record_mac = maple_pairing_reset_clear_obligation_record_mac(
            &authorization.enclave_key,
            &obligation,
        )?;
        let changed = diesel::update(
            maple_pairing_reset_clear_obligations::table
                .filter(maple_pairing_reset_clear_obligations::id.eq(obligation.id))
                .filter(maple_pairing_reset_clear_obligations::state.eq(1_i16))
                .filter(maple_pairing_reset_clear_obligations::revision.eq(prior_revision)),
        )
        .set((
            maple_pairing_reset_clear_obligations::state.eq(obligation.state),
            maple_pairing_reset_clear_obligations::revision.eq(obligation.revision),
            maple_pairing_reset_clear_obligations::acked_by_head_event_id
                .eq(obligation.acked_by_head_event_id),
            maple_pairing_reset_clear_obligations::acked_at.eq(obligation.acked_at),
            maple_pairing_reset_clear_obligations::ack_operation_id.eq(obligation.ack_operation_id),
            maple_pairing_reset_clear_obligations::ack_host_registration_lookup_digest
                .eq(&obligation.ack_host_registration_lookup_digest),
            maple_pairing_reset_clear_obligations::ack_request_mac.eq(&obligation.ack_request_mac),
            maple_pairing_reset_clear_obligations::ack_receipt_version
                .eq(obligation.ack_receipt_version),
            maple_pairing_reset_clear_obligations::ack_receipt_enc.eq(&obligation.ack_receipt_enc),
            maple_pairing_reset_clear_obligations::ack_receipt_issuer_key_id
                .eq(&obligation.ack_receipt_issuer_key_id),
            maple_pairing_reset_clear_obligations::ack_receipt_digest
                .eq(&obligation.ack_receipt_digest),
            maple_pairing_reset_clear_obligations::record_mac.eq(&obligation.record_mac),
        ))
        .execute(conn)?;
        if changed != 1 {
            return Err(DBError::MaplePairingConflict);
        }
    }

    let target_state_revision = state
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    let state_mac = maple_pairing_host_state_mac(
        &authorization.enclave_key,
        state.user_id,
        state.project_id,
        state.host_maple_device_id,
        state.revocation_stream_id,
        state.revocation_stream_generation,
        state.last_issued_revocation_sequence,
        1,
        target_state_revision,
    )?;
    let changed_state = diesel::update(
        maple_pairing_host_states::table
            .filter(maple_pairing_host_states::id.eq(state.id))
            .filter(maple_pairing_host_states::revision.eq(state.revision)),
    )
    .set((
        maple_pairing_host_states::last_acked_revocation_sequence.eq(1_i64),
        maple_pairing_host_states::revision.eq(target_state_revision),
        maple_pairing_host_states::record_mac.eq(state_mac),
    ))
    .execute(conn)?;
    if changed_state != 1 {
        return Err(DBError::MaplePairingConflict);
    }

    tombstone_maple_registration_operations_for_retirement(
        conn,
        &authorization.enclave_key,
        authorization.user_id,
        authorization.project_id,
        &head.authority_scope_digest,
        host,
        head.security_epoch,
        accepted_at,
    )?;
    let operation_lookup_digest = maple_reset_clear_ack_operation_lookup_digest(
        &authorization.enclave_key,
        &head.authority_scope_digest,
        &host_registration_lookup_digest,
        ack.operation_id,
    )?;
    let retirement_mac = maple_installation_retirement_record_mac_for_parts(
        &authorization.enclave_key,
        &head.authority_scope_digest,
        &pending.lookup_digest,
        &pending.host_identity_mac,
        head.security_epoch,
        pending.uuid,
        &pending.instruction_digest,
        &pending.chain_digest,
        &host_registration_lookup_digest,
        &operation_lookup_digest,
        &ack.request_mac,
        ack.receipt_version,
        &ack.checkpoint_issuer_key_id,
        &receipt_digest,
        accepted_at,
        accepted_at,
    )?;
    let retirement = diesel::insert_into(maple_pairing_installation_retirements::table)
        .values(NewMaplePairingInstallationRetirement {
            authority_scope_digest: head.authority_scope_digest.clone(),
            lookup_digest: pending.lookup_digest.clone(),
            host_identity_mac: pending.host_identity_mac.clone(),
            retired_security_epoch: head.security_epoch,
            final_obligation_event_id: pending.uuid,
            final_instruction_digest: pending.instruction_digest.clone(),
            final_chain_digest: pending.chain_digest.clone(),
            ack_host_registration_lookup_digest: host_registration_lookup_digest,
            ack_operation_lookup_digest: operation_lookup_digest,
            ack_request_mac: ack.request_mac.clone(),
            ack_receipt_version: ack.receipt_version,
            ack_receipt_issuer_key_id: ack.checkpoint_issuer_key_id.clone(),
            ack_receipt_digest: receipt_digest,
            retired_at: accepted_at,
            record_mac: retirement_mac,
            created_at: accepted_at,
        })
        .get_result::<MaplePairingInstallationRetirement>(conn)
        .map_err(map_maple_device_write_error)?;
    validate_maple_installation_retirement(
        &authorization.enclave_key,
        &retirement,
        &head.authority_scope_digest,
        head.security_epoch,
    )?;

    let referencing_pair = maple_pairings::table
        .filter(
            maple_pairings::controller_maple_device_id
                .eq(host.id)
                .or(maple_pairings::host_maple_device_id.eq(host.id)),
        )
        .count()
        .get_result::<i64>(conn)?;
    if referencing_pair != 0 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    diesel::delete(
        maple_pairing_host_states::table.filter(maple_pairing_host_states::id.eq(state.id)),
    )
    .execute(conn)?;
    diesel::delete(
        maple_device_registration_operations::table
            .filter(maple_device_registration_operations::maple_device_id.eq(host.id)),
    )
    .execute(conn)?;
    let removed =
        diesel::delete(maple_devices::table.filter(maple_devices::id.eq(host.id))).execute(conn)?;
    if removed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(MaplePairingOperationReceipt {
        operation_id: ack.operation_id,
        pair_id: pending.uuid,
        pairing_revision: 3,
        receipt_version: ack.receipt_version,
        receipt_enc: ack.receipt_enc.clone(),
        accepted_at,
    })
}

fn append_optional_uuid(body: &mut CanonicalBytes, value: Option<Uuid>) {
    body.append_bool(value.is_some());
    if let Some(value) = value {
        body.append_uuid(value);
    }
}

fn append_optional_bytes(body: &mut CanonicalBytes, value: Option<&[u8]>) {
    body.append_bool(value.is_some());
    if let Some(value) = value {
        body.append_bytes(value);
    }
}

fn append_optional_str(body: &mut CanonicalBytes, value: Option<&str>) {
    body.append_bool(value.is_some());
    if let Some(value) = value {
        body.append_str(value);
    }
}

fn append_optional_i16(body: &mut CanonicalBytes, value: Option<i16>) {
    body.append_bool(value.is_some());
    if let Some(value) = value {
        body.append_i16(value);
    }
}

fn append_optional_time(body: &mut CanonicalBytes, value: Option<DateTime<Utc>>) {
    body.append_bool(value.is_some());
    if let Some(value) = value {
        body.append_i64(value.timestamp_micros());
    }
}

#[derive(Clone, Copy)]
enum MapleResetClearPayloadKind {
    HostClaim,
    InstructionMaterial,
    SignedInstruction,
    Sync,
}

impl MapleResetClearPayloadKind {
    fn label(self) -> &'static str {
        match self {
            Self::HostClaim => "host_claim",
            Self::InstructionMaterial => "instruction_material",
            Self::SignedInstruction => "signed_instruction",
            Self::Sync => "sync",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn maple_reset_clear_payload_aad(
    kind: MapleResetClearPayloadKind,
    event_id: Uuid,
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    instruction_material_digest: &[u8],
    chain_digest: &[u8],
    payload_version: i16,
    issuer_key_id: Option<&str>,
    payload_digest: &[u8],
) -> Vec<u8> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_RESET_CLEAR_PAYLOAD_DOMAIN);
    body.append_str(kind.label())
        .append_uuid(event_id)
        .append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_bytes(instruction_material_digest)
        .append_bytes(chain_digest)
        .append_i16(payload_version);
    append_optional_str(&mut body, issuer_key_id);
    body.append_bytes(payload_digest);
    body.into_bytes()
}

#[allow(clippy::too_many_arguments)]
fn encrypt_maple_reset_clear_payload(
    enclave_key: &[u8],
    kind: MapleResetClearPayloadKind,
    event_id: Uuid,
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    instruction_material_digest: &[u8],
    chain_digest: &[u8],
    payload_version: i16,
    issuer_key_id: Option<&str>,
    payload_digest: &[u8],
    payload: &[u8],
) -> Result<Vec<u8>, EncryptError> {
    let key = derive_key(enclave_key, MAPLE_PAIRING_RESET_CLEAR_PAYLOAD_KEY_INFO)?;
    encrypt_aead_v1(
        &key,
        payload,
        &maple_reset_clear_payload_aad(
            kind,
            event_id,
            authority_scope_digest,
            lookup_digest,
            instruction_material_digest,
            chain_digest,
            payload_version,
            issuer_key_id,
            payload_digest,
        ),
    )
}

fn decrypt_maple_reset_clear_payload(
    enclave_key: &[u8],
    row: &MaplePairingResetClearObligation,
    kind: MapleResetClearPayloadKind,
) -> Result<Vec<u8>, EncryptError> {
    let (payload_version, payload_enc, issuer_key_id, payload_digest) = match kind {
        MapleResetClearPayloadKind::HostClaim => (
            row.host_claim_payload_version,
            row.host_claim_payload_enc.as_slice(),
            None,
            row.host_claim_digest.as_slice(),
        ),
        MapleResetClearPayloadKind::InstructionMaterial => (
            row.instruction_payload_version,
            row.instruction_payload_enc.as_slice(),
            None,
            row.instruction_digest.as_slice(),
        ),
        MapleResetClearPayloadKind::SignedInstruction => (
            row.signed_instruction_payload_version
                .ok_or(EncryptError::BadData)?,
            row.signed_instruction_payload_enc
                .as_deref()
                .ok_or(EncryptError::BadData)?,
            Some(
                row.signed_instruction_issuer_key_id
                    .as_deref()
                    .ok_or(EncryptError::BadData)?,
            ),
            row.signed_instruction_digest
                .as_deref()
                .ok_or(EncryptError::BadData)?,
        ),
        MapleResetClearPayloadKind::Sync => (
            row.sync_payload_version.ok_or(EncryptError::BadData)?,
            row.sync_payload_enc
                .as_deref()
                .ok_or(EncryptError::BadData)?,
            Some(
                row.sync_issuer_key_id
                    .as_deref()
                    .ok_or(EncryptError::BadData)?,
            ),
            row.sync_digest.as_deref().ok_or(EncryptError::BadData)?,
        ),
    };
    let key = derive_key(enclave_key, MAPLE_PAIRING_RESET_CLEAR_PAYLOAD_KEY_INFO)?;
    decrypt_aead_v1(
        &key,
        payload_enc,
        &maple_reset_clear_payload_aad(
            kind,
            row.uuid,
            &row.authority_scope_digest,
            &row.lookup_digest,
            &row.instruction_digest,
            &row.chain_digest,
            payload_version,
            issuer_key_id,
            payload_digest,
        ),
    )
}

fn maple_pairing_reset_clear_obligation_record_mac(
    enclave_key: &[u8],
    row: &MaplePairingResetClearObligation,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_RESET_CLEAR_OBLIGATION_MAC_DOMAIN);
    body.append_uuid(row.uuid)
        .append_bytes(&row.authority_scope_digest)
        .append_bytes(&row.lookup_digest)
        .append_bytes(&row.host_identity_mac)
        .append_uuid(row.reset_id)
        .append_i64(row.reset_generation)
        .append_i64(row.cumulative_reset_count);
    append_optional_uuid(&mut body, row.previous_event_id);
    append_optional_bytes(&mut body, row.previous_instruction_digest.as_deref());
    append_optional_bytes(&mut body, row.previous_chain_digest.as_deref());
    body.append_uuid(row.old_revocation_stream_id)
        .append_i64(row.old_revocation_stream_generation)
        .append_i64(row.source_security_epoch)
        .append_i64(row.source_last_issued_revocation_sequence)
        .append_uuid(row.target_revocation_stream_id)
        .append_i64(row.target_revocation_stream_generation)
        .append_i64(row.target_security_epoch)
        .append_i64(row.target_instruction_sequence)
        .append_i16(row.clear_scope)
        .append_bytes(&row.admission_set_digest)
        .append_i16(row.admission_count)
        .append_i16(row.host_claim_payload_version)
        .append_bytes(&row.host_claim_payload_enc)
        .append_bytes(&row.host_claim_digest)
        .append_i16(row.instruction_payload_version)
        .append_bytes(&row.instruction_payload_enc)
        .append_bytes(&row.instruction_digest)
        .append_bytes(&row.chain_digest)
        .append_i64(row.reset_at.timestamp_micros());
    append_optional_i16(&mut body, row.signed_instruction_payload_version);
    append_optional_bytes(&mut body, row.signed_instruction_payload_enc.as_deref());
    append_optional_str(&mut body, row.signed_instruction_issuer_key_id.as_deref());
    append_optional_bytes(&mut body, row.signed_instruction_digest.as_deref());
    append_optional_i16(&mut body, row.sync_payload_version);
    append_optional_bytes(&mut body, row.sync_payload_enc.as_deref());
    append_optional_str(&mut body, row.sync_issuer_key_id.as_deref());
    append_optional_bytes(&mut body, row.sync_digest.as_deref());
    body.append_i16(row.state).append_i64(row.revision);
    append_optional_uuid(&mut body, row.acked_by_head_event_id);
    append_optional_time(&mut body, row.acked_at);
    append_optional_uuid(&mut body, row.ack_operation_id);
    append_optional_bytes(
        &mut body,
        row.ack_host_registration_lookup_digest.as_deref(),
    );
    append_optional_bytes(&mut body, row.ack_request_mac.as_deref());
    append_optional_i16(&mut body, row.ack_receipt_version);
    append_optional_bytes(&mut body, row.ack_receipt_enc.as_deref());
    append_optional_str(&mut body, row.ack_receipt_issuer_key_id.as_deref());
    append_optional_bytes(&mut body, row.ack_receipt_digest.as_deref());
    body.append_i64(row.created_at.timestamp_micros());
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_RESET_CLEAR_OBLIGATION_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn validate_maple_pairing_reset_clear_obligation(
    enclave_key: &[u8],
    row: &MaplePairingResetClearObligation,
    authority_scope_digest: &[u8],
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_reset_clear_obligation_record_mac(enclave_key, row)?;
    let previous_shape = row.previous_event_id.is_some()
        == row.previous_instruction_digest.is_some()
        && row.previous_event_id.is_some() == row.previous_chain_digest.is_some();
    let materialized = row.signed_instruction_payload_version.is_some();
    let signed_shape = [
        row.signed_instruction_payload_enc.is_some(),
        row.signed_instruction_issuer_key_id.is_some(),
        row.signed_instruction_digest.is_some(),
        row.sync_payload_version.is_some(),
        row.sync_payload_enc.is_some(),
        row.sync_issuer_key_id.is_some(),
        row.sync_digest.is_some(),
    ]
    .into_iter()
    .all(|present| present == materialized);
    let direct_ack_present = row.ack_operation_id.is_some();
    let direct_ack_shape = [
        row.ack_host_registration_lookup_digest.is_some(),
        row.ack_request_mac.is_some(),
        row.ack_receipt_version.is_some(),
        row.ack_receipt_enc.is_some(),
        row.ack_receipt_issuer_key_id.is_some(),
        row.ack_receipt_digest.is_some(),
    ]
    .into_iter()
    .all(|present| present == direct_ack_present);
    let direct_ack_digest_is_valid = match (
        row.ack_receipt_enc.as_deref(),
        row.ack_receipt_digest.as_deref(),
    ) {
        (Some(ciphertext), Some(digest)) => {
            bool::from(sha256_digest(ciphertext).as_slice().ct_eq(digest))
        }
        (None, None) => true,
        _ => false,
    };
    let state_shape = match (row.state, row.revision, materialized) {
        (1, 1, false) | (1, 2, true) => {
            row.acked_by_head_event_id.is_none() && row.acked_at.is_none() && !direct_ack_present
        }
        (2, 2, false) => {
            row.acked_by_head_event_id
                .is_some_and(|head| head != row.uuid)
                && row.acked_at.is_some()
                && !direct_ack_present
        }
        (2, 3, true) => match row.acked_by_head_event_id {
            Some(head) if head == row.uuid => row.acked_at.is_some() && direct_ack_present,
            Some(_) => row.acked_at.is_some() && !direct_ack_present,
            None => false,
        },
        _ => false,
    };
    if row.id <= 0
        || row.uuid.is_nil()
        || row.reset_id.is_nil()
        || row.authority_scope_digest.len() != 32
        || !bool::from(
            row.authority_scope_digest
                .as_slice()
                .ct_eq(authority_scope_digest),
        )
        || row.lookup_digest.len() != 32
        || row.host_identity_mac.len() != 32
        || row.reset_generation <= 0
        || row.cumulative_reset_count != row.reset_generation
        || !previous_shape
        || row
            .previous_instruction_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || row
            .previous_chain_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || row
            .ack_host_registration_lookup_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || row.old_revocation_stream_id.is_nil()
        || row.target_revocation_stream_id.is_nil()
        || row.old_revocation_stream_id == row.target_revocation_stream_id
        || row.old_revocation_stream_generation <= 0
        || row.target_revocation_stream_generation
            != row.old_revocation_stream_generation.saturating_add(1)
        || row.source_security_epoch <= 0
        || row.target_security_epoch != row.source_security_epoch.saturating_add(1)
        || row.source_last_issued_revocation_sequence < 0
        || row.target_instruction_sequence != 1
        || row.clear_scope != 1
        || !(0..=MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION as i16)
            .contains(&row.admission_count)
        || row.admission_set_digest.len() != 32
        || row.host_claim_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || row.host_claim_payload_enc.is_empty()
        || row.host_claim_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || row.host_claim_digest.len() != 32
        || row.instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || row.instruction_payload_enc.is_empty()
        || row.instruction_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || row.instruction_digest.len() != 32
        || row.chain_digest.len() != 32
        || !signed_shape
        || !direct_ack_shape
        || !direct_ack_digest_is_valid
        || row
            .signed_instruction_payload_version
            .is_some_and(|version| version != MAPLE_PAIRING_PAYLOAD_VERSION_V1)
        || row
            .signed_instruction_payload_enc
            .as_ref()
            .is_some_and(|payload| {
                payload.is_empty() || payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
            })
        || row
            .signed_instruction_issuer_key_id
            .as_deref()
            .is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
        || row
            .signed_instruction_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || row
            .sync_payload_version
            .is_some_and(|version| version != MAPLE_PAIRING_PAYLOAD_VERSION_V1)
        || row.sync_payload_enc.as_ref().is_some_and(|payload| {
            payload.is_empty() || payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        })
        || row
            .sync_issuer_key_id
            .as_deref()
            .is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
        || row
            .sync_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || row
            .ack_operation_id
            .is_some_and(|operation_id| operation_id.is_nil())
        || row
            .ack_request_mac
            .as_ref()
            .is_some_and(|request_mac| request_mac.len() != 32)
        || row
            .ack_receipt_version
            .is_some_and(|version| version != MAPLE_PAIRING_RECEIPT_VERSION_V1)
        || row.ack_receipt_enc.as_ref().is_some_and(|receipt| {
            receipt.is_empty() || receipt.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        })
        || row
            .ack_receipt_issuer_key_id
            .as_deref()
            .is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
        || row
            .ack_receipt_digest
            .as_ref()
            .is_some_and(|digest| digest.len() != 32)
        || !state_shape
        || row.updated_at < row.created_at
        || row.reset_at < row.created_at
        || row.record_mac.len() != 32
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn maple_pairing_reset_clear_admission_record_mac(
    enclave_key: &[u8],
    row: &MaplePairingResetClearAdmission,
) -> Result<Vec<u8>, EncryptError> {
    maple_pairing_reset_clear_admission_record_mac_for_parts(
        enclave_key,
        row.obligation_uuid,
        &row.authority_scope_digest,
        &row.lookup_digest,
        row.pair_id,
        row.pairing_incarnation,
        &row.pair_authorization_digest,
        row.created_at,
    )
}

#[allow(clippy::too_many_arguments)]
fn maple_pairing_reset_clear_admission_record_mac_for_parts(
    enclave_key: &[u8],
    obligation_uuid: Uuid,
    authority_scope_digest: &[u8],
    lookup_digest: &[u8],
    pair_id: Uuid,
    pairing_incarnation: i64,
    pair_authorization_digest: &[u8],
    created_at: DateTime<Utc>,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_RESET_CLEAR_ADMISSION_MAC_DOMAIN);
    body.append_uuid(obligation_uuid)
        .append_bytes(authority_scope_digest)
        .append_bytes(lookup_digest)
        .append_uuid(pair_id)
        .append_i64(pairing_incarnation)
        .append_bytes(pair_authorization_digest)
        .append_i64(created_at.timestamp_micros());
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_RESET_CLEAR_ADMISSION_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn validate_maple_pairing_reset_clear_admission(
    enclave_key: &[u8],
    row: &MaplePairingResetClearAdmission,
    authority_scope_digest: &[u8],
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_reset_clear_admission_record_mac(enclave_key, row)?;
    if row.id <= 0
        || row.obligation_uuid.is_nil()
        || row.authority_scope_digest.len() != 32
        || !bool::from(
            row.authority_scope_digest
                .as_slice()
                .ct_eq(authority_scope_digest),
        )
        || row.lookup_digest.len() != 32
        || row.pair_id.is_nil()
        || row.pairing_incarnation <= 0
        || row.pair_authorization_digest.len() != 32
        || row.record_mac.len() != 32
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn maple_pairing_authority_hmac(
    enclave_key: &[u8],
    key_info: &[u8],
    body: CanonicalBytes,
) -> Result<Vec<u8>, DBError> {
    maple_pairing_hmac(enclave_key, key_info, &body.into_bytes()).map_err(DBError::from)
}

fn maple_pairing_authority_account_head_mac(
    enclave_key: &[u8],
    head: &crate::models::maple_pairing_db::MaplePairingAuthorityAccountHead,
) -> Result<Vec<u8>, DBError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_ACCOUNT_HEAD_MAC_DOMAIN);
    body.append_uuid(head.user_id)
        .append_i32(head.project_id)
        .append_i32(head.org_id)
        .append_i64(head.security_epoch)
        .append_bytes(&head.authority_scope_digest)
        .append_bytes(&head.authority_inventory_digest)
        .append_i64(head.authority_row_count)
        .append_i64(head.device_count)
        .append_i64(head.device_operation_count)
        .append_i64(head.lineage_count)
        .append_i64(head.pairing_count)
        .append_i64(head.pairing_operation_count)
        .append_i64(head.host_state_count)
        .append_i64(head.revocation_event_count)
        .append_i64(head.highwater_installation_group_count)
        .append_i64(head.highwater_generation_count)
        .append_i64(head.registration_operation_tombstone_count)
        .append_i64(head.installation_retirement_count)
        .append_i64(head.reset_clear_obligation_count)
        .append_i64(head.reset_clear_admission_count)
        .append_i64(head.revision)
        .append_i64(head.created_at.timestamp_micros());
    maple_pairing_authority_hmac(
        enclave_key,
        MAPLE_PAIRING_AUTHORITY_ACCOUNT_HEAD_MAC_KEY_INFO,
        body,
    )
}

fn maple_pairing_authority_project_head_mac(
    enclave_key: &[u8],
    head: &crate::models::maple_pairing_db::MaplePairingAuthorityProjectHead,
) -> Result<Vec<u8>, DBError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_PROJECT_HEAD_MAC_DOMAIN);
    body.append_i32(head.project_id)
        .append_i32(head.org_id)
        .append_uuid(head.project_uuid)
        .append_uuid(head.subject_project_id)
        .append_bytes(&head.account_inventory_digest)
        .append_i64(head.account_count)
        .append_i64(head.revision)
        .append_i64(head.created_at.timestamp_micros());
    maple_pairing_authority_hmac(
        enclave_key,
        MAPLE_PAIRING_AUTHORITY_PROJECT_HEAD_MAC_KEY_INFO,
        body,
    )
}

#[cfg(test)]
#[test]
fn maple_pairing_authority_project_head_mac_binds_both_public_project_aliases() {
    let created_at = DateTime::from_timestamp(1_700_000_000, 0).expect("valid fixed timestamp");
    let mut head = MaplePairingAuthorityProjectHead {
        project_id: 17,
        org_id: 23,
        project_uuid: Uuid::from_u128(29),
        subject_project_id: Uuid::from_u128(31),
        account_inventory_digest: vec![0x41; 32],
        account_count: 0,
        revision: 1,
        record_mac: vec![0; 32],
        created_at,
        updated_at: created_at,
    };
    let enclave_key = [0x53; 32];
    let original = maple_pairing_authority_project_head_mac(&enclave_key, &head)
        .expect("project head MAC should succeed");

    head.project_uuid = Uuid::from_u128(37);
    let changed_uuid = maple_pairing_authority_project_head_mac(&enclave_key, &head)
        .expect("project head MAC should bind project UUID");
    assert_ne!(original, changed_uuid);

    head.project_uuid = Uuid::from_u128(29);
    head.subject_project_id = Uuid::from_u128(41);
    let changed_client_id = maple_pairing_authority_project_head_mac(&enclave_key, &head)
        .expect("project head MAC should bind subject project ID");
    assert_ne!(original, changed_client_id);
}

fn maple_pairing_authority_org_head_mac(
    enclave_key: &[u8],
    head: &crate::models::maple_pairing_db::MaplePairingAuthorityOrgHead,
) -> Result<Vec<u8>, DBError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_ORG_HEAD_MAC_DOMAIN);
    body.append_i32(head.org_id)
        .append_bool(head.global_singleton)
        .append_bytes(&head.project_inventory_digest)
        .append_i64(head.project_count)
        .append_i64(head.revision)
        .append_i64(head.created_at.timestamp_micros());
    maple_pairing_authority_hmac(
        enclave_key,
        MAPLE_PAIRING_AUTHORITY_ORG_HEAD_MAC_KEY_INFO,
        body,
    )
}

fn maple_pairing_authority_global_head_mac(
    enclave_key: &[u8],
    head: &crate::models::maple_pairing_db::MaplePairingAuthorityGlobalHead,
) -> Result<Vec<u8>, DBError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_GLOBAL_HEAD_MAC_DOMAIN);
    body.append_bool(head.singleton)
        .append_i16(head.activation_state)
        .append_bytes(&head.org_inventory_digest)
        .append_i64(head.org_count)
        .append_bytes(&head.issuer_key_inventory_digest)
        .append_i64(head.issuer_key_count)
        .append_i64(head.revision)
        .append_i64(head.created_at.timestamp_micros());
    maple_pairing_authority_hmac(
        enclave_key,
        MAPLE_PAIRING_AUTHORITY_GLOBAL_HEAD_MAC_KEY_INFO,
        body,
    )
}

fn maple_pairing_issuer_key_record_mac_for_parts(
    enclave_key: &[u8],
    key_id: &str,
    global_singleton: bool,
    algorithm: &str,
    public_key_digest: &[u8],
    created_at: DateTime<Utc>,
) -> Result<Vec<u8>, DBError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_DOMAIN);
    body.append_str(key_id)
        .append_bool(global_singleton)
        .append_str(algorithm)
        .append_bytes(public_key_digest)
        .append_i64(created_at.timestamp_micros());
    maple_pairing_authority_hmac(
        enclave_key,
        MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_KEY_INFO,
        body,
    )
}

fn maple_pairing_issuer_key_record_mac(
    enclave_key: &[u8],
    row: &MaplePairingIssuerKey,
) -> Result<Vec<u8>, DBError> {
    maple_pairing_issuer_key_record_mac_for_parts(
        enclave_key,
        &row.key_id,
        row.global_singleton,
        &row.algorithm,
        &row.public_key_digest,
        row.created_at,
    )
}

fn validate_maple_pairing_issuer_key(
    enclave_key: &[u8],
    row: &MaplePairingIssuerKey,
) -> Result<(), DBError> {
    let expected = maple_pairing_issuer_key_record_mac(enclave_key, row)?;
    if !maple_pairing_issuer_key_id_is_valid(&row.key_id)
        || !row.global_singleton
        || row.algorithm != "ed25519"
        || row.public_key_digest.len() != 32
        || row.record_mac.len() != 32
        || !maple_pairing_authority_mac_matches(&expected, &row.record_mac)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn maple_pairing_authority_mac_matches(expected: &[u8], actual: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    bool::from(expected.ct_eq(actual))
}

fn validate_maple_pairing_authority_account_head(
    enclave_key: &[u8],
    head: &crate::models::maple_pairing_db::MaplePairingAuthorityAccountHead,
) -> Result<(), DBError> {
    let expected_scope =
        maple_pairing_authority_scope_digest(enclave_key, head.user_id, head.project_id)?;
    let expected_mac = maple_pairing_authority_account_head_mac(enclave_key, head)?;
    let counts = MaplePairingAuthorityCounts {
        devices: head.device_count,
        device_operations: head.device_operation_count,
        registration_operation_tombstones: head.registration_operation_tombstone_count,
        installation_retirements: head.installation_retirement_count,
        lineages: head.lineage_count,
        pairings: head.pairing_count,
        pairing_operations: head.pairing_operation_count,
        host_states: head.host_state_count,
        revocation_events: head.revocation_event_count,
        highwater_groups: head.highwater_installation_group_count,
        highwater_generations: head.highwater_generation_count,
        reset_clear_obligations: head.reset_clear_obligation_count,
        reset_clear_admissions: head.reset_clear_admission_count,
    };
    if head.user_id.is_nil()
        || head.project_id <= 0
        || head.org_id <= 0
        || head.security_epoch <= 0
        || head.authority_scope_digest.len() != 32
        || head.authority_inventory_digest.len() != 32
        || head.revision <= 0
        || head.device_count < 0
        || head.device_count > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT
        || head.device_operation_count < 0
        || head.registration_operation_tombstone_count < 0
        || head.installation_retirement_count < 0
        || head.installation_retirement_count
            > MAPLE_PAIRING_AUTHORITY_INSTALLATION_RETIREMENT_LIMIT
        || head
            .device_operation_count
            .checked_add(head.registration_operation_tombstone_count)
            .is_none_or(|count| {
                count
                    > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT
                        * MAPLE_DEVICE_OPERATION_LIMIT_PER_DEVICE
            })
        || head.lineage_count < 0
        || head.lineage_count > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || head.pairing_count < 0
        || head.pairing_count > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || head.pairing_operation_count < 0
        || head.pairing_operation_count
            > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT * MAPLE_PAIRING_OPERATION_LIMIT_PER_PAIRING
        || head.host_state_count < 0
        || head.host_state_count > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT
        || head.revocation_event_count < 0
        || head.revocation_event_count > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || head.highwater_installation_group_count < 0
        || head.highwater_installation_group_count > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT
        || head.highwater_generation_count < head.highwater_installation_group_count
        || head.highwater_generation_count > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
        || head.reset_clear_obligation_count < 0
        || head.reset_clear_obligation_count > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_OBLIGATION_LIMIT
        || head.reset_clear_admission_count < 0
        || head.reset_clear_admission_count > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_ADMISSION_LIMIT
        || counts.total_rows() != Some(head.authority_row_count)
        || head.updated_at < head.created_at
        || !maple_pairing_authority_mac_matches(&expected_scope, &head.authority_scope_digest)
        || !maple_pairing_authority_mac_matches(&expected_mac, &head.record_mac)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

#[derive(diesel::QueryableByName)]
struct MaplePairingAuthorityTryLockResult {
    #[diesel(sql_type = diesel::sql_types::Bool)]
    acquired: bool,
}

fn try_maple_pairing_authority_lock_once(
    conn: &mut PgConnection,
) -> Result<bool, diesel::result::Error> {
    let acquired = diesel::sql_query("SELECT pg_try_advisory_xact_lock($1, $2) AS acquired")
        .bind::<diesel::sql_types::Integer, _>(MAPLE_PAIRING_AUTHORITY_LOCK_KEY_1)
        .bind::<diesel::sql_types::Integer, _>(MAPLE_PAIRING_AUTHORITY_LOCK_KEY_2)
        .get_result::<MaplePairingAuthorityTryLockResult>(conn)
        .map(|result| result.acquired);
    #[cfg(test)]
    if matches!(&acquired, Ok(false)) {
        observe_maple_pairing_authority_lock_contention_if_armed_for_test();
    }
    acquired
}

#[derive(Clone, Copy)]
enum MaplePairingAuthoritySnapshotFenceMode {
    ActiveOnly,
    Bootstrap,
}

fn acquire_maple_pairing_authority_snapshot_fence_with_mode(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    mode: MaplePairingAuthoritySnapshotFenceMode,
    expected_issuer_key_inventory_digest: Option<&[u8]>,
) -> Result<(), DBError> {
    // This must be the first SQL statement after BEGIN. The advisory lock
    // serializes cooperating app instances; SERIALIZABLE additionally gives
    // the authenticated inventory and the authorized projection/mutation one
    // stable snapshot when an out-of-band database writer races the request.
    diesel::sql_query("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE").execute(conn)?;
    diesel::sql_query(format!(
        "SET LOCAL statement_timeout = '{MAPLE_PAIRING_AUTHORITY_STATEMENT_TIMEOUT}'"
    ))
    .execute(conn)?;
    let started = Instant::now();
    loop {
        if try_maple_pairing_authority_lock_once(conn)? {
            break;
        }
        if started.elapsed() >= MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT {
            tracing::warn!(
                event = "maple_pairing_authority_lock_busy",
                wait_ms = started.elapsed().as_millis() as u64,
                "Maple pairing authority advisory lock acquisition reached its local deadline"
            );
            return Err(DBError::MaplePairingAuthorityBusy);
        }
        std::thread::sleep(MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL);
    }
    // The first `pg_try_advisory_xact_lock` SELECT fixes the SERIALIZABLE
    // snapshot while a contended caller waits in this bounded loop. The first
    // authority row read then locks and authenticates the complete global root.
    // Every cooperating authority mutation advances this row, so PostgreSQL
    // raises 40001 here instead of allowing a stale waiter to project old
    // authority state after a newer transaction has committed.
    let global = load_maple_pairing_authority_global_head(conn)?;
    match global.activation_state {
        MAPLE_PAIRING_AUTHORITY_ACTIVE => {
            validate_maple_pairing_authority_global_head(enclave_key, &global)?;
            if matches!(mode, MaplePairingAuthoritySnapshotFenceMode::ActiveOnly) {
                let expected_issuer_key_inventory_digest = expected_issuer_key_inventory_digest
                    .filter(|digest| digest.len() == 32)
                    .ok_or(DBError::MaplePairingIssuerConfigurationConflict)?;
                if !maple_pairing_authority_mac_matches(
                    expected_issuer_key_inventory_digest,
                    &global.issuer_key_inventory_digest,
                ) {
                    return Err(DBError::MaplePairingIssuerConfigurationConflict);
                }
            }
            if !AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)? {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
        MAPLE_PAIRING_AUTHORITY_PENDING
            if matches!(mode, MaplePairingAuthoritySnapshotFenceMode::Bootstrap) =>
        {
            validate_pending_maple_pairing_authority_global_head(&global)?;
            if AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)?
                || !maple_pairing_authority_leaf_tables_are_empty(conn)?
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
        _ => return Err(DBError::MaplePairingAuthorityCorrupt),
    }
    let wait = started.elapsed();
    if wait >= Duration::from_millis(250) {
        tracing::warn!(
            event = "maple_pairing_authority_lock_slow",
            wait_ms = wait.as_millis() as u64,
            "Maple pairing authority advisory lock acquisition was slow"
        );
    } else {
        tracing::debug!(
            event = "maple_pairing_authority_lock_acquired",
            wait_ms = wait.as_millis() as u64,
            "Maple pairing authority advisory lock acquired"
        );
    }
    Ok(())
}

/// Acquire the global authority lock and authenticate an Active global root
/// before any account, project, organization, device, or pairing row is read.
fn acquire_maple_pairing_authority_snapshot_fence(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    expected_issuer_key_inventory_digest: &[u8],
) -> Result<(), DBError> {
    acquire_maple_pairing_authority_snapshot_fence_with_mode(
        conn,
        enclave_key,
        MaplePairingAuthoritySnapshotFenceMode::ActiveOnly,
        Some(expected_issuer_key_inventory_digest),
    )
}

/// Startup is the only path allowed to observe the exact Pending sentinel. It
/// additionally proves the activation marker and all authority leaves/heads
/// are absent before bootstrap may inspect parent scopes.
fn acquire_maple_pairing_authority_bootstrap_snapshot_fence(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(), DBError> {
    acquire_maple_pairing_authority_snapshot_fence_with_mode(
        conn,
        enclave_key,
        MaplePairingAuthoritySnapshotFenceMode::Bootstrap,
        None,
    )
}

struct MaplePairingAuthorityTransactionTimer {
    operation: &'static str,
    started: Instant,
}

impl MaplePairingAuthorityTransactionTimer {
    fn start(operation: &'static str) -> Self {
        Self {
            operation,
            started: Instant::now(),
        }
    }
}

impl Drop for MaplePairingAuthorityTransactionTimer {
    fn drop(&mut self) {
        let duration = self.started.elapsed();
        if duration >= Duration::from_secs(1) {
            tracing::warn!(
                event = "maple_pairing_authority_transaction_slow",
                operation = self.operation,
                duration_ms = duration.as_millis() as u64,
                "Maple pairing authority transaction was slow"
            );
        } else {
            tracing::debug!(
                event = "maple_pairing_authority_transaction_complete",
                operation = self.operation,
                duration_ms = duration.as_millis() as u64,
                "Maple pairing authority transaction completed"
            );
        }
    }
}

/// Project aliases returned only after the account, project, organization, and
/// global authority chain has verified under the transaction-wide lock. The
/// private fields and constructor prevent request DTOs or ordinary project rows
/// from being substituted for authenticated authority state.
#[must_use]
pub(crate) struct MaplePairingAuthenticatedProjectIdentity {
    project_id: i32,
    org_id: i32,
    project_uuid: Uuid,
    subject_project_id: Uuid,
    _timer: MaplePairingAuthorityTransactionTimer,
}

impl MaplePairingAuthenticatedProjectIdentity {
    fn from_verified_head(
        head: &MaplePairingAuthorityProjectHead,
        timer: MaplePairingAuthorityTransactionTimer,
    ) -> Self {
        Self {
            project_id: head.project_id,
            org_id: head.org_id,
            project_uuid: head.project_uuid,
            subject_project_id: head.subject_project_id,
            _timer: timer,
        }
    }

    pub(crate) fn subject_project_id(&self) -> Uuid {
        debug_assert!(
            self.project_id > 0
                && self.org_id > 0
                && !self.project_uuid.is_nil()
                && !self.subject_project_id.is_nil()
        );
        self.subject_project_id
    }
}

struct MaplePairingAuthorityInventoryHasher(Sha256);

impl MaplePairingAuthorityInventoryHasher {
    fn new(domain: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CanonicalBytes::new(domain).into_bytes());
        Self(hasher)
    }

    fn append(&mut self, body: CanonicalBytes) {
        self.0.update(body.into_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.0.finalize().to_vec()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MaplePairingAuthorityCounts {
    devices: i64,
    device_operations: i64,
    registration_operation_tombstones: i64,
    installation_retirements: i64,
    lineages: i64,
    pairings: i64,
    pairing_operations: i64,
    host_states: i64,
    revocation_events: i64,
    highwater_groups: i64,
    highwater_generations: i64,
    reset_clear_obligations: i64,
    reset_clear_admissions: i64,
}

#[derive(Clone)]
struct MaplePairingAuthorityDeviceSummary {
    id: i64,
    uuid: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    identity_mac: Vec<u8>,
    revision: i64,
}

impl From<&MapleDevice> for MaplePairingAuthorityDeviceSummary {
    fn from(row: &MapleDevice) -> Self {
        Self {
            id: row.id,
            uuid: row.uuid,
            device_id: row.device_id,
            installation_id: row.installation_id,
            identity_mac: row.identity_mac.clone(),
            revision: row.revision,
        }
    }
}

#[derive(Clone)]
struct MaplePairingAuthorityPairSummary {
    id: i64,
    uuid: Uuid,
    user_id: Uuid,
    project_id: i32,
    controller_maple_device_id: i64,
    host_maple_device_id: i64,
    pairing_incarnation: i64,
    state: i16,
    revision: i64,
    revocation_stream_id: Option<Uuid>,
    revocation_stream_generation: Option<i64>,
    revocation_issuer_key_id: Option<String>,
    created_at: DateTime<Utc>,
    approved_at: Option<DateTime<Utc>>,
    activated_at: Option<DateTime<Utc>>,
    revoked_at: Option<DateTime<Utc>>,
}

impl From<&MaplePairing> for MaplePairingAuthorityPairSummary {
    fn from(row: &MaplePairing) -> Self {
        Self {
            id: row.id,
            uuid: row.uuid,
            user_id: row.user_id,
            project_id: row.project_id,
            controller_maple_device_id: row.controller_maple_device_id,
            host_maple_device_id: row.host_maple_device_id,
            pairing_incarnation: row.pairing_incarnation,
            state: row.state,
            revision: row.revision,
            revocation_stream_id: row.revocation_stream_id,
            revocation_stream_generation: row.revocation_stream_generation,
            revocation_issuer_key_id: row.revocation_issuer_key_id.clone(),
            created_at: row.created_at,
            approved_at: row.approved_at,
            activated_at: row.activated_at,
            revoked_at: row.revoked_at,
        }
    }
}

#[derive(Clone, Copy)]
struct MaplePairingAuthorityOperationSummary {
    actor_maple_device_id: i64,
    pairing_revision: i64,
    accepted_at: DateTime<Utc>,
}

#[derive(Clone, Copy)]
struct MaplePairingAuthorityEventSummary {
    host_maple_device_id: i64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: i64,
    issuer_sequence: i64,
    acked_at: Option<DateTime<Utc>>,
}

#[derive(Clone)]
struct MaplePairingAuthorityResetClearSummary {
    id: i64,
    uuid: Uuid,
    authority_scope_digest: Vec<u8>,
    lookup_digest: Vec<u8>,
    host_identity_mac: Vec<u8>,
    reset_id: Uuid,
    reset_generation: i64,
    cumulative_reset_count: i64,
    previous_event_id: Option<Uuid>,
    previous_instruction_digest: Option<Vec<u8>>,
    previous_chain_digest: Option<Vec<u8>>,
    old_revocation_stream_id: Uuid,
    old_revocation_stream_generation: i64,
    source_security_epoch: i64,
    source_last_issued_revocation_sequence: i64,
    target_revocation_stream_id: Uuid,
    target_revocation_stream_generation: i64,
    target_security_epoch: i64,
    target_instruction_sequence: i64,
    admission_set_digest: Vec<u8>,
    admission_count: i16,
    host_claim_digest: Vec<u8>,
    instruction_digest: Vec<u8>,
    chain_digest: Vec<u8>,
    reset_at: DateTime<Utc>,
    materialized: bool,
    state: i16,
    revision: i64,
    acked_by_head_event_id: Option<Uuid>,
    acked_at: Option<DateTime<Utc>>,
    direct_ack_operation_id: Option<Uuid>,
    direct_ack_host_registration_lookup_digest: Option<Vec<u8>>,
    record_mac: Vec<u8>,
}

impl From<&MaplePairingResetClearObligation> for MaplePairingAuthorityResetClearSummary {
    fn from(row: &MaplePairingResetClearObligation) -> Self {
        Self {
            id: row.id,
            uuid: row.uuid,
            authority_scope_digest: row.authority_scope_digest.clone(),
            lookup_digest: row.lookup_digest.clone(),
            host_identity_mac: row.host_identity_mac.clone(),
            reset_id: row.reset_id,
            reset_generation: row.reset_generation,
            cumulative_reset_count: row.cumulative_reset_count,
            previous_event_id: row.previous_event_id,
            previous_instruction_digest: row.previous_instruction_digest.clone(),
            previous_chain_digest: row.previous_chain_digest.clone(),
            old_revocation_stream_id: row.old_revocation_stream_id,
            old_revocation_stream_generation: row.old_revocation_stream_generation,
            source_security_epoch: row.source_security_epoch,
            source_last_issued_revocation_sequence: row.source_last_issued_revocation_sequence,
            target_revocation_stream_id: row.target_revocation_stream_id,
            target_revocation_stream_generation: row.target_revocation_stream_generation,
            target_security_epoch: row.target_security_epoch,
            target_instruction_sequence: row.target_instruction_sequence,
            admission_set_digest: row.admission_set_digest.clone(),
            admission_count: row.admission_count,
            host_claim_digest: row.host_claim_digest.clone(),
            instruction_digest: row.instruction_digest.clone(),
            chain_digest: row.chain_digest.clone(),
            reset_at: row.reset_at,
            materialized: row.signed_instruction_payload_version.is_some(),
            state: row.state,
            revision: row.revision,
            acked_by_head_event_id: row.acked_by_head_event_id,
            acked_at: row.acked_at,
            direct_ack_operation_id: row.ack_operation_id,
            direct_ack_host_registration_lookup_digest: row
                .ack_host_registration_lookup_digest
                .clone(),
            record_mac: row.record_mac.clone(),
        }
    }
}

fn append_maple_pairing_reset_generation_counts(
    leaf: &mut CanonicalBytes,
    reset_generation: i64,
    cumulative_reset_count: i64,
) {
    leaf.append_i64(reset_generation)
        .append_i64(cumulative_reset_count);
}

#[cfg(test)]
#[test]
fn maple_pairing_reset_clear_inventory_digest_binds_cumulative_count() {
    let mut canonical = CanonicalBytes::new("test.maple-reset-clear-counts");
    append_maple_pairing_reset_generation_counts(&mut canonical, 7, 7);
    let canonical_digest = Sha256::digest(canonical.into_bytes());

    // SQL requires equality today, but the authenticated inventory transcript
    // must remain sensitive to each field independently if storage checks are
    // bypassed or a future migration changes that relational rule.
    let mut mismatched = CanonicalBytes::new("test.maple-reset-clear-counts");
    append_maple_pairing_reset_generation_counts(&mut mismatched, 7, 8);
    assert_ne!(
        canonical_digest.as_slice(),
        Sha256::digest(mismatched.into_bytes()).as_slice()
    );
}

fn validate_maple_pairing_reset_clear_admission_aggregate(
    obligation: &MaplePairingAuthorityResetClearSummary,
    aggregate: CanonicalBytes,
    actual_count: i64,
) -> Result<(), DBError> {
    if actual_count != i64::from(obligation.admission_count) {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let expected: [u8; 32] = Sha256::digest(aggregate.into_bytes()).into();
    if !maple_pairing_authority_mac_matches(&expected, &obligation.admission_set_digest) {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

impl MaplePairingAuthorityCounts {
    fn total_rows(self) -> Option<i64> {
        self.devices
            .checked_add(self.device_operations)
            .and_then(|value| value.checked_add(self.registration_operation_tombstones))
            .and_then(|value| value.checked_add(self.installation_retirements))
            .and_then(|value| value.checked_add(self.lineages))
            .and_then(|value| value.checked_add(self.pairings))
            .and_then(|value| value.checked_add(self.pairing_operations))
            .and_then(|value| value.checked_add(self.host_states))
            .and_then(|value| value.checked_add(self.revocation_events))
            .and_then(|value| value.checked_add(self.highwater_generations))
            .and_then(|value| value.checked_add(self.reset_clear_obligations))
            .and_then(|value| value.checked_add(self.reset_clear_admissions))
    }

    fn append_to(self, body: &mut CanonicalBytes) {
        body.append_i64(self.total_rows().unwrap_or(-1))
            .append_i64(self.devices)
            .append_i64(self.device_operations)
            .append_i64(self.registration_operation_tombstones)
            .append_i64(self.installation_retirements)
            .append_i64(self.lineages)
            .append_i64(self.pairings)
            .append_i64(self.pairing_operations)
            .append_i64(self.host_states)
            .append_i64(self.revocation_events)
            .append_i64(self.highwater_groups)
            .append_i64(self.highwater_generations);
        body.append_i64(self.reset_clear_obligations)
            .append_i64(self.reset_clear_admissions);
    }
}

fn append_maple_pairing_authority_category(
    hasher: &mut MaplePairingAuthorityInventoryHasher,
    category: &str,
    count: i64,
) {
    let mut body = CanonicalBytes::new("os.maple-pair-authority-category.v1");
    body.append_str(category).append_i64(count);
    hasher.append(body);
}

fn maple_pairing_authority_account_inventory_hasher(
    user_id: Uuid,
    project_id: i32,
    org_id: i32,
    security_epoch: i64,
    authority_scope_digest: &[u8],
    counts: MaplePairingAuthorityCounts,
) -> MaplePairingAuthorityInventoryHasher {
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_ACCOUNT_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-account-inventory-header.v1");
    header
        .append_uuid(user_id)
        .append_i32(project_id)
        .append_i32(org_id)
        .append_i64(security_epoch)
        .append_bytes(authority_scope_digest);
    counts.append_to(&mut header);
    hasher.append(header);
    hasher
}

fn empty_maple_pairing_authority_account_inventory(
    user_id: Uuid,
    project_id: i32,
    org_id: i32,
    authority_scope_digest: &[u8],
) -> Vec<u8> {
    let counts = MaplePairingAuthorityCounts::default();
    let mut hasher = maple_pairing_authority_account_inventory_hasher(
        user_id,
        project_id,
        org_id,
        1,
        authority_scope_digest,
        counts,
    );
    for (category, count) in [
        ("devices", counts.devices),
        ("device_operations", counts.device_operations),
        (
            "registration_operation_tombstones",
            counts.registration_operation_tombstones,
        ),
        ("installation_retirements", counts.installation_retirements),
        ("lineages", counts.lineages),
        ("pairings", counts.pairings),
        ("pairing_operations", counts.pairing_operations),
        ("host_states", counts.host_states),
        ("revocation_highwaters", counts.highwater_generations),
        ("revocation_events", counts.revocation_events),
        ("reset_clear_obligations", counts.reset_clear_obligations),
        ("reset_clear_admissions", counts.reset_clear_admissions),
    ] {
        append_maple_pairing_authority_category(&mut hasher, category, count);
    }
    hasher.finish()
}

fn count_maple_pairing_authority_account_rows(
    conn: &mut PgConnection,
    authority_scope_digest: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<MaplePairingAuthorityCounts, DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_host_states,
        maple_pairing_installation_retirements, maple_pairing_lineages, maple_pairing_operations,
        maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_admissions,
        maple_pairing_reset_clear_obligations, maple_pairing_revocation_events,
        maple_pairing_revocation_highwaters, maple_pairings,
    };

    #[derive(diesel::QueryableByName)]
    struct DistinctCount {
        #[diesel(sql_type = diesel::sql_types::BigInt)]
        count: i64,
    }
    let highwater_groups = diesel::sql_query(
        "SELECT COUNT(DISTINCT lookup_digest)::BIGINT AS count \
         FROM maple_pairing_revocation_highwaters WHERE authority_scope_digest = $1",
    )
    .bind::<diesel::sql_types::Binary, _>(authority_scope_digest)
    .get_result::<DistinctCount>(conn)?
    .count;
    let counts = MaplePairingAuthorityCounts {
        devices: maple_devices::table
            .filter(maple_devices::user_id.eq(user_id))
            .filter(maple_devices::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        device_operations: maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(user_id))
            .filter(maple_device_registration_operations::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        registration_operation_tombstones: maple_pairing_registration_operation_tombstones::table
            .filter(
                maple_pairing_registration_operation_tombstones::authority_scope_digest
                    .eq(authority_scope_digest),
            )
            .count()
            .get_result(conn)?,
        installation_retirements: maple_pairing_installation_retirements::table
            .filter(
                maple_pairing_installation_retirements::authority_scope_digest
                    .eq(authority_scope_digest),
            )
            .count()
            .get_result(conn)?,
        lineages: maple_pairing_lineages::table
            .filter(maple_pairing_lineages::user_id.eq(user_id))
            .filter(maple_pairing_lineages::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        pairings: maple_pairings::table
            .filter(maple_pairings::user_id.eq(user_id))
            .filter(maple_pairings::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        pairing_operations: maple_pairing_operations::table
            .filter(maple_pairing_operations::user_id.eq(user_id))
            .filter(maple_pairing_operations::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        host_states: maple_pairing_host_states::table
            .filter(maple_pairing_host_states::user_id.eq(user_id))
            .filter(maple_pairing_host_states::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        revocation_events: maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(user_id))
            .filter(maple_pairing_revocation_events::project_id.eq(project_id))
            .count()
            .get_result(conn)?,
        highwater_groups,
        highwater_generations: maple_pairing_revocation_highwaters::table
            .filter(
                maple_pairing_revocation_highwaters::authority_scope_digest
                    .eq(authority_scope_digest),
            )
            .count()
            .get_result(conn)?,
        reset_clear_obligations: maple_pairing_reset_clear_obligations::table
            .filter(
                maple_pairing_reset_clear_obligations::authority_scope_digest
                    .eq(authority_scope_digest),
            )
            .count()
            .get_result(conn)?,
        reset_clear_admissions: maple_pairing_reset_clear_admissions::table
            .filter(
                maple_pairing_reset_clear_admissions::authority_scope_digest
                    .eq(authority_scope_digest),
            )
            .count()
            .get_result(conn)?,
    };
    if counts.devices > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT
        || counts.device_operations < 0
        || counts.registration_operation_tombstones < 0
        || counts.installation_retirements < 0
        || counts.installation_retirements > MAPLE_PAIRING_AUTHORITY_INSTALLATION_RETIREMENT_LIMIT
        || counts.device_operations + counts.registration_operation_tombstones
            > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT * MAPLE_DEVICE_OPERATION_LIMIT_PER_DEVICE
        || counts.lineages > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || counts.pairings > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || counts.pairing_operations
            > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT * MAPLE_PAIRING_OPERATION_LIMIT_PER_PAIRING
        || counts.host_states > MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT
        || counts.revocation_events > MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT
        || counts.highwater_groups > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT
        || counts.highwater_generations > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
        || counts.reset_clear_obligations > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_OBLIGATION_LIMIT
        || counts.reset_clear_admissions > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_ADMISSION_LIMIT
        || counts.total_rows().is_none()
    {
        tracing::warn!(
            event = "maple_pairing_authority_capacity_exceeded",
            device_count = counts.devices,
            pairing_count = counts.pairings,
            highwater_group_count = counts.highwater_groups,
            highwater_generation_count = counts.highwater_generations,
            "Maple pairing authority row count exceeded a V1 lifetime bound"
        );
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    if counts.highwater_groups * 10 >= MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT * 9
        || counts.highwater_generations * 10
            >= MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT * 9
    {
        tracing::warn!(
            event = "maple_pairing_authority_capacity_near_limit",
            highwater_group_count = counts.highwater_groups,
            highwater_generation_count = counts.highwater_generations,
            "Maple pairing authority highwater history is near its V1 lifetime bound"
        );
    }
    Ok(counts)
}

fn compute_maple_pairing_authority_account_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    head: &MaplePairingAuthorityAccountHead,
) -> Result<(MaplePairingAuthorityCounts, Vec<u8>), DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_host_states,
        maple_pairing_installation_retirements, maple_pairing_lineages, maple_pairing_operations,
        maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_admissions,
        maple_pairing_reset_clear_obligations, maple_pairing_revocation_events,
        maple_pairing_revocation_highwaters, maple_pairings,
    };
    use diesel::BoolExpressionMethods;

    validate_maple_pairing_authority_account_head(enclave_key, head)?;
    let counts = count_maple_pairing_authority_account_rows(
        conn,
        &head.authority_scope_digest,
        head.user_id,
        head.project_id,
    )?;
    let mut hasher = maple_pairing_authority_account_inventory_hasher(
        head.user_id,
        head.project_id,
        head.org_id,
        head.security_epoch,
        &head.authority_scope_digest,
        counts,
    );

    append_maple_pairing_authority_category(&mut hasher, "devices", counts.devices);
    let mut devices_by_id = BTreeMap::new();
    let mut last_id = 0_i64;
    loop {
        let rows = maple_devices::table
            .filter(maple_devices::user_id.eq(head.user_id))
            .filter(maple_devices::project_id.eq(head.project_id))
            .filter(maple_devices::id.gt(last_id))
            .order(maple_devices::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MapleDevice>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            if row.id <= 0
                || row.user_id != head.user_id
                || row.project_id != head.project_id
                || row.uuid.is_nil()
                || row.device_id.is_nil()
                || row.installation_id.is_nil()
                || row.identity_mac.len() != 32
                || row.endpoint_epoch < 0
                || row.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                || row.revision <= 0
                || !maple_device_record_mac_is_valid(enclave_key, &row)?
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-device-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_uuid(row.uuid)
                .append_uuid(row.device_id)
                .append_uuid(row.installation_id)
                .append_i64(row.revision)
                .append_bytes(&row.record_mac);
            hasher.append(leaf);
            last_id = row.id;
            let summary = MaplePairingAuthorityDeviceSummary::from(&row);
            if devices_by_id.insert(row.id, summary).is_some() {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "device_operations",
        counts.device_operations,
    );
    let mut revisions_by_device: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut registration_operation_lookups = BTreeSet::new();
    last_id = 0;
    loop {
        let rows = maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(head.user_id))
            .filter(maple_device_registration_operations::project_id.eq(head.project_id))
            .filter(maple_device_registration_operations::id.gt(last_id))
            .order(maple_device_registration_operations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MapleDeviceRegistrationOperation>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let device = devices_by_id
                .get(&row.maple_device_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            validate_maple_device_registration_operation(
                enclave_key,
                &row,
                device,
                head.user_id,
                head.project_id,
            )?;
            if row.accepted_security_epoch != head.security_epoch {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if !registration_operation_lookups.insert(row.operation_lookup_digest.clone()) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-device-operation-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_uuid(row.operation_id)
                .append_bytes(&row.request_mac)
                .append_i64(row.maple_device_id)
                .append_i64(row.device_revision)
                .append_bytes(&row.authority_scope_digest)
                .append_bytes(&row.lookup_digest)
                .append_bytes(&row.operation_lookup_digest)
                .append_i64(row.known_security_epoch)
                .append_i64(row.accepted_security_epoch)
                .append_i16(row.response_kind)
                .append_i16(row.sync_payload_version)
                .append_bytes(&row.sync_payload_enc)
                .append_str(&row.sync_issuer_key_id)
                .append_bytes(&row.sync_digest)
                .append_bytes(&row.receipt_mac)
                .append_i64(row.accepted_at.timestamp_micros());
            hasher.append(leaf);
            revisions_by_device
                .entry(row.maple_device_id)
                .or_default()
                .push(row.device_revision);
            last_id = row.id;
        }
    }
    for device in devices_by_id.values() {
        let revisions = revisions_by_device
            .get(&device.id)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if revisions.len() as i64 != device.revision
            || !revisions.iter().copied().eq(1_i64..=device.revision)
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "registration_operation_tombstones",
        counts.registration_operation_tombstones,
    );
    last_id = 0;
    let mut tombstones_seen = 0_i64;
    let mut current_epoch_tombstone_lookups = BTreeSet::new();
    loop {
        let rows = maple_pairing_registration_operation_tombstones::table
            .filter(
                maple_pairing_registration_operation_tombstones::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .filter(maple_pairing_registration_operation_tombstones::id.gt(last_id))
            .order(maple_pairing_registration_operation_tombstones::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingRegistrationOperationTombstone>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_device_registration_tombstone(
                enclave_key,
                &row,
                &head.authority_scope_digest,
                head.security_epoch,
            )?;
            if !registration_operation_lookups.insert(row.operation_lookup_digest.clone()) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if row.retired_security_epoch == head.security_epoch {
                current_epoch_tombstone_lookups.insert(row.lookup_digest.clone());
            }
            let mut leaf =
                CanonicalBytes::new("os.maple-pair-authority-registration-tombstone-leaf.v1");
            leaf.append_i64(row.id)
                .append_bytes(&row.authority_scope_digest)
                .append_bytes(&row.lookup_digest)
                .append_bytes(&row.operation_lookup_digest)
                .append_i64(row.retired_security_epoch)
                .append_bytes(&row.request_mac)
                .append_i16(row.outcome_kind)
                .append_bytes(&row.outcome_digest)
                .append_i16(row.receipt_version)
                .append_bytes(&row.receipt_digest)
                .append_u16(
                    row.referenced_issuer_key_ids
                        .len()
                        .try_into()
                        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
                );
            for key_id in &row.referenced_issuer_key_ids {
                leaf.append_str(key_id);
            }
            leaf.append_i64(row.accepted_at.timestamp_micros())
                .append_bytes(&row.record_mac)
                .append_i64(row.retired_at.timestamp_micros());
            hasher.append(leaf);
            last_id = row.id;
            tombstones_seen = tombstones_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if tombstones_seen != counts.registration_operation_tombstones {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "installation_retirements",
        counts.installation_retirements,
    );
    last_id = 0;
    let mut retired_lookups = BTreeSet::new();
    let mut retired_identities = BTreeSet::new();
    let mut retired_host_registrations = BTreeSet::new();
    let mut retirement_ack_operations = BTreeSet::new();
    let mut retirements_seen = 0_i64;
    loop {
        let rows = maple_pairing_installation_retirements::table
            .filter(
                maple_pairing_installation_retirements::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .filter(maple_pairing_installation_retirements::id.gt(last_id))
            .order(maple_pairing_installation_retirements::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingInstallationRetirement>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_installation_retirement(
                enclave_key,
                &row,
                &head.authority_scope_digest,
                head.security_epoch,
            )?;
            if !retired_lookups.insert(row.lookup_digest.clone())
                || !retired_identities.insert(row.host_identity_mac.clone())
                || !retired_host_registrations
                    .insert(row.ack_host_registration_lookup_digest.clone())
                || !retirement_ack_operations.insert(row.ack_operation_lookup_digest.clone())
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let obligation = maple_pairing_reset_clear_obligations::table
                .filter(
                    maple_pairing_reset_clear_obligations::authority_scope_digest
                        .eq(&head.authority_scope_digest),
                )
                .filter(maple_pairing_reset_clear_obligations::lookup_digest.eq(&row.lookup_digest))
                .filter(
                    maple_pairing_reset_clear_obligations::uuid.eq(row.final_obligation_event_id),
                )
                .first::<MaplePairingResetClearObligation>(conn)?;
            validate_maple_pairing_reset_clear_obligation(
                enclave_key,
                &obligation,
                &head.authority_scope_digest,
            )?;
            let ack_operation_id = obligation
                .ack_operation_id
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            let expected_ack_operation_lookup = maple_reset_clear_ack_operation_lookup_digest(
                enclave_key,
                &head.authority_scope_digest,
                &row.ack_host_registration_lookup_digest,
                ack_operation_id,
            )?;
            if obligation.state != 2
                || obligation.revision != 3
                || obligation.acked_by_head_event_id != Some(obligation.uuid)
                || obligation.target_security_epoch != row.retired_security_epoch
                || obligation.host_identity_mac != row.host_identity_mac
                || obligation.instruction_digest != row.final_instruction_digest
                || obligation.chain_digest != row.final_chain_digest
                || obligation.ack_host_registration_lookup_digest.as_deref()
                    != Some(row.ack_host_registration_lookup_digest.as_slice())
                || obligation.ack_request_mac.as_deref() != Some(row.ack_request_mac.as_slice())
                || obligation.ack_receipt_version != Some(row.ack_receipt_version)
                || obligation.ack_receipt_issuer_key_id.as_deref()
                    != Some(row.ack_receipt_issuer_key_id.as_str())
                || obligation.ack_receipt_digest.as_deref()
                    != Some(row.ack_receipt_digest.as_slice())
                || obligation.acked_at != Some(row.retired_at)
                || !maple_pairing_authority_mac_matches(
                    &expected_ack_operation_lookup,
                    &row.ack_operation_lookup_digest,
                )
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf =
                CanonicalBytes::new("os.maple-pair-authority-installation-retirement-leaf.v1");
            leaf.append_i64(row.id)
                .append_bytes(&row.authority_scope_digest)
                .append_bytes(&row.lookup_digest)
                .append_bytes(&row.host_identity_mac)
                .append_i64(row.retired_security_epoch)
                .append_uuid(row.final_obligation_event_id)
                .append_bytes(&row.final_instruction_digest)
                .append_bytes(&row.final_chain_digest)
                .append_bytes(&row.ack_host_registration_lookup_digest)
                .append_bytes(&row.ack_operation_lookup_digest)
                .append_bytes(&row.ack_request_mac)
                .append_i16(row.ack_receipt_version)
                .append_str(&row.ack_receipt_issuer_key_id)
                .append_bytes(&row.ack_receipt_digest)
                .append_i64(row.retired_at.timestamp_micros())
                .append_bytes(&row.record_mac)
                .append_i64(row.created_at.timestamp_micros());
            hasher.append(leaf);
            last_id = row.id;
            retirements_seen = retirements_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if retirements_seen != counts.installation_retirements {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    if !current_epoch_tombstone_lookups
        .iter()
        .all(|lookup| retired_lookups.contains(lookup))
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    append_maple_pairing_authority_category(&mut hasher, "lineages", counts.lineages);
    let mut lineages_by_id = BTreeMap::new();
    last_id = 0;
    loop {
        let rows = maple_pairing_lineages::table
            .filter(maple_pairing_lineages::user_id.eq(head.user_id))
            .filter(maple_pairing_lineages::project_id.eq(head.project_id))
            .filter(maple_pairing_lineages::id.gt(last_id))
            .order(maple_pairing_lineages::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingLineage>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            if row.id <= 0
                || row.user_id != head.user_id
                || row.project_id != head.project_id
                || row.controller_maple_device_id == row.host_maple_device_id
                || row.direction != 1
                || row.last_pairing_incarnation <= 0
                || !devices_by_id.contains_key(&row.controller_maple_device_id)
                || !devices_by_id.contains_key(&row.host_maple_device_id)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-lineage-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_i64(row.controller_maple_device_id)
                .append_i64(row.host_maple_device_id)
                .append_i16(row.direction)
                .append_i64(row.last_pairing_incarnation)
                .append_i64(row.created_at.timestamp_micros())
                .append_i64(row.updated_at.timestamp_micros());
            hasher.append(leaf);
            last_id = row.id;
            if lineages_by_id.insert(row.id, row).is_some() {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }

    append_maple_pairing_authority_category(&mut hasher, "pairings", counts.pairings);
    let mut pairings_by_id = BTreeMap::new();
    let mut max_incarnation_by_lineage = BTreeMap::new();
    let mut current_pair_by_lineage = BTreeMap::new();
    last_id = 0;
    loop {
        let rows = maple_pairings::table
            .filter(maple_pairings::user_id.eq(head.user_id))
            .filter(maple_pairings::project_id.eq(head.project_id))
            .filter(maple_pairings::id.gt(last_id))
            .order(maple_pairings::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MaplePairing>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_record(enclave_key, &row)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let lineage = lineages_by_id
                .get(&row.lineage_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if row.user_id != head.user_id
                || row.project_id != head.project_id
                || row.controller_maple_device_id != lineage.controller_maple_device_id
                || row.host_maple_device_id != lineage.host_maple_device_id
                || row.direction != lineage.direction
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-pairing-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_uuid(row.uuid)
                .append_uuid(row.pairing_request_id)
                .append_i64(row.lineage_id)
                .append_i64(row.controller_maple_device_id)
                .append_i64(row.host_maple_device_id)
                .append_i64(row.pairing_incarnation)
                .append_i16(row.state)
                .append_i64(row.revision)
                .append_bytes(&row.record_mac);
            hasher.append(leaf);
            max_incarnation_by_lineage
                .entry(row.lineage_id)
                .and_modify(|value: &mut i64| *value = (*value).max(row.pairing_incarnation))
                .or_insert(row.pairing_incarnation);
            if matches!(
                MaplePairingState::try_from(row.state),
                Ok(MaplePairingState::Pending)
                    | Ok(MaplePairingState::AwaitingHostCommit)
                    | Ok(MaplePairingState::Active)
            ) && current_pair_by_lineage
                .insert(row.lineage_id, row.id)
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            last_id = row.id;
            if pairings_by_id
                .insert(row.id, MaplePairingAuthorityPairSummary::from(&row))
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }
    for lineage in lineages_by_id.values() {
        if max_incarnation_by_lineage.get(&lineage.id) != Some(&lineage.last_pairing_incarnation) {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "pairing_operations",
        counts.pairing_operations,
    );
    let mut operations_by_pair: BTreeMap<
        i64,
        BTreeMap<i16, MaplePairingAuthorityOperationSummary>,
    > = BTreeMap::new();
    last_id = 0;
    loop {
        let rows = maple_pairing_operations::table
            .filter(maple_pairing_operations::user_id.eq(head.user_id))
            .filter(maple_pairing_operations::project_id.eq(head.project_id))
            .filter(maple_pairing_operations::id.gt(last_id))
            .order(maple_pairing_operations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MaplePairingOperation>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let pairing = pairings_by_id
                .get(&row.maple_pairing_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            pairing_operation_receipt(enclave_key, &row, pairing.uuid)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if row.user_id != pairing.user_id
                || row.project_id != pairing.project_id
                || !(1..=5).contains(&row.operation_kind)
                || row.request_mac.len() != 32
                || !devices_by_id.contains_key(&row.actor_maple_device_id)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-pairing-operation-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_uuid(row.operation_id)
                .append_i64(row.actor_maple_device_id)
                .append_i16(row.operation_kind)
                .append_bytes(&row.request_mac)
                .append_i64(row.maple_pairing_id)
                .append_i64(row.pairing_revision)
                .append_i16(row.receipt_version)
                .append_bytes(&row.receipt_enc)
                .append_bytes(&row.receipt_mac)
                .append_i64(row.accepted_at.timestamp_micros());
            hasher.append(leaf);
            if operations_by_pair
                .entry(row.maple_pairing_id)
                .or_default()
                .insert(
                    row.operation_kind,
                    MaplePairingAuthorityOperationSummary {
                        actor_maple_device_id: row.actor_maple_device_id,
                        pairing_revision: row.pairing_revision,
                        accepted_at: row.accepted_at,
                    },
                )
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            last_id = row.id;
        }
    }

    append_maple_pairing_authority_category(&mut hasher, "host_states", counts.host_states);
    let mut host_states_by_device = BTreeMap::new();
    last_id = 0;
    loop {
        let rows = maple_pairing_host_states::table
            .filter(maple_pairing_host_states::user_id.eq(head.user_id))
            .filter(maple_pairing_host_states::project_id.eq(head.project_id))
            .filter(maple_pairing_host_states::id.gt(last_id))
            .order(maple_pairing_host_states::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingHostState>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_host_state(enclave_key, &row)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if row.user_id != head.user_id
                || row.project_id != head.project_id
                || !devices_by_id.contains_key(&row.host_maple_device_id)
                || row.last_acked_revocation_sequence > row.last_issued_revocation_sequence
                || row.revision
                    != 1_i64
                        .checked_add(row.last_issued_revocation_sequence)
                        .and_then(|value| value.checked_add(row.last_acked_revocation_sequence))
                        .ok_or(DBError::MaplePairingAuthorityCorrupt)?
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-host-state-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_i64(row.host_maple_device_id)
                .append_uuid(row.revocation_stream_id)
                .append_i64(row.revocation_stream_generation)
                .append_i64(row.revision)
                .append_bytes(&row.record_mac);
            hasher.append(leaf);
            last_id = row.id;
            if host_states_by_device
                .insert(row.host_maple_device_id, row)
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "revocation_highwaters",
        counts.highwater_generations,
    );
    let mut latest_by_lookup = BTreeMap::new();
    let mut next_generation_by_lookup = BTreeMap::new();
    let mut stream_ids = BTreeSet::new();
    let mut highwaters_by_namespace: BTreeMap<(Vec<u8>, Uuid, i64), (i64, i64)> = BTreeMap::new();
    let mut previous_highwater_by_lookup: BTreeMap<Vec<u8>, (Uuid, i64, i64)> = BTreeMap::new();
    struct HighwaterTransition {
        lookup_digest: Vec<u8>,
        old_stream_id: Uuid,
        old_generation: i64,
        old_epoch: i64,
        target_stream_id: Uuid,
        target_generation: i64,
        target_epoch: i64,
    }
    let mut highwater_transitions: Vec<HighwaterTransition> = Vec::new();
    let mut highwater_cursor_lookup = Vec::new();
    let mut highwater_cursor_generation = 0_i64;
    let mut highwater_rows_seen = 0_i64;
    loop {
        let rows = maple_pairing_revocation_highwaters::table
            .filter(
                maple_pairing_revocation_highwaters::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .filter(
                maple_pairing_revocation_highwaters::lookup_digest
                    .gt(&highwater_cursor_lookup)
                    .or(maple_pairing_revocation_highwaters::lookup_digest
                        .eq(&highwater_cursor_lookup)
                        .and(
                            maple_pairing_revocation_highwaters::revocation_stream_generation
                                .gt(highwater_cursor_generation),
                        )),
            )
            .order((
                maple_pairing_revocation_highwaters::lookup_digest.asc(),
                maple_pairing_revocation_highwaters::revocation_stream_generation.asc(),
            ))
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingRevocationHighwater>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_revocation_highwater(enclave_key, &row)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if row.revision
                != row
                    .last_issued_revocation_sequence
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingAuthorityCorrupt)?
                || !maple_pairing_authority_mac_matches(
                    &row.authority_scope_digest,
                    &head.authority_scope_digest,
                )
                || !stream_ids.insert(row.revocation_stream_id)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let expected = next_generation_by_lookup
                .entry(row.lookup_digest.clone())
                .or_insert(1_i64);
            if row.revocation_stream_generation != *expected {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            *expected = expected
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-highwater-leaf.v1");
            leaf.append_i64(row.id)
                .append_bytes(&row.authority_scope_digest)
                .append_bytes(&row.lookup_digest)
                .append_uuid(row.revocation_stream_id)
                .append_i64(row.revocation_stream_generation)
                .append_i64(row.security_epoch)
                .append_i64(row.revision)
                .append_bytes(&row.record_mac);
            hasher.append(leaf);
            let namespace_key = (
                row.lookup_digest.clone(),
                row.revocation_stream_id,
                row.revocation_stream_generation,
            );
            if highwaters_by_namespace
                .insert(
                    namespace_key,
                    (row.security_epoch, row.last_issued_revocation_sequence),
                )
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if let Some((previous_stream_id, previous_generation, previous_epoch)) =
                previous_highwater_by_lookup.insert(
                    row.lookup_digest.clone(),
                    (
                        row.revocation_stream_id,
                        row.revocation_stream_generation,
                        row.security_epoch,
                    ),
                )
            {
                if previous_generation.checked_add(1) != Some(row.revocation_stream_generation)
                    || previous_epoch.checked_add(1) != Some(row.security_epoch)
                {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                highwater_transitions.push(HighwaterTransition {
                    lookup_digest: row.lookup_digest.clone(),
                    old_stream_id: previous_stream_id,
                    old_generation: previous_generation,
                    old_epoch: previous_epoch,
                    target_stream_id: row.revocation_stream_id,
                    target_generation: row.revocation_stream_generation,
                    target_epoch: row.security_epoch,
                });
            }
            highwater_cursor_lookup = row.lookup_digest.clone();
            highwater_cursor_generation = row.revocation_stream_generation;
            highwater_rows_seen = highwater_rows_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            latest_by_lookup.insert(row.lookup_digest.clone(), row);
        }
    }
    if highwater_rows_seen != counts.highwater_generations
        || latest_by_lookup.len() as i64 != counts.highwater_groups
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    // Load retained reset headers into bounded metadata summaries before
    // classifying highwaters and validating the mixed control/event stream.
    // Ciphertext rows are released one page at a time.
    let mut obligations_by_event: BTreeMap<Uuid, MaplePairingAuthorityResetClearSummary> =
        BTreeMap::new();
    let mut obligations_by_lookup: BTreeMap<Vec<u8>, Vec<Uuid>> = BTreeMap::new();
    let mut obligation_events_in_row_order = Vec::new();
    let mut reset_batches: BTreeMap<Uuid, (i64, i64, DateTime<Utc>)> = BTreeMap::new();
    last_id = 0;
    let mut obligations_seen = 0_i64;
    loop {
        let rows = maple_pairing_reset_clear_obligations::table
            .filter(
                maple_pairing_reset_clear_obligations::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .filter(maple_pairing_reset_clear_obligations::id.gt(last_id))
            .order(maple_pairing_reset_clear_obligations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MaplePairingResetClearObligation>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_reset_clear_obligation(
                enclave_key,
                &row,
                &head.authority_scope_digest,
            )?;
            if let Some(previous) = row.previous_event_id {
                let predecessor = obligations_by_event
                    .get(&previous)
                    .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                if predecessor.lookup_digest != row.lookup_digest
                    || !maple_pairing_authority_mac_matches(
                        &predecessor.host_identity_mac,
                        &row.host_identity_mac,
                    )
                    || !maple_pairing_authority_mac_matches(
                        &predecessor.host_claim_digest,
                        &row.host_claim_digest,
                    )
                    || predecessor.target_revocation_stream_id != row.old_revocation_stream_id
                    || predecessor.target_revocation_stream_generation
                        != row.old_revocation_stream_generation
                    || predecessor.target_security_epoch != row.source_security_epoch
                    || row.previous_instruction_digest.as_deref()
                        != Some(predecessor.instruction_digest.as_slice())
                    || row.previous_chain_digest.as_deref()
                        != Some(predecessor.chain_digest.as_slice())
                    || predecessor.reset_generation.checked_add(1) != Some(row.reset_generation)
                    || row.reset_at < predecessor.reset_at
                {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
            } else if row.reset_generation != 1 {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let summary = MaplePairingAuthorityResetClearSummary::from(&row);
            match reset_batches.entry(summary.reset_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((
                        summary.source_security_epoch,
                        summary.target_security_epoch,
                        summary.reset_at,
                    ));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if *entry.get()
                        != (
                            summary.source_security_epoch,
                            summary.target_security_epoch,
                            summary.reset_at,
                        ) =>
                {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
            obligations_by_lookup
                .entry(summary.lookup_digest.clone())
                .or_default()
                .push(summary.uuid);
            obligation_events_in_row_order.push(summary.uuid);
            if obligations_by_event.insert(summary.uuid, summary).is_some() {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            last_id = row.id;
            obligations_seen = obligations_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if obligations_seen != counts.reset_clear_obligations {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    for event_ids in obligations_by_lookup.values_mut() {
        event_ids.sort_by_key(|event_id| {
            obligations_by_event
                .get(event_id)
                .map(|obligation| obligation.reset_generation)
                .unwrap_or(i64::MAX)
        });
        let mut pending_suffix_started = false;
        for event_id in event_ids.iter() {
            let obligation = obligations_by_event
                .get(event_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            match obligation.state {
                1 => {
                    pending_suffix_started = true;
                    if obligation.acked_by_head_event_id.is_some()
                        || obligation.acked_at.is_some()
                        || obligation.direct_ack_operation_id.is_some()
                        || obligation
                            .direct_ack_host_registration_lookup_digest
                            .is_some()
                    {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                }
                2 if !pending_suffix_started => {
                    let ack_head_id = obligation
                        .acked_by_head_event_id
                        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                    let ack_head = obligations_by_event
                        .get(&ack_head_id)
                        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                    if ack_head.lookup_digest != obligation.lookup_digest
                        || ack_head.state != 2
                        || ack_head.reset_generation < obligation.reset_generation
                        || ack_head.acked_by_head_event_id != Some(ack_head.uuid)
                        || ack_head.direct_ack_operation_id.is_none()
                        || ack_head
                            .direct_ack_host_registration_lookup_digest
                            .is_none()
                        || !ack_head.materialized
                        || ack_head.acked_at != obligation.acked_at
                    {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                }
                _ => return Err(DBError::MaplePairingAuthorityCorrupt),
            }
        }
    }

    let mut obligations_by_target_namespace = BTreeMap::new();
    for obligation in obligations_by_event.values() {
        let old_namespace = highwaters_by_namespace
            .get(&(
                obligation.lookup_digest.clone(),
                obligation.old_revocation_stream_id,
                obligation.old_revocation_stream_generation,
            ))
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        let target_key = (
            obligation.lookup_digest.clone(),
            obligation.target_revocation_stream_id,
            obligation.target_revocation_stream_generation,
        );
        let target_namespace = highwaters_by_namespace
            .get(&target_key)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if old_namespace.0 != obligation.source_security_epoch
            || old_namespace.1 != obligation.source_last_issued_revocation_sequence
            || target_namespace.0 != obligation.target_security_epoch
            || target_namespace.1 < obligation.target_instruction_sequence
            || obligations_by_target_namespace
                .insert(target_key, obligation.uuid)
                .is_some()
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }
    for transition in &highwater_transitions {
        let obligation = obligations_by_target_namespace
            .get(&(
                transition.lookup_digest.clone(),
                transition.target_stream_id,
                transition.target_generation,
            ))
            .and_then(|event_id| obligations_by_event.get(event_id))
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if obligation.old_revocation_stream_id != transition.old_stream_id
            || obligation.old_revocation_stream_generation != transition.old_generation
            || obligation.source_security_epoch != transition.old_epoch
            || obligation.target_security_epoch != transition.target_epoch
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }
    if obligations_by_target_namespace.len() != highwater_transitions.len() {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    let mut live_lookups = BTreeSet::new();
    let mut lookup_by_device = BTreeMap::new();
    let mut reset_control_state_by_device = BTreeMap::new();
    for device in devices_by_id.values() {
        let lookup = maple_pairing_revocation_highwater_lookup_digest(
            enclave_key,
            head.user_id,
            head.project_id,
            device.installation_id,
        )?;
        let highwater = latest_by_lookup
            .get(&lookup)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if retired_lookups.contains(&lookup)
            || retired_identities.contains(&device.identity_mac)
            || !live_lookups.insert(lookup.clone())
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let state = host_states_by_device
            .get(&device.id)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if state.revocation_stream_id != highwater.revocation_stream_id
            || state.revocation_stream_generation != highwater.revocation_stream_generation
            || state.last_issued_revocation_sequence != highwater.last_issued_revocation_sequence
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        if lookup_by_device.insert(device.id, lookup.clone()).is_some() {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        if let Some(obligation) = obligations_by_lookup
            .get(&lookup)
            .and_then(|event_ids| event_ids.last())
            .and_then(|event_id| obligations_by_event.get(event_id))
        {
            let is_current_control = obligation.target_revocation_stream_id
                == highwater.revocation_stream_id
                && obligation.target_revocation_stream_generation
                    == highwater.revocation_stream_generation;
            if obligation.state == 1 && !is_current_control {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if is_current_control {
                if !maple_pairing_authority_mac_matches(
                    &device.identity_mac,
                    &obligation.host_identity_mac,
                ) || reset_control_state_by_device
                    .insert(device.id, obligation.state)
                    .is_some()
                {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                match obligation.state {
                    1 if state.last_issued_revocation_sequence == 1
                        && state.last_acked_revocation_sequence == 0 => {}
                    2 if state.last_issued_revocation_sequence >= 1
                        && state.last_acked_revocation_sequence >= 1 => {}
                    _ => return Err(DBError::MaplePairingAuthorityCorrupt),
                }
            }
        }
    }
    for (lookup, latest) in &latest_by_lookup {
        if !live_lookups.contains(lookup) {
            let latest_obligation = obligations_by_lookup
                .get(lookup)
                .and_then(|event_ids| event_ids.last())
                .and_then(|event_id| obligations_by_event.get(event_id));
            match latest_obligation {
                Some(obligation) if obligation.state == 1 => {
                    if obligation.target_revocation_stream_id != latest.revocation_stream_id
                        || obligation.target_revocation_stream_generation
                            != latest.revocation_stream_generation
                        || obligation.target_security_epoch != head.security_epoch
                        || latest.security_epoch != head.security_epoch
                        || latest.last_issued_revocation_sequence != 1
                    {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                }
                Some(obligation) if obligation.state == 2 => {
                    // Fully ACKed historical-only namespaces are retained as
                    // non-authoritative evidence and may end above sequence 0.
                    if !retired_lookups.contains(lookup) {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                }
                Some(_) => return Err(DBError::MaplePairingAuthorityCorrupt),
                None if latest.last_issued_revocation_sequence == 0 => {}
                None => return Err(DBError::MaplePairingAuthorityCorrupt),
            }
        }
    }
    if host_states_by_device.len() != devices_by_id.len() {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "revocation_events",
        counts.revocation_events,
    );
    let mut events_by_pair = BTreeMap::new();
    let mut events_by_host: BTreeMap<i64, Vec<MaplePairingAuthorityEventSummary>> = BTreeMap::new();
    last_id = 0;
    loop {
        let rows = maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(head.user_id))
            .filter(maple_pairing_revocation_events::project_id.eq(head.project_id))
            .filter(maple_pairing_revocation_events::id.gt(last_id))
            .order(maple_pairing_revocation_events::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .load::<MaplePairingRevocationEvent>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_revocation_record(enclave_key, &row)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let pairing = pairings_by_id
                .get(&row.maple_pairing_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            let host_state = host_states_by_device
                .get(&row.recipient_host_maple_device_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if row.user_id != head.user_id
                || row.project_id != head.project_id
                || pairing.state != MaplePairingState::Revoked.as_db()
                || pairing.host_maple_device_id != row.recipient_host_maple_device_id
                || pairing.pairing_incarnation != row.pairing_incarnation
                || pairing.revocation_stream_id != Some(row.revocation_stream_id)
                || pairing.revocation_stream_generation != Some(row.revocation_stream_generation)
                || pairing.revoked_at != Some(row.created_at)
                || pairing.revocation_issuer_key_id.as_deref() != Some(row.issuer_key_id.as_str())
                || host_state.revocation_stream_id != row.revocation_stream_id
                || host_state.revocation_stream_generation != row.revocation_stream_generation
                || row.acked_at.is_some_and(|acked| acked < row.created_at)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let summary = MaplePairingAuthorityEventSummary {
                host_maple_device_id: row.recipient_host_maple_device_id,
                revocation_stream_id: row.revocation_stream_id,
                revocation_stream_generation: row.revocation_stream_generation,
                issuer_sequence: row.issuer_sequence,
                acked_at: row.acked_at,
            };
            if events_by_pair
                .insert(row.maple_pairing_id, summary)
                .is_some()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            events_by_host
                .entry(row.recipient_host_maple_device_id)
                .or_default()
                .push(summary);
            let mut leaf = CanonicalBytes::new("os.maple-pair-authority-revocation-event-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.user_id)
                .append_i32(row.project_id)
                .append_uuid(row.uuid)
                .append_i64(row.recipient_host_maple_device_id)
                .append_uuid(row.revocation_stream_id)
                .append_i64(row.revocation_stream_generation)
                .append_i64(row.issuer_sequence)
                .append_i64(row.maple_pairing_id)
                .append_i64(row.pairing_incarnation)
                .append_bytes(&row.record_mac);
            hasher.append(leaf);
            last_id = row.id;
        }
    }
    for pairing in pairings_by_id.values() {
        let event = events_by_pair.get(&pairing.id);
        let has_event = event.is_some();
        if has_event != (pairing.state == MaplePairingState::Revoked.as_db()) {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }

        if !matches!(
            MaplePairingState::try_from(pairing.state),
            Ok(MaplePairingState::Pending) | Ok(MaplePairingState::Expired)
        ) {
            let state = host_states_by_device
                .get(&pairing.host_maple_device_id)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if pairing.revocation_stream_id != Some(state.revocation_stream_id)
                || pairing.revocation_stream_generation != Some(state.revocation_stream_generation)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }

        let operations = operations_by_pair
            .get(&pairing.id)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        let operation_matches =
            |kind: i16, revision: i64, accepted_at: DateTime<Utc>, allowed_actors: &[i64]| {
                operations.get(&kind).is_some_and(|operation| {
                    operation.pairing_revision == revision
                        && operation.accepted_at == accepted_at
                        && allowed_actors.contains(&operation.actor_maple_device_id)
                })
            };
        let create_is_valid = operation_matches(
            MAPLE_PAIRING_OPERATION_CREATE,
            1,
            pairing.created_at,
            &[pairing.controller_maple_device_id],
        );
        let (expected_operation_count, lifecycle_is_valid) =
            match MaplePairingState::try_from(pairing.state) {
                Ok(MaplePairingState::Pending) | Ok(MaplePairingState::Expired) => {
                    (1, create_is_valid)
                }
                Ok(MaplePairingState::AwaitingHostCommit) => (
                    2,
                    create_is_valid
                        && operation_matches(
                            MAPLE_PAIRING_OPERATION_APPROVE,
                            2,
                            pairing
                                .approved_at
                                .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                            &[pairing.host_maple_device_id],
                        ),
                ),
                Ok(MaplePairingState::Active) => (
                    3,
                    create_is_valid
                        && operation_matches(
                            MAPLE_PAIRING_OPERATION_APPROVE,
                            2,
                            pairing
                                .approved_at
                                .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                            &[pairing.host_maple_device_id],
                        )
                        && operation_matches(
                            MAPLE_PAIRING_OPERATION_CONFIRM,
                            3,
                            pairing
                                .activated_at
                                .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                            &[pairing.host_maple_device_id],
                        ),
                ),
                Ok(MaplePairingState::Revoked) => {
                    let event = event.ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                    let ack_is_present = event.acked_at.is_some();
                    let approved_is_valid = operation_matches(
                        MAPLE_PAIRING_OPERATION_APPROVE,
                        2,
                        pairing
                            .approved_at
                            .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                        &[pairing.host_maple_device_id],
                    );
                    let confirm_is_valid = pairing.revision != 4
                        || operation_matches(
                            MAPLE_PAIRING_OPERATION_CONFIRM,
                            3,
                            pairing
                                .activated_at
                                .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                            &[pairing.host_maple_device_id],
                        );
                    let revoke_is_valid = operation_matches(
                        MAPLE_PAIRING_OPERATION_REVOKE,
                        pairing.revision,
                        pairing
                            .revoked_at
                            .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                        &[
                            pairing.controller_maple_device_id,
                            pairing.host_maple_device_id,
                        ],
                    );
                    let ack_is_valid = match event.acked_at {
                        Some(acked_at) => operation_matches(
                            MAPLE_PAIRING_OPERATION_ACK,
                            pairing.revision,
                            acked_at,
                            &[pairing.host_maple_device_id],
                        ),
                        None => !operations.contains_key(&MAPLE_PAIRING_OPERATION_ACK),
                    };
                    let lifecycle_operation_count = if pairing.revision == 4 { 4 } else { 3 };
                    (
                        lifecycle_operation_count + i64::from(ack_is_present),
                        create_is_valid
                            && approved_is_valid
                            && confirm_is_valid
                            && revoke_is_valid
                            && ack_is_valid,
                    )
                }
                Err(()) => return Err(DBError::MaplePairingAuthorityCorrupt),
            };
        if operations.len() as i64 != expected_operation_count || !lifecycle_is_valid {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    for (host_id, state) in &host_states_by_device {
        let events = events_by_host.entry(*host_id).or_default();
        events.sort_by_key(|event| event.issuer_sequence);
        let reset_control_state = reset_control_state_by_device.get(host_id).copied();
        let first_ordinary_sequence = if reset_control_state.is_some() {
            2_i64
        } else {
            1_i64
        };
        let expected_event_count = state
            .last_issued_revocation_sequence
            .checked_sub(first_ordinary_sequence - 1)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if events.len() as i64 != expected_event_count
            || matches!(reset_control_state, Some(1))
                && (state.last_issued_revocation_sequence != 1
                    || state.last_acked_revocation_sequence != 0
                    || !events.is_empty())
            || matches!(reset_control_state, Some(2)) && state.last_acked_revocation_sequence < 1
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        for (index, event) in events.iter().enumerate() {
            let sequence = i64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(first_ordinary_sequence))
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if event.host_maple_device_id != *host_id
                || event.revocation_stream_id != state.revocation_stream_id
                || event.revocation_stream_generation != state.revocation_stream_generation
                || event.issuer_sequence != sequence
                || event.acked_at.is_some() != (sequence <= state.last_acked_revocation_sequence)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "reset_clear_obligations",
        counts.reset_clear_obligations,
    );
    for event_id in &obligation_events_in_row_order {
        let row = obligations_by_event
            .get(event_id)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        let mut leaf =
            CanonicalBytes::new("os.maple-pair-authority-reset-clear-obligation-leaf.v1");
        leaf.append_i64(row.id)
            .append_uuid(row.uuid)
            .append_bytes(&row.authority_scope_digest)
            .append_bytes(&row.lookup_digest)
            .append_bytes(&row.host_identity_mac)
            .append_uuid(row.reset_id);
        append_maple_pairing_reset_generation_counts(
            &mut leaf,
            row.reset_generation,
            row.cumulative_reset_count,
        );
        append_optional_uuid(&mut leaf, row.previous_event_id);
        append_optional_bytes(&mut leaf, row.previous_instruction_digest.as_deref());
        append_optional_bytes(&mut leaf, row.previous_chain_digest.as_deref());
        leaf.append_uuid(row.old_revocation_stream_id)
            .append_i64(row.old_revocation_stream_generation)
            .append_i64(row.source_security_epoch)
            .append_i64(row.source_last_issued_revocation_sequence)
            .append_uuid(row.target_revocation_stream_id)
            .append_i64(row.target_revocation_stream_generation)
            .append_i64(row.target_security_epoch)
            .append_i64(row.target_instruction_sequence)
            .append_bytes(&row.admission_set_digest)
            .append_i16(row.admission_count)
            .append_bytes(&row.host_claim_digest)
            .append_i16(row.state)
            .append_i64(row.revision)
            .append_bytes(&row.instruction_digest)
            .append_bytes(&row.chain_digest)
            .append_i64(row.reset_at.timestamp_micros())
            .append_bool(row.materialized);
        append_optional_uuid(&mut leaf, row.acked_by_head_event_id);
        append_optional_time(&mut leaf, row.acked_at);
        append_optional_uuid(&mut leaf, row.direct_ack_operation_id);
        append_optional_bytes(
            &mut leaf,
            row.direct_ack_host_registration_lookup_digest.as_deref(),
        );
        leaf.append_bytes(&row.record_mac);
        hasher.append(leaf);
    }

    append_maple_pairing_authority_category(
        &mut hasher,
        "reset_clear_admissions",
        counts.reset_clear_admissions,
    );
    let mut admission_counts_by_event: BTreeMap<Uuid, i64> = obligations_by_event
        .keys()
        .copied()
        .map(|event_id| (event_id, 0_i64))
        .collect();
    last_id = 0;
    let mut admissions_seen = 0_i64;
    loop {
        let rows = maple_pairing_reset_clear_admissions::table
            .filter(
                maple_pairing_reset_clear_admissions::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .filter(maple_pairing_reset_clear_admissions::id.gt(last_id))
            .order(maple_pairing_reset_clear_admissions::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingResetClearAdmission>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_reset_clear_admission(
                enclave_key,
                &row,
                &head.authority_scope_digest,
            )?;
            let obligation = obligations_by_event
                .get(&row.obligation_uuid)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if obligation.lookup_digest != row.lookup_digest {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let child_count = admission_counts_by_event
                .get_mut(&row.obligation_uuid)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            *child_count = child_count
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if *child_count > i64::from(obligation.admission_count)
                || *child_count > MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut leaf =
                CanonicalBytes::new("os.maple-pair-authority-reset-clear-admission-leaf.v1");
            leaf.append_i64(row.id)
                .append_uuid(row.obligation_uuid)
                .append_bytes(&row.authority_scope_digest)
                .append_bytes(&row.lookup_digest)
                .append_uuid(row.pair_id)
                .append_i64(row.pairing_incarnation)
                .append_bytes(&row.pair_authorization_digest)
                .append_bytes(&row.record_mac)
                .append_i64(row.created_at.timestamp_micros());
            hasher.append(leaf);
            last_id = row.id;
            admissions_seen = admissions_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if admissions_seen != counts.reset_clear_admissions {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    for (event_id, obligation) in &obligations_by_event {
        if admission_counts_by_event.get(event_id).copied()
            != Some(i64::from(obligation.admission_count))
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    // Re-scan in the aggregate's canonical order. Only one obligation's
    // bounded (<=128) aggregate transcript is retained at a time; the first
    // id-ordered pass above authenticated and inventory-hashed every leaf.
    let mut admission_cursor: Option<(Uuid, Uuid, i64)> = None;
    let mut aggregate_event_id: Option<Uuid> = None;
    let mut aggregate_body: Option<CanonicalBytes> = None;
    let mut aggregate_count = 0_i64;
    let mut canonical_admissions_seen = 0_i64;
    loop {
        let mut query = maple_pairing_reset_clear_admissions::table
            .filter(
                maple_pairing_reset_clear_admissions::authority_scope_digest
                    .eq(&head.authority_scope_digest),
            )
            .into_boxed();
        if let Some((cursor_event, cursor_pair, cursor_incarnation)) = admission_cursor {
            query = query.filter(
                maple_pairing_reset_clear_admissions::obligation_uuid
                    .gt(cursor_event)
                    .or(maple_pairing_reset_clear_admissions::obligation_uuid
                        .eq(cursor_event)
                        .and(
                            maple_pairing_reset_clear_admissions::pair_id
                                .gt(cursor_pair)
                                .or(maple_pairing_reset_clear_admissions::pair_id
                                    .eq(cursor_pair)
                                    .and(
                                        maple_pairing_reset_clear_admissions::pairing_incarnation
                                            .gt(cursor_incarnation),
                                    )),
                        )),
            );
        }
        let rows = query
            .order((
                maple_pairing_reset_clear_admissions::obligation_uuid.asc(),
                maple_pairing_reset_clear_admissions::pair_id.asc(),
                maple_pairing_reset_clear_admissions::pairing_incarnation.asc(),
            ))
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .load::<MaplePairingResetClearAdmission>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_reset_clear_admission(
                enclave_key,
                &row,
                &head.authority_scope_digest,
            )?;
            let obligation = obligations_by_event
                .get(&row.obligation_uuid)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if obligation.lookup_digest != row.lookup_digest {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if aggregate_event_id != Some(row.obligation_uuid) {
                if let (Some(previous_event_id), Some(previous_body)) =
                    (aggregate_event_id, aggregate_body.take())
                {
                    let previous = obligations_by_event
                        .get(&previous_event_id)
                        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                    validate_maple_pairing_reset_clear_admission_aggregate(
                        previous,
                        previous_body,
                        aggregate_count,
                    )?;
                }
                let mut body = CanonicalBytes::new("os.maple-reset-clear-admission-set.v1");
                body.append_u16(1).append_u16(
                    obligation
                        .admission_count
                        .try_into()
                        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
                );
                aggregate_event_id = Some(row.obligation_uuid);
                aggregate_body = Some(body);
                aggregate_count = 0;
            }
            aggregate_body
                .as_mut()
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?
                .append_uuid(row.pair_id)
                .append_u64(
                    row.pairing_incarnation
                        .try_into()
                        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
                )
                .append_bytes(&row.pair_authorization_digest);
            aggregate_count = aggregate_count
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if aggregate_count > MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            admission_cursor = Some((row.obligation_uuid, row.pair_id, row.pairing_incarnation));
            canonical_admissions_seen = canonical_admissions_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if let (Some(event_id), Some(body)) = (aggregate_event_id, aggregate_body) {
        let obligation = obligations_by_event
            .get(&event_id)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        validate_maple_pairing_reset_clear_admission_aggregate(obligation, body, aggregate_count)?;
    }
    if canonical_admissions_seen != counts.reset_clear_admissions {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    for obligation in obligations_by_event
        .values()
        .filter(|row| row.admission_count == 0)
    {
        let mut empty = CanonicalBytes::new("os.maple-reset-clear-admission-set.v1");
        empty.append_u16(1).append_u16(0);
        validate_maple_pairing_reset_clear_admission_aggregate(obligation, empty, 0)?;
    }

    // A live installation or one with a still-pending reset-clear obligation
    // is authoritative only in the account's current security epoch. Fully
    // ACKed historical-only chains remain retained but cannot be resurrected.
    for (lookup, event_ids) in &obligations_by_lookup {
        let latest_obligation = event_ids
            .last()
            .and_then(|event_id| obligations_by_event.get(event_id))
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        let latest_highwater = latest_by_lookup
            .get(lookup)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if latest_obligation.state == 1
            && (latest_highwater.security_epoch != head.security_epoch
                || latest_obligation.target_security_epoch != head.security_epoch
                || latest_obligation.target_revocation_stream_id
                    != latest_highwater.revocation_stream_id
                || latest_obligation.target_revocation_stream_generation
                    != latest_highwater.revocation_stream_generation)
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }
    for lookup in &live_lookups {
        if latest_by_lookup
            .get(lookup)
            .is_none_or(|highwater| highwater.security_epoch != head.security_epoch)
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    let expected_inventory = hasher.finish();
    Ok((counts, expected_inventory))
}

fn verify_maple_pairing_authority_account(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    head: &MaplePairingAuthorityAccountHead,
) -> Result<(), DBError> {
    let (counts, expected_inventory) =
        compute_maple_pairing_authority_account_inventory(conn, enclave_key, head)?;
    let authenticated_counts = MaplePairingAuthorityCounts {
        devices: head.device_count,
        device_operations: head.device_operation_count,
        registration_operation_tombstones: head.registration_operation_tombstone_count,
        installation_retirements: head.installation_retirement_count,
        lineages: head.lineage_count,
        pairings: head.pairing_count,
        pairing_operations: head.pairing_operation_count,
        host_states: head.host_state_count,
        revocation_events: head.revocation_event_count,
        highwater_groups: head.highwater_installation_group_count,
        highwater_generations: head.highwater_generation_count,
        reset_clear_obligations: head.reset_clear_obligation_count,
        reset_clear_admissions: head.reset_clear_admission_count,
    };
    if counts != authenticated_counts
        || counts.total_rows() != Some(head.authority_row_count)
        || !maple_pairing_authority_mac_matches(
            &expected_inventory,
            &head.authority_inventory_digest,
        )
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

#[allow(dead_code)]
fn maple_pairing_authority_project_inventory_digest(
    project_id: i32,
    org_id: i32,
    project_uuid: Uuid,
    subject_project_id: Uuid,
    accounts: &[MaplePairingAuthorityAccountHead],
) -> Vec<u8> {
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_PROJECT_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-project-inventory-header.v1");
    header
        .append_i32(project_id)
        .append_i32(org_id)
        .append_uuid(project_uuid)
        .append_uuid(subject_project_id)
        .append_i64(accounts.len() as i64);
    hasher.append(header);
    for account in accounts {
        let mut body = CanonicalBytes::new("os.maple-pair-authority-account-head-leaf.v1");
        body.append_uuid(account.user_id)
            .append_i32(account.project_id)
            .append_i32(account.org_id)
            .append_i64(account.security_epoch)
            .append_bytes(&account.authority_scope_digest)
            .append_bytes(&account.authority_inventory_digest)
            .append_i64(account.authority_row_count)
            .append_i64(account.device_count)
            .append_i64(account.device_operation_count)
            .append_i64(account.lineage_count)
            .append_i64(account.pairing_count)
            .append_i64(account.pairing_operation_count)
            .append_i64(account.host_state_count)
            .append_i64(account.revocation_event_count)
            .append_i64(account.highwater_installation_group_count)
            .append_i64(account.highwater_generation_count)
            .append_i64(account.registration_operation_tombstone_count)
            .append_i64(account.installation_retirement_count)
            .append_i64(account.reset_clear_obligation_count)
            .append_i64(account.reset_clear_admission_count)
            .append_i64(account.revision)
            .append_bytes(&account.record_mac)
            .append_i64(account.created_at.timestamp_micros());
        hasher.append(body);
    }
    hasher.finish()
}

#[allow(dead_code)]
fn maple_pairing_authority_org_inventory_digest(
    org_id: i32,
    projects: &[MaplePairingAuthorityProjectHead],
) -> Vec<u8> {
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_ORG_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-org-inventory-header.v1");
    header.append_i32(org_id).append_i64(projects.len() as i64);
    hasher.append(header);
    for project in projects {
        let mut body = CanonicalBytes::new("os.maple-pair-authority-project-head-leaf.v1");
        body.append_i32(project.project_id)
            .append_i32(project.org_id)
            .append_uuid(project.project_uuid)
            .append_uuid(project.subject_project_id)
            .append_bytes(&project.account_inventory_digest)
            .append_i64(project.account_count)
            .append_i64(project.revision)
            .append_bytes(&project.record_mac)
            .append_i64(project.created_at.timestamp_micros());
        hasher.append(body);
    }
    hasher.finish()
}

#[allow(dead_code)]
fn maple_pairing_authority_global_inventory_digest(
    orgs: &[MaplePairingAuthorityOrgHead],
) -> Vec<u8> {
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_GLOBAL_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-global-inventory-header.v1");
    header.append_i64(orgs.len() as i64);
    hasher.append(header);
    for org in orgs {
        let mut body = CanonicalBytes::new("os.maple-pair-authority-org-head-leaf.v1");
        body.append_i32(org.org_id)
            .append_bool(org.global_singleton)
            .append_bytes(&org.project_inventory_digest)
            .append_i64(org.project_count)
            .append_i64(org.revision)
            .append_bytes(&org.record_mac)
            .append_i64(org.created_at.timestamp_micros());
        hasher.append(body);
    }
    hasher.finish()
}

fn maple_pairing_issuer_key_inventory_digest_from_fingerprints(
    fingerprints: &[MaplePairingIssuerKeyFingerprintV1],
) -> Result<Vec<u8>, DBError> {
    if fingerprints.len() > MAPLE_PAIRING_MAX_ISSUER_KEYS {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_ISSUER_KEY_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-issuer-key-inventory-header.v1");
    header.append_i64(
        i64::try_from(fingerprints.len()).map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
    );
    hasher.append(header);
    let mut previous_key_id: Option<&str> = None;
    for fingerprint in fingerprints {
        if !maple_pairing_issuer_key_id_is_valid(&fingerprint.key_id)
            || previous_key_id.is_some_and(|previous| previous >= fingerprint.key_id.as_str())
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        previous_key_id = Some(&fingerprint.key_id);
        let mut leaf = CanonicalBytes::new("os.maple-pair-issuer-key-inventory-leaf.v1");
        leaf.append_str(&fingerprint.key_id)
            .append_str(fingerprint.algorithm.as_wire())
            .append_bytes(&fingerprint.public_key_digest);
        hasher.append(leaf);
    }
    Ok(hasher.finish())
}

fn compute_maple_pairing_issuer_key_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::maple_pairing_issuer_keys;

    let count = maple_pairing_issuer_keys::table
        .count()
        .get_result::<i64>(conn)?;
    if !(0..=i64::try_from(MAPLE_PAIRING_MAX_ISSUER_KEYS)
        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?)
        .contains(&count)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_ISSUER_KEY_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-issuer-key-inventory-header.v1");
    header.append_i64(count);
    hasher.append(header);
    let mut cursor = String::new();
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_issuer_keys::table
            .filter(maple_pairing_issuer_keys::key_id.gt(&cursor))
            .order(maple_pairing_issuer_keys::key_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingIssuerKey>(conn)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_issuer_key(enclave_key, &row)?;
            let mut leaf = CanonicalBytes::new("os.maple-pair-issuer-key-inventory-leaf.v1");
            leaf.append_str(&row.key_id)
                .append_str(&row.algorithm)
                .append_bytes(&row.public_key_digest);
            hasher.append(leaf);
            cursor = row.key_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((count, hasher.finish()))
}

fn append_maple_pairing_authority_account_head_leaf(
    hasher: &mut MaplePairingAuthorityInventoryHasher,
    account: &MaplePairingAuthorityAccountHead,
) {
    let mut body = CanonicalBytes::new("os.maple-pair-authority-account-head-leaf.v1");
    body.append_uuid(account.user_id)
        .append_i32(account.project_id)
        .append_i32(account.org_id)
        .append_i64(account.security_epoch)
        .append_bytes(&account.authority_scope_digest)
        .append_bytes(&account.authority_inventory_digest)
        .append_i64(account.authority_row_count)
        .append_i64(account.device_count)
        .append_i64(account.device_operation_count)
        .append_i64(account.lineage_count)
        .append_i64(account.pairing_count)
        .append_i64(account.pairing_operation_count)
        .append_i64(account.host_state_count)
        .append_i64(account.revocation_event_count)
        .append_i64(account.highwater_installation_group_count)
        .append_i64(account.highwater_generation_count)
        .append_i64(account.registration_operation_tombstone_count)
        .append_i64(account.installation_retirement_count)
        .append_i64(account.reset_clear_obligation_count)
        .append_i64(account.reset_clear_admission_count)
        .append_i64(account.revision)
        .append_bytes(&account.record_mac)
        .append_i64(account.created_at.timestamp_micros());
    hasher.append(body);
}

fn compute_maple_pairing_authority_project_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
    org_id: i32,
    project_uuid: Uuid,
    subject_project_id: Uuid,
    verify_account_leaves: bool,
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::{maple_pairing_authority_account_heads, org_projects, users};
    use diesel::OptionalExtension;

    if org_projects::table
        .filter(org_projects::id.eq(project_id))
        .filter(org_projects::org_id.eq(org_id))
        .filter(org_projects::uuid.eq(project_uuid))
        .filter(org_projects::client_id.eq(subject_project_id))
        .select(org_projects::id)
        .for_share()
        .first::<i32>(conn)
        .optional()?
        .is_none()
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let account_count = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .filter(maple_pairing_authority_account_heads::org_id.eq(org_id))
        .count()
        .get_result::<i64>(conn)?;
    if account_count
        != users::table
            .filter(users::project_id.eq(project_id))
            .count()
            .get_result::<i64>(conn)?
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_PROJECT_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-project-inventory-header.v1");
    header
        .append_i32(project_id)
        .append_i32(org_id)
        .append_uuid(project_uuid)
        .append_uuid(subject_project_id)
        .append_i64(account_count);
    hasher.append(header);
    let mut cursor = Uuid::nil();
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
            .filter(maple_pairing_authority_account_heads::org_id.eq(org_id))
            .filter(maple_pairing_authority_account_heads::user_id.gt(cursor))
            .order(maple_pairing_authority_account_heads::user_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingAuthorityAccountHead>(conn)?;
        if rows.is_empty() {
            break;
        }
        for account in rows {
            validate_maple_pairing_authority_account_head(enclave_key, &account)?;
            if users::table
                .filter(users::uuid.eq(account.user_id))
                .filter(users::project_id.eq(project_id))
                .select(users::uuid)
                .for_share()
                .first::<Uuid>(conn)
                .optional()?
                .is_none()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            if verify_account_leaves {
                verify_maple_pairing_authority_account(conn, enclave_key, &account)?;
            }
            append_maple_pairing_authority_account_head_leaf(&mut hasher, &account);
            cursor = account.user_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != account_count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((account_count, hasher.finish()))
}

fn append_maple_pairing_authority_project_head_leaf(
    hasher: &mut MaplePairingAuthorityInventoryHasher,
    project: &MaplePairingAuthorityProjectHead,
) {
    let mut body = CanonicalBytes::new("os.maple-pair-authority-project-head-leaf.v1");
    body.append_i32(project.project_id)
        .append_i32(project.org_id)
        .append_uuid(project.project_uuid)
        .append_uuid(project.subject_project_id)
        .append_bytes(&project.account_inventory_digest)
        .append_i64(project.account_count)
        .append_i64(project.revision)
        .append_bytes(&project.record_mac)
        .append_i64(project.created_at.timestamp_micros());
    hasher.append(body);
}

fn compute_maple_pairing_authority_org_inventory_shallow(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    org_id: i32,
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::{maple_pairing_authority_project_heads, org_projects, orgs};
    use diesel::OptionalExtension;

    if orgs::table
        .filter(orgs::id.eq(org_id))
        .select(orgs::id)
        .for_share()
        .first::<i32>(conn)
        .optional()?
        .is_none()
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let project_count = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
        .count()
        .get_result::<i64>(conn)?;
    if project_count
        != org_projects::table
            .filter(org_projects::org_id.eq(org_id))
            .count()
            .get_result::<i64>(conn)?
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_ORG_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-org-inventory-header.v1");
    header.append_i32(org_id).append_i64(project_count);
    hasher.append(header);
    let mut cursor = 0_i32;
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
            .filter(maple_pairing_authority_project_heads::project_id.gt(cursor))
            .order(maple_pairing_authority_project_heads::project_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingAuthorityProjectHead>(conn)?;
        if rows.is_empty() {
            break;
        }
        for project in rows {
            validate_maple_pairing_authority_project_head(enclave_key, &project)?;
            if org_projects::table
                .filter(org_projects::id.eq(project.project_id))
                .filter(org_projects::org_id.eq(project.org_id))
                .filter(org_projects::uuid.eq(project.project_uuid))
                .filter(org_projects::client_id.eq(project.subject_project_id))
                .select(org_projects::id)
                .for_share()
                .first::<i32>(conn)
                .optional()?
                .is_none()
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            append_maple_pairing_authority_project_head_leaf(&mut hasher, &project);
            cursor = project.project_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != project_count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((project_count, hasher.finish()))
}

fn compute_maple_pairing_authority_org_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    org_id: i32,
    verify_account_leaves: bool,
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::{maple_pairing_authority_project_heads, org_projects, orgs};
    use diesel::OptionalExtension;

    if orgs::table
        .filter(orgs::id.eq(org_id))
        .select(orgs::id)
        .for_share()
        .first::<i32>(conn)
        .optional()?
        .is_none()
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let project_count = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
        .count()
        .get_result::<i64>(conn)?;
    if project_count
        != org_projects::table
            .filter(org_projects::org_id.eq(org_id))
            .count()
            .get_result::<i64>(conn)?
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_ORG_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-org-inventory-header.v1");
    header.append_i32(org_id).append_i64(project_count);
    hasher.append(header);
    let mut cursor = 0_i32;
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
            .filter(maple_pairing_authority_project_heads::project_id.gt(cursor))
            .order(maple_pairing_authority_project_heads::project_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingAuthorityProjectHead>(conn)?;
        if rows.is_empty() {
            break;
        }
        for project in rows {
            validate_maple_pairing_authority_project_head(enclave_key, &project)?;
            let (account_count, expected) = compute_maple_pairing_authority_project_inventory(
                conn,
                enclave_key,
                project.project_id,
                project.org_id,
                project.project_uuid,
                project.subject_project_id,
                verify_account_leaves,
            )?;
            if project.account_count != account_count
                || !maple_pairing_authority_mac_matches(
                    &expected,
                    &project.account_inventory_digest,
                )
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            append_maple_pairing_authority_project_head_leaf(&mut hasher, &project);
            cursor = project.project_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != project_count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((project_count, hasher.finish()))
}

fn append_maple_pairing_authority_org_head_leaf(
    hasher: &mut MaplePairingAuthorityInventoryHasher,
    org: &MaplePairingAuthorityOrgHead,
) {
    let mut body = CanonicalBytes::new("os.maple-pair-authority-org-head-leaf.v1");
    body.append_i32(org.org_id)
        .append_bool(org.global_singleton)
        .append_bytes(&org.project_inventory_digest)
        .append_i64(org.project_count)
        .append_i64(org.revision)
        .append_bytes(&org.record_mac)
        .append_i64(org.created_at.timestamp_micros());
    hasher.append(body);
}

fn compute_maple_pairing_authority_global_inventory_shallow(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::{maple_pairing_authority_org_heads, orgs};

    let org_count = maple_pairing_authority_org_heads::table
        .count()
        .get_result::<i64>(conn)?;
    if org_count != orgs::table.count().get_result::<i64>(conn)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_GLOBAL_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-global-inventory-header.v1");
    header.append_i64(org_count);
    hasher.append(header);
    let mut cursor = 0_i32;
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.gt(cursor))
            .order(maple_pairing_authority_org_heads::org_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingAuthorityOrgHead>(conn)?;
        if rows.is_empty() {
            break;
        }
        for org in rows {
            validate_maple_pairing_authority_org_head(enclave_key, &org)?;
            append_maple_pairing_authority_org_head_leaf(&mut hasher, &org);
            cursor = org.org_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != org_count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((org_count, hasher.finish()))
}

fn compute_maple_pairing_authority_global_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    verify_account_leaves: bool,
) -> Result<(i64, Vec<u8>), DBError> {
    use crate::models::schema::{maple_pairing_authority_org_heads, orgs};

    let org_count = maple_pairing_authority_org_heads::table
        .count()
        .get_result::<i64>(conn)?;
    if org_count != orgs::table.count().get_result::<i64>(conn)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let mut hasher =
        MaplePairingAuthorityInventoryHasher::new(MAPLE_PAIRING_AUTHORITY_GLOBAL_INVENTORY_DOMAIN);
    let mut header = CanonicalBytes::new("os.maple-pair-authority-global-inventory-header.v1");
    header.append_i64(org_count);
    hasher.append(header);
    let mut cursor = 0_i32;
    let mut seen = 0_i64;
    loop {
        let rows = maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.gt(cursor))
            .order(maple_pairing_authority_org_heads::org_id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_share()
            .load::<MaplePairingAuthorityOrgHead>(conn)?;
        if rows.is_empty() {
            break;
        }
        for org in rows {
            validate_maple_pairing_authority_org_head(enclave_key, &org)?;
            let (project_count, expected) = compute_maple_pairing_authority_org_inventory(
                conn,
                enclave_key,
                org.org_id,
                verify_account_leaves,
            )?;
            if org.project_count != project_count
                || !maple_pairing_authority_mac_matches(&expected, &org.project_inventory_digest)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            append_maple_pairing_authority_org_head_leaf(&mut hasher, &org);
            cursor = org.org_id;
            seen = seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        }
    }
    if seen != org_count {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok((org_count, hasher.finish()))
}

fn validate_maple_pairing_authority_project_head(
    enclave_key: &[u8],
    head: &MaplePairingAuthorityProjectHead,
) -> Result<(), DBError> {
    let expected = maple_pairing_authority_project_head_mac(enclave_key, head)?;
    if head.project_id <= 0
        || head.org_id <= 0
        || head.project_uuid.is_nil()
        || head.subject_project_id.is_nil()
        || head.account_inventory_digest.len() != 32
        || head.account_count < 0
        || head.revision <= 0
        || head.updated_at < head.created_at
        || !maple_pairing_authority_mac_matches(&expected, &head.record_mac)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn validate_maple_pairing_authority_org_head(
    enclave_key: &[u8],
    head: &MaplePairingAuthorityOrgHead,
) -> Result<(), DBError> {
    let expected = maple_pairing_authority_org_head_mac(enclave_key, head)?;
    if head.org_id <= 0
        || !head.global_singleton
        || head.project_inventory_digest.len() != 32
        || head.project_count < 0
        || head.revision <= 0
        || head.updated_at < head.created_at
        || !maple_pairing_authority_mac_matches(&expected, &head.record_mac)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn validate_maple_pairing_authority_global_head(
    enclave_key: &[u8],
    head: &MaplePairingAuthorityGlobalHead,
) -> Result<(), DBError> {
    let Some(actual_mac) = head.record_mac.as_deref() else {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    };
    let expected = maple_pairing_authority_global_head_mac(enclave_key, head)?;
    if !head.singleton
        || head.activation_state != MAPLE_PAIRING_AUTHORITY_ACTIVE
        || head.org_inventory_digest.len() != 32
        || head.org_count < 0
        || head.issuer_key_inventory_digest.len() != 32
        || !(0..=i64::try_from(MAPLE_PAIRING_MAX_ISSUER_KEYS)
            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?)
            .contains(&head.issuer_key_count)
        || head.revision < 2
        || head.updated_at < head.created_at
        || !maple_pairing_authority_mac_matches(&expected, actual_mac)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn validate_pending_maple_pairing_authority_global_head(
    head: &MaplePairingAuthorityGlobalHead,
) -> Result<(), DBError> {
    let zero_digest = [0_u8; 32];
    if !head.singleton
        || head.activation_state != MAPLE_PAIRING_AUTHORITY_PENDING
        || head.org_count != 0
        || head.issuer_key_count != 0
        || head.revision != 1
        || head.record_mac.is_some()
        || head.updated_at < head.created_at
        || !maple_pairing_authority_mac_matches(&zero_digest, &head.org_inventory_digest)
        || !maple_pairing_authority_mac_matches(&zero_digest, &head.issuer_key_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn load_maple_pairing_authority_global_head(
    conn: &mut PgConnection,
) -> Result<MaplePairingAuthorityGlobalHead, DBError> {
    use crate::models::schema::maple_pairing_authority_global_heads;
    use diesel::OptionalExtension;

    maple_pairing_authority_global_heads::table
        .filter(maple_pairing_authority_global_heads::singleton.eq(true))
        .for_update()
        .first::<MaplePairingAuthorityGlobalHead>(conn)
        .optional()?
        .ok_or(DBError::MaplePairingAuthorityCorrupt)
}

fn maple_pairing_authority_leaf_tables_are_empty(conn: &mut PgConnection) -> Result<bool, DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_authority_account_heads,
        maple_pairing_authority_org_heads, maple_pairing_authority_project_heads,
        maple_pairing_host_states, maple_pairing_installation_retirements,
        maple_pairing_issuer_keys, maple_pairing_lineages, maple_pairing_operations,
        maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_admissions,
        maple_pairing_reset_clear_obligations, maple_pairing_revocation_events,
        maple_pairing_revocation_highwaters, maple_pairings,
    };

    let counts = [
        maple_devices::table.count().get_result::<i64>(conn)?,
        maple_device_registration_operations::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_registration_operation_tombstones::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_installation_retirements::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_issuer_keys::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_lineages::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairings::table.count().get_result::<i64>(conn)?,
        maple_pairing_operations::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_host_states::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_revocation_events::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_revocation_highwaters::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_reset_clear_obligations::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_reset_clear_admissions::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_authority_account_heads::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_authority_project_heads::table
            .count()
            .get_result::<i64>(conn)?,
        maple_pairing_authority_org_heads::table
            .count()
            .get_result::<i64>(conn)?,
    ];
    Ok(counts.into_iter().all(|count| count == 0))
}

fn create_empty_maple_pairing_authority_account_head(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    org_id: i32,
    created_at: DateTime<Utc>,
) -> Result<MaplePairingAuthorityAccountHead, DBError> {
    use crate::models::schema::maple_pairing_authority_account_heads;

    let authority_scope_digest =
        maple_pairing_authority_scope_digest(enclave_key, user_id, project_id)?;
    let authority_inventory_digest = empty_maple_pairing_authority_account_inventory(
        user_id,
        project_id,
        org_id,
        &authority_scope_digest,
    );
    let mut candidate = MaplePairingAuthorityAccountHead {
        user_id,
        project_id,
        org_id,
        security_epoch: 1,
        authority_scope_digest: authority_scope_digest.clone(),
        authority_inventory_digest: authority_inventory_digest.clone(),
        authority_row_count: 0,
        device_count: 0,
        device_operation_count: 0,
        lineage_count: 0,
        pairing_count: 0,
        pairing_operation_count: 0,
        host_state_count: 0,
        revocation_event_count: 0,
        highwater_installation_group_count: 0,
        highwater_generation_count: 0,
        registration_operation_tombstone_count: 0,
        installation_retirement_count: 0,
        reset_clear_obligation_count: 0,
        reset_clear_admission_count: 0,
        revision: 1,
        record_mac: Vec::new(),
        created_at,
        updated_at: created_at,
    };
    candidate.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &candidate)?;
    let row = diesel::insert_into(maple_pairing_authority_account_heads::table)
        .values(NewMaplePairingAuthorityAccountHead {
            user_id,
            project_id,
            org_id,
            security_epoch: 1,
            authority_scope_digest,
            authority_inventory_digest,
            authority_row_count: 0,
            device_count: 0,
            device_operation_count: 0,
            lineage_count: 0,
            pairing_count: 0,
            pairing_operation_count: 0,
            host_state_count: 0,
            revocation_event_count: 0,
            highwater_installation_group_count: 0,
            highwater_generation_count: 0,
            registration_operation_tombstone_count: 0,
            installation_retirement_count: 0,
            reset_clear_obligation_count: 0,
            reset_clear_admission_count: 0,
            revision: 1,
            record_mac: candidate.record_mac,
            created_at,
        })
        .get_result::<MaplePairingAuthorityAccountHead>(conn)?;
    validate_maple_pairing_authority_account_head(enclave_key, &row)?;
    Ok(row)
}

fn create_maple_pairing_authority_project_head(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project: &OrgProject,
    created_at: DateTime<Utc>,
) -> Result<MaplePairingAuthorityProjectHead, DBError> {
    use crate::models::schema::maple_pairing_authority_project_heads;

    let (account_count, account_inventory_digest) =
        compute_maple_pairing_authority_project_inventory(
            conn,
            enclave_key,
            project.id,
            project.org_id,
            project.uuid,
            project.client_id,
            false,
        )?;
    let mut candidate = MaplePairingAuthorityProjectHead {
        project_id: project.id,
        org_id: project.org_id,
        project_uuid: project.uuid,
        subject_project_id: project.client_id,
        account_inventory_digest: account_inventory_digest.clone(),
        account_count,
        revision: 1,
        record_mac: Vec::new(),
        created_at,
        updated_at: created_at,
    };
    candidate.record_mac = maple_pairing_authority_project_head_mac(enclave_key, &candidate)?;
    let row = diesel::insert_into(maple_pairing_authority_project_heads::table)
        .values(NewMaplePairingAuthorityProjectHead {
            project_id: project.id,
            org_id: project.org_id,
            project_uuid: project.uuid,
            subject_project_id: project.client_id,
            account_inventory_digest,
            account_count,
            revision: 1,
            record_mac: candidate.record_mac,
            created_at,
        })
        .get_result::<MaplePairingAuthorityProjectHead>(conn)?;
    validate_maple_pairing_authority_project_head(enclave_key, &row)?;
    Ok(row)
}

fn create_maple_pairing_authority_org_head(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    org_id: i32,
    created_at: DateTime<Utc>,
) -> Result<MaplePairingAuthorityOrgHead, DBError> {
    use crate::models::schema::maple_pairing_authority_org_heads;

    let (project_count, project_inventory_digest) =
        compute_maple_pairing_authority_org_inventory_shallow(conn, enclave_key, org_id)?;
    let mut candidate = MaplePairingAuthorityOrgHead {
        org_id,
        global_singleton: true,
        project_inventory_digest: project_inventory_digest.clone(),
        project_count,
        revision: 1,
        record_mac: Vec::new(),
        created_at,
        updated_at: created_at,
    };
    candidate.record_mac = maple_pairing_authority_org_head_mac(enclave_key, &candidate)?;
    let row = diesel::insert_into(maple_pairing_authority_org_heads::table)
        .values(NewMaplePairingAuthorityOrgHead {
            org_id,
            global_singleton: true,
            project_inventory_digest,
            project_count,
            revision: 1,
            record_mac: candidate.record_mac,
            created_at,
        })
        .get_result::<MaplePairingAuthorityOrgHead>(conn)?;
    validate_maple_pairing_authority_org_head(enclave_key, &row)?;
    Ok(row)
}

fn verify_maple_pairing_authority_tree(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(), DBError> {
    verify_maple_pairing_authority_tree_with_mode(conn, enclave_key, true)
}

fn verify_maple_pairing_issuer_key_inventory(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    global: &MaplePairingAuthorityGlobalHead,
) -> Result<(), DBError> {
    let (issuer_key_count, issuer_key_inventory_digest) =
        compute_maple_pairing_issuer_key_inventory(conn, enclave_key)?;
    if global.issuer_key_count != issuer_key_count
        || !maple_pairing_authority_mac_matches(
            &issuer_key_inventory_digest,
            &global.issuer_key_inventory_digest,
        )
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn reconcile_maple_pairing_issuer_key_registry(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    configured: &[MaplePairingIssuerKeyFingerprintV1],
) -> Result<(i64, Vec<u8>, bool), DBError> {
    use crate::models::schema::maple_pairing_issuer_keys;

    let configured_digest =
        maple_pairing_issuer_key_inventory_digest_from_fingerprints(configured)?;
    let existing = maple_pairing_issuer_keys::table
        .order(maple_pairing_issuer_keys::key_id.asc())
        .for_update()
        .load::<MaplePairingIssuerKey>(conn)?;
    if existing.len() > MAPLE_PAIRING_MAX_ISSUER_KEYS || existing.len() > configured.len() {
        return Err(DBError::MaplePairingIssuerConfigurationConflict);
    }
    for row in &existing {
        validate_maple_pairing_issuer_key(enclave_key, row)?;
        let configured_row = configured
            .binary_search_by(|candidate| candidate.key_id.as_str().cmp(&row.key_id))
            .ok()
            .map(|index| &configured[index])
            .ok_or(DBError::MaplePairingIssuerConfigurationConflict)?;
        if row.algorithm != configured_row.algorithm.as_wire()
            || row.public_key_digest.as_slice() != configured_row.public_key_digest.as_slice()
        {
            return Err(DBError::MaplePairingIssuerConfigurationConflict);
        }
    }

    let mut inserted = false;
    let created_at = maple_pairing_trusted_db_now(conn)?;
    for fingerprint in configured {
        if existing
            .binary_search_by(|row| row.key_id.as_str().cmp(&fingerprint.key_id))
            .is_ok()
        {
            continue;
        }
        let algorithm = fingerprint.algorithm.as_wire().to_string();
        let public_key_digest = fingerprint.public_key_digest.to_vec();
        let record_mac = maple_pairing_issuer_key_record_mac_for_parts(
            enclave_key,
            &fingerprint.key_id,
            true,
            &algorithm,
            &public_key_digest,
            created_at,
        )?;
        let row = diesel::insert_into(maple_pairing_issuer_keys::table)
            .values(NewMaplePairingIssuerKey {
                key_id: fingerprint.key_id.clone(),
                global_singleton: true,
                algorithm,
                public_key_digest,
                record_mac,
                created_at,
            })
            .get_result::<MaplePairingIssuerKey>(conn)
            .map_err(|error| match error {
                diesel::result::Error::DatabaseError(
                    diesel::result::DatabaseErrorKind::UniqueViolation,
                    _,
                ) => DBError::MaplePairingIssuerConfigurationConflict,
                other => DBError::QueryError(other),
            })?;
        validate_maple_pairing_issuer_key(enclave_key, &row)?;
        inserted = true;
    }
    let (count, stored_digest) = compute_maple_pairing_issuer_key_inventory(conn, enclave_key)?;
    if usize::try_from(count).ok() != Some(configured.len())
        || !maple_pairing_authority_mac_matches(&configured_digest, &stored_digest)
    {
        return Err(DBError::MaplePairingIssuerConfigurationConflict);
    }
    Ok((count, stored_digest, inserted))
}

fn update_maple_pairing_issuer_key_inventory_root(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    issuer_key_count: i64,
    issuer_key_inventory_digest: Vec<u8>,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_global_heads;

    let mut global = load_maple_pairing_authority_global_head(conn)?;
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    global.issuer_key_count = issuer_key_count;
    global.issuer_key_inventory_digest = issuer_key_inventory_digest;
    global.revision = global
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    global.record_mac = Some(maple_pairing_authority_global_head_mac(
        enclave_key,
        &global,
    )?);
    let changed = diesel::update(
        maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true))
            .filter(maple_pairing_authority_global_heads::revision.eq(global.revision - 1)),
    )
    .set((
        maple_pairing_authority_global_heads::issuer_key_inventory_digest
            .eq(&global.issuer_key_inventory_digest),
        maple_pairing_authority_global_heads::issuer_key_count.eq(global.issuer_key_count),
        maple_pairing_authority_global_heads::revision.eq(global.revision),
        maple_pairing_authority_global_heads::record_mac.eq(global.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn verify_maple_pairing_authority_scoped_chain(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    account: &MaplePairingAuthorityAccountHead,
) -> Result<MaplePairingAuthorityProjectHead, DBError> {
    use crate::models::schema::{
        maple_pairing_authority_org_heads, maple_pairing_authority_project_heads,
    };

    verify_maple_pairing_authority_account(conn, enclave_key, account)?;
    let project = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::project_id.eq(account.project_id))
        .filter(maple_pairing_authority_project_heads::org_id.eq(account.org_id))
        .for_share()
        .first::<MaplePairingAuthorityProjectHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_authority_project_head(enclave_key, &project)?;
    let (account_count, account_digest) = compute_maple_pairing_authority_project_inventory(
        conn,
        enclave_key,
        account.project_id,
        account.org_id,
        project.project_uuid,
        project.subject_project_id,
        false,
    )?;
    if project.account_count != account_count
        || !maple_pairing_authority_mac_matches(&account_digest, &project.account_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    let org = maple_pairing_authority_org_heads::table
        .filter(maple_pairing_authority_org_heads::org_id.eq(account.org_id))
        .for_share()
        .first::<MaplePairingAuthorityOrgHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_authority_org_head(enclave_key, &org)?;
    let (project_count, project_digest) =
        compute_maple_pairing_authority_org_inventory_shallow(conn, enclave_key, account.org_id)?;
    if org.project_count != project_count
        || !maple_pairing_authority_mac_matches(&project_digest, &org.project_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    let global = load_maple_pairing_authority_global_head(conn)?;
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    if !AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let (org_count, org_digest) =
        compute_maple_pairing_authority_global_inventory_shallow(conn, enclave_key)?;
    if global.org_count != org_count
        || !maple_pairing_authority_mac_matches(&org_digest, &global.org_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    verify_maple_pairing_issuer_key_inventory(conn, enclave_key, &global)?;
    Ok(project)
}

fn verify_maple_pairing_authority_global_shallow(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(), DBError> {
    let global = load_maple_pairing_authority_global_head(conn)?;
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    if !AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    let (org_count, org_digest) =
        compute_maple_pairing_authority_global_inventory_shallow(conn, enclave_key)?;
    if global.org_count != org_count
        || !maple_pairing_authority_mac_matches(&org_digest, &global.org_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    verify_maple_pairing_issuer_key_inventory(conn, enclave_key, &global)
}

fn verify_maple_pairing_authority_org_chain(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    org_id: i32,
    verify_account_leaves: bool,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_org_heads;

    let org = maple_pairing_authority_org_heads::table
        .filter(maple_pairing_authority_org_heads::org_id.eq(org_id))
        .for_share()
        .first::<MaplePairingAuthorityOrgHead>(conn)?;
    validate_maple_pairing_authority_org_head(enclave_key, &org)?;
    let (project_count, project_digest) = if verify_account_leaves {
        compute_maple_pairing_authority_org_inventory(conn, enclave_key, org_id, true)?
    } else {
        compute_maple_pairing_authority_org_inventory_shallow(conn, enclave_key, org_id)?
    };
    if org.project_count != project_count
        || !maple_pairing_authority_mac_matches(&project_digest, &org.project_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    verify_maple_pairing_authority_global_shallow(conn, enclave_key)
}

fn verify_maple_pairing_authority_project_chain(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
    org_id: i32,
    verify_account_leaves: bool,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_project_heads;

    let project = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::project_id.eq(project_id))
        .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
        .for_share()
        .first::<MaplePairingAuthorityProjectHead>(conn)?;
    validate_maple_pairing_authority_project_head(enclave_key, &project)?;
    let (account_count, account_digest) = compute_maple_pairing_authority_project_inventory(
        conn,
        enclave_key,
        project_id,
        org_id,
        project.project_uuid,
        project.subject_project_id,
        verify_account_leaves,
    )?;
    if project.account_count != account_count
        || !maple_pairing_authority_mac_matches(&account_digest, &project.account_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    verify_maple_pairing_authority_org_chain(conn, enclave_key, org_id, false)
}

fn verify_maple_pairing_authority_tree_with_mode(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    verify_account_leaves: bool,
) -> Result<(), DBError> {
    let global = load_maple_pairing_authority_global_head(conn)?;
    if !AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)? {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    let (org_count, expected_global) =
        compute_maple_pairing_authority_global_inventory(conn, enclave_key, verify_account_leaves)?;
    if global.org_count != org_count
        || !maple_pairing_authority_mac_matches(&expected_global, &global.org_inventory_digest)
    {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    verify_maple_pairing_issuer_key_inventory(conn, enclave_key, &global)
}

fn bootstrap_or_audit_maple_pairing_authority_in_tx(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    configured_issuer_keys: &[MaplePairingIssuerKeyFingerprintV1],
) -> Result<Vec<u8>, DBError> {
    use crate::models::schema::{
        maple_pairing_authority_account_heads, maple_pairing_authority_global_heads,
        maple_pairing_authority_org_heads, maple_pairing_authority_project_heads, org_projects,
        orgs, users,
    };

    let _timer = MaplePairingAuthorityTransactionTimer::start("bootstrap_or_audit");
    let global = load_maple_pairing_authority_global_head(conn)?;
    let marker_exists = AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)?;
    match global.activation_state {
        MAPLE_PAIRING_AUTHORITY_ACTIVE => {
            if !marker_exists {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            verify_maple_pairing_authority_tree(conn, enclave_key)?;
            let (issuer_key_count, issuer_key_inventory_digest, inserted) =
                reconcile_maple_pairing_issuer_key_registry(
                    conn,
                    enclave_key,
                    configured_issuer_keys,
                )?;
            if inserted {
                update_maple_pairing_issuer_key_inventory_root(
                    conn,
                    enclave_key,
                    issuer_key_count,
                    issuer_key_inventory_digest.clone(),
                )?;
                verify_maple_pairing_authority_tree(conn, enclave_key)?;
            }
            Ok(issuer_key_inventory_digest)
        }
        MAPLE_PAIRING_AUTHORITY_PENDING => {
            let zero_digest = [0_u8; 32];
            if marker_exists
                || !global.singleton
                || global.org_count != 0
                || global.issuer_key_count != 0
                || global.revision != 1
                || global.record_mac.is_some()
                || !maple_pairing_authority_mac_matches(&zero_digest, &global.org_inventory_digest)
                || !maple_pairing_authority_mac_matches(
                    &zero_digest,
                    &global.issuer_key_inventory_digest,
                )
                || !maple_pairing_authority_leaf_tables_are_empty(conn)?
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }

            let created_at = maple_pairing_trusted_db_now(conn)?;
            let mut user_cursor = Uuid::nil();
            loop {
                let page = users::table
                    .filter(users::uuid.gt(user_cursor))
                    .order(users::uuid.asc())
                    .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                    .for_update()
                    .load::<User>(conn)?;
                if page.is_empty() {
                    break;
                }
                for user in page {
                    let project = org_projects::table
                        .filter(org_projects::id.eq(user.project_id))
                        .for_update()
                        .first::<OrgProject>(conn)
                        .map_err(|error| match error {
                            diesel::result::Error::NotFound => {
                                DBError::MaplePairingAuthorityCorrupt
                            }
                            other => DBError::QueryError(other),
                        })?;
                    create_empty_maple_pairing_authority_account_head(
                        conn,
                        enclave_key,
                        user.uuid,
                        user.project_id,
                        project.org_id,
                        created_at,
                    )?;
                    user_cursor = user.uuid;
                }
            }

            let mut project_cursor = 0_i32;
            loop {
                let page = org_projects::table
                    .filter(org_projects::id.gt(project_cursor))
                    .order(org_projects::id.asc())
                    .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                    .for_update()
                    .load::<OrgProject>(conn)?;
                if page.is_empty() {
                    break;
                }
                for project in page {
                    create_maple_pairing_authority_project_head(
                        conn,
                        enclave_key,
                        &project,
                        created_at,
                    )?;
                    project_cursor = project.id;
                }
            }

            let mut org_cursor = 0_i32;
            loop {
                let page = orgs::table
                    .filter(orgs::id.gt(org_cursor))
                    .order(orgs::id.asc())
                    .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                    .for_update()
                    .load::<Org>(conn)?;
                if page.is_empty() {
                    break;
                }
                for org in page {
                    create_maple_pairing_authority_org_head(conn, enclave_key, org.id, created_at)?;
                    org_cursor = org.id;
                }
            }

            let (issuer_key_count, issuer_key_inventory_digest, _) =
                reconcile_maple_pairing_issuer_key_registry(
                    conn,
                    enclave_key,
                    configured_issuer_keys,
                )?;
            NewAppDataMigration::new(MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER).insert(conn)?;
            let mut active = global.clone();
            active.activation_state = MAPLE_PAIRING_AUTHORITY_ACTIVE;
            let (org_count, org_inventory_digest) =
                compute_maple_pairing_authority_global_inventory(conn, enclave_key, false)?;
            active.org_inventory_digest = org_inventory_digest;
            active.org_count = org_count;
            active.issuer_key_inventory_digest = issuer_key_inventory_digest.clone();
            active.issuer_key_count = issuer_key_count;
            active.revision = 2;
            active.record_mac = Some(maple_pairing_authority_global_head_mac(
                enclave_key,
                &active,
            )?);
            let changed = diesel::update(
                maple_pairing_authority_global_heads::table
                    .filter(maple_pairing_authority_global_heads::singleton.eq(true))
                    .filter(
                        maple_pairing_authority_global_heads::activation_state
                            .eq(MAPLE_PAIRING_AUTHORITY_PENDING),
                    )
                    .filter(maple_pairing_authority_global_heads::revision.eq(1_i64)),
            )
            .set((
                maple_pairing_authority_global_heads::activation_state
                    .eq(MAPLE_PAIRING_AUTHORITY_ACTIVE),
                maple_pairing_authority_global_heads::org_inventory_digest
                    .eq(active.org_inventory_digest),
                maple_pairing_authority_global_heads::org_count.eq(active.org_count),
                maple_pairing_authority_global_heads::issuer_key_inventory_digest
                    .eq(active.issuer_key_inventory_digest),
                maple_pairing_authority_global_heads::issuer_key_count.eq(active.issuer_key_count),
                maple_pairing_authority_global_heads::revision.eq(active.revision),
                maple_pairing_authority_global_heads::record_mac.eq(active.record_mac),
            ))
            .execute(conn)?;
            if changed != 1 {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            // Explicit reads make trigger-modified rows part of the same
            // authenticated activation decision.
            let _ = (
                maple_pairing_authority_account_heads::table
                    .count()
                    .get_result::<i64>(conn)?,
                maple_pairing_authority_project_heads::table
                    .count()
                    .get_result::<i64>(conn)?,
                maple_pairing_authority_org_heads::table
                    .count()
                    .get_result::<i64>(conn)?,
            );
            verify_maple_pairing_authority_tree(conn, enclave_key)?;
            Ok(issuer_key_inventory_digest)
        }
        _ => Err(DBError::MaplePairingAuthorityCorrupt),
    }
}

fn cascade_maple_pairing_authority_heads(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
) -> Result<(), DBError> {
    use crate::models::schema::{
        maple_pairing_authority_account_heads, maple_pairing_authority_global_heads,
        maple_pairing_authority_org_heads, maple_pairing_authority_project_heads,
    };

    let account = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .for_update()
        .first::<MaplePairingAuthorityAccountHead>(conn)?;
    validate_maple_pairing_authority_account_head(enclave_key, &account)?;

    let mut project = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::project_id.eq(account.project_id))
        .filter(maple_pairing_authority_project_heads::org_id.eq(account.org_id))
        .for_update()
        .first::<MaplePairingAuthorityProjectHead>(conn)?;
    validate_maple_pairing_authority_project_head(enclave_key, &project)?;
    let (account_count, account_inventory_digest) =
        compute_maple_pairing_authority_project_inventory(
            conn,
            enclave_key,
            project.project_id,
            project.org_id,
            project.project_uuid,
            project.subject_project_id,
            false,
        )?;
    project.account_inventory_digest = account_inventory_digest;
    project.account_count = account_count;
    project.revision = project
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    project.record_mac = maple_pairing_authority_project_head_mac(enclave_key, &project)?;
    let changed = diesel::update(
        maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::project_id.eq(project.project_id)),
    )
    .set((
        maple_pairing_authority_project_heads::account_inventory_digest
            .eq(&project.account_inventory_digest),
        maple_pairing_authority_project_heads::account_count.eq(project.account_count),
        maple_pairing_authority_project_heads::revision.eq(project.revision),
        maple_pairing_authority_project_heads::record_mac.eq(&project.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    let mut org = maple_pairing_authority_org_heads::table
        .filter(maple_pairing_authority_org_heads::org_id.eq(account.org_id))
        .for_update()
        .first::<MaplePairingAuthorityOrgHead>(conn)?;
    validate_maple_pairing_authority_org_head(enclave_key, &org)?;
    let (project_count, project_inventory_digest) =
        compute_maple_pairing_authority_org_inventory_shallow(conn, enclave_key, org.org_id)?;
    org.project_inventory_digest = project_inventory_digest;
    org.project_count = project_count;
    org.revision = org
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    org.record_mac = maple_pairing_authority_org_head_mac(enclave_key, &org)?;
    let changed = diesel::update(
        maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org.org_id)),
    )
    .set((
        maple_pairing_authority_org_heads::project_inventory_digest
            .eq(&org.project_inventory_digest),
        maple_pairing_authority_org_heads::project_count.eq(org.project_count),
        maple_pairing_authority_org_heads::revision.eq(org.revision),
        maple_pairing_authority_org_heads::record_mac.eq(&org.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }

    let mut global = load_maple_pairing_authority_global_head(conn)?;
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    let (org_count, org_inventory_digest) =
        compute_maple_pairing_authority_global_inventory_shallow(conn, enclave_key)?;
    global.org_inventory_digest = org_inventory_digest;
    global.org_count = org_count;
    global.revision = global
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    global.record_mac = Some(maple_pairing_authority_global_head_mac(
        enclave_key,
        &global,
    )?);
    let changed = diesel::update(
        maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true)),
    )
    .set((
        maple_pairing_authority_global_heads::org_inventory_digest.eq(&global.org_inventory_digest),
        maple_pairing_authority_global_heads::org_count.eq(global.org_count),
        maple_pairing_authority_global_heads::revision.eq(global.revision),
        maple_pairing_authority_global_heads::record_mac.eq(global.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn refresh_maple_pairing_authority_global_head(
    conn: &mut PgConnection,
    enclave_key: &[u8],
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_global_heads;

    let mut global = load_maple_pairing_authority_global_head(conn)?;
    validate_maple_pairing_authority_global_head(enclave_key, &global)?;
    let (org_count, org_inventory_digest) =
        compute_maple_pairing_authority_global_inventory_shallow(conn, enclave_key)?;
    global.org_count = org_count;
    global.org_inventory_digest = org_inventory_digest;
    global.revision = global
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    global.record_mac = Some(maple_pairing_authority_global_head_mac(
        enclave_key,
        &global,
    )?);
    let changed = diesel::update(
        maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true)),
    )
    .set((
        maple_pairing_authority_global_heads::org_inventory_digest.eq(global.org_inventory_digest),
        maple_pairing_authority_global_heads::org_count.eq(global.org_count),
        maple_pairing_authority_global_heads::revision.eq(global.revision),
        maple_pairing_authority_global_heads::record_mac.eq(global.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn refresh_maple_pairing_authority_org_and_global_heads(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    org_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_org_heads;

    let mut org = maple_pairing_authority_org_heads::table
        .filter(maple_pairing_authority_org_heads::org_id.eq(org_id))
        .for_update()
        .first::<MaplePairingAuthorityOrgHead>(conn)?;
    validate_maple_pairing_authority_org_head(enclave_key, &org)?;
    let (project_count, project_inventory_digest) =
        compute_maple_pairing_authority_org_inventory_shallow(conn, enclave_key, org_id)?;
    org.project_count = project_count;
    org.project_inventory_digest = project_inventory_digest;
    org.revision = org
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    org.record_mac = maple_pairing_authority_org_head_mac(enclave_key, &org)?;
    let changed = diesel::update(
        maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org_id)),
    )
    .set((
        maple_pairing_authority_org_heads::project_inventory_digest
            .eq(org.project_inventory_digest),
        maple_pairing_authority_org_heads::project_count.eq(org.project_count),
        maple_pairing_authority_org_heads::revision.eq(org.revision),
        maple_pairing_authority_org_heads::record_mac.eq(org.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    refresh_maple_pairing_authority_global_head(conn, enclave_key)
}

fn refresh_maple_pairing_authority_project_and_ancestors(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
    org_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_project_heads;

    let mut project = maple_pairing_authority_project_heads::table
        .filter(maple_pairing_authority_project_heads::project_id.eq(project_id))
        .filter(maple_pairing_authority_project_heads::org_id.eq(org_id))
        .for_update()
        .first::<MaplePairingAuthorityProjectHead>(conn)?;
    validate_maple_pairing_authority_project_head(enclave_key, &project)?;
    let (account_count, account_inventory_digest) =
        compute_maple_pairing_authority_project_inventory(
            conn,
            enclave_key,
            project_id,
            org_id,
            project.project_uuid,
            project.subject_project_id,
            false,
        )?;
    project.account_count = account_count;
    project.account_inventory_digest = account_inventory_digest;
    project.revision = project
        .revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    project.record_mac = maple_pairing_authority_project_head_mac(enclave_key, &project)?;
    let changed = diesel::update(
        maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::project_id.eq(project_id)),
    )
    .set((
        maple_pairing_authority_project_heads::account_inventory_digest
            .eq(project.account_inventory_digest),
        maple_pairing_authority_project_heads::account_count.eq(project.account_count),
        maple_pairing_authority_project_heads::revision.eq(project.revision),
        maple_pairing_authority_project_heads::record_mac.eq(project.record_mac),
    ))
    .execute(conn)?;
    if changed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    refresh_maple_pairing_authority_org_and_global_heads(conn, enclave_key, org_id)
}

fn commit_maple_pairing_authority_account_mutation(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<(), DBError> {
    commit_maple_pairing_authority_account_mutation_with_security_epoch(
        conn,
        enclave_key,
        user_id,
        project_id,
        None,
    )
}

fn commit_maple_pairing_authority_account_mutation_with_security_epoch(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    target_security_epoch: Option<i64>,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_account_heads;

    let mut head = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .for_update()
        .first::<MaplePairingAuthorityAccountHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_authority_account_head(enclave_key, &head)?;
    let prior_revision = head.revision;
    if let Some(target_security_epoch) = target_security_epoch {
        if target_security_epoch
            != head
                .security_epoch
                .checked_add(1)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        head.security_epoch = target_security_epoch;
        // The locked head was authenticated above. Project only the authorized
        // exact-next epoch into a provisional DB-owned MAC so the exhaustive
        // inventory pass can validate and hash that future head shape before
        // the final counts, revision, MAC, CAS, and ancestor cascade persist.
        head.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &head)?;
    }
    let (counts, inventory) =
        compute_maple_pairing_authority_account_inventory(conn, enclave_key, &head)?;
    head.authority_inventory_digest = inventory;
    head.authority_row_count = counts
        .total_rows()
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    head.device_count = counts.devices;
    head.device_operation_count = counts.device_operations;
    head.lineage_count = counts.lineages;
    head.pairing_count = counts.pairings;
    head.pairing_operation_count = counts.pairing_operations;
    head.host_state_count = counts.host_states;
    head.revocation_event_count = counts.revocation_events;
    head.highwater_installation_group_count = counts.highwater_groups;
    head.highwater_generation_count = counts.highwater_generations;
    head.registration_operation_tombstone_count = counts.registration_operation_tombstones;
    head.installation_retirement_count = counts.installation_retirements;
    head.reset_clear_obligation_count = counts.reset_clear_obligations;
    head.reset_clear_admission_count = counts.reset_clear_admissions;
    head.revision = prior_revision
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
    head.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &head)?;
    let updated = diesel::update(
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
            .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
            .filter(maple_pairing_authority_account_heads::revision.eq(prior_revision)),
    )
    .set((
        maple_pairing_authority_account_heads::security_epoch.eq(head.security_epoch),
        maple_pairing_authority_account_heads::authority_inventory_digest
            .eq(&head.authority_inventory_digest),
        maple_pairing_authority_account_heads::authority_row_count.eq(head.authority_row_count),
        maple_pairing_authority_account_heads::device_count.eq(head.device_count),
        maple_pairing_authority_account_heads::device_operation_count
            .eq(head.device_operation_count),
        maple_pairing_authority_account_heads::lineage_count.eq(head.lineage_count),
        maple_pairing_authority_account_heads::pairing_count.eq(head.pairing_count),
        maple_pairing_authority_account_heads::pairing_operation_count
            .eq(head.pairing_operation_count),
        maple_pairing_authority_account_heads::host_state_count.eq(head.host_state_count),
        maple_pairing_authority_account_heads::revocation_event_count
            .eq(head.revocation_event_count),
        maple_pairing_authority_account_heads::highwater_installation_group_count
            .eq(head.highwater_installation_group_count),
        maple_pairing_authority_account_heads::highwater_generation_count
            .eq(head.highwater_generation_count),
        maple_pairing_authority_account_heads::registration_operation_tombstone_count
            .eq(head.registration_operation_tombstone_count),
        maple_pairing_authority_account_heads::installation_retirement_count
            .eq(head.installation_retirement_count),
        maple_pairing_authority_account_heads::reset_clear_obligation_count
            .eq(head.reset_clear_obligation_count),
        maple_pairing_authority_account_heads::reset_clear_admission_count
            .eq(head.reset_clear_admission_count),
        maple_pairing_authority_account_heads::revision.eq(head.revision),
        maple_pairing_authority_account_heads::record_mac.eq(&head.record_mac),
    ))
    .execute(conn)?;
    if updated != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    cascade_maple_pairing_authority_heads(conn, enclave_key, user_id)?;
    let current = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .first::<MaplePairingAuthorityAccountHead>(conn)?;
    verify_maple_pairing_authority_scoped_chain(conn, enclave_key, &current).map(|_| ())
}

fn enter_maple_pairing_authority_account_transaction(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    expected_issuer_key_inventory_digest: &[u8],
    user_id: Uuid,
    project_id: i32,
    operation: &'static str,
) -> Result<MaplePairingAuthenticatedProjectIdentity, DBError> {
    use crate::models::schema::maple_pairing_authority_account_heads;

    let timer = MaplePairingAuthorityTransactionTimer::start(operation);
    acquire_maple_pairing_authority_snapshot_fence(
        conn,
        enclave_key,
        expected_issuer_key_inventory_digest,
    )?;
    #[cfg(test)]
    observe_maple_pairing_authority_scoped_access_if_armed_for_test(user_id);
    let head = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .first::<MaplePairingAuthorityAccountHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    let project = verify_maple_pairing_authority_scoped_chain(conn, enclave_key, &head)?;
    Ok(MaplePairingAuthenticatedProjectIdentity::from_verified_head(&project, timer))
}

pub(crate) fn create_user_with_maple_authority_in_tx(
    tx: &mut PgConnection,
    new_user: &NewUser,
    enclave_key: &[u8],
    expected_issuer_key_inventory_digest: &[u8],
) -> Result<User, DBError> {
    use crate::models::schema::{org_projects, users};

    let _timer = MaplePairingAuthorityTransactionTimer::start("create_user");
    acquire_maple_pairing_authority_snapshot_fence(
        tx,
        enclave_key,
        expected_issuer_key_inventory_digest,
    )?;
    let project = org_projects::table
        .filter(org_projects::id.eq(new_user.project_id))
        .for_update()
        .first::<OrgProject>(tx)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::OrgProjectNotFound,
            other => DBError::QueryError(other),
        })?;
    verify_maple_pairing_authority_project_chain(
        tx,
        enclave_key,
        project.id,
        project.org_id,
        false,
    )?;
    let user = new_user.insert(tx)?;
    let created_at = maple_pairing_trusted_db_now(tx)?;
    create_empty_maple_pairing_authority_account_head(
        tx,
        enclave_key,
        user.uuid,
        user.project_id,
        project.org_id,
        created_at,
    )?;
    cascade_maple_pairing_authority_heads(tx, enclave_key, user.uuid)?;
    let returned = users::table
        .filter(users::uuid.eq(user.uuid))
        .first::<User>(tx)?;
    let account = crate::models::schema::maple_pairing_authority_account_heads::table
        .filter(crate::models::schema::maple_pairing_authority_account_heads::user_id.eq(user.uuid))
        .filter(
            crate::models::schema::maple_pairing_authority_account_heads::project_id
                .eq(user.project_id),
        )
        .first::<MaplePairingAuthorityAccountHead>(tx)?;
    verify_maple_pairing_authority_scoped_chain(tx, enclave_key, &account)?;
    Ok(returned)
}

#[cfg(test)]
pub(crate) fn maple_pairing_revocation_highwater_lookup_digest_for_test(
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    host_installation_id: Uuid,
) -> Result<Vec<u8>, EncryptError> {
    maple_pairing_revocation_highwater_lookup_digest(
        enclave_key,
        user_id,
        project_id,
        host_installation_id,
    )
}

/// Commit a structurally permitted but MAC-invalid global-root revision to
/// model a privileged storage-layer edit. Normal authority APIs cannot create
/// this state because the Active snapshot fence authenticates the root first.
#[cfg(test)]
pub(crate) fn tamper_maple_pairing_authority_global_root_for_test(
    db: &(dyn DBConnection + Send + Sync),
    enclave_key: &[u8],
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_global_heads;

    let conn = &mut db.get_pool().get().map_err(|_| DBError::ConnectionError)?;
    conn.transaction::<(), DBError, _>(|tx| {
        let current = load_maple_pairing_authority_global_head(tx)?;
        validate_maple_pairing_authority_global_head(enclave_key, &current)?;
        let revision = current
            .revision
            .checked_add(1)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        let changed = diesel::update(
            maple_pairing_authority_global_heads::table
                .filter(maple_pairing_authority_global_heads::singleton.eq(true))
                .filter(maple_pairing_authority_global_heads::revision.eq(current.revision)),
        )
        .set((
            maple_pairing_authority_global_heads::revision.eq(revision),
            maple_pairing_authority_global_heads::record_mac.eq(Some(vec![0_u8; 32])),
        ))
        .execute(tx)?;
        if changed != 1 {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        Ok(())
    })
}

/// Repair only the deliberately corrupted global-root test fixture. This is
/// intentionally not an application recovery API: it bypasses the authority
/// fence and must remain cfg(test).
#[cfg(test)]
pub(crate) fn restore_maple_pairing_authority_global_root_for_test(
    db: &(dyn DBConnection + Send + Sync),
    enclave_key: &[u8],
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_global_heads;

    let conn = &mut db.get_pool().get().map_err(|_| DBError::ConnectionError)?;
    conn.transaction::<(), DBError, _>(|tx| {
        let mut current = load_maple_pairing_authority_global_head(tx)?;
        if !current.singleton
            || current.activation_state != MAPLE_PAIRING_AUTHORITY_ACTIVE
            || current.org_inventory_digest.len() != 32
            || current.org_count < 0
            || current.issuer_key_inventory_digest.len() != 32
            || !(0..=i64::try_from(MAPLE_PAIRING_MAX_ISSUER_KEYS)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?)
                .contains(&current.issuer_key_count)
            || current.revision < 2
            || current.updated_at < current.created_at
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let prior_revision = current.revision;
        current.revision = prior_revision
            .checked_add(1)
            .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        current.record_mac = Some(maple_pairing_authority_global_head_mac(
            enclave_key,
            &current,
        )?);
        let changed = diesel::update(
            maple_pairing_authority_global_heads::table
                .filter(maple_pairing_authority_global_heads::singleton.eq(true))
                .filter(maple_pairing_authority_global_heads::revision.eq(prior_revision)),
        )
        .set((
            maple_pairing_authority_global_heads::revision.eq(current.revision),
            maple_pairing_authority_global_heads::record_mac.eq(current.record_mac),
        ))
        .execute(tx)?;
        if changed != 1 {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        Ok(())
    })
}

/// Test-only clock fixture for destructive-deletion rollback regressions.
///
/// Production code never rewrites accepted request times. The deletion tests
/// need a pending record that is already beyond the 30-second skew window,
/// though, and sleeping would make the disposable security suite both slow and
/// flaky. This helper authenticates the original pair and CREATE receipt,
/// rewrites their bound timestamps together, and advances the cascading
/// authority heads in one ordinary serializable authority transaction.
#[cfg(test)]
pub(crate) fn make_maple_pairing_pending_due_for_test(
    db: &(dyn DBConnection + Send + Sync),
    authorization: &MaplePairingAuthorization,
    pair_id: Uuid,
) -> Result<(), DBError> {
    use crate::models::maple_pairing_db::MaplePairingOperation;
    use crate::models::schema::{maple_pairing_operations, maple_pairings};

    let expected_issuer_key_inventory_digest =
        db.configured_maple_pairing_issuer_key_inventory_digest()?;
    let conn = &mut db.get_pool().get().map_err(|_| DBError::ConnectionError)?;
    run_maple_pairing_authority_transaction(
        conn,
        MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
        |tx| {
            let _authority_timer = enter_maple_pairing_authority_account_transaction(
                tx,
                &authorization.enclave_key,
                &expected_issuer_key_inventory_digest,
                authorization.user_id,
                authorization.project_id,
                "make_maple_pairing_pending_due_for_test",
            )?;
            let mut pairing = maple_pairings::table
                .filter(maple_pairings::user_id.eq(authorization.user_id))
                .filter(maple_pairings::project_id.eq(authorization.project_id))
                .filter(maple_pairings::uuid.eq(pair_id))
                .filter(maple_pairings::state.eq(MaplePairingState::Pending.as_db()))
                .filter(maple_pairings::revision.eq(1))
                .for_update()
                .first::<MaplePairing>(tx)
                .map_err(|error| match error {
                    diesel::result::Error::NotFound => DBError::MaplePairingConflict,
                    other => DBError::QueryError(other),
                })?;
            validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;

            let mut operation = maple_pairing_operations::table
                .filter(maple_pairing_operations::maple_pairing_id.eq(pairing.id))
                .filter(maple_pairing_operations::operation_kind.eq(MAPLE_PAIRING_OPERATION_CREATE))
                .for_update()
                .first::<MaplePairingOperation>(tx)
                .map_err(|error| match error {
                    diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
                    other => DBError::QueryError(other),
                })?;
            pairing_operation_receipt(&authorization.enclave_key, &operation, pairing.uuid)?;

            let trusted_now = maple_pairing_trusted_db_now(tx)?;
            let expires_at = trusted_now
                .checked_sub_signed(chrono::Duration::milliseconds(
                    MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
                ))
                .ok_or(DBError::MaplePairingConflict)?;
            let created_at = expires_at
                .checked_sub_signed(chrono::Duration::seconds(1))
                .ok_or(DBError::MaplePairingConflict)?;
            pairing.created_at = created_at;
            pairing.expires_at = expires_at;
            pairing.record_mac =
                maple_pairing_record_mac_for_row(&authorization.enclave_key, &pairing)?;
            operation.accepted_at = created_at;
            operation.receipt_mac = maple_pairing_receipt_mac(
                &authorization.enclave_key,
                operation.operation_id,
                operation.user_id,
                operation.project_id,
                operation.actor_maple_device_id,
                operation.operation_kind,
                &operation.request_mac,
                operation.maple_pairing_id,
                operation.pairing_revision,
                operation.receipt_version,
                &operation.receipt_enc,
                operation.receipt_issuer_key_id.as_deref(),
                operation.accepted_at,
            )?;

            let pair_changed = diesel::update(
                maple_pairings::table
                    .filter(maple_pairings::id.eq(pairing.id))
                    .filter(maple_pairings::state.eq(MaplePairingState::Pending.as_db()))
                    .filter(maple_pairings::revision.eq(1)),
            )
            .set((
                maple_pairings::created_at.eq(pairing.created_at),
                maple_pairings::expires_at.eq(pairing.expires_at),
                maple_pairings::record_mac.eq(&pairing.record_mac),
            ))
            .execute(tx)?;
            let operation_changed = diesel::update(
                maple_pairing_operations::table
                    .filter(maple_pairing_operations::id.eq(operation.id)),
            )
            .set((
                maple_pairing_operations::accepted_at.eq(operation.accepted_at),
                maple_pairing_operations::receipt_mac.eq(&operation.receipt_mac),
            ))
            .execute(tx)?;
            if pair_changed != 1 || operation_changed != 1 {
                return Err(DBError::MaplePairingConflict);
            }
            commit_maple_pairing_authority_account_mutation(
                tx,
                &authorization.enclave_key,
                authorization.user_id,
                authorization.project_id,
            )
        },
    )
}

/// Deterministic test seam for a hostile, non-cooperating DML writer between
/// an authenticated account read and its cascading-head commit. The injected
/// callback must stay synchronous and is expected to use a second connection;
/// production paths never expose such a callback.
#[cfg(test)]
pub(crate) fn run_maple_pairing_authority_ssi_race_for_test<F>(
    db: &(dyn DBConnection + Send + Sync),
    authorization: &MaplePairingAuthorization,
    after_verified_snapshot: F,
) -> Result<(), DBError>
where
    F: FnOnce() -> Result<(), DBError>,
{
    let expected_issuer_key_inventory_digest =
        db.configured_maple_pairing_issuer_key_inventory_digest()?;
    let conn = &mut db.get_pool().get().map_err(|_| DBError::ConnectionError)?;
    run_maple_pairing_authority_transaction(
        conn,
        MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
        |tx| {
            let _authority_timer = enter_maple_pairing_authority_account_transaction(
                tx,
                &authorization.enclave_key,
                &expected_issuer_key_inventory_digest,
                authorization.user_id,
                authorization.project_id,
                "run_maple_pairing_authority_ssi_race_for_test",
            )?;
            after_verified_snapshot()?;
            commit_maple_pairing_authority_account_mutation(
                tx,
                &authorization.enclave_key,
                authorization.user_id,
                authorization.project_id,
            )
        },
    )
}

/// Populate one otherwise-empty test account at the V1 retained-highwater
/// group limit without issuing thousands of public registration requests.
/// Every synthetic first generation is authenticated and the account/ancestor
/// heads are advanced normally, so a following public operation exercises the
/// real full-inventory audit and prospective group-capacity rejection.
#[cfg(test)]
pub(crate) fn seed_maple_pairing_highwater_group_capacity_for_test(
    db: &(dyn DBConnection + Send + Sync),
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<Uuid, DBError> {
    use crate::models::maple_pairing_db::NewMaplePairingRevocationHighwater;
    use crate::models::schema::maple_pairing_revocation_highwaters;

    let expected_issuer_key_inventory_digest =
        db.configured_maple_pairing_issuer_key_inventory_digest()?;
    let conn = &mut db.get_pool().get().map_err(|_| DBError::ConnectionError)?;
    run_maple_pairing_authority_transaction(
        conn,
        MaplePairingAuthorityTransactionClass::NonReplayableMutation,
        |tx| {
            let _authority_timer = enter_maple_pairing_authority_account_transaction(
                tx,
                enclave_key,
                &expected_issuer_key_inventory_digest,
                user_id,
                project_id,
                "seed_maple_pairing_highwater_group_capacity_for_test",
            )?;
            let authority_scope_digest =
                maple_pairing_authority_scope_digest(enclave_key, user_id, project_id)?;
            let counts = count_maple_pairing_authority_account_rows(
                tx,
                &authority_scope_digest,
                user_id,
                project_id,
            )?;
            if counts != MaplePairingAuthorityCounts::default() {
                return Err(DBError::MaplePairingConflict);
            }

            let mut retained_installation_id = None;
            let mut batch = Vec::with_capacity(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE as usize);
            for _ in 0..MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT {
                let installation_id = Uuid::new_v4();
                retained_installation_id.get_or_insert(installation_id);
                let lookup_digest = maple_pairing_revocation_highwater_lookup_digest(
                    enclave_key,
                    user_id,
                    project_id,
                    installation_id,
                )?;
                let stream_id = Uuid::new_v4();
                let record_mac = maple_pairing_revocation_highwater_record_mac(
                    enclave_key,
                    &lookup_digest,
                    &authority_scope_digest,
                    stream_id,
                    1,
                    1,
                    0,
                    1,
                )?;
                batch.push(NewMaplePairingRevocationHighwater {
                    lookup_digest,
                    authority_scope_digest: authority_scope_digest.clone(),
                    revocation_stream_id: stream_id,
                    revocation_stream_generation: 1,
                    security_epoch: 1,
                    last_issued_revocation_sequence: 0,
                    revision: 1,
                    record_mac,
                });
                if batch.len() == MAPLE_PAIRING_AUTHORITY_PAGE_SIZE as usize {
                    diesel::insert_into(maple_pairing_revocation_highwaters::table)
                        .values(&batch)
                        .execute(tx)?;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                diesel::insert_into(maple_pairing_revocation_highwaters::table)
                    .values(&batch)
                    .execute(tx)?;
            }
            commit_maple_pairing_authority_account_mutation(tx, enclave_key, user_id, project_id)?;
            retained_installation_id.ok_or(DBError::MaplePairingAuthorityCorrupt)
        },
    )
}

// Explicit fields preserve the ordered canonical authenticated highwater transcript.
#[allow(clippy::too_many_arguments)]
fn maple_pairing_revocation_highwater_record_mac(
    enclave_key: &[u8],
    lookup_digest: &[u8],
    authority_scope_digest: &[u8],
    revocation_stream_id: Uuid,
    revocation_stream_generation: i64,
    security_epoch: i64,
    last_issued_revocation_sequence: i64,
    revision: i64,
) -> Result<Vec<u8>, EncryptError> {
    let mut body = CanonicalBytes::new(MAPLE_PAIRING_REVOCATION_HIGHWATER_MAC_DOMAIN);
    body.append_bytes(lookup_digest)
        .append_bytes(authority_scope_digest)
        .append_uuid(revocation_stream_id)
        .append_i64(revocation_stream_generation)
        .append_i64(security_epoch)
        .append_i64(last_issued_revocation_sequence)
        .append_i64(revision);
    maple_pairing_hmac(
        enclave_key,
        MAPLE_PAIRING_REVOCATION_HIGHWATER_MAC_KEY_INFO,
        &body.into_bytes(),
    )
}

fn validate_maple_pairing_revocation_highwater(
    enclave_key: &[u8],
    row: &crate::models::maple_pairing_db::MaplePairingRevocationHighwater,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_revocation_highwater_record_mac(
        enclave_key,
        &row.lookup_digest,
        &row.authority_scope_digest,
        row.revocation_stream_id,
        row.revocation_stream_generation,
        row.security_epoch,
        row.last_issued_revocation_sequence,
        row.revision,
    )?;
    if row.id <= 0
        || row.lookup_digest.len() != 32
        || row.authority_scope_digest.len() != 32
        || row.revocation_stream_id.is_nil()
        || row.revocation_stream_generation <= 0
        || row.security_epoch <= 0
        || row.last_issued_revocation_sequence < 0
        || row.revision <= 0
        || row.updated_at < row.created_at
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingCorrupt);
    }
    Ok(())
}

fn load_maple_pairing_revocation_highwater(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    host_installation_id: Uuid,
    exclusive: bool,
) -> Result<
    (
        Vec<u8>,
        Option<crate::models::maple_pairing_db::MaplePairingRevocationHighwater>,
    ),
    DBError,
> {
    use crate::models::maple_pairing_db::MaplePairingRevocationHighwater;
    use crate::models::schema::maple_pairing_revocation_highwaters;

    let lookup_digest = maple_pairing_revocation_highwater_lookup_digest(
        enclave_key,
        user_id,
        project_id,
        host_installation_id,
    )?;
    let authority_scope_digest =
        maple_pairing_authority_scope_digest(enclave_key, user_id, project_id)?;
    let mut expected_generation = 1_i64;
    let mut seen_stream_ids = BTreeSet::new();
    let mut latest = None;
    let mut rows_seen = 0_i64;
    let mut cursor_generation = 0_i64;
    loop {
        let rows = if exclusive {
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(&lookup_digest))
                .filter(
                    maple_pairing_revocation_highwaters::authority_scope_digest
                        .eq(&authority_scope_digest),
                )
                .filter(
                    maple_pairing_revocation_highwaters::revocation_stream_generation
                        .gt(cursor_generation),
                )
                .order(maple_pairing_revocation_highwaters::revocation_stream_generation.asc())
                .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                .for_update()
                .load::<MaplePairingRevocationHighwater>(conn)?
        } else {
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(&lookup_digest))
                .filter(
                    maple_pairing_revocation_highwaters::authority_scope_digest
                        .eq(&authority_scope_digest),
                )
                .filter(
                    maple_pairing_revocation_highwaters::revocation_stream_generation
                        .gt(cursor_generation),
                )
                .order(maple_pairing_revocation_highwaters::revocation_stream_generation.asc())
                .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                .for_share()
                .load::<MaplePairingRevocationHighwater>(conn)?
        };
        if rows.is_empty() {
            break;
        }
        for row in rows {
            validate_maple_pairing_revocation_highwater(enclave_key, &row)?;
            if row.revocation_stream_generation != expected_generation
                || row.revision
                    != row
                        .last_issued_revocation_sequence
                        .checked_add(1)
                        .ok_or(DBError::MaplePairingCorrupt)?
                || !seen_stream_ids.insert(row.revocation_stream_id)
                || !bool::from(subtle::ConstantTimeEq::ct_eq(
                    row.lookup_digest.as_slice(),
                    lookup_digest.as_slice(),
                ))
                || !bool::from(subtle::ConstantTimeEq::ct_eq(
                    row.authority_scope_digest.as_slice(),
                    authority_scope_digest.as_slice(),
                ))
            {
                return Err(DBError::MaplePairingCorrupt);
            }
            rows_seen = rows_seen
                .checked_add(1)
                .ok_or(DBError::MaplePairingCorrupt)?;
            if rows_seen > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT {
                return Err(DBError::MaplePairingAuthorityCapacityExceeded);
            }
            cursor_generation = row.revocation_stream_generation;
            expected_generation = expected_generation
                .checked_add(1)
                .ok_or(DBError::MaplePairingCorrupt)?;
            latest = Some(row);
        }
    }
    Ok((lookup_digest, latest))
}

fn insert_initial_maple_pairing_revocation_highwater(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    lookup_digest: Vec<u8>,
    authority_scope_digest: Vec<u8>,
    security_epoch: i64,
) -> Result<crate::models::maple_pairing_db::MaplePairingRevocationHighwater, DBError> {
    use crate::models::maple_pairing_db::{
        MaplePairingRevocationHighwater, NewMaplePairingRevocationHighwater,
    };
    use crate::models::schema::maple_pairing_revocation_highwaters;

    let revocation_stream_id = Uuid::new_v4();
    let revocation_stream_generation = 1;
    let record_mac = maple_pairing_revocation_highwater_record_mac(
        enclave_key,
        &lookup_digest,
        &authority_scope_digest,
        revocation_stream_id,
        revocation_stream_generation,
        security_epoch,
        0,
        1,
    )?;
    let row = diesel::insert_into(maple_pairing_revocation_highwaters::table)
        .values(NewMaplePairingRevocationHighwater {
            lookup_digest,
            authority_scope_digest,
            revocation_stream_id,
            revocation_stream_generation,
            security_epoch,
            last_issued_revocation_sequence: 0,
            revision: 1,
            record_mac,
        })
        .get_result::<MaplePairingRevocationHighwater>(conn)
        .map_err(|error| match error {
            diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::UniqueViolation,
                _,
            ) => DBError::MaplePairingConflict,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_revocation_highwater(enclave_key, &row)?;
    Ok(row)
}

fn load_maple_pairing_host_state(
    conn: &mut PgConnection,
    user_id: Uuid,
    project_id: i32,
    host_maple_device_id: i64,
    exclusive: bool,
) -> Result<Option<crate::models::maple_pairing_db::MaplePairingHostState>, DBError> {
    use crate::models::maple_pairing_db::MaplePairingHostState;
    use crate::models::schema::maple_pairing_host_states;
    use diesel::OptionalExtension;

    let query = maple_pairing_host_states::table
        .filter(maple_pairing_host_states::user_id.eq(user_id))
        .filter(maple_pairing_host_states::project_id.eq(project_id))
        .filter(maple_pairing_host_states::host_maple_device_id.eq(host_maple_device_id));
    if exclusive {
        query
            .for_update()
            .first::<MaplePairingHostState>(conn)
            .optional()
            .map_err(DBError::from)
    } else {
        query
            .for_share()
            .first::<MaplePairingHostState>(conn)
            .optional()
            .map_err(DBError::from)
    }
}

fn load_latest_pending_maple_reset_clear_obligation(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    highwater: &crate::models::maple_pairing_db::MaplePairingRevocationHighwater,
    exclusive: bool,
) -> Result<Option<MaplePairingResetClearObligation>, DBError> {
    use crate::models::schema::maple_pairing_reset_clear_obligations;
    use diesel::OptionalExtension;

    let query = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&highwater.authority_scope_digest),
        )
        .filter(maple_pairing_reset_clear_obligations::lookup_digest.eq(&highwater.lookup_digest))
        .filter(maple_pairing_reset_clear_obligations::state.eq(1_i16))
        .order((
            maple_pairing_reset_clear_obligations::reset_generation.desc(),
            maple_pairing_reset_clear_obligations::id.desc(),
        ));
    let row = if exclusive {
        query
            .for_update()
            .first::<MaplePairingResetClearObligation>(conn)
            .optional()?
    } else {
        query
            .for_share()
            .first::<MaplePairingResetClearObligation>(conn)
            .optional()?
    };
    if let Some(row) = &row {
        validate_maple_pairing_reset_clear_obligation(
            enclave_key,
            row,
            &highwater.authority_scope_digest,
        )?;
        if !maple_pairing_authority_mac_matches(&row.lookup_digest, &highwater.lookup_digest)
            || row.target_revocation_stream_id != highwater.revocation_stream_id
            || row.target_revocation_stream_generation != highwater.revocation_stream_generation
            || row.target_security_epoch != highwater.security_epoch
            || row.target_instruction_sequence != 1
            || highwater.last_issued_revocation_sequence != 1
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }
    Ok(row)
}

/// Loads the authenticated head of an installation's reset-clear chain,
/// irrespective of whether that head has already been acknowledged. Pending
/// state controls the authority gate; generation and predecessor continuity do
/// not reset after an ACK.
fn load_latest_maple_reset_clear_obligation(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    highwater: &crate::models::maple_pairing_db::MaplePairingRevocationHighwater,
    exclusive: bool,
) -> Result<Option<MaplePairingResetClearObligation>, DBError> {
    use crate::models::schema::maple_pairing_reset_clear_obligations;
    use diesel::OptionalExtension;

    let query = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&highwater.authority_scope_digest),
        )
        .filter(maple_pairing_reset_clear_obligations::lookup_digest.eq(&highwater.lookup_digest))
        .order((
            maple_pairing_reset_clear_obligations::reset_generation.desc(),
            maple_pairing_reset_clear_obligations::id.desc(),
        ));
    let row = if exclusive {
        query
            .for_update()
            .first::<MaplePairingResetClearObligation>(conn)
            .optional()?
    } else {
        query
            .for_share()
            .first::<MaplePairingResetClearObligation>(conn)
            .optional()?
    };
    if let Some(row) = &row {
        validate_maple_pairing_reset_clear_obligation(
            enclave_key,
            row,
            &highwater.authority_scope_digest,
        )?;
        if !maple_pairing_authority_mac_matches(&row.lookup_digest, &highwater.lookup_digest)
            || row.target_revocation_stream_id != highwater.revocation_stream_id
            || row.target_revocation_stream_generation != highwater.revocation_stream_generation
            || row.target_security_epoch != highwater.security_epoch
            || row.target_instruction_sequence != 1
            || highwater.last_issued_revocation_sequence < 1
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }
    Ok(row)
}

// Explicit inputs bind the host namespace, identity, and highwater state.
#[allow(clippy::too_many_arguments)]
fn seed_or_validate_maple_pairing_host_state(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    host_maple_device_id: i64,
    host_identity_mac: &[u8],
    highwater: &crate::models::maple_pairing_db::MaplePairingRevocationHighwater,
    existing: Option<crate::models::maple_pairing_db::MaplePairingHostState>,
) -> Result<crate::models::maple_pairing_db::MaplePairingHostState, DBError> {
    use crate::models::maple_pairing_db::{MaplePairingHostState, NewMaplePairingHostState};
    use crate::models::schema::maple_pairing_host_states;

    validate_maple_pairing_revocation_highwater(enclave_key, highwater)?;
    let pending_reset =
        load_latest_pending_maple_reset_clear_obligation(conn, enclave_key, highwater, true)?;
    if pending_reset.as_ref().is_some_and(|pending| {
        !maple_pairing_authority_mac_matches(&pending.host_identity_mac, host_identity_mac)
    }) {
        return Err(DBError::MaplePairingCorrupt);
    }
    if let Some(state) = existing {
        validate_maple_pairing_host_state(enclave_key, &state)?;
        if state.revocation_stream_id != highwater.revocation_stream_id
            || state.revocation_stream_generation != highwater.revocation_stream_generation
            || state.last_issued_revocation_sequence != highwater.last_issued_revocation_sequence
            || pending_reset.is_some()
                && (state.last_issued_revocation_sequence != 1
                    || state.last_acked_revocation_sequence != 0
                    || state.revision != 2)
        {
            return Err(DBError::MaplePairingCorrupt);
        }
        return Ok(state);
    }

    // Missing host state is reconstructible only from one of two proofs:
    // a fresh sequence-zero namespace, or the exact retained Pending reset
    // control instruction at sequence one. Never infer durable ACK state from
    // a high-water counter alone.
    let (last_issued, last_acked, revision) = if pending_reset.is_some() {
        (1_i64, 0_i64, 2_i64)
    } else if highwater.last_issued_revocation_sequence == 0 {
        (0_i64, 0_i64, 1_i64)
    } else {
        return Err(DBError::MaplePairingCorrupt);
    };
    let record_mac = maple_pairing_host_state_mac(
        enclave_key,
        user_id,
        project_id,
        host_maple_device_id,
        highwater.revocation_stream_id,
        highwater.revocation_stream_generation,
        last_issued,
        last_acked,
        revision,
    )?;
    let state = diesel::insert_into(maple_pairing_host_states::table)
        .values(NewMaplePairingHostState {
            user_id,
            project_id,
            host_maple_device_id,
            revocation_stream_id: highwater.revocation_stream_id,
            revocation_stream_generation: highwater.revocation_stream_generation,
            last_issued_revocation_sequence: last_issued,
            last_acked_revocation_sequence: last_acked,
            revision,
            record_mac,
        })
        .get_result::<MaplePairingHostState>(conn)
        .map_err(|error| match error {
            diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::UniqueViolation,
                _,
            ) => DBError::MaplePairingConflict,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_host_state(enclave_key, &state)?;
    Ok(state)
}

fn restore_maple_pairing_host_state_from_highwater(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    device: &MapleDevice,
) -> Result<(), DBError> {
    let (_, highwater) = load_maple_pairing_revocation_highwater(
        conn,
        enclave_key,
        device.user_id,
        device.project_id,
        device.installation_id,
        true,
    )?;
    let state =
        load_maple_pairing_host_state(conn, device.user_id, device.project_id, device.id, true)?;
    match (highwater, state) {
        (Some(highwater), state) => {
            seed_or_validate_maple_pairing_host_state(
                conn,
                enclave_key,
                device.user_id,
                device.project_id,
                device.id,
                &device.identity_mac,
                &highwater,
                state,
            )?;
            Ok(())
        }
        (None, Some(state)) => {
            validate_maple_pairing_host_state(enclave_key, &state)?;
            Err(DBError::MaplePairingCorrupt)
        }
        (None, None) => {
            use crate::models::schema::maple_pairing_authority_account_heads;
            let account_head = maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::user_id.eq(device.user_id))
                .filter(maple_pairing_authority_account_heads::project_id.eq(device.project_id))
                .for_update()
                .first::<MaplePairingAuthorityAccountHead>(conn)?;
            validate_maple_pairing_authority_account_head(enclave_key, &account_head)?;
            let (lookup_digest, _) = load_maple_pairing_revocation_highwater(
                conn,
                enclave_key,
                device.user_id,
                device.project_id,
                device.installation_id,
                true,
            )?;
            let highwater = insert_initial_maple_pairing_revocation_highwater(
                conn,
                enclave_key,
                lookup_digest,
                maple_pairing_authority_scope_digest(
                    enclave_key,
                    device.user_id,
                    device.project_id,
                )?,
                account_head.security_epoch,
            )?;
            seed_or_validate_maple_pairing_host_state(
                conn,
                enclave_key,
                device.user_id,
                device.project_id,
                device.id,
                &device.identity_mac,
                &highwater,
                None,
            )?;
            Ok(())
        }
    }
}

struct PreparedMapleDeviceRegistrationSync {
    response_kind: i16,
    payload_version: i16,
    payload: Vec<u8>,
    payload_enc: Vec<u8>,
    issuer_key_id: String,
    digest: Vec<u8>,
}

fn decode_maple_authority_digest(value: &str) -> Result<Vec<u8>, DBError> {
    let decoded = STANDARD
        .decode(value)
        .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
    if decoded.len() != 32 {
        return Err(DBError::MaplePairingMaterializationFailed);
    }
    Ok(decoded)
}

fn validate_reset_clear_instruction_against_obligation(
    enclave_key: &[u8],
    internal_project_id: i32,
    instruction: &crate::models::maple_pairings::MapleResetClearRequiredV1,
    obligation: &MaplePairingResetClearObligation,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let material_digest = decode_maple_authority_digest(&instruction.instruction_material_digest)?;
    let chain_digest = decode_maple_authority_digest(&instruction.chain_digest)?;
    let admission_digest = decode_maple_authority_digest(&instruction.admission_set_digest)?;
    let previous_material = instruction
        .previous_instruction_material_digest
        .as_deref()
        .map(decode_maple_authority_digest)
        .transpose()?;
    let previous_chain = instruction
        .previous_chain_digest
        .as_deref()
        .map(decode_maple_authority_digest)
        .transpose()?;
    let identity_public_key = instruction
        .host
        .verifying_key_bytes()
        .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
    let expected_identity_mac = maple_device_identity_mac_from_claim(
        enclave_key,
        instruction.subject_account_id,
        internal_project_id,
        &identity_public_key,
    )?;
    if instruction.event_id != obligation.uuid
        || instruction.reset_id != obligation.reset_id
        || instruction.reset_generation != pairing_u64_from_i64(obligation.reset_generation)?
        || instruction.cumulative_reset_count
            != pairing_u64_from_i64(obligation.cumulative_reset_count)?
        || instruction.source_security_epoch
            != pairing_u64_from_i64(obligation.source_security_epoch)?
        || instruction.security_epoch != pairing_u64_from_i64(obligation.target_security_epoch)?
        || instruction.issuer_sequence
            != pairing_u64_from_i64(obligation.target_instruction_sequence)?
        || instruction.source_revocation_stream_id != obligation.old_revocation_stream_id
        || instruction.source_revocation_stream_generation
            != pairing_u64_from_i64(obligation.old_revocation_stream_generation)?
        || instruction.revocation_stream_id != obligation.target_revocation_stream_id
        || instruction.revocation_stream_generation
            != pairing_u64_from_i64(obligation.target_revocation_stream_generation)?
        || instruction.admission_count
            != u16::try_from(obligation.admission_count)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?
        || instruction.previous_reset_clear_event_id != obligation.previous_event_id
        || instruction.reset_at_unix_ms != obligation.reset_at.timestamp_millis()
        || !bool::from(
            material_digest
                .as_slice()
                .ct_eq(obligation.instruction_digest.as_slice()),
        )
        || !bool::from(
            chain_digest
                .as_slice()
                .ct_eq(obligation.chain_digest.as_slice()),
        )
        || !bool::from(
            admission_digest
                .as_slice()
                .ct_eq(obligation.admission_set_digest.as_slice()),
        )
        || previous_material.as_deref() != obligation.previous_instruction_digest.as_deref()
        || previous_chain.as_deref() != obligation.previous_chain_digest.as_deref()
        || !bool::from(
            expected_identity_mac
                .as_slice()
                .ct_eq(obligation.host_identity_mac.as_slice()),
        )
    {
        return Err(DBError::MaplePairingMaterializationFailed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_maple_device_registration_sync(
    conn: &mut PgConnection,
    registration: &NewMapleDeviceRegistration,
    device: &MapleDevice,
    highwater: &MaplePairingRevocationHighwater,
    state: &MaplePairingHostState,
    pending: Option<MaplePairingResetClearObligation>,
    issuer_keyset: &MaplePairingIssuerKeySetV1,
    materialize: &MaterializeMapleDeviceRegistrationSync<'_>,
) -> Result<PreparedMapleDeviceRegistrationSync, DBError> {
    use crate::models::schema::maple_pairing_reset_clear_obligations;
    let security_epoch = pairing_u64_from_i64(highwater.security_epoch)?;
    let stream_generation = pairing_u64_from_i64(highwater.revocation_stream_generation)?;
    let issued = pairing_u64_from_i64(state.last_issued_revocation_sequence)?;
    let acked = pairing_u64_from_i64(state.last_acked_revocation_sequence)?;
    let expected_status = if pending.is_some() {
        MapleRevocationSyncStatusV1::ResetClearRequired
    } else if issued == acked {
        MapleRevocationSyncStatusV1::Ready
    } else {
        MapleRevocationSyncStatusV1::RevocationsPending
    };
    let response_kind = maple_registration_response_kind(expected_status);

    let (sync, sync_payload_version, sync_payload) = match pending {
        None => {
            let output = materialize(MapleDeviceRegistrationSyncMaterializationContext::Ordinary(
                MapleDeviceRegistrationOrdinarySyncContext {
                    account_id: registration.user_id,
                    subject_project_id: registration.subject_project_id,
                    internal_project_id: registration.project_id,
                    current_device: maple_pairing_create_device_context_from_row(device)?,
                    security_epoch,
                    revocation_stream_id: highwater.revocation_stream_id,
                    revocation_stream_generation: stream_generation,
                    last_issued_issuer_sequence: issued,
                    last_acked_issuer_sequence: acked,
                    status: expected_status,
                },
            ))
            .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
            match output {
                MapleDeviceRegistrationSyncMaterial::Ordinary {
                    sync,
                    sync_payload_version,
                    sync_payload,
                } => (sync, sync_payload_version, sync_payload),
                MapleDeviceRegistrationSyncMaterial::ResetClearRequired { .. } => {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
            }
        }
        Some(mut obligation) if obligation.revision == 1 => {
            let host_claim_payload = decrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                &obligation,
                MapleResetClearPayloadKind::HostClaim,
            )?;
            let instruction_payload = decrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                &obligation,
                MapleResetClearPayloadKind::InstructionMaterial,
            )?;
            let previous_instruction_material_digest = obligation
                .previous_instruction_digest
                .as_deref()
                .map(|value| {
                    value
                        .try_into()
                        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)
                })
                .transpose()?;
            let previous_chain_digest = obligation
                .previous_chain_digest
                .as_deref()
                .map(|value| {
                    value
                        .try_into()
                        .map_err(|_| DBError::MaplePairingAuthorityCorrupt)
                })
                .transpose()?;
            let admission_set_digest: [u8; 32] = obligation
                .admission_set_digest
                .as_slice()
                .try_into()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let instruction_material_digest: [u8; 32] = obligation
                .instruction_digest
                .as_slice()
                .try_into()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let chain_digest: [u8; 32] = obligation
                .chain_digest
                .as_slice()
                .try_into()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let output = materialize(
                MapleDeviceRegistrationSyncMaterializationContext::ResetClearRequired(Box::new(
                    MapleResetClearSyncMaterializationContext {
                        account_id: registration.user_id,
                        subject_project_id: registration.subject_project_id,
                        internal_project_id: registration.project_id,
                        event_id: obligation.uuid,
                        reset_id: obligation.reset_id,
                        reset_generation: pairing_u64_from_i64(obligation.reset_generation)?,
                        cumulative_reset_count: pairing_u64_from_i64(
                            obligation.cumulative_reset_count,
                        )?,
                        source_security_epoch: pairing_u64_from_i64(
                            obligation.source_security_epoch,
                        )?,
                        security_epoch,
                        source_revocation_stream_id: obligation.old_revocation_stream_id,
                        source_revocation_stream_generation: pairing_u64_from_i64(
                            obligation.old_revocation_stream_generation,
                        )?,
                        source_last_issued_revocation_sequence: pairing_u64_from_i64(
                            obligation.source_last_issued_revocation_sequence,
                        )?,
                        revocation_stream_id: obligation.target_revocation_stream_id,
                        revocation_stream_generation: pairing_u64_from_i64(
                            obligation.target_revocation_stream_generation,
                        )?,
                        issuer_sequence: pairing_u64_from_i64(
                            obligation.target_instruction_sequence,
                        )?,
                        previous_event_id: obligation.previous_event_id,
                        previous_instruction_material_digest,
                        previous_chain_digest,
                        admission_count: u16::try_from(obligation.admission_count)
                            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
                        admission_set_digest,
                        host_claim_payload_version: obligation.host_claim_payload_version,
                        host_claim_payload,
                        current_device: maple_pairing_create_device_context_from_row(device)?,
                        instruction_payload_version: obligation.instruction_payload_version,
                        instruction_payload,
                        instruction_material_digest,
                        chain_digest,
                        reset_at: obligation.reset_at,
                    },
                )),
            )
            .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
            let (
                sync,
                signed_instruction_payload_version,
                signed_instruction_payload,
                sync_payload_version,
                sync_payload,
            ) = match output {
                MapleDeviceRegistrationSyncMaterial::ResetClearRequired {
                    sync,
                    signed_instruction_payload_version,
                    signed_instruction_payload,
                    sync_payload_version,
                    sync_payload,
                } => (
                    sync,
                    signed_instruction_payload_version,
                    signed_instruction_payload,
                    sync_payload_version,
                    sync_payload,
                ),
                MapleDeviceRegistrationSyncMaterial::Ordinary { .. } => {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
            };
            let instruction = sync
                .reset_clear_instruction
                .as_ref()
                .ok_or(DBError::MaplePairingMaterializationFailed)?;
            validate_reset_clear_instruction_against_obligation(
                &registration.enclave_key,
                registration.project_id,
                instruction,
                &obligation,
            )?;
            if signed_instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                || sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                || signed_instruction_payload
                    != serde_json::to_vec(instruction)
                        .map_err(|_| DBError::MaplePairingMaterializationFailed)?
            {
                return Err(DBError::MaplePairingMaterializationFailed);
            }
            let signed_instruction_digest = instruction
                .event_digest()
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?
                .to_vec();
            let signed_instruction_payload_enc = encrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                MapleResetClearPayloadKind::SignedInstruction,
                obligation.uuid,
                &obligation.authority_scope_digest,
                &obligation.lookup_digest,
                &obligation.instruction_digest,
                &obligation.chain_digest,
                signed_instruction_payload_version,
                Some(&instruction.issuer_key_id),
                &signed_instruction_digest,
                &signed_instruction_payload,
            )?;
            let sync_digest = Sha256::digest(&sync_payload).to_vec();
            let sync_issuer_key_id = sync.stream_checkpoint.issuer_key_id.clone();
            let sync_payload_enc = encrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                MapleResetClearPayloadKind::Sync,
                obligation.uuid,
                &obligation.authority_scope_digest,
                &obligation.lookup_digest,
                &obligation.instruction_digest,
                &obligation.chain_digest,
                sync_payload_version,
                Some(&sync_issuer_key_id),
                &sync_digest,
                &sync_payload,
            )?;
            obligation.signed_instruction_payload_version =
                Some(signed_instruction_payload_version);
            obligation.signed_instruction_payload_enc = Some(signed_instruction_payload_enc);
            obligation.signed_instruction_issuer_key_id = Some(instruction.issuer_key_id.clone());
            obligation.signed_instruction_digest = Some(signed_instruction_digest);
            obligation.sync_payload_version = Some(sync_payload_version);
            obligation.sync_payload_enc = Some(sync_payload_enc);
            obligation.sync_issuer_key_id = Some(sync_issuer_key_id);
            obligation.sync_digest = Some(sync_digest);
            obligation.revision = 2;
            obligation.record_mac = maple_pairing_reset_clear_obligation_record_mac(
                &registration.enclave_key,
                &obligation,
            )?;
            let changed = diesel::update(
                maple_pairing_reset_clear_obligations::table
                    .filter(maple_pairing_reset_clear_obligations::id.eq(obligation.id))
                    .filter(maple_pairing_reset_clear_obligations::state.eq(1_i16))
                    .filter(maple_pairing_reset_clear_obligations::revision.eq(1_i64)),
            )
            .set((
                maple_pairing_reset_clear_obligations::signed_instruction_payload_version
                    .eq(obligation.signed_instruction_payload_version),
                maple_pairing_reset_clear_obligations::signed_instruction_payload_enc
                    .eq(&obligation.signed_instruction_payload_enc),
                maple_pairing_reset_clear_obligations::signed_instruction_issuer_key_id
                    .eq(&obligation.signed_instruction_issuer_key_id),
                maple_pairing_reset_clear_obligations::signed_instruction_digest
                    .eq(&obligation.signed_instruction_digest),
                maple_pairing_reset_clear_obligations::sync_payload_version
                    .eq(obligation.sync_payload_version),
                maple_pairing_reset_clear_obligations::sync_payload_enc
                    .eq(&obligation.sync_payload_enc),
                maple_pairing_reset_clear_obligations::sync_issuer_key_id
                    .eq(&obligation.sync_issuer_key_id),
                maple_pairing_reset_clear_obligations::sync_digest.eq(&obligation.sync_digest),
                maple_pairing_reset_clear_obligations::revision.eq(2_i64),
                maple_pairing_reset_clear_obligations::record_mac.eq(&obligation.record_mac),
            ))
            .execute(conn)?;
            if changed != 1 {
                return Err(DBError::MaplePairingConflict);
            }
            (sync, sync_payload_version, sync_payload)
        }
        Some(obligation) if obligation.revision == 2 => {
            let signed_instruction_payload = decrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                &obligation,
                MapleResetClearPayloadKind::SignedInstruction,
            )?;
            let sync_payload = decrypt_maple_reset_clear_payload(
                &registration.enclave_key,
                &obligation,
                MapleResetClearPayloadKind::Sync,
            )?;
            let sync: MapleRevocationSyncV1 = serde_json::from_slice(&sync_payload)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let instruction = sync
                .reset_clear_instruction
                .as_ref()
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            if signed_instruction_payload
                != serde_json::to_vec(instruction)
                    .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?
                || obligation.sync_digest.as_deref()
                    != Some(Sha256::digest(&sync_payload).as_slice())
                || obligation.signed_instruction_digest.as_deref()
                    != Some(
                        instruction
                            .event_digest()
                            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?
                            .as_slice(),
                    )
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            validate_reset_clear_instruction_against_obligation(
                &registration.enclave_key,
                registration.project_id,
                instruction,
                &obligation,
            )?;
            (
                sync,
                obligation
                    .sync_payload_version
                    .ok_or(DBError::MaplePairingAuthorityCorrupt)?,
                sync_payload,
            )
        }
        Some(_) => return Err(DBError::MaplePairingAuthorityCorrupt),
    };

    validate_maple_device_registration_sync(
        &registration.enclave_key,
        issuer_keyset,
        &sync,
        expected_status,
        registration.user_id,
        registration.subject_project_id,
        registration.project_id,
        device,
        highwater.security_epoch,
        highwater.revocation_stream_id,
        highwater.revocation_stream_generation,
        state.last_issued_revocation_sequence,
        state.last_acked_revocation_sequence,
    )?;
    if sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || sync_payload.is_empty()
        || sync_payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || sync_payload
            != serde_json::to_vec(&sync).map_err(|_| DBError::MaplePairingMaterializationFailed)?
    {
        return Err(DBError::MaplePairingMaterializationFailed);
    }
    let issuer_key_id = sync.stream_checkpoint.issuer_key_id.clone();
    let digest = Sha256::digest(&sync_payload).to_vec();
    let payload_enc = encrypt_maple_device_sync_payload(
        &registration.enclave_key,
        registration.user_id,
        registration.project_id,
        registration.operation_id,
        device.uuid,
        device.revision,
        highwater.security_epoch,
        response_kind,
        sync_payload_version,
        &issuer_key_id,
        &digest,
        &sync_payload,
    )?;
    Ok(PreparedMapleDeviceRegistrationSync {
        response_kind,
        payload_version: sync_payload_version,
        payload: sync_payload,
        payload_enc,
        issuer_key_id,
        digest,
    })
}

fn validate_maple_pairing_revocation_record(
    enclave_key: &[u8],
    row: &MaplePairingRevocationEvent,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_revocation_record_mac(
        enclave_key,
        row.uuid,
        row.user_id,
        row.project_id,
        row.recipient_host_maple_device_id,
        row.revocation_stream_id,
        row.revocation_stream_generation,
        row.issuer_sequence,
        row.maple_pairing_id,
        row.pairing_incarnation,
        &row.issuer_key_id,
        row.payload_version,
        &row.payload_enc,
        &row.event_digest,
        row.created_at,
        row.acked_at,
    )?;
    if row.id <= 0
        || row.uuid.is_nil()
        || row.recipient_host_maple_device_id <= 0
        || row.revocation_stream_id.is_nil()
        || row.revocation_stream_generation <= 0
        || row.issuer_sequence <= 0
        || row.maple_pairing_id <= 0
        || row.pairing_incarnation <= 0
        || !maple_pairing_issuer_key_id_is_valid(&row.issuer_key_id)
        || row.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || row.payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_REVOCATION_BYTES
        || row.event_digest.len() != 32
        || !bool::from(expected.as_slice().ct_eq(row.record_mac.as_slice()))
    {
        return Err(DBError::MaplePairingCorrupt);
    }
    Ok(())
}

fn pairing_operation_receipt(
    enclave_key: &[u8],
    operation: &crate::models::maple_pairing_db::MaplePairingOperation,
    pair_id: Uuid,
) -> Result<MaplePairingOperationReceipt, DBError> {
    use subtle::ConstantTimeEq;

    let expected = maple_pairing_receipt_mac(
        enclave_key,
        operation.operation_id,
        operation.user_id,
        operation.project_id,
        operation.actor_maple_device_id,
        operation.operation_kind,
        &operation.request_mac,
        operation.maple_pairing_id,
        operation.pairing_revision,
        operation.receipt_version,
        &operation.receipt_enc,
        operation.receipt_issuer_key_id.as_deref(),
        operation.accepted_at,
    )?;
    let receipt_issuer_shape_is_valid = if operation.operation_kind == MAPLE_PAIRING_OPERATION_ACK {
        operation
            .receipt_issuer_key_id
            .as_deref()
            .is_some_and(maple_pairing_issuer_key_id_is_valid)
    } else {
        operation.receipt_issuer_key_id.is_none()
    };
    if operation.id <= 0
        || operation.operation_id.is_nil()
        || operation.actor_maple_device_id <= 0
        || operation.maple_pairing_id <= 0
        || operation.pairing_revision <= 0
        || operation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
        || operation.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || !receipt_issuer_shape_is_valid
        || !bool::from(expected.as_slice().ct_eq(operation.receipt_mac.as_slice()))
    {
        return Err(DBError::MaplePairingCorrupt);
    }
    Ok(MaplePairingOperationReceipt {
        operation_id: operation.operation_id,
        pair_id,
        pairing_revision: operation.pairing_revision,
        receipt_version: operation.receipt_version,
        receipt_enc: operation.receipt_enc.clone(),
        accepted_at: operation.accepted_at,
    })
}

fn find_scoped_maple_device(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    registration_id: Uuid,
    exclusive: bool,
) -> Result<Option<MapleDevice>, DBError> {
    use crate::models::schema::maple_devices;
    use diesel::OptionalExtension;

    let query = maple_devices::table
        .filter(maple_devices::user_id.eq(authorization.user_id))
        .filter(maple_devices::project_id.eq(authorization.project_id))
        .filter(maple_devices::uuid.eq(registration_id));
    let row = if exclusive {
        query.for_update().first::<MapleDevice>(conn).optional()?
    } else {
        query.for_share().first::<MapleDevice>(conn).optional()?
    };
    if let Some(device) = row.as_ref() {
        if !maple_device_record_mac_is_valid(&authorization.enclave_key, device)? {
            return Err(DBError::MaplePairingCorrupt);
        }
    }
    Ok(row)
}

fn maple_pairing_create_device_context_from_row(
    device: &MapleDevice,
) -> Result<MaplePairingCreateDeviceContext, DBError> {
    Ok(MaplePairingCreateDeviceContext {
        registration_id: device.uuid,
        device_id: device.device_id,
        installation_id: device.installation_id,
        endpoint_epoch: device
            .endpoint_epoch
            .try_into()
            .map_err(|_| DBError::MaplePairingCorrupt)?,
        device_revision: device.revision,
        payload_version: device.payload_version,
        payload_enc: device.payload_enc.clone(),
        identity_mac: device.identity_mac.clone(),
        record_mac: device.record_mac.clone(),
    })
}

fn maple_registration_response_kind(status: MapleRevocationSyncStatusV1) -> i16 {
    match status {
        MapleRevocationSyncStatusV1::Ready => MAPLE_REGISTRATION_SYNC_READY,
        MapleRevocationSyncStatusV1::RevocationsPending => {
            MAPLE_REGISTRATION_SYNC_REVOCATIONS_PENDING
        }
        MapleRevocationSyncStatusV1::ResetClearRequired => {
            MAPLE_REGISTRATION_SYNC_RESET_CLEAR_REQUIRED
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_maple_device_registration_sync(
    enclave_key: &[u8],
    issuer_keyset: &MaplePairingIssuerKeySetV1,
    sync: &MapleRevocationSyncV1,
    expected_status: MapleRevocationSyncStatusV1,
    user_id: Uuid,
    subject_project_id: Uuid,
    internal_project_id: i32,
    device: &MapleDevice,
    security_epoch: i64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: i64,
    last_issued: i64,
    last_acked: i64,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    sync.verify(issuer_keyset)
        .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
    let checkpoint = &sync.stream_checkpoint;
    let claim = &checkpoint.host;
    let identity_public_key = claim
        .verifying_key_bytes()
        .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
    let expected_identity_mac = maple_device_identity_mac_from_claim(
        enclave_key,
        user_id,
        internal_project_id,
        &identity_public_key,
    )?;
    if sync.status != expected_status
        || sync.security_epoch != pairing_u64_from_i64(security_epoch)?
        || checkpoint.subject_account_id != user_id
        || checkpoint.subject_project_id != subject_project_id
        || checkpoint.security_epoch != pairing_u64_from_i64(security_epoch)?
        || checkpoint.revocation_stream_id != revocation_stream_id
        || checkpoint.revocation_stream_generation
            != pairing_u64_from_i64(revocation_stream_generation)?
        || checkpoint.last_issued_issuer_sequence != pairing_u64_from_i64(last_issued)?
        || checkpoint.last_acked_issuer_sequence != pairing_u64_from_i64(last_acked)?
        || claim.registration_id != device.uuid
        || claim.device_id != device.device_id
        || claim.installation_id != device.installation_id
        || claim.endpoint_epoch != pairing_u64_from_i64(device.endpoint_epoch)?
        || !bool::from(
            expected_identity_mac
                .as_slice()
                .ct_eq(device.identity_mac.as_slice()),
        )
    {
        return Err(DBError::MaplePairingMaterializationFailed);
    }
    Ok(())
}

/// Requires a registered installation to have no unresolved reset-clear head.
/// Callers must already hold the global authority snapshot fence and have
/// completed the authenticated account inventory audit; this focused lookup
/// then protects the exact actor/participant against a reset racing replay or
/// mutation dispatch.
fn require_no_pending_reset_clear(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    device: &MapleDevice,
    exclusive: bool,
) -> Result<(), DBError> {
    let (_, highwater) = load_maple_pairing_revocation_highwater(
        conn,
        &authorization.enclave_key,
        authorization.user_id,
        authorization.project_id,
        device.installation_id,
        exclusive,
    )?;
    let highwater = highwater.ok_or(DBError::MaplePairingCorrupt)?;
    if highwater.security_epoch <= 0 {
        return Err(DBError::MaplePairingCorrupt);
    }
    if let Some(pending) = load_latest_pending_maple_reset_clear_obligation(
        conn,
        &authorization.enclave_key,
        &highwater,
        exclusive,
    )? {
        if !maple_pairing_authority_mac_matches(&pending.host_identity_mac, &device.identity_mac) {
            return Err(DBError::MaplePairingCorrupt);
        }
        return Err(DBError::MaplePairingResetClearRequired);
    }
    Ok(())
}

fn require_maple_pairing_participants_ready(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    pairing: &MaplePairing,
    exclusive: bool,
) -> Result<(), DBError> {
    use crate::models::schema::maple_devices;

    let mut participant_ids = vec![
        pairing.controller_maple_device_id,
        pairing.host_maple_device_id,
    ];
    participant_ids.sort_unstable();
    participant_ids.dedup();
    if participant_ids.len() != 2 {
        return Err(DBError::MaplePairingCorrupt);
    }
    let query = maple_devices::table
        .filter(maple_devices::user_id.eq(authorization.user_id))
        .filter(maple_devices::project_id.eq(authorization.project_id))
        .filter(maple_devices::id.eq_any(&participant_ids))
        .order(maple_devices::id.asc());
    let participants = if exclusive {
        query.for_update().load::<MapleDevice>(conn)?
    } else {
        query.for_share().load::<MapleDevice>(conn)?
    };
    if participants.len() != 2 {
        return Err(DBError::MaplePairingCorrupt);
    }
    for participant in &participants {
        if !maple_device_record_mac_is_valid(&authorization.enclave_key, participant)? {
            return Err(DBError::MaplePairingCorrupt);
        }
        require_no_pending_reset_clear(conn, authorization, participant, exclusive)?;
    }
    Ok(())
}

fn get_prior_pairing_operation(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    actor_maple_device_id: i64,
    operation_id: Uuid,
) -> Result<Option<crate::models::maple_pairing_db::MaplePairingOperation>, DBError> {
    use crate::models::schema::maple_pairing_operations;
    use diesel::OptionalExtension;

    maple_pairing_operations::table
        .filter(maple_pairing_operations::user_id.eq(authorization.user_id))
        .filter(maple_pairing_operations::project_id.eq(authorization.project_id))
        .filter(maple_pairing_operations::actor_maple_device_id.eq(actor_maple_device_id))
        .filter(maple_pairing_operations::operation_id.eq(operation_id))
        .first(conn)
        .optional()
        .map_err(DBError::from)
}

fn replay_pairing_operation(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    operation: &crate::models::maple_pairing_db::MaplePairingOperation,
    operation_kind: i16,
    request_mac: &[u8],
) -> Result<MaplePairingOperationReceipt, DBError> {
    use crate::models::schema::maple_pairings;
    use subtle::ConstantTimeEq;

    if operation.operation_kind != operation_kind
        || !bool::from(operation.request_mac.as_slice().ct_eq(request_mac))
    {
        return Err(DBError::MaplePairingConflict);
    }
    let pairing = maple_pairings::table
        .filter(maple_pairings::id.eq(operation.maple_pairing_id))
        .filter(maple_pairings::user_id.eq(authorization.user_id))
        .filter(maple_pairings::project_id.eq(authorization.project_id))
        .first::<MaplePairing>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
            other => DBError::QueryError(other),
        })?;
    validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
    require_maple_pairing_participants_ready(conn, authorization, &pairing, false)?;
    pairing_operation_receipt(&authorization.enclave_key, operation, pairing.uuid)
}

#[allow(clippy::too_many_arguments)]
fn insert_pairing_operation(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
    actor_maple_device_id: i64,
    operation_id: Uuid,
    operation_kind: i16,
    request_mac: &[u8],
    pairing: &MaplePairing,
    receipt_version: i16,
    receipt_enc: &[u8],
    receipt_issuer_key_id: Option<&str>,
    accepted_at: DateTime<Utc>,
) -> Result<MaplePairingOperationReceipt, DBError> {
    use crate::models::maple_pairing_db::{MaplePairingOperation, NewMaplePairingOperation};
    use crate::models::schema::maple_pairing_operations;

    if operation_id.is_nil()
        || request_mac.len() != 32
        || receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
        || receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || (operation_kind == MAPLE_PAIRING_OPERATION_ACK) != receipt_issuer_key_id.is_some()
        || receipt_issuer_key_id.is_some_and(|key_id| !maple_pairing_issuer_key_id_is_valid(key_id))
    {
        return Err(DBError::MaplePairingConflict);
    }
    let operation_count = maple_pairing_operations::table
        .filter(maple_pairing_operations::maple_pairing_id.eq(pairing.id))
        .count()
        .get_result::<i64>(conn)?;
    if operation_count >= MAPLE_PAIRING_OPERATION_LIMIT_PER_PAIRING {
        return Err(DBError::MaplePairingOperationLimitExceeded);
    }
    let accepted_at = normalize_db_time(accepted_at)?;
    let receipt_mac = maple_pairing_receipt_mac(
        &authorization.enclave_key,
        operation_id,
        authorization.user_id,
        authorization.project_id,
        actor_maple_device_id,
        operation_kind,
        request_mac,
        pairing.id,
        pairing.revision,
        receipt_version,
        receipt_enc,
        receipt_issuer_key_id,
        accepted_at,
    )?;
    let operation = diesel::insert_into(maple_pairing_operations::table)
        .values(NewMaplePairingOperation {
            operation_id,
            user_id: authorization.user_id,
            project_id: authorization.project_id,
            actor_maple_device_id,
            operation_kind,
            request_mac: request_mac.to_vec(),
            maple_pairing_id: pairing.id,
            pairing_revision: pairing.revision,
            receipt_version,
            receipt_enc: receipt_enc.to_vec(),
            receipt_issuer_key_id: receipt_issuer_key_id.map(str::to_owned),
            receipt_mac,
            accepted_at,
        })
        .get_result::<MaplePairingOperation>(conn)
        .map_err(|error| match error {
            diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::UniqueViolation,
                _,
            ) => DBError::MaplePairingConflict,
            other => DBError::QueryError(other),
        })?;
    pairing_operation_receipt(&authorization.enclave_key, &operation, pairing.uuid)
}

fn expire_pending_pairings(
    conn: &mut PgConnection,
    authorization: &MaplePairingAuthorization,
) -> Result<(DateTime<Utc>, bool), DBError> {
    use crate::models::schema::maple_pairings;

    // PostgreSQL CURRENT_TIMESTAMP is stable for the transaction, so expiry
    // projection and a following approval evaluate the exact same trusted
    // instant even if the application clock changes between statements.
    let now = maple_pairing_trusted_db_now(conn)?;
    let expiry_cutoff = now
        .checked_sub_signed(chrono::Duration::milliseconds(
            MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
        ))
        .ok_or(DBError::MaplePairingConflict)?;
    let expired = maple_pairings::table
        .filter(maple_pairings::user_id.eq(authorization.user_id))
        .filter(maple_pairings::project_id.eq(authorization.project_id))
        .filter(maple_pairings::state.eq(MaplePairingState::Pending.as_db()))
        .filter(maple_pairings::expires_at.le(expiry_cutoff))
        .for_update()
        .load::<MaplePairing>(conn)?;
    let changed_any = !expired.is_empty();
    for row in expired {
        validate_maple_pairing_record(&authorization.enclave_key, &row)?;
        if row.revision != 1 || !maple_pairing_pending_is_expired(row.expires_at, now) {
            return Err(DBError::MaplePairingCorrupt);
        }
        let target_revision = 2;
        let target_state = MaplePairingState::Expired.as_db();
        let record_mac = maple_pairing_record_mac_for_parts(
            &authorization.enclave_key,
            row.uuid,
            row.pairing_request_id,
            row.user_id,
            row.project_id,
            row.lineage_id,
            row.controller_maple_device_id,
            row.host_maple_device_id,
            row.direction,
            row.pairing_incarnation,
            target_state,
            target_revision,
            &row.request_nonce_mac,
            row.revocation_stream_id,
            row.revocation_stream_generation,
            row.pair_authorization_digest.as_deref(),
            &row.ticket_issuer_key_id,
            row.authorization_issuer_key_id.as_deref(),
            row.revocation_issuer_key_id.as_deref(),
            row.payload_version,
            &row.payload_enc,
            row.created_at,
            row.expires_at,
            row.approved_at,
            row.activated_at,
            row.revoked_at,
        )?;
        let changed = diesel::update(
            maple_pairings::table
                .filter(maple_pairings::id.eq(row.id))
                .filter(maple_pairings::state.eq(MaplePairingState::Pending.as_db()))
                .filter(maple_pairings::revision.eq(1)),
        )
        .set((
            maple_pairings::state.eq(target_state),
            maple_pairings::revision.eq(target_revision),
            maple_pairings::record_mac.eq(record_mac),
        ))
        .execute(conn)?;
        if changed != 1 {
            return Err(DBError::MaplePairingConflict);
        }
    }
    Ok((now, changed_any))
}

/// Exhaustively validates a pure reset materializer result against the exact
/// authority facts locked by the surrounding SERIALIZABLE transaction. No
/// retained row, highwater, tombstone, credential, or graph mutation may occur
/// before this validator has accepted every reset target.
fn validate_reset_clear_material_against_locked_context(
    enclave_key: &[u8],
    context: &MapleResetClearUnsignedMaterializationContext,
    expected_identity_mac: &[u8],
    predecessor_claim: Option<&MapleDeviceClaimV1>,
    prepared: &mut MapleResetClearUnsignedMaterial,
) -> Result<(), DBError> {
    use subtle::ConstantTimeEq;

    let invalid = || DBError::MaplePairingMaterializationFailed;
    if context.source_security_epoch.checked_add(1) != Some(context.security_epoch)
        || context.revocation_stream_generation
            != context
                .source_revocation_stream_generation
                .checked_add(1)
                .ok_or_else(invalid)?
        || context.issuer_sequence != 1
        || context.event_id.is_nil()
        || context.reset_id.is_nil()
        || context.account_id.is_nil()
        || context.subject_project_id.is_nil()
        || prepared.host_identity_mac.len() != 32
        || expected_identity_mac.len() != 32
        || !bool::from(
            prepared
                .host_identity_mac
                .as_slice()
                .ct_eq(expected_identity_mac),
        )
        || prepared.host_claim_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || prepared.instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
        || prepared.host_claim_payload.is_empty()
        || prepared.host_claim_payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || prepared.instruction_payload.is_empty()
        || prepared.instruction_payload.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
        || !bool::from(
            sha256_digest(&prepared.host_claim_payload)
                .as_slice()
                .ct_eq(prepared.host_claim_digest.as_slice()),
        )
    {
        return Err(invalid());
    }

    let host: MapleDeviceClaimV1 =
        serde_json::from_slice(&prepared.host_claim_payload).map_err(|_| invalid())?;
    host.validate().map_err(|_| invalid())?;
    let expected_claim_identity_mac = maple_device_identity_mac_from_claim(
        enclave_key,
        context.account_id,
        context.internal_project_id,
        &host.verifying_key_bytes().map_err(|_| invalid())?,
    )?;
    if !bool::from(
        expected_claim_identity_mac
            .as_slice()
            .ct_eq(expected_identity_mac),
    ) {
        return Err(invalid());
    }
    match &context.source {
        MapleResetClearSource::LiveDevice {
            registration_id,
            device_id,
            installation_id,
            endpoint_epoch,
            identity_mac,
            ..
        } => {
            if host.registration_id != *registration_id
                || host.device_id != *device_id
                || host.installation_id != *installation_id
                || i64::try_from(host.endpoint_epoch).ok() != Some(*endpoint_epoch)
                || !bool::from(identity_mac.as_slice().ct_eq(expected_identity_mac))
            {
                return Err(invalid());
            }
        }
        MapleResetClearSource::RetainedHostClaim {
            prior_event_id,
            payload,
            payload_digest,
            identity_mac,
            prior_target_revocation_stream_id,
            prior_target_revocation_stream_generation,
            prior_target_security_epoch,
            ..
        } => {
            let retained: MapleDeviceClaimV1 =
                serde_json::from_slice(payload).map_err(|_| invalid())?;
            retained.validate().map_err(|_| invalid())?;
            let canonical_retained_payload =
                serde_json::to_vec(&retained).map_err(|_| invalid())?;
            if Some(*prior_event_id) != context.previous_event_id
                || *prior_target_revocation_stream_id != context.source_revocation_stream_id
                || *prior_target_revocation_stream_generation
                    != context.source_revocation_stream_generation
                || *prior_target_security_epoch != context.source_security_epoch
                || !bool::from(
                    sha256_digest(payload)
                        .as_slice()
                        .ct_eq(payload_digest.as_slice()),
                )
                || !bool::from(identity_mac.as_slice().ct_eq(expected_identity_mac))
                || canonical_retained_payload.as_slice() != payload.as_slice()
                || host != retained
            {
                return Err(invalid());
            }
        }
    }
    let canonical_host_claim_payload = serde_json::to_vec(&host).map_err(|_| invalid())?;
    if let Some(predecessor) = predecessor_claim {
        predecessor.validate().map_err(|_| invalid())?;
        let canonical_predecessor_payload =
            serde_json::to_vec(predecessor).map_err(|_| invalid())?;
        if predecessor != &host || canonical_predecessor_payload != canonical_host_claim_payload {
            return Err(invalid());
        }
    }

    let wire_leaves = context
        .admission_leaves
        .iter()
        .map(|leaf| MapleResetClearAdmissionLeafV1 {
            pair_id: leaf.pair_id,
            pairing_incarnation: leaf.pairing_incarnation,
            pair_authorization_digest: leaf.pair_authorization_digest,
        })
        .collect::<Vec<_>>();
    let admission_count: u16 = wire_leaves.len().try_into().map_err(|_| invalid())?;
    let admission_set_digest =
        reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, &wire_leaves)
            .map_err(|_| invalid())?;
    let instruction: MapleResetClearRequiredV1 =
        serde_json::from_slice(&prepared.instruction_payload).map_err(|_| invalid())?;
    let expected_previous_instruction = context
        .previous_instruction_material_digest
        .map(|digest| STANDARD.encode(digest));
    let expected_previous_chain = context
        .previous_chain_digest
        .map(|digest| STANDARD.encode(digest));
    if instruction.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1
        || instruction.event_id != context.event_id
        || instruction.reset_id != context.reset_id
        || instruction.reset_generation != context.reset_generation
        || instruction.cumulative_reset_count != context.cumulative_reset_count
        || instruction.source_security_epoch != context.source_security_epoch
        || instruction.security_epoch != context.security_epoch
        || instruction.subject_account_id != context.account_id
        || instruction.subject_project_id != context.subject_project_id
        || instruction.recipient_host_registration_id != host.registration_id
        || instruction.host != host
        || instruction.issuer_sequence != context.issuer_sequence
        || instruction.source_revocation_stream_id != context.source_revocation_stream_id
        || instruction.source_revocation_stream_generation
            != context.source_revocation_stream_generation
        || instruction.revocation_stream_id != context.revocation_stream_id
        || instruction.revocation_stream_generation != context.revocation_stream_generation
        || instruction.clear_scope
            != MapleResetClearScopeV1::AllPairAuthorizationsForAccountProjectHostInstallation
        || instruction.admission_count != admission_count
        || instruction.admission_set_digest != STANDARD.encode(admission_set_digest)
        || instruction.previous_reset_clear_event_id != context.previous_event_id
        || instruction.previous_instruction_material_digest != expected_previous_instruction
        || instruction.previous_chain_digest != expected_previous_chain
        || instruction.reset_at_unix_ms != context.reset_at.timestamp_millis()
        || !instruction.issuer_key_id.is_empty()
        || !instruction.issuer_signature.is_empty()
    {
        return Err(invalid());
    }
    let material_transcript =
        reset_clear_instruction_material_transcript(&instruction).map_err(|_| invalid())?;
    let material_digest = sha256_digest(&material_transcript);
    if material_transcript != prepared.instruction_material_transcript
        || !bool::from(
            material_digest
                .as_slice()
                .ct_eq(prepared.instruction_material_digest.as_slice()),
        )
        || instruction.instruction_material_digest
            != STANDARD.encode(prepared.instruction_material_digest)
    {
        return Err(invalid());
    }
    let chain_digest =
        sha256_digest(&reset_clear_chain_transcript(&instruction).map_err(|_| invalid())?);
    if !bool::from(
        chain_digest
            .as_slice()
            .ct_eq(prepared.chain_digest.as_slice()),
    ) || instruction.chain_digest != STANDARD.encode(prepared.chain_digest)
    {
        return Err(invalid());
    }

    // The callback proves that it can decrypt and validate the web-owned
    // device record, but retained host bytes are DB-canonical. JSON whitespace
    // or key ordering supplied by a callback must never fork the persistent
    // reset chain for the same typed claim.
    prepared.host_claim_payload = canonical_host_claim_payload;
    prepared.host_claim_digest = sha256_digest(&prepared.host_claim_payload);
    prepared.instruction_payload = serde_json::to_vec(&instruction).map_err(|_| invalid())?;
    Ok(())
}

struct PersistedMaplePairingResetClear {
    target_security_epoch: i64,
    reset_at: DateTime<Utc>,
}

#[derive(diesel::QueryableByName)]
struct MaplePairingResetLifecycleMaximum {
    #[diesel(sql_type = diesel::sql_types::Timestamptz)]
    reset_at: DateTime<Utc>,
}

/// Return the greatest committed lifecycle timestamp in the live authority
/// graph that this reset will retire. The account graph has already been fully
/// authenticated in this SERIALIZABLE snapshot; the later deletes create the
/// corresponding predicate/write conflicts against hostile out-of-band DML.
fn load_maple_pairing_reset_lifecycle_maximum(
    conn: &mut PgConnection,
    user_id: Uuid,
    project_id: i32,
) -> Result<DateTime<Utc>, DBError> {
    diesel::sql_query(
        r#"
        WITH reset_scope AS (
            SELECT $1::uuid AS user_id, $2::integer AS project_id
        )
        SELECT GREATEST(
            CURRENT_TIMESTAMP,
            COALESCE((SELECT MAX(updated_at) FROM maple_devices
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(accepted_at) FROM maple_device_registration_operations
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(updated_at) FROM maple_pairing_host_states
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(updated_at) FROM maple_pairing_lineages
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(updated_at) FROM maple_pairings
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(accepted_at) FROM maple_pairing_operations
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(GREATEST(created_at, COALESCE(acked_at, created_at)))
                FROM maple_pairing_revocation_events
                WHERE user_id = (SELECT user_id FROM reset_scope)
                  AND project_id = (SELECT project_id FROM reset_scope)), '-infinity'::timestamptz),
            COALESCE((SELECT MAX(updated_at) FROM maple_pairing_revocation_highwaters
                WHERE authority_scope_digest = (
                    SELECT authority_scope_digest
                    FROM maple_pairing_authority_account_heads
                    WHERE user_id = (SELECT user_id FROM reset_scope)
                      AND project_id = (SELECT project_id FROM reset_scope)
                )), '-infinity'::timestamptz)
        ) AS reset_at
        "#,
    )
    .bind::<diesel::sql_types::Uuid, _>(user_id)
    .bind::<diesel::sql_types::Integer, _>(project_id)
    .get_result::<MaplePairingResetLifecycleMaximum>(conn)
    .map(|row| row.reset_at)
    .map_err(DBError::QueryError)
}

/// Rotate every currently registered installation into a fresh revocation
/// namespace and append a persistent reset-clear obligation before destructive
/// reset removes its recoverable state. Unresolved prior obligations are
/// rotated again even when their original device graph is already gone.
fn persist_maple_pairing_reset_clear_obligations_for_user(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    subject_project_id: Uuid,
    minimum_reset_at: DateTime<Utc>,
    build_material: &BuildResetClearMaterial<'_>,
) -> Result<PersistedMaplePairingResetClear, DBError> {
    use crate::models::maple_pairing_db::{
        MaplePairingRevocationHighwater, NewMaplePairingRevocationHighwater,
    };
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_authority_account_heads,
        maple_pairing_host_states, maple_pairing_registration_operation_tombstones,
        maple_pairing_reset_clear_admissions, maple_pairing_reset_clear_obligations,
        maple_pairing_revocation_events, maple_pairing_revocation_highwaters, maple_pairings,
    };
    use subtle::ConstantTimeEq;

    #[derive(Clone)]
    struct ResetPrior {
        id: i64,
        uuid: Uuid,
        reset_generation: i64,
        cumulative_reset_count: i64,
        instruction_digest: Vec<u8>,
        chain_digest: Vec<u8>,
        host_claim_digest: Vec<u8>,
        admission_count: i16,
        state: i16,
    }

    impl From<&MaplePairingResetClearObligation> for ResetPrior {
        fn from(row: &MaplePairingResetClearObligation) -> Self {
            Self {
                id: row.id,
                uuid: row.uuid,
                reset_generation: row.reset_generation,
                cumulative_reset_count: row.cumulative_reset_count,
                instruction_digest: row.instruction_digest.clone(),
                chain_digest: row.chain_digest.clone(),
                host_claim_digest: row.host_claim_digest.clone(),
                admission_count: row.admission_count,
                state: row.state,
            }
        }
    }

    #[derive(Clone)]
    enum ResetSourceRow {
        Live(MapleDevice),
        Retained { prior_id: i64 },
    }

    #[derive(Clone)]
    struct ResetTarget {
        lookup_digest: Vec<u8>,
        identity_mac: Vec<u8>,
        source: ResetSourceRow,
        prior: Option<ResetPrior>,
        highwater: MaplePairingRevocationHighwater,
    }

    struct PreparedResetTarget {
        lookup_digest: Vec<u8>,
        identity_mac: Vec<u8>,
        prior: Option<ResetPrior>,
        highwater: MaplePairingRevocationHighwater,
        target_stream_generation: i64,
        target_stream_id: Uuid,
        event_id: Uuid,
        live_device_id: Option<i64>,
        admission_leaves: Vec<MapleResetClearAdmissionMaterial>,
        prepared: MapleResetClearUnsignedMaterial,
    }

    let account_head = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .for_update()
        .first::<MaplePairingAuthorityAccountHead>(conn)?;
    validate_maple_pairing_authority_account_head(enclave_key, &account_head)?;
    let source_security_epoch = account_head.security_epoch;
    let target_security_epoch = source_security_epoch
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    let authority_scope_digest = account_head.authority_scope_digest.clone();
    let reset_id = Uuid::new_v4();
    let mut reset_at = maple_pairing_trusted_db_now(conn)?.max(minimum_reset_at);

    let counts = count_maple_pairing_authority_account_rows(
        conn,
        &authority_scope_digest,
        user_id,
        project_id,
    )?;
    let devices = maple_devices::table
        .filter(maple_devices::user_id.eq(user_id))
        .filter(maple_devices::project_id.eq(project_id))
        .order(maple_devices::id.asc())
        .for_update()
        .load::<MapleDevice>(conn)?;
    let mut targets = BTreeMap::<Vec<u8>, ResetTarget>::new();
    let mut lookup_by_live_device_id = BTreeMap::<i64, Vec<u8>>::new();
    for device in devices {
        let device_row_id = device.id;
        if !maple_device_record_mac_is_valid(enclave_key, &device)? {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let (lookup_digest, highwater) = load_maple_pairing_revocation_highwater(
            conn,
            enclave_key,
            user_id,
            project_id,
            device.installation_id,
            true,
        )?;
        let highwater = highwater.ok_or(DBError::MaplePairingAuthorityCorrupt)?;
        if highwater.security_epoch != source_security_epoch {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let prior = load_latest_maple_reset_clear_obligation(conn, enclave_key, &highwater, true)?;
        if prior.as_ref().is_some_and(|row| {
            !maple_pairing_authority_mac_matches(&row.host_identity_mac, &device.identity_mac)
        }) {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        // A reset-clear ACK permanently retires this enrollment. Its live row
        // must have been consumed atomically by the ACK path; seeing it again
        // is corruption, never authority to rotate or resurrect it.
        if prior.as_ref().is_some_and(|row| row.state != 1) {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        if let Some(prior) = &prior {
            reset_at = reset_at.max(prior.reset_at);
        }
        if targets
            .insert(
                lookup_digest.clone(),
                ResetTarget {
                    lookup_digest: lookup_digest.clone(),
                    identity_mac: device.identity_mac.clone(),
                    source: ResetSourceRow::Live(device),
                    prior: prior.as_ref().map(ResetPrior::from),
                    highwater,
                },
            )
            .is_some()
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        if lookup_by_live_device_id
            .insert(device_row_id, lookup_digest)
            .is_some()
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
    }

    // Page the compact highwater inventory and retain only the latest namespace
    // per lookup. This avoids retaining up to 4,096 ciphertext-bearing reset
    // rows merely to find the current chain head.
    let mut latest_highwater_by_lookup = BTreeMap::new();
    let mut highwater_cursor_lookup = Vec::new();
    let mut highwater_cursor_generation = 0_i64;
    loop {
        let page = maple_pairing_revocation_highwaters::table
            .filter(
                maple_pairing_revocation_highwaters::authority_scope_digest
                    .eq(&authority_scope_digest),
            )
            .filter(
                maple_pairing_revocation_highwaters::lookup_digest
                    .gt(&highwater_cursor_lookup)
                    .or(maple_pairing_revocation_highwaters::lookup_digest
                        .eq(&highwater_cursor_lookup)
                        .and(
                            maple_pairing_revocation_highwaters::revocation_stream_generation
                                .gt(highwater_cursor_generation),
                        )),
            )
            .order((
                maple_pairing_revocation_highwaters::lookup_digest.asc(),
                maple_pairing_revocation_highwaters::revocation_stream_generation.asc(),
            ))
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_update()
            .load::<MaplePairingRevocationHighwater>(conn)?;
        if page.is_empty() {
            break;
        }
        for row in page {
            validate_maple_pairing_revocation_highwater(enclave_key, &row)?;
            if !maple_pairing_authority_mac_matches(
                &row.authority_scope_digest,
                &authority_scope_digest,
            ) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            highwater_cursor_lookup = row.lookup_digest.clone();
            highwater_cursor_generation = row.revocation_stream_generation;
            latest_highwater_by_lookup.insert(row.lookup_digest.clone(), row);
        }
    }

    // A missed reset may leave no live device. Rotate only an unresolved latest
    // chain head; acknowledged historical-only lookups are intentionally not
    // resurrected. Loading the highest generation (of any state) prevents an
    // older Pending ancestor from being mistaken for the current head.
    for highwater in latest_highwater_by_lookup.into_values() {
        if targets.contains_key(&highwater.lookup_digest) {
            continue;
        }
        let Some(prior) =
            load_latest_maple_reset_clear_obligation(conn, enclave_key, &highwater, true)?
        else {
            continue;
        };
        if prior.state != 1 {
            continue;
        }
        if highwater.security_epoch != source_security_epoch
            || highwater.last_issued_revocation_sequence < 1
        {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        reset_at = reset_at.max(prior.reset_at);
        let prior_summary = ResetPrior::from(&prior);
        targets.insert(
            highwater.lookup_digest.clone(),
            ResetTarget {
                lookup_digest: highwater.lookup_digest.clone(),
                identity_mac: prior.host_identity_mac,
                source: ResetSourceRow::Retained { prior_id: prior.id },
                prior: Some(prior_summary),
                highwater,
            },
        );
    }

    // Lock and authenticate every live registration operation before any
    // materializer runs. A reset has one lifecycle timestamp, clamped past all
    // predecessor resets and every accepted operation that it retires. This
    // remains truthful even if the database wall clock moved backwards.
    let mut operation_cursor = 0_i64;
    loop {
        let page = maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(user_id))
            .filter(maple_device_registration_operations::project_id.eq(project_id))
            .filter(maple_device_registration_operations::id.gt(operation_cursor))
            .order(maple_device_registration_operations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .for_update()
            .load::<MapleDeviceRegistrationOperation>(conn)?;
        if page.is_empty() {
            break;
        }
        for operation in page {
            operation_cursor = operation.id;
            let target = targets
                .get(&operation.lookup_digest)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            let device = match &target.source {
                ResetSourceRow::Live(device) if device.id == operation.maple_device_id => device,
                _ => return Err(DBError::MaplePairingAuthorityCorrupt),
            };
            validate_maple_device_registration_operation(
                enclave_key,
                &operation,
                &MaplePairingAuthorityDeviceSummary::from(device),
                user_id,
                project_id,
            )?;
            if operation.accepted_security_epoch != source_security_epoch {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            reset_at = reset_at.max(operation.accepted_at);
        }
    }

    let target_count: i64 = targets
        .len()
        .try_into()
        .map_err(|_| DBError::MaplePairingAuthorityCapacityExceeded)?;
    let prospective_generations = counts
        .highwater_generations
        .checked_add(target_count)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    let prospective_obligations = counts
        .reset_clear_obligations
        .checked_add(target_count)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    if prospective_generations > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
        || prospective_obligations > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_OBLIGATION_LIMIT
    {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }

    type ResetClearAdmissionKey = (Uuid, u64);
    type ResetClearAdmissionLeaves = BTreeMap<ResetClearAdmissionKey, [u8; 32]>;
    type ResetClearAdmissionLeavesByHost = BTreeMap<Vec<u8>, ResetClearAdmissionLeaves>;
    let mut admission_leaves_by_host = ResetClearAdmissionLeavesByHost::new();

    // A missed-reset successor carries the exact cumulative admission set of
    // its unresolved predecessor. An acknowledged predecessor still remains in
    // the generation chain but its cleared admission set is not re-admitted.
    for target in targets.values() {
        let Some(prior) = target.prior.as_ref().filter(|prior| prior.state == 1) else {
            continue;
        };
        let rows = maple_pairing_reset_clear_admissions::table
            .filter(maple_pairing_reset_clear_admissions::obligation_uuid.eq(prior.uuid))
            .filter(
                maple_pairing_reset_clear_admissions::authority_scope_digest
                    .eq(&authority_scope_digest),
            )
            .order((
                maple_pairing_reset_clear_admissions::pair_id.asc(),
                maple_pairing_reset_clear_admissions::pairing_incarnation.asc(),
            ))
            .limit(MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION + 1)
            .for_update()
            .load::<MaplePairingResetClearAdmission>(conn)?;
        if rows.len() as i64 != i64::from(prior.admission_count) {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let leaves = admission_leaves_by_host
            .entry(target.lookup_digest.clone())
            .or_default();
        for row in rows {
            validate_maple_pairing_reset_clear_admission(
                enclave_key,
                &row,
                &authority_scope_digest,
            )?;
            if row.obligation_uuid != prior.uuid
                || !maple_pairing_authority_mac_matches(&row.lookup_digest, &target.lookup_digest)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let incarnation = pairing_u64_from_i64(row.pairing_incarnation)?;
            let digest: [u8; 32] = row
                .pair_authorization_digest
                .as_slice()
                .try_into()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if leaves.insert((row.pair_id, incarnation), digest).is_some() {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }

    let mut acknowledged_revoked_pairings = BTreeSet::new();
    let mut event_cursor = 0_i64;
    loop {
        let page = maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(user_id))
            .filter(maple_pairing_revocation_events::project_id.eq(project_id))
            .filter(maple_pairing_revocation_events::id.gt(event_cursor))
            .order(maple_pairing_revocation_events::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .for_update()
            .load::<MaplePairingRevocationEvent>(conn)?;
        if page.is_empty() {
            break;
        }
        for event in page {
            validate_maple_pairing_revocation_record(enclave_key, &event)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            event_cursor = event.id;
            if event.acked_at.is_some()
                && !acknowledged_revoked_pairings.insert(event.maple_pairing_id)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
        }
    }
    let mut pairing_cursor = 0_i64;
    loop {
        let page = maple_pairings::table
            .filter(maple_pairings::user_id.eq(user_id))
            .filter(maple_pairings::project_id.eq(project_id))
            .filter(maple_pairings::id.gt(pairing_cursor))
            .order(maple_pairings::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .for_update()
            .load::<MaplePairing>(conn)?;
        if page.is_empty() {
            break;
        }
        for pairing in page {
            validate_maple_pairing_record(enclave_key, &pairing)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            pairing_cursor = pairing.id;
            let state = MaplePairingState::try_from(pairing.state)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let requires_clear = matches!(
                state,
                MaplePairingState::AwaitingHostCommit | MaplePairingState::Active
            ) || (state == MaplePairingState::Revoked
                && !acknowledged_revoked_pairings.contains(&pairing.id));
            if requires_clear {
                let digest: [u8; 32] = pairing
                    .pair_authorization_digest
                    .as_deref()
                    .ok_or(DBError::MaplePairingAuthorityCorrupt)?
                    .try_into()
                    .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
                let host_lookup = lookup_by_live_device_id
                    .get(&pairing.host_maple_device_id)
                    .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                let leaves = admission_leaves_by_host
                    .entry(host_lookup.clone())
                    .or_default();
                let incarnation = pairing_u64_from_i64(pairing.pairing_incarnation)?;
                match leaves.insert((pairing.uuid, incarnation), digest) {
                    None => {}
                    Some(previous) if previous == digest => {}
                    Some(_) => return Err(DBError::MaplePairingAuthorityCorrupt),
                }
                if leaves.len() > MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION as usize
                {
                    return Err(DBError::MaplePairingAuthorityCapacityExceeded);
                }
            }
        }
    }
    let admission_additions: i64 = admission_leaves_by_host
        .values()
        .try_fold(0_i64, |total, leaves| {
            total.checked_add(leaves.len() as i64)
        })
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    if counts
        .reset_clear_admissions
        .checked_add(admission_additions)
        .is_none_or(|value| value > MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_ADMISSION_LIMIT)
    {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    reset_at = reset_at.max(load_maple_pairing_reset_lifecycle_maximum(
        conn, user_id, project_id,
    )?);

    // Materialize and exhaustively validate every target before the first
    // authority/credential mutation. Only bounded admission metadata and the
    // bounded prepared payload for each reset target survive this phase.
    let mut prepared_targets = Vec::with_capacity(targets.len());
    for target in targets.values() {
        let previous = target.prior.as_ref();
        let reset_generation = previous
            .map(|row| row.reset_generation.checked_add(1))
            .unwrap_or(Some(1_i64))
            .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
        let cumulative_reset_count = previous
            .map(|row| row.cumulative_reset_count.checked_add(1))
            .unwrap_or(Some(1_i64))
            .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
        if cumulative_reset_count != reset_generation {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        let target_stream_generation = target
            .highwater
            .revocation_stream_generation
            .checked_add(1)
            .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
        let target_stream_id = Uuid::new_v4();
        let event_id = Uuid::new_v4();
        let admission_leaves = admission_leaves_by_host
            .remove(&target.lookup_digest)
            .unwrap_or_default()
            .into_iter()
            .map(
                |((pair_id, pairing_incarnation), pair_authorization_digest)| {
                    MapleResetClearAdmissionMaterial {
                        pair_id,
                        pairing_incarnation,
                        pair_authorization_digest,
                    }
                },
            )
            .collect::<Vec<_>>();
        let (source, predecessor_claim) = match &target.source {
            ResetSourceRow::Live(device) => {
                let predecessor_claim = if let Some(previous) = previous {
                    let row = maple_pairing_reset_clear_obligations::table
                        .filter(maple_pairing_reset_clear_obligations::id.eq(previous.id))
                        .for_update()
                        .first::<MaplePairingResetClearObligation>(conn)?;
                    validate_maple_pairing_reset_clear_obligation(
                        enclave_key,
                        &row,
                        &authority_scope_digest,
                    )?;
                    let payload = decrypt_maple_reset_clear_payload(
                        enclave_key,
                        &row,
                        MapleResetClearPayloadKind::HostClaim,
                    )?;
                    if !bool::from(
                        sha256_digest(&payload)
                            .as_slice()
                            .ct_eq(previous.host_claim_digest.as_slice()),
                    ) {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                    Some(
                        serde_json::from_slice::<MapleDeviceClaimV1>(&payload)
                            .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
                    )
                } else {
                    None
                };
                (
                    MapleResetClearSource::LiveDevice {
                        registration_id: device.uuid,
                        device_id: device.device_id,
                        installation_id: device.installation_id,
                        revision: device.revision,
                        endpoint_epoch: device.endpoint_epoch,
                        payload_version: device.payload_version,
                        payload_enc: device.payload_enc.clone(),
                        identity_mac: device.identity_mac.clone(),
                        record_mac: device.record_mac.clone(),
                    },
                    predecessor_claim,
                )
            }
            ResetSourceRow::Retained { prior_id } => {
                let prior = maple_pairing_reset_clear_obligations::table
                    .filter(maple_pairing_reset_clear_obligations::id.eq(*prior_id))
                    .for_update()
                    .first::<MaplePairingResetClearObligation>(conn)?;
                validate_maple_pairing_reset_clear_obligation(
                    enclave_key,
                    &prior,
                    &authority_scope_digest,
                )?;
                let payload = decrypt_maple_reset_clear_payload(
                    enclave_key,
                    &prior,
                    MapleResetClearPayloadKind::HostClaim,
                )?;
                if !bool::from(
                    sha256_digest(&payload)
                        .as_slice()
                        .ct_eq(prior.host_claim_digest.as_slice()),
                ) {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                let claim: MapleDeviceClaimV1 = serde_json::from_slice(&payload)
                    .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
                (
                    MapleResetClearSource::RetainedHostClaim {
                        prior_event_id: prior.uuid,
                        payload_version: prior.host_claim_payload_version,
                        payload,
                        payload_digest: prior.host_claim_digest.clone(),
                        identity_mac: prior.host_identity_mac.clone(),
                        prior_target_revocation_stream_id: prior.target_revocation_stream_id,
                        prior_target_revocation_stream_generation: pairing_u64_from_i64(
                            prior.target_revocation_stream_generation,
                        )?,
                        prior_target_security_epoch: pairing_u64_from_i64(
                            prior.target_security_epoch,
                        )?,
                    },
                    Some(claim),
                )
            }
        };
        let context = MapleResetClearUnsignedMaterializationContext {
            account_id: user_id,
            subject_project_id,
            internal_project_id: project_id,
            source,
            event_id,
            reset_id,
            reset_generation: pairing_u64_from_i64(reset_generation)?,
            cumulative_reset_count: pairing_u64_from_i64(cumulative_reset_count)?,
            source_security_epoch: pairing_u64_from_i64(source_security_epoch)?,
            security_epoch: pairing_u64_from_i64(target_security_epoch)?,
            source_revocation_stream_id: target.highwater.revocation_stream_id,
            source_revocation_stream_generation: pairing_u64_from_i64(
                target.highwater.revocation_stream_generation,
            )?,
            source_last_issued_revocation_sequence: pairing_u64_from_i64(
                target.highwater.last_issued_revocation_sequence,
            )?,
            revocation_stream_id: target_stream_id,
            revocation_stream_generation: pairing_u64_from_i64(target_stream_generation)?,
            issuer_sequence: 1,
            previous_event_id: previous.map(|row| row.uuid),
            previous_instruction_material_digest: previous
                .map(|row| row.instruction_digest.as_slice().try_into())
                .transpose()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
            previous_chain_digest: previous
                .map(|row| row.chain_digest.as_slice().try_into())
                .transpose()
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?,
            admission_leaves: admission_leaves.clone(),
            reset_at,
        };
        let mut prepared = build_material(context.clone())
            .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
        validate_reset_clear_material_against_locked_context(
            enclave_key,
            &context,
            &target.identity_mac,
            predecessor_claim.as_ref(),
            &mut prepared,
        )?;
        prepared_targets.push(PreparedResetTarget {
            lookup_digest: target.lookup_digest.clone(),
            identity_mac: target.identity_mac.clone(),
            prior: previous.cloned(),
            highwater: target.highwater.clone(),
            target_stream_generation,
            target_stream_id,
            event_id,
            live_device_id: match &target.source {
                ResetSourceRow::Live(device) => Some(device.id),
                ResetSourceRow::Retained { .. } => None,
            },
            admission_leaves,
            prepared,
        });
    }

    // Retire exact operation IDs only after every pure callback result has
    // independently verified. Tombstones and live operations together remain
    // count-neutral across the later graph deletion.
    let mut operation_cursor = 0_i64;
    loop {
        let page = maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(user_id))
            .filter(maple_device_registration_operations::project_id.eq(project_id))
            .filter(maple_device_registration_operations::id.gt(operation_cursor))
            .order(maple_device_registration_operations::id.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)
            .for_update()
            .load::<MapleDeviceRegistrationOperation>(conn)?;
        if page.is_empty() {
            break;
        }
        for operation in page {
            operation_cursor = operation.id;
            let target = targets
                .get(&operation.lookup_digest)
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
            let device = match &target.source {
                ResetSourceRow::Live(device) if device.id == operation.maple_device_id => device,
                _ => return Err(DBError::MaplePairingAuthorityCorrupt),
            };
            validate_maple_device_registration_operation(
                enclave_key,
                &operation,
                &MaplePairingAuthorityDeviceSummary::from(device),
                user_id,
                project_id,
            )?;
            if operation.accepted_security_epoch != source_security_epoch {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let sync_payload = decrypt_maple_device_sync_payload(enclave_key, &operation, device)?;
            if !bool::from(
                sha256_digest(&sync_payload)
                    .as_slice()
                    .ct_eq(operation.sync_digest.as_slice()),
            ) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let sync: MapleRevocationSyncV1 = serde_json::from_slice(&sync_payload)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            if maple_registration_response_kind(sync.status) != operation.response_kind
                || sync.stream_checkpoint.issuer_key_id != operation.sync_issuer_key_id
                || !maple_pairing_issuer_key_id_is_valid(&sync.stream_checkpoint.issuer_key_id)
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let mut referenced_issuer_key_ids = BTreeSet::new();
            referenced_issuer_key_ids.insert(sync.stream_checkpoint.issuer_key_id.clone());
            if let Some(instruction) = sync.reset_clear_instruction.as_ref() {
                if sync.status != MapleRevocationSyncStatusV1::ResetClearRequired
                    || !maple_pairing_issuer_key_id_is_valid(&instruction.issuer_key_id)
                {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                referenced_issuer_key_ids.insert(instruction.issuer_key_id.clone());
            } else if sync.status == MapleRevocationSyncStatusV1::ResetClearRequired {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let referenced_issuer_key_ids =
                referenced_issuer_key_ids.into_iter().collect::<Vec<_>>();
            if !maple_pairing_issuer_key_ids_are_canonical(&referenced_issuer_key_ids, 4) {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let historical_receipt = MapleDeviceRegistrationReceipt {
                operation_id: operation.operation_id,
                registration_id: device.uuid,
                device_id: device.device_id,
                revision: operation.device_revision,
                accepted_at: operation.accepted_at,
                security_epoch: operation.accepted_security_epoch,
                response_kind: operation.response_kind,
                sync_payload_version: operation.sync_payload_version,
                sync_payload,
            };
            let receipt_plaintext = serde_json::to_vec(&historical_receipt)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1;
            let receipt_digest = sha256_digest(&receipt_plaintext).to_vec();
            let receipt_enc = encrypt_maple_device_registration_tombstone_receipt(
                enclave_key,
                &authority_scope_digest,
                &operation.lookup_digest,
                &operation.operation_lookup_digest,
                source_security_epoch,
                operation.response_kind,
                receipt_version,
                &receipt_digest,
                &receipt_plaintext,
            )?;
            let outcome_digest = maple_device_registration_operation_outcome_digest(&operation);
            let record_mac = maple_device_registration_tombstone_record_mac_for_parts(
                enclave_key,
                &authority_scope_digest,
                &operation.lookup_digest,
                &operation.operation_lookup_digest,
                source_security_epoch,
                &operation.request_mac,
                operation.response_kind,
                &outcome_digest,
                receipt_version,
                &receipt_enc,
                &receipt_digest,
                &referenced_issuer_key_ids,
                operation.accepted_at,
                reset_at,
            )?;
            diesel::insert_into(maple_pairing_registration_operation_tombstones::table)
                .values(NewMaplePairingRegistrationOperationTombstone {
                    authority_scope_digest: authority_scope_digest.clone(),
                    lookup_digest: operation.lookup_digest.clone(),
                    operation_lookup_digest: operation.operation_lookup_digest.clone(),
                    retired_security_epoch: source_security_epoch,
                    request_mac: operation.request_mac.clone(),
                    outcome_kind: operation.response_kind,
                    outcome_digest,
                    receipt_version,
                    receipt_enc,
                    receipt_digest,
                    referenced_issuer_key_ids,
                    accepted_at: operation.accepted_at,
                    record_mac,
                    retired_at: reset_at,
                })
                .execute(conn)
                .map_err(map_maple_device_write_error)?;
        }
    }

    for target in prepared_targets {
        let previous = target.prior.as_ref();
        let reset_generation = previous
            .map(|row| row.reset_generation.checked_add(1))
            .unwrap_or(Some(1_i64))
            .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
        let cumulative_reset_count = previous
            .map(|row| row.cumulative_reset_count.checked_add(1))
            .unwrap_or(Some(1_i64))
            .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
        let admission_leaves = target.admission_leaves;
        let prepared = target.prepared;
        let wire_leaves = admission_leaves
            .iter()
            .map(|leaf| MapleResetClearAdmissionLeafV1 {
                pair_id: leaf.pair_id,
                pairing_incarnation: leaf.pairing_incarnation,
                pair_authorization_digest: leaf.pair_authorization_digest,
            })
            .collect::<Vec<_>>();
        let admission_set_digest =
            reset_clear_admission_set_digest(MAPLE_PAIRING_ARTIFACT_VERSION_V1, &wire_leaves)
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
        let host_claim_payload_enc = encrypt_maple_reset_clear_payload(
            enclave_key,
            MapleResetClearPayloadKind::HostClaim,
            target.event_id,
            &authority_scope_digest,
            &target.lookup_digest,
            &prepared.instruction_material_digest,
            &prepared.chain_digest,
            prepared.host_claim_payload_version,
            None,
            &prepared.host_claim_digest,
            &prepared.host_claim_payload,
        )?;
        let instruction_payload_enc = encrypt_maple_reset_clear_payload(
            enclave_key,
            MapleResetClearPayloadKind::InstructionMaterial,
            target.event_id,
            &authority_scope_digest,
            &target.lookup_digest,
            &prepared.instruction_material_digest,
            &prepared.chain_digest,
            prepared.instruction_payload_version,
            None,
            &prepared.instruction_material_digest,
            &prepared.instruction_payload,
        )?;
        let highwater_record_mac = maple_pairing_revocation_highwater_record_mac(
            enclave_key,
            &target.lookup_digest,
            &authority_scope_digest,
            target.target_stream_id,
            target.target_stream_generation,
            target_security_epoch,
            1,
            2,
        )?;
        diesel::insert_into(maple_pairing_revocation_highwaters::table)
            .values(NewMaplePairingRevocationHighwater {
                lookup_digest: target.lookup_digest.clone(),
                authority_scope_digest: authority_scope_digest.clone(),
                revocation_stream_id: target.target_stream_id,
                revocation_stream_generation: target.target_stream_generation,
                security_epoch: target_security_epoch,
                last_issued_revocation_sequence: 1,
                revision: 2,
                record_mac: highwater_record_mac,
            })
            .execute(conn)?;
        if let Some(live_device_id) = target.live_device_id {
            let current = maple_pairing_host_states::table
                .filter(maple_pairing_host_states::user_id.eq(user_id))
                .filter(maple_pairing_host_states::project_id.eq(project_id))
                .filter(maple_pairing_host_states::host_maple_device_id.eq(live_device_id))
                .for_update()
                .first::<MaplePairingHostState>(conn)?;
            validate_maple_pairing_host_state(enclave_key, &current)?;
            if current.revocation_stream_id != target.highwater.revocation_stream_id
                || current.revocation_stream_generation
                    != target.highwater.revocation_stream_generation
                || current.last_issued_revocation_sequence
                    != target.highwater.last_issued_revocation_sequence
            {
                return Err(DBError::MaplePairingAuthorityCorrupt);
            }
            let revision = 2_i64;
            let record_mac = maple_pairing_host_state_mac(
                enclave_key,
                user_id,
                project_id,
                live_device_id,
                target.target_stream_id,
                target.target_stream_generation,
                1,
                0,
                revision,
            )?;
            let changed = diesel::update(
                maple_pairing_host_states::table
                    .filter(maple_pairing_host_states::id.eq(current.id))
                    .filter(maple_pairing_host_states::revision.eq(current.revision)),
            )
            .set((
                maple_pairing_host_states::revocation_stream_id.eq(target.target_stream_id),
                maple_pairing_host_states::revocation_stream_generation
                    .eq(target.target_stream_generation),
                maple_pairing_host_states::last_issued_revocation_sequence.eq(1_i64),
                maple_pairing_host_states::last_acked_revocation_sequence.eq(0_i64),
                maple_pairing_host_states::revision.eq(revision),
                maple_pairing_host_states::record_mac.eq(record_mac),
            ))
            .execute(conn)?;
            if changed != 1 {
                return Err(DBError::MaplePairingConflict);
            }
        }
        let candidate = MaplePairingResetClearObligation {
            id: 1,
            uuid: target.event_id,
            authority_scope_digest: authority_scope_digest.clone(),
            lookup_digest: target.lookup_digest.clone(),
            host_identity_mac: target.identity_mac.clone(),
            reset_id,
            reset_generation,
            cumulative_reset_count,
            previous_event_id: previous.map(|row| row.uuid),
            previous_instruction_digest: previous.map(|row| row.instruction_digest.clone()),
            previous_chain_digest: previous.map(|row| row.chain_digest.clone()),
            old_revocation_stream_id: target.highwater.revocation_stream_id,
            old_revocation_stream_generation: target.highwater.revocation_stream_generation,
            source_security_epoch,
            source_last_issued_revocation_sequence: target
                .highwater
                .last_issued_revocation_sequence,
            target_revocation_stream_id: target.target_stream_id,
            target_revocation_stream_generation: target.target_stream_generation,
            target_security_epoch,
            target_instruction_sequence: 1,
            clear_scope: 1,
            admission_set_digest: admission_set_digest.to_vec(),
            admission_count: admission_leaves
                .len()
                .try_into()
                .map_err(|_| DBError::MaplePairingAuthorityCapacityExceeded)?,
            host_claim_payload_version: prepared.host_claim_payload_version,
            host_claim_payload_enc,
            host_claim_digest: prepared.host_claim_digest.to_vec(),
            instruction_payload_version: prepared.instruction_payload_version,
            instruction_payload_enc,
            instruction_digest: prepared.instruction_material_digest.to_vec(),
            chain_digest: prepared.chain_digest.to_vec(),
            reset_at,
            signed_instruction_payload_version: None,
            signed_instruction_payload_enc: None,
            signed_instruction_issuer_key_id: None,
            signed_instruction_digest: None,
            sync_payload_version: None,
            sync_payload_enc: None,
            sync_issuer_key_id: None,
            sync_digest: None,
            state: 1,
            revision: 1,
            acked_by_head_event_id: None,
            acked_at: None,
            ack_operation_id: None,
            ack_host_registration_lookup_digest: None,
            ack_request_mac: None,
            ack_receipt_version: None,
            ack_receipt_enc: None,
            ack_receipt_issuer_key_id: None,
            ack_receipt_digest: None,
            record_mac: Vec::new(),
            created_at: reset_at,
            updated_at: reset_at,
        };
        let record_mac = maple_pairing_reset_clear_obligation_record_mac(enclave_key, &candidate)?;
        diesel::insert_into(maple_pairing_reset_clear_obligations::table)
            .values(NewMaplePairingResetClearObligation {
                uuid: target.event_id,
                authority_scope_digest: authority_scope_digest.clone(),
                lookup_digest: target.lookup_digest.clone(),
                host_identity_mac: target.identity_mac.clone(),
                reset_id,
                reset_generation,
                cumulative_reset_count,
                previous_event_id: previous.map(|row| row.uuid),
                previous_instruction_digest: previous.map(|row| row.instruction_digest.clone()),
                previous_chain_digest: previous.map(|row| row.chain_digest.clone()),
                old_revocation_stream_id: target.highwater.revocation_stream_id,
                old_revocation_stream_generation: target.highwater.revocation_stream_generation,
                source_security_epoch,
                source_last_issued_revocation_sequence: target
                    .highwater
                    .last_issued_revocation_sequence,
                target_revocation_stream_id: target.target_stream_id,
                target_revocation_stream_generation: target.target_stream_generation,
                target_security_epoch,
                target_instruction_sequence: 1,
                clear_scope: 1,
                admission_set_digest: admission_set_digest.to_vec(),
                admission_count: admission_leaves
                    .len()
                    .try_into()
                    .map_err(|_| DBError::MaplePairingAuthorityCapacityExceeded)?,
                host_claim_payload_version: candidate.host_claim_payload_version,
                host_claim_payload_enc: candidate.host_claim_payload_enc,
                host_claim_digest: candidate.host_claim_digest,
                instruction_payload_version: candidate.instruction_payload_version,
                instruction_payload_enc: candidate.instruction_payload_enc,
                instruction_digest: candidate.instruction_digest,
                chain_digest: candidate.chain_digest,
                reset_at,
                state: 1,
                revision: 1,
                record_mac,
                created_at: reset_at,
            })
            .execute(conn)?;
        for leaf in admission_leaves {
            let incarnation = i64::try_from(leaf.pairing_incarnation)
                .map_err(|_| DBError::MaplePairingAuthorityCorrupt)?;
            let child_mac = maple_pairing_reset_clear_admission_record_mac_for_parts(
                enclave_key,
                target.event_id,
                &authority_scope_digest,
                &target.lookup_digest,
                leaf.pair_id,
                incarnation,
                &leaf.pair_authorization_digest,
                reset_at,
            )?;
            diesel::insert_into(maple_pairing_reset_clear_admissions::table)
                .values(NewMaplePairingResetClearAdmission {
                    obligation_uuid: target.event_id,
                    authority_scope_digest: authority_scope_digest.clone(),
                    lookup_digest: target.lookup_digest.clone(),
                    pair_id: leaf.pair_id,
                    pairing_incarnation: incarnation,
                    pair_authorization_digest: leaf.pair_authorization_digest.to_vec(),
                    record_mac: child_mac,
                    created_at: reset_at,
                })
                .execute(conn)?;
        }
    }

    // The caller deletes the now-retired live graph and then commits this exact
    // epoch through the authenticated account head/cascade in the same DB
    // transaction. Returning the value makes a forgotten durable head advance
    // impossible to hide behind a local assignment.
    Ok(PersistedMaplePairingResetClear {
        target_security_epoch,
        reset_at,
    })
}

#[cfg(test)]
fn ensure_maple_pairing_revocation_reset_capacity_for_counts(
    current_generations: i64,
    rotating_installations: i64,
) -> Result<(), DBError> {
    let prospective_generations = current_generations
        .checked_add(rotating_installations)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    if current_generations < 0
        || rotating_installations < 0
        || prospective_generations > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
    {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    Ok(())
}

fn ensure_maple_pairing_revocation_registration_capacity_for_counts(
    current_groups: i64,
    current_generations: i64,
    retained_group_exists: bool,
) -> Result<(), DBError> {
    if current_groups < 0
        || current_generations < current_groups
        || current_groups > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT
        || current_generations > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
    {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    if retained_group_exists {
        return Ok(());
    }
    let prospective_groups = current_groups
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    let prospective_generations = current_generations
        .checked_add(1)
        .ok_or(DBError::MaplePairingAuthorityCapacityExceeded)?;
    if prospective_groups > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT
        || prospective_generations > MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT
    {
        return Err(DBError::MaplePairingAuthorityCapacityExceeded);
    }
    Ok(())
}

fn ensure_maple_pairing_revocation_registration_capacity(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
    installation_id: Uuid,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_revocation_highwaters;

    let authority_scope_digest =
        maple_pairing_authority_scope_digest(enclave_key, user_id, project_id)?;
    let lookup_digest = maple_pairing_revocation_highwater_lookup_digest(
        enclave_key,
        user_id,
        project_id,
        installation_id,
    )?;
    let retained_group_exists = maple_pairing_revocation_highwaters::table
        .filter(
            maple_pairing_revocation_highwaters::authority_scope_digest.eq(&authority_scope_digest),
        )
        .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(lookup_digest))
        .count()
        .get_result::<i64>(conn)?
        > 0;
    let counts = count_maple_pairing_authority_account_rows(
        conn,
        &authority_scope_digest,
        user_id,
        project_id,
    )?;
    if let Err(error) = ensure_maple_pairing_revocation_registration_capacity_for_counts(
        counts.highwater_groups,
        counts.highwater_generations,
        retained_group_exists,
    ) {
        tracing::warn!(
            event = "maple_pairing_authority_registration_capacity_exceeded",
            current_group_count = counts.highwater_groups,
            current_generation_count = counts.highwater_generations,
            "Maple pairing authority registration would exceed a V1 highwater lifetime bound"
        );
        return Err(error);
    }
    Ok(())
}

fn verify_maple_pairing_authority_deletion_safe(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    head: &MaplePairingAuthorityAccountHead,
) -> Result<(), DBError> {
    use crate::models::schema::{
        maple_pairing_host_states, maple_pairing_reset_clear_obligations, maple_pairings,
    };
    use diesel::OptionalExtension;

    // The full authenticated account inventory proves both completeness and
    // row integrity before the terminal predicate is evaluated. In particular,
    // it proves exact pair/event/operation linkage and the contiguous current
    // revocation stream/ACK prefix.
    verify_maple_pairing_authority_account(conn, enclave_key, head)?;
    let authorization = MaplePairingAuthorization {
        user_id: head.user_id,
        project_id: head.project_id,
        auth_credential_kind: String::new(),
        auth_binding: [0_u8; 32],
        enclave_key: enclave_key.to_vec(),
    };
    let (_, expired_any) = expire_pending_pairings(conn, &authorization)?;
    if expired_any {
        // Expiry changes authenticated pair rows. Advance the complete
        // cascading root before evaluating/consuming the terminal account so
        // every intermediate state remains a valid authenticated snapshot.
        commit_maple_pairing_authority_account_mutation(
            conn,
            enclave_key,
            head.user_id,
            head.project_id,
        )?;
        let current = crate::models::schema::maple_pairing_authority_account_heads::table
            .filter(
                crate::models::schema::maple_pairing_authority_account_heads::user_id
                    .eq(head.user_id),
            )
            .filter(
                crate::models::schema::maple_pairing_authority_account_heads::project_id
                    .eq(head.project_id),
            )
            .for_update()
            .first::<MaplePairingAuthorityAccountHead>(conn)?;
        verify_maple_pairing_authority_account(conn, enclave_key, &current)?;
    }
    let blocking_pair = maple_pairings::table
        .filter(maple_pairings::user_id.eq(head.user_id))
        .filter(maple_pairings::project_id.eq(head.project_id))
        .filter(maple_pairings::state.eq_any([
            MaplePairingState::Pending.as_db(),
            MaplePairingState::AwaitingHostCommit.as_db(),
            MaplePairingState::Active.as_db(),
        ]))
        .select(maple_pairings::id)
        .for_share()
        .first::<i64>(conn)
        .optional()?;
    let unacked_host = maple_pairing_host_states::table
        .filter(maple_pairing_host_states::user_id.eq(head.user_id))
        .filter(maple_pairing_host_states::project_id.eq(head.project_id))
        .filter(
            maple_pairing_host_states::last_acked_revocation_sequence
                .ne(maple_pairing_host_states::last_issued_revocation_sequence),
        )
        .select(maple_pairing_host_states::id)
        .for_share()
        .first::<i64>(conn)
        .optional()?;
    let pending_reset = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&head.authority_scope_digest),
        )
        .filter(maple_pairing_reset_clear_obligations::state.eq(1_i16))
        .select(maple_pairing_reset_clear_obligations::id)
        .for_share()
        .first::<i64>(conn)
        .optional()?;
    if blocking_pair.is_some() || unacked_host.is_some() || pending_reset.is_some() {
        return Err(DBError::MaplePairingAuthorityDeletionBlocked);
    }
    Ok(())
}

fn delete_maple_pairing_authority_account_for_final_parent_deletion(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<(), DBError> {
    prove_maple_pairing_authority_account_deletion_safe(conn, enclave_key, user_id, project_id)?;
    consume_maple_pairing_authority_account_after_clean_proof(
        conn,
        enclave_key,
        user_id,
        project_id,
    )
}

fn prove_maple_pairing_authority_account_deletion_safe(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::maple_pairing_authority_account_heads;

    let head = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .for_update()
        .first::<MaplePairingAuthorityAccountHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    verify_maple_pairing_authority_deletion_safe(conn, enclave_key, &head)
}

fn consume_maple_pairing_authority_account_after_clean_proof(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    user_id: Uuid,
    project_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::{
        maple_pairing_authority_account_heads, maple_pairing_installation_retirements,
        maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_admissions,
        maple_pairing_reset_clear_obligations, maple_pairing_revocation_highwaters,
    };

    let head = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .for_update()
        .first::<MaplePairingAuthorityAccountHead>(conn)
        .map_err(|error| match error {
            diesel::result::Error::NotFound => DBError::MaplePairingAuthorityCorrupt,
            other => DBError::QueryError(other),
        })?;
    // The destructive phase is deliberately separate from the proof phase.
    // Revalidate the authenticated head, but do not run expiry or cascade an
    // ancestor after another account head may already have been consumed.
    validate_maple_pairing_authority_account_head(enclave_key, &head)?;
    delete_maple_pairing_state_for_user(conn, user_id)?;
    diesel::delete(
        maple_pairing_reset_clear_admissions::table.filter(
            maple_pairing_reset_clear_admissions::authority_scope_digest
                .eq(&head.authority_scope_digest),
        ),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_reset_clear_obligations::table.filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&head.authority_scope_digest),
        ),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_registration_operation_tombstones::table.filter(
            maple_pairing_registration_operation_tombstones::authority_scope_digest
                .eq(&head.authority_scope_digest),
        ),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_installation_retirements::table.filter(
            maple_pairing_installation_retirements::authority_scope_digest
                .eq(&head.authority_scope_digest),
        ),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_revocation_highwaters::table.filter(
            maple_pairing_revocation_highwaters::authority_scope_digest
                .eq(&head.authority_scope_digest),
        ),
    )
    .execute(conn)?;
    let removed = diesel::delete(
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
            .filter(maple_pairing_authority_account_heads::project_id.eq(project_id)),
    )
    .execute(conn)?;
    if removed != 1 {
        return Err(DBError::MaplePairingAuthorityCorrupt);
    }
    Ok(())
}

fn prove_maple_pairing_authority_accounts_for_project(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::users;

    let mut cursor = Uuid::nil();
    loop {
        let user_ids = users::table
            .filter(users::project_id.eq(project_id))
            .filter(users::uuid.gt(cursor))
            .order(users::uuid.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .select(users::uuid)
            .for_update()
            .load::<Uuid>(conn)?;
        if user_ids.is_empty() {
            break;
        }
        for user_id in user_ids {
            prove_maple_pairing_authority_account_deletion_safe(
                conn,
                enclave_key,
                user_id,
                project_id,
            )?;
            cursor = user_id;
        }
    }
    Ok(())
}

fn consume_maple_pairing_authority_accounts_for_project_after_clean_proof(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
) -> Result<(), DBError> {
    use crate::models::schema::users;

    let mut cursor = Uuid::nil();
    loop {
        let user_ids = users::table
            .filter(users::project_id.eq(project_id))
            .filter(users::uuid.gt(cursor))
            .order(users::uuid.asc())
            .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
            .select(users::uuid)
            .for_update()
            .load::<Uuid>(conn)?;
        if user_ids.is_empty() {
            break;
        }
        for user_id in user_ids {
            consume_maple_pairing_authority_account_after_clean_proof(
                conn,
                enclave_key,
                user_id,
                project_id,
            )?;
            cursor = user_id;
        }
    }
    Ok(())
}

fn consume_maple_pairing_authority_accounts_for_project(
    conn: &mut PgConnection,
    enclave_key: &[u8],
    project_id: i32,
) -> Result<(), DBError> {
    // Keep every account head and parent present until all accounts have
    // materialized due expiry and passed the authenticated terminal proof.
    prove_maple_pairing_authority_accounts_for_project(conn, enclave_key, project_id)?;
    consume_maple_pairing_authority_accounts_for_project_after_clean_proof(
        conn,
        enclave_key,
        project_id,
    )
}

/// Remove raw Maple pairing/device rows in dependency order. The caller owns
/// retention semantics: password reset keeps the rotated highwaters, while a
/// verified-clean final parent deletion consumes highwaters and the account
/// head after this helper returns.
fn delete_maple_pairing_state_for_user(
    conn: &mut PgConnection,
    user_id: Uuid,
) -> Result<(), DBError> {
    use crate::models::schema::{
        maple_device_registration_operations, maple_devices, maple_pairing_host_states,
        maple_pairing_lineages, maple_pairing_operations, maple_pairing_revocation_events,
        maple_pairings,
    };

    diesel::delete(
        maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(user_id)),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_operations::table.filter(maple_pairing_operations::user_id.eq(user_id)),
    )
    .execute(conn)?;
    diesel::delete(
        maple_pairing_host_states::table.filter(maple_pairing_host_states::user_id.eq(user_id)),
    )
    .execute(conn)?;
    diesel::delete(maple_pairings::table.filter(maple_pairings::user_id.eq(user_id)))
        .execute(conn)?;
    diesel::delete(
        maple_pairing_lineages::table.filter(maple_pairing_lineages::user_id.eq(user_id)),
    )
    .execute(conn)?;
    diesel::delete(
        maple_device_registration_operations::table
            .filter(maple_device_registration_operations::user_id.eq(user_id)),
    )
    .execute(conn)?;
    diesel::delete(maple_devices::table.filter(maple_devices::user_id.eq(user_id)))
        .execute(conn)?;
    Ok(())
}

#[cfg(test)]
mod maple_pairing_db_unit_tests {
    use super::*;

    #[test]
    fn account_head_epoch_projection_requires_a_provisional_mac() {
        let enclave_key = [0x53; 32];
        let user_id = Uuid::from_u128(29);
        let project_id = 17;
        let org_id = 23;
        let created_at = DateTime::from_timestamp(1_700_000_000, 0).expect("valid fixed timestamp");
        let authority_scope_digest =
            maple_pairing_authority_scope_digest(&enclave_key, user_id, project_id)
                .expect("authority scope digest should succeed");
        let authority_inventory_digest = empty_maple_pairing_authority_account_inventory(
            user_id,
            project_id,
            org_id,
            &authority_scope_digest,
        );
        let mut head = MaplePairingAuthorityAccountHead {
            user_id,
            project_id,
            org_id,
            security_epoch: 1,
            authority_scope_digest,
            authority_inventory_digest,
            authority_row_count: 0,
            device_count: 0,
            device_operation_count: 0,
            lineage_count: 0,
            pairing_count: 0,
            pairing_operation_count: 0,
            host_state_count: 0,
            revocation_event_count: 0,
            highwater_installation_group_count: 0,
            highwater_generation_count: 0,
            registration_operation_tombstone_count: 0,
            installation_retirement_count: 0,
            reset_clear_obligation_count: 0,
            reset_clear_admission_count: 0,
            revision: 1,
            record_mac: Vec::new(),
            created_at,
            updated_at: created_at,
        };
        head.record_mac = maple_pairing_authority_account_head_mac(&enclave_key, &head)
            .expect("initial account-head MAC should succeed");
        validate_maple_pairing_authority_account_head(&enclave_key, &head)
            .expect("initial account head should validate");

        let stale_record_mac = head.record_mac.clone();
        head.security_epoch = head
            .security_epoch
            .checked_add(1)
            .expect("security epoch should advance");
        assert!(matches!(
            validate_maple_pairing_authority_account_head(&enclave_key, &head),
            Err(DBError::MaplePairingAuthorityCorrupt)
        ));

        head.record_mac = maple_pairing_authority_account_head_mac(&enclave_key, &head)
            .expect("provisional account-head MAC should succeed");
        assert_ne!(head.record_mac, stale_record_mac);
        validate_maple_pairing_authority_account_head(&enclave_key, &head)
            .expect("epoch-projected account head should validate after provisional re-MAC");
    }

    fn timestamp_test_pairing(state: MaplePairingState, revision: i64) -> MaplePairing {
        let created_at = DateTime::from_timestamp_millis(2_000_000_000_000).unwrap();
        let approved_at = created_at + chrono::Duration::seconds(1);
        let activated_at = approved_at + chrono::Duration::seconds(1);
        let revoked_at = activated_at + chrono::Duration::seconds(1);
        MaplePairing {
            id: 1,
            uuid: Uuid::new_v4(),
            pairing_request_id: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            project_id: 1,
            lineage_id: 1,
            controller_maple_device_id: 1,
            host_maple_device_id: 2,
            direction: 1,
            pairing_incarnation: 1,
            state: state.as_db(),
            revision,
            request_nonce_mac: vec![1; 32],
            revocation_stream_id: (!matches!(
                state,
                MaplePairingState::Pending | MaplePairingState::Expired
            ))
            .then(Uuid::new_v4),
            revocation_stream_generation: (!matches!(
                state,
                MaplePairingState::Pending | MaplePairingState::Expired
            ))
            .then_some(1),
            pair_authorization_digest: None,
            ticket_issuer_key_id: "test-issuer".to_string(),
            authorization_issuer_key_id: None,
            revocation_issuer_key_id: None,
            payload_version: 1,
            payload_enc: vec![],
            record_mac: vec![0; 32],
            created_at,
            expires_at: created_at + chrono::Duration::minutes(10),
            approved_at: (!matches!(
                state,
                MaplePairingState::Pending | MaplePairingState::Expired
            ))
            .then_some(approved_at),
            activated_at: matches!(state, MaplePairingState::Active)
                .then_some(activated_at)
                .or_else(|| {
                    (state == MaplePairingState::Revoked && revision == 4).then_some(activated_at)
                }),
            revoked_at: (state == MaplePairingState::Revoked).then_some(revoked_at),
            updated_at: revoked_at,
        }
    }

    #[test]
    fn pending_expiry_and_approval_share_the_exact_clock_skew_boundary() {
        assert_eq!(MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS, 30_000);
        let now = DateTime::from_timestamp_millis(2_000_000_000_000).unwrap();
        let exact_expiry_cutoff = now - chrono::Duration::milliseconds(30_000);
        assert!(maple_pairing_pending_is_expired(exact_expiry_cutoff, now));
        assert!(!maple_pairing_pending_is_expired(
            exact_expiry_cutoff + chrono::Duration::microseconds(1),
            now
        ));

        let expires_at = now;
        assert!(!maple_pairing_approval_is_timely(
            expires_at,
            expires_at + chrono::Duration::milliseconds(30_000)
        ));
        assert!(maple_pairing_approval_is_timely(
            expires_at,
            expires_at + chrono::Duration::milliseconds(30_000) - chrono::Duration::microseconds(1)
        ));
        assert!(!maple_pairing_approval_is_timely(
            expires_at,
            expires_at + chrono::Duration::milliseconds(30_000) + chrono::Duration::microseconds(1)
        ));

        assert!(maple_pairing_time_is_near_trusted_now(
            now - chrono::Duration::milliseconds(30_000),
            now
        ));
        assert!(maple_pairing_time_is_near_trusted_now(
            now + chrono::Duration::milliseconds(30_000),
            now
        ));
        assert!(!maple_pairing_time_is_near_trusted_now(
            now - chrono::Duration::milliseconds(30_001),
            now
        ));
        assert!(!maple_pairing_time_is_near_trusted_now(
            now + chrono::Duration::milliseconds(30_001),
            now
        ));
    }

    #[test]
    fn revocation_highwater_capacity_checks_exact_v1_boundaries() {
        assert!(
            ensure_maple_pairing_revocation_registration_capacity_for_counts(
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT - 1,
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT - 1,
                false,
            )
            .is_ok()
        );
        assert!(matches!(
            ensure_maple_pairing_revocation_registration_capacity_for_counts(
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT,
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT,
                false,
            ),
            Err(DBError::MaplePairingAuthorityCapacityExceeded)
        ));
        assert!(matches!(
            ensure_maple_pairing_revocation_registration_capacity_for_counts(
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT - 1,
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT,
                false,
            ),
            Err(DBError::MaplePairingAuthorityCapacityExceeded)
        ));
        assert!(
            ensure_maple_pairing_revocation_registration_capacity_for_counts(
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GROUP_LIMIT,
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT,
                true,
            )
            .is_ok(),
            "re-registering an already retained installation allocates no new highwater row"
        );

        assert!(ensure_maple_pairing_revocation_reset_capacity_for_counts(
            MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT - 1,
            1,
        )
        .is_ok());
        assert!(ensure_maple_pairing_revocation_reset_capacity_for_counts(
            MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT - 2,
            2,
        )
        .is_ok());
        assert!(matches!(
            ensure_maple_pairing_revocation_reset_capacity_for_counts(
                MAPLE_PAIRING_AUTHORITY_HIGHWATER_GENERATION_LIMIT,
                1,
            ),
            Err(DBError::MaplePairingAuthorityCapacityExceeded)
        ));
        assert!(matches!(
            ensure_maple_pairing_revocation_reset_capacity_for_counts(i64::MAX, 1),
            Err(DBError::MaplePairingAuthorityCapacityExceeded)
        ));
    }

    #[test]
    fn pairing_lifecycle_timestamps_must_be_monotonic() {
        for (state, revision) in [
            (MaplePairingState::Pending, 1),
            (MaplePairingState::AwaitingHostCommit, 2),
            (MaplePairingState::Active, 3),
            (MaplePairingState::Expired, 2),
            (MaplePairingState::Revoked, 3),
            (MaplePairingState::Revoked, 4),
        ] {
            let row = timestamp_test_pairing(state, revision);
            assert!(
                maple_pairing_lifecycle_timestamps_are_ordered(&row),
                "valid {state:?} lifecycle should be ordered"
            );
        }

        let mut awaiting = timestamp_test_pairing(MaplePairingState::AwaitingHostCommit, 2);
        awaiting.approved_at = Some(awaiting.created_at - chrono::Duration::microseconds(1));
        assert!(!maple_pairing_lifecycle_timestamps_are_ordered(&awaiting));

        let mut active = timestamp_test_pairing(MaplePairingState::Active, 3);
        active.activated_at = active
            .approved_at
            .map(|approved_at| approved_at - chrono::Duration::microseconds(1));
        assert!(!maple_pairing_lifecycle_timestamps_are_ordered(&active));

        let mut revoked = timestamp_test_pairing(MaplePairingState::Revoked, 4);
        revoked.revoked_at = revoked
            .activated_at
            .map(|activated_at| activated_at - chrono::Duration::microseconds(1));
        assert!(!maple_pairing_lifecycle_timestamps_are_ordered(&revoked));
    }

    fn serialization_failure() -> diesel::result::Error {
        diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::SerializationFailure,
            Box::new("test serialization failure".to_string()),
        )
    }

    #[test]
    fn authority_busy_classification_does_not_depend_on_database_error_text() {
        assert_eq!(MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT, Duration::from_secs(5));
        assert_eq!(
            MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL,
            Duration::from_millis(10)
        );
        let unrelated_query_cancellation =
            DBError::QueryError(diesel::result::Error::DatabaseError(
                diesel::result::DatabaseErrorKind::UnableToSendCommand,
                Box::new("cancelación localizada no clasificada".to_string()),
            ));
        assert!(matches!(
            finish_maple_pairing_authority_transaction::<()>(
                Err(unrelated_query_cancellation),
                MaplePairingAuthorityTransactionClass::ReadOnly,
            ),
            Err(DBError::QueryError(_))
        ));
    }

    #[test]
    fn serialization_failure_detection_walks_typed_model_error_sources() {
        let wrapped_errors = [
            DBError::UserError(UserError::DatabaseError(serialization_failure())),
            DBError::OrgError(OrgError::DatabaseError(serialization_failure())),
            DBError::OrgProjectError(OrgProjectError::DatabaseError(serialization_failure())),
            DBError::UserSeedWrappingError(UserSeedWrappingError::DatabaseError(
                serialization_failure(),
            )),
            DBError::AppDataMigrationError(AppDataMigrationError::DatabaseError(
                serialization_failure(),
            )),
        ];
        for wrapped in &wrapped_errors {
            assert!(is_serialization_failure(wrapped));
        }
        assert!(!is_serialization_failure(&DBError::MaplePairingConflict));
    }

    #[test]
    fn serialization_failure_detection_inspects_commit_and_rollback_errors() {
        let commit_failed = DBError::QueryError(diesel::result::Error::RollbackErrorOnCommit {
            rollback_error: Box::new(diesel::result::Error::NotFound),
            commit_error: Box::new(serialization_failure()),
        });
        assert!(is_serialization_failure(&commit_failed));

        let rollback_failed = DBError::UserError(UserError::DatabaseError(
            diesel::result::Error::RollbackErrorOnCommit {
                rollback_error: Box::new(serialization_failure()),
                commit_error: Box::new(diesel::result::Error::NotFound),
            },
        ));
        assert!(is_serialization_failure(&rollback_failed));
    }

    #[test]
    fn only_retry_safe_authority_transactions_map_serialization_to_busy() {
        let read = finish_maple_pairing_authority_transaction::<()>(
            Err(DBError::QueryError(serialization_failure())),
            MaplePairingAuthorityTransactionClass::ReadOnly,
        );
        assert!(matches!(read, Err(DBError::MaplePairingAuthorityBusy)));

        let replay_safe = finish_maple_pairing_authority_transaction::<()>(
            Err(DBError::UserError(UserError::DatabaseError(
                serialization_failure(),
            ))),
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
        );
        assert!(matches!(
            replay_safe,
            Err(DBError::MaplePairingAuthorityBusy)
        ));

        let non_replayable = finish_maple_pairing_authority_transaction::<()>(
            Err(DBError::OrgError(OrgError::DatabaseError(
                serialization_failure(),
            ))),
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
        );
        assert!(matches!(&non_replayable, Err(DBError::OrgError(_))));
        assert!(is_serialization_failure(
            non_replayable.as_ref().unwrap_err()
        ));
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DBError {
    #[error("Database connection error")]
    ConnectionError,
    #[error("Database query error: {0}")]
    QueryError(#[from] diesel::result::Error),
    #[error("User error: {0}")]
    UserError(#[from] UserError),
    #[error("User not found")]
    UserNotFound,
    #[error("Enclave secret error: {0}")]
    EnclaveSecretError(#[from] EnclaveSecretError),
    #[error("Email verification error: {0}")]
    EmailVerificationError(#[from] EmailVerificationError),
    #[error("Email verification not found")]
    EmailVerificationNotFound,
    #[error("Password reset error: {0}")]
    PasswordResetError(#[from] PasswordResetError),
    #[error("Password reset request not found")]
    PasswordResetRequestNotFound,
    #[error("Account deletion error: {0}")]
    AccountDeletionError(#[from] AccountDeletionError),
    #[error("Account deletion request not found")]
    AccountDeletionRequestNotFound,
    #[error("Encryption error: {0}")]
    EncryptionError(#[from] crate::encrypt::EncryptError),
    #[error("OAuth error: {0}")]
    OAuthError(#[from] OAuthError),
    #[error("Token usage error: {0}")]
    TokenUsageError(#[from] TokenUsageError),
    #[error("User API key error: {0}")]
    UserApiKeyError(#[from] UserApiKeyError),
    #[error("Org error: {0}")]
    OrgError(#[from] OrgError),
    #[error("Org not found")]
    OrgNotFound,
    #[error("Org project error: {0}")]
    OrgProjectError(#[from] OrgProjectError),
    #[error("Org project not found")]
    OrgProjectNotFound,
    #[error("Org project secret error: {0}")]
    OrgProjectSecretError(#[from] OrgProjectSecretError),
    #[error("Org project secret not found")]
    OrgProjectSecretNotFound,
    #[error("Invite code error: {0}")]
    InviteCodeError(#[from] InviteCodeError),
    #[error("Invite code not found")]
    InviteCodeNotFound,
    #[error("Platform invite code error: {0}")]
    PlatformInviteCodeError(#[from] PlatformInviteCodeError),
    #[error("Platform invite code not found")]
    PlatformInviteCodeNotFound,
    #[error("Invalid invite code")]
    InvalidInviteCode,
    #[error("Platform user error: {0}")]
    PlatformUserError(#[from] PlatformUserError),
    #[error("Platform user not found")]
    PlatformUserNotFound,
    #[error("Platform email verification error: {0}")]
    PlatformEmailVerificationError(#[from] PlatformEmailVerificationError),
    #[error("Platform email verification not found")]
    PlatformEmailVerificationNotFound,
    #[error("Platform password reset error: {0}")]
    PlatformPasswordResetError(#[from] PlatformPasswordResetError),
    #[error("Platform password reset request not found")]
    PlatformPasswordResetRequestNotFound,
    #[error("Org membership error: {0}")]
    OrgMembershipError(#[from] OrgMembershipError),
    #[error("Org membership not found")]
    OrgMembershipNotFound,
    #[error("Project setting error: {0}")]
    ProjectSettingError(#[from] ProjectSettingError),
    #[error("Project setting not found")]
    ProjectSettingNotFound,
    #[error("Responses API error: {0}")]
    ResponsesError(#[from] crate::models::responses::ResponsesError),
    #[error("User seed wrapping error: {0}")]
    UserSeedWrappingError(#[from] UserSeedWrappingError),
    #[error("App data migration error: {0}")]
    AppDataMigrationError(#[from] AppDataMigrationError),
    #[error("Credential state changed during update")]
    StaleCredentialState,
    #[error("Maple device registration conflicts with an existing operation or device")]
    MapleDeviceRegistrationConflict,
    #[error("Maple device limit reached for this account and project")]
    MapleDeviceLimitExceeded,
    #[error("Maple device registration operation limit reached")]
    MapleDeviceOperationLimitExceeded,
    #[error("Maple device security epoch is stale")]
    MapleDeviceSecurityEpochStale,
    #[error("Maple installation enrollment is retired")]
    MapleInstallationRetired,
    #[error("Maple remote access must be cleared on the host before this operation can continue")]
    MaplePairingResetClearRequired,
    #[error("Maple pairing conflicts with current state or a prior operation")]
    MaplePairingConflict,
    #[error("Maple pairing not found")]
    MaplePairingNotFound,
    #[error("Maple pairing limit reached")]
    MaplePairingLimitExceeded,
    #[error("Maple pairing operation limit reached")]
    MaplePairingOperationLimitExceeded,
    #[error("Maple pairing persisted state failed authentication")]
    MaplePairingCorrupt,
    #[error("Maple pairing authority is temporarily busy")]
    MaplePairingAuthorityBusy,
    #[error("Maple pairing authority hierarchy failed authentication")]
    MaplePairingAuthorityCorrupt,
    #[error("Maple pairing authority prevents destructive parent deletion")]
    MaplePairingAuthorityDeletionBlocked,
    #[error("Maple pairing authority lifetime capacity is exhausted")]
    MaplePairingAuthorityCapacityExceeded,
    #[error("Maple pairing artifact materialization failed")]
    MaplePairingMaterializationFailed,
    #[error("Maple pairing issuer configuration conflicts with durable key identity")]
    MaplePairingIssuerConfigurationConflict,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaplePairingAuthorityTransactionClass {
    ReadOnly,
    ReplaySafeMutation,
    NonReplayableMutation,
}

impl MaplePairingAuthorityTransactionClass {
    fn is_retry_safe(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReplaySafeMutation)
    }

    fn telemetry_label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::ReplaySafeMutation => "replay_safe_mutation",
            Self::NonReplayableMutation => "non_replayable_mutation",
        }
    }
}

fn finish_maple_pairing_authority_transaction<T>(
    result: Result<T, DBError>,
    class: MaplePairingAuthorityTransactionClass,
) -> Result<T, DBError> {
    if result
        .as_ref()
        .err()
        .is_some_and(|error| is_serialization_failure(error))
    {
        trace_maple_pairing_authority_serialization_failure(class);
        if class.is_retry_safe() {
            return Err(DBError::MaplePairingAuthorityBusy);
        }
    }
    result
}

/// Detect PostgreSQL serialization aborts through typed error wrappers.
///
/// Authority transactions intentionally use several model-layer errors in
/// addition to `DBError::QueryError`. Walking `Error::source` keeps the
/// classification exact without depending on display text or accidentally
/// missing a newly nested Diesel error.
pub(crate) fn is_serialization_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(source) = current {
        if source
            .downcast_ref::<diesel::result::Error>()
            .is_some_and(diesel_error_is_serialization_failure)
        {
            return true;
        }
        current = source.source();
    }
    false
}

fn diesel_error_is_serialization_failure(error: &diesel::result::Error) -> bool {
    match error {
        diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::SerializationFailure,
            _,
        ) => true,
        diesel::result::Error::RollbackErrorOnCommit {
            rollback_error,
            commit_error,
        } => {
            diesel_error_is_serialization_failure(rollback_error)
                || diesel_error_is_serialization_failure(commit_error)
        }
        _ => false,
    }
}

fn trace_maple_pairing_authority_serialization_failure(
    class: MaplePairingAuthorityTransactionClass,
) {
    tracing::warn!(
        event = "maple_pairing_authority_serialization_failure",
        transaction_class = class.telemetry_label(),
        retry_safe = class.is_retry_safe(),
        "Maple pairing authority serializable transaction aborted"
    );
}

/// Observe a commit-time abort around a non-idempotent outer transaction.
/// The original typed error is retained so callers cannot accidentally expose
/// this path as retry-safe merely because the transaction used authority
/// internals. Signup/OAuth user creation use this boundary.
pub(crate) fn finish_nonreplayable_maple_pairing_authority_transaction<T, E>(
    result: Result<T, E>,
) -> Result<T, E>
where
    E: std::error::Error + 'static,
{
    if result
        .as_ref()
        .err()
        .is_some_and(|error| is_serialization_failure(error))
    {
        trace_maple_pairing_authority_serialization_failure(
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
        );
    }
    result
}

fn run_maple_pairing_authority_transaction<T, F>(
    conn: &mut PgConnection,
    class: MaplePairingAuthorityTransactionClass,
    callback: F,
) -> Result<T, DBError>
where
    F: FnOnce(&mut PgConnection) -> Result<T, DBError>,
{
    finish_maple_pairing_authority_transaction(conn.transaction::<T, DBError, _>(callback), class)
}

#[allow(dead_code)]
pub trait DBConnection {
    fn bootstrap_or_audit_maple_pairing_authority(
        &self,
        enclave_key: &[u8],
        issuer_keyset: Option<&MaplePairingIssuerKeySetV1>,
    ) -> Result<(), DBError>;
    fn configured_maple_pairing_issuer_key_inventory_digest(&self) -> Result<Vec<u8>, DBError>;
    fn create_user(&self, new_user: NewUser, enclave_key: &[u8]) -> Result<User, DBError>;
    fn get_user_by_uuid(&self, uuid: Uuid) -> Result<User, DBError>;
    fn get_user_by_email(&self, email: String, project_id: i32) -> Result<User, DBError>;
    fn create_user_seed_wrapping(
        &self,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<UserSeedWrapping, DBError>;
    fn upsert_user_seed_wrapping(
        &self,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<UserSeedWrapping, DBError>;
    fn get_user_seed_wrappings_for_user_and_kind(
        &self,
        user_id: Uuid,
        credential_kind: &str,
    ) -> Result<Vec<UserSeedWrapping>, DBError>;
    fn get_user_seed_wrapping_by_credential(
        &self,
        user_id: Uuid,
        credential_kind: &str,
        credential_lookup_hash: &[u8],
        wrapping_version: i16,
    ) -> Result<Option<UserSeedWrapping>, DBError>;
    fn delete_user_seed_wrappings_for_user(&self, user_id: Uuid) -> Result<usize, DBError>;
    fn delete_user_seed_wrappings_for_user_and_kind(
        &self,
        user_id: Uuid,
        credential_kind: &str,
    ) -> Result<usize, DBError>;
    fn get_app_data_migration(&self, name: &str) -> Result<Option<AppDataMigration>, DBError>;
    fn app_data_migration_exists(&self, name: &str) -> Result<bool, DBError>;
    fn create_app_data_migration(
        &self,
        new_migration: NewAppDataMigration,
    ) -> Result<AppDataMigration, DBError>;
    fn get_pool(&self) -> &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>;
    fn create_enclave_secret(&self, new_secret: NewEnclaveSecret)
        -> Result<EnclaveSecret, DBError>;
    fn get_enclave_secret_by_id(&self, id: i32) -> Result<Option<EnclaveSecret>, DBError>;
    fn get_enclave_secret_by_key(&self, key: &str) -> Result<Option<EnclaveSecret>, DBError>;
    fn get_all_enclave_secrets(&self) -> Result<Vec<EnclaveSecret>, DBError>;
    fn update_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError>;
    fn delete_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError>;
    fn create_email_verification(
        &self,
        new_verification: NewEmailVerification,
    ) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_id(&self, id: i32) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_user_id(
        &self,
        user_id: Uuid,
    ) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_code(&self, code: Uuid) -> Result<EmailVerification, DBError>;
    fn update_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError>;
    fn delete_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError>;
    fn verify_email(&self, verification: &mut EmailVerification) -> Result<(), DBError>;
    fn create_password_reset_request(
        &self,
        new_request: NewPasswordResetRequest,
    ) -> Result<PasswordResetRequest, DBError>;
    fn get_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PasswordResetRequest>, DBError>;
    fn update_user_password_and_seed_wrap(
        &self,
        user: &User,
        expected_password_enc: &[u8],
        new_password_enc: Vec<u8>,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<(), DBError>;
    fn mark_password_reset_as_complete(
        &self,
        request: &PasswordResetRequest,
    ) -> Result<(), DBError>;
    fn complete_destructive_password_reset(
        &self,
        user: &User,
        reset_request: &PasswordResetRequest,
        new_password_enc: Vec<u8>,
        new_wrapping: NewUserSeedWrapping,
        enclave_key: &[u8],
        build_reset_clear_material: &BuildResetClearMaterial<'_>,
    ) -> Result<(), DBError>;

    fn register_maple_device(
        &self,
        registration: NewMapleDeviceRegistration,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &MaterializeMapleDeviceRegistrationSync<'_>,
    ) -> Result<MapleDeviceRegistrationReceipt, DBError>;
    fn list_maple_devices(
        &self,
        authorization: MapleDeviceListAuthorization,
        limit: i64,
        after: Option<MapleDeviceListCursor>,
    ) -> Result<MapleDeviceListPage, DBError>;
    /// Authenticates every durable issuer-bearing pairing record and returns
    /// the exact sorted key IDs that a verification keyset must retain.
    fn audit_maple_pairing_issuer_key_references(
        &self,
        enclave_key: &[u8],
    ) -> Result<Vec<String>, DBError>;
    fn replay_maple_reset_clear_ack(
        &self,
        authorization: MaplePairingAuthorization,
        host_registration_id: Uuid,
        operation_id: Uuid,
        request_mac: Vec<u8>,
    ) -> Result<Option<MaplePairingOperationReceipt>, DBError>;
    fn replay_maple_pairing_operation(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        operation_id: Uuid,
        operation_kind: MaplePairingOperationKind,
        request_mac: Vec<u8>,
    ) -> Result<Option<MaplePairingOperationReceipt>, DBError>;
    fn create_maple_pairing(
        &self,
        request: NewMaplePairingRequest,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &MaterializeMaplePairingCreate<'_>,
    ) -> Result<MaplePairingOperationReceipt, DBError>;
    fn list_maple_pairings(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        role: MaplePairingRole,
        states: Vec<MaplePairingState>,
        limit: i64,
        after: Option<MaplePairingCursor>,
    ) -> Result<Vec<MaplePairing>, DBError>;
    fn get_maple_pairing(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        pair_id: Uuid,
    ) -> Result<Option<MaplePairing>, DBError>;
    fn approve_maple_pairing(
        &self,
        mutation: MaplePairingApproval,
    ) -> Result<MaplePairingOperationReceipt, DBError>;
    fn confirm_maple_pairing(
        &self,
        mutation: MaplePairingConfirmation,
    ) -> Result<MaplePairingOperationReceipt, DBError>;
    fn revoke_maple_pairing(
        &self,
        mutation: MaplePairingRevocation,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &dyn Fn(
            MaplePairingRevocationContext,
        ) -> Result<
            MaplePairingRevocationMaterial,
            MaplePairingMaterializationError,
        >,
    ) -> Result<MaplePairingOperationReceipt, DBError>;
    fn list_maple_pairing_revocations(
        &self,
        authorization: MaplePairingAuthorization,
        host_registration_id: Uuid,
        expected_revocation_stream_id: Uuid,
        expected_revocation_stream_generation: u64,
        after_issuer_sequence: u64,
        limit: i64,
    ) -> Result<MaplePairingRevocationPage, DBError>;
    fn ack_maple_pairing_revocation(
        &self,
        ack: MaplePairingRevocationAck,
    ) -> Result<MaplePairingOperationReceipt, DBError>;

    // Account Deletion methods
    fn create_account_deletion_request(
        &self,
        new_request: NewAccountDeletionRequest,
    ) -> Result<AccountDeletionRequest, DBError>;
    fn get_account_deletion_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<AccountDeletionRequest>, DBError>;
    fn mark_account_deletion_as_complete(
        &self,
        request: &AccountDeletionRequest,
    ) -> Result<(), DBError>;
    fn delete_user(&self, user: &User, enclave_key: &[u8]) -> Result<(), DBError>;
    fn mark_and_delete_user(
        &self,
        user: &User,
        deletion_request: &AccountDeletionRequest,
        enclave_key: &[u8],
    ) -> Result<(), DBError>;

    // OAuth Provider methods
    fn create_oauth_provider(
        &self,
        new_provider: NewOAuthProvider,
    ) -> Result<OAuthProvider, DBError>;
    fn get_oauth_provider_by_id(&self, id: i32) -> Result<Option<OAuthProvider>, DBError>;
    fn get_oauth_provider_by_name(&self, name: &str) -> Result<Option<OAuthProvider>, DBError>;
    fn get_all_oauth_providers(&self) -> Result<Vec<OAuthProvider>, DBError>;
    fn update_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError>;
    fn delete_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError>;

    // User OAuth Connection methods
    fn create_user_oauth_connection(
        &self,
        new_connection: NewUserOAuthConnection,
    ) -> Result<UserOAuthConnection, DBError>;
    fn get_user_oauth_connection_by_id(
        &self,
        id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError>;
    fn get_user_oauth_connection_by_user_and_provider(
        &self,
        user_id: Uuid,
        provider_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError>;
    fn get_project_user_oauth_connection_by_provider_subject(
        &self,
        provider_id: i32,
        provider_user_id: &str,
        project_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError>;
    fn get_all_user_oauth_connections_for_user(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<UserOAuthConnection>, DBError>;
    fn update_user_oauth_connection(&self, connection: &UserOAuthConnection)
        -> Result<(), DBError>;
    fn delete_user_oauth_connection(&self, connection: &UserOAuthConnection)
        -> Result<(), DBError>;

    fn create_token_usage(&self, new_usage: NewTokenUsage) -> Result<TokenUsage, DBError>;

    fn update_user(&self, user: &User) -> Result<(), DBError>;
    // New org-related methods
    fn create_org(&self, new_org: NewOrg, enclave_key: &[u8]) -> Result<Org, DBError>;
    fn get_org_by_id(&self, id: i32) -> Result<Org, DBError>;
    fn get_org_by_uuid(&self, uuid: Uuid) -> Result<Org, DBError>;
    fn get_org_by_name(&self, name: &str) -> Result<Option<Org>, DBError>;
    fn get_all_orgs(&self) -> Result<Vec<Org>, DBError>;
    fn update_org(&self, org: &Org) -> Result<(), DBError>;
    fn delete_org(&self, org: &Org, enclave_key: &[u8]) -> Result<(), DBError>;

    // Org project methods
    fn create_org_project(
        &self,
        new_project: NewOrgProject,
        enclave_key: &[u8],
    ) -> Result<OrgProject, DBError>;
    fn get_org_project_by_id(&self, id: i32) -> Result<OrgProject, DBError>;
    fn get_org_project_by_uuid(&self, uuid: Uuid) -> Result<OrgProject, DBError>;
    fn get_org_project_by_client_id(&self, client_id: Uuid) -> Result<OrgProject, DBError>;
    fn get_org_project_by_name_and_org(
        &self,
        name: &str,
        org_id: i32,
    ) -> Result<Option<OrgProject>, DBError>;
    fn get_all_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError>;
    fn get_active_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError>;
    fn update_org_project(&self, project: &OrgProject) -> Result<(), DBError>;
    fn delete_org_project(&self, project: &OrgProject, enclave_key: &[u8]) -> Result<(), DBError>;

    // Org project secret methods
    fn create_org_project_secret(
        &self,
        new_secret: NewOrgProjectSecret,
    ) -> Result<OrgProjectSecret, DBError>;
    fn get_org_project_secret_by_id(&self, id: i32) -> Result<OrgProjectSecret, DBError>;
    fn get_org_project_secret_by_key_name_and_project(
        &self,
        key_name: &str,
        project_id: i32,
    ) -> Result<Option<OrgProjectSecret>, DBError>;
    fn get_all_org_project_secrets_for_project(
        &self,
        project_id: i32,
    ) -> Result<Vec<OrgProjectSecret>, DBError>;
    fn update_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError>;
    fn delete_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError>;

    // Invite code methods
    fn create_invite_code(&self, new_invite: NewInviteCode) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_id(&self, id: i32) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_code(&self, code: Uuid) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_email_and_org(
        &self,
        email: &str,
        org_id: i32,
    ) -> Result<Option<InviteCode>, DBError>;
    fn get_all_invite_codes_for_org(&self, org_id: i32) -> Result<Vec<InviteCode>, DBError>;
    fn mark_invite_code_as_used(&self, invite: &InviteCode) -> Result<(), DBError>;
    fn update_invite_code(&self, invite: &InviteCode) -> Result<(), DBError>;
    fn delete_invite_code(&self, invite: &InviteCode) -> Result<(), DBError>;

    // Platform user methods
    fn create_platform_user(&self, new_user: NewPlatformUser) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_id(&self, id: i32) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_uuid(&self, uuid: Uuid) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_email(&self, email: &str) -> Result<Option<PlatformUser>, DBError>;
    fn update_platform_user(&self, user: &PlatformUser) -> Result<(), DBError>;
    fn update_platform_user_password(
        &self,
        user: &PlatformUser,
        new_password_enc: Vec<u8>,
    ) -> Result<(), DBError>;

    // Org membership methods
    fn create_org_membership(
        &self,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError>;
    fn get_org_membership_by_platform_user_and_org(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembership, DBError>;

    fn get_org_membership_by_platform_user_and_org_with_user(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembershipWithUser, DBError>;
    fn get_all_org_memberships_for_platform_user(
        &self,
        platform_user_id: Uuid,
    ) -> Result<Vec<OrgMembership>, DBError>;
    fn get_all_org_memberships_for_org(&self, org_id: i32) -> Result<Vec<OrgMembership>, DBError>;
    fn get_all_org_memberships_with_users_for_org(
        &self,
        org_id: i32,
    ) -> Result<Vec<OrgMembershipWithUser>, DBError>;
    fn update_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError>;
    fn delete_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError>;
    fn update_membership_role(
        &self,
        membership: &mut OrgMembership,
        new_role: OrgRole,
    ) -> Result<(), DBError>;
    fn delete_membership_with_owner_check(&self, membership: &OrgMembership)
        -> Result<(), DBError>;

    // project-scoped methods
    fn get_users_for_project(
        &self,
        project_id: i32,
        page: Option<i64>,
        per_page: Option<i64>,
    ) -> Result<(Vec<User>, i64), DBError>;

    fn create_org_with_owner(
        &self,
        new_org: NewOrg,
        owner_id: Uuid,
        enclave_key: &[u8],
    ) -> Result<Org, DBError>;

    fn accept_invite_transaction(
        &self,
        invite: &InviteCode,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError>;

    // Project settings methods
    fn get_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
    ) -> Result<Option<ProjectSetting>, DBError>;

    fn update_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
        settings: serde_json::Value,
    ) -> Result<ProjectSetting, DBError>;

    fn get_project_email_settings(&self, project_id: i32)
        -> Result<Option<EmailSettings>, DBError>;

    fn update_project_email_settings(
        &self,
        project_id: i32,
        settings: EmailSettings,
    ) -> Result<ProjectSetting, DBError>;

    fn get_project_oauth_settings(&self, project_id: i32)
        -> Result<Option<OAuthSettings>, DBError>;

    fn update_project_oauth_settings(
        &self,
        project_id: i32,
        settings: OAuthSettings,
    ) -> Result<ProjectSetting, DBError>;

    // Platform email verification methods
    fn create_platform_email_verification(
        &self,
        new_verification: NewPlatformEmailVerification,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_id(
        &self,
        id: i32,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_platform_user_id(
        &self,
        platform_user_id: Uuid,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_code(
        &self,
        code: Uuid,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn update_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError>;

    fn delete_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError>;

    fn verify_platform_email(
        &self,
        verification: &mut PlatformEmailVerification,
    ) -> Result<(), DBError>;

    // Platform password reset methods
    fn create_platform_password_reset_request(
        &self,
        new_request: NewPlatformPasswordResetRequest,
    ) -> Result<PlatformPasswordResetRequest, DBError>;

    fn get_platform_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PlatformPasswordResetRequest>, DBError>;

    fn mark_platform_password_reset_as_complete(
        &self,
        request: &PlatformPasswordResetRequest,
    ) -> Result<(), DBError>;

    // User API key methods
    fn create_user_api_key(&self, new_key: NewUserApiKey) -> Result<UserApiKey, DBError>;
    fn get_user_api_key_by_id(&self, id: i32) -> Result<Option<UserApiKey>, DBError>;
    fn get_user_api_key_by_hash(&self, key_hash: &str) -> Result<Option<UserApiKey>, DBError>;
    fn get_user_by_api_key_hash(&self, key_hash: &str) -> Result<Option<User>, DBError>;
    fn get_all_user_api_keys_for_user(&self, user_id: Uuid) -> Result<Vec<UserApiKey>, DBError>;
    fn delete_user_api_key(&self, id: i32, user_id: Uuid) -> Result<(), DBError>;
    fn delete_user_api_key_by_name(&self, name: &str, user_id: Uuid) -> Result<(), DBError>;

    // Platform invite code methods
    fn validate_platform_invite_code(&self, code: Uuid) -> Result<PlatformInviteCode, DBError>;

    // ---------- Responses API helpers ----------

    // Conversations
    fn create_conversation(
        &self,
        new_conversation: NewConversation,
    ) -> Result<Conversation, DBError>;
    fn create_conversation_project(
        &self,
        new_project: NewConversationProject,
    ) -> Result<ConversationProject, DBError>;
    fn get_conversation_by_id_and_user(
        &self,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<Conversation, DBError>;
    fn get_conversation_by_uuid_and_user(
        &self,
        conversation_uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Conversation, DBError>;
    fn get_conversation_project_by_id_and_user(
        &self,
        project_id: i64,
        user_id: Uuid,
    ) -> Result<ConversationProject, DBError>;
    fn get_conversation_project_by_uuid_and_user(
        &self,
        project_uuid: Uuid,
        user_id: Uuid,
    ) -> Result<ConversationProject, DBError>;
    fn update_conversation_metadata(
        &self,
        conversation_id: i64,
        user_id: Uuid,
        metadata_enc: Vec<u8>,
    ) -> Result<Conversation, DBError>;
    fn update_conversation(
        &self,
        conversation_id: i64,
        user_id: Uuid,
        metadata_enc: Option<Vec<u8>>,
        project_id: Option<Option<i64>>,
        is_pinned: Option<bool>,
    ) -> Result<Conversation, DBError>;
    fn batch_update_conversation_project(
        &self,
        conversation_uuids: &[Uuid],
        user_id: Uuid,
        target_project_id: Option<i64>,
    ) -> Result<(), DBError>;
    #[allow(clippy::too_many_arguments)]
    fn list_conversations(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
        project_filter: ConversationProjectFilter,
        pinned: Option<bool>,
    ) -> Result<Vec<Conversation>, DBError>;
    fn list_conversation_projects(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<ConversationProject>, DBError>;
    fn update_conversation_project(
        &self,
        project_id: i64,
        user_id: Uuid,
        name_enc: Option<Vec<u8>>,
        instruction_update: ProjectInstructionUpdate,
    ) -> Result<ConversationProject, DBError>;
    fn delete_conversation(&self, conversation_id: i64, user_id: Uuid) -> Result<(), DBError>;
    fn delete_all_conversations(&self, user_id: Uuid) -> Result<(), DBError>;
    fn delete_conversation_project(&self, project_id: i64, user_id: Uuid) -> Result<(), DBError>;

    // Responses (job tracker)
    fn create_response(&self, new_response: NewResponse) -> Result<Response, DBError>;
    fn get_response_by_uuid_and_user(&self, uuid: Uuid, user_id: Uuid)
        -> Result<Response, DBError>;
    fn update_response_status(
        &self,
        id: i64,
        status: ResponseStatus,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), DBError>;
    fn update_response_status_if_current(
        &self,
        id: i64,
        current_status: ResponseStatus,
        new_status: ResponseStatus,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<bool, DBError>;
    fn cancel_response(&self, uuid: Uuid, user_id: Uuid) -> Result<Response, DBError>;
    fn delete_response(&self, uuid: Uuid, user_id: Uuid) -> Result<(), DBError>;

    #[allow(clippy::too_many_arguments)]
    fn create_conversation_with_response_and_message(
        &self,
        conversation_uuid: Uuid,
        user_id: Uuid,
        metadata_enc: Option<Vec<u8>>,
        response: Option<NewResponse>,
        first_message_content: Vec<u8>,
        first_message_tokens: i32,
        message_uuid: Uuid,
        assistant_message_uuid: Option<Uuid>,
    ) -> Result<(Conversation, Option<Response>, UserMessage), DBError>;

    // User instructions methods
    fn get_default_user_instruction(
        &self,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError>;
    fn get_project_instruction(
        &self,
        project_id: i64,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError>;
    fn get_project_instruction_for_conversation(
        &self,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError>;
    fn get_user_instruction_by_uuid_and_user(
        &self,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<UserInstruction, DBError>;
    fn create_user_instruction(
        &self,
        new_instruction: NewUserInstruction,
    ) -> Result<UserInstruction, DBError>;
    fn update_user_instruction(
        &self,
        id: i64,
        user_id: Uuid,
        name_enc: Vec<u8>,
        prompt_enc: Vec<u8>,
        prompt_tokens: i32,
        is_default: bool,
    ) -> Result<UserInstruction, DBError>;
    fn delete_user_instruction(&self, id: i64, user_id: Uuid) -> Result<(), DBError>;
    fn list_user_instructions(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<UserInstruction>, DBError>;
    fn set_default_user_instruction(
        &self,
        id: i64,
        user_id: Uuid,
    ) -> Result<UserInstruction, DBError>;

    // User messages
    fn create_user_message(&self, new_msg: NewUserMessage) -> Result<UserMessage, DBError>;
    fn update_user_message_prompt_tokens(&self, id: i64, prompt_tokens: i32)
        -> Result<(), DBError>;
    fn get_user_message(&self, id: i64, user_id: Uuid) -> Result<UserMessage, DBError>;
    fn get_user_message_by_uuid(&self, uuid: Uuid, user_id: Uuid) -> Result<UserMessage, DBError>;

    // Assistant messages
    fn create_assistant_message(
        &self,
        new_msg: NewAssistantMessage,
    ) -> Result<AssistantMessage, DBError>;
    fn get_assistant_message_by_uuid(
        &self,
        message_uuid: Uuid,
    ) -> Result<Option<AssistantMessage>, DBError>;
    fn update_assistant_message(
        &self,
        message_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        completion_tokens: i32,
        status: String,
        finish_reason: Option<String>,
    ) -> Result<AssistantMessage, DBError>;

    // Reasoning items
    fn create_reasoning_item(&self, new_item: NewReasoningItem) -> Result<ReasoningItem, DBError>;
    fn update_reasoning_item(
        &self,
        item_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        reasoning_tokens: i32,
        status: String,
    ) -> Result<ReasoningItem, DBError>;

    // Tool calls / outputs
    fn create_tool_call(&self, new_call: NewToolCall) -> Result<ToolCall, DBError>;
    fn get_tool_call_by_uuid(&self, uuid: Uuid, user_id: Uuid) -> Result<ToolCall, DBError>;
    fn create_tool_output(&self, new_output: NewToolOutput) -> Result<ToolOutput, DBError>;

    // Context reconstruction
    fn get_conversation_context_messages(
        &self,
        conversation_id: i64,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<RawThreadMessage>, DBError>;
    fn get_response_context_messages(
        &self,
        response_id: i64,
    ) -> Result<Vec<RawThreadMessage>, DBError>;

    // Optimized context reconstruction (metadata-based)
    fn get_conversation_context_metadata(
        &self,
        conversation_id: i64,
    ) -> Result<Vec<RawThreadMessageMetadata>, DBError>;
    fn get_messages_by_ids(
        &self,
        conversation_id: i64,
        message_ids: &[(String, i64)],
    ) -> Result<Vec<RawThreadMessage>, DBError>;

    // Delete operation for user messages
    fn delete_user_message(&self, id: Uuid, user_id: Uuid) -> Result<(), DBError>;
}

pub(crate) struct PostgresConnection {
    db: Pool<ConnectionManager<PgConnection>>,
    maple_pairing_issuer_key_inventory_digest: OnceLock<[u8; 32]>,
}

impl PostgresConnection {
    fn require_configured_maple_pairing_issuer_keyset(
        &self,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
    ) -> Result<Vec<u8>, DBError> {
        let fingerprints = issuer_keyset
            .fingerprints()
            .map_err(|_| DBError::MaplePairingIssuerConfigurationConflict)?;
        let supplied_digest =
            maple_pairing_issuer_key_inventory_digest_from_fingerprints(&fingerprints)?;
        let configured_digest = self.configured_maple_pairing_issuer_key_inventory_digest()?;
        if !maple_pairing_authority_mac_matches(&supplied_digest, &configured_digest) {
            return Err(DBError::MaplePairingIssuerConfigurationConflict);
        }
        Ok(configured_digest)
    }
}

impl DBConnection for PostgresConnection {
    fn bootstrap_or_audit_maple_pairing_authority(
        &self,
        enclave_key: &[u8],
        issuer_keyset: Option<&MaplePairingIssuerKeySetV1>,
    ) -> Result<(), DBError> {
        let configured_issuer_keys = issuer_keyset
            .map(MaplePairingIssuerKeySetV1::fingerprints)
            .transpose()
            .map_err(|_| DBError::MaplePairingIssuerConfigurationConflict)?
            .unwrap_or_default();
        let expected_inventory_digest =
            maple_pairing_issuer_key_inventory_digest_from_fingerprints(&configured_issuer_keys)?;
        let expected_inventory_digest: [u8; 32] = expected_inventory_digest
            .try_into()
            .map_err(|_| DBError::MaplePairingIssuerConfigurationConflict)?;
        if let Err(candidate) = self
            .maple_pairing_issuer_key_inventory_digest
            .set(expected_inventory_digest)
        {
            if self.maple_pairing_issuer_key_inventory_digest.get() != Some(&candidate) {
                return Err(DBError::MaplePairingIssuerConfigurationConflict);
            }
        }
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let inventory_digest = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
            |tx| {
                acquire_maple_pairing_authority_bootstrap_snapshot_fence(tx, enclave_key)?;
                bootstrap_or_audit_maple_pairing_authority_in_tx(
                    tx,
                    enclave_key,
                    &configured_issuer_keys,
                )
            },
        )?;
        if inventory_digest.as_slice() != expected_inventory_digest.as_slice() {
            return Err(DBError::MaplePairingAuthorityCorrupt);
        }
        Ok(())
    }

    fn configured_maple_pairing_issuer_key_inventory_digest(&self) -> Result<Vec<u8>, DBError> {
        self.maple_pairing_issuer_key_inventory_digest
            .get()
            .map(|digest| digest.to_vec())
            .ok_or(DBError::MaplePairingAuthorityCorrupt)
    }

    fn create_user(&self, new_user: NewUser, enclave_key: &[u8]) -> Result<User, DBError> {
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
            |tx| {
                create_user_with_maple_authority_in_tx(
                    tx,
                    &new_user,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )
            },
        );
        if let Err(ref e) = result {
            error!("Failed to create user: {:?}", e);
        }
        result
    }

    fn get_user_by_uuid(&self, uuid: Uuid) -> Result<User, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = User::get_by_uuid(conn, uuid)?.ok_or(DBError::UserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get user by UUID: {:?}", e);
        }
        result
    }

    fn get_user_by_email(&self, email: String, project_id: i32) -> Result<User, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = User::get_by_email(conn, email, project_id)?.ok_or(DBError::UserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get user by email: {:?}", e);
        }
        result
    }

    fn create_user_seed_wrapping(
        &self,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<UserSeedWrapping, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_wrapping.insert(conn).map_err(DBError::from)
    }

    fn upsert_user_seed_wrapping(
        &self,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<UserSeedWrapping, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_wrapping
            .upsert_by_credential(conn)
            .map_err(DBError::from)
    }

    fn get_user_seed_wrappings_for_user_and_kind(
        &self,
        user_id: Uuid,
        credential_kind: &str,
    ) -> Result<Vec<UserSeedWrapping>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserSeedWrapping::get_for_user_and_kind(conn, user_id, credential_kind)
            .map_err(DBError::from)
    }

    fn get_user_seed_wrapping_by_credential(
        &self,
        user_id: Uuid,
        credential_kind: &str,
        credential_lookup_hash: &[u8],
        wrapping_version: i16,
    ) -> Result<Option<UserSeedWrapping>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserSeedWrapping::get_by_credential(
            conn,
            user_id,
            credential_kind,
            credential_lookup_hash,
            wrapping_version,
        )
        .map_err(DBError::from)
    }

    fn delete_user_seed_wrappings_for_user(&self, user_id: Uuid) -> Result<usize, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserSeedWrapping::delete_for_user(conn, user_id).map_err(DBError::from)
    }

    fn delete_user_seed_wrappings_for_user_and_kind(
        &self,
        user_id: Uuid,
        credential_kind: &str,
    ) -> Result<usize, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserSeedWrapping::delete_for_user_and_kind(conn, user_id, credential_kind)
            .map_err(DBError::from)
    }

    fn get_app_data_migration(&self, name: &str) -> Result<Option<AppDataMigration>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        AppDataMigration::get(conn, name).map_err(DBError::from)
    }

    fn app_data_migration_exists(&self, name: &str) -> Result<bool, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        AppDataMigration::exists(conn, name).map_err(DBError::from)
    }

    fn create_app_data_migration(
        &self,
        new_migration: NewAppDataMigration,
    ) -> Result<AppDataMigration, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_migration.insert(conn).map_err(DBError::from)
    }

    fn get_pool(&self) -> &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>> {
        &self.db
    }

    fn create_enclave_secret(
        &self,
        new_secret: NewEnclaveSecret,
    ) -> Result<EnclaveSecret, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_secret.insert(conn).map_err(DBError::from)
    }

    fn get_enclave_secret_by_id(&self, id: i32) -> Result<Option<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_enclave_secret_by_key(&self, key: &str) -> Result<Option<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_by_key(conn, key).map_err(DBError::from)
    }

    fn get_all_enclave_secrets(&self) -> Result<Vec<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_all(conn).map_err(DBError::from)
    }

    fn update_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        secret.update(conn).map_err(DBError::from)
    }

    fn delete_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        secret.delete(conn).map_err(DBError::from)
    }

    fn create_email_verification(
        &self,
        new_verification: NewEmailVerification,
    ) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_verification.insert(conn).map_err(DBError::from)
    }

    fn get_email_verification_by_id(&self, id: i32) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_id(conn, id)?.ok_or(DBError::EmailVerificationNotFound)
    }

    fn get_email_verification_by_user_id(
        &self,
        user_id: Uuid,
    ) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_user_id(conn, user_id)?.ok_or(DBError::EmailVerificationNotFound)
    }

    fn get_email_verification_by_code(&self, code: Uuid) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_verification_code(conn, code)?
            .ok_or(DBError::EmailVerificationNotFound)
    }

    fn update_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        verification.update(conn).map_err(DBError::from)
    }

    fn delete_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        verification.delete(conn).map_err(DBError::from)
    }

    fn verify_email(&self, verification: &mut EmailVerification) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.verify(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to verify email: {:?}", e);
        }
        result
    }

    fn create_password_reset_request(
        &self,
        new_request: NewPasswordResetRequest,
    ) -> Result<PasswordResetRequest, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_request.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create password reset request: {:?}", e);
        }
        result
    }

    fn get_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PasswordResetRequest>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PasswordResetRequest::get_by_user_id_and_code(conn, user_id, &encrypted_code)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get password reset request: {:?}", e);
        }
        result
    }

    fn update_user_password_and_seed_wrap(
        &self,
        user: &User,
        expected_password_enc: &[u8],
        new_password_enc: Vec<u8>,
        new_wrapping: NewUserSeedWrapping,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        conn.transaction::<_, DBError, _>(|conn| {
            let updated_user_count = diesel::update(
                users::table
                    .filter(users::uuid.eq(user.uuid))
                    .filter(users::password_enc.eq(Some(expected_password_enc.to_vec()))),
            )
            .set((
                users::password_enc.eq(Some(new_password_enc)),
                users::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)?;
            if updated_user_count != 1 {
                return Err(DBError::StaleCredentialState);
            }

            UserSeedWrapping::delete_for_user_and_kind(
                conn,
                user.uuid,
                CredentialKind::Password.as_str(),
            )?;
            new_wrapping.insert(conn)?;
            Ok(())
        })
    }

    fn list_maple_pairing_revocations(
        &self,
        authorization: MaplePairingAuthorization,
        host_registration_id: Uuid,
        expected_revocation_stream_id: Uuid,
        expected_revocation_stream_generation: u64,
        after_issuer_sequence: u64,
        limit: i64,
    ) -> Result<MaplePairingRevocationPage, DBError> {
        use crate::models::maple_pairing_db::MaplePairingRevocationPageEntry;
        use crate::models::schema::{
            maple_device_registration_operations, maple_pairing_revocation_events, maple_pairings,
        };
        use diesel::JoinOnDsl;

        let after =
            i64::try_from(after_issuer_sequence).map_err(|_| DBError::MaplePairingConflict)?;
        let expected_generation = i64::try_from(expected_revocation_stream_generation)
            .map_err(|_| DBError::MaplePairingConflict)?;
        let is_discovery = expected_revocation_stream_id.is_nil()
            && expected_generation == 0
            && after_issuer_sequence == 0;
        let is_established = !expected_revocation_stream_id.is_nil() && expected_generation > 0;
        if !is_discovery && !is_established {
            return Err(DBError::MaplePairingConflict);
        }
        let limit = limit.clamp(1, MAPLE_PAIRING_REVOCATION_QUERY_LIMIT);
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReadOnly,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "list_maple_pairing_revocations",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(&authorization),
                    false,
                )?;
                let Some(host) =
                    find_scoped_maple_device(tx, &authorization, host_registration_id, false)?
                else {
                    return Err(DBError::MaplePairingNotFound);
                };
                let (_, highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    host.installation_id,
                    false,
                )?;
                let host_state = load_maple_pairing_host_state(
                    tx,
                    authorization.user_id,
                    authorization.project_id,
                    host.id,
                    false,
                )?;
                let highwater = highwater.ok_or(DBError::MaplePairingCorrupt)?;
                let state = host_state.ok_or(DBError::MaplePairingCorrupt)?;
                validate_maple_pairing_host_state(&authorization.enclave_key, &state)?;
                if state.revocation_stream_id != highwater.revocation_stream_id
                    || state.revocation_stream_generation != highwater.revocation_stream_generation
                    || state.last_issued_revocation_sequence
                        != highwater.last_issued_revocation_sequence
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                if is_established
                    && (expected_revocation_stream_id != highwater.revocation_stream_id
                        || expected_generation != highwater.revocation_stream_generation)
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let last_issued = pairing_u64_from_i64(state.last_issued_revocation_sequence)?;
                let last_acked = pairing_u64_from_i64(state.last_acked_revocation_sequence)?;
                let (
                    reset_clear_sync_payload,
                    reset_clear_lifecycle_floor,
                    pending_reset_clear_occupies_sequence_one,
                ) = if let Some(pending) = load_latest_pending_maple_reset_clear_obligation(
                    tx,
                    &authorization.enclave_key,
                    &highwater,
                    false,
                )? {
                    if pending.revision != 2
                        || state.last_issued_revocation_sequence != 1
                        || state.last_acked_revocation_sequence != 0
                        || state.revision != 2
                    {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                    let max_registration_acceptance = maple_device_registration_operations::table
                        .filter(maple_device_registration_operations::maple_device_id.eq(host.id))
                        .select(diesel::dsl::max(
                            maple_device_registration_operations::accepted_at,
                        ))
                        .first::<Option<DateTime<Utc>>>(tx)?;
                    let lifecycle_floor = max_registration_acceptance
                        .map_or(pending.reset_at, |accepted| accepted.max(pending.reset_at));
                    (
                        Some(decrypt_maple_reset_clear_payload(
                            &authorization.enclave_key,
                            &pending,
                            MapleResetClearPayloadKind::Sync,
                        )?),
                        Some(lifecycle_floor),
                        true,
                    )
                } else {
                    (None, None, false)
                };
                // A host may page only from a durably acknowledged prefix. This
                // prevents a signed-but-buggy client from skipping an unapplied
                // revocation by advancing its read cursor without an ACK CAS.
                if after_issuer_sequence > last_acked {
                    return Err(DBError::MaplePairingConflict);
                }
                let joined_rows = maple_pairing_revocation_events::table
                    .inner_join(maple_pairings::table.on(
                        maple_pairings::id.eq(maple_pairing_revocation_events::maple_pairing_id),
                    ))
                    .filter(maple_pairing_revocation_events::user_id.eq(authorization.user_id))
                    .filter(
                        maple_pairing_revocation_events::project_id.eq(authorization.project_id),
                    )
                    .filter(
                        maple_pairing_revocation_events::recipient_host_maple_device_id.eq(host.id),
                    )
                    .filter(
                        maple_pairing_revocation_events::revocation_stream_id
                            .eq(highwater.revocation_stream_id),
                    )
                    .filter(
                        maple_pairing_revocation_events::revocation_stream_generation
                            .eq(highwater.revocation_stream_generation),
                    )
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .filter(maple_pairings::host_maple_device_id.eq(host.id))
                    .filter(maple_pairing_revocation_events::issuer_sequence.gt(after))
                    .order(maple_pairing_revocation_events::issuer_sequence.asc())
                    .limit(limit)
                    .select((
                        maple_pairing_revocation_events::all_columns,
                        maple_pairings::all_columns,
                    ))
                    .load::<(MaplePairingRevocationEvent, MaplePairing)>(tx)?;
                let mut expected = after.checked_add(1).ok_or(DBError::MaplePairingCorrupt)?;
                let mut events = Vec::with_capacity(joined_rows.len());
                for (event, pairing) in joined_rows {
                    validate_maple_pairing_revocation_record(&authorization.enclave_key, &event)?;
                    validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                    if event.issuer_sequence != expected {
                        return Err(DBError::MaplePairingCorrupt);
                    }
                    if event.maple_pairing_id != pairing.id
                        || event.user_id != pairing.user_id
                        || event.project_id != pairing.project_id
                        || event.recipient_host_maple_device_id != pairing.host_maple_device_id
                        || event.pairing_incarnation != pairing.pairing_incarnation
                        || event.revocation_stream_id != highwater.revocation_stream_id
                        || event.revocation_stream_generation
                            != highwater.revocation_stream_generation
                        || pairing.revocation_stream_id != Some(highwater.revocation_stream_id)
                        || pairing.revocation_stream_generation
                            != Some(highwater.revocation_stream_generation)
                        || pairing.state != MaplePairingState::Revoked.as_db()
                    {
                        return Err(DBError::MaplePairingCorrupt);
                    }
                    expected = expected
                        .checked_add(1)
                        .ok_or(DBError::MaplePairingCorrupt)?;
                    events.push(MaplePairingRevocationPageEntry { event, pairing });
                }
                if events.last().is_some_and(|entry| {
                    u64::try_from(entry.event.issuer_sequence)
                        .map(|sequence| sequence > last_issued)
                        .unwrap_or(true)
                }) {
                    return Err(DBError::MaplePairingCorrupt);
                }
                let last_issued_i64 =
                    i64::try_from(last_issued).map_err(|_| DBError::MaplePairingCorrupt)?;
                // A pending reset-clear instruction is the authenticated item
                // at sequence one; it is carried in reset_clear_sync_payload,
                // not represented by a revocation-event row. Preserve the
                // ordinary event-gap proof for every other namespace, and
                // reject any row that tries to alias the reserved instruction.
                if pending_reset_clear_occupies_sequence_one {
                    if !events.is_empty() {
                        return Err(DBError::MaplePairingCorrupt);
                    }
                } else if events.is_empty() && after < last_issued_i64 {
                    return Err(DBError::MaplePairingCorrupt);
                }
                Ok(MaplePairingRevocationPage {
                    events,
                    reset_clear_sync_payload,
                    reset_clear_lifecycle_floor,
                    security_epoch: pairing_u64_from_i64(highwater.security_epoch)?,
                    revocation_stream_id: highwater.revocation_stream_id,
                    revocation_stream_generation: pairing_u64_from_i64(
                        highwater.revocation_stream_generation,
                    )?,
                    last_issued_revocation_sequence: last_issued,
                    last_acked_revocation_sequence: last_acked,
                })
            },
        )
    }

    fn ack_maple_pairing_revocation(
        &self,
        ack: MaplePairingRevocationAck,
    ) -> Result<MaplePairingOperationReceipt, DBError> {
        use crate::models::schema::{
            maple_pairing_authority_account_heads, maple_pairing_host_states,
            maple_pairing_revocation_events, maple_pairings,
        };
        use subtle::ConstantTimeEq;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authorization = &ack.authorization;
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "ack_maple_pairing_revocation",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(authorization),
                    true,
                )?;
                let head = maple_pairing_authority_account_heads::table
                    .filter(
                        maple_pairing_authority_account_heads::user_id.eq(authorization.user_id),
                    )
                    .filter(
                        maple_pairing_authority_account_heads::project_id
                            .eq(authorization.project_id),
                    )
                    .for_update()
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                validate_maple_pairing_authority_account_head(&authorization.enclave_key, &head)?;
                if let Some(receipt) = replay_maple_reset_clear_ack_in_transaction(
                    tx,
                    authorization,
                    &head,
                    ack.host_registration_id,
                    ack.operation_id,
                    &ack.request_mac,
                )? {
                    return Ok(receipt);
                }
                let host =
                    find_scoped_maple_device(tx, authorization, ack.host_registration_id, true)?
                        .ok_or(DBError::MaplePairingNotFound)?;
                let (_, highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    host.installation_id,
                    true,
                )?;
                let highwater = highwater.ok_or(DBError::MaplePairingCorrupt)?;
                let state = load_maple_pairing_host_state(
                    tx,
                    authorization.user_id,
                    authorization.project_id,
                    host.id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                validate_maple_pairing_host_state(&authorization.enclave_key, &state)?;
                if let Some(pending) = load_latest_pending_maple_reset_clear_obligation(
                    tx,
                    &authorization.enclave_key,
                    &highwater,
                    true,
                )? {
                    let receipt = acknowledge_pending_maple_reset_clear(
                        tx, &ack, &head, &host, &highwater, &state, &pending,
                    )?;
                    commit_maple_pairing_authority_account_mutation(
                        tx,
                        &authorization.enclave_key,
                        authorization.user_id,
                        authorization.project_id,
                    )?;
                    return Ok(receipt);
                }
                if let Some(prior) =
                    get_prior_pairing_operation(tx, authorization, host.id, ack.operation_id)?
                {
                    return replay_pairing_operation(
                        tx,
                        authorization,
                        &prior,
                        MAPLE_PAIRING_OPERATION_ACK,
                        &ack.request_mac,
                    );
                }

                let issuer_sequence = i64::try_from(ack.issuer_sequence)
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let expected_previous = i64::try_from(ack.expected_previous_issuer_sequence)
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let expected_generation = i64::try_from(ack.revocation_stream_generation)
                    .map_err(|_| DBError::MaplePairingConflict)?;
                if issuer_sequence <= 0
                    || issuer_sequence != expected_previous.saturating_add(1)
                    || ack.revocation_stream_id.is_nil()
                    || expected_generation <= 0
                    || ack.event_id.is_nil()
                    || ack.event_digest.len() != 32
                    || ack.request_mac.len() != 32
                    || !maple_pairing_issuer_key_id_is_valid(&ack.checkpoint_issuer_key_id)
                    || ack.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
                    || ack.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                {
                    return Err(DBError::MaplePairingConflict);
                }
                if state.revocation_stream_id != highwater.revocation_stream_id
                    || state.revocation_stream_generation != highwater.revocation_stream_generation
                    || state.last_issued_revocation_sequence
                        != highwater.last_issued_revocation_sequence
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                if ack.revocation_stream_id != highwater.revocation_stream_id
                    || expected_generation != highwater.revocation_stream_generation
                    || state.last_acked_revocation_sequence != expected_previous
                    || issuer_sequence > state.last_issued_revocation_sequence
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let event = maple_pairing_revocation_events::table
                    .filter(maple_pairing_revocation_events::user_id.eq(authorization.user_id))
                    .filter(
                        maple_pairing_revocation_events::project_id.eq(authorization.project_id),
                    )
                    .filter(
                        maple_pairing_revocation_events::recipient_host_maple_device_id.eq(host.id),
                    )
                    .filter(
                        maple_pairing_revocation_events::revocation_stream_id
                            .eq(ack.revocation_stream_id),
                    )
                    .filter(
                        maple_pairing_revocation_events::revocation_stream_generation
                            .eq(expected_generation),
                    )
                    .filter(maple_pairing_revocation_events::uuid.eq(ack.event_id))
                    .filter(maple_pairing_revocation_events::issuer_sequence.eq(issuer_sequence))
                    .for_update()
                    .first::<MaplePairingRevocationEvent>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingNotFound,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_revocation_record(&authorization.enclave_key, &event)?;
                if event.revocation_stream_id != highwater.revocation_stream_id
                    || event.revocation_stream_generation != highwater.revocation_stream_generation
                    || event.acked_at.is_some()
                    || !bool::from(event.event_digest.as_slice().ct_eq(&ack.event_digest))
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let accepted_at = normalize_db_time(ack.accepted_at)?;
                let trusted_now = maple_pairing_trusted_db_now(tx)?;
                if accepted_at < event.created_at
                    || !maple_pairing_time_is_near_trusted_now(accepted_at, trusted_now)
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let event_record_mac = maple_pairing_revocation_record_mac(
                    &authorization.enclave_key,
                    event.uuid,
                    event.user_id,
                    event.project_id,
                    event.recipient_host_maple_device_id,
                    event.revocation_stream_id,
                    event.revocation_stream_generation,
                    event.issuer_sequence,
                    event.maple_pairing_id,
                    event.pairing_incarnation,
                    &event.issuer_key_id,
                    event.payload_version,
                    &event.payload_enc,
                    &event.event_digest,
                    event.created_at,
                    Some(accepted_at),
                )?;
                let changed_event = diesel::update(
                    maple_pairing_revocation_events::table
                        .filter(maple_pairing_revocation_events::id.eq(event.id))
                        .filter(
                            maple_pairing_revocation_events::revocation_stream_id
                                .eq(ack.revocation_stream_id),
                        )
                        .filter(
                            maple_pairing_revocation_events::revocation_stream_generation
                                .eq(expected_generation),
                        )
                        .filter(maple_pairing_revocation_events::acked_at.is_null()),
                )
                .set((
                    maple_pairing_revocation_events::acked_at.eq(Some(accepted_at)),
                    maple_pairing_revocation_events::record_mac.eq(event_record_mac),
                ))
                .execute(tx)?;
                let target_state_revision = state
                    .revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                let host_state_record_mac = maple_pairing_host_state_mac(
                    &authorization.enclave_key,
                    state.user_id,
                    state.project_id,
                    state.host_maple_device_id,
                    state.revocation_stream_id,
                    state.revocation_stream_generation,
                    state.last_issued_revocation_sequence,
                    issuer_sequence,
                    target_state_revision,
                )?;
                let changed_state = diesel::update(
                    maple_pairing_host_states::table
                        .filter(maple_pairing_host_states::id.eq(state.id))
                        .filter(
                            maple_pairing_host_states::revocation_stream_id
                                .eq(ack.revocation_stream_id),
                        )
                        .filter(
                            maple_pairing_host_states::revocation_stream_generation
                                .eq(expected_generation),
                        )
                        .filter(
                            maple_pairing_host_states::last_acked_revocation_sequence
                                .eq(expected_previous),
                        )
                        .filter(maple_pairing_host_states::revision.eq(state.revision)),
                )
                .set((
                    maple_pairing_host_states::last_acked_revocation_sequence.eq(issuer_sequence),
                    maple_pairing_host_states::revision.eq(target_state_revision),
                    maple_pairing_host_states::record_mac.eq(host_state_record_mac),
                ))
                .execute(tx)?;
                if changed_event != 1 || changed_state != 1 {
                    return Err(DBError::MaplePairingConflict);
                }
                let pairing = maple_pairings::table
                    .filter(maple_pairings::id.eq(event.maple_pairing_id))
                    .filter(maple_pairings::host_maple_device_id.eq(host.id))
                    .first::<MaplePairing>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingConflict,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                if pairing.revocation_stream_id != Some(highwater.revocation_stream_id)
                    || pairing.revocation_stream_generation
                        != Some(highwater.revocation_stream_generation)
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                let receipt = insert_pairing_operation(
                    tx,
                    authorization,
                    host.id,
                    ack.operation_id,
                    MAPLE_PAIRING_OPERATION_ACK,
                    &ack.request_mac,
                    &pairing,
                    ack.receipt_version,
                    &ack.receipt_enc,
                    Some(&ack.checkpoint_issuer_key_id),
                    accepted_at,
                )?;
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                )?;
                Ok(receipt)
            },
        )
    }

    fn revoke_maple_pairing(
        &self,
        mutation: MaplePairingRevocation,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &dyn Fn(
            MaplePairingRevocationContext,
        ) -> Result<
            MaplePairingRevocationMaterial,
            MaplePairingMaterializationError,
        >,
    ) -> Result<MaplePairingOperationReceipt, DBError> {
        use crate::models::maple_pairing_db::{
            NewMaplePairingRevocationEvent, MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST,
        };
        use crate::models::schema::{
            maple_devices, maple_pairing_host_states, maple_pairing_operations,
            maple_pairing_revocation_events, maple_pairing_revocation_highwaters, maple_pairings,
        };

        let expected_issuer_key_inventory_digest =
            self.require_configured_maple_pairing_issuer_keyset(issuer_keyset)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authorization = &mutation.authorization;
                let project_identity = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "revoke_maple_pairing",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(authorization),
                    true,
                )?;
                let actor = find_scoped_maple_device(
                    tx,
                    authorization,
                    mutation.actor_registration_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, authorization, &actor, true)?;
                if let Some(prior) =
                    get_prior_pairing_operation(tx, authorization, actor.id, mutation.operation_id)?
                {
                    return replay_pairing_operation(
                        tx,
                        authorization,
                        &prior,
                        MAPLE_PAIRING_OPERATION_REVOKE,
                        &mutation.request_mac,
                    );
                }

                let expected_actor_role = match mutation.actor_role {
                    MaplePairingRole::Controller => WireMaplePairingRole::Controller,
                    MaplePairingRole::Host => WireMaplePairingRole::Host,
                };
                let revoke_request = &mutation.revoke_request;
                if revoke_request.validate().is_err()
                    || revoke_request.operation_id != mutation.operation_id
                    || revoke_request.asserted_account_id != authorization.user_id
                    || revoke_request.asserted_project_id != project_identity.subject_project_id()
                    || revoke_request.actor_registration_id != mutation.actor_registration_id
                    || revoke_request.actor_role != expected_actor_role
                    || revoke_request.pairing_request_id != mutation.pairing_request_id
                    || revoke_request.pair_id != mutation.pair_id
                    || revoke_request.expected_pairing_revision
                        != mutation.expected_pairing_revision
                    || revoke_request.pairing_incarnation != mutation.pairing_incarnation
                    || revoke_request.revocation_stream_id != mutation.expected_revocation_stream_id
                    || revoke_request.revocation_stream_generation
                        != mutation.expected_revocation_stream_generation
                    || mutation.request_mac.len() != 32
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let revoke_request_transcript = revoke_request
                    .transcript()
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let expected_request_mac = maple_pairing_request_operation_mac(
                    &authorization.enclave_key,
                    &revoke_request_transcript,
                    &revoke_request.signature,
                )
                .map_err(|_| DBError::MaplePairingConflict)?;
                if !maple_pairing_authority_mac_matches(
                    &expected_request_mac,
                    &mutation.request_mac,
                ) {
                    return Err(DBError::MaplePairingConflict);
                }

                let incarnation = pairing_incarnation_to_i64(mutation.pairing_incarnation)?;
                let expected_generation =
                    i64::try_from(mutation.expected_revocation_stream_generation)
                        .map_err(|_| DBError::MaplePairingConflict)?;
                let current = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .filter(maple_pairings::uuid.eq(mutation.pair_id))
                    .filter(maple_pairings::pairing_request_id.eq(mutation.pairing_request_id))
                    .for_update()
                    .first::<MaplePairing>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingNotFound,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_record(&authorization.enclave_key, &current)?;
                require_maple_pairing_participants_ready(tx, authorization, &current, true)?;
                let create_operation = maple_pairing_operations::table
                    .filter(maple_pairing_operations::maple_pairing_id.eq(current.id))
                    .filter(
                        maple_pairing_operations::operation_kind.eq(MAPLE_PAIRING_OPERATION_CREATE),
                    )
                    .for_share()
                    .first::<MaplePairingOperation>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
                        other => DBError::QueryError(other),
                    })?;
                pairing_operation_receipt(
                    &authorization.enclave_key,
                    &create_operation,
                    current.uuid,
                )?;
                let approval_operation = maple_pairing_operations::table
                    .filter(maple_pairing_operations::maple_pairing_id.eq(current.id))
                    .filter(
                        maple_pairing_operations::operation_kind
                            .eq(MAPLE_PAIRING_OPERATION_APPROVE),
                    )
                    .for_share()
                    .first::<MaplePairingOperation>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
                        other => DBError::QueryError(other),
                    })?;
                pairing_operation_receipt(
                    &authorization.enclave_key,
                    &approval_operation,
                    current.uuid,
                )?;
                let actor_is_participant = match mutation.actor_role {
                    MaplePairingRole::Controller => current.controller_maple_device_id == actor.id,
                    MaplePairingRole::Host => current.host_maple_device_id == actor.id,
                };
                let target_revision = mutation
                    .expected_pairing_revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                let valid_transition = (current.state
                    == MaplePairingState::AwaitingHostCommit.as_db()
                    && current.revision == 2
                    && target_revision == 3)
                    || (current.state == MaplePairingState::Active.as_db()
                        && current.revision == 3
                        && target_revision == 4);
                if !actor_is_participant
                    || !valid_transition
                    || current.revision != mutation.expected_pairing_revision
                    || current.pairing_incarnation != incarnation
                    || mutation.expected_revocation_stream_id.is_nil()
                    || expected_generation <= 0
                    || current.revocation_stream_id != Some(mutation.expected_revocation_stream_id)
                    || current.revocation_stream_generation != Some(expected_generation)
                {
                    return Err(DBError::MaplePairingConflict);
                }

                let controller = maple_devices::table
                    .filter(maple_devices::id.eq(current.controller_maple_device_id))
                    .filter(maple_devices::user_id.eq(authorization.user_id))
                    .filter(maple_devices::project_id.eq(authorization.project_id))
                    .for_update()
                    .first::<MapleDevice>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
                        other => DBError::QueryError(other),
                    })?;
                if !maple_device_record_mac_is_valid(&authorization.enclave_key, &controller)? {
                    return Err(DBError::MaplePairingCorrupt);
                }

                let host = maple_devices::table
                    .filter(maple_devices::id.eq(current.host_maple_device_id))
                    .filter(maple_devices::user_id.eq(authorization.user_id))
                    .filter(maple_devices::project_id.eq(authorization.project_id))
                    .for_update()
                    .first::<MapleDevice>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingCorrupt,
                        other => DBError::QueryError(other),
                    })?;
                if !maple_device_record_mac_is_valid(&authorization.enclave_key, &host)? {
                    return Err(DBError::MaplePairingCorrupt);
                }
                require_no_pending_reset_clear(tx, authorization, &host, true)?;
                let (_, highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    host.installation_id,
                    true,
                )?;
                let highwater = highwater.ok_or(DBError::MaplePairingCorrupt)?;
                let host_state = load_maple_pairing_host_state(
                    tx,
                    authorization.user_id,
                    authorization.project_id,
                    current.host_maple_device_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingCorrupt)?;
                validate_maple_pairing_host_state(&authorization.enclave_key, &host_state)?;
                if host_state.revocation_stream_id != highwater.revocation_stream_id
                    || host_state.revocation_stream_generation
                        != highwater.revocation_stream_generation
                    || host_state.last_issued_revocation_sequence
                        != highwater.last_issued_revocation_sequence
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                if highwater.revocation_stream_id != mutation.expected_revocation_stream_id
                    || highwater.revocation_stream_generation != expected_generation
                {
                    return Err(DBError::MaplePairingConflict);
                }
                if current.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1 {
                    return Err(DBError::MaplePairingCorrupt);
                }
                let pairing_incarnation = pairing_u64_from_i64(current.pairing_incarnation)?;
                let revocation_stream_generation =
                    pairing_u64_from_i64(highwater.revocation_stream_generation)?;
                let stored_payload = decrypt_maple_pairing_payload(
                    &authorization.enclave_key,
                    &current.payload_enc,
                    MaplePairingPayloadCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        pairing_request_id: current.pairing_request_id,
                        pair_id: current.uuid,
                        pairing_incarnation,
                        revocation_stream_id: Some(highwater.revocation_stream_id),
                        revocation_stream_generation: Some(revocation_stream_generation),
                        payload_version: current.payload_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingCorrupt)?;
                let stored_authorization = stored_payload
                    .pair_authorization
                    .as_ref()
                    .ok_or(DBError::MaplePairingCorrupt)?;
                if stored_payload.revocation.is_some() {
                    return Err(DBError::MaplePairingCorrupt);
                }
                let issuer_sequence = highwater
                    .last_issued_revocation_sequence
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                let predecessor = current
                    .activated_at
                    .or(current.approved_at)
                    .ok_or(DBError::MaplePairingCorrupt)?;
                // Revocation must never be blocked by a wall-clock step after the
                // host became active. Clamp the trusted database instant to the
                // last committed lifecycle timestamp.
                let revoked_at = std::cmp::max(maple_pairing_trusted_db_now(tx)?, predecessor);
                let material = materialize(MaplePairingRevocationContext {
                    pairing_request_id: current.pairing_request_id,
                    pair_id: current.uuid,
                    pairing_incarnation,
                    target_revision,
                    revocation_stream_id: highwater.revocation_stream_id,
                    revocation_stream_generation,
                    issuer_sequence: pairing_u64_from_i64(issuer_sequence)?,
                    revoked_at,
                })
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let MaplePairingRevocationMaterial {
                    request_ticket: ticket,
                    pair_authorization,
                    revocation,
                    response,
                } = material;

                // Callback output is untrusted even when every individual
                // signature is valid. First require the callback to reproduce
                // the exact authority already authenticated by the durable
                // payload, then independently verify the complete issuer and
                // participant chain before constructing any ciphertext.
                if ticket != stored_payload.request_ticket
                    || pair_authorization != *stored_authorization
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
                let approved_at = current.approved_at.ok_or(DBError::MaplePairingCorrupt)?;
                let verified_ticket = ticket
                    .verify_unexpired(
                        issuer_keyset,
                        approved_at.timestamp_millis(),
                        MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
                    )
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                pair_authorization
                    .verify_against_ticket(issuer_keyset, &verified_ticket)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                revocation
                    .verify_against_authorization(issuer_keyset, &pair_authorization)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;

                let controller_request = ticket.controller_request();
                let controller_request_transcript = controller_request
                    .transcript()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let expected_create_request_mac = maple_pairing_request_operation_mac(
                    &authorization.enclave_key,
                    &controller_request_transcript,
                    &controller_request.signature,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_approval_request = pair_authorization.host_approval_request();
                let host_approval_request_transcript = host_approval_request
                    .transcript()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let expected_approval_request_mac = maple_pairing_request_operation_mac(
                    &authorization.enclave_key,
                    &host_approval_request_transcript,
                    &host_approval_request.signature,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                if create_operation.operation_id != ticket.controller_request_operation_id
                    || create_operation.user_id != authorization.user_id
                    || create_operation.project_id != authorization.project_id
                    || create_operation.actor_maple_device_id != current.controller_maple_device_id
                    || create_operation.maple_pairing_id != current.id
                    || create_operation.pairing_revision != 1
                    || create_operation.accepted_at != current.created_at
                    || !maple_pairing_authority_mac_matches(
                        &expected_create_request_mac,
                        &create_operation.request_mac,
                    )
                    || approval_operation.operation_id
                        != pair_authorization.host_approval_operation_id
                    || approval_operation.user_id != authorization.user_id
                    || approval_operation.project_id != authorization.project_id
                    || approval_operation.actor_maple_device_id != current.host_maple_device_id
                    || approval_operation.maple_pairing_id != current.id
                    || approval_operation.pairing_revision != 2
                    || approval_operation.accepted_at != approved_at
                    || !maple_pairing_authority_mac_matches(
                        &expected_approval_request_mac,
                        &approval_operation.request_mac,
                    )
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }

                let controller_claim_key = pair_authorization
                    .controller
                    .verifying_key_bytes()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_claim_key = pair_authorization
                    .host
                    .verifying_key_bytes()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let controller_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &controller_claim_key,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &host_claim_key,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let controller_epoch = i64::try_from(pair_authorization.controller.endpoint_epoch)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_epoch = i64::try_from(pair_authorization.host.endpoint_epoch)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let authorization_digest = pair_authorization
                    .digest()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let ticket_nonce = STANDARD
                    .decode(&ticket.pairing_request_nonce)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                if ticket_nonce.len() != 32
                    || STANDARD.encode(&ticket_nonce) != ticket.pairing_request_nonce
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
                let ticket_nonce_mac = maple_pairing_request_nonce_mac(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    controller.uuid,
                    &ticket_nonce,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let stored_authorization_digest = current
                    .pair_authorization_digest
                    .as_deref()
                    .ok_or(DBError::MaplePairingCorrupt)?;
                let stored_authorization_issuer = current
                    .authorization_issuer_key_id
                    .as_deref()
                    .ok_or(DBError::MaplePairingCorrupt)?;
                let actor_claim_key = match mutation.actor_role {
                    MaplePairingRole::Controller => controller_claim_key,
                    MaplePairingRole::Host => host_claim_key,
                };
                revoke_request
                    .verify_signature(&actor_claim_key)
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;

                let participant_claims_match = pair_authorization.controller.registration_id
                    == controller.uuid
                    && pair_authorization.controller.device_id == controller.device_id
                    && pair_authorization.controller.installation_id == controller.installation_id
                    && controller_epoch <= controller.endpoint_epoch
                    && maple_pairing_authority_mac_matches(
                        &controller_identity_mac,
                        &controller.identity_mac,
                    )
                    && pair_authorization.host.registration_id == host.uuid
                    && pair_authorization.host.device_id == host.device_id
                    && pair_authorization.host.installation_id == host.installation_id
                    && host_epoch <= host.endpoint_epoch
                    && maple_pairing_authority_mac_matches(&host_identity_mac, &host.identity_mac);
                let expected_revoked_at_ms = revoked_at.timestamp_millis();
                if !participant_claims_match
                    || current.direction != MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST
                    || ticket.subject_account_id != authorization.user_id
                    || ticket.subject_project_id != project_identity.subject_project_id()
                    || ticket.pairing_request_id != current.pairing_request_id
                    || ticket.pair_id != current.uuid
                    || ticket.direction != MaplePairingDirection::ControllerToHost
                    || ticket.execution_target_id != host.uuid
                    || ticket.pairing_incarnation != pairing_incarnation
                    || ticket.created_at_unix_ms != current.created_at.timestamp_millis()
                    || ticket.expires_at_unix_ms != current.expires_at.timestamp_millis()
                    || ticket.issuer_key_id != current.ticket_issuer_key_id
                    || pair_authorization.subject_account_id != authorization.user_id
                    || pair_authorization.subject_project_id
                        != project_identity.subject_project_id()
                    || pair_authorization.pairing_request_id != current.pairing_request_id
                    || pair_authorization.pair_id != current.uuid
                    || pair_authorization.direction != MaplePairingDirection::ControllerToHost
                    || pair_authorization.execution_target_id != host.uuid
                    || pair_authorization.pairing_incarnation != pairing_incarnation
                    || pair_authorization.revocation_stream_id != highwater.revocation_stream_id
                    || pair_authorization.revocation_stream_generation
                        != revocation_stream_generation
                    || pair_authorization.approved_at_unix_ms != approved_at.timestamp_millis()
                    || pair_authorization.issuer_key_id != stored_authorization_issuer
                    || !maple_pairing_authority_mac_matches(
                        &ticket_nonce_mac,
                        &current.request_nonce_mac,
                    )
                    || !maple_pairing_authority_mac_matches(
                        &authorization_digest,
                        stored_authorization_digest,
                    )
                    || revocation.subject_account_id != authorization.user_id
                    || revocation.subject_project_id != project_identity.subject_project_id()
                    || revocation.recipient_host_registration_id != host.uuid
                    || revocation.issuer_sequence != pairing_u64_from_i64(issuer_sequence)?
                    || revocation.revocation_stream_id != highwater.revocation_stream_id
                    || revocation.revocation_stream_generation != revocation_stream_generation
                    || revocation.pairing_request_id != current.pairing_request_id
                    || revocation.pair_id != current.uuid
                    || revocation.direction != MaplePairingDirection::ControllerToHost
                    || revocation.execution_target_id != host.uuid
                    || revocation.pairing_incarnation != pairing_incarnation
                    || revocation.revoked_by_registration_id != actor.uuid
                    || revocation.revoked_by_role != expected_actor_role
                    || revocation.reason_code != revoke_request.reason_code
                    || revocation.revoked_at_unix_ms != expected_revoked_at_ms
                    || !maple_pairing_issuer_key_id_is_valid(&revocation.issuer_key_id)
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }

                let expected_response = MaplePairingMutationResponse {
                    protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
                    operation_id: mutation.operation_id,
                    pairing: MaplePairingStatusV1 {
                        pairing_request_id: current.pairing_request_id,
                        pair_id: current.uuid,
                        state: WireMaplePairingState::Revoked,
                        revision: target_revision,
                        pairing_incarnation,
                        revocation_stream_id: Some(highwater.revocation_stream_id),
                        revocation_stream_generation: Some(revocation_stream_generation),
                        direction: MaplePairingDirection::ControllerToHost,
                        execution_target_id: host.uuid,
                        controller_registration_id: controller.uuid,
                        host_registration_id: host.uuid,
                        created_at_unix_ms: current.created_at.timestamp_millis(),
                        expires_at_unix_ms: current.expires_at.timestamp_millis(),
                        approved_at_unix_ms: Some(approved_at.timestamp_millis()),
                        activated_at_unix_ms: current
                            .activated_at
                            .map(|activated_at| activated_at.timestamp_millis()),
                        revoked_at_unix_ms: Some(expected_revoked_at_ms),
                        request_ticket: Some(ticket.clone()),
                        pair_authorization: if mutation.actor_role == MaplePairingRole::Controller
                            && current.activated_at.is_none()
                        {
                            None
                        } else {
                            Some(pair_authorization.clone())
                        },
                        revocation: Some(revocation.clone()),
                    },
                };
                if response != expected_response {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }

                let event_id = revocation.event_id;
                let revocation_issuer_key_id = revocation.issuer_key_id.clone();
                let pair_payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1;
                let pair_payload_enc = encrypt_maple_pairing_payload(
                    &authorization.enclave_key,
                    &StoredMaplePairingPayloadV1 {
                        request_ticket: ticket,
                        pair_authorization: Some(pair_authorization),
                        revocation: Some(revocation.clone()),
                    },
                    MaplePairingPayloadCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        pairing_request_id: current.pairing_request_id,
                        pair_id: current.uuid,
                        pairing_incarnation,
                        revocation_stream_id: Some(highwater.revocation_stream_id),
                        revocation_stream_generation: Some(revocation_stream_generation),
                        payload_version: pair_payload_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let event_payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1;
                let event_payload_enc = encrypt_maple_pairing_revocation_payload(
                    &authorization.enclave_key,
                    &revocation,
                    MaplePairingRevocationPayloadCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        host_registration_id: host.uuid,
                        revocation_stream_id: highwater.revocation_stream_id,
                        revocation_stream_generation,
                        event_id,
                        issuer_sequence: pairing_u64_from_i64(issuer_sequence)?,
                        pair_id: current.uuid,
                        pairing_incarnation,
                        payload_version: event_payload_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let event_digest = revocation
                    .digest()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?
                    .to_vec();
                let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1;
                let receipt_enc = encrypt_maple_pairing_receipt(
                    &authorization.enclave_key,
                    &response,
                    MaplePairingReceiptCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        actor_registration_id: actor.uuid,
                        operation_id: mutation.operation_id,
                        operation_kind: MAPLE_PAIRING_OPERATION_REVOKE,
                        pair_id: current.uuid,
                        pairing_revision: target_revision,
                        receipt_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                if event_id.is_nil()
                    || pair_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                    || event_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_REVOCATION_BYTES
                    || event_digest.len() != 32
                    || receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
                let event_record_mac = maple_pairing_revocation_record_mac(
                    &authorization.enclave_key,
                    event_id,
                    authorization.user_id,
                    authorization.project_id,
                    current.host_maple_device_id,
                    highwater.revocation_stream_id,
                    highwater.revocation_stream_generation,
                    issuer_sequence,
                    current.id,
                    current.pairing_incarnation,
                    &revocation_issuer_key_id,
                    event_payload_version,
                    &event_payload_enc,
                    &event_digest,
                    revoked_at,
                    None,
                )?;
                let event = diesel::insert_into(maple_pairing_revocation_events::table)
                    .values(NewMaplePairingRevocationEvent {
                        uuid: event_id,
                        user_id: authorization.user_id,
                        project_id: authorization.project_id,
                        recipient_host_maple_device_id: current.host_maple_device_id,
                        revocation_stream_id: highwater.revocation_stream_id,
                        revocation_stream_generation: highwater.revocation_stream_generation,
                        issuer_sequence,
                        maple_pairing_id: current.id,
                        pairing_incarnation: current.pairing_incarnation,
                        issuer_key_id: revocation_issuer_key_id.clone(),
                        payload_version: event_payload_version,
                        payload_enc: event_payload_enc.clone(),
                        event_digest: event_digest.clone(),
                        record_mac: event_record_mac,
                        created_at: revoked_at,
                        acked_at: None,
                    })
                    .get_result::<MaplePairingRevocationEvent>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::DatabaseError(
                            diesel::result::DatabaseErrorKind::UniqueViolation,
                            _,
                        ) => DBError::MaplePairingConflict,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_revocation_record(&authorization.enclave_key, &event)?;

                let target_highwater_revision = highwater
                    .revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                let target_highwater_mac = maple_pairing_revocation_highwater_record_mac(
                    &authorization.enclave_key,
                    &highwater.lookup_digest,
                    &highwater.authority_scope_digest,
                    highwater.revocation_stream_id,
                    highwater.revocation_stream_generation,
                    highwater.security_epoch,
                    issuer_sequence,
                    target_highwater_revision,
                )?;
                let highwater_changed = diesel::update(
                    maple_pairing_revocation_highwaters::table
                        .filter(maple_pairing_revocation_highwaters::id.eq(highwater.id))
                        .filter(
                            maple_pairing_revocation_highwaters::revocation_stream_id
                                .eq(highwater.revocation_stream_id),
                        )
                        .filter(
                            maple_pairing_revocation_highwaters::revocation_stream_generation
                                .eq(highwater.revocation_stream_generation),
                        )
                        .filter(
                            maple_pairing_revocation_highwaters::last_issued_revocation_sequence
                                .eq(highwater.last_issued_revocation_sequence),
                        )
                        .filter(
                            maple_pairing_revocation_highwaters::revision.eq(highwater.revision),
                        ),
                )
                .set((
                    maple_pairing_revocation_highwaters::last_issued_revocation_sequence
                        .eq(issuer_sequence),
                    maple_pairing_revocation_highwaters::revision.eq(target_highwater_revision),
                    maple_pairing_revocation_highwaters::record_mac.eq(target_highwater_mac),
                ))
                .execute(tx)?;

                let target_host_state_revision = host_state
                    .revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                let target_host_state_mac = maple_pairing_host_state_mac(
                    &authorization.enclave_key,
                    host_state.user_id,
                    host_state.project_id,
                    host_state.host_maple_device_id,
                    host_state.revocation_stream_id,
                    host_state.revocation_stream_generation,
                    issuer_sequence,
                    host_state.last_acked_revocation_sequence,
                    target_host_state_revision,
                )?;
                let host_state_changed = diesel::update(
                    maple_pairing_host_states::table
                        .filter(maple_pairing_host_states::id.eq(host_state.id))
                        .filter(
                            maple_pairing_host_states::revocation_stream_id
                                .eq(highwater.revocation_stream_id),
                        )
                        .filter(
                            maple_pairing_host_states::revocation_stream_generation
                                .eq(highwater.revocation_stream_generation),
                        )
                        .filter(
                            maple_pairing_host_states::last_issued_revocation_sequence
                                .eq(host_state.last_issued_revocation_sequence),
                        )
                        .filter(maple_pairing_host_states::revision.eq(host_state.revision)),
                )
                .set((
                    maple_pairing_host_states::last_issued_revocation_sequence.eq(issuer_sequence),
                    maple_pairing_host_states::revision.eq(target_host_state_revision),
                    maple_pairing_host_states::record_mac.eq(target_host_state_mac),
                ))
                .execute(tx)?;
                if highwater_changed != 1 || host_state_changed != 1 {
                    return Err(DBError::MaplePairingConflict);
                }

                let target_state = MaplePairingState::Revoked.as_db();
                let record_mac = maple_pairing_record_mac_for_parts(
                    &authorization.enclave_key,
                    current.uuid,
                    current.pairing_request_id,
                    current.user_id,
                    current.project_id,
                    current.lineage_id,
                    current.controller_maple_device_id,
                    current.host_maple_device_id,
                    current.direction,
                    current.pairing_incarnation,
                    target_state,
                    target_revision,
                    &current.request_nonce_mac,
                    current.revocation_stream_id,
                    current.revocation_stream_generation,
                    current.pair_authorization_digest.as_deref(),
                    &current.ticket_issuer_key_id,
                    current.authorization_issuer_key_id.as_deref(),
                    Some(&revocation_issuer_key_id),
                    pair_payload_version,
                    &pair_payload_enc,
                    current.created_at,
                    current.expires_at,
                    current.approved_at,
                    current.activated_at,
                    Some(revoked_at),
                )?;
                let pairing = diesel::update(
                    maple_pairings::table
                        .filter(maple_pairings::id.eq(current.id))
                        .filter(maple_pairings::state.eq(current.state))
                        .filter(maple_pairings::revision.eq(mutation.expected_pairing_revision)),
                )
                .set((
                    maple_pairings::state.eq(target_state),
                    maple_pairings::revision.eq(target_revision),
                    maple_pairings::payload_version.eq(pair_payload_version),
                    maple_pairings::payload_enc.eq(pair_payload_enc),
                    maple_pairings::revocation_issuer_key_id.eq(Some(revocation_issuer_key_id)),
                    maple_pairings::record_mac.eq(record_mac),
                    maple_pairings::revoked_at.eq(Some(revoked_at)),
                ))
                .get_result::<MaplePairing>(tx)
                .map_err(|error| match error {
                    diesel::result::Error::NotFound => DBError::MaplePairingConflict,
                    other => DBError::QueryError(other),
                })?;
                validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                if pairing.revocation_stream_id != Some(highwater.revocation_stream_id)
                    || pairing.revocation_stream_generation
                        != Some(highwater.revocation_stream_generation)
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                let receipt = insert_pairing_operation(
                    tx,
                    authorization,
                    actor.id,
                    mutation.operation_id,
                    MAPLE_PAIRING_OPERATION_REVOKE,
                    &mutation.request_mac,
                    &pairing,
                    receipt_version,
                    &receipt_enc,
                    None,
                    revoked_at,
                )?;
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                )?;
                Ok(receipt)
            },
        )
    }

    fn approve_maple_pairing(
        &self,
        mutation: MaplePairingApproval,
    ) -> Result<MaplePairingOperationReceipt, DBError> {
        use crate::models::schema::maple_pairings;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authorization = &mutation.authorization;
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "approve_maple_pairing",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(authorization),
                    true,
                )?;
                let host = find_scoped_maple_device(
                    tx,
                    authorization,
                    mutation.host_registration_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, authorization, &host, true)?;
                if let Some(prior) =
                    get_prior_pairing_operation(tx, authorization, host.id, mutation.operation_id)?
                {
                    return replay_pairing_operation(
                        tx,
                        authorization,
                        &prior,
                        MAPLE_PAIRING_OPERATION_APPROVE,
                        &mutation.request_mac,
                    );
                }
                let (trusted_now, _) = expire_pending_pairings(tx, authorization)?;
                let incarnation = pairing_incarnation_to_i64(mutation.pairing_incarnation)?;
                let expected_generation =
                    i64::try_from(mutation.expected_revocation_stream_generation)
                        .map_err(|_| DBError::MaplePairingConflict)?;
                let approved_at = normalize_db_time(mutation.approved_at)?;
                let current = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .filter(maple_pairings::uuid.eq(mutation.pair_id))
                    .filter(maple_pairings::pairing_request_id.eq(mutation.pairing_request_id))
                    .filter(maple_pairings::host_maple_device_id.eq(host.id))
                    .for_update()
                    .first::<MaplePairing>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingNotFound,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_record(&authorization.enclave_key, &current)?;
                require_maple_pairing_participants_ready(tx, authorization, &current, true)?;
                let target_revision = mutation
                    .expected_pairing_revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                if current.state != MaplePairingState::Pending.as_db()
                    || current.revision != mutation.expected_pairing_revision
                    || current.revision != 1
                    || target_revision != 2
                    || current.pairing_incarnation != incarnation
                    || mutation.expected_revocation_stream_id.is_nil()
                    || expected_generation <= 0
                    || !maple_pairing_approval_is_timely(current.expires_at, trusted_now)
                    || approved_at < current.created_at
                    || !maple_pairing_time_is_near_trusted_now(approved_at, trusted_now)
                    || mutation.request_mac.len() != 32
                    || !maple_pairing_issuer_key_id_is_valid(&mutation.authorization_issuer_key_id)
                    || mutation.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                    || mutation.payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                    || mutation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
                    || mutation.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let (_, highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    host.installation_id,
                    true,
                )?;
                let highwater = highwater.ok_or(DBError::MaplePairingCorrupt)?;
                let host_state = load_maple_pairing_host_state(
                    tx,
                    authorization.user_id,
                    authorization.project_id,
                    host.id,
                    true,
                )?
                .ok_or(DBError::MaplePairingCorrupt)?;
                validate_maple_pairing_host_state(&authorization.enclave_key, &host_state)?;
                if host_state.revocation_stream_id != highwater.revocation_stream_id
                    || host_state.revocation_stream_generation
                        != highwater.revocation_stream_generation
                    || host_state.last_issued_revocation_sequence
                        != highwater.last_issued_revocation_sequence
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                if highwater.revocation_stream_id != mutation.expected_revocation_stream_id
                    || highwater.revocation_stream_generation != expected_generation
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let target_state = MaplePairingState::AwaitingHostCommit.as_db();
                let record_mac = maple_pairing_record_mac_for_parts(
                    &authorization.enclave_key,
                    current.uuid,
                    current.pairing_request_id,
                    current.user_id,
                    current.project_id,
                    current.lineage_id,
                    current.controller_maple_device_id,
                    current.host_maple_device_id,
                    current.direction,
                    current.pairing_incarnation,
                    target_state,
                    target_revision,
                    &current.request_nonce_mac,
                    Some(highwater.revocation_stream_id),
                    Some(highwater.revocation_stream_generation),
                    Some(&mutation.pair_authorization_digest),
                    &current.ticket_issuer_key_id,
                    Some(&mutation.authorization_issuer_key_id),
                    None,
                    mutation.payload_version,
                    &mutation.payload_enc,
                    current.created_at,
                    current.expires_at,
                    Some(approved_at),
                    current.activated_at,
                    current.revoked_at,
                )?;
                let pairing = diesel::update(
                    maple_pairings::table
                        .filter(maple_pairings::id.eq(current.id))
                        .filter(maple_pairings::state.eq(MaplePairingState::Pending.as_db()))
                        .filter(maple_pairings::revision.eq(mutation.expected_pairing_revision)),
                )
                .set((
                    maple_pairings::state.eq(target_state),
                    maple_pairings::revision.eq(target_revision),
                    maple_pairings::payload_version.eq(mutation.payload_version),
                    maple_pairings::payload_enc.eq(mutation.payload_enc.clone()),
                    maple_pairings::revocation_stream_id.eq(Some(highwater.revocation_stream_id)),
                    maple_pairings::revocation_stream_generation
                        .eq(Some(highwater.revocation_stream_generation)),
                    maple_pairings::pair_authorization_digest
                        .eq(Some(mutation.pair_authorization_digest)),
                    maple_pairings::authorization_issuer_key_id
                        .eq(Some(mutation.authorization_issuer_key_id)),
                    maple_pairings::record_mac.eq(record_mac),
                    maple_pairings::approved_at.eq(Some(approved_at)),
                ))
                .get_result::<MaplePairing>(tx)
                .map_err(|error| match error {
                    diesel::result::Error::NotFound => DBError::MaplePairingConflict,
                    other => DBError::QueryError(other),
                })?;
                validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                let receipt = insert_pairing_operation(
                    tx,
                    authorization,
                    host.id,
                    mutation.operation_id,
                    MAPLE_PAIRING_OPERATION_APPROVE,
                    &mutation.request_mac,
                    &pairing,
                    mutation.receipt_version,
                    &mutation.receipt_enc,
                    None,
                    approved_at,
                )?;
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                )?;
                Ok(receipt)
            },
        )
    }

    fn confirm_maple_pairing(
        &self,
        mutation: MaplePairingConfirmation,
    ) -> Result<MaplePairingOperationReceipt, DBError> {
        use crate::models::schema::maple_pairings;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authorization = &mutation.authorization;
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "confirm_maple_pairing",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(authorization),
                    true,
                )?;
                let host = find_scoped_maple_device(
                    tx,
                    authorization,
                    mutation.host_registration_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, authorization, &host, true)?;
                if let Some(prior) =
                    get_prior_pairing_operation(tx, authorization, host.id, mutation.operation_id)?
                {
                    return replay_pairing_operation(
                        tx,
                        authorization,
                        &prior,
                        MAPLE_PAIRING_OPERATION_CONFIRM,
                        &mutation.request_mac,
                    );
                }

                let incarnation = pairing_incarnation_to_i64(mutation.pairing_incarnation)?;
                let activated_at = normalize_db_time(mutation.activated_at)?;
                let trusted_now = maple_pairing_trusted_db_now(tx)?;
                let current = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .filter(maple_pairings::uuid.eq(mutation.pair_id))
                    .filter(maple_pairings::pairing_request_id.eq(mutation.pairing_request_id))
                    .filter(maple_pairings::host_maple_device_id.eq(host.id))
                    .for_update()
                    .first::<MaplePairing>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::MaplePairingNotFound,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_record(&authorization.enclave_key, &current)?;
                require_maple_pairing_participants_ready(tx, authorization, &current, true)?;
                let target_revision = mutation
                    .expected_pairing_revision
                    .checked_add(1)
                    .ok_or(DBError::MaplePairingConflict)?;
                if current.state != MaplePairingState::AwaitingHostCommit.as_db()
                    || current.revision != mutation.expected_pairing_revision
                    || current.revision != 2
                    || target_revision != 3
                    || current.pairing_incarnation != incarnation
                    || current
                        .approved_at
                        .is_none_or(|approved_at| activated_at < approved_at)
                    || !maple_pairing_time_is_near_trusted_now(activated_at, trusted_now)
                    || mutation.request_mac.len() != 32
                    || mutation.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                    || mutation.payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                    || mutation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1
                    || mutation.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let target_state = MaplePairingState::Active.as_db();
                let record_mac = maple_pairing_record_mac_for_parts(
                    &authorization.enclave_key,
                    current.uuid,
                    current.pairing_request_id,
                    current.user_id,
                    current.project_id,
                    current.lineage_id,
                    current.controller_maple_device_id,
                    current.host_maple_device_id,
                    current.direction,
                    current.pairing_incarnation,
                    target_state,
                    target_revision,
                    &current.request_nonce_mac,
                    current.revocation_stream_id,
                    current.revocation_stream_generation,
                    current.pair_authorization_digest.as_deref(),
                    &current.ticket_issuer_key_id,
                    current.authorization_issuer_key_id.as_deref(),
                    current.revocation_issuer_key_id.as_deref(),
                    mutation.payload_version,
                    &mutation.payload_enc,
                    current.created_at,
                    current.expires_at,
                    current.approved_at,
                    Some(activated_at),
                    current.revoked_at,
                )?;
                let pairing = diesel::update(
                    maple_pairings::table
                        .filter(maple_pairings::id.eq(current.id))
                        .filter(
                            maple_pairings::state.eq(MaplePairingState::AwaitingHostCommit.as_db()),
                        )
                        .filter(maple_pairings::revision.eq(mutation.expected_pairing_revision)),
                )
                .set((
                    maple_pairings::state.eq(target_state),
                    maple_pairings::revision.eq(target_revision),
                    maple_pairings::payload_version.eq(mutation.payload_version),
                    maple_pairings::payload_enc.eq(mutation.payload_enc.clone()),
                    maple_pairings::record_mac.eq(record_mac),
                    maple_pairings::activated_at.eq(Some(activated_at)),
                ))
                .get_result::<MaplePairing>(tx)
                .map_err(|error| match error {
                    diesel::result::Error::NotFound => DBError::MaplePairingConflict,
                    other => DBError::QueryError(other),
                })?;
                validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                let receipt = insert_pairing_operation(
                    tx,
                    authorization,
                    host.id,
                    mutation.operation_id,
                    MAPLE_PAIRING_OPERATION_CONFIRM,
                    &mutation.request_mac,
                    &pairing,
                    mutation.receipt_version,
                    &mutation.receipt_enc,
                    None,
                    activated_at,
                )?;
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                )?;
                Ok(receipt)
            },
        )
    }

    fn list_maple_pairings(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        role: MaplePairingRole,
        states: Vec<MaplePairingState>,
        limit: i64,
        after: Option<MaplePairingCursor>,
    ) -> Result<Vec<MaplePairing>, DBError> {
        use crate::models::schema::maple_pairings;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let limit = limit.clamp(1, MAPLE_PAIRING_LIST_QUERY_LIMIT);
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "list_maple_pairings",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(&authorization),
                    true,
                )?;
                let actor =
                    find_scoped_maple_device(tx, &authorization, actor_registration_id, false)?
                        .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, &authorization, &actor, false)?;
                let (_, expired_any) = expire_pending_pairings(tx, &authorization)?;

                let state_values: Vec<i16> =
                    states.into_iter().map(|state| state.as_db()).collect();
                if state_values.is_empty() {
                    if expired_any {
                        commit_maple_pairing_authority_account_mutation(
                            tx,
                            &authorization.enclave_key,
                            authorization.user_id,
                            authorization.project_id,
                        )?;
                    }
                    return Ok(Vec::new());
                }
                // The account/project row count is hard-bounded. Authenticate the
                // complete bounded set before applying participant, state, or
                // cursor projections so a storage-layer edit cannot hide a row by
                // changing one of those MAC-bound fields.
                let rows = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .order(maple_pairings::id.desc())
                    .limit(MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT + 1)
                    .load::<MaplePairing>(tx)?;
                if rows.len()
                    > usize::try_from(MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT)
                        .map_err(|_| DBError::MaplePairingCorrupt)?
                {
                    return Err(DBError::MaplePairingCorrupt);
                }
                for row in &rows {
                    validate_maple_pairing_record(&authorization.enclave_key, row)?;
                }
                let after_internal_id = after
                    .map(|cursor| {
                        rows.iter()
                            .find(|row| row.uuid == cursor.pair_id)
                            .map(|row| row.id)
                            .ok_or(DBError::MaplePairingConflict)
                    })
                    .transpose()?;
                let result: Vec<MaplePairing> = rows
                    .into_iter()
                    .filter(|row| match role {
                        MaplePairingRole::Controller => row.controller_maple_device_id == actor.id,
                        MaplePairingRole::Host => row.host_maple_device_id == actor.id,
                    })
                    .filter(|row| state_values.contains(&row.state))
                    .filter(|row| after_internal_id.is_none_or(|internal_id| row.id < internal_id))
                    .take(usize::try_from(limit).unwrap_or(1))
                    .collect();
                for pairing in &result {
                    require_maple_pairing_participants_ready(tx, &authorization, pairing, false)?;
                }
                if expired_any {
                    commit_maple_pairing_authority_account_mutation(
                        tx,
                        &authorization.enclave_key,
                        authorization.user_id,
                        authorization.project_id,
                    )?;
                }
                Ok(result)
            },
        )
    }

    fn get_maple_pairing(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        pair_id: Uuid,
    ) -> Result<Option<MaplePairing>, DBError> {
        use crate::models::schema::maple_pairings;
        use diesel::OptionalExtension;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "get_maple_pairing",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(&authorization),
                    true,
                )?;
                let Some(actor) =
                    find_scoped_maple_device(tx, &authorization, actor_registration_id, false)?
                else {
                    return Ok(None);
                };
                require_no_pending_reset_clear(tx, &authorization, &actor, false)?;
                let (_, expired_any) = expire_pending_pairings(tx, &authorization)?;
                let row = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .filter(maple_pairings::uuid.eq(pair_id))
                    .first::<MaplePairing>(tx)
                    .optional()?;
                if let Some(row) = row.as_ref() {
                    validate_maple_pairing_record(&authorization.enclave_key, row)?;
                }
                let result = row.filter(|row| {
                    row.controller_maple_device_id == actor.id
                        || row.host_maple_device_id == actor.id
                });
                if let Some(pairing) = result.as_ref() {
                    require_maple_pairing_participants_ready(tx, &authorization, pairing, false)?;
                }
                if expired_any {
                    commit_maple_pairing_authority_account_mutation(
                        tx,
                        &authorization.enclave_key,
                        authorization.user_id,
                        authorization.project_id,
                    )?;
                }
                Ok(result)
            },
        )
    }

    fn audit_maple_pairing_issuer_key_references(
        &self,
        enclave_key: &[u8],
    ) -> Result<Vec<String>, DBError> {
        use crate::models::maple_pairing_db::MaplePairingOperation;
        use crate::models::schema::{
            maple_device_registration_operations, maple_pairing_installation_retirements,
            maple_pairing_issuer_keys, maple_pairing_operations,
            maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_obligations,
            maple_pairing_revocation_events, maple_pairings,
        };
        use diesel::JoinOnDsl;

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReadOnly,
            |tx| {
                let _authority_timer = MaplePairingAuthorityTransactionTimer::start(
                    "audit_maple_pairing_issuer_key_references",
                );
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                // A single authenticated snapshot is required for key retirement.
                // SHARE blocks pairing/event/receipt writes for the short duration
                // of the audit without blocking other readers.
                diesel::sql_query(
                    "LOCK TABLE maple_pairings, maple_pairing_revocation_events, \
                     maple_pairing_operations, maple_device_registration_operations, \
                     maple_pairing_registration_operation_tombstones, \
                     maple_pairing_reset_clear_obligations, \
                     maple_pairing_installation_retirements, \
                     maple_pairing_issuer_keys IN SHARE MODE",
                )
                .execute(tx)?;
                verify_maple_pairing_authority_tree(tx, enclave_key)?;

                let issuer_rows = maple_pairing_issuer_keys::table
                    .order(maple_pairing_issuer_keys::key_id.asc())
                    .load::<MaplePairingIssuerKey>(tx)?;
                if issuer_rows.len() > MAPLE_PAIRING_MAX_ISSUER_KEYS {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                let mut registered_key_ids = BTreeSet::new();
                for row in &issuer_rows {
                    validate_maple_pairing_issuer_key(enclave_key, row)?;
                    if !registered_key_ids.insert(row.key_id.clone()) {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                }

                let mut referenced_key_ids = BTreeSet::new();
                let retain_reference = |referenced_key_ids: &mut BTreeSet<String>, key_id: &str| {
                    if !registered_key_ids.contains(key_id) {
                        return Err(DBError::MaplePairingAuthorityCorrupt);
                    }
                    referenced_key_ids.insert(key_id.to_string());
                    Ok(())
                };
                let mut last_pairing_id = 0_i64;
                loop {
                    let pairings = maple_pairings::table
                        .filter(maple_pairings::id.gt(last_pairing_id))
                        .order(maple_pairings::id.asc())
                        .limit(256)
                        .load::<MaplePairing>(tx)?;
                    if pairings.is_empty() {
                        break;
                    }
                    for pairing in pairings {
                        validate_maple_pairing_record(enclave_key, &pairing)?;
                        retain_reference(&mut referenced_key_ids, &pairing.ticket_issuer_key_id)?;
                        if let Some(key_id) = pairing.authorization_issuer_key_id.as_ref() {
                            retain_reference(&mut referenced_key_ids, key_id)?;
                        }
                        if let Some(key_id) = pairing.revocation_issuer_key_id.as_ref() {
                            retain_reference(&mut referenced_key_ids, key_id)?;
                        }
                        last_pairing_id = pairing.id;
                    }
                }

                let mut last_registration_operation_id = 0_i64;
                loop {
                    let rows = maple_device_registration_operations::table
                        .filter(
                            maple_device_registration_operations::id
                                .gt(last_registration_operation_id),
                        )
                        .order(maple_device_registration_operations::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .load::<MapleDeviceRegistrationOperation>(tx)?;
                    if rows.is_empty() {
                        break;
                    }
                    for row in rows {
                        if !maple_pairing_issuer_key_id_is_valid(&row.sync_issuer_key_id) {
                            return Err(DBError::MaplePairingAuthorityCorrupt);
                        }
                        retain_reference(&mut referenced_key_ids, &row.sync_issuer_key_id)?;
                        last_registration_operation_id = row.id;
                    }
                }

                let mut last_tombstone_id = 0_i64;
                loop {
                    let rows = maple_pairing_registration_operation_tombstones::table
                        .filter(
                            maple_pairing_registration_operation_tombstones::id
                                .gt(last_tombstone_id),
                        )
                        .order(maple_pairing_registration_operation_tombstones::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .load::<MaplePairingRegistrationOperationTombstone>(tx)?;
                    if rows.is_empty() {
                        break;
                    }
                    for row in rows {
                        if !maple_pairing_issuer_key_ids_are_canonical(
                            &row.referenced_issuer_key_ids,
                            4,
                        ) {
                            return Err(DBError::MaplePairingAuthorityCorrupt);
                        }
                        for key_id in &row.referenced_issuer_key_ids {
                            retain_reference(&mut referenced_key_ids, key_id)?;
                        }
                        last_tombstone_id = row.id;
                    }
                }

                let mut last_obligation_id = 0_i64;
                loop {
                    let rows = maple_pairing_reset_clear_obligations::table
                        .filter(maple_pairing_reset_clear_obligations::id.gt(last_obligation_id))
                        .order(maple_pairing_reset_clear_obligations::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .load::<MaplePairingResetClearObligation>(tx)?;
                    if rows.is_empty() {
                        break;
                    }
                    for row in rows {
                        for key_id in [
                            row.signed_instruction_issuer_key_id.as_ref(),
                            row.sync_issuer_key_id.as_ref(),
                            row.ack_receipt_issuer_key_id.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if !maple_pairing_issuer_key_id_is_valid(key_id) {
                                return Err(DBError::MaplePairingAuthorityCorrupt);
                            }
                            retain_reference(&mut referenced_key_ids, key_id)?;
                        }
                        last_obligation_id = row.id;
                    }
                }

                let mut last_retirement_id = 0_i64;
                loop {
                    let rows = maple_pairing_installation_retirements::table
                        .filter(maple_pairing_installation_retirements::id.gt(last_retirement_id))
                        .order(maple_pairing_installation_retirements::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .load::<MaplePairingInstallationRetirement>(tx)?;
                    if rows.is_empty() {
                        break;
                    }
                    for row in rows {
                        if !maple_pairing_issuer_key_id_is_valid(&row.ack_receipt_issuer_key_id) {
                            return Err(DBError::MaplePairingAuthorityCorrupt);
                        }
                        retain_reference(&mut referenced_key_ids, &row.ack_receipt_issuer_key_id)?;
                        last_retirement_id = row.id;
                    }
                }

                let mut last_operation_id = 0_i64;
                loop {
                    let operations =
                        maple_pairing_operations::table
                            .inner_join(maple_pairings::table.on(
                                maple_pairings::id.eq(maple_pairing_operations::maple_pairing_id),
                            ))
                            .filter(maple_pairing_operations::id.gt(last_operation_id))
                            .order(maple_pairing_operations::id.asc())
                            .limit(256)
                            .select((
                                maple_pairing_operations::all_columns,
                                maple_pairings::all_columns,
                            ))
                            .load::<(MaplePairingOperation, MaplePairing)>(tx)?;
                    if operations.is_empty() {
                        break;
                    }
                    for (operation, pairing) in operations {
                        validate_maple_pairing_record(enclave_key, &pairing)?;
                        pairing_operation_receipt(enclave_key, &operation, pairing.uuid)?;
                        if operation.maple_pairing_id != pairing.id
                            || operation.user_id != pairing.user_id
                            || operation.project_id != pairing.project_id
                        {
                            return Err(DBError::MaplePairingCorrupt);
                        }
                        if let Some(key_id) = operation.receipt_issuer_key_id.as_ref() {
                            retain_reference(&mut referenced_key_ids, key_id)?;
                        }
                        last_operation_id = operation.id;
                    }
                }

                let mut last_event_id = 0_i64;
                loop {
                    let events = maple_pairing_revocation_events::table
                        .inner_join(
                            maple_pairings::table.on(maple_pairings::id
                                .eq(maple_pairing_revocation_events::maple_pairing_id)),
                        )
                        .filter(maple_pairing_revocation_events::id.gt(last_event_id))
                        .order(maple_pairing_revocation_events::id.asc())
                        .limit(256)
                        .select((
                            maple_pairing_revocation_events::all_columns,
                            maple_pairings::all_columns,
                        ))
                        .load::<(MaplePairingRevocationEvent, MaplePairing)>(tx)?;
                    if events.is_empty() {
                        break;
                    }
                    for (event, pairing) in events {
                        validate_maple_pairing_revocation_record(enclave_key, &event)?;
                        validate_maple_pairing_record(enclave_key, &pairing)?;
                        if pairing.id != event.maple_pairing_id
                            || pairing.pairing_incarnation != event.pairing_incarnation
                            || pairing.revocation_stream_id != Some(event.revocation_stream_id)
                            || pairing.revocation_stream_generation
                                != Some(event.revocation_stream_generation)
                            || pairing.revocation_issuer_key_id.as_ref()
                                != Some(&event.issuer_key_id)
                        {
                            return Err(DBError::MaplePairingCorrupt);
                        }
                        retain_reference(&mut referenced_key_ids, &event.issuer_key_id)?;
                        last_event_id = event.id;
                    }
                }

                if !referenced_key_ids.is_subset(&registered_key_ids) {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                Ok(referenced_key_ids.into_iter().collect())
            },
        )
    }

    fn replay_maple_reset_clear_ack(
        &self,
        authorization: MaplePairingAuthorization,
        host_registration_id: Uuid,
        operation_id: Uuid,
        request_mac: Vec<u8>,
    ) -> Result<Option<MaplePairingOperationReceipt>, DBError> {
        use crate::models::schema::maple_pairing_authority_account_heads;

        if host_registration_id.is_nil() || operation_id.is_nil() || request_mac.len() != 32 {
            return Err(DBError::MaplePairingConflict);
        }
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReadOnly,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "replay_maple_reset_clear_ack",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(&authorization),
                    false,
                )?;
                let head = maple_pairing_authority_account_heads::table
                    .filter(
                        maple_pairing_authority_account_heads::user_id.eq(authorization.user_id),
                    )
                    .filter(
                        maple_pairing_authority_account_heads::project_id
                            .eq(authorization.project_id),
                    )
                    .for_share()
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                validate_maple_pairing_authority_account_head(&authorization.enclave_key, &head)?;
                replay_maple_reset_clear_ack_in_transaction(
                    tx,
                    &authorization,
                    &head,
                    host_registration_id,
                    operation_id,
                    &request_mac,
                )
            },
        )
    }

    fn replay_maple_pairing_operation(
        &self,
        authorization: MaplePairingAuthorization,
        actor_registration_id: Uuid,
        operation_id: Uuid,
        operation_kind: MaplePairingOperationKind,
        request_mac: Vec<u8>,
    ) -> Result<Option<MaplePairingOperationReceipt>, DBError> {
        if operation_id.is_nil() || request_mac.len() != 32 {
            return Err(DBError::MaplePairingConflict);
        }
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReadOnly,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "replay_maple_pairing_operation",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(&authorization),
                    false,
                )?;
                let actor =
                    find_scoped_maple_device(tx, &authorization, actor_registration_id, false)?
                        .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, &authorization, &actor, false)?;
                let Some(operation) =
                    get_prior_pairing_operation(tx, &authorization, actor.id, operation_id)?
                else {
                    return Ok(None);
                };
                replay_pairing_operation(
                    tx,
                    &authorization,
                    &operation,
                    operation_kind.as_db(),
                    &request_mac,
                )
                .map(Some)
            },
        )
    }

    fn create_maple_pairing(
        &self,
        request: NewMaplePairingRequest,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &MaterializeMaplePairingCreate<'_>,
    ) -> Result<MaplePairingOperationReceipt, DBError> {
        use crate::models::maple_pairing_db::{
            MaplePairingLineage, NewMaplePairing, NewMaplePairingLineage,
            MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST,
        };
        use crate::models::maple_pairings::{
            MaplePairingDirection, MaplePairingState as WireMaplePairingState,
            MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        };
        use crate::models::schema::{maple_pairing_lineages, maple_pairings};
        use diesel::sql_types::BigInt;
        use diesel::OptionalExtension;

        #[derive(diesel::QueryableByName)]
        struct ReservedIncarnation {
            #[diesel(sql_type = BigInt)]
            incarnation: i64,
        }

        let expected_issuer_key_inventory_digest =
            self.require_configured_maple_pairing_issuer_keyset(issuer_keyset)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authorization = &request.authorization;
                let project_identity = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "create_maple_pairing",
                )?;
                lock_maple_user_and_validate_credential(
                    tx,
                    &pairing_authorization_as_device(authorization),
                    true,
                )?;
                let controller = find_scoped_maple_device(
                    tx,
                    authorization,
                    request.controller_registration_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                let host = find_scoped_maple_device(
                    tx,
                    authorization,
                    request.host_registration_id,
                    true,
                )?
                .ok_or(DBError::MaplePairingNotFound)?;
                require_no_pending_reset_clear(tx, authorization, &controller, true)?;
                require_no_pending_reset_clear(tx, authorization, &host, true)?;
                if let Some(prior) = get_prior_pairing_operation(
                    tx,
                    authorization,
                    controller.id,
                    request.operation_id,
                )? {
                    return replay_pairing_operation(
                        tx,
                        authorization,
                        &prior,
                        MAPLE_PAIRING_OPERATION_CREATE,
                        &request.request_mac,
                    );
                }
                let (trusted_now, _) = expire_pending_pairings(tx, authorization)?;

                // Endpoint generations are authorization fences for a fresh
                // create. Compare them while both device rows remain locked so a
                // concurrent device refresh cannot race the route's signed-state
                // validation. Exact accepted operations replay above this check.
                let expected_controller_endpoint_epoch =
                    i64::try_from(request.expected_controller_endpoint_epoch)
                        .map_err(|_| DBError::MaplePairingConflict)?;
                let expected_host_endpoint_epoch =
                    i64::try_from(request.expected_host_endpoint_epoch)
                        .map_err(|_| DBError::MaplePairingConflict)?;
                if controller.endpoint_epoch != expected_controller_endpoint_epoch
                    || host.endpoint_epoch != expected_host_endpoint_epoch
                {
                    return Err(DBError::MaplePairingConflict);
                }

                if request.operation_id.is_nil()
                    || controller.id == host.id
                    || request.request_mac.len() != 32
                    || request.create_request.operation_id != request.operation_id
                    || request.create_request.asserted_account_id != authorization.user_id
                    || request.subject_project_id != project_identity.subject_project_id()
                    || request.create_request.asserted_project_id
                        != project_identity.subject_project_id()
                    || request.create_request.controller_registration_id != controller.uuid
                    || request.create_request.controller_device_id != controller.device_id
                    || request.create_request.controller_installation_id
                        != controller.installation_id
                    || request.create_request.controller_endpoint_epoch
                        != request.expected_controller_endpoint_epoch
                    || request.create_request.host_registration_id != host.uuid
                    || request.create_request.host_device_id != host.device_id
                    || request.create_request.host_installation_id != host.installation_id
                    || request.create_request.host_endpoint_epoch
                        != request.expected_host_endpoint_epoch
                    || request.create_request.direction != MaplePairingDirection::ControllerToHost
                    || request.create_request.execution_target_id != host.uuid
                {
                    return Err(DBError::MaplePairingConflict);
                }

                // Do not allocate an incarnation or invoke the issuer until the
                // database has independently authenticated the exact signed
                // request and rebound both public identities to the locked
                // device rows. Endpoint UUID/epoch equality alone is not an
                // identity proof: a callback can otherwise substitute a fresh
                // valid Ed25519 key while preserving every public identifier.
                request
                    .create_request
                    .validate()
                    .and_then(|_| request.create_request.verify_signature())
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let request_transcript = request
                    .create_request
                    .transcript()
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let expected_request_mac = maple_pairing_request_operation_mac(
                    &authorization.enclave_key,
                    &request_transcript,
                    &request.create_request.signature,
                )
                .map_err(|_| DBError::MaplePairingConflict)?;
                if !maple_pairing_authority_mac_matches(&expected_request_mac, &request.request_mac)
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let controller_request_key = request
                    .create_request
                    .controller_identity_key_bytes()
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let host_request_key = request
                    .create_request
                    .host_identity_key_bytes()
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let controller_request_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &controller_request_key,
                )?;
                let host_request_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &host_request_key,
                )?;
                if !maple_pairing_authority_mac_matches(
                    &controller_request_identity_mac,
                    &controller.identity_mac,
                ) || !maple_pairing_authority_mac_matches(
                    &host_request_identity_mac,
                    &host.identity_mac,
                ) {
                    return Err(DBError::MaplePairingConflict);
                }
                let decoded_nonce = STANDARD
                    .decode(&request.create_request.pairing_request_nonce)
                    .map_err(|_| DBError::MaplePairingConflict)?;
                if decoded_nonce.len() != 32
                    || STANDARD.encode(&decoded_nonce)
                        != request.create_request.pairing_request_nonce
                {
                    return Err(DBError::MaplePairingConflict);
                }
                let request_nonce_mac = maple_pairing_request_nonce_mac(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    controller.uuid,
                    &decoded_nonce,
                )?;

                let pairing_count = maple_pairings::table
                    .filter(maple_pairings::user_id.eq(authorization.user_id))
                    .filter(maple_pairings::project_id.eq(authorization.project_id))
                    .count()
                    .get_result::<i64>(tx)?;
                if pairing_count >= MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT {
                    return Err(DBError::MaplePairingLimitExceeded);
                }

                // PostgreSQL sequences are nontransactional by design. This is
                // the first allocation/signing point: exact replay, participant
                // gates, endpoint/epoch checks, and quotas have all completed.
                // A callback failure leaves only a never-reused gap and no
                // durable or publishable authority artifact.
                let reserved = diesel::sql_query(
                    "SELECT nextval('maple_pairing_incarnation_seq') AS incarnation",
                )
                .get_result::<ReservedIncarnation>(tx)?;
                let incarnation = reserved.incarnation;
                let pairing_incarnation = pairing_u64_from_i64(incarnation)?;
                let created_at = trusted_now;
                let expires_at = created_at
                    .checked_add_signed(chrono::Duration::milliseconds(
                        crate::models::maple_pairings::MAPLE_PAIR_REQUEST_MAX_TTL_MS,
                    ))
                    .ok_or(DBError::MaplePairingConflict)?;
                let request_mac: [u8; 32] = request
                    .request_mac
                    .as_slice()
                    .try_into()
                    .map_err(|_| DBError::MaplePairingConflict)?;
                let device_context = |device: &MapleDevice| -> Result<_, DBError> {
                    Ok(MaplePairingCreateDeviceContext {
                        registration_id: device.uuid,
                        device_id: device.device_id,
                        installation_id: device.installation_id,
                        endpoint_epoch: device
                            .endpoint_epoch
                            .try_into()
                            .map_err(|_| DBError::MaplePairingCorrupt)?,
                        device_revision: device.revision,
                        payload_version: device.payload_version,
                        payload_enc: device.payload_enc.clone(),
                        identity_mac: device.identity_mac.clone(),
                        record_mac: device.record_mac.clone(),
                    })
                };
                let material = materialize(MaplePairingCreateMaterializationContext {
                    account_id: authorization.user_id,
                    subject_project_id: project_identity.subject_project_id(),
                    operation_id: request.operation_id,
                    request_mac,
                    create_request: request.create_request.clone(),
                    controller: device_context(&controller)?,
                    host: device_context(&host)?,
                    pairing_incarnation,
                    created_at,
                    expires_at,
                })
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let MaplePairingCreateMaterial {
                    request_ticket: ticket,
                    response,
                } = material;
                ticket
                    .verify_unexpired(
                        issuer_keyset,
                        trusted_now.timestamp_millis(),
                        MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS,
                    )
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let controller_ticket_key = ticket
                    .controller
                    .verifying_key_bytes()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_ticket_key = ticket
                    .host
                    .verifying_key_bytes()
                    .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let controller_ticket_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &controller_ticket_key,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let host_ticket_identity_mac = maple_device_identity_mac_from_claim(
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                    &host_ticket_key,
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let expected_created_ms = created_at.timestamp_millis();
                let expected_expires_ms = expires_at.timestamp_millis();
                if ticket.subject_account_id != authorization.user_id
                    || ticket.subject_project_id != project_identity.subject_project_id()
                    || ticket.controller_request_operation_id != request.operation_id
                    || ticket.controller_request() != request.create_request
                    || ticket.controller.registration_id != controller.uuid
                    || ticket.controller.device_id != controller.device_id
                    || ticket.controller.installation_id != controller.installation_id
                    || ticket.controller.endpoint_epoch
                        != request.expected_controller_endpoint_epoch
                    || ticket.host.registration_id != host.uuid
                    || ticket.host.device_id != host.device_id
                    || ticket.host.installation_id != host.installation_id
                    || ticket.host.endpoint_epoch != request.expected_host_endpoint_epoch
                    || ticket.pairing_incarnation != pairing_incarnation
                    || ticket.created_at_unix_ms != expected_created_ms
                    || ticket.expires_at_unix_ms != expected_expires_ms
                    || ticket.pairing_request_id.is_nil()
                    || ticket.pair_id.is_nil()
                    || !maple_pairing_authority_mac_matches(
                        &controller_ticket_identity_mac,
                        &controller.identity_mac,
                    )
                    || !maple_pairing_authority_mac_matches(
                        &host_ticket_identity_mac,
                        &host.identity_mac,
                    )
                    || !maple_pairing_issuer_key_id_is_valid(&ticket.issuer_key_id)
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }

                let expected_response = MaplePairingMutationResponse {
                    protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
                    operation_id: request.operation_id,
                    pairing: MaplePairingStatusV1 {
                        pairing_request_id: ticket.pairing_request_id,
                        pair_id: ticket.pair_id,
                        state: WireMaplePairingState::Pending,
                        revision: 1,
                        pairing_incarnation,
                        revocation_stream_id: None,
                        revocation_stream_generation: None,
                        direction: MaplePairingDirection::ControllerToHost,
                        execution_target_id: host.uuid,
                        controller_registration_id: controller.uuid,
                        host_registration_id: host.uuid,
                        created_at_unix_ms: expected_created_ms,
                        expires_at_unix_ms: expected_expires_ms,
                        approved_at_unix_ms: None,
                        activated_at_unix_ms: None,
                        revoked_at_unix_ms: None,
                        request_ticket: Some(ticket.clone()),
                        pair_authorization: None,
                        revocation: None,
                    },
                };
                if response != expected_response {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }
                let payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1;
                let payload_enc = encrypt_maple_pairing_payload(
                    &authorization.enclave_key,
                    &StoredMaplePairingPayloadV1 {
                        request_ticket: ticket.clone(),
                        pair_authorization: None,
                        revocation: None,
                    },
                    MaplePairingPayloadCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        pairing_request_id: ticket.pairing_request_id,
                        pair_id: ticket.pair_id,
                        pairing_incarnation,
                        revocation_stream_id: None,
                        revocation_stream_generation: None,
                        payload_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1;
                let receipt_enc = encrypt_maple_pairing_receipt(
                    &authorization.enclave_key,
                    &response,
                    MaplePairingReceiptCryptoContext {
                        account_id: authorization.user_id,
                        project_id: authorization.project_id,
                        actor_registration_id: controller.uuid,
                        operation_id: request.operation_id,
                        operation_kind: MAPLE_PAIRING_OPERATION_CREATE,
                        pair_id: ticket.pair_id,
                        pairing_revision: 1,
                        receipt_version,
                    },
                )
                .map_err(|_| DBError::MaplePairingMaterializationFailed)?;
                if payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                    || receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES
                {
                    return Err(DBError::MaplePairingMaterializationFailed);
                }

                let lineage = maple_pairing_lineages::table
                    .filter(maple_pairing_lineages::user_id.eq(authorization.user_id))
                    .filter(maple_pairing_lineages::project_id.eq(authorization.project_id))
                    .filter(maple_pairing_lineages::controller_maple_device_id.eq(controller.id))
                    .filter(maple_pairing_lineages::host_maple_device_id.eq(host.id))
                    .filter(
                        maple_pairing_lineages::direction
                            .eq(MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST),
                    )
                    .for_update()
                    .first::<MaplePairingLineage>(tx)
                    .optional()?;

                let lineage = match lineage {
                    Some(lineage) => lineage,
                    None => diesel::insert_into(maple_pairing_lineages::table)
                        .values(NewMaplePairingLineage {
                            user_id: authorization.user_id,
                            project_id: authorization.project_id,
                            controller_maple_device_id: controller.id,
                            host_maple_device_id: host.id,
                            direction: MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST,
                            last_pairing_incarnation: 0,
                        })
                        .get_result::<MaplePairingLineage>(tx)
                        .map_err(|error| match error {
                            diesel::result::Error::DatabaseError(
                                diesel::result::DatabaseErrorKind::UniqueViolation,
                                _,
                            ) => DBError::MaplePairingConflict,
                            other => DBError::QueryError(other),
                        })?,
                };
                if lineage.last_pairing_incarnation >= incarnation {
                    return Err(DBError::MaplePairingConflict);
                }
                let lineage_pairings = maple_pairings::table
                    .filter(maple_pairings::lineage_id.eq(lineage.id))
                    .for_update()
                    .load::<MaplePairing>(tx)?;
                for pairing in &lineage_pairings {
                    validate_maple_pairing_record(&authorization.enclave_key, pairing)?;
                }
                let live_exists = lineage_pairings.iter().any(|pairing| {
                    [
                        MaplePairingState::Pending.as_db(),
                        MaplePairingState::AwaitingHostCommit.as_db(),
                        MaplePairingState::Active.as_db(),
                    ]
                    .contains(&pairing.state)
                });
                if live_exists {
                    return Err(DBError::MaplePairingConflict);
                }
                let updated_lineage = diesel::update(
                    maple_pairing_lineages::table
                        .filter(maple_pairing_lineages::id.eq(lineage.id))
                        .filter(
                            maple_pairing_lineages::last_pairing_incarnation
                                .eq(lineage.last_pairing_incarnation),
                        ),
                )
                .set(maple_pairing_lineages::last_pairing_incarnation.eq(incarnation))
                .execute(tx)?;
                if updated_lineage != 1 {
                    return Err(DBError::MaplePairingConflict);
                }

                let record_mac = maple_pairing_record_mac_for_parts(
                    &authorization.enclave_key,
                    ticket.pair_id,
                    ticket.pairing_request_id,
                    authorization.user_id,
                    authorization.project_id,
                    lineage.id,
                    controller.id,
                    host.id,
                    MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST,
                    incarnation,
                    MaplePairingState::Pending.as_db(),
                    1,
                    &request_nonce_mac,
                    None,
                    None,
                    None,
                    &ticket.issuer_key_id,
                    None,
                    None,
                    payload_version,
                    &payload_enc,
                    created_at,
                    expires_at,
                    None,
                    None,
                    None,
                )?;
                let pairing = diesel::insert_into(maple_pairings::table)
                    .values(NewMaplePairing {
                        uuid: ticket.pair_id,
                        pairing_request_id: ticket.pairing_request_id,
                        user_id: authorization.user_id,
                        project_id: authorization.project_id,
                        lineage_id: lineage.id,
                        controller_maple_device_id: controller.id,
                        host_maple_device_id: host.id,
                        direction: MAPLE_PAIRING_DIRECTION_CONTROLLER_TO_HOST,
                        pairing_incarnation: incarnation,
                        state: MaplePairingState::Pending.as_db(),
                        revision: 1,
                        request_nonce_mac: request_nonce_mac.clone(),
                        revocation_stream_id: None,
                        revocation_stream_generation: None,
                        pair_authorization_digest: None,
                        ticket_issuer_key_id: ticket.issuer_key_id.clone(),
                        authorization_issuer_key_id: None,
                        revocation_issuer_key_id: None,
                        payload_version,
                        payload_enc: payload_enc.clone(),
                        record_mac,
                        created_at,
                        expires_at,
                        approved_at: None,
                        activated_at: None,
                        revoked_at: None,
                    })
                    .get_result::<MaplePairing>(tx)
                    .map_err(|error| match error {
                        diesel::result::Error::DatabaseError(
                            diesel::result::DatabaseErrorKind::UniqueViolation,
                            _,
                        ) => DBError::MaplePairingConflict,
                        other => DBError::QueryError(other),
                    })?;
                validate_maple_pairing_record(&authorization.enclave_key, &pairing)?;
                let receipt = insert_pairing_operation(
                    tx,
                    authorization,
                    controller.id,
                    request.operation_id,
                    MAPLE_PAIRING_OPERATION_CREATE,
                    &request.request_mac,
                    &pairing,
                    receipt_version,
                    &receipt_enc,
                    None,
                    created_at,
                )?;
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &authorization.enclave_key,
                    authorization.user_id,
                    authorization.project_id,
                )?;
                #[cfg(test)]
                if take_maple_pairing_create_before_commit_failure_for_test(request.operation_id) {
                    return Err(DBError::MaplePairingAuthorityBusy);
                }
                Ok(receipt)
            },
        )
    }

    fn mark_password_reset_as_complete(
        &self,
        request: &PasswordResetRequest,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = request.mark_as_reset(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to mark password reset request as complete: {:?}", e);
        }
        result
    }

    fn complete_destructive_password_reset(
        &self,
        user: &User,
        reset_request: &PasswordResetRequest,
        new_password_enc: Vec<u8>,
        new_wrapping: NewUserSeedWrapping,
        enclave_key: &[u8],
        build_reset_clear_material: &BuildResetClearMaterial<'_>,
    ) -> Result<(), DBError> {
        use crate::models::schema::{
            agent_schedule_runs, agent_schedules, agents, conversation_projects,
            conversation_summaries, conversations, memory_blocks, notification_events,
            password_reset_requests, push_devices, user_embeddings, user_instructions, user_kv,
            user_oauth_connections, user_preferences, user_seed_wrappings, users,
        };

        debug!("Completing destructive password reset");
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |conn| {
                let user_id = user.uuid;
                let project_identity = enter_maple_pairing_authority_account_transaction(
                    conn,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                    user_id,
                    user.project_id,
                    "complete_destructive_password_reset",
                )?;
                let locked_user = users::table
                    .filter(users::uuid.eq(user_id))
                    .for_update()
                    .first::<User>(conn)?;
                if locked_user.project_id != user.project_id {
                    return Err(DBError::StaleCredentialState);
                }
                let locked_reset_request = password_reset_requests::table
                    .filter(password_reset_requests::id.eq(reset_request.id))
                    .filter(password_reset_requests::user_id.eq(user_id))
                    .filter(password_reset_requests::is_reset.eq(false))
                    .filter(password_reset_requests::expiration_time.gt(diesel::dsl::now))
                    .for_update()
                    .first::<PasswordResetRequest>(conn)
                    .map_err(|error| match error {
                        diesel::result::Error::NotFound => DBError::PasswordResetRequestNotFound,
                        other => DBError::QueryError(other),
                    })?;

                // Persist and exhaustively validate every unsigned reset-clear
                // instruction before the reset request, credentials, or live
                // authority graph are mutated. The authenticated project-head
                // identity is the sole source of the public project UUID.
                let persisted_reset = persist_maple_pairing_reset_clear_obligations_for_user(
                    conn,
                    enclave_key,
                    user_id,
                    locked_user.project_id,
                    project_identity.subject_project_id(),
                    locked_user.updated_at.max(locked_reset_request.created_at),
                    build_reset_clear_material,
                )?;

                let consumed_reset_count = diesel::update(
                    password_reset_requests::table
                        .filter(password_reset_requests::id.eq(locked_reset_request.id))
                        .filter(password_reset_requests::user_id.eq(user_id))
                        .filter(password_reset_requests::is_reset.eq(false))
                        .filter(password_reset_requests::expiration_time.gt(diesel::dsl::now)),
                )
                .set(password_reset_requests::is_reset.eq(true))
                .execute(conn)?;
                if consumed_reset_count != 1 {
                    return Err(DBError::PasswordResetRequestNotFound);
                }

                diesel::update(
                    password_reset_requests::table
                        .filter(password_reset_requests::user_id.eq(user_id))
                        .filter(password_reset_requests::id.ne(locked_reset_request.id))
                        .filter(password_reset_requests::is_reset.eq(false)),
                )
                .set(password_reset_requests::is_reset.eq(true))
                .execute(conn)?;

                diesel::delete(
                    user_seed_wrappings::table.filter(user_seed_wrappings::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(
                    user_oauth_connections::table
                        .filter(user_oauth_connections::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(user_embeddings::table.filter(user_embeddings::user_id.eq(user_id)))
                    .execute(conn)?;
                diesel::delete(
                    agent_schedule_runs::table.filter(agent_schedule_runs::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(agent_schedules::table.filter(agent_schedules::user_id.eq(user_id)))
                    .execute(conn)?;
                diesel::delete(agents::table.filter(agents::user_id.eq(user_id))).execute(conn)?;
                diesel::delete(
                    notification_events::table.filter(notification_events::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(push_devices::table.filter(push_devices::user_id.eq(user_id)))
                    .execute(conn)?;
                // The persistent obligations/highwaters/tombstones are rooted
                // outside this live graph and survive until an exact clear ACK.
                delete_maple_pairing_state_for_user(conn, user_id)?;
                commit_maple_pairing_authority_account_mutation_with_security_epoch(
                    conn,
                    enclave_key,
                    user_id,
                    locked_user.project_id,
                    Some(persisted_reset.target_security_epoch),
                )?;
                diesel::delete(memory_blocks::table.filter(memory_blocks::user_id.eq(user_id)))
                    .execute(conn)?;
                diesel::delete(
                    user_preferences::table.filter(user_preferences::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(user_kv::table.filter(user_kv::user_id.eq(user_id)))
                    .execute(conn)?;
                diesel::delete(
                    user_instructions::table.filter(user_instructions::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(
                    conversation_projects::table.filter(conversation_projects::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(
                    conversation_summaries::table
                        .filter(conversation_summaries::user_id.eq(user_id)),
                )
                .execute(conn)?;
                diesel::delete(conversations::table.filter(conversations::user_id.eq(user_id)))
                    .execute(conn)?;

                diesel::update(users::table.filter(users::uuid.eq(user_id)))
                    .set((
                        users::password_enc.eq(Some(new_password_enc)),
                        users::updated_at.eq(persisted_reset.reset_at),
                    ))
                    .execute(conn)?;

                if new_wrapping.user_id != user_id {
                    return Err(DBError::StaleCredentialState);
                }
                // Every prior wrapping was deleted above, so insert the one
                // replacement directly with the same clamped lifecycle time
                // used by the reset obligations, tombstones, and user row.
                diesel::insert_into(user_seed_wrappings::table)
                    .values((
                        user_seed_wrappings::user_id.eq(new_wrapping.user_id),
                        user_seed_wrappings::credential_kind.eq(&new_wrapping.credential_kind),
                        user_seed_wrappings::credential_lookup_hash
                            .eq(&new_wrapping.credential_lookup_hash),
                        user_seed_wrappings::wrapping_version.eq(new_wrapping.wrapping_version),
                        user_seed_wrappings::seed_enc.eq(&new_wrapping.seed_enc),
                        user_seed_wrappings::created_at.eq(persisted_reset.reset_at),
                        user_seed_wrappings::updated_at.eq(persisted_reset.reset_at),
                    ))
                    .execute(conn)?;

                Ok(())
            },
        )
    }

    fn register_maple_device(
        &self,
        registration: NewMapleDeviceRegistration,
        issuer_keyset: &MaplePairingIssuerKeySetV1,
        materialize: &MaterializeMapleDeviceRegistrationSync<'_>,
    ) -> Result<MapleDeviceRegistrationReceipt, DBError> {
        use crate::models::schema::{
            maple_device_registration_operations, maple_devices,
            maple_pairing_authority_account_heads, maple_pairing_installation_retirements,
            maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_obligations,
        };
        use diesel::{BoolExpressionMethods, OptionalExtension};
        use subtle::ConstantTimeEq;

        let expected_issuer_key_inventory_digest =
            self.require_configured_maple_pairing_issuer_keyset(issuer_keyset)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let authenticated_project = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &registration.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    registration.user_id,
                    registration.project_id,
                    "register_maple_device",
                )?;
                if authenticated_project.subject_project_id() != registration.subject_project_id {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                // Serializing registration operations per user makes idempotency,
                // compare-and-swap updates, quotas, and credential revocation
                // deterministic under concurrency.
                lock_maple_user_and_validate_credential(
                    tx,
                    &MapleDeviceListAuthorization {
                        user_id: registration.user_id,
                        project_id: registration.project_id,
                        auth_credential_kind: registration.auth_credential_kind.clone(),
                        auth_binding: registration.auth_binding,
                        enclave_key: registration.enclave_key.clone(),
                    },
                    true,
                )?;

                let head = maple_pairing_authority_account_heads::table
                    .filter(maple_pairing_authority_account_heads::user_id.eq(registration.user_id))
                    .filter(
                        maple_pairing_authority_account_heads::project_id
                            .eq(registration.project_id),
                    )
                    .for_update()
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                validate_maple_pairing_authority_account_head(&registration.enclave_key, &head)?;
                let authority_scope_digest = head.authority_scope_digest.clone();
                let lookup_digest = maple_pairing_revocation_highwater_lookup_digest(
                    &registration.enclave_key,
                    registration.user_id,
                    registration.project_id,
                    registration.installation_id,
                )?;
                let operation_lookup_digest = maple_device_registration_operation_lookup_digest(
                    &registration.enclave_key,
                    &authority_scope_digest,
                    registration.operation_id,
                )?;

                // Lifetime replay is resolved before any live-device, epoch,
                // retired-lineage, quota, or materializer gate. A changed body
                // under the same operation identifier remains a conflict.
                let tombstone = maple_pairing_registration_operation_tombstones::table
                    .filter(
                        maple_pairing_registration_operation_tombstones::authority_scope_digest
                            .eq(&authority_scope_digest),
                    )
                    .filter(
                        maple_pairing_registration_operation_tombstones::operation_lookup_digest
                            .eq(&operation_lookup_digest),
                    )
                    .for_share()
                    .first::<MaplePairingRegistrationOperationTombstone>(tx)
                    .optional()?;
                if let Some(tombstone) = tombstone {
                    return replay_maple_device_registration_tombstone(
                        &registration.enclave_key,
                        issuer_keyset,
                        &tombstone,
                        registration.user_id,
                        registration.subject_project_id,
                        &authority_scope_digest,
                        &lookup_digest,
                        head.security_epoch,
                        &registration.request_mac,
                    );
                }

                let prior_operation = maple_device_registration_operations::table
                    .filter(
                        maple_device_registration_operations::authority_scope_digest
                            .eq(&authority_scope_digest),
                    )
                    .filter(
                        maple_device_registration_operations::operation_lookup_digest
                            .eq(&operation_lookup_digest),
                    )
                    .for_share()
                    .first::<MapleDeviceRegistrationOperation>(tx)
                    .optional()?;
                if let Some(prior_operation) = prior_operation {
                    let device = maple_devices::table
                        .filter(maple_devices::id.eq(prior_operation.maple_device_id))
                        .filter(maple_devices::user_id.eq(registration.user_id))
                        .filter(maple_devices::project_id.eq(registration.project_id))
                        .for_share()
                        .first::<MapleDevice>(tx)?;
                    return replay_live_maple_device_registration_operation(
                        &registration.enclave_key,
                        issuer_keyset,
                        &prior_operation,
                        &device,
                        registration.user_id,
                        registration.subject_project_id,
                        registration.project_id,
                        &registration.request_mac,
                    );
                }

                // Exact retries intentionally bypass current-state and quota checks
                // above. Every new operation must supply a complete, internally
                // consistent target so the ciphertext can be AEAD-bound to the
                // caller-selected registration UUID and target revision.
                if registration.operation_id.is_nil()
                    || registration.registration_id.is_nil()
                    || registration.device_id.is_nil()
                    || registration.installation_id.is_nil()
                    || registration.request_mac.len() != 32
                    || registration.identity_mac.len() != 32
                    || registration.payload_enc.len() > MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES
                    || registration.endpoint_epoch < 0
                    || registration.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1
                    || registration.revision <= 0
                    || registration
                        .expected_revision
                        .is_some_and(|revision| revision <= 0)
                {
                    return Err(DBError::MapleDeviceRegistrationConflict);
                }

                // A lineage ACK is terminal for both the stable installation
                // instance, retained endpoint identity, and the retired host
                // registration namespace. This check must dominate stale-epoch
                // and materialization work.
                let host_registration_lookup_digest =
                    maple_reset_clear_ack_host_registration_lookup_digest(
                        &registration.enclave_key,
                        &authority_scope_digest,
                        registration.registration_id,
                    )?;
                let retirement = maple_pairing_installation_retirements::table
                    .filter(
                        maple_pairing_installation_retirements::authority_scope_digest
                            .eq(&authority_scope_digest),
                    )
                    .filter(
                        maple_pairing_installation_retirements::lookup_digest
                            .eq(&lookup_digest)
                            .or(maple_pairing_installation_retirements::host_identity_mac
                                .eq(&registration.identity_mac))
                            .or(maple_pairing_installation_retirements::ack_host_registration_lookup_digest
                                .eq(&host_registration_lookup_digest)),
                    )
                    .for_share()
                    .first::<MaplePairingInstallationRetirement>(tx)
                    .optional()?;
                if let Some(retirement) = retirement {
                    validate_maple_installation_retirement(
                        &registration.enclave_key,
                        &retirement,
                        &authority_scope_digest,
                        head.security_epoch,
                    )?;
                    return Err(DBError::MapleInstallationRetired);
                }
                if registration.known_security_epoch != head.security_epoch {
                    return Err(DBError::MapleDeviceSecurityEpochStale);
                }

                let matching_devices = maple_devices::table
                    .filter(maple_devices::user_id.eq(registration.user_id))
                    .filter(maple_devices::project_id.eq(registration.project_id))
                    .filter(
                        maple_devices::uuid
                            .eq(registration.registration_id)
                            .or(maple_devices::device_id.eq(registration.device_id))
                            .or(maple_devices::installation_id.eq(registration.installation_id))
                            .or(maple_devices::identity_mac.eq(&registration.identity_mac)),
                    )
                    .for_update()
                    .load::<MapleDevice>(tx)?;

                for device in &matching_devices {
                    let expected_record_mac =
                        maple_device_record_mac_for_row(&registration.enclave_key, device)?;
                    let record_matches: bool = expected_record_mac
                        .as_slice()
                        .ct_eq(device.record_mac.as_slice())
                        .into();
                    if !record_matches {
                        return Err(DBError::MapleDeviceRegistrationConflict);
                    }
                }

                // A new accepted operation creates exactly one revision-one
                // row. Existing rows can only replay their accepted operation;
                // endpoint refresh is not an authority-recovery primitive.
                if !matching_devices.is_empty()
                    || registration.expected_revision.is_some()
                    || registration.revision != 1
                {
                    return Err(DBError::MapleDeviceRegistrationConflict);
                }

                let (_, retained_highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &registration.enclave_key,
                    registration.user_id,
                    registration.project_id,
                    registration.installation_id,
                    true,
                )?;
                let pending_recovery = match retained_highwater.as_ref() {
                    Some(highwater) => {
                        let pending = load_latest_pending_maple_reset_clear_obligation(
                            tx,
                            &registration.enclave_key,
                            highwater,
                            true,
                        )?
                        .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                        if !bool::from(
                            pending
                                .host_identity_mac
                                .as_slice()
                                .ct_eq(registration.identity_mac.as_slice()),
                        ) {
                            return Err(DBError::MapleDeviceRegistrationConflict);
                        }
                        Some(pending)
                    }
                    None => {
                        // The identity itself must also be fresh. A Pending
                        // obligation may recover only its original lookup.
                        let retained_identity = maple_pairing_reset_clear_obligations::table
                            .filter(
                                maple_pairing_reset_clear_obligations::authority_scope_digest
                                    .eq(&authority_scope_digest),
                            )
                            .filter(
                                maple_pairing_reset_clear_obligations::host_identity_mac
                                    .eq(&registration.identity_mac),
                            )
                            .select(maple_pairing_reset_clear_obligations::id)
                            .for_share()
                            .first::<i64>(tx)
                            .optional()?;
                        if retained_identity.is_some() {
                            return Err(DBError::MapleDeviceRegistrationConflict);
                        }
                        ensure_maple_pairing_revocation_registration_capacity(
                            tx,
                            &registration.enclave_key,
                            registration.user_id,
                            registration.project_id,
                            registration.installation_id,
                        )?;
                        None
                    }
                };
                let device_count = maple_devices::table
                    .filter(maple_devices::user_id.eq(registration.user_id))
                    .filter(maple_devices::project_id.eq(registration.project_id))
                    .count()
                    .get_result::<i64>(tx)?;
                if device_count >= MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT {
                    return Err(DBError::MapleDeviceLimitExceeded);
                }

                let target_record_mac = maple_device_record_mac_for_registration(&registration)?;
                let device = diesel::insert_into(maple_devices::table)
                    .values(NewMapleDevice {
                        uuid: registration.registration_id,
                        user_id: registration.user_id,
                        project_id: registration.project_id,
                        device_id: registration.device_id,
                        installation_id: registration.installation_id,
                        identity_mac: registration.identity_mac.clone(),
                        endpoint_epoch: registration.endpoint_epoch,
                        payload_version: registration.payload_version,
                        payload_enc: registration.payload_enc.clone(),
                        record_mac: target_record_mac.clone(),
                        revision: registration.revision,
                    })
                    .get_result::<MapleDevice>(tx)
                    .map_err(map_maple_device_write_error)?;

                // Do not let a trigger-modified RETURNING tuple become new trusted
                // enclave state or feed a freshly authenticated receipt.
                if !maple_device_returned_row_matches(
                    &registration,
                    &device,
                    &target_record_mac,
                    None,
                ) {
                    return Err(DBError::MapleDeviceRegistrationConflict);
                }
                // A credential reset removes device-scoped state but deliberately
                // preserves the pseudonymous allocation fence. Re-registering the
                // same stable installation restores both host cursors to that
                // checkpoint before this device can participate in a new pair.
                restore_maple_pairing_host_state_from_highwater(
                    tx,
                    &registration.enclave_key,
                    &device,
                )?;

                let (_, highwater) = load_maple_pairing_revocation_highwater(
                    tx,
                    &registration.enclave_key,
                    registration.user_id,
                    registration.project_id,
                    registration.installation_id,
                    true,
                )?;
                let highwater = highwater.ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                let state = load_maple_pairing_host_state(
                    tx,
                    registration.user_id,
                    registration.project_id,
                    device.id,
                    true,
                )?
                .ok_or(DBError::MaplePairingAuthorityCorrupt)?;
                let pending = load_latest_pending_maple_reset_clear_obligation(
                    tx,
                    &registration.enclave_key,
                    &highwater,
                    true,
                )?;
                if pending_recovery.is_some() != pending.is_some() {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                let prepared = prepare_maple_device_registration_sync(
                    tx,
                    &registration,
                    &device,
                    &highwater,
                    &state,
                    pending,
                    issuer_keyset,
                    materialize,
                )?;

                // PostgreSQL TIMESTAMPTZ has microsecond precision. Normalize
                // and clamp recovery past its reset lifecycle so later
                // retirement timestamps remain truthful under a backward DB
                // clock.
                let mut accepted_at = maple_pairing_trusted_db_now(tx)?;
                if let Some(recovery) = pending_recovery.as_ref() {
                    accepted_at = accepted_at.max(recovery.reset_at);
                }
                let device_summary = MaplePairingAuthorityDeviceSummary::from(&device);
                let receipt_mac = maple_device_registration_operation_receipt_mac_for_parts(
                    &registration.enclave_key,
                    registration.user_id,
                    registration.project_id,
                    registration.operation_id,
                    &registration.request_mac,
                    &device_summary,
                    device.revision,
                    &authority_scope_digest,
                    &lookup_digest,
                    &operation_lookup_digest,
                    registration.known_security_epoch,
                    head.security_epoch,
                    prepared.response_kind,
                    prepared.payload_version,
                    &prepared.payload_enc,
                    &prepared.issuer_key_id,
                    &prepared.digest,
                    accepted_at,
                )?;
                let accepted_operation =
                    diesel::insert_into(maple_device_registration_operations::table)
                        .values(NewMapleDeviceRegistrationOperation {
                            operation_id: registration.operation_id,
                            user_id: registration.user_id,
                            project_id: registration.project_id,
                            request_mac: registration.request_mac.clone(),
                            maple_device_id: device.id,
                            device_revision: device.revision,
                            authority_scope_digest: authority_scope_digest.clone(),
                            lookup_digest: lookup_digest.clone(),
                            operation_lookup_digest: operation_lookup_digest.clone(),
                            known_security_epoch: registration.known_security_epoch,
                            accepted_security_epoch: head.security_epoch,
                            response_kind: prepared.response_kind,
                            sync_payload_version: prepared.payload_version,
                            sync_payload_enc: prepared.payload_enc.clone(),
                            sync_issuer_key_id: prepared.issuer_key_id.clone(),
                            sync_digest: prepared.digest.clone(),
                            receipt_mac: receipt_mac.clone(),
                            accepted_at,
                        })
                        .get_result::<MapleDeviceRegistrationOperation>(tx)
                        .map_err(map_maple_device_write_error)?;

                let request_mac_matches: bool = accepted_operation
                    .request_mac
                    .as_slice()
                    .ct_eq(registration.request_mac.as_slice())
                    .into();
                let receipt_mac_matches: bool = accepted_operation
                    .receipt_mac
                    .as_slice()
                    .ct_eq(receipt_mac.as_slice())
                    .into();
                if !request_mac_matches
                    || !receipt_mac_matches
                    || accepted_operation.id <= 0
                    || accepted_operation.operation_id != registration.operation_id
                    || accepted_operation.user_id != registration.user_id
                    || accepted_operation.project_id != registration.project_id
                    || accepted_operation.maple_device_id != device.id
                    || accepted_operation.device_revision != device.revision
                    || accepted_operation.authority_scope_digest != authority_scope_digest
                    || accepted_operation.lookup_digest != lookup_digest
                    || accepted_operation.operation_lookup_digest != operation_lookup_digest
                    || accepted_operation.known_security_epoch != registration.known_security_epoch
                    || accepted_operation.accepted_security_epoch != head.security_epoch
                    || accepted_operation.response_kind != prepared.response_kind
                    || accepted_operation.sync_payload_version != prepared.payload_version
                    || accepted_operation.sync_payload_enc != prepared.payload_enc
                    || accepted_operation.sync_issuer_key_id != prepared.issuer_key_id
                    || accepted_operation.sync_digest != prepared.digest
                    || accepted_operation.accepted_at != accepted_at
                {
                    return Err(DBError::MapleDeviceRegistrationConflict);
                }

                let receipt = replay_live_maple_device_registration_operation(
                    &registration.enclave_key,
                    issuer_keyset,
                    &accepted_operation,
                    &device,
                    registration.user_id,
                    registration.subject_project_id,
                    registration.project_id,
                    &registration.request_mac,
                )?;
                if receipt.sync_payload != prepared.payload {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                commit_maple_pairing_authority_account_mutation(
                    tx,
                    &registration.enclave_key,
                    registration.user_id,
                    registration.project_id,
                )?;
                #[cfg(test)]
                pause_maple_device_registration_before_commit_if_armed_for_test(
                    registration.operation_id,
                );
                Ok(receipt)
            },
        )
    }

    fn list_maple_devices(
        &self,
        authorization: MapleDeviceListAuthorization,
        limit: i64,
        after: Option<MapleDeviceListCursor>,
    ) -> Result<MapleDeviceListPage, DBError> {
        use crate::models::maple_pairing_db::MaplePairingAuthorityAccountHead;
        use crate::models::schema::{maple_devices, maple_pairing_authority_account_heads};

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let limit = limit.clamp(1, MAPLE_DEVICE_LIST_QUERY_LIMIT);
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReadOnly,
            |tx| {
                let _authority_timer = enter_maple_pairing_authority_account_transaction(
                    tx,
                    &authorization.enclave_key,
                    &expected_issuer_key_inventory_digest,
                    authorization.user_id,
                    authorization.project_id,
                    "list_maple_devices",
                )?;
                lock_maple_user_and_validate_credential(tx, &authorization, false)?;
                let mut query = maple_devices::table
                    .filter(maple_devices::user_id.eq(authorization.user_id))
                    .filter(maple_devices::project_id.eq(authorization.project_id))
                    .into_boxed();

                if let Some(after) = after {
                    query = query.filter(maple_devices::uuid.lt(after.registration_id));
                }

                let devices = query
                    .order(maple_devices::uuid.desc())
                    .limit(limit)
                    .load::<MapleDevice>(tx)
                    .map_err(DBError::from)?;
                let head = maple_pairing_authority_account_heads::table
                    .filter(
                        maple_pairing_authority_account_heads::user_id.eq(authorization.user_id),
                    )
                    .filter(
                        maple_pairing_authority_account_heads::project_id
                            .eq(authorization.project_id),
                    )
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                validate_maple_pairing_authority_account_head(&authorization.enclave_key, &head)?;
                Ok(MapleDeviceListPage {
                    security_epoch: pairing_u64_from_i64(head.security_epoch)?,
                    devices,
                })
            },
        )
    }

    // OAuth Provider method implementations
    fn create_oauth_provider(
        &self,
        new_provider: NewOAuthProvider,
    ) -> Result<OAuthProvider, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_provider.insert(conn).map_err(DBError::from)
    }

    fn get_oauth_provider_by_id(&self, id: i32) -> Result<Option<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_oauth_provider_by_name(&self, name: &str) -> Result<Option<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_by_name(conn, name).map_err(DBError::from)
    }

    fn get_all_oauth_providers(&self) -> Result<Vec<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_all(conn).map_err(DBError::from)
    }

    fn update_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        provider.update(conn).map_err(DBError::from)
    }

    fn delete_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        provider.delete(conn).map_err(DBError::from)
    }

    // User OAuth Connection method implementations
    fn create_user_oauth_connection(
        &self,
        new_connection: NewUserOAuthConnection,
    ) -> Result<UserOAuthConnection, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_connection.insert(conn).map_err(DBError::from)
    }

    fn get_user_oauth_connection_by_id(
        &self,
        id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_user_oauth_connection_by_user_and_provider(
        &self,
        user_id: Uuid,
        provider_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_by_user_and_provider(conn, user_id, provider_id)
            .map_err(DBError::from)
    }

    fn get_project_user_oauth_connection_by_provider_subject(
        &self,
        provider_id: i32,
        provider_user_id: &str,
        project_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_by_provider_subject_and_project(
            conn,
            provider_id,
            provider_user_id,
            project_id,
        )
        .map_err(DBError::from)
    }

    fn get_all_user_oauth_connections_for_user(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_all_for_user(conn, user_id).map_err(DBError::from)
    }

    fn update_user_oauth_connection(
        &self,
        connection: &UserOAuthConnection,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        connection.update(conn).map_err(DBError::from)
    }

    fn delete_user_oauth_connection(
        &self,
        connection: &UserOAuthConnection,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        connection.delete(conn).map_err(DBError::from)
    }

    fn create_token_usage(&self, new_usage: NewTokenUsage) -> Result<TokenUsage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_usage.insert(conn).map_err(DBError::from)
    }

    fn update_user(&self, user: &User) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        user.update(conn).map_err(DBError::from)
    }

    // Org implementations
    fn create_org(&self, new_org: NewOrg, enclave_key: &[u8]) -> Result<Org, DBError> {
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("create_org");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                verify_maple_pairing_authority_global_shallow(tx, enclave_key)?;
                let org = new_org.insert(tx).map_err(DBError::from)?;
                let created_at = maple_pairing_trusted_db_now(tx)?;
                create_maple_pairing_authority_org_head(tx, enclave_key, org.id, created_at)?;
                refresh_maple_pairing_authority_global_head(tx, enclave_key)?;
                verify_maple_pairing_authority_global_shallow(tx, enclave_key)?;
                Ok(org)
            },
        );
        if let Err(ref e) = result {
            error!("Failed to create org: {:?}", e);
        }
        result
    }

    fn get_org_by_id(&self, id: i32) -> Result<Org, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_id(conn, id)?.ok_or(DBError::OrgNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org by ID: {:?}", e);
        }
        result
    }

    fn get_org_by_uuid(&self, uuid: Uuid) -> Result<Org, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_uuid(conn, uuid)?.ok_or(DBError::OrgNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org by UUID: {:?}", e);
        }
        result
    }

    fn get_org_by_name(&self, name: &str) -> Result<Option<Org>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_name(conn, name).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get org by name: {:?}", e);
        }
        result
    }

    fn get_all_orgs(&self) -> Result<Vec<Org>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_all(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all orgs: {:?}", e);
        }
        result
    }

    fn update_org(&self, org: &Org) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = org.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org: {:?}", e);
        }
        result
    }

    fn delete_org(&self, org: &Org, enclave_key: &[u8]) -> Result<(), DBError> {
        use crate::models::schema::{
            maple_pairing_authority_org_heads, maple_pairing_authority_project_heads, org_projects,
            orgs,
        };

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("delete_org");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                let locked_org = orgs::table
                    .filter(orgs::id.eq(org.id))
                    .for_update()
                    .first::<Org>(tx)?;
                verify_maple_pairing_authority_org_chain(tx, enclave_key, locked_org.id, true)?;
                // Phase one spans the complete org subtree. No account, project
                // head, or parent is consumed until every account has passed its
                // authenticated terminal proof.
                let mut project_cursor = 0_i32;
                loop {
                    let projects = org_projects::table
                        .filter(org_projects::org_id.eq(locked_org.id))
                        .filter(org_projects::id.gt(project_cursor))
                        .order(org_projects::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .for_update()
                        .load::<OrgProject>(tx)?;
                    if projects.is_empty() {
                        break;
                    }
                    for project in projects {
                        prove_maple_pairing_authority_accounts_for_project(
                            tx,
                            enclave_key,
                            project.id,
                        )?;
                        project_cursor = project.id;
                    }
                }
                // Phase two consumes the already-proven subtree in deterministic
                // project/account order, without expiry or ancestor recomputation.
                project_cursor = 0;
                loop {
                    let projects = org_projects::table
                        .filter(org_projects::org_id.eq(locked_org.id))
                        .filter(org_projects::id.gt(project_cursor))
                        .order(org_projects::id.asc())
                        .limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)
                        .for_update()
                        .load::<OrgProject>(tx)?;
                    if projects.is_empty() {
                        break;
                    }
                    for project in projects {
                        consume_maple_pairing_authority_accounts_for_project_after_clean_proof(
                            tx,
                            enclave_key,
                            project.id,
                        )?;
                        let removed =
                            diesel::delete(maple_pairing_authority_project_heads::table.filter(
                                maple_pairing_authority_project_heads::project_id.eq(project.id),
                            ))
                            .execute(tx)?;
                        if removed != 1 {
                            return Err(DBError::MaplePairingAuthorityCorrupt);
                        }
                        project_cursor = project.id;
                    }
                }
                let removed = diesel::delete(
                    maple_pairing_authority_org_heads::table
                        .filter(maple_pairing_authority_org_heads::org_id.eq(locked_org.id)),
                )
                .execute(tx)?;
                if removed != 1 {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                locked_org.delete(tx).map_err(DBError::from)?;
                refresh_maple_pairing_authority_global_head(tx, enclave_key)?;
                verify_maple_pairing_authority_global_shallow(tx, enclave_key)
            },
        );
        if let Err(ref e) = result {
            error!("Failed to delete org: {:?}", e);
        }
        result
    }

    // Org project implementations
    fn create_org_project(
        &self,
        new_project: NewOrgProject,
        enclave_key: &[u8],
    ) -> Result<OrgProject, DBError> {
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("create_org_project");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                verify_maple_pairing_authority_org_chain(
                    tx,
                    enclave_key,
                    new_project.org_id,
                    false,
                )?;
                let project = new_project.insert(tx).map_err(DBError::from)?;
                let created_at = maple_pairing_trusted_db_now(tx)?;
                create_maple_pairing_authority_project_head(tx, enclave_key, &project, created_at)?;
                refresh_maple_pairing_authority_org_and_global_heads(
                    tx,
                    enclave_key,
                    project.org_id,
                )?;
                verify_maple_pairing_authority_org_chain(tx, enclave_key, project.org_id, false)?;
                Ok(project)
            },
        );
        if let Err(ref e) = result {
            error!("Failed to create org project: {:?}", e);
        }
        result
    }

    fn get_org_project_by_id(&self, id: i32) -> Result<OrgProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_id(conn, id)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by ID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_uuid(&self, uuid: Uuid) -> Result<OrgProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_uuid(conn, uuid)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by UUID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_client_id(&self, client_id: Uuid) -> Result<OrgProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgProject::get_by_client_id(conn, client_id)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by client ID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_name_and_org(
        &self,
        name: &str,
        org_id: i32,
    ) -> Result<Option<OrgProject>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_name_and_org(conn, name, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get org project by name and org: {:?}", e);
        }
        result
    }

    fn get_all_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org projects for org: {:?}", e);
        }
        result
    }

    fn get_active_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_active_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get active org projects for org: {:?}", e);
        }
        result
    }

    fn update_org_project(&self, project: &OrgProject) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = project.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org project: {:?}", e);
        }
        result
    }

    fn delete_org_project(&self, project: &OrgProject, enclave_key: &[u8]) -> Result<(), DBError> {
        use crate::models::schema::{maple_pairing_authority_project_heads, org_projects};

        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("delete_org_project");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                let locked_project = org_projects::table
                    .filter(org_projects::id.eq(project.id))
                    .for_update()
                    .first::<OrgProject>(tx)?;
                verify_maple_pairing_authority_project_chain(
                    tx,
                    enclave_key,
                    locked_project.id,
                    locked_project.org_id,
                    true,
                )?;
                consume_maple_pairing_authority_accounts_for_project(
                    tx,
                    enclave_key,
                    locked_project.id,
                )?;
                let removed = diesel::delete(maple_pairing_authority_project_heads::table.filter(
                    maple_pairing_authority_project_heads::project_id.eq(locked_project.id),
                ))
                .execute(tx)?;
                if removed != 1 {
                    return Err(DBError::MaplePairingAuthorityCorrupt);
                }
                locked_project.delete(tx).map_err(DBError::from)?;
                refresh_maple_pairing_authority_org_and_global_heads(
                    tx,
                    enclave_key,
                    locked_project.org_id,
                )?;
                verify_maple_pairing_authority_org_chain(
                    tx,
                    enclave_key,
                    locked_project.org_id,
                    false,
                )
            },
        );
        if let Err(ref e) = result {
            error!("Failed to delete org project: {:?}", e);
        }
        result
    }

    // Org project secret implementations
    fn create_org_project_secret(
        &self,
        new_secret: NewOrgProjectSecret,
    ) -> Result<OrgProjectSecret, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_secret.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create org project secret: {:?}", e);
        }
        result
    }

    fn get_org_project_secret_by_id(&self, id: i32) -> Result<OrgProjectSecret, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgProjectSecret::get_by_id(conn, id)?.ok_or(DBError::OrgProjectSecretNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project secret by ID: {:?}", e);
        }
        result
    }

    fn get_org_project_secret_by_key_name_and_project(
        &self,
        key_name: &str,
        project_id: i32,
    ) -> Result<Option<OrgProjectSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProjectSecret::get_by_key_name_and_project(conn, key_name, project_id)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org project secret by key name and project: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_project_secrets_for_project(
        &self,
        project_id: i32,
    ) -> Result<Vec<OrgProjectSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProjectSecret::get_all_for_project(conn, project_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org project secrets for project: {:?}", e);
        }
        result
    }

    fn update_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = secret.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org project secret: {:?}", e);
        }
        result
    }

    fn delete_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = secret.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org project secret: {:?}", e);
        }
        result
    }

    // Invite code implementations
    fn create_invite_code(&self, new_invite: NewInviteCode) -> Result<InviteCode, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_invite.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create invite code: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_id(&self, id: i32) -> Result<InviteCode, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_id(conn, id)?.ok_or(DBError::InviteCodeNotFound);
        if let Err(ref e) = result {
            error!("Failed to get invite code by ID: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_code(&self, code: Uuid) -> Result<InviteCode, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_code(conn, code)?.ok_or(DBError::InviteCodeNotFound);
        if let Err(ref e) = result {
            error!("Failed to get invite code by code: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_email_and_org(
        &self,
        email: &str,
        org_id: i32,
    ) -> Result<Option<InviteCode>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_email_and_org(conn, email, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get invite code by email and org: {:?}", e);
        }
        result
    }

    fn get_all_invite_codes_for_org(&self, org_id: i32) -> Result<Vec<InviteCode>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all invite codes for org: {:?}", e);
        }
        result
    }

    fn mark_invite_code_as_used(&self, invite: &InviteCode) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.mark_as_used(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to mark invite code as used: {:?}", e);
        }
        result
    }

    fn update_invite_code(&self, invite: &InviteCode) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update invite code: {:?}", e);
        }
        result
    }

    fn delete_invite_code(&self, invite: &InviteCode) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete invite code: {:?}", e);
        }
        result
    }

    // Platform user methods
    fn create_platform_user(&self, new_user: NewPlatformUser) -> Result<PlatformUser, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_user.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create platform user: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_id(&self, id: i32) -> Result<PlatformUser, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_id(conn, id)?.ok_or(DBError::PlatformUserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform user by ID: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_uuid(&self, uuid: Uuid) -> Result<PlatformUser, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_uuid(conn, uuid)?.ok_or(DBError::PlatformUserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform user by UUID: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_email(&self, email: &str) -> Result<Option<PlatformUser>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_email(conn, email).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get platform user by email: {:?}", e);
        }
        result
    }

    fn update_platform_user(&self, user: &PlatformUser) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform user: {:?}", e);
        }
        result
    }

    fn update_platform_user_password(
        &self,
        user: &PlatformUser,
        new_password_enc: Vec<u8>,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user
            .update_password(conn, new_password_enc)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform user password: {:?}", e);
        }
        result
    }

    // Org membership methods
    fn create_org_membership(
        &self,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_membership
            .insert(conn)
            .map_err(DBError::OrgMembershipError);
        if let Err(ref e) = result {
            error!("Failed to create org membership: {:?}", e);
        }
        result
    }

    fn get_org_membership_by_platform_user_and_org(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembership, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_by_platform_user_and_org(conn, platform_user_id, org_id)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org membership by platform user and org: {:?}",
                e
            );
        }
        result
    }

    fn get_org_membership_by_platform_user_and_org_with_user(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembershipWithUser, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::get_by_platform_user_and_org_with_user(conn, platform_user_id, org_id)
                .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org membership with user info by platform user and org: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_memberships_for_platform_user(
        &self,
        platform_user_id: Uuid,
    ) -> Result<Vec<OrgMembership>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::get_all_for_platform_user(conn, platform_user_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get all org memberships for platform user: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_memberships_for_org(&self, org_id: i32) -> Result<Vec<OrgMembership>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org memberships for org: {:?}", e);
        }
        result
    }

    fn get_all_org_memberships_with_users_for_org(
        &self,
        org_id: i32,
    ) -> Result<Vec<OrgMembershipWithUser>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_all_with_users_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get all org memberships with users for org: {:?}",
                e
            );
        }
        result
    }

    fn update_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = membership.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org membership: {:?}", e);
        }
        result
    }

    fn delete_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = membership.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org membership: {:?}", e);
        }
        result
    }

    fn update_membership_role(
        &self,
        membership: &mut OrgMembership,
        new_role: OrgRole,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::update_role_with_owner_check(conn, membership, new_role)
            .map_err(|e| match e {
                OrgMembershipError::DatabaseError(diesel::result::Error::RollbackTransaction) => {
                    DBError::OrgMembershipError(OrgMembershipError::DatabaseError(
                        diesel::result::Error::RollbackTransaction,
                    ))
                }
                _ => DBError::from(e),
            });
        if let Err(ref e) = result {
            error!("Failed to update org membership role: {:?}", e);
        }
        result
    }

    fn delete_membership_with_owner_check(
        &self,
        membership: &OrgMembership,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::delete_with_owner_check(conn, membership).map_err(|e| match e {
                OrgMembershipError::DatabaseError(diesel::result::Error::RollbackTransaction) => {
                    DBError::OrgMembershipError(OrgMembershipError::DatabaseError(
                        diesel::result::Error::RollbackTransaction,
                    ))
                }
                _ => DBError::from(e),
            });
        if let Err(ref e) = result {
            error!("Failed to delete org membership: {:?}", e);
        }
        result
    }

    // New project-scoped methods
    fn get_users_for_project(
        &self,
        project_id: i32,
        page: Option<i64>,
        per_page: Option<i64>,
    ) -> Result<(Vec<User>, i64), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Get total count first
        let total = User::get_count_for_project(conn, project_id)?;

        // Default to first page with 10 items per page
        let page = page.unwrap_or(0);
        let per_page = per_page.unwrap_or(10);

        let users = User::get_all_for_project(conn, project_id, page, per_page)?;

        Ok((users, total))
    }

    fn create_org_with_owner(
        &self,
        new_org: NewOrg,
        owner_id: Uuid,
        enclave_key: &[u8],
    ) -> Result<Org, DBError> {
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::NonReplayableMutation,
            |conn| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("create_org_with_owner");
                acquire_maple_pairing_authority_snapshot_fence(
                    conn,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                verify_maple_pairing_authority_global_shallow(conn, enclave_key)?;
                // Create the organization
                let org = new_org.insert(conn).map_err(DBError::from)?;
                let created_at = maple_pairing_trusted_db_now(conn)?;
                create_maple_pairing_authority_org_head(conn, enclave_key, org.id, created_at)?;

                // Create ownership membership
                let new_membership = NewOrgMembership::new(owner_id, org.id, OrgRole::Owner);
                new_membership.insert(conn)?;

                refresh_maple_pairing_authority_global_head(conn, enclave_key)?;
                verify_maple_pairing_authority_global_shallow(conn, enclave_key)?;

                Ok(org)
            },
        )
    }

    fn accept_invite_transaction(
        &self,
        invite: &InviteCode,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        conn.transaction(|conn| {
            // Create the membership
            let membership = new_membership.insert(conn)?;

            // Mark invite as used
            invite.mark_as_used(conn)?;

            Ok(membership)
        })
    }

    // Project settings methods
    fn get_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
    ) -> Result<Option<ProjectSetting>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ProjectSetting::get_by_project_and_category(conn, project_id, category)
            .map_err(DBError::from)
    }

    fn update_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
        settings: serde_json::Value,
    ) -> Result<ProjectSetting, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, category.clone())?
        {
            existing.settings = settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            let new_settings = NewProjectSetting {
                project_id,
                category: category.as_str().to_string(),
                settings,
            };
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    fn get_project_email_settings(
        &self,
        project_id: i32,
    ) -> Result<Option<EmailSettings>, DBError> {
        let settings = self.get_project_settings(project_id, SettingCategory::Email)?;

        match settings {
            Some(s) => s.get_email_settings().map(Some).map_err(DBError::from),
            None => Ok(None),
        }
    }

    fn update_project_email_settings(
        &self,
        project_id: i32,
        settings: EmailSettings,
    ) -> Result<ProjectSetting, DBError> {
        let new_settings = NewProjectSetting::new_email_settings(project_id, settings)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, SettingCategory::Email)?
        {
            existing.settings = new_settings.settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    fn get_project_oauth_settings(
        &self,
        project_id: i32,
    ) -> Result<Option<OAuthSettings>, DBError> {
        let settings = self.get_project_settings(project_id, SettingCategory::OAuth)?;

        match settings {
            Some(s) => s.get_oauth_settings().map(Some).map_err(DBError::from),
            None => Ok(None),
        }
    }

    fn update_project_oauth_settings(
        &self,
        project_id: i32,
        settings: OAuthSettings,
    ) -> Result<ProjectSetting, DBError> {
        let new_settings = NewProjectSetting::new_oauth_settings(project_id, settings)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, SettingCategory::OAuth)?
        {
            existing.settings = new_settings.settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    // Platform email verification implementations
    fn create_platform_email_verification(
        &self,
        new_verification: NewPlatformEmailVerification,
    ) -> Result<PlatformEmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_verification.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create platform email verification: {:?}", e);
        }
        result
    }

    fn get_platform_email_verification_by_id(
        &self,
        id: i32,
    ) -> Result<PlatformEmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_id(conn, id)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform email verification by ID: {:?}", e);
        }
        result
    }

    fn get_platform_email_verification_by_platform_user_id(
        &self,
        platform_user_id: Uuid,
    ) -> Result<PlatformEmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_platform_user_id(conn, platform_user_id)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!(
                "Failed to get platform email verification by platform user ID: {:?}",
                e
            );
        }
        result
    }

    fn get_platform_email_verification_by_code(
        &self,
        code: Uuid,
    ) -> Result<PlatformEmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_verification_code(conn, code)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform email verification by code: {:?}", e);
        }
        result
    }

    fn update_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform email verification: {:?}", e);
        }
        result
    }

    fn delete_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete platform email verification: {:?}", e);
        }
        result
    }

    fn verify_platform_email(
        &self,
        verification: &mut PlatformEmailVerification,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.verify(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to verify platform email: {:?}", e);
        }
        result
    }

    // Platform password reset implementations
    fn create_platform_password_reset_request(
        &self,
        new_request: NewPlatformPasswordResetRequest,
    ) -> Result<PlatformPasswordResetRequest, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_request.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create platform password reset request: {:?}", e);
        }
        result
    }

    fn get_platform_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PlatformPasswordResetRequest>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            PlatformPasswordResetRequest::get_by_user_id_and_code(conn, user_id, &encrypted_code)
                .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get platform password reset request: {:?}", e);
        }
        result
    }

    fn mark_platform_password_reset_as_complete(
        &self,
        request: &PlatformPasswordResetRequest,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = request.mark_as_reset(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to mark platform password reset request as complete: {:?}",
                e
            );
        }
        result
    }

    // Platform invite code implementations
    fn validate_platform_invite_code(&self, code: Uuid) -> Result<PlatformInviteCode, DBError> {
        debug!("Validating platform invite code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformInviteCode::validate_code(conn, code).map_err(|e| match e {
            PlatformInviteCodeError::InviteCodeNotFound(_) => DBError::PlatformInviteCodeNotFound,
            _ => DBError::from(e),
        });
        if let Err(ref e) = result {
            error!("Failed to validate platform invite code: {:?}", e);
        }
        result
    }

    // User API key implementations
    fn create_user_api_key(&self, new_key: NewUserApiKey) -> Result<UserApiKey, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_key.insert(conn).map_err(DBError::from)
    }

    fn get_user_api_key_by_id(&self, id: i32) -> Result<Option<UserApiKey>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserApiKey::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_user_api_key_by_hash(&self, key_hash: &str) -> Result<Option<UserApiKey>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserApiKey::get_by_key_hash(conn, key_hash).map_err(DBError::from)
    }

    fn get_user_by_api_key_hash(&self, key_hash: &str) -> Result<Option<User>, DBError> {
        use crate::models::schema::{user_api_keys, users};
        use diesel::prelude::*;

        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Single JOIN query to get user directly from API key hash
        users::table
            .inner_join(user_api_keys::table.on(users::uuid.eq(user_api_keys::user_id)))
            .filter(user_api_keys::key_hash.eq(key_hash))
            .select(users::all_columns)
            .first::<User>(conn)
            .optional()
            .map_err(DBError::from)
    }

    fn get_all_user_api_keys_for_user(&self, user_id: Uuid) -> Result<Vec<UserApiKey>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserApiKey::get_all_for_user(conn, user_id).map_err(DBError::from)
    }

    fn delete_user_api_key(&self, id: i32, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        // First verify the key belongs to the user
        // Use the same error for both "not found" and "unauthorized" to prevent information disclosure
        match UserApiKey::get_by_id(conn, id)? {
            Some(api_key) if api_key.user_id == user_id => {
                UserApiKey::delete_by_id(conn, id).map_err(DBError::from)
            }
            _ => Err(DBError::UserApiKeyError(UserApiKeyError::NotFound)),
        }
    }

    fn delete_user_api_key_by_name(&self, name: &str, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        // First find the key by name and user_id, then delete it
        // Use the same error for both "not found" and "unauthorized" to prevent information disclosure
        UserApiKey::delete_by_name_and_user(conn, name, user_id).map_err(|e| match e {
            UserApiKeyError::NotFound => DBError::UserApiKeyError(UserApiKeyError::NotFound),
            e => DBError::from(e),
        })
    }

    // Account Deletion implementations
    fn create_account_deletion_request(
        &self,
        new_request: NewAccountDeletionRequest,
    ) -> Result<AccountDeletionRequest, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_request.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create account deletion request: {:?}", e);
        }
        result
    }

    fn get_account_deletion_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<AccountDeletionRequest>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            AccountDeletionRequest::get_by_user_id_and_code(conn, user_id, &encrypted_code)
                .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get account deletion request: {:?}", e);
        }
        result
    }

    fn mark_account_deletion_as_complete(
        &self,
        request: &AccountDeletionRequest,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = request.mark_as_deleted(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to mark account deletion request as complete: {:?}",
                e
            );
        }
        result
    }

    fn delete_user(&self, user: &User, enclave_key: &[u8]) -> Result<(), DBError> {
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("delete_user");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                let locked_user = users::table
                    .filter(users::uuid.eq(user.uuid))
                    .for_update()
                    .first::<User>(tx)?;
                let account = crate::models::schema::maple_pairing_authority_account_heads::table
                    .filter(
                        crate::models::schema::maple_pairing_authority_account_heads::user_id
                            .eq(locked_user.uuid),
                    )
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                verify_maple_pairing_authority_scoped_chain(tx, enclave_key, &account)?;
                let org_id = account.org_id;
                delete_maple_pairing_authority_account_for_final_parent_deletion(
                    tx,
                    enclave_key,
                    locked_user.uuid,
                    locked_user.project_id,
                )?;
                locked_user.delete(tx).map_err(DBError::from)?;
                refresh_maple_pairing_authority_project_and_ancestors(
                    tx,
                    enclave_key,
                    account.project_id,
                    org_id,
                )?;
                verify_maple_pairing_authority_project_chain(
                    tx,
                    enclave_key,
                    account.project_id,
                    org_id,
                    false,
                )
            },
        )
    }

    fn mark_and_delete_user(
        &self,
        user: &User,
        deletion_request: &AccountDeletionRequest,
        enclave_key: &[u8],
    ) -> Result<(), DBError> {
        debug!(
            "Deleting user and marking deletion request as complete for user: {}",
            user.uuid
        );
        let expected_issuer_key_inventory_digest =
            self.configured_maple_pairing_issuer_key_inventory_digest()?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Run both operations in a transaction to ensure atomicity
        run_maple_pairing_authority_transaction(
            conn,
            MaplePairingAuthorityTransactionClass::ReplaySafeMutation,
            |tx| {
                let _timer = MaplePairingAuthorityTransactionTimer::start("mark_and_delete_user");
                acquire_maple_pairing_authority_snapshot_fence(
                    tx,
                    enclave_key,
                    &expected_issuer_key_inventory_digest,
                )?;
                let locked_user = users::table
                    .filter(users::uuid.eq(user.uuid))
                    .for_update()
                    .first::<User>(tx)?;
                let account = crate::models::schema::maple_pairing_authority_account_heads::table
                    .filter(
                        crate::models::schema::maple_pairing_authority_account_heads::user_id
                            .eq(locked_user.uuid),
                    )
                    .first::<MaplePairingAuthorityAccountHead>(tx)?;
                verify_maple_pairing_authority_scoped_chain(tx, enclave_key, &account)?;
                let org_id = account.org_id;
                delete_maple_pairing_authority_account_for_final_parent_deletion(
                    tx,
                    enclave_key,
                    locked_user.uuid,
                    locked_user.project_id,
                )?;

                // Mark complete only after the authenticated terminal predicate
                // has passed, within the same serializable transaction.
                deletion_request
                    .mark_as_deleted(tx)
                    .map_err(DBError::from)?;

                // Then delete the locked user.
                locked_user.delete(tx).map_err(DBError::from)?;
                refresh_maple_pairing_authority_project_and_ancestors(
                    tx,
                    enclave_key,
                    account.project_id,
                    org_id,
                )?;
                verify_maple_pairing_authority_project_chain(
                    tx,
                    enclave_key,
                    account.project_id,
                    org_id,
                    false,
                )
            },
        )
    }

    // ---------- Responses API implementations ----------

    // Conversations
    fn create_conversation(
        &self,
        new_conversation: NewConversation,
    ) -> Result<Conversation, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_conversation.insert(conn).map_err(DBError::from)
    }

    fn create_conversation_project(
        &self,
        new_project: NewConversationProject,
    ) -> Result<ConversationProject, DBError> {
        use crate::models::schema::users;
        use diesel::prelude::*;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        conn.transaction::<ConversationProject, ResponsesError, _>(|tx| {
            users::table
                .filter(users::uuid.eq(new_project.user_id))
                .select(users::id)
                .for_update()
                .first::<i32>(tx)?;

            let existing_project_count =
                ConversationProject::count_for_user(tx, new_project.user_id)?;
            validate_conversation_project_limit(existing_project_count)?;

            new_project.insert(tx)
        })
        .map_err(DBError::from)
    }

    fn get_conversation_by_id_and_user(
        &self,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<Conversation, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::get_by_id_and_user(conn, conversation_id, user_id).map_err(DBError::from)
    }

    fn get_conversation_by_uuid_and_user(
        &self,
        conversation_uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Conversation, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::get_by_uuid_and_user(conn, conversation_uuid, user_id).map_err(DBError::from)
    }

    fn get_conversation_project_by_id_and_user(
        &self,
        project_id: i64,
        user_id: Uuid,
    ) -> Result<ConversationProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ConversationProject::get_by_id_and_user(conn, project_id, user_id).map_err(DBError::from)
    }

    fn get_conversation_project_by_uuid_and_user(
        &self,
        project_uuid: Uuid,
        user_id: Uuid,
    ) -> Result<ConversationProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ConversationProject::get_by_uuid_and_user(conn, project_uuid, user_id)
            .map_err(DBError::from)
    }

    fn update_conversation_metadata(
        &self,
        conversation_id: i64,
        user_id: Uuid,
        metadata_enc: Vec<u8>,
    ) -> Result<Conversation, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::update_metadata(conn, conversation_id, user_id, metadata_enc)
            .map_err(DBError::from)
    }

    fn update_conversation(
        &self,
        conversation_id: i64,
        user_id: Uuid,
        metadata_enc: Option<Vec<u8>>,
        project_id: Option<Option<i64>>,
        is_pinned: Option<bool>,
    ) -> Result<Conversation, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::update(
            conn,
            conversation_id,
            user_id,
            metadata_enc,
            project_id,
            is_pinned,
        )
        .map_err(DBError::from)
    }

    fn batch_update_conversation_project(
        &self,
        conversation_uuids: &[Uuid],
        user_id: Uuid,
        target_project_id: Option<i64>,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::batch_update_project(conn, conversation_uuids, user_id, target_project_id)
            .map_err(DBError::from)
    }

    fn list_conversations(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
        project_filter: ConversationProjectFilter,
        pinned: Option<bool>,
    ) -> Result<Vec<Conversation>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::list_for_user(conn, user_id, limit, after, order, project_filter, pinned)
            .map_err(DBError::from)
    }

    fn list_conversation_projects(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<ConversationProject>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ConversationProject::list_for_user(conn, user_id, limit, after, order)
            .map_err(DBError::from)
    }

    fn update_conversation_project(
        &self,
        project_id: i64,
        user_id: Uuid,
        name_enc: Option<Vec<u8>>,
        instruction_update: ProjectInstructionUpdate,
    ) -> Result<ConversationProject, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::{conversation_projects, user_instructions};
        use diesel::prelude::*;

        conn.transaction(|tx| {
            let target = conversation_projects::table
                .filter(conversation_projects::id.eq(project_id))
                .filter(conversation_projects::user_id.eq(user_id));

            let existing_project = target
                .first::<ConversationProject>(tx)
                .optional()?
                .ok_or(diesel::result::Error::NotFound)?;

            if let Some(name_enc) = name_enc {
                diesel::update(target)
                    .set((
                        conversation_projects::name_enc.eq(name_enc),
                        conversation_projects::updated_at.eq(diesel::dsl::now),
                    ))
                    .execute(tx)?;
            }

            match instruction_update {
                ProjectInstructionUpdate::Unchanged => {}
                ProjectInstructionUpdate::Set {
                    prompt_enc,
                    prompt_tokens,
                } => {
                    let existing_instruction = user_instructions::table
                        .filter(user_instructions::user_id.eq(user_id))
                        .filter(user_instructions::project_id.eq(Some(project_id)))
                        .first::<UserInstruction>(tx)
                        .optional()?;

                    if let Some(instruction) = existing_instruction {
                        diesel::update(
                            user_instructions::table
                                .filter(user_instructions::id.eq(instruction.id)),
                        )
                        .set((
                            user_instructions::prompt_enc.eq(prompt_enc),
                            user_instructions::prompt_tokens.eq(prompt_tokens),
                            user_instructions::is_default.eq(false),
                            user_instructions::updated_at.eq(diesel::dsl::now),
                        ))
                        .execute(tx)?;
                    } else {
                        let new_instruction = NewUserInstruction {
                            uuid: Uuid::new_v4(),
                            user_id,
                            project_id: Some(project_id),
                            name_enc: None,
                            prompt_enc,
                            prompt_tokens,
                            is_default: false,
                        };

                        diesel::insert_into(user_instructions::table)
                            .values(&new_instruction)
                            .execute(tx)?;
                    }

                    diesel::update(target)
                        .set(conversation_projects::updated_at.eq(diesel::dsl::now))
                        .execute(tx)?;
                }
                ProjectInstructionUpdate::Clear => {
                    diesel::delete(
                        user_instructions::table
                            .filter(user_instructions::user_id.eq(user_id))
                            .filter(user_instructions::project_id.eq(Some(project_id))),
                    )
                    .execute(tx)?;

                    diesel::update(target)
                        .set(conversation_projects::updated_at.eq(diesel::dsl::now))
                        .execute(tx)?;
                }
            }

            conversation_projects::table
                .filter(conversation_projects::id.eq(existing_project.id))
                .filter(conversation_projects::user_id.eq(user_id))
                .first::<ConversationProject>(tx)
        })
        .map_err(|e| match e {
            diesel::result::Error::NotFound => {
                DBError::ResponsesError(ResponsesError::ConversationProjectNotFound)
            }
            _ => DBError::ResponsesError(ResponsesError::DatabaseError(e)),
        })
    }

    fn delete_conversation(&self, conversation_id: i64, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::delete_by_id_and_user(conn, conversation_id, user_id).map_err(DBError::from)
    }

    fn delete_all_conversations(&self, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Conversation::delete_all_for_user(conn, user_id).map_err(DBError::from)
    }

    fn delete_conversation_project(&self, project_id: i64, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ConversationProject::delete_by_id_and_user(conn, project_id, user_id).map_err(DBError::from)
    }

    // Responses (job tracker) implementations
    fn create_response(&self, new_response: NewResponse) -> Result<Response, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_response.insert(conn).map_err(DBError::from)
    }

    fn get_response_by_uuid_and_user(
        &self,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Response, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Response::get_by_uuid_and_user(conn, uuid, user_id).map_err(DBError::from)
    }

    fn update_response_status(
        &self,
        id: i64,
        status: ResponseStatus,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Response::update_status(conn, id, status, completed_at).map_err(DBError::from)
    }

    fn update_response_status_if_current(
        &self,
        id: i64,
        current_status: ResponseStatus,
        new_status: ResponseStatus,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<bool, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Response::update_status_if_current(conn, id, current_status, new_status, completed_at)
            .map(|rows| rows > 0)
            .map_err(DBError::from)
    }

    fn cancel_response(&self, uuid: Uuid, user_id: Uuid) -> Result<Response, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Response::cancel_by_uuid_and_user(conn, uuid, user_id).map_err(DBError::from)
    }

    fn delete_response(&self, uuid: Uuid, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        Response::delete_by_uuid_and_user(conn, uuid, user_id).map_err(DBError::from)
    }

    fn create_conversation_with_response_and_message(
        &self,
        conversation_uuid: Uuid,
        user_id: Uuid,
        metadata_enc: Option<Vec<u8>>,
        response: Option<NewResponse>,
        first_message_content: Vec<u8>,
        first_message_tokens: i32,
        message_uuid: Uuid,
        assistant_message_uuid: Option<Uuid>,
    ) -> Result<(Conversation, Option<Response>, UserMessage), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        NewConversation::create_with_response_and_message(
            conn,
            conversation_uuid,
            user_id,
            metadata_enc,
            response,
            first_message_content,
            first_message_tokens,
            message_uuid,
            assistant_message_uuid,
        )
        .map_err(DBError::from)
    }

    // User instructions implementations
    fn get_default_user_instruction(
        &self,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        user_instructions::table
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null())
            .filter(user_instructions::is_default.eq(true))
            .first::<UserInstruction>(conn)
            .optional()
            .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn get_project_instruction(
        &self,
        project_id: i64,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        user_instructions::table
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.eq(Some(project_id)))
            .first::<UserInstruction>(conn)
            .optional()
            .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn get_project_instruction_for_conversation(
        &self,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<Option<UserInstruction>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::{conversations, user_instructions};
        use diesel::prelude::*;

        let project_id = conversations::table
            .filter(conversations::id.eq(conversation_id))
            .filter(conversations::user_id.eq(user_id))
            .select(conversations::project_id)
            .first::<Option<i64>>(conn)
            .optional()
            .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))?
            .ok_or(DBError::ResponsesError(
                ResponsesError::ConversationNotFound,
            ))?;

        match project_id {
            Some(project_id) => user_instructions::table
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.eq(Some(project_id)))
                .first::<UserInstruction>(conn)
                .optional()
                .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e))),
            None => Ok(None),
        }
    }

    fn get_user_instruction_by_uuid_and_user(
        &self,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<UserInstruction, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        user_instructions::table
            .filter(user_instructions::uuid.eq(uuid))
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null())
            .first::<UserInstruction>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => {
                    DBError::ResponsesError(ResponsesError::SystemPromptNotFound)
                }
                _ => DBError::ResponsesError(ResponsesError::DatabaseError(e)),
            })
    }

    fn create_user_instruction(
        &self,
        new_instruction: NewUserInstruction,
    ) -> Result<UserInstruction, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        conn.transaction(|tx| {
            // If this instruction should be default, clear other defaults first
            if new_instruction.is_default {
                diesel::update(
                    user_instructions::table
                        .filter(user_instructions::user_id.eq(new_instruction.user_id))
                        .filter(user_instructions::project_id.is_null())
                        .filter(user_instructions::is_default.eq(true)),
                )
                .set(user_instructions::is_default.eq(false))
                .execute(tx)?;
            }

            diesel::insert_into(user_instructions::table)
                .values(&new_instruction)
                .get_result(tx)
        })
        .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn update_user_instruction(
        &self,
        id: i64,
        user_id: Uuid,
        name_enc: Vec<u8>,
        prompt_enc: Vec<u8>,
        prompt_tokens: i32,
        is_default: bool,
    ) -> Result<UserInstruction, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        conn.transaction(|tx| {
            // If setting this as default, clear other defaults first
            if is_default {
                diesel::update(
                    user_instructions::table
                        .filter(user_instructions::user_id.eq(user_id))
                        .filter(user_instructions::project_id.is_null())
                        .filter(user_instructions::is_default.eq(true))
                        .filter(user_instructions::id.ne(id)),
                )
                .set(user_instructions::is_default.eq(false))
                .execute(tx)?;
            }

            diesel::update(
                user_instructions::table
                    .filter(user_instructions::id.eq(id))
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null()),
            )
            .set((
                user_instructions::name_enc.eq(Some(name_enc)),
                user_instructions::prompt_enc.eq(prompt_enc),
                user_instructions::prompt_tokens.eq(prompt_tokens),
                user_instructions::is_default.eq(is_default),
                user_instructions::updated_at.eq(diesel::dsl::now),
            ))
            .get_result(tx)
        })
        .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn delete_user_instruction(&self, id: i64, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        diesel::delete(
            user_instructions::table
                .filter(user_instructions::id.eq(id))
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.is_null()),
        )
        .execute(conn)
        .map(|rows| {
            if rows == 0 {
                Err(DBError::ResponsesError(
                    ResponsesError::SystemPromptNotFound,
                ))
            } else {
                Ok(())
            }
        })?
    }

    fn list_user_instructions(
        &self,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<UserInstruction>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        let mut query = user_instructions::table
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null())
            .into_boxed();

        if let Some(after_uuid) = after {
            let cursor_instruction = user_instructions::table
                .filter(user_instructions::uuid.eq(after_uuid))
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.is_null())
                .select((user_instructions::updated_at, user_instructions::id))
                .first::<(DateTime<Utc>, i64)>(conn)
                .optional()?;

            if let Some((updated_at, id)) = cursor_instruction {
                if order == "desc" {
                    query = query.filter(
                        user_instructions::updated_at.lt(updated_at).or(
                            user_instructions::updated_at
                                .eq(updated_at)
                                .and(user_instructions::id.lt(id)),
                        ),
                    );
                } else {
                    query = query.filter(
                        user_instructions::updated_at.gt(updated_at).or(
                            user_instructions::updated_at
                                .eq(updated_at)
                                .and(user_instructions::id.gt(id)),
                        ),
                    );
                }
            }
        }

        if order == "desc" {
            query = query.order((
                user_instructions::updated_at.desc(),
                user_instructions::id.desc(),
            ));
        } else {
            query = query.order((
                user_instructions::updated_at.asc(),
                user_instructions::id.asc(),
            ));
        }

        query
            .limit(limit)
            .load::<UserInstruction>(conn)
            .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn set_default_user_instruction(
        &self,
        id: i64,
        user_id: Uuid,
    ) -> Result<UserInstruction, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        use crate::models::schema::user_instructions;
        use diesel::prelude::*;

        conn.transaction(|tx| {
            // Clear all defaults for this user
            diesel::update(
                user_instructions::table
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null())
                    .filter(user_instructions::is_default.eq(true)),
            )
            .set(user_instructions::is_default.eq(false))
            .execute(tx)?;

            // Set this one as default
            diesel::update(
                user_instructions::table
                    .filter(user_instructions::id.eq(id))
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null()),
            )
            .set((
                user_instructions::is_default.eq(true),
                user_instructions::updated_at.eq(diesel::dsl::now),
            ))
            .get_result(tx)
        })
        .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    // User messages
    fn create_user_message(&self, new_msg: NewUserMessage) -> Result<UserMessage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_msg.insert(conn).map_err(DBError::from)
    }

    fn update_user_message_prompt_tokens(
        &self,
        id: i64,
        prompt_tokens: i32,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        use crate::models::schema::user_messages;
        use diesel::prelude::*;

        diesel::update(user_messages::table.filter(user_messages::id.eq(id)))
            .set(user_messages::prompt_tokens.eq(prompt_tokens))
            .execute(conn)
            .map(|_| ())
            .map_err(|e| DBError::ResponsesError(ResponsesError::DatabaseError(e)))
    }

    fn get_user_message(&self, id: i64, user_id: Uuid) -> Result<UserMessage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserMessage::get_by_id_and_user(conn, id, user_id).map_err(DBError::from)
    }

    fn get_user_message_by_uuid(&self, uuid: Uuid, user_id: Uuid) -> Result<UserMessage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserMessage::get_by_uuid_and_user(conn, uuid, user_id).map_err(DBError::from)
    }

    // Assistant messages
    fn create_assistant_message(
        &self,
        new_msg: NewAssistantMessage,
    ) -> Result<AssistantMessage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_msg.insert(conn).map_err(DBError::from)
    }

    fn get_assistant_message_by_uuid(
        &self,
        message_uuid: Uuid,
    ) -> Result<Option<AssistantMessage>, DBError> {
        use crate::models::schema::assistant_messages::dsl::*;
        use diesel::{ExpressionMethods, OptionalExtension, QueryDsl, RunQueryDsl};
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        assistant_messages
            .filter(uuid.eq(message_uuid))
            .first::<AssistantMessage>(conn)
            .optional()
            .map_err(DBError::from)
    }

    fn update_assistant_message(
        &self,
        message_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        completion_tokens: i32,
        status: String,
        finish_reason: Option<String>,
    ) -> Result<AssistantMessage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        AssistantMessage::update(
            conn,
            message_uuid,
            content_enc,
            completion_tokens,
            status,
            finish_reason,
        )
        .map_err(DBError::from)
    }

    // Reasoning items
    fn create_reasoning_item(&self, new_item: NewReasoningItem) -> Result<ReasoningItem, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_item.insert(conn).map_err(DBError::from)
    }

    fn update_reasoning_item(
        &self,
        item_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        reasoning_tokens: i32,
        status: String,
    ) -> Result<ReasoningItem, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ReasoningItem::update(conn, item_uuid, content_enc, reasoning_tokens, status)
            .map_err(DBError::from)
    }

    // Tool calls / outputs
    fn create_tool_call(&self, new_call: NewToolCall) -> Result<ToolCall, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_call.insert(conn).map_err(DBError::from)
    }

    fn get_tool_call_by_uuid(&self, uuid: Uuid, user_id: Uuid) -> Result<ToolCall, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ToolCall::get_by_uuid(conn, uuid, user_id).map_err(DBError::from)
    }

    fn create_tool_output(&self, new_output: NewToolOutput) -> Result<ToolOutput, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_output.insert(conn).map_err(DBError::from)
    }

    // Context reconstruction
    fn get_conversation_context_messages(
        &self,
        conversation_id: i64,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<RawThreadMessage>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        RawThreadMessage::get_conversation_context(conn, conversation_id, limit, after, order)
            .map_err(DBError::from)
    }

    fn get_response_context_messages(
        &self,
        response_id: i64,
    ) -> Result<Vec<RawThreadMessage>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        RawThreadMessage::get_response_context(conn, response_id).map_err(DBError::from)
    }

    // Optimized context reconstruction (metadata-based)
    fn get_conversation_context_metadata(
        &self,
        conversation_id: i64,
    ) -> Result<Vec<RawThreadMessageMetadata>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        RawThreadMessageMetadata::get_conversation_context_metadata(conn, conversation_id)
            .map_err(DBError::from)
    }

    fn get_messages_by_ids(
        &self,
        conversation_id: i64,
        message_ids: &[(String, i64)],
    ) -> Result<Vec<RawThreadMessage>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        RawThreadMessage::get_messages_by_ids(conn, conversation_id, message_ids)
            .map_err(DBError::from)
    }

    fn delete_user_message(&self, id: Uuid, user_id: Uuid) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        // First get the message by UUID to find its ID
        let msg = UserMessage::get_by_uuid_and_user(conn, id, user_id)?;
        UserMessage::delete_by_id_and_user(conn, msg.id, user_id).map_err(DBError::from)
    }

    // Maintenance
}

pub(crate) fn setup_db(url: String) -> Arc<dyn DBConnection + Send + Sync> {
    info!("Connecting to database...");
    let manager = ConnectionManager::<PgConnection>::new(url);

    let pool = Pool::builder()
        .max_size(20) // Increased from 1 to support concurrent operations
        .min_idle(Some(5)) // Keep 5 connections ready for faster response
        .connection_timeout(std::time::Duration::from_secs(30))
        .idle_timeout(Some(std::time::Duration::from_secs(600))) // 10 minutes
        .max_lifetime(Some(std::time::Duration::from_secs(1800))) // 30 minutes
        .test_on_check_out(true)
        .build(manager)
        .expect("Unable to build DB connection pool");

    info!("Connected to database with pool size: 20, min idle: 5");
    Arc::new(PostgresConnection {
        db: pool,
        maple_pairing_issuer_key_inventory_digest: OnceLock::new(),
    })
}
