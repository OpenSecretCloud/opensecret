//! Explicit application projection for the first transport-v2 user slice.
//!
//! This module does not re-enter the transport-v1 router. It validates a small
//! exact operation allowlist, calls shared transport-neutral application
//! functions, and returns plaintext logical results for the gateway to encrypt
//! through the request's original session lease.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::Query;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::response::IntoResponse;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{DateTime, Utc};
use secp256k1::SecretKey;
use serde::de::{DeserializeOwned, Error as _, IgnoredAny, MapAccess, Visitor};
use serde::Deserializer as _;
use serde::Serialize;
use sha2::{Digest, Sha256};
use validator::Validate;
use zeroize::{Zeroize, Zeroizing};

use crate::bounded_json::BoundedJsonBuffer;
use crate::db::DBError;
use crate::email::send_platform_invite_email;
use crate::encrypt::{decrypt_with_key, encrypt_with_key};
use crate::jwt::{
    issue_transport_v2_platform_tokens, issue_transport_v2_user_tokens,
    validate_transport_v2_platform_resumption, validate_transport_v2_user_resumption, AuthContext,
};
use crate::kv::StoreError;
use crate::models::project_settings::{EmailSettings, OAuthSettings};
use crate::models::responses::{ConversationProjectFilter, NewConversation, ResponseStatus};
use crate::provider_cache::{
    derive_tinfoil_cache_namespace, CacheNamespaceRoot, DerivedCacheNamespace,
};
use crate::tokens::count_tokens;
use crate::web::login_routes::{
    authenticate_login, logout_data, password_reset_confirm_data, password_reset_request_data,
    register_and_authenticate, verify_email_data, AuthResponse, Credentials, LogoutRequest,
    PasswordResetConfirmPayload, PasswordResetRequestPayload, RefreshResponse, RegisterCredentials,
};
use crate::web::oauth_routes::{
    apple_native_authenticate, initiate_oauth_data, oauth_callback_authenticate,
    AppleNativeSignInRequest, OAuthAuthRequest, OAuthCallbackRequest,
};
use crate::web::openai::{
    openai_embeddings_v2_data, openai_model_catalog_data, openai_models_v2_data,
    openai_nonstream_chat_completion_v2_data, openai_stream_chat_completion_v2_data,
    openai_transcription_v2_data, openai_tts_v2_data, EmbeddingRequest, TTSRequest,
    TranscriptionRequest,
};
use crate::web::openai_auth::AuthMethod as OpenAiAuthMethod;
use crate::web::platform::common::{
    CreateInviteRequest, CreateOrgRequest, CreateProjectRequest, CreateSecretRequest,
    UpdateEmailSettingsRequest, UpdateMembershipRequest, UpdateOAuthSettingsRequest,
    UpdateProjectRequest,
};
use crate::web::platform::login_routes::{
    authenticate_platform_login, platform_logout_data, platform_password_reset_confirm_data,
    platform_password_reset_request_data, register_platform_user_data, verify_platform_email_data,
    PlatformAuthResponse, PlatformLoginRequest, PlatformLogoutRequest,
    PlatformPasswordResetConfirmPayload, PlatformPasswordResetRequestPayload,
    PlatformRefreshResponse, PlatformRegisterRequest,
};
use crate::web::platform::me_routes::{
    platform_change_password_data, request_platform_verification_data,
    PlatformChangePasswordRequest,
};
use crate::web::protected_routes::{
    confirm_account_deletion_data, create_api_key_data, decrypt_data_value, delete_all_kv_values,
    delete_api_key_by_name, delete_kv_value, encrypt_data_value, initiate_account_deletion_data,
    list_bounded_api_keys_data, map_password_change_error, private_key_bytes_data,
    private_key_data, protected_user_data, public_key_data, put_kv_value,
    request_new_verification_code_data, sign_message_data, third_party_token_data,
    verify_password_change_request, ChangePasswordRequest, ConfirmAccountDeletionRequest,
    CreateApiKeyRequest, DecryptDataRequest, DerivationPathQuery, EncryptDataRequest,
    InitiateAccountDeletionRequest, KvValue, PublicKeyQuery, SignMessageRequest,
    ThirdPartyTokenRequest,
};
use crate::web::responses::conversions::ReasoningContentItem;
use crate::web::responses::handlers::{
    responses_stream_v2_data, ContentPart, InputTokenDetails, OutputItem, OutputTokenDetails,
    ResponseUsage, ResponsesCreateRequest, ResponsesRetrieveResponse,
};
use crate::web::responses::types::ConversationContent;
use crate::web::responses::{
    constants::{
        DEFAULT_PAGINATION_LIMIT, DEFAULT_TOOL_FUNCTION_NAME, MAX_PAGINATION_LIMIT,
        OBJECT_TYPE_CONVERSATION, OBJECT_TYPE_CONVERSATION_DELETED,
        OBJECT_TYPE_CONVERSATION_PROJECT, OBJECT_TYPE_LIST, OBJECT_TYPE_LIST_DELETED,
        OBJECT_TYPE_RESPONSE, ROLE_ASSISTANT, ROLE_USER, STATUS_CANCELLED, STATUS_COMPLETED,
    },
    conversation_projects::{
        create_conversation_project_with_name_data, delete_conversation_project_by_uuid_data,
        validate_project_name, ConversationProjectListItem, ConversationProjectListResponse,
        ConversationProjectResponse, CreateConversationProjectRequest,
        ListConversationProjectsParams, UpdateConversationProjectRequest,
    },
    conversations::{
        validate_metadata, BatchDeleteConversationsRequest, BatchDeleteConversationsResponse,
        BatchDeleteItemResult, BatchUpdateConversationProjectRequest,
        BatchUpdateConversationProjectResponse, ConversationItemListResponse,
        ConversationListResponse, ConversationResponse, CreateConversationRequest,
        ListConversationsParams, ListItemsParams, UpdateConversationRequest,
        MAX_CONVERSATION_BATCH_SIZE,
    },
    instructions::{
        create_instruction_with_content_data, validate_instruction_content,
        CreateInstructionRequest, InstructionListResponse, InstructionResponse,
        ListInstructionsParams, UpdateInstructionRequest,
    },
    ConversationItem, DeletedObjectResponse, MessageContent, NullableField,
};
use crate::web::web_routes::{
    execute_web_extract, execute_web_search, parse_web_extract_request, parse_web_search_request,
    WebRouteError,
};
use crate::{
    ApiError, AppState, VerifiedUserAuthentication, ERROR_CODE_HEADER, ERROR_CONTRACT_HEADER,
};

use super::envelope::{
    decode_canonical_api_key_name_path, decode_canonical_conversation_path,
    decode_canonical_conversation_project_path, decode_canonical_instruction_path,
    decode_canonical_kv_item_path, decode_canonical_platform_resource_path,
    decode_canonical_platform_verify_email_path, decode_canonical_response_path,
    decode_canonical_verify_email_path, ConversationItemPath, Credential, EncodedBytes,
    EnvelopeLimits, HeaderField, InstructionItemPath, LogicalMethod, PlatformResourcePath,
    RequestEnvelope, RequestId, ResponseItemPath, ResponseMode,
};
use super::platform_resources::{self, PlatformResourceError, PlatformResourceKind};
use super::session::{
    AuthenticationReservation, AuthenticationStartError, AuthorityState, BoundAuthority,
    BoundPrincipal,
};
use super::session_cache::V2SessionLease;
use super::stored_conversations::{self, ProjectAssignmentUpdate, StoredConversationError};
use super::stored_resources::{self, StoredResourceError};
use super::streaming::{LogicalStreamResponse, StreamExecutionGuard};

const JSON_CONTENT_TYPE: &[u8] = b"application/json";
const AES_GCM_NONCE_AND_TAG_BYTES: usize = 12 + 16;
const ENCRYPTED_DATA_JSON_OVERHEAD_BYTES: usize = "{\"encrypted_data\":\"".len() + "\"}".len();
const MAX_ENCRYPTED_DATA_BASE64_BYTES: usize =
    ((EnvelopeLimits::RESPONSE.logical_body_bytes - ENCRYPTED_DATA_JSON_OVERHEAD_BYTES) / 4) * 4;
const MAX_ENCRYPTION_PLAINTEXT_BYTES: usize =
    (MAX_ENCRYPTED_DATA_BASE64_BYTES / 4) * 3 - AES_GCM_NONCE_AND_TAG_BYTES;
const MAX_V2_KV_LIST_ROWS: usize = 65_536;
const MAX_V2_API_KEY_LIST_ROWS: usize = 65_536;
const MAX_CONVERSATION_JSON_DEPTH: usize = 64;
const MAX_CONVERSATION_JSON_STRUCTURAL_TOKENS: usize = 131_072;
const MAX_OAUTH_STATE_BYTES: usize = 4 * 1024;
const MAX_OAUTH_CODE_BYTES: usize = 16 * 1024;
const MAX_APPLE_IDENTITY_TOKEN_BYTES: usize = 64 * 1024;
const MAX_APPLE_OPTIONAL_FIELD_BYTES: usize = 4 * 1024;

type SensitiveBytes = Zeroizing<Vec<u8>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JsonShapeError {
    TooLarge,
    Malformed,
}

pub(crate) enum OperationPreparation {
    Unsupported,
    Rejected(LogicalUnaryResponse),
    Ready(UserOperation),
}

pub(crate) enum UserOperation {
    Login {
        body: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
    Register {
        body: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
    Resume {
        credential: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
    OAuthInitiate {
        provider: OAuthProviderName,
        body: SensitiveBytes,
    },
    OAuthCallback {
        provider: OAuthProviderName,
        body: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
    AppleNativeOAuth {
        body: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
    UserPasswordResetRequest {
        body: SensitiveBytes,
    },
    UserPasswordResetConfirm {
        body: SensitiveBytes,
    },
    VerifyEmail {
        code: uuid::Uuid,
    },
    PlatformLogin {
        body: SensitiveBytes,
    },
    PlatformRegister {
        body: SensitiveBytes,
    },
    PlatformResume {
        credential: SensitiveBytes,
    },
    PlatformVerifyEmail {
        code: uuid::Uuid,
    },
    PlatformPasswordResetRequest {
        body: SensitiveBytes,
    },
    PlatformPasswordResetConfirm {
        body: SensitiveBytes,
    },
    PlatformLogout {
        authority: BoundPlatformAuthority,
        body: SensitiveBytes,
    },
    PlatformRequestVerification {
        authority: BoundPlatformAuthority,
    },
    PlatformChangePassword {
        authority: BoundPlatformAuthority,
        body: SensitiveBytes,
    },
    PlatformControl {
        authority: BoundPlatformAuthority,
        operation: PlatformControlOperation,
    },
    Logout {
        body: SensitiveBytes,
    },
    ChangePassword {
        authority: BoundUserAuthority,
        body: SensitiveBytes,
    },
    Protected {
        authority: BoundUserAuthority,
        operation: ProtectedUserOperation,
    },
    Inference {
        authority: InferenceAuthority,
        operation: InferenceOperation,
    },
    Responses {
        authority: BoundUserAuthority,
        body: SensitiveBytes,
        headers: HeaderMap,
    },
}

pub(crate) enum PlatformControlOperation {
    GetMe,
    CreateOrganization {
        body: SensitiveBytes,
    },
    ListOrganizations,
    DeleteOrganization {
        org_id: uuid::Uuid,
    },
    CreateProject {
        org_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    ListProjects {
        org_id: uuid::Uuid,
    },
    GetProject {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
    },
    UpdateProject {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    DeleteProject {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
    },
    CreateSecret {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    ListSecrets {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
    },
    DeleteSecret {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
        key_name: Zeroizing<String>,
    },
    GetEmailSettings {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
    },
    UpdateEmailSettings {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    GetOAuthSettings {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
    },
    UpdateOAuthSettings {
        org_id: uuid::Uuid,
        project_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    ListMemberships {
        org_id: uuid::Uuid,
    },
    UpdateMembership {
        org_id: uuid::Uuid,
        user_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    DeleteMembership {
        org_id: uuid::Uuid,
        user_id: uuid::Uuid,
    },
    CreateInvite {
        org_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    ListInvites {
        org_id: uuid::Uuid,
    },
    GetInvite {
        org_id: uuid::Uuid,
        invite_code: uuid::Uuid,
    },
    DeleteInvite {
        org_id: uuid::Uuid,
        invite_code: uuid::Uuid,
    },
    AcceptInvite {
        invite_code: uuid::Uuid,
    },
}

impl PlatformControlOperation {
    const fn requires_stored_output_reservation(&self) -> bool {
        matches!(
            self,
            Self::GetMe
                | Self::ListOrganizations
                | Self::ListProjects { .. }
                | Self::GetProject { .. }
                | Self::UpdateProject { .. }
                | Self::ListSecrets { .. }
                | Self::GetEmailSettings { .. }
                | Self::GetOAuthSettings { .. }
                | Self::ListMemberships { .. }
                | Self::UpdateMembership { .. }
                | Self::ListInvites { .. }
                | Self::GetInvite { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OAuthProviderName {
    Github,
    Google,
    Apple,
}

impl OAuthProviderName {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Github => "github",
            Self::Google => "google",
            Self::Apple => "apple",
        }
    }
}

pub(crate) enum InferenceOperation {
    Models,
    ModelCatalog,
    Chat {
        body: SensitiveBytes,
        headers: HeaderMap,
        stream: bool,
    },
    TextToSpeech {
        body: SensitiveBytes,
    },
    Transcription {
        body: SensitiveBytes,
    },
    Embeddings {
        body: SensitiveBytes,
    },
    WebSearch {
        body: SensitiveBytes,
    },
    WebExtract {
        body: SensitiveBytes,
    },
}

impl InferenceOperation {
    const fn is_streaming(&self) -> bool {
        matches!(self, Self::Chat { stream: true, .. })
    }
}

pub(crate) enum InferenceAuthority {
    Public,
    User(BoundUserAuthority),
    ApiKey(BoundApiKeyAuthority),
    AuthenticateApiKey {
        credential: SensitiveBytes,
        cache_namespace_root: CacheNamespaceRoot,
    },
}

#[derive(Clone)]
pub(crate) struct BoundApiKeyAuthority {
    api_key_id: i32,
    user_id: uuid::Uuid,
    cache_namespace: DerivedCacheNamespace,
}

pub(crate) enum ProtectedUserOperation {
    GetUser,
    RequestVerification,
    GetPrivateKey {
        query: Option<String>,
    },
    GetPrivateKeyBytes {
        query: Option<String>,
    },
    GetPublicKey {
        query: Option<String>,
    },
    SignMessage {
        body: SensitiveBytes,
    },
    IssueThirdPartyToken {
        body: SensitiveBytes,
    },
    EncryptData {
        body: SensitiveBytes,
    },
    DecryptData {
        body: SensitiveBytes,
    },
    PutKv {
        key: Zeroizing<String>,
        body: SensitiveBytes,
    },
    GetKv {
        key: Zeroizing<String>,
    },
    ListKv,
    DeleteKv {
        key: Zeroizing<String>,
    },
    DeleteAllKv,
    CreateApiKey {
        body: SensitiveBytes,
    },
    ListApiKeys,
    DeleteApiKey {
        name: Zeroizing<String>,
    },
    CreateConversationProject {
        body: SensitiveBytes,
    },
    ListConversationProjects {
        query: Option<String>,
    },
    GetConversationProject {
        project_id: uuid::Uuid,
    },
    UpdateConversationProject {
        project_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    DeleteConversationProject {
        project_id: uuid::Uuid,
    },
    CreateInstruction {
        body: SensitiveBytes,
    },
    ListInstructions {
        query: Option<String>,
    },
    GetInstruction {
        instruction_id: uuid::Uuid,
    },
    UpdateInstruction {
        instruction_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    DeleteInstruction {
        instruction_id: uuid::Uuid,
    },
    SetDefaultInstruction {
        instruction_id: uuid::Uuid,
    },
    CreateConversation {
        body: SensitiveBytes,
    },
    ListConversations {
        query: Option<String>,
    },
    GetConversation {
        conversation_id: uuid::Uuid,
    },
    UpdateConversation {
        conversation_id: uuid::Uuid,
        body: SensitiveBytes,
    },
    DeleteConversation {
        conversation_id: uuid::Uuid,
    },
    DeleteAllConversations,
    BatchDeleteConversations {
        body: SensitiveBytes,
    },
    BatchUpdateConversationProject {
        body: SensitiveBytes,
    },
    ListConversationItems {
        conversation_id: uuid::Uuid,
        query: Option<String>,
    },
    GetConversationItem {
        conversation_id: uuid::Uuid,
        item_id: uuid::Uuid,
    },
    GetStoredResponse {
        response_id: uuid::Uuid,
    },
    CancelStoredResponse {
        response_id: uuid::Uuid,
    },
    DeleteStoredResponse {
        response_id: uuid::Uuid,
    },
    RequestAccountDeletion {
        body: SensitiveBytes,
    },
    ConfirmAccountDeletion {
        body: SensitiveBytes,
    },
}

impl ProtectedUserOperation {
    const fn session_effect_on_success(&self) -> SessionEffect {
        if matches!(self, Self::ConfirmAccountDeletion { .. }) {
            SessionEffect::Close
        } else {
            SessionEffect::Retain
        }
    }
}

#[derive(Clone)]
pub(crate) struct BoundUserAuthority {
    user_id: uuid::Uuid,
    project_id: i32,
    auth_context: AuthContext,
    cache_namespace: DerivedCacheNamespace,
}

#[derive(Clone)]
pub(crate) struct BoundPlatformAuthority {
    platform_user_id: uuid::Uuid,
}

impl UserOperation {
    pub(crate) const fn requires_authentication_transition(&self) -> bool {
        matches!(
            self,
            Self::Login { .. }
                | Self::Register { .. }
                | Self::Resume { .. }
                | Self::OAuthCallback { .. }
                | Self::AppleNativeOAuth { .. }
                | Self::PlatformLogin { .. }
                | Self::PlatformRegister { .. }
                | Self::PlatformResume { .. }
                | Self::Inference {
                    authority: InferenceAuthority::AuthenticateApiKey { .. },
                    ..
                }
        )
    }

    pub(crate) const fn requires_provider_output_reservation(&self) -> bool {
        matches!(
            self,
            Self::Inference { .. }
                | Self::Responses { .. }
                | Self::OAuthCallback { .. }
                | Self::AppleNativeOAuth { .. }
        )
    }

    pub(crate) const fn is_streaming(&self) -> bool {
        matches!(
            self,
            Self::Inference {
                operation: InferenceOperation::Chat { stream: true, .. },
                ..
            } | Self::Responses { .. }
        )
    }

    pub(crate) const fn requires_stored_output_reservation(&self) -> bool {
        matches!(
            self,
            Self::Protected {
                operation: ProtectedUserOperation::GetKv { .. }
                    | ProtectedUserOperation::ListKv
                    | ProtectedUserOperation::ListApiKeys
                    | ProtectedUserOperation::ListConversationProjects { .. }
                    | ProtectedUserOperation::GetConversationProject { .. }
                    | ProtectedUserOperation::UpdateConversationProject { .. }
                    | ProtectedUserOperation::ListInstructions { .. }
                    | ProtectedUserOperation::GetInstruction { .. }
                    | ProtectedUserOperation::UpdateInstruction { .. }
                    | ProtectedUserOperation::SetDefaultInstruction { .. }
                    | ProtectedUserOperation::ListConversations { .. }
                    | ProtectedUserOperation::GetConversation { .. }
                    | ProtectedUserOperation::UpdateConversation { .. }
                    | ProtectedUserOperation::ListConversationItems { .. }
                    | ProtectedUserOperation::GetConversationItem { .. }
                    | ProtectedUserOperation::GetStoredResponse { .. }
                    | ProtectedUserOperation::CancelStoredResponse { .. },
                ..
            }
        ) || matches!(
            self,
            Self::PlatformControl { operation, .. }
                if operation.requires_stored_output_reservation()
        )
    }

    const fn session_effect_on_success(&self) -> SessionEffect {
        match self {
            Self::Logout { .. }
            | Self::ChangePassword { .. }
            | Self::PlatformLogout { .. }
            | Self::PlatformChangePassword { .. } => SessionEffect::Close,
            Self::Protected { operation, .. } => operation.session_effect_on_success(),
            Self::PlatformControl { .. } => SessionEffect::Retain,
            Self::Inference { .. } => SessionEffect::Retain,
            Self::Responses { .. } => SessionEffect::Retain,
            _ => SessionEffect::Retain,
        }
    }
}

pub(crate) struct LogicalUnaryResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: Vec<HeaderField>,
    pub(crate) body: Option<Zeroizing<Vec<u8>>>,
}

#[derive(Serialize)]
struct ConversationProjectCreationResponsePreflight<'a> {
    id: uuid::Uuid,
    object: &'static str,
    name: &'a str,
    instructions: Option<&'a str>,
    created_at: i64,
    updated_at: i64,
}

#[derive(Serialize)]
struct InstructionResponsePreflight<'a> {
    id: uuid::Uuid,
    object: &'static str,
    name: &'a str,
    prompt: &'a str,
    prompt_tokens: i32,
    is_default: bool,
    created_at: i64,
    updated_at: i64,
}

#[derive(Serialize)]
struct ConversationResponsePreflight<'a> {
    id: uuid::Uuid,
    object: &'static str,
    metadata: Option<&'a serde_json::Value>,
    project_id: Option<uuid::Uuid>,
    pinned: bool,
    created_at: i64,
    last_activity_at: i64,
}

#[derive(Serialize)]
struct CancelledResponsePreflight<'a> {
    id: uuid::Uuid,
    object: &'static str,
    created_at: i64,
    status: &'static str,
    model: &'a str,
    usage: Option<serde_json::Value>,
    output: &'static [serde_json::Value],
}

fn preflight_conversation_project_creation_response(name: &str) -> Result<(), ApiError> {
    preflight_conversation_project_response_with_limit(
        name,
        None,
        EnvelopeLimits::RESPONSE.logical_body_bytes,
    )
}

fn preflight_conversation_project_response(
    name: &str,
    instructions: Option<&str>,
) -> Result<(), ApiError> {
    preflight_conversation_project_response_with_limit(
        name,
        instructions,
        EnvelopeLimits::RESPONSE.logical_body_bytes,
    )
}

fn preflight_conversation_project_response_with_limit(
    name: &str,
    instructions: Option<&str>,
    logical_body_bytes: usize,
) -> Result<(), ApiError> {
    let candidate = ConversationProjectCreationResponsePreflight {
        id: uuid::Uuid::nil(),
        object: OBJECT_TYPE_CONVERSATION_PROJECT,
        name,
        instructions,
        created_at: i64::MIN,
        updated_at: i64::MIN,
    };
    // Creation assigns the UUID and timestamps in PostgreSQL. Prove before
    // insertion that the request-derived response fits; the actual UUID has
    // the same width and real timestamps are shorter. Drop the bounded copy
    // before entering storage so the preflight does not inflate the mutation.
    let response =
        LogicalUnaryResponse::json_with_limit(StatusCode::OK, &candidate, logical_body_bytes)?;
    drop(response);
    Ok(())
}

fn preflight_instruction_response(
    name: &str,
    prompt: &str,
    is_default: bool,
) -> Result<(), ApiError> {
    preflight_instruction_response_with_limit(
        name,
        prompt,
        is_default,
        EnvelopeLimits::RESPONSE.logical_body_bytes,
    )
}

fn preflight_instruction_response_with_limit(
    name: &str,
    prompt: &str,
    is_default: bool,
    logical_body_bytes: usize,
) -> Result<(), ApiError> {
    let candidate = InstructionResponsePreflight {
        id: uuid::Uuid::nil(),
        object: "instruction",
        name,
        prompt,
        prompt_tokens: i32::MIN,
        is_default,
        created_at: i64::MIN,
        updated_at: i64::MIN,
    };
    let response =
        LogicalUnaryResponse::json_with_limit(StatusCode::OK, &candidate, logical_body_bytes)?;
    drop(response);
    Ok(())
}

fn preflight_conversation_response(
    metadata: Option<&serde_json::Value>,
    project_id: Option<uuid::Uuid>,
    pinned: bool,
) -> Result<(), ApiError> {
    let candidate = ConversationResponsePreflight {
        id: uuid::Uuid::nil(),
        object: OBJECT_TYPE_CONVERSATION,
        metadata,
        project_id,
        pinned,
        created_at: i64::MIN,
        last_activity_at: i64::MIN,
    };
    let response = LogicalUnaryResponse::json(StatusCode::OK, &candidate)?;
    drop(response);
    Ok(())
}

fn preflight_batch_delete_conversations_response() -> Result<(), ApiError> {
    let candidate = BatchDeleteConversationsResponse {
        object: OBJECT_TYPE_LIST,
        data: (0..MAX_CONVERSATION_BATCH_SIZE)
            .map(|_| BatchDeleteItemResult {
                id: uuid::Uuid::nil(),
                object: OBJECT_TYPE_CONVERSATION_DELETED,
                deleted: false,
                error: Some("delete_failed"),
            })
            .collect(),
    };
    let response = LogicalUnaryResponse::json(StatusCode::OK, &candidate)?;
    drop(response);
    Ok(())
}

fn preflight_cancelled_response(model: &str) -> Result<(), ApiError> {
    let candidate = CancelledResponsePreflight {
        id: uuid::Uuid::nil(),
        object: OBJECT_TYPE_RESPONSE,
        created_at: i64::MIN,
        status: STATUS_CANCELLED,
        model,
        usage: None,
        output: &[],
    };
    let response = LogicalUnaryResponse::json(StatusCode::OK, &candidate)?;
    drop(response);
    Ok(())
}

impl LogicalUnaryResponse {
    pub(crate) fn json<T: Serialize>(status: StatusCode, value: &T) -> Result<Self, ApiError> {
        Self::json_with_limit(status, value, EnvelopeLimits::default().logical_body_bytes)
    }

    fn json_with_limit<T: Serialize>(
        status: StatusCode,
        value: &T,
        logical_body_bytes: usize,
    ) -> Result<Self, ApiError> {
        let mut buffer = BoundedJsonBuffer::new(logical_body_bytes);
        if let Err(error) = serde_json::to_writer(&mut buffer, value) {
            if buffer.exceeded() {
                return Err(ApiError::PayloadTooLarge);
            }
            tracing::error!(
                "Could not serialize transport-v2 logical response: {:?}",
                error
            );
            return Err(ApiError::InternalServerError);
        }
        let body = buffer.into_bytes();
        Ok(Self {
            status,
            headers: vec![HeaderField {
                name: "content-type".to_owned(),
                value_base64: EncodedBytes::from_bytes(JSON_CONTENT_TYPE.to_vec()),
            }],
            body: Some(body),
        })
    }

    pub(crate) fn api_error(error: ApiError) -> Self {
        match Self::json(error.status_code(), &error.response_body()) {
            Ok(mut response) => {
                let api_response = error.into_response();
                for name in [ERROR_CONTRACT_HEADER, ERROR_CODE_HEADER] {
                    if let Some(value) = api_response.headers().get(name) {
                        response.headers.push(HeaderField {
                            name: name.to_owned(),
                            value_base64: EncodedBytes::from_bytes(value.as_bytes().to_vec()),
                        });
                    }
                }
                response
            }
            Err(_) => Self::protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "Internal server error",
            ),
        }
    }

    pub(crate) fn protocol_error(status: StatusCode, code: &str, message: &str) -> Self {
        #[derive(Serialize)]
        struct ErrorBody<'a> {
            error: ErrorDetails<'a>,
        }

        #[derive(Serialize)]
        struct ErrorDetails<'a> {
            code: &'a str,
            message: &'a str,
        }

        let body = serde_json::to_vec(&ErrorBody {
            error: ErrorDetails { code, message },
        })
        .expect("fixed transport-v2 error body must serialize");
        Self {
            status,
            headers: vec![HeaderField {
                name: "content-type".to_owned(),
                value_base64: EncodedBytes::from_bytes(JSON_CONTENT_TYPE.to_vec()),
            }],
            body: Some(Zeroizing::new(body)),
        }
    }
}

pub(crate) enum LogicalApplicationResponse {
    Unary(LogicalUnaryResponse),
    Stream(LogicalStreamResponse),
}

pub(crate) struct ApplicationOutcome {
    pub(crate) response: LogicalApplicationResponse,
    pub(crate) session_effect: SessionEffect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SessionEffect {
    Retain,
    NewlyBound,
    Close,
}

impl ApplicationOutcome {
    fn success(response: LogicalUnaryResponse, session_effect: SessionEffect) -> Self {
        Self {
            response: LogicalApplicationResponse::Unary(response),
            session_effect,
        }
    }

    fn stream(response: LogicalStreamResponse, session_effect: SessionEffect) -> Self {
        Self {
            response: LogicalApplicationResponse::Stream(response),
            session_effect,
        }
    }

    fn error(error: ApiError) -> Self {
        Self {
            response: LogicalApplicationResponse::Unary(LogicalUnaryResponse::api_error(error)),
            session_effect: SessionEffect::Retain,
        }
    }

    fn closing_error(error: ApiError) -> Self {
        Self {
            response: LogicalApplicationResponse::Unary(LogicalUnaryResponse::api_error(error)),
            session_effect: SessionEffect::Close,
        }
    }

    fn error_with_effect(error: ApiError, session_effect: SessionEffect) -> Self {
        Self {
            response: LogicalApplicationResponse::Unary(LogicalUnaryResponse::api_error(error)),
            session_effect,
        }
    }
}

pub(crate) fn prepare_user_operation(
    envelope: RequestEnvelope,
    authority: AuthorityState,
) -> OperationPreparation {
    let RequestEnvelope {
        response_mode,
        credential,
        cache_namespace_root_base64,
        mut request,
        ..
    } = envelope;

    #[derive(Clone, Copy)]
    enum Route {
        Login,
        Register,
        Resume,
        GithubOAuthInitiate,
        GithubOAuthCallback,
        GoogleOAuthInitiate,
        GoogleOAuthCallback,
        AppleOAuthInitiate,
        AppleOAuthCallback,
        AppleNativeOAuth,
        UserPasswordResetRequest,
        UserPasswordResetConfirm,
        VerifyEmail,
        PlatformLogin,
        PlatformRegister,
        PlatformResume,
        PlatformVerifyEmail,
        PlatformPasswordResetRequest,
        PlatformPasswordResetConfirm,
        PlatformLogout,
        PlatformRequestVerification,
        PlatformChangePassword,
        PlatformMe,
        CreatePlatformOrganization,
        ListPlatformOrganizations,
        PlatformResource,
        GetUser,
        RequestVerification,
        GetPrivateKey,
        GetPrivateKeyBytes,
        GetPublicKey,
        SignMessage,
        IssueThirdPartyToken,
        EncryptData,
        DecryptData,
        GetKv,
        ListKv,
        PutKv,
        DeleteKv,
        DeleteAllKv,
        CreateApiKey,
        ListApiKeys,
        DeleteApiKey,
        CreateConversationProject,
        ListConversationProjects,
        GetConversationProject,
        UpdateConversationProject,
        DeleteConversationProject,
        CreateInstruction,
        ListInstructions,
        GetInstruction,
        UpdateInstruction,
        DeleteInstruction,
        SetDefaultInstruction,
        CreateConversation,
        ListConversations,
        GetConversation,
        UpdateConversation,
        DeleteConversation,
        DeleteAllConversations,
        BatchDeleteConversations,
        BatchUpdateConversationProject,
        ListConversationItems,
        GetConversationItem,
        GetStoredResponse,
        CancelStoredResponse,
        DeleteStoredResponse,
        RequestAccountDeletion,
        ConfirmAccountDeletion,
        Logout,
        ChangePassword,
        Models,
        ModelCatalog,
        ChatCompletions,
        ResponsesCreate,
        TextToSpeech,
        Transcription,
        Embeddings,
        WebSearch,
        WebExtract,
    }

    let kv_key = match decode_canonical_kv_item_path(request.method, &request.path) {
        Ok(key) => key,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let api_key_name = match decode_canonical_api_key_name_path(request.method, &request.path) {
        Ok(name) => name,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let verification_code = match decode_canonical_verify_email_path(request.method, &request.path)
    {
        Ok(code) => code,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let platform_verification_code =
        match decode_canonical_platform_verify_email_path(request.method, &request.path) {
            Ok(code) => code,
            Err(_) => {
                request.path.zeroize();
                return rejected_bad_request();
            }
        };
    let platform_resource_path =
        match decode_canonical_platform_resource_path(request.method, &request.path) {
            Ok(path) => path,
            Err(_) => {
                request.path.zeroize();
                return rejected_bad_request();
            }
        };
    let conversation_project_id =
        match decode_canonical_conversation_project_path(request.method, &request.path) {
            Ok(project_id) => project_id,
            Err(_) => {
                request.path.zeroize();
                return rejected_bad_request();
            }
        };
    let instruction_path = match decode_canonical_instruction_path(request.method, &request.path) {
        Ok(path) => path,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let conversation_path = match decode_canonical_conversation_path(request.method, &request.path)
    {
        Ok(path) => path,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let response_path = match decode_canonical_response_path(request.method, &request.path) {
        Ok(path) => path,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let route = match (request.method, request.path.as_str()) {
        (LogicalMethod::Post, "/login") => Some(Route::Login),
        (LogicalMethod::Post, "/register") => Some(Route::Register),
        (LogicalMethod::Post, "/refresh") => Some(Route::Resume),
        (LogicalMethod::Post, "/auth/github") => Some(Route::GithubOAuthInitiate),
        (LogicalMethod::Post, "/auth/github/callback") => Some(Route::GithubOAuthCallback),
        (LogicalMethod::Post, "/auth/google") => Some(Route::GoogleOAuthInitiate),
        (LogicalMethod::Post, "/auth/google/callback") => Some(Route::GoogleOAuthCallback),
        (LogicalMethod::Post, "/auth/apple") => Some(Route::AppleOAuthInitiate),
        (LogicalMethod::Post, "/auth/apple/callback") => Some(Route::AppleOAuthCallback),
        (LogicalMethod::Post, "/auth/apple/native") => Some(Route::AppleNativeOAuth),
        (LogicalMethod::Post, "/password-reset/request") => Some(Route::UserPasswordResetRequest),
        (LogicalMethod::Post, "/password-reset/confirm") => Some(Route::UserPasswordResetConfirm),
        (LogicalMethod::Get, _) if verification_code.is_some() => Some(Route::VerifyEmail),
        (LogicalMethod::Post, "/platform/login") => Some(Route::PlatformLogin),
        (LogicalMethod::Post, "/platform/register") => Some(Route::PlatformRegister),
        (LogicalMethod::Post, "/platform/refresh") => Some(Route::PlatformResume),
        (LogicalMethod::Get, _) if platform_verification_code.is_some() => {
            Some(Route::PlatformVerifyEmail)
        }
        (LogicalMethod::Post, "/platform/password-reset/request") => {
            Some(Route::PlatformPasswordResetRequest)
        }
        (LogicalMethod::Post, "/platform/password-reset/confirm") => {
            Some(Route::PlatformPasswordResetConfirm)
        }
        (LogicalMethod::Post, "/platform/logout") => Some(Route::PlatformLogout),
        (LogicalMethod::Post, "/platform/request_verification") => {
            Some(Route::PlatformRequestVerification)
        }
        (LogicalMethod::Post, "/platform/change-password") => Some(Route::PlatformChangePassword),
        (LogicalMethod::Get, "/platform/me") => Some(Route::PlatformMe),
        (LogicalMethod::Post, "/platform/orgs") => Some(Route::CreatePlatformOrganization),
        (LogicalMethod::Get, "/platform/orgs") => Some(Route::ListPlatformOrganizations),
        (LogicalMethod::Get, "/protected/user") => Some(Route::GetUser),
        (LogicalMethod::Post, "/protected/request_verification") => {
            Some(Route::RequestVerification)
        }
        (LogicalMethod::Get, "/protected/private_key") => Some(Route::GetPrivateKey),
        (LogicalMethod::Get, "/protected/private_key_bytes") => Some(Route::GetPrivateKeyBytes),
        (LogicalMethod::Get, "/protected/public_key") => Some(Route::GetPublicKey),
        (LogicalMethod::Post, "/protected/sign_message") => Some(Route::SignMessage),
        (LogicalMethod::Post, "/protected/third_party_token") => Some(Route::IssueThirdPartyToken),
        (LogicalMethod::Post, "/protected/encrypt") => Some(Route::EncryptData),
        (LogicalMethod::Post, "/protected/decrypt") => Some(Route::DecryptData),
        (LogicalMethod::Get, "/protected/kv") => Some(Route::ListKv),
        (LogicalMethod::Delete, "/protected/kv") => Some(Route::DeleteAllKv),
        (LogicalMethod::Post, "/protected/api-keys") => Some(Route::CreateApiKey),
        (LogicalMethod::Get, "/protected/api-keys") => Some(Route::ListApiKeys),
        (LogicalMethod::Post, "/v1/conversation-projects") => {
            Some(Route::CreateConversationProject)
        }
        (LogicalMethod::Get, "/v1/conversation-projects") => Some(Route::ListConversationProjects),
        (LogicalMethod::Post, "/v1/instructions") => Some(Route::CreateInstruction),
        (LogicalMethod::Get, "/v1/instructions") => Some(Route::ListInstructions),
        (LogicalMethod::Post, "/v1/conversations") => Some(Route::CreateConversation),
        (LogicalMethod::Get, "/v1/conversations") => Some(Route::ListConversations),
        (LogicalMethod::Delete, "/v1/conversations") => Some(Route::DeleteAllConversations),
        (LogicalMethod::Post, "/v1/conversations/batch-delete") => {
            Some(Route::BatchDeleteConversations)
        }
        (LogicalMethod::Post, "/v1/conversations/batch-update-project") => {
            Some(Route::BatchUpdateConversationProject)
        }
        (LogicalMethod::Post, "/protected/delete-account/request") => {
            Some(Route::RequestAccountDeletion)
        }
        (LogicalMethod::Post, "/protected/delete-account/confirm") => {
            Some(Route::ConfirmAccountDeletion)
        }
        (LogicalMethod::Post, "/protected/change_password") => Some(Route::ChangePassword),
        (LogicalMethod::Post, "/logout") => Some(Route::Logout),
        (LogicalMethod::Get, "/v1/models") => Some(Route::Models),
        (LogicalMethod::Get, "/v1/models/catalog") => Some(Route::ModelCatalog),
        (LogicalMethod::Post, "/v1/chat/completions") => Some(Route::ChatCompletions),
        (LogicalMethod::Post, "/v1/responses") => Some(Route::ResponsesCreate),
        (LogicalMethod::Post, "/v1/audio/speech") => Some(Route::TextToSpeech),
        (LogicalMethod::Post, "/v1/audio/transcriptions") => Some(Route::Transcription),
        (LogicalMethod::Post, "/v1/embeddings") => Some(Route::Embeddings),
        (LogicalMethod::Post, "/v1/web/search") => Some(Route::WebSearch),
        (LogicalMethod::Post, "/v1/web/extract") => Some(Route::WebExtract),
        (LogicalMethod::Get, _) if kv_key.is_some() => Some(Route::GetKv),
        (LogicalMethod::Put, _) if kv_key.is_some() => Some(Route::PutKv),
        (LogicalMethod::Delete, _) if kv_key.is_some() => Some(Route::DeleteKv),
        (LogicalMethod::Delete, _) if api_key_name.is_some() => Some(Route::DeleteApiKey),
        (LogicalMethod::Delete, _) if conversation_project_id.is_some() => {
            Some(Route::DeleteConversationProject)
        }
        (LogicalMethod::Get, _) if conversation_project_id.is_some() => {
            Some(Route::GetConversationProject)
        }
        (LogicalMethod::Post, _) if conversation_project_id.is_some() => {
            Some(Route::UpdateConversationProject)
        }
        (LogicalMethod::Get, _)
            if matches!(instruction_path, Some(InstructionItemPath::Item(_))) =>
        {
            Some(Route::GetInstruction)
        }
        (LogicalMethod::Post, _)
            if matches!(instruction_path, Some(InstructionItemPath::Item(_))) =>
        {
            Some(Route::UpdateInstruction)
        }
        (LogicalMethod::Delete, _)
            if matches!(instruction_path, Some(InstructionItemPath::Item(_))) =>
        {
            Some(Route::DeleteInstruction)
        }
        (LogicalMethod::Post, _)
            if matches!(instruction_path, Some(InstructionItemPath::SetDefault(_))) =>
        {
            Some(Route::SetDefaultInstruction)
        }
        (LogicalMethod::Get, _)
            if matches!(
                conversation_path,
                Some(ConversationItemPath::Conversation(_))
            ) =>
        {
            Some(Route::GetConversation)
        }
        (LogicalMethod::Post, _)
            if matches!(
                conversation_path,
                Some(ConversationItemPath::Conversation(_))
            ) =>
        {
            Some(Route::UpdateConversation)
        }
        (LogicalMethod::Delete, _)
            if matches!(
                conversation_path,
                Some(ConversationItemPath::Conversation(_))
            ) =>
        {
            Some(Route::DeleteConversation)
        }
        (LogicalMethod::Get, _)
            if matches!(conversation_path, Some(ConversationItemPath::Items(_))) =>
        {
            Some(Route::ListConversationItems)
        }
        (LogicalMethod::Get, _)
            if matches!(conversation_path, Some(ConversationItemPath::Item { .. })) =>
        {
            Some(Route::GetConversationItem)
        }
        (LogicalMethod::Get, _) if matches!(response_path, Some(ResponseItemPath::Item(_))) => {
            Some(Route::GetStoredResponse)
        }
        (LogicalMethod::Delete, _) if matches!(response_path, Some(ResponseItemPath::Item(_))) => {
            Some(Route::DeleteStoredResponse)
        }
        (LogicalMethod::Post, _) if matches!(response_path, Some(ResponseItemPath::Cancel(_))) => {
            Some(Route::CancelStoredResponse)
        }
        (_, _) if platform_resource_path.is_some() => Some(Route::PlatformResource),
        _ => None,
    };
    if kv_key.is_some()
        || api_key_name.is_some()
        || verification_code.is_some()
        || platform_verification_code.is_some()
        || platform_resource_path.is_some()
        || conversation_project_id.is_some()
        || instruction_path.is_some()
        || conversation_path.is_some()
        || response_path.is_some()
    {
        request.path.zeroize();
    }
    let Some(route) = route else {
        return OperationPreparation::Unsupported;
    };

    let expected_response_mode = match route {
        Route::ChatCompletions => {
            let Some(body) = request.body_base64.as_ref() else {
                return rejected_bad_request();
            };
            match chat_stream_requested(body.as_slice()) {
                Ok(true) => ResponseMode::Stream,
                Ok(false) => ResponseMode::Unary,
                Err(()) => return rejected_bad_request(),
            }
        }
        Route::ResponsesCreate => ResponseMode::Stream,
        _ => ResponseMode::Unary,
    };
    if response_mode != expected_response_mode {
        return rejected_bad_request();
    }

    if cache_namespace_root_base64.is_some()
        && !matches!(
            route,
            Route::Login
                | Route::Register
                | Route::Resume
                | Route::GithubOAuthCallback
                | Route::GoogleOAuthCallback
                | Route::AppleOAuthCallback
                | Route::AppleNativeOAuth
                | Route::Models
                | Route::ModelCatalog
                | Route::ChatCompletions
                | Route::TextToSpeech
                | Route::Transcription
                | Route::Embeddings
        )
    {
        return rejected_bad_request();
    }

    match route {
        Route::Login | Route::Register => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(LogicalUnaryResponse::protocol_error(
                    StatusCode::CONFLICT,
                    "session_already_bound",
                    "Session is already authenticated",
                ));
            }
            let Some(cache_namespace_root) = cache_namespace_root_base64 else {
                return rejected_bad_request();
            };

            let body = request
                .body_base64
                .expect("validated body presence")
                .into_bytes();
            let body = Zeroizing::new(body);
            if matches!(route, Route::Login) {
                OperationPreparation::Ready(UserOperation::Login {
                    body,
                    cache_namespace_root,
                })
            } else {
                OperationPreparation::Ready(UserOperation::Register {
                    body,
                    cache_namespace_root,
                })
            }
        }
        Route::Resume => {
            if request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(LogicalUnaryResponse::protocol_error(
                    StatusCode::CONFLICT,
                    "session_already_bound",
                    "Session is already authenticated",
                ));
            }
            let Some(Credential::Resumption { value_base64 }) = credential else {
                return rejected_bad_request();
            };
            if value_base64.is_empty() {
                return rejected_bad_request();
            }
            let Some(cache_namespace_root) = cache_namespace_root_base64 else {
                return rejected_bad_request();
            };
            OperationPreparation::Ready(UserOperation::Resume {
                credential: Zeroizing::new(value_base64.into_bytes()),
                cache_namespace_root,
            })
        }
        Route::GithubOAuthInitiate | Route::GoogleOAuthInitiate | Route::AppleOAuthInitiate => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            match authority {
                AuthorityState::Anonymous => {}
                AuthorityState::Bound(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AlreadyBound,
                    ));
                }
                AuthorityState::Authenticating(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AuthenticationInProgress,
                    ));
                }
                AuthorityState::Closing => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::Closing,
                    ));
                }
            }
            let provider = match route {
                Route::GithubOAuthInitiate => OAuthProviderName::Github,
                Route::GoogleOAuthInitiate => OAuthProviderName::Google,
                Route::AppleOAuthInitiate => OAuthProviderName::Apple,
                _ => unreachable!("OAuth initiation route group is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::OAuthInitiate {
                provider,
                body: Zeroizing::new(
                    request
                        .body_base64
                        .expect("validated OAuth initiation body")
                        .into_bytes(),
                ),
            })
        }
        Route::GithubOAuthCallback
        | Route::GoogleOAuthCallback
        | Route::AppleOAuthCallback
        | Route::AppleNativeOAuth => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            match authority {
                AuthorityState::Anonymous => {}
                AuthorityState::Bound(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AlreadyBound,
                    ));
                }
                AuthorityState::Authenticating(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AuthenticationInProgress,
                    ));
                }
                AuthorityState::Closing => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::Closing,
                    ));
                }
            }
            let Some(cache_namespace_root) = cache_namespace_root_base64 else {
                return rejected_bad_request();
            };
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated OAuth callback body")
                    .into_bytes(),
            );
            match route {
                Route::GithubOAuthCallback => {
                    OperationPreparation::Ready(UserOperation::OAuthCallback {
                        provider: OAuthProviderName::Github,
                        body,
                        cache_namespace_root,
                    })
                }
                Route::GoogleOAuthCallback => {
                    OperationPreparation::Ready(UserOperation::OAuthCallback {
                        provider: OAuthProviderName::Google,
                        body,
                        cache_namespace_root,
                    })
                }
                Route::AppleOAuthCallback => {
                    OperationPreparation::Ready(UserOperation::OAuthCallback {
                        provider: OAuthProviderName::Apple,
                        body,
                        cache_namespace_root,
                    })
                }
                Route::AppleNativeOAuth => {
                    OperationPreparation::Ready(UserOperation::AppleNativeOAuth {
                        body,
                        cache_namespace_root,
                    })
                }
                _ => unreachable!("OAuth callback route group is exhaustive"),
            }
        }
        Route::UserPasswordResetRequest | Route::UserPasswordResetConfirm => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(authentication_required_response());
            }
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated password-reset body")
                    .into_bytes(),
            );
            if matches!(route, Route::UserPasswordResetRequest) {
                OperationPreparation::Ready(UserOperation::UserPasswordResetRequest { body })
            } else {
                OperationPreparation::Ready(UserOperation::UserPasswordResetConfirm { body })
            }
        }
        Route::PlatformLogin | Route::PlatformRegister => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(authentication_start_error(
                    AuthenticationStartError::AlreadyBound,
                ));
            }
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated platform authentication body")
                    .into_bytes(),
            );
            if matches!(route, Route::PlatformLogin) {
                OperationPreparation::Ready(UserOperation::PlatformLogin { body })
            } else {
                OperationPreparation::Ready(UserOperation::PlatformRegister { body })
            }
        }
        Route::PlatformResume => {
            if request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(authentication_start_error(
                    AuthenticationStartError::AlreadyBound,
                ));
            }
            let Some(Credential::Resumption { value_base64 }) = credential else {
                return rejected_bad_request();
            };
            if value_base64.is_empty() {
                return rejected_bad_request();
            }
            OperationPreparation::Ready(UserOperation::PlatformResume {
                credential: Zeroizing::new(value_base64.into_bytes()),
            })
        }
        Route::PlatformVerifyEmail => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            match authority {
                AuthorityState::Anonymous => {}
                AuthorityState::Bound(bound)
                    if matches!(bound.principal(), BoundPrincipal::Platform { .. }) => {}
                AuthorityState::Bound(_) => {
                    return OperationPreparation::Rejected(authentication_required_response());
                }
                AuthorityState::Authenticating(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AuthenticationInProgress,
                    ));
                }
                AuthorityState::Closing => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::Closing,
                    ));
                }
            }
            OperationPreparation::Ready(UserOperation::PlatformVerifyEmail {
                code: platform_verification_code
                    .expect("classified platform verification route must have a code"),
            })
        }
        Route::PlatformPasswordResetRequest | Route::PlatformPasswordResetConfirm => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            if !matches!(authority, AuthorityState::Anonymous) {
                return OperationPreparation::Rejected(authentication_required_response());
            }
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated platform password-reset body")
                    .into_bytes(),
            );
            if matches!(route, Route::PlatformPasswordResetRequest) {
                OperationPreparation::Ready(UserOperation::PlatformPasswordResetRequest { body })
            } else {
                OperationPreparation::Ready(UserOperation::PlatformPasswordResetConfirm { body })
            }
        }
        Route::PlatformLogout | Route::PlatformChangePassword => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_platform_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated platform account body")
                    .into_bytes(),
            );
            if matches!(route, Route::PlatformLogout) {
                OperationPreparation::Ready(UserOperation::PlatformLogout { authority, body })
            } else {
                OperationPreparation::Ready(UserOperation::PlatformChangePassword {
                    authority,
                    body,
                })
            }
        }
        Route::PlatformRequestVerification => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_platform_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::PlatformRequestVerification { authority })
        }
        Route::PlatformMe
        | Route::CreatePlatformOrganization
        | Route::ListPlatformOrganizations
        | Route::PlatformResource => {
            if credential.is_some() || request.query.is_some() {
                return rejected_bad_request();
            }

            let requires_body = matches!(route, Route::CreatePlatformOrganization)
                || matches!(
                    (&platform_resource_path, request.method),
                    (
                        Some(
                            PlatformResourcePath::Projects(_)
                                | PlatformResourcePath::Secrets { .. }
                                | PlatformResourcePath::Invites(_)
                        ),
                        LogicalMethod::Post
                    ) | (
                        Some(
                            PlatformResourcePath::Project { .. }
                                | PlatformResourcePath::Membership { .. }
                        ),
                        LogicalMethod::Patch
                    ) | (
                        Some(
                            PlatformResourcePath::EmailSettings { .. }
                                | PlatformResourcePath::OAuthSettings { .. }
                        ),
                        LogicalMethod::Put
                    )
                );

            let body = if requires_body {
                if !has_exact_json_content_type(&request.headers)
                    || request
                        .body_base64
                        .as_ref()
                        .is_none_or(EncodedBytes::is_empty)
                {
                    return rejected_bad_request();
                }
                Some(Zeroizing::new(
                    request
                        .body_base64
                        .take()
                        .expect("validated platform resource body")
                        .into_bytes(),
                ))
            } else {
                if !request.headers.is_empty() || request.body_base64.is_some() {
                    return rejected_bad_request();
                }
                None
            };

            let authority = match bound_platform_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = match (route, platform_resource_path, request.method) {
                (Route::PlatformMe, None, LogicalMethod::Get) => PlatformControlOperation::GetMe,
                (Route::CreatePlatformOrganization, None, LogicalMethod::Post) => {
                    PlatformControlOperation::CreateOrganization {
                        body: body.expect("organization creation requires a body"),
                    }
                }
                (Route::ListPlatformOrganizations, None, LogicalMethod::Get) => {
                    PlatformControlOperation::ListOrganizations
                }
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Organization(org_id)),
                    LogicalMethod::Delete,
                ) => PlatformControlOperation::DeleteOrganization { org_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Projects(org_id)),
                    LogicalMethod::Post,
                ) => PlatformControlOperation::CreateProject {
                    org_id,
                    body: body.expect("project creation requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Projects(org_id)),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::ListProjects { org_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Project { org_id, project_id }),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::GetProject { org_id, project_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Project { org_id, project_id }),
                    LogicalMethod::Patch,
                ) => PlatformControlOperation::UpdateProject {
                    org_id,
                    project_id,
                    body: body.expect("project update requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Project { org_id, project_id }),
                    LogicalMethod::Delete,
                ) => PlatformControlOperation::DeleteProject { org_id, project_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Secrets { org_id, project_id }),
                    LogicalMethod::Post,
                ) => PlatformControlOperation::CreateSecret {
                    org_id,
                    project_id,
                    body: body.expect("secret creation requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Secrets { org_id, project_id }),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::ListSecrets { org_id, project_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Secret {
                        org_id,
                        project_id,
                        key_name,
                    }),
                    LogicalMethod::Delete,
                ) => PlatformControlOperation::DeleteSecret {
                    org_id,
                    project_id,
                    key_name,
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::EmailSettings { org_id, project_id }),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::GetEmailSettings { org_id, project_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::EmailSettings { org_id, project_id }),
                    LogicalMethod::Put,
                ) => PlatformControlOperation::UpdateEmailSettings {
                    org_id,
                    project_id,
                    body: body.expect("email settings update requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::OAuthSettings { org_id, project_id }),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::GetOAuthSettings { org_id, project_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::OAuthSettings { org_id, project_id }),
                    LogicalMethod::Put,
                ) => PlatformControlOperation::UpdateOAuthSettings {
                    org_id,
                    project_id,
                    body: body.expect("OAuth settings update requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Memberships(org_id)),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::ListMemberships { org_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Membership { org_id, user_id }),
                    LogicalMethod::Patch,
                ) => PlatformControlOperation::UpdateMembership {
                    org_id,
                    user_id,
                    body: body.expect("membership update requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Membership { org_id, user_id }),
                    LogicalMethod::Delete,
                ) => PlatformControlOperation::DeleteMembership { org_id, user_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Invites(org_id)),
                    LogicalMethod::Post,
                ) => PlatformControlOperation::CreateInvite {
                    org_id,
                    body: body.expect("invite creation requires a body"),
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Invites(org_id)),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::ListInvites { org_id },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Invite {
                        org_id,
                        invite_code,
                    }),
                    LogicalMethod::Get,
                ) => PlatformControlOperation::GetInvite {
                    org_id,
                    invite_code,
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::Invite {
                        org_id,
                        invite_code,
                    }),
                    LogicalMethod::Delete,
                ) => PlatformControlOperation::DeleteInvite {
                    org_id,
                    invite_code,
                },
                (
                    Route::PlatformResource,
                    Some(PlatformResourcePath::AcceptInvite(invite_code)),
                    LogicalMethod::Post,
                ) => PlatformControlOperation::AcceptInvite { invite_code },
                _ => return rejected_bad_request(),
            };
            OperationPreparation::Ready(UserOperation::PlatformControl {
                authority,
                operation,
            })
        }
        Route::VerifyEmail => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            match authority {
                AuthorityState::Anonymous => {}
                AuthorityState::Bound(bound)
                    if matches!(bound.principal(), BoundPrincipal::User { .. }) => {}
                AuthorityState::Bound(_) => {
                    return OperationPreparation::Rejected(authentication_required_response());
                }
                AuthorityState::Authenticating(_) => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::AuthenticationInProgress,
                    ));
                }
                AuthorityState::Closing => {
                    return OperationPreparation::Rejected(authentication_start_error(
                        AuthenticationStartError::Closing,
                    ));
                }
            }
            OperationPreparation::Ready(UserOperation::VerifyEmail {
                code: verification_code.expect("classified verification route must have a code"),
            })
        }
        Route::GetUser | Route::GetPrivateKey | Route::GetPrivateKeyBytes | Route::GetPublicKey => {
            if credential.is_some() || !request.headers.is_empty() || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            if matches!(route, Route::GetUser) && request.query.is_some() {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = match route {
                Route::GetUser => ProtectedUserOperation::GetUser,
                Route::GetPrivateKey => ProtectedUserOperation::GetPrivateKey {
                    query: request.query,
                },
                Route::GetPrivateKeyBytes => ProtectedUserOperation::GetPrivateKeyBytes {
                    query: request.query,
                },
                Route::GetPublicKey => ProtectedUserOperation::GetPublicKey {
                    query: request.query,
                },
                _ => unreachable!("fixed protected GET classifier is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::RequestVerification => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::RequestVerification,
            })
        }
        Route::SignMessage
        | Route::IssueThirdPartyToken
        | Route::EncryptData
        | Route::DecryptData
        | Route::RequestAccountDeletion
        | Route::ConfirmAccountDeletion
        | Route::ChangePassword => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = request
                .body_base64
                .expect("validated protected JSON body presence")
                .into_bytes();
            let body = Zeroizing::new(body);
            if matches!(route, Route::ChangePassword) {
                return OperationPreparation::Ready(UserOperation::ChangePassword {
                    authority,
                    body,
                });
            }
            let operation = match route {
                Route::SignMessage => ProtectedUserOperation::SignMessage { body },
                Route::IssueThirdPartyToken => {
                    ProtectedUserOperation::IssueThirdPartyToken { body }
                }
                Route::EncryptData => ProtectedUserOperation::EncryptData { body },
                Route::DecryptData => ProtectedUserOperation::DecryptData { body },
                Route::RequestAccountDeletion => {
                    ProtectedUserOperation::RequestAccountDeletion { body }
                }
                Route::ConfirmAccountDeletion => {
                    ProtectedUserOperation::ConfirmAccountDeletion { body }
                }
                _ => unreachable!("fixed protected POST classifier is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::Logout => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            if let Err(rejection) = bound_user_authority(authority) {
                return OperationPreparation::Rejected(rejection);
            }
            OperationPreparation::Ready(UserOperation::Logout {
                body: Zeroizing::new(
                    request
                        .body_base64
                        .expect("validated logout body presence")
                        .into_bytes(),
                ),
            })
        }
        Route::PutKv => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = request
                .body_base64
                .expect("validated KV value body presence")
                .into_bytes();
            let operation = ProtectedUserOperation::PutKv {
                key: kv_key.expect("classified KV item route must have a decoded key"),
                body: Zeroizing::new(body),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::GetKv | Route::ListKv => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = if matches!(route, Route::GetKv) {
                ProtectedUserOperation::GetKv {
                    key: kv_key.expect("classified KV item route must have a decoded key"),
                }
            } else {
                ProtectedUserOperation::ListKv
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::DeleteKv | Route::DeleteAllKv => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = if matches!(route, Route::DeleteKv) {
                ProtectedUserOperation::DeleteKv {
                    key: kv_key.expect("classified KV item route must have a decoded key"),
                }
            } else {
                ProtectedUserOperation::DeleteAllKv
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::CreateApiKey => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = request
                .body_base64
                .expect("validated API-key creation body presence")
                .into_bytes();
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::CreateApiKey {
                    body: Zeroizing::new(body),
                },
            })
        }
        Route::ListApiKeys => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::ListApiKeys,
            })
        }
        Route::DeleteApiKey => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::DeleteApiKey {
                    name: api_key_name
                        .expect("classified API-key item route must have a decoded name"),
                },
            })
        }
        Route::CreateConversationProject => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = request
                .body_base64
                .expect("validated conversation-project creation body presence")
                .into_bytes();
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::CreateConversationProject {
                    body: Zeroizing::new(body),
                },
            })
        }
        Route::ListConversationProjects => {
            if credential.is_some() || !request.headers.is_empty() || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::ListConversationProjects {
                    query: request.query,
                },
            })
        }
        Route::GetConversationProject => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::GetConversationProject {
                    project_id: conversation_project_id
                        .expect("classified conversation-project route must have an ID"),
                },
            })
        }
        Route::UpdateConversationProject => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::UpdateConversationProject {
                    project_id: conversation_project_id
                        .expect("classified conversation-project route must have an ID"),
                    body: Zeroizing::new(
                        request
                            .body_base64
                            .expect("validated conversation-project update body")
                            .into_bytes(),
                    ),
                },
            })
        }
        Route::DeleteConversationProject => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::DeleteConversationProject {
                    project_id: conversation_project_id
                        .expect("classified conversation-project route must have an ID"),
                },
            })
        }
        Route::CreateInstruction => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::CreateInstruction {
                    body: Zeroizing::new(
                        request
                            .body_base64
                            .expect("validated instruction creation body")
                            .into_bytes(),
                    ),
                },
            })
        }
        Route::ListInstructions => {
            if credential.is_some() || !request.headers.is_empty() || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::ListInstructions {
                    query: request.query,
                },
            })
        }
        Route::GetInstruction | Route::DeleteInstruction | Route::SetDefaultInstruction => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let instruction_id = match instruction_path
                .expect("classified instruction route must have a decoded path")
            {
                InstructionItemPath::Item(id) | InstructionItemPath::SetDefault(id) => id,
            };
            let operation = match route {
                Route::GetInstruction => ProtectedUserOperation::GetInstruction { instruction_id },
                Route::DeleteInstruction => {
                    ProtectedUserOperation::DeleteInstruction { instruction_id }
                }
                Route::SetDefaultInstruction => {
                    ProtectedUserOperation::SetDefaultInstruction { instruction_id }
                }
                _ => unreachable!("instruction route group is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::UpdateInstruction => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let Some(InstructionItemPath::Item(instruction_id)) = instruction_path else {
                unreachable!("classified instruction update must use an item path")
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::UpdateInstruction {
                    instruction_id,
                    body: Zeroizing::new(
                        request
                            .body_base64
                            .expect("validated instruction update body")
                            .into_bytes(),
                    ),
                },
            })
        }
        Route::CreateConversation
        | Route::BatchDeleteConversations
        | Route::BatchUpdateConversationProject => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let body = Zeroizing::new(
                request
                    .body_base64
                    .expect("validated conversation mutation body")
                    .into_bytes(),
            );
            let operation = match route {
                Route::CreateConversation => ProtectedUserOperation::CreateConversation { body },
                Route::BatchDeleteConversations => {
                    ProtectedUserOperation::BatchDeleteConversations { body }
                }
                Route::BatchUpdateConversationProject => {
                    ProtectedUserOperation::BatchUpdateConversationProject { body }
                }
                _ => unreachable!("conversation collection mutation group is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::UpdateConversation => {
            if credential.is_some()
                || request.query.is_some()
                || !has_exact_json_content_type(&request.headers)
                || request
                    .body_base64
                    .as_ref()
                    .is_none_or(EncodedBytes::is_empty)
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let Some(ConversationItemPath::Conversation(conversation_id)) = conversation_path
            else {
                unreachable!("classified conversation update must use a conversation path")
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::UpdateConversation {
                    conversation_id,
                    body: Zeroizing::new(
                        request
                            .body_base64
                            .expect("validated conversation update body")
                            .into_bytes(),
                    ),
                },
            })
        }
        Route::ListConversations | Route::ListConversationItems => {
            if credential.is_some() || !request.headers.is_empty() || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = if matches!(route, Route::ListConversations) {
                ProtectedUserOperation::ListConversations {
                    query: request.query,
                }
            } else {
                let Some(ConversationItemPath::Items(conversation_id)) = conversation_path else {
                    unreachable!("classified conversation-item list must use an items path")
                };
                ProtectedUserOperation::ListConversationItems {
                    conversation_id,
                    query: request.query,
                }
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::GetConversation
        | Route::DeleteConversation
        | Route::DeleteAllConversations
        | Route::GetConversationItem
        | Route::GetStoredResponse
        | Route::CancelStoredResponse
        | Route::DeleteStoredResponse => {
            if credential.is_some()
                || request.query.is_some()
                || !request.headers.is_empty()
                || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = match route {
                Route::GetConversation | Route::DeleteConversation => {
                    let Some(ConversationItemPath::Conversation(conversation_id)) =
                        conversation_path
                    else {
                        unreachable!("classified conversation item must use a conversation path")
                    };
                    if matches!(route, Route::GetConversation) {
                        ProtectedUserOperation::GetConversation { conversation_id }
                    } else {
                        ProtectedUserOperation::DeleteConversation { conversation_id }
                    }
                }
                Route::DeleteAllConversations => ProtectedUserOperation::DeleteAllConversations,
                Route::GetConversationItem => {
                    let Some(ConversationItemPath::Item {
                        conversation_id,
                        item_id,
                    }) = conversation_path
                    else {
                        unreachable!("classified conversation item must use an item path")
                    };
                    ProtectedUserOperation::GetConversationItem {
                        conversation_id,
                        item_id,
                    }
                }
                Route::GetStoredResponse | Route::DeleteStoredResponse => {
                    let Some(ResponseItemPath::Item(response_id)) = response_path else {
                        unreachable!("classified stored response must use an item path")
                    };
                    if matches!(route, Route::GetStoredResponse) {
                        ProtectedUserOperation::GetStoredResponse { response_id }
                    } else {
                        ProtectedUserOperation::DeleteStoredResponse { response_id }
                    }
                }
                Route::CancelStoredResponse => {
                    let Some(ResponseItemPath::Cancel(response_id)) = response_path else {
                        unreachable!("classified response cancellation must use a cancel path")
                    };
                    ProtectedUserOperation::CancelStoredResponse { response_id }
                }
                _ => unreachable!("bodyless conversation/response group is exhaustive"),
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation,
            })
        }
        Route::Models => {
            if request.body_base64.is_some()
                || prepare_inference_headers(request.headers, false).is_err()
            {
                return rejected_bad_request();
            }
            let authority = if matches!(authority, AuthorityState::Anonymous)
                && credential.is_none()
                && cache_namespace_root_base64.is_none()
            {
                InferenceAuthority::Public
            } else {
                match prepare_inference_authority(
                    authority,
                    credential,
                    cache_namespace_root_base64,
                ) {
                    Ok(authority) => authority,
                    Err(rejection) => return OperationPreparation::Rejected(rejection),
                }
            };
            OperationPreparation::Ready(UserOperation::Inference {
                authority,
                operation: InferenceOperation::Models,
            })
        }
        Route::ModelCatalog
        | Route::ChatCompletions
        | Route::TextToSpeech
        | Route::Transcription
        | Route::Embeddings => {
            let operation = match route {
                Route::ModelCatalog => {
                    if request.body_base64.is_some()
                        || prepare_inference_headers(request.headers, false).is_err()
                    {
                        return rejected_bad_request();
                    }
                    InferenceOperation::ModelCatalog
                }
                Route::ChatCompletions => {
                    let Some(body) = request.body_base64.take() else {
                        return rejected_bad_request();
                    };
                    if body.is_empty() {
                        return rejected_bad_request();
                    }
                    let headers = match prepare_inference_headers(request.headers, true) {
                        Ok(headers) => headers,
                        Err(()) => return rejected_bad_request(),
                    };
                    InferenceOperation::Chat {
                        body: Zeroizing::new(body.into_bytes()),
                        headers,
                        stream: response_mode == ResponseMode::Stream,
                    }
                }
                Route::TextToSpeech | Route::Transcription | Route::Embeddings => {
                    if prepare_inference_headers(request.headers, true).is_err() {
                        return rejected_bad_request();
                    }
                    let Some(body) = request.body_base64.take() else {
                        return rejected_bad_request();
                    };
                    if body.is_empty() {
                        return rejected_bad_request();
                    }
                    let body = Zeroizing::new(body.into_bytes());
                    match route {
                        Route::TextToSpeech => InferenceOperation::TextToSpeech { body },
                        Route::Transcription => InferenceOperation::Transcription { body },
                        Route::Embeddings => InferenceOperation::Embeddings { body },
                        _ => unreachable!("typed OpenAI route group is exhaustive"),
                    }
                }
                _ => unreachable!("OpenAI unary route group is exhaustive"),
            };

            let authority = match prepare_inference_authority(
                authority,
                credential,
                cache_namespace_root_base64,
            ) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Inference {
                authority,
                operation,
            })
        }
        Route::ResponsesCreate => {
            if credential.is_some() || cache_namespace_root_base64.is_some() {
                return rejected_bad_request();
            }
            let Some(body) = request.body_base64.take() else {
                return rejected_bad_request();
            };
            if body.is_empty() {
                return rejected_bad_request();
            }
            let headers = match prepare_inference_headers(request.headers, true) {
                Ok(headers) => headers,
                Err(()) => return rejected_bad_request(),
            };
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            OperationPreparation::Ready(UserOperation::Responses {
                authority,
                body: Zeroizing::new(body.into_bytes()),
                headers,
            })
        }
        Route::WebSearch | Route::WebExtract => {
            if credential.is_some()
                || cache_namespace_root_base64.is_some()
                || prepare_inference_headers(request.headers, true).is_err()
            {
                return rejected_bad_request();
            }
            let Some(body) = request.body_base64.take() else {
                return rejected_bad_request();
            };
            if body.is_empty() {
                return rejected_bad_request();
            }
            let authority = match bound_user_authority(authority) {
                Ok(authority) => authority,
                Err(rejection) => return OperationPreparation::Rejected(rejection),
            };
            let operation = if matches!(route, Route::WebSearch) {
                InferenceOperation::WebSearch {
                    body: Zeroizing::new(body.into_bytes()),
                }
            } else {
                InferenceOperation::WebExtract {
                    body: Zeroizing::new(body.into_bytes()),
                }
            };
            OperationPreparation::Ready(UserOperation::Inference {
                authority: InferenceAuthority::User(authority),
                operation,
            })
        }
    }
}

fn bound_user_authority(
    authority: AuthorityState,
) -> Result<BoundUserAuthority, LogicalUnaryResponse> {
    let AuthorityState::Bound(bound) = authority else {
        return Err(authentication_required_response());
    };
    let BoundPrincipal::User {
        user_id,
        project_id,
        auth_context,
        cache_namespace,
    } = bound.principal()
    else {
        return Err(authentication_required_response());
    };
    Ok(BoundUserAuthority {
        user_id: *user_id,
        project_id: *project_id,
        auth_context: auth_context.clone(),
        cache_namespace: cache_namespace.clone(),
    })
}

fn bound_platform_authority(
    authority: AuthorityState,
) -> Result<BoundPlatformAuthority, LogicalUnaryResponse> {
    let AuthorityState::Bound(bound) = authority else {
        return Err(authentication_required_response());
    };
    let BoundPrincipal::Platform { platform_user_id } = bound.principal() else {
        return Err(authentication_required_response());
    };
    Ok(BoundPlatformAuthority {
        platform_user_id: *platform_user_id,
    })
}

fn prepare_inference_authority(
    authority: AuthorityState,
    credential: Option<Credential>,
    cache_namespace_root: Option<CacheNamespaceRoot>,
) -> Result<InferenceAuthority, LogicalUnaryResponse> {
    match authority {
        AuthorityState::Anonymous => {
            let Some(Credential::ApiKey { value_base64 }) = credential else {
                return Err(authentication_required_response());
            };
            let Some(cache_namespace_root) = cache_namespace_root else {
                return Err(bad_request_response());
            };
            if value_base64.is_empty() {
                return Err(bad_request_response());
            }
            Ok(InferenceAuthority::AuthenticateApiKey {
                credential: Zeroizing::new(value_base64.into_bytes()),
                cache_namespace_root,
            })
        }
        AuthorityState::Bound(bound) => {
            if credential.is_some() || cache_namespace_root.is_some() {
                return Err(LogicalUnaryResponse::protocol_error(
                    StatusCode::CONFLICT,
                    "session_already_bound",
                    "Session is already authenticated",
                ));
            }
            match bound.principal() {
                BoundPrincipal::User {
                    user_id,
                    project_id,
                    auth_context,
                    cache_namespace,
                } => Ok(InferenceAuthority::User(BoundUserAuthority {
                    user_id: *user_id,
                    project_id: *project_id,
                    auth_context: auth_context.clone(),
                    cache_namespace: cache_namespace.clone(),
                })),
                BoundPrincipal::ApiKey {
                    api_key_id,
                    user_id,
                    cache_namespace,
                } => Ok(InferenceAuthority::ApiKey(BoundApiKeyAuthority {
                    api_key_id: *api_key_id,
                    user_id: *user_id,
                    cache_namespace: cache_namespace.clone(),
                })),
                BoundPrincipal::Platform { .. } => Err(authentication_required_response()),
            }
        }
        AuthorityState::Authenticating(_) => Err(authentication_start_error(
            AuthenticationStartError::AuthenticationInProgress,
        )),
        AuthorityState::Closing => Err(authentication_start_error(
            AuthenticationStartError::Closing,
        )),
    }
}

pub(crate) fn begin_authentication_transition(
    operation: &UserOperation,
    lease: &V2SessionLease,
    request_id: RequestId,
) -> Result<Option<AuthenticationReservation>, LogicalUnaryResponse> {
    if !operation.requires_authentication_transition() {
        return Ok(None);
    }

    lease
        .state()
        .begin_authentication(request_id)
        .map(Some)
        .map_err(authentication_start_error)
}

fn authentication_start_error(error: AuthenticationStartError) -> LogicalUnaryResponse {
    let (code, message) = match error {
        AuthenticationStartError::AuthenticationInProgress => (
            "authentication_in_progress",
            "Session authentication is already in progress",
        ),
        AuthenticationStartError::AlreadyBound => {
            ("session_already_bound", "Session is already authenticated")
        }
        AuthenticationStartError::Closing => ("session_closed", "Session is closed"),
    };
    LogicalUnaryResponse::protocol_error(StatusCode::CONFLICT, code, message)
}

pub(crate) async fn execute_user_operation(
    app_state: Arc<AppState>,
    lease: V2SessionLease,
    operation: UserOperation,
    authentication: Option<AuthenticationReservation>,
    monotonic_now: Instant,
    stream_guard: Option<StreamExecutionGuard>,
) -> ApplicationOutcome {
    debug_assert_eq!(operation.is_streaming(), stream_guard.is_some());
    let session_effect_on_success = operation.session_effect_on_success();
    match operation {
        UserOperation::Login {
            body,
            cache_namespace_root,
        } => {
            let parsed =
                serde_json::from_slice::<Credentials>(&body).map_err(|_| ApiError::BadRequest);
            let credentials = match parsed {
                Ok(credentials) => credentials,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let verified = match authenticate_login(Arc::clone(&app_state), credentials).await {
                Ok(verified) => verified,
                Err(error) => return ApplicationOutcome::error(error),
            };
            finish_user_binding(
                &app_state,
                &lease,
                verified,
                authentication.expect("login requires authentication reservation"),
                monotonic_now,
                UserAuthResponseKind::Login,
                cache_namespace_root,
            )
        }
        UserOperation::Register {
            body,
            cache_namespace_root,
        } => {
            let parsed = serde_json::from_slice::<RegisterCredentials>(&body)
                .map_err(|_| ApiError::BadRequest);
            let credentials = match parsed {
                Ok(credentials) => credentials,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let verified =
                match register_and_authenticate(Arc::clone(&app_state), credentials).await {
                    Ok(verified) => verified,
                    Err(error) => return ApplicationOutcome::error(error),
                };
            finish_user_binding(
                &app_state,
                &lease,
                verified,
                authentication.expect("registration requires authentication reservation"),
                monotonic_now,
                UserAuthResponseKind::Login,
                cache_namespace_root,
            )
        }
        UserOperation::Resume {
            mut credential,
            cache_namespace_root,
        } => {
            let bytes = std::mem::take(&mut *credential);
            let credential = match String::from_utf8(bytes) {
                Ok(credential) => Zeroizing::new(credential),
                Err(error) => {
                    let mut bytes = error.into_bytes();
                    bytes.zeroize();
                    return ApplicationOutcome::error(ApiError::InvalidJwt);
                }
            };
            let verified = match validate_transport_v2_user_resumption(&credential, &app_state) {
                Ok(verified) => verified,
                Err(error) => return ApplicationOutcome::error(error),
            };
            finish_user_binding(
                &app_state,
                &lease,
                verified,
                authentication.expect("resumption requires authentication reservation"),
                monotonic_now,
                UserAuthResponseKind::Refresh,
                cache_namespace_root,
            )
        }
        UserOperation::OAuthInitiate { provider, body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_provider_json_body::<OAuthAuthRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let response = match initiate_oauth_data(
                &app_state,
                request,
                provider.as_str(),
                Some(lease.state().session_id()),
            )
            .await
            .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::OAuthCallback {
            provider,
            body,
            cache_namespace_root,
        } => {
            let request = match parse_provider_json_body::<OAuthCallbackRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if let Err(error) = validate_oauth_callback_request(&request) {
                return ApplicationOutcome::error(error);
            }
            let verified = match oauth_callback_authenticate(
                &app_state,
                request,
                provider.as_str(),
                Some(lease.state().session_id()),
            )
            .await
            {
                Ok(verified) => verified,
                Err(error) => return ApplicationOutcome::error(error),
            };
            finish_user_binding(
                &app_state,
                &lease,
                verified,
                authentication.expect("OAuth callback requires authentication reservation"),
                monotonic_now,
                UserAuthResponseKind::Login,
                cache_namespace_root,
            )
        }
        UserOperation::AppleNativeOAuth {
            body,
            cache_namespace_root,
        } => {
            let request = match parse_provider_json_body::<AppleNativeSignInRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if let Err(error) = validate_apple_native_request(&request) {
                return ApplicationOutcome::error(error);
            }
            let verified = match apple_native_authenticate(&app_state, request).await {
                Ok(verified) => verified,
                Err(error) => return ApplicationOutcome::error(error),
            };
            finish_user_binding(
                &app_state,
                &lease,
                verified,
                authentication.expect("Apple native OAuth requires authentication reservation"),
                monotonic_now,
                UserAuthResponseKind::Login,
                cache_namespace_root,
            )
        }
        UserOperation::UserPasswordResetRequest { body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_json_body::<PasswordResetRequestPayload>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let response = match password_reset_request_data(&app_state, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::UserPasswordResetConfirm { body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_json_body::<PasswordResetConfirmPayload>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let response = match password_reset_confirm_data(&app_state, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::PlatformLogin { body } => {
            let request = match parse_json_body::<PlatformLoginRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let platform_user =
                match authenticate_platform_login(Arc::clone(&app_state), request).await {
                    Ok(platform_user) => platform_user,
                    Err(error) => return ApplicationOutcome::error(error),
                };
            finish_platform_binding(
                &app_state,
                &lease,
                platform_user,
                authentication.expect("platform login requires authentication reservation"),
                monotonic_now,
                PlatformAuthResponseKind::Login,
            )
        }
        UserOperation::PlatformRegister { body } => {
            let request = match parse_json_body::<PlatformRegisterRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let platform_user =
                match register_platform_user_data(Arc::clone(&app_state), request).await {
                    Ok(platform_user) => platform_user,
                    Err(error) => return ApplicationOutcome::error(error),
                };
            finish_platform_binding(
                &app_state,
                &lease,
                platform_user,
                authentication.expect("platform registration requires authentication reservation"),
                monotonic_now,
                PlatformAuthResponseKind::Login,
            )
        }
        UserOperation::PlatformResume { mut credential } => {
            let bytes = std::mem::take(&mut *credential);
            let credential = match String::from_utf8(bytes) {
                Ok(credential) => Zeroizing::new(credential),
                Err(error) => {
                    let mut bytes = error.into_bytes();
                    bytes.zeroize();
                    return ApplicationOutcome::error(ApiError::InvalidJwt);
                }
            };
            let platform_user =
                match validate_transport_v2_platform_resumption(&credential, &app_state) {
                    Ok(platform_user) => platform_user,
                    Err(error) => return ApplicationOutcome::error(error),
                };
            finish_platform_binding(
                &app_state,
                &lease,
                platform_user,
                authentication.expect("platform resumption requires authentication reservation"),
                monotonic_now,
                PlatformAuthResponseKind::Refresh,
            )
        }
        UserOperation::PlatformVerifyEmail { code } => {
            debug_assert!(authentication.is_none());
            let response = match verify_platform_email_data(&app_state, code)
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::PlatformPasswordResetRequest { body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_json_body::<PlatformPasswordResetRequestPayload>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let response = match platform_password_reset_request_data(&app_state, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::PlatformPasswordResetConfirm { body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_json_body::<PlatformPasswordResetConfirmPayload>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let response = match platform_password_reset_confirm_data(&app_state, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::PlatformLogout { authority, body } => {
            debug_assert!(authentication.is_none());
            if let Err(outcome) = revalidate_bound_platform_user(&app_state, &authority) {
                return outcome;
            }
            let request = match parse_json_body::<PlatformLogoutRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let response =
                match LogicalUnaryResponse::json(StatusCode::OK, &platform_logout_data(request)) {
                    Ok(response) => response,
                    Err(error) => return ApplicationOutcome::error(error),
                };
            ApplicationOutcome::success(response, session_effect_on_success)
        }
        UserOperation::PlatformRequestVerification { authority } => {
            debug_assert!(authentication.is_none());
            let platform_user = match revalidate_bound_platform_user(&app_state, &authority) {
                Ok(platform_user) => platform_user,
                Err(outcome) => return outcome,
            };
            let response = match request_platform_verification_data(&app_state, &platform_user)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::PlatformChangePassword { authority, body } => {
            debug_assert!(authentication.is_none());
            let platform_user = match revalidate_bound_platform_user(&app_state, &authority) {
                Ok(platform_user) => platform_user,
                Err(outcome) => return outcome,
            };
            let request = match parse_json_body::<PlatformChangePasswordRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let response = match platform_change_password_data(&app_state, &platform_user, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, session_effect_on_success)
        }
        UserOperation::PlatformControl {
            authority,
            operation,
        } => execute_platform_control_operation(&app_state, authority, operation).await,
        UserOperation::VerifyEmail { code } => {
            debug_assert!(authentication.is_none());
            let response = match verify_email_data(&app_state, code)
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        UserOperation::Logout { body } => {
            debug_assert!(authentication.is_none());
            let request = match parse_json_body::<LogoutRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let response = match LogicalUnaryResponse::json(StatusCode::OK, &logout_data(request)) {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, session_effect_on_success)
        }
        UserOperation::ChangePassword { authority, body } => {
            debug_assert!(authentication.is_none());
            let user = match app_state.verify_bound_user(
                authority.user_id,
                authority.project_id,
                &authority.auth_context,
            ) {
                Ok(user) => user,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        return ApplicationOutcome::closing_error(error);
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            let mut request = match parse_json_body::<ChangePasswordRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if let Err(error) =
                verify_password_change_request(&app_state, &user, &mut request).await
            {
                return bound_user_error_outcome(&app_state, &authority, error);
            }

            let new_password = std::mem::take(&mut request.new_password);
            let prepared = match app_state
                .prepare_user_password_and_seed_wrap(&user, &authority.auth_context, new_password)
                .await
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    let error = map_password_change_error(error);
                    return bound_user_error_outcome(&app_state, &authority, error);
                }
            };

            let issued = match issue_transport_v2_user_tokens(
                &user,
                prepared.new_auth_context(),
                &app_state,
            ) {
                Ok(issued) => issued,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let mut access_token = issued.access_token;
            let mut resumption_token = issued.resumption_token;
            #[derive(Serialize)]
            struct ChangePasswordResponse<'a> {
                message: &'static str,
                access_token: &'a str,
                refresh_token: &'a str,
            }
            let response = LogicalUnaryResponse::json(
                StatusCode::OK,
                &ChangePasswordResponse {
                    message: "Password changed successfully",
                    access_token: &access_token,
                    refresh_token: &resumption_token,
                },
            );
            access_token.zeroize();
            resumption_token.zeroize();
            let response = match response {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };

            let new_auth_context =
                match app_state.commit_prepared_user_password_and_seed_wrap(&user, prepared) {
                    Ok(auth_context) => auth_context,
                    Err(error) => {
                        return ApplicationOutcome::closing_error(map_password_change_error(error));
                    }
                };
            if let Err(error) =
                app_state.verify_seed_wrap_for_auth_context(&user, &new_auth_context)
            {
                return ApplicationOutcome::closing_error(map_password_change_error(error));
            }

            ApplicationOutcome::success(response, session_effect_on_success)
        }
        UserOperation::Protected {
            authority,
            operation,
        } => {
            debug_assert!(authentication.is_none());
            let user = match app_state.verify_bound_user(
                authority.user_id,
                authority.project_id,
                &authority.auth_context,
            ) {
                Ok(user) => user,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        return ApplicationOutcome::closing_error(error);
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            let response = match execute_protected_user_operation(
                &app_state,
                &user,
                &authority.auth_context,
                operation,
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        return ApplicationOutcome::closing_error(error);
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            ApplicationOutcome::success(response, session_effect_on_success)
        }
        UserOperation::Inference {
            authority,
            operation,
        } => {
            execute_inference_operation(
                &app_state,
                operation,
                authority,
                authentication,
                monotonic_now,
                stream_guard,
            )
            .await
        }
        UserOperation::Responses {
            authority,
            body,
            headers,
        } => {
            debug_assert!(authentication.is_none());
            let Some(stream_guard) = stream_guard else {
                return ApplicationOutcome::error(ApiError::InternalServerError);
            };
            let user = match app_state.verify_bound_user(
                authority.user_id,
                authority.project_id,
                &authority.auth_context,
            ) {
                Ok(user) => user,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        return ApplicationOutcome::closing_error(error);
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            let request = match parse_provider_json_body::<ResponsesCreateRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            match responses_stream_v2_data(
                Arc::clone(&app_state),
                headers,
                user,
                authority.auth_context,
                request,
                authority.cache_namespace,
                stream_guard,
            )
            .await
            {
                Ok(stream) => ApplicationOutcome::stream(
                    LogicalStreamResponse::sse(stream),
                    SessionEffect::Retain,
                ),
                Err(error) => ApplicationOutcome::error(error),
            }
        }
    }
}

async fn execute_platform_control_operation(
    app_state: &AppState,
    authority: BoundPlatformAuthority,
    operation: PlatformControlOperation,
) -> ApplicationOutcome {
    macro_rules! resource_value {
        ($future:expr) => {
            match $future.await {
                Ok(value) => value,
                Err(error) => return platform_resource_error_outcome(error),
            }
        };
    }

    macro_rules! json_response {
        ($value:expr) => {
            match LogicalUnaryResponse::json(StatusCode::OK, &$value) {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            }
        };
    }

    let pool = app_state.db.get_pool();
    let actor_id = authority.platform_user_id;
    let logical_body_limit = EnvelopeLimits::DEFAULT.logical_body_bytes;
    let response = match operation {
        PlatformControlOperation::GetMe => {
            let value = resource_value!(platform_resources::get_me(
                pool,
                actor_id,
                logical_body_limit
            ));
            json_response!(value)
        }
        PlatformControlOperation::CreateOrganization { body } => {
            let request = match parse_json_body::<CreateOrgRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let value =
                resource_value!(platform_resources::create_org(pool, actor_id, request.name));
            json_response!(value)
        }
        PlatformControlOperation::ListOrganizations => {
            let value = resource_value!(platform_resources::list_orgs(
                pool,
                actor_id,
                logical_body_limit
            ));
            json_response!(value)
        }
        PlatformControlOperation::DeleteOrganization { org_id } => {
            resource_value!(platform_resources::delete_org(pool, actor_id, org_id));
            json_response!(serde_json::json!({
                "message": "Organization deleted successfully"
            }))
        }
        PlatformControlOperation::CreateProject { org_id, body } => {
            let request = match parse_json_body::<CreateProjectRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let value = resource_value!(platform_resources::create_project(
                pool,
                actor_id,
                org_id,
                request.name,
                request.description,
            ));
            json_response!(value)
        }
        PlatformControlOperation::ListProjects { org_id } => {
            let value = resource_value!(platform_resources::list_projects(
                pool,
                actor_id,
                org_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::GetProject { org_id, project_id } => {
            let value = resource_value!(platform_resources::get_project(
                pool,
                actor_id,
                org_id,
                project_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::UpdateProject {
            org_id,
            project_id,
            body,
        } => {
            let request = match parse_json_body::<UpdateProjectRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let value = resource_value!(platform_resources::update_project(
                pool,
                actor_id,
                org_id,
                project_id,
                request.name,
                request.description,
                request.status,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::DeleteProject { org_id, project_id } => {
            resource_value!(platform_resources::delete_project(
                pool, actor_id, org_id, project_id
            ));
            json_response!(serde_json::json!({
                "message": "Project deleted successfully"
            }))
        }
        PlatformControlOperation::CreateSecret {
            org_id,
            project_id,
            body,
        } => {
            let mut request = match parse_json_body::<CreateSecretRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                request.secret.zeroize();
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let encoded_secret = Zeroizing::new(std::mem::take(&mut request.secret));
            let secret_bytes = match STANDARD.decode(encoded_secret.as_bytes()) {
                Ok(secret) => Zeroizing::new(secret),
                Err(_) => return ApplicationOutcome::error(ApiError::BadRequest),
            };
            let enclave_key = match SecretKey::from_slice(&app_state.enclave_key) {
                Ok(key) => key,
                Err(_) => return ApplicationOutcome::error(ApiError::InternalServerError),
            };
            let encrypted_secret =
                Zeroizing::new(encrypt_with_key(&enclave_key, &secret_bytes).await);
            let value = resource_value!(platform_resources::create_secret(
                pool,
                actor_id,
                org_id,
                project_id,
                request.key_name,
                &encrypted_secret,
            ));
            json_response!(value)
        }
        PlatformControlOperation::ListSecrets { org_id, project_id } => {
            let value = resource_value!(platform_resources::list_secrets(
                pool,
                actor_id,
                org_id,
                project_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::DeleteSecret {
            org_id,
            project_id,
            key_name,
        } => {
            resource_value!(platform_resources::delete_secret(
                pool, actor_id, org_id, project_id, &key_name,
            ));
            json_response!(serde_json::json!({
                "message": "Secret deleted successfully"
            }))
        }
        PlatformControlOperation::GetEmailSettings { org_id, project_id } => {
            let value = resource_value!(platform_resources::get_email_settings(
                pool,
                actor_id,
                org_id,
                project_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::UpdateEmailSettings {
            org_id,
            project_id,
            body,
        } => {
            let request = match parse_json_body::<UpdateEmailSettingsRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let settings = EmailSettings {
                provider: request.provider,
                send_from: request.send_from,
                email_verification_url: request.email_verification_url,
            };
            let value = resource_value!(platform_resources::update_email_settings(
                pool, actor_id, org_id, project_id, settings,
            ));
            json_response!(value)
        }
        PlatformControlOperation::GetOAuthSettings { org_id, project_id } => {
            let value = resource_value!(platform_resources::get_oauth_settings(
                pool,
                actor_id,
                org_id,
                project_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::UpdateOAuthSettings {
            org_id,
            project_id,
            body,
        } => {
            let request = match parse_json_body::<UpdateOAuthSettingsRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err()
                || (request.google_oauth_enabled && request.google_oauth_settings.is_none())
                || (request.github_oauth_enabled && request.github_oauth_settings.is_none())
                || (request.apple_oauth_enabled && request.apple_oauth_settings.is_none())
            {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let settings = OAuthSettings {
                google_oauth_enabled: request.google_oauth_enabled,
                github_oauth_enabled: request.github_oauth_enabled,
                apple_oauth_enabled: request.apple_oauth_enabled,
                google_oauth_settings: request.google_oauth_settings,
                github_oauth_settings: request.github_oauth_settings,
                apple_oauth_settings: request.apple_oauth_settings,
            };
            let value = resource_value!(platform_resources::update_oauth_settings(
                pool, actor_id, org_id, project_id, settings,
            ));
            json_response!(value)
        }
        PlatformControlOperation::ListMemberships { org_id } => {
            let value = resource_value!(platform_resources::list_memberships(
                pool,
                actor_id,
                org_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::UpdateMembership {
            org_id,
            user_id,
            body,
        } => {
            let request = match parse_json_body::<UpdateMembershipRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let value = resource_value!(platform_resources::update_membership(
                pool,
                actor_id,
                org_id,
                user_id,
                request.role,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::DeleteMembership { org_id, user_id } => {
            resource_value!(platform_resources::delete_membership(
                pool, actor_id, org_id, user_id,
            ));
            json_response!(serde_json::json!({
                "message": "Membership deleted successfully"
            }))
        }
        PlatformControlOperation::CreateInvite { org_id, body } => {
            let request = match parse_json_body::<CreateInviteRequest>(body) {
                Ok(request) => request,
                Err(error) => return ApplicationOutcome::error(error),
            };
            if request.validate().is_err() {
                return ApplicationOutcome::error(ApiError::BadRequest);
            }
            let created = match platform_resources::create_invite(
                pool,
                actor_id,
                org_id,
                request.email,
                request.role,
                logical_body_limit,
            )
            .await
            {
                Ok(created) => created,
                Err(PlatformResourceError::NotFound(PlatformResourceKind::Organization)) => {
                    return ApplicationOutcome::error(ApiError::BadRequest);
                }
                Err(error) => return platform_resource_error_outcome(error),
            };
            let response = json_response!(created.response);
            let dispatch = created.dispatch;
            let app_mode = app_state.app_mode.clone();
            let resend_api_key = app_state.resend_api_key.clone();
            tokio::spawn(async move {
                if send_platform_invite_email(
                    app_mode,
                    resend_api_key,
                    dispatch.email,
                    dispatch.organization_name,
                    dispatch.invite_code,
                    dispatch.organization_id,
                )
                .await
                .is_err()
                {
                    tracing::error!("Failed to send platform invitation email");
                }
            });
            response
        }
        PlatformControlOperation::ListInvites { org_id } => {
            let value = resource_value!(platform_resources::list_invites(
                pool,
                actor_id,
                org_id,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::GetInvite {
            org_id,
            invite_code,
        } => {
            let value = resource_value!(platform_resources::get_invite(
                pool,
                actor_id,
                org_id,
                invite_code,
                logical_body_limit,
            ));
            json_response!(value)
        }
        PlatformControlOperation::DeleteInvite {
            org_id,
            invite_code,
        } => {
            resource_value!(platform_resources::delete_invite(
                pool,
                actor_id,
                org_id,
                invite_code,
            ));
            json_response!(serde_json::json!({
                "message": "Invite deleted successfully"
            }))
        }
        PlatformControlOperation::AcceptInvite { invite_code } => {
            resource_value!(platform_resources::accept_invite(
                pool,
                actor_id,
                invite_code,
                logical_body_limit,
            ));
            json_response!(serde_json::json!({
                "message": "Invite accepted successfully"
            }))
        }
    };

    ApplicationOutcome::success(response, SessionEffect::Retain)
}

fn platform_resource_error_outcome(error: PlatformResourceError) -> ApplicationOutcome {
    match error {
        PlatformResourceError::NotFound(PlatformResourceKind::Actor) => {
            ApplicationOutcome::closing_error(ApiError::Unauthorized)
        }
        PlatformResourceError::NotFound(_) => ApplicationOutcome::error(ApiError::NotFound),
        PlatformResourceError::Unauthorized | PlatformResourceError::VerifiedEmailRequired => {
            ApplicationOutcome::error(ApiError::Unauthorized)
        }
        PlatformResourceError::Validation
        | PlatformResourceError::Conflict
        | PlatformResourceError::LastOwner
        | PlatformResourceError::InviteAlreadyUsed
        | PlatformResourceError::InviteExpired => ApplicationOutcome::error(ApiError::BadRequest),
        PlatformResourceError::OutputTooLarge => {
            ApplicationOutcome::error(ApiError::PayloadTooLarge)
        }
        PlatformResourceError::InconsistentSnapshot
        | PlatformResourceError::Connection
        | PlatformResourceError::Database(_) => {
            tracing::error!("Failed to execute bounded platform resource operation");
            ApplicationOutcome::error(ApiError::InternalServerError)
        }
    }
}

async fn execute_inference_operation(
    app_state: &Arc<AppState>,
    operation: InferenceOperation,
    authority: InferenceAuthority,
    authentication: Option<AuthenticationReservation>,
    monotonic_now: Instant,
    stream_guard: Option<StreamExecutionGuard>,
) -> ApplicationOutcome {
    match authority {
        InferenceAuthority::Public => {
            debug_assert!(authentication.is_none());
            debug_assert!(stream_guard.is_none());
            if !matches!(operation, InferenceOperation::Models) {
                return ApplicationOutcome::error(ApiError::Unauthorized);
            }
            let response = match openai_models_v2_data(app_state)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, SessionEffect::Retain)
        }
        InferenceAuthority::User(authority) => {
            debug_assert!(authentication.is_none());
            let user = match app_state.verify_bound_user(
                authority.user_id,
                authority.project_id,
                &authority.auth_context,
            ) {
                Ok(user) => user,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        return ApplicationOutcome::closing_error(error);
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            execute_inference_for_bound_user(
                app_state,
                &user,
                OpenAiAuthMethod::Jwt,
                &authority.cache_namespace,
                operation,
                stream_guard,
                SessionEffect::Retain,
            )
            .await
        }
        InferenceAuthority::ApiKey(authority) => {
            debug_assert!(authentication.is_none());
            let user = match app_state
                .db
                .revalidate_user_api_key_owner(authority.api_key_id, authority.user_id)
            {
                Ok(Some(user)) => user,
                Ok(None) => return ApplicationOutcome::closing_error(ApiError::Unauthorized),
                Err(error) => {
                    tracing::error!(
                        api_key_id = authority.api_key_id,
                        "Failed to revalidate a bound API-key authority: {error:?}"
                    );
                    return ApplicationOutcome::error(ApiError::InternalServerError);
                }
            };
            execute_inference_for_bound_user(
                app_state,
                &user,
                OpenAiAuthMethod::ApiKey,
                &authority.cache_namespace,
                operation,
                stream_guard,
                SessionEffect::Retain,
            )
            .await
        }
        InferenceAuthority::AuthenticateApiKey {
            credential,
            cache_namespace_root,
        } => {
            let Some(authentication) = authentication else {
                return ApplicationOutcome::error(ApiError::InternalServerError);
            };
            let key_hash = match canonical_api_key_hash(credential) {
                Ok(key_hash) => key_hash,
                Err(error) => return ApplicationOutcome::error(error),
            };
            let resolved = match app_state.db.resolve_user_api_key_by_hash(&key_hash) {
                Ok(Some(resolved)) => resolved,
                Ok(None) => return ApplicationOutcome::error(ApiError::Unauthorized),
                Err(error) => {
                    tracing::error!("Failed to authenticate a v2 API key: {error:?}");
                    return ApplicationOutcome::error(ApiError::InternalServerError);
                }
            };
            drop(key_hash);
            let cache_namespace =
                derive_tinfoil_cache_namespace(&cache_namespace_root, resolved.user.get_id());
            drop(cache_namespace_root);
            let authority = BoundAuthority::api_key(
                resolved.api_key_id,
                resolved.user.get_id(),
                cache_namespace.clone(),
            );
            if authentication.commit_at(authority, monotonic_now).is_err() {
                return ApplicationOutcome::error(ApiError::InternalServerError);
            }

            // Once the authority commits, every application outcome must carry
            // NewlyBound. The gateway can then close the session if this first
            // authenticated response cannot be encrypted for the client.
            execute_inference_for_bound_user(
                app_state,
                &resolved.user,
                OpenAiAuthMethod::ApiKey,
                &cache_namespace,
                operation,
                stream_guard,
                SessionEffect::NewlyBound,
            )
            .await
        }
    }
}

async fn execute_inference_for_bound_user(
    app_state: &Arc<AppState>,
    user: &crate::User,
    auth_method: OpenAiAuthMethod,
    cache_namespace: &DerivedCacheNamespace,
    operation: InferenceOperation,
    stream_guard: Option<StreamExecutionGuard>,
    session_effect: SessionEffect,
) -> ApplicationOutcome {
    if operation.is_streaming() {
        let Some(stream_guard) = stream_guard else {
            return ApplicationOutcome::error_with_effect(
                ApiError::InternalServerError,
                session_effect,
            );
        };
        let InferenceOperation::Chat {
            body,
            headers,
            stream: true,
        } = operation
        else {
            unreachable!("only Chat Completions currently supports inference streaming")
        };
        let body = match parse_provider_json_body::<serde_json::Value>(body) {
            Ok(body) => body,
            Err(error) => return ApplicationOutcome::error_with_effect(error, session_effect),
        };
        return match openai_stream_chat_completion_v2_data(
            app_state,
            user,
            auth_method,
            body,
            &headers,
            cache_namespace,
            stream_guard,
        )
        .await
        {
            Ok(stream) => {
                ApplicationOutcome::stream(LogicalStreamResponse::sse(stream), session_effect)
            }
            Err(error) => ApplicationOutcome::error_with_effect(error, session_effect),
        };
    }

    debug_assert!(stream_guard.is_none());
    let response =
        execute_inference_for_user(app_state, user, auth_method, cache_namespace, operation).await;
    ApplicationOutcome::success(response, session_effect)
}

async fn execute_inference_for_user(
    app_state: &Arc<AppState>,
    user: &crate::User,
    auth_method: OpenAiAuthMethod,
    cache_namespace: &DerivedCacheNamespace,
    operation: InferenceOperation,
) -> LogicalUnaryResponse {
    let result = match operation {
        InferenceOperation::Models => openai_models_v2_data(app_state)
            .await
            .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value)),
        InferenceOperation::ModelCatalog => {
            let value = openai_model_catalog_data(app_state, user).await;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        InferenceOperation::Chat {
            body,
            headers,
            stream: false,
        } => {
            let body = match parse_provider_json_body::<serde_json::Value>(body) {
                Ok(body) => body,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            openai_nonstream_chat_completion_v2_data(
                app_state,
                user,
                auth_method,
                body,
                &headers,
                cache_namespace,
            )
            .await
            .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
        }
        InferenceOperation::TextToSpeech { body } => {
            let request = match parse_provider_json_body::<TTSRequest>(body) {
                Ok(request) => request,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            openai_tts_v2_data(app_state, user, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
        }
        InferenceOperation::Transcription { body } => {
            let request = match parse_provider_json_body::<TranscriptionRequest>(body) {
                Ok(request) => request,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            openai_transcription_v2_data(app_state, user, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
        }
        InferenceOperation::Embeddings { body } => {
            let request = match parse_provider_json_body::<EmbeddingRequest>(body) {
                Ok(request) => request,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            openai_embeddings_v2_data(app_state, user, auth_method, request)
                .await
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
        }
        InferenceOperation::WebSearch { body } => {
            let body = match parse_provider_json_body::<serde_json::Value>(body) {
                Ok(body) => body,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            let request = match parse_web_search_request(body) {
                Ok(request) => request,
                Err(error) => {
                    return logical_web_error(error).unwrap_or_else(LogicalUnaryResponse::api_error)
                }
            };
            match execute_web_search(app_state, user, request).await {
                Ok(value) => LogicalUnaryResponse::json(StatusCode::OK, &value),
                Err(error) => logical_web_error(error),
            }
        }
        InferenceOperation::WebExtract { body } => {
            let body = match parse_provider_json_body::<serde_json::Value>(body) {
                Ok(body) => body,
                Err(error) => return LogicalUnaryResponse::api_error(error),
            };
            let request = match parse_web_extract_request(body) {
                Ok(request) => request,
                Err(error) => {
                    return logical_web_error(error).unwrap_or_else(LogicalUnaryResponse::api_error)
                }
            };
            match execute_web_extract(app_state, user, request).await {
                Ok(value) => LogicalUnaryResponse::json(StatusCode::OK, &value),
                Err(error) => logical_web_error(error),
            }
        }
        InferenceOperation::Chat { stream: true, .. } => {
            unreachable!("streaming Chat Completions uses the stream dispatcher")
        }
    };

    match result {
        Ok(response) => response,
        Err(error) => LogicalUnaryResponse::api_error(error),
    }
}

fn logical_web_error(error: WebRouteError) -> Result<LogicalUnaryResponse, ApiError> {
    let (status, body) = error.into_logical_parts();
    LogicalUnaryResponse::json(status, &body)
}

fn canonical_api_key_hash(mut credential: SensitiveBytes) -> Result<Zeroizing<String>, ApiError> {
    let bytes = std::mem::take(&mut *credential);
    let value = match String::from_utf8(bytes) {
        Ok(value) => Zeroizing::new(value),
        Err(error) => {
            let mut bytes = error.into_bytes();
            bytes.zeroize();
            return Err(ApiError::Unauthorized);
        }
    };
    let key_id = uuid::Uuid::parse_str(&value).map_err(|_| ApiError::Unauthorized)?;
    let canonical = Zeroizing::new(key_id.hyphenated().to_string());
    Ok(Zeroizing::new(hex::encode(Sha256::digest(
        canonical.as_bytes(),
    ))))
}

fn bound_user_error_outcome(
    app_state: &AppState,
    authority: &BoundUserAuthority,
    error: ApiError,
) -> ApplicationOutcome {
    if matches!(
        app_state.verify_bound_user(
            authority.user_id,
            authority.project_id,
            &authority.auth_context,
        ),
        Err(ApiError::Unauthorized | ApiError::InvalidJwt)
    ) {
        ApplicationOutcome::closing_error(error)
    } else {
        ApplicationOutcome::error(error)
    }
}

fn revalidate_bound_platform_user(
    app_state: &AppState,
    authority: &BoundPlatformAuthority,
) -> Result<crate::models::platform_users::PlatformUser, ApplicationOutcome> {
    app_state
        .db
        .get_platform_user_by_uuid(authority.platform_user_id)
        .map_err(|error| match error {
            DBError::PlatformUserNotFound => {
                ApplicationOutcome::closing_error(ApiError::Unauthorized)
            }
            _ => {
                tracing::error!(
                    platform_user_id = %authority.platform_user_id,
                    "Failed to revalidate bound platform user: {error:?}"
                );
                ApplicationOutcome::error(ApiError::InternalServerError)
            }
        })
}

async fn execute_protected_user_operation(
    app_state: &AppState,
    user: &crate::User,
    auth_context: &AuthContext,
    operation: ProtectedUserOperation,
) -> Result<LogicalUnaryResponse, ApiError> {
    match operation {
        ProtectedUserOperation::GetUser => {
            let value = protected_user_data(app_state, user)?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::RequestVerification => {
            let value = request_new_verification_code_data(app_state, user).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::GetPrivateKey { query } => {
            let query = parse_logical_query::<DerivationPathQuery>(query)?;
            let value = private_key_data(app_state, user, auth_context, query)?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::GetPrivateKeyBytes { query } => {
            let query = parse_logical_query::<DerivationPathQuery>(query)?;
            let value = private_key_bytes_data(app_state, user, auth_context, query).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::GetPublicKey { query } => {
            let query = parse_logical_query::<PublicKeyQuery>(query)?;
            let value = public_key_data(app_state, user, auth_context, query).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::SignMessage { body } => {
            let request = parse_json_body::<SignMessageRequest>(body)?;
            let value = sign_message_data(app_state, user, auth_context, request).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::IssueThirdPartyToken { body } => {
            let request = parse_json_body::<ThirdPartyTokenRequest>(body)?;
            let value = third_party_token_data(app_state, user, request)?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::EncryptData { body } => {
            let mut request = parse_json_body::<EncryptDataRequest>(body)?;
            if let Err(error) = validate_encryption_plaintext_length(request.data.len()) {
                request.data.zeroize();
                return Err(error);
            }
            let value = encrypt_data_value(app_state, user, auth_context, request).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::DecryptData { body } => {
            let mut request = parse_json_body::<DecryptDataRequest>(body)?;
            if let Err(error) = validate_encrypted_data_base64_length(request.encrypted_data.len())
            {
                request.encrypted_data.zeroize();
                return Err(error);
            }
            let value = decrypt_data_value(app_state, user, auth_context, request).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &*value)
        }
        ProtectedUserOperation::PutKv { key, body } => {
            let value = parse_json_body::<KvValue>(body)?;
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            put_kv_value(app_state, user, auth_context, &key, value.as_str()).await?;
            Ok(response)
        }
        ProtectedUserOperation::GetKv { key } => {
            let value = app_state
                .get_bounded_kv(
                    user,
                    auth_context,
                    &key,
                    EnvelopeLimits::default().logical_body_bytes,
                )
                .await
                .map_err(map_bounded_kv_read_error)?;
            LogicalUnaryResponse::json(StatusCode::OK, &value.as_deref())
        }
        ProtectedUserOperation::ListKv => {
            let values = app_state
                .list_bounded_kv(
                    user,
                    auth_context,
                    EnvelopeLimits::default().logical_body_bytes,
                    MAX_V2_KV_LIST_ROWS,
                )
                .await
                .map_err(map_bounded_kv_read_error)?;
            LogicalUnaryResponse::json(StatusCode::OK, &*values)
        }
        ProtectedUserOperation::DeleteKv { key } => {
            let response = LogicalUnaryResponse::json(
                StatusCode::OK,
                &serde_json::json!({ "message": "Resource deleted successfully" }),
            )?;
            delete_kv_value(app_state, user, auth_context, &key).await?;
            Ok(response)
        }
        ProtectedUserOperation::DeleteAllKv => {
            let response = LogicalUnaryResponse::json(
                StatusCode::OK,
                &serde_json::json!({
                    "message": "All key-value pairs deleted successfully"
                }),
            )?;
            delete_all_kv_values(app_state, user.uuid).await?;
            Ok(response)
        }
        ProtectedUserOperation::CreateApiKey { body } => {
            let request = parse_json_body::<CreateApiKeyRequest>(body)?;
            let value = create_api_key_data(app_state, user, request).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::ListApiKeys => {
            let value = list_bounded_api_keys_data(
                app_state,
                user,
                EnvelopeLimits::default().logical_body_bytes,
                MAX_V2_API_KEY_LIST_ROWS,
            )?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::DeleteApiKey { name } => {
            let response = LogicalUnaryResponse::json(
                StatusCode::OK,
                &serde_json::json!({ "success": true }),
            )?;
            delete_api_key_by_name(app_state, user, &name)?;
            Ok(response)
        }
        ProtectedUserOperation::CreateConversationProject { body } => {
            let request = parse_json_body::<CreateConversationProjectRequest>(body)?;
            let name = Zeroizing::new(validate_project_name(&request.name)?);
            preflight_conversation_project_creation_response(&name)?;
            let value =
                create_conversation_project_with_name_data(app_state, user, auth_context, name)
                    .await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::ListConversationProjects { query } => {
            let params = parse_logical_query::<ListConversationProjectsParams>(query)?;
            let limit = if params.limit <= 0 {
                DEFAULT_PAGINATION_LIMIT
            } else {
                params.limit.min(MAX_PAGINATION_LIMIT)
            };
            let (mut projects, has_more) = stored_resources::list_projects(
                app_state.db.get_pool(),
                user.uuid,
                limit,
                params.after,
                &params.order,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_project_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let first_id = projects.first().map(|project| project.uuid);
            let last_id = projects.last().map(|project| project.uuid);
            let mut data = Vec::with_capacity(projects.len());
            for project in &mut projects {
                let mut name = decrypt_required_string(
                    &user_key,
                    &project.name_enc,
                    "conversation project name",
                )?;
                data.push(ConversationProjectListItem {
                    id: project.uuid,
                    object: OBJECT_TYPE_CONVERSATION_PROJECT,
                    name: std::mem::take(&mut *name),
                    created_at: project.created_at.timestamp(),
                    updated_at: project.updated_at.timestamp(),
                });
            }
            let value = ConversationProjectListResponse {
                object: OBJECT_TYPE_LIST,
                data,
                has_more,
                first_id,
                last_id,
            };
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::GetConversationProject { project_id } => {
            let mut stored = stored_resources::get_project(
                app_state.db.get_pool(),
                user.uuid,
                project_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_project_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let mut name = decrypt_required_string(
                &user_key,
                &stored.project.name_enc,
                "conversation project name",
            )?;
            let mut instructions = decrypt_optional_string(
                &user_key,
                stored.prompt_enc.as_ref(),
                "conversation project instruction",
            )?;
            let value = ConversationProjectResponse {
                id: stored.project.uuid,
                object: OBJECT_TYPE_CONVERSATION_PROJECT,
                name: std::mem::take(&mut *name),
                instructions: instructions
                    .as_mut()
                    .map(|value| std::mem::take(&mut **value)),
                created_at: stored.project.created_at.timestamp(),
                updated_at: stored.project.updated_at.timestamp(),
            };
            stored.zeroize();
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::UpdateConversationProject { project_id, body } => {
            let mut request = parse_json_body::<UpdateConversationProjectRequest>(body)?;
            if request.name.is_none() && request.instructions.is_missing() {
                return Err(ApiError::BadRequest);
            }

            let mut stored = stored_resources::get_project(
                app_state.db.get_pool(),
                user.uuid,
                project_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_project_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;

            let supplied_name = request.name.is_some();
            let mut final_name = if let Some(name) = request.name.take() {
                let name = Zeroizing::new(name);
                Zeroizing::new(validate_project_name(&name)?)
            } else {
                decrypt_required_string(
                    &user_key,
                    &stored.project.name_enc,
                    "conversation project name",
                )?
            };

            enum InstructionMutation {
                Unchanged,
                Set,
                Clear,
            }
            let (instruction_mutation, mut final_instructions) =
                match std::mem::take(&mut request.instructions) {
                    NullableField::Missing => (
                        InstructionMutation::Unchanged,
                        decrypt_optional_string(
                            &user_key,
                            stored.prompt_enc.as_ref(),
                            "conversation project instruction",
                        )?,
                    ),
                    NullableField::Null => (InstructionMutation::Clear, None),
                    NullableField::Value(prompt) => {
                        if prompt.trim().is_empty() {
                            return Err(ApiError::BadRequest);
                        }
                        (InstructionMutation::Set, Some(Zeroizing::new(prompt)))
                    }
                };

            preflight_conversation_project_response(
                &final_name,
                final_instructions.as_deref().map(String::as_str),
            )?;

            let name_enc = if supplied_name {
                Some(encrypt_with_key(&user_key, final_name.as_bytes()).await)
            } else {
                None
            };
            let instruction_update = match instruction_mutation {
                InstructionMutation::Unchanged => {
                    crate::models::responses::ProjectInstructionUpdate::Unchanged
                }
                InstructionMutation::Clear => {
                    crate::models::responses::ProjectInstructionUpdate::Clear
                }
                InstructionMutation::Set => {
                    let prompt = final_instructions
                        .as_ref()
                        .expect("set instruction must retain its plaintext");
                    crate::models::responses::ProjectInstructionUpdate::Set {
                        prompt_enc: encrypt_with_key(&user_key, prompt.as_bytes()).await,
                        prompt_tokens: count_tokens(prompt).min(i32::MAX as usize) as i32,
                    }
                }
            };
            let metadata = stored_resources::update_project(
                app_state.db.get_pool(),
                user.uuid,
                project_id,
                stored.project.updated_at,
                name_enc,
                instruction_update,
            )
            .map_err(map_project_storage_error)?;
            stored.zeroize();

            let value = ConversationProjectResponse {
                id: metadata.uuid,
                object: OBJECT_TYPE_CONVERSATION_PROJECT,
                name: std::mem::take(&mut *final_name),
                instructions: final_instructions
                    .as_mut()
                    .map(|value| std::mem::take(&mut **value)),
                created_at: metadata.created_at.timestamp(),
                updated_at: metadata.updated_at.timestamp(),
            };
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::DeleteConversationProject { project_id } => {
            let value = DeletedObjectResponse::conversation_project(project_id);
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            delete_conversation_project_by_uuid_data(app_state, user, project_id)?;
            Ok(response)
        }
        ProtectedUserOperation::CreateInstruction { body } => {
            let mut request = parse_json_body::<CreateInstructionRequest>(body)?;
            validate_instruction_content(&request.name, &request.prompt)?;
            preflight_instruction_response(&request.name, &request.prompt, request.is_default)?;
            let name = Zeroizing::new(std::mem::take(&mut request.name));
            let prompt = Zeroizing::new(std::mem::take(&mut request.prompt));
            let value = create_instruction_with_content_data(
                app_state,
                user,
                auth_context,
                name,
                prompt,
                request.is_default,
            )
            .await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::ListInstructions { query } => {
            let params = parse_logical_query::<ListInstructionsParams>(query)?;
            let limit = if params.limit <= 0 {
                DEFAULT_PAGINATION_LIMIT
            } else {
                params.limit.min(MAX_PAGINATION_LIMIT)
            };
            let (mut instructions, has_more) = stored_resources::list_instructions(
                app_state.db.get_pool(),
                user.uuid,
                limit,
                params.after,
                &params.order,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_instruction_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let first_id = instructions.first().map(|instruction| instruction.uuid);
            let last_id = instructions.last().map(|instruction| instruction.uuid);
            let mut data = Vec::with_capacity(instructions.len());
            for instruction in &mut instructions {
                let (mut name, mut prompt) = decrypt_instruction_content(&user_key, instruction)?;
                data.push(InstructionResponse {
                    id: instruction.uuid,
                    object: "instruction",
                    name: std::mem::take(&mut *name),
                    prompt: std::mem::take(&mut *prompt),
                    prompt_tokens: instruction.prompt_tokens,
                    is_default: instruction.is_default,
                    created_at: instruction.created_at.timestamp(),
                    updated_at: instruction.updated_at.timestamp(),
                });
            }
            let value = InstructionListResponse {
                object: OBJECT_TYPE_LIST,
                data,
                has_more,
                first_id,
                last_id,
            };
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::GetInstruction { instruction_id } => {
            let mut instruction = stored_resources::get_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_instruction_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let (mut name, mut prompt) = decrypt_instruction_content(&user_key, &instruction)?;
            let value = InstructionResponse {
                id: instruction.uuid,
                object: "instruction",
                name: std::mem::take(&mut *name),
                prompt: std::mem::take(&mut *prompt),
                prompt_tokens: instruction.prompt_tokens,
                is_default: instruction.is_default,
                created_at: instruction.created_at.timestamp(),
                updated_at: instruction.updated_at.timestamp(),
            };
            instruction.zeroize();
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::UpdateInstruction {
            instruction_id,
            body,
        } => {
            let mut request = parse_json_body::<UpdateInstructionRequest>(body)?;
            if request.name.is_none() && request.prompt.is_none() && request.is_default.is_none() {
                return Err(ApiError::BadRequest);
            }
            let mut instruction = stored_resources::get_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_instruction_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let (mut current_name, mut current_prompt) =
                decrypt_instruction_content(&user_key, &instruction)?;
            let mut final_name = request
                .name
                .take()
                .map(Zeroizing::new)
                .unwrap_or_else(|| Zeroizing::new(std::mem::take(&mut *current_name)));
            let mut final_prompt = request
                .prompt
                .take()
                .map(Zeroizing::new)
                .unwrap_or_else(|| Zeroizing::new(std::mem::take(&mut *current_prompt)));
            validate_instruction_content(&final_name, &final_prompt)?;
            let is_default = request.is_default.unwrap_or(instruction.is_default);
            preflight_instruction_response(&final_name, &final_prompt, is_default)?;
            let name_enc = encrypt_with_key(&user_key, final_name.as_bytes()).await;
            let prompt_enc = encrypt_with_key(&user_key, final_prompt.as_bytes()).await;
            let prompt_tokens = count_tokens(&final_prompt).min(i32::MAX as usize) as i32;
            let metadata = stored_resources::update_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
                instruction.updated_at,
                stored_resources::InstructionUpdateCiphertext {
                    name_enc,
                    prompt_enc,
                    prompt_tokens,
                    is_default,
                },
            )
            .map_err(map_instruction_storage_error)?;
            instruction.zeroize();
            let value = InstructionResponse {
                id: metadata.uuid,
                object: "instruction",
                name: std::mem::take(&mut *final_name),
                prompt: std::mem::take(&mut *final_prompt),
                prompt_tokens: metadata.prompt_tokens,
                is_default: metadata.is_default,
                created_at: metadata.created_at.timestamp(),
                updated_at: metadata.updated_at.timestamp(),
            };
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::DeleteInstruction { instruction_id } => {
            let value = DeletedObjectResponse {
                id: instruction_id,
                object: "instruction.deleted",
                deleted: true,
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            let deleted = stored_resources::delete_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
            )
            .map_err(map_instruction_storage_error)?;
            debug_assert_eq!(deleted, instruction_id);
            Ok(response)
        }
        ProtectedUserOperation::SetDefaultInstruction { instruction_id } => {
            let mut instruction = stored_resources::get_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_instruction_storage_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let (mut name, mut prompt) = decrypt_instruction_content(&user_key, &instruction)?;
            preflight_instruction_response(&name, &prompt, true)?;
            let metadata = stored_resources::set_default_instruction(
                app_state.db.get_pool(),
                user.uuid,
                instruction_id,
                instruction.updated_at,
            )
            .map_err(map_instruction_storage_error)?;
            instruction.zeroize();
            let value = InstructionResponse {
                id: metadata.uuid,
                object: "instruction",
                name: std::mem::take(&mut *name),
                prompt: std::mem::take(&mut *prompt),
                prompt_tokens: metadata.prompt_tokens,
                is_default: metadata.is_default,
                created_at: metadata.created_at.timestamp(),
                updated_at: metadata.updated_at.timestamp(),
            };
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::CreateConversation { body } => {
            let request = parse_conversation_json_body::<CreateConversationRequest>(body)?;
            if request
                .items
                .as_ref()
                .is_some_and(|items| !items.is_empty())
            {
                return Err(ApiError::BadRequest);
            }

            let mut metadata = request.metadata.unwrap_or_else(|| serde_json::json!({}));
            validate_metadata(&metadata)?;
            if metadata.get("title").is_none() {
                metadata["title"] = serde_json::json!("New Conversation");
            }
            let project_id = match request.project_id {
                Some(project_uuid) => Some(
                    stored_conversations::resolve_project_id(
                        app_state.db.get_pool(),
                        user.uuid,
                        project_uuid,
                    )
                    .map_err(map_stored_conversation_error)?,
                ),
                None => None,
            };
            let pinned = request.pinned.unwrap_or(false);
            preflight_conversation_response(Some(&metadata), request.project_id, pinned)?;

            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let metadata_enc = Some(encrypt_json_value(&user_key, &metadata).await?);
            let conversation_uuid = uuid::Uuid::new_v4();
            let mut conversation = app_state
                .db
                .create_conversation(NewConversation {
                    uuid: conversation_uuid,
                    user_id: user.uuid,
                    project_id,
                    is_pinned: pinned,
                    metadata_enc,
                })
                .map_err(crate::web::responses::error_mapping::map_generic_db_error)?;
            let mut value = ConversationResponse {
                id: conversation.uuid,
                object: OBJECT_TYPE_CONVERSATION,
                metadata: Some(metadata),
                project_id: request.project_id,
                pinned: conversation.is_pinned,
                created_at: conversation.created_at.timestamp(),
                last_activity_at: conversation.last_activity_at.timestamp(),
            };
            if let Some(ciphertext) = conversation.metadata_enc.as_mut() {
                ciphertext.zeroize();
            }
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            zeroize_conversation_response(&mut value);
            response
        }
        ProtectedUserOperation::ListConversations { query } => {
            let params = parse_logical_query::<ListConversationsParams>(query)?;
            params.validate()?;
            let limit = if params.limit <= 0 {
                DEFAULT_PAGINATION_LIMIT
            } else {
                params.limit.min(MAX_PAGINATION_LIMIT)
            };
            let project_filter = if params.unassigned_project == Some(true) {
                ConversationProjectFilter::Unassigned
            } else if let Some(project_uuid) = params.project_id {
                ConversationProjectFilter::Assigned(
                    stored_conversations::resolve_project_id(
                        app_state.db.get_pool(),
                        user.uuid,
                        project_uuid,
                    )
                    .map_err(map_stored_conversation_error)?,
                )
            } else {
                ConversationProjectFilter::Any
            };
            let (stored, has_more) = stored_conversations::list_conversations(
                app_state.db.get_pool(),
                user.uuid,
                limit,
                params.after,
                &params.order,
                project_filter,
                params.pinned,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let first_id = stored.first().map(|value| value.conversation.uuid);
            let last_id = stored.last().map(|value| value.conversation.uuid);
            let mut data = Vec::with_capacity(stored.len());
            for value in &stored {
                let metadata = decrypt_optional_json(
                    &user_key,
                    value.conversation.metadata_enc.as_ref(),
                    "conversation metadata",
                )?;
                data.push(stored_conversation_response(value, metadata));
            }
            let mut value = ConversationListResponse {
                object: OBJECT_TYPE_LIST,
                data,
                has_more,
                first_id,
                last_id,
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value
                .data
                .iter_mut()
                .for_each(zeroize_conversation_response);
            response
        }
        ProtectedUserOperation::GetConversation { conversation_id } => {
            let stored = stored_conversations::get_conversation(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let metadata = decrypt_optional_json(
                &user_key,
                stored.conversation.metadata_enc.as_ref(),
                "conversation metadata",
            )?;
            let mut value = stored_conversation_response(&stored, metadata);
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            zeroize_conversation_response(&mut value);
            response
        }
        ProtectedUserOperation::UpdateConversation {
            conversation_id,
            body,
        } => {
            let mut request = parse_conversation_json_body::<UpdateConversationRequest>(body)?;
            if request.metadata.is_none()
                && request.project_id.is_missing()
                && request.pinned.is_none()
            {
                return Err(ApiError::BadRequest);
            }
            let stored = stored_conversations::get_conversation(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;

            let supplied_metadata = request.metadata.is_some();
            let final_metadata = match request.metadata.take() {
                Some(metadata) => {
                    validate_metadata(&metadata)?;
                    Some(metadata)
                }
                None => decrypt_optional_json(
                    &user_key,
                    stored.conversation.metadata_enc.as_ref(),
                    "conversation metadata",
                )?,
            };
            let (project_update, final_project_uuid) = match request.project_id {
                NullableField::Missing => (ProjectAssignmentUpdate::Unchanged, stored.project_uuid),
                NullableField::Null => (ProjectAssignmentUpdate::Set(None), None),
                NullableField::Value(project_uuid) => {
                    let project_id = stored_conversations::resolve_project_id(
                        app_state.db.get_pool(),
                        user.uuid,
                        project_uuid,
                    )
                    .map_err(map_stored_conversation_error)?;
                    (
                        ProjectAssignmentUpdate::Set(Some(project_id)),
                        Some(project_uuid),
                    )
                }
            };
            let pinned = request.pinned.unwrap_or(stored.conversation.is_pinned);
            preflight_conversation_response(final_metadata.as_ref(), final_project_uuid, pinned)?;
            let metadata_enc = if supplied_metadata {
                Some(
                    encrypt_json_value(
                        &user_key,
                        final_metadata
                            .as_ref()
                            .expect("supplied metadata must remain available"),
                    )
                    .await?,
                )
            } else {
                None
            };
            let metadata = stored_conversations::update_conversation(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
                metadata_enc,
                project_update,
                request.pinned,
            )
            .map_err(map_stored_conversation_error)?;
            let mut value = ConversationResponse {
                id: metadata.uuid,
                object: OBJECT_TYPE_CONVERSATION,
                metadata: final_metadata,
                project_id: metadata.project_uuid,
                pinned: metadata.is_pinned,
                created_at: metadata.created_at.timestamp(),
                last_activity_at: metadata.last_activity_at.timestamp(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            zeroize_conversation_response(&mut value);
            response
        }
        ProtectedUserOperation::DeleteConversation { conversation_id } => {
            let value = DeletedObjectResponse::conversation(conversation_id);
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            stored_conversations::delete_conversation(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
            )
            .map_err(map_stored_conversation_error)?;
            Ok(response)
        }
        ProtectedUserOperation::DeleteAllConversations => {
            let value = serde_json::json!({
                "object": OBJECT_TYPE_LIST_DELETED,
                "deleted": true
            });
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            stored_conversations::delete_all_conversations(app_state.db.get_pool(), user.uuid)
                .map_err(map_stored_conversation_error)?;
            Ok(response)
        }
        ProtectedUserOperation::BatchDeleteConversations { body } => {
            let request = parse_json_body::<BatchDeleteConversationsRequest>(body)?;
            if request.ids.is_empty() || request.ids.len() > MAX_CONVERSATION_BATCH_SIZE {
                return Err(ApiError::BadRequest);
            }
            preflight_batch_delete_conversations_response()?;
            let mut data = Vec::with_capacity(request.ids.len());
            for conversation_id in request.ids {
                let lookup = stored_conversations::lookup_conversation_id(
                    app_state.db.get_pool(),
                    user.uuid,
                    conversation_id,
                );
                let Ok(internal_id) = lookup else {
                    data.push(BatchDeleteItemResult {
                        id: conversation_id,
                        object: OBJECT_TYPE_CONVERSATION_DELETED,
                        deleted: false,
                        error: Some("not_found"),
                    });
                    continue;
                };
                data.push(match stored_conversations::delete_conversation_by_internal_id(
                    app_state.db.get_pool(),
                    user.uuid,
                    internal_id,
                ) {
                    Ok(()) => BatchDeleteItemResult {
                        id: conversation_id,
                        object: OBJECT_TYPE_CONVERSATION_DELETED,
                        deleted: true,
                        error: None,
                    },
                    Err(error) => {
                        tracing::error!(?error, %conversation_id, "Failed to batch-delete conversation");
                        BatchDeleteItemResult {
                            id: conversation_id,
                            object: OBJECT_TYPE_CONVERSATION_DELETED,
                            deleted: false,
                            error: Some("delete_failed"),
                        }
                    }
                });
            }
            LogicalUnaryResponse::json(
                StatusCode::OK,
                &BatchDeleteConversationsResponse {
                    object: OBJECT_TYPE_LIST,
                    data,
                },
            )
        }
        ProtectedUserOperation::BatchUpdateConversationProject { body } => {
            let request = parse_json_body::<BatchUpdateConversationProjectRequest>(body)?;
            if request.ids.is_empty() || request.ids.len() > MAX_CONVERSATION_BATCH_SIZE {
                return Err(ApiError::BadRequest);
            }
            let target_project_id = match request.project_id {
                NullableField::Missing => return Err(ApiError::BadRequest),
                NullableField::Null => None,
                NullableField::Value(project_uuid) => Some(
                    stored_conversations::resolve_project_id(
                        app_state.db.get_pool(),
                        user.uuid,
                        project_uuid,
                    )
                    .map_err(map_stored_conversation_error)?,
                ),
            };
            let value = BatchUpdateConversationProjectResponse { success: true };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            stored_conversations::batch_update_conversation_project(
                app_state.db.get_pool(),
                user.uuid,
                &request.ids,
                target_project_id,
            )
            .map_err(map_stored_conversation_error)?;
            Ok(response)
        }
        ProtectedUserOperation::ListConversationItems {
            conversation_id,
            query,
        } => {
            let params = parse_logical_query::<ListItemsParams>(query)?;
            let limit = if params.limit <= 0 {
                DEFAULT_PAGINATION_LIMIT
            } else {
                params.limit.min(MAX_PAGINATION_LIMIT)
            };
            let (stored, has_more) = stored_conversations::list_conversation_items(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
                limit,
                params.after,
                &params.order,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let data = stored
                .into_iter()
                .map(|item| stored_item_to_conversation_item(item, &user_key))
                .collect::<Result<Vec<_>, _>>()?;
            let first_id = data.first().map(conversation_item_id);
            let last_id = data.last().map(conversation_item_id);
            let mut value = ConversationItemListResponse {
                object: OBJECT_TYPE_LIST,
                data,
                has_more,
                first_id,
                last_id,
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value.data.iter_mut().for_each(zeroize_conversation_item);
            response
        }
        ProtectedUserOperation::GetConversationItem {
            conversation_id,
            item_id,
        } => {
            let stored = stored_conversations::get_conversation_item(
                app_state.db.get_pool(),
                user.uuid,
                conversation_id,
                item_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let mut value = stored_item_to_conversation_item(stored, &user_key)?;
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            zeroize_conversation_item(&mut value);
            response
        }
        ProtectedUserOperation::GetStoredResponse { response_id } => {
            let stored = stored_conversations::get_stored_response(
                app_state.db.get_pool(),
                user.uuid,
                response_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            let user_key = app_state
                .get_user_key(user, auth_context, None, None)
                .await
                .map_err(|_| crate::web::responses::error_mapping::map_key_retrieval_error())?;
            let mut value = stored_response_to_wire(stored, &user_key)?;
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            zeroize_stored_response(&mut value);
            response
        }
        ProtectedUserOperation::CancelStoredResponse { response_id } => {
            let metadata = stored_conversations::get_cancelable_response_metadata(
                app_state.db.get_pool(),
                user.uuid,
                response_id,
                EnvelopeLimits::default().logical_body_bytes,
            )
            .map_err(map_stored_conversation_error)?;
            preflight_cancelled_response(&metadata.model)?;
            let mut value = ResponsesRetrieveResponse {
                id: metadata.uuid,
                object: OBJECT_TYPE_RESPONSE,
                created_at: metadata.created_at.timestamp(),
                status: STATUS_CANCELLED.to_owned(),
                model: metadata.model,
                usage: None,
                output: Vec::new(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            zeroize_stored_response(&mut value);
            let transitioned = stored_conversations::transition_stored_response_to_cancelled(
                app_state.db.get_pool(),
                user.uuid,
                response_id,
            )
            .map_err(map_stored_conversation_error)?;
            debug_assert_eq!(transitioned, response_id);
            let _ = app_state.cancellation_broadcast.send(response_id);
            Ok(response)
        }
        ProtectedUserOperation::DeleteStoredResponse { response_id } => {
            let value = DeletedObjectResponse::response(response_id);
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value)?;
            stored_conversations::delete_stored_response(
                app_state.db.get_pool(),
                user.uuid,
                response_id,
            )
            .map_err(map_stored_conversation_error)?;
            Ok(response)
        }
        ProtectedUserOperation::RequestAccountDeletion { body } => {
            let request = parse_json_body::<InitiateAccountDeletionRequest>(body)?;
            let value = initiate_account_deletion_data(app_state, user, request).await?;
            LogicalUnaryResponse::json(StatusCode::OK, &value)
        }
        ProtectedUserOperation::ConfirmAccountDeletion { body } => {
            let request = parse_json_body::<ConfirmAccountDeletionRequest>(body)?;
            let response = LogicalUnaryResponse::json(
                StatusCode::OK,
                &serde_json::json!({
                    "message": "Your account has been successfully deleted."
                }),
            )?;
            confirm_account_deletion_data(app_state, user, request).await?;
            Ok(response)
        }
    }
}

fn map_bounded_kv_read_error(error: StoreError) -> ApiError {
    if matches!(error, StoreError::OutputTooLarge) {
        ApiError::PayloadTooLarge
    } else {
        tracing::error!("Error reading bounded key-value output");
        ApiError::InternalServerError
    }
}

fn map_project_storage_error(error: StoredResourceError) -> ApiError {
    match error {
        StoredResourceError::ConversationProjectNotFound => ApiError::NotFound,
        StoredResourceError::OutputTooLarge => ApiError::PayloadTooLarge,
        StoredResourceError::StaleResource => ApiError::Conflict,
        error => {
            tracing::error!(?error, "Failed to read bounded conversation-project state");
            ApiError::InternalServerError
        }
    }
}

fn map_instruction_storage_error(error: StoredResourceError) -> ApiError {
    match error {
        StoredResourceError::InstructionNotFound => ApiError::NotFound,
        StoredResourceError::OutputTooLarge => ApiError::PayloadTooLarge,
        StoredResourceError::StaleResource => ApiError::Conflict,
        error => {
            tracing::error!(?error, "Failed to read bounded instruction state");
            ApiError::InternalServerError
        }
    }
}

fn map_stored_conversation_error(error: StoredConversationError) -> ApiError {
    match error {
        StoredConversationError::ConversationNotFound
        | StoredConversationError::ConversationProjectNotFound
        | StoredConversationError::ConversationItemNotFound
        | StoredConversationError::ResponseNotFound => ApiError::NotFound,
        StoredConversationError::Validation => ApiError::BadRequest,
        StoredConversationError::OutputTooLarge => ApiError::PayloadTooLarge,
        error => {
            tracing::error!(?error, "Failed to access bounded conversation state");
            ApiError::InternalServerError
        }
    }
}

fn decrypt_optional_json(
    user_key: &secp256k1::SecretKey,
    ciphertext: Option<&Vec<u8>>,
    description: &'static str,
) -> Result<Option<serde_json::Value>, ApiError> {
    let Some(ciphertext) = ciphertext else {
        return Ok(None);
    };
    let plaintext = Zeroizing::new(decrypt_with_key(user_key, ciphertext).map_err(|_| {
        tracing::error!(description, "Failed to decrypt bounded stored JSON");
        ApiError::InternalServerError
    })?);
    match validate_json_shape(&plaintext) {
        Ok(()) => {}
        Err(JsonShapeError::TooLarge) => return Err(ApiError::PayloadTooLarge),
        Err(JsonShapeError::Malformed) => {
            tracing::error!(description, "Malformed bounded stored JSON");
            return Err(ApiError::InternalServerError);
        }
    }
    serde_json::from_slice(&plaintext).map(Some).map_err(|_| {
        tracing::error!(description, "Failed to parse bounded stored JSON");
        ApiError::InternalServerError
    })
}

async fn encrypt_json_value(
    user_key: &secp256k1::SecretKey,
    value: &serde_json::Value,
) -> Result<Vec<u8>, ApiError> {
    let plaintext = Zeroizing::new(serde_json::to_vec(value).map_err(|_| {
        tracing::error!("Failed to serialize conversation metadata");
        ApiError::InternalServerError
    })?);
    Ok(encrypt_with_key(user_key, &plaintext).await)
}

fn stored_conversation_response(
    stored: &stored_conversations::StoredConversation,
    metadata: Option<serde_json::Value>,
) -> ConversationResponse {
    ConversationResponse {
        id: stored.conversation.uuid,
        object: OBJECT_TYPE_CONVERSATION,
        metadata,
        project_id: stored.project_uuid,
        pinned: stored.conversation.is_pinned,
        created_at: stored.conversation.created_at.timestamp(),
        last_activity_at: stored.conversation.last_activity_at.timestamp(),
    }
}

fn zeroize_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(value) => value.zeroize(),
        serde_json::Value::Array(values) => values.iter_mut().for_each(zeroize_json_value),
        serde_json::Value::Object(values) => {
            for (mut key, mut value) in std::mem::take(values) {
                key.zeroize();
                zeroize_json_value(&mut value);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn zeroize_conversation_response(value: &mut ConversationResponse) {
    if let Some(metadata) = value.metadata.as_mut() {
        zeroize_json_value(metadata);
    }
}

fn zeroize_conversation_item(item: &mut ConversationItem) {
    match item {
        ConversationItem::Message {
            status,
            role,
            content,
            ..
        } => {
            if let Some(status) = status.as_mut() {
                status.zeroize();
            }
            role.zeroize();
            for part in content {
                match part {
                    ConversationContent::Text { text }
                    | ConversationContent::InputText { text }
                    | ConversationContent::OutputText { text } => text.zeroize(),
                    ConversationContent::InputImage { image_url } => image_url.zeroize(),
                    ConversationContent::InputFile { filename } => filename.zeroize(),
                }
            }
        }
        ConversationItem::FunctionToolCall {
            name,
            arguments,
            status,
            ..
        } => {
            name.zeroize();
            arguments.zeroize();
            if let Some(status) = status.as_mut() {
                status.zeroize();
            }
        }
        ConversationItem::FunctionToolCallOutput { output, status, .. } => {
            output.zeroize();
            if let Some(status) = status.as_mut() {
                status.zeroize();
            }
        }
        ConversationItem::Reasoning {
            content, status, ..
        } => {
            for ReasoningContentItem::Text { text } in content {
                text.zeroize();
            }
            if let Some(status) = status.as_mut() {
                status.zeroize();
            }
        }
    }
}

fn zeroize_stored_response(value: &mut ResponsesRetrieveResponse) {
    value.status.zeroize();
    value.model.zeroize();
    for item in &mut value.output {
        item.output_type.zeroize();
        item.id.zeroize();
        item.status.zeroize();
        if let Some(role) = item.role.as_mut() {
            role.zeroize();
        }
        if let Some(content) = item.content.as_mut() {
            for part in content {
                part.part_type.zeroize();
                part.text.zeroize();
            }
        }
        for value in [
            &mut item.call_id,
            &mut item.name,
            &mut item.arguments,
            &mut item.output,
        ] {
            if let Some(value) = value.as_mut() {
                value.zeroize();
            }
        }
    }
}

fn conversation_item_id(item: &ConversationItem) -> uuid::Uuid {
    match item {
        ConversationItem::Message { id, .. }
        | ConversationItem::FunctionToolCall { id, .. }
        | ConversationItem::FunctionToolCallOutput { id, .. }
        | ConversationItem::Reasoning { id, .. } => *id,
    }
}

fn stored_item_to_conversation_item(
    mut item: stored_conversations::StoredConversationItem,
    user_key: &secp256k1::SecretKey,
) -> Result<ConversationItem, ApiError> {
    let mut content = decrypt_optional_string(
        user_key,
        item.content_enc.as_ref(),
        "conversation item content",
    )?
    .unwrap_or_else(|| Zeroizing::new(String::new()));
    let id = item.uuid;
    let created_at = Some(item.created_at.timestamp());
    let status = item.status.take();

    match item.message_type.as_str() {
        "user" => {
            let message_content =
                serde_json::from_str::<MessageContent>(&content).map_err(|_| {
                    tracing::error!("Failed to parse bounded user-message content");
                    ApiError::InternalServerError
                })?;
            Ok(ConversationItem::Message {
                id,
                status,
                role: ROLE_USER.to_owned(),
                content: Vec::<ConversationContent>::from(message_content),
                created_at,
            })
        }
        "assistant" => {
            let content = if content.is_empty() {
                Vec::new()
            } else {
                vec![ConversationContent::OutputText {
                    text: std::mem::take(&mut *content),
                }]
            };
            Ok(ConversationItem::Message {
                id,
                status,
                role: ROLE_ASSISTANT.to_owned(),
                content,
                created_at,
            })
        }
        "tool_call" => Ok(ConversationItem::FunctionToolCall {
            id,
            call_id: item.tool_call_id.ok_or_else(|| {
                tracing::error!("Stored tool call is missing its call ID");
                ApiError::InternalServerError
            })?,
            name: item
                .tool_name
                .take()
                .unwrap_or_else(|| DEFAULT_TOOL_FUNCTION_NAME.to_owned()),
            arguments: std::mem::take(&mut *content),
            status,
            created_at,
        }),
        "tool_output" => Ok(ConversationItem::FunctionToolCallOutput {
            id,
            call_id: item.tool_call_id.ok_or_else(|| {
                tracing::error!("Stored tool output is missing its call ID");
                ApiError::InternalServerError
            })?,
            output: std::mem::take(&mut *content),
            status,
            created_at,
        }),
        "reasoning" => {
            let content = if content.is_empty() {
                Vec::new()
            } else {
                vec![ReasoningContentItem::Text {
                    text: std::mem::take(&mut *content),
                }]
            };
            Ok(ConversationItem::Reasoning {
                id,
                content,
                status,
                created_at,
            })
        }
        unknown => {
            tracing::error!(
                message_type = unknown,
                "Unknown bounded conversation-item type"
            );
            Err(ApiError::InternalServerError)
        }
    }
}

fn response_status_string(status: ResponseStatus) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn stored_response_to_wire(
    stored: stored_conversations::StoredResponse,
    user_key: &secp256k1::SecretKey,
) -> Result<ResponsesRetrieveResponse, ApiError> {
    let completed = stored.response.status == ResponseStatus::Completed;
    let mut input_tokens = 0i32;
    let mut output_tokens = 0i32;
    let mut reasoning_tokens = 0i32;
    let mut output = Vec::new();

    for mut item in stored.items {
        let tokens = item.token_count.unwrap_or_default();
        if completed {
            match item.message_type.as_str() {
                "user" | "tool_call" | "tool_output" => {
                    input_tokens = input_tokens.saturating_add(tokens);
                }
                "assistant" => {
                    output_tokens = output_tokens.saturating_add(tokens);
                }
                "reasoning" => {
                    output_tokens = output_tokens.saturating_add(tokens);
                    reasoning_tokens = reasoning_tokens.saturating_add(tokens);
                }
                _ => {}
            }
        }

        let status = item
            .status
            .take()
            .unwrap_or_else(|| STATUS_COMPLETED.to_owned());
        match item.message_type.as_str() {
            "user" => {}
            "assistant" => {
                let content = decrypt_optional_string(
                    user_key,
                    item.content_enc.as_ref(),
                    "assistant message content",
                )?
                .map_or_else(Vec::new, |mut text| {
                    vec![ContentPart {
                        part_type: "output_text".to_owned(),
                        annotations: Vec::new(),
                        logprobs: Vec::new(),
                        text: std::mem::take(&mut *text),
                    }]
                });
                output.push(OutputItem {
                    id: item.uuid.to_string(),
                    output_type: "message".to_owned(),
                    status,
                    role: Some(ROLE_ASSISTANT.to_owned()),
                    content: Some(content),
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                });
            }
            "tool_call" => {
                let arguments = decrypt_optional_string(
                    user_key,
                    item.content_enc.as_ref(),
                    "tool call arguments",
                )?
                .map(|mut value| std::mem::take(&mut *value));
                output.push(OutputItem {
                    id: item.uuid.to_string(),
                    output_type: "tool_call".to_owned(),
                    status,
                    role: None,
                    content: None,
                    call_id: Some(item.tool_call_id.unwrap_or(item.uuid).to_string()),
                    name: Some(
                        item.tool_name
                            .take()
                            .unwrap_or_else(|| DEFAULT_TOOL_FUNCTION_NAME.to_owned()),
                    ),
                    arguments,
                    output: None,
                });
            }
            "tool_output" => {
                let tool_output =
                    decrypt_optional_string(user_key, item.content_enc.as_ref(), "tool output")?
                        .map(|mut value| std::mem::take(&mut *value));
                output.push(OutputItem {
                    id: item.uuid.to_string(),
                    output_type: "tool_output".to_owned(),
                    status,
                    role: None,
                    content: None,
                    call_id: item.tool_call_id.map(|id| id.to_string()),
                    name: None,
                    arguments: None,
                    output: tool_output,
                });
            }
            "reasoning" => output.push(OutputItem {
                id: item.uuid.to_string(),
                output_type: "reasoning".to_owned(),
                status,
                role: None,
                content: Some(Vec::new()),
                call_id: None,
                name: None,
                arguments: None,
                output: None,
            }),
            _ => {}
        }
    }

    let usage = completed.then(|| ResponseUsage {
        input_tokens,
        input_tokens_details: InputTokenDetails { cached_tokens: 0 },
        output_tokens,
        output_tokens_details: OutputTokenDetails { reasoning_tokens },
        total_tokens: input_tokens.saturating_add(output_tokens),
    });

    Ok(ResponsesRetrieveResponse {
        id: stored.response.uuid,
        object: OBJECT_TYPE_RESPONSE,
        created_at: stored.response.created_at.timestamp(),
        status: response_status_string(stored.response.status),
        model: stored.response.model,
        usage,
        output,
    })
}

fn decrypt_instruction_content(
    user_key: &secp256k1::SecretKey,
    instruction: &stored_resources::InstructionCiphertextRow,
) -> Result<(Zeroizing<String>, Zeroizing<String>), ApiError> {
    let name = decrypt_required_string(
        user_key,
        instruction
            .name_enc
            .as_ref()
            .ok_or(ApiError::InternalServerError)?,
        "instruction name",
    )?;
    let prompt = decrypt_required_string(user_key, &instruction.prompt_enc, "instruction prompt")?;
    Ok((name, prompt))
}

fn decrypt_required_string(
    user_key: &secp256k1::SecretKey,
    ciphertext: &Vec<u8>,
    description: &'static str,
) -> Result<Zeroizing<String>, ApiError> {
    decrypt_optional_string(user_key, Some(ciphertext), description)?
        .ok_or(ApiError::InternalServerError)
}

fn decrypt_optional_string(
    user_key: &secp256k1::SecretKey,
    ciphertext: Option<&Vec<u8>>,
    description: &'static str,
) -> Result<Option<Zeroizing<String>>, ApiError> {
    let Some(ciphertext) = ciphertext else {
        return Ok(None);
    };
    let mut plaintext = Zeroizing::new(decrypt_with_key(user_key, ciphertext).map_err(|_| {
        tracing::error!(description, "Failed to decrypt bounded stored output");
        ApiError::InternalServerError
    })?);
    let value = match String::from_utf8(std::mem::take(&mut *plaintext)) {
        Ok(value) => value,
        Err(error) => {
            let mut invalid_bytes = error.into_bytes();
            let value = String::from_utf8_lossy(&invalid_bytes).into_owned();
            invalid_bytes.zeroize();
            value
        }
    };
    Ok(Some(Zeroizing::new(value)))
}

fn parse_json_body<T: DeserializeOwned>(body: SensitiveBytes) -> Result<T, ApiError> {
    serde_json::from_slice::<T>(&body).map_err(|_| ApiError::BadRequest)
}

/// Bounds JSON container amplification before serde allocates an object tree.
///
/// Strings are skipped without allocation, so structural punctuation inside a
/// metadata value cannot inflate the count. Full syntax validation remains the
/// responsibility of serde immediately after this preflight.
fn validate_json_shape(bytes: &[u8]) -> Result<(), JsonShapeError> {
    let mut depth = 0usize;
    let mut structural_tokens = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match *byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.checked_add(1).ok_or(JsonShapeError::TooLarge)?;
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(JsonShapeError::TooLarge)?;
                if depth > MAX_CONVERSATION_JSON_DEPTH
                    || structural_tokens > MAX_CONVERSATION_JSON_STRUCTURAL_TOKENS
                {
                    return Err(JsonShapeError::TooLarge);
                }
            }
            b'}' | b']' => {
                depth = depth.checked_sub(1).ok_or(JsonShapeError::Malformed)?;
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(JsonShapeError::TooLarge)?;
                if structural_tokens > MAX_CONVERSATION_JSON_STRUCTURAL_TOKENS {
                    return Err(JsonShapeError::TooLarge);
                }
            }
            b',' | b':' => {
                structural_tokens = structural_tokens
                    .checked_add(1)
                    .ok_or(JsonShapeError::TooLarge)?;
                if structural_tokens > MAX_CONVERSATION_JSON_STRUCTURAL_TOKENS {
                    return Err(JsonShapeError::TooLarge);
                }
            }
            _ => {}
        }
    }

    if in_string || depth != 0 {
        Err(JsonShapeError::Malformed)
    } else {
        Ok(())
    }
}

/// Read only the top-level Chat Completions `stream` flag while skipping all
/// prompt/tool content without allocating a second JSON object tree.
fn chat_stream_requested(bytes: &[u8]) -> Result<bool, ()> {
    if validate_json_shape(bytes).is_err() {
        return Err(());
    }

    struct StreamProbe;

    impl<'de> Visitor<'de> for StreamProbe {
        type Value = bool;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a Chat Completions JSON object")
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut stream = None;
            while let Some(key) = map.next_key::<String>()? {
                let key = Zeroizing::new(key);
                if key.as_str() == "stream" {
                    if stream.is_some() {
                        return Err(M::Error::custom("duplicate stream field"));
                    }
                    stream = Some(map.next_value::<bool>()?);
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(stream.unwrap_or(false))
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let stream = deserializer.deserialize_map(StreamProbe).map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    Ok(stream)
}

fn parse_conversation_json_body<T: DeserializeOwned>(body: SensitiveBytes) -> Result<T, ApiError> {
    match validate_json_shape(&body) {
        Ok(()) => parse_json_body(body),
        Err(JsonShapeError::TooLarge) => Err(ApiError::PayloadTooLarge),
        Err(JsonShapeError::Malformed) => Err(ApiError::BadRequest),
    }
}

fn parse_provider_json_body<T: DeserializeOwned>(body: SensitiveBytes) -> Result<T, ApiError> {
    match validate_json_shape(&body) {
        Ok(()) => parse_json_body(body),
        Err(JsonShapeError::TooLarge) => Err(ApiError::PayloadTooLarge),
        Err(JsonShapeError::Malformed) => Err(ApiError::BadRequest),
    }
}

fn validate_oauth_callback_request(request: &OAuthCallbackRequest) -> Result<(), ApiError> {
    if request.code.is_empty() || request.state.is_empty() {
        return Err(ApiError::BadRequest);
    }
    if request.code.len() > MAX_OAUTH_CODE_BYTES || request.state.len() > MAX_OAUTH_STATE_BYTES {
        return Err(ApiError::PayloadTooLarge);
    }
    Ok(())
}

fn validate_apple_native_request(request: &AppleNativeSignInRequest) -> Result<(), ApiError> {
    let oversized_optional_field = [
        request.user_identifier.as_deref(),
        request.email.as_deref(),
        request.given_name.as_deref(),
        request.family_name.as_deref(),
        request.nonce.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|value| value.len() > MAX_APPLE_OPTIONAL_FIELD_BYTES);
    if request.identity_token.is_empty() {
        return Err(ApiError::BadRequest);
    }
    if request.identity_token.len() > MAX_APPLE_IDENTITY_TOKEN_BYTES || oversized_optional_field {
        return Err(ApiError::PayloadTooLarge);
    }
    Ok(())
}

fn parse_logical_query<T: DeserializeOwned>(query: Option<String>) -> Result<T, ApiError> {
    let path_and_query = match query {
        Some(query) => format!("/?{query}"),
        None => "/".to_owned(),
    };
    let uri = path_and_query
        .parse::<Uri>()
        .map_err(|_| ApiError::BadRequest)?;
    Query::<T>::try_from_uri(&uri)
        .map(|Query(value)| value)
        .map_err(|_| ApiError::BadRequest)
}

fn validate_encryption_plaintext_length(length: usize) -> Result<(), ApiError> {
    if length > MAX_ENCRYPTION_PLAINTEXT_BYTES {
        Err(ApiError::PayloadTooLarge)
    } else {
        Ok(())
    }
}

fn validate_encrypted_data_base64_length(length: usize) -> Result<(), ApiError> {
    if length > MAX_ENCRYPTED_DATA_BASE64_BYTES {
        Err(ApiError::PayloadTooLarge)
    } else {
        Ok(())
    }
}

enum UserAuthResponseKind {
    Login,
    Refresh,
}

enum PlatformAuthResponseKind {
    Login,
    Refresh,
}

fn finish_platform_binding(
    app_state: &AppState,
    lease: &V2SessionLease,
    platform_user: crate::models::platform_users::PlatformUser,
    authentication: AuthenticationReservation,
    monotonic_now: Instant,
    response_kind: PlatformAuthResponseKind,
) -> ApplicationOutcome {
    let issued = match issue_transport_v2_platform_tokens(&platform_user, app_state) {
        Ok(issued) => issued,
        Err(error) => return ApplicationOutcome::error(error),
    };
    let authentication_expires_at = match monotonic_authentication_expiry(
        issued.access_expires_at,
        monotonic_now,
        lease.state().absolute_expires_at(),
    ) {
        Ok(expiry) => expiry,
        Err(error) => return ApplicationOutcome::error(error),
    };

    let mut access_token = issued.access_token;
    let mut resumption_token = issued.resumption_token;
    let response = match response_kind {
        PlatformAuthResponseKind::Login => {
            let mut value = PlatformAuthResponse {
                id: platform_user.uuid,
                email: platform_user.email.clone(),
                name: platform_user.name.clone(),
                access_token: access_token.clone(),
                refresh_token: resumption_token.clone(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value.access_token.zeroize();
            value.refresh_token.zeroize();
            response
        }
        PlatformAuthResponseKind::Refresh => {
            let mut value = PlatformRefreshResponse {
                access_token: access_token.clone(),
                refresh_token: resumption_token.clone(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value.access_token.zeroize();
            value.refresh_token.zeroize();
            response
        }
    };
    access_token.zeroize();
    resumption_token.zeroize();
    let response = match response {
        Ok(response) => response,
        Err(error) => return ApplicationOutcome::error(error),
    };

    if authentication
        .commit_at(
            BoundAuthority::platform(platform_user.uuid, authentication_expires_at),
            monotonic_now,
        )
        .is_err()
    {
        return ApplicationOutcome::error(ApiError::InternalServerError);
    }
    ApplicationOutcome::success(response, SessionEffect::NewlyBound)
}

fn finish_user_binding(
    app_state: &AppState,
    lease: &V2SessionLease,
    verified: VerifiedUserAuthentication,
    authentication: AuthenticationReservation,
    monotonic_now: Instant,
    response_kind: UserAuthResponseKind,
    cache_namespace_root: CacheNamespaceRoot,
) -> ApplicationOutcome {
    let issued =
        match issue_transport_v2_user_tokens(&verified.user, &verified.auth_context, app_state) {
            Ok(issued) => issued,
            Err(error) => return ApplicationOutcome::error(error),
        };
    let authentication_expires_at = match monotonic_authentication_expiry(
        issued.access_expires_at,
        monotonic_now,
        lease.state().absolute_expires_at(),
    ) {
        Ok(expiry) => expiry,
        Err(error) => return ApplicationOutcome::error(error),
    };

    let mut access_token = issued.access_token;
    let mut resumption_token = issued.resumption_token;
    let response = match response_kind {
        UserAuthResponseKind::Login => {
            let mut value = AuthResponse {
                id: verified.user.get_id(),
                email: verified.user.get_email().map(str::to_owned),
                access_token: access_token.clone(),
                refresh_token: resumption_token.clone(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value.access_token.zeroize();
            value.refresh_token.zeroize();
            response
        }
        UserAuthResponseKind::Refresh => {
            let mut value = RefreshResponse {
                access_token: access_token.clone(),
                refresh_token: resumption_token.clone(),
            };
            let response = LogicalUnaryResponse::json(StatusCode::OK, &value);
            value.access_token.zeroize();
            value.refresh_token.zeroize();
            response
        }
    };
    access_token.zeroize();
    resumption_token.zeroize();
    let response = match response {
        Ok(response) => response,
        Err(error) => return ApplicationOutcome::error(error),
    };

    let cache_namespace =
        derive_tinfoil_cache_namespace(&cache_namespace_root, verified.user.get_id());
    drop(cache_namespace_root);
    let authority =
        BoundAuthority::verified_user(&verified, authentication_expires_at, cache_namespace);
    if authentication.commit_at(authority, monotonic_now).is_err() {
        return ApplicationOutcome::error(ApiError::InternalServerError);
    }

    ApplicationOutcome::success(response, SessionEffect::NewlyBound)
}

fn monotonic_authentication_expiry(
    wall_expiry: DateTime<Utc>,
    monotonic_now: Instant,
    absolute_session_expiry: Instant,
) -> Result<Instant, ApiError> {
    let remaining = wall_expiry
        .signed_duration_since(Utc::now())
        .to_std()
        .map_err(|_| ApiError::InternalServerError)?;
    let authentication_expiry = monotonic_now
        .checked_add(remaining)
        .ok_or(ApiError::InternalServerError)?;
    let capped = authentication_expiry.min(absolute_session_expiry);
    if capped <= monotonic_now {
        return Err(ApiError::InternalServerError);
    }
    Ok(capped)
}

fn rejected_bad_request() -> OperationPreparation {
    OperationPreparation::Rejected(bad_request_response())
}

fn bad_request_response() -> LogicalUnaryResponse {
    LogicalUnaryResponse::protocol_error(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "Invalid request",
    )
}

fn rejected_authentication_required() -> OperationPreparation {
    OperationPreparation::Rejected(authentication_required_response())
}

fn authentication_required_response() -> LogicalUnaryResponse {
    LogicalUnaryResponse::protocol_error(
        StatusCode::UNAUTHORIZED,
        "authentication_required",
        "Authentication required",
    )
}

fn has_exact_json_content_type(headers: &[HeaderField]) -> bool {
    let [header] = headers else {
        return false;
    };
    if header.name != "content-type" {
        return false;
    }
    is_json_content_type(header.value_base64.as_slice())
}

fn is_json_content_type(value: &[u8]) -> bool {
    let Ok(value) = std::str::from_utf8(value) else {
        return false;
    };
    let mut parts = value.split(';');
    if !parts
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json")
    {
        return false;
    }

    let mut saw_charset = false;
    for parameter in parts {
        let Some((name, value)) = parameter.trim().split_once('=') else {
            return false;
        };
        if saw_charset
            || !name.trim().eq_ignore_ascii_case("charset")
            || !value.trim().eq_ignore_ascii_case("utf-8")
        {
            return false;
        }
        saw_charset = true;
    }
    true
}

fn prepare_inference_headers(
    headers: Vec<HeaderField>,
    require_json_content_type: bool,
) -> Result<HeaderMap, ()> {
    let mut prepared = HeaderMap::new();
    let mut saw_content_type = false;

    for header in headers {
        let name = HeaderName::from_bytes(header.name.as_bytes()).map_err(|_| ())?;
        if is_forbidden_inference_header(&name) {
            return Err(());
        }
        if name == header::CONTENT_TYPE {
            if saw_content_type || !is_json_content_type(header.value_base64.as_slice()) {
                return Err(());
            }
            saw_content_type = true;
        }
        let mut value = HeaderValue::from_bytes(header.value_base64.as_slice()).map_err(|_| ())?;
        value.set_sensitive(true);
        prepared.append(name, value);
    }

    if require_json_content_type && !saw_content_type {
        return Err(());
    }
    Ok(prepared)
}

fn is_forbidden_inference_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "authorization"
            | "proxy-authorization"
            | "proxy-authenticate"
            | "cookie"
            | "set-cookie"
            | "host"
            | "content-length"
            | "content-encoding"
            | "content-md5"
            | "digest"
            | "accept-encoding"
            | "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "x-session-id"
            | "x-api-key"
            | "api-key"
            | "x-openai-api-key"
            | "x-tinfoil-api-key"
            | "x-goog-api-key"
            | "x-anthropic-api-key"
            | "openai-organization"
            | "openai-project"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jwt::AuthMethod;
    use crate::transport_v2::envelope::{LogicalRequest, UnaryResponseEnvelope, Version2};
    use crate::web::protected_routes::EncryptDataResponse;

    fn envelope(
        method: LogicalMethod,
        path: &str,
        headers: Vec<HeaderField>,
        body: Option<&[u8]>,
        credential: Option<Credential>,
    ) -> RequestEnvelope {
        let cache_namespace_root_base64 = (matches!(path, "/login" | "/register")
            || matches!(credential, Some(Credential::Resumption { .. })))
        .then(|| CacheNamespaceRoot::from_bytes([0x30; 32]));
        RequestEnvelope {
            version: Version2,
            request_id: RequestId::from_bytes([0x31; 16]),
            response_mode: ResponseMode::Unary,
            credential,
            cache_namespace_root_base64,
            request: LogicalRequest {
                method,
                path: path.to_owned(),
                query: None,
                headers,
                body_base64: body.map(|body| EncodedBytes::from_bytes(body.to_vec())),
            },
        }
    }

    fn json_header() -> Vec<HeaderField> {
        vec![HeaderField {
            name: "content-type".to_owned(),
            value_base64: EncodedBytes::from_bytes(JSON_CONTENT_TYPE.to_vec()),
        }]
    }

    fn logical_header_value<'a>(
        response: &'a LogicalUnaryResponse,
        name: &str,
    ) -> Option<&'a [u8]> {
        response
            .headers
            .iter()
            .find(|header| header.name == name)
            .map(|header| header.value_base64.as_slice())
    }

    fn bound_user_authority() -> AuthorityState {
        let auth_context = AuthContext::new(AuthMethod::Password, 7, [0x32; 32]);
        let cache_namespace = derive_tinfoil_cache_namespace(
            &CacheNamespaceRoot::from_bytes([0x34; 32]),
            uuid::Uuid::from_bytes([0x33; 16]),
        );
        AuthorityState::Bound(BoundAuthority::user(
            uuid::Uuid::from_bytes([0x33; 16]),
            7,
            &auth_context,
            Instant::now() + std::time::Duration::from_secs(60),
            cache_namespace,
        ))
    }

    #[test]
    fn exact_user_operation_contracts_are_admitted() {
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/login",
                    json_header(),
                    Some(br#"{"id":"00000000-0000-0000-0000-000000000000","password":"p","client_id":"00000000-0000-0000-0000-000000000000"}"#),
                    None,
                ),
                AuthorityState::Anonymous,
            ),
            OperationPreparation::Ready(UserOperation::Login { .. })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/register",
                    json_header(),
                    Some(br#"{"password":"p","client_id":"00000000-0000-0000-0000-000000000000","inviteCode":"ignored"}"#),
                    None,
                ),
                AuthorityState::Anonymous,
            ),
            OperationPreparation::Ready(UserOperation::Register { .. })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/refresh",
                    Vec::new(),
                    None,
                    Some(Credential::Resumption {
                        value_base64: EncodedBytes::from_bytes(b"token".to_vec()),
                    }),
                ),
                AuthorityState::Anonymous,
            ),
            OperationPreparation::Ready(UserOperation::Resume { .. })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Get,
                    "/protected/user",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::GetUser,
                ..
            })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/protected/request_verification",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::RequestVerification,
                ..
            })
        ));
    }

    #[test]
    fn exact_password_reset_and_platform_lifecycle_contracts_are_admitted() {
        for (path, expected_request) in [
            ("/password-reset/request", true),
            ("/password-reset/confirm", false),
        ] {
            let prepared = prepare_user_operation(
                envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None),
                AuthorityState::Anonymous,
            );
            assert!(matches!(
                (expected_request, prepared),
                (
                    true,
                    OperationPreparation::Ready(UserOperation::UserPasswordResetRequest { .. })
                ) | (
                    false,
                    OperationPreparation::Ready(UserOperation::UserPasswordResetConfirm { .. })
                )
            ));
        }

        for (path, expected_login) in [("/platform/login", true), ("/platform/register", false)] {
            let prepared = prepare_user_operation(
                envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None),
                AuthorityState::Anonymous,
            );
            assert!(matches!(
                (expected_login, prepared),
                (
                    true,
                    OperationPreparation::Ready(UserOperation::PlatformLogin { .. })
                ) | (
                    false,
                    OperationPreparation::Ready(UserOperation::PlatformRegister { .. })
                )
            ));
        }

        let mut resume = envelope(
            LogicalMethod::Post,
            "/platform/refresh",
            Vec::new(),
            None,
            Some(Credential::Resumption {
                value_base64: EncodedBytes::from_bytes(b"platform-resumption".to_vec()),
            }),
        );
        resume.cache_namespace_root_base64 = None;
        assert!(matches!(
            prepare_user_operation(resume, AuthorityState::Anonymous),
            OperationPreparation::Ready(UserOperation::PlatformResume { .. })
        ));

        let verification_path = "/platform/verify-email/123e4567-e89b-12d3-a456-426614174000";
        for authority in [AuthorityState::Anonymous, bound_platform_authority()] {
            assert!(matches!(
                prepare_user_operation(
                    envelope(
                        LogicalMethod::Get,
                        verification_path,
                        Vec::new(),
                        None,
                        None,
                    ),
                    authority,
                ),
                OperationPreparation::Ready(UserOperation::PlatformVerifyEmail { .. })
            ));
        }

        for (path, expected_request) in [
            ("/platform/password-reset/request", true),
            ("/platform/password-reset/confirm", false),
        ] {
            let prepared = prepare_user_operation(
                envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None),
                AuthorityState::Anonymous,
            );
            assert!(matches!(
                (expected_request, prepared),
                (
                    true,
                    OperationPreparation::Ready(UserOperation::PlatformPasswordResetRequest { .. })
                ) | (
                    false,
                    OperationPreparation::Ready(UserOperation::PlatformPasswordResetConfirm { .. })
                )
            ));
        }

        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/platform/logout",
                    json_header(),
                    Some(b"{}"),
                    None,
                ),
                bound_platform_authority(),
            ),
            OperationPreparation::Ready(UserOperation::PlatformLogout { .. })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/platform/request_verification",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_platform_authority(),
            ),
            OperationPreparation::Ready(UserOperation::PlatformRequestVerification { .. })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/platform/change-password",
                    json_header(),
                    Some(b"{}"),
                    None,
                ),
                bound_platform_authority(),
            ),
            OperationPreparation::Ready(UserOperation::PlatformChangePassword { .. })
        ));
    }

    #[test]
    fn platform_lifecycle_rejects_wrong_authority_and_transplanted_metadata() {
        for path in [
            "/platform/logout",
            "/platform/request_verification",
            "/platform/change-password",
        ] {
            let bodyful = path != "/platform/request_verification";
            let request = envelope(
                LogicalMethod::Post,
                path,
                if bodyful { json_header() } else { Vec::new() },
                bodyful.then_some(b"{}".as_slice()),
                None,
            );
            for authority in [AuthorityState::Anonymous, bound_user_authority()] {
                assert!(matches!(
                    prepare_user_operation(request.clone(), authority),
                    OperationPreparation::Rejected(response)
                        if response.status == StatusCode::UNAUTHORIZED
                ));
            }

            let mut transplanted = request;
            transplanted.request.query = Some("admin=true".to_owned());
            assert!(matches!(
                prepare_user_operation(transplanted, bound_platform_authority()),
                OperationPreparation::Rejected(response)
                    if response.status == StatusCode::BAD_REQUEST
            ));
        }

        let platform_login = envelope(
            LogicalMethod::Post,
            "/platform/login",
            json_header(),
            Some(b"{}"),
            None,
        );
        assert!(matches!(
            prepare_user_operation(platform_login, bound_platform_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::CONFLICT
        ));

        let platform_verification = envelope(
            LogicalMethod::Get,
            "/platform/verify-email/123e4567-e89b-12d3-a456-426614174000",
            Vec::new(),
            None,
            None,
        );
        assert!(matches!(
            prepare_user_operation(platform_verification, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));
    }

    #[test]
    fn platform_authentication_and_terminal_session_effects_are_classified() {
        for operation in [
            UserOperation::PlatformLogin {
                body: Zeroizing::new(Vec::new()),
            },
            UserOperation::PlatformRegister {
                body: Zeroizing::new(Vec::new()),
            },
            UserOperation::PlatformResume {
                credential: Zeroizing::new(Vec::new()),
            },
        ] {
            assert!(operation.requires_authentication_transition());
            assert_eq!(operation.session_effect_on_success(), SessionEffect::Retain);
        }

        let authority = BoundPlatformAuthority {
            platform_user_id: uuid::Uuid::nil(),
        };
        assert_eq!(
            UserOperation::PlatformLogout {
                authority: authority.clone(),
                body: Zeroizing::new(Vec::new()),
            }
            .session_effect_on_success(),
            SessionEffect::Close
        );
        assert_eq!(
            UserOperation::PlatformChangePassword {
                authority,
                body: Zeroizing::new(Vec::new()),
            }
            .session_effect_on_success(),
            SessionEffect::Close
        );
    }

    #[test]
    fn verification_resend_rejects_unbound_and_transplanted_metadata() {
        let request = || {
            envelope(
                LogicalMethod::Post,
                "/protected/request_verification",
                Vec::new(),
                None,
                None,
            )
        };
        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_header = request();
        with_header.request.headers = json_header();
        assert!(matches!(
            prepare_user_operation(with_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_body = request();
        with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(with_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn account_deletion_request_requires_a_bound_user_and_exact_json_request() {
        let request = || {
            envelope(
                LogicalMethod::Post,
                "/protected/delete-account/request",
                json_header(),
                Some(br#"{"hashed_secret":"client-hash"}"#),
                None,
            )
        };
        let OperationPreparation::Ready(UserOperation::Protected { operation, .. }) =
            prepare_user_operation(request(), bound_user_authority())
        else {
            panic!("exact account deletion request must be admitted");
        };
        assert!(matches!(
            &operation,
            ProtectedUserOperation::RequestAccountDeletion { .. }
        ));
        assert_eq!(operation.session_effect_on_success(), SessionEffect::Retain);
        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_extra_header = request();
        with_extra_header.request.headers.push(HeaderField {
            name: "accept".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_extra_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut without_body = request();
        without_body.request.body_base64 = None;
        assert!(matches!(
            prepare_user_operation(without_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn account_deletion_confirmation_is_exact_and_terminal_only_on_success() {
        let request = || {
            envelope(
                LogicalMethod::Post,
                "/protected/delete-account/confirm",
                json_header(),
                Some(
                    br#"{"confirmation_code":"123e4567-e89b-12d3-a456-426614174000","plaintext_secret":"client-secret"}"#,
                ),
                None,
            )
        };
        let OperationPreparation::Ready(UserOperation::Protected { operation, .. }) =
            prepare_user_operation(request(), bound_user_authority())
        else {
            panic!("exact account deletion confirmation must be admitted");
        };
        assert!(matches!(
            &operation,
            ProtectedUserOperation::ConfirmAccountDeletion { .. }
        ));
        assert_eq!(operation.session_effect_on_success(), SessionEffect::Close);

        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_extra_header = request();
        with_extra_header.request.headers.push(HeaderField {
            name: "accept".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_extra_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut without_body = request();
        without_body.request.body_base64 = None;
        assert!(matches!(
            prepare_user_operation(without_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_empty_body = request();
        with_empty_body.request.body_base64 = Some(EncodedBytes::from_bytes(Vec::new()));
        assert!(matches!(
            prepare_user_operation(with_empty_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut streaming = request();
        streaming.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(streaming, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn logout_is_exact_bound_and_terminal_only_on_success() {
        let request = || {
            envelope(
                LogicalMethod::Post,
                "/logout",
                json_header(),
                Some(br#"{"refresh_token":"resumption-token"}"#),
                None,
            )
        };
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(request(), bound_user_authority())
        else {
            panic!("exact user logout must be admitted");
        };
        assert!(matches!(&operation, UserOperation::Logout { .. }));
        assert_eq!(operation.session_effect_on_success(), SessionEffect::Close);
        let parsed = parse_json_body::<LogoutRequest>(Zeroizing::new(
            br#"{"refresh_token":"resumption-token","push_device_id":"123e4567-e89b-12d3-a456-426614174000"}"#
                .to_vec(),
        ))
        .expect("the released Rust SDK logout extension must remain accepted");
        assert_eq!(
            logout_data(parsed),
            serde_json::json!({ "message": "Logged out successfully" })
        );

        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));
        assert!(matches!(
            prepare_user_operation(
                request(),
                AuthorityState::Bound(BoundAuthority::platform(
                    uuid::Uuid::from_bytes([0x42; 16]),
                    Instant::now() + std::time::Duration::from_secs(60),
                )),
            ),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));
        assert!(matches!(
            prepare_user_operation(
                request(),
                AuthorityState::Bound(BoundAuthority::api_key(
                    17,
                    uuid::Uuid::from_bytes([0x43; 16]),
                    derive_tinfoil_cache_namespace(
                        &CacheNamespaceRoot::from_bytes([0x44; 32]),
                        uuid::Uuid::from_bytes([0x43; 16]),
                    ),
                )),
            ),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_extra_header = request();
        with_extra_header.request.headers.push(HeaderField {
            name: "accept".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_extra_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut without_body = request();
        without_body.request.body_base64 = None;
        assert!(matches!(
            prepare_user_operation(without_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_empty_body = request();
        with_empty_body.request.body_base64 = Some(EncodedBytes::from_bytes(Vec::new()));
        assert!(matches!(
            prepare_user_operation(with_empty_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut streaming = request();
        streaming.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(streaming, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn password_change_is_exact_bound_and_terminal_only_on_success() {
        let request = || {
            envelope(
                LogicalMethod::Post,
                "/protected/change_password",
                json_header(),
                Some(br#"{"current_password":"old","new_password":"new"}"#),
                None,
            )
        };
        let OperationPreparation::Ready(operation) =
            prepare_user_operation(request(), bound_user_authority())
        else {
            panic!("exact password change must be admitted");
        };
        assert!(matches!(&operation, UserOperation::ChangePassword { .. }));
        assert_eq!(operation.session_effect_on_success(), SessionEffect::Close);

        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_extra_header = request();
        with_extra_header.request.headers.push(HeaderField {
            name: "accept".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"application/json".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_extra_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut without_body = request();
        without_body.request.body_base64 = None;
        assert!(matches!(
            prepare_user_operation(without_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_empty_body = request();
        with_empty_body.request.body_base64 = Some(EncodedBytes::from_bytes(Vec::new()));
        assert!(matches!(
            prepare_user_operation(with_empty_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut streaming = request();
        streaming.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(streaming, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn email_verification_is_a_canonical_code_operation_without_rebinding() {
        let path = "/verify-email/123e4567-e89b-12d3-a456-426614174000";
        let request = || envelope(LogicalMethod::Get, path, Vec::new(), None, None);
        for authority in [AuthorityState::Anonymous, bound_user_authority()] {
            assert!(matches!(
                prepare_user_operation(request(), authority),
                OperationPreparation::Ready(UserOperation::VerifyEmail { code })
                    if code == uuid::Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
            ));
        }

        for authority in [bound_api_key_authority(), bound_platform_authority()] {
            assert_eq!(
                rejected_status(prepare_user_operation(request(), authority)),
                StatusCode::UNAUTHORIZED
            );
        }

        assert!(matches!(
            prepare_user_operation(
                request(),
                AuthorityState::Authenticating(RequestId::from_bytes([0x41; 16])),
            ),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::CONFLICT
        ));
        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Closing),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::CONFLICT
        ));

        let mut with_query = request();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(with_query, AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_header = request();
        with_header.request.headers = json_header();
        assert!(matches!(
            prepare_user_operation(with_header, AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_body = request();
        with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(with_body, AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut with_credential = request();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(with_credential, AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let uppercase = envelope(
            LogicalMethod::Get,
            "/verify-email/123E4567-E89B-12D3-A456-426614174000",
            Vec::new(),
            None,
            None,
        );
        assert!(matches!(
            prepare_user_operation(uppercase, AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn exact_sensitive_user_operation_contracts_are_admitted() {
        let mut private_key = envelope(
            LogicalMethod::Get,
            "/protected/private_key",
            Vec::new(),
            None,
            None,
        );
        let private_key_query =
            "seed_phrase_derivation_path=m%2F83696968%27%2F39%27%2F0%27%2F12%27%2F0%27".to_owned();
        private_key.request.query = Some(private_key_query.clone());
        assert!(matches!(
            prepare_user_operation(private_key, bound_user_authority()),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::GetPrivateKey { query: Some(query) },
                ..
            }) if query == private_key_query
        ));

        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Get,
                    "/protected/private_key_bytes",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::GetPrivateKeyBytes { .. },
                ..
            })
        ));

        let mut public_key = envelope(
            LogicalMethod::Get,
            "/protected/public_key",
            Vec::new(),
            None,
            None,
        );
        let public_key_query = "algorithm=schnorr".to_owned();
        public_key.request.query = Some(public_key_query.clone());
        assert!(matches!(
            prepare_user_operation(public_key, bound_user_authority()),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::GetPublicKey { query: Some(query) },
                ..
            }) if query == public_key_query
        ));

        for (path, expected) in [
            ("/protected/sign_message", "sign"),
            ("/protected/third_party_token", "third_party"),
            ("/protected/encrypt", "encrypt"),
            ("/protected/decrypt", "decrypt"),
        ] {
            let prepared = prepare_user_operation(
                envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None),
                bound_user_authority(),
            );
            assert!(
                matches!(
                    (&prepared, expected),
                    (
                        OperationPreparation::Ready(UserOperation::Protected {
                            operation: ProtectedUserOperation::SignMessage { .. },
                            ..
                        }),
                        "sign"
                    ) | (
                        OperationPreparation::Ready(UserOperation::Protected {
                            operation: ProtectedUserOperation::IssueThirdPartyToken { .. },
                            ..
                        }),
                        "third_party"
                    ) | (
                        OperationPreparation::Ready(UserOperation::Protected {
                            operation: ProtectedUserOperation::EncryptData { .. },
                            ..
                        }),
                        "encrypt"
                    ) | (
                        OperationPreparation::Ready(UserOperation::Protected {
                            operation: ProtectedUserOperation::DecryptData { .. },
                            ..
                        }),
                        "decrypt"
                    )
                ),
                "{path}"
            );
        }
    }

    #[test]
    fn exact_kv_mutation_contracts_are_admitted_with_decoded_keys() {
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Put,
                    "/protected/kv/key%2Fpart",
                    json_header(),
                    Some(br#""value""#),
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::PutKv { key, .. },
                ..
            }) if &*key == "key/part"
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Put,
                    "/protected/kv/empty",
                    json_header(),
                    Some(br#""""#),
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::PutKv { .. },
                ..
            })
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Delete,
                    "/protected/kv/%252F",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::DeleteKv { key },
                ..
            }) if &*key == "%2F"
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Delete,
                    "/protected/kv",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::DeleteAllKv,
                ..
            })
        ));
    }

    #[test]
    fn exact_api_key_mutation_contracts_are_admitted() {
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/protected/api-keys",
                    json_header(),
                    Some(br#"{"name":"Production Key-1_test"}"#),
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::CreateApiKey { .. },
                ..
            })
        ));

        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Delete,
                    "/protected/api-keys/Production%20Key%2D1%5Ftest",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::DeleteApiKey { name },
                ..
            }) if &*name == "Production Key-1_test"
        ));

        let list = prepare_user_operation(
            envelope(
                LogicalMethod::Get,
                "/protected/api-keys",
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        assert!(matches!(
            &list,
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::ListApiKeys,
                ..
            })
        ));
        let OperationPreparation::Ready(list) = list else {
            unreachable!("matched ready API-key list operation")
        };
        assert!(list.requires_stored_output_reservation());
    }

    #[test]
    fn conversation_project_family_is_exact_bound_and_classified_by_storage_risk() {
        let create = || {
            envelope(
                LogicalMethod::Post,
                "/v1/conversation-projects",
                json_header(),
                Some(br#"{"name":"  Project name  "}"#),
                None,
            )
        };
        let create_operation = prepare_user_operation(create(), bound_user_authority());
        assert!(matches!(
            &create_operation,
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::CreateConversationProject { .. },
                ..
            })
        ));
        let OperationPreparation::Ready(create_operation) = create_operation else {
            unreachable!("matched ready conversation-project creation")
        };
        assert!(!create_operation.requires_stored_output_reservation());

        let project_id = "123e4567-e89b-12d3-a456-426614174000";
        let delete = || {
            envelope(
                LogicalMethod::Delete,
                &format!("/v1/conversation-projects/{project_id}"),
                Vec::new(),
                None,
                None,
            )
        };
        let delete_operation = prepare_user_operation(delete(), bound_user_authority());
        assert!(matches!(
            &delete_operation,
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::DeleteConversationProject {
                    project_id: decoded,
                },
                ..
            }) if *decoded == uuid::Uuid::parse_str(project_id).unwrap()
        ));
        let OperationPreparation::Ready(delete_operation) = delete_operation else {
            unreachable!("matched ready conversation-project deletion")
        };
        assert!(!delete_operation.requires_stored_output_reservation());

        for request in [create(), delete()] {
            assert!(matches!(
                prepare_user_operation(request, AuthorityState::Anonymous),
                OperationPreparation::Rejected(response)
                    if response.status == StatusCode::UNAUTHORIZED
            ));
        }

        let mut create_with_query = create();
        create_with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(create_with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut create_with_credential = create();
        create_with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(create_with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut delete_with_body = delete();
        delete_with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(delete_with_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut streaming = create();
        streaming.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(streaming, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let list = prepare_user_operation(
            envelope(
                LogicalMethod::Get,
                "/v1/conversation-projects",
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        let get = prepare_user_operation(
            envelope(
                LogicalMethod::Get,
                &format!("/v1/conversation-projects/{project_id}"),
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        let update = prepare_user_operation(
            envelope(
                LogicalMethod::Post,
                &format!("/v1/conversation-projects/{project_id}"),
                json_header(),
                Some(br#"{"instructions":null}"#),
                None,
            ),
            bound_user_authority(),
        );
        for operation in [list, get, update] {
            let OperationPreparation::Ready(operation) = operation else {
                panic!("bounded conversation-project operation must be admitted");
            };
            assert!(operation.requires_stored_output_reservation());
        }
    }

    #[test]
    fn instruction_family_is_exact_bound_and_classified_by_storage_risk() {
        let instruction_id = "123e4567-e89b-12d3-a456-426614174000";
        let item_path = format!("/v1/instructions/{instruction_id}");
        let create = prepare_user_operation(
            envelope(
                LogicalMethod::Post,
                "/v1/instructions",
                json_header(),
                Some(br#"{"name":"n","prompt":"p"}"#),
                None,
            ),
            bound_user_authority(),
        );
        let delete = prepare_user_operation(
            envelope(LogicalMethod::Delete, &item_path, Vec::new(), None, None),
            bound_user_authority(),
        );
        for operation in [create, delete] {
            let OperationPreparation::Ready(operation) = operation else {
                panic!("fixed-output instruction mutation must be admitted");
            };
            assert!(!operation.requires_stored_output_reservation());
        }

        let list = prepare_user_operation(
            envelope(
                LogicalMethod::Get,
                "/v1/instructions",
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        let get = prepare_user_operation(
            envelope(LogicalMethod::Get, &item_path, Vec::new(), None, None),
            bound_user_authority(),
        );
        let update = prepare_user_operation(
            envelope(
                LogicalMethod::Post,
                &item_path,
                json_header(),
                Some(br#"{"is_default":false}"#),
                None,
            ),
            bound_user_authority(),
        );
        let set_default = prepare_user_operation(
            envelope(
                LogicalMethod::Post,
                &format!("{item_path}/set-default"),
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        for operation in [list, get, update, set_default] {
            let OperationPreparation::Ready(operation) = operation else {
                panic!("stored instruction operation must be admitted");
            };
            assert!(operation.requires_stored_output_reservation());
        }

        let mut transplanted = envelope(
            LogicalMethod::Post,
            &item_path,
            json_header(),
            Some(br#"{"name":"n"}"#),
            None,
        );
        transplanted.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"stolen".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(transplanted, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
        assert!(matches!(
            prepare_user_operation(
                envelope(LogicalMethod::Get, &item_path, Vec::new(), None, None),
                AuthorityState::Anonymous,
            ),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));
    }

    #[test]
    fn conversation_item_and_stored_response_families_are_exact_and_classified_by_storage_risk() {
        let conversation = "123e4567-e89b-12d3-a456-426614174000";
        let item = "223e4567-e89b-12d3-a456-426614174000";
        let response = "323e4567-e89b-12d3-a456-426614174000";
        let conversation_path = format!("/v1/conversations/{conversation}");
        let items_path = format!("{conversation_path}/items");
        let response_path = format!("/v1/responses/{response}");

        let cases = [
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/conversations",
                    json_header(),
                    Some(br#"{}"#),
                    None,
                ),
                "create_conversation",
                false,
            ),
            (
                {
                    let mut request = envelope(
                        LogicalMethod::Get,
                        "/v1/conversations",
                        Vec::new(),
                        None,
                        None,
                    );
                    request.request.query = Some("limit=10&order=desc".to_owned());
                    request
                },
                "list_conversations",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Get,
                    &conversation_path,
                    Vec::new(),
                    None,
                    None,
                ),
                "get_conversation",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    &conversation_path,
                    json_header(),
                    Some(br#"{"pinned":true}"#),
                    None,
                ),
                "update_conversation",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Delete,
                    &conversation_path,
                    Vec::new(),
                    None,
                    None,
                ),
                "delete_conversation",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Delete,
                    "/v1/conversations",
                    Vec::new(),
                    None,
                    None,
                ),
                "delete_all_conversations",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/conversations/batch-delete",
                    json_header(),
                    Some(br#"{"ids":["123e4567-e89b-12d3-a456-426614174000"]}"#),
                    None,
                ),
                "batch_delete_conversations",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/conversations/batch-update-project",
                    json_header(),
                    Some(br#"{"ids":["123e4567-e89b-12d3-a456-426614174000"],"project_id":null}"#),
                    None,
                ),
                "batch_update_conversation_project",
                false,
            ),
            (
                {
                    let mut request =
                        envelope(LogicalMethod::Get, &items_path, Vec::new(), None, None);
                    request.request.query = Some("limit=5&include=ignored".to_owned());
                    request
                },
                "list_conversation_items",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Get,
                    &format!("{items_path}/{item}"),
                    Vec::new(),
                    None,
                    None,
                ),
                "get_conversation_item",
                true,
            ),
            (
                envelope(LogicalMethod::Get, &response_path, Vec::new(), None, None),
                "get_stored_response",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    &format!("{response_path}/cancel"),
                    Vec::new(),
                    None,
                    None,
                ),
                "cancel_stored_response",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Delete,
                    &response_path,
                    Vec::new(),
                    None,
                    None,
                ),
                "delete_stored_response",
                false,
            ),
        ];

        for (request, expected, stored_output) in cases {
            let prepared = prepare_user_operation(request, bound_user_authority());
            let OperationPreparation::Ready(operation) = prepared else {
                panic!("{expected} must be admitted");
            };
            let UserOperation::Protected {
                operation: protected,
                ..
            } = &operation
            else {
                panic!("{expected} must be a protected user operation");
            };
            let actual = match protected {
                ProtectedUserOperation::CreateConversation { .. } => "create_conversation",
                ProtectedUserOperation::ListConversations { .. } => "list_conversations",
                ProtectedUserOperation::GetConversation { .. } => "get_conversation",
                ProtectedUserOperation::UpdateConversation { .. } => "update_conversation",
                ProtectedUserOperation::DeleteConversation { .. } => "delete_conversation",
                ProtectedUserOperation::DeleteAllConversations => "delete_all_conversations",
                ProtectedUserOperation::BatchDeleteConversations { .. } => {
                    "batch_delete_conversations"
                }
                ProtectedUserOperation::BatchUpdateConversationProject { .. } => {
                    "batch_update_conversation_project"
                }
                ProtectedUserOperation::ListConversationItems { .. } => "list_conversation_items",
                ProtectedUserOperation::GetConversationItem { .. } => "get_conversation_item",
                ProtectedUserOperation::GetStoredResponse { .. } => "get_stored_response",
                ProtectedUserOperation::CancelStoredResponse { .. } => "cancel_stored_response",
                ProtectedUserOperation::DeleteStoredResponse { .. } => "delete_stored_response",
                _ => panic!("unexpected operation for {expected}"),
            };
            assert_eq!(actual, expected);
            assert_eq!(
                operation.requires_stored_output_reservation(),
                stored_output
            );
        }

        for request in [
            envelope(
                LogicalMethod::Get,
                &conversation_path,
                Vec::new(),
                None,
                None,
            ),
            envelope(LogicalMethod::Get, &response_path, Vec::new(), None, None),
        ] {
            assert!(matches!(
                prepare_user_operation(request, AuthorityState::Anonymous),
                OperationPreparation::Rejected(response)
                    if response.status == StatusCode::UNAUTHORIZED
            ));
        }

        let mut transplanted_body = envelope(
            LogicalMethod::Delete,
            &conversation_path,
            Vec::new(),
            None,
            None,
        );
        transplanted_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(transplanted_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let disabled_item_post = envelope(
            LogicalMethod::Post,
            &items_path,
            json_header(),
            Some(br#"{"items":[]}"#),
            None,
        );
        assert!(matches!(
            prepare_user_operation(disabled_item_post, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn conversation_project_creation_preflight_matches_wire_shape_and_bounds_before_insert() {
        let name = "quoted \\\" project ☃";
        let candidate = ConversationProjectCreationResponsePreflight {
            id: uuid::Uuid::nil(),
            object: OBJECT_TYPE_CONVERSATION_PROJECT,
            name,
            instructions: None,
            created_at: i64::MIN,
            updated_at: i64::MIN,
        };
        let actual = crate::web::responses::conversation_projects::ConversationProjectResponse {
            id: uuid::Uuid::nil(),
            object: OBJECT_TYPE_CONVERSATION_PROJECT,
            name: name.to_owned(),
            instructions: None,
            created_at: i64::MIN,
            updated_at: i64::MIN,
        };
        assert_eq!(
            serde_json::to_vec(&candidate).unwrap(),
            serde_json::to_vec(&actual).unwrap(),
        );
        assert!(matches!(
            preflight_conversation_project_response_with_limit(name, None, 8),
            Err(ApiError::PayloadTooLarge)
        ));
    }

    #[test]
    fn instruction_preflight_matches_wire_shape_and_bounds_before_default_mutation() {
        let name = "  quoted \" instruction  ";
        let prompt = "  preserve prompt whitespace ☃  ";
        let candidate = InstructionResponsePreflight {
            id: uuid::Uuid::nil(),
            object: "instruction",
            name,
            prompt,
            prompt_tokens: i32::MIN,
            is_default: true,
            created_at: i64::MIN,
            updated_at: i64::MIN,
        };
        let actual = InstructionResponse {
            id: uuid::Uuid::nil(),
            object: "instruction",
            name: name.to_owned(),
            prompt: prompt.to_owned(),
            prompt_tokens: i32::MIN,
            is_default: true,
            created_at: i64::MIN,
            updated_at: i64::MIN,
        };
        assert_eq!(
            serde_json::to_vec(&candidate).unwrap(),
            serde_json::to_vec(&actual).unwrap(),
        );
        assert!(matches!(
            preflight_instruction_response_with_limit(name, prompt, true, 8),
            Err(ApiError::PayloadTooLarge)
        ));
    }

    fn stored_item_fixture(
        message_type: &str,
        uuid: uuid::Uuid,
        content_enc: Option<Vec<u8>>,
        token_count: i32,
    ) -> stored_conversations::StoredConversationItem {
        stored_conversations::StoredConversationItem {
            message_type: message_type.to_owned(),
            id: 1,
            type_rank: 1,
            uuid,
            content_enc,
            status: Some(STATUS_COMPLETED.to_owned()),
            created_at: Utc::now(),
            model: None,
            token_count: Some(token_count),
            tool_call_id: None,
            finish_reason: None,
            tool_name: None,
        }
    }

    #[tokio::test]
    async fn bounded_conversation_items_preserve_each_existing_wire_variant() {
        let user_key = secp256k1::SecretKey::from_slice(&[0x51; 32]).unwrap();
        let user_id = uuid::Uuid::from_bytes([0x52; 16]);
        let assistant_id = uuid::Uuid::from_bytes([0x53; 16]);
        let tool_call_id = uuid::Uuid::from_bytes([0x54; 16]);
        let tool_output_id = uuid::Uuid::from_bytes([0x55; 16]);
        let reasoning_id = uuid::Uuid::from_bytes([0x56; 16]);

        let user_plaintext =
            Zeroizing::new(serde_json::to_vec(&MessageContent::Text("hello".to_owned())).unwrap());
        let user = stored_item_fixture(
            "user",
            user_id,
            Some(encrypt_with_key(&user_key, &user_plaintext).await),
            1,
        );
        let user_timestamp = user.created_at.timestamp();

        let assistant = stored_item_fixture(
            "assistant",
            assistant_id,
            Some(encrypt_with_key(&user_key, b"answer").await),
            2,
        );
        let assistant_timestamp = assistant.created_at.timestamp();

        let mut tool_call = stored_item_fixture(
            "tool_call",
            tool_call_id,
            Some(encrypt_with_key(&user_key, br#"{"city":"Oslo"}"#).await),
            3,
        );
        tool_call.tool_call_id = Some(tool_call_id);
        tool_call.tool_name = Some("weather".to_owned());
        let tool_call_timestamp = tool_call.created_at.timestamp();

        let mut tool_output = stored_item_fixture(
            "tool_output",
            tool_output_id,
            Some(encrypt_with_key(&user_key, b"sunny").await),
            4,
        );
        tool_output.tool_call_id = Some(tool_call_id);
        let tool_output_timestamp = tool_output.created_at.timestamp();

        let reasoning = stored_item_fixture(
            "reasoning",
            reasoning_id,
            Some(encrypt_with_key(&user_key, b"private thought").await),
            5,
        );
        let reasoning_timestamp = reasoning.created_at.timestamp();

        let actual = [user, assistant, tool_call, tool_output, reasoning]
            .into_iter()
            .map(|item| {
                serde_json::to_value(stored_item_to_conversation_item(item, &user_key).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                serde_json::json!({
                    "type": "message", "id": user_id, "status": "completed", "role": "user",
                    "content": [{"type":"input_text","text":"hello"}],
                    "created_at": user_timestamp
                }),
                serde_json::json!({
                    "type": "message", "id": assistant_id, "status": "completed", "role": "assistant",
                    "content": [{"type":"output_text","text":"answer"}],
                    "created_at": assistant_timestamp
                }),
                serde_json::json!({
                    "type": "function_call", "id": tool_call_id, "call_id": tool_call_id,
                    "name": "weather", "arguments": "{\"city\":\"Oslo\"}",
                    "status": "completed", "created_at": tool_call_timestamp
                }),
                serde_json::json!({
                    "type": "function_call_output", "id": tool_output_id, "call_id": tool_call_id,
                    "output": "sunny", "status": "completed", "created_at": tool_output_timestamp
                }),
                serde_json::json!({
                    "type": "reasoning", "id": reasoning_id,
                    "content": [{"type":"text","text":"private thought"}],
                    "status": "completed", "created_at": reasoning_timestamp
                }),
            ]
        );
    }

    #[tokio::test]
    async fn bounded_stored_response_preserves_output_filtering_and_usage_math() {
        let user_key = secp256k1::SecretKey::from_slice(&[0x61; 32]).unwrap();
        let response_id = uuid::Uuid::from_bytes([0x62; 16]);
        let user_item_id = uuid::Uuid::from_bytes([0x64; 16]);
        let assistant_id = uuid::Uuid::from_bytes([0x65; 16]);
        let empty_assistant_id = uuid::Uuid::from_bytes([0x69; 16]);
        let call_id = uuid::Uuid::from_bytes([0x66; 16]);
        let output_id = uuid::Uuid::from_bytes([0x67; 16]);
        let reasoning_id = uuid::Uuid::from_bytes([0x68; 16]);

        let user = stored_item_fixture("user", user_item_id, None, 3);
        let assistant = stored_item_fixture(
            "assistant",
            assistant_id,
            Some(encrypt_with_key(&user_key, b"answer").await),
            5,
        );
        let empty_assistant = stored_item_fixture("assistant", empty_assistant_id, None, 0);
        let mut tool_call = stored_item_fixture(
            "tool_call",
            call_id,
            Some(encrypt_with_key(&user_key, b"{}").await),
            2,
        );
        tool_call.tool_call_id = Some(call_id);
        tool_call.tool_name = Some("lookup".to_owned());
        let mut tool_output = stored_item_fixture(
            "tool_output",
            output_id,
            Some(encrypt_with_key(&user_key, b"ok").await),
            4,
        );
        tool_output.tool_call_id = Some(call_id);
        let reasoning = stored_item_fixture("reasoning", reasoning_id, None, 7);
        let created_at = Utc::now();
        let stored = stored_conversations::StoredResponse {
            response: stored_conversations::StoredResponseMetadata {
                id: 1,
                uuid: response_id,
                conversation_id: 2,
                status: ResponseStatus::Completed,
                model: "test-model".to_owned(),
                created_at,
            },
            items: vec![
                user,
                assistant,
                empty_assistant,
                tool_call,
                tool_output,
                reasoning,
            ],
        };

        let actual =
            serde_json::to_value(stored_response_to_wire(stored, &user_key).unwrap()).unwrap();
        assert_eq!(actual["id"], serde_json::json!(response_id));
        assert_eq!(actual["status"], "completed");
        assert_eq!(actual["model"], "test-model");
        assert_eq!(actual["usage"]["input_tokens"], 9);
        assert_eq!(actual["usage"]["output_tokens"], 12);
        assert_eq!(
            actual["usage"]["output_tokens_details"]["reasoning_tokens"],
            7
        );
        assert_eq!(actual["usage"]["total_tokens"], 21);
        assert_eq!(
            actual["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "message",
                "message",
                "tool_call",
                "tool_output",
                "reasoning"
            ]
        );
        assert_eq!(actual["output"][1]["content"], serde_json::json!([]));
        assert_eq!(actual["output"][4]["content"], serde_json::json!([]));
        assert!(actual["output"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["id"] != serde_json::json!(user_item_id)));
    }

    #[test]
    fn conversation_json_shape_is_bounded_before_value_allocation() {
        assert_eq!(
            validate_json_shape(
                br#"{"metadata":{"title":"{[,:","escaped":"quote: \" ok"},"future":true}"#
            ),
            Ok(())
        );

        let too_deep = format!(
            "{}0{}",
            "[".repeat(MAX_CONVERSATION_JSON_DEPTH + 1),
            "]".repeat(MAX_CONVERSATION_JSON_DEPTH + 1)
        );
        assert_eq!(
            validate_json_shape(too_deep.as_bytes()),
            Err(JsonShapeError::TooLarge)
        );

        let mut too_wide = String::from("[");
        for index in 0..=MAX_CONVERSATION_JSON_STRUCTURAL_TOKENS {
            if index != 0 {
                too_wide.push(',');
            }
            too_wide.push('0');
        }
        too_wide.push(']');
        assert_eq!(
            validate_json_shape(too_wide.as_bytes()),
            Err(JsonShapeError::TooLarge)
        );
        assert_eq!(
            validate_json_shape(br#"{"metadata":"unterminated}"#),
            Err(JsonShapeError::Malformed)
        );

        let parsed = parse_conversation_json_body::<UpdateConversationRequest>(Zeroizing::new(
            br#"{"pinned":true,"future":{"shape":"ignored"}}"#.to_vec(),
        ))
        .expect("unknown fields must remain accepted after structural preflight");
        assert_eq!(parsed.pinned, Some(true));
    }

    #[test]
    fn api_key_administration_rejects_anonymous_and_transplanted_metadata() {
        let create = || {
            envelope(
                LogicalMethod::Post,
                "/protected/api-keys",
                json_header(),
                Some(br#"{"name":"Production Key"}"#),
                None,
            )
        };
        assert!(matches!(
            prepare_user_operation(create(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut create_with_query = create();
        create_with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(create_with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut create_with_credential = create();
        create_with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(create_with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut create_stream = create();
        create_stream.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(create_stream, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let delete = || {
            envelope(
                LogicalMethod::Delete,
                "/protected/api-keys/Production%20Key",
                Vec::new(),
                None,
                None,
            )
        };
        let mut delete_with_body = delete();
        delete_with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(delete_with_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut delete_with_header = delete();
        delete_with_header.request.headers = json_header();
        assert!(matches!(
            prepare_user_operation(delete_with_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut delete_with_query = delete();
        delete_with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(delete_with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let list = || {
            envelope(
                LogicalMethod::Get,
                "/protected/api-keys",
                Vec::new(),
                None,
                None,
            )
        };
        assert!(matches!(
            prepare_user_operation(list(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut list_with_query = list();
        list_with_query.request.query = Some("transplanted=true".to_owned());
        assert!(matches!(
            prepare_user_operation(list_with_query, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut list_with_header = list();
        list_with_header.request.headers = json_header();
        assert!(matches!(
            prepare_user_operation(list_with_header, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut list_with_body = list();
        list_with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert!(matches!(
            prepare_user_operation(list_with_body, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));

        let mut list_with_credential = list();
        list_with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert!(matches!(
            prepare_user_operation(list_with_credential, bound_user_authority()),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::BAD_REQUEST
        ));
    }

    #[test]
    fn exact_kv_read_contracts_are_admitted_and_require_stored_output_reservation() {
        let item = prepare_user_operation(
            envelope(
                LogicalMethod::Get,
                "/protected/kv/key%2Fpart",
                Vec::new(),
                None,
                None,
            ),
            bound_user_authority(),
        );
        assert!(matches!(
            &item,
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::GetKv { key },
                ..
            }) if &**key == "key/part"
        ));
        let OperationPreparation::Ready(item) = item else {
            unreachable!("validated item read must be ready");
        };
        assert!(item.requires_stored_output_reservation());

        let list = prepare_user_operation(
            envelope(LogicalMethod::Get, "/protected/kv", Vec::new(), None, None),
            bound_user_authority(),
        );
        assert!(matches!(
            &list,
            OperationPreparation::Ready(UserOperation::Protected {
                operation: ProtectedUserOperation::ListKv,
                ..
            })
        ));
        let OperationPreparation::Ready(list) = list else {
            unreachable!("validated list read must be ready");
        };
        assert!(list.requires_stored_output_reservation());
    }

    #[test]
    fn kv_value_keeps_the_existing_json_string_wire_shape() {
        let wire = "\"line\\né\"";
        let value: KvValue = serde_json::from_slice(wire.as_bytes()).unwrap();
        assert_eq!(value.as_str(), "line\né");
        assert_eq!(serde_json::to_vec(&value).unwrap(), wire.as_bytes());
    }

    #[test]
    fn kv_reads_keep_existing_null_string_and_list_wire_shapes() {
        let missing = LogicalUnaryResponse::json(StatusCode::OK, &Option::<&str>::None).unwrap();
        assert_eq!(&*missing.body.unwrap(), b"null");

        let present = LogicalUnaryResponse::json(StatusCode::OK, &Some("line\né")).unwrap();
        assert_eq!(&*present.body.unwrap(), "\"line\\né\"".as_bytes());

        let empty =
            LogicalUnaryResponse::json(StatusCode::OK, &Vec::<crate::kv::KVPair>::new()).unwrap();
        assert_eq!(&*empty.body.unwrap(), b"[]");

        let rows = vec![crate::kv::KVPair {
            key: "key".to_owned(),
            value: "value".to_owned(),
            created_at: 1,
            updated_at: 2,
        }];
        let list = LogicalUnaryResponse::json(StatusCode::OK, &rows).unwrap();
        assert_eq!(
            &*list.body.unwrap(),
            br#"[{"key":"key","value":"value","created_at":1,"updated_at":2}]"#
        );
    }

    #[test]
    fn coded_api_errors_preserve_the_established_error_contract_inside_v2() {
        let response = LogicalUnaryResponse::api_error(ApiError::SessionNotFound);

        assert_eq!(response.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            response.body.as_ref().map(|body| body.as_slice()),
            Some(br#"{"status":400,"message":"Bad Request"}"#.as_slice())
        );
        assert_eq!(response.headers.len(), 3);
        assert_eq!(
            logical_header_value(&response, "content-type"),
            Some(JSON_CONTENT_TYPE)
        );
        assert_eq!(
            logical_header_value(&response, ERROR_CONTRACT_HEADER),
            Some(b"1".as_slice())
        );
        assert_eq!(
            logical_header_value(&response, ERROR_CODE_HEADER),
            Some(b"session_not_found".as_slice())
        );
    }

    #[test]
    fn uncoded_api_errors_include_only_the_established_contract_header_inside_v2() {
        let response = LogicalUnaryResponse::api_error(ApiError::BadRequest);

        assert_eq!(response.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            response.body.as_ref().map(|body| body.as_slice()),
            Some(br#"{"status":400,"message":"Bad Request"}"#.as_slice())
        );
        assert_eq!(response.headers.len(), 2);
        assert_eq!(
            logical_header_value(&response, "content-type"),
            Some(JSON_CONTENT_TYPE)
        );
        assert_eq!(
            logical_header_value(&response, ERROR_CONTRACT_HEADER),
            Some(b"1".as_slice())
        );
        assert_eq!(logical_header_value(&response, ERROR_CODE_HEADER), None);
    }

    #[test]
    fn api_key_list_keeps_existing_json_wire_shape() {
        use crate::web::protected_routes::{ApiKeyInfo, ListApiKeysResponse};

        let empty =
            LogicalUnaryResponse::json(StatusCode::OK, &ListApiKeysResponse { keys: Vec::new() })
                .unwrap();
        assert_eq!(&*empty.body.unwrap(), br#"{"keys":[]}"#);

        let populated = LogicalUnaryResponse::json(
            StatusCode::OK,
            &ListApiKeysResponse {
                keys: vec![ApiKeyInfo {
                    name: "Production Key".to_owned(),
                    created_at: "2026-08-29T12:34:56Z".parse().unwrap(),
                }],
            },
        )
        .unwrap();
        assert_eq!(
            &*populated.body.unwrap(),
            br#"{"keys":[{"name":"Production Key","created_at":"2026-08-29T12:34:56Z"}]}"#
        );
    }

    #[test]
    fn near_miss_user_operations_are_rejected_before_dispatch() {
        let mut wrong_mode = envelope(
            LogicalMethod::Post,
            "/login",
            json_header(),
            Some(b"{}"),
            None,
        );
        wrong_mode.response_mode = ResponseMode::Auto;
        assert!(matches!(
            prepare_user_operation(wrong_mode, AuthorityState::Anonymous),
            OperationPreparation::Rejected(_)
        ));

        for request in [
            envelope(LogicalMethod::Post, "/login", Vec::new(), Some(b"{}"), None),
            envelope(LogicalMethod::Post, "/login", json_header(), None, None),
            envelope(
                LogicalMethod::Post,
                "/refresh",
                Vec::new(),
                Some(b"{}"),
                Some(Credential::Resumption {
                    value_base64: EncodedBytes::from_bytes(b"token".to_vec()),
                }),
            ),
            envelope(
                LogicalMethod::Get,
                "/protected/user",
                Vec::new(),
                Some(b""),
                None,
            ),
        ] {
            assert!(matches!(
                prepare_user_operation(request, AuthorityState::Anonymous),
                OperationPreparation::Rejected(_)
            ));
        }
    }

    #[test]
    fn register_preserves_legacy_unknown_invite_code_tolerance() {
        let parsed: RegisterCredentials = serde_json::from_slice(
            br#"{"password":"p","client_id":"00000000-0000-0000-0000-000000000000","inviteCode":"legacy"}"#,
        )
        .expect("legacy SDK inviteCode remains ignored by application payload parsing");
        assert_eq!(parsed.password, "p");
    }

    #[test]
    fn bound_user_contract_rejects_rebinding_and_metadata_transplants() {
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Post,
                    "/login",
                    json_header(),
                    Some(b"{}"),
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Rejected(_)
        ));

        let mut query = envelope(
            LogicalMethod::Get,
            "/protected/user",
            Vec::new(),
            None,
            None,
        );
        query.request.query = Some("x=1".to_owned());
        for request in [
            query,
            envelope(
                LogicalMethod::Get,
                "/protected/user",
                json_header(),
                None,
                None,
            ),
            envelope(
                LogicalMethod::Get,
                "/protected/user",
                Vec::new(),
                Some(b""),
                None,
            ),
            envelope(
                LogicalMethod::Get,
                "/protected/user",
                Vec::new(),
                None,
                Some(Credential::ApiKey {
                    value_base64: EncodedBytes::from_bytes(b"key".to_vec()),
                }),
            ),
        ] {
            assert!(matches!(
                prepare_user_operation(request, bound_user_authority()),
                OperationPreparation::Rejected(_)
            ));
        }
    }

    #[test]
    fn sensitive_user_contracts_reject_unbound_and_transplanted_metadata() {
        for path in [
            "/protected/private_key",
            "/protected/private_key_bytes",
            "/protected/public_key",
        ] {
            assert!(matches!(
                prepare_user_operation(
                    envelope(LogicalMethod::Get, path, Vec::new(), None, None),
                    AuthorityState::Anonymous,
                ),
                OperationPreparation::Rejected(_)
            ));
            assert!(matches!(
                prepare_user_operation(
                    envelope(LogicalMethod::Get, path, json_header(), None, None),
                    bound_user_authority(),
                ),
                OperationPreparation::Rejected(_)
            ));
        }

        for path in [
            "/protected/sign_message",
            "/protected/third_party_token",
            "/protected/encrypt",
            "/protected/decrypt",
        ] {
            assert!(matches!(
                prepare_user_operation(
                    envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None),
                    AuthorityState::Anonymous,
                ),
                OperationPreparation::Rejected(_)
            ));

            let mut query = envelope(LogicalMethod::Post, path, json_header(), Some(b"{}"), None);
            query.request.query = Some("x=1".to_owned());
            for request in [
                envelope(LogicalMethod::Post, path, Vec::new(), Some(b"{}"), None),
                envelope(LogicalMethod::Post, path, json_header(), None, None),
                query,
                envelope(
                    LogicalMethod::Post,
                    path,
                    json_header(),
                    Some(b"{}"),
                    Some(Credential::Resumption {
                        value_base64: EncodedBytes::from_bytes(b"token".to_vec()),
                    }),
                ),
            ] {
                assert!(matches!(
                    prepare_user_operation(request, bound_user_authority()),
                    OperationPreparation::Rejected(_)
                ));
            }
        }
    }

    #[test]
    fn kv_mutations_reject_unbound_and_transplanted_metadata() {
        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Put,
                    "/protected/kv/key",
                    json_header(),
                    Some(br#""value""#),
                    None,
                ),
                AuthorityState::Anonymous,
            ),
            OperationPreparation::Rejected(_)
        ));

        let mut put_query = envelope(
            LogicalMethod::Put,
            "/protected/kv/key",
            json_header(),
            Some(br#""value""#),
            None,
        );
        put_query.request.query = Some("x=1".to_owned());
        for request in [
            envelope(
                LogicalMethod::Put,
                "/protected/kv/key",
                Vec::new(),
                Some(br#""value""#),
                None,
            ),
            envelope(
                LogicalMethod::Put,
                "/protected/kv/key",
                json_header(),
                None,
                None,
            ),
            put_query,
            envelope(
                LogicalMethod::Put,
                "/protected/kv/key",
                json_header(),
                Some(br#""value""#),
                Some(Credential::Resumption {
                    value_base64: EncodedBytes::from_bytes(b"token".to_vec()),
                }),
            ),
        ] {
            assert!(matches!(
                prepare_user_operation(request, bound_user_authority()),
                OperationPreparation::Rejected(_)
            ));
        }

        let mut delete_query = envelope(
            LogicalMethod::Delete,
            "/protected/kv/key",
            Vec::new(),
            None,
            None,
        );
        delete_query.request.query = Some("x=1".to_owned());
        for request in [
            envelope(
                LogicalMethod::Delete,
                "/protected/kv/key",
                json_header(),
                None,
                None,
            ),
            envelope(
                LogicalMethod::Delete,
                "/protected/kv/key",
                Vec::new(),
                Some(b""),
                None,
            ),
            delete_query,
            envelope(
                LogicalMethod::Delete,
                "/protected/kv",
                Vec::new(),
                None,
                Some(Credential::ApiKey {
                    value_base64: EncodedBytes::from_bytes(b"key".to_vec()),
                }),
            ),
        ] {
            assert!(matches!(
                prepare_user_operation(request, bound_user_authority()),
                OperationPreparation::Rejected(_)
            ));
        }

        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Delete,
                    "/protected/kv/%2f",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Rejected(_)
        ));
    }

    #[test]
    fn kv_reads_reject_unbound_and_transplanted_metadata() {
        for path in ["/protected/kv/key", "/protected/kv"] {
            assert!(matches!(
                prepare_user_operation(
                    envelope(LogicalMethod::Get, path, Vec::new(), None, None),
                    AuthorityState::Anonymous,
                ),
                OperationPreparation::Rejected(_)
            ));

            let mut query = envelope(LogicalMethod::Get, path, Vec::new(), None, None);
            query.request.query = Some("x=1".to_owned());
            for request in [
                envelope(LogicalMethod::Get, path, json_header(), None, None),
                envelope(LogicalMethod::Get, path, Vec::new(), Some(b""), None),
                query,
                envelope(
                    LogicalMethod::Get,
                    path,
                    Vec::new(),
                    None,
                    Some(Credential::ApiKey {
                        value_base64: EncodedBytes::from_bytes(b"key".to_vec()),
                    }),
                ),
            ] {
                assert!(matches!(
                    prepare_user_operation(request, bound_user_authority()),
                    OperationPreparation::Rejected(_)
                ));
            }
        }
    }

    #[test]
    fn logical_query_parsing_matches_axum_query_semantics() {
        let no_derivation = parse_logical_query::<DerivationPathQuery>(None)
            .expect("absent derivation query uses the existing default key options");
        assert!(no_derivation
            .key_options
            .seed_phrase_derivation_path
            .is_none());
        assert!(no_derivation
            .key_options
            .private_key_derivation_path
            .is_none());

        let parsed = parse_logical_query::<DerivationPathQuery>(Some(
            "private_key_derivation_path=m%2F44%27%2F0%27%2F0%27%2F0%2F0".to_owned(),
        ))
        .expect("valid encoded derivation path query");
        assert_eq!(
            parsed.key_options.private_key_derivation_path.as_deref(),
            Some("m/44'/0'/0'/0/0")
        );
        assert!(
            parse_logical_query::<PublicKeyQuery>(Some("algorithm=schnorr".to_owned())).is_ok()
        );
        assert!(parse_logical_query::<PublicKeyQuery>(None).is_err());
        assert!(
            parse_logical_query::<PublicKeyQuery>(Some("algorithm=unknown".to_owned())).is_err()
        );
    }

    #[test]
    fn oversized_logical_json_response_fails_before_gateway_encryption() {
        assert!(LogicalUnaryResponse::json_with_limit(StatusCode::OK, &"abc", 5).is_ok());
        assert!(matches!(
            LogicalUnaryResponse::json_with_limit(StatusCode::OK, &"abcd", 5),
            Err(ApiError::PayloadTooLarge)
        ));

        let escaped = "\0";
        assert!(LogicalUnaryResponse::json_with_limit(StatusCode::OK, &escaped, 8).is_ok());
        assert!(matches!(
            LogicalUnaryResponse::json_with_limit(StatusCode::OK, &escaped, 7),
            Err(ApiError::PayloadTooLarge)
        ));

        let mut buffer = BoundedJsonBuffer::new(7);
        assert!(serde_json::to_writer(&mut buffer, &escaped).is_err());
        assert!(buffer.exceeded());
        assert!(buffer.len() <= 7);
    }

    #[test]
    fn crypto_utility_response_bounds_are_exact() {
        let empty_response = EncryptDataResponse {
            encrypted_data: String::new(),
        };
        assert_eq!(
            serde_json::to_vec(&empty_response).unwrap().len(),
            ENCRYPTED_DATA_JSON_OVERHEAD_BYTES
        );

        let max_encrypted_bytes = MAX_ENCRYPTION_PLAINTEXT_BYTES + AES_GCM_NONCE_AND_TAG_BYTES;
        let encoded_bytes = max_encrypted_bytes.div_ceil(3) * 4;
        assert_eq!(encoded_bytes, MAX_ENCRYPTED_DATA_BASE64_BYTES);
        assert_eq!(
            encoded_bytes + ENCRYPTED_DATA_JSON_OVERHEAD_BYTES,
            EnvelopeLimits::DEFAULT.logical_body_bytes - 3
        );

        let next_encoded_bytes = (max_encrypted_bytes + 1).div_ceil(3) * 4;
        assert_eq!(
            next_encoded_bytes + ENCRYPTED_DATA_JSON_OVERHEAD_BYTES,
            EnvelopeLimits::DEFAULT.logical_body_bytes + 1
        );

        let response_envelope_overhead = serde_json::to_vec(&UnaryResponseEnvelope {
            version: Version2,
            request_id: RequestId::from_bytes([0x31; 16]),
            status: StatusCode::OK.as_u16(),
            headers: json_header(),
            body_base64: Some(EncodedBytes::from_bytes(Vec::new())),
        })
        .unwrap()
        .len();
        let logical_response_bytes = encoded_bytes + ENCRYPTED_DATA_JSON_OVERHEAD_BYTES;
        let response_record_plaintext_bytes =
            response_envelope_overhead + logical_response_bytes.div_ceil(3) * 4;
        let response_record_bytes = response_record_plaintext_bytes + 12 + 16;
        let encrypted_outer_response_bytes =
            "{\"encrypted\":\"\"}".len() + response_record_bytes.div_ceil(3) * 4;
        assert_eq!(encrypted_outer_response_bytes, 52_196_060);
        assert!(encrypted_outer_response_bytes <= 50 * 1024 * 1024);

        assert!(validate_encryption_plaintext_length(MAX_ENCRYPTION_PLAINTEXT_BYTES).is_ok());
        assert!(matches!(
            validate_encryption_plaintext_length(MAX_ENCRYPTION_PLAINTEXT_BYTES + 1),
            Err(ApiError::PayloadTooLarge)
        ));
        assert!(validate_encrypted_data_base64_length(MAX_ENCRYPTED_DATA_BASE64_BYTES).is_ok());
        assert!(matches!(
            validate_encrypted_data_base64_length(MAX_ENCRYPTED_DATA_BASE64_BYTES + 1),
            Err(ApiError::PayloadTooLarge)
        ));
    }

    fn api_key_credential(value: &[u8]) -> Credential {
        Credential::ApiKey {
            value_base64: EncodedBytes::from_bytes(value.to_vec()),
        }
    }

    fn api_key_binding_envelope(
        method: LogicalMethod,
        path: &str,
        headers: Vec<HeaderField>,
        body: Option<&[u8]>,
    ) -> RequestEnvelope {
        let mut request = envelope(
            method,
            path,
            headers,
            body,
            Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
        );
        request.cache_namespace_root_base64 = Some(CacheNamespaceRoot::from_bytes([0xa1; 32]));
        request
    }

    fn bound_api_key_authority() -> AuthorityState {
        let user_id = uuid::Uuid::from_bytes([0xa2; 16]);
        let cache_namespace =
            derive_tinfoil_cache_namespace(&CacheNamespaceRoot::from_bytes([0xa3; 32]), user_id);
        AuthorityState::Bound(BoundAuthority::api_key(17, user_id, cache_namespace))
    }

    fn bound_platform_authority() -> AuthorityState {
        AuthorityState::Bound(BoundAuthority::platform(
            uuid::Uuid::from_bytes([0xa4; 16]),
            Instant::now() + std::time::Duration::from_secs(60),
        ))
    }

    #[test]
    fn complete_platform_control_matrix_is_exactly_session_bound() {
        let org = "123e4567-e89b-12d3-a456-426614174000";
        let project = "223e4567-e89b-12d3-a456-426614174000";
        let user = "323e4567-e89b-12d3-a456-426614174000";
        let invite = "423e4567-e89b-12d3-a456-426614174000";
        let cases = [
            (LogicalMethod::Get, "/platform/me".to_owned(), false, true),
            (
                LogicalMethod::Post,
                "/platform/orgs".to_owned(),
                true,
                false,
            ),
            (LogicalMethod::Get, "/platform/orgs".to_owned(), false, true),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}"),
                false,
                false,
            ),
            (
                LogicalMethod::Post,
                format!("/platform/orgs/{org}/projects"),
                true,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects"),
                false,
                true,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects/{project}"),
                false,
                true,
            ),
            (
                LogicalMethod::Patch,
                format!("/platform/orgs/{org}/projects/{project}"),
                true,
                true,
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/projects/{project}"),
                false,
                false,
            ),
            (
                LogicalMethod::Post,
                format!("/platform/orgs/{org}/projects/{project}/secrets"),
                true,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects/{project}/secrets"),
                false,
                true,
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/projects/{project}/secrets/API_KEY"),
                false,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects/{project}/settings/email"),
                false,
                true,
            ),
            (
                LogicalMethod::Put,
                format!("/platform/orgs/{org}/projects/{project}/settings/email"),
                true,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects/{project}/settings/oauth"),
                false,
                true,
            ),
            (
                LogicalMethod::Put,
                format!("/platform/orgs/{org}/projects/{project}/settings/oauth"),
                true,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/memberships"),
                false,
                true,
            ),
            (
                LogicalMethod::Patch,
                format!("/platform/orgs/{org}/memberships/{user}"),
                true,
                true,
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/memberships/{user}"),
                false,
                false,
            ),
            (
                LogicalMethod::Post,
                format!("/platform/orgs/{org}/invites"),
                true,
                false,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/invites"),
                false,
                true,
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/invites/{invite}"),
                false,
                true,
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/invites/{invite}"),
                false,
                false,
            ),
            (
                LogicalMethod::Post,
                format!("/platform/accept_invite/{invite}"),
                false,
                false,
            ),
        ];

        assert_eq!(cases.len(), 24);
        for (method, path, has_body, requires_stored_output) in cases {
            let preparation = prepare_user_operation(
                envelope(
                    method,
                    &path,
                    has_body.then(json_header).unwrap_or_default(),
                    has_body.then_some(b"{}".as_slice()),
                    None,
                ),
                bound_platform_authority(),
            );
            let OperationPreparation::Ready(operation) = preparation else {
                panic!("platform control route was not admitted: {method:?} {path}");
            };
            assert!(matches!(&operation, UserOperation::PlatformControl { .. }));
            assert_eq!(
                operation.requires_stored_output_reservation(),
                requires_stored_output,
                "{method:?} {path}"
            );
        }
    }

    #[test]
    fn platform_control_rejects_authority_and_request_transplants() {
        let exact = || envelope(LogicalMethod::Get, "/platform/me", Vec::new(), None, None);
        for authority in [
            AuthorityState::Anonymous,
            bound_user_authority(),
            bound_api_key_authority(),
        ] {
            assert_eq!(
                rejected_status(prepare_user_operation(exact(), authority)),
                StatusCode::UNAUTHORIZED
            );
        }

        let mut with_query = exact();
        with_query.request.query = Some("transplanted=true".to_owned());
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_query,
                bound_platform_authority()
            )),
            StatusCode::BAD_REQUEST
        );

        let mut with_body = exact();
        with_body.request.body_base64 = Some(EncodedBytes::from_bytes(b"{}".to_vec()));
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_body,
                bound_platform_authority()
            )),
            StatusCode::BAD_REQUEST
        );

        let mut with_header = exact();
        with_header.request.headers = json_header();
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_header,
                bound_platform_authority()
            )),
            StatusCode::BAD_REQUEST
        );

        let mut with_credential = exact();
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"transplanted".to_vec()),
        });
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_credential,
                bound_platform_authority()
            )),
            StatusCode::BAD_REQUEST
        );
    }

    fn inference_kind(operation: &InferenceOperation) -> &'static str {
        match operation {
            InferenceOperation::Models => "models",
            InferenceOperation::ModelCatalog => "catalog",
            InferenceOperation::Chat { .. } => "chat",
            InferenceOperation::TextToSpeech { .. } => "speech",
            InferenceOperation::Transcription { .. } => "transcription",
            InferenceOperation::Embeddings { .. } => "embeddings",
            InferenceOperation::WebSearch { .. } => "web-search",
            InferenceOperation::WebExtract { .. } => "web-extract",
        }
    }

    fn rejected_status(preparation: OperationPreparation) -> StatusCode {
        match preparation {
            OperationPreparation::Rejected(response) => response.status,
            OperationPreparation::Unsupported => panic!("expected a classified rejection"),
            OperationPreparation::Ready(_) => panic!("expected the request to be rejected"),
        }
    }

    fn oauth_envelope(path: &str, body: &[u8], with_cache_root: bool) -> RequestEnvelope {
        let mut request = envelope(LogicalMethod::Post, path, json_header(), Some(body), None);
        if with_cache_root {
            request.cache_namespace_root_base64 = Some(CacheNamespaceRoot::from_bytes([0xa5; 32]));
        }
        request
    }

    #[test]
    fn exact_seven_route_oauth_matrix_is_session_bound_at_callback() {
        let initiate_body =
            br#"{"client_id":"00000000-0000-0000-0000-000000000000","invite_code":"ignored"}"#;
        for (path, expected_provider) in [
            ("/auth/github", OAuthProviderName::Github),
            ("/auth/google", OAuthProviderName::Google),
            ("/auth/apple", OAuthProviderName::Apple),
        ] {
            let OperationPreparation::Ready(operation) = prepare_user_operation(
                oauth_envelope(path, initiate_body, false),
                AuthorityState::Anonymous,
            ) else {
                panic!("OAuth initiation route {path} must be admitted");
            };
            assert!(matches!(
                &operation,
                UserOperation::OAuthInitiate { provider, .. }
                    if *provider == expected_provider
            ));
            assert!(!operation.requires_authentication_transition());
            assert!(!operation.requires_provider_output_reservation());
        }

        let callback_body = br#"{"code":"code","state":"state"}"#;
        for (path, expected_provider) in [
            ("/auth/github/callback", OAuthProviderName::Github),
            ("/auth/google/callback", OAuthProviderName::Google),
            ("/auth/apple/callback", OAuthProviderName::Apple),
        ] {
            let OperationPreparation::Ready(operation) = prepare_user_operation(
                oauth_envelope(path, callback_body, true),
                AuthorityState::Anonymous,
            ) else {
                panic!("OAuth callback route {path} must be admitted");
            };
            assert!(matches!(
                &operation,
                UserOperation::OAuthCallback { provider, .. }
                    if *provider == expected_provider
            ));
            assert!(operation.requires_authentication_transition());
            assert!(operation.requires_provider_output_reservation());
        }

        let native_body =
            br#"{"identity_token":"token","client_id":"00000000-0000-0000-0000-000000000000"}"#;
        let OperationPreparation::Ready(operation) = prepare_user_operation(
            oauth_envelope("/auth/apple/native", native_body, true),
            AuthorityState::Anonymous,
        ) else {
            panic!("Apple native OAuth route must be admitted");
        };
        assert!(matches!(&operation, UserOperation::AppleNativeOAuth { .. }));
        assert!(operation.requires_authentication_transition());
        assert!(operation.requires_provider_output_reservation());

        assert!(serde_json::from_slice::<OAuthAuthRequest>(initiate_body).is_ok());
    }

    #[test]
    fn oauth_projection_rejects_transplants_and_noncanonical_shapes() {
        let callback_body = br#"{"code":"code","state":"state"}"#;
        let initiation_body = br#"{"client_id":"00000000-0000-0000-0000-000000000000"}"#;

        assert_eq!(
            rejected_status(prepare_user_operation(
                oauth_envelope("/auth/github/callback", callback_body, false),
                AuthorityState::Anonymous,
            )),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            rejected_status(prepare_user_operation(
                oauth_envelope("/auth/github", initiation_body, true),
                AuthorityState::Anonymous,
            )),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            rejected_status(prepare_user_operation(
                oauth_envelope("/auth/github", initiation_body, false),
                bound_user_authority(),
            )),
            StatusCode::CONFLICT
        );
        assert_eq!(
            rejected_status(prepare_user_operation(
                oauth_envelope("/auth/github/callback", callback_body, true),
                bound_user_authority(),
            )),
            StatusCode::CONFLICT
        );

        let mut with_query = oauth_envelope("/auth/google/callback", callback_body, true);
        with_query.request.query = Some("next=elsewhere".to_owned());
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_query,
                AuthorityState::Anonymous
            )),
            StatusCode::BAD_REQUEST
        );

        let mut with_credential = oauth_envelope("/auth/apple/native", callback_body, true);
        with_credential.credential = Some(Credential::Resumption {
            value_base64: EncodedBytes::from_bytes(b"copied".to_vec()),
        });
        assert_eq!(
            rejected_status(prepare_user_operation(
                with_credential,
                AuthorityState::Anonymous
            )),
            StatusCode::BAD_REQUEST
        );

        let mut streaming = oauth_envelope("/auth/apple/callback", callback_body, true);
        streaming.response_mode = ResponseMode::Stream;
        assert_eq!(
            rejected_status(prepare_user_operation(streaming, AuthorityState::Anonymous)),
            StatusCode::BAD_REQUEST
        );

        let trailing_slash = oauth_envelope("/auth/github/", initiation_body, false);
        assert!(matches!(
            prepare_user_operation(trailing_slash, AuthorityState::Anonymous),
            OperationPreparation::Unsupported
        ));
        let wrong_method = envelope(LogicalMethod::Get, "/auth/github", Vec::new(), None, None);
        assert!(matches!(
            prepare_user_operation(wrong_method, AuthorityState::Anonymous),
            OperationPreparation::Unsupported
        ));
    }

    #[test]
    fn oauth_callback_controlled_fields_are_bounded_before_provider_work() {
        let valid_callback = OAuthCallbackRequest {
            code: "code".to_owned(),
            state: "state".to_owned(),
        };
        assert!(validate_oauth_callback_request(&valid_callback).is_ok());
        assert!(matches!(
            validate_oauth_callback_request(&OAuthCallbackRequest {
                code: String::new(),
                state: "state".to_owned(),
            }),
            Err(ApiError::BadRequest)
        ));
        assert!(matches!(
            validate_oauth_callback_request(&OAuthCallbackRequest {
                code: "code".to_owned(),
                state: "s".repeat(MAX_OAUTH_STATE_BYTES + 1),
            }),
            Err(ApiError::PayloadTooLarge)
        ));

        let valid_native = AppleNativeSignInRequest {
            identity_token: "token".to_owned(),
            user_identifier: None,
            email: None,
            given_name: None,
            family_name: None,
            client_id: uuid::Uuid::nil(),
            nonce: None,
        };
        assert!(validate_apple_native_request(&valid_native).is_ok());
        assert!(matches!(
            validate_apple_native_request(&AppleNativeSignInRequest {
                identity_token: String::new(),
                ..valid_native.clone()
            }),
            Err(ApiError::BadRequest)
        ));
        assert!(matches!(
            validate_apple_native_request(&AppleNativeSignInRequest {
                identity_token: "t".repeat(MAX_APPLE_IDENTITY_TOKEN_BYTES + 1),
                ..valid_native.clone()
            }),
            Err(ApiError::PayloadTooLarge)
        ));
        assert!(matches!(
            validate_apple_native_request(&AppleNativeSignInRequest {
                nonce: Some("n".repeat(MAX_APPLE_OPTIONAL_FIELD_BYTES + 1)),
                ..valid_native
            }),
            Err(ApiError::PayloadTooLarge)
        ));
    }

    #[test]
    fn exact_eight_route_unary_inference_matrix_is_classified_and_reserved() {
        let cases = [
            (
                envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None),
                AuthorityState::Anonymous,
                "models",
                true,
            ),
            (
                envelope(
                    LogicalMethod::Get,
                    "/v1/models/catalog",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
                "catalog",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/chat/completions",
                    json_header(),
                    Some(br#"{"model":"model","messages":[]}"#),
                    None,
                ),
                bound_user_authority(),
                "chat",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/audio/speech",
                    json_header(),
                    Some(br#"{"model":"model","input":"hello","voice":"alloy"}"#),
                    None,
                ),
                bound_user_authority(),
                "speech",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/audio/transcriptions",
                    json_header(),
                    Some(br#"{"model":"model","file":"AA=="}"#),
                    None,
                ),
                bound_user_authority(),
                "transcription",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/embeddings",
                    json_header(),
                    Some(br#"{"model":"model","input":"hello"}"#),
                    None,
                ),
                bound_user_authority(),
                "embeddings",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/web/search",
                    json_header(),
                    Some(br#"{"query":"maple"}"#),
                    None,
                ),
                bound_user_authority(),
                "web-search",
                false,
            ),
            (
                envelope(
                    LogicalMethod::Post,
                    "/v1/web/extract",
                    json_header(),
                    Some(br#"{"urls":["https://example.com/"]}"#),
                    None,
                ),
                bound_user_authority(),
                "web-extract",
                false,
            ),
        ];

        for (mut request, authority, expected_kind, expects_public_authority) in cases {
            request.request.query = Some("preview=1".to_owned());
            request.request.headers.push(HeaderField {
                name: "x-provider-beta".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"preview".to_vec()),
            });
            let OperationPreparation::Ready(operation) = prepare_user_operation(request, authority)
            else {
                panic!("{expected_kind} must be admitted by the exact unary matrix");
            };
            assert!(
                operation.requires_provider_output_reservation(),
                "{expected_kind} must reserve bounded provider-output capacity"
            );
            let UserOperation::Inference {
                authority,
                operation,
            } = operation
            else {
                panic!("{expected_kind} must classify as inference");
            };
            assert_eq!(inference_kind(&operation), expected_kind);
            assert_eq!(
                matches!(authority, InferenceAuthority::Public),
                expects_public_authority,
                "only the public models route may use public authority"
            );
            if !expects_public_authority {
                assert!(matches!(authority, InferenceAuthority::User(_)));
            }
        }
    }

    #[test]
    fn exact_six_route_api_key_first_binding_matrix_requires_transition() {
        let cases = [
            (
                api_key_binding_envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None),
                "models",
            ),
            (
                api_key_binding_envelope(
                    LogicalMethod::Get,
                    "/v1/models/catalog",
                    Vec::new(),
                    None,
                ),
                "catalog",
            ),
            (
                api_key_binding_envelope(
                    LogicalMethod::Post,
                    "/v1/chat/completions",
                    json_header(),
                    Some(br#"{"model":"model","messages":[]}"#),
                ),
                "chat",
            ),
            (
                api_key_binding_envelope(
                    LogicalMethod::Post,
                    "/v1/audio/speech",
                    json_header(),
                    Some(br#"{"model":"model","input":"hello","voice":"alloy"}"#),
                ),
                "speech",
            ),
            (
                api_key_binding_envelope(
                    LogicalMethod::Post,
                    "/v1/audio/transcriptions",
                    json_header(),
                    Some(br#"{"model":"model","file":"AA=="}"#),
                ),
                "transcription",
            ),
            (
                api_key_binding_envelope(
                    LogicalMethod::Post,
                    "/v1/embeddings",
                    json_header(),
                    Some(br#"{"model":"model","input":"hello"}"#),
                ),
                "embeddings",
            ),
        ];

        for (mut request, expected_kind) in cases {
            request.request.query = Some("provider=tinfoil".to_owned());
            request.request.headers.push(HeaderField {
                name: "x-provider-beta".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"preview".to_vec()),
            });
            let OperationPreparation::Ready(operation) =
                prepare_user_operation(request, AuthorityState::Anonymous)
            else {
                panic!("{expected_kind} must admit an encrypted first-use API key");
            };
            assert!(operation.requires_authentication_transition());
            assert!(operation.requires_provider_output_reservation());
            let UserOperation::Inference {
                authority:
                    InferenceAuthority::AuthenticateApiKey {
                        credential,
                        cache_namespace_root: _,
                    },
                operation,
            } = operation
            else {
                panic!("{expected_kind} must classify as an API-key binding operation");
            };
            assert_eq!(
                credential.as_slice(),
                b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
            );
            assert_eq!(inference_kind(&operation), expected_kind);
        }
    }

    #[test]
    fn public_models_is_anonymous_only_when_no_explicit_authority_is_supplied() {
        let OperationPreparation::Ready(UserOperation::Inference {
            authority: InferenceAuthority::Public,
            operation: InferenceOperation::Models,
        }) = prepare_user_operation(
            envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None),
            AuthorityState::Anonymous,
        )
        else {
            panic!("models without credentials must remain public");
        };

        let OperationPreparation::Ready(UserOperation::Inference {
            authority: InferenceAuthority::User(_),
            operation: InferenceOperation::Models,
        }) = prepare_user_operation(
            envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None),
            bound_user_authority(),
        )
        else {
            panic!("a user-bound session may list public models");
        };

        let OperationPreparation::Ready(UserOperation::Inference {
            authority: InferenceAuthority::ApiKey(_),
            operation: InferenceOperation::Models,
        }) = prepare_user_operation(
            envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None),
            bound_api_key_authority(),
        )
        else {
            panic!("an API-key-bound session may list public models");
        };

        assert_eq!(
            rejected_status(prepare_user_operation(
                envelope(
                    LogicalMethod::Get,
                    "/v1/models",
                    Vec::new(),
                    None,
                    Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
                ),
                AuthorityState::Anonymous,
            )),
            StatusCode::BAD_REQUEST,
            "an explicit first-use API key requires its cache root"
        );

        let mut root_without_credential =
            envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None);
        root_without_credential.cache_namespace_root_base64 =
            Some(CacheNamespaceRoot::from_bytes([0xa5; 32]));
        assert_eq!(
            rejected_status(prepare_user_operation(
                root_without_credential,
                AuthorityState::Anonymous,
            )),
            StatusCode::UNAUTHORIZED,
            "a cache root alone cannot select an authority"
        );

        assert_eq!(
            rejected_status(prepare_user_operation(
                envelope(
                    LogicalMethod::Get,
                    "/v1/models",
                    Vec::new(),
                    None,
                    Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",)),
                ),
                bound_user_authority(),
            )),
            StatusCode::CONFLICT,
            "a bound session cannot be rebound with an API key"
        );

        assert_eq!(
            rejected_status(prepare_user_operation(
                envelope(LogicalMethod::Get, "/v1/models", Vec::new(), None, None),
                bound_platform_authority(),
            )),
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn web_routes_are_user_only_and_reject_authentication_transplants() {
        for path in ["/v1/web/search", "/v1/web/extract"] {
            for authority in [
                AuthorityState::Anonymous,
                bound_api_key_authority(),
                bound_platform_authority(),
            ] {
                assert_eq!(
                    rejected_status(prepare_user_operation(
                        envelope(
                            LogicalMethod::Post,
                            path,
                            json_header(),
                            Some(br#"{"query":"maple"}"#),
                            None,
                        ),
                        authority,
                    )),
                    StatusCode::UNAUTHORIZED
                );
            }

            let with_credential = envelope(
                LogicalMethod::Post,
                path,
                json_header(),
                Some(br#"{"query":"maple"}"#),
                Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
            );
            assert_eq!(
                rejected_status(prepare_user_operation(
                    with_credential,
                    AuthorityState::Anonymous,
                )),
                StatusCode::BAD_REQUEST
            );

            let mut with_root = envelope(
                LogicalMethod::Post,
                path,
                json_header(),
                Some(br#"{"query":"maple"}"#),
                None,
            );
            with_root.cache_namespace_root_base64 =
                Some(CacheNamespaceRoot::from_bytes([0xa6; 32]));
            assert_eq!(
                rejected_status(prepare_user_operation(with_root, bound_user_authority())),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn cache_namespace_root_is_required_only_for_authority_binding() {
        let mut login = envelope(
            LogicalMethod::Post,
            "/login",
            json_header(),
            Some(b"{}"),
            None,
        );
        login.cache_namespace_root_base64 = None;
        let mut register = envelope(
            LogicalMethod::Post,
            "/register",
            json_header(),
            Some(b"{}"),
            None,
        );
        register.cache_namespace_root_base64 = None;
        let mut resume = envelope(
            LogicalMethod::Post,
            "/refresh",
            Vec::new(),
            None,
            Some(Credential::Resumption {
                value_base64: EncodedBytes::from_bytes(b"resumption".to_vec()),
            }),
        );
        resume.cache_namespace_root_base64 = None;
        let api_key_without_root = envelope(
            LogicalMethod::Get,
            "/v1/models/catalog",
            Vec::new(),
            None,
            Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
        );

        for request in [login, register, resume, api_key_without_root] {
            assert_eq!(
                rejected_status(prepare_user_operation(request, AuthorityState::Anonymous)),
                StatusCode::BAD_REQUEST
            );
        }

        let mut protected_transplant = envelope(
            LogicalMethod::Get,
            "/protected/user",
            Vec::new(),
            None,
            None,
        );
        protected_transplant.cache_namespace_root_base64 =
            Some(CacheNamespaceRoot::from_bytes([0xa7; 32]));
        assert_eq!(
            rejected_status(prepare_user_operation(
                protected_transplant,
                bound_user_authority(),
            )),
            StatusCode::BAD_REQUEST
        );

        for authority in [bound_user_authority(), bound_api_key_authority()] {
            let follow_up = envelope(
                LogicalMethod::Get,
                "/v1/models/catalog",
                Vec::new(),
                None,
                None,
            );
            assert!(matches!(
                prepare_user_operation(follow_up, authority),
                OperationPreparation::Ready(UserOperation::Inference { .. })
            ));
        }

        let mut user_root_transplant = envelope(
            LogicalMethod::Get,
            "/v1/models/catalog",
            Vec::new(),
            None,
            None,
        );
        user_root_transplant.cache_namespace_root_base64 =
            Some(CacheNamespaceRoot::from_bytes([0xa8; 32]));
        let mut api_key_root_transplant = envelope(
            LogicalMethod::Get,
            "/v1/models/catalog",
            Vec::new(),
            None,
            None,
        );
        api_key_root_transplant.cache_namespace_root_base64 =
            Some(CacheNamespaceRoot::from_bytes([0xa9; 32]));

        for (authority, request) in [
            (
                bound_user_authority(),
                envelope(
                    LogicalMethod::Get,
                    "/v1/models/catalog",
                    Vec::new(),
                    None,
                    Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
                ),
            ),
            (bound_user_authority(), user_root_transplant),
            (
                bound_api_key_authority(),
                envelope(
                    LogicalMethod::Get,
                    "/v1/models/catalog",
                    Vec::new(),
                    None,
                    Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")),
                ),
            ),
            (bound_api_key_authority(), api_key_root_transplant),
        ] {
            assert_eq!(
                rejected_status(prepare_user_operation(request, authority)),
                StatusCode::CONFLICT
            );
        }
    }

    #[test]
    fn chat_classifier_requires_mode_to_match_the_top_level_stream_flag() {
        let mut streaming = envelope(
            LogicalMethod::Post,
            "/v1/chat/completions",
            json_header(),
            Some(br#"{"model":"model","messages":[],"stream":true}"#),
            None,
        );
        streaming.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(streaming, bound_user_authority()),
            OperationPreparation::Ready(UserOperation::Inference {
                operation: InferenceOperation::Chat { stream: true, .. },
                ..
            })
        ));

        let mut mismatched_stream = envelope(
            LogicalMethod::Post,
            "/v1/chat/completions",
            json_header(),
            Some(br#"{"stream":true}"#),
            None,
        );
        mismatched_stream.response_mode = ResponseMode::Unary;
        assert!(matches!(
            prepare_user_operation(mismatched_stream, bound_user_authority()),
            OperationPreparation::Rejected(_)
        ));

        let mut mismatched_unary = envelope(
            LogicalMethod::Post,
            "/v1/chat/completions",
            json_header(),
            Some(br#"{"stream":false}"#),
            None,
        );
        mismatched_unary.response_mode = ResponseMode::Stream;
        assert!(matches!(
            prepare_user_operation(mismatched_unary, bound_user_authority()),
            OperationPreparation::Rejected(_)
        ));

        for invalid in [
            br#"{"stream":"yes"}"#.as_slice(),
            br#"{"stream":true,"stream":true}"#.as_slice(),
            br#"[]"#.as_slice(),
        ] {
            let mut request = envelope(
                LogicalMethod::Post,
                "/v1/chat/completions",
                json_header(),
                Some(invalid),
                None,
            );
            request.response_mode = ResponseMode::Stream;
            assert!(matches!(
                prepare_user_operation(request, bound_user_authority()),
                OperationPreparation::Rejected(_)
            ));
        }

        let mut automatic = envelope(
            LogicalMethod::Post,
            "/v1/chat/completions",
            json_header(),
            Some(br#"{"stream":true}"#),
            None,
        );
        automatic.response_mode = ResponseMode::Auto;
        assert!(matches!(
            prepare_user_operation(automatic, bound_user_authority()),
            OperationPreparation::Rejected(_)
        ));
    }

    #[test]
    fn chat_classifier_rejects_missing_body_and_unsafe_headers() {
        for body in [None, Some(&b""[..])] {
            assert!(matches!(
                prepare_user_operation(
                    envelope(
                        LogicalMethod::Post,
                        "/v1/chat/completions",
                        json_header(),
                        body,
                        None,
                    ),
                    bound_user_authority(),
                ),
                OperationPreparation::Rejected(_)
            ));
        }

        for forbidden in [
            "authorization",
            "proxy-authorization",
            "proxy-authenticate",
            "cookie",
            "set-cookie",
            "host",
            "content-length",
            "content-encoding",
            "content-md5",
            "digest",
            "accept-encoding",
            "connection",
            "keep-alive",
            "proxy-connection",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "x-session-id",
            "x-api-key",
            "api-key",
            "x-openai-api-key",
            "x-tinfoil-api-key",
            "x-goog-api-key",
            "x-anthropic-api-key",
            "openai-organization",
            "openai-project",
        ] {
            let mut headers = json_header();
            headers.push(HeaderField {
                name: forbidden.to_owned(),
                value_base64: EncodedBytes::from_bytes(b"opaque".to_vec()),
            });
            assert!(
                prepare_inference_headers(headers, true).is_err(),
                "{forbidden} must never cross the provider boundary"
            );
        }

        let duplicate_content_type = vec![json_header()[0].clone(), json_header()[0].clone()];
        assert!(prepare_inference_headers(duplicate_content_type, true).is_err());
        assert!(prepare_inference_headers(
            vec![HeaderField {
                name: "invalid header name".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"opaque".to_vec()),
            }],
            true,
        )
        .is_err());
        assert!(prepare_inference_headers(Vec::new(), true).is_err());
        assert!(prepare_inference_headers(Vec::new(), false).is_ok());
        assert!(prepare_inference_headers(
            vec![HeaderField {
                name: "invalid header name".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"opaque".to_vec()),
            }],
            false
        )
        .is_err());

        let safe_headers = vec![
            json_header()[0].clone(),
            HeaderField {
                name: "x-client-hint".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"first".to_vec()),
            },
            HeaderField {
                name: "x-client-hint".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"second".to_vec()),
            },
        ];
        let prepared = prepare_inference_headers(safe_headers, true)
            .expect("safe opaque headers are retained");
        let values = prepared
            .get_all("x-client-hint")
            .iter()
            .map(|value| value.as_bytes())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![&b"first"[..], &b"second"[..]]);
        assert!(prepared
            .get_all("x-client-hint")
            .iter()
            .all(HeaderValue::is_sensitive));
    }

    #[test]
    fn responses_create_requires_bound_user_and_explicit_stream_mode() {
        let request = || {
            let mut request = envelope(
                LogicalMethod::Post,
                "/v1/responses",
                json_header(),
                Some(br#"{"model":"model","input":"hello","conversation":"123e4567-e89b-12d3-a456-426614174000","stream":false}"#),
                None,
            );
            request.response_mode = ResponseMode::Stream;
            request
        };

        let OperationPreparation::Ready(operation) =
            prepare_user_operation(request(), bound_user_authority())
        else {
            panic!("valid Responses create stream must be admitted");
        };
        assert!(matches!(&operation, UserOperation::Responses { .. }));
        assert!(operation.is_streaming());

        let mut unary = request();
        unary.response_mode = ResponseMode::Unary;
        assert!(matches!(
            prepare_user_operation(unary, bound_user_authority()),
            OperationPreparation::Rejected(_)
        ));
        assert!(matches!(
            prepare_user_operation(request(), AuthorityState::Anonymous),
            OperationPreparation::Rejected(response)
                if response.status == StatusCode::UNAUTHORIZED
        ));

        let mut with_query = request();
        with_query.request.query = Some("admin=true".to_owned());
        with_query.request.headers.push(HeaderField {
            name: "x-provider-beta".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"preview".to_vec()),
        });
        let OperationPreparation::Ready(UserOperation::Responses { headers, .. }) =
            prepare_user_operation(with_query, bound_user_authority())
        else {
            panic!("Responses query and safe extension headers must preserve the raw contract");
        };
        assert_eq!(
            headers.get("x-provider-beta").map(HeaderValue::as_bytes),
            Some(&b"preview"[..])
        );

        let mut with_credential = request();
        with_credential.credential =
            Some(api_key_credential(b"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"));
        assert!(matches!(
            prepare_user_operation(with_credential, bound_user_authority()),
            OperationPreparation::Rejected(_)
        ));
    }

    #[test]
    fn canonical_api_key_hash_matches_v1_uuid_canonicalization() {
        let parsed = uuid::Uuid::parse_str("AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE").unwrap();
        let expected = hex::encode(Sha256::digest(parsed.to_string().as_bytes()));

        for representation in [
            parsed.to_string(),
            parsed.to_string().to_ascii_uppercase(),
            parsed.simple().to_string(),
        ] {
            let hash = canonical_api_key_hash(Zeroizing::new(representation.into_bytes()))
                .expect("all UUID spellings accepted by v1 canonicalize identically");
            assert_eq!(&*hash, &expected);
        }

        assert!(matches!(
            canonical_api_key_hash(Zeroizing::new(vec![0xff])),
            Err(ApiError::Unauthorized)
        ));
        assert!(matches!(
            canonical_api_key_hash(Zeroizing::new(b"not-a-uuid".to_vec())),
            Err(ApiError::Unauthorized)
        ));
    }
}
