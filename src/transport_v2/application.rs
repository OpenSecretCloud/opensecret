//! Explicit application projection for the first transport-v2 user slice.
//!
//! This module does not re-enter the transport-v1 router. It validates a small
//! exact operation allowlist, calls shared transport-neutral application
//! functions, and returns plaintext logical results for the gateway to encrypt
//! through the request's original session lease.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::Query;
use axum::http::StatusCode;
use axum::http::Uri;
use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::Serialize;
use zeroize::{Zeroize, Zeroizing};

use crate::bounded_json::BoundedJsonBuffer;
use crate::jwt::{
    issue_transport_v2_user_tokens, validate_transport_v2_user_resumption, AuthContext,
};
use crate::kv::StoreError;
use crate::web::login_routes::{
    authenticate_login, register_and_authenticate, verify_email_data, AuthResponse, Credentials,
    RefreshResponse, RegisterCredentials,
};
use crate::web::protected_routes::{
    confirm_account_deletion_data, create_api_key_data, decrypt_data_value, delete_all_kv_values,
    delete_api_key_by_name, delete_kv_value, encrypt_data_value, initiate_account_deletion_data,
    list_bounded_api_keys_data, private_key_bytes_data, private_key_data, protected_user_data,
    public_key_data, put_kv_value, request_new_verification_code_data, sign_message_data,
    third_party_token_data, ConfirmAccountDeletionRequest, CreateApiKeyRequest, DecryptDataRequest,
    DerivationPathQuery, EncryptDataRequest, InitiateAccountDeletionRequest, KvValue,
    PublicKeyQuery, SignMessageRequest, ThirdPartyTokenRequest,
};
use crate::{ApiError, AppState, VerifiedUserAuthentication};

use super::envelope::{
    decode_canonical_api_key_name_path, decode_canonical_kv_item_path,
    decode_canonical_verify_email_path, Credential, EncodedBytes, EnvelopeLimits, HeaderField,
    LogicalMethod, RequestEnvelope, RequestId, ResponseMode,
};
use super::session::{
    AuthenticationReservation, AuthenticationStartError, AuthorityState, BoundAuthority,
    BoundPrincipal,
};
use super::session_cache::V2SessionLease;

const JSON_CONTENT_TYPE: &[u8] = b"application/json";
const AES_GCM_NONCE_AND_TAG_BYTES: usize = 12 + 16;
const ENCRYPTED_DATA_JSON_OVERHEAD_BYTES: usize = "{\"encrypted_data\":\"".len() + "\"}".len();
const MAX_ENCRYPTED_DATA_BASE64_BYTES: usize =
    ((EnvelopeLimits::DEFAULT.logical_body_bytes - ENCRYPTED_DATA_JSON_OVERHEAD_BYTES) / 4) * 4;
const MAX_ENCRYPTION_PLAINTEXT_BYTES: usize =
    (MAX_ENCRYPTED_DATA_BASE64_BYTES / 4) * 3 - AES_GCM_NONCE_AND_TAG_BYTES;
const MAX_V2_KV_LIST_ROWS: usize = 65_536;
const MAX_V2_API_KEY_LIST_ROWS: usize = 65_536;

type SensitiveBytes = Zeroizing<Vec<u8>>;

pub(crate) enum OperationPreparation {
    Unsupported,
    Rejected(LogicalUnaryResponse),
    Ready(UserOperation),
}

pub(crate) enum UserOperation {
    Login {
        body: SensitiveBytes,
    },
    Register {
        body: SensitiveBytes,
    },
    Resume {
        credential: SensitiveBytes,
    },
    VerifyEmail {
        code: uuid::Uuid,
    },
    Protected {
        authority: BoundUserAuthority,
        operation: ProtectedUserOperation,
    },
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
}

impl UserOperation {
    pub(crate) const fn requires_authentication_transition(&self) -> bool {
        matches!(
            self,
            Self::Login { .. } | Self::Register { .. } | Self::Resume { .. }
        )
    }

    pub(crate) const fn requires_stored_output_reservation(&self) -> bool {
        matches!(
            self,
            Self::Protected {
                operation: ProtectedUserOperation::GetKv { .. }
                    | ProtectedUserOperation::ListKv
                    | ProtectedUserOperation::ListApiKeys,
                ..
            }
        )
    }
}

pub(crate) struct LogicalUnaryResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: Vec<HeaderField>,
    pub(crate) body: Option<Zeroizing<Vec<u8>>>,
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

    pub(crate) fn api_error(error: &ApiError) -> Self {
        match Self::json(error.status_code(), &error.response_body()) {
            Ok(response) => response,
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

pub(crate) struct ApplicationOutcome {
    pub(crate) response: LogicalUnaryResponse,
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
            response,
            session_effect,
        }
    }

    fn error(error: ApiError) -> Self {
        Self {
            response: LogicalUnaryResponse::api_error(&error),
            session_effect: SessionEffect::Retain,
        }
    }

    fn closing_error(error: ApiError) -> Self {
        Self {
            response: LogicalUnaryResponse::api_error(&error),
            session_effect: SessionEffect::Close,
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
        mut request,
        ..
    } = envelope;

    #[derive(Clone, Copy)]
    enum Route {
        Login,
        Register,
        Resume,
        VerifyEmail,
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
        RequestAccountDeletion,
        ConfirmAccountDeletion,
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
    let route = match (request.method, request.path.as_str()) {
        (LogicalMethod::Post, "/login") => Some(Route::Login),
        (LogicalMethod::Post, "/register") => Some(Route::Register),
        (LogicalMethod::Post, "/refresh") => Some(Route::Resume),
        (LogicalMethod::Get, _) if verification_code.is_some() => Some(Route::VerifyEmail),
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
        (LogicalMethod::Post, "/protected/delete-account/request") => {
            Some(Route::RequestAccountDeletion)
        }
        (LogicalMethod::Post, "/protected/delete-account/confirm") => {
            Some(Route::ConfirmAccountDeletion)
        }
        (LogicalMethod::Get, _) if kv_key.is_some() => Some(Route::GetKv),
        (LogicalMethod::Put, _) if kv_key.is_some() => Some(Route::PutKv),
        (LogicalMethod::Delete, _) if kv_key.is_some() => Some(Route::DeleteKv),
        (LogicalMethod::Delete, _) if api_key_name.is_some() => Some(Route::DeleteApiKey),
        _ => None,
    };
    if kv_key.is_some() || api_key_name.is_some() || verification_code.is_some() {
        request.path.zeroize();
    }
    let Some(route) = route else {
        return OperationPreparation::Unsupported;
    };

    if response_mode != ResponseMode::Unary {
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

            let body = request
                .body_base64
                .expect("validated body presence")
                .into_bytes();
            let body = Zeroizing::new(body);
            if matches!(route, Route::Login) {
                OperationPreparation::Ready(UserOperation::Login { body })
            } else {
                OperationPreparation::Ready(UserOperation::Register { body })
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
            OperationPreparation::Ready(UserOperation::Resume {
                credential: Zeroizing::new(value_base64.into_bytes()),
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
                AuthorityState::Anonymous | AuthorityState::Bound(_) => {}
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
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
        | Route::ConfirmAccountDeletion => {
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
                Err(rejection) => return rejection,
            };
            let body = request
                .body_base64
                .expect("validated protected JSON body presence")
                .into_bytes();
            let body = Zeroizing::new(body);
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
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
                Err(rejection) => return rejection,
            };
            OperationPreparation::Ready(UserOperation::Protected {
                authority,
                operation: ProtectedUserOperation::DeleteApiKey {
                    name: api_key_name
                        .expect("classified API-key item route must have a decoded name"),
                },
            })
        }
    }
}

fn bound_user_authority(
    authority: AuthorityState,
) -> Result<BoundUserAuthority, OperationPreparation> {
    let AuthorityState::Bound(bound) = authority else {
        return Err(rejected_authentication_required());
    };
    let BoundPrincipal::User {
        user_id,
        project_id,
        auth_context,
    } = bound.principal()
    else {
        return Err(rejected_authentication_required());
    };
    Ok(BoundUserAuthority {
        user_id: *user_id,
        project_id: *project_id,
        auth_context: auth_context.clone(),
    })
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
) -> ApplicationOutcome {
    match operation {
        UserOperation::Login { body } => {
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
            )
        }
        UserOperation::Register { body } => {
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
            )
        }
        UserOperation::Resume { mut credential } => {
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
            )
        }
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
        UserOperation::Protected {
            authority,
            operation,
        } => {
            debug_assert!(authentication.is_none());
            let session_effect = operation.session_effect_on_success();
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
            ApplicationOutcome::success(response, session_effect)
        }
    }
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

fn parse_json_body<T: DeserializeOwned>(body: SensitiveBytes) -> Result<T, ApiError> {
    serde_json::from_slice::<T>(&body).map_err(|_| ApiError::BadRequest)
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

fn finish_user_binding(
    app_state: &AppState,
    lease: &V2SessionLease,
    verified: VerifiedUserAuthentication,
    authentication: AuthenticationReservation,
    monotonic_now: Instant,
    response_kind: UserAuthResponseKind,
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

    let authority = BoundAuthority::verified_user(&verified, authentication_expires_at);
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
    OperationPreparation::Rejected(LogicalUnaryResponse::protocol_error(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "Invalid request",
    ))
}

fn rejected_authentication_required() -> OperationPreparation {
    OperationPreparation::Rejected(LogicalUnaryResponse::protocol_error(
        StatusCode::UNAUTHORIZED,
        "authentication_required",
        "Authentication required",
    ))
}

fn has_exact_json_content_type(headers: &[HeaderField]) -> bool {
    let [header] = headers else {
        return false;
    };
    if header.name != "content-type" {
        return false;
    }
    let Ok(value) = std::str::from_utf8(header.value_base64.as_slice()) else {
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
        RequestEnvelope {
            version: Version2,
            request_id: RequestId::from_bytes([0x31; 16]),
            response_mode: ResponseMode::Unary,
            credential,
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

    fn bound_user_authority() -> AuthorityState {
        let auth_context = AuthContext::new(AuthMethod::Password, 7, [0x32; 32]);
        AuthorityState::Bound(BoundAuthority::user(
            uuid::Uuid::from_bytes([0x33; 16]),
            7,
            &auth_context,
            Instant::now() + std::time::Duration::from_secs(60),
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
}
