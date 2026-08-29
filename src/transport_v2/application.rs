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
use crate::web::login_routes::{
    authenticate_login, register_and_authenticate, AuthResponse, Credentials, RefreshResponse,
    RegisterCredentials,
};
use crate::web::protected_routes::{
    decrypt_data_value, delete_all_kv_values, delete_kv_value, encrypt_data_value,
    private_key_bytes_data, private_key_data, protected_user_data, public_key_data, put_kv_value,
    sign_message_data, third_party_token_data, DecryptDataRequest, DerivationPathQuery,
    EncryptDataRequest, KvValue, PublicKeyQuery, SignMessageRequest, ThirdPartyTokenRequest,
};
use crate::{ApiError, AppState, VerifiedUserAuthentication};

use super::envelope::{
    decode_canonical_kv_item_path, Credential, EncodedBytes, EnvelopeLimits, HeaderField,
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
    Protected {
        authority: BoundUserAuthority,
        operation: ProtectedUserOperation,
    },
}

pub(crate) enum ProtectedUserOperation {
    GetUser,
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
    DeleteKv {
        key: Zeroizing<String>,
    },
    DeleteAllKv,
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
    pub(crate) bound_session: bool,
}

impl ApplicationOutcome {
    fn success(response: LogicalUnaryResponse, bound_session: bool) -> Self {
        Self {
            response,
            bound_session,
        }
    }

    fn error(error: ApiError) -> Self {
        Self {
            response: LogicalUnaryResponse::api_error(&error),
            bound_session: false,
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
        GetUser,
        GetPrivateKey,
        GetPrivateKeyBytes,
        GetPublicKey,
        SignMessage,
        IssueThirdPartyToken,
        EncryptData,
        DecryptData,
        PutKv,
        DeleteKv,
        DeleteAllKv,
    }

    let kv_key = match decode_canonical_kv_item_path(request.method, &request.path) {
        Ok(key) => key,
        Err(_) => {
            request.path.zeroize();
            return rejected_bad_request();
        }
    };
    let route = match (request.method, request.path.as_str()) {
        (LogicalMethod::Post, "/login") => Some(Route::Login),
        (LogicalMethod::Post, "/register") => Some(Route::Register),
        (LogicalMethod::Post, "/refresh") => Some(Route::Resume),
        (LogicalMethod::Get, "/protected/user") => Some(Route::GetUser),
        (LogicalMethod::Get, "/protected/private_key") => Some(Route::GetPrivateKey),
        (LogicalMethod::Get, "/protected/private_key_bytes") => Some(Route::GetPrivateKeyBytes),
        (LogicalMethod::Get, "/protected/public_key") => Some(Route::GetPublicKey),
        (LogicalMethod::Post, "/protected/sign_message") => Some(Route::SignMessage),
        (LogicalMethod::Post, "/protected/third_party_token") => Some(Route::IssueThirdPartyToken),
        (LogicalMethod::Post, "/protected/encrypt") => Some(Route::EncryptData),
        (LogicalMethod::Post, "/protected/decrypt") => Some(Route::DecryptData),
        (LogicalMethod::Delete, "/protected/kv") => Some(Route::DeleteAllKv),
        (LogicalMethod::Put, _) if kv_key.is_some() => Some(Route::PutKv),
        (LogicalMethod::Delete, _) if kv_key.is_some() => Some(Route::DeleteKv),
        _ => None,
    };
    if kv_key.is_some() {
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
        Route::SignMessage
        | Route::IssueThirdPartyToken
        | Route::EncryptData
        | Route::DecryptData => {
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
                        lease.state().close();
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
                        lease.state().close();
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            ApplicationOutcome::success(response, false)
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

    ApplicationOutcome::success(response, true)
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

        assert!(matches!(
            prepare_user_operation(
                envelope(
                    LogicalMethod::Get,
                    "/protected/kv/key%2Fpart",
                    Vec::new(),
                    None,
                    None,
                ),
                bound_user_authority(),
            ),
            OperationPreparation::Unsupported
        ));
    }

    #[test]
    fn kv_value_keeps_the_existing_json_string_wire_shape() {
        let wire = "\"line\\né\"";
        let value: KvValue = serde_json::from_slice(wire.as_bytes()).unwrap();
        assert_eq!(value.as_str(), "line\né");
        assert_eq!(serde_json::to_vec(&value).unwrap(), wire.as_bytes());
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
