//! Explicit application projection for the first transport-v2 user slice.
//!
//! This module does not re-enter the transport-v1 router. It validates a small
//! exact operation allowlist, calls shared transport-neutral application
//! functions, and returns plaintext logical results for the gateway to encrypt
//! through the request's original session lease.

use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use chrono::{DateTime, Utc};
use serde::Serialize;
use zeroize::{Zeroize, Zeroizing};

use crate::jwt::{
    issue_transport_v2_user_tokens, validate_transport_v2_user_resumption, AuthContext,
};
use crate::web::login_routes::{
    authenticate_login, register_and_authenticate, AuthResponse, Credentials, RefreshResponse,
    RegisterCredentials,
};
use crate::web::protected_routes::protected_user_data;
use crate::{ApiError, AppState, VerifiedUserAuthentication};

use super::envelope::{
    Credential, EncodedBytes, HeaderField, LogicalMethod, RequestEnvelope, RequestId, ResponseMode,
};
use super::session::{
    AuthenticationReservation, AuthenticationStartError, AuthorityState, BoundAuthority,
    BoundPrincipal,
};
use super::session_cache::V2SessionLease;

const JSON_CONTENT_TYPE: &[u8] = b"application/json";

pub(crate) enum OperationPreparation {
    Unsupported,
    Rejected(LogicalUnaryResponse),
    Ready(UserOperation),
}

pub(crate) enum UserOperation {
    Login { body: Vec<u8> },
    Register { body: Vec<u8> },
    Resume { credential: Vec<u8> },
    GetUser(BoundUserAuthority),
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
    pub(crate) body: Option<Vec<u8>>,
}

impl LogicalUnaryResponse {
    pub(crate) fn json<T: Serialize>(status: StatusCode, value: &T) -> Result<Self, ApiError> {
        let body = serde_json::to_vec(value).map_err(|error| {
            tracing::error!(
                "Could not serialize transport-v2 logical response: {:?}",
                error
            );
            ApiError::InternalServerError
        })?;
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
            body: Some(body),
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
        request,
        ..
    } = envelope;

    let operation = match (request.method, request.path.as_str()) {
        (LogicalMethod::Post, "/login") => 0,
        (LogicalMethod::Post, "/register") => 1,
        (LogicalMethod::Post, "/refresh") => 2,
        (LogicalMethod::Get, "/protected/user") => 3,
        _ => return OperationPreparation::Unsupported,
    };

    if response_mode != ResponseMode::Unary || request.query.is_some() {
        return rejected_bad_request();
    }

    match operation {
        0 | 1 => {
            if credential.is_some()
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
            if operation == 0 {
                OperationPreparation::Ready(UserOperation::Login { body })
            } else {
                OperationPreparation::Ready(UserOperation::Register { body })
            }
        }
        2 => {
            if !request.headers.is_empty() || request.body_base64.is_some() {
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
                credential: value_base64.into_bytes(),
            })
        }
        3 => {
            if credential.is_some() || !request.headers.is_empty() || request.body_base64.is_some()
            {
                return rejected_bad_request();
            }
            let AuthorityState::Bound(bound) = authority else {
                return rejected_authentication_required();
            };
            let BoundPrincipal::User {
                user_id,
                project_id,
                auth_context,
            } = bound.principal()
            else {
                return rejected_authentication_required();
            };
            OperationPreparation::Ready(UserOperation::GetUser(BoundUserAuthority {
                user_id: *user_id,
                project_id: *project_id,
                auth_context: auth_context.clone(),
            }))
        }
        _ => unreachable!("operation classifier is exhaustive"),
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
) -> ApplicationOutcome {
    match operation {
        UserOperation::Login { mut body } => {
            let parsed =
                serde_json::from_slice::<Credentials>(&body).map_err(|_| ApiError::BadRequest);
            body.zeroize();
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
        UserOperation::Register { mut body } => {
            let parsed = serde_json::from_slice::<RegisterCredentials>(&body)
                .map_err(|_| ApiError::BadRequest);
            body.zeroize();
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
        UserOperation::Resume { credential } => {
            let credential = match String::from_utf8(credential) {
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
        UserOperation::GetUser(bound) => {
            debug_assert!(authentication.is_none());
            let user = match app_state.verify_bound_user(
                bound.user_id,
                bound.project_id,
                &bound.auth_context,
            ) {
                Ok(user) => user,
                Err(error) => {
                    if matches!(error, ApiError::Unauthorized | ApiError::InvalidJwt) {
                        lease.state().close();
                    }
                    return ApplicationOutcome::error(error);
                }
            };
            let response = match protected_user_data(&app_state, &user)
                .and_then(|value| LogicalUnaryResponse::json(StatusCode::OK, &value))
            {
                Ok(response) => response,
                Err(error) => return ApplicationOutcome::error(error),
            };
            ApplicationOutcome::success(response, false)
        }
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
    use crate::transport_v2::envelope::{LogicalRequest, Version2};

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
            OperationPreparation::Ready(UserOperation::GetUser(_))
        ));
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
}
