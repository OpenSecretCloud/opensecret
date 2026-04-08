use crate::User;
use crate::{
    db::DBError,
    email::{send_hello_email, send_verification_email},
    jwt::{validate_token, CustomClaims, NewToken, TokenType},
    models::email_verification::NewEmailVerification,
    models::push_devices::PushDevice,
};
use crate::{jwt::USER_REFRESH, web::encryption_middleware::EncryptedResponse};
use crate::{
    web::encryption_middleware::{decrypt_request, encrypt_response},
    Error,
};
use crate::{ApiError, AppState};
use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    routing::{get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::spawn;
use tracing::{debug, error, info};
use uuid::Uuid;

#[derive(Deserialize, Clone)]
pub struct PasswordResetRequestPayload {
    email: String,
    hashed_secret: String,
    client_id: Uuid,
}

#[derive(Deserialize, Clone)]
pub struct PasswordResetConfirmPayload {
    email: String,
    alphanumeric_code: String,
    plaintext_secret: String,
    new_password: String,
    client_id: Uuid,
}

#[derive(Deserialize, Clone)]
pub struct Credentials {
    pub email: Option<String>,
    pub id: Option<Uuid>,
    pub password: String,
    pub client_id: Uuid,
}

#[derive(Deserialize, Clone)]
pub struct RegisterCredentials {
    pub name: Option<String>,
    pub email: Option<String>,
    pub password: String,
    pub client_id: Uuid,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/login",
            post(login).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<Credentials>,
            )),
        )
        .route(
            "/register",
            post(register).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<RegisterCredentials>,
            )),
        )
        .route(
            "/logout",
            post(logout).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<LogoutRequest>,
            )),
        )
        .route(
            "/refresh",
            post(refresh_token).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<RefreshRequest>,
            )),
        )
        .route(
            "/verify-email/:code",
            get(verify_email).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/password-reset/request",
            post(password_reset_request).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PasswordResetRequestPayload>,
            )),
        )
        .route(
            "/password-reset/confirm",
            post(password_reset_confirm).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PasswordResetConfirmPayload>,
            )),
        )
        .with_state(app_state)
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub id: Uuid,
    pub email: Option<String>,
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RefreshRequest {
    refresh_token: String,
}

#[derive(Serialize)]
pub struct RefreshResponse {
    access_token: String,
    refresh_token: String,
}

#[derive(Deserialize, Clone)]
pub struct LogoutRequest {
    refresh_token: String,
    #[serde(default)]
    push_device_id: Option<Uuid>,
}

pub async fn login(
    State(data): State<Arc<AppState>>,
    Extension(creds): Extension<Credentials>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<AuthResponse>>, ApiError> {
    debug!("Entering login function");
    tracing::trace!("call login");

    let auth_response = login_internal(data.clone(), creds).await?;
    let result = encrypt_response(&data, &session_id, &auth_response).await;
    debug!("Exiting login function");
    result
}

async fn login_internal(data: Arc<AppState>, creds: Credentials) -> Result<AuthResponse, ApiError> {
    // First get the project by client_id and verify it's active
    let project = data
        .db
        .get_org_project_by_client_id(creds.client_id)
        .map_err(|_| ApiError::BadRequest)?;

    // Get user based on provided credentials, scoped to project
    let user = match (&creds.email, &creds.id) {
        (Some(email), _) => {
            // Try email first if provided
            match data.db.get_user_by_email(email.clone(), project.id) {
                Ok(user) => user,
                Err(DBError::UserNotFound) => {
                    error!("User not found by email: {email}");
                    return Err(ApiError::InvalidUsernameOrPassword);
                }
                Err(e) => {
                    error!("Error fetching user by email: {:?}", e);
                    return Err(ApiError::InternalServerError);
                }
            }
        }
        (None, Some(id)) => {
            // Only allow ID-based login for guest users
            match data.db.get_user_by_uuid(*id) {
                Ok(user) => {
                    if !user.is_guest() {
                        error!("ID-based login not allowed for users with email addresses");
                        return Err(ApiError::InvalidUsernameOrPassword);
                    }
                    // Verify user belongs to the specified project
                    if user.project_id != project.id {
                        error!("User does not belong to specified project");
                        return Err(ApiError::InvalidUsernameOrPassword);
                    }
                    user
                }
                Err(DBError::UserNotFound) => {
                    error!("User not found by ID: {id}");
                    return Err(ApiError::InvalidUsernameOrPassword);
                }
                Err(e) => {
                    error!("Error fetching user by ID: {:?}", e);
                    return Err(ApiError::InternalServerError);
                }
            }
        }
        (None, None) => {
            error!("Neither email nor ID provided for login");
            return Err(ApiError::InvalidUsernameOrPassword);
        }
    };

    // Check if the user is an OAuth-only user
    if user.password_enc.is_none() {
        error!("Attempted password login for OAuth-only user");
        return Err(ApiError::InvalidUsernameOrPassword);
    }

    // Proceed with password authentication
    match data
        .authenticate_user(creds.email, creds.id, creds.password, project.id)
        .await
    {
        Ok(Some(authenticated_user)) => {
            let access_token = NewToken::new(&authenticated_user, TokenType::Access, &data)?;
            let refresh_token = NewToken::new(&authenticated_user, TokenType::Refresh, &data)?;
            let auth_response = AuthResponse {
                id: authenticated_user.get_id(),
                email: authenticated_user.get_email().map(|s| s.to_string()),
                access_token: access_token.token,
                refresh_token: refresh_token.token,
            };
            Ok(auth_response)
        }
        Ok(None) => {
            error!("Invalid password attempt");
            Err(ApiError::InvalidUsernameOrPassword)
        }
        Err(e) => {
            error!("Error authenticating user: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

pub async fn logout(
    State(data): State<Arc<AppState>>,
    Extension(logout_request): Extension<LogoutRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering logout function");
    info!("Logout request received");
    // TODO actually delete the refresh token

    if let Some((user_id, push_device_id)) = logout_push_revoke_target(
        &logout_request,
        validate_token(&logout_request.refresh_token, &data, USER_REFRESH),
    ) {
        match data.db.get_pool().get() {
            Ok(mut conn) => {
                if let Err(revoke_error) =
                    PushDevice::revoke_by_uuid_and_user(&mut conn, push_device_id, user_id)
                {
                    error!(
                        "Failed to revoke push device during logout cleanup: {:?}",
                        revoke_error
                    );
                }
            }
            Err(pool_error) => {
                error!(
                    "Failed to get DB connection during logout push cleanup: {:?}",
                    pool_error
                );
            }
        }
    }

    tracing::trace!("Logout request received");
    let response = json!({ "message": "Logged out successfully" });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting logout function");
    result
}

fn logout_push_revoke_target(
    logout_request: &LogoutRequest,
    claims: Result<CustomClaims, ApiError>,
) -> Option<(Uuid, Uuid)> {
    let push_device_id = logout_request.push_device_id?;
    let claims = claims.ok()?;
    let user_id = Uuid::parse_str(&claims.sub).ok()?;
    Some((user_id, push_device_id))
}

pub async fn register(
    State(data): State<Arc<AppState>>,
    Extension(creds): Extension<RegisterCredentials>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<AuthResponse>>, ApiError> {
    debug!("Entering register function");
    tracing::trace!("call register");

    let user = match data.register_user(creds.clone()).await {
        Ok(user) => user,
        Err(Error::UserAlreadyExists) => {
            tracing::warn!("Cannot register user that already exists");
            return Err(ApiError::EmailAlreadyExists);
        }
        Err(e) => {
            tracing::error!("Error registering user: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    // Handle new user registration
    handle_new_user_registration(&data, &user, true).await?;

    // After registration, proceed with login
    let login_result = login_internal(
        data.clone(),
        Credentials {
            email: creds.email,
            id: Some(user.uuid),
            password: creds.password,
            client_id: creds.client_id,
        },
    )
    .await?;

    let result = encrypt_response(&data, &session_id, &login_result).await;
    debug!("Exiting register function");
    result
}

pub async fn handle_new_user_registration(
    data: &AppState,
    user: &User,
    requires_email_verification: bool,
) -> Result<(), ApiError> {
    // Only handle email verification if user has an email
    if requires_email_verification && !user.is_guest() {
        // Create email verification entry
        let new_verification = NewEmailVerification::new(user.uuid, 24, false);
        let verification = match data.db.create_email_verification(new_verification) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Error creating email verification: {:?}", e);
                return Err(ApiError::InternalServerError);
            }
        };

        // Send verification email in the background
        if let Some(email) = user.get_email() {
            let email = email.to_string();
            let verification_code = verification.verification_code;
            let data = data.clone();
            let project_id = user.project_id;
            spawn(async move {
                if let Err(e) =
                    send_verification_email(&data, project_id, email, verification_code).await
                {
                    tracing::error!("Could not send verification email: {e}");
                }
            });
        }
    }

    // Only send welcome email if user has an email
    if !user.is_guest() {
        let welcome_email = user.get_email().unwrap().to_string(); // Safe to unwrap since we checked is_guest()
        let data = data.clone();
        let project_id = user.project_id;
        spawn(async move {
            if let Err(e) = send_hello_email(&data, project_id, welcome_email).await {
                tracing::error!("Could not schedule welcome email: {e}");
            }
        });
    }

    Ok(())
}

pub async fn refresh_token(
    State(data): State<Arc<AppState>>,
    Extension(refresh_request): Extension<RefreshRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<RefreshResponse>>, ApiError> {
    debug!("Entering refresh_token function");
    info!("Refresh token request received");

    let claims = validate_token(&refresh_request.refresh_token, &data, USER_REFRESH)?;

    // Audience check is now handled by validate_token
    let user_id = Uuid::parse_str(&claims.sub).map_err(|_| ApiError::InvalidJwt)?;

    let user = data
        .get_user(user_id)
        .await
        .map_err(|_| ApiError::Unauthorized)?;

    let new_access_token = NewToken::new(&user, TokenType::Access, &data)?;
    let new_refresh_token = NewToken::new(&user, TokenType::Refresh, &data)?;

    let response = RefreshResponse {
        access_token: new_access_token.token,
        refresh_token: new_refresh_token.token,
    };
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting refresh_token function");
    result
}

pub async fn verify_email(
    State(data): State<Arc<AppState>>,
    Path(code): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering verify_email function");
    let verification = match data.db.get_email_verification_by_code(code) {
        Ok(v) => v,
        Err(DBError::EmailVerificationNotFound) => return Err(ApiError::BadRequest),
        Err(_) => return Err(ApiError::InternalServerError),
    };

    if verification.is_expired() {
        return Err(ApiError::BadRequest);
    }

    if verification.is_verified {
        let response = json!({
            "message": "Email already verified"
        });
        return encrypt_response(&data, &session_id, &response).await;
    }

    let mut verification = verification;
    if data.db.verify_email(&mut verification).is_err() {
        return Err(ApiError::InternalServerError);
    }

    let response = json!({
        "message": "Email verified successfully"
    });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting verify_email function");
    result
}

pub async fn password_reset_request(
    State(data): State<Arc<AppState>>,
    Extension(payload): Extension<PasswordResetRequestPayload>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering password_reset_request function");

    // Get project by client_id
    let project = data
        .db
        .get_org_project_by_client_id(payload.client_id)
        .map_err(|_| ApiError::BadRequest)?;

    // Check if user exists and is not an OAuth-only user
    match data.db.get_user_by_email(payload.email.clone(), project.id) {
        Ok(user) => {
            if user.password_enc.is_none() {
                error!("OAuth-only user attempted to reset password");
                // Still return success to not leak information about the account
                let response = json!({
                    "message": "If an account with that email exists, we have sent a password reset link."
                });
                return encrypt_response(&data, &session_id, &response).await;
            }
        }
        Err(DBError::UserNotFound) => {
            // User doesn't exist, but we don't want to leak this information
            let response = json!({
                "message": "If an account with that email exists, we have sent a password reset link."
            });
            return encrypt_response(&data, &session_id, &response).await;
        }
        Err(e) => {
            error!("Error in password reset request: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    }

    // Proceed with password reset request
    let _ = data
        .create_password_reset_request(payload.email.clone(), payload.hashed_secret, project.id)
        .await
        .map_err(|e| {
            error!("Error in create_password_reset_request: {:?}", e);
            // We don't expose this error to the user
        });

    let response = json!({
        "message": "If an account with that email exists, we have sent a password reset link."
    });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting password_reset_request function");
    result
}

pub async fn password_reset_confirm(
    State(data): State<Arc<AppState>>,
    Extension(payload): Extension<PasswordResetConfirmPayload>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering password_reset_confirm function");

    // Get project by client_id
    let project = data
        .db
        .get_org_project_by_client_id(payload.client_id)
        .map_err(|_| ApiError::BadRequest)?;

    // Check if user exists and is not an OAuth-only user
    match data.db.get_user_by_email(payload.email.clone(), project.id) {
        Ok(user) => {
            if user.password_enc.is_none() {
                error!("OAuth-only user attempted to reset password");
                return Err(ApiError::InvalidUsernameOrPassword);
            }
        }
        Err(DBError::UserNotFound) => {
            error!("User not found in password reset confirm");
            return Err(ApiError::InvalidUsernameOrPassword);
        }
        Err(e) => {
            error!("Error in password reset confirm: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    }

    // Proceed with password reset confirmation
    data.confirm_password_reset(
        payload.email,
        payload.alphanumeric_code,
        payload.plaintext_secret,
        payload.new_password,
        project.id,
    )
    .await
    .map_err(|e| match e {
        crate::Error::PasswordResetExpired => ApiError::BadRequest,
        crate::Error::InvalidPasswordResetSecret => ApiError::BadRequest,
        crate::Error::InvalidPasswordResetRequest => ApiError::BadRequest,
        _ => ApiError::InternalServerError,
    })?;

    let response = json!({
        "message": "Password reset successful. You can now log in with your new password."
    });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting password_reset_confirm function");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_claims(user_id: Uuid) -> CustomClaims {
        CustomClaims {
            sub: user_id.to_string(),
            aud: Some(USER_REFRESH.to_string()),
            azp: None,
            role: None,
        }
    }

    #[test]
    fn logout_push_revoke_target_requires_push_device_id() {
        let request = LogoutRequest {
            refresh_token: "refresh-token".to_string(),
            push_device_id: None,
        };

        assert_eq!(
            logout_push_revoke_target(&request, Ok(sample_claims(Uuid::new_v4()))),
            None
        );
    }

    #[test]
    fn logout_push_revoke_target_rejects_invalid_user_id() {
        let push_device_id = Uuid::new_v4();
        let request = LogoutRequest {
            refresh_token: "refresh-token".to_string(),
            push_device_id: Some(push_device_id),
        };
        let claims = CustomClaims {
            sub: "not-a-uuid".to_string(),
            aud: Some(USER_REFRESH.to_string()),
            azp: None,
            role: None,
        };

        assert_eq!(logout_push_revoke_target(&request, Ok(claims)), None);
    }

    #[test]
    fn logout_push_revoke_target_returns_user_and_device_ids() {
        let user_id = Uuid::new_v4();
        let push_device_id = Uuid::new_v4();
        let request = LogoutRequest {
            refresh_token: "refresh-token".to_string(),
            push_device_id: Some(push_device_id),
        };

        assert_eq!(
            logout_push_revoke_target(&request, Ok(sample_claims(user_id))),
            Some((user_id, push_device_id))
        );
    }
}
