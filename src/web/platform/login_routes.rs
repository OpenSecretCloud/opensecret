use crate::{
    email::send_platform_verification_email,
    jwt::{NewToken, TokenType, PLATFORM_REFRESH},
    models::{
        org_memberships::OrgRole, platform_email_verification::NewPlatformEmailVerification,
        platform_users::NewPlatformUser,
    },
    web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse},
    ApiError, AppState, Error,
};
use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    routing::{get, post},
    Extension, Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::spawn;
use tracing::{debug, error, info};
use uuid::Uuid;
use validator::Validate;

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformLoginRequest {
    #[validate(email(message = "Invalid email format"))]
    #[validate(length(max = 255, message = "Email must not exceed 255 characters"))]
    pub email: String,

    #[validate(length(
        min = 8,
        max = 64,
        message = "Password must be between 8 and 64 characters"
    ))]
    pub password: String,
}

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformRegisterRequest {
    #[validate(email(message = "Invalid email format"))]
    #[validate(length(max = 255, message = "Email must not exceed 255 characters"))]
    pub email: String,

    #[validate(length(
        min = 8,
        max = 64,
        message = "Password must be between 8 and 64 characters"
    ))]
    pub password: String,

    #[validate(length(max = 50, message = "Name must not exceed 50 characters"))]
    pub name: Option<String>,

    #[validate(length(min = 1, message = "Invite code is required"))]
    pub invite_code: String,
}

#[derive(Serialize)]
pub struct PlatformAuthResponse {
    pub id: Uuid,
    pub email: String,
    pub name: Option<String>,
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformRefreshRequest {
    #[validate(length(min = 1, message = "Refresh token cannot be empty"))]
    pub refresh_token: String,
}

#[derive(Serialize)]
pub struct PlatformRefreshResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformLogoutRequest {
    #[validate(length(min = 1, message = "Refresh token cannot be empty"))]
    pub refresh_token: String,
}

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformPasswordResetRequestPayload {
    #[validate(email(message = "Invalid email format"))]
    #[validate(length(max = 255, message = "Email must not exceed 255 characters"))]
    pub email: String,

    #[validate(length(min = 1, message = "Hashed secret cannot be empty"))]
    pub hashed_secret: String,
}

#[derive(Deserialize, Clone, Validate)]
pub struct PlatformPasswordResetConfirmPayload {
    #[validate(email(message = "Invalid email format"))]
    #[validate(length(max = 255, message = "Email must not exceed 255 characters"))]
    pub email: String,

    #[validate(length(min = 1, message = "Alphanumeric code cannot be empty"))]
    pub alphanumeric_code: String,

    #[validate(length(min = 1, message = "Plaintext secret cannot be empty"))]
    pub plaintext_secret: String,

    #[validate(length(
        min = 8,
        max = 64,
        message = "New password must be between 8 and 64 characters"
    ))]
    pub new_password: String,
}

#[derive(Debug, Serialize)]
pub struct PlatformOrg {
    pub id: i32,
    pub uuid: Uuid,
    pub name: String,
    pub role: OrgRole,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct PlatformProject {
    pub id: i32,
    pub uuid: Uuid,
    pub client_id: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/platform/login",
            post(login_platform_user).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformLoginRequest>,
            )),
        )
        .route(
            "/platform/register",
            post(register_platform_user).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformRegisterRequest>,
            )),
        )
        .route(
            "/platform/refresh",
            post(refresh_platform_token).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformRefreshRequest>,
            )),
        )
        .route(
            "/platform/logout",
            post(logout_platform_user).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformLogoutRequest>,
            )),
        )
        .route(
            "/platform/verify-email/:code",
            get(verify_platform_email)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/platform/password-reset/request",
            post(platform_password_reset_request).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformPasswordResetRequestPayload>,
            )),
        )
        .route(
            "/platform/password-reset/confirm",
            post(platform_password_reset_confirm).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<PlatformPasswordResetConfirmPayload>,
            )),
        )
        .with_state(app_state)
}

pub async fn login_platform_user(
    State(data): State<Arc<AppState>>,
    Extension(login_request): Extension<PlatformLoginRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<PlatformAuthResponse>>, ApiError> {
    debug!("Entering login_platform_user function");

    // Validate request
    if let Err(errors) = login_request.validate() {
        error!("Validation error: {:?}", errors);
        return Err(ApiError::BadRequest);
    }

    let auth_response = login_internal_platform(data.clone(), login_request).await?;
    let result = encrypt_response(&data, &session_id, &auth_response).await;
    debug!("Exiting login_platform_user function");
    result
}

async fn login_internal_platform(
    data: Arc<AppState>,
    login_request: PlatformLoginRequest,
) -> Result<PlatformAuthResponse, ApiError> {
    // Authenticate the platform user
    match data
        .authenticate_platform_user(&login_request.email, login_request.password)
        .await
    {
        Ok(Some(platform_user)) => {
            // Generate tokens
            let access_token =
                NewToken::new_for_platform_user(&platform_user, TokenType::Access, &data)?;
            let refresh_token =
                NewToken::new_for_platform_user(&platform_user, TokenType::Refresh, &data)?;

            let auth_response = PlatformAuthResponse {
                id: platform_user.uuid,
                email: platform_user.email,
                name: platform_user.name,
                access_token: access_token.token,
                refresh_token: refresh_token.token,
            };
            Ok(auth_response)
        }
        Ok(None) => {
            error!("Invalid login attempt for platform user");
            Err(ApiError::InvalidUsernameOrPassword)
        }
        Err(e) => {
            error!("Error authenticating platform user: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

pub async fn register_platform_user(
    State(data): State<Arc<AppState>>,
    Extension(register_request): Extension<PlatformRegisterRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<PlatformAuthResponse>>, ApiError> {
    debug!("Entering register_platform_user function");

    // Validate request
    if let Err(errors) = register_request.validate() {
        error!("Validation error: {:?}", errors);
        return Err(ApiError::BadRequest);
    }

    // Check if user already exists
    if data
        .db
        .get_platform_user_by_email(&register_request.email)?
        .is_some()
    {
        return Err(ApiError::EmailAlreadyExists);
    }

    // Validate invite code
    let invite_code = match Uuid::parse_str(&register_request.invite_code) {
        Ok(code) => code,
        Err(e) => {
            error!("Invalid invite code format: {:?}", e);
            return Err(ApiError::BadRequest);
        }
    };

    // Check if invite code is valid
    if let Err(e) = data.db.validate_platform_invite_code(invite_code) {
        error!("Invalid invite code: {:?}", e);
        return Err(ApiError::BadRequest);
    }

    // Hash and encrypt the password
    let password_hash = password_auth::generate_hash(register_request.password);
    let secret_key = secp256k1::SecretKey::from_slice(&data.enclave_key)
        .map_err(|_| ApiError::InternalServerError)?;
    let encrypted_password =
        crate::encrypt::encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

    // Create the platform user
    let new_platform_user =
        NewPlatformUser::new(register_request.email.clone(), Some(encrypted_password))
            .with_name(register_request.name.unwrap_or_default());

    let platform_user = data
        .db
        .create_platform_user(new_platform_user)
        .map_err(|e| {
            error!("Failed to create platform user: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Create platform email verification
    let new_verification = match NewPlatformEmailVerification::new(platform_user.uuid, 24, false) {
        Ok(v) => v,
        Err(e) => {
            error!("Error creating platform email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };
    let verification = match data.db.create_platform_email_verification(new_verification) {
        Ok(v) => v,
        Err(e) => {
            error!("Error creating platform email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    // Send verification email in background
    let email = register_request.email.clone();
    let verification_code = verification.verification_code;
    let app_state = data.clone();

    spawn(async move {
        if let Err(e) = send_platform_verification_email(
            &app_state,
            app_state.resend_api_key.clone(),
            email,
            verification_code,
        )
        .await
        {
            error!("Could not send verification email: {}", e);
        }
    });

    // Generate tokens
    let access_token = NewToken::new_for_platform_user(&platform_user, TokenType::Access, &data)?;
    let refresh_token = NewToken::new_for_platform_user(&platform_user, TokenType::Refresh, &data)?;

    let response = PlatformAuthResponse {
        id: platform_user.uuid,
        email: platform_user.email,
        name: platform_user.name,
        access_token: access_token.token,
        refresh_token: refresh_token.token,
    };

    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting register_platform_user function");
    result
}

pub async fn refresh_platform_token(
    State(data): State<Arc<AppState>>,
    Extension(refresh_request): Extension<PlatformRefreshRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<PlatformRefreshResponse>>, ApiError> {
    debug!("Entering refresh_platform_token function");

    // Validate request
    if let Err(errors) = refresh_request.validate() {
        error!("Validation error: {:?}", errors);
        return Err(ApiError::BadRequest);
    }

    let claims =
        crate::jwt::validate_token(&refresh_request.refresh_token, &data, PLATFORM_REFRESH)?;
    let platform_user_id = Uuid::parse_str(&claims.sub).map_err(|_| ApiError::InvalidJwt)?;

    // Verify platform user still exists
    let platform_user = data.db.get_platform_user_by_uuid(platform_user_id)?;

    // Generate new tokens
    let new_access_token =
        NewToken::new_for_platform_user(&platform_user, TokenType::Access, &data)?;
    let new_refresh_token =
        NewToken::new_for_platform_user(&platform_user, TokenType::Refresh, &data)?;

    let response = PlatformRefreshResponse {
        access_token: new_access_token.token,
        refresh_token: new_refresh_token.token,
    };

    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting refresh_platform_token function");
    result
}

pub async fn verify_platform_email(
    State(data): State<Arc<AppState>>,
    Path(code): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering verify_platform_email function");

    // Retrieve the verification record using the code
    let verification = match data.db.get_platform_email_verification_by_code(code) {
        Ok(verification) => verification,
        Err(crate::db::DBError::PlatformEmailVerificationNotFound) => {
            error!("Platform email verification code not found: {}", code);
            return Err(ApiError::BadRequest);
        }
        Err(e) => {
            error!("Error retrieving platform email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    // Check if verification is already marked as verified
    if verification.is_verified {
        let response = json!({
            "message": "Email already verified"
        });
        return encrypt_response(&data, &session_id, &response).await;
    }

    // Check if verification is expired
    if verification.expires_at < Utc::now() {
        error!(
            "verification is expired for user: {}",
            verification.platform_user_id
        );
        return Err(ApiError::BadRequest);
    }

    // Mark the verification as verified
    let mut verification_to_update = verification.clone();
    if let Err(e) = data.db.verify_platform_email(&mut verification_to_update) {
        error!("Error verifying platform email: {:?}", e);
        return Err(ApiError::InternalServerError);
    }

    let response = json!({
        "message": "Email verified successfully"
    });

    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting verify_platform_email function");
    result
}

pub async fn logout_platform_user(
    State(data): State<Arc<AppState>>,
    Extension(logout_request): Extension<PlatformLogoutRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering logout_platform_user function");
    info!("Platform logout request received");

    // TODO: Implement token invalidation logic here when needed
    tracing::trace!(
        "Platform logout request for refresh token: {}",
        logout_request.refresh_token
    );

    let response = json!({ "message": "Logged out successfully" });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting logout_platform_user function");
    result
}

pub async fn platform_password_reset_request(
    State(data): State<Arc<AppState>>,
    Extension(payload): Extension<PlatformPasswordResetRequestPayload>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering platform_password_reset_request function");

    // Validate request
    if let Err(errors) = payload.validate() {
        error!("Validation error: {:?}", errors);
        return Err(ApiError::BadRequest);
    }

    // Check if user exists and is not an OAuth-only user
    match data.db.get_platform_user_by_email(&payload.email) {
        Ok(Some(user)) => {
            if user.password_enc.is_none() {
                error!("OAuth-only platform user attempted to reset password");
                // Still return success to not leak information about the account
                let response = json!({
                    "message": "If an account with that email exists, we have sent a password reset link."
                });
                return encrypt_response(&data, &session_id, &response).await;
            }
        }
        Ok(None) => {
            // User doesn't exist, but we don't want to leak this information
            let response = json!({
                "message": "If an account with that email exists, we have sent a password reset link."
            });
            return encrypt_response(&data, &session_id, &response).await;
        }
        Err(e) => {
            error!("Error in platform password reset request: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    }

    // Process the password reset request
    let _ = data
        .create_platform_password_reset_request(payload.email.clone(), payload.hashed_secret)
        .await
        .map_err(|e| {
            error!("Error in create_platform_password_reset_request: {:?}", e);
            // We don't expose this error to the user
        });

    let response = json!({
        "message": "If an account with that email exists, we have sent a password reset link."
    });
    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting platform_password_reset_request function");
    result
}

pub async fn platform_password_reset_confirm(
    State(data): State<Arc<AppState>>,
    Extension(payload): Extension<PlatformPasswordResetConfirmPayload>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering platform_password_reset_confirm function");

    // Validate request
    if let Err(errors) = payload.validate() {
        error!("Validation error: {:?}", errors);
        return Err(ApiError::BadRequest);
    }

    // Check if user exists and is not an OAuth-only user
    match data.db.get_platform_user_by_email(&payload.email) {
        Ok(Some(user)) => {
            if user.password_enc.is_none() {
                error!("OAuth-only platform user attempted to reset password");
                return Err(ApiError::InvalidUsernameOrPassword);
            }
        }
        Ok(None) => {
            error!("Platform user not found in password reset confirm");
            return Err(ApiError::InvalidUsernameOrPassword);
        }
        Err(e) => {
            error!("Error in platform password reset confirm: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    }

    // Proceed with password reset confirmation
    data.confirm_platform_password_reset(
        payload.email,
        payload.alphanumeric_code,
        payload.plaintext_secret,
        payload.new_password,
    )
    .await
    .map_err(|e| match e {
        Error::PasswordResetExpired => ApiError::BadRequest,
        Error::InvalidPasswordResetSecret => ApiError::BadRequest,
        Error::InvalidPasswordResetRequest => ApiError::BadRequest,
        _ => ApiError::InternalServerError,
    })?;

    let response = json!({
        "message": "Password reset successful. You can now log in with your new password."
    });

    let result = encrypt_response(&data, &session_id, &response).await;
    debug!("Exiting platform_password_reset_confirm function");
    result
}
