use crate::aws_credentials::AwsCredentialManager;
use crate::encrypt::generate_random_bytes_from_enclave;
use crate::Error;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::header,
    middleware::Next,
    response::IntoResponse,
};
use chrono::{Duration, Utc};
use jwt_compact::{alg::Es256k, prelude::*, AlgorithmExt};
use secp256k1::{All, PublicKey, Secp256k1, SecretKey};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use uuid::Uuid;

use crate::db::DBError;
use crate::{ApiError, AppState};
use jsonwebtoken::{
    encode as jwt_encode, Algorithm as JwtAlgorithm, EncodingKey, Header as JwtHeader,
};
use url::Url;

use crate::models::{platform_users::PlatformUser, users::User};

pub const USER_ACCESS: &str = "access";
pub const USER_REFRESH: &str = "refresh";

pub const PLATFORM_ACCESS: &str = "platform_access";
pub const PLATFORM_REFRESH: &str = "platform_refresh";

#[derive(Debug, Clone)]
pub enum TokenType {
    Access,
    Refresh,
    ThirdParty { aud: Option<String>, azp: String },
}

#[derive(Debug, Clone)]
pub struct NewToken {
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct JwtKeys {
    signing_key: SecretKey, // For ES256K
    secp: Secp256k1<All>,
}

impl JwtKeys {
    pub fn new(secret_bytes: Vec<u8>) -> Result<Self, Error> {
        // check for size before slicing
        if secret_bytes.len() < 32 {
            return Err(Error::EncryptionError(
                "Insufficient key length: must be at least 32 bytes".to_string(),
            ));
        }

        let secp = Secp256k1::new(); // Creates All context
        let signing_key = SecretKey::from_slice(&secret_bytes[..32])
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        Ok(Self { signing_key, secp })
    }

    pub fn public_key(&self) -> PublicKey {
        PublicKey::from_secret_key(&self.secp, &self.signing_key)
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct CustomClaims {
    pub sub: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

impl TokenType {
    pub fn validate_third_party_audience(aud: &str) -> Result<(), ApiError> {
        // Validate third party audience can't use our internal audience types
        const RESERVED_AUDIENCES: [&str; 4] =
            [USER_ACCESS, USER_REFRESH, PLATFORM_ACCESS, PLATFORM_REFRESH];

        // 1. Check for reserved audiences
        if RESERVED_AUDIENCES.contains(&aud) {
            tracing::error!(
                "Third-party tokens cannot use internal audience types: {}",
                aud
            );
            return Err(ApiError::BadRequest);
        }

        // 2. Check length limit (max 50 characters)
        const MAX_AUDIENCE_LENGTH: usize = 50;
        if aud.len() > MAX_AUDIENCE_LENGTH {
            tracing::error!(
                "Audience value exceeds maximum length of {}: {} (length: {})",
                MAX_AUDIENCE_LENGTH,
                aud,
                aud.len()
            );
            return Err(ApiError::BadRequest);
        }

        // 3. Check for null bytes which can cause issues in some systems
        if aud.contains('\0') {
            tracing::error!("Audience contains null bytes which is not allowed");
            return Err(ApiError::BadRequest);
        }

        // 4. Check for character set restrictions - only allow alphanumeric, dots, dashes, colons, and slashes
        // This helps prevent injection attacks while still allowing typical URL characters
        if !aud.chars().all(|c| {
            c.is_alphanumeric()
                || c == '.'
                || c == '-'
                || c == ':'
                || c == '/'
                || c == '_'
                || c == '~'
                || c == '?'
                || c == '&'
                || c == '='
                || c == '+'
                || c == '%'
                || c == '#'
        }) {
            tracing::error!("Audience contains disallowed characters: {}", aud);
            return Err(ApiError::BadRequest);
        }

        // 5. Parse the URL to ensure it's valid
        let _url = Url::parse(aud).map_err(|e| {
            tracing::error!("Invalid audience URL format: {}, error: {:?}", aud, e);
            ApiError::BadRequest
        })?;

        Ok(())
    }
}

impl NewToken {
    /// Attempts to generate a token for third-party authentication using a project-specific JWT key.
    /// Falls back to the default JWT key if no project-specific key exists.
    fn get_third_party_token(
        azp: &str,
        app_state: &AppState,
        header: &Header,
        claims: &Claims<CustomClaims>,
    ) -> Result<String, ApiError> {
        use crate::web::platform::common::THIRD_PARTY_JWT_SECRET;

        // Parse the "azp" value which should be the project client_id
        let project_client_id = Uuid::parse_str(azp).map_err(|e| {
            tracing::error!(
                "Invalid project client_id format in azp: {}, error: {:?}",
                azp,
                e
            );
            ApiError::BadRequest
        })?;

        // Look up the project by client_id (not UUID)
        let project = app_state
            .db
            .get_org_project_by_client_id(project_client_id)
            .map_err(|e| {
                tracing::error!(
                    "Error looking up project with client_id {}: {:?}",
                    project_client_id,
                    e
                );
                match e {
                    DBError::OrgProjectNotFound => ApiError::BadRequest,
                    _ => ApiError::InternalServerError,
                }
            })?;

        // Look up a custom JWT secret for this project
        match app_state
            .db
            .get_org_project_secret_by_key_name_and_project(THIRD_PARTY_JWT_SECRET, project.id)
        {
            Ok(Some(secret)) => {
                // Decrypt the custom JWT secret using the enclave key
                let secret_key =
                    secp256k1::SecretKey::from_slice(&app_state.enclave_key).map_err(|e| {
                        tracing::error!("Failed to create secret key from enclave key: {:?}", e);
                        ApiError::InternalServerError
                    })?;

                let decrypted_key =
                    crate::encrypt::decrypt_with_key(&secret_key, &secret.secret_enc).map_err(
                        |e| {
                            tracing::error!(
                                "Failed to decrypt custom JWT secret for project {}: {:?}",
                                project_client_id,
                                e
                            );
                            ApiError::InternalServerError
                        },
                    )?;

                // For custom secrets, use HS256 algorithm (HMAC with shared secret)
                // This is what third-party services like Supabase expect
                tracing::debug!(
                    "Using custom JWT secret with HS256 for project {}",
                    project_client_id
                );

                // Create HS256 header
                let jwt_header = JwtHeader::new(JwtAlgorithm::HS256);

                // Create encoding key from the decrypted secret
                let encoding_key = EncodingKey::from_secret(&decrypted_key);

                // Encode the token using HS256
                jwt_encode(&jwt_header, claims, &encoding_key).map_err(|e| {
                    tracing::error!("Error creating HS256 token with custom secret: {:?}", e);
                    ApiError::InternalServerError
                })
            }
            Ok(None) => {
                // No custom secret found, use the default key
                tracing::debug!(
                    "No custom JWT secret found for project {}, using default",
                    project_client_id
                );
                let es256k = Es256k::<Sha256>::new(app_state.config.jwt_keys.secp.clone());

                es256k
                    .token(header, claims, &app_state.config.jwt_keys.signing_key)
                    .map_err(|e| {
                        tracing::error!("Error creating token: {:?}", e);
                        ApiError::InternalServerError
                    })
            }
            Err(e) => {
                // Database error looking up the secret
                tracing::error!(
                    "Database error looking up custom JWT secret for project {}: {:?}",
                    project_client_id,
                    e
                );
                Err(ApiError::InternalServerError)
            }
        }
    }

    pub fn new(user: &User, token_type: TokenType, app_state: &AppState) -> Result<Self, ApiError> {
        let (aud, azp, role, duration) = match &token_type {
            TokenType::Access => (
                Some(USER_ACCESS.to_string()),
                None,
                None,
                Duration::minutes(app_state.config.access_token_maxage),
            ),
            TokenType::Refresh => (
                Some(USER_REFRESH.to_string()),
                None,
                None,
                Duration::days(app_state.config.refresh_token_maxage),
            ),
            TokenType::ThirdParty { aud, azp } => {
                // Validate the audience URL against allowed domains
                if aud.is_some() {
                    TokenType::validate_third_party_audience(aud.as_ref().expect("just checked"))?;
                }

                (
                    aud.clone(),
                    Some(azp.clone()),
                    Some("authenticated".to_string()),
                    Duration::hours(1),
                )
            }
        };

        let custom_claims = CustomClaims {
            sub: user.get_id().to_string(),
            aud,
            azp,
            role,
        };

        tracing::debug!("Creating new token with claims: {:?}", custom_claims);

        // Account for clock drift by setting issued_at 1 minute in the past
        let now = Utc::now();
        let iat = now - Duration::minutes(1);
        let exp = iat + duration;

        let mut claims = Claims::new(custom_claims);
        claims.issued_at = Some(iat);
        claims.expiration = Some(exp);
        claims.not_before = Some(iat);

        // Create header with typ field
        let header = Header::empty().with_token_type("JWT");

        // Check if we need to use a custom JWT secret for third-party tokens
        let token_string = if let TokenType::ThirdParty { azp, .. } = &token_type {
            // Try to get the third-party token using project-specific key or fall back to default key
            Self::get_third_party_token(azp, app_state, &header, &claims)?
        } else {
            // For normal user tokens, use the default key
            let es256k = Es256k::<Sha256>::new(app_state.config.jwt_keys.secp.clone());

            es256k
                .token(&header, &claims, &app_state.config.jwt_keys.signing_key)
                .map_err(|e| {
                    tracing::error!("Error creating token: {:?}", e);
                    ApiError::InternalServerError
                })?
        };

        tracing::debug!("Successfully created token");

        Ok(Self {
            token: token_string,
        })
    }

    pub fn new_for_platform_user(
        user: &PlatformUser,
        token_type: TokenType,
        app_state: &AppState,
    ) -> Result<Self, ApiError> {
        let (aud, azp, duration) = match token_type {
            TokenType::Access => (
                PLATFORM_ACCESS.to_string(),
                None,
                Duration::minutes(app_state.config.access_token_maxage),
            ),
            TokenType::Refresh => (
                PLATFORM_REFRESH.to_string(),
                None,
                Duration::days(app_state.config.refresh_token_maxage),
            ),
            TokenType::ThirdParty { .. } => {
                // Platform users cannot create third-party tokens
                return Err(ApiError::BadRequest);
            }
        };

        let custom_claims = CustomClaims {
            sub: user.uuid.to_string(),
            aud: Some(aud),
            azp,
            role: None,
        };

        tracing::debug!(
            "Creating new platform token with claims: {:?}",
            custom_claims
        );

        // Account for clock drift by setting issued_at 1 minute in the past
        let now = Utc::now();
        let iat = now - Duration::minutes(1);
        let exp = iat + duration;

        let mut claims = Claims::new(custom_claims);
        claims.issued_at = Some(iat);
        claims.expiration = Some(exp);
        claims.not_before = Some(iat);

        let header = Header::empty().with_token_type("JWT");
        let es256k = Es256k::<Sha256>::new(app_state.config.jwt_keys.secp.clone());

        let token_string = es256k
            .token(&header, &claims, &app_state.config.jwt_keys.signing_key)
            .map_err(|e| {
                tracing::error!("Error creating token: {:?}", e);
                ApiError::InternalServerError
            })?;

        tracing::debug!("Successfully created platform token");

        Ok(Self {
            token: token_string,
        })
    }
}

pub async fn generate_jwt_secret(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
) -> Result<Vec<u8>, Error> {
    tracing::info!("Generating new JWT secret");
    if let Some(cred_manager) = aws_credential_manager.read().await.as_ref().cloned() {
        let aws_creds = cred_manager
            .get_credentials()
            .await
            .expect("should have creds");

        generate_random_bytes_from_enclave(
            &aws_creds.region,
            &aws_creds.access_key_id,
            &aws_creds.secret_access_key,
            &aws_creds.token,
            32,
        )
        .await
        .map_err(|e| Error::EncryptionError(e.to_string()))
    } else {
        Ok(crate::encrypt::generate_random::<32>().to_vec())
    }
}

pub async fn validate_jwt(
    State(data): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> impl IntoResponse {
    tracing::debug!("Entering validate_jwt");
    let token = match req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|auth_header| auth_header.to_str().ok())
        .and_then(|auth_value| auth_value.strip_prefix("Bearer ").map(ToString::to_string))
    {
        Some(token) => token,
        None => return ApiError::InvalidJwt.into_response(),
    };

    tracing::trace!("Validating JWT");

    let claims = match validate_token(&token, &data, USER_ACCESS) {
        Ok(claims) => claims,
        Err(_) => return ApiError::InvalidJwt.into_response(),
    };

    let user_uuid: Uuid = match Uuid::parse_str(&claims.sub) {
        Ok(uuid) => uuid,
        Err(e) => {
            tracing::error!("Error parsing user uuid: {:?}", e);
            return ApiError::InvalidJwt.into_response();
        }
    };

    let user = match data.get_user(user_uuid).await {
        Ok(user) => user,
        Err(e) => {
            tracing::error!("Error getting user: {:?}", e);
            return ApiError::InternalServerError.into_response();
        }
    };

    req.extensions_mut().insert(user);
    tracing::debug!("Exiting validate_jwt");
    next.run(req).await
}

pub async fn validate_platform_jwt(
    State(data): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> impl IntoResponse {
    tracing::debug!("Entering validate_platform_jwt");
    let token = match req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|auth_header| auth_header.to_str().ok())
        .and_then(|auth_value| auth_value.strip_prefix("Bearer ").map(ToString::to_string))
    {
        Some(token) => token,
        None => return ApiError::InvalidJwt.into_response(),
    };

    tracing::trace!("Validating platform JWT");

    let claims = match validate_token(&token, &data, PLATFORM_ACCESS) {
        Ok(claims) => claims,
        Err(_) => return ApiError::InvalidJwt.into_response(),
    };

    let platform_user_id: Uuid = match Uuid::parse_str(&claims.sub) {
        Ok(uuid) => uuid,
        Err(e) => {
            tracing::error!("Error parsing platform user uuid: {:?}", e);
            return ApiError::InvalidJwt.into_response();
        }
    };

    let platform_user = match data.db.get_platform_user_by_uuid(platform_user_id) {
        Ok(user) => user,
        Err(e) => {
            tracing::error!("Error getting platform user: {:?}", e);
            return ApiError::Unauthorized.into_response();
        }
    };

    req.extensions_mut().insert(platform_user);
    tracing::debug!("Exiting validate_platform_jwt");
    next.run(req).await
}

pub(crate) fn validate_token(
    original_token: &str,
    data: &AppState,
    expected_audience: &str,
) -> Result<CustomClaims, ApiError> {
    // Try ES256K first
    let es256k = Es256k::<Sha256>::new(data.config.jwt_keys.secp.clone());
    let public_key = data.config.jwt_keys.public_key();

    tracing::trace!("Attempting to validate ES256K token");

    // First parse the token with the correct type
    let parsed_token = match UntrustedToken::new(original_token) {
        Ok(token) => token,
        Err(e) => {
            tracing::error!("Failed to parse token: {:?}", e);
            return Err(ApiError::InvalidJwt);
        }
    };

    // Deserialize claims first
    let token: Token<CustomClaims> = match es256k.validator(&public_key).validate(&parsed_token) {
        Ok(token) => {
            tracing::trace!("ES256K signature validation successful");

            // Only validate expiration, not maturity
            let time_options = TimeOptions::default();
            if let Err(e) = token.claims().validate_expiration(&time_options) {
                tracing::error!("Token expired: {:?}", e);
                return Err(ApiError::InvalidJwt);
            }

            // Validate audience with proper type annotation
            let claims: &Claims<CustomClaims> = token.claims();
            if let Some(audience) = &claims.custom.aud {
                if audience != expected_audience {
                    tracing::error!(
                        "Invalid audience: got {}, expected {}",
                        audience,
                        expected_audience
                    );
                    return Err(ApiError::InvalidJwt);
                }
            } else {
                tracing::error!("Missing audience in token, expected {}", expected_audience);
                return Err(ApiError::InvalidJwt);
            }

            token
        }
        Err(e) => {
            tracing::debug!("ES256K validation failed: {:?}", e);
            return Err(ApiError::InvalidJwt);
        }
    };

    // Return the claims
    Ok(token.claims().custom.clone())
}
