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
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Duration, Utc};
use jwt_compact::{alg::Es256k, prelude::*, AlgorithmExt};
use secp256k1::{All, PublicKey, Secp256k1, SecretKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::db::DBError;
use crate::{ApiError, AppState, VerifiedUserAuthentication};
use jsonwebtoken::{
    encode as jwt_encode, Algorithm as JwtAlgorithm, EncodingKey, Header as JwtHeader,
};
use url::Url;

use crate::models::{platform_users::PlatformUser, users::User};

pub const USER_ACCESS: &str = "access";
pub const USER_REFRESH: &str = "refresh";

pub const PLATFORM_ACCESS: &str = "platform_access";
pub const PLATFORM_REFRESH: &str = "platform_refresh";

pub(crate) const TRANSPORT_V2_USER_ACCESS_AUDIENCE: &str =
    "urn:opensecret:internal:transport-v2:user:access-descriptor";
pub(crate) const TRANSPORT_V2_USER_RESUMPTION_AUDIENCE: &str =
    "urn:opensecret:internal:transport-v2:user:resumption";
pub(crate) const TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE: &str =
    "urn:opensecret:internal:transport-v2:user:native-handoff";
pub(crate) const TRANSPORT_V2_NATIVE_HANDOFF_GRANT_MAX_BYTES: usize = 4_096;
pub(crate) const TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE: &str =
    "urn:opensecret:internal:transport-v2:platform:access-descriptor";
pub(crate) const TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE: &str =
    "urn:opensecret:internal:transport-v2:platform:resumption";
const TRANSPORT_V2_TOKEN_ISSUER: &str = "urn:opensecret:transport-v2";
const TRANSPORT_V2_TOKEN_VERSION: u8 = 2;
const TRANSPORT_V2_NATIVE_HANDOFF_TTL: Duration = Duration::minutes(5);
const TRANSPORT_V2_NATIVE_HANDOFF_CLOCK_LEEWAY: Duration = Duration::seconds(30);
const TRANSPORT_V2_NATIVE_HANDOFF_ATTEMPT_DOMAIN: &[u8] =
    b"opensecret:transport-v2:native-handoff:attempt:v1\0";

pub const USER_TOKEN_FORMAT_V2: u8 = 2;

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
    #[serde(rename = "tf", skip_serializing_if = "Option::is_none")]
    pub token_format: Option<u8>,
    #[serde(rename = "am", skip_serializing_if = "Option::is_none")]
    pub auth_method: Option<String>,
    #[serde(rename = "pid", skip_serializing_if = "Option::is_none")]
    pub project_id: Option<i32>,
    #[serde(rename = "ab", skip_serializing_if = "Option::is_none")]
    pub auth_binding: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMethod {
    Password,
    OAuth,
}

impl AuthMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMethod::Password => "password",
            AuthMethod::OAuth => "oauth",
        }
    }

    fn from_str(value: &str) -> Result<Self, ApiError> {
        match value {
            "password" => Ok(AuthMethod::Password),
            "oauth" => Ok(AuthMethod::OAuth),
            _ => Err(ApiError::InvalidJwt),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthContext {
    pub token_format: u8,
    pub method: AuthMethod,
    pub project_id: i32,
    pub auth_binding: [u8; 32],
}

impl AuthContext {
    pub fn new(method: AuthMethod, project_id: i32, auth_binding: [u8; 32]) -> Self {
        Self {
            token_format: USER_TOKEN_FORMAT_V2,
            method,
            project_id,
            auth_binding,
        }
    }

    pub fn apply_to_claims(&self, claims: &mut CustomClaims) {
        claims.token_format = Some(self.token_format);
        claims.auth_method = Some(self.method.as_str().to_string());
        claims.project_id = Some(self.project_id);
        claims.auth_binding = Some(URL_SAFE_NO_PAD.encode(self.auth_binding));
    }

    pub fn from_claims(claims: &CustomClaims) -> Result<Self, ApiError> {
        if claims.token_format != Some(USER_TOKEN_FORMAT_V2) {
            tracing::error!("Missing or invalid user token format");
            return Err(ApiError::InvalidJwt);
        }

        let method = claims
            .auth_method
            .as_deref()
            .ok_or(ApiError::InvalidJwt)
            .and_then(AuthMethod::from_str)?;

        let project_id = claims.project_id.ok_or(ApiError::InvalidJwt)?;

        let auth_binding_claim = claims.auth_binding.as_ref().ok_or(ApiError::InvalidJwt)?;
        let auth_binding_bytes = URL_SAFE_NO_PAD
            .decode(auth_binding_claim)
            .map_err(|_| ApiError::InvalidJwt)?;
        let auth_binding: [u8; 32] = auth_binding_bytes
            .try_into()
            .map_err(|_| ApiError::InvalidJwt)?;

        Ok(Self {
            token_format: USER_TOKEN_FORMAT_V2,
            method,
            project_id,
            auth_binding,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TransportV2TokenKind {
    AccessDescriptor,
    Resumption,
    NativeHandoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TransportV2PrincipalKind {
    User,
    Platform,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TransportV2UserClaims {
    sub: String,
    iss: String,
    aud: String,
    tv: u8,
    tk: TransportV2TokenKind,
    pk: TransportV2PrincipalKind,
    #[serde(rename = "tf")]
    token_format: u8,
    #[serde(rename = "am")]
    auth_method: String,
    #[serde(rename = "pid")]
    project_id: i32,
    #[serde(rename = "ab")]
    auth_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TransportV2PlatformClaims {
    sub: String,
    iss: String,
    aud: String,
    tv: u8,
    tk: TransportV2TokenKind,
    pk: TransportV2PrincipalKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TransportV2NativeHandoffClaims {
    sub: String,
    iss: String,
    aud: String,
    tv: u8,
    tk: TransportV2TokenKind,
    pk: TransportV2PrincipalKind,
    #[serde(rename = "tf")]
    token_format: u8,
    #[serde(rename = "am")]
    auth_method: String,
    #[serde(rename = "pid")]
    project_id: i32,
    #[serde(rename = "ab")]
    auth_binding: String,
    #[serde(rename = "sid")]
    target_session_id: String,
    #[serde(rename = "jti")]
    native_attempt_commitment: String,
}

pub(crate) struct IssuedTransportV2UserTokens {
    pub(crate) access_token: String,
    pub(crate) resumption_token: String,
    pub(crate) access_expires_at: DateTime<Utc>,
}

pub(crate) struct IssuedTransportV2PlatformTokens {
    pub(crate) access_token: String,
    pub(crate) resumption_token: String,
    pub(crate) access_expires_at: DateTime<Utc>,
}

pub(crate) struct IssuedTransportV2NativeHandoffGrant {
    pub(crate) grant: String,
    pub(crate) expires_at: DateTime<Utc>,
}

impl TokenType {
    pub fn validate_third_party_audience(aud: &str) -> Result<(), ApiError> {
        // Validate third party audience can't use our internal audience types
        const RESERVED_AUDIENCES: [&str; 9] = [
            USER_ACCESS,
            USER_REFRESH,
            PLATFORM_ACCESS,
            PLATFORM_REFRESH,
            TRANSPORT_V2_USER_ACCESS_AUDIENCE,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
            TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE,
            TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE,
            TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE,
        ];

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

        // 5. Reject empty audience values
        if aud.is_empty() {
            tracing::error!("Audience value cannot be empty");
            return Err(ApiError::BadRequest);
        }

        // 6. Parse as URI to ensure it's valid if it contains ':'
        if aud.contains(':') {
            Url::parse(aud).map_err(|e| {
                tracing::error!("Invalid audience URI format: {}, error: {:?}", aud, e);
                ApiError::BadRequest
            })?;
        }

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
            TokenType::Access | TokenType::Refresh => {
                tracing::error!("User access/refresh tokens require AuthContext");
                return Err(ApiError::BadRequest);
            }
        };

        let custom_claims = CustomClaims {
            sub: user.get_id().to_string(),
            aud,
            azp,
            role,
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
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

    pub fn new_with_auth_context(
        user: &User,
        token_type: TokenType,
        app_state: &AppState,
        auth_context: &AuthContext,
    ) -> Result<Self, ApiError> {
        if user.project_id != auth_context.project_id {
            tracing::error!("User token auth context project does not match user project");
            return Err(ApiError::BadRequest);
        }

        let (aud, duration) = match &token_type {
            TokenType::Access => (
                Some(USER_ACCESS.to_string()),
                Duration::minutes(app_state.config.access_token_maxage),
            ),
            TokenType::Refresh => (
                Some(USER_REFRESH.to_string()),
                Duration::days(app_state.config.refresh_token_maxage),
            ),
            TokenType::ThirdParty { .. } => {
                return Err(ApiError::BadRequest);
            }
        };

        let mut custom_claims = CustomClaims {
            sub: user.get_id().to_string(),
            aud,
            azp: None,
            role: None,
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
        };
        auth_context.apply_to_claims(&mut custom_claims);

        tracing::debug!(
            "Creating new v2 user token for user {} with audience {:?}",
            user.get_id(),
            custom_claims.aud
        );

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
                tracing::error!("Error creating v2 user token: {:?}", e);
                ApiError::InternalServerError
            })?;

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
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
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

pub(crate) fn issue_transport_v2_user_tokens(
    user: &User,
    auth_context: &AuthContext,
    app_state: &AppState,
) -> Result<IssuedTransportV2UserTokens, ApiError> {
    if user.project_id != auth_context.project_id {
        tracing::error!("Transport-v2 user token auth context project mismatch");
        return Err(ApiError::BadRequest);
    }

    let (access_token, access_expires_at) = issue_transport_v2_user_token(
        user,
        auth_context,
        app_state,
        TransportV2TokenKind::AccessDescriptor,
        TRANSPORT_V2_USER_ACCESS_AUDIENCE,
        Duration::minutes(app_state.config.access_token_maxage),
    )?;
    let (resumption_token, _) = issue_transport_v2_user_token(
        user,
        auth_context,
        app_state,
        TransportV2TokenKind::Resumption,
        TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        Duration::days(app_state.config.refresh_token_maxage),
    )?;

    Ok(IssuedTransportV2UserTokens {
        access_token,
        resumption_token,
        access_expires_at,
    })
}

fn issue_transport_v2_user_token(
    user: &User,
    auth_context: &AuthContext,
    app_state: &AppState,
    token_kind: TransportV2TokenKind,
    audience: &str,
    duration: Duration,
) -> Result<(String, DateTime<Utc>), ApiError> {
    let custom_claims = TransportV2UserClaims {
        sub: user.get_id().to_string(),
        iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
        aud: audience.to_owned(),
        tv: TRANSPORT_V2_TOKEN_VERSION,
        tk: token_kind,
        pk: TransportV2PrincipalKind::User,
        token_format: auth_context.token_format,
        auth_method: auth_context.method.as_str().to_owned(),
        project_id: auth_context.project_id,
        auth_binding: URL_SAFE_NO_PAD.encode(auth_context.auth_binding),
    };

    // Match the existing first-party clock-skew policy without changing the
    // v1 token constructors.
    let issued_at = Utc::now() - Duration::minutes(1);
    let expiration = issued_at + duration;
    let mut claims = Claims::new(custom_claims);
    claims.issued_at = Some(issued_at);
    claims.not_before = Some(issued_at);
    claims.expiration = Some(expiration);

    let header = Header::empty().with_token_type("JWT");
    let es256k = Es256k::<Sha256>::new(app_state.config.jwt_keys.secp.clone());
    let token = es256k
        .token(&header, &claims, &app_state.config.jwt_keys.signing_key)
        .map_err(|error| {
            tracing::error!("Error creating transport-v2 user token: {:?}", error);
            ApiError::InternalServerError
        })?;

    Ok((token, expiration))
}

pub(crate) fn issue_transport_v2_native_handoff_grant(
    user: &User,
    auth_context: &AuthContext,
    target_session_id: Uuid,
    native_attempt_id: Uuid,
    app_state: &AppState,
) -> Result<IssuedTransportV2NativeHandoffGrant, ApiError> {
    if user.project_id != auth_context.project_id {
        return Err(ApiError::BadRequest);
    }

    issue_transport_v2_native_handoff_grant_with_keys(
        user.get_id(),
        auth_context,
        target_session_id,
        native_attempt_id,
        &app_state.config.jwt_keys,
    )
}

fn issue_transport_v2_native_handoff_grant_with_keys(
    user_id: Uuid,
    auth_context: &AuthContext,
    target_session_id: Uuid,
    native_attempt_id: Uuid,
    jwt_keys: &JwtKeys,
) -> Result<IssuedTransportV2NativeHandoffGrant, ApiError> {
    if target_session_id.is_nil() || native_attempt_id.is_nil() {
        return Err(ApiError::BadRequest);
    }

    let custom_claims = TransportV2NativeHandoffClaims {
        sub: user_id.to_string(),
        iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
        aud: TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE.to_owned(),
        tv: TRANSPORT_V2_TOKEN_VERSION,
        tk: TransportV2TokenKind::NativeHandoff,
        pk: TransportV2PrincipalKind::User,
        token_format: auth_context.token_format,
        auth_method: auth_context.method.as_str().to_owned(),
        project_id: auth_context.project_id,
        auth_binding: URL_SAFE_NO_PAD.encode(auth_context.auth_binding),
        target_session_id: target_session_id.to_string(),
        native_attempt_commitment: native_handoff_attempt_commitment(native_attempt_id),
    };

    let issued_at = Utc::now();
    let expiration = issued_at + TRANSPORT_V2_NATIVE_HANDOFF_TTL;
    let mut claims = Claims::new(custom_claims);
    claims.issued_at = Some(issued_at);
    claims.not_before = Some(issued_at);
    claims.expiration = Some(expiration);

    let header = Header::empty().with_token_type("JWT");
    let es256k = Es256k::<Sha256>::new(jwt_keys.secp.clone());
    let grant = es256k
        .token(&header, &claims, &jwt_keys.signing_key)
        .map_err(|error| {
            tracing::error!("Error creating transport-v2 native handoff grant: {error:?}");
            ApiError::InternalServerError
        })?;
    if !is_canonical_compact_jwt(&grant) {
        tracing::error!("Issued transport-v2 native handoff grant exceeded its wire contract");
        return Err(ApiError::InternalServerError);
    }

    Ok(IssuedTransportV2NativeHandoffGrant {
        grant,
        expires_at: expiration,
    })
}

#[cfg(test)]
pub(crate) fn issue_transport_v2_native_handoff_grant_for_test(
    user_id: Uuid,
    auth_context: &AuthContext,
    target_session_id: Uuid,
    native_attempt_id: Uuid,
    jwt_keys: &JwtKeys,
) -> Result<IssuedTransportV2NativeHandoffGrant, ApiError> {
    issue_transport_v2_native_handoff_grant_with_keys(
        user_id,
        auth_context,
        target_session_id,
        native_attempt_id,
        jwt_keys,
    )
}

pub(crate) fn issue_transport_v2_platform_tokens(
    platform_user: &PlatformUser,
    app_state: &AppState,
) -> Result<IssuedTransportV2PlatformTokens, ApiError> {
    let (access_token, access_expires_at) = issue_transport_v2_platform_token(
        platform_user,
        app_state,
        TransportV2TokenKind::AccessDescriptor,
        TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE,
        Duration::minutes(app_state.config.access_token_maxage),
    )?;
    let (resumption_token, _) = issue_transport_v2_platform_token(
        platform_user,
        app_state,
        TransportV2TokenKind::Resumption,
        TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE,
        Duration::days(app_state.config.refresh_token_maxage),
    )?;

    Ok(IssuedTransportV2PlatformTokens {
        access_token,
        resumption_token,
        access_expires_at,
    })
}

fn issue_transport_v2_platform_token(
    platform_user: &PlatformUser,
    app_state: &AppState,
    token_kind: TransportV2TokenKind,
    audience: &str,
    duration: Duration,
) -> Result<(String, DateTime<Utc>), ApiError> {
    let custom_claims = TransportV2PlatformClaims {
        sub: platform_user.uuid.to_string(),
        iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
        aud: audience.to_owned(),
        tv: TRANSPORT_V2_TOKEN_VERSION,
        tk: token_kind,
        pk: TransportV2PrincipalKind::Platform,
    };

    let issued_at = Utc::now() - Duration::minutes(1);
    let expiration = issued_at + duration;
    let mut claims = Claims::new(custom_claims);
    claims.issued_at = Some(issued_at);
    claims.not_before = Some(issued_at);
    claims.expiration = Some(expiration);

    let header = Header::empty().with_token_type("JWT");
    let es256k = Es256k::<Sha256>::new(app_state.config.jwt_keys.secp.clone());
    let token = es256k
        .token(&header, &claims, &app_state.config.jwt_keys.signing_key)
        .map_err(|error| {
            tracing::error!("Error creating transport-v2 platform token: {:?}", error);
            ApiError::InternalServerError
        })?;

    Ok((token, expiration))
}

pub(crate) fn validate_transport_v2_user_resumption(
    original_token: &str,
    data: &AppState,
) -> Result<VerifiedUserAuthentication, ApiError> {
    let claims =
        validate_transport_v2_user_resumption_claims(original_token, &data.config.jwt_keys)?;
    let user_id = Uuid::parse_str(&claims.sub).map_err(|_| ApiError::InvalidJwt)?;
    let auth_context = transport_v2_auth_context(&claims)?;
    let user = match data.verify_bound_user(user_id, claims.project_id, &auth_context) {
        Ok(user) => user,
        Err(ApiError::InternalServerError) => return Err(ApiError::InternalServerError),
        Err(_) => return Err(ApiError::InvalidJwt),
    };

    Ok(VerifiedUserAuthentication { user, auth_context })
}

pub(crate) fn validate_transport_v2_native_handoff_grant(
    original_token: &str,
    expected_session_id: Uuid,
    expected_attempt_id: Uuid,
    data: &AppState,
) -> Result<VerifiedUserAuthentication, ApiError> {
    if expected_session_id.is_nil() || expected_attempt_id.is_nil() {
        return Err(ApiError::InvalidJwt);
    }
    let claims = validate_transport_v2_native_handoff_grant_claims(
        original_token,
        &data.config.jwt_keys,
        expected_session_id,
        expected_attempt_id,
    )?;
    let user_id = parse_canonical_non_nil_uuid(&claims.sub)?;
    let auth_context = transport_v2_auth_context_from_parts(
        claims.token_format,
        &claims.auth_method,
        claims.project_id,
        &claims.auth_binding,
    )?;
    let user = match data.verify_bound_user(user_id, claims.project_id, &auth_context) {
        Ok(user) => user,
        Err(ApiError::InternalServerError) => return Err(ApiError::InternalServerError),
        Err(_) => return Err(ApiError::InvalidJwt),
    };

    Ok(VerifiedUserAuthentication { user, auth_context })
}

pub(crate) fn validate_transport_v2_platform_resumption(
    original_token: &str,
    data: &AppState,
) -> Result<PlatformUser, ApiError> {
    let claims =
        validate_transport_v2_platform_resumption_claims(original_token, &data.config.jwt_keys)?;
    let platform_user_id = Uuid::parse_str(&claims.sub).map_err(|_| ApiError::InvalidJwt)?;
    data.db
        .get_platform_user_by_uuid(platform_user_id)
        .map_err(|error| match error {
            DBError::PlatformUserNotFound => ApiError::InvalidJwt,
            _ => {
                tracing::error!(
                    "Failed to load transport-v2 platform resumption principal: {:?}",
                    error
                );
                ApiError::InternalServerError
            }
        })
}

fn validate_transport_v2_platform_resumption_claims(
    original_token: &str,
    jwt_keys: &JwtKeys,
) -> Result<TransportV2PlatformClaims, ApiError> {
    let parsed_token = UntrustedToken::new(original_token).map_err(|error| {
        tracing::error!(
            "Failed to parse transport-v2 platform resumption token: {:?}",
            error
        );
        ApiError::InvalidJwt
    })?;
    let es256k = Es256k::<Sha256>::new(jwt_keys.secp.clone());
    let public_key = jwt_keys.public_key();
    let token: Token<TransportV2PlatformClaims> = es256k
        .validator(&public_key)
        .validate(&parsed_token)
        .map_err(|error| {
            tracing::debug!(
                "Transport-v2 platform resumption signature validation failed: {:?}",
                error
            );
            ApiError::InvalidJwt
        })?;
    let claims = token.claims();

    if claims.custom.iss != TRANSPORT_V2_TOKEN_ISSUER
        || claims.custom.aud != TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE
        || claims.custom.tv != TRANSPORT_V2_TOKEN_VERSION
        || claims.custom.tk != TransportV2TokenKind::Resumption
        || claims.custom.pk != TransportV2PrincipalKind::Platform
    {
        return Err(ApiError::InvalidJwt);
    }

    validate_transport_v2_token_times(claims, "platform")?;
    Ok(claims.custom.clone())
}

fn validate_transport_v2_user_resumption_claims(
    original_token: &str,
    jwt_keys: &JwtKeys,
) -> Result<TransportV2UserClaims, ApiError> {
    let parsed_token = UntrustedToken::new(original_token).map_err(|error| {
        tracing::error!("Failed to parse transport-v2 resumption token: {:?}", error);
        ApiError::InvalidJwt
    })?;
    let es256k = Es256k::<Sha256>::new(jwt_keys.secp.clone());
    let public_key = jwt_keys.public_key();
    let token: Token<TransportV2UserClaims> = es256k
        .validator(&public_key)
        .validate(&parsed_token)
        .map_err(|error| {
            tracing::debug!(
                "Transport-v2 resumption signature validation failed: {:?}",
                error
            );
            ApiError::InvalidJwt
        })?;
    let claims = token.claims();

    if claims.custom.iss != TRANSPORT_V2_TOKEN_ISSUER
        || claims.custom.aud != TRANSPORT_V2_USER_RESUMPTION_AUDIENCE
        || claims.custom.tv != TRANSPORT_V2_TOKEN_VERSION
        || claims.custom.tk != TransportV2TokenKind::Resumption
        || claims.custom.pk != TransportV2PrincipalKind::User
    {
        return Err(ApiError::InvalidJwt);
    }

    validate_transport_v2_token_times(claims, "user")?;

    Ok(claims.custom.clone())
}

fn validate_transport_v2_native_handoff_grant_claims(
    original_token: &str,
    jwt_keys: &JwtKeys,
    expected_session_id: Uuid,
    expected_attempt_id: Uuid,
) -> Result<TransportV2NativeHandoffClaims, ApiError> {
    if !is_canonical_compact_jwt(original_token) {
        return Err(ApiError::InvalidJwt);
    }
    let parsed_token = UntrustedToken::new(original_token).map_err(|error| {
        tracing::debug!("Failed to parse transport-v2 native handoff grant: {error:?}");
        ApiError::InvalidJwt
    })?;
    let es256k = Es256k::<Sha256>::new(jwt_keys.secp.clone());
    let public_key = jwt_keys.public_key();
    let token: Token<TransportV2NativeHandoffClaims> = es256k
        .validator(&public_key)
        .validate(&parsed_token)
        .map_err(|error| {
            tracing::debug!(
                "Transport-v2 native handoff grant signature validation failed: {error:?}"
            );
            ApiError::InvalidJwt
        })?;
    let claims = token.claims();

    if claims.custom.iss != TRANSPORT_V2_TOKEN_ISSUER
        || claims.custom.aud != TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE
        || claims.custom.tv != TRANSPORT_V2_TOKEN_VERSION
        || claims.custom.tk != TransportV2TokenKind::NativeHandoff
        || claims.custom.pk != TransportV2PrincipalKind::User
    {
        return Err(ApiError::InvalidJwt);
    }

    validate_transport_v2_native_handoff_times(claims)?;
    let target_session_id = parse_canonical_non_nil_uuid(&claims.custom.target_session_id)?;
    if target_session_id != expected_session_id
        || claims.custom.native_attempt_commitment
            != native_handoff_attempt_commitment(expected_attempt_id)
    {
        return Err(ApiError::InvalidJwt);
    }

    Ok(claims.custom.clone())
}

#[cfg(test)]
pub(crate) fn validate_transport_v2_native_handoff_grant_claims_for_test(
    original_token: &str,
    expected_session_id: Uuid,
    expected_attempt_id: Uuid,
    jwt_keys: &JwtKeys,
) -> Result<(Uuid, AuthContext), ApiError> {
    let claims = validate_transport_v2_native_handoff_grant_claims(
        original_token,
        jwt_keys,
        expected_session_id,
        expected_attempt_id,
    )?;
    let user_id = parse_canonical_non_nil_uuid(&claims.sub)?;
    let auth_context = transport_v2_auth_context_from_parts(
        claims.token_format,
        &claims.auth_method,
        claims.project_id,
        &claims.auth_binding,
    )?;
    Ok((user_id, auth_context))
}

fn validate_transport_v2_native_handoff_times<T>(claims: &Claims<T>) -> Result<(), ApiError> {
    let time_options = TimeOptions::from_leeway(TRANSPORT_V2_NATIVE_HANDOFF_CLOCK_LEEWAY);
    claims
        .validate_expiration(&time_options)
        .and_then(|claims| claims.validate_maturity(&time_options))
        .map_err(|error| {
            tracing::debug!("Transport-v2 native handoff time validation failed: {error:?}");
            ApiError::InvalidJwt
        })?;

    let issued_at = claims.issued_at.ok_or(ApiError::InvalidJwt)?;
    let not_before = claims.not_before.ok_or(ApiError::InvalidJwt)?;
    let expiration = claims.expiration.ok_or(ApiError::InvalidJwt)?;
    let latest_acceptable_issuance = Utc::now()
        .checked_add_signed(TRANSPORT_V2_NATIVE_HANDOFF_CLOCK_LEEWAY)
        .ok_or(ApiError::InvalidJwt)?;
    if issued_at != not_before
        || issued_at > latest_acceptable_issuance
        || expiration <= issued_at
        || expiration.signed_duration_since(issued_at) > TRANSPORT_V2_NATIVE_HANDOFF_TTL
    {
        return Err(ApiError::InvalidJwt);
    }
    Ok(())
}

fn parse_canonical_non_nil_uuid(value: &str) -> Result<Uuid, ApiError> {
    let parsed = Uuid::parse_str(value).map_err(|_| ApiError::InvalidJwt)?;
    if parsed.is_nil() || parsed.to_string() != value {
        return Err(ApiError::InvalidJwt);
    }
    Ok(parsed)
}

fn native_handoff_attempt_commitment(native_attempt_id: Uuid) -> String {
    let mut digest = Sha256::new();
    digest.update(TRANSPORT_V2_NATIVE_HANDOFF_ATTEMPT_DOMAIN);
    digest.update(native_attempt_id.as_bytes());
    URL_SAFE_NO_PAD.encode(digest.finalize())
}

fn is_canonical_compact_jwt(value: &str) -> bool {
    if value.is_empty() || value.len() > TRANSPORT_V2_NATIVE_HANDOFF_GRANT_MAX_BYTES {
        return false;
    }
    let mut segments = value.split('.');
    let canonical_segment = |segment: &str| {
        if segment.is_empty()
            || !segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return false;
        }
        URL_SAFE_NO_PAD
            .decode(segment)
            .is_ok_and(|decoded| URL_SAFE_NO_PAD.encode(decoded) == segment)
    };
    matches!(
        (segments.next(), segments.next(), segments.next(), segments.next()),
        (Some(header), Some(payload), Some(signature), None)
            if canonical_segment(header)
                && canonical_segment(payload)
                && canonical_segment(signature)
    )
}

fn validate_transport_v2_token_times<T>(
    claims: &Claims<T>,
    principal_label: &'static str,
) -> Result<(), ApiError> {
    let time_options = TimeOptions::default();
    claims
        .validate_expiration(&time_options)
        .and_then(|claims| claims.validate_maturity(&time_options))
        .map_err(|error| {
            tracing::error!(
                principal = principal_label,
                "Transport-v2 resumption time validation failed: {:?}",
                error
            );
            ApiError::InvalidJwt
        })?;

    let issued_at = claims.issued_at.ok_or(ApiError::InvalidJwt)?;
    let not_before = claims.not_before.ok_or(ApiError::InvalidJwt)?;
    let expiration = claims.expiration.ok_or(ApiError::InvalidJwt)?;
    let latest_acceptable_issuance = Utc::now()
        .checked_add_signed(time_options.leeway)
        .ok_or(ApiError::InvalidJwt)?;
    if issued_at > latest_acceptable_issuance || issued_at > not_before || not_before >= expiration
    {
        return Err(ApiError::InvalidJwt);
    }
    Ok(())
}

fn transport_v2_auth_context(claims: &TransportV2UserClaims) -> Result<AuthContext, ApiError> {
    transport_v2_auth_context_from_parts(
        claims.token_format,
        &claims.auth_method,
        claims.project_id,
        &claims.auth_binding,
    )
}

fn transport_v2_auth_context_from_parts(
    token_format: u8,
    auth_method: &str,
    project_id: i32,
    auth_binding: &str,
) -> Result<AuthContext, ApiError> {
    if token_format != USER_TOKEN_FORMAT_V2 {
        return Err(ApiError::InvalidJwt);
    }
    let method = AuthMethod::from_str(auth_method)?;
    let auth_binding_bytes = URL_SAFE_NO_PAD
        .decode(auth_binding)
        .map_err(|_| ApiError::InvalidJwt)?;
    let auth_binding: [u8; 32] = auth_binding_bytes
        .try_into()
        .map_err(|_| ApiError::InvalidJwt)?;
    Ok(AuthContext {
        token_format: USER_TOKEN_FORMAT_V2,
        method,
        project_id,
        auth_binding,
    })
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

    let (claims, access_token_expired) =
        match validate_access_token_for_auth(&token, &data, USER_ACCESS) {
            Ok(validation) => validation,
            Err(_) => return ApiError::InvalidJwt.into_response(),
        };

    let auth_context = match AuthContext::from_claims(&claims) {
        Ok(auth_context) => auth_context,
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

    if user.project_id != auth_context.project_id {
        tracing::error!("JWT auth context project does not match user project");
        return ApiError::InvalidJwt.into_response();
    }

    if let Err(e) = data.verify_seed_wrap_for_auth_context(&user, &auth_context) {
        tracing::error!(
            "JWT auth context no longer unwraps an active seed wrap: {:?}",
            e
        );
        return ApiError::InvalidJwt.into_response();
    }

    if access_token_expired {
        return ApiError::AccessTokenExpired.into_response();
    }

    req.extensions_mut().insert(auth_context);
    req.extensions_mut().insert(user);
    next.run(req).await
}

pub async fn validate_platform_jwt(
    State(data): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> impl IntoResponse {
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

    let (claims, access_token_expired) =
        match validate_access_token_for_auth(&token, &data, PLATFORM_ACCESS) {
            Ok(validation) => validation,
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

    if access_token_expired {
        return ApiError::AccessTokenExpired.into_response();
    }

    req.extensions_mut().insert(platform_user);
    next.run(req).await
}

pub(crate) fn validate_token(
    original_token: &str,
    data: &AppState,
    expected_audience: &str,
) -> Result<CustomClaims, ApiError> {
    validate_token_with_keys(original_token, &data.config.jwt_keys, expected_audience)
}

pub(crate) fn validate_access_token_for_auth(
    original_token: &str,
    data: &AppState,
    expected_audience: &str,
) -> Result<(CustomClaims, bool), ApiError> {
    if !is_access_token_audience(expected_audience) {
        return Err(ApiError::InvalidJwt);
    }
    validate_token_with_keys_for_auth(original_token, &data.config.jwt_keys, expected_audience)
}

fn validate_token_with_keys(
    original_token: &str,
    jwt_keys: &JwtKeys,
    expected_audience: &str,
) -> Result<CustomClaims, ApiError> {
    let (claims, access_token_expired) =
        validate_token_with_keys_for_auth(original_token, jwt_keys, expected_audience)?;
    if access_token_expired {
        Err(ApiError::AccessTokenExpired)
    } else {
        Ok(claims)
    }
}

fn is_access_token_audience(audience: &str) -> bool {
    audience == USER_ACCESS || audience == PLATFORM_ACCESS
}

fn validate_token_with_keys_for_auth(
    original_token: &str,
    jwt_keys: &JwtKeys,
    expected_audience: &str,
) -> Result<(CustomClaims, bool), ApiError> {
    // Try ES256K first
    let es256k = Es256k::<Sha256>::new(jwt_keys.secp.clone());
    let public_key = jwt_keys.public_key();

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
    let (token, access_token_expired): (Token<CustomClaims>, bool) =
        match es256k.validator(&public_key).validate(&parsed_token) {
            Ok(token) => {
                tracing::trace!("ES256K signature validation successful");

                // Validate the audience before classifying expiration. This ensures
                // an expired token for another audience is never a refresh signal.
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

                // Only validate expiration, not maturity. Access-token expiry is
                // carried to the authentication middleware so identity and auth
                // binding checks still run before a refresh signal is emitted.
                let time_options = TimeOptions::default();
                let access_token_expired = match token.claims().validate_expiration(&time_options) {
                    Ok(_) => false,
                    Err(jwt_compact::ValidationError::Expired)
                        if is_access_token_audience(expected_audience) =>
                    {
                        true
                    }
                    Err(e) => {
                        tracing::error!("Token expiration validation failed: {:?}", e);
                        return Err(ApiError::InvalidJwt);
                    }
                };

                (token, access_token_expired)
            }
            Err(e) => {
                tracing::debug!("ES256K validation failed: {:?}", e);
                return Err(ApiError::InvalidJwt);
            }
        };

    Ok((token.claims().custom.clone(), access_token_expired))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{decode as jwt_decode, DecodingKey, Validation};
    use serde::Serialize;

    fn test_keys(byte: u8) -> JwtKeys {
        JwtKeys::new(vec![byte; 32]).unwrap()
    }

    fn signed_test_token(
        keys: &JwtKeys,
        audience: &str,
        expiration: Option<chrono::DateTime<Utc>>,
    ) -> String {
        signed_test_token_with_auth_binding(
            keys,
            audience,
            expiration,
            Some(URL_SAFE_NO_PAD.encode([7u8; 32])),
        )
    }

    fn signed_test_token_with_auth_binding(
        keys: &JwtKeys,
        audience: &str,
        expiration: Option<chrono::DateTime<Utc>>,
        auth_binding: Option<String>,
    ) -> String {
        let now = Utc::now();
        let mut claims = Claims::new(CustomClaims {
            sub: Uuid::nil().to_string(),
            aud: Some(audience.to_string()),
            azp: None,
            role: None,
            token_format: Some(USER_TOKEN_FORMAT_V2),
            auth_method: Some(AuthMethod::Password.as_str().to_string()),
            project_id: Some(1),
            auth_binding,
        });
        claims.issued_at = Some(now - Duration::minutes(2));
        claims.not_before = Some(now - Duration::minutes(2));
        claims.expiration = expiration;

        Es256k::<Sha256>::new(keys.secp.clone())
            .token(
                &Header::empty().with_token_type("JWT"),
                &claims,
                &keys.signing_key,
            )
            .unwrap()
    }

    fn transport_v2_test_claims(
        token_kind: TransportV2TokenKind,
        audience: &str,
    ) -> TransportV2UserClaims {
        TransportV2UserClaims {
            sub: Uuid::nil().to_string(),
            iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
            aud: audience.to_owned(),
            tv: TRANSPORT_V2_TOKEN_VERSION,
            tk: token_kind,
            pk: TransportV2PrincipalKind::User,
            token_format: USER_TOKEN_FORMAT_V2,
            auth_method: AuthMethod::Password.as_str().to_owned(),
            project_id: 1,
            auth_binding: URL_SAFE_NO_PAD.encode([7_u8; 32]),
        }
    }

    fn signed_transport_v2_test_token<T: Serialize>(
        keys: &JwtKeys,
        custom: T,
        issued_at: Option<DateTime<Utc>>,
        not_before: Option<DateTime<Utc>>,
        expiration: Option<DateTime<Utc>>,
    ) -> String {
        let mut claims = Claims::new(custom);
        claims.issued_at = issued_at;
        claims.not_before = not_before;
        claims.expiration = expiration;
        Es256k::<Sha256>::new(keys.secp.clone())
            .token(
                &Header::empty().with_token_type("JWT"),
                &claims,
                &keys.signing_key,
            )
            .unwrap()
    }

    fn valid_transport_v2_resumption(keys: &JwtKeys) -> String {
        let now = Utc::now();
        signed_transport_v2_test_token(
            keys,
            transport_v2_test_claims(
                TransportV2TokenKind::Resumption,
                TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
            ),
            Some(now - Duration::minutes(2)),
            Some(now - Duration::minutes(2)),
            Some(now + Duration::minutes(5)),
        )
    }

    fn transport_v2_platform_test_claims(
        token_kind: TransportV2TokenKind,
        audience: &str,
    ) -> TransportV2PlatformClaims {
        TransportV2PlatformClaims {
            sub: Uuid::nil().to_string(),
            iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
            aud: audience.to_owned(),
            tv: TRANSPORT_V2_TOKEN_VERSION,
            tk: token_kind,
            pk: TransportV2PrincipalKind::Platform,
        }
    }

    fn valid_transport_v2_platform_resumption(keys: &JwtKeys) -> String {
        let now = Utc::now();
        signed_transport_v2_test_token(
            keys,
            transport_v2_platform_test_claims(
                TransportV2TokenKind::Resumption,
                TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE,
            ),
            Some(now - Duration::minutes(2)),
            Some(now - Duration::minutes(2)),
            Some(now + Duration::minutes(5)),
        )
    }

    fn transport_v2_native_handoff_test_claims(
        target_session_id: Uuid,
        native_attempt_id: Uuid,
    ) -> TransportV2NativeHandoffClaims {
        TransportV2NativeHandoffClaims {
            sub: Uuid::from_bytes([0x41; 16]).to_string(),
            iss: TRANSPORT_V2_TOKEN_ISSUER.to_owned(),
            aud: TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE.to_owned(),
            tv: TRANSPORT_V2_TOKEN_VERSION,
            tk: TransportV2TokenKind::NativeHandoff,
            pk: TransportV2PrincipalKind::User,
            token_format: USER_TOKEN_FORMAT_V2,
            auth_method: AuthMethod::OAuth.as_str().to_owned(),
            project_id: 7,
            auth_binding: URL_SAFE_NO_PAD.encode([0x42; 32]),
            target_session_id: target_session_id.to_string(),
            native_attempt_commitment: native_handoff_attempt_commitment(native_attempt_id),
        }
    }

    #[test]
    fn expired_user_access_token_is_recoverable_only_after_signature_and_audience_validation() {
        let trusted_keys = test_keys(1);
        let other_keys = test_keys(2);
        let expired = Utc::now() - Duration::minutes(1);

        let expired_access = signed_test_token(&trusted_keys, USER_ACCESS, Some(expired));
        assert!(matches!(
            validate_token_with_keys(&expired_access, &trusted_keys, USER_ACCESS),
            Err(ApiError::AccessTokenExpired)
        ));

        let wrong_audience = signed_test_token(&trusted_keys, USER_REFRESH, Some(expired));
        assert!(matches!(
            validate_token_with_keys(&wrong_audience, &trusted_keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));

        let wrong_signature = signed_test_token(&other_keys, USER_ACCESS, Some(expired));
        assert!(matches!(
            validate_token_with_keys(&wrong_signature, &trusted_keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));
    }

    #[test]
    fn expired_access_tokens_are_recoverable_but_refresh_tokens_are_not() {
        let keys = test_keys(3);
        let expired = Utc::now() - Duration::minutes(1);

        let refresh = signed_test_token(&keys, USER_REFRESH, Some(expired));
        assert!(matches!(
            validate_token_with_keys(&refresh, &keys, USER_REFRESH),
            Err(ApiError::InvalidJwt)
        ));

        let platform_access = signed_test_token(&keys, PLATFORM_ACCESS, Some(expired));
        assert!(matches!(
            validate_token_with_keys(&platform_access, &keys, PLATFORM_ACCESS),
            Err(ApiError::AccessTokenExpired)
        ));
    }

    #[test]
    fn expired_access_claims_are_available_for_auth_binding_validation() {
        let keys = test_keys(5);
        let malformed_binding = URL_SAFE_NO_PAD.encode([7u8; 31]);
        let token = signed_test_token_with_auth_binding(
            &keys,
            USER_ACCESS,
            Some(Utc::now() - Duration::minutes(1)),
            Some(malformed_binding),
        );

        let (claims, access_token_expired) =
            validate_token_with_keys_for_auth(&token, &keys, USER_ACCESS).unwrap();
        assert!(access_token_expired);
        assert!(matches!(
            AuthContext::from_claims(&claims),
            Err(ApiError::InvalidJwt)
        ));
    }

    #[test]
    fn valid_access_and_malformed_tokens_do_not_report_expiration() {
        let keys = test_keys(4);
        let valid = signed_test_token(&keys, USER_ACCESS, Some(Utc::now() + Duration::minutes(5)));
        let missing_expiration = signed_test_token(&keys, USER_ACCESS, None);

        assert!(validate_token_with_keys(&valid, &keys, USER_ACCESS).is_ok());
        assert!(matches!(
            validate_token_with_keys(&missing_expiration, &keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));
        assert!(matches!(
            validate_token_with_keys("not-a-jwt", &keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));
    }

    #[test]
    fn test_jsonwebtoken_hs256_round_trip() {
        let now = Utc::now();
        let claims = {
            let mut claims = Claims::new(CustomClaims {
                sub: "user-id".to_string(),
                aud: Some("https://example.com".to_string()),
                azp: Some(Uuid::nil().to_string()),
                role: Some("authenticated".to_string()),
                token_format: None,
                auth_method: None,
                project_id: None,
                auth_binding: None,
            });
            claims.issued_at = Some(now);
            claims.not_before = Some(now);
            claims.expiration = Some(now + Duration::minutes(5));
            claims
        };

        let token = jwt_encode(
            &JwtHeader::new(JwtAlgorithm::HS256),
            &claims,
            &EncodingKey::from_secret(b"super-secret"),
        )
        .expect("token should encode");

        let mut validation = Validation::new(JwtAlgorithm::HS256);
        validation.set_audience(&["https://example.com"]);

        let decoded = jwt_decode::<Claims<CustomClaims>>(
            &token,
            &DecodingKey::from_secret(b"super-secret"),
            &validation,
        )
        .expect("token should decode");

        assert_eq!(decoded.claims.custom, claims.custom);
    }

    #[test]
    fn auth_context_round_trips_through_custom_claims() {
        let auth_context = AuthContext::new(AuthMethod::Password, 7, [3u8; 32]);
        let mut claims = CustomClaims {
            sub: Uuid::nil().to_string(),
            aud: Some(USER_ACCESS.to_string()),
            azp: None,
            role: None,
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
        };

        auth_context.apply_to_claims(&mut claims);
        let parsed = AuthContext::from_claims(&claims).unwrap();

        assert_eq!(auth_context, parsed);
        let expected_binding = URL_SAFE_NO_PAD.encode([3u8; 32]);
        assert_eq!(
            claims.auth_binding.as_deref(),
            Some(expected_binding.as_str())
        );
    }

    #[test]
    fn auth_context_rejects_legacy_user_access_and_refresh_claim_shapes() {
        for audience in [USER_ACCESS, USER_REFRESH] {
            let claims = CustomClaims {
                sub: Uuid::nil().to_string(),
                aud: Some(audience.to_string()),
                azp: None,
                role: None,
                token_format: None,
                auth_method: None,
                project_id: None,
                auth_binding: None,
            };

            assert!(
                AuthContext::from_claims(&claims).is_err(),
                "legacy {audience} claims without auth context must be rejected"
            );
        }
    }

    #[test]
    fn auth_context_rejects_bad_method() {
        let mut claims = CustomClaims {
            sub: Uuid::nil().to_string(),
            aud: Some(USER_ACCESS.to_string()),
            azp: None,
            role: None,
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
        };
        AuthContext::new(AuthMethod::OAuth, 7, [4u8; 32]).apply_to_claims(&mut claims);
        claims.auth_method = Some("api_key".to_string());

        assert!(AuthContext::from_claims(&claims).is_err());
    }

    #[test]
    fn auth_context_rejects_wrong_binding_length() {
        let mut claims = CustomClaims {
            sub: Uuid::nil().to_string(),
            aud: Some(USER_ACCESS.to_string()),
            azp: None,
            role: None,
            token_format: None,
            auth_method: None,
            project_id: None,
            auth_binding: None,
        };
        AuthContext::new(AuthMethod::Password, 7, [5u8; 32]).apply_to_claims(&mut claims);
        claims.auth_binding = Some(URL_SAFE_NO_PAD.encode([5u8; 31]));

        assert!(AuthContext::from_claims(&claims).is_err());
    }

    #[test]
    fn transport_v2_tokens_are_cryptographically_separate_from_v1_tokens() {
        let keys = test_keys(6);
        let now = Utc::now();
        let resumption = valid_transport_v2_resumption(&keys);
        assert!(validate_transport_v2_user_resumption_claims(&resumption, &keys).is_ok());
        assert!(matches!(
            validate_token_with_keys(&resumption, &keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));
        assert!(matches!(
            validate_token_with_keys(&resumption, &keys, USER_REFRESH),
            Err(ApiError::InvalidJwt)
        ));

        let descriptor = signed_transport_v2_test_token(
            &keys,
            transport_v2_test_claims(
                TransportV2TokenKind::AccessDescriptor,
                TRANSPORT_V2_USER_ACCESS_AUDIENCE,
            ),
            Some(now - Duration::minutes(2)),
            Some(now - Duration::minutes(2)),
            Some(now + Duration::minutes(5)),
        );
        assert!(matches!(
            validate_transport_v2_user_resumption_claims(&descriptor, &keys),
            Err(ApiError::InvalidJwt)
        ));
        assert!(matches!(
            validate_token_with_keys(&descriptor, &keys, USER_ACCESS),
            Err(ApiError::InvalidJwt)
        ));

        for audience in [USER_ACCESS, USER_REFRESH] {
            let legacy = signed_test_token(&keys, audience, Some(now + Duration::minutes(5)));
            assert!(matches!(
                validate_transport_v2_user_resumption_claims(&legacy, &keys),
                Err(ApiError::InvalidJwt)
            ));
        }

        for audience in [
            TRANSPORT_V2_USER_ACCESS_AUDIENCE,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        ] {
            assert!(matches!(
                TokenType::validate_third_party_audience(audience),
                Err(ApiError::BadRequest)
            ));
        }
    }

    #[test]
    fn transport_v2_platform_tokens_are_separate_from_user_and_v1_domains() {
        let keys = test_keys(10);
        let now = Utc::now();
        let resumption = valid_transport_v2_platform_resumption(&keys);
        assert!(validate_transport_v2_platform_resumption_claims(&resumption, &keys).is_ok());
        assert!(matches!(
            validate_transport_v2_user_resumption_claims(&resumption, &keys),
            Err(ApiError::InvalidJwt)
        ));
        for audience in [PLATFORM_ACCESS, PLATFORM_REFRESH] {
            assert!(matches!(
                validate_token_with_keys(&resumption, &keys, audience),
                Err(ApiError::InvalidJwt)
            ));
        }

        let descriptor = signed_transport_v2_test_token(
            &keys,
            transport_v2_platform_test_claims(
                TransportV2TokenKind::AccessDescriptor,
                TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE,
            ),
            Some(now - Duration::minutes(2)),
            Some(now - Duration::minutes(2)),
            Some(now + Duration::minutes(5)),
        );
        assert!(matches!(
            validate_transport_v2_platform_resumption_claims(&descriptor, &keys),
            Err(ApiError::InvalidJwt)
        ));

        let user_resumption = valid_transport_v2_resumption(&keys);
        assert!(matches!(
            validate_transport_v2_platform_resumption_claims(&user_resumption, &keys),
            Err(ApiError::InvalidJwt)
        ));

        let legacy = signed_test_token(&keys, PLATFORM_REFRESH, Some(now + Duration::minutes(5)));
        assert!(matches!(
            validate_transport_v2_platform_resumption_claims(&legacy, &keys),
            Err(ApiError::InvalidJwt)
        ));

        for audience in [
            TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE,
            TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE,
        ] {
            assert!(matches!(
                TokenType::validate_third_party_audience(audience),
                Err(ApiError::BadRequest)
            ));
        }
    }

    #[test]
    fn transport_v2_platform_resumption_rejects_wrong_identity_purpose_and_time() {
        let keys = test_keys(11);
        let now = Utc::now();
        let sign = |claims: TransportV2PlatformClaims,
                    issued_at: Option<DateTime<Utc>>,
                    not_before: Option<DateTime<Utc>>,
                    expiration: Option<DateTime<Utc>>| {
            signed_transport_v2_test_token(&keys, claims, issued_at, not_before, expiration)
        };
        let valid_claims = || {
            transport_v2_platform_test_claims(
                TransportV2TokenKind::Resumption,
                TRANSPORT_V2_PLATFORM_RESUMPTION_AUDIENCE,
            )
        };

        let mut wrong_issuer = valid_claims();
        wrong_issuer.iss = "urn:attacker".to_owned();
        let mut wrong_audience = valid_claims();
        wrong_audience.aud = TRANSPORT_V2_PLATFORM_ACCESS_AUDIENCE.to_owned();
        let mut wrong_version = valid_claims();
        wrong_version.tv += 1;
        let mut wrong_kind = valid_claims();
        wrong_kind.tk = TransportV2TokenKind::AccessDescriptor;
        let mut wrong_principal = valid_claims();
        wrong_principal.pk = TransportV2PrincipalKind::User;

        for claims in [
            wrong_issuer,
            wrong_audience,
            wrong_version,
            wrong_kind,
            wrong_principal,
        ] {
            let token = sign(
                claims,
                Some(now - Duration::minutes(2)),
                Some(now - Duration::minutes(2)),
                Some(now + Duration::minutes(5)),
            );
            assert!(matches!(
                validate_transport_v2_platform_resumption_claims(&token, &keys),
                Err(ApiError::InvalidJwt)
            ));
        }

        let invalid_times = [
            (None, Some(now), Some(now + Duration::minutes(5))),
            (Some(now), None, Some(now + Duration::minutes(5))),
            (Some(now), Some(now), None),
            (
                Some(now - Duration::minutes(10)),
                Some(now - Duration::minutes(10)),
                Some(now - Duration::minutes(5)),
            ),
            (
                Some(now + Duration::minutes(10)),
                Some(now + Duration::minutes(10)),
                Some(now + Duration::minutes(15)),
            ),
        ];
        for (issued_at, not_before, expiration) in invalid_times {
            let token = sign(valid_claims(), issued_at, not_before, expiration);
            assert!(matches!(
                validate_transport_v2_platform_resumption_claims(&token, &keys),
                Err(ApiError::InvalidJwt)
            ));
        }
    }

    #[test]
    fn transport_v2_resumption_rejects_wrong_identity_and_purpose_claims() {
        let keys = test_keys(7);
        let now = Utc::now();
        let sign = |claims: TransportV2UserClaims| {
            signed_transport_v2_test_token(
                &keys,
                claims,
                Some(now - Duration::minutes(2)),
                Some(now - Duration::minutes(2)),
                Some(now + Duration::minutes(5)),
            )
        };

        let mut wrong_issuer = transport_v2_test_claims(
            TransportV2TokenKind::Resumption,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        );
        wrong_issuer.iss = "urn:attacker".to_owned();
        let mut wrong_audience = transport_v2_test_claims(
            TransportV2TokenKind::Resumption,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        );
        wrong_audience.aud = TRANSPORT_V2_USER_ACCESS_AUDIENCE.to_owned();
        let mut wrong_version = transport_v2_test_claims(
            TransportV2TokenKind::Resumption,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        );
        wrong_version.tv += 1;
        let mut wrong_kind = transport_v2_test_claims(
            TransportV2TokenKind::Resumption,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        );
        wrong_kind.tk = TransportV2TokenKind::AccessDescriptor;

        for token in [
            sign(wrong_issuer),
            sign(wrong_audience),
            sign(wrong_version),
            sign(wrong_kind),
        ] {
            assert!(matches!(
                validate_transport_v2_user_resumption_claims(&token, &keys),
                Err(ApiError::InvalidJwt)
            ));
        }

        let mut wrong_principal = serde_json::to_value(transport_v2_test_claims(
            TransportV2TokenKind::Resumption,
            TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
        ))
        .unwrap();
        wrong_principal["pk"] = serde_json::Value::String("platform".to_owned());
        let token = signed_transport_v2_test_token(
            &keys,
            wrong_principal,
            Some(now - Duration::minutes(2)),
            Some(now - Duration::minutes(2)),
            Some(now + Duration::minutes(5)),
        );
        assert!(matches!(
            validate_transport_v2_user_resumption_claims(&token, &keys),
            Err(ApiError::InvalidJwt)
        ));

        let other_keys = test_keys(8);
        assert!(matches!(
            validate_transport_v2_user_resumption_claims(
                &valid_transport_v2_resumption(&other_keys),
                &keys
            ),
            Err(ApiError::InvalidJwt)
        ));
    }

    #[test]
    fn transport_v2_resumption_requires_strict_freshness_and_auth_binding() {
        let keys = test_keys(9);
        let now = Utc::now();
        let claims = || {
            transport_v2_test_claims(
                TransportV2TokenKind::Resumption,
                TRANSPORT_V2_USER_RESUMPTION_AUDIENCE,
            )
        };
        let cases = [
            (None, Some(now), Some(now + Duration::minutes(5))),
            (Some(now), None, Some(now + Duration::minutes(5))),
            (Some(now), Some(now), None),
            (
                Some(now - Duration::minutes(10)),
                Some(now - Duration::minutes(10)),
                Some(now - Duration::minutes(5)),
            ),
            (
                Some(now + Duration::minutes(10)),
                Some(now + Duration::minutes(10)),
                Some(now + Duration::minutes(15)),
            ),
        ];
        for (issued_at, not_before, expiration) in cases {
            let token =
                signed_transport_v2_test_token(&keys, claims(), issued_at, not_before, expiration);
            assert!(matches!(
                validate_transport_v2_user_resumption_claims(&token, &keys),
                Err(ApiError::InvalidJwt)
            ));
        }

        let mut malformed_base64 = claims();
        malformed_base64.auth_binding = "not-base64".to_owned();
        assert!(matches!(
            transport_v2_auth_context(&malformed_base64),
            Err(ApiError::InvalidJwt)
        ));
        let mut wrong_length = claims();
        wrong_length.auth_binding = URL_SAFE_NO_PAD.encode([1_u8; 31]);
        assert!(matches!(
            transport_v2_auth_context(&wrong_length),
            Err(ApiError::InvalidJwt)
        ));
        let mut wrong_format = claims();
        wrong_format.token_format += 1;
        assert!(matches!(
            transport_v2_auth_context(&wrong_format),
            Err(ApiError::InvalidJwt)
        ));
        let mut wrong_method = claims();
        wrong_method.auth_method = "api_key".to_owned();
        assert!(matches!(
            transport_v2_auth_context(&wrong_method),
            Err(ApiError::InvalidJwt)
        ));
    }

    #[test]
    fn transport_v2_native_handoff_grant_is_exactly_session_and_attempt_bound() {
        let keys = test_keys(12);
        let target_session_id = Uuid::from_bytes([0x51; 16]);
        let native_attempt_id = Uuid::from_bytes([0x52; 16]);
        let now = Utc::now();
        let token = signed_transport_v2_test_token(
            &keys,
            transport_v2_native_handoff_test_claims(target_session_id, native_attempt_id),
            Some(now),
            Some(now),
            Some(now + TRANSPORT_V2_NATIVE_HANDOFF_TTL),
        );

        let claims = validate_transport_v2_native_handoff_grant_claims(
            &token,
            &keys,
            target_session_id,
            native_attempt_id,
        )
        .expect("valid grant should pass its exact domain and binding checks");
        let auth_context = transport_v2_auth_context_from_parts(
            claims.token_format,
            &claims.auth_method,
            claims.project_id,
            &claims.auth_binding,
        )
        .expect("valid grant should carry a complete auth context");
        assert_eq!(auth_context.method, AuthMethod::OAuth);
        assert!(is_canonical_compact_jwt(&token));
        assert!(token.len() <= TRANSPORT_V2_NATIVE_HANDOFF_GRANT_MAX_BYTES);
        let encoded_payload = token
            .split('.')
            .nth(1)
            .expect("compact JWT must have a payload");
        let payload = URL_SAFE_NO_PAD
            .decode(encoded_payload)
            .expect("signed test payload must be canonical base64url");
        assert!(!String::from_utf8_lossy(&payload).contains(&native_attempt_id.to_string()));

        for (session_id, attempt_id) in [
            (Uuid::from_bytes([0x53; 16]), native_attempt_id),
            (target_session_id, Uuid::from_bytes([0x54; 16])),
        ] {
            assert!(matches!(
                validate_transport_v2_native_handoff_grant_claims(
                    &token, &keys, session_id, attempt_id,
                ),
                Err(ApiError::InvalidJwt)
            ));
        }

        assert!(matches!(
            validate_transport_v2_native_handoff_grant_claims(
                &token,
                &test_keys(13),
                target_session_id,
                native_attempt_id,
            ),
            Err(ApiError::InvalidJwt)
        ));
        assert!(matches!(
            TokenType::validate_third_party_audience(TRANSPORT_V2_USER_NATIVE_HANDOFF_AUDIENCE),
            Err(ApiError::BadRequest)
        ));
    }

    #[test]
    fn transport_v2_native_handoff_rejects_wrong_domain_time_and_wire_shape() {
        let keys = test_keys(14);
        let target_session_id = Uuid::from_bytes([0xab; 16]);
        let native_attempt_id = Uuid::from_bytes([0xcd; 16]);
        let now = Utc::now();
        let sign = |claims: TransportV2NativeHandoffClaims,
                    issued_at: Option<DateTime<Utc>>,
                    not_before: Option<DateTime<Utc>>,
                    expiration: Option<DateTime<Utc>>| {
            signed_transport_v2_test_token(&keys, claims, issued_at, not_before, expiration)
        };
        let validate = |token: &str| {
            validate_transport_v2_native_handoff_grant_claims(
                token,
                &keys,
                target_session_id,
                native_attempt_id,
            )
        };
        let valid_claims =
            || transport_v2_native_handoff_test_claims(target_session_id, native_attempt_id);

        let mut wrong_issuer = valid_claims();
        wrong_issuer.iss = "urn:attacker".to_owned();
        let mut wrong_audience = valid_claims();
        wrong_audience.aud = TRANSPORT_V2_USER_RESUMPTION_AUDIENCE.to_owned();
        let mut wrong_version = valid_claims();
        wrong_version.tv += 1;
        let mut wrong_kind = valid_claims();
        wrong_kind.tk = TransportV2TokenKind::Resumption;
        let mut wrong_principal = valid_claims();
        wrong_principal.pk = TransportV2PrincipalKind::Platform;
        let mut noncanonical_session = valid_claims();
        noncanonical_session.target_session_id =
            noncanonical_session.target_session_id.to_uppercase();
        let mut malformed_attempt_commitment = valid_claims();
        malformed_attempt_commitment.native_attempt_commitment = Uuid::nil().to_string();

        for claims in [
            wrong_issuer,
            wrong_audience,
            wrong_version,
            wrong_kind,
            wrong_principal,
            noncanonical_session,
            malformed_attempt_commitment,
        ] {
            let token = sign(
                claims,
                Some(now),
                Some(now),
                Some(now + TRANSPORT_V2_NATIVE_HANDOFF_TTL),
            );
            assert!(matches!(validate(&token), Err(ApiError::InvalidJwt)));
        }

        let invalid_times = [
            (None, Some(now), Some(now + Duration::minutes(1))),
            (Some(now), None, Some(now + Duration::minutes(1))),
            (Some(now), Some(now), None),
            (
                Some(now - Duration::minutes(2)),
                Some(now - Duration::minutes(2)),
                Some(now - Duration::minutes(1)),
            ),
            (
                Some(now + Duration::minutes(1)),
                Some(now + Duration::minutes(1)),
                Some(now + Duration::minutes(2)),
            ),
            (
                Some(now),
                Some(now),
                Some(now + TRANSPORT_V2_NATIVE_HANDOFF_TTL + Duration::seconds(1)),
            ),
            (
                Some(now),
                Some(now + Duration::seconds(1)),
                Some(now + Duration::minutes(1)),
            ),
        ];
        for (issued_at, not_before, expiration) in invalid_times {
            let token = sign(valid_claims(), issued_at, not_before, expiration);
            assert!(matches!(validate(&token), Err(ApiError::InvalidJwt)));
        }

        let access_descriptor = signed_transport_v2_test_token(
            &keys,
            transport_v2_test_claims(
                TransportV2TokenKind::AccessDescriptor,
                TRANSPORT_V2_USER_ACCESS_AUDIENCE,
            ),
            Some(now),
            Some(now),
            Some(now + Duration::minutes(1)),
        );
        assert!(matches!(
            validate(&access_descriptor),
            Err(ApiError::InvalidJwt)
        ));

        for malformed in [
            "not-a-jwt".to_owned(),
            "a.b.c".to_owned(),
            "a.b.c.d".to_owned(),
            "a.b.c=".to_owned(),
            "a".repeat(TRANSPORT_V2_NATIVE_HANDOFF_GRANT_MAX_BYTES + 1),
        ] {
            assert!(!is_canonical_compact_jwt(&malformed));
            assert!(matches!(validate(&malformed), Err(ApiError::InvalidJwt)));
        }
    }
}
