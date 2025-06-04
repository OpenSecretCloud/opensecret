use crate::email::{
    send_account_deletion_confirmation_email, send_account_deletion_email,
    send_password_reset_confirmation_email, send_password_reset_email,
    send_platform_password_reset_confirmation_email, send_platform_password_reset_email,
};
use crate::encrypt::encrypt_key_deterministic;
use crate::encrypt::generate_random;
use crate::encrypt::{
    decrypt_with_key, decrypt_with_kms, encrypt_with_key, CustomRng, GenKeyResult,
};
use crate::jwt::validate_platform_jwt;
use crate::login_routes::RegisterCredentials;
use crate::models::account_deletion::{AccountDeletionError, NewAccountDeletionRequest};
use crate::models::password_reset::NewPasswordResetRequest;
use crate::models::platform_password_reset::NewPlatformPasswordResetRequest;
use crate::models::platform_users::PlatformUser;
use crate::sqs::SqsEventPublisher;
use crate::web::platform_login_routes;
use crate::web::{health_routes, login_routes, oauth_routes, openai_routes, protected_routes};
use crate::{attestation_routes::SessionState, web::platform_routes};

use crate::{
    aws_credentials::AwsCredentialError,
    models::enclave_secrets::NewEnclaveSecret,
    private_key::{decrypt_user_seed_to_key, generate_twelve_word_seed},
};
use crate::{billing::BillingClient, web::platform::common::PROJECT_RESEND_API_KEY};
use crate::{
    db::{setup_db, DBConnection, DBError},
    models::users::{NewUser, User},
};
use crate::{encrypt::create_new_encryption_key, jwt::validate_jwt};
use aws_credentials::{AwsCredentialManager, AwsCredentials};
use axum::{http::StatusCode, middleware::from_fn_with_state, response::IntoResponse, Json};
use base64::engine::general_purpose;
use base64::Engine as _;
use chacha20poly1305::aead::Aead;
use chacha20poly1305::KeyInit;
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use kv::{KVPair, StoreError, StoreResult};
use password_auth::{generate_hash, verify_password, VerifyError};
use rand_core::{CryptoRng, RngCore};
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::io::{Read, Write};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use subtle::ConstantTimeEq;
use tokio::spawn;
use tokio::sync::RwLock;
use tokio::task::{self};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use url::Url;
use uuid::Uuid;
use vsock::{VsockAddr, VsockStream};
use web::attestation_routes;
use x25519_dalek::{EphemeralSecret, PublicKey};

mod apple_signin;
mod aws_credentials;
mod billing;
mod db;
mod email;
mod encrypt;
mod jwt;
mod kv;
mod message_signing;
mod migrations;
mod models;
mod oauth;
mod private_key;
mod proxy_config;
mod sqs;
mod web;

use apple_signin::AppleJwtVerifier;
use oauth::{AppleProvider, GithubProvider, GoogleProvider, OAuthManager};
use proxy_config::ProxyRouter;

const ENCLAVE_KEY_NAME: &str = "enclave_key";
const OPENAI_API_KEY_NAME: &str = "openai_api_key";
const JWT_SECRET_KEY_NAME: &str = "jwt_secret";

// General secret key names
const GITHUB_CLIENT_ID_NAME: &str = "github_client_id";
const GITHUB_CLIENT_SECRET_NAME: &str = "github_client_secret";
const GOOGLE_CLIENT_ID_NAME: &str = "google_client_id";
const GOOGLE_CLIENT_SECRET_NAME: &str = "google_client_secret";
const RESEND_API_KEY_NAME: &str = "resend_api_key";

const BILLING_API_KEY_NAME: &str = "billing_api_key";
const BILLING_SERVER_URL_NAME: &str = "billing_server_url";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnclaveRequest {
    pub request_type: String,
    pub key_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParentResponse {
    pub response_type: String,
    pub response_value: serde_json::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    TaskJoin(#[from] task::JoinError),

    #[error(transparent)]
    StdIo(#[from] std::io::Error),

    #[error(transparent)]
    TryInit(#[from] tracing_subscriber::util::TryInitError),

    #[error("Database error: {0}")]
    DatabaseError(#[from] DBError),

    #[error("Private key not found")]
    PrivateKeyNotFound,

    #[error("Private key could not be generated")]
    PrivateKeyGenerationFailure,

    #[error("Private key already exists")]
    PrivateKeyAlreadyExists,

    #[error("User not found")]
    UserNotFound,

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Authentication error")]
    AuthenticationError,

    #[error("Failed to parse secret")]
    SecretParsingError,

    #[error("AWS Credential error: {0}")]
    AwsCredentialError(#[from] aws_credentials::AwsCredentialError),

    #[error("User is already verified")]
    UserAlreadyVerified,

    #[error("Builder error: {0}")]
    BuilderError(String),

    #[error("Password reset request expired")]
    PasswordResetExpired,

    #[error("Invalid password reset secret")]
    InvalidPasswordResetSecret,

    #[error("Invalid password reset request")]
    InvalidPasswordResetRequest,

    #[error("Account deletion request expired")]
    AccountDeletionExpired,

    #[error("Invalid account deletion secret")]
    InvalidAccountDeletionSecret,

    #[error("Invalid account deletion request")]
    InvalidAccountDeletionRequest,

    #[error("Password verification error: {0}")]
    PasswordVerificationError(#[from] VerifyError),

    #[error("Password is required for registration")]
    PasswordRequired,

    #[error("OAuth error: {0}")]
    OAuthError(String),

    #[error("User with this email already exists")]
    UserAlreadyExists,

    #[error("Signing error: {0}")]
    SigningError(String),

    #[error("Invalid derivation path: {0}")]
    InvalidDerivationPath(String),

    #[error("Key derivation failed: {0}")]
    KeyDerivationError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Invalid email, password, or login method")]
    InvalidUsernameOrPassword,

    #[error("Invalid JWT")]
    InvalidJwt,

    #[error("Internal server error")]
    InternalServerError,

    #[error("Bad Request")]
    BadRequest,

    #[error("Encryption error")]
    EncryptionError,

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Invalid invite code")]
    InvalidInviteCode,

    #[error("Token refresh failed")]
    RefreshFailed,

    #[error("User is already verified")]
    UserAlreadyVerified,

    #[error("No valid email found for the Oauth account")]
    NoEmailFound,

    #[error("User exists but Oauth not linked")]
    UserExistsNotLinked,

    #[error("User not found")]
    UserNotFound,

    #[error("Email already registered")]
    EmailAlreadyExists,

    #[error("Usage limit reached")]
    UsageLimitReached,

    #[error("Resource not found")]
    NotFound,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            ApiError::InvalidUsernameOrPassword => StatusCode::UNAUTHORIZED,
            ApiError::InvalidJwt => StatusCode::UNAUTHORIZED,
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::InternalServerError => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::BadRequest => StatusCode::BAD_REQUEST,
            ApiError::InvalidInviteCode => StatusCode::UNAUTHORIZED,
            ApiError::RefreshFailed => StatusCode::UNAUTHORIZED,
            ApiError::UserAlreadyVerified => StatusCode::BAD_REQUEST,
            ApiError::EncryptionError => StatusCode::BAD_REQUEST,
            ApiError::NoEmailFound => StatusCode::BAD_REQUEST,
            ApiError::UserExistsNotLinked => StatusCode::CONFLICT,
            ApiError::UserNotFound => StatusCode::NOT_FOUND,
            ApiError::EmailAlreadyExists => StatusCode::CONFLICT,
            ApiError::UsageLimitReached => StatusCode::FORBIDDEN,
            ApiError::NotFound => StatusCode::NOT_FOUND,
        };
        (
            status,
            Json(ErrorResponse {
                status: status.as_u16(),
                message: self.to_string(),
            }),
        )
            .into_response()
    }
}

impl From<DBError> for ApiError {
    fn from(err: DBError) -> Self {
        error!("Database error: {:?}", err);
        match err {
            DBError::PlatformUserNotFound => ApiError::UserNotFound,
            DBError::PlatformUserError(_) => ApiError::InternalServerError,
            DBError::OrgMembershipNotFound => ApiError::NotFound,
            DBError::OrgMembershipError(_) => ApiError::InternalServerError,
            _ => ApiError::InternalServerError,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    status: u16,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    // Subject (whom token refers to)
    pub sub: String, // This will now be the UUID
    // Expiration time (as UTC timestamp)
    pub exp: i64,
    // Issued at (as UTC timestamp)
    pub iat: i64,
    // Audience
    pub aud: String,
}

#[derive(Debug, Clone)]
pub struct Config {
    jwt_keys: jwt::JwtKeys,
    access_token_maxage: i64,
    refresh_token_maxage: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    Local,
    Dev,
    Preview,
    Prod,
    Custom(String),
}

impl AppMode {
    fn frontend_url(&self) -> &str {
        match self {
            AppMode::Local => "http://127.0.0.1:5173",
            AppMode::Dev => "https://dev.secretgpt.ai",
            AppMode::Preview => "https://preview.opensecret.cloud",
            AppMode::Prod => "https://trymaple.ai",
            AppMode::Custom(_) => "https://preview.opensecret.cloud",
        }
    }
}

impl fmt::Display for AppMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppMode::Local => write!(f, "local"),
            AppMode::Dev => write!(f, "dev"),
            AppMode::Preview => write!(f, "preview"),
            AppMode::Prod => write!(f, "prod"),
            AppMode::Custom(_) => write!(f, "custom"),
        }
    }
}

impl FromStr for AppMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" => Ok(AppMode::Local),
            "dev" => Ok(AppMode::Dev),
            "preview" => Ok(AppMode::Preview),
            "prod" => Ok(AppMode::Prod),
            "custom" => {
                // For custom mode, get the ENV_NAME
                match std::env::var("ENV_NAME") {
                    Ok(env_name) => Ok(AppMode::Custom(env_name)),
                    Err(_) => Err("ENV_NAME must be set when using custom mode".to_string()),
                }
            }
            _ => Err(format!("Invalid app mode: {}", s)),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    app_mode: AppMode,
    db: Arc<dyn DBConnection + Send + Sync>,
    config: Config,
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    enclave_key: Vec<u8>,
    proxy_router: Arc<ProxyRouter>,
    resend_api_key: Option<String>,
    ephemeral_keys: Arc<RwLock<HashMap<String, EphemeralSecret>>>,
    session_states: Arc<tokio::sync::RwLock<HashMap<Uuid, SessionState>>>,
    oauth_manager: Arc<OAuthManager>,
    sqs_publisher: Option<Arc<SqsEventPublisher>>,
    billing_client: Option<BillingClient>,
    apple_jwt_verifier: Arc<AppleJwtVerifier>,
}

#[derive(Default)]
pub struct AppStateBuilder {
    app_mode: Option<AppMode>,
    db: Option<Arc<dyn DBConnection + Send + Sync>>,
    enclave_key: Option<Vec<u8>>,
    aws_credential_manager: Option<Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>>,
    openai_api_key: Option<String>,
    openai_api_base: Option<String>,
    tinfoil_api_base: Option<String>,
    jwt_secret: Option<Vec<u8>>,
    resend_api_key: Option<String>,
    github_client_secret: Option<String>,
    github_client_id: Option<String>,
    google_client_secret: Option<String>,
    google_client_id: Option<String>,
    apple_client_secret: Option<String>,
    apple_client_id: Option<String>,
    sqs_queue_maple_events_url: Option<String>,
    sqs_publisher: Option<Arc<SqsEventPublisher>>,
    billing_api_key: Option<String>,
    billing_server_url: Option<String>,
}

impl AppStateBuilder {
    pub fn app_mode(mut self, app_mode: AppMode) -> Self {
        self.app_mode = Some(app_mode);
        self
    }

    pub fn db(mut self, db: Arc<dyn DBConnection + Send + Sync>) -> Self {
        self.db = Some(db);
        self
    }

    pub fn enclave_key(mut self, enclave_key: Vec<u8>) -> Self {
        self.enclave_key = Some(enclave_key);
        self
    }

    pub fn aws_credential_manager(
        mut self,
        aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    ) -> Self {
        self.aws_credential_manager = Some(aws_credential_manager);
        self
    }

    pub fn openai_api_key(mut self, openai_api_key: Option<String>) -> Self {
        self.openai_api_key = openai_api_key;
        self
    }

    pub fn openai_api_base(mut self, openai_api_base: String) -> Self {
        self.openai_api_base = Some(openai_api_base);
        self
    }

    pub fn tinfoil_api_base(mut self, tinfoil_api_base: Option<String>) -> Self {
        self.tinfoil_api_base = tinfoil_api_base;
        self
    }

    pub fn jwt_secret(mut self, jwt_secret: Vec<u8>) -> Self {
        self.jwt_secret = Some(jwt_secret);
        self
    }

    pub fn resend_api_key(mut self, resend_api_key: Option<String>) -> Self {
        self.resend_api_key = resend_api_key;
        self
    }

    pub fn github_client_secret(mut self, github_client_secret: Option<String>) -> Self {
        self.github_client_secret = github_client_secret;
        self
    }

    pub fn github_client_id(mut self, github_client_id: Option<String>) -> Self {
        self.github_client_id = github_client_id;
        self
    }

    pub fn google_client_secret(mut self, google_client_secret: Option<String>) -> Self {
        self.google_client_secret = google_client_secret;
        self
    }

    pub fn google_client_id(mut self, google_client_id: Option<String>) -> Self {
        self.google_client_id = google_client_id;
        self
    }

    pub fn apple_client_secret(mut self, apple_client_secret: Option<String>) -> Self {
        self.apple_client_secret = apple_client_secret;
        self
    }

    pub fn apple_client_id(mut self, apple_client_id: Option<String>) -> Self {
        self.apple_client_id = apple_client_id;
        self
    }

    pub fn sqs_queue_maple_events_url(
        mut self,
        sqs_queue_maple_events_url: Option<String>,
    ) -> Self {
        self.sqs_queue_maple_events_url = sqs_queue_maple_events_url;
        self
    }

    pub fn sqs_publisher(mut self, sqs_publisher: Option<Arc<SqsEventPublisher>>) -> Self {
        self.sqs_publisher = sqs_publisher;
        self
    }

    pub fn billing_api_key(mut self, billing_api_key: Option<String>) -> Self {
        self.billing_api_key = billing_api_key;
        self
    }

    pub fn billing_server_url(mut self, billing_server_url: Option<String>) -> Self {
        self.billing_server_url = billing_server_url;
        self
    }

    pub async fn build(self) -> Result<AppState, Error> {
        let app_mode = self
            .app_mode
            .ok_or(Error::BuilderError("app_mode is required".to_string()))?;
        let db = self
            .db
            .ok_or(Error::BuilderError("db is required".to_string()))?;
        let enclave_key = self
            .enclave_key
            .ok_or(Error::BuilderError("enclave_key is required".to_string()))?;
        let aws_credential_manager = self.aws_credential_manager.ok_or(Error::BuilderError(
            "aws_credential_manager is required".to_string(),
        ))?;
        let openai_api_base = self.openai_api_base.ok_or(Error::BuilderError(
            "openai_api_base is required".to_string(),
        ))?;
        let jwt_secret = self
            .jwt_secret
            .ok_or(Error::BuilderError("jwt_secret is required".to_string()))?;

        let config = Config {
            jwt_keys: jwt::JwtKeys::new(jwt_secret)?,
            access_token_maxage: 60,  // 60 minutes
            refresh_token_maxage: 30, // 30 days
        };

        // Log the public key in hex format
        tracing::info!(
            "JWT ES256K public key (hex): {}",
            hex::encode(config.jwt_keys.public_key().serialize())
        );

        let mut oauth_manager = OAuthManager::new();

        // Initialize GitHub provider
        let github_provider = GithubProvider::new(db.clone()).await.map_err(|e| {
            error!("Failed to initialize GitHub OAuth provider: {:?}", e);
            Error::BuilderError("Failed to initialize GitHub OAuth provider".to_string())
        })?;
        oauth_manager.add_provider("github".to_string(), Box::new(github_provider));

        // Initialize Google provider
        let google_provider = GoogleProvider::new(db.clone()).await.map_err(|e| {
            error!("Failed to initialize Google OAuth provider: {:?}", e);
            Error::BuilderError("Failed to initialize Google OAuth provider".to_string())
        })?;
        oauth_manager.add_provider("google".to_string(), Box::new(google_provider));

        // Initialize Apple provider
        let apple_provider = AppleProvider::new(db.clone()).await.map_err(|e| {
            error!("Failed to initialize Apple OAuth provider: {:?}", e);
            Error::BuilderError("Failed to initialize Apple OAuth provider".to_string())
        })?;
        oauth_manager.add_provider("apple".to_string(), Box::new(apple_provider));

        let oauth_manager = Arc::new(oauth_manager);

        // Initialize SQS publisher if URL is provided
        let sqs_publisher = if let Some(ref queue_url) = self.sqs_queue_maple_events_url {
            // Use the same region as AWS credentials if available
            let region = if let Some(creds) = aws_credential_manager.read().await.as_ref() {
                creds.get_credentials().await.map(|c| c.region)
            } else {
                None
            };

            Some(Arc::new(
                SqsEventPublisher::new(queue_url.clone(), region, aws_credential_manager.clone())
                    .await,
            ))
        } else {
            None
        };

        let billing_client = if let (Some(api_key), Some(base_url)) =
            (self.billing_api_key, self.billing_server_url)
        {
            tracing::debug!("Billing client is configured.");
            Some(BillingClient::new(api_key, base_url))
        } else {
            tracing::debug!("Billing client not configured");
            None
        };

        // Initialize the AppleJwtVerifier
        let apple_jwt_verifier = Arc::new(AppleJwtVerifier::new());

        // Initialize the ProxyRouter
        let proxy_router = Arc::new(ProxyRouter::new(
            openai_api_base.clone(),
            self.openai_api_key.clone(),
            self.tinfoil_api_base.clone(),
        ));

        Ok(AppState {
            app_mode,
            db,
            config,
            aws_credential_manager,
            enclave_key,
            proxy_router,
            resend_api_key: self.resend_api_key,
            ephemeral_keys: Arc::new(RwLock::new(HashMap::new())),
            session_states: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            oauth_manager,
            sqs_publisher,
            billing_client,
            apple_jwt_verifier,
        })
    }
}

impl AppState {
    async fn register_user(&self, creds: RegisterCredentials) -> Result<User, Error> {
        // Get project by client_id
        let project = self
            .db
            .get_org_project_by_client_id(creds.client_id)
            .map_err(|e| {
                error!(
                    "Database error during client_id ({:?}) lookup: {:?}",
                    creds.client_id, e
                );
                DBError::OrgProjectNotFound
            })?;

        // First check if user exists - only if email is provided
        if let Some(email) = &creds.email {
            match self.db.get_user_by_email(email.clone(), project.id) {
                Ok(_) => {
                    // User already exists in this project
                    return Err(Error::UserAlreadyExists);
                }
                Err(DBError::UserNotFound) => {
                    // This is what we want - user doesn't exist
                }
                Err(e) => {
                    // Some other database error
                    return Err(Error::DatabaseError(e));
                }
            }
        }

        let password = creds.password;

        // hash then encrypt with enclave key
        let password_hash = generate_hash(password);

        let secret_key = SecretKey::from_slice(&self.enclave_key.clone())
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let encrypted_pw = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

        tracing::debug!("registering new user: {:?}", creds.email);

        // Generate private key for new user
        let user_seed_words = generate_twelve_word_seed(self.aws_credential_manager.clone())
            .await?
            .to_string();

        let encrypted_key = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

        let new_user = NewUser::new(creds.email, Some(encrypted_pw), project.id, encrypted_key)
            .with_name_option(creds.name);

        let user = self.db.create_user(new_user)?;

        tracing::info!("registered new user: {:?} {:?}", user.email, user.uuid);

        Ok(user)
    }

    async fn authenticate_user(
        &self,
        user_email: Option<String>,
        user_id: Option<Uuid>,
        user_password: String,
        user_project_id: i32,
    ) -> Result<Option<User>, Error> {
        // Ensure at least one identifier is provided
        if user_email.is_none() && user_id.is_none() {
            return Err(Error::AuthenticationError);
        }

        // Try email first if provided, fall back to UUID
        let user = if let Some(email) = user_email {
            self.db.get_user_by_email(email, user_project_id)?
        } else {
            // We can safely unwrap id here because we checked above that at least one exists
            let user = self.db.get_user_by_uuid(user_id.unwrap())?;
            // Verify user belongs to the specified project
            if user.project_id != user_project_id {
                return Err(Error::AuthenticationError);
            }
            user
        };

        // Check if the user is an OAuth-only user
        if user.password_enc.is_none() {
            error!("OAuth-only user attempted password login");
            return Ok(None);
        }

        // Verify the current password
        let secret_key = SecretKey::from_slice(&self.enclave_key.clone())
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let decrypted_password_bytes =
            decrypt_with_key(&secret_key, user.password_enc.as_ref().unwrap())
                .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let decrypted_password_hash = String::from_utf8(decrypted_password_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))?;

        // Verifying the password is blocking and potentially slow, so we'll do so via
        // `spawn_blocking`.
        let res =
            task::spawn_blocking(move || verify_password(user_password, &decrypted_password_hash))
                .await?;

        match res {
            Ok(_) => Ok(Some(user)),
            Err(_) => Ok(None),
        }
    }

    async fn get_user(&self, user_uuid: Uuid) -> Result<User, Error> {
        let user = self
            .db
            .get_user_by_uuid(user_uuid)
            .map_err(|_| Error::UserNotFound)?;
        Ok(user)
    }

    /// Returns the user's private key, optionally derived using the provided derivation paths.
    ///
    /// # Arguments
    /// * `user_uuid` - The UUID of the user
    /// * `derivation_path` - Optional BIP-32 derivation path
    /// * `seed_phrase_derivation_path` - Optional BIP-85 derivation path to derive a child seed
    ///
    /// # Returns
    /// * `Result<SecretKey, Error>` - The user's private key, potentially derived through
    ///   BIP-85 and/or BIP-32 derivation paths
    async fn get_user_key(
        &self,
        user_uuid: Uuid,
        derivation_path: Option<&str>,
        seed_phrase_derivation_path: Option<&str>,
    ) -> Result<SecretKey, Error> {
        info!("Getting user key for UUID: {}", user_uuid);

        if let Some(path) = derivation_path {
            debug!("Using BIP-32 derivation path: {}", path);
        }

        if let Some(path) = seed_phrase_derivation_path {
            debug!("Using BIP-85 derivation path: {}", path);
        }

        let user = self.get_user(user_uuid).await?;
        debug!("User retrieved successfully");

        let encrypted_seed = match user.get_seed_encrypted().await {
            Some(es) => {
                debug!("Found existing encrypted seed for user");
                es
            }
            None => {
                // create seed if not already exists
                info!(
                    "No seed found for user {}, generating new private key",
                    user_uuid
                );
                let updated_user = self.generate_private_key(user_uuid).await?;
                updated_user.get_seed_encrypted().await.ok_or_else(|| {
                    error!("Seed not found after generation for user: {}", user_uuid);
                    Error::PrivateKeyGenerationFailure
                })?
            }
        };

        debug!("Decrypting user seed and deriving key");
        let user_secret_key = decrypt_user_seed_to_key(
            self.enclave_key.clone(),
            encrypted_seed,
            derivation_path,
            seed_phrase_derivation_path,
        )?;

        info!("Successfully derived key for user: {}", user_uuid);
        Ok(user_secret_key)
    }

    /// Sign a message with the user's private key, using the specified algorithm
    async fn sign_message(
        &self,
        user_uuid: Uuid,
        message_bytes: &[u8],
        algorithm: message_signing::SigningAlgorithm,
        derivation_path: Option<&str>,
        seed_phrase_derivation_path: Option<&str>,
    ) -> Result<message_signing::SignMessageResponse, Error> {
        info!(
            "Signing message for user: {}, algorithm: {:?}",
            user_uuid, algorithm
        );

        if let Some(path) = derivation_path {
            debug!("Using BIP-32 derivation path: {}", path);
        }

        if let Some(path) = seed_phrase_derivation_path {
            debug!("Using BIP-85 derivation path: {}", path);
        }

        debug!("Getting user key for message signing");
        let user_secret_key = self
            .get_user_key(user_uuid, derivation_path, seed_phrase_derivation_path)
            .await?;

        debug!("Signing message with algorithm: {:?}", algorithm);
        let result = message_signing::sign_message(&user_secret_key, message_bytes, algorithm);

        if result.is_ok() {
            info!("Message signed successfully for user: {}", user_uuid);
        } else {
            error!("Failed to sign message for user: {}", user_uuid);
        }

        result
    }

    // DEPRECATED: New users now always get a private key
    async fn generate_private_key(&self, user_uuid: Uuid) -> Result<User, Error> {
        let user = self.get_user(user_uuid).await?;

        if user.get_seed_encrypted().await.is_none() {
            let user_seed_words = generate_twelve_word_seed(self.aws_credential_manager.clone())
                .await?
                .to_string();

            let secret_key = SecretKey::from_slice(&self.enclave_key.clone())
                .map_err(|e| Error::EncryptionError(e.to_string()))?;

            let encrypted_key = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

            self.db.set_user_key(user, encrypted_key)?;

            self.get_user(user_uuid).await
        } else {
            Err(Error::PrivateKeyAlreadyExists)
        }
    }

    async fn get(&self, user_id: Uuid, key: String) -> StoreResult<Option<String>> {
        let user_key = self
            .get_user_key(user_id, None, None)
            .await
            .map_err(|_| StoreError::Unauthorized)?;
        kv::get(self.db.get_pool(), user_id, &key, &user_key)
    }

    async fn put(&self, user_id: Uuid, key: String, value: String) -> StoreResult<()> {
        let user_key = self
            .get_user_key(user_id, None, None)
            .await
            .map_err(|_| StoreError::Unauthorized)?;
        kv::put(
            self.db.get_pool(),
            user_id,
            key,
            value,
            &user_key,
            self.aws_credential_manager.clone(),
        )
        .await
    }

    async fn delete(&self, user_id: Uuid, key: String) -> StoreResult<()> {
        let user_key = self
            .get_user_key(user_id, None, None)
            .await
            .map_err(|_| StoreError::Unauthorized)?;
        kv::delete(self.db.get_pool(), user_id, &key, &user_key)
    }

    async fn list(&self, user_id: Uuid) -> StoreResult<Vec<KVPair>> {
        let user_key = self
            .get_user_key(user_id, None, None)
            .await
            .map_err(|_| StoreError::Unauthorized)?;
        kv::list(self.db.get_pool(), user_id, &user_key)
    }

    pub async fn get_aws_credentials(&self) -> Option<AwsCredentials> {
        if let Some(manager) = self.aws_credential_manager.read().await.as_ref() {
            manager.get_credentials().await
        } else {
            None
        }
    }

    pub async fn get_enclave_key(&self) -> Vec<u8> {
        self.enclave_key.clone()
    }

    pub async fn create_ephemeral_key(&self, nonce: String) -> PublicKey {
        let custom_rng = CustomRng::new();

        // Use a wrapper that implements RngCore and CryptoRng
        let mut rng_wrapper = AsyncRngWrapper::new(custom_rng);

        // Create the EphemeralSecret using the RngCore implementation
        let ephemeral_secret = EphemeralSecret::random_from_rng(&mut rng_wrapper);
        let public_key = PublicKey::from(&ephemeral_secret);

        self.ephemeral_keys
            .write()
            .await
            .insert(nonce, ephemeral_secret);

        public_key
    }

    pub async fn get_and_remove_ephemeral_secret(&self, nonce: &str) -> Option<EphemeralSecret> {
        self.ephemeral_keys.write().await.remove(nonce)
    }

    pub async fn decrypt_session_data(
        &self,
        session_id: &Uuid,
        encrypted_data: &str,
    ) -> Result<Vec<u8>, ApiError> {
        tracing::trace!("decrypting session data for session_id: {}", session_id);

        let decoded_data = general_purpose::STANDARD
            .decode(encrypted_data)
            .map_err(|e| {
                tracing::error!("Failed to decode base64 data: {:?}", e);
                ApiError::BadRequest
            })?;

        tracing::trace!("decoded session data length: {}", decoded_data.len());

        if decoded_data.len() < 12 {
            tracing::error!("Decoded data is too short");
            return Err(ApiError::BadRequest);
        }

        let (nonce, ciphertext) = decoded_data.split_at(12);
        let nonce_array: [u8; 12] = nonce.try_into().map_err(|e| {
            tracing::error!("Failed to convert nonce: {:?}", e);
            ApiError::BadRequest
        })?;

        tracing::trace!("nonce: {:?}", nonce_array);
        tracing::trace!("ciphertext length: {}", ciphertext.len());

        self.session_states
            .read()
            .await
            .get(session_id)
            .ok_or_else(|| {
                tracing::error!("Session not found: {}", session_id);
                ApiError::Unauthorized
            })
            .and_then(|state| {
                state.decrypt(ciphertext, &nonce_array).map_err(|e| {
                    tracing::error!("Decryption failed: {:?}", e);
                    e
                })
            })
    }

    pub async fn encrypt_session_data(
        &self,
        session_id: &Uuid,
        data: &[u8],
    ) -> Result<Vec<u8>, ApiError> {
        let session_states = self.session_states.read().await;
        let session_state = session_states
            .get(session_id)
            .ok_or(ApiError::Unauthorized)?;

        let session_key = session_state.get_session_key();
        let key = Key::from_slice(session_key.as_ref());

        let nonce_bytes: [u8; 12] = crate::encrypt::generate_random();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let cipher = ChaCha20Poly1305::new(key);

        let mut encrypted_data = nonce_bytes.to_vec();
        encrypted_data.extend_from_slice(
            &cipher
                .encrypt(nonce, data)
                .map_err(|_| ApiError::InternalServerError)?,
        );

        Ok(encrypted_data)
    }

    async fn create_password_reset_request(
        &self,
        email: String,
        hashed_secret: String,
        project_id: i32,
    ) -> Result<String, Error> {
        let alphanumeric_code = self.generate_alphanumeric_code();

        // Check if the user exists
        match self.db.get_user_by_email(email.clone(), project_id) {
            Ok(user) => {
                // Only proceed with email if user has one
                if user.get_email().is_some() {
                    // User exists, proceed with the actual reset request
                    let secret_key = SecretKey::from_slice(&self.enclave_key)
                        .map_err(|e| Error::EncryptionError(e.to_string()))?;
                    let encrypted_code =
                        encrypt_key_deterministic(&secret_key, alphanumeric_code.as_bytes());

                    let new_request = NewPasswordResetRequest::new(
                        user.uuid,
                        hashed_secret,
                        encrypted_code,
                        24, // 24 hours expiration
                    );

                    self.db.create_password_reset_request(new_request)?;

                    // Send the actual email in the background
                    let app_state = self.clone();
                    let user_email = email.clone();
                    let code = alphanumeric_code.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            send_password_reset_email(&app_state, project_id, user_email, code)
                                .await
                        {
                            error!("Failed to send password reset email: {:?}", e);
                        }
                    });
                }
            }
            Err(DBError::UserNotFound) => {
                // User doesn't exist, but we don't want to reveal this information
                // So we'll just log it and return as if everything was successful
                debug!("Password reset requested for non-existent email: {}", email);
            }
            Err(e) => {
                // For other errors, we should still log them but not expose them to the user
                error!("Error during password reset request: {:?}", e);
            }
        }

        // Always return the generated code, even if we didn't actually create a request
        Ok(alphanumeric_code)
    }

    async fn confirm_password_reset(
        &self,
        email: String,
        alphanumeric_code: String,
        plaintext_secret: String,
        new_password: String,
        project_id: i32,
    ) -> Result<(), Error> {
        let user = self.db.get_user_by_email(email.clone(), project_id)?;

        // Verify user has an email
        if user.get_email().is_none() {
            return Err(Error::UserNotFound);
        }

        // Deterministically encrypt the provided alphanumeric code for lookup
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;
        let encrypted_code = encrypt_key_deterministic(&secret_key, alphanumeric_code.as_bytes());

        let reset_request = self
            .db
            .get_password_reset_request_by_user_id_and_code(user.uuid, encrypted_code)?;

        if let Some(reset_request) = reset_request {
            if reset_request.is_expired() {
                warn!("Password reset expired for user: {}", user.uuid);
                return Err(Error::PasswordResetExpired);
            }

            // Hash the plaintext secret again for comparison
            let hashed_plaintext = generate_reset_hash(plaintext_secret.clone());

            // Compare the hashed values using constant-time comparison
            if hashed_plaintext
                .as_bytes()
                .ct_eq(reset_request.hashed_secret.as_bytes())
                .into()
            {
                // Password verification succeeded, continue with reset
                self.update_user_password(&user, new_password).await?;
                self.db.mark_password_reset_as_complete(&reset_request)?;

                // Send confirmation email in the background
                let app_state = self.clone();
                let user_email = user.email.clone();
                tokio::spawn(async move {
                    if let Err(e) = send_password_reset_confirmation_email(
                        &app_state,
                        project_id,
                        user_email.expect("We checked email had to exist above"),
                    )
                    .await
                    {
                        error!("Failed to send password reset confirmation email: {:?}", e);
                    }
                });

                Ok(())
            } else {
                warn!(
                    "Password verification failed for user {}. Hashes do not match.",
                    user.uuid
                );
                Err(Error::InvalidPasswordResetSecret)
            }
        } else {
            Err(Error::InvalidPasswordResetRequest)
        }
    }

    fn generate_alphanumeric_code(&self) -> String {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        const CODE_LEN: usize = 8;

        let random_bytes: [u8; CODE_LEN] = generate_random();

        random_bytes
            .iter()
            .map(|&b| CHARSET[b as usize % CHARSET.len()] as char)
            .collect()
    }

    pub async fn create_platform_password_reset_request(
        &self,
        email: String,
        hashed_secret: String,
    ) -> Result<String, Error> {
        let alphanumeric_code = self.generate_alphanumeric_code();

        // Check if the platform user exists
        match self.db.get_platform_user_by_email(&email) {
            Ok(Some(platform_user)) => {
                // Only proceed with email if user has a password
                if platform_user.password_enc.is_some() {
                    // User exists, proceed with the actual reset request
                    let secret_key = SecretKey::from_slice(&self.enclave_key)
                        .map_err(|e| Error::EncryptionError(e.to_string()))?;
                    let encrypted_code =
                        encrypt_key_deterministic(&secret_key, alphanumeric_code.as_bytes());

                    let new_request = match NewPlatformPasswordResetRequest::new(
                        platform_user.uuid,
                        hashed_secret,
                        encrypted_code,
                        24, // 24 hours expiration
                    ) {
                        Ok(req) => req,
                        Err(e) => {
                            return Err(Error::EncryptionError(format!(
                                "Invalid expiration hours: {}",
                                e
                            )))
                        }
                    };

                    self.db
                        .create_platform_password_reset_request(new_request)?;

                    // Send the actual email in the background
                    let app_state = self.clone();
                    let user_email = email.clone();
                    let code = alphanumeric_code.clone();
                    spawn(async move {
                        if let Err(e) = send_platform_password_reset_email(
                            &app_state,
                            app_state.resend_api_key.clone(),
                            user_email,
                            code,
                        )
                        .await
                        {
                            error!("Failed to send platform password reset email: {:?}", e);
                        }
                    });
                }
            }
            Ok(None) => {
                // User doesn't exist, but we don't want to reveal this information
                // So we'll just log it and return as if everything was successful
                debug!(
                    "Password reset requested for non-existent platform email: {}",
                    email
                );
            }
            Err(e) => {
                // For other errors, we should still log them but not expose them to the user
                error!("Error during platform password reset request: {:?}", e);
            }
        }

        // Always return the generated code, even if we didn't actually create a request
        Ok(alphanumeric_code)
    }

    pub async fn confirm_platform_password_reset(
        &self,
        email: String,
        alphanumeric_code: String,
        plaintext_secret: String,
        new_password: String,
    ) -> Result<(), Error> {
        trace!("Confirm platform password reset for {email} with code {alphanumeric_code}");

        let platform_user = match self.db.get_platform_user_by_email(&email) {
            Ok(Some(user)) => user,
            Ok(None) => return Err(Error::UserNotFound),
            Err(_) => return Err(Error::UserNotFound),
        };

        // Deterministically encrypt the provided alphanumeric code for lookup
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;
        let encrypted_code = encrypt_key_deterministic(&secret_key, alphanumeric_code.as_bytes());

        let reset_request = self
            .db
            .get_platform_password_reset_request_by_user_id_and_code(
                platform_user.uuid,
                encrypted_code,
            )?;

        if let Some(reset_request) = reset_request {
            if reset_request.is_expired() {
                warn!(
                    "Platform password reset expired for user: {}",
                    platform_user.uuid
                );
                return Err(Error::PasswordResetExpired);
            }

            // Hash the plaintext secret again for comparison
            let hashed_plaintext = generate_reset_hash(plaintext_secret.clone());

            // Compare the hashed values using constant-time comparison
            if hashed_plaintext
                .as_bytes()
                .ct_eq(reset_request.hashed_secret.as_bytes())
                .into()
            {
                // Password verification succeeded, continue with reset
                // Hash the new password
                let password_hash = password_auth::generate_hash(new_password.clone());

                // Encrypt the hashed password
                let secret_key = SecretKey::from_slice(&self.enclave_key)
                    .map_err(|e| Error::EncryptionError(e.to_string()))?;
                let encrypted_password =
                    encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

                // Update the platform user's password
                self.db
                    .update_platform_user_password(&platform_user, encrypted_password)?;
                self.db
                    .mark_platform_password_reset_as_complete(&reset_request)?;

                // Send confirmation email in the background
                let user_email = platform_user.email.clone();
                let app_state = self.clone();
                spawn(async move {
                    if let Err(e) = send_platform_password_reset_confirmation_email(
                        &app_state,
                        app_state.resend_api_key.clone(),
                        user_email,
                    )
                    .await
                    {
                        error!(
                            "Failed to send platform password reset confirmation email: {:?}",
                            e
                        );
                    }
                });

                Ok(())
            } else {
                warn!(
                    "Password verification failed for platform user {}. Hashes do not match.",
                    platform_user.uuid
                );
                Err(Error::InvalidPasswordResetSecret)
            }
        } else {
            Err(Error::InvalidPasswordResetRequest)
        }
    }

    pub async fn update_user_password(
        &self,
        user: &User,
        new_password: String,
    ) -> Result<(), Error> {
        // Hash the new password
        let password_hash = password_auth::generate_hash(new_password);

        // Encrypt the hashed password
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;
        let encrypted_password = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

        // Update the user's password
        self.db
            .update_user_password(user, Some(encrypted_password))
            .map_err(Error::from)
    }

    pub async fn update_platform_user_password(
        &self,
        user: &PlatformUser,
        new_password: String,
    ) -> Result<(), Error> {
        // Hash the new password
        let password_hash = password_auth::generate_hash(new_password);

        // Encrypt the hashed password
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;
        let encrypted_password = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

        // Update the platform user's password
        self.db
            .update_platform_user_password(user, encrypted_password)
            .map_err(Error::from)
    }

    pub fn frontend_url(&self) -> String {
        self.app_mode.frontend_url().to_string()
    }

    pub fn oauth_callback_url(&self, provider: &str) -> Result<String, url::ParseError> {
        let base_url = Url::parse(self.frontend_url().as_str())?;
        Ok(base_url
            .join(&format!("/auth/{}/callback", provider))?
            .to_string())
    }

    pub async fn create_account_deletion_request(
        &self,
        user_id: Uuid,
        hashed_secret: String,
        project_id: i32,
    ) -> Result<(), Error> {
        // Generate a random UUID for the confirmation code
        let confirmation_code = Uuid::new_v4();

        // Check if the user exists
        match self.db.get_user_by_uuid(user_id) {
            Ok(user) => {
                // Only proceed if user belongs to the project
                if user.project_id != project_id {
                    return Err(Error::UserNotFound);
                }

                // User exists, proceed with the actual deletion request
                let secret_key = SecretKey::from_slice(&self.enclave_key)
                    .map_err(|e| Error::EncryptionError(e.to_string()))?;
                let encrypted_code =
                    encrypt_key_deterministic(&secret_key, confirmation_code.as_bytes());

                let new_request = NewAccountDeletionRequest::new(
                    user.uuid,
                    project_id,
                    hashed_secret,
                    encrypted_code,
                    24, // 24 hours expiration
                );

                self.db.create_account_deletion_request(new_request)?;

                // Send the account deletion email in the background
                let app_state = self.clone();
                let user_email = user.email.clone();
                let code = confirmation_code.to_string();
                tokio::spawn(async move {
                    if let Some(email) = user_email {
                        if let Err(e) =
                            send_account_deletion_email(&app_state, project_id, email, code).await
                        {
                            error!("Failed to send account deletion email: {:?}", e);
                        }
                    }
                });
            }
            Err(DBError::UserNotFound) => {
                // User doesn't exist, but we don't want to reveal this information
                // So we'll just log it and return as if everything was successful
                debug!(
                    "Account deletion requested for non-existent user: {}",
                    user_id
                );
            }
            Err(e) => {
                // For other errors, we should still log them but not expose them to the user
                error!("Error during account deletion request: {:?}", e);
            }
        }

        // Return Ok regardless of whether we actually created a request
        // The confirmation code is only delivered via email
        Ok(())
    }

    pub async fn confirm_account_deletion(
        &self,
        user_id: Uuid,
        confirmation_code: String,
        plaintext_secret: String,
    ) -> Result<(), Error> {
        let user = self.db.get_user_by_uuid(user_id)?;

        // Parse the confirmation code UUID from string
        let confirmation_code_uuid = match Uuid::parse_str(&confirmation_code) {
            Ok(uuid) => uuid,
            Err(_) => return Err(Error::InvalidAccountDeletionRequest),
        };

        // Deterministically encrypt the provided confirmation code for lookup
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;
        let encrypted_code =
            encrypt_key_deterministic(&secret_key, confirmation_code_uuid.as_bytes());

        let deletion_request = self
            .db
            .get_account_deletion_request_by_user_id_and_code(user.uuid, encrypted_code)?;

        if let Some(deletion_request) = deletion_request {
            // Since we're filtering by expiration_time at the database level,
            // this check serves as an additional validation and converts
            // the model-specific error to an application-level error
            if let Err(AccountDeletionError::RequestExpired) = deletion_request.is_expired() {
                warn!("Account deletion request expired for user: {}", user.uuid);
                return Err(Error::AccountDeletionExpired);
            }

            // Hash the plaintext secret again for comparison
            let hashed_plaintext = generate_reset_hash(plaintext_secret.clone());

            // Compare the hashed values using constant-time comparison
            if hashed_plaintext
                .as_bytes()
                .ct_eq(deletion_request.hashed_secret.as_bytes())
                .into()
            {
                // Secret verification succeeded, proceed with account deletion
                // Use the transaction-based method for atomicity
                self.db.mark_and_delete_user(&user, &deletion_request)?;

                // Send confirmation email in the background if the user has an email
                if let Some(email) = user.email.clone() {
                    let app_state = self.clone();
                    let project_id = user.project_id;
                    tokio::spawn(async move {
                        if let Err(e) =
                            send_account_deletion_confirmation_email(&app_state, project_id, email)
                                .await
                        {
                            error!(
                                "Failed to send account deletion confirmation email: {:?}",
                                e
                            );
                        }
                    });
                }

                Ok(())
            } else {
                warn!(
                    "Secret verification failed for user {}. Hashes do not match.",
                    user.uuid
                );
                Err(Error::InvalidAccountDeletionSecret)
            }
        } else {
            Err(Error::InvalidAccountDeletionRequest)
        }
    }

    async fn authenticate_platform_user(
        &self,
        email: &str,
        password: String,
    ) -> Result<Option<PlatformUser>, Error> {
        // Get the platform user
        let platform_user = match self.db.get_platform_user_by_email(email)? {
            Some(user) => user,
            None => {
                warn!("Could not find platform user by email: {email}");
                return Ok(None);
            }
        };

        // Check if this is an OAuth-only user (no password)
        if platform_user.password_enc.is_none() {
            error!("OAuth-only platform user attempted password login");
            return Ok(None);
        }

        let secret_key = SecretKey::from_slice(&self.enclave_key.clone())
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let decrypted_password_bytes =
            decrypt_with_key(&secret_key, platform_user.password_enc.as_ref().unwrap())
                .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let decrypted_password_hash = String::from_utf8(decrypted_password_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))?;

        // Verifying the password is blocking and potentially slow, so we'll do so via
        // `spawn_blocking`.
        let res = task::spawn_blocking(move || verify_password(password, &decrypted_password_hash))
            .await?;

        match res {
            Ok(_) => Ok(Some(platform_user)),
            Err(_) => Ok(None),
        }
    }

    pub async fn get_project_secret(
        &self,
        project_id: i32,
        key_name: &str,
    ) -> Result<Option<String>, Error> {
        // Get the encrypted secret
        let secret = match self
            .db
            .get_org_project_secret_by_key_name_and_project(key_name, project_id)?
        {
            Some(s) => s,
            None => return Ok(None),
        };

        // Decrypt the secret using the enclave key
        let secret_key = SecretKey::from_slice(&self.enclave_key)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        let decrypted_bytes = decrypt_with_key(&secret_key, &secret.secret_enc)
            .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Always return base64 encoded bytes
        Ok(Some(general_purpose::STANDARD.encode(&decrypted_bytes)))
    }

    pub async fn get_project_resend_api_key(
        &self,
        project_id: i32,
    ) -> Result<Option<String>, Error> {
        // Get the project's Resend API key
        self.get_project_secret(project_id, PROJECT_RESEND_API_KEY)
            .await
    }
}

async fn get_secret(key_name: &str) -> Result<String, Error> {
    let cid = 3;
    let port = 8003;

    let sock_addr = VsockAddr::new(cid, port);
    let mut stream = VsockStream::connect(&sock_addr)?;

    let request = EnclaveRequest {
        request_type: "SecretsManager".to_string(),
        key_name: Some(key_name.to_string()),
    };
    let request_json = serde_json::to_string(&request)?;
    stream.write_all(request_json.as_bytes())?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;

    let parent_response: ParentResponse = serde_json::from_str(&response)?;
    if parent_response.response_type == "secret" {
        let secret_json: Value =
            serde_json::from_str(parent_response.response_value.as_str().unwrap())?;

        // Assuming the secret is always a JSON object with a single key-value pair
        if let Some((_, value)) = secret_json.as_object().and_then(|obj| obj.iter().next()) {
            Ok(value.as_str().unwrap_or_default().to_string())
        } else {
            Err(Error::SecretParsingError)
        }
    } else {
        Err(Error::AuthenticationError)
    }
}

async fn get_or_create_enclave_key(
    app_mode: &AppMode,
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<GenKeyResult, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    let aws_kms_key_id = get_kms_key_id(app_mode);

    // Check if the key has been initialized before
    let existing_key = db.get_enclave_secret_by_key(ENCLAVE_KEY_NAME)?;

    let key_res = if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        // Decrypt the existing key
        let decrypted_key = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        GenKeyResult {
            key: decrypted_key,
            encrypted_key: encrypted_key.value.clone(),
        }
    } else {
        // Create a new encryption key
        create_new_encryption_key(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &aws_kms_key_id,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?
    };

    // Store the encrypted version of the key if it's new
    if existing_key.is_none() {
        let new_secret =
            NewEnclaveSecret::new(ENCLAVE_KEY_NAME.to_string(), key_res.encrypted_key.clone());
        db.create_enclave_secret(new_secret)?;
    }

    Ok(key_res)
}

async fn retrieve_openai_api_key(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<String, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(OPENAI_API_KEY_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted api key");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
    } else {
        Err(Error::EncryptionError(
            "OpenAI API key not found in the database".to_string(),
        ))
    }
}

async fn get_or_create_jwt_secret(
    app_mode: &AppMode,
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
    enclave_key: &[u8],
) -> Result<Vec<u8>, Error> {
    match app_mode {
        AppMode::Local => {
            // For local mode, use environment variable
            Ok(std::env::var("JWT_SECRET")
                .expect("JWT_SECRET must be set in local mode")
                .into_bytes())
        }
        _ => {
            // Check if JWT secret exists in enclave_secrets
            if let Some(encrypted_jwt_secret) = db.get_enclave_secret_by_key(JWT_SECRET_KEY_NAME)? {
                // Decrypt existing JWT secret
                let secret_key = SecretKey::from_slice(enclave_key)
                    .map_err(|e| Error::EncryptionError(e.to_string()))?;
                decrypt_with_key(&secret_key, &encrypted_jwt_secret.value)
                    .map_err(|e| Error::EncryptionError(e.to_string()))
            } else {
                // Generate new JWT secret
                let jwt_secret = jwt::generate_jwt_secret(aws_credential_manager.clone()).await?;

                // Encrypt and store the new JWT secret
                let secret_key = SecretKey::from_slice(enclave_key)
                    .map_err(|e| Error::EncryptionError(e.to_string()))?;
                let encrypted_jwt_secret = encrypt_with_key(&secret_key, &jwt_secret).await;

                let new_secret =
                    NewEnclaveSecret::new(JWT_SECRET_KEY_NAME.to_string(), encrypted_jwt_secret);
                db.create_enclave_secret(new_secret)?;

                Ok(jwt_secret)
            }
        }
    }
}

fn get_kms_key_id(app_mode: &AppMode) -> String {
    match app_mode {
        AppMode::Prod => "alias/open-secret-prod-enclave".to_string(),
        AppMode::Preview => "alias/open-secret-preview1-enclave".to_string(),
        AppMode::Dev => "alias/open-secret-dev-enclave".to_string(),
        AppMode::Custom(env_name) => format!("alias/open-secret-{}-enclave", env_name),
        AppMode::Local => unreachable!("shouldn't use kms in local mode"),
    }
}

fn is_default_openai_domain(domain: &str) -> bool {
    domain.contains("openai.com")
}

async fn retrieve_resend_api_key(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(RESEND_API_KEY_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted Resend API key");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("Resend API key not found in the database");
        Ok(None)
    }
}

async fn retrieve_github_client_id(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(GITHUB_CLIENT_ID_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted GitHub client ID");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("GitHub client ID not found in the database");
        Ok(None)
    }
}

async fn retrieve_github_client_secret(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(GITHUB_CLIENT_SECRET_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted GitHub client secret");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("GitHub client secret not found in the database");
        Ok(None)
    }
}

async fn retrieve_google_client_secret(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(GOOGLE_CLIENT_SECRET_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted Google client secret");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("Google client secret not found in the database");
        Ok(None)
    }
}

async fn retrieve_google_client_id(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(GOOGLE_CLIENT_ID_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted Google client ID");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("Google client ID not found in the database");
        Ok(None)
    }
}

struct AsyncRngWrapper {
    inner: CustomRng,
}

impl AsyncRngWrapper {
    fn new(inner: CustomRng) -> Self {
        AsyncRngWrapper { inner }
    }
}

impl RngCore for AsyncRngWrapper {
    fn next_u32(&mut self) -> u32 {
        futures::executor::block_on(self.inner.next_u32())
    }

    fn next_u64(&mut self) -> u64 {
        futures::executor::block_on(self.inner.next_u64())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        futures::executor::block_on(self.inner.fill_bytes(dest))
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        futures::executor::block_on(self.inner.fill_bytes(dest));
        Ok(())
    }
}

// Implement CryptoRng for AsyncRngWrapper
impl CryptoRng for AsyncRngWrapper {}

pub fn generate_reset_hash(password: String) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    format!("{:x}", hasher.finalize())
}

async fn retrieve_billing_api_key(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the key already exists in the db
    let existing_key = db.get_enclave_secret_by_key(BILLING_API_KEY_NAME)?;

    if let Some(ref encrypted_key) = existing_key {
        // Convert the stored bytes back to base64
        let base64_encrypted_key = general_purpose::STANDARD.encode(&encrypted_key.value);

        debug!("trying to decrypt base64 encrypted billing API key");

        // Decrypt the existing key
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_key,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("Billing API key not found in the database");
        Ok(None)
    }
}

async fn retrieve_billing_server_url(
    aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
) -> Result<Option<String>, Error> {
    let creds = aws_credential_manager
        .read()
        .await
        .clone()
        .expect("non-local mode should have creds")
        .get_credentials()
        .await
        .expect("non-local mode should have creds");

    // check if the url already exists in the db
    let existing_url = db.get_enclave_secret_by_key(BILLING_SERVER_URL_NAME)?;

    if let Some(ref encrypted_url) = existing_url {
        // Convert the stored bytes back to base64
        let base64_encrypted_url = general_purpose::STANDARD.encode(&encrypted_url.value);

        debug!("trying to decrypt base64 encrypted billing server URL");

        // Decrypt the existing url
        let decrypted_bytes = decrypt_with_kms(
            &creds.region,
            &creds.access_key_id,
            &creds.secret_access_key,
            &creds.token,
            &base64_encrypted_url,
        )
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

        // Convert the decrypted bytes to a UTF-8 string
        String::from_utf8(decrypted_bytes)
            .map_err(|e| Error::EncryptionError(format!("Failed to decode UTF-8: {}", e)))
            .map(Some)
    } else {
        tracing::info!("Billing server URL not found in the database");
        Ok(None)
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Add debug logs for entrypoints and exit points
    tracing::debug!("Starting application");

    // Load .env file
    dotenv::dotenv().ok();

    let app_mode = std::env::var("APP_MODE")
        .unwrap_or_else(|_| "local".to_string())
        .parse::<AppMode>()
        .expect("Invalid APP_MODE");

    tracing_subscriber::registry()
        .with(EnvFilter::new(std::env::var("RUST_LOG").unwrap_or_else(
            |_| {
                "sg_backend=debug,axum_login=debug,tower_sessions=debug,sqlx=warn,tower_http=debug"
                    .into()
            },
        )))
        .with(tracing_subscriber::fmt::layer().with_ansi(false))
        .try_init()?;

    let aws_credential_manager = if app_mode != AppMode::Local {
        Arc::new(RwLock::new(Some(AwsCredentialManager::new())))
    } else {
        Arc::new(RwLock::new(None))
    };

    if app_mode != AppMode::Local {
        // Wait for initial credentials with a timeout
        let timeout = Duration::from_secs(60); // 1 minute timeout
        match tokio::time::timeout(
            timeout,
            aws_credential_manager
                .read()
                .await
                .as_ref()
                .expect("non-local mode should have creds")
                .wait_for_credentials(),
        )
        .await
        {
            Ok(_) => tracing::info!("Initial AWS credentials fetched successfully"),
            Err(_) => {
                tracing::error!("Timed out waiting for initial AWS credentials");
                return Err(Error::AwsCredentialError(AwsCredentialError::Timeout));
            }
        }

        // Spawn a task to refresh AWS credentials
        let refresh_manager = aws_credential_manager.clone();
        tokio::spawn(async move {
            let refresh_interval = Duration::from_secs(5 * 60 * 60); // 5 hours
            let retry_interval = Duration::from_secs(5); // 5 seconds

            loop {
                tracing::info!("Refreshing AWS credentials");

                let fetch_res = refresh_manager
                    .read()
                    .await
                    .as_ref()
                    .expect("non-local mode should have creds")
                    .fetch_credentials()
                    .await;

                match fetch_res {
                    Ok(_creds) => {
                        tracing::info!("AWS credentials refreshed successfully");
                        tokio::time::sleep(refresh_interval).await;
                    }
                    Err(e) => {
                        tracing::error!("Failed to refresh AWS credentials: {:?}", e);
                        tracing::info!("Retrying in 5 seconds...");
                        tokio::time::sleep(retry_interval).await;
                    }
                }
            }
        });
    }

    let pg_url = if app_mode != AppMode::Local {
        // Fetch database URL from Secrets Manager using the new AwsCredentialManager
        let secret_name = match app_mode {
            AppMode::Prod => "opensecret_prod_database_url",
            AppMode::Preview => "opensecret_preview1_database_url",
            AppMode::Dev => "opensecret_dev_database_url",
            AppMode::Custom(ref env_name) => {
                let name = format!("opensecret_{}_database_url", env_name);
                Box::leak(name.into_boxed_str())
            }
            AppMode::Local => unreachable!("just checked"),
        };
        match get_secret(secret_name).await {
            Ok(encrypted_url) => {
                let creds = aws_credential_manager
                    .read()
                    .await
                    .clone()
                    .expect("non-local mode should have creds")
                    .get_credentials()
                    .await
                    .expect("should have just waited for credentials");

                tracing::info!("Retrieved and decrypting database URL from Secrets Manager");
                let url_vec = decrypt_with_kms(
                    &creds.region,
                    &creds.access_key_id,
                    &creds.secret_access_key,
                    &creds.token,
                    &encrypted_url,
                )
                .map_err(|e| {
                    tracing::error!("Failed to decrypt database URL: {:?}", e);
                    Error::EncryptionError(e.to_string())
                })?;

                String::from_utf8(url_vec).expect("should parse url")
            }
            Err(e) => {
                tracing::error!(
                    "Failed to retrieve database URL from Secrets Manager: {:?}",
                    e
                );
                return Err(e);
            }
        }
    } else {
        std::env::var("DATABASE_URL").expect("DATABASE_URL must be set")
    };

    let db = setup_db(pg_url);

    // enclave secret retrieval
    let enclave_key = if app_mode != AppMode::Local {
        let enclave_key =
            get_or_create_enclave_key(&app_mode, aws_credential_manager.clone(), db.clone())
                .await?;
        enclave_key.key
    } else {
        let enclave_key =
            std::env::var("ENCLAVE_SECRET_MOCK").expect("needs ENCLAVE_SECRET_MOCK in local mode");
        let enclave_key: [u8; 32] = hex::decode(enclave_key)
            .unwrap()
            .try_into()
            .expect("ENCLAVE_SECRET_MOCK must be 32 bytes");
        enclave_key.to_vec()
    };

    let openai_api_base =
        env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com".to_string());

    let openai_api_key = if is_default_openai_domain(&openai_api_base) {
        if app_mode != AppMode::Local {
            Some(
                retrieve_openai_api_key(aws_credential_manager.clone(), db.clone())
                    .await
                    .expect("OpenAI API key should be retrieved correctly"),
            )
        } else {
            Some(
                std::env::var("OPENAI_API_KEY")
                    .expect("OPENAI_API_KEY must be set for OpenAI domain"),
            )
        }
    } else {
        None // No API key needed if not using OpenAI's domain
    };

    // Tinfoil API base is always from environment (like OpenAI API base)
    let tinfoil_api_base = env::var("TINFOIL_API_BASE").ok();

    let jwt_secret = get_or_create_jwt_secret(
        &app_mode,
        aws_credential_manager.clone(),
        db.clone(),
        &enclave_key,
    )
    .await?;

    let resend_api_key = if app_mode != AppMode::Local {
        retrieve_resend_api_key(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("RESEND_API_KEY").ok()
    };

    let github_client_secret = if app_mode != AppMode::Local {
        retrieve_github_client_secret(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("GITHUB_CLIENT_SECRET").ok()
    };

    let github_client_id = if app_mode != AppMode::Local {
        retrieve_github_client_id(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("GITHUB_CLIENT_ID").ok()
    };

    let google_client_secret = if app_mode != AppMode::Local {
        retrieve_google_client_secret(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("GOOGLE_CLIENT_SECRET").ok()
    };

    let google_client_id = if app_mode != AppMode::Local {
        retrieve_google_client_id(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("GOOGLE_CLIENT_ID").ok()
    };

    let sqs_queue_maple_events_url = if app_mode != AppMode::Local {
        // Get from database if in enclave mode
        if let Some(ref encrypted_url) =
            db.get_enclave_secret_by_key("sqs_queue_maple_events_url")?
        {
            let creds = aws_credential_manager
                .read()
                .await
                .clone()
                .expect("non-local mode should have creds")
                .get_credentials()
                .await
                .expect("should have just waited for credentials");

            // Decrypt the URL
            let url_vec = decrypt_with_kms(
                &creds.region,
                &creds.access_key_id,
                &creds.secret_access_key,
                &creds.token,
                &general_purpose::STANDARD.encode(&encrypted_url.value),
            )
            .map_err(|e| {
                tracing::error!("Failed to decrypt SQS queue URL: {:?}", e);
                Error::EncryptionError(e.to_string())
            })?;

            Some(String::from_utf8(url_vec).expect("should parse url"))
        } else {
            // URL not found in database - this is optional so we'll return None
            None
        }
    } else {
        // In local mode, get from environment variable
        std::env::var("SQS_QUEUE_MAPLE_EVENTS_URL").ok()
    };

    let billing_api_key = if app_mode != AppMode::Local {
        // Get from database if in enclave mode
        retrieve_billing_api_key(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("BILLING_API_KEY").ok()
    };

    let billing_server_url = if app_mode != AppMode::Local {
        // Get from database if in enclave mode
        retrieve_billing_server_url(aws_credential_manager.clone(), db.clone()).await?
    } else {
        std::env::var("BILLING_SERVER_URL").ok()
    };

    let app_state = AppStateBuilder::default()
        .app_mode(app_mode.clone())
        .db(db)
        .enclave_key(enclave_key)
        .aws_credential_manager(aws_credential_manager)
        .openai_api_key(openai_api_key)
        .openai_api_base(openai_api_base)
        .tinfoil_api_base(tinfoil_api_base)
        .jwt_secret(jwt_secret)
        .resend_api_key(resend_api_key)
        .github_client_secret(github_client_secret.clone())
        .github_client_id(github_client_id.clone())
        .google_client_secret(google_client_secret.clone())
        .google_client_id(google_client_id.clone())
        .sqs_queue_maple_events_url(sqs_queue_maple_events_url)
        .billing_api_key(billing_api_key)
        .billing_server_url(billing_server_url)
        .build()
        .await?;
    tracing::info!("App state created, app_mode: {:?}", app_mode);

    let app_state = Arc::new(app_state);

    // Run migrations before starting the server
    migrations::run_migrations(
        &app_state,
        github_client_secret.clone(),
        google_client_secret.clone(),
        github_client_id.clone(),
        google_client_id.clone(),
    )
    .await?;

    let cors = CorsLayer::new()
        // allow all method types
        .allow_methods(Any)
        // allow all headers
        .allow_headers(Any)
        // allow requests from any origin
        .allow_origin(Any);

    let app = protected_routes(app_state.clone())
        .route_layer(from_fn_with_state(app_state.clone(), validate_jwt))
        .merge(health_routes())
        .merge(login_routes(app_state.clone()))
        .merge(
            openai_routes(app_state.clone())
                .route_layer(from_fn_with_state(app_state.clone(), validate_jwt)),
        )
        .merge(attestation_routes::router(app_state.clone()))
        .merge(oauth_routes(app_state.clone()))
        .merge(platform_login_routes(app_state.clone()))
        .merge(
            platform_routes(app_state.clone())
                .route_layer(from_fn_with_state(app_state.clone(), validate_platform_jwt)),
        )
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    tracing::info!("Listening on http://localhost:3000");

    Ok(axum::serve(listener, app.into_make_service()).await?)
}
