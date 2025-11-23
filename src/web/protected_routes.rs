use crate::encrypt;
use crate::jwt::{NewToken, TokenType};
use crate::message_signing::SigningAlgorithm;
use crate::private_key::{decrypt_user_seed_to_mnemonic, VALID_BIP39_WORD_COUNTS};
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::Error;
use crate::KVPair;
use crate::{
    db::DBError, email::send_verification_email, models::email_verification::NewEmailVerification,
    models::users::User, ApiError, AppState,
};

// BIP-85 constants
const BIP85_PURPOSE: u32 = 83696968; // BIP-85 purpose value
const BIP85_APPLICATION_BIP39: u32 = 39; // Application number for BIP-39 mnemonics
use axum::middleware::from_fn_with_state;
use axum::{
    extract::{Path, Query, State},
    routing::{delete, get, post, put},
    Router,
};
use axum::{Extension, Json};
use base64::{engine::general_purpose, Engine as _};
use bitcoin::bip32::DerivationPath;
use chrono::{DateTime, Utc};
use secp256k1::Secp256k1;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::str::FromStr;
use std::sync::Arc;
use tokio::spawn;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoginMethod {
    Email,
    Github,
    Google,
    Apple,
    Guest,
}

// Update AppUser struct to include login_method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUser {
    pub id: Uuid,
    pub name: Option<String>,
    pub email: Option<String>,
    pub email_verified: bool,
    pub login_method: LoginMethod,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<&User> for AppUser {
    fn from(user: &User) -> Self {
        AppUser {
            id: user.uuid,
            name: user.name.clone(),
            email: user.email.clone(),
            // This will be set separately
            email_verified: false,
            // This will be updated for oauth
            login_method: if user.is_guest() {
                LoginMethod::Guest
            } else {
                LoginMethod::Email
            },
            created_at: user.created_at,
            updated_at: user.updated_at,
        }
    }
}

#[derive(Serialize)]
pub struct ProtectedUserData {
    pub user: AppUser,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChangePasswordRequest {
    pub current_password: String,
    pub new_password: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InitiateAccountDeletionRequest {
    pub hashed_secret: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfirmAccountDeletionRequest {
    pub confirmation_code: String,
    pub plaintext_secret: String,
}

#[derive(Debug, Serialize)]
pub struct PrivateKeyResponse {
    /// Root mnemonic or derived mnemonic if BIP-85 path is specified
    mnemonic: String,
}

/// Response struct for the private key bytes endpoint.
/// Contains the private key encoded as a hexadecimal string.
#[derive(Debug, Serialize)]
pub struct PrivateKeyBytesResponse {
    /// The private key as a 64-character hexadecimal string (32 bytes).
    /// This is the standard secp256k1 private key format.
    private_key: String,
}

/// Structure for key derivation options.
/// Supports both BIP-32 and BIP-85 derivation paths.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct KeyOptions {
    /// BIP-85 derivation path to derive a child seed phrase
    /// Format: "m/83696968'/39'/0'/12'/0'" where:
    ///   - 83696968' is the BIP-85 purpose
    ///   - 39' is for BIP-39 mnemonics
    ///   - 0' is the coin type
    ///   - 12', 18', or 24' is the word count (only these word counts are supported)
    ///   - 0' is the index
    pub seed_phrase_derivation_path: Option<String>,

    /// BIP-32 derivation path for deriving a private key
    /// Format: "m/44'/0'/0'/0/0"
    pub private_key_derivation_path: Option<String>,
}

impl KeyOptions {
    /// Validates both derivation paths if present.
    pub fn validate(&self) -> Result<(), ApiError> {
        debug!("Validating key options");

        // Validate private_key_derivation_path if present
        if let Some(ref path) = self.private_key_derivation_path {
            debug!("Validating BIP-32 derivation path: {}", path);
            validate_path(path)?;
            info!("BIP-32 derivation path validation successful: {}", path);
        }

        // Validate seed_phrase_derivation_path if present
        if let Some(ref path) = self.seed_phrase_derivation_path {
            debug!("Validating BIP-85 derivation path: {}", path);
            validate_path(path)?;
            validate_bip85_path(path)?;
            info!("BIP-85 derivation path validation successful: {}", path);
        }

        debug!("Key options validation completed successfully");
        Ok(())
    }
}

/// Validates a BIP-85 path to ensure it follows the correct format and contains valid values.
/// Format: m/PURPOSE'/APP'/LANGUAGE'/WORDS'/INDEX' where:
/// - PURPOSE' is the BIP-85 purpose value (BIP85_PURPOSE = 83696968', must be hardened)
/// - APP' is for BIP-39 mnemonics (BIP85_APPLICATION_BIP39 = 39', must be hardened)
/// - LANGUAGE' is typically 0' for English (must be hardened)
/// - WORDS' must be one of VALID_BIP39_WORD_COUNTS (12', 18', 24', must be hardened)
/// - INDEX' is the derivation index (must be hardened)
pub fn validate_bip85_path(path: &str) -> Result<(), ApiError> {
    debug!("Validating BIP-85 path: {}", path);

    // Split the path into segments
    let segments: Vec<&str> = path.split('/').collect();

    // Path must have exactly 6 segments: m/PURPOSE'/APP'/LANGUAGE'/WORDS'/INDEX'
    if segments.len() != 6 {
        error!(
            "BIP-85 path must have exactly 6 segments, found {}: {}",
            segments.len(),
            path
        );
        return Err(ApiError::BadRequest);
    }

    // First segment should be "m" (master key)
    if segments[0] != "m" {
        error!(
            "BIP-85 path must start with 'm', found '{}': {}",
            segments[0], path
        );
        return Err(ApiError::BadRequest);
    }

    // Check purpose is BIP85_PURPOSE (83696968')
    let purpose = segments[1];
    if !purpose.starts_with(&BIP85_PURPOSE.to_string())
        || !(purpose.ends_with('\'') || purpose.ends_with('h'))
    {
        error!(
            "BIP-85 path purpose must be {}' or {}h, found '{}': {}",
            BIP85_PURPOSE, BIP85_PURPOSE, purpose, path
        );
        return Err(ApiError::BadRequest);
    }

    // Check application is BIP85_APPLICATION_BIP39 (39' for BIP-39 mnemonic)
    let application = segments[2];
    if !application.starts_with(&BIP85_APPLICATION_BIP39.to_string())
        || !(application.ends_with('\'') || application.ends_with('h'))
    {
        error!(
            "BIP-85 path application must be {}' or {}h for BIP-39 mnemonics, found '{}': {}",
            BIP85_APPLICATION_BIP39, BIP85_APPLICATION_BIP39, application, path
        );
        return Err(ApiError::BadRequest);
    }

    // Check language is hardened
    let language = segments[3];
    if !(language.ends_with('\'') || language.ends_with('h')) {
        error!(
            "BIP-85 path language must be hardened, found '{}': {}",
            language, path
        );
        return Err(ApiError::BadRequest);
    }

    debug!("BIP-85 path language component valid: {}", language);

    // Check word count is valid (12, 18, 24) and hardened
    let word_count = segments[4];
    let word_count_value = match word_count.trim_end_matches(&['\'', 'h'][..]).parse::<u32>() {
        Ok(value) => value,
        Err(_) => {
            error!(
                "BIP-85 path word count must be a number, found '{}': {}",
                word_count, path
            );
            return Err(ApiError::BadRequest);
        }
    };

    // Validate word count is one of the valid options
    if !VALID_BIP39_WORD_COUNTS.contains(&word_count_value) {
        error!(
            "BIP-85 path word count must be one of {:?}, found {}: {}",
            VALID_BIP39_WORD_COUNTS, word_count_value, path
        );
        return Err(ApiError::BadRequest);
    }

    debug!("BIP-85 path word count valid: {} words", word_count_value);

    // Check word count is hardened
    if !(word_count.ends_with('\'') || word_count.ends_with('h')) {
        error!(
            "BIP-85 path word count must be hardened, found '{}': {}",
            word_count, path
        );
        return Err(ApiError::BadRequest);
    }

    // Check index is hardened
    let index = segments[5];
    if !(index.ends_with('\'') || index.ends_with('h')) {
        error!(
            "BIP-85 path index must be hardened, found '{}': {}",
            index, path
        );
        return Err(ApiError::BadRequest);
    }

    // Try to parse the index to ensure it's a valid number
    let index_value = match index.trim_end_matches(&['\'', 'h'][..]).parse::<u32>() {
        Ok(value) => value,
        Err(_) => {
            error!(
                "BIP-85 path index must be a valid number, found '{}': {}",
                index, path
            );
            return Err(ApiError::BadRequest);
        }
    };

    debug!("BIP-85 path index valid: {}", index_value);
    info!("BIP-85 path validation successful: {}", path);

    Ok(())
}

/// Query parameters for endpoints that accept a derivation path.
/// The derivation path should follow BIP32 format (e.g., "m/44'/0'/0'/0/0").
#[derive(Debug, Clone, Deserialize)]
pub struct DerivationPathQuery {
    #[serde(default, flatten)]
    pub key_options: KeyOptions,
}

impl DerivationPathQuery {
    /// Validates that the derivation paths follow BIP32 and BIP-85 format if present.
    pub fn validate(&self) -> Result<(), ApiError> {
        // Use the KeyOptions validation
        self.key_options.validate()
    }
}

/// Helper function to validate a derivation path
fn validate_path(path: &str) -> Result<(), ApiError> {
    // Allow empty path or "m" alone
    if path.is_empty() || path == "m" {
        return Ok(());
    }

    // For non-empty paths, validate using bitcoin library's DerivationPath
    DerivationPath::from_str(path).map_err(|e| {
        error!("Invalid derivation path format: {}", e);
        ApiError::BadRequest
    })?;

    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
pub struct SignMessageRequest {
    pub message_base64: String,
    pub algorithm: SigningAlgorithm,
    #[serde(default)]
    pub key_options: KeyOptions,
}

#[derive(Debug, Serialize)]
pub struct SignMessageResponseJson {
    pub signature: String,
    pub message_hash: String,
}

#[derive(Debug, Serialize)]
pub struct PublicKeyResponseJson {
    pub public_key: String,
    pub algorithm: SigningAlgorithm,
}

#[derive(Debug, Deserialize)]
pub struct PublicKeyQuery {
    algorithm: SigningAlgorithm,
    #[serde(default, flatten)]
    key_options: KeyOptions,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConvertGuestRequest {
    pub email: String,
    pub password: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThirdPartyTokenRequest {
    pub audience: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ThirdPartyTokenResponse {
    pub token: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncryptDataRequest {
    pub data: String,
    #[serde(default)]
    pub key_options: KeyOptions,
}

#[derive(Debug, Serialize)]
pub struct EncryptDataResponse {
    pub encrypted_data: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecryptDataRequest {
    pub encrypted_data: String,
    #[serde(default)]
    pub key_options: KeyOptions,
}

// API Key Management Types

// Validation function for API key names
// More restrictive than project names since they're used in URL paths
fn validate_api_key_name(name: &str) -> Result<(), validator::ValidationError> {
    use validator::ValidationError;

    // First check for leading/trailing whitespace
    if name != name.trim() {
        return Err(
            ValidationError::new("whitespace").with_message(std::borrow::Cow::Borrowed(
                "Name cannot have leading or trailing whitespace",
            )),
        );
    }

    // After confirming no leading/trailing space, check length
    if name.is_empty() || name.len() > 50 {
        return Err(ValidationError::new("length")
            .with_message(std::borrow::Cow::Borrowed("Name must be 1-50 characters")));
    }

    // Check for valid characters (ASCII only for URL safety)
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '_')
    {
        return Err(ValidationError::new("invalid_characters").with_message(
            std::borrow::Cow::Borrowed(
                "Only ASCII alphanumeric characters, spaces, hyphens, and underscores are allowed",
            ),
        ));
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CreateApiKeyRequest {
    #[validate(length(min = 1, max = 50))]
    #[validate(custom(function = "validate_api_key_name"))]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateApiKeyResponse {
    pub key: String, // UUID format with dashes - only returned on creation
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListApiKeysResponse {
    pub keys: Vec<ApiKeyInfo>,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/protected/user",
            get(user_protected).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/kv/:key",
            get(get_kv).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/kv/:key",
            put(put_kv).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<String>,
            )),
        )
        .route(
            "/protected/kv/:key",
            delete(delete_kv).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/kv",
            get(list_kv).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/kv",
            delete(delete_all_kv)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/request_verification",
            post(request_new_verification_code)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/change_password",
            post(change_password).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<ChangePasswordRequest>,
            )),
        )
        .route(
            "/protected/delete-account/request",
            post(initiate_account_deletion).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<InitiateAccountDeletionRequest>,
            )),
        )
        .route(
            "/protected/delete-account/confirm",
            post(confirm_account_deletion).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<ConfirmAccountDeletionRequest>,
            )),
        )
        .route(
            "/protected/private_key",
            get(get_private_key)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/private_key_bytes",
            get(get_private_key_bytes)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/sign_message",
            post(sign_message).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<SignMessageRequest>,
            )),
        )
        .route(
            "/protected/public_key",
            get(get_public_key).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/convert_guest",
            post(convert_guest_to_email).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<ConvertGuestRequest>,
            )),
        )
        .route(
            "/protected/third_party_token",
            post(generate_third_party_token).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<ThirdPartyTokenRequest>,
            )),
        )
        .route(
            "/protected/encrypt",
            post(encrypt_data).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<EncryptDataRequest>,
            )),
        )
        .route(
            "/protected/decrypt",
            post(decrypt_data).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<DecryptDataRequest>,
            )),
        )
        // API Key management endpoints
        .route(
            "/protected/api-keys",
            post(create_api_key).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<CreateApiKeyRequest>,
            )),
        )
        .route(
            "/protected/api-keys",
            get(list_api_keys).layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/protected/api-keys/:name",
            delete(delete_api_key)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state.clone())
}

pub async fn user_protected(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ProtectedUserData>>, ApiError> {
    debug!("Entering user_protected function");
    tracing::info!("user_protected request");
    let mut app_user: AppUser = AppUser::from(&user);

    // Set email verification status - only if not a guest user
    app_user.email_verified = if user.is_guest() {
        false
    } else {
        match data.db.get_email_verification_by_user_id(user.uuid) {
            Ok(verification) => verification.is_verified,
            Err(DBError::EmailVerificationNotFound) => false,
            Err(e) => {
                tracing::error!("Error checking email verification: {:?}", e);
                return Err(ApiError::InternalServerError);
            }
        }
    };

    // Determine login method
    if user.password_enc.is_none() {
        // This is an OAuth user - find out which provider
        match data.db.get_all_user_oauth_connections_for_user(user.uuid) {
            Ok(connections) => {
                if let Some(connection) = connections.first() {
                    // Get the provider details
                    match data.db.get_oauth_provider_by_id(connection.provider_id) {
                        Ok(Some(provider)) => {
                            // Set the login method based on the provider name
                            app_user.login_method = match provider.name.as_str() {
                                "github" => LoginMethod::Github,
                                "google" => LoginMethod::Google,
                                "apple" => LoginMethod::Apple,
                                // Add other providers here as they're supported
                                _ => {
                                    tracing::error!("Unknown OAuth provider: {}", provider.name);
                                    return Err(ApiError::InternalServerError);
                                }
                            };
                        }
                        Ok(None) => {
                            tracing::error!(
                                "OAuth provider not found for id: {}",
                                connection.provider_id
                            );
                            return Err(ApiError::InternalServerError);
                        }
                        Err(e) => {
                            tracing::error!("Error fetching OAuth provider: {:?}", e);
                            return Err(ApiError::InternalServerError);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Error fetching OAuth connections: {:?}", e);
                return Err(ApiError::InternalServerError);
            }
        }
    }

    let response = ProtectedUserData { user: app_user };

    debug!("Exiting user_protected function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn get_kv(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Path(key): Path<String>,
) -> Result<Json<EncryptedResponse<Option<String>>>, ApiError> {
    debug!("Entering get_kv function");

    let value = match data.get(user.uuid, key).await {
        Ok(kv) => kv,
        Err(e) => {
            tracing::error!("Error getting key-value pair: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };
    debug!("Exiting get_kv function");
    encrypt_response(&data, &session_id, &value).await
}

pub async fn put_kv(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Path(key): Path<String>,
    Extension(value): Extension<String>,
) -> Result<Json<EncryptedResponse<String>>, ApiError> {
    debug!("Entering put_kv function");
    info!("Putting key-value pair for user");
    tracing::trace!("putting key-value pair: {} = {}", key, value);

    match data.put(user.uuid, key, value.clone()).await {
        Ok(kv) => kv,
        Err(e) => {
            tracing::error!("Error putting key-value pair: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    debug!("Exiting put_kv function");
    encrypt_response(&data, &session_id, &value).await
}

pub async fn delete_kv(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Path(key): Path<String>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering delete_kv function");

    match data.delete(user.uuid, key).await {
        Ok(_) => {
            let response = json!({ "message": "Resource deleted successfully" });
            debug!("Exiting delete_kv function");
            encrypt_response(&data, &session_id, &response).await
        }
        Err(e) => {
            tracing::error!("Error deleting key-value pair: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

pub async fn delete_all_kv(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering delete_all_kv function");
    info!("Deleting all key-value pairs for user");

    match data.delete_all(user.uuid).await {
        Ok(_) => {
            let response = json!({
                "message": "All key-value pairs deleted successfully"
            });
            debug!("Exiting delete_all_kv function");
            encrypt_response(&data, &session_id, &response).await
        }
        Err(e) => {
            tracing::error!("Error deleting all key-value pairs: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

pub async fn list_kv(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<Vec<KVPair>>>, ApiError> {
    debug!("Entering list_kv function");

    let kvs = match data.list(user.uuid).await {
        Ok(kvs) => kvs,
        Err(e) => {
            tracing::error!("Error listing key-value pairs: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };
    debug!("Exiting list_kv function");
    encrypt_response(&data, &session_id, &kvs).await
}

pub async fn request_new_verification_code(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering request_new_verification_code function");

    // First check if user has an email
    let email = match user.get_email() {
        Some(email) => email.to_string(),
        None => {
            let response = json!({ "error": "No email associated with account" });
            return encrypt_response(&data, &session_id, &response).await;
        }
    };

    // Check if the user is already verified
    match data.db.get_email_verification_by_user_id(user.uuid) {
        Ok(verification) => {
            if verification.is_verified {
                let response = json!({ "error": "User is already verified" });
                return encrypt_response(&data, &session_id, &response).await;
            }
            // Delete the old verification
            if let Err(e) = data.db.delete_email_verification(&verification) {
                tracing::error!("Error deleting old verification: {:?}", e);
                return Err(ApiError::InternalServerError);
            }
        }
        Err(DBError::EmailVerificationNotFound) => {
            // This is fine, we'll create a new verification
        }
        Err(e) => {
            tracing::error!("Error checking email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    }

    // Create a new verification entry
    let new_verification = NewEmailVerification::new(user.uuid, 24, false); // 24 hours expiration
    let verification = match data.db.create_email_verification(new_verification) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Error creating email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    // Send the new verification email
    if let Err(e) = send_verification_email(
        &data,
        user.project_id,
        email,
        verification.verification_code,
    )
    .await
    {
        tracing::error!("Error sending verification email: {:?}", e);
        return Err(ApiError::InternalServerError);
    }

    let response = json!({ "message": "New verification code sent successfully" });
    debug!("Exiting request_new_verification_code function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn change_password(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(change_request): Extension<ChangePasswordRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering change_password function");

    // Check if user is an OAuth-only user
    if user.password_enc.is_none() {
        error!("OAuth-only user attempted to change password");
        return Err(ApiError::InvalidUsernameOrPassword);
    }

    // Get email if it exists
    let email = user.get_email().map(|e| e.to_string());

    match data
        .authenticate_user(
            email,
            Some(user.uuid),
            change_request.current_password,
            user.project_id,
        )
        .await
    {
        Ok(Some(authenticated_user)) if authenticated_user.uuid == user.uuid => {
            // Current password is correct, proceed with password change
            match data
                .update_user_password(&user, change_request.new_password)
                .await
            {
                Ok(()) => {
                    let response = json!({ "message": "Password changed successfully" });
                    debug!("Exiting change_password function");
                    encrypt_response(&data, &session_id, &response).await
                }
                Err(e) => {
                    error!("Error changing password: {:?}", e);
                    Err(ApiError::InternalServerError)
                }
            }
        }
        _ => {
            // Current password is incorrect
            Err(ApiError::InvalidUsernameOrPassword)
        }
    }
}

pub async fn get_private_key(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Query(query): Query<DerivationPathQuery>,
) -> Result<Json<EncryptedResponse<PrivateKeyResponse>>, ApiError> {
    info!("Entering get_private_key function for user: {}", user.uuid);

    // Log received derivation paths for debugging purposes
    if let Some(path) = &query.key_options.seed_phrase_derivation_path {
        info!("Received BIP-85 derivation path: {}", path);
    }
    if let Some(path) = &query.key_options.private_key_derivation_path {
        info!("Received BIP-32 derivation path: {}", path);
    }

    // Validate paths if present
    debug!("Validating derivation paths");
    query.validate()?;

    // First try to get the existing encrypted seed
    debug!("Retrieving encrypted seed for user");
    let encrypted_seed = match user.get_seed_encrypted().await {
        Some(seed) => {
            debug!("Found existing encrypted seed");
            seed
        }
        None => {
            // Only generate a new key if one doesn't exist
            info!(
                "No seed found, generating new private key for user: {}",
                user.uuid
            );
            data.generate_private_key(user.uuid)
                .await
                .map_err(|e| {
                    error!("Failed to generate private key: {:?}", e);
                    ApiError::InternalServerError
                })?
                .get_seed_encrypted()
                .await
                .ok_or_else(|| {
                    error!("Private key not found after generation: {}", user.uuid);
                    ApiError::InternalServerError
                })?
        }
    };

    // Check if BIP-85 derivation is requested
    if let Some(bip85_path) = &query.key_options.seed_phrase_derivation_path {
        info!("BIP-85 derivation requested with path: {}", bip85_path);
        // Derive a child mnemonic
        let child_mnemonic = crate::private_key::decrypt_and_derive_bip85_mnemonic(
            data.enclave_key.clone(),
            encrypted_seed,
            bip85_path,
        )
        .map_err(|e| {
            error!("BIP-85 derivation error: {:?}", e);
            ApiError::BadRequest
        })?;

        let word_count = child_mnemonic.word_count();
        info!(
            "Successfully derived BIP-85 mnemonic with {} words",
            word_count
        );

        let response = PrivateKeyResponse {
            mnemonic: child_mnemonic.to_string(),
        };

        debug!("Encrypting response with derived mnemonic");
        info!("Exiting get_private_key function with derived BIP-85 mnemonic");
        encrypt_response(&data, &session_id, &response).await
    } else {
        // Return root mnemonic
        info!("Returning root mnemonic (no BIP-85 derivation)");
        let mnemonic = decrypt_user_seed_to_mnemonic(data.enclave_key.clone(), encrypted_seed)
            .map_err(|e| {
                error!("Failed to decrypt user seed: {:?}", e);
                ApiError::InternalServerError
            })?;

        let word_count = mnemonic.word_count();
        debug!("Root mnemonic has {} words", word_count);

        let response = PrivateKeyResponse {
            mnemonic: mnemonic.to_string(),
        };

        debug!("Encrypting response with root mnemonic");
        info!("Exiting get_private_key function with root mnemonic");
        encrypt_response(&data, &session_id, &response).await
    }
}

pub async fn get_private_key_bytes(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Query(query): Query<DerivationPathQuery>,
) -> Result<Json<EncryptedResponse<PrivateKeyBytesResponse>>, ApiError> {
    info!(
        "Entering get_private_key_bytes function for user: {}",
        user.uuid
    );

    // Log received derivation paths for debugging purposes
    if let Some(path) = &query.key_options.seed_phrase_derivation_path {
        info!("Received BIP-85 derivation path: {}", path);
    }
    if let Some(path) = &query.key_options.private_key_derivation_path {
        info!("Received BIP-32 derivation path: {}", path);
    }

    // Validate derivation path if present
    debug!("Validating derivation paths");
    query.validate()?;

    // Use the method that supports both BIP-85 and BIP-32 derivation
    debug!("Getting user key with provided derivation paths");
    let secret_key = data
        .get_user_key(
            user.uuid,
            query.key_options.private_key_derivation_path.as_deref(),
            query.key_options.seed_phrase_derivation_path.as_deref(),
        )
        .await
        .map_err(|e| match e {
            Error::InvalidDerivationPath(msg) => {
                error!("Invalid derivation path: {}", msg);
                ApiError::BadRequest
            }
            Error::KeyDerivationError(msg) => {
                error!("Failed to derive key: {}", msg);
                ApiError::BadRequest
            }
            _ => {
                error!("Failed to get user key: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    info!("Successfully retrieved private key for user: {}", user.uuid);

    // Convert key to string
    debug!("Converting private key to string format");
    let response = PrivateKeyBytesResponse {
        private_key: secret_key.display_secret().to_string(),
    };

    debug!("Encrypting private key response");
    info!("Exiting get_private_key_bytes function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn sign_message(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(sign_request): Extension<SignMessageRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<SignMessageResponseJson>>, ApiError> {
    info!("Entering sign_message function for user: {}", user.uuid);
    debug!("Sign request algorithm: {:?}", sign_request.algorithm);

    // Validate key_options
    debug!("Validating key options");
    let validation_query = DerivationPathQuery {
        key_options: sign_request.key_options.clone(),
    };
    validation_query.validate()?;

    // Extract derivation paths from key_options
    let derivation_path = sign_request
        .key_options
        .private_key_derivation_path
        .as_deref();
    let seed_phrase_derivation_path = sign_request
        .key_options
        .seed_phrase_derivation_path
        .as_deref();

    // Log derivation paths if present
    if let Some(path) = derivation_path {
        info!("Using BIP-32 derivation path: {}", path);
    }

    if let Some(path) = seed_phrase_derivation_path {
        info!("Using BIP-85 derivation path: {}", path);
    }

    debug!("Decoding base64 message");
    let message_bytes = general_purpose::STANDARD
        .decode(&sign_request.message_base64)
        .map_err(|e| {
            error!("Failed to decode base64 message: {:?}", e);
            ApiError::BadRequest
        })?;

    debug!(
        "Signing message with algorithm: {:?}",
        sign_request.algorithm
    );
    let response = data
        .sign_message(
            user.uuid,
            &message_bytes,
            sign_request.algorithm,
            derivation_path,
            seed_phrase_derivation_path,
        )
        .await
        .map_err(|e| {
            error!("Error signing message: {:?}", e);
            ApiError::InternalServerError
        })?;

    info!("Message signed successfully for user: {}", user.uuid);
    debug!("Creating JSON response");
    let json_response = SignMessageResponseJson {
        signature: response.signature.to_string(),
        message_hash: hex::encode(response.message_hash),
    };

    debug!("Encrypting signed message response");
    info!("Exiting sign_message function");
    encrypt_response(&data, &session_id, &json_response).await
}

pub async fn get_public_key(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
    Query(query): Query<PublicKeyQuery>,
) -> Result<Json<EncryptedResponse<PublicKeyResponseJson>>, ApiError> {
    debug!("Entering get_public_key function");

    // Validate the key_options
    let validation_query = DerivationPathQuery {
        key_options: query.key_options.clone(),
    };
    validation_query.validate()?;

    // Extract derivation paths from key_options
    let derivation_path = query.key_options.private_key_derivation_path.as_deref();
    let seed_phrase_derivation_path = query.key_options.seed_phrase_derivation_path.as_deref();

    let user_secret_key = data
        .get_user_key(user.uuid, derivation_path, seed_phrase_derivation_path)
        .await
        .map_err(|e| {
            error!("Error getting user key: {:?}", e);
            ApiError::InternalServerError
        })?;

    let secp = Secp256k1::new();
    let public_key = user_secret_key.public_key(&secp);

    // Format public key according to algorithm
    let public_key_str = match query.algorithm {
        SigningAlgorithm::Schnorr => {
            let (xonly_pubkey, _parity) = public_key.x_only_public_key();
            xonly_pubkey.to_string()
        }
        SigningAlgorithm::Ecdsa => {
            // For ECDSA, use the compressed format
            public_key.to_string()
        }
    };

    let response = PublicKeyResponseJson {
        public_key: public_key_str,
        algorithm: query.algorithm,
    };

    debug!("Exiting get_public_key function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn convert_guest_to_email(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(convert_request): Extension<ConvertGuestRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering convert_guest_to_email function");

    // Check if user is eligible for conversion
    if !user.is_guest() {
        error!("User already has an email address");
        return Err(ApiError::BadRequest);
    }

    if user.password_enc.is_none() {
        error!("OAuth users cannot be converted");
        return Err(ApiError::BadRequest);
    }

    // Check if email is already taken
    if data
        .db
        .get_user_by_email(convert_request.email.clone(), user.project_id)
        .is_ok()
    {
        error!("Email address already in use");
        return Err(ApiError::EmailAlreadyExists);
    }

    // Hash and encrypt the new password
    let password_hash = password_auth::generate_hash(convert_request.password);
    let secret_key =
        SecretKey::from_slice(&data.enclave_key).map_err(|_| ApiError::InternalServerError)?;
    let encrypted_password = encrypt::encrypt_with_key(&secret_key, password_hash.as_bytes()).await;

    // Update the user with new email, password, and optional name
    let mut updated_user = user.clone();
    updated_user.email = Some(convert_request.email);
    updated_user.password_enc = Some(encrypted_password);
    updated_user.name = convert_request.name;

    // Save the changes
    if let Err(e) = data.db.update_user(&updated_user) {
        error!("Failed to update user: {:?}", e);
        return Err(ApiError::InternalServerError);
    }

    // Create email verification entry
    let new_verification = NewEmailVerification::new(updated_user.uuid, 24, false);
    let verification = match data.db.create_email_verification(new_verification) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Error creating email verification: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    // Send verification email
    if let Some(email) = updated_user.get_email() {
        let email = email.to_string();
        let verification_code = verification.verification_code;
        let data = data.clone();
        let project_id = updated_user.project_id;
        spawn(async move {
            if let Err(e) =
                send_verification_email(&data, project_id, email, verification_code).await
            {
                tracing::error!("Could not send verification email: {e}");
            }
        });
    }

    let response = json!({
        "message": "Successfully converted guest account to email account. Please check your email for verification."
    });

    debug!("Exiting convert_guest_to_email function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn generate_third_party_token(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(request): Extension<ThirdPartyTokenRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ThirdPartyTokenResponse>>, ApiError> {
    debug!("Entering generate_third_party_token function");
    info!(
        "Generating third party token for user {} with audience {:?}",
        user.uuid, request.audience
    );

    // Validate the audience URL
    if let Some(audience) = request.audience.as_ref() {
        if url::Url::parse(audience).is_err() {
            error!("Invalid audience URL provided: {}", audience);
            return Err(ApiError::BadRequest);
        }
    }

    debug!("Audience URL validation successful");

    let project = data.db.get_org_project_by_id(user.project_id)?;

    let token = match NewToken::new(
        &user,
        TokenType::ThirdParty {
            aud: request.audience,
            azp: project.client_id.to_string(),
        },
        &data,
    ) {
        Ok(token) => {
            info!(
                "Successfully generated third party token for user {}",
                user.uuid
            );
            token
        }
        Err(e) => {
            error!("Failed to generate third party token: {:?}", e);
            return Err(e);
        }
    };

    let response = ThirdPartyTokenResponse { token: token.token };

    debug!("Exiting generate_third_party_token function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn encrypt_data(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(request): Extension<EncryptDataRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<EncryptDataResponse>>, ApiError> {
    debug!("Entering encrypt_data function");
    info!("Encrypting data for user {}", user.uuid);

    // Validate key_options
    let validation_query = DerivationPathQuery {
        key_options: request.key_options.clone(),
    };
    validation_query.validate()?;

    // Extract derivation paths from key_options
    let derivation_path = request.key_options.private_key_derivation_path.as_deref();
    let seed_phrase_derivation_path = request.key_options.seed_phrase_derivation_path.as_deref();

    // Get the user's key
    let user_key = data
        .get_user_key(user.uuid, derivation_path, seed_phrase_derivation_path)
        .await
        .map_err(|e| match e {
            Error::InvalidDerivationPath(msg) => {
                error!("Invalid derivation path: {}", msg);
                ApiError::BadRequest
            }
            Error::KeyDerivationError(msg) => {
                error!("Failed to derive key: {}", msg);
                ApiError::BadRequest
            }
            _ => {
                error!("Failed to get user key: {:?}", e);
                ApiError::InternalServerError
            }
        })?;

    // Encrypt the data
    let data_bytes = request.data.as_bytes();
    let encrypted_data = encrypt::encrypt_with_key(&user_key, data_bytes).await;

    // Convert to base64 for easy transport
    let encrypted_data_base64 = general_purpose::STANDARD.encode(&encrypted_data);

    let response = EncryptDataResponse {
        encrypted_data: encrypted_data_base64,
    };

    debug!("Exiting encrypt_data function");
    encrypt_response(&data, &session_id, &response).await
}

pub async fn decrypt_data(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(request): Extension<DecryptDataRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<String>>, ApiError> {
    debug!("Entering decrypt_data function");
    info!(
        "Decrypting data for user {} with key options: {:?}",
        user.uuid, request.key_options
    );

    // Validate key_options
    let validation_query = DerivationPathQuery {
        key_options: request.key_options.clone(),
    };
    validation_query.validate()?;
    debug!("Derivation path validation successful");

    // Extract derivation paths from key_options
    let derivation_path = request.key_options.private_key_derivation_path.as_deref();
    let seed_phrase_derivation_path = request.key_options.seed_phrase_derivation_path.as_deref();

    // Get the user's key
    let user_key = data
        .get_user_key(user.uuid, derivation_path, seed_phrase_derivation_path)
        .await
        .map_err(|e| match e {
            Error::InvalidDerivationPath(msg) => {
                error!("Invalid derivation path: {}", msg);
                ApiError::BadRequest
            }
            Error::KeyDerivationError(msg) => {
                error!("Failed to derive key: {}", msg);
                ApiError::BadRequest
            }
            _ => {
                error!("Failed to get user key: {e}");
                ApiError::InternalServerError
            }
        })?;
    debug!("Successfully retrieved user key");

    // Decode the base64 encrypted data
    let encrypted_data = match general_purpose::STANDARD.decode(&request.encrypted_data) {
        Ok(data) => data,
        Err(e) => {
            error!("Failed to decode base64 data: {e}");
            return Err(ApiError::BadRequest);
        }
    };

    // Decrypt the data
    let decrypted_data = match encrypt::decrypt_with_key(&user_key, &encrypted_data) {
        Ok(data) => {
            debug!("Successfully decrypted data");
            data
        }
        Err(e) => {
            error!("Decryption failed: {e}");
            return Err(ApiError::BadRequest);
        }
    };

    // Convert decrypted bytes to string
    let decrypted_string = match String::from_utf8(decrypted_data) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to convert decrypted data to string: {e}");
            return Err(ApiError::BadRequest);
        }
    };

    debug!("Exiting decrypt_data function, preparing encrypted response");
    encrypt_response(&data, &session_id, &decrypted_string).await
}

pub async fn initiate_account_deletion(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(delete_request): Extension<InitiateAccountDeletionRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering initiate_account_deletion function");
    info!("User {} is initiating account deletion request", user.uuid);

    // Create the account deletion request
    match data
        .create_account_deletion_request(
            user.uuid,
            delete_request.hashed_secret.clone(),
            user.project_id,
        )
        .await
    {
        Ok(_) => {
            // Success - we return a generic message regardless of whether the
            // request was actually created and email sent
            let response = json!({
                "message": "We have sent a confirmation code to your email."
            });

            let result = encrypt_response(&data, &session_id, &response).await;
            debug!("Exiting initiate_account_deletion function");
            result
        }
        Err(e) => {
            error!("Error in initiating account deletion: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

pub async fn confirm_account_deletion(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(confirm_request): Extension<ConfirmAccountDeletionRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Entering confirm_account_deletion function");
    info!("User {} is confirming account deletion", user.uuid);

    // Confirm the account deletion
    match data
        .confirm_account_deletion(
            user.uuid,
            confirm_request.confirmation_code.clone(),
            confirm_request.plaintext_secret.clone(),
        )
        .await
    {
        Ok(_) => {
            let response = json!({
                "message": "Your account has been successfully deleted."
            });

            let result = encrypt_response(&data, &session_id, &response).await;
            debug!("Exiting confirm_account_deletion function");
            result
        }
        Err(e) => match e {
            Error::AccountDeletionExpired => {
                warn!("Account deletion confirmation has expired: {:?}", e);
                Err(ApiError::BadRequest)
            }
            Error::InvalidAccountDeletionSecret => {
                warn!("Invalid account deletion secret: {:?}", e);
                Err(ApiError::BadRequest)
            }
            Error::InvalidAccountDeletionRequest => {
                warn!("Invalid account deletion request: {:?}", e);
                Err(ApiError::BadRequest)
            }
            _ => {
                error!("Error in confirming account deletion: {:?}", e);
                Err(ApiError::InternalServerError)
            }
        },
    }
}

// API Key Management Handlers

pub async fn create_api_key(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(request): Extension<CreateApiKeyRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<CreateApiKeyResponse>>, ApiError> {
    debug!("Creating new API key for user: {}", user.uuid);

    // Validate the request
    if let Err(e) = request.validate() {
        error!("API key request validation failed: {:?}", e);
        return Err(ApiError::BadRequest);
    }

    // Use the name directly (validation ensures no leading/trailing whitespace)
    let name = request.name.clone();

    // Generate a random UUID for the API key
    let random_bytes: [u8; 16] =
        crate::encrypt::generate_random_enclave::<16>(data.aws_credential_manager.clone()).await;
    let api_key_uuid = Uuid::from_bytes(random_bytes);

    // Hash the UUID string (with dashes) using SHA-256
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(api_key_uuid.to_string().as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());

    // Create the database record
    let new_key =
        crate::models::user_api_keys::NewUserApiKey::new(user.uuid, key_hash, name.clone());

    let api_key_record = data.db.create_user_api_key(new_key).map_err(|e| {
        match e {
            DBError::UserApiKeyError(
                crate::models::user_api_keys::UserApiKeyError::DuplicateName,
            ) => {
                error!("API key with name '{}' already exists for user", name);
                ApiError::Conflict // 409 - name already in use
            }
            _ => {
                error!("Failed to create API key: {:?}", e);
                ApiError::InternalServerError
            }
        }
    })?;

    let response = CreateApiKeyResponse {
        key: api_key_uuid.to_string(), // Return the actual UUID to the user (only time it's shown)
        name: api_key_record.name,
        created_at: api_key_record.created_at,
    };

    info!("Created API key '{}' for user {}", response.name, user.uuid);
    encrypt_response(&data, &session_id, &response).await
}

pub async fn list_api_keys(
    State(data): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<ListApiKeysResponse>>, ApiError> {
    debug!("Listing API keys for user: {}", user.uuid);

    let api_keys = data
        .db
        .get_all_user_api_keys_for_user(user.uuid)
        .map_err(|e| {
            error!("Failed to list API keys: {:?}", e);
            ApiError::InternalServerError
        })?;

    let keys: Vec<ApiKeyInfo> = api_keys
        .into_iter()
        .map(|key| ApiKeyInfo {
            name: key.name,
            created_at: key.created_at,
        })
        .collect();

    let response = ListApiKeysResponse { keys };

    encrypt_response(&data, &session_id, &response).await
}

pub async fn delete_api_key(
    State(data): State<Arc<AppState>>,
    Path(name): Path<String>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Deleting API key '{}' for user: {}", name, user.uuid);

    data.db
        .delete_user_api_key_by_name(&name, user.uuid)
        .map_err(|e| {
            error!("Failed to delete API key: {:?}", e);
            match e {
                DBError::UserApiKeyError(
                    crate::models::user_api_keys::UserApiKeyError::NotFound,
                ) => ApiError::NotFound,
                _ => ApiError::InternalServerError,
            }
        })?;

    info!("Deleted API key '{}' for user {}", name, user.uuid);

    let response = json!({ "success": true });
    encrypt_response(&data, &session_id, &response).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt::{decrypt_with_key, encrypt_with_key};
    use secp256k1::SecretKey;

    #[test]
    fn test_derivation_path_validation() {
        // Test valid paths
        let valid_paths = vec![
            "m/44'/0'/0'/0/0", // Standard BIP44 absolute
            "44'/0'/0'/0/0",   // Standard BIP44 relative
            "m/84'/0'/0'/0/0", // Standard BIP84 absolute
            "84'/0'/0'/0/0",   // Standard BIP84 relative
            "m/49'/0'/0'/0/0", // Standard BIP49 absolute
            "0/0",             // Simple relative path
            "m/0/0",           // Simple absolute path
            "0",               // Single level relative
            "m/0",             // Single level absolute
            "m",               // Master key
            "",                // Empty path
            "0'",              // Hardened child
            "m/0'",            // Hardened child absolute
            "0h",              // Hardened child (alternate notation)
            "m/0h",            // Hardened child absolute (alternate notation)
        ];

        for path in valid_paths {
            let query = DerivationPathQuery {
                key_options: KeyOptions {
                    private_key_derivation_path: Some(path.to_string()),
                    seed_phrase_derivation_path: None,
                },
            };
            assert!(query.validate().is_ok(), "Path should be valid: {}", path);
        }

        // Test invalid paths
        let invalid_paths = vec![
            "invalid",      // Random string
            "m//0",         // Double slash
            "m/x/0",        // Invalid character
            "m/0'/x/0",     // Invalid character after hardened
            "M/0/0",        // Wrong case for master key
            "m/2147483648", // Exceeds u32::MAX
            "n/0/0",        // Wrong master key letter
            "/0/0",         // Slash without master key
        ];

        for path in invalid_paths {
            let query = DerivationPathQuery {
                key_options: KeyOptions {
                    private_key_derivation_path: Some(path.to_string()),
                    seed_phrase_derivation_path: None,
                },
            };
            assert!(
                query.validate().is_err(),
                "Path should be invalid: {}",
                path
            );
        }

        // Test None path
        let query = DerivationPathQuery {
            key_options: KeyOptions {
                private_key_derivation_path: None,
                seed_phrase_derivation_path: None,
            },
        };
        assert!(query.validate().is_ok(), "None path should be valid");

        // Test valid BIP-85 paths - only 12, 18, and 24 word phrases are supported
        let valid_bip85_paths = vec![
            "m/83696968'/39'/0'/12'/0'",   // Standard BIP-85 for 12-word mnemonic
            "m/83696968'/39'/0'/18'/0'",   // Standard BIP-85 for 18-word mnemonic
            "m/83696968'/39'/0'/24'/0'",   // Standard BIP-85 for 24-word mnemonic
            "m/83696968'/39'/0'/12'/5'",   // BIP-85 with non-zero index
            "m/83696968'/39'/0'/12'/255'", // BIP-85 with large index
            "m/83696968'/39'/1'/12'/0'",   // BIP-85 with non-English language
            // Test alternate hardened notation
            "m/83696968h/39h/0h/12h/0h", // Using 'h' notation instead of '
        ];

        for path in valid_bip85_paths {
            let query = DerivationPathQuery {
                key_options: KeyOptions {
                    private_key_derivation_path: None,
                    seed_phrase_derivation_path: Some(path.to_string()),
                },
            };
            assert!(
                query.validate().is_ok(),
                "BIP-85 path should be valid: {}",
                path
            );
        }

        // Test invalid BIP-85 paths
        let invalid_bip85_paths = vec![
            "m/44'/0'/0'/0/0",             // Not a BIP-85 path (wrong purpose)
            "m/83696968/39'/0'/12'/0'",    // Purpose not hardened
            "m/83696968'/39/0'/12'/0'",    // Application not hardened
            "m/83696968'/39'/0/12'/0'",    // Language not hardened
            "m/83696968'/39'/0'/12/0'",    // Word count not hardened
            "m/83696968'/39'/0'/12'/0",    // Index not hardened
            "m/83696968'/39'/0'/13'/0'",   // Invalid word count (not 12,18,24)
            "m/83696968'/39'/0'/15'/0'",   // Invalid word count (not 12,18,24)
            "m/83696968'/39'/0'/21'/0'",   // Invalid word count (not 12,18,24)
            "m/83696968'/39'/0'/256'/0'",  // Word count too large
            "m/83696968'",                 // Incomplete path (too few segments)
            "m/83696968'/39'",             // Incomplete path (too few segments)
            "m/83696968'/39'/0'",          // Incomplete path (too few segments)
            "m/83696968'/39'/0'/12'",      // Incomplete path (too few segments)
            "83696968'/39'/0'/12'/0'",     // Missing 'm' prefix
            "m/83696968'/38'/0'/12'/0'",   // Wrong application (not 39)
            "m/83696968'/39'/0'/12'/0'/0", // Extra segment
            "m/83696968'/39'/ABC'/12'/0'", // Non-numeric language
            "m/83696968'/39'/0'/ABC'/0'",  // Non-numeric word count
            "m/83696968'/39'/0'/12'/ABC'", // Non-numeric index
        ];

        for path in invalid_bip85_paths {
            let query = DerivationPathQuery {
                key_options: KeyOptions {
                    private_key_derivation_path: None,
                    seed_phrase_derivation_path: Some(path.to_string()),
                },
            };
            assert!(
                query.validate().is_err(),
                "BIP-85 path should be invalid: {}",
                path
            );
        }
    }

    #[tokio::test]
    async fn test_encrypt_decrypt_empty_array() {
        // Mock a private key for testing
        let test_key = SecretKey::from_slice(&[0x42; 32]).unwrap();

        // Test with empty array as JSON string "[]"
        let empty_array = "[]".as_bytes();

        // First encryption
        let encrypted1 = encrypt_with_key(&test_key, empty_array).await;
        // Second encryption of the same data
        let encrypted2 = encrypt_with_key(&test_key, empty_array).await;

        // Since we use a random nonce, the two encryptions should be different
        assert_ne!(encrypted1, encrypted2);

        // Decrypt the first encrypted value
        let decrypted1 = decrypt_with_key(&test_key, &encrypted1).unwrap();
        assert_eq!(decrypted1, empty_array);

        // Decrypt the second encrypted value - this should also work
        let decrypted2 = decrypt_with_key(&test_key, &encrypted2).unwrap();
        assert_eq!(decrypted2, empty_array);

        // Test with multiple consecutive decryption attempts on the same encrypted values
        let decrypted1_again = decrypt_with_key(&test_key, &encrypted1).unwrap();
        assert_eq!(decrypted1_again, empty_array);

        let decrypted2_again = decrypt_with_key(&test_key, &encrypted2).unwrap();
        assert_eq!(decrypted2_again, empty_array);
    }

    #[tokio::test]
    async fn test_encrypt_decrypt_base64_payload() {
        // Test with the specific base64 payloads from the user report
        let test_key = SecretKey::from_slice(&[0x42; 32]).unwrap();

        // Recreate the two payloads mentioned in the report
        let payload1 = general_purpose::STANDARD
            .decode("eoW3a/gBol+uMLr6DmMwMjYHxiS31/cSet3sAnTi")
            .unwrap();
        let payload2 = general_purpose::STANDARD
            .decode("YWF2+/2fKK1UBTaFqpAzGh/7y/QrxzfE5/FkqLfe")
            .unwrap();

        // Attempt to decrypt both payloads
        match decrypt_with_key(&test_key, &payload1) {
            Ok(decrypted) => {
                // If decryption succeeds, check if it's an empty array
                let as_str = String::from_utf8_lossy(&decrypted);
                println!("Payload 1 decrypted: {}", as_str);
            }
            Err(e) => {
                // Expected - our test key won't match the real key, but we're testing the process
                println!("Payload 1 decryption failed as expected: {:?}", e);
            }
        }

        match decrypt_with_key(&test_key, &payload2) {
            Ok(decrypted) => {
                // If decryption succeeds, check if it's an empty array
                let as_str = String::from_utf8_lossy(&decrypted);
                println!("Payload 2 decrypted: {}", as_str);
            }
            Err(e) => {
                // Expected - our test key won't match the real key, but we're testing the process
                println!("Payload 2 decryption failed as expected: {:?}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_multiple_empty_array_encryptions() {
        // Simulate the exact user scenario: multiple different private keys,
        // each encrypting "[]" twice, then decrypting both versions
        let empty_array = "[]".as_bytes();

        // Run 10 iterations with different keys
        for i in 0..10 {
            // Generate a valid key for each iteration
            // Use a pattern that ensures a valid secp256k1 key (non-zero, less than curve order)
            let base_key = [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14,
                0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
                0x29, 0x30, 0x31, 0x32,
            ];
            let mut key_bytes = base_key;
            key_bytes[0] = (i + 1) as u8; // Change first byte based on iteration

            let private_key = SecretKey::from_slice(&key_bytes)
                .unwrap_or_else(|e| panic!("Failed to create key in iteration {}: {:?}", i, e));

            // Encrypt the empty array twice with the same key
            let encrypted1 = encrypt_with_key(&private_key, empty_array).await;
            let encrypted2 = encrypt_with_key(&private_key, empty_array).await;

            // The encryptions should be different due to random nonce
            assert_ne!(
                encrypted1, encrypted2,
                "Iteration {}: Encryptions should differ",
                i
            );

            // Decrypt both and verify they both correctly decrypt to "[]"
            let decrypted1 = decrypt_with_key(&private_key, &encrypted1)
                .unwrap_or_else(|e| panic!("Iteration {}: First decryption failed: {:?}", i, e));

            let decrypted2 = decrypt_with_key(&private_key, &encrypted2)
                .unwrap_or_else(|e| panic!("Iteration {}: Second decryption failed: {:?}", i, e));

            // Both decryptions should match the original empty array
            assert_eq!(
                decrypted1, empty_array,
                "Iteration {}: First decryption didn't match original data",
                i
            );
            assert_eq!(
                decrypted2, empty_array,
                "Iteration {}: Second decryption didn't match original data",
                i
            );

            // Print debug info
            println!("Iteration {}: Both decryptions successful", i);
        }
    }
}

#[cfg(test)]
mod api_key_validation_tests {
    use super::*;

    #[test]
    fn test_valid_api_key_names() {
        let max_length_name = "a".repeat(50);
        let valid_names = vec![
            "my-api-key",
            "test_key_123",
            "Production Key",
            "key-with_spaces and-dashes",
            "UPPERCASE",
            "lowercase",
            "MixedCase",
            "123numeric",
            "a",                      // single character
            max_length_name.as_str(), // max length
        ];

        for name in valid_names {
            assert!(
                validate_api_key_name(name).is_ok(),
                "Expected '{}' to be valid",
                name
            );
        }
    }

    #[test]
    fn test_invalid_api_key_names_whitespace() {
        let invalid_names = vec![
            "   ",            // only spaces
            "\t",             // tab
            "\n",             // newline
            "  leading",      // leading space
            "trailing  ",     // trailing space
            " both ",         // both leading and trailing
            "\tleading_tab",  // leading tab
            "trailing_tab\t", // trailing tab
        ];

        for name in invalid_names {
            let result = validate_api_key_name(name);
            assert!(
                result.is_err(),
                "Expected '{}' to be invalid",
                name.escape_debug()
            );
            if let Err(e) = result {
                assert_eq!(
                    e.code,
                    "whitespace",
                    "Expected whitespace error for '{}'",
                    name.escape_debug()
                );
            }
        }
    }

    #[test]
    fn test_invalid_api_key_names_length() {
        let too_long = "a".repeat(51);
        let way_too_long = "x".repeat(100);
        let invalid_names = vec![
            "",                    // empty
            too_long.as_str(),     // too long (51 chars)
            way_too_long.as_str(), // way too long
        ];

        for name in invalid_names {
            let result = validate_api_key_name(name);
            assert!(
                result.is_err(),
                "Expected '{}' (len={}) to be invalid",
                if name.len() > 20 { "[truncated]" } else { name },
                name.len()
            );
            if let Err(e) = result {
                assert_eq!(
                    e.code,
                    "length",
                    "Expected length error for name of length {}",
                    name.len()
                );
            }
        }
    }

    #[test]
    fn test_invalid_api_key_names_characters() {
        let invalid_names = vec![
            "café",             // accented character
            "测试",             // Chinese characters
            "тест",             // Cyrillic
            "key!",             // exclamation mark
            "key@host",         // at symbol
            "key#hash",         // hash
            "key$money",        // dollar sign
            "key%percent",      // percent
            "key^caret",        // caret
            "key&and",          // ampersand
            "key*star",         // asterisk
            "key(paren",        // parenthesis
            "key[bracket",      // bracket
            "key{brace",        // brace
            "key|pipe",         // pipe
            "key\\backslash",   // backslash
            "key/slash",        // forward slash
            "key:colon",        // colon
            "key;semicolon",    // semicolon
            "key'quote",        // single quote
            "key\"doublequote", // double quote
            "key<less",         // less than
            "key>greater",      // greater than
            "key?question",     // question mark
            "key,comma",        // comma
            "key.period",       // period
            "😀emoji",          // emoji
            "key\0null",        // null character
            "hello\nworld",     // newline in middle
            "hello\tworld",     // tab in middle
        ];

        for name in invalid_names {
            let result = validate_api_key_name(name);
            assert!(
                result.is_err(),
                "Expected '{}' to be invalid due to invalid characters",
                name.escape_debug()
            );
            if let Err(e) = result {
                assert_eq!(
                    e.code,
                    "invalid_characters",
                    "Expected invalid_characters error for '{}'",
                    name.escape_debug()
                );
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test that internal spaces are allowed
        assert!(validate_api_key_name("valid name").is_ok());
        assert!(validate_api_key_name("multiple  spaces  inside").is_ok());

        // Test all allowed characters together
        assert!(validate_api_key_name("aZ_0-9 mix").is_ok());

        // Test exact boundary lengths
        let exactly_50 = "x".repeat(50);
        let exactly_51 = "x".repeat(51);
        assert!(validate_api_key_name(&exactly_50).is_ok()); // exactly 50
        assert!(validate_api_key_name(&exactly_51).is_err()); // exactly 51
    }

    #[test]
    fn test_security_concerns() {
        // Path traversal attempts
        assert!(validate_api_key_name("../etc/passwd").is_err());
        assert!(validate_api_key_name("..\\windows\\system32").is_err());

        // SQL injection attempts
        assert!(validate_api_key_name("'; DROP TABLE users--").is_err());
        assert!(validate_api_key_name("1' OR '1'='1").is_err());

        // XSS attempts
        assert!(validate_api_key_name("<script>alert(1)</script>").is_err());
        assert!(validate_api_key_name("javascript:alert(1)").is_err());

        // Command injection attempts
        assert!(validate_api_key_name("key; rm -rf /").is_err());
        assert!(validate_api_key_name("key`whoami`").is_err());
        assert!(validate_api_key_name("key$(whoami)").is_err());
        assert!(validate_api_key_name("key&&whoami").is_err());
        assert!(validate_api_key_name("key||whoami").is_err());
    }
}
