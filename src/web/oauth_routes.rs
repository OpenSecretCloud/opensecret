use crate::apple_signin::{generate_apple_client_secret, validate_apple_native_token};
use crate::models::oauth::NewUserOAuthConnection;
use crate::oauth::OAuthState;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::login_routes::handle_new_user_registration;
use crate::web::platform::common::{
    PROJECT_APPLE_OAUTH_SECRET, PROJECT_GITHUB_OAUTH_SECRET, PROJECT_GOOGLE_OAUTH_SECRET,
};
use crate::GithubProvider;
use crate::GoogleProvider;
use crate::{decrypt_with_key, private_key::generate_twelve_word_seed};
use crate::{encrypt, DBError};
use crate::{encrypt::encrypt_with_key, models::email_verification::NewEmailVerification};
use crate::{
    jwt::{NewToken, TokenType},
    models::{
        project_settings::OAuthProviderSettings,
        users::{NewUser, User},
    },
    ApiError, AppState,
};
use axum::{
    extract::{Extension, State},
    routing::post,
    Json, Router,
};
use base64::Engine as _;
use oauth2::TokenResponse;
use oauth2::{
    basic::{BasicClient, BasicTokenType},
    AuthorizationCode, EmptyExtraTokenFields, StandardTokenResponse,
};
use reqwest::header::AUTHORIZATION;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, trace};
use uuid::Uuid;

pub fn router(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/auth/github",
            post(|state, ext1, ext2| initiate_oauth(state, ext1, ext2, "github")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthAuthRequest>,
                ),
            ),
        )
        .route(
            "/auth/github/callback",
            post(|state, ext1, ext2| oauth_callback(state, ext1, ext2, "github")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthCallbackRequest>,
                ),
            ),
        )
        .route(
            "/auth/google",
            post(|state, ext1, ext2| initiate_oauth(state, ext1, ext2, "google")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthAuthRequest>,
                ),
            ),
        )
        .route(
            "/auth/google/callback",
            post(|state, ext1, ext2| oauth_callback(state, ext1, ext2, "google")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthCallbackRequest>,
                ),
            ),
        )
        .route(
            "/auth/apple",
            post(|state, ext1, ext2| initiate_oauth(state, ext1, ext2, "apple")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthAuthRequest>,
                ),
            ),
        )
        .route(
            "/auth/apple/callback",
            post(|state, ext1, ext2| oauth_callback(state, ext1, ext2, "apple")).layer(
                axum::middleware::from_fn_with_state(
                    app_state.clone(),
                    decrypt_request::<OAuthCallbackRequest>,
                ),
            ),
        )
        .route(
            "/auth/apple/native",
            post(handle_apple_native_signin).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<AppleNativeSignInRequest>,
            )),
        )
        .with_state(app_state)
}

#[derive(Serialize)]
pub struct OAuthOAuthCallbackResponse {
    auth_url: String,
    state: String,
}

#[derive(Deserialize, Clone)]
pub struct OAuthAuthRequest {
    pub client_id: Uuid,
}

#[derive(Deserialize, Clone)]
pub struct OAuthCallbackRequest {
    pub code: String,
    pub state: String,
}

#[derive(Serialize)]
pub struct OAuthCallbackResponse {
    id: Uuid,
    email: String,
    access_token: String,
    refresh_token: String,
}

#[derive(Deserialize, Clone)]
pub struct AppleNativeSignInRequest {
    pub identity_token: String,          // JWT from Apple
    pub user_identifier: Option<String>, // Apple's unique user ID (sub) - Optional since we can extract from token
    pub email: Option<String>,           // Email (only provided on first sign-in)
    pub given_name: Option<String>,      // First name (only provided on first sign-in)
    pub family_name: Option<String>,     // Last name (only provided on first sign-in)
    pub client_id: Uuid,                 // Your app's client ID for the project
    pub nonce: Option<String>,           // Optional nonce for preventing replay attacks
}

#[derive(Deserialize, Clone, Debug)]
struct GithubUser {
    id: i64,
    login: String,
    name: Option<String>,
    email: Option<String>,
}

#[derive(Deserialize)]
struct GithubEmail {
    email: String,
    primary: bool,
    verified: bool,
}

#[derive(Deserialize, Clone, Debug)]
struct GoogleUser {
    sub: String,
    email: String,
    email_verified: bool,
    name: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
struct AppleUser {
    sub: String,
    email: Option<String>,
    email_verified: Option<bool>,
    name: Option<String>,
}

#[derive(Deserialize, Debug)]
struct AppleTokenResponse {
    access_token: String,
    #[allow(dead_code)]
    token_type: String,
    expires_in: u64,
    refresh_token: Option<String>,
    id_token: String,
}

impl AppleTokenResponse {
    // Convert to OAuth2 StandardTokenResponse
    fn to_standard_token(&self) -> StandardTokenResponse<EmptyExtraTokenFields, BasicTokenType> {
        // Create the base token
        let mut token = StandardTokenResponse::new(
            oauth2::AccessToken::new(self.access_token.clone()),
            BasicTokenType::Bearer,
            EmptyExtraTokenFields {},
        );

        // Add refresh token if available
        if let Some(ref rt) = self.refresh_token {
            token.set_refresh_token(Some(oauth2::RefreshToken::new(rt.clone())));
        }

        // Add expiration
        let expires = std::time::Duration::from_secs(self.expires_in);
        token.set_expires_in(Some(&expires));

        token
    }
}

async fn get_project_oauth_client(
    app_state: &AppState,
    project_id: i32,
    provider_name: &str,
) -> Result<BasicClient, ApiError> {
    // Get project OAuth settings
    let oauth_settings = app_state
        .db
        .get_project_oauth_settings(project_id)?
        .ok_or_else(|| {
            error!("OAuth settings not found for project");
            ApiError::BadRequest
        })?;

    // Get provider settings based on provider name
    let (enabled, provider_settings, secret_key) = match provider_name {
        "github" => {
            let enabled = oauth_settings.github_oauth_enabled;
            let settings = oauth_settings.github_oauth_settings;
            let secret = app_state
                .db
                .get_org_project_secret_by_key_name_and_project(
                    PROJECT_GITHUB_OAUTH_SECRET,
                    project_id,
                )?;
            (enabled, settings, secret)
        }
        "google" => {
            let enabled = oauth_settings.google_oauth_enabled;
            let settings = oauth_settings.google_oauth_settings;
            let secret = app_state
                .db
                .get_org_project_secret_by_key_name_and_project(
                    PROJECT_GOOGLE_OAUTH_SECRET,
                    project_id,
                )?;
            (enabled, settings, secret)
        }
        "apple" => {
            let enabled = oauth_settings.apple_oauth_enabled;
            let mut apple_settings = oauth_settings.apple_oauth_settings.clone();

            // For web-based Apple OAuth flow, append .services to the client ID
            // This is because Apple requires Services ID for web flow, not App ID
            if let Some(ref mut settings) = apple_settings {
                if !settings.client_id.ends_with(".services") {
                    settings.client_id = format!("{}.services", settings.client_id);
                    debug!(
                        "Modified Apple client ID for OAuth client: {}",
                        settings.client_id
                    );
                }
            }

            let secret = app_state
                .db
                .get_org_project_secret_by_key_name_and_project(
                    PROJECT_APPLE_OAUTH_SECRET,
                    project_id,
                )?;

            // Convert AppleOAuthSettings to OAuthProviderSettings for the tuple
            let standard_settings = apple_settings.as_ref().map(|apple| OAuthProviderSettings {
                client_id: apple.client_id.clone(),
                redirect_url: apple.redirect_url.clone(),
            });

            (enabled, standard_settings, secret)
        }
        _ => {
            error!("Unsupported OAuth provider: {}", provider_name);
            return Err(ApiError::BadRequest);
        }
    };

    // Verify OAuth is enabled for this provider
    if !enabled {
        error!("{} OAuth is not enabled for this project", provider_name);
        return Err(ApiError::BadRequest);
    }

    // Get provider settings
    let provider_settings = provider_settings.ok_or_else(|| {
        error!("{} OAuth settings not configured", provider_name);
        ApiError::BadRequest
    })?;

    // Get and decrypt client secret
    let secret = secret_key.ok_or_else(|| {
        error!("{} OAuth secret not found", provider_name);
        ApiError::BadRequest
    })?;

    let secret_key = SecretKey::from_slice(&app_state.enclave_key).map_err(|_| {
        error!("Failed to create secret key from enclave key");
        ApiError::InternalServerError
    })?;

    let client_secret = String::from_utf8(
        decrypt_with_key(&secret_key, &secret.secret_enc).map_err(|_| {
            error!("Failed to decrypt OAuth client secret");
            ApiError::InternalServerError
        })?,
    )
    .map_err(|_| {
        error!("Failed to parse decrypted OAuth client secret as UTF-8");
        ApiError::InternalServerError
    })?;

    // Get the OAuth provider
    let oauth_provider = app_state
        .oauth_manager
        .get_provider(provider_name)
        .ok_or_else(|| {
            error!("{} provider not initialized", provider_name);
            ApiError::InternalServerError
        })?;

    // For Apple, generate a JWT client secret instead of using the raw key
    if provider_name == "apple" {
        debug!("Generating Apple JWT client secret");

        // Get the Apple settings
        let apple_settings = oauth_settings
            .apple_oauth_settings
            .as_ref()
            .ok_or_else(|| {
                error!("Apple OAuth settings not configured");
                ApiError::BadRequest
            })?;

        // The client_secret is the private key in base64 - decode it first
        let private_key_pem = base64::engine::general_purpose::STANDARD
            .decode(&client_secret)
            .map_err(|_e| {
                error!("Failed to base64 decode the Apple private key");
                ApiError::InternalServerError
            })?;

        // Convert to UTF-8 string as it should be a PEM-encoded private key
        let private_key_pem_str = String::from_utf8(private_key_pem).map_err(|_e| {
            error!("Failed to convert Apple private key to string");
            ApiError::InternalServerError
        })?;

        // Make sure client_id for JWT includes .services suffix
        let client_id_with_services = if !apple_settings.client_id.ends_with(".services") {
            format!("{}.services", apple_settings.client_id)
        } else {
            apple_settings.client_id.clone()
        };

        // Check that Team ID and Key ID are configured
        let team_id = apple_settings.team_id.as_ref().ok_or_else(|| {
            error!("Apple OAuth Team ID is not configured");
            ApiError::BadRequest
        })?;

        let key_id = apple_settings.key_id.as_ref().ok_or_else(|| {
            error!("Apple OAuth Key ID is not configured");
            ApiError::BadRequest
        })?;

        // Generate the JWT client secret
        let client_secret_jwt = generate_apple_client_secret(
            team_id,
            key_id,
            &private_key_pem_str,
            &client_id_with_services,
        )
        .map_err(|_e| {
            error!("Failed to generate Apple client secret JWT");
            ApiError::InternalServerError
        })?;

        // Log the OAuth URL and redirect URL being used
        debug!(
            "Building Apple OAuth client with Client ID: {}, Redirect URL: {}",
            client_id_with_services, apple_settings.redirect_url
        );

        // Use the same client ID for the OAuth client as in the JWT's sub claim
        let client_id_for_client = client_id_with_services.clone();

        // Build and return the client with the JWT as client_secret
        // Important: Use the same client ID with .services suffix for building the client
        oauth_provider
            .build_client(
                client_id_for_client, // Use the same client ID as in the JWT
                client_secret_jwt,
                apple_settings.redirect_url.clone(),
            )
            .await
            .map_err(|e| {
                error!("Failed to build Apple OAuth client: {:?}", e);
                ApiError::InternalServerError
            })
    } else {
        // For other providers, use the client_secret as-is
        oauth_provider
            .build_client(
                provider_settings.client_id.clone(),
                client_secret,
                provider_settings.redirect_url.clone(),
            )
            .await
            .map_err(|_| ApiError::InternalServerError)
    }
}

pub async fn initiate_oauth(
    State(app_state): State<Arc<AppState>>,
    Extension(auth_request): Extension<OAuthAuthRequest>,
    Extension(session_id): Extension<Uuid>,
    provider_name: &str,
) -> Result<Json<EncryptedResponse<OAuthOAuthCallbackResponse>>, ApiError> {
    debug!("Entering init {} auth function", provider_name);

    // Get project
    let project = app_state
        .db
        .get_org_project_by_client_id(auth_request.client_id)
        .map_err(|_| ApiError::BadRequest)?;

    // Get OAuth client for this project
    let oauth_client = get_project_oauth_client(&app_state, project.id, provider_name).await?;

    // Get the OAuth provider
    let oauth_provider = app_state
        .oauth_manager
        .get_provider(provider_name)
        .ok_or_else(|| {
            error!("{} provider not initialized", provider_name);
            ApiError::InternalServerError
        })?;

    // Generate initial authorization URL to get the CSRF token
    let (initial_url, csrf_token) = oauth_provider.generate_authorize_url(&oauth_client).await;

    // Create our state that includes both CSRF token and client_id
    let state = OAuthState {
        csrf_token: csrf_token.secret().clone(),
        client_id: project.client_id,
    };

    // Store the complete state in the provider
    oauth_provider
        .store_state(csrf_token.secret(), state.clone())
        .await;

    let state_json = serde_json::to_string(&state).map_err(|_| ApiError::InternalServerError)?;
    let state_base64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(state_json);

    // Replace the CSRF token in the URL with our encoded state
    let auth_url = initial_url.replace(csrf_token.secret(), &state_base64);

    let response = OAuthOAuthCallbackResponse {
        auth_url,
        state: state_base64,
    };

    debug!("Exiting init {} auth function", provider_name);
    encrypt_response(&app_state, &session_id, &response).await
}

pub async fn oauth_callback(
    State(app_state): State<Arc<AppState>>,
    Extension(callback_request): Extension<OAuthCallbackRequest>,
    Extension(session_id): Extension<Uuid>,
    provider_name: &str,
) -> Result<Json<EncryptedResponse<OAuthCallbackResponse>>, ApiError> {
    debug!("Entering {} callback function", provider_name);
    debug!(
        "Received code (redacted, len={})",
        callback_request.code.len()
    );
    debug!("Received state: {}", callback_request.state);

    // Decode and parse the state
    let state_json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&callback_request.state)
        .map_err(|e| {
            error!("Could not parse state: {:?}", e);
            ApiError::BadRequest
        })?;
    debug!("Parsed state from request");
    let state: OAuthState = serde_json::from_slice(&state_json).map_err(|e| {
        error!("Could not parse OAuthState: {:?}", e);
        ApiError::BadRequest
    })?;
    debug!("Converted state to OAuthState");

    // Get the OAuth provider
    let oauth_provider = app_state
        .oauth_manager
        .get_provider(provider_name)
        .ok_or_else(|| {
            error!("{} provider not initialized", provider_name);
            ApiError::InternalServerError
        })?;

    // Validate the complete state (both CSRF token and client_id)
    let is_valid = oauth_provider.validate_state(&state).await;

    if !is_valid {
        error!("Invalid state in {} callback", provider_name);
        return Err(ApiError::BadRequest);
    }

    // Get project (we can trust the client_id now since we validated it against our stored state)
    debug!("Getting project from client_id: {:?}", state.client_id);
    let project = app_state
        .db
        .get_org_project_by_client_id(state.client_id)
        .map_err(|e| {
            error!(
                "Could not get project by client {:?} id: {:?}",
                state.client_id, e
            );
            ApiError::BadRequest
        })?;

    // Get OAuth client for this project
    let oauth_client = get_project_oauth_client(&app_state, project.id, provider_name).await?;

    // Exchange the code for an access token
    debug!(
        "Exchanging authorization code for access token with provider: {}",
        provider_name
    );

    // For Apple, add extra debugging for token exchange
    // Return both the token and the ID token for Apple
    let (token, apple_id_token): (
        StandardTokenResponse<EmptyExtraTokenFields, BasicTokenType>,
        Option<String>,
    ) = if provider_name == "apple" {
        debug!("Setting up Apple token exchange");

        // Perform the token exchange
        let token_url = "https://appleid.apple.com/auth/token";

        // Let's try a custom approach matching your successful Postman request
        let client = reqwest::Client::new();

        // Get the parameters from the OAuth client
        let client_id = oauth_client.client_id().as_str();

        // We need to regenerate the JWT here
        // First get the project OAuth settings again
        let oauth_settings = app_state
            .db
            .get_project_oauth_settings(project.id)?
            .ok_or_else(|| {
                error!("OAuth settings not found for project");
                ApiError::BadRequest
            })?;

        // Then get the Apple-specific settings
        let apple_settings = oauth_settings
            .apple_oauth_settings
            .as_ref()
            .ok_or_else(|| {
                error!("Apple OAuth settings not configured");
                ApiError::BadRequest
            })?;

        // Decrypt the private key again
        let secret = app_state
            .db
            .get_org_project_secret_by_key_name_and_project(PROJECT_APPLE_OAUTH_SECRET, project.id)?
            .ok_or_else(|| {
                error!("Apple OAuth secret not found");
                ApiError::BadRequest
            })?;

        let secret_key = SecretKey::from_slice(&app_state.enclave_key).map_err(|_| {
            error!("Failed to create secret key from enclave key");
            ApiError::InternalServerError
        })?;

        let client_secret_raw = String::from_utf8(
            decrypt_with_key(&secret_key, &secret.secret_enc).map_err(|_| {
                error!("Failed to decrypt OAuth client secret");
                ApiError::InternalServerError
            })?,
        )
        .map_err(|_| {
            error!("Failed to parse decrypted OAuth client secret as UTF-8");
            ApiError::InternalServerError
        })?;

        // The client_secret is the private key in base64 - decode it first
        let private_key_pem = base64::engine::general_purpose::STANDARD
            .decode(&client_secret_raw)
            .map_err(|_e| {
                error!("Failed to base64 decode the Apple private key");
                ApiError::InternalServerError
            })?;

        // Convert to UTF-8 string as it should be a PEM-encoded private key
        let private_key_pem_str = String::from_utf8(private_key_pem).map_err(|_e| {
            error!("Failed to convert Apple private key to string");
            ApiError::InternalServerError
        })?;

        // Make sure client_id for JWT includes .services suffix
        let client_id_with_services = if !apple_settings.client_id.ends_with(".services") {
            format!("{}.services", apple_settings.client_id)
        } else {
            apple_settings.client_id.clone()
        };

        // Check that Team ID and Key ID are configured for token exchange
        let team_id = apple_settings.team_id.as_ref().ok_or_else(|| {
            error!("Apple OAuth Team ID is not configured for token exchange");
            ApiError::BadRequest
        })?;

        let key_id = apple_settings.key_id.as_ref().ok_or_else(|| {
            error!("Apple OAuth Key ID is not configured for token exchange");
            ApiError::BadRequest
        })?;

        // Generate the JWT client secret
        let client_secret = generate_apple_client_secret(
            team_id,
            key_id,
            &private_key_pem_str,
            &client_id_with_services,
        )
        .map_err(|_e| {
            error!("Failed to generate Apple client secret JWT");
            ApiError::InternalServerError
        })?;

        let redirect_uri = oauth_client
            .redirect_url()
            .ok_or_else(|| {
                error!("OAuth redirect URL not configured");
                ApiError::InternalServerError
            })?
            .as_str();

        // Make sure the client_id parameter matches what's in the JWT's sub claim
        // Apple requires these to be exactly the same
        // - client_id parameter must match JWT's sub claim exactly
        let client_id_param = if apple_settings.client_id.ends_with(".services") {
            // Settings already have .services suffix
            client_id
        } else {
            // Use the same value with .services suffix that we used in the JWT
            client_id_with_services.as_str()
        };

        // Build the form data for the token request
        let params = [
            ("client_id", client_id_param),
            ("client_secret", &client_secret),
            ("code", &callback_request.code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", redirect_uri),
        ];

        // Create the request
        let request = client
            .post(token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params);

        // Send the request
        let response = request.send().await.map_err(|e| {
            error!("HTTP request failed: {:?}", e);
            ApiError::InternalServerError
        })?;

        // Check the response
        let status = response.status();

        // Check if successful
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!("Error response from Apple: {}", error_text);

            // Try to parse the error as JSON for more details
            if let Ok(error_json) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_json.get("error") {
                    error!("Error type: {}", error_obj);

                    // If it's an invalid_client error, log diagnostic info
                    if error_obj.as_str() == Some("invalid_client") {
                        error!("invalid_client error detected - this typically means the client_secret JWT is invalid");
                        error!("JWT verification failed - check Team ID, Key ID, and private key configuration");
                    }
                }
            }

            error!("Apple OAuth token exchange failed. Possible issues:");
            error!("1. Client ID might be wrong (check if it needs .services suffix)");
            error!("2. Team ID or Key ID might not match the private key");
            error!("3. Private key might be invalid or in wrong format");
            error!("4. The authorization code might be invalid or expired");
            error!("5. Redirect URI might not match the one used in the authorization request");

            return Err(ApiError::InternalServerError);
        }

        // Parse the successful response
        let token_json = response.text().await.map_err(|e| {
            error!("Failed to read response body: {:?}", e);
            ApiError::InternalServerError
        })?;

        // Parse the JSON using our custom AppleTokenResponse struct
        let apple_token: AppleTokenResponse = serde_json::from_str(&token_json).map_err(|e| {
            error!("Failed to parse Apple token response: {:?}", e);
            ApiError::InternalServerError
        })?;

        debug!("Successfully parsed Apple token response");

        // Convert AppleTokenResponse to StandardTokenResponse
        let token = apple_token.to_standard_token();

        // Extract id_token for return
        let id_token = apple_token.id_token;

        // Return both the token and the ID token
        (token, Some(id_token))
    } else {
        // For other providers, use the standard OAuth client
        let token = oauth_client
            .exchange_code(AuthorizationCode::new(callback_request.code.clone()))
            .request_async(oauth2::reqwest::async_http_client)
            .await
            .map_err(|e| {
                error!("Failed to exchange code for access token: {:?}", e);
                ApiError::InternalServerError
            })?;

        // No ID token for non-Apple providers
        (token, None)
    };

    debug!("Successfully obtained {} token", provider_name);

    let access_token = token.access_token().secret().to_string();

    // Fetch user information and find or create the user
    let user = match provider_name {
        "github" => {
            debug!("Access token obtained, fetching GitHub user");
            // Get GitHub provider for GitHub-specific operations
            let github_provider = oauth_provider.as_github().ok_or_else(|| {
                error!("Failed to get GithubProvider");
                ApiError::InternalServerError
            })?;
            let github_user = match fetch_github_user(&access_token, github_provider).await {
                Ok(user) => {
                    debug!("Successfully fetched GitHub user");
                    user
                }
                Err(e) => {
                    error!("Failed to fetch GitHub user: {:?}", e);
                    return Err(e);
                }
            };

            find_or_create_user_from_oauth(
                &app_state,
                github_user.email.clone().unwrap_or_default(),
                github_user.id.to_string(),
                "github",
                access_token,
                github_user.name.clone().or(Some(github_user.login.clone())),
                project.id,
            )
            .await?
        }
        "google" => {
            debug!("Access token obtained, fetching Google user");
            // Get Google provider for Google-specific operations
            let google_provider = oauth_provider.as_google().ok_or_else(|| {
                error!("Failed to get GoogleProvider");
                ApiError::InternalServerError
            })?;
            let google_user = match fetch_google_user(&access_token, google_provider).await {
                Ok(user) => {
                    debug!("Successfully fetched Google user");
                    user
                }
                Err(e) => {
                    error!("Failed to fetch Google user: {:?}", e);
                    return Err(e);
                }
            };

            find_or_create_user_from_oauth(
                &app_state,
                google_user.email.clone(),
                google_user.sub.clone(),
                "google",
                access_token,
                google_user.name.clone(),
                project.id,
            )
            .await?
        }
        "apple" => {
            debug!("Access token obtained, processing Apple ID token");

            // Verify we have a valid Apple provider
            let _apple_provider = oauth_provider.as_apple().ok_or_else(|| {
                error!("Failed to get AppleProvider");
                ApiError::InternalServerError
            })?;

            // Get project OAuth settings first
            let oauth_settings = app_state
                .db
                .get_project_oauth_settings(project.id)?
                .ok_or_else(|| {
                    error!("OAuth settings not found for project");
                    ApiError::BadRequest
                })?;

            // Get Apple client ID from project OAuth settings
            let mut client_id = oauth_settings
                .apple_oauth_settings
                .ok_or_else(|| {
                    error!("Apple OAuth settings not configured");
                    ApiError::BadRequest
                })?
                .client_id;

            // For web-based OAuth flow, append .services to the client ID
            // This is because Apple requires Services ID for web flow, not App ID
            if !client_id.ends_with(".services") {
                client_id = format!("{}.services", client_id);
                debug!("Modified Apple client ID for web flow: {}", client_id);
            }

            // For Apple, we need to extract the ID token from the token exchange response
            debug!("Processing Apple OAuth token response to get ID token");

            // Get the ID token that was returned directly from the token exchange
            // Use unwrap_or_default instead of a match since we know it exists
            let id_token = apple_id_token.ok_or_else(|| {
                error!("Missing Apple ID token that should have been returned from token exchange");
                ApiError::InternalServerError
            })?;

            // Verify the token just like we do for native sign-in
            let apple_user = match fetch_apple_user(&app_state, &id_token, &client_id).await {
                Ok(user) => {
                    debug!("Successfully verified Apple ID token");
                    user
                }
                Err(e) => {
                    error!("Failed to verify Apple ID token: {:?}", e);
                    return Err(e);
                }
            };

            // For Apple, we need a special approach:
            // 1. First, try to find any existing users with this Apple ID
            // 2. If found, just use that user - no need for email
            // 3. Only if this is a first-time user, we need an email

            let sub = apple_user.sub.clone();

            // Get the Apple provider from the database to get its ID
            let apple_db_provider = app_state
                .db
                .get_oauth_provider_by_name("apple")
                .map_err(|e| {
                    error!("Failed to get Apple OAuth provider: {:?}", e);
                    ApiError::InternalServerError
                })?
                .ok_or_else(|| {
                    error!("Apple OAuth provider not found");
                    ApiError::InternalServerError
                })?;

            // Directly query for a user with this Apple ID
            let existing_user = if let Some(connection) = app_state
                .db
                .get_user_oauth_connection_by_provider_and_provider_user_id(
                    apple_db_provider.id,
                    &sub,
                )? {
                // Found a connection - get the user
                debug!("Found existing connection for Apple ID: {}", sub);

                let user = app_state.db.get_user_by_uuid(connection.user_id)?;

                // CRITICAL: Verify user belongs to the current project
                if user.project_id != project.id {
                    debug!("User found but belongs to different project");
                    // Treat as if no connection exists for this project
                    None
                } else {
                    // Update the connection with the new token
                    // For Apple web flow, use the refresh token if available, otherwise empty string
                    let token_to_store = token
                        .refresh_token()
                        .map(|rt| rt.secret().to_string())
                        .unwrap_or_else(|| "".to_string());

                    update_provider_connection(
                        &app_state,
                        &user,
                        apple_db_provider.id,
                        &token_to_store,
                    )
                    .await?;

                    Some(user)
                }
            } else {
                // No existing connection found
                debug!("No existing connection found for Apple ID: {}", sub);
                None
            };

            // If user was found, return that user
            if let Some(user) = existing_user {
                debug!("Using existing user for Apple OAuth");
                user
            } else {
                // For new users, we absolutely need a valid email (not empty)
                debug!(
                    "No existing user found with Apple ID: {}, creating new user",
                    sub
                );

                // Make sure we have a non-empty email
                let email = apple_user
                    .email
                    .clone()
                    .filter(|e| !e.is_empty())
                    .ok_or_else(|| {
                        error!("No valid email found in Apple token for new user");
                        ApiError::NoEmailFound
                    })?;

                // For Apple web flow, use the refresh token if available, otherwise empty string
                let token_to_store = token
                    .refresh_token()
                    .map(|rt| rt.secret().to_string())
                    .unwrap_or_else(|| "".to_string());

                find_or_create_user_from_oauth(
                    &app_state,
                    email,
                    sub,
                    "apple",
                    token_to_store, // Store refresh token instead of access token
                    apple_user.name.clone(),
                    project.id,
                )
                .await?
            }
        }
        _ => {
            error!("Unsupported provider: {}", provider_name);
            return Err(ApiError::InternalServerError);
        }
    };

    // Generate JWT tokens
    let access_token = NewToken::new(&user, TokenType::Access, &app_state).map_err(|e| {
        error!("Failed to generate access token: {:?}", e);
        ApiError::InternalServerError
    })?;
    let refresh_token = NewToken::new(&user, TokenType::Refresh, &app_state).map_err(|e| {
        error!("Failed to generate refresh token: {:?}", e);
        ApiError::InternalServerError
    })?;

    let auth_response = OAuthCallbackResponse {
        id: user.get_id(),
        email: user
            .get_email()
            .expect("OAuth user must have email")
            .to_string(),
        access_token: access_token.token,
        refresh_token: refresh_token.token,
    };

    debug!("Exiting {} callback function", provider_name);
    encrypt_response(&app_state, &session_id, &auth_response).await
}

async fn fetch_github_user(
    access_token: &str,
    github_provider: &GithubProvider,
) -> Result<GithubUser, ApiError> {
    let client = reqwest::Client::new();
    let user_url = &github_provider.user_info_url;

    debug!("Sending request to GitHub API: {}", user_url);
    let response = client
        .get(user_url)
        .header("Authorization", format!("token {}", access_token))
        .header("User-Agent", "OpenSecret")
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send request to GitHub API: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Get status and headers before consuming the response
    let status = response.status();
    let headers = response.headers().clone();
    debug!("GitHub API response status: {}", status);
    trace!("GitHub API response headers: {:?}", headers);

    if !status.is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error body".to_string());
        error!(
            "GitHub API returned non-success status: {} {}",
            status,
            status.canonical_reason().unwrap_or("")
        );
        error!("Error response body: {}", error_body);
        return Err(ApiError::InternalServerError);
    }

    let user_body = response.text().await.map_err(|e| {
        error!("Failed to read GitHub user response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    trace!("GitHub user response body: {}", user_body);

    let mut github_user: GithubUser = serde_json::from_str(&user_body).map_err(|e| {
        error!("Failed to parse GitHub user JSON: {:?}", e);
        error!("GitHub user response body: {}", user_body);
        ApiError::InternalServerError
    })?;

    // If the email is not public, fetch the email separately
    if github_user.email.is_none() {
        let emails_url = "https://api.github.com/user/emails";
        debug!("Fetching GitHub user emails: {}", emails_url);
        let emails_response = client
            .get(emails_url)
            .header("Authorization", format!("token {}", access_token))
            .header("User-Agent", "OpenSecret")
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request for GitHub user emails: {:?}", e);
                ApiError::InternalServerError
            })?;

        let emails_status = emails_response.status();
        let emails_headers = emails_response.headers().clone();
        trace!("GitHub emails API response status: {}", emails_status);
        trace!("GitHub emails API response headers: {:?}", emails_headers);

        if !emails_status.is_success() {
            let error_body = emails_response
                .text()
                .await
                .unwrap_or_else(|_| "Unable to read error body".to_string());
            error!(
                "GitHub API returned non-success status for emails: {} {}",
                emails_status,
                emails_status.canonical_reason().unwrap_or("")
            );
            error!("Error response body for emails: {}", error_body);
            return Err(ApiError::InternalServerError);
        }

        let emails_body = emails_response.text().await.map_err(|e| {
            error!("Failed to read GitHub emails response body: {:?}", e);
            ApiError::InternalServerError
        })?;

        trace!("GitHub emails response body: {}", emails_body);

        let emails: Vec<GithubEmail> = serde_json::from_str(&emails_body).map_err(|e| {
            error!("Failed to parse GitHub emails JSON: {:?}", e);
            error!("GitHub emails response body: {}", emails_body);
            ApiError::InternalServerError
        })?;

        github_user.email = emails
            .into_iter()
            .find(|e| e.primary && e.verified)
            .map(|e| e.email);
    }

    // If we still don't have an email, return an error
    if github_user.email.is_none() {
        error!("No valid email found for GitHub user");
        return Err(ApiError::NoEmailFound);
    }

    Ok(github_user)
}

async fn fetch_google_user(
    access_token: &str,
    google_provider: &GoogleProvider,
) -> Result<GoogleUser, ApiError> {
    let client = reqwest::Client::new();
    let user_url = &google_provider.user_info_url;

    debug!("Sending request to Google API: {}", user_url);
    let response = client
        .get(user_url)
        .header(AUTHORIZATION, format!("Bearer {}", access_token))
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send request to Google API: {:?}", e);
            ApiError::InternalServerError
        })?;

    let status = response.status();
    if !status.is_success() {
        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error body".to_string());
        error!(
            "Google API returned non-success status: {} {}",
            status, error_body
        );
        return Err(ApiError::InternalServerError);
    }

    let google_user: GoogleUser = response.json().await.map_err(|e| {
        error!("Failed to parse Google user JSON: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Ensure email is present and verified
    if google_user.email.is_empty() || !google_user.email_verified {
        error!("Google user email is not present or not verified");
        return Err(ApiError::BadRequest);
    }

    Ok(google_user)
}

async fn fetch_apple_user(
    app_state: &AppState,
    id_token: &str,
    client_id: &str,
) -> Result<AppleUser, ApiError> {
    debug!("Parsing and validating Apple ID token");

    // Validate the token using the shared verifier (no nonce validation for now in web flow)
    let claims =
        validate_apple_native_token(&app_state.apple_jwt_verifier, id_token, client_id, None)
            .await?;

    // Create an AppleUser from the claims
    let apple_user = AppleUser {
        sub: claims.sub,
        email: claims.email,
        email_verified: claims.email_verified,
        name: None, // Apple doesn't include name in JWT, name comes from frontend
    };

    // Check if we have an email and it's verified
    if let Some(email) = &apple_user.email {
        if !email.is_empty() && apple_user.email_verified.unwrap_or(false) {
            return Ok(apple_user);
        }
    }

    // If we reach here, we either don't have an email or it's not verified
    error!("Apple user email is not present or not verified");
    Err(ApiError::NoEmailFound)
}

async fn find_or_create_user_from_oauth(
    app_state: &AppState,
    email: String,
    provider_user_id: String,
    provider_name: &str,
    access_token: String,
    user_name: Option<String>,
    project_id: i32,
) -> Result<User, ApiError> {
    let provider = app_state
        .db
        .get_oauth_provider_by_name(provider_name)
        .map_err(|e| {
            error!("Failed to get {} OAuth provider: {:?}", provider_name, e);
            ApiError::InternalServerError
        })?
        .ok_or_else(|| {
            error!("{} OAuth provider not found", provider_name);
            ApiError::InternalServerError
        })?;

    // Try to find the user by email
    match app_state.db.get_user_by_email(email.clone(), project_id) {
        Ok(existing_user) => {
            // User exists, check if they have a connection with the provider
            let existing_connection = app_state
                .db
                .get_user_oauth_connection_by_user_and_provider(existing_user.uuid, provider.id)
                .map_err(|e| {
                    error!("Failed to get existing OAuth connection: {:?}", e);
                    ApiError::InternalServerError
                })?;

            if existing_connection.is_some() {
                // User has already linked their account, update the token
                update_provider_connection(app_state, &existing_user, provider.id, &access_token)
                    .await?;
                Ok(existing_user)
            } else {
                // User exists but hasn't linked the provider before
                error!("User exists but hasn't linked {} before", provider_name);
                Err(ApiError::UserExistsNotLinked)
            }
        }
        Err(DBError::UserNotFound) => {
            // Generate private key for new user
            let user_seed_words =
                generate_twelve_word_seed(app_state.aws_credential_manager.clone())
                    .await
                    .map_err(|_e| ApiError::InternalServerError)?
                    .to_string();

            let secret_key = SecretKey::from_slice(&app_state.enclave_key.clone())
                .map_err(|_e| ApiError::EncryptionError)?;

            let encrypted_key = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

            // Create new user
            let new_user = NewUser::new(Some(email.clone()), None, project_id, encrypted_key)
                .with_name(user_name.unwrap_or_default());

            let user = app_state.db.create_user(new_user).map_err(|e| {
                error!("Failed to create new user: {:?}", e);
                ApiError::InternalServerError
            })?;

            // Create connection for the new user
            create_provider_connection(
                app_state,
                &user,
                provider.id,
                provider_user_id,
                &access_token,
            )
            .await?;

            // Create email verification entry as already verified
            let new_verification = NewEmailVerification::new(user.uuid, 24, true);
            app_state
                .db
                .create_email_verification(new_verification)
                .map_err(|e| {
                    error!("Error creating email verification: {:?}", e);
                    ApiError::InternalServerError
                })?;

            // Handle new user registration
            handle_new_user_registration(app_state, &user, false).await?;

            Ok(user)
        }
        Err(e) => {
            error!("Database error when fetching user: {:?}", e);
            Err(ApiError::InternalServerError)
        }
    }
}

/// Handler for Apple native sign-in (iOS Sign in with Apple)
pub async fn handle_apple_native_signin(
    State(app_state): State<Arc<AppState>>,
    Extension(request): Extension<AppleNativeSignInRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<OAuthCallbackResponse>>, ApiError> {
    debug!("Handling Apple native sign-in");

    // Get project
    let project = app_state
        .db
        .get_org_project_by_client_id(request.client_id)
        .map_err(|_| ApiError::BadRequest)?;

    // Get project OAuth settings
    let oauth_settings = app_state
        .db
        .get_project_oauth_settings(project.id)?
        .ok_or_else(|| {
            error!("OAuth settings not found for project");
            ApiError::BadRequest
        })?;

    // Verify Apple OAuth is enabled
    if !oauth_settings.apple_oauth_enabled {
        error!("Apple OAuth is not enabled for this project");
        return Err(ApiError::BadRequest);
    }

    // Get Apple client ID from project settings
    let apple_oauth_settings = oauth_settings.apple_oauth_settings.ok_or_else(|| {
        error!("Apple OAuth settings not configured");
        ApiError::BadRequest
    })?;

    // Use the client ID from OAuth settings
    let client_id = apple_oauth_settings.client_id;

    // Verify the Apple JWT token using the shared verifier with nonce if provided
    debug!("Verifying Apple identity token");
    let claims = validate_apple_native_token(
        &app_state.apple_jwt_verifier,
        &request.identity_token,
        &client_id,
        request.nonce.as_deref(),
    )
    .await?;

    // If user_identifier is provided, verify it matches the sub from the token
    if let Some(user_id) = &request.user_identifier {
        if *user_id != claims.sub {
            error!("User identifier mismatch in Apple token");
            return Err(ApiError::BadRequest);
        }
    }

    // Use the sub claim from the token as the verified user identifier
    let verified_user_id = claims.sub.clone();

    // Get the Apple provider from the database
    let apple_provider = app_state
        .db
        .get_oauth_provider_by_name("apple")
        .map_err(|e| {
            error!("Failed to get Apple OAuth provider: {:?}", e);
            ApiError::InternalServerError
        })?
        .ok_or_else(|| {
            error!("Apple OAuth provider not found");
            ApiError::InternalServerError
        })?;

    // For Apple native flow, we don't need to store any tokens
    // The iOS device handles authentication and token management
    let access_token = "".to_string(); // Empty string instead of storing the ID token

    // For Apple, we need a special approach:
    // 1. First, try to find any existing users with this Apple ID
    // 2. If found, just use that user - no need for email
    // 3. Only if this is a first-time user, we need an email

    // Directly query for existing connection with this Apple ID
    if let Some(connection) = app_state
        .db
        .get_user_oauth_connection_by_provider_and_provider_user_id(
            apple_provider.id,
            &verified_user_id,
        )?
    {
        // Found a connection - get the user
        debug!(
            "Found existing connection for Apple ID: {}",
            verified_user_id
        );

        let user = app_state.db.get_user_by_uuid(connection.user_id)?;

        // CRITICAL: Verify user belongs to the current project
        if user.project_id != project.id {
            debug!("User found but belongs to different project, treating as new user");
            // Fall through to create new user for this project
        } else {
            // Update the connection with an empty string instead of storing the ID token
            update_provider_connection(&app_state, &user, apple_provider.id, &access_token).await?;

            // Generate JWT tokens
            let access_token =
                NewToken::new(&user, TokenType::Access, &app_state).map_err(|e| {
                    error!("Failed to generate access token: {:?}", e);
                    ApiError::InternalServerError
                })?;

            let refresh_token =
                NewToken::new(&user, TokenType::Refresh, &app_state).map_err(|e| {
                    error!("Failed to generate refresh token: {:?}", e);
                    ApiError::InternalServerError
                })?;

            let auth_response = OAuthCallbackResponse {
                id: user.get_id(),
                email: user
                    .get_email()
                    .expect("OAuth user must have email")
                    .to_string(),
                access_token: access_token.token,
                refresh_token: refresh_token.token,
            };

            debug!("Apple sign-in successful for existing user");
            return encrypt_response(&app_state, &session_id, &auth_response).await;
        }
    }

    // If we get here, user doesn't exist - need to create new user
    debug!(
        "No existing user found with Apple ID: {}, creating new user",
        verified_user_id
    );

    // For new users, we absolutely need an email
    // Determine the email to use - prioritize the one from the token if available
    // Make sure we actually have an email - don't allow empty strings
    let email = claims
        .email
        .or(request.email.clone())
        .filter(|e| !e.is_empty())
        .ok_or_else(|| {
            error!("No valid email found in Apple token or request for new user");
            ApiError::NoEmailFound
        })?;

    // Construct a name from given_name and family_name if provided
    // and ensure it's None if empty or just whitespace
    let user_name = match (request.given_name.clone(), request.family_name.clone()) {
        (Some(given), Some(family)) => {
            let combined = format!("{} {}", given.trim(), family.trim())
                .trim()
                .to_string();
            if combined.is_empty() {
                None
            } else {
                Some(combined)
            }
        }
        (Some(given), None) => {
            let trimmed = given.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        (None, Some(family)) => {
            let trimmed = family.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        (None, None) => None,
    };

    // Create the new user
    let user = find_or_create_user_from_oauth(
        &app_state,
        email,
        verified_user_id,
        "apple",
        access_token,
        user_name,
        project.id,
    )
    .await?;

    // Generate JWT tokens
    let access_token = NewToken::new(&user, TokenType::Access, &app_state).map_err(|e| {
        error!("Failed to generate access token: {:?}", e);
        ApiError::InternalServerError
    })?;

    let refresh_token = NewToken::new(&user, TokenType::Refresh, &app_state).map_err(|e| {
        error!("Failed to generate refresh token: {:?}", e);
        ApiError::InternalServerError
    })?;

    let auth_response = OAuthCallbackResponse {
        id: user.get_id(),
        email: user
            .get_email()
            .expect("OAuth user must have email")
            .to_string(),
        access_token: access_token.token,
        refresh_token: refresh_token.token,
    };

    debug!("Apple native sign-in successful for new user");
    encrypt_response(&app_state, &session_id, &auth_response).await
}

async fn update_provider_connection(
    app_state: &AppState,
    user: &User,
    provider_id: i32,
    token: &str,
) -> Result<(), ApiError> {
    let mut connection = app_state
        .db
        .get_user_oauth_connection_by_user_and_provider(user.uuid, provider_id)
        .map_err(|e| {
            error!("Failed to get existing OAuth connection: {:?}", e);
            ApiError::InternalServerError
        })?
        .expect("Connection should exist");

    // Only encrypt and update token if it's not empty
    if !token.is_empty() {
        let encrypted_token = encrypt_access_token(app_state, token).await?;
        connection.access_token_enc = encrypted_token;
    }

    app_state
        .db
        .update_user_oauth_connection(&connection)
        .map_err(|e| {
            error!("Failed to update OAuth connection: {:?}", e);
            ApiError::InternalServerError
        })?;

    Ok(())
}

async fn create_provider_connection(
    app_state: &AppState,
    user: &User,
    provider_id: i32,
    provider_user_id: String,
    access_token: &str,
) -> Result<(), ApiError> {
    // Only encrypt/store the token if it's not empty
    let encrypted_token = if !access_token.is_empty() {
        Some(encrypt_access_token(app_state, access_token).await?)
    } else {
        None
    };

    let new_connection = NewUserOAuthConnection {
        user_id: user.uuid,
        provider_id,
        provider_user_id,
        access_token_enc: encrypted_token.unwrap_or_default(),
        refresh_token_enc: None,
        expires_at: None,
    };

    app_state
        .db
        .create_user_oauth_connection(new_connection)
        .map_err(|e| {
            error!("Failed to create new OAuth connection: {:?}", e);
            ApiError::InternalServerError
        })?;

    Ok(())
}

async fn encrypt_access_token(
    app_state: &AppState,
    access_token: &str,
) -> Result<Vec<u8>, ApiError> {
    let secret_key = SecretKey::from_slice(&app_state.enclave_key).map_err(|e| {
        error!("Failed to create SecretKey from enclave key: {:?}", e);
        ApiError::InternalServerError
    })?;
    Ok(encrypt::encrypt_with_key(&secret_key, access_token.as_bytes()).await)
}
