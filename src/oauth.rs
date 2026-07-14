use crate::db::DBConnection;
use crate::models::oauth::NewOAuthProvider;
use crate::Error;
use async_trait::async_trait;
use oauth2::{
    basic::BasicClient, AuthUrl, ClientId, ClientSecret, CsrfToken, RedirectUrl, Scope, TokenUrl,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, info};
use uuid::Uuid;

// OAuth redirects normally complete within seconds. Ten minutes leaves room for a user to
// authenticate without retaining abandoned CSRF state for the encrypted session lifetime.
const OAUTH_STATE_TTL: Duration = Duration::from_secs(10 * 60);
// This limit applies independently to each configured provider.
const OAUTH_STATE_CAPACITY: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OAuthState {
    pub csrf_token: String,
    pub client_id: Uuid,
}

#[derive(Debug, Clone)]
struct OAuthStateStore {
    inner: Arc<Mutex<OAuthStateStoreInner>>,
    ttl: Duration,
    capacity: usize,
}

#[derive(Debug, Default)]
struct OAuthStateStoreInner {
    entries: HashMap<String, StoredOAuthState>,
}

#[derive(Debug)]
struct StoredOAuthState {
    state: OAuthState,
    expires_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("OAuth state capacity reached")]
pub struct OAuthStateCapacityError;

impl OAuthStateStore {
    fn new() -> Self {
        Self::with_limits(OAUTH_STATE_TTL, OAUTH_STATE_CAPACITY)
    }

    fn with_limits(ttl: Duration, capacity: usize) -> Self {
        assert!(capacity > 0, "OAuth state capacity must be positive");
        Self {
            inner: Arc::new(Mutex::new(OAuthStateStoreInner::default())),
            ttl,
            capacity,
        }
    }

    async fn store(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.store_at(csrf_token, state, Instant::now()).await
    }

    async fn store_at(
        &self,
        csrf_token: &str,
        state: OAuthState,
        now: Instant,
    ) -> Result<(), OAuthStateCapacityError> {
        let mut inner = self.inner.lock().await;
        inner
            .entries
            .retain(|_, stored_state| stored_state.expires_at > now);

        if !inner.entries.contains_key(csrf_token) && inner.entries.len() >= self.capacity {
            return Err(OAuthStateCapacityError);
        }

        inner.entries.insert(
            csrf_token.to_string(),
            StoredOAuthState {
                state,
                expires_at: now
                    .checked_add(self.ttl)
                    .expect("OAuth state TTL must fit in Instant"),
            },
        );
        Ok(())
    }

    async fn consume(&self, state: &OAuthState) -> bool {
        self.consume_at(state, Instant::now()).await
    }

    async fn consume_at(&self, state: &OAuthState, now: Instant) -> bool {
        let mut inner = self.inner.lock().await;
        inner
            .entries
            .retain(|_, stored_state| stored_state.expires_at > now);

        let matches = inner
            .entries
            .get(&state.csrf_token)
            .is_some_and(|stored_state| stored_state.state == *state);
        if matches {
            inner.entries.remove(&state.csrf_token);
        }
        matches
    }

    #[cfg(test)]
    async fn len_at(&self, now: Instant) -> usize {
        let mut inner = self.inner.lock().await;
        inner
            .entries
            .retain(|_, stored_state| stored_state.expires_at > now);
        inner.entries.len()
    }
}

#[derive(Debug, Clone)]
pub struct GithubProvider {
    pub auth_url: String,
    pub token_url: String,
    pub user_info_url: String,
    state_store: OAuthStateStore,
}

impl GithubProvider {
    pub async fn new(db: Arc<dyn DBConnection + Send + Sync>) -> Result<Self, Error> {
        let auth_url = "https://github.com/login/oauth/authorize".to_string();
        let token_url = "https://github.com/login/oauth/access_token".to_string();
        let user_info_url = "https://api.github.com/user".to_string();

        let provider = Self {
            auth_url,
            token_url,
            user_info_url,
            state_store: OAuthStateStore::new(),
        };

        // Ensure the provider exists in the database
        provider.ensure_provider_exists(db).await?;

        info!("GitHub OAuth provider initialized successfully");
        Ok(provider)
    }

    pub async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        let auth_url = AuthUrl::new(self.auth_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid auth URL: {}", e)))?;
        let token_url = TokenUrl::new(self.token_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid token URL: {}", e)))?;

        Ok(BasicClient::new(
            ClientId::new(client_id),
            Some(ClientSecret::new(client_secret)),
            auth_url,
            Some(token_url),
        )
        .set_redirect_uri(
            RedirectUrl::new(redirect_url)
                .map_err(|e| Error::OAuthError(format!("Invalid redirect URL: {}", e)))?,
        ))
    }

    pub async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        let (auth_url, csrf_token) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("user:email".to_string()))
            .url();

        (auth_url.to_string(), csrf_token)
    }

    pub async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.state_store.store(csrf_token, state).await
    }

    pub async fn consume_state(&self, state: &OAuthState) -> bool {
        self.state_store.consume(state).await
    }

    async fn ensure_provider_exists(
        &self,
        db: Arc<dyn DBConnection + Send + Sync>,
    ) -> Result<(), Error> {
        debug!("Checking if GitHub OAuth provider exists in the database");
        let existing_provider = db.get_oauth_provider_by_name("github")?;

        if existing_provider.is_none() {
            info!("GitHub OAuth provider not found in database, creating new entry");
            let new_provider = NewOAuthProvider {
                name: "github".to_string(),
                auth_url: self.auth_url.clone(),
                token_url: self.token_url.clone(),
                user_info_url: self.user_info_url.clone(),
            };

            match db.create_oauth_provider(new_provider) {
                Ok(_) => info!("GitHub OAuth provider successfully added to database"),
                Err(e) => {
                    error!(
                        "Failed to create GitHub OAuth provider in database: {:?}",
                        e
                    );
                    return Err(e.into());
                }
            }
        } else {
            debug!("GitHub OAuth provider already exists in database");
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GoogleProvider {
    pub auth_url: String,
    pub token_url: String,
    pub user_info_url: String,
    state_store: OAuthStateStore,
}

impl GoogleProvider {
    pub async fn new(db: Arc<dyn DBConnection + Send + Sync>) -> Result<Self, Error> {
        let auth_url = "https://accounts.google.com/o/oauth2/v2/auth".to_string();
        let token_url = "https://oauth2.googleapis.com/token".to_string();
        let user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo".to_string();

        let provider = Self {
            auth_url,
            token_url,
            user_info_url,
            state_store: OAuthStateStore::new(),
        };

        // Ensure the provider exists in the database
        provider.ensure_provider_exists(db).await?;

        info!("Google OAuth provider initialized successfully");
        Ok(provider)
    }

    pub async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        let auth_url = AuthUrl::new(self.auth_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid auth URL: {}", e)))?;
        let token_url = TokenUrl::new(self.token_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid token URL: {}", e)))?;

        Ok(BasicClient::new(
            ClientId::new(client_id),
            Some(ClientSecret::new(client_secret)),
            auth_url,
            Some(token_url),
        )
        .set_redirect_uri(
            RedirectUrl::new(redirect_url)
                .map_err(|e| Error::OAuthError(format!("Invalid redirect URL: {}", e)))?,
        ))
    }

    pub async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        let (auth_url, csrf_token) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("email".to_string()))
            .add_scope(Scope::new("profile".to_string()))
            .url();

        (auth_url.to_string(), csrf_token)
    }

    pub async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.state_store.store(csrf_token, state).await
    }

    pub async fn consume_state(&self, state: &OAuthState) -> bool {
        self.state_store.consume(state).await
    }

    async fn ensure_provider_exists(
        &self,
        db: Arc<dyn DBConnection + Send + Sync>,
    ) -> Result<(), Error> {
        debug!("Checking if Google OAuth provider exists in the database");
        let existing_provider = db.get_oauth_provider_by_name("google")?;

        if existing_provider.is_none() {
            info!("Google OAuth provider not found in database, creating new entry");
            let new_provider = NewOAuthProvider {
                name: "google".to_string(),
                auth_url: self.auth_url.clone(),
                token_url: self.token_url.clone(),
                user_info_url: self.user_info_url.clone(),
            };

            match db.create_oauth_provider(new_provider) {
                Ok(_) => info!("Google OAuth provider successfully added to database"),
                Err(e) => {
                    error!(
                        "Failed to create Google OAuth provider in database: {:?}",
                        e
                    );
                    return Err(e.into());
                }
            }
        } else {
            debug!("Google OAuth provider already exists in database");
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AppleProvider {
    pub auth_url: String,
    pub token_url: String,
    pub jwks_url: String,
    state_store: OAuthStateStore,
    // No need for user_info_url as Apple doesn't have a separate endpoint - all info is in the ID token
}

impl AppleProvider {
    pub async fn new(db: Arc<dyn DBConnection + Send + Sync>) -> Result<Self, Error> {
        let auth_url = "https://appleid.apple.com/auth/authorize".to_string();
        let token_url = "https://appleid.apple.com/auth/token".to_string();
        let jwks_url = "https://appleid.apple.com/auth/keys".to_string();

        let provider = Self {
            auth_url,
            token_url,
            jwks_url,
            state_store: OAuthStateStore::new(),
        };

        // Ensure the provider exists in the database
        provider.ensure_provider_exists(db).await?;

        info!("Apple OAuth provider initialized successfully");
        Ok(provider)
    }

    pub async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        let auth_url = AuthUrl::new(self.auth_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid auth URL: {}", e)))?;
        let token_url = TokenUrl::new(self.token_url.clone())
            .map_err(|e| Error::OAuthError(format!("Invalid token URL: {}", e)))?;

        Ok(BasicClient::new(
            ClientId::new(client_id),
            Some(ClientSecret::new(client_secret)),
            auth_url,
            Some(token_url),
        )
        .set_redirect_uri(
            RedirectUrl::new(redirect_url)
                .map_err(|e| Error::OAuthError(format!("Invalid redirect URL: {}", e)))?,
        ))
    }

    pub async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        // For Apple Sign In, we'll primarily use the JS SDK on the frontend
        // This URL is fallback for non-JS environments
        let (auth_url, csrf_token) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("name".to_string()))
            .add_scope(Scope::new("email".to_string()))
            .add_extra_param("response_type", "code id_token") // Apple best practice
            .add_extra_param("response_mode", "form_post") // Required by Apple
            .url();

        (auth_url.to_string(), csrf_token)
    }

    pub async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.state_store.store(csrf_token, state).await
    }

    pub async fn consume_state(&self, state: &OAuthState) -> bool {
        self.state_store.consume(state).await
    }

    async fn ensure_provider_exists(
        &self,
        db: Arc<dyn DBConnection + Send + Sync>,
    ) -> Result<(), Error> {
        debug!("Checking if Apple OAuth provider exists in the database");
        let existing_provider = db.get_oauth_provider_by_name("apple")?;

        if existing_provider.is_none() {
            info!("Apple OAuth provider not found in database, creating new entry");
            // Set a dummy user_info_url as Apple doesn't have one, everything comes from the JWT
            let new_provider = NewOAuthProvider {
                name: "apple".to_string(),
                auth_url: self.auth_url.clone(),
                token_url: self.token_url.clone(),
                user_info_url: self.jwks_url.clone(), // Use JWKS URL in place of user_info_url
            };

            match db.create_oauth_provider(new_provider) {
                Ok(_) => info!("Apple OAuth provider successfully added to database"),
                Err(e) => {
                    error!("Failed to create Apple OAuth provider in database: {:?}", e);
                    return Err(e.into());
                }
            }
        } else {
            debug!("Apple OAuth provider already exists in database");
        }

        Ok(())
    }
}

#[async_trait]
pub trait OAuthProvider: Send + Sync + 'static {
    fn as_github(&self) -> Option<&GithubProvider> {
        None
    }

    fn as_google(&self) -> Option<&GoogleProvider> {
        None
    }

    fn as_apple(&self) -> Option<&AppleProvider> {
        None
    }

    async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken);
    async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError>;
    async fn consume_state(&self, state: &OAuthState) -> bool;
    async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error>;
}

pub struct OAuthManager {
    providers: HashMap<String, Box<dyn OAuthProvider>>,
}

impl OAuthManager {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn add_provider(&mut self, name: String, provider: Box<dyn OAuthProvider>) {
        self.providers.insert(name, provider);
    }

    pub fn get_provider(&self, name: &str) -> Option<&dyn OAuthProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }
}

#[async_trait]
impl OAuthProvider for GithubProvider {
    fn as_github(&self) -> Option<&GithubProvider> {
        Some(self)
    }

    async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        self.generate_authorize_url(client).await
    }

    async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.store_state(csrf_token, state).await
    }

    async fn consume_state(&self, state: &OAuthState) -> bool {
        self.consume_state(state).await
    }

    async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        self.build_client(client_id, client_secret, redirect_url)
            .await
    }
}

#[async_trait]
impl OAuthProvider for GoogleProvider {
    fn as_google(&self) -> Option<&GoogleProvider> {
        Some(self)
    }

    async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        self.generate_authorize_url(client).await
    }

    async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.store_state(csrf_token, state).await
    }

    async fn consume_state(&self, state: &OAuthState) -> bool {
        self.consume_state(state).await
    }

    async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        self.build_client(client_id, client_secret, redirect_url)
            .await
    }
}

#[async_trait]
impl OAuthProvider for AppleProvider {
    fn as_apple(&self) -> Option<&AppleProvider> {
        Some(self)
    }

    async fn generate_authorize_url(&self, client: &BasicClient) -> (String, CsrfToken) {
        self.generate_authorize_url(client).await
    }

    async fn store_state(
        &self,
        csrf_token: &str,
        state: OAuthState,
    ) -> Result<(), OAuthStateCapacityError> {
        self.store_state(csrf_token, state).await
    }

    async fn consume_state(&self, state: &OAuthState) -> bool {
        self.consume_state(state).await
    }

    async fn build_client(
        &self,
        client_id: String,
        client_secret: String,
        redirect_url: String,
    ) -> Result<BasicClient, Error> {
        self.build_client(client_id, client_secret, redirect_url)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Barrier;

    fn state(csrf_token: &str, client_id: u128) -> OAuthState {
        OAuthState {
            csrf_token: csrf_token.to_string(),
            client_id: Uuid::from_u128(client_id),
        }
    }

    #[tokio::test]
    async fn oauth_state_expires_at_ttl() {
        let ttl = Duration::from_secs(10);
        let store = OAuthStateStore::with_limits(ttl, 2);
        let now = Instant::now();
        let oauth_state = state("expired", 1);

        store
            .store_at(&oauth_state.csrf_token, oauth_state.clone(), now)
            .await
            .unwrap();

        assert!(!store.consume_at(&oauth_state, now + ttl).await);
        assert_eq!(store.len_at(now + ttl).await, 0);
    }

    #[tokio::test]
    async fn oauth_state_store_rejects_new_entry_at_capacity() {
        let store = OAuthStateStore::with_limits(Duration::from_secs(60), 2);
        let now = Instant::now();
        let first = state("first", 1);
        let second = state("second", 2);
        let third = state("third", 3);

        store
            .store_at(&first.csrf_token, first.clone(), now)
            .await
            .unwrap();
        store
            .store_at(&second.csrf_token, second.clone(), now)
            .await
            .unwrap();

        assert_eq!(
            store.store_at(&third.csrf_token, third.clone(), now).await,
            Err(OAuthStateCapacityError)
        );

        assert_eq!(store.len_at(now).await, 2);
        assert!(store.consume_at(&first, now).await);
        assert!(store.consume_at(&second, now).await);
        assert!(!store.consume_at(&third, now).await);
    }

    #[tokio::test]
    async fn oauth_state_store_replaces_existing_entry_at_capacity() {
        let store = OAuthStateStore::with_limits(Duration::from_secs(60), 1);
        let now = Instant::now();
        let original = state("shared", 1);
        let replacement = state("shared", 2);

        store
            .store_at(&original.csrf_token, original.clone(), now)
            .await
            .unwrap();
        store
            .store_at(&replacement.csrf_token, replacement.clone(), now)
            .await
            .unwrap();

        assert_eq!(store.len_at(now).await, 1);
        assert!(!store.consume_at(&original, now).await);
        assert!(store.consume_at(&replacement, now).await);
    }

    #[tokio::test]
    async fn oauth_state_store_prunes_expired_entries_before_capacity_check() {
        let ttl = Duration::from_secs(10);
        let store = OAuthStateStore::with_limits(ttl, 2);
        let now = Instant::now();
        let expired = state("expired", 1);
        let live = state("live", 2);
        let newest = state("newest", 3);

        store
            .store_at(&expired.csrf_token, expired.clone(), now)
            .await
            .unwrap();
        store
            .store_at(&live.csrf_token, live.clone(), now + Duration::from_secs(5))
            .await
            .unwrap();
        store
            .store_at(&newest.csrf_token, newest.clone(), now + ttl)
            .await
            .unwrap();

        assert_eq!(store.len_at(now + ttl).await, 2);
        assert!(!store.consume_at(&expired, now + ttl).await);
        assert!(store.consume_at(&live, now + ttl).await);
        assert!(store.consume_at(&newest, now + ttl).await);
    }

    #[tokio::test]
    async fn mismatched_oauth_callback_does_not_consume_valid_state() {
        let store = OAuthStateStore::with_limits(Duration::from_secs(60), 2);
        let now = Instant::now();
        let valid = state("csrf-token", 1);
        let mismatched = state("csrf-token", 2);

        store
            .store_at(&valid.csrf_token, valid.clone(), now)
            .await
            .unwrap();

        assert!(!store.consume_at(&mismatched, now).await);
        assert!(store.consume_at(&valid, now).await);
    }

    #[tokio::test]
    async fn valid_oauth_state_can_only_be_consumed_once() {
        let store = OAuthStateStore::with_limits(Duration::from_secs(60), 2);
        let now = Instant::now();
        let oauth_state = state("one-time", 1);

        store
            .store_at(&oauth_state.csrf_token, oauth_state.clone(), now)
            .await
            .unwrap();

        assert!(store.consume_at(&oauth_state, now).await);
        assert!(!store.consume_at(&oauth_state, now).await);
    }

    #[tokio::test]
    async fn concurrent_oauth_callbacks_have_one_winner() {
        let store = OAuthStateStore::with_limits(Duration::from_secs(60), 2);
        let now = Instant::now();
        let oauth_state = state("raced", 1);
        store
            .store_at(&oauth_state.csrf_token, oauth_state.clone(), now)
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(3));
        let first = {
            let store = store.clone();
            let barrier = barrier.clone();
            let oauth_state = oauth_state.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                store.consume_at(&oauth_state, now).await
            })
        };
        let second = {
            let store = store.clone();
            let barrier = barrier.clone();
            let oauth_state = oauth_state.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                store.consume_at(&oauth_state, now).await
            })
        };

        barrier.wait().await;
        let consumed = [first.await.unwrap(), second.await.unwrap()]
            .into_iter()
            .filter(|was_consumed| *was_consumed)
            .count();

        assert_eq!(consumed, 1);
    }
}
