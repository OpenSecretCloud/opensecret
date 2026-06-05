use crate::models::org_project_secrets::NewOrgProjectSecret;
use crate::models::project_settings::{EmailSettings, OAuthProviderSettings, OAuthSettings};
use crate::web::platform::{
    PROJECT_GITHUB_OAUTH_SECRET, PROJECT_GOOGLE_OAUTH_SECRET, PROJECT_RESEND_API_KEY,
};
use crate::{AppMode, AppState, Error};
use secp256k1::SecretKey;
use std::sync::Arc;
use tracing::{debug, error, info};

#[cfg(feature = "seed-wrap-translation")]
use crate::{
    db::DBError,
    encrypt::{decrypt_with_key, EncryptError},
    models::{
        app_data_migrations::{AppDataMigration, AppDataMigrationError, NewAppDataMigration},
        oauth::{OAuthError, OAuthProvider, UserOAuthConnection},
        schema::{user_seed_wrappings, users},
        user_seed_wrappings::{NewUserSeedWrapping, UserSeedWrappingError},
        users::User,
    },
    seed_wrapping::{
        compute_oauth_auth_binding, compute_password_auth_binding, encrypt_seed_v1,
        normalize_email_login_identifier, normalize_guest_login_identifier,
        oauth_credential_lookup_hash, password_credential_lookup_hash, CredentialKind,
        PasswordLoginIdentifierKind, SEED_WRAP_VERSION_V1,
    },
};
#[cfg(feature = "seed-wrap-translation")]
use diesel::{
    pg::PgConnection,
    prelude::*,
    sql_query,
    sql_types::{BigInt, Integer, Text},
    Connection, QueryableByName,
};
#[cfg(feature = "seed-wrap-translation")]
use uuid::Uuid;

#[cfg(feature = "seed-wrap-translation")]
const AEAD_SEED_WRAPPINGS_MIGRATION: &str = "aead_seed_wrappings_v1";

// TODO remove migration code now that this ran successfully
pub async fn run_migrations(
    app_state: &Arc<AppState>,
    github_client_secret: Option<String>,
    google_client_secret: Option<String>,
    github_client_id: Option<String>,
    google_client_id: Option<String>,
) -> Result<(), Error> {
    debug!("Starting migrations");
    #[cfg(feature = "seed-wrap-translation")]
    migrate_aead_seed_wrappings_v1(app_state)?;

    migrate_maple_project_settings(
        app_state,
        github_client_secret,
        google_client_secret,
        github_client_id,
        google_client_id,
    )
    .await?;
    debug!("Migrations completed successfully");
    Ok(())
}

#[cfg(feature = "seed-wrap-translation")]
#[derive(Debug, thiserror::Error)]
enum SeedWrapTranslationError {
    #[error("Diesel error: {0}")]
    Diesel(#[from] diesel::result::Error),
    #[error("App data migration error: {0}")]
    AppDataMigration(#[from] AppDataMigrationError),
    #[error("OAuth error: {0}")]
    OAuth(#[from] OAuthError),
    #[error("User seed wrapping error: {0}")]
    UserSeedWrapping(#[from] UserSeedWrappingError),
    #[error("Encryption error: {0}")]
    Encrypt(#[from] EncryptError),
    #[error("Invalid UTF-8 in decrypted database value: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("User {0} has no legacy seed_enc")]
    MissingSeed(Uuid),
    #[error("User {0} has no password or OAuth credential to migrate")]
    NoUsableCredential(Uuid),
    #[error("OAuth connection {connection_id} references missing provider {provider_id}")]
    MissingOAuthProvider {
        connection_id: i32,
        provider_id: i32,
    },
    #[error(
        "Duplicate OAuth provider subject for provider {provider_id}, subject {provider_user_id}, project {project_id}: {connection_count} connections"
    )]
    DuplicateOAuthSubject {
        provider_id: i32,
        provider_user_id: String,
        project_id: i32,
        connection_count: i64,
    },
    #[error(
        "Seed wrap postflight mismatch for {credential_kind}: expected {expected_count}, actual {actual_count}"
    )]
    PostflightWrapCountMismatch {
        credential_kind: String,
        expected_count: i64,
        actual_count: i64,
    },
}

#[cfg(feature = "seed-wrap-translation")]
impl From<SeedWrapTranslationError> for Error {
    fn from(error: SeedWrapTranslationError) -> Self {
        match error {
            SeedWrapTranslationError::Diesel(e) => Error::DatabaseError(DBError::QueryError(e)),
            SeedWrapTranslationError::AppDataMigration(e) => Error::DatabaseError(DBError::from(e)),
            SeedWrapTranslationError::OAuth(e) => Error::DatabaseError(DBError::from(e)),
            SeedWrapTranslationError::UserSeedWrapping(e) => Error::DatabaseError(DBError::from(e)),
            SeedWrapTranslationError::Encrypt(e) => Error::EncryptionError(e.to_string()),
            SeedWrapTranslationError::Utf8(e) => {
                Error::EncryptionError(format!("Invalid decrypted UTF-8: {e}"))
            }
            SeedWrapTranslationError::MissingSeed(user_uuid) => {
                Error::EncryptionError(format!("User {user_uuid} has no legacy seed_enc"))
            }
            SeedWrapTranslationError::NoUsableCredential(user_uuid) => {
                Error::EncryptionError(format!("User {user_uuid} has no usable credential"))
            }
            SeedWrapTranslationError::MissingOAuthProvider {
                connection_id,
                provider_id,
            } => Error::EncryptionError(format!(
                "OAuth connection {connection_id} references missing provider {provider_id}"
            )),
            SeedWrapTranslationError::DuplicateOAuthSubject {
                provider_id,
                provider_user_id,
                project_id,
                connection_count,
            } => Error::EncryptionError(format!(
                "Duplicate OAuth provider subject for provider {provider_id}, subject {provider_user_id}, project {project_id}: {connection_count} connections"
            )),
            SeedWrapTranslationError::PostflightWrapCountMismatch {
                credential_kind,
                expected_count,
                actual_count,
            } => Error::EncryptionError(format!(
                "Seed wrap postflight mismatch for {credential_kind}: expected {expected_count}, actual {actual_count}"
            )),
        }
    }
}

#[cfg(feature = "seed-wrap-translation")]
#[derive(QueryableByName)]
#[diesel(check_for_backend(diesel::pg::Pg))]
struct DuplicateOAuthSubject {
    #[diesel(sql_type = Integer)]
    provider_id: i32,
    #[diesel(sql_type = Text)]
    provider_user_id: String,
    #[diesel(sql_type = Integer)]
    project_id: i32,
    #[diesel(sql_type = BigInt)]
    connection_count: i64,
}

#[cfg(feature = "seed-wrap-translation")]
fn migrate_aead_seed_wrappings_v1(app_state: &Arc<AppState>) -> Result<(), Error> {
    info!("Checking AEAD seed wrapping translation migration");

    let legacy_secret_key = SecretKey::from_slice(&app_state.enclave_key)
        .map_err(|e| Error::EncryptionError(e.to_string()))?;

    let pool = app_state.db.get_pool();
    let conn = &mut pool
        .get()
        .map_err(|_| Error::DatabaseError(DBError::ConnectionError))?;

    conn.transaction::<_, SeedWrapTranslationError, _>(|conn| {
        sql_query("SELECT pg_advisory_xact_lock(hashtext($1))")
            .bind::<Text, _>(AEAD_SEED_WRAPPINGS_MIGRATION)
            .execute(conn)?;

        if AppDataMigration::get(conn, AEAD_SEED_WRAPPINGS_MIGRATION)?.is_some() {
            info!("AEAD seed wrapping translation already completed; skipping");
            return Ok(());
        }

        validate_no_duplicate_oauth_subjects_by_project(conn)?;

        let all_users = users::table.order(users::id.asc()).load::<User>(conn)?;
        let mut expected_wraps = 0usize;
        let mut expected_password_wraps = 0usize;
        let mut expected_oauth_wraps = 0usize;

        for user in all_users {
            let legacy_seed_enc = user
                .seed_encrypted()
                .ok_or(SeedWrapTranslationError::MissingSeed(user.uuid))?;
            let plaintext_seed = decrypt_with_key(&legacy_secret_key, legacy_seed_enc)?;
            let mut user_wraps = 0usize;

            if let Some(password_enc) = user.password_enc.as_ref() {
                let verifier_bytes = decrypt_with_key(&legacy_secret_key, password_enc)?;
                let verifier = String::from_utf8(verifier_bytes)?;
                upsert_password_seed_wrap(conn, app_state, &user, &verifier, &plaintext_seed)?;
                user_wraps += 1;
                expected_password_wraps += 1;
            }

            let oauth_connections = UserOAuthConnection::get_all_for_user(conn, user.uuid)?;
            for connection in oauth_connections {
                let provider = OAuthProvider::get_by_id(conn, connection.provider_id)?.ok_or(
                    SeedWrapTranslationError::MissingOAuthProvider {
                        connection_id: connection.id,
                        provider_id: connection.provider_id,
                    },
                )?;

                upsert_oauth_seed_wrap(
                    conn,
                    app_state,
                    &user,
                    &provider,
                    &connection,
                    &plaintext_seed,
                )?;
                user_wraps += 1;
                expected_oauth_wraps += 1;
            }

            if user_wraps == 0 {
                return Err(SeedWrapTranslationError::NoUsableCredential(user.uuid));
            }
            expected_wraps += user_wraps;
        }

        validate_seed_wrap_postflight_count(
            conn,
            CredentialKind::Password.as_str(),
            expected_password_wraps,
        )?;
        validate_seed_wrap_postflight_count(
            conn,
            CredentialKind::OAuth.as_str(),
            expected_oauth_wraps,
        )?;

        NewAppDataMigration::new(AEAD_SEED_WRAPPINGS_MIGRATION).insert(conn)?;
        info!(
            "AEAD seed wrapping translation completed with {} seed wraps",
            expected_wraps
        );
        Ok(())
    })
    .map_err(Error::from)
}

#[cfg(feature = "seed-wrap-translation")]
fn validate_no_duplicate_oauth_subjects_by_project(
    conn: &mut PgConnection,
) -> Result<(), SeedWrapTranslationError> {
    let duplicates = sql_query(
        r#"
        SELECT
            user_oauth_connections.provider_id AS provider_id,
            user_oauth_connections.provider_user_id AS provider_user_id,
            users.project_id AS project_id,
            COUNT(*)::BIGINT AS connection_count
        FROM user_oauth_connections
        INNER JOIN users ON users.uuid = user_oauth_connections.user_id
        GROUP BY
            user_oauth_connections.provider_id,
            user_oauth_connections.provider_user_id,
            users.project_id
        HAVING COUNT(*) > 1
        LIMIT 1
        "#,
    )
    .load::<DuplicateOAuthSubject>(conn)?;

    if let Some(duplicate) = duplicates.into_iter().next() {
        return Err(SeedWrapTranslationError::DuplicateOAuthSubject {
            provider_id: duplicate.provider_id,
            provider_user_id: duplicate.provider_user_id,
            project_id: duplicate.project_id,
            connection_count: duplicate.connection_count,
        });
    }

    Ok(())
}

#[cfg(feature = "seed-wrap-translation")]
fn validate_seed_wrap_postflight_count(
    conn: &mut PgConnection,
    credential_kind: &str,
    expected_count: usize,
) -> Result<(), SeedWrapTranslationError> {
    let actual_count = user_seed_wrappings::table
        .filter(user_seed_wrappings::credential_kind.eq(credential_kind))
        .count()
        .get_result::<i64>(conn)?;
    let expected_count = expected_count as i64;

    if actual_count != expected_count {
        return Err(SeedWrapTranslationError::PostflightWrapCountMismatch {
            credential_kind: credential_kind.to_string(),
            expected_count,
            actual_count,
        });
    }

    Ok(())
}

#[cfg(feature = "seed-wrap-translation")]
fn upsert_password_seed_wrap(
    conn: &mut PgConnection,
    app_state: &AppState,
    user: &User,
    decrypted_password_verifier: &str,
    plaintext_seed: &[u8],
) -> Result<(), SeedWrapTranslationError> {
    let (identifier_kind, normalized_identifier) = match user.get_email() {
        Some(email) => (
            PasswordLoginIdentifierKind::Email,
            normalize_email_login_identifier(email),
        ),
        None => (
            PasswordLoginIdentifierKind::GuestUuid,
            normalize_guest_login_identifier(user.uuid),
        ),
    };

    let auth_binding = compute_password_auth_binding(
        &app_state.enclave_key,
        user.project_id,
        user.uuid,
        identifier_kind,
        &normalized_identifier,
        decrypted_password_verifier,
    )?;
    let lookup_hash =
        password_credential_lookup_hash(&app_state.enclave_key, user.project_id, user.uuid)?;
    let seed_enc = encrypt_seed_v1(
        &app_state.enclave_key,
        plaintext_seed,
        user.uuid,
        user.project_id,
        CredentialKind::Password,
        &auth_binding,
    )?;

    NewUserSeedWrapping::new(
        user.uuid,
        CredentialKind::Password.as_str(),
        lookup_hash.as_bytes().to_vec(),
        SEED_WRAP_VERSION_V1,
        seed_enc,
    )
    .upsert_by_credential(conn)?;

    Ok(())
}

#[cfg(feature = "seed-wrap-translation")]
fn upsert_oauth_seed_wrap(
    conn: &mut PgConnection,
    app_state: &AppState,
    user: &User,
    provider: &OAuthProvider,
    connection: &UserOAuthConnection,
    plaintext_seed: &[u8],
) -> Result<(), SeedWrapTranslationError> {
    let auth_binding = compute_oauth_auth_binding(
        &app_state.enclave_key,
        user.project_id,
        user.uuid,
        &provider.name,
        &connection.provider_user_id,
    )?;
    let lookup_hash = oauth_credential_lookup_hash(
        &app_state.enclave_key,
        user.project_id,
        user.uuid,
        &provider.name,
        &connection.provider_user_id,
    )?;
    let seed_enc = encrypt_seed_v1(
        &app_state.enclave_key,
        plaintext_seed,
        user.uuid,
        user.project_id,
        CredentialKind::OAuth,
        &auth_binding,
    )?;

    NewUserSeedWrapping::new(
        user.uuid,
        CredentialKind::OAuth.as_str(),
        lookup_hash.as_bytes().to_vec(),
        SEED_WRAP_VERSION_V1,
        seed_enc,
    )
    .upsert_by_credential(conn)?;

    Ok(())
}

async fn migrate_maple_project_settings(
    app_state: &Arc<AppState>,
    github_client_secret: Option<String>,
    google_client_secret: Option<String>,
    github_client_id: Option<String>,
    google_client_id: Option<String>,
) -> Result<(), Error> {
    info!("Checking Maple project settings migration");

    // Get OpenSecret org
    let org = app_state
        .db
        .get_org_by_name("OpenSecret")?
        .expect("OpenSecret organization must exist");

    // Get Maple project
    let maple = app_state
        .db
        .get_org_project_by_name_and_org("Maple", org.id)?
        .expect("Maple project must exist");

    // Check if email settings already exist
    let needs_email_migration = app_state.db.get_project_email_settings(maple.id)?.is_none();
    let needs_oauth_migration = app_state.db.get_project_oauth_settings(maple.id)?.is_none();

    if needs_email_migration || needs_oauth_migration {
        info!("Starting Maple project settings migration");
        perform_maple_settings_migration(
            app_state,
            maple.id,
            github_client_secret,
            google_client_secret,
            github_client_id,
            google_client_id,
        )
        .await?;
    } else {
        debug!("Maple project settings already exist, skipping migration");
    }

    Ok(())
}

async fn perform_maple_settings_migration(
    app_state: &Arc<AppState>,
    maple_id: i32,
    github_client_secret: Option<String>,
    google_client_secret: Option<String>,
    github_client_id: Option<String>,
    google_client_id: Option<String>,
) -> Result<(), Error> {
    // Get base URLs based on app mode
    let (verification_base_url, oauth_base_url) = match app_state.app_mode {
        AppMode::Local => ("http://127.0.0.1:5173/verify", "http://127.0.0.1:5173"),
        AppMode::Dev => (
            "https://dev.secretgpt.ai/verify",
            "https://dev.secretgpt.ai",
        ),
        AppMode::Preview => (
            "https://opensecret.cloud/verify",
            "https://preview.opensecret.cloud",
        ),
        AppMode::Prod => ("https://trymaple.ai/verify", "https://trymaple.ai"),
        AppMode::Custom(_) => (
            "https://preview.opensecret.cloud/verify",
            "https://preview.opensecret.cloud",
        ),
    };

    // Create email settings with the correct from_email based on app_mode
    let send_from = match app_state.app_mode {
        AppMode::Local => "local@email.trymaple.ai",
        AppMode::Dev => "dev@email.trymaple.ai",
        AppMode::Preview => "preview@email.trymaple.ai",
        AppMode::Prod => "hello@email.trymaple.ai",
        AppMode::Custom(_) => "preview@email.trymaple.ai",
    }
    .to_string();

    // Migrate email settings if needed
    if app_state.db.get_project_email_settings(maple_id)?.is_none() {
        let email_settings = EmailSettings {
            provider: "resend".to_string(),
            send_from,
            email_verification_url: verification_base_url.to_string(),
        };

        // Update project settings
        app_state
            .db
            .update_project_email_settings(maple_id, email_settings)?;

        // Get the global Resend API key which used to be meant for just Maple
        if let Some(resend_api_key) = &app_state.resend_api_key {
            migrate_project_secret(app_state, maple_id, PROJECT_RESEND_API_KEY, resend_api_key)
                .await?;
            info!("Successfully migrated Resend API key to Maple project secrets");
        } else {
            error!("No Resend API key found during migration");
        }
    }

    // Migrate OAuth settings if needed
    if app_state.db.get_project_oauth_settings(maple_id)?.is_none() {
        // Create OAuth settings with both providers enabled if credentials exist
        let oauth_settings = OAuthSettings {
            google_oauth_enabled: google_client_id.is_some() && google_client_secret.is_some(),
            github_oauth_enabled: github_client_id.is_some() && github_client_secret.is_some(),
            apple_oauth_enabled: false, // Apple auth is new, so disabled by default in migrations
            google_oauth_settings: google_client_id.map(|client_id| OAuthProviderSettings {
                client_id,
                redirect_url: format!("{}/auth/google/callback", oauth_base_url),
            }),
            github_oauth_settings: github_client_id.map(|client_id| OAuthProviderSettings {
                client_id,
                redirect_url: format!("{}/auth/github/callback", oauth_base_url),
            }),
            apple_oauth_settings: None, // No Apple OAuth settings during migration
        };

        app_state
            .db
            .update_project_oauth_settings(maple_id, oauth_settings)?;

        // Migrate OAuth secrets
        if let Some(secret) = github_client_secret {
            migrate_project_secret(app_state, maple_id, PROJECT_GITHUB_OAUTH_SECRET, &secret)
                .await?;
            info!("Successfully migrated GitHub OAuth secret to Maple project secrets");
        }

        if let Some(secret) = google_client_secret {
            migrate_project_secret(app_state, maple_id, PROJECT_GOOGLE_OAUTH_SECRET, &secret)
                .await?;
            info!("Successfully migrated Google OAuth secret to Maple project secrets");
        }
    }

    info!("Successfully completed Maple project settings migration");
    Ok(())
}

async fn migrate_project_secret(
    app_state: &Arc<AppState>,
    project_id: i32,
    key_name: &str,
    secret_value: &str,
) -> Result<(), Error> {
    // Encrypt the secret with the enclave key
    let secret_key = SecretKey::from_slice(&app_state.enclave_key)
        .map_err(|e| Error::EncryptionError(e.to_string()))?;
    let encrypted_value =
        crate::encrypt::encrypt_with_key(&secret_key, secret_value.as_bytes()).await;

    // Create project secret
    let new_secret = NewOrgProjectSecret::new(project_id, key_name.to_string(), encrypted_value);
    app_state.db.create_org_project_secret(new_secret)?;

    Ok(())
}
