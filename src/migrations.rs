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
        compute_oauth_auth_binding, compute_password_auth_binding, decrypt_seed_v1,
        encrypt_seed_v1, normalize_email_login_identifier, normalize_guest_login_identifier,
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
    #[error(
        "Seed wrap postflight mismatch for user {user_uuid}, {credential_kind}: expected {expected_count}, actual {actual_count}"
    )]
    UserPostflightWrapCountMismatch {
        user_uuid: Uuid,
        credential_kind: String,
        expected_count: i64,
        actual_count: i64,
    },
    #[error("Duplicate seed wrap credential rows exist after translation")]
    DuplicateSeedWrapCredential,
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
            SeedWrapTranslationError::UserPostflightWrapCountMismatch {
                user_uuid,
                credential_kind,
                expected_count,
                actual_count,
            } => Error::EncryptionError(format!(
                "Seed wrap postflight mismatch for user {user_uuid}, {credential_kind}: expected {expected_count}, actual {actual_count}"
            )),
            SeedWrapTranslationError::DuplicateSeedWrapCredential => Error::EncryptionError(
                "Duplicate seed wrap credential rows exist after translation".to_string(),
            ),
        }
    }
}

#[cfg(feature = "seed-wrap-translation")]
#[derive(QueryableByName)]
#[diesel(check_for_backend(diesel::pg::Pg))]
struct SqlCount {
    #[diesel(sql_type = BigInt)]
    count: i64,
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
            let mut user_password_wraps = 0usize;
            let mut user_oauth_wraps = 0usize;

            if let Some(password_enc) = user.password_enc.as_ref() {
                let verifier_bytes = decrypt_with_key(&legacy_secret_key, password_enc)?;
                let verifier = String::from_utf8(verifier_bytes)?;
                upsert_password_seed_wrap(conn, app_state, &user, &verifier, &plaintext_seed)?;
                user_wraps += 1;
                user_password_wraps += 1;
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
                user_oauth_wraps += 1;
                expected_oauth_wraps += 1;
            }

            if user_wraps == 0 {
                return Err(SeedWrapTranslationError::NoUsableCredential(user.uuid));
            }
            validate_user_seed_wrap_postflight_count(
                conn,
                user.uuid,
                CredentialKind::Password.as_str(),
                user_password_wraps,
            )?;
            validate_user_seed_wrap_postflight_count(
                conn,
                user.uuid,
                CredentialKind::OAuth.as_str(),
                user_oauth_wraps,
            )?;
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
        validate_no_duplicate_seed_wrap_credentials(conn)?;

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
fn validate_user_seed_wrap_postflight_count(
    conn: &mut PgConnection,
    user_uuid: Uuid,
    credential_kind: &str,
    expected_count: usize,
) -> Result<(), SeedWrapTranslationError> {
    let actual_count = user_seed_wrappings::table
        .filter(user_seed_wrappings::user_id.eq(user_uuid))
        .filter(user_seed_wrappings::credential_kind.eq(credential_kind))
        .count()
        .get_result::<i64>(conn)?;
    let expected_count = expected_count as i64;

    if actual_count != expected_count {
        return Err(SeedWrapTranslationError::UserPostflightWrapCountMismatch {
            user_uuid,
            credential_kind: credential_kind.to_string(),
            expected_count,
            actual_count,
        });
    }

    Ok(())
}

#[cfg(feature = "seed-wrap-translation")]
fn validate_no_duplicate_seed_wrap_credentials(
    conn: &mut PgConnection,
) -> Result<(), SeedWrapTranslationError> {
    let duplicate_count = sql_query(
        r#"
        SELECT COUNT(*)::BIGINT AS count
        FROM (
            SELECT 1
            FROM user_seed_wrappings
            GROUP BY user_id, credential_kind, credential_lookup_hash, wrapping_version
            HAVING COUNT(*) > 1
            LIMIT 1
        ) duplicate_groups
        "#,
    )
    .get_result::<SqlCount>(conn)?
    .count;

    if duplicate_count > 0 {
        return Err(SeedWrapTranslationError::DuplicateSeedWrapCredential);
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
    let verified_seed = decrypt_seed_v1(
        &app_state.enclave_key,
        &seed_enc,
        user.uuid,
        user.project_id,
        CredentialKind::Password,
        &auth_binding,
    )?;
    if verified_seed != plaintext_seed {
        return Err(SeedWrapTranslationError::Encrypt(
            EncryptError::FailedToDecrypt,
        ));
    }

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
    let verified_seed = decrypt_seed_v1(
        &app_state.enclave_key,
        &seed_enc,
        user.uuid,
        user.project_id,
        CredentialKind::OAuth,
        &auth_binding,
    )?;
    if verified_seed != plaintext_seed {
        return Err(SeedWrapTranslationError::Encrypt(
            EncryptError::FailedToDecrypt,
        ));
    }

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

#[cfg(all(test, feature = "seed-wrap-translation"))]
mod tests {
    use super::*;
    use crate::{
        db::setup_db,
        encrypt::encrypt_with_key,
        models::{
            oauth::NewUserOAuthConnection,
            org_projects::OrgProject,
            schema::{app_data_migrations, org_projects},
            user_seed_wrappings::UserSeedWrapping,
            users::NewUser,
        },
        private_key::generate_twelve_word_seed,
        AppStateBuilder,
    };
    use diesel::{ExpressionMethods, QueryDsl, RunQueryDsl};
    use password_auth::generate_hash;
    use tokio::sync::RwLock;

    #[tokio::test]
    #[ignore = "requires AEAD_TRANSLATION_TEST_DATABASE_URL pointing at an empty disposable migrated local Postgres"]
    async fn db_seed_wrap_translation_is_atomic_idempotent_and_creates_expected_wraps() {
        let Some(database_url) = translation_test_database_url() else {
            eprintln!("skipping: AEAD_TRANSLATION_TEST_DATABASE_URL is not set");
            return;
        };

        let app_state = build_local_translation_app_state(database_url).await;
        assert_empty_translation_state(&app_state);

        let project = first_active_project(&app_state);
        let marker = Uuid::new_v4();

        let duplicate_subject = format!("duplicate-google-sub-{marker}");
        let duplicate_a = insert_legacy_oauth_user(
            &app_state,
            project.id,
            format!("translation-duplicate-a-{marker}@example.com"),
            "google",
            duplicate_subject.clone(),
        )
        .await;
        let duplicate_b = insert_legacy_oauth_user(
            &app_state,
            project.id,
            format!("translation-duplicate-b-{marker}@example.com"),
            "google",
            duplicate_subject,
        )
        .await;

        let duplicate_result = migrate_aead_seed_wrappings_v1(&app_state);
        assert!(
            matches!(duplicate_result, Err(Error::EncryptionError(message)) if message.contains("Duplicate OAuth provider subject")),
            "duplicate OAuth provider subjects in one project must fail before translation"
        );
        assert!(!migration_marker_exists(&app_state));
        assert_eq!(total_seed_wrap_count(&app_state), 0);
        delete_test_users(&app_state, &[&duplicate_a.user, &duplicate_b.user]);

        let valid_before_failure = insert_legacy_password_user(
            &app_state,
            project.id,
            Some(format!(
                "translation-valid-before-failure-{marker}@example.com"
            )),
            "valid-password-before-rollback",
        )
        .await;
        let invalid_no_credential = insert_legacy_no_credential_user(
            &app_state,
            project.id,
            format!("translation-no-credential-{marker}@example.com"),
        )
        .await;

        let rollback_result = migrate_aead_seed_wrappings_v1(&app_state);
        assert!(
            matches!(rollback_result, Err(Error::EncryptionError(message)) if message.contains("has no usable credential")),
            "translation must fail when a legacy user has no password or OAuth credential"
        );
        assert!(!migration_marker_exists(&app_state));
        assert_eq!(
            total_seed_wrap_count(&app_state),
            0,
            "seed wraps written before a later user failure must roll back with the transaction"
        );
        delete_test_users(
            &app_state,
            &[&valid_before_failure.user, &invalid_no_credential],
        );

        let password_user = insert_legacy_password_user(
            &app_state,
            project.id,
            Some(format!("translation-password-{marker}@example.com")),
            "password-user-password",
        )
        .await;
        let guest_user =
            insert_legacy_password_user(&app_state, project.id, None, "guest-user-password").await;
        let oauth_user = insert_legacy_oauth_user(
            &app_state,
            project.id,
            format!("translation-oauth-{marker}@example.com"),
            "google",
            format!("translation-google-sub-{marker}"),
        )
        .await;
        let multi_user = insert_legacy_password_user(
            &app_state,
            project.id,
            Some(format!("translation-multi-{marker}@example.com")),
            "multi-user-password",
        )
        .await;
        let multi_oauth_subject = format!("translation-multi-github-sub-{marker}");
        insert_oauth_connection(
            &app_state,
            &multi_user.user,
            "github",
            multi_oauth_subject.clone(),
        );

        migrate_aead_seed_wrappings_v1(&app_state).expect(
            "translation should complete for password, guest, OAuth, and multi-credential users",
        );

        assert!(migration_marker_exists(&app_state));
        assert_eq!(total_seed_wrap_count(&app_state), 5);
        assert_eq!(
            seed_wrap_count_for_user_and_kind(
                &app_state,
                password_user.user.uuid,
                CredentialKind::Password
            ),
            1
        );
        assert_eq!(
            seed_wrap_count_for_user_and_kind(
                &app_state,
                guest_user.user.uuid,
                CredentialKind::Password
            ),
            1
        );
        assert_eq!(
            seed_wrap_count_for_user_and_kind(
                &app_state,
                oauth_user.user.uuid,
                CredentialKind::OAuth
            ),
            1
        );
        assert_eq!(
            seed_wrap_count_for_user_and_kind(
                &app_state,
                multi_user.user.uuid,
                CredentialKind::Password
            ),
            1
        );
        assert_eq!(
            seed_wrap_count_for_user_and_kind(
                &app_state,
                multi_user.user.uuid,
                CredentialKind::OAuth
            ),
            1
        );

        assert_legacy_seed_bytes_unchanged(
            &app_state,
            password_user.user.uuid,
            &password_user.legacy_seed_enc,
        );
        assert_legacy_seed_bytes_unchanged(
            &app_state,
            guest_user.user.uuid,
            &guest_user.legacy_seed_enc,
        );
        assert_legacy_seed_bytes_unchanged(
            &app_state,
            oauth_user.user.uuid,
            &oauth_user.legacy_seed_enc,
        );
        assert_legacy_seed_bytes_unchanged(
            &app_state,
            multi_user.user.uuid,
            &multi_user.legacy_seed_enc,
        );

        assert_password_wrap_decrypts(&app_state, &password_user);
        assert_password_wrap_decrypts(&app_state, &guest_user);
        assert_oauth_wrap_decrypts(&app_state, &oauth_user);
        assert_oauth_wrap_decrypts(
            &app_state,
            &LegacyOAuthFixture {
                user: multi_user.user.clone(),
                seed_words: multi_user.seed_words.clone(),
                legacy_seed_enc: multi_user.legacy_seed_enc.clone(),
                provider_name: "github".to_string(),
                provider_user_id: multi_oauth_subject,
            },
        );

        let seed_wraps_before_second_run = seed_wrap_snapshots(&app_state);
        overwrite_legacy_seed(&app_state, password_user.user.uuid, b"tampered legacy seed").await;
        migrate_aead_seed_wrappings_v1(&app_state)
            .expect("completed marker should make the second translation call skip");
        assert_eq!(
            seed_wrap_snapshots(&app_state),
            seed_wraps_before_second_run,
            "completed marker must prevent rerunning translation from tampered legacy seed_enc"
        );

        delete_test_users(
            &app_state,
            &[
                &password_user.user,
                &guest_user.user,
                &oauth_user.user,
                &multi_user.user,
            ],
        );
        delete_translation_marker(&app_state);
    }

    #[derive(Debug)]
    struct LegacyPasswordFixture {
        user: User,
        seed_words: String,
        password_hash: String,
        legacy_seed_enc: Vec<u8>,
    }

    #[derive(Debug)]
    struct LegacyOAuthFixture {
        user: User,
        seed_words: String,
        legacy_seed_enc: Vec<u8>,
        provider_name: String,
        provider_user_id: String,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct SeedWrapSnapshot {
        user_id: Uuid,
        credential_kind: String,
        credential_lookup_hash: Vec<u8>,
        wrapping_version: i16,
        seed_enc: Vec<u8>,
    }

    fn translation_test_database_url() -> Option<String> {
        let database_url = std::env::var("AEAD_TRANSLATION_TEST_DATABASE_URL").ok()?;
        assert!(
            database_url.contains("localhost") || database_url.contains("127.0.0.1"),
            "AEAD_TRANSLATION_TEST_DATABASE_URL must point at local Postgres"
        );
        assert!(
            database_url.contains("aead") && database_url.contains("scratch"),
            "AEAD_TRANSLATION_TEST_DATABASE_URL must point at a disposable AEAD scratch database"
        );
        Some(database_url)
    }

    async fn build_local_translation_app_state(database_url: String) -> Arc<AppState> {
        let db = setup_db(database_url);
        Arc::new(
            AppStateBuilder::default()
                .app_mode(AppMode::Local)
                .db(db)
                .enclave_key([42u8; 32].to_vec())
                .aws_credential_manager(Arc::new(RwLock::new(None)))
                .openai_api_base("http://localhost:9".to_string())
                .jwt_secret([24u8; 32].to_vec())
                .build()
                .await
                .expect("local test app state should build"),
        )
    }

    fn assert_empty_translation_state(app_state: &Arc<AppState>) {
        let conn = &mut test_conn(app_state);
        let user_count = users::table
            .count()
            .get_result::<i64>(conn)
            .expect("user count should load");
        let seed_wrap_count = user_seed_wrappings::table
            .count()
            .get_result::<i64>(conn)
            .expect("seed wrap count should load");
        let marker = AppDataMigration::get(conn, AEAD_SEED_WRAPPINGS_MIGRATION)
            .expect("migration marker lookup should not fail");

        assert_eq!(
            user_count, 0,
            "translation migration test requires an empty user table"
        );
        assert_eq!(
            seed_wrap_count, 0,
            "translation migration test requires an empty seed-wrap table"
        );
        assert!(
            marker.is_none(),
            "translation migration test requires no completed marker"
        );
    }

    fn first_active_project(app_state: &Arc<AppState>) -> OrgProject {
        let conn = &mut test_conn(app_state);
        org_projects::table
            .filter(org_projects::status.eq("active"))
            .order(org_projects::id.asc())
            .first::<OrgProject>(conn)
            .expect("test database should contain at least one active project")
    }

    async fn insert_legacy_password_user(
        app_state: &Arc<AppState>,
        project_id: i32,
        email: Option<String>,
        password: &str,
    ) -> LegacyPasswordFixture {
        let seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
            .await
            .expect("test seed should generate")
            .to_string();
        let password_hash = generate_hash(password);
        let secret_key = SecretKey::from_slice(&app_state.enclave_key)
            .expect("test enclave key should be valid");
        let password_enc = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;
        let legacy_seed_enc = encrypt_with_key(&secret_key, seed_words.as_bytes()).await;
        let user = NewUser::new(
            email,
            Some(password_enc),
            project_id,
            legacy_seed_enc.clone(),
        )
        .insert(&mut test_conn(app_state))
        .expect("legacy password user should insert");

        LegacyPasswordFixture {
            user,
            seed_words,
            password_hash,
            legacy_seed_enc,
        }
    }

    async fn insert_legacy_oauth_user(
        app_state: &Arc<AppState>,
        project_id: i32,
        email: String,
        provider_name: &str,
        provider_user_id: String,
    ) -> LegacyOAuthFixture {
        let seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
            .await
            .expect("test seed should generate")
            .to_string();
        let secret_key = SecretKey::from_slice(&app_state.enclave_key)
            .expect("test enclave key should be valid");
        let legacy_seed_enc = encrypt_with_key(&secret_key, seed_words.as_bytes()).await;
        let user = NewUser::new(Some(email), None, project_id, legacy_seed_enc.clone())
            .insert(&mut test_conn(app_state))
            .expect("legacy OAuth user should insert");

        insert_oauth_connection(app_state, &user, provider_name, provider_user_id.clone());

        LegacyOAuthFixture {
            user,
            seed_words,
            legacy_seed_enc,
            provider_name: provider_name.to_string(),
            provider_user_id,
        }
    }

    async fn insert_legacy_no_credential_user(
        app_state: &Arc<AppState>,
        project_id: i32,
        email: String,
    ) -> User {
        let seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
            .await
            .expect("test seed should generate")
            .to_string();
        let secret_key = SecretKey::from_slice(&app_state.enclave_key)
            .expect("test enclave key should be valid");
        let legacy_seed_enc = encrypt_with_key(&secret_key, seed_words.as_bytes()).await;
        NewUser::new(Some(email), None, project_id, legacy_seed_enc)
            .insert(&mut test_conn(app_state))
            .expect("legacy no-credential user should insert")
    }

    fn insert_oauth_connection(
        app_state: &Arc<AppState>,
        user: &User,
        provider_name: &str,
        provider_user_id: String,
    ) {
        let provider = app_state
            .db
            .get_oauth_provider_by_name(provider_name)
            .expect("test OAuth provider lookup should succeed")
            .expect("test OAuth provider should exist after AppState build");

        NewUserOAuthConnection::new(
            user.uuid,
            provider.id,
            provider_user_id,
            Vec::new(),
            None,
            None,
        )
        .insert(&mut test_conn(app_state))
        .expect("legacy OAuth connection should insert");
    }

    fn assert_password_wrap_decrypts(app_state: &Arc<AppState>, fixture: &LegacyPasswordFixture) {
        let (identifier_kind, normalized_identifier) = match fixture.user.get_email() {
            Some(email) => (
                PasswordLoginIdentifierKind::Email,
                normalize_email_login_identifier(email),
            ),
            None => (
                PasswordLoginIdentifierKind::GuestUuid,
                normalize_guest_login_identifier(fixture.user.uuid),
            ),
        };
        let auth_binding = compute_password_auth_binding(
            &app_state.enclave_key,
            fixture.user.project_id,
            fixture.user.uuid,
            identifier_kind,
            &normalized_identifier,
            &fixture.password_hash,
        )
        .expect("password auth binding should compute");
        let wrap = only_seed_wrap_for_user_and_kind(
            app_state,
            fixture.user.uuid,
            CredentialKind::Password,
        );
        let decrypted_seed = decrypt_seed_v1(
            &app_state.enclave_key,
            &wrap.seed_enc,
            fixture.user.uuid,
            fixture.user.project_id,
            CredentialKind::Password,
            &auth_binding,
        )
        .expect("migrated password wrap should decrypt");

        assert_eq!(decrypted_seed, fixture.seed_words.as_bytes());
    }

    fn assert_oauth_wrap_decrypts(app_state: &Arc<AppState>, fixture: &LegacyOAuthFixture) {
        let auth_binding = compute_oauth_auth_binding(
            &app_state.enclave_key,
            fixture.user.project_id,
            fixture.user.uuid,
            &fixture.provider_name,
            &fixture.provider_user_id,
        )
        .expect("OAuth auth binding should compute");
        let wrap =
            only_seed_wrap_for_user_and_kind(app_state, fixture.user.uuid, CredentialKind::OAuth);
        let decrypted_seed = decrypt_seed_v1(
            &app_state.enclave_key,
            &wrap.seed_enc,
            fixture.user.uuid,
            fixture.user.project_id,
            CredentialKind::OAuth,
            &auth_binding,
        )
        .expect("migrated OAuth wrap should decrypt");

        assert_eq!(decrypted_seed, fixture.seed_words.as_bytes());
    }

    fn assert_legacy_seed_bytes_unchanged(
        app_state: &Arc<AppState>,
        user_uuid: Uuid,
        expected_seed_enc: &[u8],
    ) {
        let user = User::get_by_uuid(&mut test_conn(app_state), user_uuid)
            .expect("user lookup should not fail")
            .expect("user should still exist");
        assert_eq!(
            user.seed_encrypted(),
            Some(expected_seed_enc),
            "translation must leave legacy users.seed_enc unchanged"
        );
    }

    fn total_seed_wrap_count(app_state: &Arc<AppState>) -> i64 {
        user_seed_wrappings::table
            .count()
            .get_result::<i64>(&mut test_conn(app_state))
            .expect("seed wrap count should load")
    }

    fn seed_wrap_count_for_user_and_kind(
        app_state: &Arc<AppState>,
        user_uuid: Uuid,
        credential_kind: CredentialKind,
    ) -> i64 {
        user_seed_wrappings::table
            .filter(user_seed_wrappings::user_id.eq(user_uuid))
            .filter(user_seed_wrappings::credential_kind.eq(credential_kind.as_str()))
            .count()
            .get_result::<i64>(&mut test_conn(app_state))
            .expect("seed wrap count should load")
    }

    fn only_seed_wrap_for_user_and_kind(
        app_state: &Arc<AppState>,
        user_uuid: Uuid,
        credential_kind: CredentialKind,
    ) -> UserSeedWrapping {
        let wraps = user_seed_wrappings::table
            .filter(user_seed_wrappings::user_id.eq(user_uuid))
            .filter(user_seed_wrappings::credential_kind.eq(credential_kind.as_str()))
            .load::<UserSeedWrapping>(&mut test_conn(app_state))
            .expect("seed wraps should load");
        assert_eq!(wraps.len(), 1);
        wraps.into_iter().next().expect("just asserted one wrap")
    }

    fn seed_wrap_snapshots(app_state: &Arc<AppState>) -> Vec<SeedWrapSnapshot> {
        let mut snapshots = user_seed_wrappings::table
            .load::<UserSeedWrapping>(&mut test_conn(app_state))
            .expect("seed wraps should load")
            .into_iter()
            .map(|wrap| SeedWrapSnapshot {
                user_id: wrap.user_id,
                credential_kind: wrap.credential_kind,
                credential_lookup_hash: wrap.credential_lookup_hash,
                wrapping_version: wrap.wrapping_version,
                seed_enc: wrap.seed_enc,
            })
            .collect::<Vec<_>>();

        snapshots.sort_by(|left, right| {
            (
                left.user_id,
                &left.credential_kind,
                &left.credential_lookup_hash,
                left.wrapping_version,
            )
                .cmp(&(
                    right.user_id,
                    &right.credential_kind,
                    &right.credential_lookup_hash,
                    right.wrapping_version,
                ))
        });
        snapshots
    }

    fn migration_marker_exists(app_state: &Arc<AppState>) -> bool {
        AppDataMigration::exists(&mut test_conn(app_state), AEAD_SEED_WRAPPINGS_MIGRATION)
            .expect("migration marker lookup should not fail")
    }

    async fn overwrite_legacy_seed(
        app_state: &Arc<AppState>,
        user_uuid: Uuid,
        plaintext_seed: &[u8],
    ) {
        let secret_key = SecretKey::from_slice(&app_state.enclave_key)
            .expect("test enclave key should be valid");
        let seed_enc = encrypt_with_key(&secret_key, plaintext_seed).await;
        let updated_rows = diesel::update(users::table)
            .filter(users::uuid.eq(user_uuid))
            .set(users::seed_enc.eq(Some(seed_enc)))
            .execute(&mut test_conn(app_state))
            .expect("legacy seed tamper should update");
        assert_eq!(updated_rows, 1);
    }

    fn delete_test_users(app_state: &Arc<AppState>, users: &[&User]) {
        for user in users {
            app_state
                .db
                .delete_user(user)
                .expect("test user cleanup should delete");
        }
    }

    fn delete_translation_marker(app_state: &Arc<AppState>) {
        diesel::delete(
            app_data_migrations::table
                .filter(app_data_migrations::name.eq(AEAD_SEED_WRAPPINGS_MIGRATION)),
        )
        .execute(&mut test_conn(app_state))
        .expect("translation marker cleanup should delete");
    }

    fn test_conn(
        app_state: &Arc<AppState>,
    ) -> diesel::r2d2::PooledConnection<diesel::r2d2::ConnectionManager<PgConnection>> {
        app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available")
    }
}
