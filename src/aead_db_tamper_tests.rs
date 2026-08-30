use crate::{
    db::{setup_db, DBError},
    encrypt::{encrypt_key_deterministic, encrypt_with_key},
    generate_reset_hash,
    jwt::{
        issue_transport_v2_platform_tokens, issue_transport_v2_user_tokens,
        validate_transport_v2_platform_resumption, validate_transport_v2_user_resumption,
    },
    login_routes::RegisterCredentials,
    models::{
        account_deletion::NewAccountDeletionRequest,
        email_verification::NewEmailVerification,
        oauth::NewUserOAuthConnection,
        org_projects::OrgProject,
        password_reset::NewPasswordResetRequest,
        platform_users::NewPlatformUser,
        responses::{
            NewAssistantMessage, NewConversation, NewConversationProject, NewReasoningItem,
            NewResponse, NewToolCall, NewToolOutput, NewUserInstruction, NewUserMessage,
            ProjectInstructionUpdate, ResponseStatus, ResponsesError,
        },
        schema::{
            account_deletion_requests, assistant_messages, conversation_projects,
            conversation_summaries, conversations, org_projects, password_reset_requests,
            platform_users, reasoning_items, responses, tool_calls, tool_outputs,
            user_instructions, user_messages, user_oauth_connections, user_seed_wrappings,
        },
        user_api_keys::{NewUserApiKey, UserApiKey, UserApiKeyError},
        user_kv::{NewUserKV, UserKV},
        user_seed_wrappings::NewUserSeedWrapping,
        users::{NewUser, User},
    },
    private_key::generate_twelve_word_seed,
    seed_wrapping::{password_reset_code_mac, CredentialKind},
    transport_v2::stored_resources::{self, InstructionUpdateCiphertext, StoredResourceError},
    AppMode, AppState, AppStateBuilder, Error,
};
use chrono::Utc;
use diesel::{ExpressionMethods, QueryDsl, RunQueryDsl};
use password_auth::generate_hash;
use secp256k1::SecretKey;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

fn test_credential(label: &str) -> &'static str {
    Box::leak(format!("aead-test-credential-{label}").into_boxed_str())
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_platform_v2_resumption_revalidates_live_platform_user() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let platform_user = app_state
        .db
        .create_platform_user(NewPlatformUser::new(
            format!("transport-v2-platform-{marker}@example.com"),
            Some(vec![0x41; 32]),
        ))
        .expect("platform user should be created");
    let issued = issue_transport_v2_platform_tokens(&platform_user, &app_state)
        .expect("platform v2 credentials should issue");
    let resumed = validate_transport_v2_platform_resumption(&issued.resumption_token, &app_state)
        .expect("live platform principal should resume");
    assert_eq!(resumed.uuid, platform_user.uuid);

    let mut connection = app_state.db.get_pool().get().expect("database connection");
    diesel::delete(platform_users::table.filter(platform_users::id.eq(platform_user.id)))
        .execute(&mut connection)
        .expect("platform test user should be deleted");
    assert!(
        validate_transport_v2_platform_resumption(&issued.resumption_token, &app_state).is_err(),
        "deleted platform principals must not resume"
    );
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_registration_creates_initial_seed_wrap_and_login_works() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-registration-{marker}@example.com");
    let password = test_credential("registration-before-login");

    let user = app_state
        .register_user(RegisterCredentials {
            name: Some("AEAD Registration Test".to_string()),
            email: Some(email.clone()),
            password: password.to_string(),
            client_id: project.client_id,
        })
        .await
        .expect("registration should create user and initial seed wrap");

    let password_wraps = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(user.uuid, CredentialKind::Password.as_str())
        .expect("registered user's password seed wraps should load");
    assert_eq!(
        password_wraps.len(),
        1,
        "password registration should commit exactly one initial password seed wrap"
    );

    let login = app_state
        .authenticate_user(Some(email), None, password.to_string(), project.id)
        .await
        .expect("registered user login should not error")
        .expect("registered password should verify and unwrap");
    app_state
        .get_user_key(&login.user, &login.auth_context, None, None)
        .await
        .expect("registered user's auth context should derive a user key");

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_seed_wrap_substitution_fails_before_issuing_password_session() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-tamper-victim-{marker}@example.com");
    let attacker_email = format!("aead-tamper-attacker-{marker}@example.com");
    let victim_password = test_credential("victim-before-tamper");
    let attacker_password = test_credential("attacker-before-tamper");

    let victim = create_password_wrapped_user(
        &app_state,
        project.id,
        victim_email.clone(),
        victim_password,
    )
    .await;
    let attacker = create_password_wrapped_user(
        &app_state,
        project.id,
        attacker_email.clone(),
        attacker_password,
    )
    .await;

    let attacker_login_before_tamper = app_state
        .authenticate_user(
            Some(attacker_email.clone()),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await
        .expect("untampered attacker login should not error");
    assert!(
        attacker_login_before_tamper.is_some(),
        "untampered attacker login should verify and unwrap"
    );

    copy_victim_seed_wrap_ciphertext_to_attacker(&app_state, &victim, &attacker);

    let attacker_login_after_tamper = app_state
        .authenticate_user(
            Some(attacker_email),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await;

    assert!(
        matches!(attacker_login_after_tamper, Err(Error::AuthenticationError)),
        "tampered attacker row must fail before issuing a password session"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_verifier_substitution_fails_before_issuing_victim_session() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-pw-tamper-victim-{marker}@example.com");
    let attacker_email = format!("aead-pw-tamper-attacker-{marker}@example.com");
    let victim_password = test_credential("victim-before-verifier-row-tamper");
    let attacker_password = test_credential("attacker-before-verifier-row-tamper");

    let victim = create_password_wrapped_user(
        &app_state,
        project.id,
        victim_email.clone(),
        victim_password,
    )
    .await;
    let attacker =
        create_password_wrapped_user(&app_state, project.id, attacker_email, attacker_password)
            .await;

    let victim_login_before_tamper = app_state
        .authenticate_user(
            Some(victim_email.clone()),
            None,
            victim_password.to_string(),
            project.id,
        )
        .await
        .expect("untampered victim login should not error");
    assert!(
        victim_login_before_tamper.is_some(),
        "untampered victim login should verify and unwrap"
    );

    copy_attacker_password_verifier_to_victim(&app_state, &attacker, &victim);

    let victim_shell_login_after_tamper = app_state
        .authenticate_user(
            Some(victim_email),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await;

    assert!(
        matches!(
            victim_shell_login_after_tamper,
            Err(Error::AuthenticationError)
        ),
        "copied attacker password verifier must not produce a victim session"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_victim_password_verifier_copy_to_attacker_does_not_issue_attacker_session() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-pw-copy-victim-{marker}@example.com");
    let attacker_email = format!("aead-pw-copy-attacker-{marker}@example.com");
    let victim_password = test_credential("victim-before-copy-to-attacker");
    let attacker_password = test_credential("attacker-before-copy-to-attacker");

    let victim =
        create_password_wrapped_user(&app_state, project.id, victim_email, victim_password).await;
    let attacker = create_password_wrapped_user(
        &app_state,
        project.id,
        attacker_email.clone(),
        attacker_password,
    )
    .await;

    copy_victim_password_verifier_to_attacker(&app_state, &victim, &attacker);

    let attacker_password_login_after_tamper = app_state
        .authenticate_user(
            Some(attacker_email.clone()),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await
        .expect("attacker login with attacker password should not error");
    assert!(
        attacker_password_login_after_tamper.is_none(),
        "attacker password must not verify after victim password verifier is copied into attacker row"
    );

    let victim_password_login_in_attacker_row = app_state
        .authenticate_user(
            Some(attacker_email),
            None,
            victim_password.to_string(),
            project.id,
        )
        .await;
    assert!(
        matches!(
            victim_password_login_in_attacker_row,
            Err(Error::AuthenticationError)
        ),
        "even the password that matches the copied verifier must not unwrap the attacker's seed context"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_copied_kv_rows_do_not_decrypt_under_attacker_auth_context() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-kv-copy-victim-{marker}@example.com");
    let attacker_email = format!("aead-kv-copy-attacker-{marker}@example.com");
    let victim_password = test_credential("victim-before-kv-copy");
    let attacker_password = test_credential("attacker-before-kv-copy");

    let victim = create_password_wrapped_user(
        &app_state,
        project.id,
        victim_email.clone(),
        victim_password,
    )
    .await;
    let attacker = create_password_wrapped_user(
        &app_state,
        project.id,
        attacker_email.clone(),
        attacker_password,
    )
    .await;

    let victim_login = app_state
        .authenticate_user(
            Some(victim_email),
            None,
            victim_password.to_string(),
            project.id,
        )
        .await
        .expect("victim login should not error")
        .expect("victim password should verify and unwrap");
    let attacker_login = app_state
        .authenticate_user(
            Some(attacker_email),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await
        .expect("attacker login should not error")
        .expect("attacker password should verify and unwrap");

    app_state
        .put(
            &victim_login.user,
            &victim_login.auth_context,
            "copied-kv-secret",
            "victim plaintext must not leak",
        )
        .await
        .expect("victim KV insert should succeed");

    copy_victim_kv_rows_to_attacker(&app_state, &victim, &attacker);

    let attacker_list = app_state
        .list(&attacker_login.user, &attacker_login.auth_context)
        .await;

    assert!(
        matches!(attacker_list, Err(crate::kv::StoreError::DecryptionError)),
        "copied victim KV ciphertext must fail under the attacker's authenticated user key"
    );

    let attacker_get = app_state
        .get(
            &attacker_login.user,
            &attacker_login.auth_context,
            "copied-kv-secret",
        )
        .await
        .expect("attacker get should not error for a missing attacker-encrypted key");
    assert!(
        attacker_get.is_none(),
        "attacker lookup with the plaintext key must not match the copied victim encrypted key"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_bounded_kv_reads_preserve_wire_data_and_enforce_limits() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner_email = format!("aead-bounded-kv-owner-{marker}@example.com");
    let other_email = format!("aead-bounded-kv-other-{marker}@example.com");
    let owner_password = test_credential("bounded-kv-owner");
    let other_password = test_credential("bounded-kv-other");

    let owner =
        create_password_wrapped_user(&app_state, project.id, owner_email.clone(), owner_password)
            .await;
    let other =
        create_password_wrapped_user(&app_state, project.id, other_email.clone(), other_password)
            .await;

    let owner_login = app_state
        .authenticate_user(
            Some(owner_email),
            None,
            owner_password.to_string(),
            project.id,
        )
        .await
        .expect("owner login should not error")
        .expect("owner password should verify and unwrap");
    let other_login = app_state
        .authenticate_user(
            Some(other_email),
            None,
            other_password.to_string(),
            project.id,
        )
        .await
        .expect("other-user login should not error")
        .expect("other-user password should verify and unwrap");

    let entries = [
        ("bounded-kv-alpha", "first value"),
        ("bounded-kv-beta", "quoted \"value\" and snowman ☃"),
    ];
    for (key, value) in entries {
        app_state
            .put(&owner_login.user, &owner_login.auth_context, key, value)
            .await
            .expect("owner KV insert should succeed");
    }

    let present = app_state
        .get_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            entries[1].0,
            entries[1].1.len(),
        )
        .await
        .expect("bounded GET at the exact plaintext limit should succeed")
        .expect("stored owner value should be present");
    assert_eq!(present.as_str(), entries[1].1);

    let missing = app_state
        .get_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            "bounded-kv-missing",
            0,
        )
        .await
        .expect("bounded GET for a missing key should succeed");
    assert!(
        missing.is_none(),
        "missing bounded GET should preserve null semantics"
    );

    let undersized_get = app_state
        .get_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            entries[1].0,
            entries[1].1.len() - 1,
        )
        .await;
    assert!(matches!(
        undersized_get,
        Err(crate::kv::StoreError::OutputTooLarge)
    ));

    let aggregate_plaintext_len = entries
        .iter()
        .map(|(key, value)| key.len() + value.len())
        .sum();
    let mut bounded_list = app_state
        .list_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            aggregate_plaintext_len,
            entries.len(),
        )
        .await
        .expect("bounded LIST at the exact aggregate limit should succeed");
    let mut legacy_list = app_state
        .list(&owner_login.user, &owner_login.auth_context)
        .await
        .expect("legacy LIST comparison should succeed");

    bounded_list.sort_by(|left, right| left.key.cmp(&right.key));
    legacy_list.sort_by(|left, right| left.key.cmp(&right.key));
    assert_eq!(bounded_list.len(), entries.len());
    assert_eq!(bounded_list.len(), legacy_list.len());
    for (actual, expected) in bounded_list.iter().zip(&legacy_list) {
        assert_eq!(actual.key, expected.key);
        assert_eq!(actual.value, expected.value);
        assert_eq!(actual.created_at, expected.created_at);
        assert_eq!(actual.updated_at, expected.updated_at);
        assert!(actual.created_at > 0);
        assert!(actual.updated_at >= actual.created_at);
        assert_eq!(
            serde_json::to_value(actual).expect("bounded KV pair should serialize"),
            serde_json::json!({
                "key": expected.key,
                "value": expected.value,
                "created_at": expected.created_at,
                "updated_at": expected.updated_at,
            })
        );
    }

    let undersized_list = app_state
        .list_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            aggregate_plaintext_len - 1,
            entries.len(),
        )
        .await;
    assert!(matches!(
        undersized_list,
        Err(crate::kv::StoreError::OutputTooLarge)
    ));

    let row_limited_list = app_state
        .list_bounded_kv(
            &owner_login.user,
            &owner_login.auth_context,
            aggregate_plaintext_len,
            entries.len() - 1,
        )
        .await;
    assert!(matches!(
        row_limited_list,
        Err(crate::kv::StoreError::OutputTooLarge)
    ));

    let other_get = app_state
        .get_bounded_kv(
            &other_login.user,
            &other_login.auth_context,
            entries[0].0,
            entries[0].1.len(),
        )
        .await
        .expect("cross-user bounded GET should not error");
    assert!(
        other_get.is_none(),
        "bounded GET must remain scoped to its user"
    );
    let other_list = app_state
        .list_bounded_kv(&other_login.user, &other_login.auth_context, 0, 0)
        .await
        .expect("empty cross-user bounded LIST should succeed");
    assert!(
        other_list.is_empty(),
        "bounded LIST must remain scoped to its user"
    );

    let _ = app_state.db.delete_user(&owner);
    let _ = app_state.db.delete_user(&other);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_bounded_api_key_list_preserves_metadata_and_enforces_limits() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner_email = format!("aead-bounded-api-keys-owner-{marker}@example.com");
    let other_email = format!("aead-bounded-api-keys-other-{marker}@example.com");
    let owner = create_password_wrapped_user(
        &app_state,
        project.id,
        owner_email,
        test_credential("bounded-api-keys-owner"),
    )
    .await;
    let other = create_password_wrapped_user(
        &app_state,
        project.id,
        other_email,
        test_credential("bounded-api-keys-other"),
    )
    .await;

    let names = ["bounded API key-1", "bounded API key_2"];
    for (index, name) in names.iter().enumerate() {
        app_state
            .db
            .create_user_api_key(NewUserApiKey::new(
                owner.uuid,
                format!("{:064x}", marker.as_u128().wrapping_add(index as u128)),
                (*name).to_owned(),
            ))
            .expect("API-key metadata insert should succeed");
    }

    let aggregate_name_bytes = names.iter().map(|name| name.len()).sum();
    let mut conn = app_state
        .db
        .get_pool()
        .get()
        .expect("bounded API-key test connection should open");
    let mut bounded = UserApiKey::get_bounded_list_for_user(
        &mut conn,
        owner.uuid,
        aggregate_name_bytes,
        names.len(),
    )
    .expect("bounded API-key list should succeed at its exact limits");
    let mut legacy = app_state
        .db
        .get_all_user_api_keys_for_user(owner.uuid)
        .expect("legacy API-key list comparison should succeed");
    bounded.sort_by(|left, right| left.name.cmp(&right.name));
    legacy.sort_by(|left, right| left.name.cmp(&right.name));
    assert_eq!(bounded.len(), names.len());
    for (actual, expected) in bounded.iter().zip(&legacy) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.created_at, expected.created_at);
    }

    assert!(matches!(
        UserApiKey::get_bounded_list_for_user(
            &mut conn,
            owner.uuid,
            aggregate_name_bytes - 1,
            names.len(),
        ),
        Err(UserApiKeyError::OutputTooLarge)
    ));
    assert!(matches!(
        UserApiKey::get_bounded_list_for_user(
            &mut conn,
            owner.uuid,
            aggregate_name_bytes,
            names.len() - 1,
        ),
        Err(UserApiKeyError::OutputTooLarge)
    ));
    let other_rows = UserApiKey::get_bounded_list_for_user(&mut conn, other.uuid, 0, 0)
        .expect("empty other-user API-key list should succeed");
    assert!(
        other_rows.is_empty(),
        "API-key list must remain user-scoped"
    );
    drop(conn);

    let response = crate::web::protected_routes::list_bounded_api_keys_data(
        &app_state,
        &owner,
        aggregate_name_bytes,
        names.len(),
    )
    .expect("bounded API-key response data should succeed");
    assert_eq!(response.keys.len(), names.len());
    for key in &response.keys {
        assert!(names.contains(&key.name.as_str()));
        assert!(key.created_at.timestamp() > 0);
    }

    let _ = app_state.db.delete_user(&owner);
    let _ = app_state.db.delete_user(&other);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_email_verification_preserves_success_idempotency_and_expiry_contracts() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let user = create_password_wrapped_user(
        &app_state,
        project.id,
        format!("aead-email-verification-{marker}@example.com"),
        test_credential("email-verification"),
    )
    .await;

    let verification = app_state
        .db
        .create_email_verification(NewEmailVerification::new(user.uuid, 24, false))
        .expect("unverified email code should insert");
    let first =
        crate::web::login_routes::verify_email_data(&app_state, verification.verification_code)
            .expect("fresh email code should verify");
    assert_eq!(
        first,
        serde_json::json!({ "message": "Email verified successfully" })
    );
    assert!(
        app_state
            .db
            .get_email_verification_by_code(verification.verification_code)
            .expect("verified email row should remain readable")
            .is_verified
    );

    let repeated =
        crate::web::login_routes::verify_email_data(&app_state, verification.verification_code)
            .expect("verified code should be application-idempotent");
    assert_eq!(
        repeated,
        serde_json::json!({ "message": "Email already verified" })
    );

    let expired = app_state
        .db
        .create_email_verification(NewEmailVerification::new(user.uuid, -1, false))
        .expect("expired email code should insert for the contract test");
    assert!(matches!(
        crate::web::login_routes::verify_email_data(&app_state, expired.verification_code),
        Err(crate::ApiError::BadRequest)
    ));
    assert!(
        !app_state
            .db
            .get_email_verification_by_code(expired.verification_code)
            .expect("expired email row should remain readable")
            .is_verified
    );

    let expired_verified = app_state
        .db
        .create_email_verification(NewEmailVerification::new(user.uuid, -1, true))
        .expect("expired verified email code should insert for the contract test");
    assert!(matches!(
        crate::web::login_routes::verify_email_data(&app_state, expired_verified.verification_code,),
        Err(crate::ApiError::BadRequest)
    ));
    assert!(
        app_state
            .db
            .get_email_verification_by_code(expired_verified.verification_code)
            .expect("expired verified email row should remain readable")
            .is_verified
    );
    assert!(matches!(
        crate::web::login_routes::verify_email_data(&app_state, Uuid::new_v4()),
        Err(crate::ApiError::BadRequest)
    ));

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_account_deletion_request_preserves_generic_response_and_distinct_attempts() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id))
        .expect("guest deletion-request test user should insert");

    for hashed_secret in ["first-client-hash", "second-client-hash"] {
        let value = crate::web::protected_routes::initiate_account_deletion_data(
            &app_state,
            &user,
            crate::web::protected_routes::InitiateAccountDeletionRequest {
                hashed_secret: hashed_secret.to_owned(),
            },
        )
        .await
        .expect("account deletion request should preserve its generic success response");
        assert_eq!(
            value,
            serde_json::json!({
                "message": "We have sent a confirmation code to your email."
            })
        );
    }

    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let stored = account_deletion_requests::table
        .filter(account_deletion_requests::user_id.eq(user.uuid))
        .order(account_deletion_requests::id.asc())
        .select((
            account_deletion_requests::project_id,
            account_deletion_requests::hashed_secret,
            account_deletion_requests::encrypted_code,
            account_deletion_requests::expiration_time,
            account_deletion_requests::is_deleted,
        ))
        .load::<(i32, String, Vec<u8>, chrono::DateTime<Utc>, bool)>(conn)
        .expect("account deletion requests should remain readable");
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[0].0, project.id);
    assert_eq!(stored[0].1, "first-client-hash");
    assert!(!stored[0].4);
    assert_eq!(stored[1].0, project.id);
    assert_eq!(stored[1].1, "second-client-hash");
    assert!(!stored[1].4);
    assert_ne!(stored[0].2, stored[1].2);
    let now = Utc::now();
    for (_, _, _, expiration_time, _) in &stored {
        let remaining = expiration_time.signed_duration_since(now);
        assert!(remaining > chrono::Duration::hours(23));
        assert!(remaining <= chrono::Duration::hours(24));
    }

    diesel::delete(
        account_deletion_requests::table.filter(account_deletion_requests::user_id.eq(user.uuid)),
    )
    .execute(conn)
    .expect("test account deletion requests should delete");
    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_account_deletion_confirmation_preserves_failures_and_completed_audit_row() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id))
        .expect("guest deletion-confirmation test user should insert");
    let confirmation_code = Uuid::new_v4();
    let sibling_code = Uuid::new_v4();
    let plaintext_secret = format!("deletion-secret-{}", Uuid::new_v4());
    let sibling_secret = format!("sibling-deletion-secret-{}", Uuid::new_v4());
    let invalid_secret = format!("invalid-deletion-secret-{}", Uuid::new_v4());
    let enclave_key = SecretKey::from_slice(&app_state.enclave_key)
        .expect("test enclave key must be a valid secp256k1 secret");
    let selected = app_state
        .db
        .create_account_deletion_request(NewAccountDeletionRequest::new(
            user.uuid,
            project.id,
            generate_reset_hash(plaintext_secret.clone()),
            encrypt_key_deterministic(&enclave_key, confirmation_code.as_bytes()),
            24,
        ))
        .expect("selected deletion request should insert");
    let sibling = app_state
        .db
        .create_account_deletion_request(NewAccountDeletionRequest::new(
            user.uuid,
            project.id,
            generate_reset_hash(sibling_secret),
            encrypt_key_deterministic(&enclave_key, sibling_code.as_bytes()),
            24,
        ))
        .expect("sibling deletion request should insert");

    let invalid_code = crate::web::protected_routes::confirm_account_deletion_data(
        &app_state,
        &user,
        crate::web::protected_routes::ConfirmAccountDeletionRequest {
            confirmation_code: "not-a-uuid".to_owned(),
            plaintext_secret: plaintext_secret.clone(),
        },
    )
    .await;
    assert!(matches!(invalid_code, Err(crate::ApiError::BadRequest)));

    let invalid_secret = crate::web::protected_routes::confirm_account_deletion_data(
        &app_state,
        &user,
        crate::web::protected_routes::ConfirmAccountDeletionRequest {
            confirmation_code: confirmation_code.to_string(),
            plaintext_secret: invalid_secret,
        },
    )
    .await;
    assert!(matches!(invalid_secret, Err(crate::ApiError::BadRequest)));
    app_state
        .db
        .get_user_by_uuid(user.uuid)
        .expect("failed confirmation must preserve the user");

    crate::web::protected_routes::confirm_account_deletion_data(
        &app_state,
        &user,
        crate::web::protected_routes::ConfirmAccountDeletionRequest {
            confirmation_code: confirmation_code.to_string(),
            plaintext_secret,
        },
    )
    .await
    .expect("valid account deletion confirmation should succeed");
    assert!(matches!(
        app_state.db.get_user_by_uuid(user.uuid),
        Err(DBError::UserNotFound)
    ));

    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let rows = account_deletion_requests::table
        .filter(account_deletion_requests::id.eq_any([selected.id, sibling.id]))
        .order(account_deletion_requests::id.asc())
        .select((
            account_deletion_requests::id,
            account_deletion_requests::is_deleted,
            account_deletion_requests::completed_at,
        ))
        .load::<(i32, bool, Option<chrono::DateTime<Utc>>)>(conn)
        .expect("orphan-preserving deletion audit rows should remain readable");
    assert_eq!(rows.len(), 2);
    let selected_row = rows
        .iter()
        .find(|(id, _, _)| *id == selected.id)
        .expect("selected deletion audit row should remain");
    assert!(selected_row.1);
    assert!(selected_row.2.is_some());
    let sibling_row = rows
        .iter()
        .find(|(id, _, _)| *id == sibling.id)
        .expect("sibling deletion audit row should remain");
    assert!(!sibling_row.1);
    assert!(sibling_row.2.is_none());

    diesel::delete(
        account_deletion_requests::table
            .filter(account_deletion_requests::id.eq_any([selected.id, sibling.id])),
    )
    .execute(conn)
    .expect("test account deletion audit rows should delete");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_change_invalidates_old_auth_context_and_preserves_seed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-password-change-{marker}@example.com");
    let old_password = test_credential("old-before-change");
    let new_password = test_credential("new-after-change");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;

    let old_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login should not error")
        .expect("old password login should verify and unwrap");
    let old_key = app_state
        .get_user_key(&old_login.user, &old_login.auth_context, None, None)
        .await
        .expect("old key should derive before password change");
    app_state
        .verify_bound_user(old_login.user.uuid, project.id, &old_login.auth_context)
        .expect("transport-v2 bound authority should be live before password change");

    let new_auth_context = app_state
        .update_user_password_and_seed_wrap(
            &old_login.user,
            &old_login.auth_context,
            new_password.to_string(),
        )
        .await
        .expect("password change should rewrap seed");

    let old_context_after_change =
        app_state.verify_seed_wrap_for_auth_context(&old_login.user, &old_login.auth_context);
    assert!(
        matches!(old_context_after_change, Err(Error::AuthenticationError)),
        "old password auth context must not unwrap after password change"
    );
    assert!(matches!(
        app_state.verify_bound_user(old_login.user.uuid, project.id, &old_login.auth_context,),
        Err(crate::ApiError::Unauthorized)
    ));

    let old_password_login_after_change = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login after change should not error");
    assert!(
        old_password_login_after_change.is_none(),
        "old password must not authenticate after password change"
    );

    let new_password_login = app_state
        .authenticate_user(Some(email), None, new_password.to_string(), project.id)
        .await
        .expect("new password login should not error")
        .expect("new password should verify and unwrap");
    let new_key = app_state
        .get_user_key(
            &new_password_login.user,
            &new_password_login.auth_context,
            None,
            None,
        )
        .await
        .expect("new key should derive after password change");

    app_state
        .verify_seed_wrap_for_auth_context(&new_password_login.user, &new_auth_context)
        .expect("new auth context returned by password change should unwrap");
    assert_eq!(
        old_key.secret_bytes(),
        new_key.secret_bytes(),
        "normal password change must preserve the existing user seed"
    );

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_change_activates_only_new_v2_resumption_after_commit() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-v2-password-change-{marker}@example.com");
    let old_password = test_credential("old-before-v2-change");
    let new_password = test_credential("new-after-v2-change");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;
    let old_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login should not error")
        .expect("old password login should verify and unwrap");
    let old_tokens =
        issue_transport_v2_user_tokens(&old_login.user, &old_login.auth_context, &app_state)
            .expect("old v2 credentials should issue");

    let prepared = app_state
        .prepare_user_password_and_seed_wrap(
            &old_login.user,
            &old_login.auth_context,
            new_password.to_string(),
        )
        .await
        .expect("replacement password state should prepare");
    let replacement_tokens =
        issue_transport_v2_user_tokens(&old_login.user, prepared.new_auth_context(), &app_state)
            .expect("replacement v2 credentials should issue before commit");
    assert!(
        validate_transport_v2_user_resumption(&replacement_tokens.resumption_token, &app_state)
            .is_err(),
        "a preissued replacement resumption credential must remain dormant before commit"
    );

    let new_auth_context = app_state
        .commit_prepared_user_password_and_seed_wrap(&old_login.user, prepared)
        .expect("replacement password state should commit");
    app_state
        .verify_seed_wrap_for_auth_context(&old_login.user, &new_auth_context)
        .expect("committed replacement auth context should unwrap");

    assert!(
        validate_transport_v2_user_resumption(&old_tokens.resumption_token, &app_state).is_err(),
        "the prior resumption credential must stop validating after commit"
    );
    let resumed =
        validate_transport_v2_user_resumption(&replacement_tokens.resumption_token, &app_state)
            .expect("the replacement resumption credential should validate after commit");
    assert_eq!(resumed.user.uuid, old_login.user.uuid);
    assert_eq!(resumed.auth_context, new_auth_context);

    let new_login = app_state
        .authenticate_user(Some(email), None, new_password.to_string(), project.id)
        .await
        .expect("new password login should not error")
        .expect("new password should authenticate after commit");
    assert_eq!(new_login.auth_context, new_auth_context);

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_change_deletes_tampered_stale_password_wraps() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-password-change-stale-wrap-{marker}@example.com");
    let old_password = test_credential("old-before-stale-wrap-change");
    let new_password = test_credential("new-after-stale-wrap-change");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;

    let old_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login should not error")
        .expect("old password login should verify and unwrap");

    tamper_password_wrap_lookup_hash(&app_state, &user, marker.as_bytes().to_vec());

    app_state
        .update_user_password_and_seed_wrap(
            &old_login.user,
            &old_login.auth_context,
            new_password.to_string(),
        )
        .await
        .expect("password change should delete stale wraps and rewrap seed");

    let old_context_after_change =
        app_state.verify_seed_wrap_for_auth_context(&old_login.user, &old_login.auth_context);
    assert!(
        matches!(old_context_after_change, Err(Error::AuthenticationError)),
        "old auth context must not unwrap after lookup-hash-tampered password wrap replacement"
    );

    let password_wraps = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(user.uuid, CredentialKind::Password.as_str())
        .expect("post-change password wraps should load");
    assert_eq!(
        password_wraps.len(),
        1,
        "password change must delete every existing password wrap before inserting the replacement"
    );

    let new_password_login = app_state
        .authenticate_user(Some(email), None, new_password.to_string(), project.id)
        .await
        .expect("new password login should not error")
        .expect("new password should verify and unwrap");
    app_state
        .verify_seed_wrap_for_auth_context(
            &new_password_login.user,
            &new_password_login.auth_context,
        )
        .expect("new auth context should unwrap the replacement wrap");

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_conversation_project_uuid_delete_is_owner_scoped_and_cascades() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let org_project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner =
        create_response_transaction_test_user(&app_state, org_project.id, marker, "project-owner");
    let attacker = create_response_transaction_test_user(
        &app_state,
        org_project.id,
        marker,
        "project-attacker",
    );
    let project_uuid = Uuid::new_v4();
    let project = app_state
        .db
        .create_conversation_project(NewConversationProject {
            uuid: project_uuid,
            user_id: owner.uuid,
            name_enc: vec![0x5a; 1024 * 1024],
        })
        .expect("owner conversation project should insert");
    let conversation_uuid = Uuid::new_v4();
    app_state
        .db
        .create_conversation(NewConversation {
            uuid: conversation_uuid,
            user_id: owner.uuid,
            project_id: Some(project.id),
            is_pinned: false,
            metadata_enc: None,
        })
        .expect("assigned conversation should insert");

    assert!(matches!(
        app_state
            .db
            .delete_conversation_project_by_uuid_and_user(project_uuid, attacker.uuid),
        Err(DBError::ResponsesError(
            ResponsesError::ConversationProjectNotFound
        ))
    ));
    app_state
        .db
        .get_conversation_project_by_uuid_and_user(project_uuid, owner.uuid)
        .expect("foreign deletion must preserve the owner project");

    let deleted = app_state
        .db
        .delete_conversation_project_by_uuid_and_user(project_uuid, owner.uuid)
        .expect("owner-scoped UUID deletion should succeed");
    assert_eq!(deleted, project_uuid);
    assert!(matches!(
        app_state
            .db
            .get_conversation_project_by_uuid_and_user(project_uuid, owner.uuid),
        Err(DBError::ResponsesError(
            ResponsesError::ConversationProjectNotFound
        ))
    ));
    assert!(matches!(
        app_state
            .db
            .get_conversation_by_uuid_and_user(conversation_uuid, owner.uuid),
        Err(DBError::ResponsesError(
            ResponsesError::ConversationNotFound
        ))
    ));

    let _ = app_state.db.delete_user(&owner);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_v2_project_and_instruction_reads_are_bounded_scoped_and_sentinel_safe() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let org_project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner =
        create_response_transaction_test_user(&app_state, org_project.id, marker, "bounded-owner");
    let other =
        create_response_transaction_test_user(&app_state, org_project.id, marker, "bounded-other");

    let safe_project = app_state
        .db
        .create_conversation_project(NewConversationProject {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            name_enc: vec![0x11; 32],
        })
        .expect("safe project should insert");
    app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: Some(safe_project.id),
            name_enc: None,
            prompt_enc: vec![0x22; 33],
            prompt_tokens: 1,
            is_default: false,
        })
        .expect("safe project instruction should insert");
    app_state
        .db
        .create_conversation_project(NewConversationProject {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            name_enc: vec![0x33; 1024 * 1024],
        })
        .expect("oversized sentinel project should insert");

    let project =
        stored_resources::get_project(app_state.db.get_pool(), owner.uuid, safe_project.uuid, 9)
            .expect("project should fit the exact aggregate plaintext limit");
    assert_eq!(project.project.uuid, safe_project.uuid);
    assert_eq!(project.prompt_enc.as_ref().map(Vec::len), Some(33));
    assert!(matches!(
        stored_resources::get_project(app_state.db.get_pool(), owner.uuid, safe_project.uuid, 8,),
        Err(StoredResourceError::OutputTooLarge)
    ));
    assert!(matches!(
        stored_resources::get_project(
            app_state.db.get_pool(),
            other.uuid,
            safe_project.uuid,
            usize::MAX,
        ),
        Err(StoredResourceError::ConversationProjectNotFound)
    ));
    let (project_page, project_has_more) =
        stored_resources::list_projects(app_state.db.get_pool(), owner.uuid, 1, None, "asc", 4)
            .expect("oversized lookahead project must not reject the safe returned page");
    assert!(project_has_more);
    assert_eq!(project_page.len(), 1);
    assert_eq!(project_page[0].uuid, safe_project.uuid);

    let safe_instruction = app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: None,
            name_enc: Some(vec![0x44; 32]),
            prompt_enc: vec![0x55; 33],
            prompt_tokens: 1,
            is_default: false,
        })
        .expect("safe global instruction should insert");
    let oversized_instruction = app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: None,
            name_enc: Some(vec![0x66; 1024 * 1024]),
            prompt_enc: vec![0x77; 1024 * 1024],
            prompt_tokens: 1,
            is_default: false,
        })
        .expect("oversized sentinel instruction should insert");

    let instruction = stored_resources::get_instruction(
        app_state.db.get_pool(),
        owner.uuid,
        safe_instruction.uuid,
        9,
    )
    .expect("instruction should fit the exact aggregate plaintext limit");
    assert_eq!(instruction.uuid, safe_instruction.uuid);
    assert!(matches!(
        stored_resources::get_instruction(
            app_state.db.get_pool(),
            other.uuid,
            safe_instruction.uuid,
            usize::MAX,
        ),
        Err(StoredResourceError::InstructionNotFound)
    ));
    let (instruction_page, instruction_has_more) =
        stored_resources::list_instructions(app_state.db.get_pool(), owner.uuid, 1, None, "asc", 9)
            .expect("oversized lookahead instruction must not reject the safe returned page");
    assert!(instruction_has_more);
    assert_eq!(instruction_page.len(), 1);
    assert_eq!(instruction_page[0].uuid, safe_instruction.uuid);

    assert!(matches!(
        stored_resources::delete_instruction(
            app_state.db.get_pool(),
            other.uuid,
            oversized_instruction.uuid,
        ),
        Err(StoredResourceError::InstructionNotFound)
    ));
    assert_eq!(
        stored_resources::delete_instruction(
            app_state.db.get_pool(),
            owner.uuid,
            oversized_instruction.uuid,
        )
        .expect("owner narrow deletion should succeed"),
        oversized_instruction.uuid
    );
    let remaining = user_instructions::table
        .filter(user_instructions::uuid.eq(oversized_instruction.uuid))
        .count()
        .get_result::<i64>(
            &mut app_state
                .db
                .get_pool()
                .get()
                .expect("verification connection should open"),
        )
        .expect("instruction deletion should be queryable");
    assert_eq!(remaining, 0);

    let _ = app_state.db.delete_user(&owner);
    let _ = app_state.db.delete_user(&other);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_v2_project_and_instruction_mutations_are_atomic_and_reject_stale_state() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let org_project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner =
        create_response_transaction_test_user(&app_state, org_project.id, marker, "mutation-owner");

    let project = app_state
        .db
        .create_conversation_project(NewConversationProject {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            name_enc: vec![0x11; 32],
        })
        .expect("project should insert");
    app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: Some(project.id),
            name_enc: None,
            prompt_enc: vec![0x22; 33],
            prompt_tokens: 1,
            is_default: false,
        })
        .expect("project instruction should insert");

    let updated_project = stored_resources::update_project(
        app_state.db.get_pool(),
        owner.uuid,
        project.uuid,
        project.updated_at,
        Some(vec![0x31; 34]),
        ProjectInstructionUpdate::Set {
            prompt_enc: vec![0x32; 35],
            prompt_tokens: 2,
        },
    )
    .expect("current project snapshot should update atomically");
    assert_eq!(updated_project.uuid, project.uuid);

    let stored_project = stored_resources::get_project(
        app_state.db.get_pool(),
        owner.uuid,
        project.uuid,
        usize::MAX,
    )
    .expect("updated project should remain readable");
    assert_eq!(stored_project.project.name_enc, vec![0x31; 34]);
    assert_eq!(stored_project.prompt_enc, Some(vec![0x32; 35]));

    assert!(matches!(
        stored_resources::update_project(
            app_state.db.get_pool(),
            owner.uuid,
            project.uuid,
            project.updated_at - chrono::Duration::seconds(1),
            Some(vec![0x41; 36]),
            ProjectInstructionUpdate::Set {
                prompt_enc: vec![0x42; 37],
                prompt_tokens: 3,
            },
        ),
        Err(StoredResourceError::StaleResource)
    ));
    let unchanged_project = stored_resources::get_project(
        app_state.db.get_pool(),
        owner.uuid,
        project.uuid,
        usize::MAX,
    )
    .expect("stale project update must leave both rows unchanged");
    assert_eq!(unchanged_project.project.name_enc, vec![0x31; 34]);
    assert_eq!(unchanged_project.prompt_enc, Some(vec![0x32; 35]));

    let first_instruction = app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: None,
            name_enc: Some(vec![0x51; 32]),
            prompt_enc: vec![0x52; 33],
            prompt_tokens: 4,
            is_default: true,
        })
        .expect("first general instruction should insert");
    let second_instruction = app_state
        .db
        .create_user_instruction(NewUserInstruction {
            uuid: Uuid::new_v4(),
            user_id: owner.uuid,
            project_id: None,
            name_enc: Some(vec![0x61; 32]),
            prompt_enc: vec![0x62; 33],
            prompt_tokens: 5,
            is_default: false,
        })
        .expect("second general instruction should insert");

    let updated_instruction = stored_resources::update_instruction(
        app_state.db.get_pool(),
        owner.uuid,
        second_instruction.uuid,
        second_instruction.updated_at,
        InstructionUpdateCiphertext {
            name_enc: vec![0x71; 34],
            prompt_enc: vec![0x72; 35],
            prompt_tokens: 6,
            is_default: true,
        },
    )
    .expect("current instruction snapshot should update and become default atomically");
    assert!(updated_instruction.is_default);

    let mut conn = app_state
        .db
        .get_pool()
        .get()
        .expect("verification connection should open");
    let instruction_state = user_instructions::table
        .filter(user_instructions::uuid.eq_any([first_instruction.uuid, second_instruction.uuid]))
        .select((
            user_instructions::uuid,
            user_instructions::name_enc,
            user_instructions::prompt_enc,
            user_instructions::prompt_tokens,
            user_instructions::is_default,
        ))
        .load::<(Uuid, Option<Vec<u8>>, Vec<u8>, i32, bool)>(&mut conn)
        .expect("instruction mutation should be queryable");
    let first_state = instruction_state
        .iter()
        .find(|row| row.0 == first_instruction.uuid)
        .expect("first instruction should remain");
    let second_state = instruction_state
        .iter()
        .find(|row| row.0 == second_instruction.uuid)
        .expect("second instruction should remain");
    assert!(!first_state.4);
    assert_eq!(second_state.1, Some(vec![0x71; 34]));
    assert_eq!(second_state.2, vec![0x72; 35]);
    assert_eq!(second_state.3, 6);
    assert!(second_state.4);

    assert!(matches!(
        stored_resources::update_instruction(
            app_state.db.get_pool(),
            owner.uuid,
            second_instruction.uuid,
            second_instruction.updated_at - chrono::Duration::seconds(1),
            InstructionUpdateCiphertext {
                name_enc: vec![0x81; 36],
                prompt_enc: vec![0x82; 37],
                prompt_tokens: 7,
                is_default: false,
            },
        ),
        Err(StoredResourceError::StaleResource)
    ));
    let second_after_stale = stored_resources::get_instruction(
        app_state.db.get_pool(),
        owner.uuid,
        second_instruction.uuid,
        usize::MAX,
    )
    .expect("stale instruction update must leave the row unchanged");
    assert_eq!(second_after_stale.name_enc, Some(vec![0x71; 34]));
    assert_eq!(second_after_stale.prompt_enc, vec![0x72; 35]);
    assert_eq!(second_after_stale.prompt_tokens, 6);
    assert!(second_after_stale.is_default);

    let first_after_clear = stored_resources::get_instruction(
        app_state.db.get_pool(),
        owner.uuid,
        first_instruction.uuid,
        usize::MAX,
    )
    .expect("cleared default instruction should remain readable");
    let first_default = stored_resources::set_default_instruction(
        app_state.db.get_pool(),
        owner.uuid,
        first_instruction.uuid,
        first_after_clear.updated_at,
    )
    .expect("current instruction snapshot should become default atomically");
    assert!(first_default.is_default);

    assert!(matches!(
        stored_resources::set_default_instruction(
            app_state.db.get_pool(),
            owner.uuid,
            second_instruction.uuid,
            second_instruction.updated_at - chrono::Duration::seconds(1),
        ),
        Err(StoredResourceError::StaleResource)
    ));
    let default_state = user_instructions::table
        .filter(user_instructions::uuid.eq_any([first_instruction.uuid, second_instruction.uuid]))
        .select((user_instructions::uuid, user_instructions::is_default))
        .load::<(Uuid, bool)>(&mut conn)
        .expect("default state should be queryable");
    assert_eq!(
        default_state
            .iter()
            .find(|row| row.0 == first_instruction.uuid)
            .map(|row| row.1),
        Some(true)
    );
    assert_eq!(
        default_state
            .iter()
            .find(|row| row.0 == second_instruction.uuid)
            .map(|row| row.1),
        Some(false)
    );

    let _ = app_state.db.delete_user(&owner);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_password_change_cannot_commit_after_destructive_reset_rotates_seed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-password-change-reset-race-{marker}@example.com");
    let old_password = test_credential("old-before-reset-race");
    let reset_password = test_credential("reset-after-reset-race");
    let racing_password = test_credential("racing-after-stale-change");
    let reset_code = "RACE0001";
    let reset_secret = format!("reset-race-secret-{marker}");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;

    let old_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login should not error")
        .expect("old password login should verify and unwrap");
    let old_seed = app_state
        .decrypt_seed_for_auth_context(&old_login.user, &old_login.auth_context)
        .expect("old auth context should unwrap before reset");
    let expected_old_password_enc = old_login
        .user
        .password_enc
        .clone()
        .expect("password user should have old password verifier ciphertext");
    let (racing_password_hash, racing_password_enc) = app_state
        .encrypt_user_password_verifier(zeroize::Zeroizing::new(racing_password.to_string()))
        .await
        .expect("racing password verifier should encrypt");
    let stale_racing_wrapping = app_state
        .new_password_seed_wrapping_for_user(&old_login.user, &racing_password_hash, &old_seed)
        .expect("stale racing password wrap should be constructible before reset");

    insert_valid_reset_request_for_user(&app_state, project.id, &user, reset_code, &reset_secret);
    app_state
        .confirm_password_reset(
            email.clone(),
            reset_code.to_string(),
            reset_secret,
            reset_password.to_string(),
            project.id,
        )
        .await
        .expect("destructive reset should win the race and commit first");

    let stale_password_change_result = app_state.db.update_user_password_and_seed_wrap(
        &old_login.user,
        &expected_old_password_enc,
        racing_password_enc,
        stale_racing_wrapping,
    );
    assert!(
        matches!(
            stale_password_change_result,
            Err(DBError::StaleCredentialState)
        ),
        "password change must fail if destructive reset changed the credential row after old auth proof"
    );

    let old_context_after_reset =
        app_state.verify_seed_wrap_for_auth_context(&old_login.user, &old_login.auth_context);
    assert!(
        matches!(old_context_after_reset, Err(Error::AuthenticationError)),
        "old auth context must remain invalid after the stale password change is rejected"
    );

    let racing_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            racing_password.to_string(),
            project.id,
        )
        .await
        .expect("racing password login should not error");
    assert!(
        racing_login.is_none(),
        "rejected stale password change must not make the racing password valid"
    );

    let reset_login = app_state
        .authenticate_user(Some(email), None, reset_password.to_string(), project.id)
        .await
        .expect("reset password login should not error")
        .expect("reset password should remain valid");
    app_state
        .verify_seed_wrap_for_auth_context(&reset_login.user, &reset_login.auth_context)
        .expect("reset auth context should still unwrap");

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_oauth_seed_wrap_substitution_fails_for_attacker_provider_subject() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-oauth-tamper-victim-{marker}@example.com");
    let attacker_email = format!("aead-oauth-tamper-attacker-{marker}@example.com");
    let victim_provider_subject = format!("victim-google-sub-{marker}");
    let attacker_provider_subject = format!("attacker-google-sub-{marker}");

    let victim = create_oauth_wrapped_user(
        &app_state,
        project.id,
        victim_email,
        "google",
        victim_provider_subject,
    )
    .await;
    let attacker = create_oauth_wrapped_user(
        &app_state,
        project.id,
        attacker_email,
        "google",
        attacker_provider_subject.clone(),
    )
    .await;

    let attacker_auth_context = app_state
        .oauth_auth_context_for_user(&attacker, "google", &attacker_provider_subject)
        .expect("attacker OAuth auth context should compute");
    app_state
        .verify_seed_wrap_for_auth_context(&attacker, &attacker_auth_context)
        .expect("untampered attacker OAuth wrap should verify");

    copy_victim_seed_wrap_ciphertext_to_attacker_for_kind(
        &app_state,
        &victim,
        &attacker,
        CredentialKind::OAuth,
    );

    let attacker_unwrap_after_tamper =
        app_state.verify_seed_wrap_for_auth_context(&attacker, &attacker_auth_context);

    assert!(
        matches!(
            attacker_unwrap_after_tamper,
            Err(Error::AuthenticationError)
        ),
        "copied victim OAuth seed wrap must not unwrap for attacker provider subject"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_oauth_connection_remap_fails_before_victim_seed_unwrap() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-oauth-remap-victim-{marker}@example.com");
    let attacker_email = format!("aead-oauth-remap-attacker-{marker}@example.com");
    let victim_provider_subject = format!("victim-google-remap-sub-{marker}");
    let attacker_provider_subject = format!("attacker-google-remap-sub-{marker}");

    let victim = create_oauth_wrapped_user(
        &app_state,
        project.id,
        victim_email,
        "google",
        victim_provider_subject,
    )
    .await;
    let attacker = create_oauth_wrapped_user(
        &app_state,
        project.id,
        attacker_email,
        "google",
        attacker_provider_subject.clone(),
    )
    .await;

    remap_attacker_oauth_connection_to_victim(
        &app_state,
        &attacker,
        &victim,
        &attacker_provider_subject,
    );

    let google_provider = app_state
        .db
        .get_oauth_provider_by_name("google")
        .expect("test OAuth provider lookup should succeed")
        .expect("test OAuth provider should exist after AppState build");
    let remapped_connection = app_state
        .db
        .get_project_user_oauth_connection_by_provider_subject(
            google_provider.id,
            &attacker_provider_subject,
            project.id,
        )
        .expect("remapped OAuth subject lookup should not error")
        .expect("remapped OAuth subject should resolve to a connection");
    assert_eq!(
        remapped_connection.user_id, victim.uuid,
        "DB tamper precondition should map attacker subject to victim user"
    );

    let victim_auth_context_from_attacker_subject = app_state
        .oauth_auth_context_for_user(&victim, "google", &attacker_provider_subject)
        .expect("victim OAuth auth context should compute from remapped subject");
    let victim_unwrap_after_connection_remap = app_state
        .verify_seed_wrap_for_auth_context(&victim, &victim_auth_context_from_attacker_subject);

    assert!(
        matches!(
            victim_unwrap_after_connection_remap,
            Err(Error::AuthenticationError)
        ),
        "remapped attacker OAuth subject must not unwrap victim seed"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_copied_password_reset_row_mac_fails_for_victim() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-reset-tamper-victim-{marker}@example.com");
    let attacker_email = format!("aead-reset-tamper-attacker-{marker}@example.com");
    let victim_password = test_credential("victim-before-reset-row-tamper");
    let attacker_password = test_credential("attacker-before-reset-row-tamper");
    let reset_code = "R3S3T123";
    let reset_secret = format!("reset-secret-{marker}");
    let attempted_new_password = test_credential("attacker-reset-row-new");

    let victim = create_password_wrapped_user(
        &app_state,
        project.id,
        victim_email.clone(),
        victim_password,
    )
    .await;
    let attacker =
        create_password_wrapped_user(&app_state, project.id, attacker_email, attacker_password)
            .await;

    insert_copied_attacker_reset_request_for_victim(
        &app_state,
        project.id,
        &attacker,
        &victim,
        reset_code,
        &reset_secret,
    );

    let victim_reset_after_row_copy = app_state
        .confirm_password_reset(
            victim_email.clone(),
            reset_code.to_string(),
            reset_secret.clone(),
            attempted_new_password.to_string(),
            project.id,
        )
        .await;

    assert!(
        matches!(
            victim_reset_after_row_copy,
            Err(Error::InvalidPasswordResetRequest)
        ),
        "victim reset must not find a row containing an attacker-bound reset-code MAC"
    );

    let victim_login_after_failed_reset = app_state
        .authenticate_user(
            Some(victim_email),
            None,
            victim_password.to_string(),
            project.id,
        )
        .await
        .expect("victim login after failed reset should not error");
    assert!(
        victim_login_after_failed_reset.is_some(),
        "failed copied reset-row attempt must leave victim password usable"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_destructive_password_reset_invalidates_old_auth_context_and_rotates_seed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-destructive-reset-{marker}@example.com");
    let old_password = test_credential("old-before-destructive-reset");
    let new_password = test_credential("new-after-destructive-reset");
    let reset_code = "R0T8KEY1";
    let reset_secret = format!("destructive-reset-secret-{marker}");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;
    let api_key_name = format!("reset-preserved-api-key-{marker}");
    app_state
        .db
        .create_user_api_key(NewUserApiKey::new(
            user.uuid,
            format!("test-api-key-hash-{marker}"),
            api_key_name.clone(),
        ))
        .expect("test API key should insert before destructive reset");

    let old_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login should not error")
        .expect("old password login should verify and unwrap");
    let old_key = app_state
        .get_user_key(&old_login.user, &old_login.auth_context, None, None)
        .await
        .expect("old key should derive before destructive reset");
    let old_authenticated_seed = app_state
        .decrypt_seed_for_auth_context(&old_login.user, &old_login.auth_context)
        .expect("old auth context should unwrap old seed before destructive reset");

    insert_valid_reset_request_for_user(&app_state, project.id, &user, reset_code, &reset_secret);
    app_state
        .confirm_password_reset(
            email.clone(),
            reset_code.to_string(),
            reset_secret,
            new_password.to_string(),
            project.id,
        )
        .await
        .expect("destructive password reset should complete");

    let old_context_after_reset =
        app_state.verify_seed_wrap_for_auth_context(&old_login.user, &old_login.auth_context);
    assert!(
        matches!(old_context_after_reset, Err(Error::AuthenticationError)),
        "old auth context must not unwrap after destructive reset"
    );

    let old_password_login_after_reset = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            old_password.to_string(),
            project.id,
        )
        .await
        .expect("old password login after reset should not error");
    assert!(
        old_password_login_after_reset.is_none(),
        "old password must not authenticate after destructive reset"
    );

    let new_password_login = app_state
        .authenticate_user(Some(email), None, new_password.to_string(), project.id)
        .await
        .expect("new password login should not error")
        .expect("new password should verify and unwrap after reset");
    let new_key = app_state
        .get_user_key(
            &new_password_login.user,
            &new_password_login.auth_context,
            None,
            None,
        )
        .await
        .expect("new key should derive after destructive reset");

    assert_ne!(
        old_key.secret_bytes(),
        new_key.secret_bytes(),
        "destructive password reset must rotate the user seed"
    );

    let new_authenticated_seed = app_state
        .decrypt_seed_for_auth_context(&new_password_login.user, &new_password_login.auth_context)
        .expect("new auth context should unwrap new seed after destructive reset");
    assert_ne!(
        old_authenticated_seed, new_authenticated_seed,
        "destructive reset must generate a fresh seed"
    );

    let api_keys_after_reset = app_state
        .db
        .get_all_user_api_keys_for_user(user.uuid)
        .expect("API keys should load after destructive reset");
    assert_eq!(
        api_keys_after_reset.len(),
        1,
        "destructive reset should preserve existing user API keys in this release"
    );
    assert_eq!(api_keys_after_reset[0].name, api_key_name);

    let remaining_wraps = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(user.uuid, CredentialKind::Password.as_str())
        .expect("post-reset seed wraps should load");
    assert_eq!(
        remaining_wraps.len(),
        1,
        "destructive reset should leave exactly one password seed wrap"
    );

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_destructive_password_reset_consumes_other_active_reset_requests() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-reset-consumes-stale-{marker}@example.com");
    let old_password = test_credential("old-before-reset-consume");
    let first_new_password = test_credential("new-after-first-reset-consume");
    let second_new_password = test_credential("new-after-stale-reset-consume");
    let first_reset_code = "CONSUME1";
    let first_reset_secret = format!("first-reset-secret-{marker}");
    let stale_reset_code = "CONSUME2";
    let stale_reset_secret = format!("stale-reset-secret-{marker}");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;

    insert_valid_reset_request_for_user(
        &app_state,
        project.id,
        &user,
        first_reset_code,
        &first_reset_secret,
    );
    insert_valid_reset_request_for_user(
        &app_state,
        project.id,
        &user,
        stale_reset_code,
        &stale_reset_secret,
    );
    assert_eq!(
        active_password_reset_request_count(&app_state, user.uuid),
        2,
        "test setup should create two active reset requests"
    );

    app_state
        .confirm_password_reset(
            email.clone(),
            first_reset_code.to_string(),
            first_reset_secret,
            first_new_password.to_string(),
            project.id,
        )
        .await
        .expect("first destructive password reset should complete");

    assert_eq!(
        active_password_reset_request_count(&app_state, user.uuid),
        0,
        "successful destructive reset must invalidate every other active reset request"
    );

    let stale_reset_result = app_state
        .confirm_password_reset(
            email.clone(),
            stale_reset_code.to_string(),
            stale_reset_secret,
            second_new_password.to_string(),
            project.id,
        )
        .await;
    assert!(
        matches!(stale_reset_result, Err(Error::InvalidPasswordResetRequest)),
        "stale reset credential must not trigger a second destructive reset"
    );

    let first_new_login = app_state
        .authenticate_user(
            Some(email.clone()),
            None,
            first_new_password.to_string(),
            project.id,
        )
        .await
        .expect("first new password login should not error")
        .expect("first new password should still authenticate");
    app_state
        .verify_seed_wrap_for_auth_context(&first_new_login.user, &first_new_login.auth_context)
        .expect("first reset password auth context should still unwrap");

    let stale_new_password_login = app_state
        .authenticate_user(
            Some(email),
            None,
            second_new_password.to_string(),
            project.id,
        )
        .await
        .expect("stale reset password login should not error");
    assert!(
        stale_new_password_login.is_none(),
        "stale reset attempt must not rotate the account to the stale new password"
    );

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_destructive_password_reset_wipes_response_storage_cascade() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-reset-cascade-{marker}@example.com");
    let old_password = test_credential("old-before-reset-cascade");
    let new_password = test_credential("new-after-reset-cascade");
    let reset_code = "CASCADE1";
    let reset_secret = format!("cascade-reset-secret-{marker}");

    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;
    insert_response_storage_stack_for_user(&app_state, user.uuid);
    assert_response_storage_counts(&app_state, user.uuid, 1);

    insert_valid_reset_request_for_user(&app_state, project.id, &user, reset_code, &reset_secret);
    app_state
        .confirm_password_reset(
            email,
            reset_code.to_string(),
            reset_secret,
            new_password.to_string(),
            project.id,
        )
        .await
        .expect("destructive password reset should complete with response storage rows present");

    assert_response_storage_counts(&app_state, user.uuid, 0);

    let _ = app_state.db.delete_user(&user);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_atomic_response_tool_items_overwrite_child_linkage_and_order_timestamps() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner = create_response_transaction_test_user(&app_state, project.id, marker, "owner");
    let hostile = create_response_transaction_test_user(&app_state, project.id, marker, "hostile");
    let owner_conversation = insert_response_transaction_test_conversation(&app_state, owner.uuid);
    let hostile_conversation =
        insert_response_transaction_test_conversation(&app_state, hostile.uuid);

    let response_uuid = Uuid::new_v4();
    let message_uuid = Uuid::new_v4();
    let first_call_uuid = Uuid::new_v4();
    let first_output_uuid = Uuid::new_v4();
    let second_call_uuid = Uuid::new_v4();
    let second_output_uuid = Uuid::new_v4();

    let persisted = app_state
        .db
        .create_response_with_message_and_tool_items(
            response_transaction_test_response(response_uuid, owner.uuid, owner_conversation),
            NewUserMessage {
                uuid: message_uuid,
                conversation_id: hostile_conversation,
                response_id: Some(i64::MAX),
                user_id: hostile.uuid,
                content_enc: vec![1, 2, 3],
                prompt_tokens: 3,
            },
            vec![
                response_transaction_test_tool_pair(
                    first_call_uuid,
                    first_output_uuid,
                    hostile.uuid,
                    hostile_conversation,
                ),
                response_transaction_test_tool_pair(
                    second_call_uuid,
                    second_output_uuid,
                    hostile.uuid,
                    hostile_conversation,
                ),
            ],
        )
        .expect("atomic response transaction should succeed");

    assert_eq!(persisted.response.uuid, response_uuid);
    assert_eq!(persisted.response.user_id, owner.uuid);
    assert_eq!(persisted.response.conversation_id, owner_conversation);
    assert_eq!(persisted.user_message.uuid, message_uuid);
    assert_eq!(persisted.user_message.user_id, owner.uuid);
    assert_eq!(persisted.user_message.conversation_id, owner_conversation);
    assert_eq!(
        persisted.user_message.response_id,
        Some(persisted.response.id)
    );
    assert_eq!(persisted.tool_items.len(), 2);

    let mut previous_created_at = persisted.user_message.created_at;
    for pair in &persisted.tool_items {
        assert_eq!(pair.tool_call.user_id, owner.uuid);
        assert_eq!(pair.tool_call.conversation_id, owner_conversation);
        assert_eq!(pair.tool_call.response_id, Some(persisted.response.id));
        assert_eq!(
            pair.tool_call
                .created_at
                .signed_duration_since(previous_created_at),
            chrono::Duration::microseconds(1),
            "each tool call must immediately follow the preceding durable item"
        );

        assert_eq!(pair.tool_output.user_id, owner.uuid);
        assert_eq!(pair.tool_output.conversation_id, owner_conversation);
        assert_eq!(pair.tool_output.response_id, Some(persisted.response.id));
        assert_eq!(pair.tool_output.tool_call_fk, pair.tool_call.id);
        assert_eq!(
            pair.tool_output
                .created_at
                .signed_duration_since(pair.tool_call.created_at),
            chrono::Duration::microseconds(1),
            "each tool output must immediately follow its directly linked call"
        );

        previous_created_at = pair.tool_output.created_at;
    }
    assert_eq!(persisted.last_item_created_at, previous_created_at);

    assert_response_transaction_row_counts(
        &app_state,
        response_uuid,
        message_uuid,
        &[first_call_uuid, second_call_uuid],
        &[first_output_uuid, second_output_uuid],
        (1, 1, 2, 2),
    );

    // Text-only Responses requests use the same atomic insert path with no
    // precomputed tool items; keep that compatibility path covered explicitly.
    let text_response_uuid = Uuid::new_v4();
    let text_message_uuid = Uuid::new_v4();
    let text_only = app_state
        .db
        .create_response_with_message_and_tool_items(
            response_transaction_test_response(text_response_uuid, owner.uuid, owner_conversation),
            NewUserMessage {
                uuid: text_message_uuid,
                conversation_id: owner_conversation,
                response_id: None,
                user_id: owner.uuid,
                content_enc: vec![10, 11, 12],
                prompt_tokens: 3,
            },
            Vec::new(),
        )
        .expect("text-only atomic response transaction should succeed");

    assert!(text_only.tool_items.is_empty());
    assert_eq!(
        text_only.last_item_created_at,
        text_only.user_message.created_at
    );
    assert_response_transaction_row_counts(
        &app_state,
        text_response_uuid,
        text_message_uuid,
        &[],
        &[],
        (1, 1, 0, 0),
    );

    let _ = app_state.db.delete_user(&owner);
    let _ = app_state.db.delete_user(&hostile);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_atomic_response_tool_items_reject_wrong_owner_without_partial_rows() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let conversation_owner =
        create_response_transaction_test_user(&app_state, project.id, marker, "conversation-owner");
    let requester =
        create_response_transaction_test_user(&app_state, project.id, marker, "requester");
    let conversation_id =
        insert_response_transaction_test_conversation(&app_state, conversation_owner.uuid);

    let response_uuid = Uuid::new_v4();
    let message_uuid = Uuid::new_v4();
    let call_uuid = Uuid::new_v4();
    let output_uuid = Uuid::new_v4();
    let result = app_state.db.create_response_with_message_and_tool_items(
        response_transaction_test_response(response_uuid, requester.uuid, conversation_id),
        NewUserMessage {
            uuid: message_uuid,
            conversation_id,
            response_id: None,
            user_id: requester.uuid,
            content_enc: vec![4, 5, 6],
            prompt_tokens: 3,
        },
        vec![response_transaction_test_tool_pair(
            call_uuid,
            output_uuid,
            requester.uuid,
            conversation_id,
        )],
    );

    assert!(
        matches!(
            result,
            Err(DBError::ResponsesError(
                ResponsesError::ConversationNotFound
            ))
        ),
        "a response must not be created in another user's conversation"
    );
    assert_response_transaction_row_counts(
        &app_state,
        response_uuid,
        message_uuid,
        &[call_uuid],
        &[output_uuid],
        (0, 0, 0, 0),
    );

    let _ = app_state.db.delete_user(&conversation_owner);
    let _ = app_state.db.delete_user(&requester);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_atomic_response_tool_items_roll_back_on_mid_batch_constraint_failure() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let owner = create_response_transaction_test_user(&app_state, project.id, marker, "rollback");
    let conversation_id = insert_response_transaction_test_conversation(&app_state, owner.uuid);

    let response_uuid = Uuid::new_v4();
    let message_uuid = Uuid::new_v4();
    let first_call_uuid = Uuid::new_v4();
    let second_call_uuid = Uuid::new_v4();
    let duplicate_output_uuid = Uuid::new_v4();
    let result = app_state.db.create_response_with_message_and_tool_items(
        response_transaction_test_response(response_uuid, owner.uuid, conversation_id),
        NewUserMessage {
            uuid: message_uuid,
            conversation_id,
            response_id: None,
            user_id: owner.uuid,
            content_enc: vec![7, 8, 9],
            prompt_tokens: 3,
        },
        vec![
            response_transaction_test_tool_pair(
                first_call_uuid,
                duplicate_output_uuid,
                owner.uuid,
                conversation_id,
            ),
            response_transaction_test_tool_pair(
                second_call_uuid,
                duplicate_output_uuid,
                owner.uuid,
                conversation_id,
            ),
        ],
    );

    assert!(
        matches!(
            result,
            Err(DBError::ResponsesError(ResponsesError::DatabaseError(
                diesel::result::Error::DatabaseError(
                    diesel::result::DatabaseErrorKind::UniqueViolation,
                    _
                )
            )))
        ),
        "the second output must fail specifically on its duplicate UUID"
    );
    assert_response_transaction_row_counts(
        &app_state,
        response_uuid,
        message_uuid,
        &[first_call_uuid, second_call_uuid],
        &[duplicate_output_uuid],
        (0, 0, 0, 0),
    );

    let _ = app_state.db.delete_user(&owner);
}

fn create_response_transaction_test_user(
    app_state: &AppState,
    project_id: i32,
    marker: Uuid,
    label: &str,
) -> User {
    app_state
        .db
        .create_user(NewUser::new(
            Some(format!("atomic-response-{label}-{marker}@example.com")),
            None,
            project_id,
        ))
        .expect("response transaction test user should insert")
}

fn insert_response_transaction_test_conversation(app_state: &AppState, user_id: Uuid) -> i64 {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    NewConversation {
        uuid: Uuid::new_v4(),
        user_id,
        project_id: None,
        is_pinned: false,
        metadata_enc: None,
    }
    .insert(conn)
    .expect("response transaction test conversation should insert")
    .id
}

fn response_transaction_test_response(
    uuid: Uuid,
    user_id: Uuid,
    conversation_id: i64,
) -> NewResponse {
    NewResponse {
        uuid,
        user_id,
        conversation_id,
        status: ResponseStatus::InProgress,
        model: "atomic-response-transaction-test".to_string(),
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        tool_choice: None,
        parallel_tool_calls: false,
        store: true,
        metadata_enc: None,
    }
}

fn response_transaction_test_tool_pair(
    call_uuid: Uuid,
    output_uuid: Uuid,
    hostile_user_id: Uuid,
    hostile_conversation_id: i64,
) -> (NewToolCall, NewToolOutput) {
    (
        NewToolCall {
            uuid: call_uuid,
            conversation_id: hostile_conversation_id,
            response_id: Some(i64::MAX),
            user_id: hostile_user_id,
            name: "read_image".to_string(),
            arguments_enc: Some(vec![10, 11, 12]),
            argument_tokens: 3,
            status: "completed".to_string(),
            created_at: Utc::now(),
        },
        NewToolOutput {
            uuid: output_uuid,
            conversation_id: hostile_conversation_id,
            response_id: Some(i64::MAX),
            user_id: hostile_user_id,
            tool_call_fk: i64::MAX,
            output_enc: vec![13, 14, 15],
            output_tokens: 3,
            status: "completed".to_string(),
            error: None,
            created_at: Utc::now(),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn assert_response_transaction_row_counts(
    app_state: &AppState,
    response_uuid: Uuid,
    message_uuid: Uuid,
    call_uuids: &[Uuid],
    output_uuids: &[Uuid],
    expected: (i64, i64, i64, i64),
) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    let response_count = responses::table
        .filter(responses::uuid.eq(response_uuid))
        .count()
        .get_result::<i64>(conn)
        .expect("response count should query");
    let message_count = user_messages::table
        .filter(user_messages::uuid.eq(message_uuid))
        .count()
        .get_result::<i64>(conn)
        .expect("user message count should query");
    let call_count = tool_calls::table
        .filter(tool_calls::uuid.eq_any(call_uuids))
        .count()
        .get_result::<i64>(conn)
        .expect("tool call count should query");
    let output_count = tool_outputs::table
        .filter(tool_outputs::uuid.eq_any(output_uuids))
        .count()
        .get_result::<i64>(conn)
        .expect("tool output count should query");

    assert_eq!(
        (response_count, message_count, call_count, output_count),
        expected,
        "response transaction row counts"
    );
}

async fn build_local_test_app_state(database_url: String) -> AppState {
    let db = setup_db(database_url);
    AppStateBuilder::default()
        .app_mode(AppMode::Local)
        .db(db)
        .enclave_key([42u8; 32].to_vec())
        .aws_credential_manager(Arc::new(RwLock::new(None)))
        .openai_api_base("http://localhost:9".to_string())
        .tinfoil_api_base("http://localhost:9".to_string())
        .jwt_secret([24u8; 32].to_vec())
        .build()
        .await
        .expect("local test app state should build")
}

fn first_active_project(app_state: &AppState) -> OrgProject {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    org_projects::table
        .filter(org_projects::status.eq("active"))
        .order(org_projects::id.asc())
        .first::<OrgProject>(conn)
        .expect("test database should contain at least one active project")
}

fn insert_response_storage_stack_for_user(app_state: &AppState, user_id: Uuid) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    let conversation_project = NewConversationProject {
        uuid: Uuid::new_v4(),
        user_id,
        name_enc: vec![25, 26, 27],
    }
    .insert(conn)
    .expect("test conversation project should insert");

    let conversation = NewConversation {
        uuid: Uuid::new_v4(),
        user_id,
        project_id: Some(conversation_project.id),
        is_pinned: false,
        metadata_enc: Some(vec![1, 2, 3]),
    }
    .insert(conn)
    .expect("test conversation should insert");

    let response = NewResponse {
        uuid: Uuid::new_v4(),
        user_id,
        conversation_id: conversation.id,
        status: ResponseStatus::Completed,
        model: "aead-reset-cascade-test".to_string(),
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        tool_choice: None,
        parallel_tool_calls: false,
        store: true,
        metadata_enc: Some(vec![4, 5, 6]),
    }
    .insert(conn)
    .expect("test response should insert");

    NewUserMessage {
        uuid: Uuid::new_v4(),
        conversation_id: conversation.id,
        response_id: Some(response.id),
        user_id,
        content_enc: vec![7, 8, 9],
        prompt_tokens: 3,
    }
    .insert(conn)
    .expect("test user message should insert");

    let assistant_message = NewAssistantMessage {
        uuid: Uuid::new_v4(),
        conversation_id: conversation.id,
        response_id: Some(response.id),
        user_id,
        content_enc: Some(vec![10, 11, 12]),
        completion_tokens: 3,
        status: "completed".to_string(),
        finish_reason: Some("stop".to_string()),
        created_at: Utc::now(),
    }
    .insert(conn)
    .expect("test assistant message should insert");

    let tool_call = NewToolCall {
        uuid: Uuid::new_v4(),
        conversation_id: conversation.id,
        response_id: Some(response.id),
        user_id,
        name: "aead_reset_test_tool".to_string(),
        arguments_enc: Some(vec![13, 14, 15]),
        argument_tokens: 3,
        status: "completed".to_string(),
        created_at: Utc::now(),
    }
    .insert(conn)
    .expect("test tool call should insert");

    NewToolOutput {
        uuid: Uuid::new_v4(),
        conversation_id: conversation.id,
        response_id: Some(response.id),
        user_id,
        tool_call_fk: tool_call.id,
        output_enc: vec![16, 17, 18],
        output_tokens: 3,
        status: "completed".to_string(),
        error: None,
        created_at: Utc::now(),
    }
    .insert(conn)
    .expect("test tool output should insert");

    NewReasoningItem {
        uuid: Uuid::new_v4(),
        conversation_id: conversation.id,
        response_id: Some(response.id),
        assistant_message_id: Some(assistant_message.id),
        user_id,
        content_enc: Some(vec![19, 20, 21]),
        summary_enc: Some(vec![22, 23, 24]),
        reasoning_tokens: 3,
        status: "completed".to_string(),
        created_at: Utc::now(),
    }
    .insert(conn)
    .expect("test reasoning item should insert");

    let summary_time = Utc::now();
    diesel::insert_into(conversation_summaries::table)
        .values((
            conversation_summaries::user_id.eq(user_id),
            conversation_summaries::conversation_id.eq(conversation.id),
            conversation_summaries::from_created_at.eq(summary_time),
            conversation_summaries::to_created_at.eq(summary_time),
            conversation_summaries::message_count.eq(1),
            conversation_summaries::content_enc.eq(vec![28, 29, 30]),
            conversation_summaries::content_tokens.eq(3),
        ))
        .execute(conn)
        .expect("test conversation summary should insert");
}

fn assert_response_storage_counts(app_state: &AppState, user_id: Uuid, expected_count: i64) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    let conversation_count = conversations::table
        .filter(conversations::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("conversation count should query");
    let conversation_project_count = conversation_projects::table
        .filter(conversation_projects::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("conversation project count should query");
    let conversation_summary_count = conversation_summaries::table
        .filter(conversation_summaries::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("conversation summary count should query");
    let response_count = responses::table
        .filter(responses::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("response count should query");
    let user_message_count = user_messages::table
        .filter(user_messages::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("user message count should query");
    let assistant_message_count = assistant_messages::table
        .filter(assistant_messages::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("assistant message count should query");
    let tool_call_count = tool_calls::table
        .filter(tool_calls::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("tool call count should query");
    let tool_output_count = tool_outputs::table
        .filter(tool_outputs::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("tool output count should query");
    let reasoning_item_count = reasoning_items::table
        .filter(reasoning_items::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("reasoning item count should query");

    assert_eq!(conversation_count, expected_count, "conversation row count");
    assert_eq!(
        conversation_project_count, expected_count,
        "conversation project row count"
    );
    assert_eq!(
        conversation_summary_count, expected_count,
        "conversation summary row count"
    );
    assert_eq!(response_count, expected_count, "response row count");
    assert_eq!(user_message_count, expected_count, "user message row count");
    assert_eq!(
        assistant_message_count, expected_count,
        "assistant message row count"
    );
    assert_eq!(tool_call_count, expected_count, "tool call row count");
    assert_eq!(tool_output_count, expected_count, "tool output row count");
    assert_eq!(
        reasoning_item_count, expected_count,
        "reasoning item row count"
    );
}

async fn create_password_wrapped_user(
    app_state: &AppState,
    project_id: i32,
    email: String,
    password: &str,
) -> User {
    let secret_key =
        SecretKey::from_slice(&app_state.enclave_key).expect("test enclave key should be valid");
    let password_hash = generate_hash(password);
    let password_enc = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;
    let user_seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
        .await
        .expect("test seed should generate")
        .to_string();

    let user = app_state
        .db
        .create_user(NewUser::new(Some(email), Some(password_enc), project_id))
        .expect("test user should insert");

    app_state
        .create_password_seed_wrap_for_user(&user, &password_hash, user_seed_words.as_bytes())
        .expect("test user seed wrap should insert");

    user
}

async fn create_oauth_wrapped_user(
    app_state: &AppState,
    project_id: i32,
    email: String,
    provider_name: &str,
    provider_user_id: String,
) -> User {
    let user_seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
        .await
        .expect("test seed should generate")
        .to_string();

    let user = app_state
        .db
        .create_user(NewUser::new(Some(email), None, project_id))
        .expect("test OAuth user should insert");

    let provider = app_state
        .db
        .get_oauth_provider_by_name(provider_name)
        .expect("test OAuth provider lookup should succeed")
        .expect("test OAuth provider should exist after AppState build");

    app_state
        .db
        .create_user_oauth_connection(NewUserOAuthConnection {
            user_id: user.uuid,
            provider_id: provider.id,
            provider_user_id: provider_user_id.clone(),
            access_token_enc: Vec::new(),
            refresh_token_enc: None,
            expires_at: None,
        })
        .expect("test OAuth connection should insert");

    app_state
        .create_oauth_seed_wrap_for_user(
            &user,
            provider_name,
            &provider_user_id,
            user_seed_words.as_bytes(),
        )
        .expect("test OAuth seed wrap should insert");

    user
}

fn copy_attacker_password_verifier_to_victim(app_state: &AppState, attacker: &User, victim: &User) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    victim
        .update_password(conn, attacker.password_enc.clone())
        .expect("tampered victim password verifier should update");
}

fn copy_victim_password_verifier_to_attacker(app_state: &AppState, victim: &User, attacker: &User) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    attacker
        .update_password(conn, victim.password_enc.clone())
        .expect("tampered attacker password verifier should update");
}

fn copy_victim_kv_rows_to_attacker(app_state: &AppState, victim: &User, attacker: &User) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let victim_rows = UserKV::get_all_for_user(conn, victim.uuid)
        .expect("victim KV rows should load before copy");
    assert!(
        !victim_rows.is_empty(),
        "DB tamper precondition requires at least one victim KV row"
    );

    for row in victim_rows {
        NewUserKV::new(attacker.uuid, row.key_enc, row.value_enc)
            .insert(conn)
            .expect("copied victim KV row should insert for attacker");
    }
}

fn remap_attacker_oauth_connection_to_victim(
    app_state: &AppState,
    attacker: &User,
    victim: &User,
    attacker_provider_subject: &str,
) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    let updated_rows = diesel::update(user_oauth_connections::table)
        .filter(user_oauth_connections::user_id.eq(attacker.uuid))
        .filter(user_oauth_connections::provider_user_id.eq(attacker_provider_subject))
        .set(user_oauth_connections::user_id.eq(victim.uuid))
        .execute(conn)
        .expect("tampered OAuth connection should update");

    assert_eq!(
        updated_rows, 1,
        "DB tamper precondition should move exactly one OAuth connection"
    );
}

fn insert_copied_attacker_reset_request_for_victim(
    app_state: &AppState,
    project_id: i32,
    attacker: &User,
    victim: &User,
    reset_code: &str,
    reset_secret: &str,
) {
    let attacker_reset_code_mac = password_reset_code_mac(
        &app_state.enclave_key,
        project_id,
        attacker.uuid,
        reset_code,
    )
    .expect("attacker reset-code MAC should compute");
    let copied_request = NewPasswordResetRequest::new(
        victim.uuid,
        generate_reset_hash(reset_secret.to_string()),
        attacker_reset_code_mac.to_vec(),
        24,
    );

    app_state
        .db
        .create_password_reset_request(copied_request)
        .expect("copied reset request row should insert");
}

fn insert_valid_reset_request_for_user(
    app_state: &AppState,
    project_id: i32,
    user: &User,
    reset_code: &str,
    reset_secret: &str,
) {
    let reset_code_mac =
        password_reset_code_mac(&app_state.enclave_key, project_id, user.uuid, reset_code)
            .expect("reset-code MAC should compute");
    let request = NewPasswordResetRequest::new(
        user.uuid,
        generate_reset_hash(reset_secret.to_string()),
        reset_code_mac.to_vec(),
        24,
    );

    app_state
        .db
        .create_password_reset_request(request)
        .expect("valid reset request row should insert");
}

fn active_password_reset_request_count(app_state: &AppState, user_id: Uuid) -> i64 {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    password_reset_requests::table
        .filter(password_reset_requests::user_id.eq(user_id))
        .filter(password_reset_requests::is_reset.eq(false))
        .count()
        .get_result::<i64>(conn)
        .expect("active password reset request count should query")
}

fn tamper_password_wrap_lookup_hash(app_state: &AppState, user: &User, new_lookup_hash: Vec<u8>) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");

    let updated_rows = diesel::update(user_seed_wrappings::table)
        .filter(user_seed_wrappings::user_id.eq(user.uuid))
        .filter(user_seed_wrappings::credential_kind.eq(CredentialKind::Password.as_str()))
        .set(user_seed_wrappings::credential_lookup_hash.eq(new_lookup_hash))
        .execute(conn)
        .expect("tampered password wrap lookup hash should update");

    assert_eq!(
        updated_rows, 1,
        "DB tamper precondition should mutate exactly one password seed wrap"
    );
}

fn copy_victim_seed_wrap_ciphertext_to_attacker(
    app_state: &AppState,
    victim: &User,
    attacker: &User,
) {
    copy_victim_seed_wrap_ciphertext_to_attacker_for_kind(
        app_state,
        victim,
        attacker,
        CredentialKind::Password,
    );
}

fn copy_victim_seed_wrap_ciphertext_to_attacker_for_kind(
    app_state: &AppState,
    victim: &User,
    attacker: &User,
    credential_kind: CredentialKind,
) {
    let victim_wraps = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(victim.uuid, credential_kind.as_str())
        .expect("victim seed wraps should load");
    assert_eq!(
        victim_wraps.len(),
        1,
        "tamper helper expects exactly one victim wrap for the credential kind"
    );
    let victim_wrap = victim_wraps.into_iter().next().unwrap();

    let attacker_wraps = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(attacker.uuid, credential_kind.as_str())
        .expect("attacker seed wraps should load");
    assert_eq!(
        attacker_wraps.len(),
        1,
        "tamper helper expects exactly one attacker wrap for the credential kind"
    );
    let attacker_wrap = attacker_wraps.into_iter().next().unwrap();

    app_state
        .db
        .upsert_user_seed_wrapping(NewUserSeedWrapping::new(
            attacker.uuid,
            credential_kind.as_str(),
            attacker_wrap.credential_lookup_hash,
            attacker_wrap.wrapping_version,
            victim_wrap.seed_enc,
        ))
        .expect("tampered attacker seed wrap should update");
}
