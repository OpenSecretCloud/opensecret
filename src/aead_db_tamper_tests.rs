use crate::{
    db::setup_db,
    encrypt::{decrypt_with_key, encrypt_with_key},
    generate_reset_hash,
    login_routes::RegisterCredentials,
    models::{
        oauth::NewUserOAuthConnection,
        org_projects::OrgProject,
        password_reset::NewPasswordResetRequest,
        schema::{org_projects, user_oauth_connections, users},
        user_kv::{NewUserKV, UserKV},
        user_seed_wrappings::NewUserSeedWrapping,
        users::{NewUser, User},
    },
    private_key::{generate_twelve_word_seed, plaintext_user_seed_to_mnemonic},
    seed_wrapping::{password_reset_code_mac, CredentialKind},
    AppMode, AppState, AppStateBuilder, Error,
};
use diesel::{ExpressionMethods, QueryDsl, RunQueryDsl};
use password_auth::generate_hash;
use secp256k1::SecretKey;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

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
    let password = "registration-password-before-login";

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
    let victim_password = "victim-password-before-tamper";
    let attacker_password = "attacker-password-before-tamper";

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
    let victim_password = "victim-password-before-password-row-tamper";
    let attacker_password = "attacker-password-before-password-row-tamper";

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
    let victim_password = "victim-password-before-copy-to-attacker";
    let attacker_password = "attacker-password-before-copy-to-attacker";

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
async fn db_legacy_seed_substitution_does_not_change_authenticated_attacker_key() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-legacy-seed-victim-{marker}@example.com");
    let attacker_email = format!("aead-legacy-seed-attacker-{marker}@example.com");
    let victim_password = "victim-password-before-legacy-seed-tamper";
    let attacker_password = "attacker-password-before-legacy-seed-tamper";

    let victim =
        create_password_wrapped_user(&app_state, project.id, victim_email, victim_password).await;
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
        .expect("untampered attacker login should not error")
        .expect("untampered attacker login should verify and unwrap");
    let attacker_key_before = app_state
        .get_user_key(
            &attacker_login_before_tamper.user,
            &attacker_login_before_tamper.auth_context,
            None,
            None,
        )
        .await
        .expect("untampered attacker key should derive");

    copy_victim_legacy_seed_to_attacker(&app_state, &victim, &attacker);

    let attacker_login_after_tamper = app_state
        .authenticate_user(
            Some(attacker_email),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await
        .expect("legacy seed tamper should not make attacker login error")
        .expect("attacker password login should still verify against attacker wrap");
    let attacker_key_after = app_state
        .get_user_key(
            &attacker_login_after_tamper.user,
            &attacker_login_after_tamper.auth_context,
            None,
            None,
        )
        .await
        .expect("attacker key should still derive from authenticated seed wrap");

    assert_eq!(
        attacker_key_before.secret_bytes(),
        attacker_key_after.secret_bytes(),
        "copied legacy users.seed_enc must not affect authenticated attacker seed unwrap"
    );

    let _ = app_state.db.delete_user(&victim);
    let _ = app_state.db.delete_user(&attacker);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_legacy_seed_substitution_does_not_export_victim_mnemonic() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let victim_email = format!("aead-private-key-victim-{marker}@example.com");
    let attacker_email = format!("aead-private-key-attacker-{marker}@example.com");
    let victim_password = "victim-password-before-private-key-export-tamper";
    let attacker_password = "attacker-password-before-private-key-export-tamper";

    let victim =
        create_password_wrapped_user(&app_state, project.id, victim_email, victim_password).await;
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
        .expect("untampered attacker login should not error")
        .expect("untampered attacker login should verify and unwrap");
    let attacker_seed_before = app_state
        .decrypt_seed_for_auth_context(
            &attacker_login_before_tamper.user,
            &attacker_login_before_tamper.auth_context,
        )
        .expect("untampered attacker seed should unwrap");
    let attacker_mnemonic_before = plaintext_user_seed_to_mnemonic(&attacker_seed_before)
        .expect("untampered attacker seed should parse as mnemonic")
        .to_string();

    let legacy_secret_key = SecretKey::from_slice(&app_state.enclave_key)
        .expect("test enclave key should be a valid legacy SecretKey");
    let victim_legacy_seed = decrypt_with_key(
        &legacy_secret_key,
        victim
            .seed_encrypted()
            .expect("victim legacy seed should exist"),
    )
    .expect("victim legacy seed should decrypt");
    let victim_mnemonic = plaintext_user_seed_to_mnemonic(&victim_legacy_seed)
        .expect("victim legacy seed should parse as mnemonic")
        .to_string();
    assert_ne!(
        attacker_mnemonic_before, victim_mnemonic,
        "test precondition should use different victim and attacker seeds"
    );

    copy_victim_legacy_seed_to_attacker(&app_state, &victim, &attacker);

    let attacker_login_after_tamper = app_state
        .authenticate_user(
            Some(attacker_email),
            None,
            attacker_password.to_string(),
            project.id,
        )
        .await
        .expect("legacy seed tamper should not make attacker login error")
        .expect("attacker password login should still verify against attacker wrap");
    let exported_seed_after_tamper = app_state
        .decrypt_seed_for_auth_context(
            &attacker_login_after_tamper.user,
            &attacker_login_after_tamper.auth_context,
        )
        .expect("attacker private-key export seed should still unwrap");
    let exported_mnemonic_after_tamper =
        plaintext_user_seed_to_mnemonic(&exported_seed_after_tamper)
            .expect("attacker private-key export seed should parse as mnemonic")
            .to_string();

    assert_eq!(
        exported_mnemonic_after_tamper, attacker_mnemonic_before,
        "copied legacy users.seed_enc must not change the authenticated private-key export seed"
    );
    assert_ne!(
        exported_mnemonic_after_tamper, victim_mnemonic,
        "copied legacy users.seed_enc must not export the victim mnemonic"
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
    let victim_password = "victim-password-before-kv-copy";
    let attacker_password = "attacker-password-before-kv-copy";

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
            "copied-kv-secret".to_string(),
            "victim plaintext must not leak".to_string(),
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
            "copied-kv-secret".to_string(),
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
async fn db_password_change_invalidates_old_auth_context_and_preserves_seed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("aead-password-change-{marker}@example.com");
    let old_password = "old-password-before-change";
    let new_password = "new-password-after-change";

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
async fn db_guest_conversion_invalidates_old_auth_context_and_preserves_seed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let guest_password = "guest-password-before-conversion";
    let email_password = "email-password-after-guest-conversion";
    let email = format!("aead-guest-conversion-{marker}@example.com");

    let guest = create_guest_wrapped_user(&app_state, project.id, guest_password).await;

    let guest_login = app_state
        .authenticate_user(
            None,
            Some(guest.uuid),
            guest_password.to_string(),
            project.id,
        )
        .await
        .expect("guest login should not error")
        .expect("guest password should verify and unwrap");
    let guest_key_before = app_state
        .get_user_key(&guest_login.user, &guest_login.auth_context, None, None)
        .await
        .expect("guest key should derive before conversion");

    let (updated_user, new_auth_context) = app_state
        .convert_guest_to_email_and_seed_wrap(
            &guest_login.user,
            &guest_login.auth_context,
            email.clone(),
            email_password.to_string(),
            Some("Converted Guest".to_string()),
        )
        .await
        .expect("guest conversion should rewrap seed");

    let old_context_after_conversion =
        app_state.verify_seed_wrap_for_auth_context(&guest_login.user, &guest_login.auth_context);
    assert!(
        matches!(
            old_context_after_conversion,
            Err(Error::AuthenticationError)
        ),
        "old guest auth context must not unwrap after guest-to-email conversion"
    );

    let old_guest_login_after_conversion = app_state
        .authenticate_user(
            None,
            Some(guest.uuid),
            guest_password.to_string(),
            project.id,
        )
        .await
        .expect("old guest login after conversion should not error");
    assert!(
        old_guest_login_after_conversion.is_none(),
        "old guest password must not authenticate after conversion"
    );

    let email_login = app_state
        .authenticate_user(Some(email), None, email_password.to_string(), project.id)
        .await
        .expect("email login after conversion should not error")
        .expect("email password should verify and unwrap after conversion");
    let email_key_after = app_state
        .get_user_key(&email_login.user, &email_login.auth_context, None, None)
        .await
        .expect("email key should derive after conversion");

    app_state
        .verify_seed_wrap_for_auth_context(&updated_user, &new_auth_context)
        .expect("new auth context returned by guest conversion should unwrap");
    assert_eq!(
        guest_key_before.secret_bytes(),
        email_key_after.secret_bytes(),
        "guest-to-email conversion must preserve the existing user seed"
    );

    let _ = app_state.db.delete_user(&updated_user);
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
    let victim_password = "victim-password-before-reset-row-tamper";
    let attacker_password = "attacker-password-before-reset-row-tamper";
    let reset_code = "R3S3T123";
    let reset_secret = format!("reset-secret-{marker}");
    let attempted_new_password = "attacker-reset-row-new-password";

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
    let old_password = "old-password-before-destructive-reset";
    let new_password = "new-password-after-destructive-reset";
    let reset_code = "R0T8KEY1";
    let reset_secret = format!("destructive-reset-secret-{marker}");

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
        .expect("old key should derive before destructive reset");

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

async fn build_local_test_app_state(database_url: String) -> AppState {
    let db = setup_db(database_url);
    AppStateBuilder::default()
        .app_mode(AppMode::Local)
        .db(db)
        .enclave_key([42u8; 32].to_vec())
        .aws_credential_manager(Arc::new(RwLock::new(None)))
        .openai_api_base("http://localhost:9".to_string())
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
    let legacy_seed_enc = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

    let user = app_state
        .db
        .create_user(NewUser::new(
            Some(email),
            Some(password_enc),
            project_id,
            legacy_seed_enc,
        ))
        .expect("test user should insert");

    app_state
        .create_password_seed_wrap_for_user(&user, &password_hash, user_seed_words.as_bytes())
        .expect("test user seed wrap should insert");

    user
}

async fn create_guest_wrapped_user(app_state: &AppState, project_id: i32, password: &str) -> User {
    let secret_key =
        SecretKey::from_slice(&app_state.enclave_key).expect("test enclave key should be valid");
    let password_hash = generate_hash(password);
    let password_enc = encrypt_with_key(&secret_key, password_hash.as_bytes()).await;
    let user_seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
        .await
        .expect("test seed should generate")
        .to_string();
    let legacy_seed_enc = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

    let user = app_state
        .db
        .create_user(NewUser::new(
            None,
            Some(password_enc),
            project_id,
            legacy_seed_enc,
        ))
        .expect("test guest user should insert");

    app_state
        .create_password_seed_wrap_for_user(&user, &password_hash, user_seed_words.as_bytes())
        .expect("test guest seed wrap should insert");

    user
}

async fn create_oauth_wrapped_user(
    app_state: &AppState,
    project_id: i32,
    email: String,
    provider_name: &str,
    provider_user_id: String,
) -> User {
    let secret_key =
        SecretKey::from_slice(&app_state.enclave_key).expect("test enclave key should be valid");
    let user_seed_words = generate_twelve_word_seed(app_state.aws_credential_manager.clone())
        .await
        .expect("test seed should generate")
        .to_string();
    let legacy_seed_enc = encrypt_with_key(&secret_key, user_seed_words.as_bytes()).await;

    let user = app_state
        .db
        .create_user(NewUser::new(Some(email), None, project_id, legacy_seed_enc))
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

fn copy_victim_legacy_seed_to_attacker(app_state: &AppState, victim: &User, attacker: &User) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let victim_legacy_seed = victim
        .seed_encrypted()
        .expect("victim legacy seed should exist")
        .to_vec();

    let updated_rows = diesel::update(users::table)
        .filter(users::uuid.eq(attacker.uuid))
        .set(users::seed_enc.eq(Some(victim_legacy_seed)))
        .execute(conn)
        .expect("tampered attacker legacy seed should update");

    assert_eq!(
        updated_rows, 1,
        "DB tamper precondition should update exactly one attacker user row"
    );
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
    let victim_wrap = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(victim.uuid, credential_kind.as_str())
        .expect("victim seed wraps should load")
        .into_iter()
        .next()
        .expect("victim credential wrap should exist");
    let attacker_wrap = app_state
        .db
        .get_user_seed_wrappings_for_user_and_kind(attacker.uuid, credential_kind.as_str())
        .expect("attacker seed wraps should load")
        .into_iter()
        .next()
        .expect("attacker credential wrap should exist");

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
