use crate::db::{
    clear_maple_pairing_authority_scoped_access_observer_for_test,
    fail_next_maple_pairing_create_before_commit_for_test, make_maple_pairing_pending_due_for_test,
    maple_pairing_revocation_highwater_lookup_digest_for_test,
    observe_maple_pairing_authority_scoped_access_for_test,
    observe_next_maple_pairing_authority_lock_contention_for_test,
    pause_next_maple_device_registration_before_commit_for_test,
    restore_maple_pairing_authority_global_root_for_test,
    run_maple_pairing_authority_ssi_race_for_test,
    seed_maple_pairing_highwater_group_capacity_for_test,
    tamper_maple_pairing_authority_global_root_for_test, MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT,
    MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES,
};
use crate::{
    db::{setup_db, DBConnection, DBError},
    encrypt::encrypt_with_key,
    generate_reset_hash,
    login_routes::RegisterCredentials,
    models::{
        maple_devices::{
            MapleDevice, MapleDeviceListAuthorization, MapleDeviceListCursor,
            MapleDeviceRegistrationReceipt, NewMapleDeviceRegistration,
        },
        maple_pairing_db::{
            MaplePairing, MaplePairingApproval, MaplePairingAuthorityAccountHead,
            MaplePairingAuthorization, MaplePairingConfirmation, MaplePairingCursor,
            MaplePairingOperationKind, MaplePairingOperationReceipt, MaplePairingRevocation,
            MaplePairingRevocationAck, MaplePairingRevocationContext,
            MaplePairingRevocationHighwater, MaplePairingRevocationMaterial, MaplePairingRole,
            MaplePairingState, NewMaplePairingRequest, StoredMaplePairingPayloadV1,
        },
        maple_pairings::{
            sign_pair_authorization, sign_pair_request_ticket, sign_pair_revocation,
            ApproveMaplePairingRequest, CreateMaplePairingRequest, Ed25519MaplePairingIssuer,
            MaplePairAuthorizationV1, MaplePairRevocationV1, MaplePairingDirection,
            MaplePairingIssuer, MaplePairingIssuerKeySetV1, MaplePairingMutationResponse,
            MaplePairingRole as WireMaplePairingRole, MaplePairingState as WireMaplePairingState,
            MaplePairingStatusV1, MapleRevocationSyncV1, RevokeMaplePairingRequest,
            MAPLE_PAIRING_ARTIFACT_VERSION_V1, MAPLE_PAIRING_PROTOCOL_VERSION_V1,
            MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
        },
        oauth::NewUserOAuthConnection,
        org_projects::{NewOrgProject, OrgProject},
        orgs::NewOrg,
        password_reset::NewPasswordResetRequest,
        responses::{
            NewAssistantMessage, NewConversation, NewConversationProject, NewReasoningItem,
            NewResponse, NewToolCall, NewToolOutput, NewUserMessage, ResponseStatus,
        },
        schema::{
            assistant_messages, conversation_projects, conversation_summaries, conversations,
            maple_device_registration_operations, maple_devices,
            maple_pairing_authority_account_heads, maple_pairing_authority_global_heads,
            maple_pairing_authority_org_heads, maple_pairing_authority_project_heads,
            maple_pairing_host_states, maple_pairing_installation_retirements,
            maple_pairing_lineages, maple_pairing_operations,
            maple_pairing_registration_operation_tombstones, maple_pairing_reset_clear_admissions,
            maple_pairing_reset_clear_obligations, maple_pairing_revocation_events,
            maple_pairing_revocation_highwaters, maple_pairings, org_projects, orgs,
            password_reset_requests, reasoning_items, responses, tool_calls, tool_outputs,
            user_messages, user_oauth_connections, user_seed_wrappings, users,
        },
        user_api_keys::NewUserApiKey,
        user_kv::{NewUserKV, UserKV},
        user_seed_wrappings::NewUserSeedWrapping,
        users::{NewUser, User},
    },
    private_key::generate_twelve_word_seed,
    seed_wrapping::{
        encrypt_seed_v1, password_reset_code_mac, AuthBinding, CredentialKind, SEED_WRAP_VERSION_V1,
    },
    AppMode, AppState, AppStateBuilder, Error,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::Utc;
use diesel::{Connection, ExpressionMethods, QueryDsl, RunQueryDsl};
use ed25519_dalek::{Signer, SigningKey};
use password_auth::generate_hash;
use secp256k1::SecretKey;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Barrier;
use tokio::sync::RwLock;
use uuid::Uuid;

fn test_credential(label: &str) -> &'static str {
    Box::leak(format!("aead-test-credential-{label}").into_boxed_str())
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
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
        .encrypt_user_password_verifier(racing_password.to_string())
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&attacker, &app_state.enclave_key);
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
    let stale_registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        61,
    );
    register_test_maple_device(&app_state, stale_registration.clone())
        .expect("Maple device should register before destructive reset");
    assert_maple_device_row_counts(&app_state, user.uuid, 1, 1);

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
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
    assert!(matches!(
        register_test_maple_device(&app_state, stale_registration),
        Err(DBError::StaleCredentialState)
    ));
    assert!(matches!(
        list_test_maple_devices(
            app_state.db.as_ref(),
            MapleDeviceListAuthorization {
                user_id: user.uuid,
                project_id: project.id,
                auth_credential_kind: CredentialKind::Password.as_str().to_string(),
                auth_binding: old_login.auth_context.auth_binding,
                enclave_key: app_state.enclave_key.clone(),
            },
            10,
            None,
        ),
        Err(DBError::StaleCredentialState)
    ));
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);

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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_destructive_reset_rotates_revocation_stream_and_rejects_stale_namespace() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let marker = Uuid::new_v4();
    let email = format!("maple-revocation-reset-{marker}@example.com");
    let old_password = test_credential("maple-revocation-reset-old");
    let new_password = test_credential("maple-revocation-reset-new");
    let reset_code = "STREAM01";
    let reset_secret = format!("maple-revocation-reset-secret-{marker}");
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
        .expect("old password should authenticate");
    let old_authorization = MaplePairingAuthorization {
        user_id: user.uuid,
        project_id: project.id,
        auth_credential_kind: CredentialKind::Password.as_str().to_string(),
        auth_binding: old_login.auth_context.auth_binding,
        enclave_key: app_state.enclave_key.clone(),
    };

    let controller = build_test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        old_login.auth_context.auth_binding,
        1,
        121,
    );
    let host = build_test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        old_login.auth_context.auth_binding,
        1,
        125,
    );
    register_test_maple_device(&app_state, controller.clone())
        .expect("controller should register before reset");
    register_test_maple_device(&app_state, host.clone())
        .expect("host should register before reset");
    let (old_stream_id, old_stream_generation) =
        current_maple_revocation_stream(&app_state, &old_authorization, host.registration_id);
    assert_ne!(old_stream_id, Uuid::nil());
    assert_eq!(old_stream_generation, 1);

    let (first_request_id, first_pair_id, first_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &old_authorization,
        controller.registration_id,
        host.registration_id,
        129,
    );
    let first_event_id = Uuid::new_v4();
    let (_, first_event_digest) = revoke_test_maple_pairing(
        &app_state,
        &old_authorization,
        controller.registration_id,
        MaplePairingRole::Controller,
        first_request_id,
        first_pair_id,
        3,
        first_incarnation,
        old_stream_id,
        old_stream_generation,
        first_event_id,
        &|context| {
            assert_eq!(context.revocation_stream_id, old_stream_id);
            assert_eq!(context.revocation_stream_generation, old_stream_generation);
            assert_eq!(context.issuer_sequence, 1);
        },
        &|_| {},
    )
    .expect("first pre-reset revocation should commit");
    app_state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: old_authorization.clone(),
            operation_id: Uuid::new_v4(),
            request_mac: vec![138; 32],
            host_registration_id: host.registration_id,
            revocation_stream_id: old_stream_id,
            revocation_stream_generation: old_stream_generation,
            event_id: first_event_id,
            issuer_sequence: 1,
            event_digest: first_event_digest,
            expected_previous_issuer_sequence: 0,
            checkpoint_issuer_key_id: "maple-test-issuer-current".to_string(),
            receipt_version: 1,
            receipt_enc: vec![139, 140],
            accepted_at: Utc::now(),
        })
        .expect("first pre-reset revocation should ACK");

    let (second_request_id, second_pair_id, second_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &old_authorization,
        controller.registration_id,
        host.registration_id,
        141,
    );
    let second_event_id = Uuid::new_v4();
    revoke_test_maple_pairing(
        &app_state,
        &old_authorization,
        controller.registration_id,
        MaplePairingRole::Controller,
        second_request_id,
        second_pair_id,
        3,
        second_incarnation,
        old_stream_id,
        old_stream_generation,
        second_event_id,
        &|context| {
            assert_eq!(context.revocation_stream_id, old_stream_id);
            assert_eq!(context.issuer_sequence, 2);
        },
        &|_| {},
    )
    .expect("second pre-reset revocation should remain unacknowledged");
    let old_page = app_state
        .db
        .list_maple_pairing_revocations(
            old_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        )
        .expect("pre-reset page should expose the unacknowledged suffix");
    assert_eq!(old_page.events.len(), 1);
    assert_eq!(old_page.events[0].event.uuid, second_event_id);
    assert_eq!(old_page.last_issued_revocation_sequence, 2);
    assert_eq!(old_page.last_acked_revocation_sequence, 1);

    let lookup_digest = maple_pairing_revocation_highwater_lookup_digest_for_test(
        &app_state.enclave_key,
        user.uuid,
        project.id,
        host.installation_id,
    )
    .expect("host tombstone lookup digest should derive");
    let original_highwater = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_revocation_highwaters::table
            .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(&lookup_digest))
            .first::<MaplePairingRevocationHighwater>(conn)
            .expect("host high-water tombstone should exist")
    };
    assert_eq!(
        original_highwater.last_issued_revocation_sequence, 2,
        "unacknowledged N+1 must still advance the durable allocation fence"
    );

    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(maple_pairing_revocation_highwaters::revocation_stream_id.eq(Uuid::new_v4()))
        .execute(conn)
        .expect("test should tamper the tombstone stream ID");
    }
    let stream_id_tamper = app_state
        .db
        .list_maple_pairing_revocations(
            old_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        )
        .map(|_| ());
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(
            maple_pairing_revocation_highwaters::revocation_stream_id
                .eq(original_highwater.revocation_stream_id),
        )
        .execute(conn)
        .expect("test should restore the tombstone stream ID");
    }
    assert!(
        matches!(
            &stream_id_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "stream-ID tamper should fail exhaustive authority validation: {stream_id_tamper:?}"
    );
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(
            maple_pairing_revocation_highwaters::revocation_stream_generation
                .eq(original_highwater.revocation_stream_generation + 1),
        )
        .execute(conn)
        .expect("test should tamper the tombstone stream generation");
    }
    let stream_generation_tamper = app_state
        .db
        .list_maple_pairing_revocations(
            old_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        )
        .map(|_| ());
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(
            maple_pairing_revocation_highwaters::revocation_stream_generation
                .eq(original_highwater.revocation_stream_generation),
        )
        .execute(conn)
        .expect("test should restore the tombstone stream generation");
    }
    assert!(
        matches!(
            &stream_generation_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "stream-generation tamper should fail exhaustive authority validation: {stream_generation_tamper:?}"
    );
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(maple_pairing_revocation_highwaters::record_mac.eq(vec![0; 32]))
        .execute(conn)
        .expect("test should tamper the tombstone MAC");
    }
    let record_mac_tamper = app_state
        .db
        .list_maple_pairing_revocations(
            old_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        )
        .map(|_| ());
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_highwaters::table
                .filter(maple_pairing_revocation_highwaters::id.eq(original_highwater.id)),
        )
        .set(maple_pairing_revocation_highwaters::record_mac.eq(&original_highwater.record_mac))
        .execute(conn)
        .expect("test should restore the tombstone MAC");
    }
    assert!(
        matches!(
            &record_mac_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "record-MAC tamper should fail exhaustive authority validation: {record_mac_tamper:?}"
    );

    insert_valid_reset_request_for_user(&app_state, project.id, &user, reset_code, &reset_secret);
    let first_reset = app_state.confirm_password_reset(
        email.clone(),
        reset_code.to_string(),
        reset_secret.clone(),
        new_password.to_string(),
        project.id,
    );
    let racing_replay = app_state.confirm_password_reset(
        email.clone(),
        reset_code.to_string(),
        reset_secret,
        new_password.to_string(),
        project.id,
    );
    let (first_reset, racing_replay) = tokio::join!(first_reset, racing_replay);
    assert!(
        first_reset.is_ok() ^ racing_replay.is_ok(),
        "exactly one concurrent destructive-reset attempt must commit"
    );
    assert_maple_pairing_row_counts(&app_state, user.uuid, 0, 0, 0, 0, 0);
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);

    let rotated_highwater = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_revocation_highwaters::table
            .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(&lookup_digest))
            .order(maple_pairing_revocation_highwaters::revocation_stream_generation.desc())
            .first::<MaplePairingRevocationHighwater>(conn)
            .expect("destructive reset must preserve and rotate the host tombstone")
    };
    assert_ne!(rotated_highwater.revocation_stream_id, old_stream_id);
    assert_eq!(
        rotated_highwater.revocation_stream_generation,
        i64::try_from(old_stream_generation).unwrap() + 1
    );
    assert_eq!(rotated_highwater.last_issued_revocation_sequence, 1);
    let retained_generations = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_revocation_highwaters::table
            .filter(maple_pairing_revocation_highwaters::lookup_digest.eq(&lookup_digest))
            .order(maple_pairing_revocation_highwaters::revocation_stream_generation.asc())
            .load::<MaplePairingRevocationHighwater>(conn)
            .expect("append-only host generations should load")
    };
    assert_eq!(retained_generations.len(), 2);
    assert_eq!(retained_generations[0].revocation_stream_id, old_stream_id);
    assert_eq!(
        retained_generations[0].revocation_stream_generation,
        i64::try_from(old_stream_generation).unwrap()
    );
    assert_eq!(retained_generations[0].last_issued_revocation_sequence, 2);
    assert_eq!(retained_generations[1].id, rotated_highwater.id);

    let new_login = app_state
        .authenticate_user(Some(email), None, new_password.to_string(), project.id)
        .await
        .expect("new password login should not error")
        .expect("new password should authenticate");
    let new_authorization = MaplePairingAuthorization {
        user_id: user.uuid,
        project_id: project.id,
        auth_credential_kind: CredentialKind::Password.as_str().to_string(),
        auth_binding: new_login.auth_context.auth_binding,
        enclave_key: app_state.enclave_key.clone(),
    };
    let controller_after_reset = build_test_maple_device_registration_with_id(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        controller.registration_id,
        controller.device_id,
        controller.installation_id,
        new_login.auth_context.auth_binding,
        2,
        121,
    );
    let host_after_reset = build_test_maple_device_registration_with_id(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        host.registration_id,
        host.device_id,
        host.installation_id,
        new_login.auth_context.auth_binding,
        2,
        125,
    );
    let controller_reset_receipt =
        register_test_maple_device(&app_state, controller_after_reset.clone())
            .expect("same controller installation should enter reset-clear recovery");
    let host_reset_receipt = register_test_maple_device(&app_state, host_after_reset.clone())
        .expect("same host installation should enter reset-clear recovery");
    let (reset_stream_id, reset_stream_generation) =
        current_maple_revocation_stream(&app_state, &new_authorization, host.registration_id);
    assert_eq!(reset_stream_id, rotated_highwater.revocation_stream_id);
    assert_eq!(
        reset_stream_generation,
        u64::try_from(rotated_highwater.revocation_stream_generation).unwrap()
    );
    let discovered_after_reset = app_state
        .db
        .list_maple_pairing_revocations(
            new_authorization.clone(),
            host.registration_id,
            Uuid::nil(),
            0,
            0,
            10,
        )
        .expect("explicit discovery should authenticate the reset namespace");
    assert!(discovered_after_reset.events.is_empty());
    assert_eq!(discovered_after_reset.last_issued_revocation_sequence, 1);
    assert_eq!(discovered_after_reset.last_acked_revocation_sequence, 0);
    assert!(discovered_after_reset.reset_clear_sync_payload.is_some());
    assert!(matches!(
        app_state.db.list_maple_pairing_revocations(
            new_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        ),
        Err(DBError::MaplePairingConflict)
    ));

    let shared_terminal_operation_id = Uuid::new_v4();
    let (_controller_ack, _controller_ack_receipt) = ack_test_reset_clear_registration(
        &app_state,
        &new_authorization,
        &controller_reset_receipt,
        shared_terminal_operation_id,
        151,
    );
    let (host_ack, host_ack_receipt) = ack_test_reset_clear_registration(
        &app_state,
        &new_authorization,
        &host_reset_receipt,
        shared_terminal_operation_id,
        153,
    );
    let host_ack_replay = app_state
        .db
        .ack_maple_pairing_revocation(host_ack.clone())
        .expect("exact terminal ACK retry must replay after live-host deletion");
    assert_eq!(host_ack_replay.operation_id, host_ack_receipt.operation_id);
    assert_eq!(host_ack_replay.pair_id, host_ack_receipt.pair_id);
    assert_eq!(host_ack_replay.receipt_enc, host_ack_receipt.receipt_enc);
    let mut changed_host_ack = host_ack;
    changed_host_ack.request_mac = vec![154; 32];
    assert!(matches!(
        app_state.db.ack_maple_pairing_revocation(changed_host_ack),
        Err(DBError::MaplePairingConflict)
    ));

    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
    let retired_authority_scope_digest =
        assert_reset_retirement_rows(&app_state, user.uuid, project.id, 2, 4);

    let replayed_host_registration =
        register_test_maple_device(&app_state, host_after_reset.clone())
            .expect("exact registration retry must replay from the terminal tombstone");
    assert_eq!(replayed_host_registration, host_reset_receipt);
    let mut changed_host_registration = host_after_reset.clone();
    changed_host_registration.request_mac = vec![155; 32];
    assert!(matches!(
        register_test_maple_device(&app_state, changed_host_registration),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));
    let mut reused_retired_host = host_after_reset;
    reused_retired_host.operation_id = Uuid::new_v4();
    reused_retired_host.request_mac = vec![156; 32];
    assert!(matches!(
        register_test_maple_device(&app_state, reused_retired_host),
        Err(DBError::MapleInstallationRetired)
    ));

    assert!(matches!(
        app_state.db.list_maple_pairing_revocations(
            new_authorization.clone(),
            host.registration_id,
            old_stream_id,
            old_stream_generation,
            1,
            10,
        ),
        Err(DBError::MaplePairingNotFound)
    ));

    let fresh_controller = build_test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        new_login.auth_context.auth_binding,
        2,
        157,
    );
    let fresh_host = build_test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        new_login.auth_context.auth_binding,
        2,
        158,
    );
    register_test_maple_device(&app_state, fresh_controller.clone())
        .expect("fresh controller lineage should register after reset");
    register_test_maple_device(&app_state, fresh_host.clone())
        .expect("fresh host lineage should register after reset");
    let (new_stream_id, new_stream_generation) =
        current_maple_revocation_stream(&app_state, &new_authorization, fresh_host.registration_id);

    let (new_request_id, new_pair_id, new_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &new_authorization,
        fresh_controller.registration_id,
        fresh_host.registration_id,
        159,
    );
    let new_event_id = Uuid::new_v4();
    let (_, new_event_digest) = revoke_test_maple_pairing(
        &app_state,
        &new_authorization,
        fresh_controller.registration_id,
        MaplePairingRole::Controller,
        new_request_id,
        new_pair_id,
        3,
        new_incarnation,
        new_stream_id,
        new_stream_generation,
        new_event_id,
        &|context| {
            assert_eq!(context.revocation_stream_id, new_stream_id);
            assert_eq!(context.revocation_stream_generation, new_stream_generation);
            assert_eq!(
                context.issuer_sequence, 1,
                "the fresh namespace must be gap-free after reset"
            );
        },
        &|_| {},
    )
    .expect("the first post-reset revocation should allocate sequence one");
    let new_page = app_state
        .db
        .list_maple_pairing_revocations(
            new_authorization.clone(),
            fresh_host.registration_id,
            new_stream_id,
            new_stream_generation,
            0,
            10,
        )
        .expect("the new stream should page from its own zero cursor");
    assert_eq!(new_page.events.len(), 1);
    assert_eq!(new_page.events[0].event.uuid, new_event_id);
    assert_eq!(new_page.events[0].event.issuer_sequence, 1);
    assert_eq!(new_page.last_issued_revocation_sequence, 1);
    assert_eq!(new_page.last_acked_revocation_sequence, 0);

    assert!(matches!(
        app_state.db.delete_user(&user, &app_state.enclave_key),
        Err(DBError::MaplePairingAuthorityDeletionBlocked)
    ));
    app_state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: new_authorization,
            operation_id: Uuid::new_v4(),
            request_mac: vec![169; 32],
            host_registration_id: fresh_host.registration_id,
            revocation_stream_id: new_stream_id,
            revocation_stream_generation: new_stream_generation,
            event_id: new_event_id,
            issuer_sequence: 1,
            event_digest: new_event_digest,
            expected_previous_issuer_sequence: 0,
            checkpoint_issuer_key_id: "maple-test-issuer-current".to_string(),
            receipt_version: 1,
            receipt_enc: vec![170, 171],
            accepted_at: Utc::now(),
        })
        .expect("terminal revocation ACK should make final account deletion safe");

    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("test account cleanup should remove live pairing state");
    assert_maple_pairing_row_counts(&app_state, user.uuid, 0, 0, 0, 0, 0);
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
    let remaining_retained_rows = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let account_heads = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .filter(maple_pairing_authority_account_heads::project_id.eq(project.id))
            .count()
            .get_result::<i64>(conn)
            .expect("remaining account-head count should query");
        let highwaters = maple_pairing_revocation_highwaters::table
            .filter(
                maple_pairing_revocation_highwaters::authority_scope_digest
                    .eq(&retired_authority_scope_digest),
            )
            .count()
            .get_result::<i64>(conn)
            .expect("remaining high-water count should query");
        let obligations = maple_pairing_reset_clear_obligations::table
            .filter(
                maple_pairing_reset_clear_obligations::authority_scope_digest
                    .eq(&retired_authority_scope_digest),
            )
            .count()
            .get_result::<i64>(conn)
            .expect("remaining reset-clear obligation count should query");
        let admissions = maple_pairing_reset_clear_admissions::table
            .filter(
                maple_pairing_reset_clear_admissions::authority_scope_digest
                    .eq(&retired_authority_scope_digest),
            )
            .count()
            .get_result::<i64>(conn)
            .expect("remaining reset-clear admission count should query");
        let registration_tombstones = maple_pairing_registration_operation_tombstones::table
            .filter(
                maple_pairing_registration_operation_tombstones::authority_scope_digest
                    .eq(&retired_authority_scope_digest),
            )
            .count()
            .get_result::<i64>(conn)
            .expect("remaining registration tombstone count should query");
        let retirements = maple_pairing_installation_retirements::table
            .filter(
                maple_pairing_installation_retirements::authority_scope_digest
                    .eq(&retired_authority_scope_digest),
            )
            .count()
            .get_result::<i64>(conn)
            .expect("remaining installation-retirement count should query");
        (
            account_heads,
            highwaters,
            obligations,
            admissions,
            registration_tombstones,
            retirements,
        )
    };
    assert_eq!(
        remaining_retained_rows,
        (0, 0, 0, 0, 0, 0),
        "verified-clean final account deletion must consume every scoped retained authority row"
    );
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_project_deletion_proves_all_accounts_before_consuming_any() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let org = app_state
        .db
        .create_org(
            NewOrg::new(format!("maple-project-delete-org-{marker}")),
            &app_state.enclave_key,
        )
        .expect("test org should create with its authority head");
    let project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-project-delete-{marker}")),
            &app_state.enclave_key,
        )
        .expect("test project should create with its authority head");
    let first_user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("first project user should create");
    let second_user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("second project user should create");
    let (expiring_user, blocking_user) = if first_user.uuid < second_user.uuid {
        (&first_user, &second_user)
    } else {
        (&second_user, &first_user)
    };
    let due_pending =
        create_due_pending_test_pairing_for_user(&app_state, expiring_user, project.id, 170);
    let active = create_active_test_pairing_for_user(&app_state, blocking_user, project.id, 180);

    assert!(matches!(
        app_state
            .db
            .delete_org_project(&project, &app_state.enclave_key),
        Err(DBError::MaplePairingAuthorityDeletionBlocked)
    ));
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            users::table
                .filter(users::project_id.eq(project.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            2
        );
        assert_eq!(
            maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::project_id.eq(project.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            2,
            "the earlier expiring account head must survive a later blocked account"
        );
        assert_eq!(
            maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::user_id.eq(expiring_user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            1
        );
        assert_eq!(
            maple_pairings::table
                .filter(maple_pairings::uuid.eq(due_pending.pair_id))
                .select(maple_pairings::state)
                .first::<i16>(conn)
                .unwrap(),
            MaplePairingState::Pending.as_db(),
            "expiry projected for an earlier account must roll back when a later account blocks deletion"
        );
    }

    revoke_and_ack_active_test_pairing(&app_state, &active, 190);
    app_state
        .db
        .delete_org_project(&project, &app_state.enclave_key)
        .expect("terminally acknowledged project authority should delete");
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            users::table
                .filter(users::project_id.eq(project.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::project_id.eq(project.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_pairing_authority_project_heads::table
                .filter(maple_pairing_authority_project_heads::project_id.eq(project.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
    }
    app_state
        .db
        .delete_org(&org, &app_state.enclave_key)
        .expect("empty test org should delete");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_org_deletion_preflights_the_complete_project_subtree() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let org = app_state
        .db
        .create_org(
            NewOrg::new(format!("maple-org-delete-{marker}")),
            &app_state.enclave_key,
        )
        .expect("test org should create");
    let first_project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-org-delete-a-{marker}")),
            &app_state.enclave_key,
        )
        .expect("first test project should create");
    let second_project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-org-delete-b-{marker}")),
            &app_state.enclave_key,
        )
        .expect("second test project should create");
    let expiring_user = app_state
        .db
        .create_user(
            NewUser::new(None, None, first_project.id),
            &app_state.enclave_key,
        )
        .expect("expiring project user should create");
    let blocking_user = app_state
        .db
        .create_user(
            NewUser::new(None, None, second_project.id),
            &app_state.enclave_key,
        )
        .expect("blocking project user should create");
    let due_pending =
        create_due_pending_test_pairing_for_user(&app_state, &expiring_user, first_project.id, 195);
    let active =
        create_active_test_pairing_for_user(&app_state, &blocking_user, second_project.id, 200);

    assert!(matches!(
        app_state.db.delete_org(&org, &app_state.enclave_key),
        Err(DBError::MaplePairingAuthorityDeletionBlocked)
    ));
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            org_projects::table
                .filter(org_projects::org_id.eq(org.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            2
        );
        assert_eq!(
            maple_pairing_authority_project_heads::table
                .filter(maple_pairing_authority_project_heads::org_id.eq(org.id))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            2,
            "the first project head must survive a later blocked project"
        );
        assert_eq!(
            users::table
                .filter(users::uuid.eq(expiring_user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            1
        );
        assert_eq!(
            maple_pairings::table
                .filter(maple_pairings::uuid.eq(due_pending.pair_id))
                .select(maple_pairings::state)
                .first::<i16>(conn)
                .unwrap(),
            MaplePairingState::Pending.as_db(),
            "first-project expiry must roll back when a later project blocks org deletion"
        );
    }

    revoke_and_ack_active_test_pairing(&app_state, &active, 210);
    app_state
        .db
        .delete_org(&org, &app_state.enclave_key)
        .expect("terminally acknowledged org authority should delete");
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    assert_eq!(
        orgs::table
            .filter(orgs::id.eq(org.id))
            .count()
            .get_result::<i64>(conn)
            .unwrap(),
        0
    );
    assert_eq!(
        maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org.id))
            .count()
            .get_result::<i64>(conn)
            .unwrap(),
        0
    );
    assert_eq!(
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::org_id.eq(org.id))
            .count()
            .get_result::<i64>(conn)
            .unwrap(),
        0
    );
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_tampered_global_root_rejects_before_any_scoped_access_or_mutation() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("global-root fence test user should create");
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        213,
    );
    let (head_revision_before, head_digest_before) = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select((
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::authority_inventory_digest,
            ))
            .first::<(i64, Vec<u8>)>(conn)
            .unwrap()
    };
    tamper_maple_pairing_authority_global_root_for_test(&*app_state.db, &app_state.enclave_key)
        .expect("test should commit a MAC-invalid global root revision");
    let scoped_accesses = observe_maple_pairing_authority_scoped_access_for_test(user.uuid);
    let rejected = register_test_maple_device(&app_state, registration.clone());
    clear_maple_pairing_authority_scoped_access_observer_for_test(user.uuid);
    assert!(matches!(
        rejected,
        Err(DBError::MaplePairingAuthorityCorrupt)
    ));
    assert_eq!(
        scoped_accesses.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "invalid global-root MAC must fail before the account head is read"
    );
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
                .select((
                    maple_pairing_authority_account_heads::revision,
                    maple_pairing_authority_account_heads::authority_inventory_digest,
                ))
                .first::<(i64, Vec<u8>)>(conn)
                .unwrap(),
            (head_revision_before, head_digest_before),
            "rejected request must not advance or rewrite the scoped head"
        );
        assert_eq!(
            maple_devices::table
                .filter(maple_devices::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_device_registration_operations::table
                .filter(maple_device_registration_operations::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
    }

    restore_maple_pairing_authority_global_root_for_test(&*app_state.db, &app_state.enclave_key)
        .expect("test should restore a valid authenticated global root");
    register_test_maple_device(&app_state, registration)
        .expect("same unconsumed operation should succeed after fixture repair");
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("global-root fence test user should delete cleanly");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_contended_authority_waiter_aborts_stale_snapshot_then_retries_fresh() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("stale-waiter test user should create");
    let operation_id = Uuid::new_v4();
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        operation_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        214,
    );
    let authorization = maple_device_list_authorization(&app_state, user.uuid, project.id);
    let (holder_reached, release_holder) =
        pause_next_maple_device_registration_before_commit_for_test(operation_id);
    let holder_state = app_state.clone();
    let holder =
        std::thread::spawn(move || register_test_maple_device(&holder_state, registration));
    holder_reached
        .recv_timeout(Duration::from_secs(10))
        .expect("registration holder should pause after staging authenticated heads");

    // The waiter's first failed pg_try SELECT fixes its SERIALIZABLE snapshot
    // before the holder commits. Observe that attempt before releasing the
    // holder so the global-root FOR UPDATE freshness fence deterministically
    // returns 40001/Busy instead of projecting the stale empty device list.
    let contention_observed = observe_next_maple_pairing_authority_lock_contention_for_test();
    let waiter_db = app_state.db.clone();
    let waiter_authorization = authorization.clone();
    let waiter = std::thread::spawn(move || {
        list_test_maple_devices(waiter_db.as_ref(), waiter_authorization, 32, None)
    });
    if contention_observed
        .recv_timeout(Duration::from_secs(10))
        .is_err()
    {
        let _ = release_holder.send(());
        let _ = holder.join();
        let _ = waiter.join();
        panic!("read-only waiter should contend before holder commit");
    }
    release_holder
        .send(())
        .expect("registration holder should still await release");
    let receipt = holder
        .join()
        .expect("registration holder thread should not panic")
        .expect("registration holder should commit");
    let stale_waiter = waiter
        .join()
        .expect("read-only waiter thread should not panic");
    assert!(matches!(
        stale_waiter,
        Err(DBError::MaplePairingAuthorityBusy)
    ));

    let fresh = list_test_maple_devices(app_state.db.as_ref(), authorization, 32, None)
        .expect("retry should acquire a fresh authority snapshot");
    assert_eq!(fresh.len(), 1);
    assert_eq!(fresh[0].uuid, receipt.registration_id);
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("stale-waiter test user should delete cleanly");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_out_of_band_leaf_write_aborts_verified_authority_commit() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let org = app_state
        .db
        .create_org(
            NewOrg::new(format!("maple-ssi-org-{marker}")),
            &app_state.enclave_key,
        )
        .expect("SSI test org should create");
    let project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-ssi-project-{marker}")),
            &app_state.enclave_key,
        )
        .expect("SSI test project should create");
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("SSI test user should create");
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        215,
    );
    let registered = register_test_maple_device(&app_state, registration)
        .expect("SSI test device should register");
    let authorization = maple_pairing_authorization(&app_state, user.uuid, project.id);
    let (device_id, original_payload, original_record_mac, head_revision_before) = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let device = maple_devices::table
            .filter(maple_devices::uuid.eq(registered.registration_id))
            .select((
                maple_devices::id,
                maple_devices::payload_enc,
                maple_devices::record_mac,
            ))
            .first::<(i64, Vec<u8>, Vec<u8>)>(conn)
            .unwrap();
        let revision = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select(maple_pairing_authority_account_heads::revision)
            .first::<i64>(conn)
            .unwrap();
        (device.0, device.1, device.2, revision)
    };
    let hostile_payload = vec![216, 217, 218];
    let writer_db = app_state.db.clone();
    let race =
        run_maple_pairing_authority_ssi_race_for_test(&*app_state.db, &authorization, move || {
            let conn = &mut writer_db
                .get_pool()
                .get()
                .map_err(|_| DBError::ConnectionError)?;
            conn.transaction::<(), diesel::result::Error, _>(|tx| {
                diesel::sql_query("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE").execute(tx)?;
                // T2 reads the head T1 will write and then writes the leaf T1
                // already read, creating a deterministic SSI dependency cycle.
                let _ = maple_pairing_authority_account_heads::table
                    .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
                    .select(maple_pairing_authority_account_heads::revision)
                    .first::<i64>(tx)?;
                let changed =
                    diesel::update(maple_devices::table.filter(maple_devices::id.eq(device_id)))
                        .set(maple_devices::payload_enc.eq(&hostile_payload))
                        .execute(tx)?;
                assert_eq!(changed, 1);
                Ok(())
            })
            .map_err(DBError::from)
        });
    assert!(matches!(race, Err(DBError::MaplePairingAuthorityBusy)));

    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            maple_pairing_authority_account_heads::table
                .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
                .select(maple_pairing_authority_account_heads::revision)
                .first::<i64>(conn)
                .unwrap(),
            head_revision_before,
            "the serializable authority transaction must not publish a head after SSI abort"
        );
        assert_eq!(
            maple_devices::table
                .filter(maple_devices::id.eq(device_id))
                .select(maple_devices::payload_enc)
                .first::<Vec<u8>>(conn)
                .unwrap(),
            vec![216, 217, 218],
            "the hostile out-of-band transaction should remain committed"
        );
    }
    assert!(matches!(
        list_test_maple_devices(
            app_state.db.as_ref(),
            MapleDeviceListAuthorization {
                user_id: authorization.user_id,
                project_id: authorization.project_id,
                auth_credential_kind: authorization.auth_credential_kind.clone(),
                auth_binding: authorization.auth_binding,
                enclave_key: authorization.enclave_key.clone(),
            },
            32,
            None,
        ),
        Err(DBError::MaplePairingAuthorityCorrupt)
    ));

    // Restore the exact authenticated leaf so this disposable test can use the
    // normal verified-clean deletion path for cleanup.
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(maple_devices::table.filter(maple_devices::id.eq(device_id)))
            .set((
                maple_devices::payload_enc.eq(original_payload),
                maple_devices::record_mac.eq(original_record_mac),
            ))
            .execute(conn)
            .unwrap();
    }
    list_test_maple_devices(
        app_state.db.as_ref(),
        MapleDeviceListAuthorization {
            user_id: authorization.user_id,
            project_id: authorization.project_id,
            auth_credential_kind: authorization.auth_credential_kind.clone(),
            auth_binding: authorization.auth_binding,
            enclave_key: authorization.enclave_key.clone(),
        },
        32,
        None,
    )
    .expect("restored authority leaf should authenticate");
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("restored SSI test user should delete");
    app_state
        .db
        .delete_org_project(&project, &app_state.enclave_key)
        .expect("empty SSI test project should delete");
    app_state
        .db
        .delete_org(&org, &app_state.enclave_key)
        .expect("empty SSI test org should delete");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_failed_precommit_pair_create_publishes_one_higher_incarnation_only_on_retry() {
    use subtle::ConstantTimeEq;

    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let org = app_state
        .db
        .create_org(
            NewOrg::new(format!("maple-create-abort-org-{marker}")),
            &app_state.enclave_key,
        )
        .expect("create-abort test org should create");
    let project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-create-abort-project-{marker}")),
            &app_state.enclave_key,
        )
        .expect("create-abort test project should create");
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("create-abort test user should create");
    let controller = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        222,
    );
    let host = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        223,
    );
    register_test_maple_device(&app_state, controller.clone())
        .expect("create-abort controller should register");
    register_test_maple_device(&app_state, host.clone())
        .expect("create-abort host should register");
    let authorization = maple_pairing_authorization(&app_state, user.uuid, project.id);
    let devices = list_test_maple_devices(
        app_state.db.as_ref(),
        MapleDeviceListAuthorization {
            user_id: authorization.user_id,
            project_id: authorization.project_id,
            auth_credential_kind: authorization.auth_credential_kind.clone(),
            auth_binding: authorization.auth_binding,
            enclave_key: authorization.enclave_key.clone(),
        },
        32,
        None,
    )
    .expect("create-abort participants should load");
    let controller_epoch = devices
        .iter()
        .find(|device| device.uuid == controller.registration_id)
        .map(|device| u64::try_from(device.endpoint_epoch).unwrap())
        .unwrap();
    let host_epoch = devices
        .iter()
        .find(|device| device.uuid == host.registration_id)
        .map(|device| u64::try_from(device.endpoint_epoch).unwrap())
        .unwrap();
    let operation_id = Uuid::new_v4();
    let head_before = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select((
                maple_pairing_authority_account_heads::authority_inventory_digest,
                maple_pairing_authority_account_heads::authority_row_count,
                maple_pairing_authority_account_heads::lineage_count,
                maple_pairing_authority_account_heads::pairing_count,
                maple_pairing_authority_account_heads::pairing_operation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, i64, i64, i64, Vec<u8>)>(conn)
            .unwrap()
    };
    let ancestor_heads_before = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let project_head = maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::project_id.eq(project.id))
            .select((
                maple_pairing_authority_project_heads::account_inventory_digest,
                maple_pairing_authority_project_heads::account_count,
                maple_pairing_authority_project_heads::revision,
                maple_pairing_authority_project_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, Vec<u8>)>(conn)
            .unwrap();
        let org_head = maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org.id))
            .select((
                maple_pairing_authority_org_heads::project_inventory_digest,
                maple_pairing_authority_org_heads::project_count,
                maple_pairing_authority_org_heads::revision,
                maple_pairing_authority_org_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, Vec<u8>)>(conn)
            .unwrap();
        let global_head = maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true))
            .select((
                maple_pairing_authority_global_heads::org_inventory_digest,
                maple_pairing_authority_global_heads::org_count,
                maple_pairing_authority_global_heads::revision,
                maple_pairing_authority_global_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, Option<Vec<u8>>)>(conn)
            .unwrap();
        (project_head, org_head, global_head)
    };
    // PostgreSQL nextval is nontransactional: the aborted signed candidate
    // intentionally burns the incarnation observed by the pure materializer.
    // Only the later committed/published CREATE establishes authority, and its
    // exact replay must neither reserve nor sign again.
    let request = test_maple_pairing_create_request(
        &app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        operation_id,
        Some(controller_epoch),
        Some(host_epoch),
        232,
    );
    let first_incarnation = std::cell::Cell::new(0_u64);
    let signed_artifact_count = std::cell::Cell::new(0_u8);
    fail_next_maple_pairing_create_before_commit_for_test(operation_id);
    let first_attempt = create_test_maple_pairing_observed(&app_state, request.clone(), &|value| {
        first_incarnation.set(value);
        signed_artifact_count.set(signed_artifact_count.get() + 1);
    });
    assert!(matches!(
        first_attempt,
        Err(DBError::MaplePairingAuthorityBusy)
    ));
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        assert_eq!(
            maple_pairings::table
                .filter(maple_pairings::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_pairing_operations::table
                .filter(maple_pairing_operations::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_pairing_lineages::table
                .filter(maple_pairing_lineages::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        let head_after_abort = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select((
                maple_pairing_authority_account_heads::authority_inventory_digest,
                maple_pairing_authority_account_heads::authority_row_count,
                maple_pairing_authority_account_heads::lineage_count,
                maple_pairing_authority_account_heads::pairing_count,
                maple_pairing_authority_account_heads::pairing_operation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, i64, i64, i64, Vec<u8>)>(conn)
            .unwrap();
        assert!(bool::from(
            head_before
                .0
                .as_slice()
                .ct_eq(head_after_abort.0.as_slice())
        ));
        assert_eq!(head_before.1, head_after_abort.1);
        assert_eq!(head_before.2, head_after_abort.2);
        assert_eq!(head_before.3, head_after_abort.3);
        assert_eq!(head_before.4, head_after_abort.4);
        assert_eq!(head_before.5, head_after_abort.5);
        assert!(bool::from(
            head_before
                .6
                .as_slice()
                .ct_eq(head_after_abort.6.as_slice())
        ));
        let ancestor_heads_after = (
            maple_pairing_authority_project_heads::table
                .filter(maple_pairing_authority_project_heads::project_id.eq(project.id))
                .select((
                    maple_pairing_authority_project_heads::account_inventory_digest,
                    maple_pairing_authority_project_heads::account_count,
                    maple_pairing_authority_project_heads::revision,
                    maple_pairing_authority_project_heads::record_mac,
                ))
                .first::<(Vec<u8>, i64, i64, Vec<u8>)>(conn)
                .unwrap(),
            maple_pairing_authority_org_heads::table
                .filter(maple_pairing_authority_org_heads::org_id.eq(org.id))
                .select((
                    maple_pairing_authority_org_heads::project_inventory_digest,
                    maple_pairing_authority_org_heads::project_count,
                    maple_pairing_authority_org_heads::revision,
                    maple_pairing_authority_org_heads::record_mac,
                ))
                .first::<(Vec<u8>, i64, i64, Vec<u8>)>(conn)
                .unwrap(),
            maple_pairing_authority_global_heads::table
                .filter(maple_pairing_authority_global_heads::singleton.eq(true))
                .select((
                    maple_pairing_authority_global_heads::org_inventory_digest,
                    maple_pairing_authority_global_heads::org_count,
                    maple_pairing_authority_global_heads::revision,
                    maple_pairing_authority_global_heads::record_mac,
                ))
                .first::<(Vec<u8>, i64, i64, Option<Vec<u8>>)>(conn)
                .unwrap(),
        );
        assert_eq!(
            ancestor_heads_after, ancestor_heads_before,
            "aborted staged CREATE must not advance or rewrite project, org, or global heads"
        );
    }

    let second_incarnation = std::cell::Cell::new(0_u64);
    let committed = create_test_maple_pairing_observed(&app_state, request.clone(), &|value| {
        second_incarnation.set(value);
        signed_artifact_count.set(signed_artifact_count.get() + 1);
    })
    .expect("same operation retry should commit exactly once");
    assert!(second_incarnation.get() > first_incarnation.get());
    let replayed = create_test_maple_pairing_observed(&app_state, request, &|_| {
        signed_artifact_count.set(signed_artifact_count.get() + 1);
    })
    .expect("exact committed retry should replay");
    assert_eq!(committed.operation_id, replayed.operation_id);
    assert_eq!(committed.pair_id, replayed.pair_id);
    assert_eq!(committed.pairing_revision, replayed.pairing_revision);
    assert_eq!(committed.receipt_version, replayed.receipt_version);
    assert_eq!(committed.receipt_enc, replayed.receipt_enc);
    assert_eq!(committed.accepted_at, replayed.accepted_at);
    assert_eq!(
        signed_artifact_count.get(),
        2,
        "the failed unpublished attempt and the higher committed retry sign once each; DB replay signs nothing"
    );
    assert_eq!(pairing_row_count(&app_state, user.uuid), 1);
    make_maple_pairing_pending_due_for_test(&*app_state.db, &authorization, committed.pair_id)
        .expect("committed pending test pair should become deletion-safe");
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("expired create-abort test user should delete");
    app_state
        .db
        .delete_org_project(&project, &app_state.enclave_key)
        .expect("empty create-abort test project should delete");
    app_state
        .db
        .delete_org(&org, &app_state.enclave_key)
        .expect("empty create-abort test org should delete");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_highwater_group_capacity_rejection_is_atomic_at_exact_limit() {
    use subtle::ConstantTimeEq;

    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let marker = Uuid::new_v4();
    let email = format!("maple-capacity-{marker}@example.com");
    let old_password = test_credential("maple-capacity-old-password");
    let org = app_state
        .db
        .create_org(
            NewOrg::new(format!("maple-capacity-org-{marker}")),
            &app_state.enclave_key,
        )
        .expect("capacity test org should create");
    let project = app_state
        .db
        .create_org_project(
            NewOrgProject::new(org.id, format!("maple-capacity-project-{marker}")),
            &app_state.enclave_key,
        )
        .expect("capacity test project should create");
    let user =
        create_password_wrapped_user(&app_state, project.id, email.clone(), old_password).await;
    let _retained_installation_id = match seed_maple_pairing_highwater_group_capacity_for_test(
        &*app_state.db,
        &app_state.enclave_key,
        user.uuid,
        project.id,
    ) {
        Ok(installation_id) => installation_id,
        Err(error) => {
            app_state
                .db
                .delete_user(&user, &app_state.enclave_key)
                .expect("failed group-capacity fixture account should clean up");
            app_state
                .db
                .delete_org_project(&project, &app_state.enclave_key)
                .expect("failed group-capacity fixture project should clean up");
            app_state
                .db
                .delete_org(&org, &app_state.enclave_key)
                .expect("failed group-capacity fixture org should clean up");
            panic!("authenticated fixture should reach the exact highwater group limit: {error:?}");
        }
    };

    let fresh_registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        220,
    );
    let (head_before, credential_before) = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let head = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select((
                maple_pairing_authority_account_heads::authority_inventory_digest,
                maple_pairing_authority_account_heads::authority_row_count,
                maple_pairing_authority_account_heads::device_count,
                maple_pairing_authority_account_heads::highwater_installation_group_count,
                maple_pairing_authority_account_heads::highwater_generation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, i64, i64, i64, Vec<u8>)>(conn)
            .unwrap();
        let credential = users::table
            .filter(users::uuid.eq(user.uuid))
            .select((users::password_enc, users::updated_at))
            .first::<(Option<Vec<u8>>, chrono::DateTime<Utc>)>(conn)
            .unwrap();
        (head, credential)
    };
    assert_eq!(head_before.1, 1024);
    assert_eq!(head_before.2, 0);
    assert_eq!(head_before.3, 1024);
    assert_eq!(head_before.4, 1024);

    assert!(matches!(
        register_test_maple_device(&app_state, fresh_registration),
        Err(DBError::MaplePairingAuthorityCapacityExceeded)
    ));

    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let head_after = maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .select((
                maple_pairing_authority_account_heads::authority_inventory_digest,
                maple_pairing_authority_account_heads::authority_row_count,
                maple_pairing_authority_account_heads::device_count,
                maple_pairing_authority_account_heads::highwater_installation_group_count,
                maple_pairing_authority_account_heads::highwater_generation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first::<(Vec<u8>, i64, i64, i64, i64, i64, Vec<u8>)>(conn)
            .unwrap();
        let credential_after = users::table
            .filter(users::uuid.eq(user.uuid))
            .select((users::password_enc, users::updated_at))
            .first::<(Option<Vec<u8>>, chrono::DateTime<Utc>)>(conn)
            .unwrap();
        assert!(bool::from(
            head_before.0.as_slice().ct_eq(head_after.0.as_slice())
        ));
        assert_eq!(head_before.1, head_after.1);
        assert_eq!(head_before.2, head_after.2);
        assert_eq!(head_before.3, head_after.3);
        assert_eq!(head_before.4, head_after.4);
        assert_eq!(head_before.5, head_after.5);
        assert!(bool::from(
            head_before.6.as_slice().ct_eq(head_after.6.as_slice())
        ));
        assert_eq!(credential_before, credential_after);
        assert_eq!(
            maple_devices::table
                .filter(maple_devices::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(
            maple_device_registration_operations::table
                .filter(maple_device_registration_operations::user_id.eq(user.uuid))
                .count()
                .get_result::<i64>(conn)
                .unwrap(),
            0
        );
        assert_eq!(head_after.4, 1024);
    }

    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("verified-clean capacity fixture account should delete");
    app_state
        .db
        .delete_org_project(&project, &app_state.enclave_key)
        .expect("empty capacity fixture project should delete");
    app_state
        .db
        .delete_org(&org, &app_state.enclave_key)
        .expect("empty capacity fixture org should delete");
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
    let load_authority_head = || {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(user.uuid))
            .filter(maple_pairing_authority_account_heads::project_id.eq(project.id))
            .first::<MaplePairingAuthorityAccountHead>(conn)
            .expect("test authority account head should load")
    };
    let authority_head_before = load_authority_head();

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

    let authority_head_after = load_authority_head();
    assert_eq!(
        authority_head_after.security_epoch,
        authority_head_before
            .security_epoch
            .checked_add(1)
            .expect("test security epoch should advance"),
        "a destructive reset must advance the authenticated security epoch exactly once"
    );
    assert_eq!(
        authority_head_after.revision,
        authority_head_before
            .revision
            .checked_add(1)
            .expect("test authority revision should advance"),
        "a destructive reset must commit the account authority head exactly once"
    );
    assert_ne!(
        authority_head_after.authority_inventory_digest,
        authority_head_before.authority_inventory_digest,
        "even an empty authority inventory must bind the new security epoch"
    );
    assert!(
        [
            authority_head_after.authority_row_count,
            authority_head_after.device_count,
            authority_head_after.device_operation_count,
            authority_head_after.lineage_count,
            authority_head_after.pairing_count,
            authority_head_after.pairing_operation_count,
            authority_head_after.host_state_count,
            authority_head_after.revocation_event_count,
            authority_head_after.highwater_installation_group_count,
            authority_head_after.highwater_generation_count,
            authority_head_after.registration_operation_tombstone_count,
            authority_head_after.installation_retirement_count,
            authority_head_after.reset_clear_obligation_count,
            authority_head_after.reset_clear_admission_count,
        ]
        .into_iter()
        .all(|count| count == 0),
        "a no-device destructive reset must retain an empty authenticated authority inventory"
    );
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
    assert_maple_pairing_row_counts(&app_state, user.uuid, 0, 0, 0, 0, 0);
    let retained_reset_rows = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let authority_scope_digest = &authority_head_after.authority_scope_digest;
        (
            maple_pairing_revocation_highwaters::table
                .filter(
                    maple_pairing_revocation_highwaters::authority_scope_digest
                        .eq(authority_scope_digest),
                )
                .count()
                .get_result::<i64>(conn)
                .expect("no-device highwater count should query"),
            maple_pairing_reset_clear_obligations::table
                .filter(
                    maple_pairing_reset_clear_obligations::authority_scope_digest
                        .eq(authority_scope_digest),
                )
                .count()
                .get_result::<i64>(conn)
                .expect("no-device reset-clear obligation count should query"),
            maple_pairing_reset_clear_admissions::table
                .filter(
                    maple_pairing_reset_clear_admissions::authority_scope_digest
                        .eq(authority_scope_digest),
                )
                .count()
                .get_result::<i64>(conn)
                .expect("no-device reset-clear admission count should query"),
        )
    };
    assert_eq!(
        retained_reset_rows,
        (0, 0, 0),
        "a no-device destructive reset must create no highwater, obligation, or admission rows"
    );

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
    let authority_head_after_stale_retry = load_authority_head();
    assert_eq!(
        (
            authority_head_after_stale_retry.security_epoch,
            authority_head_after_stale_retry.revision,
            authority_head_after_stale_retry.authority_inventory_digest,
            authority_head_after_stale_retry.record_mac,
        ),
        (
            authority_head_after.security_epoch,
            authority_head_after.revision,
            authority_head_after.authority_inventory_digest.clone(),
            authority_head_after.record_mac.clone(),
        ),
        "a consumed reset request must not advance or rewrite the authority head"
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
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

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_registration_is_idempotent_and_changed_replay_conflicts() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("device registration test user should insert");
    let operation_id = Uuid::new_v4();
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        operation_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        7,
    );

    let first = register_test_maple_device(&app_state, registration.clone())
        .expect("first device registration should succeed");
    let mut retried = registration.clone();
    // The enclave encryption nonce changes on a retry, while the authenticated
    // structured request fingerprint remains identical.
    retried.payload_enc = vec![90, 91, 92];
    let replay = register_test_maple_device(&app_state, retried)
        .expect("exact operation replay should return its original receipt");
    assert_eq!(replay, first);

    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("tamper test database connection should be available");
    assert!(diesel::update(
        maple_devices::table.filter(maple_devices::uuid.eq(first.registration_id)),
    )
    .set(maple_devices::payload_enc.eq(vec![0u8; MAPLE_DEVICE_MAX_ENCRYPTED_PAYLOAD_BYTES + 1]),)
    .execute(conn)
    .is_err());
    let original_operation_revision = maple_device_registration_operations::table
        .filter(maple_device_registration_operations::operation_id.eq(registration.operation_id))
        .select(maple_device_registration_operations::device_revision)
        .first::<i64>(conn)
        .expect("original operation device revision should query");
    diesel::update(
        maple_device_registration_operations::table.filter(
            maple_device_registration_operations::operation_id.eq(registration.operation_id),
        ),
    )
    .set(maple_device_registration_operations::device_revision.eq(99))
    .execute(conn)
    .expect("operation receipt revision tamper should write");
    let operation_revision_tamper = register_test_maple_device(&app_state, registration.clone());
    diesel::update(
        maple_device_registration_operations::table.filter(
            maple_device_registration_operations::operation_id.eq(registration.operation_id),
        ),
    )
    .set(maple_device_registration_operations::device_revision.eq(original_operation_revision))
    .execute(conn)
    .expect("operation receipt revision should restore for cleanup");
    assert!(
        matches!(
            &operation_revision_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "operation revision tamper should fail exhaustive authority validation: {operation_revision_tamper:?}"
    );
    assert_eq!(
        register_test_maple_device(&app_state, registration.clone())
            .expect("restored operation row should replay"),
        first,
    );

    let original_endpoint_epoch = maple_devices::table
        .filter(maple_devices::uuid.eq(first.registration_id))
        .select(maple_devices::endpoint_epoch)
        .first::<i64>(conn)
        .expect("original device endpoint epoch should query");
    diesel::update(maple_devices::table.filter(maple_devices::uuid.eq(first.registration_id)))
        .set(maple_devices::endpoint_epoch.eq(999))
        .execute(conn)
        .expect("clear record field tamper should write");
    let endpoint_epoch_tamper = register_test_maple_device(&app_state, registration.clone());
    diesel::update(maple_devices::table.filter(maple_devices::uuid.eq(first.registration_id)))
        .set(maple_devices::endpoint_epoch.eq(original_endpoint_epoch))
        .execute(conn)
        .expect("device endpoint epoch should restore for cleanup");
    assert!(
        matches!(
            &endpoint_epoch_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "endpoint epoch tamper should fail exhaustive authority validation: {endpoint_epoch_tamper:?}"
    );
    assert_eq!(
        register_test_maple_device(&app_state, registration.clone())
            .expect("restored device row should replay"),
        first,
    );

    let mut changed = registration;
    changed.request_mac = vec![8; 32];
    let conflict = register_test_maple_device(&app_state, changed);
    assert!(matches!(
        conflict,
        Err(DBError::MapleDeviceRegistrationConflict)
    ));
    assert_maple_device_row_counts(&app_state, user.uuid, 1, 1);

    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("account deletion should cascade through Maple device records");
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_registration_rejects_trigger_modified_returning_rows() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("trigger tamper test user should insert");
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("trigger tamper database connection should be available");

    diesel::sql_query("DROP TRIGGER IF EXISTS maple_device_test_mutate_row ON maple_devices")
        .execute(conn)
        .unwrap();
    diesel::sql_query("DROP FUNCTION IF EXISTS maple_device_test_mutate_row()")
        .execute(conn)
        .unwrap();
    diesel::sql_query(
        "CREATE FUNCTION maple_device_test_mutate_row() RETURNS trigger LANGUAGE plpgsql AS $$ \
         BEGIN NEW.endpoint_epoch := NEW.endpoint_epoch + 1; RETURN NEW; END $$",
    )
    .execute(conn)
    .unwrap();
    diesel::sql_query(
        "CREATE TRIGGER maple_device_test_mutate_row BEFORE INSERT ON maple_devices \
         FOR EACH ROW EXECUTE FUNCTION maple_device_test_mutate_row()",
    )
    .execute(conn)
    .unwrap();

    let row_tampered = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        71,
    );
    assert!(matches!(
        register_test_maple_device(&app_state, row_tampered),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);

    diesel::sql_query("DROP TRIGGER maple_device_test_mutate_row ON maple_devices")
        .execute(conn)
        .unwrap();
    diesel::sql_query("DROP FUNCTION maple_device_test_mutate_row()")
        .execute(conn)
        .unwrap();

    diesel::sql_query(
        "DROP TRIGGER IF EXISTS maple_device_test_mutate_operation \
         ON maple_device_registration_operations",
    )
    .execute(conn)
    .unwrap();
    diesel::sql_query("DROP FUNCTION IF EXISTS maple_device_test_mutate_operation()")
        .execute(conn)
        .unwrap();
    diesel::sql_query(
        "CREATE FUNCTION maple_device_test_mutate_operation() RETURNS trigger LANGUAGE plpgsql AS $$ \
         BEGIN NEW.accepted_at := NEW.accepted_at + interval '1 second'; RETURN NEW; END $$",
    )
    .execute(conn)
    .unwrap();
    diesel::sql_query(
        "CREATE TRIGGER maple_device_test_mutate_operation BEFORE INSERT \
         ON maple_device_registration_operations FOR EACH ROW \
         EXECUTE FUNCTION maple_device_test_mutate_operation()",
    )
    .execute(conn)
    .unwrap();

    let operation_tampered = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        72,
    );
    assert!(matches!(
        register_test_maple_device(&app_state, operation_tampered),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);

    diesel::sql_query(
        "DROP TRIGGER maple_device_test_mutate_operation \
         ON maple_device_registration_operations",
    )
    .execute(conn)
    .unwrap();
    diesel::sql_query("DROP FUNCTION maple_device_test_mutate_operation()")
        .execute(conn)
        .unwrap();
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .unwrap();
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_registration_is_create_only_after_initial_acceptance() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("CAS test user should insert");
    let create = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        41,
    );

    let mut invalid_create = create.clone();
    invalid_create.expected_revision = Some(1);
    assert!(matches!(
        register_test_maple_device(&app_state, invalid_create),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));
    let created = register_test_maple_device(&app_state, create.clone())
        .expect("initial device creation should succeed");
    assert_eq!(created.registration_id, create.registration_id);
    assert_eq!(created.revision, 1);

    let mut wrong_target =
        next_maple_device_registration(&create, Uuid::new_v4(), create.endpoint_epoch + 1, 42);
    wrong_target.registration_id = Uuid::new_v4();
    assert!(matches!(
        register_test_maple_device(&app_state, wrong_target),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));

    let mut changed_identity =
        next_maple_device_registration(&create, Uuid::new_v4(), create.endpoint_epoch + 1, 43);
    changed_identity.identity_mac = vec![99; 32];
    assert!(matches!(
        register_test_maple_device(&app_state, changed_identity),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));

    let lower_epoch =
        next_maple_device_registration(&create, Uuid::new_v4(), create.endpoint_epoch - 1, 44);
    assert!(matches!(
        register_test_maple_device(&app_state, lower_epoch),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));

    let mut stale =
        next_maple_device_registration(&create, Uuid::new_v4(), create.endpoint_epoch + 2, 45);
    stale.expected_revision = Some(2);
    stale.revision = 3;
    assert!(matches!(
        register_test_maple_device(&app_state, stale),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));

    let update = next_maple_device_registration(&create, Uuid::new_v4(), create.endpoint_epoch, 46);
    assert!(matches!(
        register_test_maple_device(&app_state, update),
        Err(DBError::MapleDeviceRegistrationConflict)
    ));

    assert_maple_device_row_counts(&app_state, user.uuid, 1, 1);
    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_registration_concurrent_retries_commit_once() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("concurrent registration test user should insert");
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        19,
    );
    let barrier = Arc::new(Barrier::new(8));
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let state = app_state.clone();
        let barrier = barrier.clone();
        let registration = registration.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            tokio::task::spawn_blocking(move || {
                register_test_maple_device_with_bounded_busy_retry(&state, registration)
            })
            .await
            .expect("registration task should join")
            .expect("concurrent exact retry should succeed")
        }));
    }

    let mut receipts = Vec::new();
    for task in tasks {
        receipts.push(task.await.expect("registration coordinator should join"));
    }
    assert!(receipts.windows(2).all(|pair| pair[0] == pair[1]));
    assert_maple_device_row_counts(&app_state, user.uuid, 1, 1);

    let updates = [
        next_maple_device_registration(
            &registration,
            Uuid::new_v4(),
            registration.endpoint_epoch + 1,
            20,
        ),
        next_maple_device_registration(
            &registration,
            Uuid::new_v4(),
            registration.endpoint_epoch + 2,
            21,
        ),
    ];
    let barrier = Arc::new(Barrier::new(2));
    let mut tasks = Vec::new();
    for update in updates {
        let state = app_state.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            tokio::task::spawn_blocking(move || {
                register_test_maple_device_with_bounded_busy_retry(&state, update)
            })
            .await
            .expect("competing CAS task should join")
        }));
    }
    let results = futures::future::join_all(tasks)
        .await
        .into_iter()
        .map(|result| result.expect("CAS coordinator should join"))
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 0);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(DBError::MapleDeviceRegistrationConflict)))
            .count(),
        2
    );
    assert_maple_device_row_counts(&app_state, user.uuid, 1, 1);

    let _ = app_state.db.delete_user(&user, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_quota_is_fail_closed() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let device_quota_user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("device quota test user should insert");
    for marker in 1..=MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT {
        register_test_maple_device(
            &app_state,
            test_maple_device_registration(
                &app_state,
                device_quota_user.uuid,
                project.id,
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                u8::try_from(marker).expect("device quota marker fits in a byte"),
            ),
        )
        .expect("device within account quota should register");
    }
    assert!(matches!(
        register_test_maple_device(
            &app_state,
            test_maple_device_registration(
                &app_state,
                device_quota_user.uuid,
                project.id,
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                200,
            )
        ),
        Err(DBError::MapleDeviceLimitExceeded)
    ));
    assert_maple_device_row_counts(
        &app_state,
        device_quota_user.uuid,
        MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT,
        MAPLE_DEVICE_LIMIT_PER_ACCOUNT_PROJECT,
    );

    let _ = app_state
        .db
        .delete_user(&device_quota_user, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_keyset_pages_are_bounded_and_account_scoped() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let victim = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("paged device owner should insert");
    let other = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("other device owner should insert");

    let mut expected = Vec::new();
    for marker in 0..5u8 {
        let registration = test_maple_device_registration(
            &app_state,
            victim.uuid,
            project.id,
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            marker,
        );
        expected.push(
            register_test_maple_device(&app_state, registration)
                .expect("paged device should register")
                .registration_id,
        );
    }

    // The same device and installation identifiers remain independent under a
    // different authenticated account.
    let victim_first = list_test_maple_devices(
        app_state.db.as_ref(),
        maple_device_list_authorization(&app_state, victim.uuid, project.id),
        1,
        None,
    )
    .expect("victim first row should list")
    .pop()
    .expect("victim should have a device");
    register_test_maple_device(
        &app_state,
        test_maple_device_registration(
            &app_state,
            other.uuid,
            project.id,
            Uuid::new_v4(),
            victim_first.device_id,
            victim_first.installation_id,
            88,
        ),
    )
    .expect("other account may use independently scoped device identifiers");

    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let page = list_test_maple_devices(
            app_state.db.as_ref(),
            maple_device_list_authorization(&app_state, victim.uuid, project.id),
            2,
            cursor,
        )
        .expect("keyset page should load");
        assert!(page.len() <= 2);
        if page.is_empty() {
            break;
        }
        cursor = page.last().map(|row| MapleDeviceListCursor {
            registration_id: row.uuid,
        });
        seen.extend(page.into_iter().map(|row| row.uuid));
    }
    seen.sort_unstable();
    expected.sort_unstable();
    assert_eq!(seen, expected);
    assert_eq!(
        list_test_maple_devices(
            app_state.db.as_ref(),
            maple_device_list_authorization(&app_state, other.uuid, project.id),
            10,
            None,
        )
        .expect("other account device list should load")
        .len(),
        1
    );
    // Cursors are immutable tuples authenticated by the encrypted API layer;
    // the DB applies the tuple only after account/project scoping, so even a
    // tuple copied from another account cannot disclose its rows.
    let victim_cursor = MapleDeviceListCursor {
        registration_id: victim_first.uuid,
    };
    assert!(list_test_maple_devices(
        app_state.db.as_ref(),
        maple_device_list_authorization(&app_state, other.uuid, project.id),
        2,
        Some(victim_cursor),
    )
    .expect("foreign tuple remains scoped to the authenticated account")
    .iter()
    .all(|row| row.user_id == other.uuid));

    let _ = app_state.db.delete_user(&victim, &app_state.enclave_key);
    let _ = app_state.db.delete_user(&other, &app_state.enclave_key);
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_maple_device_list_readers_coexist_but_order_after_credential_writers() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("list lock test user should insert");
    let registration = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        91,
    );
    register_test_maple_device(&app_state, registration)
        .expect("list lock test device should register");

    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("list lock test database connection should be available");
    conn.transaction::<_, diesel::result::Error, _>(|tx| {
        users::table
            .filter(users::uuid.eq(user.uuid))
            .select(users::uuid)
            .for_share()
            .first::<Uuid>(tx)?;

        let db = app_state.db.clone();
        let authorization = maple_device_list_authorization(&app_state, user.uuid, project.id);
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            result_tx
                .send(list_test_maple_devices(
                    db.as_ref(),
                    authorization,
                    10,
                    None,
                ))
                .unwrap();
        });
        let rows = result_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("two list FOR SHARE locks must coexist")
            .expect("concurrent list should succeed");
        assert_eq!(rows.len(), 1);
        handle.join().unwrap();
        Ok(())
    })
    .unwrap();

    let mut pending = None;
    conn.transaction::<_, diesel::result::Error, _>(|tx| {
        users::table
            .filter(users::uuid.eq(user.uuid))
            .select(users::uuid)
            .for_update()
            .first::<Uuid>(tx)?;

        let db = app_state.db.clone();
        let authorization = maple_device_list_authorization(&app_state, user.uuid, project.id);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            result_tx
                .send(list_test_maple_devices(
                    db.as_ref(),
                    authorization,
                    10,
                    None,
                ))
                .unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            result_rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "list FOR SHARE must wait behind a credential writer lock"
        );
        pending = Some((handle, result_rx));
        Ok(())
    })
    .unwrap();

    let (handle, result_rx) = pending.unwrap();
    let rows = result_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("list should resume after credential writer commits")
        .expect("ordered list should succeed");
    assert_eq!(rows.len(), 1);
    handle.join().unwrap();

    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .unwrap();
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_pairing_materializers_require_the_process_pinned_verifier_keyset() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = create_password_wrapped_user(
        &app_state,
        project.id,
        format!("maple-verifier-fence-{}@example.com", Uuid::new_v4()),
        test_credential("maple-verifier-fence"),
    )
    .await;
    let mut remapped_keyset = test_maple_pairing_issuer_keyset(&[
        "maple-test-issuer-current",
        "maple-test-issuer-future",
        "maple-test-issuer-old",
        "maple-test-issuer-revocation",
    ]);
    let remapped_current = Ed25519MaplePairingIssuer::new(
        "maple-test-issuer-current".to_string(),
        SigningKey::from_bytes(&[9; 32]),
    )
    .expect("remapped test issuer should construct")
    .public_key_entry();
    let current = remapped_keyset
        .keys
        .iter_mut()
        .find(|key| key.key_id == "maple-test-issuer-current")
        .expect("current issuer should exist");
    *current = remapped_current;
    remapped_keyset
        .validate()
        .expect("same-ID remapped test keyset should be internally well-formed");

    let controller = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        81,
    );
    let rejected_registration = DBConnection::register_maple_device(
        app_state.db.as_ref(),
        controller.clone(),
        &remapped_keyset,
        &|_| panic!("verifier mismatch must reject before registration materialization"),
    );
    assert!(matches!(
        rejected_registration,
        Err(DBError::MaplePairingIssuerConfigurationConflict)
    ));
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);

    let host = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        83,
    );
    register_test_maple_device(&app_state, controller.clone())
        .expect("controller should register with the pinned verifier set");
    register_test_maple_device(&app_state, host.clone())
        .expect("host should register with the pinned verifier set");
    let authorization = maple_pairing_authorization(&app_state, user.uuid, project.id);
    let request = test_maple_pairing_create_request(
        &app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        Uuid::new_v4(),
        None,
        None,
        85,
    );
    let rejected_create = DBConnection::create_maple_pairing(
        app_state.db.as_ref(),
        request,
        &remapped_keyset,
        &|_| panic!("verifier mismatch must reject before CREATE materialization"),
    );
    assert!(matches!(
        rejected_create,
        Err(DBError::MaplePairingIssuerConfigurationConflict)
    ));
    assert_eq!(pairing_row_count(&app_state, user.uuid), 0);

    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("verifier-fence test account should clean up");
}

#[tokio::test]
#[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
async fn db_pairing_typed_material_substitution_rolls_back_every_authority_head() {
    let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
        return;
    };

    let app_state = build_local_test_app_state(database_url).await;
    let project = first_active_project(&app_state);
    let user = create_password_wrapped_user(
        &app_state,
        project.id,
        format!("maple-material-boundary-{}@example.com", Uuid::new_v4()),
        test_credential("maple-material-boundary"),
    )
    .await;
    let controller = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        171,
    );
    let host = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        173,
    );
    register_test_maple_device(&app_state, controller.clone())
        .expect("material-boundary controller should register");
    register_test_maple_device(&app_state, host.clone())
        .expect("material-boundary host should register");
    let authorization = maple_pairing_authorization(&app_state, user.uuid, project.id);
    let issuer = app_state
        .maple_pairing_issuer
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer");
    let keyset = app_state
        .maple_pairing_issuer_keyset
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer keyset");
    let locked_controller_key =
        test_maple_device_identity_key(controller.device_id, controller.installation_id);

    for substitution in 0_u8..3 {
        let request = test_maple_pairing_create_request(
            &app_state,
            &authorization,
            controller.registration_id,
            host.registration_id,
            Uuid::new_v4(),
            None,
            None,
            175_u8.wrapping_add(substitution),
        );
        let before = maple_pairing_create_rollback_snapshot(&app_state, &authorization);
        let enclave_key = app_state.enclave_key.clone();
        let result = DBConnection::create_maple_pairing(
            app_state.db.as_ref(),
            request,
            keyset,
            &|context| {
                let mut material = crate::web::maple_pairings::materialize_maple_pairing_create(
                    &enclave_key,
                    issuer,
                    project.id,
                    context,
                )?;
                match substitution {
                    0 | 1 => {
                        let fresh_key = SigningKey::from_bytes(&[201_u8 + substitution; 32]);
                        let fresh_public_key = fresh_key.verifying_key().to_bytes();
                        if substitution == 0 {
                            material.request_ticket.controller.identity_public_key =
                                STANDARD.encode(fresh_public_key);
                            material.request_ticket.controller.endpoint_id =
                                hex::encode(fresh_public_key);
                        } else {
                            material.request_ticket.host.identity_public_key =
                                STANDARD.encode(fresh_public_key);
                            material.request_ticket.host.endpoint_id =
                                hex::encode(fresh_public_key);
                        }
                        let mut embedded_request = material.request_ticket.controller_request();
                        embedded_request.signature = STANDARD.encode(
                            if substitution == 0 {
                                fresh_key.sign(
                                    &embedded_request
                                        .transcript()
                                        .expect("fresh controller transcript should encode"),
                                )
                            } else {
                                locked_controller_key.sign(
                                    &embedded_request
                                        .transcript()
                                        .expect("fresh host transcript should encode"),
                                )
                            }
                            .to_bytes(),
                        );
                        material.request_ticket.controller_request_digest = STANDARD.encode(
                            embedded_request
                                .digest()
                                .expect("substituted controller request digest should encode"),
                        );
                        material.request_ticket.controller_request_signature =
                            embedded_request.signature;
                        material.request_ticket.issuer_key_id.clear();
                        material.request_ticket.issuer_signature.clear();
                        material.request_ticket =
                            sign_pair_request_ticket(material.request_ticket, issuer)
                                .expect("substituted request ticket should remain issuer-valid");
                        material.response.pairing.request_ticket =
                            Some(material.request_ticket.clone());
                    }
                    2 => material.response.operation_id = Uuid::new_v4(),
                    _ => unreachable!(),
                }
                Ok(material)
            },
        );
        assert!(matches!(
            result,
            Err(DBError::MaplePairingMaterializationFailed)
        ));
        assert_eq!(
            maple_pairing_create_rollback_snapshot(&app_state, &authorization),
            before,
            "CREATE substitution {substitution} must leave every row/head unchanged (a sequence gap is allowed)"
        );
    }

    let active = create_active_test_pairing_for_user(&app_state, &user, project.id, 181);
    for substitution in 0_u8..7 {
        let before = maple_pairing_mutation_snapshot(
            &app_state,
            &active.authorization,
            active.pair_id,
            active.host_registration_id,
            active.revocation_stream_id,
            active.revocation_stream_generation,
        );
        let result = revoke_test_maple_pairing(
            &app_state,
            &active.authorization,
            active.controller_registration_id,
            MaplePairingRole::Controller,
            active.pairing_request_id,
            active.pair_id,
            3,
            active.pairing_incarnation,
            active.revocation_stream_id,
            active.revocation_stream_generation,
            Uuid::new_v4(),
            &|_| {},
            &|material| {
                match substitution {
                    0 => material.revocation.issuer_sequence += 1,
                    1 => material.revocation.reason_code = "trusted_but_wrong_reason".to_string(),
                    2 => {
                        material.revocation.revoked_by_registration_id =
                            material.revocation.host.registration_id;
                        material.revocation.revoked_by_role = WireMaplePairingRole::Host;
                    }
                    3 => material.revocation.revoked_at_unix_ms += 1,
                    4 => {
                        let untrusted_issuer = Ed25519MaplePairingIssuer::new(
                            "maple-test-issuer-untrusted".to_string(),
                            SigningKey::from_bytes(&[9; 32]),
                        )
                        .expect("untrusted test issuer should construct");
                        material.revocation.issuer_key_id.clear();
                        material.revocation.issuer_signature.clear();
                        material.revocation =
                            sign_pair_revocation(material.revocation.clone(), &untrusted_issuer)
                                .expect("wrong-key revocation should remain internally signed");
                    }
                    5 => material.pair_authorization.host.endpoint_epoch += 1,
                    6 => material.response.pairing.revocation = None,
                    _ => unreachable!(),
                }
                if substitution <= 3 {
                    let trusted_issuer = test_maple_pairing_revocation_issuer();
                    material.revocation.issuer_key_id.clear();
                    material.revocation.issuer_signature.clear();
                    material.revocation =
                        sign_pair_revocation(material.revocation.clone(), &trusted_issuer)
                            .expect("hostile locked-field revocation should remain issuer-valid");
                }
            },
        );
        assert!(matches!(
            result,
            Err(DBError::MaplePairingMaterializationFailed)
        ));
        assert_eq!(
            maple_pairing_mutation_snapshot(
                &app_state,
                &active.authorization,
                active.pair_id,
                active.host_registration_id,
                active.revocation_stream_id,
                active.revocation_stream_generation,
            ),
            before,
            "REVOKE substitution {substitution} must roll back pairing/event/op/highwater/host-state and every ancestor head"
        );
    }
}

#[tokio::test]
#[ignore = "requires MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL pointing at isolated disposable migrated local Postgres"]
async fn db_maple_pairing_lifecycle_is_ordered_and_destructive_cleanup_is_complete() {
    let Some(database_url) = std::env::var("MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL").ok() else {
        eprintln!("skipping: MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL is not set");
        return;
    };

    let initial_keyset = test_maple_pairing_issuer_keyset(&[
        "maple-test-issuer-current",
        "maple-test-issuer-old",
        "maple-test-issuer-revocation",
    ]);
    let app_state =
        build_local_test_app_state_with_keyset(database_url.clone(), Arc::new(initial_keyset))
            .await
            .expect("isolated issuer-rotation fixture should bootstrap the initial keyset");
    let project = first_active_project(&app_state);
    let user = app_state
        .db
        .create_user(NewUser::new(None, None, project.id), &app_state.enclave_key)
        .expect("pairing test user should insert");
    let controller = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        81,
    );
    let host = test_maple_device_registration(
        &app_state,
        user.uuid,
        project.id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        82,
    );
    register_test_maple_device(&app_state, controller.clone())
        .expect("controller registration should commit");
    register_test_maple_device(&app_state, host.clone()).expect("host registration should commit");
    let authorization = maple_pairing_authorization(&app_state, user.uuid, project.id);
    let (host_stream_id, host_stream_generation) =
        current_maple_revocation_stream(&app_state, &authorization, host.registration_id);
    let (controller_stream_id, controller_stream_generation) =
        current_maple_revocation_stream(&app_state, &authorization, controller.registration_id);

    let (first_request_id, first_pair_id, first_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        91,
    );
    let backwards_activation = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        let approved_at = maple_pairings::table
            .filter(maple_pairings::uuid.eq(first_pair_id))
            .select(maple_pairings::approved_at)
            .first::<Option<chrono::DateTime<Utc>>>(conn)
            .expect("active pairing approval timestamp should query")
            .expect("active pairing must retain its approval timestamp");
        diesel::update(maple_pairings::table.filter(maple_pairings::uuid.eq(first_pair_id)))
            .set(
                maple_pairings::activated_at
                    .eq(Some(approved_at - chrono::Duration::microseconds(1))),
            )
            .execute(conn)
    };
    assert!(matches!(
        backwards_activation,
        Err(diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::CheckViolation,
            _
        ))
    ));
    let first_event_id = Uuid::new_v4();
    let (revoked, first_digest) = revoke_test_maple_pairing(
        &app_state,
        &authorization,
        controller.registration_id,
        MaplePairingRole::Controller,
        first_request_id,
        first_pair_id,
        3,
        first_incarnation,
        host_stream_id,
        host_stream_generation,
        first_event_id,
        &|context| {
            assert_eq!(context.target_revision, 4);
            assert_eq!(context.issuer_sequence, 1);
        },
        &|_| {},
    )
    .expect("active pairing should revoke atomically");
    assert_eq!(revoked.pairing_revision, 4);

    let referenced_key_ids = app_state
        .db
        .audit_maple_pairing_issuer_key_references(&app_state.enclave_key)
        .expect("durable issuer-key references should authenticate");
    assert_eq!(
        referenced_key_ids,
        vec![
            "maple-test-issuer-current".to_string(),
            "maple-test-issuer-revocation".to_string(),
        ],
        "the current signing and revocation keys must remain durably referenced"
    );

    let issuer_root_before_rotation = maple_pairing_issuer_inventory_state(&app_state);
    assert_eq!(issuer_root_before_rotation.0, 3);
    let missing_old_keyset = test_maple_pairing_issuer_keyset(&[
        "maple-test-issuer-current",
        "maple-test-issuer-revocation",
    ]);
    let missing_old_result =
        build_local_test_app_state_with_keyset(database_url.clone(), Arc::new(missing_old_keyset))
            .await;
    assert!(matches!(
        missing_old_result,
        Err(Error::BuilderError(message))
            if message
                == "Maple pairing issuer keyset conflicts with the authenticated lifetime registry"
    ));
    assert_eq!(
        maple_pairing_issuer_inventory_state(&app_state),
        issuer_root_before_rotation,
        "a key-omission startup failure must not mutate the authenticated registry root"
    );

    let retained_rotation_keyset = test_maple_pairing_issuer_keyset(&[
        "maple-test-issuer-current",
        "maple-test-issuer-future",
        "maple-test-issuer-old",
        "maple-test-issuer-revocation",
    ]);
    let retained_state = build_local_test_app_state_with_keyset(
        database_url.clone(),
        Arc::new(retained_rotation_keyset),
    )
    .await
    .expect("a rotation keyset retaining every durable issuer must start");
    let issuer_root_after_rotation = maple_pairing_issuer_inventory_state(&retained_state);
    assert_eq!(issuer_root_after_rotation.0, 4);
    assert_ne!(
        issuer_root_after_rotation.1, issuer_root_before_rotation.1,
        "appending a public key must change the authenticated issuer inventory digest"
    );
    assert_eq!(
        issuer_root_after_rotation.2,
        issuer_root_before_rotation.2 + 1,
        "one atomic registry expansion must advance the global authority root exactly once"
    );
    assert!(matches!(
        app_state
            .db
            .audit_maple_pairing_issuer_key_references(&app_state.enclave_key),
        Err(DBError::MaplePairingIssuerConfigurationConflict)
    ));
    let app_state = retained_state;

    assert_fresh_pairing_rejects_stale_endpoint_epoch(
        &app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        109,
    );

    let unacked = app_state
        .db
        .list_maple_pairing_revocations(
            authorization.clone(),
            host.registration_id,
            host_stream_id,
            host_stream_generation,
            0,
            10,
        )
        .expect("host revocation page should load");
    assert_eq!(unacked.events.len(), 1);
    assert_eq!(unacked.events[0].event.uuid, first_event_id);
    assert_eq!(unacked.events[0].event.issuer_sequence, 1);
    assert!(unacked.events[0].event.acked_at.is_none());
    assert_eq!(unacked.last_issued_revocation_sequence, 1);
    assert_eq!(unacked.last_acked_revocation_sequence, 0);

    let acked = app_state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: authorization.clone(),
            operation_id: Uuid::new_v4(),
            request_mac: vec![100; 32],
            host_registration_id: host.registration_id,
            revocation_stream_id: host_stream_id,
            revocation_stream_generation: host_stream_generation,
            event_id: first_event_id,
            issuer_sequence: 1,
            event_digest: first_digest,
            expected_previous_issuer_sequence: 0,
            checkpoint_issuer_key_id: "maple-test-issuer-current".to_string(),
            receipt_version: 1,
            receipt_enc: vec![101, 102],
            accepted_at: Utc::now(),
        })
        .expect("durable host commit should acknowledge exactly the next revocation");
    assert_eq!(acked.pair_id, first_pair_id);
    assert_eq!(acked.pairing_revision, 4);

    let (active_request_id, active_pair_id, active_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &authorization,
        host.registration_id,
        controller.registration_id,
        103,
    );
    let failed = revoke_test_maple_pairing(
        &app_state,
        &authorization,
        host.registration_id,
        MaplePairingRole::Controller,
        active_request_id,
        active_pair_id,
        3,
        active_incarnation,
        controller_stream_id,
        controller_stream_generation,
        Uuid::new_v4(),
        &|_| {},
        &|material| material.response.operation_id = Uuid::new_v4(),
    );
    assert!(matches!(
        failed,
        Err(DBError::MaplePairingMaterializationFailed)
    ));
    let still_active = app_state
        .db
        .get_maple_pairing(authorization.clone(), host.registration_id, active_pair_id)
        .expect("pairing status should load")
        .expect("pairing should remain visible to its participant");
    assert_eq!(still_active.state, MaplePairingState::Active.as_db());
    assert_eq!(still_active.revision, 3);
    let rolled_back_page = app_state
        .db
        .list_maple_pairing_revocations(
            authorization.clone(),
            controller.registration_id,
            controller_stream_id,
            controller_stream_generation,
            0,
            10,
        )
        .expect("reverse host revocation page should load");
    assert!(rolled_back_page.events.is_empty());
    assert_eq!(rolled_back_page.last_issued_revocation_sequence, 0);

    let (last_request_id, last_pair_id, last_incarnation) = create_and_activate_test_pairing(
        &app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        105,
    );
    let last_event_id = Uuid::new_v4();
    let (_, last_digest) = revoke_test_maple_pairing(
        &app_state,
        &authorization,
        controller.registration_id,
        MaplePairingRole::Controller,
        last_request_id,
        last_pair_id,
        3,
        last_incarnation,
        host_stream_id,
        host_stream_generation,
        last_event_id,
        &|context| assert_eq!(context.issuer_sequence, 2),
        &|_| {},
    )
    .expect("second lineage incarnation should revoke");
    let final_page = app_state
        .db
        .list_maple_pairing_revocations(
            authorization.clone(),
            host.registration_id,
            host_stream_id,
            host_stream_generation,
            1,
            10,
        )
        .expect("second revocation page should load contiguously");
    assert_eq!(final_page.events.len(), 1);
    assert_eq!(final_page.events[0].event.uuid, last_event_id);
    assert_eq!(final_page.events[0].event.issuer_sequence, 2);
    assert!(final_page.events[0].event.acked_at.is_none());
    assert_eq!(final_page.last_issued_revocation_sequence, 2);
    assert_eq!(final_page.last_acked_revocation_sequence, 1);

    let blocked_delete = app_state.db.delete_user(&user, &app_state.enclave_key);
    assert!(matches!(
        blocked_delete,
        Err(DBError::MaplePairingAuthorityDeletionBlocked)
    ));
    assert_maple_pairing_row_counts(&app_state, user.uuid, 2, 3, 12, 2, 2);
    assert_maple_device_row_counts(&app_state, user.uuid, 2, 2);

    revoke_and_ack_active_test_pairing(
        &app_state,
        &ActiveTestPairing {
            authorization: authorization.clone(),
            controller_registration_id: host.registration_id,
            host_registration_id: controller.registration_id,
            pairing_request_id: active_request_id,
            pair_id: active_pair_id,
            pairing_incarnation: active_incarnation,
            revocation_stream_id: controller_stream_id,
            revocation_stream_generation: controller_stream_generation,
        },
        113,
    );
    app_state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: authorization.clone(),
            operation_id: Uuid::new_v4(),
            request_mac: vec![120; 32],
            host_registration_id: host.registration_id,
            revocation_stream_id: host_stream_id,
            revocation_stream_generation: host_stream_generation,
            event_id: last_event_id,
            issuer_sequence: 2,
            event_digest: last_digest,
            expected_previous_issuer_sequence: 1,
            checkpoint_issuer_key_id: "maple-test-issuer-current".to_string(),
            receipt_version: 1,
            receipt_enc: vec![121, 122],
            accepted_at: Utc::now(),
        })
        .expect("terminal host commit should acknowledge the second revocation");

    let first_pair_page = app_state
        .db
        .list_maple_pairings(
            maple_pairing_authorization(&app_state, user.uuid, project.id),
            controller.registration_id,
            MaplePairingRole::Controller,
            vec![MaplePairingState::Revoked],
            1,
            None,
        )
        .expect("newest revoked pair page should load");
    assert_eq!(first_pair_page.len(), 1);
    assert_eq!(first_pair_page[0].uuid, last_pair_id);
    let second_pair_page = app_state
        .db
        .list_maple_pairings(
            maple_pairing_authorization(&app_state, user.uuid, project.id),
            controller.registration_id,
            MaplePairingRole::Controller,
            vec![MaplePairingState::Revoked],
            1,
            Some(MaplePairingCursor {
                pair_id: last_pair_id,
            }),
        )
        .expect("older revoked pair page should use the cursor's internal insertion id");
    assert_eq!(second_pair_page.len(), 1);
    assert_eq!(second_pair_page[0].uuid, first_pair_id);

    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(maple_pairings::table.filter(maple_pairings::uuid.eq(first_pair_id)))
            .set(maple_pairings::authorization_issuer_key_id.eq(Some("maple-test-issuer-old")))
            .execute(conn)
            .expect("test should tamper the pair issuer reference with another registered key");
    }
    let pair_issuer_tamper = app_state
        .db
        .audit_maple_pairing_issuer_key_references(&app_state.enclave_key);
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(maple_pairings::table.filter(maple_pairings::uuid.eq(first_pair_id)))
            .set(maple_pairings::authorization_issuer_key_id.eq(Some("maple-test-issuer-current")))
            .execute(conn)
            .expect("test should restore the pair issuer reference");
    }
    assert!(
        matches!(
            &pair_issuer_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "pair issuer-reference tamper should fail exhaustive authority validation: {pair_issuer_tamper:?}"
    );
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_events::table
                .filter(maple_pairing_revocation_events::uuid.eq(first_event_id)),
        )
        .set(maple_pairing_revocation_events::issuer_key_id.eq("maple-test-issuer-current"))
        .execute(conn)
        .expect("test should tamper the event issuer reference with another registered key");
    }
    let event_issuer_tamper = app_state
        .db
        .audit_maple_pairing_issuer_key_references(&app_state.enclave_key);
    {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::update(
            maple_pairing_revocation_events::table
                .filter(maple_pairing_revocation_events::uuid.eq(first_event_id)),
        )
        .set(maple_pairing_revocation_events::issuer_key_id.eq("maple-test-issuer-revocation"))
        .execute(conn)
        .expect("test should restore the event issuer reference");
    }
    assert!(
        matches!(
            &event_issuer_tamper,
            Err(DBError::MaplePairingAuthorityCorrupt)
        ),
        "event issuer-reference tamper should fail exhaustive authority validation: {event_issuer_tamper:?}"
    );
    app_state
        .db
        .audit_maple_pairing_issuer_key_references(&app_state.enclave_key)
        .expect("restored issuer references should authenticate");

    assert_maple_pairing_row_counts(&app_state, user.uuid, 2, 3, 15, 2, 3);
    assert_maple_device_row_counts(&app_state, user.uuid, 2, 2);
    let direct_parent_delete = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        diesel::delete(users::table.filter(users::uuid.eq(user.uuid))).execute(conn)
    };
    assert!(matches!(
        direct_parent_delete,
        Err(diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::ForeignKeyViolation,
            _
        ))
    ));
    assert_maple_pairing_row_counts(&app_state, user.uuid, 2, 3, 15, 2, 3);
    assert_maple_device_row_counts(&app_state, user.uuid, 2, 2);
    app_state
        .db
        .delete_user(&user, &app_state.enclave_key)
        .expect("account deletion must explicitly clear restricted pairing children");
    assert_maple_pairing_row_counts(&app_state, user.uuid, 0, 0, 0, 0, 0);
    assert_maple_device_row_counts(&app_state, user.uuid, 0, 0);
}

async fn build_local_test_app_state(database_url: String) -> AppState {
    let db = setup_db(database_url);
    let keyset = Arc::new(test_maple_pairing_issuer_keyset(&[
        "maple-test-issuer-current",
        "maple-test-issuer-future",
        "maple-test-issuer-old",
        "maple-test-issuer-revocation",
    ]));
    AppStateBuilder::default()
        .app_mode(AppMode::Local)
        .db(db)
        .enclave_key([42u8; 32].to_vec())
        .aws_credential_manager(Arc::new(RwLock::new(None)))
        .openai_api_base("http://localhost:9".to_string())
        .tinfoil_api_base("http://localhost:9".to_string())
        .jwt_secret([24u8; 32].to_vec())
        .maple_pairing_issuer(Some(test_maple_pairing_issuer()))
        .maple_pairing_issuer_keyset(Some(keyset))
        .build()
        .await
        .expect("local test app state should build")
}

async fn build_local_test_app_state_with_keyset(
    database_url: String,
    keyset: Arc<MaplePairingIssuerKeySetV1>,
) -> Result<AppState, Error> {
    let db = setup_db(database_url);
    AppStateBuilder::default()
        .app_mode(AppMode::Local)
        .db(db)
        .enclave_key([42u8; 32].to_vec())
        .aws_credential_manager(Arc::new(RwLock::new(None)))
        .openai_api_base("http://localhost:9".to_string())
        .tinfoil_api_base("http://localhost:9".to_string())
        .jwt_secret([24u8; 32].to_vec())
        .maple_pairing_issuer(Some(test_maple_pairing_issuer()))
        .maple_pairing_issuer_keyset(Some(keyset))
        .build()
        .await
}

fn test_maple_pairing_issuer() -> Arc<dyn MaplePairingIssuer> {
    Arc::new(
        Ed25519MaplePairingIssuer::new(
            "maple-test-issuer-current".to_string(),
            SigningKey::from_bytes(&[1; 32]),
        )
        .expect("test Maple pairing issuer should construct"),
    )
}

fn test_maple_pairing_revocation_issuer() -> Ed25519MaplePairingIssuer {
    Ed25519MaplePairingIssuer::new(
        "maple-test-issuer-revocation".to_string(),
        SigningKey::from_bytes(&[4; 32]),
    )
    .expect("test Maple revocation issuer should construct")
}

fn test_maple_device_identity_key(device_id: Uuid, installation_id: Uuid) -> SigningKey {
    let mut seed = [0_u8; 32];
    seed[..16].copy_from_slice(device_id.as_bytes());
    seed[16..].copy_from_slice(installation_id.as_bytes());
    SigningKey::from_bytes(&seed)
}

fn register_test_maple_device(
    app_state: &AppState,
    registration: NewMapleDeviceRegistration,
) -> Result<MapleDeviceRegistrationReceipt, DBError> {
    let issuer = app_state
        .maple_pairing_issuer
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer");
    let keyset = app_state
        .maple_pairing_issuer_keyset
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer keyset");
    let enclave_key = registration.enclave_key.clone();
    DBConnection::register_maple_device(app_state.db.as_ref(), registration, keyset, &|context| {
        crate::web::maple_devices::materialize_maple_device_registration_sync(
            &enclave_key,
            issuer,
            keyset,
            context,
        )
    })
}

fn register_test_maple_device_with_bounded_busy_retry(
    app_state: &AppState,
    registration: NewMapleDeviceRegistration,
) -> Result<MapleDeviceRegistrationReceipt, DBError> {
    const MAX_ATTEMPTS: usize = 8;
    for attempt in 0..MAX_ATTEMPTS {
        match register_test_maple_device(app_state, registration.clone()) {
            Err(DBError::MaplePairingAuthorityBusy) if attempt + 1 < MAX_ATTEMPTS => {
                std::thread::sleep(Duration::from_millis(
                    5 * u64::try_from(attempt + 1).expect("retry count should fit"),
                ));
            }
            result => return result,
        }
    }
    unreachable!("the final bounded attempt always returns")
}

fn list_test_maple_devices(
    db: &(dyn DBConnection + Send + Sync),
    authorization: MapleDeviceListAuthorization,
    limit: i64,
    after: Option<MapleDeviceListCursor>,
) -> Result<Vec<MapleDevice>, DBError> {
    DBConnection::list_maple_devices(db, authorization, limit, after).map(|page| page.devices)
}

fn ack_test_reset_clear_registration(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    receipt: &MapleDeviceRegistrationReceipt,
    operation_id: Uuid,
    marker: u8,
) -> (MaplePairingRevocationAck, MaplePairingOperationReceipt) {
    let sync: MapleRevocationSyncV1 = serde_json::from_slice(&receipt.sync_payload)
        .expect("reset-clear registration sync should decode");
    let instruction = sync
        .reset_clear_instruction
        .as_ref()
        .expect("recovery registration must carry the reset-clear instruction");
    let ack = MaplePairingRevocationAck {
        authorization: authorization.clone(),
        operation_id,
        request_mac: vec![marker; 32],
        host_registration_id: receipt.registration_id,
        revocation_stream_id: sync.stream_checkpoint.revocation_stream_id,
        revocation_stream_generation: sync.stream_checkpoint.revocation_stream_generation,
        event_id: instruction.event_id,
        issuer_sequence: instruction.issuer_sequence,
        event_digest: instruction
            .event_digest()
            .expect("reset-clear event digest should encode")
            .to_vec(),
        expected_previous_issuer_sequence: 0,
        checkpoint_issuer_key_id: sync.stream_checkpoint.issuer_key_id.clone(),
        receipt_version: 1,
        receipt_enc: vec![marker, marker.wrapping_add(1)],
        accepted_at: receipt.accepted_at,
    };
    let accepted = app_state
        .db
        .ack_maple_pairing_revocation(ack.clone())
        .expect("reset-clear ACK should retire the recovered installation");
    (ack, accepted)
}

fn assert_reset_retirement_rows(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
    expected_retirements: i64,
    expected_tombstones: i64,
) -> Vec<u8> {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let authority_scope_digest = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(project_id))
        .select(maple_pairing_authority_account_heads::authority_scope_digest)
        .first::<Vec<u8>>(conn)
        .expect("test authority scope should load");
    let retirements = maple_pairing_installation_retirements::table
        .filter(
            maple_pairing_installation_retirements::authority_scope_digest
                .eq(&authority_scope_digest),
        )
        .count()
        .get_result::<i64>(conn)
        .expect("retirement count should query");
    let tombstones = maple_pairing_registration_operation_tombstones::table
        .filter(
            maple_pairing_registration_operation_tombstones::authority_scope_digest
                .eq(&authority_scope_digest),
        )
        .count()
        .get_result::<i64>(conn)
        .expect("registration tombstone count should query");
    let terminal_obligations = maple_pairing_reset_clear_obligations::table
        .filter(
            maple_pairing_reset_clear_obligations::authority_scope_digest
                .eq(&authority_scope_digest),
        )
        .filter(maple_pairing_reset_clear_obligations::state.eq(2_i16))
        .count()
        .get_result::<i64>(conn)
        .expect("terminal reset-clear obligation count should query");
    assert_eq!(retirements, expected_retirements);
    assert_eq!(tombstones, expected_tombstones);
    assert_eq!(terminal_obligations, expected_retirements);
    authority_scope_digest
}

#[allow(clippy::too_many_arguments)]
fn test_maple_pairing_create_request(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    controller_registration_id: Uuid,
    host_registration_id: Uuid,
    operation_id: Uuid,
    controller_endpoint_epoch: Option<u64>,
    host_endpoint_epoch: Option<u64>,
    marker: u8,
) -> NewMaplePairingRequest {
    let project = app_state
        .db
        .get_org_project_by_id(authorization.project_id)
        .expect("test Maple project should load");
    let devices = list_test_maple_devices(
        app_state.db.as_ref(),
        MapleDeviceListAuthorization {
            user_id: authorization.user_id,
            project_id: authorization.project_id,
            auth_credential_kind: authorization.auth_credential_kind.clone(),
            auth_binding: authorization.auth_binding,
            enclave_key: authorization.enclave_key.clone(),
        },
        32,
        None,
    )
    .expect("test pairing participants should load");
    let controller = devices
        .iter()
        .find(|device| device.uuid == controller_registration_id)
        .expect("test pairing controller should exist");
    let host = devices
        .iter()
        .find(|device| device.uuid == host_registration_id)
        .expect("test pairing host should exist");
    let controller_key =
        test_maple_device_identity_key(controller.device_id, controller.installation_id);
    let host_key = test_maple_device_identity_key(host.device_id, host.installation_id);
    let controller_endpoint_epoch = controller_endpoint_epoch.unwrap_or_else(|| {
        u64::try_from(controller.endpoint_epoch).expect("controller endpoint epoch should fit")
    });
    let host_endpoint_epoch = host_endpoint_epoch.unwrap_or_else(|| {
        u64::try_from(host.endpoint_epoch).expect("host endpoint epoch should fit")
    });
    let mut create_request = CreateMaplePairingRequest {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
        operation_id,
        asserted_account_id: authorization.user_id,
        asserted_project_id: project.client_id,
        controller_registration_id,
        controller_device_id: controller.device_id,
        controller_installation_id: controller.installation_id,
        controller_endpoint_id: hex::encode(controller_key.verifying_key().as_bytes()),
        controller_endpoint_epoch,
        host_registration_id,
        host_device_id: host.device_id,
        host_installation_id: host.installation_id,
        host_endpoint_id: hex::encode(host_key.verifying_key().as_bytes()),
        host_endpoint_epoch,
        direction: MaplePairingDirection::ControllerToHost,
        execution_target_id: host_registration_id,
        pairing_request_nonce: STANDARD.encode([marker; 32]),
        protocol_min: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        protocol_max: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        signature: STANDARD.encode([0_u8; 64]),
    };
    create_request.signature = STANDARD.encode(
        controller_key
            .sign(
                &create_request
                    .transcript()
                    .expect("test create request transcript should encode"),
            )
            .to_bytes(),
    );
    let request_mac = crate::web::maple_pairings::request_operation_mac(
        &authorization.enclave_key,
        &create_request
            .transcript()
            .expect("test create request transcript should re-encode"),
        &create_request.signature,
    )
    .expect("test create request MAC should derive");
    NewMaplePairingRequest {
        authorization: authorization.clone(),
        subject_project_id: project.client_id,
        operation_id,
        request_mac: request_mac.to_vec(),
        create_request,
        controller_registration_id,
        expected_controller_endpoint_epoch: controller_endpoint_epoch,
        host_registration_id,
        expected_host_endpoint_epoch: host_endpoint_epoch,
    }
}

fn create_test_maple_pairing(
    app_state: &AppState,
    request: NewMaplePairingRequest,
) -> Result<MaplePairingOperationReceipt, DBError> {
    create_test_maple_pairing_observed(app_state, request, &|_| {})
}

fn create_test_maple_pairing_observed(
    app_state: &AppState,
    request: NewMaplePairingRequest,
    observe_incarnation: &dyn Fn(u64),
) -> Result<MaplePairingOperationReceipt, DBError> {
    let issuer = app_state
        .maple_pairing_issuer
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer");
    let keyset = app_state
        .maple_pairing_issuer_keyset
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer keyset");
    let enclave_key = request.authorization.enclave_key.clone();
    let internal_project_id = request.authorization.project_id;
    DBConnection::create_maple_pairing(app_state.db.as_ref(), request, keyset, &|context| {
        observe_incarnation(context.pairing_incarnation);
        crate::web::maple_pairings::materialize_maple_pairing_create(
            &enclave_key,
            issuer,
            internal_project_id,
            context,
        )
    })
}

fn test_maple_pairing_issuer_keyset(key_ids: &[&str]) -> MaplePairingIssuerKeySetV1 {
    let mut keys = key_ids
        .iter()
        .map(|key_id| {
            // Issuer key IDs are immutable key identities across rotations;
            // adding/removing peers in the keyset must never remap an ID.
            let marker = match *key_id {
                "maple-test-issuer-current" => 1,
                "maple-test-issuer-future" => 2,
                "maple-test-issuer-old" => 3,
                "maple-test-issuer-revocation" => 4,
                other => panic!("unexpected test issuer key ID: {other}"),
            };
            Ed25519MaplePairingIssuer::new(
                (*key_id).to_string(),
                SigningKey::from_bytes(&[marker; 32]),
            )
            .expect("test issuer should construct")
            .public_key_entry()
        })
        .collect::<Vec<_>>();
    keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
    MaplePairingIssuerKeySetV1 {
        version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
        keys,
    }
}

fn test_maple_device_registration(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    marker: u8,
) -> NewMapleDeviceRegistration {
    let auth_binding = AuthBinding::from_bytes([73u8; 32]);
    let seed_enc = encrypt_seed_v1(
        &app_state.enclave_key,
        b"maple-device-test-seed",
        user_id,
        project_id,
        CredentialKind::Password,
        &auth_binding,
    )
    .expect("test Maple credential wrap should encrypt");
    app_state
        .db
        .upsert_user_seed_wrapping(NewUserSeedWrapping::new(
            user_id,
            CredentialKind::Password.as_str(),
            vec![73u8; 32],
            SEED_WRAP_VERSION_V1,
            seed_enc,
        ))
        .expect("test Maple credential wrap should upsert");
    build_test_maple_device_registration(
        app_state,
        user_id,
        project_id,
        operation_id,
        device_id,
        installation_id,
        *AuthBinding::from_bytes([73u8; 32]).as_bytes(),
        1,
        marker,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_test_maple_device_registration(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    auth_binding: [u8; 32],
    known_security_epoch: i64,
    marker: u8,
) -> NewMapleDeviceRegistration {
    build_test_maple_device_registration_with_id(
        app_state,
        user_id,
        project_id,
        operation_id,
        Uuid::new_v4(),
        device_id,
        installation_id,
        auth_binding,
        known_security_epoch,
        marker,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_test_maple_device_registration_with_id(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
    operation_id: Uuid,
    registration_id: Uuid,
    device_id: Uuid,
    installation_id: Uuid,
    auth_binding: [u8; 32],
    known_security_epoch: i64,
    marker: u8,
) -> NewMapleDeviceRegistration {
    let project = app_state
        .db
        .get_org_project_by_id(project_id)
        .expect("test Maple project should load");
    let endpoint_epoch = i64::from(marker) + 1;
    let identity_key = test_maple_device_identity_key(device_id, installation_id);
    let (identity_mac, payload_enc) = crate::web::maple_devices::build_test_maple_device_payload(
        &app_state.enclave_key,
        user_id,
        project_id,
        registration_id,
        device_id,
        installation_id,
        1,
        u64::try_from(endpoint_epoch).expect("test endpoint epoch should fit"),
        1,
        identity_key.verifying_key().to_bytes(),
    )
    .expect("test Maple device payload should encrypt");
    NewMapleDeviceRegistration {
        user_id,
        subject_project_id: project.client_id,
        project_id,
        operation_id,
        request_mac: vec![marker; 32],
        auth_credential_kind: CredentialKind::Password.as_str().to_string(),
        auth_binding,
        enclave_key: app_state.enclave_key.clone(),
        registration_id,
        device_id,
        installation_id,
        identity_mac,
        endpoint_epoch,
        expected_revision: None,
        known_security_epoch,
        payload_version: 1,
        payload_enc,
        revision: 1,
    }
}

fn next_maple_device_registration(
    current: &NewMapleDeviceRegistration,
    operation_id: Uuid,
    endpoint_epoch: i64,
    marker: u8,
) -> NewMapleDeviceRegistration {
    NewMapleDeviceRegistration {
        operation_id,
        request_mac: vec![marker; 32],
        endpoint_epoch,
        expected_revision: Some(current.revision),
        payload_enc: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
        revision: current.revision + 1,
        ..current.clone()
    }
}

fn maple_device_list_authorization(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
) -> MapleDeviceListAuthorization {
    MapleDeviceListAuthorization {
        user_id,
        project_id,
        auth_credential_kind: CredentialKind::Password.as_str().to_string(),
        auth_binding: [73u8; 32],
        enclave_key: app_state.enclave_key.clone(),
    }
}

fn maple_pairing_authorization(
    app_state: &AppState,
    user_id: Uuid,
    project_id: i32,
) -> MaplePairingAuthorization {
    MaplePairingAuthorization {
        user_id,
        project_id,
        auth_credential_kind: CredentialKind::Password.as_str().to_string(),
        auth_binding: [73u8; 32],
        enclave_key: app_state.enclave_key.clone(),
    }
}

fn current_maple_revocation_stream(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    host_registration_id: Uuid,
) -> (Uuid, u64) {
    let page = app_state
        .db
        .list_maple_pairing_revocations(
            authorization.clone(),
            host_registration_id,
            Uuid::nil(),
            0,
            0,
            1,
        )
        .expect("signed-discovery sentinel should reveal the current authenticated stream");
    (page.revocation_stream_id, page.revocation_stream_generation)
}

fn assert_fresh_pairing_rejects_stale_endpoint_epoch(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    controller_registration_id: Uuid,
    host_registration_id: Uuid,
    marker: u8,
) {
    let devices = list_test_maple_devices(
        app_state.db.as_ref(),
        maple_device_list_authorization(app_state, authorization.user_id, authorization.project_id),
        32,
        None,
    )
    .expect("pairing participants should load");
    let controller = devices
        .iter()
        .find(|device| device.uuid == controller_registration_id)
        .expect("controller registration should exist");
    let host = devices
        .iter()
        .find(|device| device.uuid == host_registration_id)
        .expect("host registration should exist");
    let current_controller_epoch =
        u64::try_from(controller.endpoint_epoch).expect("controller endpoint epoch should fit");
    let stale_controller_epoch = current_controller_epoch
        .checked_sub(1)
        .expect("test controller endpoint epoch should be positive");
    let row_count_before = pairing_row_count(app_state, authorization.user_id);
    let request = test_maple_pairing_create_request(
        app_state,
        authorization,
        controller_registration_id,
        host_registration_id,
        Uuid::new_v4(),
        Some(stale_controller_epoch),
        Some(u64::try_from(host.endpoint_epoch).expect("host endpoint epoch should fit")),
        marker,
    );
    let result = create_test_maple_pairing(app_state, request);
    assert!(matches!(result, Err(DBError::MaplePairingConflict)));
    assert_eq!(
        pairing_row_count(app_state, authorization.user_id),
        row_count_before,
        "a stale fresh create must not persist a pair"
    );
}

fn create_and_activate_test_pairing(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    controller_registration_id: Uuid,
    host_registration_id: Uuid,
    marker: u8,
) -> (Uuid, Uuid, u64) {
    let request = test_maple_pairing_create_request(
        app_state,
        authorization,
        controller_registration_id,
        host_registration_id,
        Uuid::new_v4(),
        None,
        None,
        marker,
    );
    let request_mac = request.request_mac.clone();
    let created = create_test_maple_pairing(app_state, request.clone())
        .expect("pairing creation should commit");
    assert_eq!(created.pairing_revision, 1);
    let replayed = create_test_maple_pairing(app_state, request)
        .expect("exact create retry should replay its receipt");
    assert_eq!(replayed.operation_id, created.operation_id);
    assert_eq!(replayed.pair_id, created.pair_id);
    assert_eq!(replayed.pairing_revision, created.pairing_revision);
    assert_eq!(replayed.receipt_enc, created.receipt_enc);

    let pairing = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairings::table
            .filter(maple_pairings::uuid.eq(created.pair_id))
            .first::<MaplePairing>(conn)
            .expect("created test pairing should load")
    };
    let pairing_request_id = pairing.pairing_request_id;
    let pair_id = pairing.uuid;
    let pairing_incarnation =
        u64::try_from(pairing.pairing_incarnation).expect("pairing incarnation should fit");
    let pairing_count_before_replay = pairing_row_count(app_state, authorization.user_id);
    let max_incarnation_before_replay = max_pairing_incarnation(app_state, authorization.user_id);
    let replayed_before_current_endpoint_validation = app_state
        .db
        .replay_maple_pairing_operation(
            authorization.clone(),
            controller_registration_id,
            created.operation_id,
            MaplePairingOperationKind::Create,
            request_mac.clone(),
        )
        .expect("pre-validation replay lookup should succeed")
        .expect("accepted create operation should be replayable");
    assert_eq!(
        replayed_before_current_endpoint_validation.receipt_enc,
        created.receipt_enc
    );
    assert_eq!(
        pairing_row_count(app_state, authorization.user_id),
        pairing_count_before_replay,
        "pre-validation replay must not create another pair"
    );
    assert_eq!(
        max_pairing_incarnation(app_state, authorization.user_id),
        max_incarnation_before_replay,
        "pre-validation replay must not reserve or persist a new incarnation"
    );
    assert!(app_state
        .db
        .replay_maple_pairing_operation(
            authorization.clone(),
            controller_registration_id,
            Uuid::new_v4(),
            MaplePairingOperationKind::Create,
            request_mac.clone(),
        )
        .expect("unknown operation lookup should not error")
        .is_none());
    assert!(matches!(
        app_state.db.replay_maple_pairing_operation(
            authorization.clone(),
            controller_registration_id,
            created.operation_id,
            MaplePairingOperationKind::Approve,
            request_mac,
        ),
        Err(DBError::MaplePairingConflict)
    ));

    let (revocation_stream_id, revocation_stream_generation) =
        current_maple_revocation_stream(app_state, authorization, host_registration_id);

    let created_payload =
        crate::web::maple_pairings::decrypt_pair_payload_for_test(&app_state.enclave_key, &pairing)
            .expect("DB-owned create payload should decrypt");
    let ticket = created_payload.request_ticket;
    let host_device = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_devices::table
            .filter(maple_devices::uuid.eq(host_registration_id))
            .first::<MapleDevice>(conn)
            .expect("test host device should load")
    };
    let host_key =
        test_maple_device_identity_key(host_device.device_id, host_device.installation_id);
    let approval_operation_id = Uuid::new_v4();
    let approved_at = Utc::now();
    let mut approval_request = ApproveMaplePairingRequest {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
        operation_id: approval_operation_id,
        asserted_account_id: authorization.user_id,
        asserted_project_id: ticket.subject_project_id,
        host_registration_id,
        pairing_request_id,
        pair_id,
        expected_pairing_revision: 1,
        pairing_incarnation,
        revocation_stream_id,
        revocation_stream_generation,
        request_ticket_digest: STANDARD.encode(
            ticket
                .digest()
                .expect("test request-ticket digest should encode"),
        ),
        host_approval_nonce: STANDARD.encode([marker.wrapping_add(5); 32]),
        approved_protocol_min: ticket.protocol_min,
        approved_protocol_max: ticket.protocol_max,
        signature: STANDARD.encode([0_u8; 64]),
    };
    approval_request.signature = STANDARD.encode(
        host_key
            .sign(
                &approval_request
                    .transcript()
                    .expect("test host-approval transcript should encode"),
            )
            .to_bytes(),
    );
    let approval_request_mac = crate::web::maple_pairings::request_operation_mac(
        &authorization.enclave_key,
        &approval_request
            .transcript()
            .expect("test host-approval transcript should re-encode"),
        &approval_request.signature,
    )
    .expect("test host-approval request MAC should derive");
    let pair_authorization = sign_pair_authorization(
        MaplePairAuthorizationV1 {
            artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
            subject_account_id: ticket.subject_account_id,
            subject_project_id: ticket.subject_project_id,
            pairing_request_id,
            pair_id,
            direction: ticket.direction,
            execution_target_id: ticket.execution_target_id,
            controller: ticket.controller.clone(),
            host: ticket.host.clone(),
            pairing_request_nonce: ticket.pairing_request_nonce.clone(),
            controller_request_operation_id: ticket.controller_request_operation_id,
            controller_request_digest: ticket.controller_request_digest.clone(),
            controller_request_signature: ticket.controller_request_signature.clone(),
            request_ticket_digest: approval_request.request_ticket_digest.clone(),
            host_approval_operation_id: approval_operation_id,
            host_approval_expected_pairing_revision: 1,
            host_approval_nonce: approval_request.host_approval_nonce.clone(),
            host_approval_digest: STANDARD.encode(
                approval_request
                    .digest()
                    .expect("test host-approval digest should encode"),
            ),
            host_approval_signature: approval_request.signature.clone(),
            pairing_incarnation,
            revocation_stream_id,
            revocation_stream_generation,
            protocol_min: ticket.protocol_min,
            protocol_max: ticket.protocol_max,
            approved_at_unix_ms: approved_at.timestamp_millis(),
            issuer_key_id: String::new(),
            issuer_signature: String::new(),
        },
        app_state
            .maple_pairing_issuer
            .as_deref()
            .expect("test AppState should inject a Maple pairing issuer"),
    )
    .expect("test pair authorization should sign");
    let active_payload = StoredMaplePairingPayloadV1 {
        request_ticket: ticket,
        pair_authorization: Some(pair_authorization.clone()),
        revocation: None,
    };
    let active_payload_enc = crate::web::maple_pairings::encrypt_pair_payload_for_test(
        &app_state.enclave_key,
        &pairing,
        revocation_stream_id,
        revocation_stream_generation,
        &active_payload,
    )
    .expect("test active pairing payload should encrypt");

    let approved = app_state
        .db
        .approve_maple_pairing(MaplePairingApproval {
            authorization: authorization.clone(),
            operation_id: approval_operation_id,
            request_mac: approval_request_mac.to_vec(),
            host_registration_id,
            pairing_request_id,
            pair_id,
            expected_pairing_revision: 1,
            pairing_incarnation,
            expected_revocation_stream_id: revocation_stream_id,
            expected_revocation_stream_generation: revocation_stream_generation,
            authorization_issuer_key_id: pair_authorization.issuer_key_id.clone(),
            pair_authorization_digest: pair_authorization
                .digest()
                .expect("test pair-authorization digest should encode")
                .to_vec(),
            payload_version: 1,
            payload_enc: active_payload_enc.clone(),
            receipt_version: 1,
            receipt_enc: vec![marker.wrapping_add(9), marker.wrapping_add(10)],
            approved_at,
        })
        .expect("host approval should transition to awaiting commit");
    assert_eq!(approved.pairing_revision, 2);

    let confirmed = app_state
        .db
        .confirm_maple_pairing(MaplePairingConfirmation {
            authorization: authorization.clone(),
            operation_id: Uuid::new_v4(),
            request_mac: vec![marker.wrapping_add(11); 32],
            host_registration_id,
            pairing_request_id,
            pair_id,
            expected_pairing_revision: 2,
            pairing_incarnation,
            payload_version: 1,
            payload_enc: active_payload_enc,
            receipt_version: 1,
            receipt_enc: vec![marker.wrapping_add(14), marker.wrapping_add(15)],
            activated_at: Utc::now(),
        })
        .expect("host confirmation should activate the pair");
    assert_eq!(confirmed.pairing_revision, 3);
    (pairing_request_id, pair_id, pairing_incarnation)
}

#[allow(clippy::too_many_arguments)]
fn revoke_test_maple_pairing(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    actor_registration_id: Uuid,
    actor_role: MaplePairingRole,
    pairing_request_id: Uuid,
    pair_id: Uuid,
    expected_pairing_revision: i64,
    pairing_incarnation: u64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: u64,
    event_id: Uuid,
    observe_context: &dyn Fn(MaplePairingRevocationContext),
    mutate_material: &dyn Fn(&mut MaplePairingRevocationMaterial),
) -> Result<(MaplePairingOperationReceipt, Vec<u8>), DBError> {
    let row = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairings::table
            .filter(maple_pairings::uuid.eq(pair_id))
            .first::<MaplePairing>(conn)
            .expect("test pairing should load")
    };
    let stored_payload =
        crate::web::maple_pairings::decrypt_pair_payload_for_test(&app_state.enclave_key, &row)
            .expect("test pairing authority payload should decrypt");
    let ticket = stored_payload.request_ticket;
    let pair_authorization = stored_payload
        .pair_authorization
        .expect("active test pairing should retain an authorization");
    let actor = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_devices::table
            .filter(maple_devices::uuid.eq(actor_registration_id))
            .first::<MapleDevice>(conn)
            .expect("test revocation actor should load")
    };
    let actor_key = test_maple_device_identity_key(actor.device_id, actor.installation_id);
    let wire_actor_role = match actor_role {
        MaplePairingRole::Controller => WireMaplePairingRole::Controller,
        MaplePairingRole::Host => WireMaplePairingRole::Host,
    };
    let operation_id = Uuid::new_v4();
    let mut revoke_request = RevokeMaplePairingRequest {
        protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
        transcript_version: MAPLE_PAIRING_TRANSCRIPT_VERSION_V1,
        operation_id,
        asserted_account_id: authorization.user_id,
        asserted_project_id: ticket.subject_project_id,
        actor_registration_id,
        actor_role: wire_actor_role,
        pairing_request_id,
        pair_id,
        expected_pairing_revision,
        pairing_incarnation,
        revocation_stream_id,
        revocation_stream_generation,
        reason_code: "test_revocation".to_string(),
        signature: STANDARD.encode([0_u8; 64]),
    };
    revoke_request.signature = STANDARD.encode(
        actor_key
            .sign(
                &revoke_request
                    .transcript()
                    .expect("test revoke transcript should encode"),
            )
            .to_bytes(),
    );
    let request_mac = crate::web::maple_pairings::request_operation_mac(
        &authorization.enclave_key,
        &revoke_request
            .transcript()
            .expect("test revoke transcript should re-encode"),
        &revoke_request.signature,
    )
    .expect("test revoke request MAC should derive");
    let issuer = test_maple_pairing_revocation_issuer();
    let keyset = app_state
        .maple_pairing_issuer_keyset
        .as_deref()
        .expect("test AppState should inject a Maple pairing issuer keyset");
    let revoke_request_for_material = revoke_request.clone();
    let receipt = app_state.db.revoke_maple_pairing(
        MaplePairingRevocation {
            authorization: authorization.clone(),
            revoke_request,
            operation_id,
            request_mac: request_mac.to_vec(),
            actor_registration_id,
            actor_role,
            pairing_request_id,
            pair_id,
            expected_pairing_revision,
            pairing_incarnation,
            expected_revocation_stream_id: revocation_stream_id,
            expected_revocation_stream_generation: revocation_stream_generation,
        },
        keyset,
        &|context| {
            observe_context(context);
            let revocation = sign_pair_revocation(
                MaplePairRevocationV1 {
                    artifact_version: MAPLE_PAIRING_ARTIFACT_VERSION_V1,
                    event_id,
                    subject_account_id: pair_authorization.subject_account_id,
                    subject_project_id: pair_authorization.subject_project_id,
                    recipient_host_registration_id: pair_authorization.host.registration_id,
                    issuer_sequence: context.issuer_sequence,
                    revocation_stream_id: context.revocation_stream_id,
                    revocation_stream_generation: context.revocation_stream_generation,
                    pairing_request_id: context.pairing_request_id,
                    pair_id: context.pair_id,
                    direction: pair_authorization.direction,
                    execution_target_id: pair_authorization.execution_target_id,
                    controller: pair_authorization.controller.clone(),
                    host: pair_authorization.host.clone(),
                    pairing_incarnation: context.pairing_incarnation,
                    pair_authorization_digest: STANDARD.encode(
                        pair_authorization
                            .digest()
                            .expect("test pair-authorization digest should encode"),
                    ),
                    revoked_by_registration_id: actor_registration_id,
                    revoked_by_role: wire_actor_role,
                    reason_code: revoke_request_for_material.reason_code.clone(),
                    revoked_at_unix_ms: context.revoked_at.timestamp_millis(),
                    issuer_key_id: String::new(),
                    issuer_signature: String::new(),
                },
                &issuer,
            )
            .expect("test revocation should sign");
            let response = MaplePairingMutationResponse {
                protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1,
                operation_id,
                pairing: MaplePairingStatusV1 {
                    pairing_request_id: context.pairing_request_id,
                    pair_id: context.pair_id,
                    state: WireMaplePairingState::Revoked,
                    revision: context.target_revision,
                    pairing_incarnation: context.pairing_incarnation,
                    revocation_stream_id: Some(context.revocation_stream_id),
                    revocation_stream_generation: Some(context.revocation_stream_generation),
                    direction: pair_authorization.direction,
                    execution_target_id: pair_authorization.execution_target_id,
                    controller_registration_id: pair_authorization.controller.registration_id,
                    host_registration_id: pair_authorization.host.registration_id,
                    created_at_unix_ms: row.created_at.timestamp_millis(),
                    expires_at_unix_ms: row.expires_at.timestamp_millis(),
                    approved_at_unix_ms: row
                        .approved_at
                        .map(|approved_at| approved_at.timestamp_millis()),
                    activated_at_unix_ms: row
                        .activated_at
                        .map(|activated_at| activated_at.timestamp_millis()),
                    revoked_at_unix_ms: Some(context.revoked_at.timestamp_millis()),
                    request_ticket: Some(ticket.clone()),
                    pair_authorization: if actor_role == MaplePairingRole::Controller
                        && row.activated_at.is_none()
                    {
                        None
                    } else {
                        Some(pair_authorization.clone())
                    },
                    revocation: Some(revocation.clone()),
                },
            };
            let mut material = MaplePairingRevocationMaterial {
                request_ticket: ticket.clone(),
                pair_authorization: pair_authorization.clone(),
                revocation,
                response,
            };
            mutate_material(&mut material);
            Ok(material)
        },
    )?;
    let event_digest = {
        let conn = &mut app_state
            .db
            .get_pool()
            .get()
            .expect("test database connection should be available");
        maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::uuid.eq(event_id))
            .select(maple_pairing_revocation_events::event_digest)
            .first::<Vec<u8>>(conn)
            .expect("committed test revocation event should load")
    };
    Ok((receipt, event_digest))
}

type MaplePairingMutationPairingSnapshot = (
    i16,
    i64,
    i16,
    Vec<u8>,
    Vec<u8>,
    Option<String>,
    Option<chrono::DateTime<Utc>>,
);

#[derive(Debug, PartialEq, Eq)]
struct MaplePairingMutationSnapshot {
    pairing: MaplePairingMutationPairingSnapshot,
    highwater: (i64, i64, Vec<u8>),
    host_state: (i64, i64, i64, Vec<u8>),
    event_count: i64,
    operation_count: i64,
    lineage_count: i64,
    account_head: (i64, i64, i64, Vec<u8>),
    project_head: (i64, Vec<u8>),
    org_head: (i64, Vec<u8>),
    global_head: (i64, Option<Vec<u8>>),
}

#[derive(Debug, PartialEq, Eq)]
struct MaplePairingCreateRollbackSnapshot {
    pairing_count: i64,
    lineage_count: i64,
    operation_count: i64,
    event_count: i64,
    account_head: (i64, i64, i64, Vec<u8>),
    project_head: (i64, Vec<u8>),
    org_head: (i64, Vec<u8>),
    global_head: (i64, Option<Vec<u8>>),
}

fn maple_pairing_create_rollback_snapshot(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
) -> MaplePairingCreateRollbackSnapshot {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let org_id = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(authorization.user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(authorization.project_id))
        .select(maple_pairing_authority_account_heads::org_id)
        .first::<i32>(conn)
        .expect("snapshot authority org should load");
    MaplePairingCreateRollbackSnapshot {
        pairing_count: maple_pairings::table
            .filter(maple_pairings::user_id.eq(authorization.user_id))
            .filter(maple_pairings::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot pairing count should query"),
        lineage_count: maple_pairing_lineages::table
            .filter(maple_pairing_lineages::user_id.eq(authorization.user_id))
            .filter(maple_pairing_lineages::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot lineage count should query"),
        operation_count: maple_pairing_operations::table
            .filter(maple_pairing_operations::user_id.eq(authorization.user_id))
            .filter(maple_pairing_operations::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot operation count should query"),
        event_count: maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(authorization.user_id))
            .filter(maple_pairing_revocation_events::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot event count should query"),
        account_head: maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(authorization.user_id))
            .filter(maple_pairing_authority_account_heads::project_id.eq(authorization.project_id))
            .select((
                maple_pairing_authority_account_heads::pairing_count,
                maple_pairing_authority_account_heads::pairing_operation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot authority head should load"),
        project_head: maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::project_id.eq(authorization.project_id))
            .select((
                maple_pairing_authority_project_heads::revision,
                maple_pairing_authority_project_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot project head should load"),
        org_head: maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org_id))
            .select((
                maple_pairing_authority_org_heads::revision,
                maple_pairing_authority_org_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot org head should load"),
        global_head: maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true))
            .select((
                maple_pairing_authority_global_heads::revision,
                maple_pairing_authority_global_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot global head should load"),
    }
}

fn maple_pairing_mutation_snapshot(
    app_state: &AppState,
    authorization: &MaplePairingAuthorization,
    pair_id: Uuid,
    host_registration_id: Uuid,
    revocation_stream_id: Uuid,
    revocation_stream_generation: u64,
) -> MaplePairingMutationSnapshot {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let host_id = maple_devices::table
        .filter(maple_devices::uuid.eq(host_registration_id))
        .select(maple_devices::id)
        .first::<i64>(conn)
        .expect("snapshot host should load");
    let generation =
        i64::try_from(revocation_stream_generation).expect("snapshot stream generation should fit");
    let org_id = maple_pairing_authority_account_heads::table
        .filter(maple_pairing_authority_account_heads::user_id.eq(authorization.user_id))
        .filter(maple_pairing_authority_account_heads::project_id.eq(authorization.project_id))
        .select(maple_pairing_authority_account_heads::org_id)
        .first::<i32>(conn)
        .expect("snapshot authority org should load");
    MaplePairingMutationSnapshot {
        pairing: maple_pairings::table
            .filter(maple_pairings::uuid.eq(pair_id))
            .select((
                maple_pairings::state,
                maple_pairings::revision,
                maple_pairings::payload_version,
                maple_pairings::payload_enc,
                maple_pairings::record_mac,
                maple_pairings::revocation_issuer_key_id,
                maple_pairings::revoked_at,
            ))
            .first(conn)
            .expect("snapshot pairing should load"),
        highwater: maple_pairing_revocation_highwaters::table
            .filter(
                maple_pairing_revocation_highwaters::revocation_stream_id.eq(revocation_stream_id),
            )
            .filter(
                maple_pairing_revocation_highwaters::revocation_stream_generation.eq(generation),
            )
            .select((
                maple_pairing_revocation_highwaters::last_issued_revocation_sequence,
                maple_pairing_revocation_highwaters::revision,
                maple_pairing_revocation_highwaters::record_mac,
            ))
            .first(conn)
            .expect("snapshot highwater should load"),
        host_state: maple_pairing_host_states::table
            .filter(maple_pairing_host_states::host_maple_device_id.eq(host_id))
            .filter(maple_pairing_host_states::revocation_stream_id.eq(revocation_stream_id))
            .filter(maple_pairing_host_states::revocation_stream_generation.eq(generation))
            .select((
                maple_pairing_host_states::last_issued_revocation_sequence,
                maple_pairing_host_states::last_acked_revocation_sequence,
                maple_pairing_host_states::revision,
                maple_pairing_host_states::record_mac,
            ))
            .first(conn)
            .expect("snapshot host state should load"),
        event_count: maple_pairing_revocation_events::table
            .filter(maple_pairing_revocation_events::user_id.eq(authorization.user_id))
            .filter(maple_pairing_revocation_events::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot event count should query"),
        operation_count: maple_pairing_operations::table
            .filter(maple_pairing_operations::user_id.eq(authorization.user_id))
            .filter(maple_pairing_operations::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot operation count should query"),
        lineage_count: maple_pairing_lineages::table
            .filter(maple_pairing_lineages::user_id.eq(authorization.user_id))
            .filter(maple_pairing_lineages::project_id.eq(authorization.project_id))
            .count()
            .get_result(conn)
            .expect("snapshot lineage count should query"),
        account_head: maple_pairing_authority_account_heads::table
            .filter(maple_pairing_authority_account_heads::user_id.eq(authorization.user_id))
            .filter(maple_pairing_authority_account_heads::project_id.eq(authorization.project_id))
            .select((
                maple_pairing_authority_account_heads::pairing_count,
                maple_pairing_authority_account_heads::pairing_operation_count,
                maple_pairing_authority_account_heads::revision,
                maple_pairing_authority_account_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot authority head should load"),
        project_head: maple_pairing_authority_project_heads::table
            .filter(maple_pairing_authority_project_heads::project_id.eq(authorization.project_id))
            .select((
                maple_pairing_authority_project_heads::revision,
                maple_pairing_authority_project_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot project head should load"),
        org_head: maple_pairing_authority_org_heads::table
            .filter(maple_pairing_authority_org_heads::org_id.eq(org_id))
            .select((
                maple_pairing_authority_org_heads::revision,
                maple_pairing_authority_org_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot org head should load"),
        global_head: maple_pairing_authority_global_heads::table
            .filter(maple_pairing_authority_global_heads::singleton.eq(true))
            .select((
                maple_pairing_authority_global_heads::revision,
                maple_pairing_authority_global_heads::record_mac,
            ))
            .first(conn)
            .expect("snapshot global head should load"),
    }
}

struct DuePendingTestPairing {
    pair_id: Uuid,
}

fn create_due_pending_test_pairing_for_user(
    app_state: &AppState,
    user: &User,
    project_id: i32,
    marker: u8,
) -> DuePendingTestPairing {
    let controller = test_maple_device_registration(
        app_state,
        user.uuid,
        project_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        marker,
    );
    let host = test_maple_device_registration(
        app_state,
        user.uuid,
        project_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        marker.wrapping_add(1),
    );
    register_test_maple_device(app_state, controller.clone())
        .expect("due-pending controller should register");
    register_test_maple_device(app_state, host.clone()).expect("due-pending host should register");
    let authorization = maple_pairing_authorization(app_state, user.uuid, project_id);
    let request = test_maple_pairing_create_request(
        app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        Uuid::new_v4(),
        None,
        None,
        marker.wrapping_add(2),
    );
    let pair_id = create_test_maple_pairing(app_state, request)
        .expect("due-pending pair should create")
        .pair_id;
    make_maple_pairing_pending_due_for_test(&*app_state.db, &authorization, pair_id)
        .expect("pending pair should move beyond the trusted expiry skew window");
    DuePendingTestPairing { pair_id }
}

struct ActiveTestPairing {
    authorization: MaplePairingAuthorization,
    controller_registration_id: Uuid,
    host_registration_id: Uuid,
    pairing_request_id: Uuid,
    pair_id: Uuid,
    pairing_incarnation: u64,
    revocation_stream_id: Uuid,
    revocation_stream_generation: u64,
}

fn create_active_test_pairing_for_user(
    app_state: &AppState,
    user: &User,
    project_id: i32,
    marker: u8,
) -> ActiveTestPairing {
    let controller = test_maple_device_registration(
        app_state,
        user.uuid,
        project_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        marker,
    );
    let host = test_maple_device_registration(
        app_state,
        user.uuid,
        project_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        marker.wrapping_add(1),
    );
    register_test_maple_device(app_state, controller.clone())
        .expect("test controller should register");
    register_test_maple_device(app_state, host.clone()).expect("test host should register");
    let authorization = maple_pairing_authorization(app_state, user.uuid, project_id);
    let (pairing_request_id, pair_id, pairing_incarnation) = create_and_activate_test_pairing(
        app_state,
        &authorization,
        controller.registration_id,
        host.registration_id,
        marker.wrapping_add(2),
    );
    let (revocation_stream_id, revocation_stream_generation) =
        current_maple_revocation_stream(app_state, &authorization, host.registration_id);
    ActiveTestPairing {
        authorization,
        controller_registration_id: controller.registration_id,
        host_registration_id: host.registration_id,
        pairing_request_id,
        pair_id,
        pairing_incarnation,
        revocation_stream_id,
        revocation_stream_generation,
    }
}

fn revoke_and_ack_active_test_pairing(
    app_state: &AppState,
    pairing: &ActiveTestPairing,
    marker: u8,
) {
    let event_id = Uuid::new_v4();
    let (_, event_digest) = revoke_test_maple_pairing(
        app_state,
        &pairing.authorization,
        pairing.controller_registration_id,
        MaplePairingRole::Controller,
        pairing.pairing_request_id,
        pairing.pair_id,
        3,
        pairing.pairing_incarnation,
        pairing.revocation_stream_id,
        pairing.revocation_stream_generation,
        event_id,
        &|_| {},
        &|_| {},
    )
    .expect("test pairing should revoke");
    app_state
        .db
        .ack_maple_pairing_revocation(MaplePairingRevocationAck {
            authorization: pairing.authorization.clone(),
            operation_id: Uuid::new_v4(),
            request_mac: vec![marker.wrapping_add(5); 32],
            host_registration_id: pairing.host_registration_id,
            revocation_stream_id: pairing.revocation_stream_id,
            revocation_stream_generation: pairing.revocation_stream_generation,
            event_id,
            issuer_sequence: 1,
            event_digest,
            expected_previous_issuer_sequence: 0,
            checkpoint_issuer_key_id: "maple-test-issuer-current".to_string(),
            receipt_version: 1,
            receipt_enc: vec![marker.wrapping_add(6)],
            accepted_at: Utc::now(),
        })
        .expect("test revocation should ACK");
}

fn assert_maple_pairing_row_counts(
    app_state: &AppState,
    user_id: Uuid,
    expected_lineages: i64,
    expected_pairings: i64,
    expected_operations: i64,
    expected_host_states: i64,
    expected_revocations: i64,
) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let lineages = maple_pairing_lineages::table
        .filter(maple_pairing_lineages::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("pairing lineage count should query");
    let pairings = maple_pairings::table
        .filter(maple_pairings::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("pairing count should query");
    let operations = maple_pairing_operations::table
        .filter(maple_pairing_operations::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("pairing operation count should query");
    let host_states = maple_pairing_host_states::table
        .filter(maple_pairing_host_states::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("pairing host-state count should query");
    let revocations = maple_pairing_revocation_events::table
        .filter(maple_pairing_revocation_events::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("pairing revocation count should query");
    assert_eq!(lineages, expected_lineages);
    assert_eq!(pairings, expected_pairings);
    assert_eq!(operations, expected_operations);
    assert_eq!(host_states, expected_host_states);
    assert_eq!(revocations, expected_revocations);
}

fn pairing_row_count(app_state: &AppState, user_id: Uuid) -> i64 {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    maple_pairings::table
        .filter(maple_pairings::user_id.eq(user_id))
        .count()
        .get_result(conn)
        .expect("pairing count should query")
}

fn maple_pairing_issuer_inventory_state(app_state: &AppState) -> (i64, Vec<u8>, i64) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    maple_pairing_authority_global_heads::table
        .filter(maple_pairing_authority_global_heads::singleton.eq(true))
        .select((
            maple_pairing_authority_global_heads::issuer_key_count,
            maple_pairing_authority_global_heads::issuer_key_inventory_digest,
            maple_pairing_authority_global_heads::revision,
        ))
        .first::<(i64, Vec<u8>, i64)>(conn)
        .expect("authenticated issuer inventory state should query")
}

fn max_pairing_incarnation(app_state: &AppState, user_id: Uuid) -> Option<i64> {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    maple_pairings::table
        .filter(maple_pairings::user_id.eq(user_id))
        .select(diesel::dsl::max(maple_pairings::pairing_incarnation))
        .first(conn)
        .expect("maximum pairing incarnation should query")
}

fn assert_maple_device_row_counts(
    app_state: &AppState,
    user_id: Uuid,
    expected_devices: i64,
    expected_operations: i64,
) {
    let conn = &mut app_state
        .db
        .get_pool()
        .get()
        .expect("test database connection should be available");
    let devices = maple_devices::table
        .filter(maple_devices::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("device count should query");
    let operations = maple_device_registration_operations::table
        .filter(maple_device_registration_operations::user_id.eq(user_id))
        .count()
        .get_result::<i64>(conn)
        .expect("device operation count should query");
    assert_eq!(devices, expected_devices);
    assert_eq!(operations, expected_operations);
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
        .create_user(
            NewUser::new(Some(email), Some(password_enc), project_id),
            &app_state.enclave_key,
        )
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
        .create_user(
            NewUser::new(Some(email), None, project_id),
            &app_state.enclave_key,
        )
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
