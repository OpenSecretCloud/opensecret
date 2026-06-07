use std::{collections::BTreeSet, fs, path::Path};

const REQUEST_TIME_SCAN_ROOTS: &[&str] = &["src/main.rs", "src/web"];

const LEGACY_SEED_PATTERNS: &[&str] = &[
    "get_seed_encrypted",
    "seed_encrypted(",
    "generate_private_key(",
    "decrypt_user_seed_to_mnemonic",
    "decrypt_user_seed_to_key",
    "decrypt_and_derive_bip85_mnemonic",
];

const DESTRUCTIVE_RESET_REQUIRED_TABLES: &[&str] = &[
    "user_seed_wrappings",
    "user_embeddings",
    "agent_schedule_runs",
    "agent_schedules",
    "agents",
    "notification_events",
    "push_devices",
    "memory_blocks",
    "user_preferences",
    "user_kv",
    "user_instructions",
    "conversation_projects",
    "conversation_summaries",
    "conversations",
];

const DESTRUCTIVE_RESET_CASCADE_ENCRYPTED_TABLES: &[(&str, &str)] = &[
    ("assistant_messages", "conversations"),
    ("reasoning_items", "conversations"),
    ("responses", "conversations"),
    ("tool_calls", "conversations"),
    ("tool_outputs", "conversations"),
    ("user_messages", "conversations"),
];

const DESTRUCTIVE_RESET_UPDATED_ENCRYPTED_TABLES: &[&str] = &["users"];

const ENCRYPTED_TABLES_NOT_USER_PRIVATE_STORAGE: &[&str] = &[
    "account_deletion_requests",
    "org_project_secrets",
    "password_reset_requests",
    "platform_password_reset_requests",
    "platform_users",
    "user_oauth_connections",
];

#[test]
fn request_time_paths_do_not_use_legacy_seed_decrypt_helpers() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut findings = Vec::new();

    for root in REQUEST_TIME_SCAN_ROOTS {
        collect_forbidden_legacy_seed_matches(
            &manifest_dir.join(root),
            LEGACY_SEED_PATTERNS,
            &mut findings,
        );
    }

    assert!(
        findings.is_empty(),
        "request-time legacy seed use found:\n{}",
        findings.join("\n")
    );
}

#[test]
fn openai_compatible_routes_do_not_request_user_storage_keys() {
    let openai_routes = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/openai.rs");
    let contents =
        fs::read_to_string(&openai_routes).expect("OpenAI route source should be readable");

    assert!(
        !contents.contains("get_user_key("),
        "{} must not call get_user_key without API-key-bound seed wraps",
        openai_routes.display()
    );
}

#[test]
fn user_jwt_middleware_requires_active_seed_wrap_before_request_extensions() {
    let jwt_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/jwt.rs");
    let contents = fs::read_to_string(&jwt_source).expect("JWT source should be readable");
    let middleware_body = extract_function_body(&contents, "pub async fn validate_jwt");

    assert_patterns_in_order(
        middleware_body,
        &[
            "AuthContext::from_claims(&claims)",
            "if user.project_id != auth_context.project_id",
            "verify_seed_wrap_for_auth_context(&user, &auth_context)",
            "req.extensions_mut().insert(auth_context)",
            "req.extensions_mut().insert(user)",
        ],
    );
}

#[test]
fn openai_jwt_fallback_inserts_signed_auth_context_but_api_keys_do_not() {
    let openai_auth_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/openai_auth.rs");
    let contents =
        fs::read_to_string(&openai_auth_source).expect("OpenAI auth source should be readable");
    let middleware_body = extract_function_body(&contents, "pub async fn validate_openai_auth");

    for required_pattern in [
        "AuthContext::from_claims(&claims)",
        "if user.project_id != auth_context.project_id",
        "verify_seed_wrap_for_auth_context(&user, &auth_context)",
        "req.extensions_mut().insert(auth_context)",
        "req.extensions_mut().insert(AuthMethod::Jwt)",
    ] {
        assert!(
            middleware_body.contains(required_pattern),
            "OpenAI JWT fallback must contain `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        middleware_body,
        &[
            "AuthContext::from_claims(&claims)",
            "if user.project_id != auth_context.project_id",
            "verify_seed_wrap_for_auth_context(&user, &auth_context)",
            "req.extensions_mut().insert(auth_context)",
            "req.extensions_mut().insert(user)",
            "req.extensions_mut().insert(AuthMethod::Jwt)",
        ],
    );

    let api_key_insert = "req.extensions_mut().insert(AuthMethod::ApiKey)";
    let api_key_index = middleware_body
        .find(api_key_insert)
        .expect("OpenAI API-key branch should insert AuthMethod::ApiKey");
    let jwt_auth_context_index = middleware_body
        .find("AuthContext::from_claims(&claims)")
        .expect("OpenAI JWT fallback should parse AuthContext from claims");
    assert!(
        api_key_index < jwt_auth_context_index,
        "OpenAI API-key branch should return before JWT AuthContext parsing"
    );

    let api_key_branch = &middleware_body[..jwt_auth_context_index];
    assert!(
        !api_key_branch.contains("insert(auth_context)"),
        "OpenAI API-key auth must not synthesize or insert AuthContext without API-key-bound seed wraps"
    );
}

#[test]
fn refresh_route_preserves_signed_auth_context_without_recomputing_binding() {
    let login_routes = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/login_routes.rs");
    let contents =
        fs::read_to_string(&login_routes).expect("login route source should be readable");
    let refresh_body = extract_function_body(&contents, "pub async fn refresh_token");

    for required_pattern in [
        "AuthContext::from_claims(&claims)",
        "verify_seed_wrap_for_auth_context(&user, &auth_context)",
        "NewToken::new_with_auth_context(&user, TokenType::Access, &data, &auth_context)",
        "NewToken::new_with_auth_context(&user, TokenType::Refresh, &data, &auth_context)",
    ] {
        assert!(
            refresh_body.contains(required_pattern),
            "refresh route must contain `{required_pattern}`"
        );
    }

    for forbidden_pattern in [
        "authenticate_user",
        "password_auth_context_for_user",
        "oauth_auth_context_for_user",
        "compute_password_auth_binding",
        "compute_oauth_auth_binding",
        "password_enc",
        "provider_user_id",
        "get_user_oauth_connection",
    ] {
        assert!(
            !refresh_body.contains(forbidden_pattern),
            "refresh route must not recompute auth binding from DB state via `{forbidden_pattern}`"
        );
    }
}

#[test]
fn legacy_token_constructor_is_only_used_for_third_party_tokens() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut findings = Vec::new();

    for root in REQUEST_TIME_SCAN_ROOTS {
        collect_pattern_matches(&manifest_dir.join(root), "NewToken::new(", &mut findings);
    }

    assert_eq!(
        findings.len(),
        1,
        "expected exactly one legacy token constructor use for third-party tokens, found:\n{}",
        findings.join("\n")
    );
    assert!(
        findings[0].contains("src/web/protected_routes.rs"),
        "legacy token constructor should only be used in protected third-party token route, found {}",
        findings[0]
    );

    let protected_routes = manifest_dir.join("src/web/protected_routes.rs");
    let contents =
        fs::read_to_string(&protected_routes).expect("protected route source should be readable");
    let third_party_body =
        extract_function_body(&contents, "pub async fn generate_third_party_token");

    assert!(third_party_body.contains("NewToken::new("));
    assert!(third_party_body.contains("TokenType::ThirdParty"));
}

#[test]
fn user_token_constructor_binds_signed_auth_context_to_user_project_without_logging_binding() {
    let jwt_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/jwt.rs");
    let contents = fs::read_to_string(&jwt_source).expect("JWT source should be readable");
    let constructor_body = extract_function_body(&contents, "pub fn new_with_auth_context");

    for required_pattern in [
        "if user.project_id != auth_context.project_id",
        "return Err(ApiError::BadRequest)",
        "auth_context.apply_to_claims(&mut custom_claims)",
    ] {
        assert!(
            constructor_body.contains(required_pattern),
            "v2 user token constructor must contain `{required_pattern}`"
        );
    }

    for forbidden_pattern in [
        "Creating new v2 user token with claims",
        "{:?},\n            custom_claims",
        "{:?}\",\n            custom_claims",
    ] {
        assert!(
            !constructor_body.contains(forbidden_pattern),
            "v2 user token constructor must not log full custom claims via `{forbidden_pattern}`"
        );
    }
}

#[test]
fn destructive_password_reset_wipes_user_key_encrypted_storage_roots() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let contents = fs::read_to_string(&db_source).expect("DB source should be readable");
    let reset_marker = "debug!(\"Completing destructive password reset\");";
    let reset_marker_index = contents.find(reset_marker).unwrap_or_else(|| {
        panic!("destructive reset implementation should contain `{reset_marker}`")
    });
    let implementation_start = contents[..reset_marker_index]
        .rfind("fn complete_destructive_password_reset")
        .expect("destructive reset implementation signature should exist");
    let reset_body = extract_function_body(
        &contents[implementation_start..],
        "fn complete_destructive_password_reset",
    );

    for table_name in DESTRUCTIVE_RESET_REQUIRED_TABLES {
        assert!(
            reset_body.contains(&format!("{table_name}::table")),
            "destructive reset must delete from `{table_name}`"
        );
        assert!(
            reset_body.contains(&format!("{table_name}::user_id.eq(user_id)")),
            "destructive reset must scope `{table_name}` deletion to user_id"
        );
    }

    for required_pattern in [
        "users::password_enc.eq(Some(new_password_enc))",
        "users::seed_enc.eq(Some(new_legacy_seed_enc))",
        "new_wrapping.upsert_by_credential(conn)",
        "password_reset_requests::id.eq(reset_request.id)",
        "password_reset_requests::user_id.eq(user_id)",
        "password_reset_requests::is_reset.eq(false)",
        "password_reset_requests::expiration_time.gt(diesel::dsl::now)",
        "if consumed_reset_count != 1",
        "DBError::PasswordResetRequestNotFound",
        "password_reset_requests::id.ne(reset_request.id)",
    ] {
        assert!(
            reset_body.contains(required_pattern),
            "destructive reset must contain `{required_pattern}`"
        );
    }

    assert!(
        !reset_body.contains("user_api_keys::table"),
        "destructive password reset must preserve user_api_keys in this release"
    );
}

#[test]
fn destructive_password_reset_encrypted_schema_inventory_is_classified() {
    let schema_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models/schema.rs");
    let contents = fs::read_to_string(&schema_source).expect("schema source should be readable");
    let encrypted_tables = collect_schema_tables_with_encrypted_columns(&contents);

    let mut classified_tables = BTreeSet::new();
    classified_tables.extend(DESTRUCTIVE_RESET_REQUIRED_TABLES.iter().copied());
    classified_tables.extend(
        DESTRUCTIVE_RESET_CASCADE_ENCRYPTED_TABLES
            .iter()
            .map(|(table_name, _owner_table)| *table_name),
    );
    classified_tables.extend(DESTRUCTIVE_RESET_UPDATED_ENCRYPTED_TABLES.iter().copied());
    classified_tables.extend(ENCRYPTED_TABLES_NOT_USER_PRIVATE_STORAGE.iter().copied());

    let unclassified_tables = encrypted_tables
        .iter()
        .filter(|table_name| !classified_tables.contains(table_name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unclassified_tables.is_empty(),
        "encrypted schema tables must be classified for destructive reset handling:\n{}",
        unclassified_tables.join("\n")
    );

    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db_contents = fs::read_to_string(&db_source).expect("DB source should be readable");
    let reset_marker = "debug!(\"Completing destructive password reset\");";
    let reset_marker_index = db_contents.find(reset_marker).unwrap_or_else(|| {
        panic!("destructive reset implementation should contain `{reset_marker}`")
    });
    let implementation_start = db_contents[..reset_marker_index]
        .rfind("fn complete_destructive_password_reset")
        .expect("destructive reset implementation signature should exist");
    let reset_body = extract_function_body(
        &db_contents[implementation_start..],
        "fn complete_destructive_password_reset",
    );

    for (cascade_table, owner_table) in DESTRUCTIVE_RESET_CASCADE_ENCRYPTED_TABLES {
        assert!(
            encrypted_tables.contains(*cascade_table),
            "`{cascade_table}` should remain in the encrypted schema inventory"
        );
        assert!(
            reset_body.contains(&format!("{owner_table}::table")),
            "`{cascade_table}` is classified as cascade-covered, so destructive reset must delete owner `{owner_table}`"
        );
    }
}

#[test]
fn seed_wrap_translation_startup_path_is_feature_gated() {
    let migrations_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/migrations.rs");
    let contents =
        fs::read_to_string(&migrations_source).expect("migrations source should be readable");
    let startup_call = "migrate_aead_seed_wrappings_v1(app_state)?;";
    let startup_call_index = contents
        .find(startup_call)
        .unwrap_or_else(|| panic!("startup migration call `{startup_call}` should exist"));

    let prefix_window_start = startup_call_index.saturating_sub(120);
    let prefix_window = &contents[prefix_window_start..startup_call_index];
    assert!(
        prefix_window.contains("#[cfg(feature = \"seed-wrap-translation\")]"),
        "startup seed-wrap translation call must be feature-gated"
    );

    for required_gated_item in [
        "fn migrate_aead_seed_wrappings_v1",
        "fn validate_no_duplicate_oauth_subjects_by_project",
        "fn validate_user_seed_wrap_postflight_count",
        "fn validate_no_duplicate_seed_wrap_credentials",
        "fn validate_seed_wrap_postflight_count",
        "fn upsert_password_seed_wrap",
        "fn upsert_oauth_seed_wrap",
    ] {
        let item_index = contents
            .find(required_gated_item)
            .unwrap_or_else(|| panic!("`{required_gated_item}` should exist"));
        let item_prefix_start = item_index.saturating_sub(120);
        let item_prefix = &contents[item_prefix_start..item_index];
        assert!(
            item_prefix.contains("#[cfg(feature = \"seed-wrap-translation\")]"),
            "`{required_gated_item}` must be feature-gated"
        );
    }
}

#[test]
fn seed_wrap_translation_is_all_or_nothing_with_preflight_and_postflight() {
    let migrations_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/migrations.rs");
    let contents =
        fs::read_to_string(&migrations_source).expect("migrations source should be readable");
    let migration_body = extract_function_body(&contents, "fn migrate_aead_seed_wrappings_v1");

    for required_pattern in [
        "conn.transaction::<_, SeedWrapTranslationError, _>(|conn|",
        "SELECT pg_advisory_xact_lock(hashtext($1))",
        ".bind::<Text, _>(AEAD_SEED_WRAPPINGS_MIGRATION)",
        "AppDataMigration::get(conn, AEAD_SEED_WRAPPINGS_MIGRATION)?",
        "validate_no_duplicate_oauth_subjects_by_project(conn)?",
        "users::table.order(users::id.asc()).load::<User>(conn)?",
        ".seed_encrypted()",
        "decrypt_with_key(&legacy_secret_key, legacy_seed_enc)?",
        "upsert_password_seed_wrap(conn, app_state, &user, &verifier, &plaintext_seed)?",
        "upsert_oauth_seed_wrap(",
        "if user_wraps == 0",
        "SeedWrapTranslationError::NoUsableCredential(user.uuid)",
        "validate_user_seed_wrap_postflight_count(",
        "validate_seed_wrap_postflight_count(",
        "validate_no_duplicate_seed_wrap_credentials(conn)?",
        "NewAppDataMigration::new(AEAD_SEED_WRAPPINGS_MIGRATION).insert(conn)?",
    ] {
        assert!(
            migration_body.contains(required_pattern),
            "seed-wrap translation migration must contain `{required_pattern}`"
        );
    }

    assert_patterns_in_order(
        migration_body,
        &[
            "conn.transaction::<_, SeedWrapTranslationError, _>(|conn|",
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            "AppDataMigration::get(conn, AEAD_SEED_WRAPPINGS_MIGRATION)?",
            "validate_no_duplicate_oauth_subjects_by_project(conn)?",
            "users::table.order(users::id.asc()).load::<User>(conn)?",
            "validate_user_seed_wrap_postflight_count(",
            "validate_seed_wrap_postflight_count(",
            "validate_no_duplicate_seed_wrap_credentials(conn)?",
            "NewAppDataMigration::new(AEAD_SEED_WRAPPINGS_MIGRATION).insert(conn)?",
        ],
    );

    for helper_name in ["fn upsert_password_seed_wrap", "fn upsert_oauth_seed_wrap"] {
        let helper_body = extract_function_body(&contents, helper_name);
        for required_pattern in [
            "encrypt_seed_v1(",
            "decrypt_seed_v1(",
            "if verified_seed != plaintext_seed",
            ".upsert_by_credential(conn)?",
        ] {
            assert!(
                helper_body.contains(required_pattern),
                "translation helper `{helper_name}` must verify generated wraps before insert with `{required_pattern}`"
            );
        }
        assert_patterns_in_order(
            helper_body,
            &[
                "encrypt_seed_v1(",
                "decrypt_seed_v1(",
                "if verified_seed != plaintext_seed",
                ".upsert_by_credential(conn)?",
            ],
        );
    }
}

#[test]
fn user_password_reset_uses_mac_lookup_and_destructive_reseed() {
    let main_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let contents = fs::read_to_string(&main_source).expect("main source should be readable");
    let create_body = extract_function_body(&contents, "async fn create_password_reset_request");
    let confirm_body = extract_function_body(&contents, "async fn confirm_password_reset");

    for required_pattern in [
        "user.password_enc.is_some()",
        "password_reset_code_mac(",
        "NewPasswordResetRequest::new(",
        "reset_code_mac.to_vec()",
    ] {
        assert!(
            create_body.contains(required_pattern),
            "password reset request creation must contain `{required_pattern}`"
        );
    }

    for required_pattern in [
        "user.password_enc.is_none()",
        "password_reset_code_mac(",
        "get_password_reset_request_by_user_id_and_code(user.uuid, reset_code_mac.to_vec())",
        "generate_twelve_word_seed",
        "verify_new_password_seed_wrapping_for_user(",
        "complete_destructive_password_reset(",
    ] {
        assert!(
            confirm_body.contains(required_pattern),
            "password reset confirm must contain `{required_pattern}`"
        );
    }

    for forbidden_pattern in [
        "encrypt_with_key(&secret_key, alphanumeric_code",
        "encrypt_with_key(&secret_key, alphanumeric_code.as_bytes())",
    ] {
        assert!(
            !create_body.contains(forbidden_pattern),
            "user password reset must not store portable encrypted reset codes via `{forbidden_pattern}`"
        );
    }
}

#[test]
fn oauth_login_uses_verified_project_scoped_subject_and_pre_token_unwrap() {
    let oauth_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/oauth_routes.rs");
    let contents =
        fs::read_to_string(&oauth_source).expect("OAuth route source should be readable");
    let shared_oauth_body =
        extract_function_body(&contents, "async fn find_or_create_user_from_oauth");
    let authenticated_body = extract_function_body(&contents, "fn authenticated_oauth_user");
    let apple_native_body =
        extract_function_body(&contents, "pub async fn handle_apple_native_signin");

    for required_pattern in [
        "get_project_user_oauth_connection_by_provider_subject(",
        "provider.id",
        "&provider_user_id",
        "project_id",
        "update_provider_connection(app_state, &existing_connection, &access_token)",
        "authenticated_oauth_user(app_state, user, provider_name, &provider_user_id)",
    ] {
        assert!(
            shared_oauth_body.contains(required_pattern),
            "shared OAuth flow must contain `{required_pattern}`"
        );
    }

    for required_pattern in [
        "oauth_auth_context_for_user(&user, provider_name, provider_user_id)",
        "verify_seed_wrap_for_auth_context(&user, &auth_context)",
    ] {
        assert!(
            authenticated_body.contains(required_pattern),
            "OAuth token issuance must contain `{required_pattern}`"
        );
    }

    for required_pattern in [
        "get_project_user_oauth_connection_by_provider_subject(",
        "apple_provider.id",
        "&verified_user_id",
        "project.id",
        "update_provider_connection(&app_state, &connection, &access_token)",
        "authenticated_oauth_user(&app_state, user, \"apple\", &verified_user_id)",
    ] {
        assert!(
            apple_native_body.contains(required_pattern),
            "Apple native OAuth flow must contain `{required_pattern}`"
        );
    }
}

#[test]
fn password_credential_lifecycle_rewraps_seed_and_reissues_tokens() {
    let main_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let main_contents = fs::read_to_string(&main_source).expect("main source should be readable");
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db_contents = fs::read_to_string(&db_source).expect("DB source should be readable");
    let protected_source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/protected_routes.rs");
    let protected_contents =
        fs::read_to_string(&protected_source).expect("protected route source should be readable");

    let change_password_route =
        extract_function_body(&protected_contents, "pub async fn change_password");
    assert!(
        protected_contents.contains(
            "pub async fn change_password(\n    State(data): State<Arc<AppState>>,\n    Extension(user): Extension<User>,\n    Extension(auth_context): Extension<AuthContext>,"
        ),
        "password change route must require the current signed AuthContext extension"
    );
    for required_pattern in [
        ".authenticate_user(",
        ".update_user_password_and_seed_wrap(",
        "&auth_context",
        "NewToken::new_with_auth_context(",
        "TokenType::Access",
        "TokenType::Refresh",
        "&new_auth_context",
    ] {
        assert!(
            change_password_route.contains(required_pattern),
            "password change route must contain `{required_pattern}`"
        );
    }
    assert!(
        !change_password_route.contains("&authenticated_user.auth_context"),
        "password change must unwrap with the current signed AuthContext, not a DB-recomputed password auth context"
    );

    let password_update_helper = extract_function_body(
        &main_contents,
        "async fn update_user_password_and_seed_wrap",
    );
    for required_pattern in [
        "let expected_password_enc = user",
        "decrypt_seed_for_auth_context(user, auth_context)",
        "encrypt_user_password_verifier(new_password)",
        "new_password_seed_wrapping_for_user(user, &password_hash, &plaintext_seed)",
        "password_auth_context_for_user(user, &password_hash)",
        "expected_password_enc",
        "encrypted_password",
        "new_wrapping",
        "DBError::StaleCredentialState => Error::AuthenticationError",
        "verify_seed_wrap_for_auth_context(user, &new_auth_context)",
    ] {
        assert!(
            password_update_helper.contains(required_pattern),
            "password update helper must contain `{required_pattern}`"
        );
    }

    let password_update_start = db_contents
        .rfind("fn update_user_password_and_seed_wrap")
        .expect("password seed-wrap update implementation signature should exist");
    let password_update_db = extract_function_body(
        &db_contents[password_update_start..],
        "fn update_user_password_and_seed_wrap",
    );
    assert_patterns_in_order(
        password_update_db,
        &[
            "users::password_enc.eq(Some(expected_password_enc.to_vec()))",
            "users::password_enc.eq(Some(new_password_enc))",
            "if updated_user_count != 1",
            "DBError::StaleCredentialState",
            "UserSeedWrapping::delete_for_user_and_kind(",
            "CredentialKind::Password.as_str()",
            "new_wrapping.insert(conn)",
        ],
    );
    assert!(
        !password_update_db.contains("new_wrapping.upsert_by_credential(conn)"),
        "password change must delete all password wraps before inserting the replacement, not rely on DB-controlled lookup-hash upsert"
    );

    assert!(
        !protected_contents.contains("/protected/convert_guest")
            && !protected_contents.contains("convert_guest_to_email")
            && !main_contents.contains("convert_guest_to_email_and_seed_wrap")
            && !db_contents.contains("update_user_and_seed_wrap"),
        "guest conversion is intentionally unsupported; do not reintroduce it without revisiting the seed-wrap lifecycle"
    );
}

#[test]
fn password_registration_and_login_issue_tokens_only_after_seed_wrap_verification() {
    let main_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let main_contents = fs::read_to_string(&main_source).expect("main source should be readable");
    let login_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/login_routes.rs");
    let login_contents =
        fs::read_to_string(&login_source).expect("login route source should be readable");

    let register_user_body = extract_function_body(&main_contents, "async fn register_user");
    for required_pattern in [
        "generate_hash(password)",
        "generate_twelve_word_seed",
        "NewUser::new(creds.email, Some(encrypted_pw), project.id, encrypted_key)",
        "create_user_with_password_seed_wrap(",
    ] {
        assert!(
            register_user_body.contains(required_pattern),
            "registration helper must contain `{required_pattern}`"
        );
    }

    let create_user_wrap_body =
        extract_function_body(&main_contents, "fn create_user_with_password_seed_wrap");
    for required_pattern in [
        "conn.transaction::<_, CreateUserSeedWrapTransactionError, _>",
        "new_user.insert(conn)",
        "new_password_seed_wrapping_for_user(",
        "new_wrapping.upsert_by_credential(conn)",
    ] {
        assert!(
            create_user_wrap_body.contains(required_pattern),
            "password registration must atomically create user and seed wrap with `{required_pattern}`"
        );
    }

    let new_password_wrap_body =
        extract_function_body(&main_contents, "fn new_password_seed_wrapping_for_user");
    for required_pattern in [
        "encrypt_seed_v1(",
        "verify_new_password_seed_wrapping_for_user(",
    ] {
        assert!(
            new_password_wrap_body.contains(required_pattern),
            "new password seed wrap construction must contain `{required_pattern}`"
        );
    }

    let create_oauth_user_wrap_body =
        extract_function_body(&main_contents, "pub fn create_user_with_oauth_seed_wrap");
    for required_pattern in [
        "conn.transaction::<_, CreateUserSeedWrapTransactionError, _>",
        "new_user.insert(conn)",
        "NewUserOAuthConnection",
        "new_connection.insert(conn)",
        "new_oauth_seed_wrapping_for_user(",
        "new_wrapping.upsert_by_credential(conn)",
        "NewEmailVerification::new(user.uuid, 24, true)",
        "new_verification.insert(conn)",
    ] {
        assert!(
            create_oauth_user_wrap_body.contains(required_pattern),
            "OAuth registration must atomically create user, provider connection, seed wrap, and verified email with `{required_pattern}`"
        );
    }

    let create_oauth_wrap_body =
        extract_function_body(&main_contents, "fn new_oauth_seed_wrapping_for_user");
    for required_pattern in [
        "encrypt_seed_v1(",
        "verify_new_oauth_seed_wrapping_for_user(",
    ] {
        assert!(
            create_oauth_wrap_body.contains(required_pattern),
            "OAuth seed wrap creation must contain `{required_pattern}`"
        );
    }

    let register_route = extract_function_body(&login_contents, "pub async fn register");
    for required_pattern in [
        "data.register_user(creds.clone()).await",
        "login_internal(",
        "password: creds.password",
    ] {
        assert!(
            register_route.contains(required_pattern),
            "registration route must contain `{required_pattern}`"
        );
    }

    let authenticate_body = extract_function_body(&main_contents, "async fn authenticate_user");
    for required_pattern in [
        "decrypt_with_key(&secret_key, user.password_enc.as_ref().unwrap())",
        "verify_password(user_password, &decrypted_password_hash)",
        "password_auth_context_for_user(&user, &verifier_for_binding)",
        "verify_seed_wrap_for_auth_context(&user, &auth_context)",
        "AuthenticatedUser { user, auth_context }",
    ] {
        assert!(
            authenticate_body.contains(required_pattern),
            "password authentication must contain `{required_pattern}`"
        );
    }

    let login_internal_body = extract_function_body(&login_contents, "async fn login_internal");
    for required_pattern in [
        ".authenticate_user(",
        "NewToken::new_with_auth_context(",
        "TokenType::Access",
        "TokenType::Refresh",
        "&authenticated_user.auth_context",
    ] {
        assert!(
            login_internal_body.contains(required_pattern),
            "password login must contain `{required_pattern}`"
        );
    }
}

fn collect_forbidden_legacy_seed_matches(
    path: &Path,
    forbidden_patterns: &[&str],
    findings: &mut Vec<String>,
) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("source directory should be readable") {
            let entry = entry.expect("source directory entry should be readable");
            collect_forbidden_legacy_seed_matches(&entry.path(), forbidden_patterns, findings);
        }
        return;
    }

    if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
        return;
    }

    let contents = fs::read_to_string(path).expect("source file should be readable");
    for (line_index, line) in contents.lines().enumerate() {
        for pattern in forbidden_patterns {
            if line.contains(pattern) {
                findings.push(format!(
                    "{}:{} contains `{}`",
                    path.display(),
                    line_index + 1,
                    pattern
                ));
            }
        }
    }
}

fn collect_pattern_matches(path: &Path, pattern: &str, findings: &mut Vec<String>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("source directory should be readable") {
            let entry = entry.expect("source directory entry should be readable");
            collect_pattern_matches(&entry.path(), pattern, findings);
        }
        return;
    }

    if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
        return;
    }

    let contents = fs::read_to_string(path).expect("source file should be readable");
    for (line_index, line) in contents.lines().enumerate() {
        if line.contains(pattern) {
            findings.push(format!(
                "{}:{} contains `{}`",
                path.display(),
                line_index + 1,
                pattern
            ));
        }
    }
}

fn collect_schema_tables_with_encrypted_columns(schema_source: &str) -> BTreeSet<String> {
    let mut encrypted_tables = BTreeSet::new();
    let mut in_table_macro = false;
    let mut current_table: Option<String> = None;

    for line in schema_source.lines() {
        let trimmed = line.trim();

        if trimmed == "diesel::table! {" {
            in_table_macro = true;
            current_table = None;
            continue;
        }

        if !in_table_macro {
            continue;
        }

        if current_table.is_none() && trimmed.ends_with('{') && trimmed.contains(" (") {
            let table_name = trimmed
                .split_whitespace()
                .next()
                .expect("schema table declaration should have a table name");
            current_table = Some(table_name.to_string());
            continue;
        }

        if trimmed == "}" {
            in_table_macro = false;
            current_table = None;
            continue;
        }

        let is_encrypted_column =
            trimmed.contains("_enc ->") || trimmed.contains("encrypted_code ->");
        if is_encrypted_column {
            let table_name = current_table.as_ref().unwrap_or_else(|| {
                panic!("encrypted column `{trimmed}` should appear inside a table declaration")
            });
            encrypted_tables.insert(table_name.clone());
        }
    }

    encrypted_tables
}

fn extract_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let signature_start = source
        .find(signature)
        .unwrap_or_else(|| panic!("function signature `{signature}` should exist"));
    let body_start = source[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .unwrap_or_else(|| panic!("function `{signature}` should have a body"));

    let mut depth = 0i32;
    for (relative_index, byte) in source[body_start..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let body_end = body_start + relative_index + 1;
                    return &source[body_start..body_end];
                }
            }
            _ => {}
        }
    }

    panic!("function `{signature}` body should be balanced");
}

fn assert_patterns_in_order(source: &str, patterns: &[&str]) {
    let mut search_offset = 0usize;

    for pattern in patterns {
        let relative_index = source[search_offset..]
            .find(pattern)
            .unwrap_or_else(|| panic!("expected `{pattern}` after offset {search_offset}"));
        search_offset += relative_index + pattern.len();
    }
}
