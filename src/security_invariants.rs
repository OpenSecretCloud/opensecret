use std::{fs, path::Path};

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
        "reset_request.mark_as_reset(conn)",
    ] {
        assert!(
            reset_body.contains(required_pattern),
            "destructive reset must contain `{required_pattern}`"
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
