use std::{collections::BTreeSet, fs, path::Path};
use syn::visit::{self, Visit};

const REQUEST_TIME_SCAN_ROOTS: &[&str] = &["src/main.rs", "src/web"];
const REQUEST_TIME_LOG_SCAN_ROOTS: &[&str] = &["src/main.rs", "src/db.rs", "src/web"];

const SENSITIVE_LOG_IDENTIFIERS: &[&str] = &[
    "session_key",
    "refresh_token",
    "alphanumeric_code",
    "reset_id",
    "event_id",
    "operation_id",
    "issuer_key_id",
    "issuer_signature",
    "authority_scope_digest",
    "lookup_digest",
    "operation_lookup_digest",
    "ack_host_registration_lookup_digest",
    "ack_operation_lookup_digest",
    "request_mac",
    "record_mac",
    "host_identity_mac",
    "pair_authorization_digest",
    "admission_set_digest",
    "outcome_digest",
    "receipt_digest",
    "receipt_enc",
    "referenced_issuer_key_ids",
    "final_obligation_event_id",
    "final_instruction_digest",
    "final_chain_digest",
    "previous_instruction_digest",
    "previous_chain_digest",
    "host_claim_digest",
    "instruction_digest",
    "chain_digest",
    "signed_instruction_digest",
    "ack_receipt_digest",
    "ack_request_mac",
    "sync_digest",
    "sync_payload",
    "host_claim_payload",
    "instruction_payload",
    "signed_instruction_payload",
    "ack_receipt",
];

const SENSITIVE_LOG_MESSAGES: &[&str] = &[
    "session key:",
    "Generated session key",
    "Logout request for refresh token:",
    "Platform logout request for refresh token:",
    "with code {alphanumeric_code}",
];

const LOG_MACROS: &[&str] = &[
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "event",
    "span",
    "trace_span",
    "debug_span",
    "info_span",
    "warn_span",
    "error_span",
];

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

const DESTRUCTIVE_RESET_MAPLE_HELPER_ENCRYPTED_TABLES: &[&str] = &[
    "maple_device_registration_operations",
    "maple_devices",
    "maple_pairing_operations",
    "maple_pairing_registration_operation_tombstones",
    "maple_pairing_reset_clear_obligations",
    "maple_pairing_revocation_events",
    "maple_pairings",
];

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
fn request_time_logs_do_not_include_secrets_or_maple_authority_material() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut findings = Vec::new();

    for root in REQUEST_TIME_LOG_SCAN_ROOTS {
        collect_sensitive_log_findings_in_path(
            &manifest_dir.join(root),
            manifest_dir,
            &mut findings,
        );
    }

    assert!(
        findings.is_empty(),
        "request-time log macros must not reference secrets or Maple authority material:\n{}",
        findings.join("\n")
    );
}

#[test]
fn sensitive_log_scanner_detects_multiline_fields_and_formatted_identifiers() {
    let source = r#"
fn example(request: Request, session_key: [u8; 32], alphanumeric_code: String) {
    tracing::warn!(
        refresh_token = %request.refresh_token,
        "logout failed"
    );
    debug!("generated key: {session_key:?}");
    tracing::event!(Level::WARN, alphanumeric_code, "reset failed");
    trace!("Generated session key: redacted");
}
"#;

    let findings = collect_sensitive_log_findings("example.rs", source);

    assert_eq!(findings.len(), 4, "unexpected findings: {findings:#?}");
    for expected in ["refresh_token", "session_key", "alphanumeric_code"] {
        assert!(
            findings.iter().any(|finding| finding.contains(expected)),
            "expected a finding for `{expected}`"
        );
    }
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("Generated session key")),
        "expected the legacy exact message to remain forbidden"
    );
}

#[test]
fn sensitive_log_scanner_ignores_identifiers_outside_logging_macros() {
    let source = r#"
fn example(refresh_token: String, session_key: [u8; 32], alphanumeric_code: String) {
    let response = json!({
        "refresh_token": refresh_token,
        "session_key": session_key,
        "alphanumeric_code": alphanumeric_code,
    });
    info!(refresh_token_hash = "redacted", "token revoked");
    consume(response);
}
"#;

    assert!(collect_sensitive_log_findings("example.rs", source).is_empty());
}

#[test]
fn maple_authority_models_never_derive_blanket_debug() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let device_models = fs::read_to_string(manifest_dir.join("src/models/maple_devices.rs"))
        .expect("Maple device model source should be readable");
    let pairing_models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB model source should be readable");
    let db_source =
        fs::read_to_string(manifest_dir.join("src/db.rs")).expect("DB source should be readable");

    assert_structs_do_not_derive_debug(
        "src/models/maple_devices.rs",
        &device_models,
        &[
            "MapleDevice",
            "NewMapleDevice",
            "MapleDeviceRegistrationOperation",
            "NewMapleDeviceRegistrationOperation",
            "NewMapleDeviceRegistration",
            "MapleDeviceListAuthorization",
            "MapleDeviceRegistrationReceipt",
            "MaplePairingRegistrationOperationTombstone",
            "NewMaplePairingRegistrationOperationTombstone",
            "MaplePairingInstallationRetirement",
            "NewMaplePairingInstallationRetirement",
        ],
    );
    assert_structs_do_not_derive_debug(
        "src/models/maple_pairing_db.rs",
        &pairing_models,
        &[
            "MaplePairingAuthorityGlobalHead",
            "MaplePairingIssuerKey",
            "NewMaplePairingIssuerKey",
            "NewMaplePairingAuthorityOrgHead",
            "MaplePairingAuthorityOrgHead",
            "NewMaplePairingAuthorityProjectHead",
            "MaplePairingAuthorityProjectHead",
            "NewMaplePairingAuthorityAccountHead",
            "MaplePairingAuthorityAccountHead",
            "MaplePairingAuthorization",
            "MaplePairingLineage",
            "NewMaplePairingLineage",
            "MaplePairing",
            "NewMaplePairing",
            "MaplePairingOperation",
            "NewMaplePairingOperation",
            "MaplePairingHostState",
            "NewMaplePairingHostState",
            "MaplePairingRevocationHighwater",
            "NewMaplePairingRevocationHighwater",
            "MaplePairingResetClearObligation",
            "NewMaplePairingResetClearObligation",
            "MaplePairingResetClearAdmission",
            "NewMaplePairingResetClearAdmission",
            "MapleResetClearSyncMaterializationContext",
            "MapleDeviceRegistrationOrdinarySyncContext",
            "MapleResetClearAdmissionMaterial",
            "MapleResetClearUnsignedMaterializationContext",
            "MapleResetClearUnsignedMaterial",
            "MaplePairingRevocationEvent",
            "NewMaplePairingRevocationEvent",
            "NewMaplePairingRequest",
            "MaplePairingCreateDeviceContext",
            "MaplePairingCreateMaterializationContext",
            "MaplePairingCreateMaterial",
            "MaplePairingApproval",
            "MaplePairingConfirmation",
            "MaplePairingRevocation",
            "MaplePairingRevocationContext",
            "MaplePairingRevocationMaterial",
            "MaplePairingRevocationAck",
            "MaplePairingOperationReceipt",
            "MaplePairingRevocationPageEntry",
            "MaplePairingRevocationPage",
        ],
    );
    assert_enums_do_not_derive_debug(
        "src/models/maple_pairing_db.rs",
        &pairing_models,
        &[
            "MapleDeviceRegistrationSyncMaterializationContext",
            "MapleDeviceRegistrationSyncMaterial",
            "MapleResetClearSource",
        ],
    );
    assert_structs_do_not_derive_debug(
        "src/db.rs",
        &db_source,
        &[
            "MaplePairingAuthenticatedProjectIdentity",
            "MaplePairingAuthorityInventoryHasher",
            "MaplePairingAuthorityDeviceSummary",
            "MaplePairingAuthorityPairSummary",
            "MaplePairingAuthorityOperationSummary",
            "MaplePairingAuthorityEventSummary",
        ],
    );
}

#[test]
fn maple_pairing_custom_debug_redacts_authority_material() {
    use crate::models::maple_pairing_db::{MaplePairing, MaplePairingRevocationEvent};

    let now = chrono::DateTime::from_timestamp_micros(2_000_000_000_000_000)
        .expect("test timestamp should be valid");
    let pair_request_mac = vec![241, 242, 243, 244, 245, 246];
    let pair_payload = vec![231, 232, 233, 234, 235, 236];
    let pair_record_mac = vec![221, 222, 223, 224, 225, 226];
    let pair_authorization_digest = vec![225, 224, 223, 222, 221, 220];
    let ticket_issuer = "ticket-issuer-debug-sentinel".to_string();
    let authorization_issuer = "authorization-issuer-debug-sentinel".to_string();
    let revocation_issuer = "revocation-issuer-debug-sentinel".to_string();
    let pairing = MaplePairing {
        id: 1,
        uuid: uuid::Uuid::from_u128(1),
        pairing_request_id: uuid::Uuid::from_u128(2),
        user_id: uuid::Uuid::from_u128(3),
        project_id: 4,
        lineage_id: 5,
        controller_maple_device_id: 6,
        host_maple_device_id: 7,
        direction: 1,
        pairing_incarnation: 8,
        state: 3,
        revision: 3,
        request_nonce_mac: pair_request_mac.clone(),
        revocation_stream_id: Some(uuid::Uuid::from_u128(9)),
        revocation_stream_generation: Some(1),
        pair_authorization_digest: Some(pair_authorization_digest.clone()),
        ticket_issuer_key_id: ticket_issuer.clone(),
        authorization_issuer_key_id: Some(authorization_issuer.clone()),
        revocation_issuer_key_id: Some(revocation_issuer.clone()),
        payload_version: 1,
        payload_enc: pair_payload.clone(),
        record_mac: pair_record_mac.clone(),
        created_at: now,
        expires_at: now,
        approved_at: Some(now),
        activated_at: Some(now),
        revoked_at: None,
        updated_at: now,
    };
    let pairing_debug = format!("{pairing:?}");
    assert_debug_omits_authority_values(
        "MaplePairing",
        &pairing_debug,
        &[
            format!("{pair_request_mac:?}"),
            format!("{pair_payload:?}"),
            format!("{pair_record_mac:?}"),
            format!("{pair_authorization_digest:?}"),
            ticket_issuer,
            authorization_issuer,
            revocation_issuer,
        ],
    );

    let event_payload = vec![211, 212, 213, 214, 215, 216];
    let event_digest = vec![201, 202, 203, 204, 205, 206];
    let event_record_mac = vec![191, 192, 193, 194, 195, 196];
    let event_issuer = "event-issuer-debug-sentinel".to_string();
    let event = MaplePairingRevocationEvent {
        id: 10,
        uuid: uuid::Uuid::from_u128(11),
        user_id: uuid::Uuid::from_u128(12),
        project_id: 13,
        recipient_host_maple_device_id: 14,
        revocation_stream_id: uuid::Uuid::from_u128(15),
        revocation_stream_generation: 1,
        issuer_sequence: 1,
        maple_pairing_id: 1,
        pairing_incarnation: 8,
        issuer_key_id: event_issuer.clone(),
        payload_version: 1,
        payload_enc: event_payload.clone(),
        event_digest: event_digest.clone(),
        record_mac: event_record_mac.clone(),
        created_at: now,
        acked_at: None,
    };
    let event_debug = format!("{event:?}");
    assert_debug_omits_authority_values(
        "MaplePairingRevocationEvent",
        &event_debug,
        &[
            format!("{event_payload:?}"),
            format!("{event_digest:?}"),
            format!("{event_record_mac:?}"),
            event_issuer,
        ],
    );
}

#[test]
fn maple_device_registration_receipt_debug_redacts_exact_sync_payload() {
    use crate::models::maple_devices::MapleDeviceRegistrationReceipt;

    let sync_payload = vec![181, 182, 183, 184, 185, 186];
    let receipt = MapleDeviceRegistrationReceipt {
        operation_id: uuid::Uuid::from_u128(20),
        registration_id: uuid::Uuid::from_u128(21),
        device_id: uuid::Uuid::from_u128(22),
        revision: 1,
        accepted_at: chrono::DateTime::from_timestamp_micros(2_000_000_000_000_000)
            .expect("test timestamp should be valid"),
        security_epoch: 1,
        response_kind: 1,
        sync_payload_version: 1,
        sync_payload: sync_payload.clone(),
    };

    assert_debug_omits_authority_values(
        "MapleDeviceRegistrationReceipt",
        &format!("{receipt:?}"),
        &[format!("{sync_payload:?}")],
    );
}

#[test]
fn maple_materializer_debug_redacts_registration_sync_and_create_artifacts() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/models/maple_pairing_db.rs"
    ));
    for implementation_name in [
        "impl std::fmt::Debug for MapleDeviceRegistrationSyncMaterial",
        "impl std::fmt::Debug for MaplePairingCreateMaterial",
    ] {
        let implementation = extract_function_body(source, implementation_name);
        assert!(
            implementation.contains("[redacted]"),
            "`{implementation_name}` must retain an explicit redaction marker"
        );
        for forbidden_field in [
            ".field(\"sync\"",
            ".field(\"sync_payload\"",
            ".field(\"signed_instruction_payload\"",
            ".field(\"request_ticket\"",
            ".field(\"response\"",
            ".field(\"request_nonce_mac\"",
            ".field(\"payload_enc\"",
            ".field(\"payload_digest\"",
            ".field(\"receipt_enc\"",
            ".field(\"receipt_digest\"",
        ] {
            assert!(
                !implementation.contains(forbidden_field),
                "`{implementation_name}` must redact authority field `{forbidden_field}`"
            );
        }
    }
}

#[test]
fn maple_reset_clear_wire_contract_binds_the_complete_scope_and_redacts_debug() {
    let wire_source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/models/maple_pairings.rs"
    ));

    for required_domain in [
        "os.maple-reset-clear-admission-set.v1",
        "os.maple-reset-clear-instruction-material.v1",
        "os.maple-reset-clear-chain.v1",
        "os.maple-reset-clear-required.v1",
    ] {
        assert!(
            wire_source.contains(required_domain),
            "reset-clear transcripts must retain domain `{required_domain}`"
        );
    }
    for required_type in [
        "pub enum MapleResetClearScopeV1",
        "pub enum MapleRevocationSyncStatusV1",
        "pub struct MapleResetClearRequiredV1",
        "pub enum MapleRevocationStreamEventV1",
        "pub struct MapleRevocationSyncV1",
        "pub struct MapleResetClearAdmissionLeafV1",
    ] {
        assert!(
            wire_source.contains(required_type),
            "reset-clear wire model must retain `{required_type}`"
        );
    }

    let admission_transcript =
        extract_function_body(wire_source, "pub fn reset_clear_admission_set_transcript(");
    for required_pattern in [
        "MAPLE_RESET_CLEAR_MAX_ADMISSIONS",
        "canonical.sort_unstable()",
        "canonical.windows(2)",
        ".append_uuid(leaf.pair_id)",
        ".append_u64(leaf.pairing_incarnation)",
        ".append_bytes(&leaf.pair_authorization_digest)",
    ] {
        assert!(
            admission_transcript.contains(required_pattern),
            "private admission-set transcript must contain `{required_pattern}`"
        );
    }

    let material_transcript = extract_function_body(
        wire_source,
        "pub fn reset_clear_instruction_material_transcript(",
    );
    for required_pattern in [
        ".append_uuid(instruction.event_id)",
        ".append_uuid(instruction.reset_id)",
        ".append_u64(instruction.reset_generation)",
        ".append_u64(instruction.cumulative_reset_count)",
        ".append_u64(instruction.source_security_epoch)",
        ".append_u64(instruction.security_epoch)",
        ".append_uuid(instruction.subject_account_id)",
        ".append_uuid(instruction.subject_project_id)",
        ".append_uuid(instruction.recipient_host_registration_id)",
        "append_device_claim(&mut transcript, &instruction.host)",
        ".append_uuid(instruction.source_revocation_stream_id)",
        ".append_uuid(instruction.revocation_stream_id)",
        ".append_u16(instruction.admission_count)",
        ".append_bytes(&admission_set_digest)",
    ] {
        assert!(
            material_transcript.contains(required_pattern),
            "reset-clear instruction material must bind `{required_pattern}`"
        );
    }
    let predecessor = extract_function_body(wire_source, "fn reset_clear_predecessor(");
    assert_patterns_in_order(
        predecessor,
        &[
            "instruction.previous_reset_clear_event_id",
            "instruction.previous_instruction_material_digest.as_deref()",
            "instruction.previous_chain_digest.as_deref()",
            "Some(event_id), Some(material_digest), Some(chain_digest)",
        ],
    );
    let chain = extract_function_body(wire_source, "pub fn reset_clear_chain_transcript(");
    assert_patterns_in_order(
        chain,
        &[
            "let predecessor = reset_clear_predecessor(instruction)?",
            ".append_bool(predecessor.is_some())",
            ".append_bytes(&previous_chain_digest)",
            ".append_uuid(event_id)",
            ".append_bytes(&previous_material_digest)",
            ".append_uuid(instruction.reset_id)",
            ".append_uuid(instruction.event_id)",
            ".append_u64(instruction.reset_generation)",
            ".append_u64(instruction.cumulative_reset_count)",
        ],
    );

    let instruction_impl = extract_function_body(wire_source, "impl MapleResetClearRequiredV1");
    for required_method in [
        "pub fn validate(",
        "pub fn verify(",
        "pub fn event_digest(",
        "pub fn verify_against_checkpoint(",
        "pub fn verify_discovered_head_against_checkpoint(",
        "pub fn verify_direct_successor(",
    ] {
        assert!(
            instruction_impl.contains(required_method),
            "reset-clear artifact must retain `{required_method}`"
        );
    }
    let direct_successor =
        extract_function_body(instruction_impl, "pub fn verify_direct_successor(");
    for required_pattern in [
        "self.previous_reset_clear_event_id != Some(predecessor.event_id)",
        "linked_material != Some(predecessor_material)",
        "linked_chain != Some(predecessor_chain)",
        "predecessor.cumulative_reset_count.checked_add(1)",
        "predecessor.security_epoch != self.source_security_epoch",
        "predecessor.recipient_host_registration_id",
        "predecessor.host != self.host",
    ] {
        assert!(
            direct_successor.contains(required_pattern),
            "direct-successor verification must bind `{required_pattern}`"
        );
    }
    assert!(
        !direct_successor.contains("device_claim_is_same_identity_at_or_before"),
        "a missed-reset successor must preserve the exact full retained host claim, not accept an endpoint refresh"
    );
    for required_regression in [
        "A missed-reset successor carries the exact retained host claim.",
        "changed_host.host.endpoint_epoch",
        "verify_direct_successor(predecessor, &changed_checkpoint, &keyset)",
        ".is_err()",
    ] {
        assert!(
            wire_source.contains(required_regression),
            "wire regressions must reject a changed successor host claim via `{required_regression}`"
        );
    }

    let sync_validation = extract_function_body(wire_source, "impl MapleRevocationSyncV1");
    for required_pattern in [
        "MapleRevocationSyncStatusV1::Ready, None",
        "MapleRevocationSyncStatusV1::RevocationsPending, None",
        "MapleRevocationSyncStatusV1::ResetClearRequired, Some(instruction)",
        "instruction.verify_against_checkpoint(&self.stream_checkpoint, keyset)",
        "pub fn verify_against_registration(",
    ] {
        assert!(
            sync_validation.contains(required_pattern),
            "typed revocation sync must retain `{required_pattern}`"
        );
    }

    let event_enum = extract_function_body(wire_source, "pub enum MapleRevocationStreamEventV1");
    let event_enum_start = wire_source
        .find("pub enum MapleRevocationStreamEventV1")
        .expect("typed revocation event enum should exist");
    let event_attributes = &wire_source[event_enum_start.saturating_sub(100)..event_enum_start];
    for required_pattern in [
        "#[serde(rename = \"pair_revocation\")]",
        "#[serde(rename = \"reset_clear_required\")]",
    ] {
        assert!(
            event_enum.contains(required_pattern),
            "revocation event representation must contain `{required_pattern}`"
        );
    }
    assert!(
        event_attributes.contains("#[serde(tag = \"event_type\", content = \"event\")]"),
        "revocation events must remain adjacently tagged as `event_type,event`"
    );

    let list_response = extract_function_body(
        wire_source,
        "pub struct ListMaplePairingRevocationsResponse",
    );
    assert!(
        list_response.contains("pub revocation_sync: MapleRevocationSyncV1")
            && list_response.contains("pub events: Vec<MapleRevocationStreamEventV1>"),
        "the existing revocation page must carry typed sync and events"
    );
    for forbidden_public_leaf in [
        "MapleResetClearAdmissionLeafV1",
        "pair_authorization_digest",
        "record_mac",
        "authority_scope_digest",
        "lookup_digest",
    ] {
        assert!(
            !list_response.contains(forbidden_public_leaf),
            "revocation-list response must not expose private leaf/authentication field `{forbidden_public_leaf}`"
        );
    }

    let list_verifier =
        extract_function_body(wire_source, "impl ListMaplePairingRevocationsResponse");
    for required_progression_gate in [
        "self.events.len() > usize::from(request.effective_limit()?)",
        "request.after_issuer_sequence.checked_add(1) != Some(first.issuer_sequence())",
        "self.events.is_empty()",
        ".last_issued_issuer_sequence",
        "request.after_issuer_sequence",
        ".last_acked_issuer_sequence",
    ] {
        assert!(
            list_verifier.contains(required_progression_gate),
            "revocation paging must retain `{required_progression_gate}`"
        );
    }

    let list_verify = extract_function_body(list_verifier, "pub fn verify(");
    for required_reset_clear_gate in [
        "match self.revocation_sync.reset_clear_instruction.as_ref()",
        "Some(instruction)",
        "Some(MapleRevocationStreamEventV1::ResetClearRequired(event))",
        "if event == instruction",
        "None if self.events.iter().all(|event|",
        "event.issuer_sequence() <= checkpoint.last_acked_issuer_sequence",
        "None =>",
        "if let Some(instruction) = self.revocation_sync.reset_clear_instruction.as_ref()",
        "instruction.event_digest()?",
        ".event_digest()?",
    ] {
        assert!(
            list_verify.contains(required_reset_clear_gate),
            "revocation-page verification must retain reset-clear gate `{required_reset_clear_gate}`"
        );
    }
    assert!(
        list_verify.matches("\"reset_clear_stream_event\"").count() >= 3,
        "missing, inconsistent, or digest-mismatched pending reset-clear instructions must fail with one stable field error"
    );

    for required_regression in [
        "fn historical_reset_event_verifies_after_ack_and_after_later_events()",
        "fn unacked_reset_event_requires_exact_pending_sync_instruction()",
        "fn revocation_stream_event_json_starts_with_frozen_tag_then_content()",
        "signed_genesis_reset_and_checkpoint(&value, 1, 1)",
        "signed_genesis_reset_and_checkpoint(&value, 2, 1)",
        r#"{\"event_type\":\"reset_clear_required\",\"event\":{"#,
        r#"{\"event_type\":\"pair_revocation\",\"event\":{"#,
    ] {
        assert!(
            wire_source.contains(required_regression),
            "wire regressions must retain `{required_regression}`"
        );
    }

    let instruction_debug = extract_function_body(
        wire_source,
        "impl std::fmt::Debug for MapleResetClearRequiredV1",
    );
    for permitted_summary in [
        "artifact_version",
        "reset_generation",
        "cumulative_reset_count",
        "source_security_epoch",
        "security_epoch",
        "admission_count",
        "has_previous_reset",
        "[redacted]",
    ] {
        assert!(
            instruction_debug.contains(permitted_summary),
            "reset-clear Debug summary must retain `{permitted_summary}`"
        );
    }
    for forbidden_debug_field in [
        ".field(\"event_id\"",
        ".field(\"reset_id\"",
        ".field(\"subject_account_id\"",
        ".field(\"subject_project_id\"",
        ".field(\"recipient_host_registration_id\"",
        ".field(\"admission_set_digest\"",
        ".field(\"instruction_material_digest\"",
        ".field(\"chain_digest\"",
        ".field(\"issuer_key_id\"",
        ".field(\"issuer_signature\"",
    ] {
        assert!(
            !instruction_debug.contains(forbidden_debug_field),
            "reset-clear Debug must redact `{forbidden_debug_field}`"
        );
    }
    let leaf_debug = extract_function_body(
        wire_source,
        "impl std::fmt::Debug for MapleResetClearAdmissionLeafV1",
    );
    assert!(
        leaf_debug.contains("[redacted]")
            && !leaf_debug.contains(".field(\"pair_id\"")
            && !leaf_debug.contains(".field(\"pairing_incarnation\"")
            && !leaf_debug.contains(".field(\"pair_authorization_digest\""),
        "private admission-leaf Debug must reveal no leaf material"
    );

    for debug_impl in [
        "impl std::fmt::Debug for AckMaplePairingRevocationRequest",
        "impl std::fmt::Debug for MapleRevocationStreamEventV1",
        "impl std::fmt::Debug for ListMaplePairingRevocationsResponse",
        "impl std::fmt::Debug for AckMaplePairingRevocationResponse",
        "impl std::fmt::Debug for MapleRevocationStreamCheckpointV1",
        "impl std::fmt::Debug for MapleRevocationSyncV1",
    ] {
        let implementation = extract_function_body(wire_source, debug_impl);
        assert!(
            implementation.contains("[redacted]"),
            "`{debug_impl}` must retain an explicit redaction marker"
        );
        for forbidden_field in [
            ".field(\"operation_id\"",
            ".field(\"query_id\"",
            ".field(\"event_id\"",
            ".field(\"asserted_account_id\"",
            ".field(\"asserted_project_id\"",
            ".field(\"subject_account_id\"",
            ".field(\"subject_project_id\"",
            ".field(\"host_registration_id\"",
            ".field(\"host\"",
            ".field(\"revocation_stream_id\"",
            ".field(\"event_digest\"",
            ".field(\"issuer_key_id\"",
            ".field(\"issuer_signature\"",
            ".field(\"stream_checkpoint\"",
            ".field(\"reset_clear_instruction\"",
            ".field(\"revocation_sync\"",
            ".field(\"events\"",
        ] {
            assert!(
                !implementation.contains(forbidden_field),
                "`{debug_impl}` must redact authority field `{forbidden_field}`"
            );
        }
    }
    assert!(
        wire_source.contains("fn v6_authority_debug_is_fully_redacted()"),
        "runtime wire Debug sentinels must cover the full v6 authority surface"
    );
}

#[test]
fn maple_reset_clear_ack_preparation_consumes_one_durable_clear_proof() {
    let wire_source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/models/maple_pairings.rs"
    ));
    let integrity_contract = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/maple-pairing-authority-integrity.md"
    ));

    let ack_start = wire_source
        .find("pub struct AckMaplePairingRevocationRequest")
        .expect("prepared ACK wire request must exist");
    let ack_attributes = &wire_source[ack_start.saturating_sub(120)..ack_start];
    assert!(
        ack_attributes.contains("#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]"),
        "only an already-prepared exact ACK wire request needs Clone for retry"
    );
    let ack_transcript = extract_function_body(
        wire_source,
        "pub fn ack_pairing_revocation_request_transcript(",
    );
    for exact_request_binding in [
        ".append_uuid(request.operation_id)",
        ".append_uuid(request.host_registration_id)",
        ".append_uuid(request.revocation_stream_id)",
        ".append_u64(request.revocation_stream_generation)",
        ".append_uuid(request.event_id)",
        ".append_u64(request.issuer_sequence)",
        ".append_bytes(&event_digest)",
        ".append_u64(request.expected_previous_issuer_sequence)",
    ] {
        assert!(
            ack_transcript.contains(exact_request_binding),
            "prepared ACK transcript must retain `{exact_request_binding}`"
        );
    }

    // The SDK implementation lives in a sibling repository, so this backend
    // gate pins the cross-repository typestate contract in the integrity spec.
    for single_use_contract in [
        "linear, non-`Clone` proof wrapper",
        "consumes that proof exactly once",
        "one exact operation identifier",
        "The proof cannot prepare a second operation.",
        "prepared acknowledgement request may be cloned and retried",
        "retry must be byte-identical",
    ] {
        assert!(
            integrity_contract.contains(single_use_contract),
            "reset-clear SDK contract must retain `{single_use_contract}`"
        );
    }
}

#[test]
fn maple_reset_clear_vectors_cover_capacity_chaining_epoch_and_replay_boundaries() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/maple_pairing_v1_vectors.json"
    )))
    .expect("Maple pairing v1 vectors should remain valid JSON");

    let admission_vectors = fixture
        .get("reset_clear_admission_set_vectors")
        .expect("reset-clear admission vectors should exist");
    assert_eq!(
        admission_vectors["empty"]["leaves"]
            .as_array()
            .expect("empty admission vector should contain leaves")
            .len(),
        0,
        "the frozen vectors must cover the empty admission set"
    );
    let maximum_admissions = admission_vectors["max_128"]["leaves"]
        .as_array()
        .expect("maximum admission vector should contain leaves");
    assert_eq!(
        maximum_admissions.len(),
        128,
        "the frozen vectors must cover the complete per-obligation admission limit"
    );
    assert!(
        maximum_admissions.iter().all(|leaf| {
            leaf.get("pair_id")
                .and_then(serde_json::Value::as_str)
                .is_some()
                && leaf
                    .get("pairing_incarnation")
                    .and_then(serde_json::Value::as_u64)
                    .is_some()
                && leaf
                    .get("pair_authorization_digest")
                    .and_then(serde_json::Value::as_str)
                    .is_some()
        }),
        "every private admission vector must bind the exact pair identity and authorization digest"
    );

    let chain = fixture["reset_clear_three_reset_chain"]
        .as_array()
        .expect("three-reset chain vector should be an array");
    assert_eq!(
        chain.len(),
        3,
        "the frozen vectors must cover a three-reset chain"
    );
    let mut event_ids = BTreeSet::new();
    let mut reset_ids = BTreeSet::new();
    for (index, entry) in chain.iter().enumerate() {
        let instruction = &entry["instruction"];
        let generation = u64::try_from(index + 1).expect("test index should fit u64");
        event_ids.insert(
            instruction["event_id"]
                .as_str()
                .expect("reset chain event ID should be a string"),
        );
        reset_ids.insert(
            instruction["reset_id"]
                .as_str()
                .expect("reset chain reset ID should be a string"),
        );
        assert_eq!(instruction["reset_generation"].as_u64(), Some(generation));
        assert_eq!(
            instruction["cumulative_reset_count"].as_u64(),
            Some(generation)
        );
        assert_eq!(
            instruction["source_security_epoch"].as_u64(),
            Some(generation)
        );
        assert_eq!(
            instruction["security_epoch"].as_u64(),
            generation.checked_add(1)
        );
        assert!(entry["instruction_material_digest"].as_str().is_some());
        assert!(entry["chain_digest"].as_str().is_some());
        assert!(entry["event_digest"].as_str().is_some());
        if let Some(predecessor) = index.checked_sub(1) {
            assert_eq!(
                instruction["previous_reset_clear_event_id"],
                chain[predecessor]["instruction"]["event_id"]
            );
            assert_eq!(
                instruction["previous_instruction_material_digest"],
                chain[predecessor]["instruction_material_digest"]
            );
            assert_eq!(
                instruction["previous_chain_digest"],
                chain[predecessor]["chain_digest"]
            );
        } else {
            assert!(instruction["previous_reset_clear_event_id"].is_null());
            assert!(instruction["previous_instruction_material_digest"].is_null());
            assert!(instruction["previous_chain_digest"].is_null());
        }
    }
    assert_eq!(
        event_ids.len(),
        3,
        "every host obligation must have a fresh event ID"
    );
    assert_eq!(
        reset_ids.len(),
        3,
        "every successive reset must have a fresh reset ID"
    );
    assert_eq!(
        chain
            .iter()
            .map(|entry| entry["instruction"]["admission_count"].as_u64())
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(128)],
        "recursive vectors must cover empty, ordinary, and maximum admission sets"
    );

    let epoch_outcomes = fixture["security_epoch_outcome_vectors"]
        .as_array()
        .expect("security-epoch outcome matrix should be an array");
    let names = epoch_outcomes
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect::<BTreeSet<_>>();
    for required_case in [
        "known_epoch_stale",
        "known_epoch_ahead",
        "current_epoch_pending_reset_dominates",
        "fresh_current_epoch_after_ack",
    ] {
        assert!(
            names.contains(required_case),
            "security-epoch vectors must retain `{required_case}`"
        );
    }

    let tombstones = &fixture["registration_operation_tombstone_vectors"];
    assert_eq!(
        tombstones["storage"]["operation_lookup_digest"].as_str(),
        Some("hmac_only")
    );
    assert_eq!(
        tombstones["storage"]["raw_operation_id_stored"].as_bool(),
        Some(false)
    );
    assert_eq!(
        tombstones["exact_old_request"].as_str(),
        Some("frozen_ready_replay")
    );
    assert_eq!(
        tombstones["changed_request_same_operation_lookup"].as_str(),
        Some("conflict")
    );

    let structural = &fixture["wire_structural_assertions"];
    assert_eq!(
        structural["reset_event_json_key_order"],
        serde_json::json!(["event_type", "event"])
    );
    assert_eq!(
        structural["retained_admission_leaves_cross_wire"].as_bool(),
        Some(false)
    );
    assert_eq!(
        structural["reset_admission_public_shape"],
        serde_json::json!(["admission_count", "admission_set_digest"])
    );
    assert_eq!(
        fixture["maple_pairing_reset_clear_required_error"],
        serde_json::json!({
            "status": 409,
            "message": "Maple remote access must be cleared on the host before this operation can continue.",
            "code": "MaplePairingResetClearRequired"
        }),
        "the frozen vectors must retain the exact sanitized pending-reset error"
    );

    for response_name in [
        "register_device_response_ready",
        "register_device_response_revocations_pending",
        "register_device_response_reset_clear_required",
    ] {
        let response = &fixture[response_name];
        assert_eq!(
            response["security_epoch"], response["revocation_sync"]["security_epoch"],
            "`{response_name}` must bind one exact account epoch across registration and sync"
        );
    }
    assert_eq!(
        fixture["register_device_request_epoch_1"]["known_security_epoch"].as_u64(),
        Some(1)
    );
    assert_eq!(
        fixture["register_device_request_epoch_4"]["known_security_epoch"].as_u64(),
        Some(4)
    );
}

#[test]
fn maple_reset_clear_reuses_the_existing_routes_and_registration_epoch_fields() {
    let device_routes = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/web/maple_devices.rs"
    ));
    let pairing_routes = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/web/maple_pairings.rs"
    ));

    let pairing_router = extract_function_body(pairing_routes, "pub fn router(");
    assert_eq!(
        pairing_router.matches(".route(").count(),
        8,
        "reset-clear must reuse the existing revocation list/ACK surface, not add a ninth pairing route"
    );
    for required_route in [
        "/protected/maple/pairings/request",
        "/protected/maple/pairings/list",
        "/protected/maple/pairings/status",
        "/protected/maple/pairings/approve",
        "/protected/maple/pairings/confirm",
        "/protected/maple/pairings/revoke",
        "/protected/maple/pairings/revocations/list",
        "/protected/maple/pairings/revocations/ack",
    ] {
        assert!(
            pairing_router.contains(required_route),
            "pairing router must retain `{required_route}`"
        );
    }
    assert!(
        !pairing_router.contains("reset-clear") && !pairing_router.contains("reset_clear"),
        "reset-clear must not create a dedicated route"
    );

    let registration_request =
        extract_function_body(device_routes, "pub struct RegisterMapleDeviceRequest");
    assert!(
        registration_request.contains("pub known_security_epoch: u64"),
        "registration request must bind the client-known account security epoch"
    );
    let registration_response =
        extract_function_body(device_routes, "pub struct RegisterMapleDeviceResponse");
    for required_field in [
        "pub security_epoch: u64",
        "pub revocation_sync: MapleRevocationSyncV1",
    ] {
        assert!(
            registration_response.contains(required_field),
            "registration response must retain `{required_field}`"
        );
    }
    let response_debug = extract_function_body(
        device_routes,
        "impl std::fmt::Debug for RegisterMapleDeviceResponse",
    );
    assert!(
        response_debug.contains("[redacted]")
            && !response_debug.contains(".field(\"operation_id\"")
            && !response_debug.contains(".field(\"registration_id\"")
            && !response_debug.contains(".field(\"device_id\"")
            && !response_debug.contains(".field(\"revocation_sync\""),
        "registration response Debug must redact stable device IDs and the exact sync payload"
    );
    assert!(
        device_routes.contains("fn registration_response_debug_redacts_all_authority_material()"),
        "registration response must retain runtime Debug-redaction sentinels"
    );
    let list_response = extract_function_body(device_routes, "pub struct ListMapleDevicesResponse");
    assert!(
        list_response.contains("pub security_epoch: u64"),
        "device-list bootstrap must publish the account security epoch from its authenticated snapshot"
    );

    let registration_transcript =
        extract_function_body(device_routes, "pub(crate) fn registration_transcript(");
    assert_patterns_in_order(
        registration_transcript,
        &[
            ".append_uuid(request.asserted_account_id)",
            ".append_uuid(request.asserted_project_id)",
            ".append_u64(request.known_security_epoch)",
            ".append_uuid(request.operation_id)",
        ],
    );
    let fixture_regression = extract_function_body(
        device_routes,
        "fn frozen_security_epoch_registration_vectors_match_wire_contract()",
    );
    for required_anchor in [
        "register_device_request_epoch_1",
        "register_device_request_epoch_4",
        "registration_transcript(",
        "sha256_digest(&transcript)",
        ".verify_strict(&transcript",
        "expected_revision\\\":5,\\\"known_security_epoch",
        "list_devices_response_security_epoch",
        "assert_eq!(list_response.security_epoch, 4)",
    ] {
        assert!(
            fixture_regression.contains(required_anchor),
            "device wire fixture regression must retain `{required_anchor}`"
        );
    }
    let registration_error_map =
        extract_function_body(device_routes, "fn map_maple_registration_db_error(");
    assert!(
        !registration_error_map.contains("MaplePairingResetClearRequired"),
        "current-epoch host registration under pending reset must persist and return the exact signed sync, not lose recovery material behind the generic pending-reset error"
    );
}

#[test]
fn maple_reset_clear_required_is_one_exact_sanitized_public_conflict() {
    let main_source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs"));
    let db_source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/db.rs"));
    let pairing_routes = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/web/maple_pairings.rs"
    ));
    let message =
        "Maple remote access must be cleared on the host before this operation can continue.";

    let api_errors = extract_function_body(main_source, "pub enum ApiError");
    assert!(
        api_errors.contains(&format!(
            "#[error(\"{message}\")]\n    MaplePairingResetClearRequired"
        )),
        "the public pending-reset error must retain its exact sanitized message"
    );
    let db_errors = extract_function_body(db_source, "pub enum DBError");
    assert!(
        db_errors.contains("MaplePairingResetClearRequired"),
        "the database layer must retain the typed pending-reset error"
    );

    let api_error_impl = extract_function_body(main_source, "impl ApiError");
    assert!(
        api_error_impl.contains(
            "Self::MaplePairingResetClearRequired => Some(\"MaplePairingResetClearRequired\")"
        ),
        "pending reset-clear must retain its exact machine code"
    );
    let response_impl = extract_function_body(main_source, "impl IntoResponse for ApiError");
    assert!(
        response_impl.contains("ApiError::MaplePairingResetClearRequired => StatusCode::CONFLICT"),
        "pending reset-clear must remain HTTP 409"
    );
    let conversion = extract_function_body(main_source, "impl From<DBError> for ApiError");
    assert!(
        conversion.contains(
            "DBError::MaplePairingResetClearRequired => ApiError::MaplePairingResetClearRequired"
        ),
        "the typed database gate must map to the exact public error"
    );
    let pairing_error_map = extract_function_body(pairing_routes, "fn map_pairing_db_error(");
    assert!(
        pairing_error_map.contains(
            "DBError::MaplePairingResetClearRequired => ApiError::MaplePairingResetClearRequired"
        ),
        "pairing routes must preserve the typed pending-reset error"
    );
    assert!(
        pairing_routes
            .contains("fn unresolved_reset_clear_maps_to_the_sanitized_typed_api_error()"),
        "pairing route error mapping must retain its focused regression"
    );

    let exact_json = extract_function_body(
        main_source,
        "async fn reset_clear_required_error_is_exact_and_contains_no_authority_material(",
    );
    for required_pattern in [
        "StatusCode::CONFLICT",
        "\"status\": 409",
        message,
        "\"code\": \"MaplePairingResetClearRequired\"",
        "maple_pairing_reset_clear_required_error",
        "event_id",
        "reset_id",
        "revocation_stream",
        "issuer_key",
        "signature",
        "digest",
        "sync_payload",
        "assert!(!encoded.contains(forbidden))",
    ] {
        assert!(
            exact_json.contains(required_pattern),
            "pending reset-clear public JSON regression must retain `{required_pattern}`"
        );
    }
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
fn web_routes_remain_jwt_authenticated_and_e2ee_wrapped() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let main_source = fs::read_to_string(manifest_dir.join("src/main.rs"))
        .expect("main source should be readable");
    let web_source = fs::read_to_string(manifest_dir.join("src/web/web_routes.rs"))
        .expect("web route source should be readable");

    assert!(
        main_source.contains(
            "web_routes(app_state.clone())\n                .route_layer(from_fn_with_state(app_state.clone(), validate_jwt))"
        ),
        "provider-neutral web routes must remain behind user JWT validation"
    );

    let router_body = extract_function_body(&web_source, "pub fn router");
    assert_eq!(
        router_body.matches("decrypt_request::<Value>").count(),
        2,
        "both web routes must decrypt request bodies through session E2EE"
    );
    assert_eq!(
        web_source
            .matches("Extension(user): Extension<User>")
            .count(),
        2,
        "both web handlers must require the JWT-injected user"
    );
    for handler in ["async fn search_web", "async fn extract_web"] {
        let handler_body = extract_function_body(&web_source, handler);
        assert!(
            handler_body.contains("encrypt_response(&state, &session_id, &response)"),
            "{handler} must encrypt successful responses through session E2EE"
        );
    }
}

#[test]
fn maple_device_routes_remain_jwt_authenticated_and_e2ee_wrapped() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let main_source = fs::read_to_string(manifest_dir.join("src/main.rs"))
        .expect("main source should be readable");
    let route_source = fs::read_to_string(manifest_dir.join("src/web/maple_devices.rs"))
        .expect("Maple device route source should be readable");

    assert!(
        main_source.contains(
            "maple_devices_routes(app_state.clone())\n                .route_layer(from_fn_with_state(app_state.clone(), validate_jwt))"
        ),
        "Maple device routes must remain behind user JWT validation"
    );

    let router_body = extract_function_body(&route_source, "pub fn router");
    for encrypted_middleware in [
        "decrypt_request_bounded::<",
        "RegisterMapleDeviceRequest",
        "MAX_REGISTRATION_ENCRYPTED_BYTES",
        "MAX_REGISTRATION_PLAINTEXT_BYTES",
        "decrypt_request::<()>",
    ] {
        assert!(
            router_body.contains(encrypted_middleware),
            "Maple device router must contain `{encrypted_middleware}`"
        );
    }
    for handler in ["async fn register_device", "async fn list_devices"] {
        let handler_start = route_source
            .find(handler)
            .unwrap_or_else(|| panic!("Maple device route should contain `{handler}`"));
        let handler_signature = &route_source[handler_start
            ..route_source[handler_start..]
                .find("{")
                .map(|offset| handler_start + offset)
                .expect("Maple device handler should have a body")];
        let handler_body = extract_function_body(&route_source, handler);
        assert!(
            handler_signature.contains("Extension(user): Extension<User>"),
            "{handler} must require the JWT-injected user"
        );
        assert!(
            handler_signature.contains("Extension(auth_context): Extension<AuthContext>"),
            "{handler} must carry the verified auth context into its database transaction"
        );
        assert!(
            handler_body.contains("encrypt_response("),
            "{handler} must encrypt successful responses through session E2EE"
        );
    }
}

#[test]
fn maple_pairing_authority_bootstrap_precedes_issuer_reference_audit() {
    let main_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let contents = fs::read_to_string(main_source).expect("main source should be readable");
    let build_body = extract_function_body(&contents, "pub async fn build(self)");

    assert_patterns_in_order(
        build_body,
        &[
            "keyset.validate()",
            "let authority_issuer_keyset = self.maple_pairing_issuer_keyset.clone()",
            "bootstrap_or_audit_maple_pairing_authority(",
            "authority_issuer_keyset.as_deref()",
            ".await?",
            "Err(DBError::MaplePairingIssuerConfigurationConflict)",
            "audit_maple_pairing_issuer_key_references(",
            ".await??",
        ],
    );
}

#[test]
fn maple_pairing_issuer_registry_is_authenticated_append_only_and_process_pinned() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB models should be readable");
    let schema = fs::read_to_string(manifest_dir.join("src/models/schema.rs"))
        .expect("Diesel schema should be readable");
    let up = fs::read_to_string(
        manifest_dir.join("migrations/2026-08-13-120000_maple_pairings_v1/up.sql"),
    )
    .expect("Maple pairing migration should be readable");
    let down = fs::read_to_string(
        manifest_dir.join("migrations/2026-08-13-120000_maple_pairings_v1/down.sql"),
    )
    .expect("Maple pairing rollback should be readable");

    for required_model_field in [
        "pub issuer_key_inventory_digest: Vec<u8>",
        "pub issuer_key_count: i64",
        "pub(crate) struct MaplePairingIssuerKey",
        "pub(crate) struct NewMaplePairingIssuerKey",
        "pub key_id: String",
        "pub global_singleton: bool",
        "pub algorithm: String",
        "pub public_key_digest: Vec<u8>",
        "pub record_mac: Vec<u8>",
    ] {
        assert!(
            models.contains(required_model_field),
            "issuer registry model must retain `{required_model_field}`"
        );
    }
    for required_schema_field in [
        "issuer_key_inventory_digest -> Bytea",
        "issuer_key_count -> Int8",
        "maple_pairing_issuer_keys (key_id)",
        "global_singleton -> Bool",
        "algorithm -> Text",
        "public_key_digest -> Bytea",
        "record_mac -> Bytea",
    ] {
        assert!(
            schema.contains(required_schema_field),
            "issuer registry Diesel schema must retain `{required_schema_field}`"
        );
    }

    let global_mac = extract_function_body(&db, "fn maple_pairing_authority_global_head_mac(");
    assert_patterns_in_order(
        global_mac,
        &[
            ".append_bytes(&head.org_inventory_digest)",
            ".append_i64(head.org_count)",
            ".append_bytes(&head.issuer_key_inventory_digest)",
            ".append_i64(head.issuer_key_count)",
            ".append_i64(head.revision)",
        ],
    );

    let issuer_mac =
        extract_function_body(&db, "fn maple_pairing_issuer_key_record_mac_for_parts(");
    assert_patterns_in_order(
        issuer_mac,
        &[
            "MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_DOMAIN",
            ".append_str(key_id)",
            ".append_bool(global_singleton)",
            ".append_str(algorithm)",
            ".append_bytes(public_key_digest)",
            ".append_i64(created_at.timestamp_micros())",
            "MAPLE_PAIRING_ISSUER_KEY_RECORD_MAC_KEY_INFO",
        ],
    );
    let issuer_validation = extract_function_body(&db, "fn validate_maple_pairing_issuer_key(");
    for required_gate in [
        "maple_pairing_issuer_key_id_is_valid(&row.key_id)",
        "!row.global_singleton",
        "row.algorithm != \"ed25519\"",
        "row.public_key_digest.len() != 32",
        "row.record_mac.len() != 32",
        "maple_pairing_authority_mac_matches(&expected, &row.record_mac)",
    ] {
        assert!(
            issuer_validation.contains(required_gate),
            "issuer registry row validation must retain `{required_gate}`"
        );
    }

    let configured_inventory = extract_function_body(
        &db,
        "fn maple_pairing_issuer_key_inventory_digest_from_fingerprints(",
    );
    for required_gate in [
        "fingerprints.len() > MAPLE_PAIRING_MAX_ISSUER_KEYS",
        "MAPLE_PAIRING_ISSUER_KEY_INVENTORY_DOMAIN",
        "os.maple-pair-issuer-key-inventory-header.v1",
        "previous >= fingerprint.key_id.as_str()",
        "os.maple-pair-issuer-key-inventory-leaf.v1",
        ".append_str(&fingerprint.key_id)",
        ".append_str(fingerprint.algorithm.as_wire())",
        ".append_bytes(&fingerprint.public_key_digest)",
    ] {
        assert!(
            configured_inventory.contains(required_gate),
            "configured issuer inventory must retain `{required_gate}`"
        );
    }
    let stored_inventory =
        extract_function_body(&db, "fn compute_maple_pairing_issuer_key_inventory(");
    assert_patterns_in_order(
        stored_inventory,
        &[
            "maple_pairing_issuer_keys::table",
            ".count()",
            "MAPLE_PAIRING_ISSUER_KEY_INVENTORY_DOMAIN",
            "os.maple-pair-issuer-key-inventory-header.v1",
            ".filter(maple_pairing_issuer_keys::key_id.gt(&cursor))",
            ".order(maple_pairing_issuer_keys::key_id.asc())",
            ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
            "validate_maple_pairing_issuer_key(enclave_key, &row)?",
            "os.maple-pair-issuer-key-inventory-leaf.v1",
            ".append_str(&row.key_id)",
            ".append_str(&row.algorithm)",
            ".append_bytes(&row.public_key_digest)",
            "if seen != count",
        ],
    );
    let leaf_emptiness =
        extract_function_body(&db, "fn maple_pairing_authority_leaf_tables_are_empty(");
    assert!(
        leaf_emptiness.contains("maple_pairing_issuer_keys::table"),
        "authority bootstrap must treat any pre-activation issuer registry row as retained state"
    );

    let verify_inventory =
        extract_function_body(&db, "fn verify_maple_pairing_issuer_key_inventory(");
    for required_gate in [
        "compute_maple_pairing_issuer_key_inventory(conn, enclave_key)?",
        "global.issuer_key_count != issuer_key_count",
        "&issuer_key_inventory_digest",
        "&global.issuer_key_inventory_digest",
    ] {
        assert!(
            verify_inventory.contains(required_gate),
            "authenticated issuer inventory verification must retain `{required_gate}`"
        );
    }
    let reconcile = extract_function_body(&db, "fn reconcile_maple_pairing_issuer_key_registry(");
    assert_patterns_in_order(
        reconcile,
        &[
            "maple_pairing_issuer_key_inventory_digest_from_fingerprints(configured)?",
            "maple_pairing_issuer_keys::table",
            ".order(maple_pairing_issuer_keys::key_id.asc())",
            ".for_update()",
            "existing.len() > configured.len()",
            "validate_maple_pairing_issuer_key(enclave_key, row)?",
            ".binary_search_by(|candidate| candidate.key_id.as_str().cmp(&row.key_id))",
            "row.algorithm != configured_row.algorithm.as_wire()",
            "row.public_key_digest.as_slice() != configured_row.public_key_digest",
            "diesel::insert_into(maple_pairing_issuer_keys::table)",
            "NewMaplePairingIssuerKey",
            "compute_maple_pairing_issuer_key_inventory(conn, enclave_key)?",
            "Some(configured.len())",
            "maple_pairing_authority_mac_matches(&configured_digest, &stored_digest)",
        ],
    );
    let update_root =
        extract_function_body(&db, "fn update_maple_pairing_issuer_key_inventory_root(");
    assert_patterns_in_order(
        update_root,
        &[
            "load_maple_pairing_authority_global_head(conn)?",
            "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
            "global.issuer_key_count = issuer_key_count",
            "global.issuer_key_inventory_digest = issuer_key_inventory_digest",
            "global.revision",
            "maple_pairing_authority_global_head_mac(",
            "maple_pairing_authority_global_heads::issuer_key_inventory_digest",
            "maple_pairing_authority_global_heads::issuer_key_count",
            "maple_pairing_authority_global_heads::record_mac",
        ],
    );

    for verifier in [
        "fn verify_maple_pairing_authority_scoped_chain(",
        "fn verify_maple_pairing_authority_global_shallow(",
        "fn verify_maple_pairing_authority_tree_with_mode(",
    ] {
        let verifier_body = extract_function_body(&db, verifier);
        assert!(
            verifier_body.contains("verify_maple_pairing_issuer_key_inventory("),
            "normal authority verifier `{verifier}` must authenticate the stored issuer registry"
        );
    }

    let bootstrap =
        extract_function_body(&db, "fn bootstrap_or_audit_maple_pairing_authority_in_tx(");
    let active_start = bootstrap
        .find("MAPLE_PAIRING_AUTHORITY_ACTIVE => {")
        .expect("bootstrap must retain an Active branch");
    let pending_start = bootstrap
        .find("MAPLE_PAIRING_AUTHORITY_PENDING => {")
        .expect("bootstrap must retain a Pending branch");
    let active_branch = &bootstrap[active_start..pending_start];
    assert_patterns_in_order(
        active_branch,
        &[
            "verify_maple_pairing_authority_tree(conn, enclave_key)?",
            "reconcile_maple_pairing_issuer_key_registry(",
            "if inserted",
            "update_maple_pairing_issuer_key_inventory_root(",
            "verify_maple_pairing_authority_tree(conn, enclave_key)?",
        ],
    );
    let pending_branch = &bootstrap[pending_start..];
    assert_patterns_in_order(
        pending_branch,
        &[
            "global.issuer_key_count != 0",
            "&global.issuer_key_inventory_digest",
            "reconcile_maple_pairing_issuer_key_registry(",
            "NewAppDataMigration::new(MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER).insert(conn)?",
            "active.issuer_key_inventory_digest = issuer_key_inventory_digest.clone()",
            "active.issuer_key_count = issuer_key_count",
            "maple_pairing_authority_global_head_mac(",
            "verify_maple_pairing_authority_tree(conn, enclave_key)?",
        ],
    );

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let startup = extract_function_body(db_impl, "fn bootstrap_or_audit_maple_pairing_authority(");
    assert_patterns_in_order(
        startup,
        &[
            "MaplePairingIssuerKeySetV1::fingerprints",
            "maple_pairing_issuer_key_inventory_digest_from_fingerprints(&configured_issuer_keys)?",
            ".maple_pairing_issuer_key_inventory_digest",
            ".set(expected_inventory_digest)",
            "if self.maple_pairing_issuer_key_inventory_digest.get() != Some(&candidate)",
            "let conn = &mut self.db.get()",
            "run_maple_pairing_authority_transaction(",
            "acquire_maple_pairing_authority_bootstrap_snapshot_fence(tx, enclave_key)?",
            "bootstrap_or_audit_maple_pairing_authority_in_tx(",
            "&configured_issuer_keys",
            "if inventory_digest.as_slice() != expected_inventory_digest",
        ],
    );
    let configured_digest = extract_function_body(
        db_impl,
        "fn configured_maple_pairing_issuer_key_inventory_digest(",
    );
    assert!(
        configured_digest.contains(".get()")
            && configured_digest.contains(".ok_or(DBError::MaplePairingAuthorityCorrupt)"),
        "operational authority paths must fail closed until startup pins a configured issuer inventory"
    );
    let supplied_keyset_fence =
        extract_function_body(&db, "fn require_configured_maple_pairing_issuer_keyset(");
    assert_patterns_in_order(
        supplied_keyset_fence,
        &[
            "issuer_keyset",
            ".fingerprints()",
            "maple_pairing_issuer_key_inventory_digest_from_fingerprints(&fingerprints)?",
            "configured_maple_pairing_issuer_key_inventory_digest()?",
            "maple_pairing_authority_mac_matches(&supplied_digest, &configured_digest)",
            "return Err(DBError::MaplePairingIssuerConfigurationConflict)",
            "Ok(configured_digest)",
        ],
    );
    let create = extract_function_body(db_impl, "fn create_maple_pairing(");
    assert_patterns_in_order(
        create,
        &[
            "require_configured_maple_pairing_issuer_keyset(issuer_keyset)?",
            "let conn = &mut self.db.get()",
            "run_maple_pairing_authority_transaction(",
            "enter_maple_pairing_authority_account_transaction(",
            "materialize(",
        ],
    );
    let register = extract_function_body(db_impl, "fn register_maple_device(");
    assert_patterns_in_order(
        register,
        &[
            "require_configured_maple_pairing_issuer_keyset(issuer_keyset)?",
            "let conn = &mut self.db.get()",
            "run_maple_pairing_authority_transaction(",
            "enter_maple_pairing_authority_account_transaction(",
            "prepare_maple_device_registration_sync(",
            "issuer_keyset",
            "materialize",
        ],
    );
    assert!(
        db.contains("maple_pairing_issuer_key_inventory_digest: OnceLock<[u8; 32]>")
            && db.contains("maple_pairing_issuer_key_inventory_digest: OnceLock::new()"),
        "the configured issuer inventory must remain immutable for one DB connection lifetime"
    );

    let normalized_up = normalize_whitespace(&up);
    let global_table = normalize_whitespace(extract_sql_create_table(
        &up,
        "maple_pairing_authority_global_heads",
    ));
    for required_global_shape in [
        "issuer_key_inventory_digest BYTEA NOT NULL",
        "issuer_key_count BIGINT NOT NULL DEFAULT 0",
        "octet_length(issuer_key_inventory_digest) = 32",
        "issuer_key_count BETWEEN 0 AND 1024",
        "issuer_key_inventory_digest = decode(repeat('00', 32), 'hex')",
        "issuer_key_count = 0",
    ] {
        assert!(
            global_table.contains(required_global_shape),
            "global authority root must retain issuer shape `{required_global_shape}`"
        );
    }
    let issuer_table =
        normalize_whitespace(extract_sql_create_table(&up, "maple_pairing_issuer_keys"));
    for required_registry_shape in [
        "key_id TEXT COLLATE \"C\" PRIMARY KEY",
        "global_singleton BOOLEAN NOT NULL DEFAULT TRUE CHECK (global_singleton)",
        "algorithm TEXT NOT NULL",
        "public_key_digest BYTEA NOT NULL",
        "record_mac BYTEA NOT NULL",
        "CHECK (key_id ~ '^[a-z0-9._:-]{1,64}$')",
        "CHECK (algorithm = 'ed25519')",
        "octet_length(public_key_digest) = 32",
        "octet_length(record_mac) = 32",
        "UNIQUE (algorithm, public_key_digest)",
        "REFERENCES maple_pairing_authority_global_heads(singleton) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
    ] {
        assert!(
            issuer_table.contains(required_registry_shape),
            "issuer registry SQL must retain `{required_registry_shape}`"
        );
    }
    let issuer_guard = extract_sql_function(&up, "enforce_maple_pairing_issuer_key_mutation");
    for required_guard in [
        "IF TG_OP = 'INSERT'",
        "Maple pairing issuer key identity is immutable",
    ] {
        assert!(
            issuer_guard.contains(required_guard),
            "issuer registry mutation guard must retain `{required_guard}`"
        );
    }
    let hierarchy_guard =
        extract_sql_function(&up, "enforce_maple_pairing_authority_hierarchy_commit");
    for required_guard in [
        "max(issuer_key_count)",
        "TG_TABLE_NAME IN (",
        "'maple_pairing_issuer_keys'",
        "SELECT count(*) INTO actual_issuer_key_count",
        "root_issuer_key_count <> actual_issuer_key_count",
        "active Maple pairing issuer-key inventory count is inconsistent",
    ] {
        assert!(
            hierarchy_guard.contains(required_guard),
            "deferred issuer registry count fence must retain `{required_guard}`"
        );
    }
    for required_trigger in [
        "CREATE TRIGGER guard_maple_pairing_issuer_key_mutation BEFORE UPDATE OR DELETE ON maple_pairing_issuer_keys",
        "CREATE CONSTRAINT TRIGGER guard_maple_pairing_issuer_key_commit AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_issuer_keys DEFERRABLE INITIALLY DEFERRED",
        "CREATE TRIGGER guard_maple_pairing_issuer_keys_truncate BEFORE TRUNCATE ON maple_pairing_issuer_keys",
    ] {
        assert!(
            normalized_up.contains(required_trigger),
            "issuer registry SQL must retain `{required_trigger}`"
        );
    }
    assert_patterns_in_order(
        &down,
        &[
            "guard_maple_pairing_issuer_keys_truncate",
            "guard_maple_pairing_issuer_key_commit",
            "guard_maple_pairing_issuer_key_mutation",
            "DROP TABLE IF EXISTS maple_pairing_issuer_keys",
            "DROP TABLE IF EXISTS maple_pairing_authority_global_heads",
        ],
    );
}

#[test]
fn maple_pairing_issuer_references_are_registry_anchored_at_the_sql_boundary() {
    let up = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/up.sql"
    ));
    let normalized = normalize_whitespace(up);
    let registration_sync_issuer_fk = "CONSTRAINT maple_device_registration_operations_sync_issuer_fk FOREIGN KEY (sync_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED";
    assert!(
        normalized.contains(registration_sync_issuer_fk),
        "registration operation sync issuer must remain registry-anchored"
    );
    for (table_name, required_foreign_keys) in [
        (
            "maple_pairings",
            &[
                "CONSTRAINT maple_pairings_ticket_issuer_fk FOREIGN KEY (ticket_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
                "CONSTRAINT maple_pairings_authorization_issuer_fk FOREIGN KEY (authorization_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
                "CONSTRAINT maple_pairings_revocation_issuer_fk FOREIGN KEY (revocation_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            ][..],
        ),
        (
            "maple_pairing_operations",
            &[
                "CONSTRAINT maple_pairing_operations_receipt_issuer_fk FOREIGN KEY (receipt_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            ][..],
        ),
        (
            "maple_pairing_reset_clear_obligations",
            &[
                "CONSTRAINT maple_pairing_reset_clear_obligations_signed_instruction_issuer_fk FOREIGN KEY (signed_instruction_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
                "CONSTRAINT maple_pairing_reset_clear_obligations_sync_issuer_fk FOREIGN KEY (sync_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
                "CONSTRAINT maple_pairing_reset_clear_obligations_ack_receipt_issuer_fk FOREIGN KEY (ack_receipt_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            ][..],
        ),
        (
            "maple_pairing_installation_retirements",
            &[
                "CONSTRAINT maple_pairing_installation_retirements_ack_receipt_issuer_fk FOREIGN KEY (ack_receipt_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            ][..],
        ),
        (
            "maple_pairing_revocation_events",
            &[
                "CONSTRAINT maple_pairing_revocation_events_issuer_fk FOREIGN KEY (issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            ][..],
        ),
    ] {
        let table = normalize_whitespace(extract_sql_create_table(up, table_name));
        for required_foreign_key in required_foreign_keys {
            assert!(
                table.contains(*required_foreign_key),
                "{table_name} must remain registry-anchored by `{required_foreign_key}`"
            );
        }
    }
    assert!(
        normalized
            .matches("FOREIGN KEY (ack_receipt_issuer_key_id) REFERENCES maple_pairing_issuer_keys(key_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED")
            .count()
            >= 2,
        "both reset-clear obligations and installation retirements must anchor ACK receipt issuers"
    );

    let tombstone_guard =
        extract_sql_function(up, "enforce_maple_pairing_registration_tombstone_mutation");
    assert_patterns_in_order(
        tombstone_guard,
        &[
            "IF TG_OP = 'UPDATE'",
            "ELSIF TG_OP = 'INSERT'",
            "IF NOT maple_pairing_issuer_key_ids_are_canonical(",
            "NEW.referenced_issuer_key_ids",
            "4",
            "RETURN NEW",
            "unnest(NEW.referenced_issuer_key_ids) AS referenced(key_id)",
            "SELECT 1 FROM maple_pairing_issuer_keys registered",
            "registered.key_id = referenced.key_id",
            "Maple registration tombstone references an unknown issuer key",
            "RETURN NEW",
        ],
    );
}

#[test]
fn maple_pairing_persisted_and_wire_artifacts_reject_non_v1_versions() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let db_models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB models should be readable");
    let wire = fs::read_to_string(manifest_dir.join("src/models/maple_pairings.rs"))
        .expect("Maple pairing wire models should be readable");
    let device_routes = fs::read_to_string(manifest_dir.join("src/web/maple_devices.rs"))
        .expect("Maple device routes should be readable");
    let pairing_routes = fs::read_to_string(manifest_dir.join("src/web/maple_pairings.rs"))
        .expect("Maple pairing routes should be readable");
    let up = fs::read_to_string(
        manifest_dir.join("migrations/2026-08-13-120000_maple_pairings_v1/up.sql"),
    )
    .expect("Maple pairing migration should be readable");

    for exact_constant in [
        "pub const MAPLE_PAIRING_PAYLOAD_VERSION_V1: i16 = 1;",
        "pub const MAPLE_PAIRING_RECEIPT_VERSION_V1: i16 = 1;",
    ] {
        assert!(
            db_models.contains(exact_constant),
            "DB version constant must remain exact: `{exact_constant}`"
        );
    }
    for exact_constant in [
        "pub const MAPLE_PAIRING_PROTOCOL_VERSION_V1: u16 = 1;",
        "pub const MAPLE_PAIRING_TRANSCRIPT_VERSION_V1: u16 = 1;",
        "pub const MAPLE_PAIRING_ARTIFACT_VERSION_V1: u16 = 1;",
    ] {
        assert!(
            wire.contains(exact_constant),
            "wire version constant must remain exact: `{exact_constant}`"
        );
    }

    for (function, required_gates) in [
        (
            "fn validate_maple_device_registration_operation(",
            &["operation.sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1"][..],
        ),
        (
            "fn validate_maple_pairing_record(",
            &["row.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1"][..],
        ),
        (
            "fn validate_maple_device_registration_tombstone(",
            &[
                "row.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1",
                "receipt.sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
            ][..],
        ),
        (
            "fn validate_maple_installation_retirement(",
            &["row.ack_receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1"][..],
        ),
        (
            "fn acknowledge_pending_maple_reset_clear(",
            &["ack.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1"][..],
        ),
        (
            "fn validate_maple_pairing_reset_clear_obligation(",
            &[
                "row.host_claim_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "row.instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "version != MAPLE_PAIRING_RECEIPT_VERSION_V1",
            ][..],
        ),
        (
            "fn prepare_maple_device_registration_sync(",
            &[
                "signed_instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "sync_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
            ][..],
        ),
        (
            "fn validate_maple_pairing_revocation_record(",
            &["row.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1"][..],
        ),
        (
            "fn pairing_operation_receipt(",
            &["operation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1"][..],
        ),
        (
            "fn insert_pairing_operation(",
            &["receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1"][..],
        ),
        (
            "fn validate_reset_clear_material_against_locked_context(",
            &[
                "prepared.host_claim_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "prepared.instruction_payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "instruction.artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1",
            ][..],
        ),
    ] {
        let body = extract_function_body(&db, function);
        for required_gate in required_gates {
            assert!(
                body.contains(required_gate),
                "persisted-artifact validator `{function}` must retain exact V1 gate `{required_gate}`"
            );
        }
    }

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    for (method, required_gates) in [
        (
            "ack_maple_pairing_revocation",
            &["ack.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1"][..],
        ),
        (
            "revoke_maple_pairing",
            &[
                "let pair_payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "let event_payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1",
            ][..],
        ),
        (
            "approve_maple_pairing",
            &[
                "mutation.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "mutation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1",
            ][..],
        ),
        (
            "confirm_maple_pairing",
            &[
                "mutation.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "mutation.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1",
            ][..],
        ),
        (
            "create_maple_pairing",
            &[
                "let payload_version = MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "let receipt_version = MAPLE_PAIRING_RECEIPT_VERSION_V1",
                "protocol_version: MAPLE_PAIRING_PROTOCOL_VERSION_V1",
            ][..],
        ),
        (
            "register_maple_device",
            &[
                "registration.payload_version != MAPLE_PAIRING_PAYLOAD_VERSION_V1",
                "accepted_operation.sync_payload_version != prepared.payload_version",
            ][..],
        ),
    ] {
        let body = extract_function_body(db_impl, &format!("fn {method}("));
        for required_gate in required_gates {
            assert!(
                body.contains(required_gate),
                "DB method `{method}` must reject non-V1 material via `{required_gate}`"
            );
        }
    }

    for (source_name, source, function, required_gates) in [
        (
            "device routes",
            device_routes.as_str(),
            "fn materialize_maple_device_registration_sync(",
            &[
                "context.host_claim_payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1",
                "context.instruction_payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1",
            ][..],
        ),
        (
            "device routes",
            device_routes.as_str(),
            "fn validate_registration(",
            &[
                "request.protocol_version != PROTOCOL_VERSION_V1",
                "request.transcript_version != TRANSCRIPT_VERSION_V1",
            ][..],
        ),
        (
            "device routes",
            device_routes.as_str(),
            "pub(crate) fn decrypt_device_response(",
            &["row.payload_version != PAYLOAD_VERSION_V1"][..],
        ),
        (
            "device routes",
            device_routes.as_str(),
            "pub(crate) fn build_reset_clear_material(",
            &["*payload_version != RESET_CLEAR_PAYLOAD_VERSION_V1"][..],
        ),
        (
            "pairing routes",
            pairing_routes.as_str(),
            "fn validate_common_assertions(",
            &["protocol_version != PROTOCOL_VERSION_V1"][..],
        ),
    ] {
        let body = extract_function_body(source, function);
        for required_gate in required_gates {
            assert!(
                body.contains(required_gate),
                "{source_name} `{function}` must retain exact V1 gate `{required_gate}`"
            );
        }
    }

    for wire_boundary in [
        "pub fn reset_clear_admission_set_transcript(",
        "fn validate_reset_clear_material_shape(",
        "impl MapleRevocationStreamCheckpointV1",
        "impl MaplePairRequestTicketV1",
        "impl MaplePairAuthorizationV1",
        "impl MaplePairRevocationV1",
    ] {
        let body = extract_function_body(&wire, wire_boundary);
        assert!(
            body.contains("artifact_version != MAPLE_PAIRING_ARTIFACT_VERSION_V1"),
            "wire boundary `{wire_boundary}` must reject non-V1 artifact versions"
        );
    }
    for wire_boundary in [
        "impl ListMaplePairingRevocationsResponse",
        "impl AckMaplePairingRevocationResponse",
    ] {
        let body = extract_function_body(&wire, wire_boundary);
        assert!(
            body.contains("protocol_version != MAPLE_PAIRING_PROTOCOL_VERSION_V1"),
            "wire boundary `{wire_boundary}` must reject non-V1 protocol versions"
        );
    }
    let shared_version_gate = extract_function_body(&wire, "fn validate_versions(");
    for required_gate in [
        "protocol_version != MAPLE_PAIRING_PROTOCOL_VERSION_V1",
        "transcript_version != MAPLE_PAIRING_TRANSCRIPT_VERSION_V1",
    ] {
        assert!(
            shared_version_gate.contains(required_gate),
            "shared wire validation must retain exact V1 gate `{required_gate}`"
        );
    }

    let normalized_up = normalize_whitespace(&up);
    for exact_sql_gate in [
        "sync_payload_version = 1 AND octet_length(sync_payload_enc) BETWEEN 1 AND 65536",
        "receipt_version = 1 AND octet_length(receipt_enc) BETWEEN 1 AND 65536",
        "CONSTRAINT maple_pairings_payload_version_v1 CHECK (payload_version = 1)",
        "CONSTRAINT maple_pairing_operations_receipt_version_v1 CHECK (receipt_version = 1)",
        "host_claim_payload_version = 1 AND octet_length(host_claim_payload_enc) BETWEEN 1 AND 65536 AND instruction_payload_version = 1",
        "signed_instruction_payload_version = 1 AND octet_length(signed_instruction_payload_enc) BETWEEN 1 AND 65536",
        "sync_payload_version = 1 AND octet_length(sync_payload_enc) BETWEEN 1 AND 65536",
        "ack_receipt_version = 1 AND ack_receipt_enc IS NOT NULL",
        "AND ack_receipt_version = 1 AND ack_receipt_issuer_key_id",
        "CONSTRAINT maple_pairing_revocation_events_payload_version_v1 CHECK (payload_version = 1)",
    ] {
        assert!(
            normalized_up.contains(exact_sql_gate),
            "SQL persistence boundary must retain exact V1 gate `{exact_sql_gate}`"
        );
    }
    let registration_operation_v1 = "CONSTRAINT maple_device_registration_operations_sync_shape CHECK ( sync_payload_version = 1 AND octet_length(sync_payload_enc) BETWEEN 1 AND 65536";
    assert!(
        normalized_up.contains(registration_operation_v1),
        "registration operation persistence must retain its exact V1 sync gate"
    );
    for (table_name, exact_v1_gates) in [
        (
            "maple_pairing_registration_operation_tombstones",
            &[
                "CONSTRAINT maple_pairing_registration_operation_tombstones_receipt_shape CHECK ( receipt_version = 1 AND octet_length(receipt_enc) BETWEEN 1 AND 65536",
            ][..],
        ),
        (
            "maple_pairings",
            &["CONSTRAINT maple_pairings_payload_version_v1 CHECK (payload_version = 1)"][..],
        ),
        (
            "maple_pairing_operations",
            &["CONSTRAINT maple_pairing_operations_receipt_version_v1 CHECK (receipt_version = 1)"]
                [..],
        ),
        (
            "maple_pairing_reset_clear_obligations",
            &[
                "CONSTRAINT maple_pairing_reset_clear_obligations_unsigned_payload_shape CHECK ( host_claim_payload_version = 1",
                "AND instruction_payload_version = 1",
                "CONSTRAINT maple_pairing_reset_clear_obligations_signed_material_shape CHECK",
                "AND signed_instruction_payload_version = 1",
                "AND sync_payload_version = 1",
                "AND ack_receipt_version = 1",
            ][..],
        ),
        (
            "maple_pairing_installation_retirements",
            &[
                "CONSTRAINT maple_pairing_installation_retirements_shape CHECK",
                "AND ack_receipt_version = 1",
            ][..],
        ),
        (
            "maple_pairing_revocation_events",
            &[
                "CONSTRAINT maple_pairing_revocation_events_payload_version_v1 CHECK (payload_version = 1)",
            ][..],
        ),
    ] {
        let table = normalize_whitespace(extract_sql_create_table(&up, table_name));
        assert_patterns_in_order(&table, exact_v1_gates);
    }
}

#[test]
fn every_active_authority_entry_threads_the_pinned_issuer_inventory_digest() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let main = fs::read_to_string(manifest_dir.join("src/main.rs"))
        .expect("main source should be readable");

    let fence = extract_function_body(
        &db,
        "fn acquire_maple_pairing_authority_snapshot_fence_with_mode(",
    );
    assert_patterns_in_order(
        fence,
        &[
            "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
            "MaplePairingAuthoritySnapshotFenceMode::ActiveOnly",
            "expected_issuer_key_inventory_digest",
            ".filter(|digest| digest.len() == 32)",
            "DBError::MaplePairingIssuerConfigurationConflict",
            "maple_pairing_authority_mac_matches(",
            "expected_issuer_key_inventory_digest",
            "&global.issuer_key_inventory_digest",
            "return Err(DBError::MaplePairingIssuerConfigurationConflict)",
        ],
    );
    let active_wrapper =
        extract_function_body(&db, "fn acquire_maple_pairing_authority_snapshot_fence(");
    assert!(
        active_wrapper.contains("Some(expected_issuer_key_inventory_digest)"),
        "the Active fence wrapper must require the locally pinned issuer inventory digest"
    );
    let bootstrap_wrapper = extract_function_body(
        &db,
        "fn acquire_maple_pairing_authority_bootstrap_snapshot_fence(",
    );
    assert!(
        bootstrap_wrapper.contains("MaplePairingAuthoritySnapshotFenceMode::Bootstrap")
            && bootstrap_wrapper.contains("None"),
        "only startup bootstrap may enter without an already-reconciled configured digest"
    );

    for call_name in [
        "acquire_maple_pairing_authority_snapshot_fence(",
        "enter_maple_pairing_authority_account_transaction(",
    ] {
        let mut caller_count = 0usize;
        for (call_position, _) in db.match_indices(call_name) {
            if db[..call_position].ends_with("fn ") {
                continue;
            }
            caller_count += 1;
            let call = extract_rust_parenthesized_call(&db, call_position, call_name);
            assert!(
                call.contains("expected_issuer_key_inventory_digest"),
                "authority call `{call_name}` must thread the pinned issuer digest: `{call}`"
            );
        }
        assert!(
            caller_count >= 10,
            "expected production and test callers to exercise `{call_name}`"
        );
    }

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let operational_methods = [
        "create_user",
        "list_maple_pairing_revocations",
        "ack_maple_pairing_revocation",
        "revoke_maple_pairing",
        "approve_maple_pairing",
        "confirm_maple_pairing",
        "list_maple_pairings",
        "get_maple_pairing",
        "audit_maple_pairing_issuer_key_references",
        "replay_maple_reset_clear_ack",
        "replay_maple_pairing_operation",
        "create_maple_pairing",
        "complete_destructive_password_reset",
        "register_maple_device",
        "list_maple_devices",
        "create_org",
        "delete_org",
        "create_org_project",
        "delete_org_project",
        "create_org_with_owner",
        "delete_user",
        "mark_and_delete_user",
    ];
    for method_name in operational_methods {
        let method = extract_function_body(db_impl, &format!("fn {method_name}("));
        let configured_digest_source = match method_name {
            "create_maple_pairing" | "revoke_maple_pairing" | "register_maple_device" => {
                "require_configured_maple_pairing_issuer_keyset(issuer_keyset)?"
            }
            _ => "configured_maple_pairing_issuer_key_inventory_digest()?",
        };
        assert_patterns_in_order(
            method,
            &[
                "let expected_issuer_key_inventory_digest =",
                configured_digest_source,
                "run_maple_pairing_authority_transaction(",
                "expected_issuer_key_inventory_digest",
            ],
        );
    }
    assert_eq!(
        db_impl
            .matches("require_configured_maple_pairing_issuer_keyset(issuer_keyset)?")
            .count(),
        3,
        "only CREATE, REVOKE, and REGISTER accept a verifier keyset, and all must bind it to the process-pinned issuer inventory"
    );
    assert_eq!(
        db_impl
            .matches("run_maple_pairing_authority_transaction(")
            .count(),
        operational_methods.len() + 1,
        "every DB-implementation authority transaction must be either startup bootstrap or an enumerated digest-fenced operational method"
    );

    for signup_helper in [
        "fn create_user_with_password_seed_wrap(",
        "pub fn create_user_with_oauth_seed_wrap(",
    ] {
        let helper = extract_function_body(&main, signup_helper);
        assert_patterns_in_order(
            helper,
            &[
                "configured_maple_pairing_issuer_key_inventory_digest()?",
                "create_user_with_maple_authority_in_tx(",
                "&expected_issuer_key_inventory_digest",
            ],
        );
    }
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
        "user_oauth_connections::table",
        "user_oauth_connections::user_id.eq(user_id)",
        "password_reset_requests::id.eq(reset_request.id)",
        "password_reset_requests::user_id.eq(user_id)",
        "password_reset_requests::is_reset.eq(false)",
        "password_reset_requests::expiration_time.gt(diesel::dsl::now)",
        "if consumed_reset_count != 1",
        "DBError::PasswordResetRequestNotFound",
        "password_reset_requests::id.ne(locked_reset_request.id)",
    ] {
        assert!(
            reset_body.contains(required_pattern),
            "destructive reset must contain `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        reset_body,
        &[
            "diesel::delete(\n                    user_seed_wrappings::table",
            "users::password_enc.eq(Some(new_password_enc))",
            "if new_wrapping.user_id != user_id",
            "diesel::insert_into(user_seed_wrappings::table)",
            "user_seed_wrappings::user_id.eq(new_wrapping.user_id)",
            "user_seed_wrappings::credential_kind.eq(&new_wrapping.credential_kind)",
            "user_seed_wrappings::credential_lookup_hash",
            ".eq(&new_wrapping.credential_lookup_hash)",
            "user_seed_wrappings::wrapping_version.eq(new_wrapping.wrapping_version)",
            "user_seed_wrappings::seed_enc.eq(&new_wrapping.seed_enc)",
            "user_seed_wrappings::created_at.eq(persisted_reset.reset_at)",
            "user_seed_wrappings::updated_at.eq(persisted_reset.reset_at)",
        ],
    );
    assert!(
        !reset_body.contains("upsert_by_credential"),
        "destructive reset must directly insert the sole replacement wrapping with the clamped reset lifecycle time"
    );

    assert!(
        !reset_body.contains("users::seed_enc") && !reset_body.contains("new_legacy_seed_enc"),
        "destructive reset must not touch the removed legacy users.seed_enc bridge"
    );

    assert!(
        !reset_body.contains("user_api_keys::table"),
        "destructive password reset must preserve user_api_keys in this release"
    );
}

#[test]
fn destructive_password_reset_projects_the_exact_next_authenticated_authority_epoch() {
    let db_source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/db.rs"));
    let commit = extract_function_body(
        db_source,
        "fn commit_maple_pairing_authority_account_mutation_with_security_epoch(",
    );

    assert_patterns_in_order(
        commit,
        &[
            "validate_maple_pairing_authority_account_head(enclave_key, &head)?",
            "let prior_revision = head.revision;",
            "if let Some(target_security_epoch) = target_security_epoch",
            ".security_epoch\n                .checked_add(1)",
            "head.security_epoch = target_security_epoch;",
            "head.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &head)?;",
            "compute_maple_pairing_authority_account_inventory(conn, enclave_key, &head)?",
            "head.revision = prior_revision\n        .checked_add(1)",
            "head.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &head)?;",
            ".filter(maple_pairing_authority_account_heads::revision.eq(prior_revision))",
            "if updated != 1",
            "cascade_maple_pairing_authority_heads(conn, enclave_key, user_id)?",
            "verify_maple_pairing_authority_scoped_chain(conn, enclave_key, &current)",
        ],
    );
}

#[test]
fn maple_pairing_parent_foreign_keys_fail_closed() {
    let migration = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/up.sql"
    ));
    assert!(
        !migration.contains("ON DELETE CASCADE"),
        "the authority migration must replace every cascading Maple parent edge with NO ACTION"
    );

    let normalized = normalize_whitespace(migration);
    for required_pattern in [
        "REFERENCES users(uuid, project_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
        "REFERENCES maple_devices(id, user_id, project_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
    ] {
        assert!(
            normalized.contains(required_pattern),
            "device authority edges must contain `{required_pattern}`"
        );
    }

    for table_name in [
        "maple_pairing_lineages",
        "maple_pairings",
        "maple_pairing_operations",
        "maple_pairing_host_states",
        "maple_pairing_revocation_events",
    ] {
        let table = normalize_whitespace(extract_sql_create_table(migration, table_name));
        let required_pattern =
            "REFERENCES users(uuid, project_id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED";
        assert!(
            table.contains(required_pattern),
            "`{table_name}` must retain its raw `{required_pattern}` parent edge"
        );
    }

    for required_constraint in [
        "maple_devices_authority_account_fk",
        "maple_device_registration_operations_authority_account_fk",
        "maple_pairing_lineages_authority_account_fk",
        "maple_pairings_authority_account_fk",
        "maple_pairing_operations_authority_account_fk",
        "maple_pairing_host_states_authority_account_fk",
        "maple_pairing_revocation_events_authority_account_fk",
        "maple_pairing_revocation_highwaters_authority_scope_fk",
        "maple_pairing_registration_operation_tombstones_scope_fk",
        "maple_pairing_installation_retirements_scope_fk",
        "maple_pairing_reset_clear_obligations_scope_fk",
        "maple_pairing_reset_clear_admissions_scope_fk",
    ] {
        assert!(
            migration.contains(required_constraint),
            "Maple authority row must be anchored by `{required_constraint}`"
        );
    }
}

#[test]
fn maple_reset_clear_storage_is_authenticated_bounded_and_append_only() {
    let up = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/up.sql"
    ));
    let down = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/down.sql"
    ));

    let global_head = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_authority_global_heads",
    ));
    assert!(
        global_head.contains(
            "activation_state = 2 AND revision >= 2 AND record_mac IS NOT NULL AND octet_length(record_mac) = 32",
        ),
        "the Active root shape must explicitly reject NULL record_mac instead of accepting SQL CHECK UNKNOWN"
    );
    let pairing_rows = normalize_whitespace(extract_sql_create_table(up, "maple_pairings"));
    assert!(
        pairing_rows.contains(
            "state IN (2, 3, 5) AND pair_authorization_digest IS NOT NULL AND octet_length(pair_authorization_digest) = 32",
        ),
        "authorized pairing states must explicitly reject NULL authorization digests"
    );

    let account_head = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_authority_account_heads",
    ));
    for required_pattern in [
        "security_epoch BIGINT NOT NULL DEFAULT 1",
        "CHECK (security_epoch > 0)",
        "authority_row_count BETWEEN 0 AND 567360",
        "registration_operation_tombstone_count BIGINT NOT NULL DEFAULT 0",
        "installation_retirement_count BIGINT NOT NULL DEFAULT 0",
        "reset_clear_obligation_count BIGINT NOT NULL DEFAULT 0",
        "reset_clear_admission_count BIGINT NOT NULL DEFAULT 0",
        "registration_operation_tombstone_count BETWEEN 0 AND 32768",
        "installation_retirement_count BETWEEN 0 AND 1024",
        "reset_clear_obligation_count BETWEEN 0 AND 4096",
        "reset_clear_admission_count BETWEEN 0 AND 524288",
        "device_operation_count + registration_operation_tombstone_count <= 32768",
        "+ highwater_generation_count + registration_operation_tombstone_count + installation_retirement_count + reset_clear_obligation_count + reset_clear_admission_count",
    ] {
        assert!(
            account_head.contains(required_pattern),
            "account authority head must authenticate reset-clear capacity via `{required_pattern}`"
        );
    }

    let tombstones = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_registration_operation_tombstones",
    ));
    for required_pattern in [
        "authority_scope_digest BYTEA NOT NULL",
        "lookup_digest BYTEA NOT NULL",
        "operation_lookup_digest BYTEA NOT NULL",
        "retired_security_epoch BIGINT NOT NULL",
        "request_mac BYTEA NOT NULL",
        "outcome_kind SMALLINT NOT NULL",
        "outcome_digest BYTEA NOT NULL",
        "receipt_version SMALLINT NOT NULL",
        "receipt_enc BYTEA NOT NULL",
        "receipt_digest BYTEA NOT NULL",
        "referenced_issuer_key_ids TEXT[] NOT NULL",
        "accepted_at TIMESTAMPTZ NOT NULL",
        "record_mac BYTEA NOT NULL",
        "retired_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "receipt_version = 1",
        "maple_pairing_issuer_key_ids_are_canonical(referenced_issuer_key_ids, 4)",
        "accepted_at <= retired_at",
        "UNIQUE (authority_scope_digest, operation_lookup_digest)",
        "REFERENCES maple_pairing_authority_account_heads(authority_scope_digest) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
    ] {
        assert!(
            tombstones.contains(required_pattern),
            "retired registration-operation tombstone must retain `{required_pattern}`"
        );
    }
    assert!(
        !tombstones.contains("operation_id UUID")
            && !tombstones.contains("user_id")
            && !tombstones.contains("project_id"),
        "tombstones must remain pseudonymous and must not retain raw parent or operation IDs"
    );

    let retirements = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_installation_retirements",
    ));
    for required_pattern in [
        "authority_scope_digest BYTEA NOT NULL",
        "lookup_digest BYTEA NOT NULL",
        "host_identity_mac BYTEA NOT NULL",
        "retired_security_epoch BIGINT NOT NULL",
        "final_obligation_event_id UUID NOT NULL",
        "final_instruction_digest BYTEA NOT NULL",
        "final_chain_digest BYTEA NOT NULL",
        "ack_host_registration_lookup_digest BYTEA NOT NULL",
        "ack_operation_lookup_digest BYTEA NOT NULL",
        "ack_request_mac BYTEA NOT NULL",
        "ack_receipt_version SMALLINT NOT NULL",
        "ack_receipt_issuer_key_id TEXT NOT NULL",
        "ack_receipt_digest BYTEA NOT NULL",
        "retired_at TIMESTAMPTZ NOT NULL",
        "record_mac BYTEA NOT NULL",
        "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "created_at = retired_at",
        "UNIQUE (authority_scope_digest, lookup_digest)",
        "UNIQUE (authority_scope_digest, host_identity_mac)",
        "UNIQUE (authority_scope_digest, ack_host_registration_lookup_digest)",
        "UNIQUE (authority_scope_digest, ack_operation_lookup_digest)",
        "REFERENCES maple_pairing_authority_account_heads(authority_scope_digest) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
        ") REFERENCES maple_pairing_reset_clear_obligations( uuid, authority_scope_digest, lookup_digest, instruction_digest, chain_digest ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
    ] {
        assert!(
            retirements.contains(required_pattern),
            "terminal installation retirement must retain `{required_pattern}`"
        );
    }
    for forbidden_raw_parent in [
        "user_id UUID",
        "project_id INTEGER",
        "installation_id UUID",
        "device_id UUID",
        "operation_id UUID",
    ] {
        assert!(
            !retirements.contains(forbidden_raw_parent),
            "pseudonymous installation retirement must not retain `{forbidden_raw_parent}`"
        );
    }

    let obligations = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_reset_clear_obligations",
    ));
    for required_pattern in [
        "authority_scope_digest BYTEA NOT NULL",
        "lookup_digest BYTEA NOT NULL",
        "host_identity_mac BYTEA NOT NULL",
        "reset_id UUID NOT NULL",
        "reset_generation BIGINT NOT NULL",
        "cumulative_reset_count BIGINT NOT NULL",
        "previous_event_id UUID",
        "previous_instruction_digest BYTEA",
        "previous_chain_digest BYTEA",
        "old_revocation_stream_id UUID NOT NULL",
        "old_revocation_stream_generation BIGINT NOT NULL",
        "source_security_epoch BIGINT NOT NULL",
        "source_last_issued_revocation_sequence BIGINT NOT NULL",
        "target_revocation_stream_id UUID NOT NULL",
        "target_revocation_stream_generation BIGINT NOT NULL",
        "target_security_epoch BIGINT NOT NULL",
        "target_security_epoch = source_security_epoch + 1",
        "target_instruction_sequence = 1",
        "clear_scope = 1",
        "admission_set_digest BYTEA NOT NULL",
        "CHECK (admission_count BETWEEN 0 AND 128)",
        "host_claim_payload_version SMALLINT NOT NULL",
        "host_claim_payload_enc BYTEA NOT NULL",
        "host_claim_digest BYTEA NOT NULL",
        "instruction_payload_version SMALLINT NOT NULL",
        "instruction_payload_enc BYTEA NOT NULL",
        "instruction_digest BYTEA NOT NULL",
        "chain_digest BYTEA NOT NULL",
        "signed_instruction_payload_version SMALLINT",
        "signed_instruction_payload_enc BYTEA",
        "signed_instruction_issuer_key_id TEXT",
        "signed_instruction_digest BYTEA",
        "sync_payload_version SMALLINT",
        "sync_payload_enc BYTEA",
        "sync_issuer_key_id TEXT",
        "sync_digest BYTEA",
        "acked_by_head_event_id UUID",
        "ack_operation_id UUID",
        "ack_host_registration_lookup_digest BYTEA",
        "ack_request_mac BYTEA",
        "ack_receipt_version SMALLINT",
        "ack_receipt_enc BYTEA",
        "ack_receipt_issuer_key_id TEXT",
        "ack_receipt_digest BYTEA",
        "record_mac BYTEA NOT NULL",
        "FOREIGN KEY (authority_scope_digest) REFERENCES maple_pairing_authority_account_heads(authority_scope_digest) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
        ") REFERENCES maple_pairing_revocation_highwaters(",
        ") REFERENCES maple_pairing_reset_clear_obligations(",
    ] {
        assert!(
            obligations.contains(required_pattern),
            "reset-clear obligation must retain `{required_pattern}`"
        );
    }
    for required_state_shape in [
        "signed_instruction_payload_version IS NULL AND signed_instruction_payload_enc IS NULL AND signed_instruction_issuer_key_id IS NULL AND signed_instruction_digest IS NULL AND sync_payload_version IS NULL AND sync_payload_enc IS NULL AND sync_issuer_key_id IS NULL AND sync_digest IS NULL",
        "signed_instruction_payload_version IS NOT NULL AND signed_instruction_payload_enc IS NOT NULL AND signed_instruction_issuer_key_id IS NOT NULL AND signed_instruction_digest IS NOT NULL AND sync_payload_version IS NOT NULL AND sync_payload_enc IS NOT NULL AND sync_issuer_key_id IS NOT NULL AND sync_digest IS NOT NULL",
        "state = 1 AND revision = 1",
        "state = 1 AND revision = 2",
        "state = 2 AND revision = 2",
        "state = 2 AND revision = 3",
        "acked_by_head_event_id IS NOT NULL",
        "acked_by_head_event_id = uuid",
        "ack_operation_id IS NOT NULL",
        "ack_host_registration_lookup_digest IS NOT NULL",
        "octet_length(ack_host_registration_lookup_digest) = 32",
        "ack_request_mac IS NOT NULL",
        "ack_receipt_version IS NOT NULL",
        "ack_receipt_enc IS NOT NULL",
        "ack_receipt_issuer_key_id IS NOT NULL",
        "ack_receipt_digest IS NOT NULL",
    ] {
        assert!(
            obligations.contains(required_state_shape),
            "reset-clear state machine must retain `{required_state_shape}`"
        );
    }
    for forbidden_raw_parent in [
        "user_id UUID",
        "project_id INTEGER",
        "installation_id UUID",
        "device_id UUID",
    ] {
        assert!(
            !obligations.contains(forbidden_raw_parent),
            "pseudonymous reset-clear obligation must not retain `{forbidden_raw_parent}`"
        );
    }

    let admissions = normalize_whitespace(extract_sql_create_table(
        up,
        "maple_pairing_reset_clear_admissions",
    ));
    for required_pattern in [
        "obligation_uuid UUID NOT NULL",
        "authority_scope_digest BYTEA NOT NULL",
        "lookup_digest BYTEA NOT NULL",
        "pair_id UUID NOT NULL",
        "pairing_incarnation BIGINT NOT NULL",
        "pair_authorization_digest BYTEA NOT NULL",
        "record_mac BYTEA NOT NULL",
        "REFERENCES maple_pairing_authority_account_heads(authority_scope_digest) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
        "REFERENCES maple_pairing_reset_clear_obligations( uuid, authority_scope_digest, lookup_digest ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
        "CONSTRAINT maple_pairing_reset_clear_admissions_identity_unique UNIQUE (obligation_uuid, pair_id, pairing_incarnation)",
    ] {
        assert!(
            admissions.contains(required_pattern),
            "private reset-clear admission leaf must retain `{required_pattern}`"
        );
    }
    assert!(
        !admissions.contains("maple_pairing_reset_clear_admissions_pair_unique")
            && !admissions.contains("maple_pairing_reset_clear_admissions_incarnation_unique"),
        "admission identity must be the exact triple; pair or incarnation alone is not globally unique across lineages"
    );
    for forbidden_raw_parent in [
        "user_id UUID",
        "project_id INTEGER",
        "installation_id UUID",
        "device_id UUID",
    ] {
        assert!(
            !admissions.contains(forbidden_raw_parent),
            "reset-clear admission leaf must not retain `{forbidden_raw_parent}`"
        );
    }

    let obligation_guard =
        extract_sql_function(up, "enforce_maple_pairing_reset_clear_obligation_mutation");
    for required_pattern in [
        "NEW.state <> 1 OR NEW.revision <> 1",
        "ELSIF TG_OP = 'DELETE'",
        "IF OLD.state <> 2",
        "pending reset-clear obligation cannot be deleted",
        "reset-clear obligation identity is immutable",
        "OLD.state = 1 AND OLD.revision = 1",
        "NEW.state = 1 AND NEW.revision = 2",
        "NEW.state = 2 AND NEW.revision = 2",
        "OLD.state = 1 AND OLD.revision = 2",
        "NEW.state <> 2 OR NEW.revision <> 3",
        "acked reset-clear obligation is immutable",
    ] {
        assert!(
            obligation_guard.contains(required_pattern),
            "reset-clear row guard must retain `{required_pattern}`"
        );
    }
    let tombstone_guard =
        extract_sql_function(up, "enforce_maple_pairing_registration_tombstone_mutation");
    for required_pattern in [
        "Maple registration operation tombstone is immutable",
        "Maple registration operation tombstone deletion is out of order",
        "FROM maple_pairing_reset_clear_obligations",
        "FROM maple_device_registration_operations",
    ] {
        assert!(
            tombstone_guard.contains(required_pattern),
            "registration tombstone row guard must retain `{required_pattern}`"
        );
    }
    let retirement_guard =
        extract_sql_function(up, "enforce_maple_pairing_installation_retirement_mutation");
    for required_pattern in [
        "Maple installation retirement is immutable",
        "Maple installation retirement deletion is out of order",
        "FROM maple_pairing_reset_clear_obligations",
        "FROM maple_pairing_registration_operation_tombstones",
        "Maple installation retirement requires the exact terminal ACK head",
        "obligation_state <> 2",
        "obligation_revision <> 3",
        "obligation_ack_head IS DISTINCT FROM NEW.final_obligation_event_id",
        "obligation_ack_host_registration_lookup_digest",
        "IS DISTINCT FROM NEW.ack_host_registration_lookup_digest",
        "obligation_ack_request_mac IS DISTINCT FROM NEW.ack_request_mac",
        "obligation_ack_receipt_digest IS DISTINCT FROM NEW.ack_receipt_digest",
        "obligation_target_epoch IS DISTINCT FROM NEW.retired_security_epoch",
        "obligation_acked_at IS DISTINCT FROM NEW.retired_at",
    ] {
        assert!(
            retirement_guard.contains(required_pattern),
            "installation retirement row guard must retain `{required_pattern}`"
        );
    }
    let normalized_up = normalize_whitespace(up);
    for required_pattern in [
        "CREATE INDEX idx_maple_pairing_registration_operation_tombstones_scope ON maple_pairing_registration_operation_tombstones(authority_scope_digest, id)",
        "CREATE INDEX idx_maple_pairing_registration_operation_tombstones_lookup ON maple_pairing_registration_operation_tombstones( authority_scope_digest, lookup_digest, retired_security_epoch, id )",
        "CREATE INDEX idx_maple_pairing_installation_retirements_scope ON maple_pairing_installation_retirements(authority_scope_digest, id)",
        "CREATE INDEX idx_maple_pairing_installation_retirements_lookup ON maple_pairing_installation_retirements(authority_scope_digest, lookup_digest)",
        "CREATE INDEX idx_maple_pairing_installation_retirements_identity ON maple_pairing_installation_retirements(authority_scope_digest, host_identity_mac)",
        "CREATE UNIQUE INDEX idx_maple_pairing_reset_clear_no_forks ON maple_pairing_reset_clear_obligations(previous_event_id) WHERE previous_event_id IS NOT NULL",
        "CREATE UNIQUE INDEX idx_maple_pairing_reset_clear_ack_operation ON maple_pairing_reset_clear_obligations( authority_scope_digest, ack_host_registration_lookup_digest, ack_operation_id ) WHERE ack_operation_id IS NOT NULL",
        "CREATE INDEX idx_maple_pairing_reset_clear_obligations_scope ON maple_pairing_reset_clear_obligations(authority_scope_digest, id)",
        "CREATE INDEX idx_maple_pairing_reset_clear_obligations_current ON maple_pairing_reset_clear_obligations( authority_scope_digest, lookup_digest, state, reset_generation DESC, id DESC )",
        "CREATE INDEX idx_maple_pairing_reset_clear_admissions_scope ON maple_pairing_reset_clear_admissions(authority_scope_digest, id)",
        "CREATE INDEX idx_maple_pairing_reset_clear_admissions_canonical ON maple_pairing_reset_clear_admissions( authority_scope_digest, obligation_uuid, pair_id, pairing_incarnation )",
        "CREATE TRIGGER guard_maple_pairing_reset_clear_obligation_mutation BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_reset_clear_obligations",
        "CREATE TRIGGER guard_maple_pairing_registration_tombstone_mutation BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_registration_operation_tombstones",
        "CREATE TRIGGER guard_maple_pairing_installation_retirement_mutation BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_installation_retirements",
        "CREATE TRIGGER guard_maple_pairing_registration_tombstones_truncate BEFORE TRUNCATE ON maple_pairing_registration_operation_tombstones",
        "CREATE TRIGGER guard_maple_pairing_installation_retirements_truncate BEFORE TRUNCATE ON maple_pairing_installation_retirements",
        "CREATE TRIGGER guard_maple_pairing_reset_clear_obligations_truncate BEFORE TRUNCATE ON maple_pairing_reset_clear_obligations",
        "CREATE TRIGGER guard_maple_pairing_reset_clear_admissions_truncate BEFORE TRUNCATE ON maple_pairing_reset_clear_admissions",
    ] {
        assert!(
            normalized_up.contains(required_pattern),
            "reset-clear storage guard must retain `{required_pattern}`"
        );
    }

    assert_patterns_in_order(
        down,
        &[
            "opensecret.allow_destructive_maple_pairing_down",
            "DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_admissions_truncate",
            "DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_obligations_truncate",
            "DROP TRIGGER IF EXISTS guard_maple_pairing_registration_tombstones_truncate",
            "DROP TRIGGER IF EXISTS guard_maple_pairing_installation_retirements_truncate",
            "DROP TABLE IF EXISTS maple_pairing_reset_clear_admissions",
            "DROP TABLE IF EXISTS maple_pairing_installation_retirements",
            "DROP TABLE IF EXISTS maple_pairing_reset_clear_obligations",
            "DROP TABLE IF EXISTS maple_pairing_registration_operation_tombstones",
            "DROP COLUMN known_security_epoch",
            "ADD CONSTRAINT maple_device_registration_operations_user_id_fkey",
        ],
    );
}

#[test]
fn maple_reset_clear_inventory_authenticates_epoch_receipts_and_exact_retained_rows() {
    let db_source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/db.rs"));

    for required_constant in [
        "const MAPLE_PAIRING_AUTHORITY_PAGE_SIZE: i64 = 256;",
        "const MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE: i64 = 64;",
        "const MAPLE_PAIRING_AUTHORITY_INSTALLATION_RETIREMENT_LIMIT: i64 = 1024;",
        "const MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_OBLIGATION_LIMIT: i64 = 4096;",
        "const MAPLE_PAIRING_AUTHORITY_RESET_CLEAR_ADMISSION_LIMIT: i64 = 524_288;",
        "const MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION: i64 = 128;",
    ] {
        assert!(
            db_source.contains(required_constant),
            "reset-clear inventory must retain `{required_constant}`"
        );
    }

    let tombstone_mac = extract_function_body(
        db_source,
        "fn maple_device_registration_tombstone_record_mac_for_parts(",
    );
    for required_field in [
        ".append_bytes(authority_scope_digest)",
        ".append_bytes(lookup_digest)",
        ".append_bytes(operation_lookup_digest)",
        ".append_i64(retired_security_epoch)",
        ".append_bytes(request_mac)",
        ".append_i16(outcome_kind)",
        ".append_bytes(outcome_digest)",
        ".append_i16(receipt_version)",
        ".append_bytes(receipt_enc)",
        ".append_bytes(receipt_digest)",
        ".append_u16(",
        "referenced_issuer_key_ids",
        ".unwrap_or(u16::MAX)",
        "for key_id in referenced_issuer_key_ids",
        "body.append_str(key_id)",
        ".append_i64(accepted_at.timestamp_micros())",
        ".append_i64(retired_at.timestamp_micros())",
    ] {
        assert!(
            tombstone_mac.contains(required_field),
            "tombstone record MAC must bind `{required_field}`"
        );
    }
    let tombstone_validation = extract_function_body(
        db_source,
        "fn validate_maple_device_registration_tombstone(",
    );
    for required_gate in [
        "row.retired_security_epoch > current_security_epoch",
        "row.receipt_version != MAPLE_PAIRING_RECEIPT_VERSION_V1",
        "row.receipt_enc.is_empty()",
        "row.receipt_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES",
        "row.receipt_digest.len() != 32",
        "maple_pairing_issuer_key_ids_are_canonical(&row.referenced_issuer_key_ids, 4)",
        "row.accepted_at > row.retired_at",
        "decrypt_maple_device_registration_tombstone_receipt(enclave_key, row)",
        "sha256_digest(&receipt)",
        ".ct_eq(row.receipt_digest.as_slice())",
        "serde_json::from_slice(&receipt)",
        "referenced_issuer_key_ids.insert(sync.stream_checkpoint.issuer_key_id.clone())",
        "referenced_issuer_key_ids.insert(instruction.issuer_key_id.clone())",
        "referenced_issuer_key_ids != row.referenced_issuer_key_ids",
    ] {
        assert!(
            tombstone_validation.contains(required_gate),
            "registration tombstone validation must retain `{required_gate}`"
        );
    }
    let retirement_mac = extract_function_body(
        db_source,
        "fn maple_installation_retirement_record_mac_for_parts(",
    );
    for required_field in [
        ".append_bytes(authority_scope_digest)",
        ".append_bytes(lookup_digest)",
        ".append_bytes(host_identity_mac)",
        ".append_i64(retired_security_epoch)",
        ".append_uuid(final_obligation_event_id)",
        ".append_bytes(final_instruction_digest)",
        ".append_bytes(final_chain_digest)",
        ".append_bytes(ack_host_registration_lookup_digest)",
        ".append_bytes(ack_operation_lookup_digest)",
        ".append_bytes(ack_request_mac)",
        ".append_i16(ack_receipt_version)",
        ".append_str(ack_receipt_issuer_key_id)",
        ".append_bytes(ack_receipt_digest)",
        ".append_i64(retired_at.timestamp_micros())",
        ".append_i64(created_at.timestamp_micros())",
    ] {
        assert!(
            retirement_mac.contains(required_field),
            "installation retirement MAC must bind `{required_field}`"
        );
    }
    let retirement_validation =
        extract_function_body(db_source, "fn validate_maple_installation_retirement(");
    for required_gate in [
        "row.authority_scope_digest",
        ".ct_eq(authority_scope_digest)",
        "row.lookup_digest.len() != 32",
        "row.host_identity_mac.len() != 32",
        "row.retired_security_epoch > current_security_epoch",
        "row.final_obligation_event_id.is_nil()",
        "row.final_instruction_digest.len() != 32",
        "row.final_chain_digest.len() != 32",
        "row.ack_host_registration_lookup_digest.len() != 32",
        "row.ack_operation_lookup_digest.len() != 32",
        "row.ack_request_mac.len() != 32",
        "maple_pairing_issuer_key_id_is_valid(&row.ack_receipt_issuer_key_id)",
        "row.ack_receipt_digest.len() != 32",
        "row.retired_at != row.created_at",
        "expected.as_slice().ct_eq(row.record_mac.as_slice())",
    ] {
        assert!(
            retirement_validation.contains(required_gate),
            "installation retirement validation must retain `{required_gate}`"
        );
    }
    let obligation_mac = extract_function_body(
        db_source,
        "fn maple_pairing_reset_clear_obligation_record_mac(",
    );
    for required_field in [
        ".append_uuid(row.uuid)",
        ".append_bytes(&row.authority_scope_digest)",
        ".append_bytes(&row.lookup_digest)",
        ".append_bytes(&row.host_identity_mac)",
        ".append_uuid(row.reset_id)",
        ".append_i64(row.reset_generation)",
        ".append_i64(row.cumulative_reset_count)",
        "row.previous_event_id",
        "row.previous_instruction_digest.as_deref()",
        "row.previous_chain_digest.as_deref()",
        ".append_uuid(row.old_revocation_stream_id)",
        ".append_i64(row.old_revocation_stream_generation)",
        ".append_i64(row.source_security_epoch)",
        ".append_i64(row.source_last_issued_revocation_sequence)",
        ".append_uuid(row.target_revocation_stream_id)",
        ".append_i64(row.target_revocation_stream_generation)",
        ".append_i64(row.target_security_epoch)",
        ".append_i64(row.target_instruction_sequence)",
        ".append_i16(row.clear_scope)",
        ".append_bytes(&row.admission_set_digest)",
        ".append_i16(row.admission_count)",
        ".append_bytes(&row.host_claim_payload_enc)",
        ".append_bytes(&row.host_claim_digest)",
        ".append_bytes(&row.instruction_payload_enc)",
        ".append_bytes(&row.instruction_digest)",
        ".append_bytes(&row.chain_digest)",
        "row.signed_instruction_payload_enc.as_deref()",
        "row.signed_instruction_issuer_key_id.as_deref()",
        "row.signed_instruction_digest.as_deref()",
        "row.sync_payload_enc.as_deref()",
        "row.sync_issuer_key_id.as_deref()",
        "row.sync_digest.as_deref()",
        "row.acked_by_head_event_id",
        "row.ack_operation_id",
        "row.ack_host_registration_lookup_digest.as_deref()",
        "row.ack_request_mac.as_deref()",
        "row.ack_receipt_enc.as_deref()",
        "row.ack_receipt_issuer_key_id.as_deref()",
        "row.ack_receipt_digest.as_deref()",
        ".append_i64(row.created_at.timestamp_micros())",
    ] {
        assert!(
            obligation_mac.contains(required_field),
            "reset-clear obligation record MAC must bind `{required_field}`"
        );
    }
    let admission_mac = extract_function_body(
        db_source,
        "fn maple_pairing_reset_clear_admission_record_mac(",
    );
    assert_patterns_in_order(
        admission_mac,
        &[
            "maple_pairing_reset_clear_admission_record_mac_for_parts(",
            "row.obligation_uuid",
            "&row.authority_scope_digest",
            "&row.lookup_digest",
            "row.pair_id",
            "row.pairing_incarnation",
            "&row.pair_authorization_digest",
            "row.created_at",
        ],
    );
    let admission_mac_for_parts = extract_function_body(
        db_source,
        "fn maple_pairing_reset_clear_admission_record_mac_for_parts(",
    );
    for required_field in [
        ".append_uuid(obligation_uuid)",
        ".append_bytes(authority_scope_digest)",
        ".append_bytes(lookup_digest)",
        ".append_uuid(pair_id)",
        ".append_i64(pairing_incarnation)",
        ".append_bytes(pair_authorization_digest)",
        ".append_i64(created_at.timestamp_micros())",
    ] {
        assert!(
            admission_mac_for_parts.contains(required_field),
            "reset-clear admission record MAC must bind `{required_field}`"
        );
    }
    let admission_aggregate = extract_function_body(
        db_source,
        "fn validate_maple_pairing_reset_clear_admission_aggregate(",
    );
    for required_gate in [
        "actual_count != i64::from(obligation.admission_count)",
        "Sha256::digest(aggregate.into_bytes())",
        "maple_pairing_authority_mac_matches(&expected, &obligation.admission_set_digest)",
    ] {
        assert!(
            admission_aggregate.contains(required_gate),
            "reset-clear admission aggregate must retain `{required_gate}`"
        );
    }

    let head_mac = extract_function_body(db_source, "fn maple_pairing_authority_account_head_mac(");
    for required_field in [
        ".append_i64(head.security_epoch)",
        ".append_i64(head.registration_operation_tombstone_count)",
        ".append_i64(head.installation_retirement_count)",
        ".append_i64(head.reset_clear_obligation_count)",
        ".append_i64(head.reset_clear_admission_count)",
    ] {
        assert!(
            head_mac.contains(required_field),
            "account-head MAC must bind `{required_field}`"
        );
    }
    for required_count_binding in [
        ".checked_add(self.installation_retirements)",
        ".append_i64(self.installation_retirements)",
        "(\"installation_retirements\", counts.installation_retirements)",
        "installation_retirements: maple_pairing_installation_retirements::table",
        "head.installation_retirement_count = counts.installation_retirements",
    ] {
        assert!(
            db_source.contains(required_count_binding),
            "retirement count must remain capacity-, inventory-, and head-bound via `{required_count_binding}`"
        );
    }
    let inventory_header = extract_function_body(
        db_source,
        "fn maple_pairing_authority_account_inventory_hasher(",
    );
    let inventory_header_signature = extract_method_signature(
        db_source,
        "maple_pairing_authority_account_inventory_hasher",
        ')',
    );
    assert!(
        inventory_header_signature.contains("security_epoch: i64"),
        "account inventory header must accept a signed 64-bit security epoch"
    );
    for required_field in [
        ".append_i64(security_epoch)",
        "counts.append_to(&mut header)",
    ] {
        assert!(
            inventory_header.contains(required_field),
            "account inventory header must bind `{required_field}`"
        );
    }
    let account_creation = extract_function_body(
        db_source,
        "fn create_empty_maple_pairing_authority_account_head(",
    );
    let account_candidate = account_creation
        .split_once("let mut candidate = MaplePairingAuthorityAccountHead {")
        .expect("account-head creation must build the MAC candidate explicitly")
        .1
        .split_once("};")
        .expect("account-head MAC candidate must have a bounded struct literal")
        .0;
    assert!(
        account_candidate.contains("security_epoch: 1"),
        "the account-head MAC candidate itself must begin at security epoch one"
    );
    assert_patterns_in_order(
        account_creation,
        &[
            "let mut candidate = MaplePairingAuthorityAccountHead {",
            "security_epoch: 1",
            "candidate.record_mac = maple_pairing_authority_account_head_mac(enclave_key, &candidate)?",
            "NewMaplePairingAuthorityAccountHead {",
            "security_epoch: 1",
        ],
    );

    let registration_receipt = extract_function_body(
        db_source,
        "fn maple_device_registration_operation_receipt_mac_for_parts(",
    );
    for required_field in [
        ".append_bytes(authority_scope_digest)",
        ".append_bytes(lookup_digest)",
        ".append_bytes(operation_lookup_digest)",
        ".append_i64(known_security_epoch)",
        ".append_i64(accepted_security_epoch)",
        ".append_i16(response_kind)",
        ".append_i16(sync_payload_version)",
        ".append_bytes(sync_payload_enc)",
        ".append_str(sync_issuer_key_id)",
        ".append_bytes(sync_digest)",
    ] {
        assert!(
            registration_receipt.contains(required_field),
            "registration receipt MAC must bind `{required_field}`"
        );
    }
    assert!(
        db_source
            .matches("maple_device_registration_operation_receipt_mac_for_parts(")
            .count()
            >= 3,
        "the validation wrapper and new-operation publication must use the complete v6 registration receipt MAC"
    );

    let registration_validation = extract_function_body(
        db_source,
        "fn validate_maple_device_registration_operation(",
    );
    for required_gate in [
        "operation.accepted_security_epoch != operation.known_security_epoch",
        "operation.sync_payload_enc.is_empty()",
        "operation.sync_payload_enc.len() > MAPLE_PAIRING_MAX_ENCRYPTED_PAYLOAD_BYTES",
        "maple_pairing_issuer_key_id_is_valid(&operation.sync_issuer_key_id)",
        "operation.sync_digest.len() != 32",
        ".ct_eq(operation.authority_scope_digest.as_slice())",
        ".ct_eq(operation.lookup_digest.as_slice())",
        ".ct_eq(operation.operation_lookup_digest.as_slice())",
        ".ct_eq(operation.receipt_mac.as_slice())",
    ] {
        assert!(
            registration_validation.contains(required_gate),
            "registration operation validation must retain `{required_gate}`"
        );
    }

    let inventory = extract_function_body(
        db_source,
        "fn compute_maple_pairing_authority_account_inventory(",
    );
    assert_patterns_in_order(
        inventory,
        &[
            "head.security_epoch",
            "\"device_operations\"",
            "validate_maple_device_registration_operation(",
            ".append_bytes(&row.authority_scope_digest)",
            ".append_bytes(&row.lookup_digest)",
            ".append_bytes(&row.operation_lookup_digest)",
            ".append_i64(row.known_security_epoch)",
            ".append_i64(row.accepted_security_epoch)",
            ".append_i16(row.response_kind)",
            ".append_i16(row.sync_payload_version)",
            ".append_bytes(&row.sync_payload_enc)",
            ".append_str(&row.sync_issuer_key_id)",
            ".append_bytes(&row.sync_digest)",
            ".append_bytes(&row.receipt_mac)",
        ],
    );

    let tombstones_start = inventory
        .find("\"registration_operation_tombstones\"")
        .expect("tombstone inventory category should exist");
    let tombstones_end = inventory[tombstones_start..]
        .find("\"installation_retirements\"")
        .map(|offset| tombstones_start + offset)
        .expect("tombstone inventory should precede installation retirements");
    let tombstones = &inventory[tombstones_start..tombstones_end];
    for required_pattern in [
        ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
        "validate_maple_device_registration_tombstone(",
        "registration_operation_lookups.insert(row.operation_lookup_digest.clone())",
        ".append_bytes(&row.authority_scope_digest)",
        ".append_bytes(&row.lookup_digest)",
        ".append_bytes(&row.operation_lookup_digest)",
        ".append_i64(row.retired_security_epoch)",
        ".append_bytes(&row.request_mac)",
        ".append_i16(row.outcome_kind)",
        ".append_bytes(&row.outcome_digest)",
        ".append_i16(row.receipt_version)",
        ".append_bytes(&row.receipt_digest)",
        "row.referenced_issuer_key_ids",
        "leaf.append_str(key_id)",
        ".append_i64(row.accepted_at.timestamp_micros())",
        ".append_bytes(&row.record_mac)",
        ".append_i64(row.retired_at.timestamp_micros())",
        "tombstones_seen != counts.registration_operation_tombstones",
    ] {
        assert!(
            tombstones.contains(required_pattern),
            "tombstone inventory must page, validate, and hash exact rows via `{required_pattern}`"
        );
    }

    let retirements_start = inventory
        .find("\"installation_retirements\"")
        .expect("installation retirement inventory category should exist");
    let retirements_end = inventory[retirements_start..]
        .find("\"lineages\"")
        .map(|offset| retirements_start + offset)
        .expect("installation retirement inventory should precede lineages");
    let retirements = &inventory[retirements_start..retirements_end];
    for required_pattern in [
        ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
        "validate_maple_installation_retirement(",
        "retired_lookups.insert(row.lookup_digest.clone())",
        "retired_identities.insert(row.host_identity_mac.clone())",
        "retired_host_registrations",
        ".insert(row.ack_host_registration_lookup_digest.clone())",
        "retirement_ack_operations.insert(row.ack_operation_lookup_digest.clone())",
        "maple_pairing_reset_clear_obligations::uuid",
        ".eq(row.final_obligation_event_id)",
        "expected_ack_operation_lookup",
        "obligation.state != 2",
        "obligation.revision != 3",
        "obligation.acked_by_head_event_id != Some(obligation.uuid)",
        "obligation.target_security_epoch != row.retired_security_epoch",
        "obligation.host_identity_mac != row.host_identity_mac",
        "obligation.instruction_digest != row.final_instruction_digest",
        "obligation.chain_digest != row.final_chain_digest",
        "obligation.ack_host_registration_lookup_digest.as_deref()",
        "Some(row.ack_host_registration_lookup_digest.as_slice())",
        "obligation.ack_request_mac.as_deref() != Some(row.ack_request_mac.as_slice())",
        "obligation.ack_receipt_version != Some(row.ack_receipt_version)",
        "obligation.ack_receipt_issuer_key_id.as_deref()",
        "obligation.ack_receipt_digest.as_deref()",
        "obligation.acked_at != Some(row.retired_at)",
        ".append_bytes(&row.authority_scope_digest)",
        ".append_bytes(&row.lookup_digest)",
        ".append_bytes(&row.host_identity_mac)",
        ".append_i64(row.retired_security_epoch)",
        ".append_uuid(row.final_obligation_event_id)",
        ".append_bytes(&row.final_instruction_digest)",
        ".append_bytes(&row.final_chain_digest)",
        ".append_bytes(&row.ack_host_registration_lookup_digest)",
        ".append_bytes(&row.ack_operation_lookup_digest)",
        ".append_bytes(&row.ack_request_mac)",
        ".append_i16(row.ack_receipt_version)",
        ".append_str(&row.ack_receipt_issuer_key_id)",
        ".append_bytes(&row.ack_receipt_digest)",
        ".append_i64(row.retired_at.timestamp_micros())",
        ".append_bytes(&row.record_mac)",
        ".append_i64(row.created_at.timestamp_micros())",
        "retirements_seen != counts.installation_retirements",
    ] {
        assert!(
            retirements.contains(required_pattern),
            "installation retirement inventory must validate and hash exact terminal proof via `{required_pattern}`"
        );
    }
    for required_same_epoch_proof in [
        "let mut current_epoch_tombstone_lookups = BTreeSet::new()",
        "row.retired_security_epoch == head.security_epoch",
        "current_epoch_tombstone_lookups.insert(row.lookup_digest.clone())",
        ".all(|lookup| retired_lookups.contains(lookup))",
    ] {
        assert!(
            inventory.contains(required_same_epoch_proof),
            "same-epoch tombstones must require a retained retirement proof via `{required_same_epoch_proof}`"
        );
    }
    for required_retired_lineage_gate in [
        "retired_lookups.contains(&lookup)",
        "retired_identities.contains(&device.identity_mac)",
        "Some(obligation) if obligation.state == 2",
        "if !retired_lookups.contains(lookup)",
    ] {
        assert!(
            inventory.contains(required_retired_lineage_gate),
            "inventory must reject live/ACKed authority without exact retirement proof via `{required_retired_lineage_gate}`"
        );
    }

    let obligations_start = inventory
        .find("\"reset_clear_obligations\"")
        .expect("reset-clear obligation inventory category should exist");
    let obligations_end = inventory[obligations_start..]
        .find("\"reset_clear_admissions\"")
        .map(|offset| obligations_start + offset)
        .expect("obligation inventory should precede admissions");
    let obligations = &inventory[obligations_start..obligations_end];
    for required_pattern in [
        ".append_uuid(row.uuid)",
        ".append_bytes(&row.authority_scope_digest)",
        ".append_bytes(&row.lookup_digest)",
        ".append_bytes(&row.host_identity_mac)",
        "row.previous_event_id",
        "row.previous_instruction_digest.as_deref()",
        "row.previous_chain_digest.as_deref()",
        ".append_uuid(row.old_revocation_stream_id)",
        ".append_i64(row.old_revocation_stream_generation)",
        ".append_i64(row.source_security_epoch)",
        ".append_i64(row.source_last_issued_revocation_sequence)",
        ".append_uuid(row.target_revocation_stream_id)",
        ".append_i64(row.target_revocation_stream_generation)",
        ".append_i64(row.target_security_epoch)",
        ".append_i64(row.target_instruction_sequence)",
        ".append_bytes(&row.admission_set_digest)",
        ".append_i16(row.admission_count)",
        ".append_bytes(&row.instruction_digest)",
        ".append_bytes(&row.chain_digest)",
        ".append_bytes(&row.record_mac)",
    ] {
        assert!(
            obligations.contains(required_pattern),
            "obligation inventory leaf must hash exact authenticated state via `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        obligations,
        &[
            "append_maple_pairing_reset_generation_counts(",
            "row.reset_generation",
            "row.cumulative_reset_count",
        ],
    );
    let reset_generation_counts = extract_function_body(
        db_source,
        "fn append_maple_pairing_reset_generation_counts(",
    );
    assert_patterns_in_order(
        reset_generation_counts,
        &[
            ".append_i64(reset_generation)",
            ".append_i64(cumulative_reset_count)",
        ],
    );
    for required_pattern in [
        ".limit(MAPLE_PAIRING_AUTHORITY_CIPHERTEXT_PAGE_SIZE)",
        "validate_maple_pairing_reset_clear_obligation(",
        "predecessor.lookup_digest != row.lookup_digest",
        "&predecessor.host_identity_mac",
        "predecessor.target_security_epoch != row.source_security_epoch",
        "row.previous_instruction_digest.as_deref()",
        "Some(predecessor.instruction_digest.as_slice())",
        "row.previous_chain_digest.as_deref()",
        "Some(predecessor.chain_digest.as_slice())",
        "predecessor.reset_generation.checked_add(1) != Some(row.reset_generation)",
        "row.reset_at < predecessor.reset_at",
        "row.reset_generation != 1",
        "obligations_seen != counts.reset_clear_obligations",
        "event_ids.sort_by_key(",
        "obligations_by_target_namespace.len() != highwater_transitions.len()",
    ] {
        assert!(
            inventory.contains(required_pattern),
            "obligation inventory must page, validate, and hash the authenticated chain via `{required_pattern}`"
        );
    }

    let admissions = &inventory[obligations_end..];
    for required_pattern in [
        ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
        "validate_maple_pairing_reset_clear_admission(",
        "obligation.lookup_digest != row.lookup_digest",
        ".append_uuid(row.obligation_uuid)",
        ".append_bytes(&row.authority_scope_digest)",
        ".append_bytes(&row.lookup_digest)",
        ".append_uuid(row.pair_id)",
        ".append_i64(row.pairing_incarnation)",
        ".append_bytes(&row.pair_authorization_digest)",
        ".append_bytes(&row.record_mac)",
        "admissions_seen != counts.reset_clear_admissions",
        "*child_count > i64::from(obligation.admission_count)",
        "*child_count > MAPLE_PAIRING_RESET_CLEAR_ADMISSION_LIMIT_PER_OBLIGATION",
        "admission_counts_by_event.get(event_id).copied()",
        "CanonicalBytes::new(\"os.maple-reset-clear-admission-set.v1\")",
        "maple_pairing_reset_clear_admissions::obligation_uuid.asc()",
        "maple_pairing_reset_clear_admissions::pair_id.asc()",
        "maple_pairing_reset_clear_admissions::pairing_incarnation.asc()",
        "validate_maple_pairing_reset_clear_admission_aggregate(",
        "canonical_admissions_seen != counts.reset_clear_admissions",
        "row.admission_count == 0",
        "latest_obligation.state == 1",
        "latest_highwater.security_epoch != head.security_epoch",
    ] {
        assert!(
            admissions.contains(required_pattern),
            "admission inventory must page, validate, aggregate, and classify exact rows via `{required_pattern}`"
        );
    }
}

#[test]
fn maple_reset_clear_pending_control_is_a_complete_highwater_host_union() {
    let db_source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/db.rs"));

    let pending_loader = extract_function_body(
        db_source,
        "fn load_latest_pending_maple_reset_clear_obligation(",
    );
    for required_gate in [
        "authority_scope_digest\n                .eq(&highwater.authority_scope_digest)",
        "lookup_digest.eq(&highwater.lookup_digest)",
        "state.eq(1_i16)",
        "reset_generation.desc()",
        "validate_maple_pairing_reset_clear_obligation(",
        "&highwater.authority_scope_digest",
        "&row.lookup_digest, &highwater.lookup_digest",
        "row.target_revocation_stream_id != highwater.revocation_stream_id",
        "row.target_revocation_stream_generation != highwater.revocation_stream_generation",
        "row.target_security_epoch != highwater.security_epoch",
        "row.target_instruction_sequence != 1",
        "highwater.last_issued_revocation_sequence != 1",
    ] {
        assert!(
            pending_loader.contains(required_gate),
            "pending reset-clear lookup must retain `{required_gate}`"
        );
    }

    let seed = extract_function_body(db_source, "fn seed_or_validate_maple_pairing_host_state(");
    assert_patterns_in_order(
        seed,
        &[
            "validate_maple_pairing_revocation_highwater(enclave_key, highwater)?",
            "load_latest_pending_maple_reset_clear_obligation(conn, enclave_key, highwater, true)?",
            "&pending.host_identity_mac, host_identity_mac",
            "if let Some(state) = existing",
            "validate_maple_pairing_host_state(enclave_key, &state)?",
            "pending_reset.is_some()",
            "state.last_issued_revocation_sequence != 1",
            "state.last_acked_revocation_sequence != 0",
            "state.revision != 2",
            "let (last_issued, last_acked, revision) = if pending_reset.is_some()",
            "(1_i64, 0_i64, 2_i64)",
            "highwater.last_issued_revocation_sequence == 0",
            "(0_i64, 0_i64, 1_i64)",
            "return Err(DBError::MaplePairingCorrupt)",
        ],
    );

    let inventory = extract_function_body(
        db_source,
        "fn compute_maple_pairing_authority_account_inventory(",
    );
    for required_union_gate in [
        "let mut highwater_transitions:",
        "previous_epoch.checked_add(1) != Some(row.security_epoch)",
        "let mut obligations_by_target_namespace = BTreeMap::new()",
        "obligations_by_target_namespace.len() != highwater_transitions.len()",
        "let mut live_lookups = BTreeSet::new()",
        "let mut reset_control_state_by_device = BTreeMap::new()",
        "if obligation.state == 1 && !is_current_control",
        "&device.identity_mac",
        "&obligation.host_identity_mac",
        "1 if state.last_issued_revocation_sequence == 1",
        "&& state.last_acked_revocation_sequence == 0",
        "for (lookup, latest) in &latest_by_lookup",
        "if !live_lookups.contains(lookup)",
        "Some(obligation) if obligation.state == 1",
        "latest_obligation.target_security_epoch != head.security_epoch",
        "latest.last_issued_revocation_sequence != 1",
        "None if latest.last_issued_revocation_sequence == 0",
        "None => return Err(DBError::MaplePairingAuthorityCorrupt)",
    ] {
        assert!(
            inventory.contains(required_union_gate),
            "account inventory must classify the complete live/retained reset-control union via `{required_union_gate}`"
        );
    }

    let pending_gate = extract_function_body(db_source, "fn require_no_pending_reset_clear(");
    assert_patterns_in_order(
        pending_gate,
        &[
            "load_maple_pairing_revocation_highwater(",
            "highwater.ok_or(DBError::MaplePairingCorrupt)?",
            "load_latest_pending_maple_reset_clear_obligation(",
            "&pending.host_identity_mac, &device.identity_mac",
            "return Err(DBError::MaplePairingResetClearRequired)",
        ],
    );
}

#[test]
fn maple_pairing_authority_root_and_destructive_down_are_fail_closed() {
    let up = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/up.sql"
    ));
    let down = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/2026-08-13-120000_maple_pairings_v1/down.sql"
    ));
    let destructive_down_setting = "opensecret.allow_destructive_maple_pairing_down";

    let first_down_statement = down
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("--"))
        .expect("destructive down migration must contain its fail-closed guard");
    assert_eq!(
        first_down_statement, "DO $$",
        "the disposable-database guard must be the first executable down-migration statement"
    );
    assert!(
        !up.contains(destructive_down_setting),
        "the disposable rollback GUC must never bypass Active-root or parent-head guards"
    );
    assert_eq!(
        down.matches(destructive_down_setting).count(),
        1,
        "only the explicitly destructive down script may consult its disposable-test GUC"
    );
    assert_patterns_in_order(
        down,
        &[
            "current_setting(",
            destructive_down_setting,
            "IS DISTINCT FROM 'disposable-test-only'",
            "DELETE FROM app_data_migrations",
        ],
    );

    let global_guard = extract_sql_function(up, "enforce_maple_pairing_authority_global_mutation");
    for required_pattern in [
        "IF TG_OP = 'INSERT'",
        "Maple pairing authority root cannot be recreated",
        "ELSIF TG_OP = 'DELETE'",
        "Maple pairing authority root cannot be removed",
        "ELSIF OLD.activation_state = 2",
        "IF NEW.activation_state <> 2",
        "active Maple pairing authority root cannot be downgraded",
        "pending Maple pairing authority root permits only revision-2 activation",
    ] {
        assert!(
            global_guard.contains(required_pattern),
            "Active-root guard must contain `{required_pattern}`"
        );
    }
    assert!(
        !global_guard.contains("current_setting")
            && !global_guard.contains("allow_destructive_maple_pairing_down"),
        "Active-root UPDATE/DELETE rejection must be unconditional"
    );
    assert!(
        normalize_whitespace(up).contains(
            "CREATE TRIGGER guard_maple_pairing_authority_global_mutation BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_authority_global_heads"
        ),
        "the unconditional Active-root DML guard must cover INSERT, UPDATE, and DELETE"
    );

    let marker_guard = extract_sql_function(up, "enforce_maple_pairing_authority_marker_mutation");
    assert!(
        marker_guard.contains("maple_pairing_authority_v1_activated")
            && marker_guard.contains("active Maple pairing authority marker is immutable")
            && !marker_guard.contains("current_setting"),
        "the Active activation marker must be unconditionally immutable"
    );
    let hierarchy_guard =
        extract_sql_function(up, "enforce_maple_pairing_authority_hierarchy_commit");
    let scoped_head_guard =
        extract_sql_function(up, "enforce_maple_pairing_authority_scoped_head_mutation");
    for immutable_project_identity in [
        "NEW.project_id IS DISTINCT FROM OLD.project_id",
        "NEW.org_id IS DISTINCT FROM OLD.org_id",
        "NEW.project_uuid IS DISTINCT FROM OLD.project_uuid",
        "NEW.subject_project_id IS DISTINCT FROM OLD.subject_project_id",
    ] {
        assert!(
            scoped_head_guard.contains(immutable_project_identity),
            "project authority-head identity must be immutable via `{immutable_project_identity}`"
        );
    }
    let normalized_up = normalize_whitespace(up);
    for exact_project_identity_constraint in [
        "ADD CONSTRAINT maple_pairing_authority_projects_identity_unique UNIQUE (id, org_id, uuid, client_id)",
        "CONSTRAINT maple_pairing_authority_project_identity_unique UNIQUE (project_id, org_id, project_uuid, subject_project_id)",
        "FOREIGN KEY (project_id, org_id, project_uuid, subject_project_id) REFERENCES org_projects(id, org_id, uuid, client_id)",
    ] {
        assert!(
            normalized_up.contains(exact_project_identity_constraint),
            "project authority hierarchy must bind the exact internal/public identity tuple via `{exact_project_identity_constraint}`"
        );
    }
    assert!(
        normalize_whitespace(down).contains(
            "ALTER TABLE org_projects DROP CONSTRAINT IF EXISTS maple_pairing_authority_projects_identity_unique, DROP CONSTRAINT IF EXISTS maple_pairing_authority_projects_scope_unique"
        ),
        "destructive rollback must remove both Maple project identity anchors"
    );
    for required_pattern in [
        "IF root_count <> 1",
        "Maple pairing authority root cardinality is invalid",
        "pending Maple pairing authority must remain empty and unmarked",
        "active Maple pairing authority marker is missing",
        "TG_TABLE_NAME = 'maple_pairing_authority_global_heads'",
        "OLD.activation_state = 1",
        "NEW.activation_state = 2",
        "active Maple pairing authority hierarchy is incomplete",
        "TG_TABLE_NAME IN ('orgs', 'maple_pairing_authority_org_heads')",
        "TG_TABLE_NAME IN ('users', 'maple_pairing_authority_account_heads')",
        "parent_exists IS DISTINCT FROM head_exists",
        "active Maple pairing project ancestry is incomplete",
        "active Maple pairing account ancestry is incomplete",
        "unknown Maple pairing authority hierarchy relation",
    ] {
        assert!(
            hierarchy_guard.contains(required_pattern),
            "deferred authority hierarchy guard must contain `{required_pattern}`"
        );
    }
    assert!(
        !hierarchy_guard.contains("current_setting"),
        "hierarchy completeness checks must not have a custom-GUC bypass"
    );
    for retained_table in [
        "maple_pairing_registration_operation_tombstones",
        "maple_pairing_installation_retirements",
        "maple_pairing_reset_clear_obligations",
        "maple_pairing_reset_clear_admissions",
    ] {
        let emptiness_probe = format!("EXISTS (SELECT 1 FROM {retained_table} LIMIT 1)");
        assert!(
            global_guard.contains(&emptiness_probe),
            "Pending-to-Active root transition must reject retained leaf `{retained_table}`"
        );
        assert!(
            hierarchy_guard.contains(&emptiness_probe),
            "Pending hierarchy sentinel must reject retained leaf `{retained_table}`"
        );
    }
    let activation_scan = hierarchy_guard
        .find("IF TG_TABLE_NAME = 'maple_pairing_authority_global_heads'")
        .expect("Pending-to-Active transition must establish the complete hierarchy base case");
    let steady_state_scopes = hierarchy_guard
        .find("IF TG_TABLE_NAME IN ('orgs', 'maple_pairing_authority_org_heads')")
        .expect("steady-state hierarchy guard must branch on indexed OLD/NEW scopes");
    assert!(
        activation_scan < steady_state_scopes,
        "complete activation proof must precede steady-state scoped preservation"
    );
    let activation_branch = &hierarchy_guard[activation_scan..steady_state_scopes];
    for relation in ["orgs", "org_projects", "users"] {
        assert!(
            activation_branch.contains(&format!("LEFT JOIN {relation}"))
                || activation_branch.contains(&format!("FROM {relation}")),
            "one-time activation must completely scan `{relation}`"
        );
    }
    let steady_state_branch = &hierarchy_guard[steady_state_scopes..];
    assert!(
        !steady_state_branch.contains("LEFT JOIN"),
        "Active steady-state events must use indexed scoped equivalence/ancestry checks, not global anti-joins"
    );
    assert_eq!(
        steady_state_branch.matches("IF parent_exists THEN").count(),
        4,
        "project/account ancestry must be required only while the scoped parent remains, including deferred DELETE events"
    );
    for required_pattern in [
        "IF TG_OP <> 'INSERT'",
        "IF TG_OP <> 'DELETE'",
        "OLD.org_id",
        "NEW.org_id",
        "OLD.project_id",
        "NEW.project_id",
        "OLD.user_id",
        "NEW.user_id",
        "OLD.uuid",
        "NEW.uuid",
        "OLD.client_id",
        "NEW.client_id",
        "OLD.project_uuid",
        "NEW.project_uuid",
        "OLD.subject_project_id",
        "NEW.subject_project_id",
    ] {
        assert!(
            steady_state_branch.contains(required_pattern),
            "steady-state hierarchy preservation must check OLD/NEW indexed scope via `{required_pattern}`"
        );
    }
    let normalized_scopes = normalize_whitespace(steady_state_branch);
    assert!(
        normalized_scopes.contains(
            "IF TG_TABLE_NAME = 'org_projects' THEN IF TG_OP = 'UPDATE' THEN IF NEW.id IS DISTINCT FROM OLD.id OR NEW.org_id IS DISTINCT FROM OLD.org_id OR NEW.uuid IS DISTINCT FROM OLD.uuid OR NEW.client_id IS DISTINCT FROM OLD.client_id THEN RAISE EXCEPTION 'active Maple pairing project identity cannot be replaced'; END IF; END IF; END IF;"
        ),
        "project-parent identity fields must be resolved only inside relation- and operation-specific procedural branches"
    );
    assert!(
        !normalized_scopes.contains("IF TG_TABLE_NAME = 'org_projects' AND TG_OP = 'UPDATE' AND"),
        "table-specific OLD/NEW fields must not be hidden behind boolean short-circuiting"
    );
    for required_pattern in [
        "IF TG_TABLE_NAME = 'orgs' THEN scope_org_id := OLD.id; ELSE scope_org_id := OLD.org_id; END IF;",
        "IF TG_TABLE_NAME = 'orgs' THEN scope_org_id := NEW.id; ELSE scope_org_id := NEW.org_id; END IF;",
        "IF TG_TABLE_NAME = 'org_projects' THEN scope_project_id := OLD.id; scope_project_uuid := OLD.uuid; scope_subject_project_id := OLD.client_id; ELSE scope_project_id := OLD.project_id; scope_project_uuid := OLD.project_uuid; scope_subject_project_id := OLD.subject_project_id; END IF;",
        "IF TG_TABLE_NAME = 'org_projects' THEN scope_project_id := NEW.id; scope_project_uuid := NEW.uuid; scope_subject_project_id := NEW.client_id; ELSE scope_project_id := NEW.project_id; scope_project_uuid := NEW.project_uuid; scope_subject_project_id := NEW.subject_project_id; END IF;",
        "IF TG_TABLE_NAME = 'users' THEN scope_user_id := OLD.uuid;",
        "IF TG_TABLE_NAME = 'users' THEN scope_user_id := NEW.uuid;",
    ] {
        assert!(
            normalized_scopes.contains(required_pattern),
            "deferred hierarchy guard must derive the correct parent/head OLD/NEW key via `{required_pattern}`"
        );
    }
    for exact_tuple_pattern in [
        "AND h.project_uuid = p.uuid AND h.subject_project_id = p.client_id",
        "AND p.uuid = h.project_uuid AND p.client_id = h.subject_project_id",
        "AND uuid = scope_project_uuid AND client_id = scope_subject_project_id",
        "AND project_uuid = scope_project_uuid AND subject_project_id = scope_subject_project_id",
        "active Maple pairing project identity cannot be replaced",
    ] {
        assert!(
            normalize_whitespace(hierarchy_guard).contains(exact_tuple_pattern),
            "project hierarchy checks must compare the exact authenticated alias tuple via `{exact_tuple_pattern}`"
        );
    }
    for (trigger_name, table_name) in [
        (
            "guard_maple_pairing_authority_global_commit",
            "maple_pairing_authority_global_heads",
        ),
        (
            "guard_maple_pairing_authority_org_head_commit",
            "maple_pairing_authority_org_heads",
        ),
        (
            "guard_maple_pairing_authority_project_head_commit",
            "maple_pairing_authority_project_heads",
        ),
        (
            "guard_maple_pairing_authority_account_head_commit",
            "maple_pairing_authority_account_heads",
        ),
        ("guard_maple_pairing_authority_org_parent_commit", "orgs"),
        (
            "guard_maple_pairing_authority_project_parent_commit",
            "org_projects",
        ),
        ("guard_maple_pairing_authority_user_parent_commit", "users"),
        (
            "guard_maple_pairing_authority_marker_commit",
            "app_data_migrations",
        ),
    ] {
        let expected = format!(
            "CREATE CONSTRAINT TRIGGER {trigger_name} AFTER INSERT OR UPDATE OR DELETE ON {table_name} DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit()"
        );
        assert!(
            normalized_up.contains(&expected),
            "authority hierarchy relation `{table_name}` must retain deferred per-row guard `{trigger_name}`"
        );
        assert!(
            down.contains(&format!("DROP TRIGGER IF EXISTS {trigger_name}")),
            "destructive rollback must explicitly detach deferred guard `{trigger_name}`"
        );
    }

    for (trigger_name, table_name) in [
        (
            "guard_maple_pairing_authority_global_truncate",
            "maple_pairing_authority_global_heads",
        ),
        (
            "guard_maple_pairing_authority_org_head_truncate",
            "maple_pairing_authority_org_heads",
        ),
        (
            "guard_maple_pairing_authority_project_head_truncate",
            "maple_pairing_authority_project_heads",
        ),
        (
            "guard_maple_pairing_authority_account_head_truncate",
            "maple_pairing_authority_account_heads",
        ),
        ("guard_maple_pairing_authority_user_truncate", "users"),
        (
            "guard_maple_pairing_authority_project_truncate",
            "org_projects",
        ),
        ("guard_maple_pairing_authority_org_truncate", "orgs"),
        (
            "guard_maple_pairing_authority_marker_truncate",
            "app_data_migrations",
        ),
    ] {
        let expected = format!(
            "CREATE TRIGGER {trigger_name} BEFORE TRUNCATE ON {table_name} EXECUTE FUNCTION forbid_maple_pairing_authority_truncate()"
        );
        assert!(
            normalized_up.contains(&expected),
            "authority hierarchy relation `{table_name}` must reject TRUNCATE through `{trigger_name}`"
        );
        assert!(
            down.contains(&format!("DROP TRIGGER IF EXISTS {trigger_name}")),
            "destructive rollback must explicitly detach truncate guard `{trigger_name}`"
        );
    }

    for trigger_name in [
        "guard_maple_pairing_authority_user_parent_commit",
        "guard_maple_pairing_authority_project_parent_commit",
        "guard_maple_pairing_authority_org_parent_commit",
        "guard_maple_pairing_authority_user_truncate",
        "guard_maple_pairing_authority_project_truncate",
        "guard_maple_pairing_authority_org_truncate",
    ] {
        assert!(
            down.contains(&format!("DROP TRIGGER IF EXISTS {trigger_name}")),
            "destructive rollback must remove parent guard `{trigger_name}`"
        );
    }
    for function_name in [
        "forbid_maple_pairing_authority_truncate",
        "enforce_maple_pairing_authority_hierarchy_commit",
        "enforce_maple_pairing_authority_marker_mutation",
        "enforce_maple_pairing_authority_scoped_head_mutation",
        "enforce_maple_pairing_authority_global_mutation",
    ] {
        assert!(
            down.contains(&format!("DROP FUNCTION IF EXISTS {function_name}()")),
            "destructive rollback must remove authority function `{function_name}`"
        );
    }

    let normalized_down = normalize_whitespace(down);
    for restored_edge in [
        "ADD CONSTRAINT maple_devices_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE",
        "ADD CONSTRAINT maple_devices_project_id_fkey FOREIGN KEY (project_id) REFERENCES org_projects(id) ON DELETE CASCADE",
        "ADD CONSTRAINT maple_device_registration_operations_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE",
        "ADD CONSTRAINT maple_device_registration_operations_project_id_fkey FOREIGN KEY (project_id) REFERENCES org_projects(id) ON DELETE CASCADE",
        "ADD CONSTRAINT maple_device_registration_operations_scoped_device_fk FOREIGN KEY (maple_device_id, user_id, project_id) REFERENCES maple_devices(id, user_id, project_id) ON DELETE CASCADE",
    ] {
        assert!(
            normalized_down.contains(restored_edge),
            "rollback must restore the prior Maple-device edge `{restored_edge}`"
        );
    }
}

#[test]
fn maple_pairing_project_identity_is_mac_bound_and_precedes_materialization() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db =
        fs::read_to_string(manifest_dir.join("src/db.rs")).expect("DB source should be readable");
    let models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB model source should be readable");
    let schema = fs::read_to_string(manifest_dir.join("src/models/schema.rs"))
        .expect("generated schema source should be readable");

    for (source, anchor, fields) in [
        (
            models.as_str(),
            "pub(crate) struct NewMaplePairingAuthorityProjectHead",
            [
                "pub project_id: i32",
                "pub org_id: i32",
                "pub project_uuid: Uuid",
                "pub subject_project_id: Uuid",
                "pub account_inventory_digest: Vec<u8>",
            ],
        ),
        (
            models.as_str(),
            "pub(crate) struct MaplePairingAuthorityProjectHead",
            [
                "pub project_id: i32",
                "pub org_id: i32",
                "pub project_uuid: Uuid",
                "pub subject_project_id: Uuid",
                "pub account_inventory_digest: Vec<u8>",
            ],
        ),
        (
            schema.as_str(),
            "maple_pairing_authority_project_heads (project_id)",
            [
                "project_id -> Int4",
                "org_id -> Int4",
                "project_uuid -> Uuid",
                "subject_project_id -> Uuid",
                "account_inventory_digest -> Bytea",
            ],
        ),
    ] {
        let project_head = source
            .split_once(anchor)
            .unwrap_or_else(|| panic!("project-head declaration `{anchor}` must exist"))
            .1
            .split_once("\n}")
            .unwrap_or_else(|| panic!("project-head declaration `{anchor}` must be bounded"))
            .0;
        assert_patterns_in_order(project_head, &fields);
    }

    let head_mac = extract_function_body(&db, "fn maple_pairing_authority_project_head_mac(");
    assert_patterns_in_order(
        head_mac,
        &[
            "CanonicalBytes::new(MAPLE_PAIRING_AUTHORITY_PROJECT_HEAD_MAC_DOMAIN)",
            ".append_i32(head.project_id)",
            ".append_i32(head.org_id)",
            ".append_uuid(head.project_uuid)",
            ".append_uuid(head.subject_project_id)",
            ".append_bytes(&head.account_inventory_digest)",
            ".append_i64(head.account_count)",
            ".append_i64(head.revision)",
            ".append_i64(head.created_at.timestamp_micros())",
        ],
    );
    let project_inventory =
        extract_function_body(&db, "fn compute_maple_pairing_authority_project_inventory(");
    assert_patterns_in_order(
        project_inventory,
        &[
            ".filter(org_projects::id.eq(project_id))",
            ".filter(org_projects::org_id.eq(org_id))",
            ".filter(org_projects::uuid.eq(project_uuid))",
            ".filter(org_projects::client_id.eq(subject_project_id))",
            ".append_i32(project_id)",
            ".append_i32(org_id)",
            ".append_uuid(project_uuid)",
            ".append_uuid(subject_project_id)",
            ".append_i64(account_count)",
        ],
    );
    for org_leaf in [
        extract_function_body(&db, "fn maple_pairing_authority_org_inventory_digest("),
        extract_function_body(&db, "fn append_maple_pairing_authority_project_head_leaf("),
    ] {
        assert_patterns_in_order(
            org_leaf,
            &[
                ".append_i32(project.project_id)",
                ".append_i32(project.org_id)",
                ".append_uuid(project.project_uuid)",
                ".append_uuid(project.subject_project_id)",
                ".append_bytes(&project.account_inventory_digest)",
            ],
        );
    }

    assert_struct_does_not_derive(
        "src/db.rs",
        &db,
        "MaplePairingAuthenticatedProjectIdentity",
        &["Clone", "Copy", "Debug", "Serialize", "Deserialize"],
    );
    let authenticated_identity = db
        .split_once("pub(crate) struct MaplePairingAuthenticatedProjectIdentity")
        .expect("authenticated project identity must exist")
        .1
        .split_once("\n}")
        .expect("authenticated project identity must have a bounded declaration")
        .0;
    assert!(
        !authenticated_identity.contains("pub ") && !authenticated_identity.contains("pub(crate)"),
        "authenticated project identity fields must remain private"
    );
    assert_patterns_in_order(
        authenticated_identity,
        &[
            "project_id: i32",
            "org_id: i32",
            "project_uuid: Uuid",
            "subject_project_id: Uuid",
        ],
    );
    let authenticated_identity_impl = db
        .split_once("impl MaplePairingAuthenticatedProjectIdentity")
        .expect("authenticated project identity implementation must exist")
        .1
        .split_once("struct MaplePairingAuthorityInventoryHasher")
        .expect("authenticated project identity implementation must stay bounded")
        .0;
    assert!(
        authenticated_identity_impl.contains("fn from_verified_head(")
            && !authenticated_identity_impl.contains("pub(crate) fn from_verified_head(")
            && !authenticated_identity_impl.contains("pub fn from_verified_head("),
        "authenticated project identity construction must remain private to verified DB state"
    );
    assert!(
        authenticated_identity_impl.contains("pub(crate) fn subject_project_id(&self) -> Uuid")
            && authenticated_identity_impl
                .matches("pub(crate) fn ")
                .count()
                == 1,
        "only the authenticated public project alias may be exposed crate-wide"
    );

    let authority_entry =
        extract_function_body(&db, "fn enter_maple_pairing_authority_account_transaction(");
    assert_patterns_in_order(
        authority_entry,
        &[
            "acquire_maple_pairing_authority_snapshot_fence(",
            "expected_issuer_key_inventory_digest",
            "verify_maple_pairing_authority_scoped_chain(conn, enclave_key, &head)",
            "MaplePairingAuthenticatedProjectIdentity::from_verified_head(&project, timer)",
        ],
    );

    let db_impl = db
        .split_once("impl DBConnection for PostgresConnection")
        .expect("Postgres DB implementation must exist")
        .1;
    let create_pairing = extract_function_body(db_impl, "fn create_maple_pairing(");
    assert_patterns_in_order(
        create_pairing,
        &[
            "let project_identity = enter_maple_pairing_authority_account_transaction(",
            "request.subject_project_id != project_identity.subject_project_id()",
            "request.create_request.asserted_project_id",
            "!= project_identity.subject_project_id()",
            "SELECT nextval('maple_pairing_incarnation_seq') AS incarnation",
            "materialize(MaplePairingCreateMaterializationContext",
            "subject_project_id: project_identity.subject_project_id()",
        ],
    );
    let before_nextval = create_pairing
        .split_once("SELECT nextval('maple_pairing_incarnation_seq') AS incarnation")
        .expect("pairing create must reserve its incarnation after validation")
        .0;
    assert!(
        !before_nextval.contains("materialize(MaplePairingCreateMaterializationContext"),
        "a wrong project assertion must fail before callback invocation"
    );
}

#[test]
fn maple_pairing_generated_schema_patch_is_single_purpose_and_tracks_generator_order() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let diesel_config = fs::read_to_string(manifest_dir.join("diesel.toml"))
        .expect("Diesel configuration should be readable");
    let schema_patch = fs::read_to_string(manifest_dir.join("src/models/schema.patch"))
        .expect("generated schema patch should be readable");
    let schema = fs::read_to_string(manifest_dir.join("src/models/schema.rs"))
        .expect("generated schema should be readable");
    let maple_devices = fs::read_to_string(manifest_dir.join("src/models/maple_devices.rs"))
        .expect("Maple device models should be readable");
    let migration = fs::read_to_string(
        manifest_dir.join("migrations/2026-08-13-120000_maple_pairings_v1/up.sql"),
    )
    .expect("Maple pairing migration should be readable");

    assert_patterns_in_order(
        &diesel_config,
        &[
            "[print_schema]",
            "file = \"src/models/schema.rs\"",
            "patch_file = \"src/models/schema.patch\"",
            "[migrations_directory]",
        ],
    );
    assert_eq!(
        schema_patch
            .lines()
            .filter(|line| line.starts_with("@@ "))
            .count(),
        1,
        "the generated schema correction must remain a single contextual hunk"
    );
    let changed_lines: Vec<_> = schema_patch
        .lines()
        .filter(|line| {
            (line.starts_with('+') && !line.starts_with("+++"))
                || (line.starts_with('-') && !line.starts_with("---"))
        })
        .collect();
    assert_eq!(
        changed_lines,
        vec![
            "-        referenced_issuer_key_ids -> Array<Nullable<Text>>,",
            "+        referenced_issuer_key_ids -> Array<Text>,",
        ],
        "the schema patch may correct only the canonical non-null issuer-key array type"
    );
    assert_eq!(
        schema
            .matches("referenced_issuer_key_ids -> Array<Text>,")
            .count(),
        1,
        "the checked schema must contain exactly one non-null issuer-key array"
    );
    assert!(
        !schema.contains("referenced_issuer_key_ids -> Array<Nullable<Text>>,"),
        "the checked schema must not expose nullable issuer-key array elements"
    );

    assert_patterns_in_order(
        &schema,
        &[
            "maple_pairing_host_states (id)",
            "maple_pairing_installation_retirements (id)",
            "maple_pairing_issuer_keys (key_id)",
            "maple_pairing_lineages (id)",
            "maple_pairing_registration_operation_tombstones (id)",
            "maple_pairing_reset_clear_admissions (id)",
            "maple_pairing_reset_clear_obligations (id)",
            "maple_pairing_revocation_events (id)",
        ],
    );
    let maple_joinables: Vec<_> = schema
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("diesel::joinable!(maple_"))
        .collect();
    assert_eq!(
        maple_joinables,
        vec![
            "diesel::joinable!(maple_device_registration_operations -> maple_pairing_issuer_keys (sync_issuer_key_id));",
            "diesel::joinable!(maple_pairing_authority_org_heads -> maple_pairing_authority_global_heads (global_singleton));",
            "diesel::joinable!(maple_pairing_authority_org_heads -> orgs (org_id));",
            "diesel::joinable!(maple_pairing_authority_project_heads -> maple_pairing_authority_org_heads (org_id));",
            "diesel::joinable!(maple_pairing_installation_retirements -> maple_pairing_issuer_keys (ack_receipt_issuer_key_id));",
            "diesel::joinable!(maple_pairing_issuer_keys -> maple_pairing_authority_global_heads (global_singleton));",
            "diesel::joinable!(maple_pairing_operations -> maple_pairing_issuer_keys (receipt_issuer_key_id));",
            "diesel::joinable!(maple_pairing_revocation_events -> maple_pairing_issuer_keys (issuer_key_id));",
        ],
        "checked Maple joinables must remain the exact Diesel-generated set and order"
    );

    let registration_table = schema
        .split_once("maple_device_registration_operations (id)")
        .expect("registration-operation schema should exist")
        .1
        .split_once("\n    }\n}")
        .expect("registration-operation schema should be bounded")
        .0;
    let registration_model = maple_devices
        .split_once("pub(crate) struct MapleDeviceRegistrationOperation {")
        .expect("registration-operation model should exist")
        .1
        .split_once("\n}")
        .expect("registration-operation model should be bounded")
        .0;
    let physical_order = [
        "request_mac",
        "maple_device_id",
        "device_revision",
        "receipt_mac",
        "accepted_at",
        "authority_scope_digest",
        "lookup_digest",
        "operation_lookup_digest",
        "known_security_epoch",
        "accepted_security_epoch",
        "response_kind",
        "sync_payload_version",
        "sync_payload_enc",
        "sync_issuer_key_id",
        "sync_digest",
    ];
    assert_patterns_in_order(registration_table, &physical_order);
    assert_patterns_in_order(registration_model, &physical_order);

    let canonical_issuer_ids =
        extract_sql_function(&migration, "maple_pairing_issuer_key_ids_are_canonical");
    assert_patterns_in_order(
        canonical_issuer_ids,
        &[
            "SELECT key_ids IS NOT NULL",
            "AND array_ndims(key_ids) = 1",
            "AND cardinality(key_ids) BETWEEN 1 AND maximum_count",
            "AND array_position(key_ids, NULL) IS NULL",
            "AND key_ids = ARRAY(",
        ],
    );
}

#[test]
fn maple_pairing_authority_sql_regression_harness_covers_deferred_guards() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_dir = manifest_dir.join(".agents/skills/validate-opensecret/scripts");
    let disposable_runner = fs::read_to_string(script_dir.join("disposable_db_tests.sh"))
        .expect("disposable database test runner should be readable");
    let authority_runner =
        fs::read_to_string(script_dir.join("maple_pairing_authority_hierarchy_sql_tests.sh"))
            .expect("Maple authority SQL test runner should be readable");
    let fixture = fs::read_to_string(script_dir.join("maple_pairing_authority_hierarchy_case.sql"))
        .expect("Maple authority SQL case fixture should be readable");

    let cases = [
        ("success", "activation_complete"),
        ("failure", "activation_missing_head"),
        ("success", "project_mutable_updates"),
        ("success", "valid_scoped_lifecycle"),
        ("failure", "parent_head_mismatch"),
        ("failure", "head_parent_mismatch"),
        ("failure", "project_parent_move_mismatch"),
        ("failure", "project_internal_id_mutation"),
        ("failure", "project_uuid_mutation"),
        ("failure", "project_client_id_mutation"),
        ("failure", "project_head_alias_mutation"),
        ("failure", "project_head_identity_mismatch"),
        ("failure", "project_alias_reinsert"),
        ("failure", "missing_ancestor"),
        ("failure", "active_marker_delete"),
        ("failure", "active_root_delete"),
        ("failure", "active_root_downgrade"),
        ("failure", "truncate_guard"),
        ("failure", "tombstone_null_issuer_key_id"),
        ("failure", "tombstone_unknown_issuer_key_id"),
        ("success", "steady_state_scoped_no_global_scan"),
    ];
    for (expectation, test_case) in cases {
        assert!(
            authority_runner.contains(&format!("expect_{expectation} {test_case}")),
            "the authority SQL runner must execute `{test_case}` as an expected {expectation}"
        );
        assert!(
            fixture.contains(&format!("'{test_case}'")),
            "the authority SQL fixture must recognize `{test_case}`"
        );
    }
    let executed_case_count = authority_runner
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("expect_success ") || line.starts_with("expect_failure "))
        .count();
    assert_eq!(
        executed_case_count,
        cases.len(),
        "the disposable regression count must change whenever an authority SQL case is added or removed"
    );

    assert_patterns_in_order(
        &disposable_runner,
        &[
            "diesel migration run --locked-schema",
            "diesel migration redo --locked-schema",
            "bash \"$script_dir/maple_pairing_authority_hierarchy_sql_tests.sh\"",
            "test \"$authority_sql_count\" -eq 21",
        ],
    );
    assert!(
        disposable_runner.contains("authority_sql_count=0")
            && disposable_runner.contains("\"$authority_sql_count\" \"$aead_count\""),
        "disposable evidence must report the exact authority SQL case count"
    );
    let issuer_rotation_runner = disposable_runner
        .split_once("# Registry expansion is lifetime-persistent by design.")
        .expect("the disposable runner must isolate lifetime issuer-registry expansion")
        .1;
    assert_patterns_in_order(
        issuer_rotation_runner,
        &[
            "createdb -h \"$pgsockets\" -p \"$pgport\" -U \"$admin_user\"",
            "--owner=opensecret_user \"$issuer_rotation_test_database\"",
            "export MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL=\"postgres://opensecret_user:password@127.0.0.1:${pgport}/${issuer_rotation_test_database}\"",
            "diesel migration run --locked-schema",
            "--database-url \"$MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL\"",
            "SELECT count(*) FROM __diesel_schema_migrations",
            "SELECT current_database()",
            "= \"$issuer_rotation_test_database\"",
            "cargo test --locked --all-features aead_db_tamper_tests",
        ],
    );

    for required_named_constraint in [
        "SET CONSTRAINTS guard_maple_pairing_authority_org_head_commit IMMEDIATE;",
        "SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;",
        "SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit IMMEDIATE;",
    ] {
        assert!(
            fixture.contains(required_named_constraint),
            "the regression fixture must deterministically fire `{required_named_constraint}`"
        );
    }

    let project_mutable_updates = fixture
        .split_once("\\if :project_mutable_updates")
        .expect("mutable project/head success branch should exist")
        .1
        .split_once("\\endif")
        .expect("mutable project/head success branch should be bounded")
        .0;
    assert_patterns_in_order(
        project_mutable_updates,
        &[
            "UPDATE org_projects",
            "SET description =",
            "SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE",
            "SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit DEFERRED",
            "UPDATE maple_pairing_authority_project_heads",
            "SET account_inventory_digest =",
            "account_count =",
            "revision = revision + 1",
            "record_mac =",
            "SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit IMMEDIATE",
            "SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit DEFERRED",
        ],
    );
    assert!(
        authority_runner.contains("expect_success project_mutable_updates"),
        "mutable project metadata and project-head state must remain a successful named-constraint regression"
    );

    let missing_ancestor = fixture
        .split_once("\\if :missing_ancestor")
        .expect("missing-ancestor regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("missing-ancestor regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        missing_ancestor,
        &[
            "DISABLE TRIGGER guard_maple_pairing_authority_org_head_commit",
            "DELETE FROM maple_pairing_authority_org_heads",
            "UPDATE maple_pairing_authority_project_heads",
            "SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit IMMEDIATE",
        ],
    );
    assert!(
        !missing_ancestor.contains("ENABLE TRIGGER guard_maple_pairing_authority_org_head_commit"),
        "the expected rollback must restore the disabled trigger without DDL after pending events"
    );
    assert_patterns_in_order(
        &authority_runner,
        &[
            "expect_failure missing_ancestor",
            "'active Maple pairing project ancestry is incomplete'",
        ],
    );

    let truncate_guard = fixture
        .split_once("\\if :truncate_guard")
        .expect("TRUNCATE regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("TRUNCATE regression branch should be bounded")
        .0;
    assert!(
        truncate_guard.contains("TRUNCATE maple_pairing_authority_account_heads CASCADE;"),
        "the disposable TRUNCATE regression must reach the production guard across the FK closure"
    );
    for forbidden_bypass in [
        "DISABLE TRIGGER",
        "session_replication_role",
        "opensecret.allow_destructive_maple_pairing_down",
    ] {
        assert!(
            !truncate_guard.contains(forbidden_bypass),
            "the TRUNCATE regression must not bypass production enforcement"
        );
    }
    assert_patterns_in_order(
        &authority_runner,
        &[
            "expect_failure truncate_guard",
            "'TRUNCATE of Maple pairing authority state is forbidden'",
        ],
    );

    let active_root_delete = fixture
        .split_once("\\if :active_root_delete")
        .expect("Active-root DELETE regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("Active-root DELETE regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        active_root_delete,
        &[
            "SET LOCAL opensecret.allow_destructive_maple_pairing_down",
            "'disposable-test-only'",
            "DELETE FROM maple_pairing_authority_global_heads",
        ],
    );
    let active_root_downgrade = fixture
        .split_once("\\if :active_root_downgrade")
        .expect("Active-root downgrade regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("Active-root downgrade regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        active_root_downgrade,
        &[
            "SET LOCAL opensecret.allow_destructive_maple_pairing_down",
            "'disposable-test-only'",
            "UPDATE maple_pairing_authority_global_heads",
            "SET activation_state = 1",
        ],
    );
    for (test_case, expected_failure) in [
        (
            "expect_failure active_root_delete",
            "'Maple pairing authority root cannot be removed'",
        ),
        (
            "expect_failure active_root_downgrade",
            "'active Maple pairing authority root cannot be downgraded'",
        ),
    ] {
        assert_patterns_in_order(&authority_runner, &[test_case, expected_failure]);
    }

    let tombstone_null_issuer_key_id = fixture
        .split_once("\\if :tombstone_null_issuer_key_id")
        .expect("NULL issuer-key tombstone regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("NULL issuer-key tombstone regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        tombstone_null_issuer_key_id,
        &[
            "INSERT INTO users",
            "INSERT INTO maple_pairing_authority_account_heads",
            "SET CONSTRAINTS ALL IMMEDIATE",
            "SET CONSTRAINTS ALL DEFERRED",
            "INSERT INTO maple_pairing_registration_operation_tombstones",
            "ARRAY['issuer-a', NULL]::TEXT[]",
        ],
    );
    assert_patterns_in_order(
        &authority_runner,
        &[
            "expect_failure tombstone_null_issuer_key_id",
            "'violates check constraint \"maple_pairing_registration_operation_tombstones_receipt_shape\"'",
        ],
    );

    let tombstone_unknown_issuer_key_id = fixture
        .split_once("\\if :tombstone_unknown_issuer_key_id")
        .expect("unknown issuer-key tombstone regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("unknown issuer-key tombstone regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        tombstone_unknown_issuer_key_id,
        &[
            "INSERT INTO users",
            "INSERT INTO maple_pairing_authority_account_heads",
            "SET CONSTRAINTS ALL IMMEDIATE",
            "SET CONSTRAINTS ALL DEFERRED",
            "INSERT INTO maple_pairing_registration_operation_tombstones",
            "ARRAY['issuer-a']::TEXT[]",
        ],
    );
    assert_patterns_in_order(
        &authority_runner,
        &[
            "expect_failure tombstone_unknown_issuer_key_id",
            "'Maple registration tombstone references an unknown issuer key'",
        ],
    );

    let scoped_case = fixture
        .split_once("\\if :steady_state_scoped_no_global_scan")
        .expect("steady-state scoped regression branch should exist")
        .1
        .split_once("\\endif")
        .expect("steady-state scoped regression branch should be bounded")
        .0;
    assert_patterns_in_order(
        scoped_case,
        &[
            "DISABLE TRIGGER guard_maple_pairing_authority_org_parent_commit",
            "INSERT INTO orgs",
            "ENABLE TRIGGER guard_maple_pairing_authority_org_parent_commit",
            "INSERT INTO users",
            "INSERT INTO maple_pairing_authority_account_heads",
            "SET CONSTRAINTS ALL IMMEDIATE",
        ],
    );
    assert!(
        authority_runner.contains("expect_success steady_state_scoped_no_global_scan"),
        "an unrelated hierarchy gap must not make scoped steady-state verification scan globally"
    );
}

#[test]
fn maple_pairing_authority_bootstrap_is_one_way_and_never_recreates_active_state() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("db source should be readable");
    let global_loader = extract_function_body(&db, "fn load_maple_pairing_authority_global_head(");
    assert!(
        global_loader.contains(".optional()?")
            && global_loader.contains(".ok_or(DBError::MaplePairingAuthorityCorrupt)"),
        "a missing SQL-seeded global sentinel must fail closed"
    );
    assert!(
        !global_loader.contains("insert_into"),
        "application code must never lazily recreate a missing global sentinel"
    );

    let bootstrap =
        extract_function_body(&db, "fn bootstrap_or_audit_maple_pairing_authority_in_tx(");
    let active_marker = "MAPLE_PAIRING_AUTHORITY_ACTIVE => {";
    let pending_marker = "MAPLE_PAIRING_AUTHORITY_PENDING => {";
    let active_start = bootstrap
        .find(active_marker)
        .expect("bootstrap must have an explicit Active branch");
    let pending_start = bootstrap
        .find(pending_marker)
        .expect("bootstrap must have an explicit Pending branch");
    assert!(
        active_start < pending_start,
        "bootstrap branch ordering unexpectedly changed"
    );
    let active_branch = &bootstrap[active_start..pending_start];
    for required_pattern in [
        "if !marker_exists",
        "DBError::MaplePairingAuthorityCorrupt",
        "verify_maple_pairing_authority_tree(conn, enclave_key)",
    ] {
        assert!(
            active_branch.contains(required_pattern),
            "Active authority audit must contain `{required_pattern}`"
        );
    }
    for forbidden_pattern in [
        "insert_into",
        "create_empty_",
        "create_maple_pairing_authority_",
    ] {
        assert!(
            !active_branch.contains(forbidden_pattern),
            "Active authority audit must never recreate missing state via `{forbidden_pattern}`"
        );
    }

    let unknown_state_start = bootstrap[pending_start..]
        .find("_ => Err(DBError::MaplePairingAuthorityCorrupt)")
        .map(|offset| pending_start + offset)
        .expect("unknown or downgraded authority states must fail closed");
    let pending_branch = &bootstrap[pending_start..unknown_state_start];
    for required_pattern in [
        "if marker_exists",
        "maple_pairing_authority_leaf_tables_are_empty(conn)?",
        "NewAppDataMigration::new(MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER).insert(conn)?",
        ".eq(MAPLE_PAIRING_AUTHORITY_PENDING)",
        ".filter(maple_pairing_authority_global_heads::revision.eq(1_i64))",
        "if changed != 1",
        "verify_maple_pairing_authority_tree(conn, enclave_key)",
    ] {
        assert!(
            pending_branch.contains(required_pattern),
            "one-time Pending activation must contain `{required_pattern}`"
        );
    }
    assert!(
        !bootstrap.contains("insert_into(maple_pairing_authority_global_heads"),
        "only the SQL migration may create the global authority sentinel"
    );

    let leaf_emptiness =
        extract_function_body(&db, "fn maple_pairing_authority_leaf_tables_are_empty(");
    for retained_table in [
        "maple_pairing_registration_operation_tombstones::table",
        "maple_pairing_installation_retirements::table",
        "maple_pairing_reset_clear_obligations::table",
        "maple_pairing_reset_clear_admissions::table",
    ] {
        assert!(
            leaf_emptiness.contains(retained_table),
            "bootstrap must fail closed when retained authority leaf `{retained_table}` exists"
        );
    }
}

#[test]
fn maple_pairing_authority_lock_has_bounded_global_serialization() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("db source should be readable");
    for required_constant in [
        "const MAPLE_PAIRING_AUTHORITY_LOCK_KEY_1: i32 = 0x4d41_504c",
        "const MAPLE_PAIRING_AUTHORITY_LOCK_KEY_2: i32 = 0x4155_5448",
        "const MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT: Duration = Duration::from_secs(5)",
        "const MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(10)",
        "const MAPLE_PAIRING_AUTHORITY_STATEMENT_TIMEOUT: &str = \"30s\"",
    ] {
        assert!(
            db.contains(required_constant),
            "authority serialization policy must contain `{required_constant}`"
        );
    }

    let try_lock_result_start = db
        .find("#[derive(diesel::QueryableByName)]\nstruct MaplePairingAuthorityTryLockResult")
        .expect("typed advisory try-lock result should exist");
    let try_lock_result_end = db[try_lock_result_start..]
        .find("\n\nfn try_maple_pairing_authority_lock_once(")
        .map(|offset| try_lock_result_start + offset)
        .expect("typed advisory try-lock result should precede its query helper");
    let try_lock_result = &db[try_lock_result_start..try_lock_result_end];
    for required_pattern in [
        "#[derive(diesel::QueryableByName)]",
        "#[diesel(sql_type = diesel::sql_types::Bool)]",
        "acquired: bool",
    ] {
        assert!(
            try_lock_result.contains(required_pattern),
            "typed advisory try-lock result must contain `{required_pattern}`"
        );
    }
    let try_lock = extract_function_body(&db, "fn try_maple_pairing_authority_lock_once(");
    for required_pattern in [
        "SELECT pg_try_advisory_xact_lock($1, $2) AS acquired",
        "MAPLE_PAIRING_AUTHORITY_LOCK_KEY_1",
        "MAPLE_PAIRING_AUTHORITY_LOCK_KEY_2",
        ".get_result::<MaplePairingAuthorityTryLockResult>(conn)",
        ".map(|result| result.acquired)",
        "if matches!(&acquired, Ok(false))",
        "observe_maple_pairing_authority_lock_contention_if_armed_for_test()",
        "acquired",
    ] {
        assert!(
            try_lock.contains(required_pattern),
            "locale-independent advisory try-lock must contain `{required_pattern}`"
        );
    }

    let fence_mode = db
        .find("enum MaplePairingAuthoritySnapshotFenceMode")
        .expect("authority snapshot-fence mode should exist");
    let fence_mode_end = db[fence_mode..]
        .find("\n}\n\nfn acquire_maple_pairing_authority_snapshot_fence_with_mode(")
        .map(|offset| fence_mode + offset + 2)
        .expect("authority snapshot-fence mode should precede the core helper");
    let fence_mode = &db[fence_mode..fence_mode_end];
    for required_mode in ["ActiveOnly", "Bootstrap"] {
        assert!(
            fence_mode.contains(required_mode),
            "authority snapshot-fence mode must retain `{required_mode}`"
        );
    }

    let fence_body = extract_function_body(
        &db,
        "fn acquire_maple_pairing_authority_snapshot_fence_with_mode(",
    );
    for required_pattern in [
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        "SET LOCAL statement_timeout",
        "MAPLE_PAIRING_AUTHORITY_STATEMENT_TIMEOUT",
        "let started = Instant::now()",
        "if try_maple_pairing_authority_lock_once(conn)?",
        "started.elapsed() >= MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT",
        "event = \"maple_pairing_authority_lock_busy\"",
        "return Err(DBError::MaplePairingAuthorityBusy)",
        "std::thread::sleep(MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL)",
        "let global = load_maple_pairing_authority_global_head(conn)?",
        "match global.activation_state",
        "MAPLE_PAIRING_AUTHORITY_ACTIVE =>",
        "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
        "AppDataMigration::exists(conn, MAPLE_PAIRING_AUTHORITY_ACTIVATION_MARKER)?",
        "MAPLE_PAIRING_AUTHORITY_PENDING",
        "MaplePairingAuthoritySnapshotFenceMode::Bootstrap",
        "validate_pending_maple_pairing_authority_global_head(&global)?",
        "!maple_pairing_authority_leaf_tables_are_empty(conn)?",
        "_ => return Err(DBError::MaplePairingAuthorityCorrupt)",
    ] {
        assert!(
            fence_body.contains(required_pattern),
            "bounded authenticated authority snapshot fence must contain `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        fence_body,
        &[
            "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            "SET LOCAL statement_timeout",
            "let started = Instant::now()",
            "if try_maple_pairing_authority_lock_once(conn)?",
            "started.elapsed() >= MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT",
            "return Err(DBError::MaplePairingAuthorityBusy)",
            "std::thread::sleep(MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL)",
            "load_maple_pairing_authority_global_head(conn)?",
            "match global.activation_state",
            "MAPLE_PAIRING_AUTHORITY_ACTIVE =>",
            "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
            "MAPLE_PAIRING_AUTHORITY_PENDING",
            "MaplePairingAuthoritySnapshotFenceMode::Bootstrap",
            "validate_pending_maple_pairing_authority_global_head(&global)?",
        ],
    );
    let global_fence = fence_body
        .find("let global = load_maple_pairing_authority_global_head(conn)?")
        .expect("authority snapshot fence must lock its global root after advisory acquisition");
    let before_global_fence = &fence_body[..global_fence];
    for forbidden_authority_read in [
        "::table",
        ".first::<",
        ".load::<",
        "verify_maple_pairing_authority_",
        "validate_maple_pairing_authority_",
    ] {
        assert!(
            !before_global_fence.contains(forbidden_authority_read),
            "the pg_try polling phase must not read authority state via `{forbidden_authority_read}` before the global-root freshness fence"
        );
    }
    let legacy_lock_error_mapper = ["map_maple_pairing_authority_", "lock_error"].concat();
    for forbidden_pattern in [
        "SET LOCAL lock_timeout",
        "SELECT pg_advisory_xact_lock",
        ".message()",
        ".to_string()",
    ] {
        assert!(
            !fence_body.contains(forbidden_pattern) && !try_lock.contains(forbidden_pattern),
            "advisory-lock Busy classification must not depend on `{forbidden_pattern}`"
        );
    }
    assert!(
        !db.contains(&legacy_lock_error_mapper),
        "the removed database-error-text lock mapper must not be reintroduced"
    );
    for forbidden_scoped_relation in [
        "maple_pairing_authority_account_heads",
        "maple_pairing_authority_project_heads",
        "maple_pairing_authority_org_heads",
        "users::table",
        "org_projects::table",
        "orgs::table",
    ] {
        assert!(
            !fence_body.contains(forbidden_scoped_relation),
            "the global-root freshness fence must run before scoped relation `{forbidden_scoped_relation}` is accessed"
        );
    }

    let global_loader = extract_function_body(&db, "fn load_maple_pairing_authority_global_head(");
    assert_patterns_in_order(
        global_loader,
        &[
            "maple_pairing_authority_global_heads::table",
            "maple_pairing_authority_global_heads::singleton.eq(true)",
            ".for_update()",
            ".first::<MaplePairingAuthorityGlobalHead>(conn)",
            ".optional()?",
            ".ok_or(DBError::MaplePairingAuthorityCorrupt)",
        ],
    );

    let active_wrapper =
        extract_function_body(&db, "fn acquire_maple_pairing_authority_snapshot_fence(");
    assert_patterns_in_order(
        active_wrapper,
        &[
            "acquire_maple_pairing_authority_snapshot_fence_with_mode(",
            "conn",
            "enclave_key",
            "MaplePairingAuthoritySnapshotFenceMode::ActiveOnly",
            "Some(expected_issuer_key_inventory_digest)",
        ],
    );
    assert!(
        !active_wrapper.contains("MaplePairingAuthoritySnapshotFenceMode::Bootstrap"),
        "ordinary authority paths must never opt into Pending bootstrap mode"
    );
    let bootstrap_wrapper = extract_function_body(
        &db,
        "fn acquire_maple_pairing_authority_bootstrap_snapshot_fence(",
    );
    assert_patterns_in_order(
        bootstrap_wrapper,
        &[
            "acquire_maple_pairing_authority_snapshot_fence_with_mode(",
            "conn",
            "enclave_key",
            "MaplePairingAuthoritySnapshotFenceMode::Bootstrap",
            "None",
        ],
    );
    assert_eq!(
        db.matches("MaplePairingAuthoritySnapshotFenceMode::Bootstrap")
            .count(),
        2,
        "only the core Pending branch and startup-only wrapper may name bootstrap mode"
    );

    let no_text_regression = extract_function_body(
        &db,
        "fn authority_busy_classification_does_not_depend_on_database_error_text(",
    );
    for required_pattern in [
        "MAPLE_PAIRING_AUTHORITY_LOCK_TIMEOUT, Duration::from_secs(5)",
        "MAPLE_PAIRING_AUTHORITY_LOCK_RETRY_INTERVAL",
        "Duration::from_millis(10)",
        "DatabaseErrorKind::UnableToSendCommand",
        "Err(DBError::QueryError(_))",
    ] {
        assert!(
            no_text_regression.contains(required_pattern),
            "locale-independent Busy regression must contain `{required_pattern}`"
        );
    }

    let db_tests = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/aead_db_tamper_tests.rs"),
    )
    .expect("database tamper regression source should be readable");
    let stale_waiter = extract_function_body(
        &db_tests,
        "async fn db_contended_authority_waiter_aborts_stale_snapshot_then_retries_fresh(",
    );
    assert_patterns_in_order(
        stale_waiter,
        &[
            "pause_next_maple_device_registration_before_commit_for_test(operation_id)",
            "holder_reached",
            ".recv_timeout(Duration::from_secs(10))",
            "observe_next_maple_pairing_authority_lock_contention_for_test()",
            "list_test_maple_devices(waiter_db.as_ref(), waiter_authorization, 32, None)",
            "contention_observed",
            ".recv_timeout(Duration::from_secs(10))",
            "release_holder",
            ".send(())",
            "Err(DBError::MaplePairingAuthorityBusy)",
            "list_test_maple_devices(app_state.db.as_ref(), authorization, 32, None)",
            "assert_eq!(fresh.len(), 1)",
        ],
    );
    let list_adapter = extract_function_body(&db_tests, "fn list_test_maple_devices(");
    assert_patterns_in_order(
        list_adapter,
        &[
            "DBConnection::list_maple_devices(db, authorization, limit, after)",
            ".map(|page| page.devices)",
        ],
    );
    let issuer_rotation = extract_function_body(
        &db_tests,
        "async fn db_maple_pairing_lifecycle_is_ordered_and_destructive_cleanup_is_complete(",
    );
    assert_patterns_in_order(
        issuer_rotation,
        &[
            "MAPLE_ISSUER_ROTATION_TEST_DATABASE_URL",
            "let initial_keyset = test_maple_pairing_issuer_keyset(&[",
            "\"maple-test-issuer-current\"",
            "\"maple-test-issuer-old\"",
            "\"maple-test-issuer-revocation\"",
            "let issuer_root_before_rotation = maple_pairing_issuer_inventory_state(&app_state)",
            "assert_eq!(issuer_root_before_rotation.0, 3)",
            "let missing_old_keyset = test_maple_pairing_issuer_keyset(&[",
            "Maple pairing issuer keyset conflicts with the authenticated lifetime registry",
            "maple_pairing_issuer_inventory_state(&app_state)",
            "issuer_root_before_rotation",
            "let retained_rotation_keyset = test_maple_pairing_issuer_keyset(&[",
            "\"maple-test-issuer-future\"",
            "let issuer_root_after_rotation = maple_pairing_issuer_inventory_state(&retained_state)",
            "assert_eq!(issuer_root_after_rotation.0, 4)",
            "issuer_root_after_rotation.1",
            "issuer_root_before_rotation.1",
            "issuer_root_after_rotation.2",
            "issuer_root_before_rotation.2 + 1",
            "audit_maple_pairing_issuer_key_references(&app_state.enclave_key)",
            "Err(DBError::MaplePairingIssuerConfigurationConflict)",
            "let app_state = retained_state",
        ],
    );

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let register = extract_function_body(db_impl, "fn register_maple_device(");
    assert_patterns_in_order(
        register,
        &[
            "enter_maple_pairing_authority_account_transaction(",
            "commit_maple_pairing_authority_account_mutation(",
            "pause_maple_device_registration_before_commit_if_armed_for_test(",
            "Ok(receipt)",
        ],
    );
}

#[test]
fn maple_pairing_authority_lock_enters_serializable_before_any_database_access() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("DB source should be readable");
    let active_fence_call = "acquire_maple_pairing_authority_snapshot_fence(";
    let bootstrap_fence_call = "acquire_maple_pairing_authority_bootstrap_snapshot_fence(";
    let fence_body = extract_function_body(
        &db,
        "fn acquire_maple_pairing_authority_snapshot_fence_with_mode(",
    );
    let first_sql_query = fence_body
        .find("diesel::sql_query")
        .expect("authority snapshot fence must execute its transaction-isolation statement");
    let serializable = fence_body
        .find("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        .expect("authority snapshot fence must select SERIALIZABLE isolation");
    assert!(
        first_sql_query < serializable,
        "SERIALIZABLE must be the first SQL statement in the authority snapshot fence"
    );
    assert_patterns_in_order(
        fence_body,
        &[
            "diesel::sql_query",
            "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            ".execute(conn)?",
            "SET LOCAL statement_timeout",
            "try_maple_pairing_authority_lock_once(conn)?",
            "load_maple_pairing_authority_global_head(conn)?",
            "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
        ],
    );
    assert!(
        !fence_body.contains("SET LOCAL lock_timeout")
            && !fence_body.contains("SELECT pg_advisory_xact_lock"),
        "authority locking must remain a locale-independent bounded pg_try loop"
    );

    let validate_global =
        extract_function_body(&db, "fn validate_maple_pairing_authority_global_head(");
    for required_pattern in [
        "head.record_mac.as_deref()",
        "maple_pairing_authority_global_head_mac(enclave_key, head)",
        "head.activation_state != MAPLE_PAIRING_AUTHORITY_ACTIVE",
        "maple_pairing_authority_mac_matches(&expected, actual_mac)",
    ] {
        assert!(
            validate_global.contains(required_pattern),
            "normal authority paths must authenticate an Active global root via `{required_pattern}`"
        );
    }
    for verifier in [
        "fn verify_maple_pairing_authority_scoped_chain(",
        "fn verify_maple_pairing_authority_global_shallow(",
        "fn verify_maple_pairing_authority_tree_with_mode(",
    ] {
        let verifier_body = extract_function_body(&db, verifier);
        assert_patterns_in_order(
            verifier_body,
            &[
                "load_maple_pairing_authority_global_head(conn)?",
                "validate_maple_pairing_authority_global_head(enclave_key, &global)?",
            ],
        );
        assert!(
            !verifier_body.contains("MAPLE_PAIRING_AUTHORITY_PENDING"),
            "normal verifier `{verifier}` must never accept a Pending root"
        );
    }
    let bootstrap =
        extract_function_body(&db, "fn bootstrap_or_audit_maple_pairing_authority_in_tx(");
    assert!(
        bootstrap.contains("MAPLE_PAIRING_AUTHORITY_PENDING => {")
            && bootstrap.contains("MAPLE_PAIRING_AUTHORITY_ACTIVE => {")
            && db.matches("MAPLE_PAIRING_AUTHORITY_PENDING => {").count() == 1,
        "Pending-root handling must remain isolated to one-time bootstrap"
    );

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let startup = extract_function_body(db_impl, "fn bootstrap_or_audit_maple_pairing_authority(");
    assert_patterns_in_order(
        startup,
        &[
            "run_maple_pairing_authority_transaction(",
            "acquire_maple_pairing_authority_bootstrap_snapshot_fence(tx, enclave_key)",
            "bootstrap_or_audit_maple_pairing_authority_in_tx(",
            "&configured_issuer_keys",
        ],
    );
    assert!(
        !startup.contains("acquire_maple_pairing_authority_snapshot_fence("),
        "startup must use the isolated bootstrap fence rather than requiring an already-Active root"
    );

    for (fence_call, fence_kind, minimum_callers) in [
        (active_fence_call, "Active", 8usize),
        (bootstrap_fence_call, "bootstrap", 1usize),
    ] {
        let mut direct_caller_count = 0usize;
        for (call_position, _) in db.match_indices(fence_call) {
            if db[..call_position].ends_with("fn ") {
                continue;
            }
            direct_caller_count += 1;
            let call = extract_rust_parenthesized_call(&db, call_position, fence_call);
            if fence_kind == "Active" {
                assert!(
                    call.contains("expected_issuer_key_inventory_digest"),
                    "direct Active authority-fence caller must pass the locally pinned issuer inventory digest"
                );
            }
            let caller_prefix = rust_function_body_prefix_before(&db, call_position);
            for forbidden_database_access in [
                "diesel::sql_query",
                "::table",
                ".first::<",
                ".load::<",
                ".get_result::<",
                ".execute(",
                ".insert(",
                ".update(",
                ".delete(",
            ] {
                assert!(
                    !caller_prefix.contains(forbidden_database_access),
                    "direct {fence_kind} authority-fence caller accessed the database via `{forbidden_database_access}` before selecting SERIALIZABLE and authenticating the global root"
                );
            }
        }
        assert!(
            direct_caller_count >= minimum_callers,
            "expected at least {minimum_callers} direct {fence_kind} authority-fence callers"
        );
    }

    assert_eq!(
        db.matches("acquire_maple_pairing_authority_snapshot_fence_with_mode(")
            .count(),
        3,
        "only the core definition and its Active/bootstrap wrappers may choose a snapshot-fence mode"
    );
}

#[test]
fn maple_pairing_create_gates_before_in_transaction_materialization_and_publication() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB models should be readable");
    let web = fs::read_to_string(manifest_dir.join("src/web/maple_pairings.rs"))
        .expect("Maple pairing route source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let create = extract_function_body(db_impl, "fn create_maple_pairing(");
    let create_signature = extract_method_signature(db_impl, "create_maple_pairing", '{');
    assert!(
        create_signature.contains("issuer_keyset: &MaplePairingIssuerKeySetV1")
            && create_signature.contains("materialize: &MaterializeMaplePairingCreate<'_>"),
        "DB CREATE must receive the immutable issuer verifier separately from the untrusted materializer callback"
    );

    let materializer_start = models
        .find("pub type MaterializeMaplePairingCreate")
        .expect("pairing CREATE materializer alias must exist");
    let materializer_end = models[materializer_start..]
        .find(';')
        .map(|offset| materializer_start + offset + 1)
        .expect("pairing CREATE materializer alias must terminate");
    let materializer = &models[materializer_start..materializer_end];
    assert!(
        materializer.contains("dyn Fn(")
            && materializer.contains("MaplePairingCreateMaterializationContext")
            && materializer
                .contains("Result<MaplePairingCreateMaterial, MaplePairingMaterializationError>",)
            && !materializer.contains("Future"),
        "pairing CREATE materialization must remain a synchronous, pure callback boundary"
    );
    assert!(
        !materializer.contains("bool"),
        "the materializer boundary must not treat callback-supplied booleans as proof of issuer or nonce verification"
    );

    assert_patterns_in_order(
        create,
        &[
            "enter_maple_pairing_authority_account_transaction(",
            "lock_maple_user_and_validate_credential(",
            "find_scoped_maple_device(",
            "request.controller_registration_id",
            "find_scoped_maple_device(",
            "request.host_registration_id",
            "require_no_pending_reset_clear(tx, authorization, &controller, true)?",
            "require_no_pending_reset_clear(tx, authorization, &host, true)?",
            "if let Some(prior) = get_prior_pairing_operation(",
            "return replay_pairing_operation(",
            "let (trusted_now, _) = expire_pending_pairings(tx, authorization)?",
            "request\n                    .create_request\n                    .validate()",
            "request.create_request.verify_signature()",
            "let request_transcript = request",
            "let expected_request_mac = maple_pairing_request_operation_mac(",
            "let controller_request_key = request",
            "controller_identity_key_bytes()",
            "let host_request_key = request",
            "host_identity_key_bytes()",
            "let controller_request_identity_mac = maple_device_identity_mac_from_claim(",
            "let host_request_identity_mac = maple_device_identity_mac_from_claim(",
            "let decoded_nonce = STANDARD",
            "let request_nonce_mac = maple_pairing_request_nonce_mac(",
            "let pairing_count = maple_pairings::table",
            "if pairing_count >= MAPLE_PAIRING_LIMIT_PER_ACCOUNT_PROJECT",
            "SELECT nextval('maple_pairing_incarnation_seq') AS incarnation",
            "let material = materialize(MaplePairingCreateMaterializationContext",
            ".map_err(|_| DBError::MaplePairingMaterializationFailed)?",
            "ticket\n                    .verify_unexpired(",
            "issuer_keyset",
            "trusted_now.timestamp_millis()",
            "MAPLE_PAIRING_CLOCK_SKEW_GRACE_MS",
            "let controller_ticket_key = ticket",
            "let host_ticket_key = ticket",
            "let controller_ticket_identity_mac = maple_device_identity_mac_from_claim(",
            "let host_ticket_identity_mac = maple_device_identity_mac_from_claim(",
            "let expected_response = MaplePairingMutationResponse {",
            "if response != expected_response",
            "let payload_enc = encrypt_maple_pairing_payload(",
            "let receipt_enc = encrypt_maple_pairing_receipt(",
            "let lineage = maple_pairing_lineages::table",
            "let pairing = diesel::insert_into(maple_pairings::table)",
            "let receipt = insert_pairing_operation(",
            "commit_maple_pairing_authority_account_mutation(",
            "take_maple_pairing_create_before_commit_failure_for_test(request.operation_id)",
            "Ok(receipt)",
        ],
    );
    assert_eq!(
        create.matches("materialize(").count(),
        1,
        "fresh CREATE must invoke the pure signer/materializer at exactly one post-gate point"
    );
    assert!(
        create.find("require_no_pending_reset_clear(")
            < create.find("SELECT nextval('maple_pairing_incarnation_seq')"),
        "pending reset-clear must reject CREATE before any incarnation allocation"
    );
    assert!(
        create.find("return replay_pairing_operation(") < create.find("materialize("),
        "an exact committed CREATE replay must return without invoking fresh materialization"
    );
    assert!(
        create.find("materialize(") < create.find("diesel::insert_into(maple_pairings::table)"),
        "materialization failure may burn only the nontransactional sequence value, not insert authority rows"
    );
    assert!(
        !create.contains("issuer_verified") && !create.contains("request_nonce_binding_verified"),
        "callback-supplied verification booleans are forgeable and cannot authorize persisted pairing authority"
    );
    assert!(
        create.contains("ticket\n                    .verify_unexpired(")
            && create.contains("maple_pairing_request_nonce_mac(")
            && create.contains("maple_pairing_request_operation_mac(")
            && create
                .matches("maple_device_identity_mac_from_claim(")
                .count()
                >= 4
            && create.contains("maple_pairing_authority_mac_matches(")
            && create.contains("encrypt_maple_pairing_payload(")
            && create.contains("encrypt_maple_pairing_receipt("),
        "DB must independently verify the request/ticket chain, constant-time bind both participant identities, and own durable AEAD"
    );
    let before_nextval = create
        .split_once("SELECT nextval('maple_pairing_incarnation_seq') AS incarnation")
        .expect("pairing create must reserve its incarnation after request verification")
        .0;
    assert_eq!(
        before_nextval
            .matches("maple_device_identity_mac_from_claim(")
            .count(),
        2,
        "both signed request endpoint keys must be rebound to locked identity MACs before nextval"
    );

    assert!(
        !db.contains("fn reserve_maple_pairing_incarnation(")
            && !web.contains("reserve_maple_pairing_incarnation(")
            && !web.contains("fn replay_or_create_pairing<T>("),
        "the standalone reservation and route-level replay/prebuild path must stay removed"
    );
    let web_materializer =
        extract_function_body(&web, "pub(crate) fn materialize_maple_pairing_create(");
    assert_patterns_in_order(
        web_materializer,
        &[
            "let expected_request_mac = request_operation_mac(",
            ".transcript()",
            "&context.create_request.signature",
            "expected_request_mac.ct_eq(&context.request_mac)",
            ".verify_signature()",
            "let controller = create_materialization_device_response(",
            "context.controller",
            "let host = create_materialization_device_response(",
            "context.host",
            "let pairing_request_id = Uuid::new_v4()",
            "let pair_id = Uuid::new_v4()",
            "sign_pair_request_ticket(",
            "MaplePairRequestTicketV1 {",
            "pairing_incarnation: context.pairing_incarnation",
            "issuer_key_id: String::new()",
            "issuer_signature: String::new()",
            "issuer,",
            "let response = MaplePairingMutationResponse {",
            "Ok(MaplePairingCreateMaterial {",
        ],
    );
    let material_fields = web_materializer
        .split_once("Ok(MaplePairingCreateMaterial {")
        .expect("production CREATE materializer must return the complete DB material bundle")
        .1;
    assert_patterns_in_order(material_fields, &["request_ticket: ticket", "response"]);
    for forbidden_callback_field in [
        "payload_enc",
        "payload_digest",
        "receipt_enc",
        "receipt_digest",
        "request_nonce_mac",
    ] {
        assert!(
            !material_fields.contains(forbidden_callback_field),
            "CREATE callback output must not carry DB-owned {forbidden_callback_field}"
        );
    }

    let request_pairing = extract_function_body(&web, "async fn request_pairing(");
    assert_patterns_in_order(
        request_pairing,
        &[
            "let (issuer, keyset) = require_pairing_crypto(&state)?",
            "let materialize = move |context: MaplePairingCreateMaterializationContext|",
            "materialize_maple_pairing_create(",
            "&enclave_key",
            "issuer.as_ref()",
            "internal_project_id",
            "context",
            ".create_maple_pairing(",
            "NewMaplePairingRequest {",
            "create_request: request.clone()",
            "keyset.as_ref()",
            "&materialize",
            "decrypt_receipt(",
            "encrypt_response(&state, &session_id, &response)",
        ],
    );
    assert_eq!(
        request_pairing.matches("encrypt_response(").count(),
        1,
        "only the committed DB receipt may cross the encrypted response boundary"
    );
    assert!(
        !request_pairing.contains("sign_pair_request_ticket(")
            && request_pairing
                .matches("materialize_maple_pairing_create(")
                .count()
                == 1,
        "the handler must delegate exact CREATE signing to the production materializer once"
    );
    assert!(
        !web_materializer.contains("issuer_verified")
            && !web_materializer.contains("request_nonce_binding_verified")
            && !request_pairing.contains("issuer_verified")
            && !request_pairing.contains("request_nonce_binding_verified"),
        "the route and materializer must not smuggle forgeable verification booleans into DB"
    );
}

#[test]
fn maple_pairing_revoke_uses_typed_authority_and_db_owned_ciphertext() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let models = fs::read_to_string(manifest_dir.join("src/models/maple_pairing_db.rs"))
        .expect("Maple pairing DB models should be readable");
    let web = fs::read_to_string(manifest_dir.join("src/web/maple_pairings.rs"))
        .expect("Maple pairing route source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let revoke = extract_function_body(db_impl, "fn revoke_maple_pairing(");
    let revoke_signature = extract_method_signature(db_impl, "revoke_maple_pairing", '{');
    assert!(
        revoke_signature.contains("issuer_keyset: &MaplePairingIssuerKeySetV1"),
        "DB REVOKE must receive the process-pinned issuer verifier independently"
    );

    let mutation = models
        .split_once("pub struct MaplePairingRevocation {")
        .expect("typed pairing revocation mutation must exist")
        .1
        .split_once('}')
        .expect("typed pairing revocation mutation must terminate")
        .0;
    assert!(
        mutation.contains("pub revoke_request: RevokeMaplePairingRequest"),
        "DB REVOKE must receive the exact actor-signed wire request"
    );
    let material = models
        .split_once("pub struct MaplePairingRevocationMaterial {")
        .expect("typed pairing revocation material must exist")
        .1
        .split_once('}')
        .expect("typed pairing revocation material must terminate")
        .0;
    for required_typed_field in [
        "pub request_ticket: MaplePairRequestTicketV1",
        "pub pair_authorization: MaplePairAuthorizationV1",
        "pub revocation: MaplePairRevocationV1",
        "pub response: MaplePairingMutationResponse",
    ] {
        assert!(
            material.contains(required_typed_field),
            "REVOKE callback must return {required_typed_field}"
        );
    }
    for forbidden_opaque_field in [
        "payload_enc",
        "event_digest",
        "issuer_key_id",
        "receipt_enc",
    ] {
        assert!(
            !material.contains(forbidden_opaque_field),
            "REVOKE callback must not return opaque {forbidden_opaque_field}"
        );
    }

    assert_patterns_in_order(
        revoke,
        &[
            "require_configured_maple_pairing_issuer_keyset(issuer_keyset)?",
            "let project_identity = enter_maple_pairing_authority_account_transaction(",
            "let revoke_request = &mutation.revoke_request",
            "revoke_request.validate().is_err()",
            "let expected_request_mac = maple_pairing_request_operation_mac(",
            "let current = maple_pairings::table",
            "let create_operation = maple_pairing_operations::table",
            "pairing_operation_receipt(",
            "let approval_operation = maple_pairing_operations::table",
            "pairing_operation_receipt(",
            "let controller = maple_devices::table",
            "let host = maple_devices::table",
            "let stored_payload = decrypt_maple_pairing_payload(",
            "let material = materialize(MaplePairingRevocationContext",
            "let MaplePairingRevocationMaterial {",
            "if ticket != stored_payload.request_ticket",
            ".verify_unexpired(",
            ".verify_against_ticket(issuer_keyset, &verified_ticket)",
            ".verify_against_authorization(issuer_keyset, &pair_authorization)",
            "let controller_request = ticket.controller_request()",
            "let expected_create_request_mac = maple_pairing_request_operation_mac(",
            "let host_approval_request = pair_authorization.host_approval_request()",
            "let expected_approval_request_mac = maple_pairing_request_operation_mac(",
            "let controller_identity_mac = maple_device_identity_mac_from_claim(",
            "let host_identity_mac = maple_device_identity_mac_from_claim(",
            "revoke_request\n                    .verify_signature(&actor_claim_key)",
            "let expected_response = MaplePairingMutationResponse {",
            "if response != expected_response",
            "let pair_payload_enc = encrypt_maple_pairing_payload(",
            "let event_payload_enc = encrypt_maple_pairing_revocation_payload(",
            "let event_digest = revocation",
            "let receipt_enc = encrypt_maple_pairing_receipt(",
            "diesel::insert_into(maple_pairing_revocation_events::table)",
            "diesel::update(",
            "insert_pairing_operation(",
            "commit_maple_pairing_authority_account_mutation(",
        ],
    );
    let before_first_mutation = revoke
        .split_once("diesel::insert_into(maple_pairing_revocation_events::table)")
        .expect("REVOKE must insert its event only after validation")
        .0;
    assert!(
        before_first_mutation.contains("revocation.reason_code != revoke_request.reason_code")
            && before_first_mutation.contains("revocation.issuer_sequence")
            && before_first_mutation.contains("revocation.revocation_stream_id")
            && before_first_mutation.contains("revocation.revoked_at_unix_ms")
            && before_first_mutation.contains("revocation.revoked_by_registration_id")
            && before_first_mutation.contains(
                "create_operation.operation_id != ticket.controller_request_operation_id",
            )
            && before_first_mutation.contains(
                "approval_operation.operation_id\n                        != pair_authorization.host_approval_operation_id",
            )
            && before_first_mutation.contains("expected_create_request_mac")
            && before_first_mutation.contains("expected_approval_request_mac")
            && before_first_mutation.contains("maple_pairing_authority_mac_matches(")
            && before_first_mutation.contains("DBError::MaplePairingMaterializationFailed"),
        "all signed REVOKE semantics and constant-time bindings must fail closed before mutation"
    );

    let web_revoke = extract_function_body(&web, "async fn revoke_pairing(");
    assert!(
        web_revoke.contains("Ok(MaplePairingRevocationMaterial {")
            && web_revoke.contains("request_ticket: ticket_for_material.clone()")
            && web_revoke.contains("pair_authorization: authorization_for_material.clone()")
            && web_revoke.contains("revocation,")
            && web_revoke.contains("response,")
            && !web_revoke.contains("encrypt_pair_payload(")
            && !web_revoke.contains("encrypt_revocation_payload(")
            && !web_revoke.contains("encrypt_receipt("),
        "the REVOKE callback must return typed artifacts only; DB owns every durable ciphertext"
    );
    assert_patterns_in_order(
        web_revoke,
        &[
            ".revoke_maple_pairing(",
            "revoke_request: request.clone()",
            "keyset.as_ref()",
            "&materialize",
            "decrypt_receipt(",
        ],
    );
}

#[test]
fn maple_pairing_routes_consume_device_pages_and_bind_the_signed_approval_digest() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let web = fs::read_to_string(manifest_dir.join("src/web/maple_pairings.rs"))
        .expect("Maple pairing route source should be readable");

    let load_devices = extract_function_body(&web, "fn load_devices(");
    assert_patterns_in_order(
        load_devices,
        &[
            "let page = state",
            ".list_maple_devices(",
            "device_list_authorization(state, user, auth_context)",
            "33",
            "None",
            ".map_err(map_pairing_db_error)?",
            "page.devices",
            ".into_iter()",
            "decrypt_device_response(",
            ".collect()",
        ],
    );

    let approval = extract_function_body(&web, "async fn approve_pairing(");
    assert_patterns_in_order(
        approval,
        &[
            "sign_pair_authorization(",
            "pair_authorization.verify_against_ticket(keyset, &verified_ticket)",
            "pair_authorization: Some(pair_authorization.clone())",
            "pair_authorization.issuer_key_id.clone()",
            "stored_wire(pair_authorization.digest())?.to_vec()",
            ".approve_maple_pairing(MaplePairingApproval {",
            "authorization_issuer_key_id",
            "pair_authorization_digest",
        ],
    );

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let persist_approval = extract_function_body(db_impl, "fn approve_maple_pairing(");
    assert_patterns_in_order(
        persist_approval,
        &[
            "Some(&mutation.pair_authorization_digest)",
            "maple_pairings::pair_authorization_digest",
            ".eq(Some(mutation.pair_authorization_digest))",
            "validate_maple_pairing_record(&authorization.enclave_key, &pairing)?",
        ],
    );
}

#[test]
fn maple_device_listing_returns_epoch_and_page_from_one_authority_snapshot() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let db = fs::read_to_string(manifest_dir.join("src/db.rs"))
        .expect("Maple pairing DB source should be readable");
    let models = fs::read_to_string(manifest_dir.join("src/models/maple_devices.rs"))
        .expect("Maple device models should be readable");
    let web = fs::read_to_string(manifest_dir.join("src/web/maple_devices.rs"))
        .expect("Maple device route source should be readable");

    let page = extract_function_body(&models, "pub struct MapleDeviceListPage");
    assert!(
        page.contains("pub security_epoch: u64") && page.contains("pub devices: Vec<MapleDevice>"),
        "the internal list result must carry the validated epoch beside its rows"
    );
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let list = extract_function_body(db_impl, "fn list_maple_devices(");
    assert_patterns_in_order(
        list,
        &[
            "enter_maple_pairing_authority_account_transaction(",
            "lock_maple_user_and_validate_credential(tx, &authorization, false)?",
            "let devices = query",
            ".limit(limit)",
            ".load::<MapleDevice>(tx)",
            "let head = maple_pairing_authority_account_heads::table",
            "validate_maple_pairing_authority_account_head(",
            "Ok(MapleDeviceListPage {",
            "security_epoch: pairing_u64_from_i64(head.security_epoch)?",
            "devices",
        ],
    );
    let route = extract_function_body(&web, "async fn list_devices(");
    assert_patterns_in_order(
        route,
        &[
            "let page = state",
            ".list_maple_devices(",
            "let security_epoch = page.security_epoch",
            "let mut rows = page.devices",
            "rows.len() > usize::from(limit)",
            "rows.truncate(usize::from(limit))",
            "ListMapleDevicesResponse {",
            "security_epoch",
            "devices",
            "next_cursor",
            "has_more",
        ],
    );
}

#[test]
fn maple_pairing_parent_crud_requires_the_enclave_key_and_authority_transaction() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("db source should be readable");
    let db_trait = extract_function_body(&db, "pub trait DBConnection");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");

    for method_name in [
        "create_user",
        "delete_user",
        "mark_and_delete_user",
        "create_org",
        "delete_org",
        "create_org_project",
        "delete_org_project",
        "create_org_with_owner",
    ] {
        let trait_signature = extract_method_signature(db_trait, method_name, ';');
        assert!(
            trait_signature.contains("enclave_key: &[u8]"),
            "DB trait method `{method_name}` must require the enclave authority key"
        );

        let implementation_signature = extract_method_signature(db_impl, method_name, '{');
        assert!(
            implementation_signature.contains("enclave_key: &[u8]")
                && !implementation_signature.contains("_enclave_key"),
            "PostgreSQL method `{method_name}` must consume the enclave authority key"
        );
        let implementation = extract_function_body(db_impl, &format!("fn {method_name}("));
        assert!(
            implementation.contains("run_maple_pairing_authority_transaction("),
            "PostgreSQL method `{method_name}` must atomically mutate its parent and authority heads"
        );
        assert!(
            implementation.contains("acquire_maple_pairing_authority_snapshot_fence(")
                || implementation.contains("create_user_with_maple_authority_in_tx("),
            "PostgreSQL method `{method_name}` must enter the globally serialized, Active-root-authenticated authority transaction"
        );
    }

    let create_user_in_tx =
        extract_function_body(&db, "fn create_user_with_maple_authority_in_tx(");
    assert_patterns_in_order(
        create_user_in_tx,
        &[
            "acquire_maple_pairing_authority_snapshot_fence(",
            "expected_issuer_key_inventory_digest",
            ".for_update()",
            "verify_maple_pairing_authority_project_chain(",
            "new_user.insert(",
            "create_empty_maple_pairing_authority_account_head(",
            "cascade_maple_pairing_authority_heads(",
            "verify_maple_pairing_authority_scoped_chain(",
        ],
    );
}

#[test]
fn maple_pairing_serialization_failure_is_busy_only_for_retry_safe_outer_transactions() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("DB source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let main_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let main = fs::read_to_string(main_source).expect("main source should be readable");

    for required_pattern in [
        "enum MaplePairingAuthorityTransactionClass",
        "ReadOnly",
        "ReplaySafeMutation",
        "NonReplayableMutation",
    ] {
        assert!(
            db.contains(required_pattern),
            "authority transaction classification must contain `{required_pattern}`"
        );
    }
    let class_policy = extract_function_body(&db, "impl MaplePairingAuthorityTransactionClass");
    for required_pattern in [
        "matches!(self, Self::ReadOnly | Self::ReplaySafeMutation)",
        "Self::ReadOnly => \"read_only\"",
        "Self::ReplaySafeMutation => \"replay_safe_mutation\"",
        "Self::NonReplayableMutation => \"non_replayable_mutation\"",
    ] {
        assert!(
            class_policy.contains(required_pattern),
            "authority transaction retry/telemetry policy must contain `{required_pattern}`"
        );
    }

    let serialization_classifier =
        extract_function_body(&db, "pub(crate) fn is_serialization_failure(");
    for required_pattern in [
        "while let Some(source) = current",
        ".downcast_ref::<diesel::result::Error>()",
        ".is_some_and(diesel_error_is_serialization_failure)",
        "current = source.source()",
    ] {
        assert!(
            serialization_classifier.contains(required_pattern),
            "SQLSTATE 40001 classifier must walk typed nested sources via `{required_pattern}`"
        );
    }
    assert!(
        !serialization_classifier.contains("message()")
            && !serialization_classifier.contains("to_string()"),
        "serialization-failure classification must not depend on database error text"
    );
    let diesel_classifier = extract_function_body(&db, "fn diesel_error_is_serialization_failure(");
    for required_pattern in [
        "DatabaseErrorKind::SerializationFailure",
        "diesel::result::Error::RollbackErrorOnCommit",
        "rollback_error",
        "commit_error",
        "diesel_error_is_serialization_failure(rollback_error)",
        "diesel_error_is_serialization_failure(commit_error)",
    ] {
        assert!(
            diesel_classifier.contains(required_pattern),
            "Diesel 40001 classifier must recursively inspect commit/rollback envelopes via `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        diesel_classifier,
        &[
            "DatabaseErrorKind::SerializationFailure",
            ") => true",
            "RollbackErrorOnCommit",
            "diesel_error_is_serialization_failure(rollback_error)",
            "diesel_error_is_serialization_failure(commit_error)",
            "_ => false",
        ],
    );
    let typed_source_regression = extract_function_body(
        &db,
        "fn serialization_failure_detection_walks_typed_model_error_sources(",
    );
    for required_wrapper in [
        "DBError::UserError(UserError::DatabaseError(serialization_failure()))",
        "DBError::OrgError(OrgError::DatabaseError(serialization_failure()))",
        "DBError::OrgProjectError(OrgProjectError::DatabaseError(serialization_failure()))",
        "DBError::UserSeedWrappingError(UserSeedWrappingError::DatabaseError(",
        "DBError::AppDataMigrationError(AppDataMigrationError::DatabaseError(",
        "assert!(!is_serialization_failure(&DBError::MaplePairingConflict))",
    ] {
        assert!(
            typed_source_regression.contains(required_wrapper),
            "typed SQLSTATE 40001 regression coverage must retain `{required_wrapper}`"
        );
    }
    let commit_rollback_regression = extract_function_body(
        &db,
        "fn serialization_failure_detection_inspects_commit_and_rollback_errors(",
    );
    assert_eq!(
        commit_rollback_regression
            .matches("diesel::result::Error::RollbackErrorOnCommit")
            .count(),
        2,
        "nested Diesel commit and rollback envelopes must both remain under runtime regression coverage"
    );
    for required_nested_failure in [
        "rollback_error: Box::new(diesel::result::Error::NotFound)",
        "commit_error: Box::new(serialization_failure())",
        "rollback_error: Box::new(serialization_failure())",
        "commit_error: Box::new(diesel::result::Error::NotFound)",
    ] {
        assert!(
            commit_rollback_regression.contains(required_nested_failure),
            "nested SQLSTATE 40001 regression coverage must retain `{required_nested_failure}`"
        );
    }
    let result_mapper =
        extract_function_body(&db, "fn finish_maple_pairing_authority_transaction<T>(");
    assert_patterns_in_order(
        result_mapper,
        &[
            "is_serialization_failure(error)",
            "trace_maple_pairing_authority_serialization_failure(class)",
            "if class.is_retry_safe()",
            "DBError::MaplePairingAuthorityBusy",
            "result",
        ],
    );
    let telemetry = extract_function_body(
        &db,
        "fn trace_maple_pairing_authority_serialization_failure(",
    );
    for required_pattern in [
        "event = \"maple_pairing_authority_serialization_failure\"",
        "transaction_class = class.telemetry_label()",
        "retry_safe = class.is_retry_safe()",
    ] {
        assert!(
            telemetry.contains(required_pattern),
            "serialization-failure telemetry must contain `{required_pattern}`"
        );
    }

    let nonreplayable_custom = extract_function_body(
        &db,
        "pub(crate) fn finish_nonreplayable_maple_pairing_authority_transaction<T, E>(",
    );
    for required_pattern in [
        "is_serialization_failure(error)",
        "trace_maple_pairing_authority_serialization_failure(",
        "MaplePairingAuthorityTransactionClass::NonReplayableMutation",
        "result",
    ] {
        assert!(
            nonreplayable_custom.contains(required_pattern),
            "custom signup transaction mapper must conservatively retain and trace via `{required_pattern}`"
        );
    }
    assert!(
        !nonreplayable_custom.contains("DBError::MaplePairingAuthorityBusy"),
        "nonreplayable custom transactions must never expose a retry-safe Busy result"
    );
    let signup_error_start = main
        .find("enum CreateUserSeedWrapTransactionError")
        .expect("custom signup transaction error should exist");
    let signup_error_derive = main[..signup_error_start]
        .rfind("#[derive(")
        .expect("custom signup transaction error should derive thiserror::Error");
    let signup_error_end = main[signup_error_start..]
        .find("impl From<CreateUserSeedWrapTransactionError> for Error")
        .map(|offset| signup_error_start + offset)
        .expect("custom signup transaction error conversion should exist");
    let signup_error = &main[signup_error_derive..signup_error_end];
    for required_source in [
        "#[derive(Debug, thiserror::Error)]",
        "Diesel(#[from] diesel::result::Error)",
        "Database(#[from] DBError)",
        "User(#[from] UserError)",
        "UserSeedWrapping(#[from] UserSeedWrappingError)",
    ] {
        assert!(
            signup_error.contains(required_source),
            "custom signup commit/callback errors must remain in the typed source chain via `{required_source}`"
        );
    }
    assert_eq!(
        main.matches("conn.transaction::<_, CreateUserSeedWrapTransactionError, _>(")
            .count(),
        2,
        "password and OAuth signup must retain their custom atomic transaction error type"
    );
    assert_eq!(
        main.matches("finish_nonreplayable_maple_pairing_authority_transaction(")
            .count(),
        2,
        "both custom signup transactions must observe commit-time serialization aborts conservatively"
    );
    for signup_helper in [
        "fn create_user_with_password_seed_wrap(",
        "pub fn create_user_with_oauth_seed_wrap(",
    ] {
        let signup = extract_function_body(&main, signup_helper);
        assert_patterns_in_order(
            signup,
            &[
                "finish_nonreplayable_maple_pairing_authority_transaction(",
                "conn.transaction::<_, CreateUserSeedWrapTransactionError, _>(|conn|",
                "create_user_with_maple_authority_in_tx(",
                "),",
                ".map_err(Error::from)",
            ],
        );
        assert!(
            !signup.contains("MaplePairingAuthorityBusy"),
            "signup helper `{signup_helper}` must retain its original nonreplayable error instead of inviting retry"
        );
    }

    let runner = extract_function_body(&db, "fn run_maple_pairing_authority_transaction<T, F>(");
    for required_pattern in [
        "finish_maple_pairing_authority_transaction(",
        "conn.transaction::<T, DBError, _>(callback)",
    ] {
        assert!(
            runner.contains(required_pattern),
            "outer authority transaction runner must contain `{required_pattern}`"
        );
    }
    assert_eq!(
        runner.matches("callback").count(),
        1,
        "the outer authority runner must invoke its FnOnce callback exactly once"
    );
    for forbidden_retry_pattern in ["loop {", "while ", "thread::sleep", "tokio::time::sleep"] {
        assert!(
            !runner.contains(forbidden_retry_pattern),
            "authority transaction runner must not retry internally via `{forbidden_retry_pattern}`"
        );
    }

    let expected_classes = [
        (
            "MaplePairingAuthorityTransactionClass::ReadOnly",
            &[
                "list_maple_pairing_revocations",
                "audit_maple_pairing_issuer_key_references",
                "replay_maple_reset_clear_ack",
                "replay_maple_pairing_operation",
                "list_maple_devices",
            ][..],
        ),
        (
            "MaplePairingAuthorityTransactionClass::ReplaySafeMutation",
            &[
                "ack_maple_pairing_revocation",
                "revoke_maple_pairing",
                "approve_maple_pairing",
                "confirm_maple_pairing",
                "list_maple_pairings",
                "get_maple_pairing",
                "create_maple_pairing",
                "complete_destructive_password_reset",
                "register_maple_device",
                "delete_org",
                "delete_org_project",
                "delete_user",
                "mark_and_delete_user",
            ][..],
        ),
        (
            "MaplePairingAuthorityTransactionClass::NonReplayableMutation",
            &[
                "bootstrap_or_audit_maple_pairing_authority",
                "create_user",
                "create_org",
                "create_org_project",
                "create_org_with_owner",
            ][..],
        ),
    ];
    let mut classified_callers = 0usize;
    for (class_name, methods) in expected_classes {
        for method_name in methods {
            let implementation = extract_function_body(db_impl, &format!("fn {method_name}("));
            assert_patterns_in_order(
                implementation,
                &["run_maple_pairing_authority_transaction(", class_name, "|"],
            );
            classified_callers += 1;
        }
    }
    for (function_signature, class_name) in [
        (
            "pub(crate) fn make_maple_pairing_pending_due_for_test(",
            "MaplePairingAuthorityTransactionClass::ReplaySafeMutation",
        ),
        (
            "pub(crate) fn run_maple_pairing_authority_ssi_race_for_test<F>(",
            "MaplePairingAuthorityTransactionClass::ReplaySafeMutation",
        ),
        (
            "pub(crate) fn seed_maple_pairing_highwater_group_capacity_for_test(",
            "MaplePairingAuthorityTransactionClass::NonReplayableMutation",
        ),
    ] {
        let test_transaction = extract_function_body(&db, function_signature);
        assert_patterns_in_order(
            test_transaction,
            &["run_maple_pairing_authority_transaction(", class_name, "|"],
        );
        classified_callers += 1;
    }
    assert_eq!(
        db.matches("run_maple_pairing_authority_transaction(")
            .count(),
        classified_callers,
        "every outer authority transaction must have an explicit, reviewed retry-safety class"
    );
}

#[test]
fn maple_pairing_parent_deletion_requires_one_authenticated_clean_deletion_path() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("DB source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");

    let deletion_predicate =
        extract_function_body(&db, "fn verify_maple_pairing_authority_deletion_safe(");
    for required_pattern in [
        "verify_maple_pairing_authority_account(conn, enclave_key, head)?",
        "expire_pending_pairings(conn, &authorization)?",
        "commit_maple_pairing_authority_account_mutation(",
        "verify_maple_pairing_authority_account(conn, enclave_key, &current)?",
        "MaplePairingState::Pending.as_db()",
        "MaplePairingState::AwaitingHostCommit.as_db()",
        "MaplePairingState::Active.as_db()",
        "last_acked_revocation_sequence",
        "last_issued_revocation_sequence",
        "let pending_reset = maple_pairing_reset_clear_obligations::table",
        "maple_pairing_reset_clear_obligations::state.eq(1_i16)",
        "pending_reset.is_some()",
        "DBError::MaplePairingAuthorityDeletionBlocked",
    ] {
        assert!(
            deletion_predicate.contains(required_pattern),
            "authenticated terminal-deletion predicate must contain `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        deletion_predicate,
        &[
            "verify_maple_pairing_authority_account(conn, enclave_key, head)?",
            "expire_pending_pairings(conn, &authorization)?",
            "if expired_any",
            "commit_maple_pairing_authority_account_mutation(",
            ".for_update()",
            ".first::<MaplePairingAuthorityAccountHead>(conn)?",
            "verify_maple_pairing_authority_account(conn, enclave_key, &current)?",
            "let blocking_pair",
            "let unacked_host",
            "let pending_reset",
            "DBError::MaplePairingAuthorityDeletionBlocked",
        ],
    );

    let account_deletion = extract_function_body(
        &db,
        "fn delete_maple_pairing_authority_account_for_final_parent_deletion(",
    );
    assert_patterns_in_order(
        account_deletion,
        &[
            "prove_maple_pairing_authority_account_deletion_safe(",
            "consume_maple_pairing_authority_account_after_clean_proof(",
        ],
    );
    for forbidden_raw_cleanup in [
        "delete_maple_pairing_state_for_user(",
        "maple_pairing_revocation_highwaters::table",
        "maple_pairing_authority_account_heads::table",
    ] {
        assert!(
            !account_deletion.contains(forbidden_raw_cleanup),
            "the single-account coordinator must delegate, not perform raw cleanup via `{forbidden_raw_cleanup}`"
        );
    }

    let account_proof = extract_function_body(
        &db,
        "fn prove_maple_pairing_authority_account_deletion_safe(",
    );
    assert_patterns_in_order(
        account_proof,
        &[
            ".for_update()",
            ".first::<MaplePairingAuthorityAccountHead>(conn)",
            "verify_maple_pairing_authority_deletion_safe(conn, enclave_key, &head)",
        ],
    );
    for forbidden_destructive_pattern in [
        "delete_maple_pairing_state_for_user(",
        "diesel::delete(",
        "consume_maple_pairing_authority_account_after_clean_proof(",
    ] {
        assert!(
            !account_proof.contains(forbidden_destructive_pattern),
            "clean-deletion proof must be nondestructive via `{forbidden_destructive_pattern}`"
        );
    }

    let account_consumption = extract_function_body(
        &db,
        "fn consume_maple_pairing_authority_account_after_clean_proof(",
    );
    assert_patterns_in_order(
        account_consumption,
        &[
            ".for_update()",
            ".first::<MaplePairingAuthorityAccountHead>(conn)",
            "validate_maple_pairing_authority_account_head(enclave_key, &head)?",
            "delete_maple_pairing_state_for_user(conn, user_id)?",
            "maple_pairing_reset_clear_admissions::table",
            "maple_pairing_reset_clear_obligations::table",
            "maple_pairing_registration_operation_tombstones::table",
            "maple_pairing_installation_retirements::table",
            "maple_pairing_revocation_highwaters::table",
            "maple_pairing_authority_account_heads::table",
            "if removed != 1",
        ],
    );
    for forbidden_reproof_pattern in [
        "verify_maple_pairing_authority_deletion_safe(",
        "expire_pending_pairings(",
        "commit_maple_pairing_authority_account_mutation(",
        "cascade_maple_pairing_authority_heads(",
    ] {
        assert!(
            !account_consumption.contains(forbidden_reproof_pattern),
            "destructive phase must not mutate/re-run the terminal proof via `{forbidden_reproof_pattern}`"
        );
    }
    assert_eq!(
        db.matches("verify_maple_pairing_authority_deletion_safe(")
            .count(),
        2,
        "only the nondestructive account-proof helper may evaluate the clean-deletion predicate"
    );

    let project_proof = extract_function_body(
        &db,
        "fn prove_maple_pairing_authority_accounts_for_project(",
    );
    for required_pattern in [
        ".filter(users::project_id.eq(project_id))",
        ".filter(users::uuid.gt(cursor))",
        ".order(users::uuid.asc())",
        ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
        ".for_update()",
        "prove_maple_pairing_authority_account_deletion_safe(",
        "cursor = user_id",
    ] {
        assert!(
            project_proof.contains(required_pattern),
            "project/org proof pass must page authenticated accounts via `{required_pattern}`"
        );
    }
    assert!(
        !project_proof.contains("consume_maple_pairing_authority_account_after_clean_proof("),
        "project proof pass must not consume an account"
    );

    let project_consumption = extract_function_body(
        &db,
        "fn consume_maple_pairing_authority_accounts_for_project_after_clean_proof(",
    );
    for required_pattern in [
        ".filter(users::project_id.eq(project_id))",
        ".filter(users::uuid.gt(cursor))",
        ".order(users::uuid.asc())",
        ".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)",
        ".for_update()",
        "consume_maple_pairing_authority_account_after_clean_proof(",
        "cursor = user_id",
    ] {
        assert!(
            project_consumption.contains(required_pattern),
            "project destructive pass must page already-proven accounts via `{required_pattern}`"
        );
    }
    for forbidden_proof_pattern in [
        "prove_maple_pairing_authority_account_deletion_safe(",
        "verify_maple_pairing_authority_deletion_safe(",
    ] {
        assert!(
            !project_consumption.contains(forbidden_proof_pattern),
            "project destructive pass must not interleave proof via `{forbidden_proof_pattern}`"
        );
    }

    let project_deletion = extract_function_body(
        &db,
        "fn consume_maple_pairing_authority_accounts_for_project(",
    );
    assert_patterns_in_order(
        project_deletion,
        &[
            "prove_maple_pairing_authority_accounts_for_project(",
            "consume_maple_pairing_authority_accounts_for_project_after_clean_proof(",
        ],
    );

    for method_name in ["delete_user", "mark_and_delete_user"] {
        let implementation = extract_function_body(db_impl, &format!("fn {method_name}("));
        assert!(
            implementation
                .contains("delete_maple_pairing_authority_account_for_final_parent_deletion("),
            "parent deletion method `{method_name}` must consume each account through the authenticated clean-deletion helper"
        );
        for forbidden_raw_cleanup in [
            "delete_maple_pairing_state_for_user(",
            "delete_maple_pairing_state_for_project(",
        ] {
            assert!(
                !implementation.contains(forbidden_raw_cleanup),
                "parent deletion method `{method_name}` must not bypass the authenticated clean-deletion predicate via `{forbidden_raw_cleanup}`"
            );
        }

        let parent_delete = match method_name {
            "delete_user" | "mark_and_delete_user" => "locked_user.delete(tx)",
            "delete_org_project" => "locked_project.delete(tx)",
            "delete_org" => "locked_org.delete(tx)",
            _ => unreachable!("parent deletion method list is exhaustive"),
        };
        assert_patterns_in_order(
            implementation,
            &[
                "acquire_maple_pairing_authority_snapshot_fence(",
                "expected_issuer_key_inventory_digest",
                "verify_maple_pairing_authority_scoped_chain(",
                "delete_maple_pairing_authority_account_for_final_parent_deletion(",
                parent_delete,
                "verify_maple_pairing_authority_project_chain(",
            ],
        );
    }

    let delete_project = extract_function_body(db_impl, "fn delete_org_project(");
    assert_patterns_in_order(
        delete_project,
        &[
            "acquire_maple_pairing_authority_snapshot_fence(",
            "expected_issuer_key_inventory_digest",
            "verify_maple_pairing_authority_project_chain(",
            "consume_maple_pairing_authority_accounts_for_project(",
            "maple_pairing_authority_project_heads::table",
            "locked_project.delete(tx)",
            "verify_maple_pairing_authority_org_chain(",
        ],
    );

    let delete_org = extract_function_body(db_impl, "fn delete_org(");
    assert_eq!(
        delete_org
            .matches(".filter(org_projects::id.gt(project_cursor))")
            .count(),
        2,
        "org deletion must independently page the complete proof and destructive passes"
    );
    assert_eq!(
        delete_org
            .matches(".limit(MAPLE_PAIRING_AUTHORITY_PAGE_SIZE)")
            .count(),
        2,
        "both org-subtree passes must remain bounded"
    );
    assert_patterns_in_order(
        delete_org,
        &[
            "acquire_maple_pairing_authority_snapshot_fence(",
            "expected_issuer_key_inventory_digest",
            "verify_maple_pairing_authority_org_chain(",
            "prove_maple_pairing_authority_accounts_for_project(",
            "project_cursor = 0",
            "consume_maple_pairing_authority_accounts_for_project_after_clean_proof(",
            "maple_pairing_authority_project_heads::table",
            "maple_pairing_authority_org_heads::table",
            "locked_org.delete(tx)",
            "refresh_maple_pairing_authority_global_head(",
            "verify_maple_pairing_authority_global_shallow(",
        ],
    );
    let destructive_org_phase = delete_org
        .find("consume_maple_pairing_authority_accounts_for_project_after_clean_proof(")
        .expect("org deletion must have a second, destructive pass");
    assert!(
        !delete_org[..destructive_org_phase].contains("diesel::delete(")
            && !delete_org[..destructive_org_phase].contains(".delete(tx)"),
        "org deletion must prove the entire subtree before deleting any account or parent"
    );
}

#[test]
fn maple_pairing_device_and_pairing_control_plane_use_authority_lock_wrapper() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("db source should be readable");
    let authority_wrapper =
        extract_function_body(&db, "fn enter_maple_pairing_authority_account_transaction(");
    for required_pattern in [
        "acquire_maple_pairing_authority_snapshot_fence(",
        "expected_issuer_key_inventory_digest",
        "maple_pairing_authority_account_heads::table",
        "verify_maple_pairing_authority_scoped_chain(",
    ] {
        assert!(
            authority_wrapper.contains(required_pattern),
            "Maple control-plane authority wrapper must authenticate its scoped chain via `{required_pattern}`"
        );
    }
    assert!(
        !authority_wrapper.contains("verify_maple_pairing_authority_tree("),
        "account-scoped control-plane requests must not rescan the global authority tree"
    );

    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    for method_name in [
        "register_maple_device",
        "list_maple_devices",
        "replay_maple_reset_clear_ack",
        "replay_maple_pairing_operation",
        "create_maple_pairing",
        "list_maple_pairings",
        "get_maple_pairing",
        "approve_maple_pairing",
        "confirm_maple_pairing",
        "revoke_maple_pairing",
        "list_maple_pairing_revocations",
        "ack_maple_pairing_revocation",
    ] {
        let body = extract_function_body(db_impl, &format!("fn {method_name}("));
        assert!(
            body.contains("enter_maple_pairing_authority_account_transaction(")
                && body.contains("expected_issuer_key_inventory_digest"),
            "Maple control-plane method `{method_name}` must enter the globally locked, issuer-digest-fenced scoped-chain wrapper"
        );
    }

    let list_revocations = extract_function_body(db_impl, "fn list_maple_pairing_revocations(");
    assert_patterns_in_order(
        list_revocations,
        &[
            "pending_reset_clear_occupies_sequence_one,",
            "load_latest_pending_maple_reset_clear_obligation(",
            "pending.revision != 2",
            "state.last_issued_revocation_sequence != 1",
            "state.last_acked_revocation_sequence != 0",
            "state.revision != 2",
            "decrypt_maple_reset_clear_payload(",
            "Some(lifecycle_floor)",
            "true,",
            "} else {",
            "(None, None, false)",
            "if pending_reset_clear_occupies_sequence_one",
            "if !events.is_empty()",
            "return Err(DBError::MaplePairingCorrupt)",
            "else if events.is_empty() && after < last_issued_i64",
        ],
    );
    assert_eq!(
        list_revocations
            .matches("pending_reset_clear_occupies_sequence_one")
            .count(),
        2,
        "LIST must define the authenticated pending-reset sequence-one flag once and consume it once in the event-gap proof"
    );

    let issuer_audit =
        extract_function_body(db_impl, "fn audit_maple_pairing_issuer_key_references(");
    for required_pattern in [
        "acquire_maple_pairing_authority_snapshot_fence(",
        "expected_issuer_key_inventory_digest",
        "LOCK TABLE maple_pairings, maple_pairing_revocation_events",
        "maple_pairing_operations, maple_device_registration_operations",
        "maple_pairing_registration_operation_tombstones",
        "maple_pairing_reset_clear_obligations",
        "maple_pairing_installation_retirements",
        "maple_pairing_issuer_keys IN SHARE MODE",
        "verify_maple_pairing_authority_tree(",
        "let issuer_rows = maple_pairing_issuer_keys::table",
        "validate_maple_pairing_issuer_key(enclave_key, row)?",
        "registered_key_ids.insert(row.key_id.clone())",
        "let retain_reference =",
        "if !registered_key_ids.contains(key_id)",
        "&pairing.ticket_issuer_key_id",
        "pairing.authorization_issuer_key_id.as_ref()",
        "pairing.revocation_issuer_key_id.as_ref()",
        "&row.sync_issuer_key_id",
        "maple_pairing_issuer_key_ids_are_canonical(",
        "for key_id in &row.referenced_issuer_key_ids",
        "row.signed_instruction_issuer_key_id.as_ref()",
        "row.sync_issuer_key_id.as_ref()",
        "row.ack_receipt_issuer_key_id.as_ref()",
        "&row.ack_receipt_issuer_key_id",
        "operation.receipt_issuer_key_id.as_ref()",
        "&event.issuer_key_id",
        "referenced_key_ids.is_subset(&registered_key_ids)",
    ] {
        assert!(
            issuer_audit.contains(required_pattern),
            "global issuer-reference audit must contain `{required_pattern}`"
        );
    }
    assert_patterns_in_order(
        issuer_audit,
        &[
            "acquire_maple_pairing_authority_snapshot_fence(",
            "expected_issuer_key_inventory_digest",
            "LOCK TABLE maple_pairings, maple_pairing_revocation_events",
            "maple_pairing_installation_retirements",
            "maple_pairing_issuer_keys IN SHARE MODE",
            "verify_maple_pairing_authority_tree(tx, enclave_key)",
        ],
    );
}

#[test]
fn maple_device_exact_registration_replay_is_read_only() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("DB source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let register = extract_function_body(db_impl, "fn register_maple_device(");
    let tombstone_replay = extract_function_body(register, "if let Some(tombstone)");
    let live_replay = extract_function_body(register, "if let Some(prior_operation)");

    for (replay_kind, exact_replay) in [
        ("retained tombstone", tombstone_replay),
        ("live operation", live_replay),
    ] {
        for forbidden_pattern in [
            "restore_maple_pairing_host_state_from_highwater(",
            "seed_or_validate_maple_pairing_host_state(",
            "insert_initial_maple_pairing_revocation_highwater(",
            "commit_maple_pairing_authority_account_mutation(",
            "diesel::insert_into(",
            "diesel::update(",
            "diesel::delete(",
            ".execute(",
        ] {
            assert!(
                !exact_replay.contains(forbidden_pattern),
                "exact {replay_kind} registration replay must not mutate authority state via `{forbidden_pattern}`"
            );
        }
    }
    assert!(
        tombstone_replay.contains("return replay_maple_device_registration_tombstone("),
        "exact retired registration replay must return the authenticated tombstone receipt"
    );
    assert!(
        live_replay.contains("return replay_live_maple_device_registration_operation("),
        "exact live registration replay must return the authenticated prior receipt"
    );

    assert_patterns_in_order(
        register,
        &[
            "let tombstone = maple_pairing_registration_operation_tombstones::table",
            "if let Some(tombstone) = tombstone",
            "return replay_maple_device_registration_tombstone(",
            "let prior_operation = maple_device_registration_operations::table",
            "if let Some(prior_operation) = prior_operation",
            "return replay_live_maple_device_registration_operation(",
            "if registration.operation_id.is_nil()",
            "let host_registration_lookup_digest =",
            "maple_reset_clear_ack_host_registration_lookup_digest(",
            "registration.registration_id",
            "let retirement = maple_pairing_installation_retirements::table",
            "maple_pairing_installation_retirements::lookup_digest",
            ".or(maple_pairing_installation_retirements::host_identity_mac",
            ".or(maple_pairing_installation_retirements::ack_host_registration_lookup_digest",
            ".eq(&host_registration_lookup_digest)",
            "return Err(DBError::MapleInstallationRetired)",
            "if registration.known_security_epoch != head.security_epoch",
            "prepare_maple_device_registration_sync(",
        ],
    );

    let tombstone_helper =
        extract_function_body(&db, "fn replay_maple_device_registration_tombstone(");
    for required_gate in [
        "row.request_mac.as_slice().ct_eq(request_mac)",
        "return Err(DBError::MapleDeviceRegistrationConflict)",
        "row.lookup_digest.as_slice().ct_eq(lookup_digest)",
        "decrypt_maple_device_registration_tombstone_receipt(enclave_key, row)",
        "sync.verify_against_registration(",
    ] {
        assert!(
            tombstone_helper.contains(required_gate),
            "retained registration replay must authenticate exact history via `{required_gate}`"
        );
    }
    for pending_only_gate in [
        "load_latest_pending_maple_reset_clear_obligation(",
        ".ok_or(DBError::MaplePairingAuthorityCorrupt)?",
        ".host_identity_mac",
        ".ct_eq(registration.identity_mac.as_slice())",
        "let retained_identity = maple_pairing_reset_clear_obligations::table",
        "if retained_identity.is_some()",
    ] {
        assert!(
            register.contains(pending_only_gate),
            "registration recovery must be Pending-only and identity-bound via `{pending_only_gate}`"
        );
    }

    let web_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/maple_devices.rs");
    let web = fs::read_to_string(web_source).expect("Maple device web source should be readable");
    let register_route = extract_function_body(&web, "async fn register_device(");
    assert_patterns_in_order(
        register_route,
        &[
            "let issuer_keyset = require_pairing_keyset(&state)?",
            ".register_maple_device(",
            "let (issuer, materializer_keyset) =",
            "require_pairing_crypto(&state)",
            "materialize_maple_device_registration_sync(",
        ],
    );
    assert_eq!(
        register_route
            .matches("require_pairing_crypto(&state)")
            .count(),
        1,
        "registration must demand the active signer only inside fresh materialization, after exact replay and retirement gates"
    );
}

#[test]
fn maple_reset_clear_ack_replay_precedes_live_identity_and_reuses_exact_receipt() {
    let db_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/db.rs");
    let db = fs::read_to_string(db_source).expect("DB source should be readable");
    let db_impl = extract_function_body(&db, "impl DBConnection for PostgresConnection");
    let replay_wrapper = extract_function_body(db_impl, "fn replay_maple_reset_clear_ack(");
    for required_wrapper_gate in [
        "MaplePairingAuthorityTransactionClass::ReadOnly",
        "enter_maple_pairing_authority_account_transaction(",
        "replay_maple_reset_clear_ack_in_transaction(",
    ] {
        assert!(
            replay_wrapper.contains(required_wrapper_gate),
            "reset-clear ACK replay wrapper must retain `{required_wrapper_gate}`"
        );
    }
    let replay = extract_function_body(&db, "fn replay_maple_reset_clear_ack_in_transaction(");
    for required_gate in [
        "maple_reset_clear_ack_host_registration_lookup_digest(",
        "host_registration_id",
        "maple_reset_clear_ack_operation_lookup_digest(",
        "&host_registration_lookup_digest",
        "maple_pairing_installation_retirements::ack_host_registration_lookup_digest",
        "maple_pairing_installation_retirements::ack_operation_lookup_digest",
        "validate_maple_installation_retirement(",
        ".ack_host_registration_lookup_digest",
        ".ct_eq(host_registration_lookup_digest.as_slice())",
        "retirement.ack_request_mac.as_slice().ct_eq(request_mac)",
        "return Err(DBError::MaplePairingConflict)",
        "validate_maple_pairing_reset_clear_obligation(",
        "obligation.state != 2",
        "obligation.revision != 3",
        "obligation.acked_by_head_event_id != Some(obligation.uuid)",
        "obligation.ack_operation_id != Some(operation_id)",
        "obligation.ack_host_registration_lookup_digest.as_deref()",
        "retirement.ack_host_registration_lookup_digest.as_slice()",
        "obligation.ack_request_mac.as_deref() != Some(request_mac)",
        "sha256_digest(&receipt_enc)",
        ".ct_eq(retirement.ack_receipt_digest.as_slice())",
        "receipt_enc,",
    ] {
        assert!(
            replay.contains(required_gate),
            "reset-clear ACK replay must authenticate exact retained history via `{required_gate}`"
        );
    }
    let host_lookup = extract_function_body(
        &db,
        "fn maple_reset_clear_ack_host_registration_lookup_digest(",
    );
    assert_patterns_in_order(
        host_lookup,
        &[
            "MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_DOMAIN",
            ".append_bytes(authority_scope_digest)",
            ".append_uuid(host_registration_id)",
            "MAPLE_RESET_CLEAR_ACK_HOST_LOOKUP_KEY_INFO",
        ],
    );
    let operation_lookup =
        extract_function_body(&db, "fn maple_reset_clear_ack_operation_lookup_digest(");
    assert_patterns_in_order(
        operation_lookup,
        &[
            "MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_DOMAIN",
            ".append_bytes(authority_scope_digest)",
            ".append_bytes(host_registration_lookup_digest)",
            ".append_uuid(operation_id)",
            "MAPLE_RESET_CLEAR_ACK_OPERATION_LOOKUP_KEY_INFO",
        ],
    );
    let ack_mutation_helper =
        extract_function_body(&db, "fn acknowledge_pending_maple_reset_clear(");
    assert_patterns_in_order(
        ack_mutation_helper,
        &[
            "maple_reset_clear_ack_host_registration_lookup_digest(",
            "host.uuid",
            "obligation.ack_operation_id = Some(ack.operation_id)",
            "obligation.ack_host_registration_lookup_digest =",
            "Some(host_registration_lookup_digest.clone())",
            "maple_reset_clear_ack_operation_lookup_digest(",
            "&host_registration_lookup_digest",
            "NewMaplePairingInstallationRetirement",
            "ack_host_registration_lookup_digest: host_registration_lookup_digest",
            "ack_operation_lookup_digest: operation_lookup_digest",
        ],
    );
    let ack_mutation = extract_function_body(db_impl, "fn ack_maple_pairing_revocation(");
    assert_patterns_in_order(
        ack_mutation,
        &[
            "let head = maple_pairing_authority_account_heads::table",
            "validate_maple_pairing_authority_account_head(",
            "replay_maple_reset_clear_ack_in_transaction(",
            "return Ok(receipt)",
            "find_scoped_maple_device(",
        ],
    );
    assert_eq!(
        db.matches("replay_maple_reset_clear_ack_in_transaction(")
            .count(),
        3,
        "the exact ACK helper must have one definition plus read-only and mutation-transaction callers"
    );

    let web_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/maple_pairings.rs");
    let web = fs::read_to_string(web_source).expect("Maple pairing web source should be readable");
    let web_replay = extract_function_body(&web, "fn replay_reset_clear_ack_if_present(");
    assert_patterns_in_order(
        web_replay,
        &[
            ".replay_maple_reset_clear_ack(",
            "decrypt_receipt(",
            "response.verify_against_request(request, keyset)",
            "Ok(Some(response))",
        ],
    );
    let ack = extract_function_body(&web, "async fn ack_revocation(");
    assert_patterns_in_order(
        ack,
        &[
            "request_operation_mac(",
            "replay_reset_clear_ack_if_present(",
            "return encrypt_response(&state, &session_id, &response).await",
            "load_devices(&state, &user, &auth_context)",
            "Err(ApiError::NotFound)",
            "replay_reset_clear_ack_if_present(",
            "return encrypt_response(&state, &session_id, &response).await",
            "verify_device_signature(host, &transcript, &request.signature)",
            "require_pairing_crypto(&state)",
            "let page_result = state",
            ".list_maple_pairing_revocations(",
            "Err(DBError::MaplePairingNotFound)",
            "replay_reset_clear_ack_if_present(",
            "return encrypt_response(&state, &session_id, &response).await",
            "let event_digest = canonical_b64_32(",
        ],
    );
    assert_eq!(
        ack.matches("replay_reset_clear_ack_if_present(").count(),
        3,
        "ACK web preflight, missing-device, and missing-list fallbacks must all resolve exact retained replay"
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
    classified_tables.extend(
        DESTRUCTIVE_RESET_MAPLE_HELPER_ENCRYPTED_TABLES
            .iter()
            .copied(),
    );
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
fn seed_wrap_translation_code_and_build_targets_are_removed_after_rollout() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let migrations_source = manifest_dir.join("src/migrations.rs");
    let migrations_contents =
        fs::read_to_string(&migrations_source).expect("migrations source should be readable");
    let cargo_contents =
        fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("Cargo.toml should be readable");

    let mut checked_files = vec![
        ("src/migrations.rs", migrations_contents),
        ("Cargo.toml", cargo_contents),
    ];

    for repo_root_file in ["flake.nix", "justfile"] {
        if let Ok(contents) = fs::read_to_string(manifest_dir.join(repo_root_file)) {
            checked_files.push((repo_root_file, contents));
        }
    }

    for forbidden_pattern in [
        "seed-wrap-translation",
        "migrate_aead_seed_wrappings_v1",
        "AEAD_SEED_WRAPPINGS_MIGRATION",
        "SeedWrapTranslationError",
        "opensecret-aead-translation",
    ] {
        for (file_name, contents) in &checked_files {
            assert!(
                !contents.contains(forbidden_pattern),
                "post-rollout serving source/build file `{file_name}` must not contain `{forbidden_pattern}`"
            );
        }
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
        "legacy_seed_enc",
        "encrypt_with_key(&secret_key, user_seed_words.as_bytes())",
    ] {
        assert!(
            !create_body.contains(forbidden_pattern) && !confirm_body.contains(forbidden_pattern),
            "user password reset must not store portable encrypted reset/seed material via `{forbidden_pattern}`"
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
        "NewUser::new(creds.email, Some(encrypted_pw), project.id)",
        "create_user_with_password_seed_wrap(",
    ] {
        assert!(
            register_user_body.contains(required_pattern),
            "registration helper must contain `{required_pattern}`"
        );
    }
    assert!(
        !register_user_body.contains("encrypt_with_key(&secret_key, user_seed_words.as_bytes())"),
        "registration must not write legacy users.seed_enc ciphertext"
    );

    let create_user_wrap_body =
        extract_function_body(&main_contents, "fn create_user_with_password_seed_wrap");
    for required_pattern in [
        "conn.transaction::<_, CreateUserSeedWrapTransactionError, _>",
        "create_user_with_maple_authority_in_tx(",
        "&self.enclave_key",
        "new_password_seed_wrapping_for_user(",
        "new_wrapping.upsert_by_credential(conn)",
    ] {
        assert!(
            create_user_wrap_body.contains(required_pattern),
            "password registration must atomically create user and seed wrap with `{required_pattern}`"
        );
    }
    assert!(
        !create_user_wrap_body.contains("new_user.insert(conn)"),
        "password registration must not bypass authenticated Maple authority-head creation"
    );

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
        "create_user_with_maple_authority_in_tx(",
        "&self.enclave_key",
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
    assert!(
        !create_oauth_user_wrap_body.contains("new_user.insert(conn)"),
        "OAuth registration must not bypass authenticated Maple authority-head creation"
    );

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

fn collect_sensitive_log_findings(source_path: &str, source: &str) -> Vec<String> {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("{source_path} should parse as Rust source: {error}"));
    let mut visitor = SensitiveLogVisitor {
        source_path,
        findings: Vec::new(),
    };
    visitor.visit_file(&syntax);
    visitor.findings
}

fn collect_sensitive_log_findings_in_path(
    path: &Path,
    manifest_dir: &Path,
    findings: &mut Vec<String>,
) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("source directory should be readable") {
            let entry = entry.expect("source directory entry should be readable");
            collect_sensitive_log_findings_in_path(&entry.path(), manifest_dir, findings);
        }
        return;
    }

    if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
        return;
    }

    let contents = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("{} should be readable", path.display()));
    let relative_path = path.strip_prefix(manifest_dir).unwrap_or(path);
    findings.extend(collect_sensitive_log_findings(
        &relative_path.display().to_string(),
        &contents,
    ));
}

struct SensitiveLogVisitor<'a> {
    source_path: &'a str,
    findings: Vec<String>,
}

impl<'ast> Visit<'ast> for SensitiveLogVisitor<'_> {
    fn visit_macro(&mut self, log_macro: &'ast syn::Macro) {
        let Some(macro_name) = log_macro
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
        else {
            return;
        };

        if LOG_MACROS.contains(&macro_name.as_str()) {
            let body = log_macro.tokens.to_string();
            let mut sensitive_references = SENSITIVE_LOG_IDENTIFIERS
                .iter()
                .filter(|identifier| contains_identifier(&body, identifier))
                .map(|identifier| format!("identifier `{identifier}`"))
                .collect::<Vec<_>>();
            sensitive_references.extend(
                SENSITIVE_LOG_MESSAGES
                    .iter()
                    .filter(|message| body.contains(*message))
                    .map(|message| format!("message `{message}`")),
            );

            if !sensitive_references.is_empty() {
                self.findings.push(format!(
                    "{}: `{}!` references {}",
                    self.source_path,
                    macro_name,
                    sensitive_references.join(", ")
                ));
            }
        }

        visit::visit_macro(self, log_macro);
    }
}

fn contains_identifier(source: &str, identifier: &str) -> bool {
    source.match_indices(identifier).any(|(start, _)| {
        let before = source[..start].chars().next_back();
        let after = source[start + identifier.len()..].chars().next();
        !before.is_some_and(is_identifier_character) && !after.is_some_and(is_identifier_character)
    })
}

fn is_identifier_character(character: char) -> bool {
    character == '_' || character.is_alphanumeric()
}

fn assert_structs_do_not_derive_debug(source_path: &str, source: &str, struct_names: &[&str]) {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("{source_path} should parse as Rust source: {error}"));
    let structs = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.to_string(), item)),
            _ => None,
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    for struct_name in struct_names {
        let item = structs.get(*struct_name).unwrap_or_else(|| {
            panic!("authority-bearing struct `{struct_name}` should exist in {source_path}")
        });
        let mut derives_debug = false;
        for attribute in &item.attrs {
            if !attribute.path().is_ident("derive") {
                continue;
            }
            attribute
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("Debug") {
                        derives_debug = true;
                    }
                    Ok(())
                })
                .unwrap_or_else(|error| {
                    panic!(
                        "derive attributes for `{struct_name}` in {source_path} should parse: {error}"
                    )
                });
        }
        assert!(
            !derives_debug,
            "authority-bearing struct `{struct_name}` in {source_path} must not derive blanket Debug"
        );
    }
}

fn assert_struct_does_not_derive(
    source_path: &str,
    source: &str,
    struct_name: &str,
    forbidden_derives: &[&str],
) {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("{source_path} should parse as Rust source: {error}"));
    let item = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == struct_name => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!("authority-bearing struct `{struct_name}` should exist in {source_path}")
        });

    let mut found = BTreeSet::new();
    for attribute in &item.attrs {
        if !attribute.path().is_ident("derive") {
            continue;
        }
        attribute
            .parse_nested_meta(|meta| {
                if let Some(identifier) = meta.path.get_ident() {
                    found.insert(identifier.to_string());
                }
                Ok(())
            })
            .unwrap_or_else(|error| {
                panic!(
                    "derive attributes for `{struct_name}` in {source_path} should parse: {error}"
                )
            });
    }
    for forbidden in forbidden_derives {
        assert!(
            !found.contains(*forbidden),
            "authority-bearing struct `{struct_name}` in {source_path} must not derive `{forbidden}`"
        );
    }
}

fn assert_enums_do_not_derive_debug(source_path: &str, source: &str, enum_names: &[&str]) {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("{source_path} should parse as Rust source: {error}"));
    let enums = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Enum(item) => Some((item.ident.to_string(), item)),
            _ => None,
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    for enum_name in enum_names {
        let item = enums.get(*enum_name).unwrap_or_else(|| {
            panic!("authority-bearing enum `{enum_name}` should exist in {source_path}")
        });
        let mut derives_debug = false;
        for attribute in &item.attrs {
            if !attribute.path().is_ident("derive") {
                continue;
            }
            attribute
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("Debug") {
                        derives_debug = true;
                    }
                    Ok(())
                })
                .unwrap_or_else(|error| {
                    panic!(
                        "derive attributes for `{enum_name}` in {source_path} should parse: {error}"
                    )
                });
        }
        assert!(
            !derives_debug,
            "authority-bearing enum `{enum_name}` in {source_path} must not derive blanket Debug"
        );
    }
}

fn assert_debug_omits_authority_values(
    type_name: &str,
    debug_output: &str,
    forbidden_values: &[String],
) {
    assert!(
        debug_output.contains("[redacted]"),
        "{type_name} Debug output should explicitly mark authority material as redacted"
    );
    for forbidden in forbidden_values {
        assert!(
            !debug_output.contains(forbidden),
            "{type_name} Debug output leaked authority value `{forbidden}`"
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

fn extract_sql_create_table<'a>(source: &'a str, table_name: &str) -> &'a str {
    let declaration = format!("CREATE TABLE {table_name} (");
    let table_start = source
        .find(&declaration)
        .unwrap_or_else(|| panic!("SQL table declaration `{declaration}` should exist"));
    let relative_end = source[table_start..]
        .find("\n);")
        .unwrap_or_else(|| panic!("SQL table `{table_name}` should have a terminating `);`"));
    &source[table_start..table_start + relative_end + 3]
}

fn extract_sql_function<'a>(source: &'a str, function_name: &str) -> &'a str {
    let declaration = format!("CREATE OR REPLACE FUNCTION {function_name}(");
    let function_start = source
        .find(&declaration)
        .unwrap_or_else(|| panic!("SQL function declaration `{declaration}` should exist"));
    let relative_end = source[function_start..]
        .find("\n\nCREATE ")
        .unwrap_or_else(|| {
            panic!("SQL function `{function_name}` should precede another DDL item")
        });
    &source[function_start..function_start + relative_end]
}

fn normalize_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_method_signature<'a>(source: &'a str, method_name: &str, terminator: char) -> &'a str {
    let declaration = format!("fn {method_name}(");
    let method_start = source
        .find(&declaration)
        .unwrap_or_else(|| panic!("method declaration `{declaration}` should exist"));
    let relative_end = source[method_start..]
        .find(terminator)
        .unwrap_or_else(|| panic!("method `{method_name}` should have terminator `{terminator}`"));
    &source[method_start..method_start + relative_end]
}

fn rust_function_body_prefix_before(source: &str, position: usize) -> &str {
    let function_start = source[..position]
        .rfind("fn ")
        .unwrap_or_else(|| panic!("Rust call at byte {position} should be inside a function"));
    let body_start = source[function_start..position]
        .find('{')
        .map(|offset| function_start + offset + 1)
        .unwrap_or_else(|| panic!("Rust function before byte {position} should have a body"));
    &source[body_start..position]
}

fn extract_rust_parenthesized_call<'a>(
    source: &'a str,
    call_position: usize,
    call_name: &str,
) -> &'a str {
    let open_paren = call_position + call_name.len() - 1;
    assert_eq!(
        source.as_bytes().get(open_paren),
        Some(&b'('),
        "Rust call `{call_name}` should end at its opening parenthesis"
    );
    let mut depth = 0i32;
    for (relative_index, byte) in source[open_paren..].bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return &source[call_position..open_paren + relative_index + 1];
                }
            }
            _ => {}
        }
    }
    panic!("Rust call `{call_name}` should have balanced parentheses");
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
