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
