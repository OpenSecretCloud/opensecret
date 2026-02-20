use crate::nearai::error::NearAiError;
use crate::nearai::models::AttestationInfo;
use dcap_qvl::collateral::get_collateral;
use dcap_qvl::quote::{Report, TDReport10};
use dcap_qvl::verify::ring::verify as verify_tdx;
use secp256k1::PublicKey;
use sha2::{Digest, Sha256};
use sha3::Keccak256;
use std::time::{SystemTime, UNIX_EPOCH};

pub const INTEL_PCCS_URL: &str = "https://api.trustedservices.intel.com/tdx/certification/v4";

pub struct VerifiedTdxQuote {
    pub report_data: [u8; 64],
    pub mr_config_id: [u8; 48],
}

pub async fn verify_tdx_quote(intel_quote_hex: &str) -> Result<VerifiedTdxQuote, NearAiError> {
    let quote_bytes = hex::decode(intel_quote_hex.trim_start_matches("0x"))?;

    let collateral = get_collateral(INTEL_PCCS_URL, &quote_bytes)
        .await
        .map_err(|e| NearAiError::Tdx(e.to_string()))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| NearAiError::Tdx(e.to_string()))?
        .as_secs();

    let verified =
        verify_tdx(&quote_bytes, &collateral, now).map_err(|e| NearAiError::Tdx(e.to_string()))?;

    // dcap-qvl returns a TCB status (e.g. UpToDate/OutOfDate) and fails verification only for
    // invalid states like Revoked; we rely on that behavior and do not enforce a stricter policy.

    extract_verified_tdx_quote(&verified.report)
        .ok_or_else(|| NearAiError::Tdx("expected TDX TD report (TD10/TD15)".to_string()))
}

fn extract_verified_tdx_quote(report: &Report) -> Option<VerifiedTdxQuote> {
    // TD15 extends TD10. For v1 we only need TD10-common fields.
    let td10: &TDReport10 = match report {
        Report::TD10(td10) => td10,
        Report::TD15(td15) => &td15.base,
        _ => return None,
    };

    Some(VerifiedTdxQuote {
        report_data: td10.report_data,
        mr_config_id: td10.mr_config_id,
    })
}

pub fn verify_report_data_binding(
    report_data: &[u8; 64],
    signing_address: &str,
    request_nonce: &[u8; 32],
) -> Result<(), NearAiError> {
    let signing_address_bytes = parse_eth_address_bytes(signing_address)?;
    let mut expected_addr_field = [0u8; 32];
    expected_addr_field[..signing_address_bytes.len()].copy_from_slice(&signing_address_bytes);

    if report_data[..32] != expected_addr_field {
        return Err(NearAiError::Attestation(
            "TDX report_data does not bind signing_address".to_string(),
        ));
    }

    if report_data[32..] != request_nonce[..] {
        return Err(NearAiError::Attestation(
            "TDX report_data does not embed request_nonce".to_string(),
        ));
    }

    Ok(())
}

pub fn normalize_secp256k1_pubkey_hex(pubkey_hex: &str) -> Result<String, NearAiError> {
    let bytes = hex::decode(pubkey_hex.trim_start_matches("0x"))?;
    let normalized = match bytes.as_slice() {
        [0x04, rest @ ..] if rest.len() == 64 => rest.to_vec(),
        _ if bytes.len() == 64 => bytes,
        _ => {
            return Err(NearAiError::Attestation(format!(
                "unexpected secp256k1 public key length: {}",
                bytes.len()
            )))
        }
    };

    Ok(hex::encode(normalized))
}

pub fn secp256k1_pubkey_from_64hex(pubkey_hex: &str) -> Result<PublicKey, NearAiError> {
    let key_bytes = hex::decode(pubkey_hex.trim_start_matches("0x"))?;
    let key_65 = if key_bytes.len() == 65 {
        key_bytes
    } else if key_bytes.len() == 64 {
        let mut prefixed = Vec::with_capacity(65);
        prefixed.push(0x04);
        prefixed.extend_from_slice(&key_bytes);
        prefixed
    } else {
        return Err(NearAiError::Attestation(format!(
            "unexpected secp256k1 public key length: {}",
            key_bytes.len()
        )));
    };

    Ok(PublicKey::from_slice(&key_65)?)
}

pub fn verify_signing_pubkey_matches_address(
    signing_public_key_hex: &str,
    signing_address: &str,
) -> Result<String, NearAiError> {
    let normalized_pubkey_hex = normalize_secp256k1_pubkey_hex(signing_public_key_hex)?;
    let pubkey_bytes = hex::decode(&normalized_pubkey_hex)?;

    let derived_addr = ethereum_address_from_uncompressed_pubkey_64(&pubkey_bytes);
    let expected_addr = parse_eth_address_bytes(signing_address)?;

    if derived_addr != expected_addr {
        return Err(NearAiError::Attestation(
            "signing_public_key does not match signing_address".to_string(),
        ));
    }

    Ok(normalized_pubkey_hex)
}

pub fn verify_compose_manifest(
    mr_config_id: &[u8; 48],
    info: &AttestationInfo,
) -> Result<(), NearAiError> {
    let tcb_info = info
        .tcb_info
        .as_ref()
        .ok_or_else(|| NearAiError::Attestation("missing info.tcb_info".to_string()))?;

    let tcb_info_obj: serde_json::Value = if let Some(s) = tcb_info.as_str() {
        serde_json::from_str(s)?
    } else {
        tcb_info.clone()
    };

    let app_compose = tcb_info_obj
        .get("app_compose")
        .and_then(|v| v.as_str())
        .ok_or_else(|| NearAiError::Attestation("missing tcb_info.app_compose".to_string()))?;

    let app_compose_obj: serde_json::Value = serde_json::from_str(app_compose).map_err(|e| {
        NearAiError::Attestation(format!("tcb_info.app_compose is not valid JSON: {e}"))
    })?;
    let docker_compose_file = app_compose_obj
        .get("docker_compose_file")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            NearAiError::Attestation("tcb_info.app_compose missing docker_compose_file".to_string())
        })?;
    if docker_compose_file.trim().is_empty() {
        return Err(NearAiError::Attestation(
            "tcb_info.app_compose docker_compose_file is empty".to_string(),
        ));
    }

    let compose_hash = Sha256::digest(app_compose.as_bytes());
    let expected_prefix = format!("01{}", hex::encode(compose_hash));
    let mr_config_hex = hex::encode(mr_config_id);

    if !mr_config_hex.starts_with(&expected_prefix) {
        return Err(NearAiError::Attestation(
            "mr_config_id does not match app_compose sha256".to_string(),
        ));
    }

    Ok(())
}

fn parse_eth_address_bytes(addr: &str) -> Result<[u8; 20], NearAiError> {
    let addr = addr.trim_start_matches("0x");
    let bytes = hex::decode(addr)?;
    if bytes.len() != 20 {
        return Err(NearAiError::Attestation(format!(
            "expected 20-byte Ethereum address, got {} bytes",
            bytes.len()
        )));
    }

    Ok(bytes.try_into().expect("checked length is 20 bytes"))
}

fn ethereum_address_from_uncompressed_pubkey_64(pubkey_64: &[u8]) -> [u8; 20] {
    // Ethereum address is keccak256(uncompressed_pubkey_without_prefix)[12..].
    let hash = Keccak256::digest(pubkey_64);
    hash[12..32].try_into().expect("slice length is 20 bytes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcap_qvl::quote::{TDReport10, TDReport15};
    use secp256k1::{PublicKey, Secp256k1, SecretKey};
    use serde_json::json;

    fn td10_with_fields(report_data: [u8; 64], mr_config_id: [u8; 48]) -> TDReport10 {
        TDReport10 {
            tee_tcb_svn: [0u8; 16],
            mr_seam: [0u8; 48],
            mr_signer_seam: [0u8; 48],
            seam_attributes: [0u8; 8],
            td_attributes: [0u8; 8],
            xfam: [0u8; 8],
            mr_td: [0u8; 48],
            mr_config_id,
            mr_owner: [0u8; 48],
            mr_owner_config: [0u8; 48],
            rt_mr0: [0u8; 48],
            rt_mr1: [0u8; 48],
            rt_mr2: [0u8; 48],
            rt_mr3: [0u8; 48],
            report_data,
        }
    }

    #[test]
    fn test_extract_verified_tdx_quote_supports_td10_and_td15() {
        let report_data = [0xAAu8; 64];
        let mr_config_id = [0xBBu8; 48];
        let td10 = td10_with_fields(report_data, mr_config_id);

        let r10 = Report::TD10(td10);
        let extracted10 = extract_verified_tdx_quote(&r10).expect("td10 should extract");
        assert_eq!(extracted10.report_data, report_data);
        assert_eq!(extracted10.mr_config_id, mr_config_id);

        let td15 = TDReport15 {
            base: td10,
            tee_tcb_svn2: [0u8; 16],
            mr_service_td: [0u8; 48],
        };
        let r15 = Report::TD15(td15);
        let extracted15 = extract_verified_tdx_quote(&r15).expect("td15 should extract");
        assert_eq!(extracted15.report_data, report_data);
        assert_eq!(extracted15.mr_config_id, mr_config_id);
    }

    #[test]
    fn test_verify_report_data_binding_ok_and_mismatch() {
        let addr_bytes = [0x11u8; 20];
        let signing_address = format!("0x{}", hex::encode(addr_bytes));
        let nonce = [0x22u8; 32];

        let mut report_data = [0u8; 64];
        report_data[..20].copy_from_slice(&addr_bytes);
        report_data[32..].copy_from_slice(&nonce);

        assert!(verify_report_data_binding(&report_data, &signing_address, &nonce).is_ok());

        let mut bad_addr = report_data;
        bad_addr[0] ^= 0xff;
        assert!(verify_report_data_binding(&bad_addr, &signing_address, &nonce).is_err());

        let mut bad_nonce = report_data;
        bad_nonce[63] ^= 0x01;
        assert!(verify_report_data_binding(&bad_nonce, &signing_address, &nonce).is_err());
    }

    #[test]
    fn test_pubkey_normalization_and_address_binding() {
        let mut sk_bytes = [0u8; 32];
        sk_bytes[31] = 1;
        let sk = SecretKey::from_slice(&sk_bytes).unwrap();
        let secp = Secp256k1::new();
        let pk = PublicKey::from_secret_key(&secp, &sk);
        let pk_uncompressed = pk.serialize_uncompressed();

        let pk_65_hex = hex::encode(pk_uncompressed);
        let pk_64_hex = hex::encode(&pk_uncompressed[1..]);

        assert_eq!(
            normalize_secp256k1_pubkey_hex(&pk_65_hex).unwrap(),
            pk_64_hex
        );
        assert_eq!(
            normalize_secp256k1_pubkey_hex(&pk_64_hex).unwrap(),
            pk_64_hex
        );

        assert!(secp256k1_pubkey_from_64hex(&pk_65_hex).is_ok());
        assert!(secp256k1_pubkey_from_64hex(&pk_64_hex).is_ok());

        let derived_addr = ethereum_address_from_uncompressed_pubkey_64(&pk_uncompressed[1..]);
        let signing_address = format!("0x{}", hex::encode(derived_addr));

        let normalized = verify_signing_pubkey_matches_address(&pk_65_hex, &signing_address)
            .expect("pubkey should match derived address");
        assert_eq!(normalized, pk_64_hex);

        let wrong_address = "0x0000000000000000000000000000000000000000";
        assert!(verify_signing_pubkey_matches_address(&pk_65_hex, wrong_address).is_err());
    }

    #[test]
    fn test_verify_compose_manifest_accepts_object_and_string() {
        let docker_compose_file = "version: '3'\nservices:\n  app:\n    image: example";
        let app_compose = json!({"docker_compose_file": docker_compose_file}).to_string();
        let tcb = json!({"app_compose": app_compose});

        let info_obj = AttestationInfo {
            tcb_info: Some(tcb.clone()),
        };
        let info_str = AttestationInfo {
            tcb_info: Some(serde_json::Value::String(tcb.to_string())),
        };

        let compose_hash = Sha256::digest(app_compose.as_bytes());
        let mut mr_config_id = [0u8; 48];
        mr_config_id[0] = 0x01;
        mr_config_id[1..33].copy_from_slice(&compose_hash);

        assert!(verify_compose_manifest(&mr_config_id, &info_obj).is_ok());
        assert!(verify_compose_manifest(&mr_config_id, &info_str).is_ok());

        let mut bad = mr_config_id;
        bad[1] ^= 0x01;
        assert!(verify_compose_manifest(&bad, &info_obj).is_err());
    }
}
