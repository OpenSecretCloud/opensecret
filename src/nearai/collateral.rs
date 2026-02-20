// Custom collateral fetching for Intel DCAP TDX quote verification.
//
// This replaces `dcap_qvl::collateral::get_collateral` because that function
// builds its own reqwest client with the `hickory-dns` feature enabled, which
// bypasses /etc/hosts and requires a real DNS server. Inside Nitro enclaves
// there is no DNS server -- all external hosts are resolved via /etc/hosts
// entries pointing to local vsock traffic forwarders.
//
// The logic here is adapted from dcap-qvl 0.3.12 collateral.rs:
//   https://crates.io/crates/dcap-qvl/0.3.12
//   Source: src/collateral.rs (get_collateral, get_collateral_for_fmspc_impl, PcsEndpoints)
//
// We only replicate the HTTP fetching and QuoteCollateralV3 assembly.
// All cryptographic verification remains in dcap_qvl::verify::ring::verify.

use crate::nearai::error::NearAiError;
use dcap_qvl::quote::Quote;
use dcap_qvl::QuoteCollateralV3;
use serde::Deserialize;
use tracing::debug;
use x509_parser::prelude::*;

#[derive(Deserialize)]
struct TcbInfoResponse {
    #[serde(rename = "tcbInfo")]
    tcb_info: serde_json::Value,
    signature: String,
}

#[derive(Deserialize)]
struct QeIdentityResponse {
    #[serde(rename = "enclaveIdentity")]
    enclave_identity: serde_json::Value,
    signature: String,
}

fn url_decode_header(response: &reqwest::Response, name: &str) -> Result<String, NearAiError> {
    let value = response
        .headers()
        .get(name)
        .ok_or_else(|| NearAiError::Tdx(format!("missing response header: {name}")))?
        .to_str()
        .map_err(|e| NearAiError::Tdx(format!("non-ascii header {name}: {e}")))?;

    // URL-decode the header value (Intel PCS percent-encodes certificate chains)
    let decoded = percent_decode(value);
    Ok(decoded)
}

fn percent_decode(input: &str) -> String {
    let mut result = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                result.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(result).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).to_string())
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Fetch DCAP collateral from Intel PCS using the provided reqwest client.
///
/// This is a drop-in replacement for `dcap_qvl::collateral::get_collateral`
/// that accepts an externally-built reqwest client (without hickory-dns).
pub async fn get_collateral(
    client: &reqwest::Client,
    pccs_url: &str,
    quote_bytes: &[u8],
) -> Result<QuoteCollateralV3, NearAiError> {
    let parsed_quote = Quote::parse(quote_bytes).map_err(|e| NearAiError::Tdx(format!("{e:#}")))?;

    let fmspc = hex::encode_upper(
        parsed_quote
            .fmspc()
            .map_err(|e| NearAiError::Tdx(format!("{e:#}")))?,
    );
    let ca = parsed_quote
        .ca()
        .map_err(|e| NearAiError::Tdx(format!("{e:#}")))?;
    let tee = if parsed_quote.header.is_sgx() {
        "sgx"
    } else {
        "tdx"
    };

    let base = pccs_url
        .trim_end_matches('/')
        .trim_end_matches("/sgx/certification/v4")
        .trim_end_matches("/tdx/certification/v4");

    let mk_url =
        |t: &str, path: &str| -> String { format!("{}/{}/certification/v4/{}", base, t, path) };

    // PCK CRL
    let pckcrl_url = mk_url("sgx", &format!("pckcrl?ca={}&encoding=der", ca));
    let response = client
        .get(&pckcrl_url)
        .send()
        .await
        .map_err(|e| NearAiError::Tdx(format!("PCK CRL fetch failed: {e:#}")))?;
    let pck_crl_issuer_chain = url_decode_header(&response, "SGX-PCK-CRL-Issuer-Chain")?;
    let pck_crl = response
        .bytes()
        .await
        .map_err(|e| NearAiError::Tdx(format!("PCK CRL read failed: {e:#}")))?
        .to_vec();

    // TCB Info
    let tcb_url = mk_url(tee, &format!("tcb?fmspc={}", fmspc));
    let response = client
        .get(&tcb_url)
        .send()
        .await
        .map_err(|e| NearAiError::Tdx(format!("TCB Info fetch failed: {e:#}")))?;
    let tcb_info_issuer_chain = url_decode_header(&response, "SGX-TCB-Info-Issuer-Chain")
        .or_else(|_| url_decode_header(&response, "TCB-Info-Issuer-Chain"))?;
    let raw_tcb_info = response
        .text()
        .await
        .map_err(|e| NearAiError::Tdx(format!("TCB Info read failed: {e:#}")))?;

    // QE Identity
    let qe_url = mk_url(tee, "qe/identity?update=standard");
    let response = client
        .get(&qe_url)
        .send()
        .await
        .map_err(|e| NearAiError::Tdx(format!("QE Identity fetch failed: {e:#}")))?;
    let qe_identity_issuer_chain =
        url_decode_header(&response, "SGX-Enclave-Identity-Issuer-Chain")?;
    let raw_qe_identity = response
        .text()
        .await
        .map_err(|e| NearAiError::Tdx(format!("QE Identity read failed: {e:#}")))?;

    // Root CA CRL -- try the PCCS rootcacrl endpoint first (works for caching
    // services like Phala PCCS). If that fails (Intel PCS doesn't serve it),
    // extract the CRL distribution point URL from the root certificate in the
    // QE identity issuer chain and fetch it directly.
    // See dcap-qvl 0.3.12 collateral.rs lines 353-382 for the original logic.
    let root_ca_crl = {
        let rootcacrl_url = mk_url("sgx", "rootcacrl");
        let rootcacrl_result = client.get(&rootcacrl_url).send().await;
        match rootcacrl_result {
            Ok(resp) if resp.status().is_success() => {
                let bytes = resp
                    .bytes()
                    .await
                    .map_err(|e| NearAiError::Tdx(format!("Root CA CRL read failed: {e:#}")))?;
                // PCCS returns hex-encoded CRL; try to hex-decode, otherwise use raw DER.
                if let Ok(hex_str) = std::str::from_utf8(&bytes) {
                    hex::decode(hex_str).unwrap_or_else(|_| bytes.to_vec())
                } else {
                    bytes.to_vec()
                }
            }
            _ => {
                debug!("rootcacrl endpoint unavailable, extracting CRL URL from issuer chain");
                let crl_url = extract_root_ca_crl_url(&qe_identity_issuer_chain)?;
                let resp =
                    client.get(&crl_url).send().await.map_err(|e| {
                        NearAiError::Tdx(format!("Root CA CRL fetch failed: {e:#}"))
                    })?;
                if !resp.status().is_success() {
                    return Err(NearAiError::Tdx(format!(
                        "Root CA CRL fetch from {} returned {}",
                        crl_url,
                        resp.status()
                    )));
                }
                resp.bytes()
                    .await
                    .map_err(|e| NearAiError::Tdx(format!("Root CA CRL read failed: {e:#}")))?
                    .to_vec()
            }
        }
    };

    // Parse TCB Info
    let tcb_info_resp: TcbInfoResponse = serde_json::from_str(&raw_tcb_info)?;
    let tcb_info = tcb_info_resp.tcb_info.to_string();
    let tcb_info_signature = hex::decode(&tcb_info_resp.signature)
        .map_err(|e| NearAiError::Tdx(format!("TCB Info signature hex decode failed: {e}")))?;

    // Parse QE Identity
    let qe_identity_resp: QeIdentityResponse = serde_json::from_str(&raw_qe_identity)?;
    let qe_identity = qe_identity_resp.enclave_identity.to_string();
    let qe_identity_signature = hex::decode(&qe_identity_resp.signature)
        .map_err(|e| NearAiError::Tdx(format!("QE Identity signature hex decode failed: {e}")))?;

    // PCK certificate chain (cert_type 5 embeds it in the quote)
    let pck_certificate_chain = parsed_quote
        .raw_cert_chain()
        .ok()
        .map(|chain| String::from_utf8_lossy(chain).to_string());

    Ok(QuoteCollateralV3 {
        pck_crl_issuer_chain,
        root_ca_crl,
        pck_crl,
        tcb_info_issuer_chain,
        tcb_info,
        tcb_info_signature,
        qe_identity_issuer_chain,
        qe_identity,
        qe_identity_signature,
        pck_certificate_chain,
    })
}

/// Extract the CRL distribution point URL from the root certificate in a PEM
/// issuer chain. The root cert is the last in the chain. This mirrors dcap-qvl's
/// `extract_crl_url` (collateral.rs lines 116-145).
fn extract_root_ca_crl_url(issuer_chain_pem: &str) -> Result<String, NearAiError> {
    let pem_certs: Vec<_> = Pem::iter_from_buffer(issuer_chain_pem.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| NearAiError::Tdx(format!("failed to parse PEM issuer chain: {e}")))?;

    let root_pem = pem_certs
        .last()
        .ok_or_else(|| NearAiError::Tdx("empty issuer chain".to_string()))?;

    let (_, cert) = X509Certificate::from_der(root_pem.contents.as_ref())
        .map_err(|e| NearAiError::Tdx(format!("failed to parse root cert DER: {e}")))?;

    for ext in cert.extensions() {
        if let ParsedExtension::CRLDistributionPoints(cdp) = ext.parsed_extension() {
            for dp in cdp.iter() {
                if let Some(DistributionPointName::FullName(names)) = &dp.distribution_point {
                    for name in names {
                        if let GeneralName::URI(uri) = name {
                            return Ok(uri.to_string());
                        }
                    }
                }
            }
        }
    }

    Err(NearAiError::Tdx(
        "no CRL distribution point found in root certificate".to_string(),
    ))
}
