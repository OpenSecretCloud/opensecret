use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AttestationReport {
    pub gateway_attestation: AttestationBaseInfo,

    #[serde(default)]
    pub model_attestations: Vec<AttestationBaseInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttestationBaseInfo {
    pub signing_address: String,

    #[serde(default)]
    pub signing_public_key: Option<String>,

    #[serde(default)]
    pub signing_algo: Option<String>,

    pub intel_quote: String,

    #[serde(default)]
    pub nvidia_payload: Option<String>,

    pub info: AttestationInfo,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttestationInfo {
    #[serde(default)]
    pub tcb_info: Option<serde_json::Value>,
}
