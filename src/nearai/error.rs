use thiserror::Error;

#[derive(Debug, Error)]
pub enum NearAiError {
    #[error("Near.AI is not configured")]
    Unconfigured,

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("HTTP {status} from {url}: {body}")]
    HttpStatus {
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Hex decode error: {0}")]
    Hex(#[from] hex::FromHexError),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("Attestation verification error: {0}")]
    Attestation(String),

    #[error("TDX quote verification error: {0}")]
    Tdx(String),

    #[error("NVIDIA NRAS verification error: {0}")]
    Nras(String),

    #[error("JWT verification error: {0}")]
    Jwt(String),

    #[error("NRAS JWT kid not found in JWKS: {0}")]
    JwtKidNotFound(String),
}

impl From<secp256k1::Error> for NearAiError {
    fn from(e: secp256k1::Error) -> Self {
        Self::Crypto(e.to_string())
    }
}

impl From<aes_gcm::Error> for NearAiError {
    fn from(e: aes_gcm::Error) -> Self {
        Self::Crypto(e.to_string())
    }
}

impl From<hkdf::InvalidLength> for NearAiError {
    fn from(e: hkdf::InvalidLength) -> Self {
        Self::Crypto(e.to_string())
    }
}

impl From<sha2::digest::InvalidLength> for NearAiError {
    fn from(e: sha2::digest::InvalidLength) -> Self {
        Self::Crypto(e.to_string())
    }
}

impl From<jsonwebtoken::errors::Error> for NearAiError {
    fn from(e: jsonwebtoken::errors::Error) -> Self {
        Self::Jwt(e.to_string())
    }
}
