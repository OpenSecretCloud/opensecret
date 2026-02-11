use reqwest::StatusCode;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use url::Url;
use uuid::Uuid;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum OsFlagsError {
    #[error("Invalid base URL: {0}")]
    InvalidBaseUrl(#[from] url::ParseError),

    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Service error: {status} - {message}")]
    Service { status: StatusCode, message: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct UserFlagsResponse {
    pub user_uuid: Uuid,
    pub flags: HashMap<String, bool>,
    #[serde(default)]
    pub reasons: HashMap<String, String>,
}

#[derive(Clone)]
pub struct OsFlagsClient {
    client: reqwest::Client,
    base_url: Url,
    api_key: Option<Arc<String>>,
}

impl OsFlagsClient {
    pub fn new(base_url: String, api_key: Option<String>) -> Result<Self, OsFlagsError> {
        let base_url = format!("{}/", base_url.trim_end_matches('/'));
        let base_url = Url::parse(&base_url)?;

        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(100)
            .user_agent("opensecret/os-flags-client")
            .build()?;

        Ok(Self {
            client,
            base_url,
            api_key: api_key.map(Arc::new),
        })
    }

    pub async fn get_user_flags(
        &self,
        user_uuid: Uuid,
        keys: Option<&[&str]>,
        include_reasons: bool,
    ) -> Result<UserFlagsResponse, OsFlagsError> {
        let mut url = self
            .base_url
            .join(&format!("v1/users/{}/flags", user_uuid))?;

        {
            let mut qp = url.query_pairs_mut();
            if let Some(keys) = keys {
                if !keys.is_empty() {
                    qp.append_pair("keys", &keys.join(","));
                }
            }
            if include_reasons {
                qp.append_pair("include_reasons", "1");
            }
        }

        let mut req = self.client.get(url);
        if let Some(api_key) = &self.api_key {
            req = req.bearer_auth(api_key.as_str());
        }

        let resp = req.send().await?;
        let status = resp.status();

        if status.is_success() {
            Ok(resp.json::<UserFlagsResponse>().await?)
        } else {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.as_str())
                        .map(ToOwned::to_owned)
                })
                .unwrap_or(text);

            Err(OsFlagsError::Service { status, message })
        }
    }

    pub async fn is_enabled(&self, user_uuid: Uuid, key: &str) -> Result<bool, OsFlagsError> {
        let keys = [key];
        let resp = self.get_user_flags(user_uuid, Some(&keys), false).await?;
        Ok(resp.flags.get(key).copied().unwrap_or(false))
    }
}

impl std::fmt::Debug for OsFlagsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsFlagsClient")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}
