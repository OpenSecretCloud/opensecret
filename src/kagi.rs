//! Minimal Kagi Search API client
//!
//! This module provides a lightweight client for the Kagi Search API.
//! Only includes what we actually use - no bloat from auto-generated code.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

const KAGI_API_BASE: &str = "https://kagi.com/api/v1";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum KagiError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
}

/// Kagi API client with reusable HTTP client and stored API key
#[derive(Clone)]
pub struct KagiClient {
    client: reqwest::Client,
    api_key: Arc<String>,
}

impl KagiClient {
    /// Create a new Kagi client with the given API key
    pub fn new(api_key: String) -> Result<Self, KagiError> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(100)
            .user_agent("OpenAPI-Generator/0.1.0/rust")
            .build()
            .map_err(KagiError::Request)?;

        Ok(Self {
            client,
            api_key: Arc::new(api_key),
        })
    }

    /// Execute a search query
    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse, KagiError> {
        let url = format!("{}/search", KAGI_API_BASE);

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.api_key.as_str())
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(KagiError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let search_response = response.json::<SearchResponse>().await?;
        Ok(search_response)
    }
}

impl std::fmt::Debug for KagiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KagiClient")
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow: Option<Workflow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens: Option<LensConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

impl SearchRequest {
    pub fn new(query: String) -> Self {
        Self {
            query,
            workflow: None,
            lens_id: None,
            lens: None,
            timeout: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Workflow {
    Search,
    Images,
    Videos,
    News,
    Podcasts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<Meta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<SearchData>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Meta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub podcast: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub podcast_creator: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub news: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adjacent_question: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_answer: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interesting_news: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interesting_finds: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infobox: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_tracking: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_records: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weather: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_search: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub listicle: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_archive: Option<Vec<SearchResult>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<SearchResultImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub props: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResultImage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<i32>,
}
