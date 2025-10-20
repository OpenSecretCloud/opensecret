//! Minimal Brave Search API client
//!
//! This module provides a lightweight client for the Brave Search API.
//! Only includes what we actually use - no bloat from auto-generated code.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

const BRAVE_API_BASE: &str = "https://api.search.brave.com/res/v1";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum BraveError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
}

/// Brave API client with reusable HTTP client and stored API key
#[derive(Clone)]
pub struct BraveClient {
    client: reqwest::Client,
    api_key: Arc<String>,
}

impl BraveClient {
    /// Create a new Brave client with the given API key
    pub fn new(api_key: String) -> Result<Self, BraveError> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(100)
            .user_agent("OpenSecret/0.1.0")
            .build()
            .map_err(BraveError::Request)?;

        Ok(Self {
            client,
            api_key: Arc::new(api_key),
        })
    }

    /// Execute a web search query
    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse, BraveError> {
        let url = format!("{}/web/search", BRAVE_API_BASE);

        let mut query_params = vec![("q", request.query.clone())];

        if let Some(country) = &request.country {
            query_params.push(("country", country.clone()));
        }
        if let Some(search_lang) = &request.search_lang {
            query_params.push(("search_lang", search_lang.clone()));
        }
        if let Some(count) = request.count {
            query_params.push(("count", count.to_string()));
        }
        if let Some(offset) = request.offset {
            query_params.push(("offset", offset.to_string()));
        }
        if let Some(safesearch) = &request.safesearch {
            query_params.push(("safesearch", safesearch.clone()));
        }
        if let Some(freshness) = &request.freshness {
            query_params.push(("freshness", freshness.clone()));
        }
        if let Some(summary) = request.summary {
            query_params.push(("summary", if summary { "1" } else { "0" }.to_string()));
        }

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json")
            .query(&query_params)
            .send()
            .await?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(BraveError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let search_response = response.json::<SearchResponse>().await?;
        Ok(search_response)
    }

    /// Execute a summarizer query using a key from web search
    pub async fn summarizer(&self, key: &str) -> Result<SummarizerSearchResponse, BraveError> {
        let url = format!("{}/summarizer/search", BRAVE_API_BASE);

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json")
            .query(&[("key", key)])
            .send()
            .await?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(BraveError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let summarizer_response = response.json::<SummarizerSearchResponse>().await?;
        Ok(summarizer_response)
    }
}

impl std::fmt::Debug for BraveClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BraveClient")
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safesearch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<bool>,
}

impl SearchRequest {
    pub fn new(query: String) -> Self {
        Self {
            query,
            country: None,
            search_lang: None,
            count: Some(10),
            offset: None,
            safesearch: None,
            freshness: None,
            summary: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<QueryInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web: Option<WebResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub news: Option<NewsResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infobox: Option<Infobox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summarizer: Option<Summarizer>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct QueryInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub altered: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct WebResults {
    #[serde(rename = "type")]
    pub results_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<SearchResult>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct NewsResults {
    #[serde(rename = "type")]
    pub results_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<NewsResult>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_snippets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct NewsResult {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Infobox {
    #[serde(rename = "type")]
    pub infobox_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_desc: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Summarizer {
    #[serde(rename = "type")]
    pub summarizer_type: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SummarizerSearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<SummaryItem>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SummaryItem {
    #[serde(rename = "type")]
    pub item_type: String,
    /// For type="token", data is a string with the text
    /// For type="enum_item", data is a SummaryEntity object
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}
