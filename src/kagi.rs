//! Minimal client for Kagi's v1 Search and Extract APIs.
//!
//! Search intentionally does not request inline extraction. Callers can inspect
//! the search results and then explicitly extract only the pages they need.

use reqwest::{header, StatusCode};
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize};
use std::{fmt, sync::Arc, time::Duration};
use url::Url;

const KAGI_API_BASE: &str = "https://kagi.com/api/v1/";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const SEARCH_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const EXTRACT_REQUEST_TIMEOUT: Duration = Duration::from_secs(35);
const SEARCH_RESPONSE_LIMIT_BYTES: usize = 1024 * 1024;
const EXTRACT_RESPONSE_LIMIT_BYTES: usize = 5 * 1024 * 1024;
const ERROR_MESSAGE_LIMIT_CHARS: usize = 4 * 1024;
const MAX_EXTRACT_URLS: usize = 3;
const TRACE_HEADER: &str = "x-kagi-trace";
const TRACE_ID_LIMIT_CHARS: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum KagiError {
    #[error("Kagi API key cannot be empty")]
    InvalidApiKey,

    #[error("Kagi search query cannot be empty")]
    InvalidQuery,

    #[error("Kagi extract requires between 1 and {MAX_EXTRACT_URLS} URLs (received {count})")]
    InvalidUrlCount { count: usize },

    #[error("Kagi extract URL at index {index} is invalid: {reason}")]
    InvalidUrl { index: usize, reason: String },

    #[error("invalid Kagi API base URL: {0}")]
    InvalidBaseUrl(#[from] url::ParseError),

    #[error("Kagi {operation} request failed: {source}")]
    Request {
        operation: &'static str,
        #[source]
        source: reqwest::Error,
    },

    #[error("Kagi {operation} response exceeded {limit_bytes} bytes (trace ID: {trace_id})")]
    ResponseTooLarge {
        operation: &'static str,
        limit_bytes: usize,
        trace_id: String,
    },

    #[error("Kagi {operation} API returned HTTP {status} (trace ID: {trace_id})")]
    Api {
        operation: &'static str,
        status: StatusCode,
        message: String,
        trace_id: String,
    },

    #[error("invalid Kagi {operation} response (trace ID: {trace_id})")]
    InvalidResponse {
        operation: &'static str,
        message: String,
        trace_id: String,
    },
}

/// Kagi API client with a reusable connection pool and redacted credentials.
#[derive(Clone)]
pub struct KagiClient {
    client: reqwest::Client,
    api_key: Arc<str>,
    base_url: Url,
}

impl KagiClient {
    pub fn new(api_key: String) -> Result<Self, KagiError> {
        Self::new_with_base_url_inner(api_key, KAGI_API_BASE)
    }

    #[cfg(test)]
    fn new_with_base_url(api_key: String, base_url: &str) -> Result<Self, KagiError> {
        Self::new_with_base_url_inner(api_key, base_url)
    }

    fn new_with_base_url_inner(api_key: String, base_url: &str) -> Result<Self, KagiError> {
        let api_key = api_key.trim().to_owned();
        if api_key.is_empty() {
            return Err(KagiError::InvalidApiKey);
        }

        let base_url = Url::parse(&format!("{}/", base_url.trim_end_matches('/')))?;
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(EXTRACT_REQUEST_TIMEOUT)
            .pool_max_idle_per_host(100)
            .user_agent("OpenSecret/0.1.0")
            .build()
            .map_err(|source| KagiError::Request {
                operation: "client initialization",
                source,
            })?;

        Ok(Self {
            client,
            api_key: Arc::from(api_key),
            base_url,
        })
    }

    /// Search Kagi for web and news results without fetching page contents.
    pub async fn search(&self, query: &str) -> Result<SearchResponse, KagiError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(KagiError::InvalidQuery);
        }

        let request = SearchRequest {
            query,
            workflow: "search",
            format: "json",
            limit: 10,
            safe_search: true,
        };

        let response = self
            .client
            .post(self.endpoint("search")?)
            .bearer_auth(self.api_key.as_ref())
            .header(header::ACCEPT, "application/json")
            .json(&request)
            .timeout(SEARCH_REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|source| KagiError::Request {
                operation: "search",
                source,
            })?;

        parse_response(response, "search", SEARCH_RESPONSE_LIMIT_BYTES).await
    }

    /// Extract Markdown from one to three HTTPS URLs.
    pub async fn extract(&self, urls: &[String]) -> Result<ExtractResponse, KagiError> {
        validate_extract_urls(urls)?;

        let request = ExtractRequest {
            pages: urls.iter().map(|url| PageInput { url }).collect(),
            format: "json",
        };

        let response = self
            .client
            .post(self.endpoint("extract")?)
            .bearer_auth(self.api_key.as_ref())
            .header(header::ACCEPT, "application/json")
            .json(&request)
            .timeout(EXTRACT_REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|source| KagiError::Request {
                operation: "extract",
                source,
            })?;

        parse_response(response, "extract", EXTRACT_RESPONSE_LIMIT_BYTES).await
    }

    fn endpoint(&self, path: &'static str) -> Result<Url, KagiError> {
        Ok(self.base_url.join(path)?)
    }
}

impl fmt::Debug for KagiClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KagiClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Meta {
    #[serde(default, deserialize_with = "deserialize_optional_trace")]
    pub trace: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchResponse {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub meta: Meta,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub data: SearchData,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchData {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub search: Vec<SearchResult>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub news: Vec<SearchResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    #[serde(default)]
    pub snippet: Option<String>,
    #[serde(default)]
    pub time: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExtractResponse {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub meta: Meta,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub data: Vec<ExtractPage>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub errors: Vec<ErrorDetail>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExtractPage {
    pub url: String,
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ErrorDetail {
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    workflow: &'static str,
    format: &'static str,
    limit: u8,
    safe_search: bool,
}

#[derive(Serialize)]
struct ExtractRequest<'a> {
    pages: Vec<PageInput<'a>>,
    format: &'static str,
}

#[derive(Serialize)]
struct PageInput<'a> {
    url: &'a str,
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_optional_trace<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.map(|trace| sanitize_trace_id(&trace)))
}

pub(crate) fn sanitize_trace_id(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | ':')
        })
        .take(TRACE_ID_LIMIT_CHARS)
        .collect();
    if sanitized.is_empty() {
        "unavailable".to_owned()
    } else {
        sanitized
    }
}

fn validate_extract_urls(urls: &[String]) -> Result<(), KagiError> {
    if urls.is_empty() || urls.len() > MAX_EXTRACT_URLS {
        return Err(KagiError::InvalidUrlCount { count: urls.len() });
    }

    for (index, raw_url) in urls.iter().enumerate() {
        let url = Url::parse(raw_url).map_err(|error| KagiError::InvalidUrl {
            index,
            reason: error.to_string(),
        })?;

        if url.scheme() != "https" {
            return Err(KagiError::InvalidUrl {
                index,
                reason: "URL must use HTTPS".to_owned(),
            });
        }
        if url.host_str().is_none() {
            return Err(KagiError::InvalidUrl {
                index,
                reason: "URL must include a host".to_owned(),
            });
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(KagiError::InvalidUrl {
                index,
                reason: "URL must not contain credentials".to_owned(),
            });
        }
    }

    Ok(())
}

async fn parse_response<T: DeserializeOwned>(
    response: reqwest::Response,
    operation: &'static str,
    response_limit: usize,
) -> Result<T, KagiError> {
    let status = response.status();
    let header_trace = response
        .headers()
        .get(TRACE_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(sanitize_trace_id);
    let body = read_bounded_body(response, operation, response_limit, &header_trace).await?;
    let body_trace = trace_from_body(&body);
    let trace_id = body_trace
        .or(header_trace)
        .unwrap_or_else(|| "unavailable".to_owned());

    if !status.is_success() {
        return Err(KagiError::Api {
            operation,
            status,
            message: api_error_message(&body),
            trace_id,
        });
    }

    serde_json::from_slice(&body).map_err(|error| KagiError::InvalidResponse {
        operation,
        message: error.to_string(),
        trace_id,
    })
}

async fn read_bounded_body(
    mut response: reqwest::Response,
    operation: &'static str,
    response_limit: usize,
    trace: &Option<String>,
) -> Result<Vec<u8>, KagiError> {
    if response
        .content_length()
        .is_some_and(|length| length > response_limit as u64)
    {
        return Err(KagiError::ResponseTooLarge {
            operation,
            limit_bytes: response_limit,
            trace_id: trace.clone().unwrap_or_else(|| "unavailable".to_owned()),
        });
    }

    let initial_capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or(0)
        .min(response_limit);
    let mut body = Vec::with_capacity(initial_capacity);

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|source| KagiError::Request { operation, source })?
    {
        if body.len().saturating_add(chunk.len()) > response_limit {
            return Err(KagiError::ResponseTooLarge {
                operation,
                limit_bytes: response_limit,
                trace_id: trace.clone().unwrap_or_else(|| "unavailable".to_owned()),
            });
        }
        body.extend_from_slice(&chunk);
    }

    Ok(body)
}

fn trace_from_body(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("meta")?
        .get("trace")?
        .as_str()
        .map(sanitize_trace_id)
}

fn api_error_message(body: &[u8]) -> String {
    let parsed = serde_json::from_slice::<serde_json::Value>(body).ok();
    let mut messages = Vec::new();

    if let Some(value) = parsed.as_ref() {
        collect_error_messages(value.get("error"), &mut messages);
        collect_error_messages(value.get("errors"), &mut messages);
        if messages.is_empty() {
            collect_string(value.get("message"), &mut messages);
        }
    }

    let message = if messages.is_empty() {
        let text = String::from_utf8_lossy(body).trim().to_owned();
        if text.is_empty() {
            "empty error response".to_owned()
        } else {
            text
        }
    } else {
        messages.join("; ")
    };

    truncate_chars(&message, ERROR_MESSAGE_LIMIT_CHARS)
}

fn collect_error_messages(value: Option<&serde_json::Value>, messages: &mut Vec<String>) {
    let Some(value) = value else {
        return;
    };

    match value {
        serde_json::Value::String(message) => messages.push(message.clone()),
        serde_json::Value::Array(errors) => {
            for error in errors {
                collect_error_messages(Some(error), messages);
            }
        }
        serde_json::Value::Object(error) => {
            let code = error.get("code").and_then(serde_json::Value::as_str);
            let message = error.get("message").and_then(serde_json::Value::as_str);
            match (code, message) {
                (Some(code), Some(message)) => messages.push(format!("{code}: {message}")),
                (Some(code), None) => messages.push(code.to_owned()),
                (None, Some(message)) => messages.push(message.to_owned()),
                (None, None) => {}
            }
        }
        _ => {}
    }
}

fn collect_string(value: Option<&serde_json::Value>, messages: &mut Vec<String>) {
    if let Some(message) = value.and_then(serde_json::Value::as_str) {
        messages.push(message.to_owned());
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode as AxumStatusCode},
        response::Response,
        routing::post,
        Json, Router,
    };
    use serde_json::{json, Value};
    use tokio::net::TcpListener;

    async fn test_server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn search_sends_v1_contract_and_parses_web_and_news() {
        async fn handler(headers: HeaderMap, Json(body): Json<Value>) -> Json<Value> {
            assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
            assert_eq!(
                body,
                json!({
                    "query": "current rust release",
                    "workflow": "search",
                    "format": "json",
                    "limit": 10,
                    "safe_search": true
                })
            );
            Json(json!({
                "meta": { "trace": "search-trace" },
                "data": {
                    "search": [{
                        "url": "https://www.rust-lang.org/",
                        "title": "Rust",
                        "snippet": "A language",
                        "time": "2026-07-01"
                    }],
                    "news": [{
                        "url": "https://blog.rust-lang.org/",
                        "title": "Rust blog"
                    }]
                }
            }))
        }

        let router = Router::new().route("/api/v1/search", post(handler));
        let (base_url, server) = test_server(router).await;
        let client =
            KagiClient::new_with_base_url("secret".to_owned(), &format!("{base_url}/api/v1"))
                .unwrap();

        let response = client.search(" current rust release ").await.unwrap();
        assert_eq!(response.meta.trace.as_deref(), Some("search-trace"));
        assert_eq!(response.data.search.len(), 1);
        assert_eq!(response.data.search[0].title, "Rust");
        assert_eq!(response.data.news.len(), 1);
        assert_eq!(response.data.news[0].snippet, None);
        server.abort();
    }

    #[tokio::test]
    async fn extract_parses_page_and_top_level_errors() {
        async fn handler(headers: HeaderMap, Json(body): Json<Value>) -> Json<Value> {
            assert_eq!(headers.get("authorization").unwrap(), "Bearer secret");
            assert_eq!(
                body,
                json!({
                    "pages": [
                        { "url": "https://example.com/one" },
                        { "url": "https://example.com/two" }
                    ],
                    "format": "json"
                })
            );
            Json(json!({
                "meta": { "trace": "extract-trace" },
                "data": [
                    {
                        "url": "https://example.com/one",
                        "markdown": "# One"
                    },
                    {
                        "url": "https://example.com/two",
                        "error": "No data returned from crawlers"
                    }
                ],
                "errors": [{
                    "code": "crawler.empty",
                    "url": "https://kagi.com/docs/errors/crawler.empty",
                    "message": "One page failed",
                    "location": "pages[1]"
                }]
            }))
        }

        let router = Router::new().route("/api/v1/extract", post(handler));
        let (base_url, server) = test_server(router).await;
        let client =
            KagiClient::new_with_base_url("secret".to_owned(), &format!("{base_url}/api/v1"))
                .unwrap();
        let urls = vec![
            "https://example.com/one".to_owned(),
            "https://example.com/two".to_owned(),
        ];

        let response = client.extract(&urls).await.unwrap();
        assert_eq!(response.meta.trace.as_deref(), Some("extract-trace"));
        assert_eq!(response.data[0].markdown.as_deref(), Some("# One"));
        assert_eq!(
            response.data[1].error.as_deref(),
            Some("No data returned from crawlers")
        );
        assert_eq!(response.errors[0].code, "crawler.empty");
        server.abort();
    }

    #[tokio::test]
    async fn api_errors_include_message_and_body_trace() {
        async fn handler() -> (AxumStatusCode, Json<Value>) {
            (
                AxumStatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "meta": { "trace": "rate-trace" },
                    "error": [{
                        "code": "rate_limit",
                        "message": "Too many requests"
                    }]
                })),
            )
        }

        let router = Router::new().route("/api/v1/search", post(handler));
        let (base_url, server) = test_server(router).await;
        let client =
            KagiClient::new_with_base_url("secret".to_owned(), &format!("{base_url}/api/v1"))
                .unwrap();

        let error = client.search("anything").await.unwrap_err();
        match error {
            KagiError::Api {
                status,
                message,
                trace_id,
                ..
            } => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(message, "rate_limit: Too many requests");
                assert_eq!(trace_id, "rate-trace");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        server.abort();
    }

    #[tokio::test]
    async fn rejects_oversized_responses_before_deserializing() {
        #[derive(Clone)]
        struct LargeBody(Arc<Vec<u8>>);

        async fn handler(State(body): State<LargeBody>) -> Response {
            Response::builder()
                .header(TRACE_HEADER, "large-trace")
                .body(Body::from(body.0.as_ref().clone()))
                .unwrap()
        }

        let body = LargeBody(Arc::new(vec![b'x'; SEARCH_RESPONSE_LIMIT_BYTES + 1]));
        let router = Router::new()
            .route("/api/v1/search", post(handler))
            .with_state(body);
        let (base_url, server) = test_server(router).await;
        let client =
            KagiClient::new_with_base_url("secret".to_owned(), &format!("{base_url}/api/v1"))
                .unwrap();

        let error = client.search("anything").await.unwrap_err();
        match error {
            KagiError::ResponseTooLarge {
                limit_bytes,
                trace_id,
                ..
            } => {
                assert_eq!(limit_bytes, SEARCH_RESPONSE_LIMIT_BYTES);
                assert_eq!(trace_id, "large-trace");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        server.abort();
    }

    #[tokio::test]
    async fn extract_rejects_invalid_urls_before_sending() {
        let client =
            KagiClient::new_with_base_url("secret".to_owned(), "http://127.0.0.1:1/api/v1")
                .unwrap();

        assert!(matches!(
            client.extract(&[]).await,
            Err(KagiError::InvalidUrlCount { count: 0 })
        ));
        assert!(matches!(
            client.extract(&["http://example.com".to_owned()]).await,
            Err(KagiError::InvalidUrl { .. })
        ));
        assert!(matches!(
            client
                .extract(&["https://user:password@example.com".to_owned()])
                .await,
            Err(KagiError::InvalidUrl { .. })
        ));
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = KagiClient::new("top-secret-value".to_owned()).unwrap();
        let debug = format!("{client:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("top-secret-value"));
    }

    #[test]
    fn nullable_optional_collections_decode_as_empty() {
        let search: SearchResponse = serde_json::from_value(json!({
            "meta": null,
            "data": { "search": null, "news": null }
        }))
        .unwrap();
        assert!(search.data.search.is_empty());
        assert!(search.data.news.is_empty());

        let extract: ExtractResponse = serde_json::from_value(json!({
            "meta": null,
            "data": null,
            "errors": null
        }))
        .unwrap();
        assert!(extract.data.is_empty());
        assert!(extract.errors.is_empty());
    }

    #[test]
    fn trace_ids_are_sanitized_and_bounded_during_deserialization() {
        let response: SearchResponse = serde_json::from_value(json!({
            "meta": { "trace": format!("trace\n{}", "x".repeat(200)) },
            "data": {}
        }))
        .unwrap();
        let trace = response.meta.trace.unwrap();
        assert!(!trace.contains('\n'));
        assert!(trace.chars().count() <= TRACE_ID_LIMIT_CHARS);
        assert_eq!(sanitize_trace_id("\n\t"), "unavailable");
    }

    #[test]
    fn api_error_display_does_not_expose_response_message() {
        let error = KagiError::Api {
            operation: "search",
            status: StatusCode::BAD_REQUEST,
            message: "private query echoed by provider".to_string(),
            trace_id: "safe-trace".to_string(),
        };
        let display = error.to_string();
        assert!(display.contains("HTTP 400"));
        assert!(display.contains("safe-trace"));
        assert!(!display.contains("private query"));
    }
}
