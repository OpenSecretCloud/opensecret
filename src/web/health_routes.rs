use super::openai::{
    read_provider_body_limited, safe_log_preview, MAX_MODELS_RESPONSE_BYTES,
    MAX_PROVIDER_ERROR_BODY_BYTES,
};
use crate::AppState;
use axum::{http::StatusCode, Router};
use axum::{routing::get, Json};
use serde::Serialize;
use std::sync::Arc;

const API_VERSION: &str = "v1";

pub fn router_with_state(state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/health-check", get(health_check))
        .route("/health-check-extended", get(health_check_extended))
        .with_state(state)
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Serialize)]
pub struct ExtendedHealthResponse {
    pub status: String,
    pub version: String,
    pub outbound_connectivity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_check: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl HealthResponse {
    /// Fabricate a status: pass response without checking database connectivity
    pub fn new_ok() -> Self {
        Self {
            status: String::from("pass"),
            version: String::from(API_VERSION),
        }
    }
}

/// Health check endpoint following the IETF draft standard
/// <https://datatracker.ietf.org/doc/html/draft-inadarei-api-health-check>
pub async fn health_check() -> Result<Json<HealthResponse>, (StatusCode, String)> {
    Ok(Json(HealthResponse::new_ok()))
}

/// Extended health check that tests outbound connectivity via model listing
pub async fn health_check_extended(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<Json<ExtendedHealthResponse>, (StatusCode, String)> {
    use hyper::{Body, Client};
    use hyper_tls::HttpsConnector;
    use std::time::Duration;
    use tokio::time::timeout;

    // Create a fresh HTTP client to test actual connectivity
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(15))
        .build::<_, Body>(https);

    // Try to fetch models directly from the tinfoil proxy with a timeout
    let timeout_duration = Duration::from_secs(5);

    let tinfoil_proxy = state.proxy_router.get_tinfoil_proxy();

    let result = timeout(
        timeout_duration,
        fetch_models_directly(&client, &tinfoil_proxy),
    )
    .await;

    match result {
        Ok(Ok(model_count)) => Ok(Json(ExtendedHealthResponse {
            status: "pass".to_string(),
            version: API_VERSION.to_string(),
            outbound_connectivity: true,
            model_check: Some(format!(
                "Successfully fetched {} models from {}",
                model_count, tinfoil_proxy.provider_name
            )),
            error: None,
        })),
        Ok(Err(e)) => {
            // Failed to fetch models
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Failed to fetch models from {}: {}",
                    tinfoil_proxy.provider_name, e
                ),
            ))
        }
        Err(_) => {
            // Timeout occurred
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Model fetch from {} timed out after 5 seconds",
                    tinfoil_proxy.provider_name
                ),
            ))
        }
    }
}

/// Helper function to fetch models directly without caching
async fn fetch_models_directly(
    client: &hyper::Client<hyper_tls::HttpsConnector<hyper::client::HttpConnector>>,
    proxy_config: &crate::proxy_config::ProxyConfig,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    use hyper::{Body, Request};

    let mut req = Request::builder()
        .method("GET")
        .uri(format!("{}/v1/models", proxy_config.base_url));

    if let Some(api_key) = &proxy_config.api_key {
        if !api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }
    }

    let req = req.body(Body::empty())?;
    let res = client.request(req).await?;

    if !res.status().is_success() {
        let status = res.status();
        let body_excerpt = read_provider_error_excerpt(res.into_body()).await?;
        return Err(format!("HTTP {}: {}", status, body_excerpt).into());
    }

    count_models_in_body(res.into_body()).await
}

async fn count_models_in_body(
    body: hyper::Body,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    count_models_in_body_with_limit(body, MAX_MODELS_RESPONSE_BYTES).await
}

async fn count_models_in_body_with_limit(
    body: hyper::Body,
    limit: usize,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let body_bytes = read_provider_body_limited(body, limit).await?;
    let models_response: serde_json::Value = serde_json::from_slice(&body_bytes)?;

    // Count the models
    let model_count = models_response
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

    Ok(model_count)
}

async fn read_provider_error_excerpt(
    body: hyper::Body,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    read_provider_error_excerpt_with_limit(body, MAX_PROVIDER_ERROR_BODY_BYTES).await
}

async fn read_provider_error_excerpt_with_limit(
    body: hyper::Body,
    limit: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let body_bytes = read_provider_body_limited(body, limit).await?;
    Ok(safe_log_preview(&String::from_utf8_lossy(&body_bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn body_from_chunks(chunks: Vec<Bytes>) -> hyper::Body {
        hyper::Body::wrap_stream(futures::stream::iter(
            chunks.into_iter().map(Result::<Bytes, std::io::Error>::Ok),
        ))
    }

    #[tokio::test]
    async fn chunked_models_response_is_counted_within_limit() {
        let body = body_from_chunks(vec![
            Bytes::from_static(br#"{"data":[{"id":"one"},"#),
            Bytes::from_static(br#"{"id":"two"}]}"#),
        ]);

        let model_count = count_models_in_body_with_limit(body, 64)
            .await
            .expect("a chunked models response within the limit should be accepted");

        assert_eq!(model_count, 2);
    }

    #[tokio::test]
    async fn chunked_models_response_is_rejected_over_limit() {
        let body = body_from_chunks(vec![
            Bytes::from_static(b"abcd"),
            Bytes::from_static(b"efghi"),
        ]);

        let error = count_models_in_body_with_limit(body, 8)
            .await
            .expect_err("a models response over the aggregate limit should be rejected");

        assert_eq!(
            error.to_string(),
            "provider response body exceeded the 8-byte limit"
        );
    }

    #[tokio::test]
    async fn chunked_provider_error_is_rejected_over_limit() {
        let body = body_from_chunks(vec![
            Bytes::from_static(b"abcd"),
            Bytes::from_static(b"efghi"),
        ]);

        let error = read_provider_error_excerpt_with_limit(body, 8)
            .await
            .expect_err("an error response over the aggregate limit should be rejected");

        assert_eq!(
            error.to_string(),
            "provider response body exceeded the 8-byte limit"
        );
    }

    #[tokio::test]
    async fn chunked_provider_error_is_reduced_to_a_safe_excerpt() {
        let body = body_from_chunks(vec![
            Bytes::from(vec![b'a'; 100]),
            Bytes::from(vec![b'b'; 51]),
            Bytes::from_static(b"\nignored"),
        ]);

        let excerpt = read_provider_error_excerpt_with_limit(body, 256)
            .await
            .expect("a bounded error response should produce an excerpt");

        assert_eq!(excerpt.chars().count(), 153);
        assert!(excerpt.ends_with("..."));
        assert!(!excerpt.contains('\n'));
    }
}
