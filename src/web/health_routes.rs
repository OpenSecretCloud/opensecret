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

    // Try to fetch models directly from the proxy with a timeout
    let timeout_duration = Duration::from_secs(5);

    // We'll test against the default proxy (Continuum)
    let default_proxy = state.proxy_router.get_default_proxy();

    let result = timeout(
        timeout_duration,
        fetch_models_directly(&client, &default_proxy),
    )
    .await;

    match result {
        Ok(Ok(model_count)) => Ok(Json(ExtendedHealthResponse {
            status: "pass".to_string(),
            version: API_VERSION.to_string(),
            outbound_connectivity: true,
            model_check: Some(format!(
                "Successfully fetched {} models from {}",
                model_count, default_proxy.provider_name
            )),
            error: None,
        })),
        Ok(Err(e)) => {
            // Failed to fetch models
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Failed to fetch models from {}: {}",
                    default_proxy.provider_name, e
                ),
            ))
        }
        Err(_) => {
            // Timeout occurred
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Model fetch from {} timed out after 5 seconds",
                    default_proxy.provider_name
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
        let body_bytes = hyper::body::to_bytes(res.into_body()).await?;
        let body_str = String::from_utf8_lossy(&body_bytes);
        return Err(format!("HTTP {}: {}", status, body_str).into());
    }

    let body_bytes = hyper::body::to_bytes(res.into_body()).await?;
    let models_response: serde_json::Value = serde_json::from_slice(&body_bytes)?;

    // Count the models
    let model_count = models_response
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

    Ok(model_count)
}
