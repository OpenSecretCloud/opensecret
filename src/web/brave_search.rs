use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info};
use url::form_urlencoded;

use crate::{
    AppState,
    proxy_config::ProxyConfig,
};

#[derive(Debug, Deserialize)]
pub struct BraveSearchQuery {
    pub q: String,
    pub count: Option<i32>,
    pub offset: Option<i32>,
    pub safesearch: Option<String>,
    pub freshness: Option<String>,
    pub text_decorations: Option<bool>,
    pub spellcheck: Option<bool>,
    pub result_filter: Option<String>,
    pub goggles_id: Option<String>,
    pub units: Option<String>,
    pub country: Option<String>,
    pub search_lang: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct BraveSearchError {
    pub error: String,
    pub message: String,
}

pub async fn brave_search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<BraveSearchQuery>,
) -> Result<Response, StatusCode> {
    debug!("Brave Search request: q={}", params.q);

    // Get Brave Search configuration
    let brave_config = state
        .proxy_router
        .get_brave_search_config()
        .ok_or_else(|| {
            error!("Brave Search API key not configured");
            StatusCode::SERVICE_UNAVAILABLE
        })?;

    // Build the search request
    let search_response = perform_brave_search(&brave_config, params).await?;

    Ok(Json(search_response).into_response())
}

async fn perform_brave_search(
    config: &ProxyConfig,
    params: BraveSearchQuery,
) -> Result<Value, StatusCode> {
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, Body>(https);

    // Build query parameters
    let mut query_params = vec![("q", params.q)];
    
    if let Some(count) = params.count {
        query_params.push(("count", count.to_string()));
    }
    if let Some(offset) = params.offset {
        query_params.push(("offset", offset.to_string()));
    }
    if let Some(safesearch) = params.safesearch {
        query_params.push(("safesearch", safesearch));
    }
    if let Some(freshness) = params.freshness {
        query_params.push(("freshness", freshness));
    }
    if let Some(text_decorations) = params.text_decorations {
        query_params.push(("text_decorations", text_decorations.to_string()));
    }
    if let Some(spellcheck) = params.spellcheck {
        query_params.push(("spellcheck", spellcheck.to_string()));
    }
    if let Some(result_filter) = params.result_filter {
        query_params.push(("result_filter", result_filter));
    }
    if let Some(goggles_id) = params.goggles_id {
        query_params.push(("goggles_id", goggles_id));
    }
    if let Some(units) = params.units {
        query_params.push(("units", units));
    }
    if let Some(country) = params.country {
        query_params.push(("country", country));
    }
    if let Some(search_lang) = params.search_lang {
        query_params.push(("search_lang", search_lang));
    }

    // Add any extra parameters from the flattened map
    for (key, value) in params.extra.iter() {
        if let Some(s) = value.as_str() {
            query_params.push((key.as_str(), s.to_string()));
        } else {
            query_params.push((key.as_str(), value.to_string()));
        }
    }

    // Build the URL with query parameters using form_urlencoded
    let query_string = form_urlencoded::Serializer::new(String::new())
        .extend_pairs(query_params.iter())
        .finish();

    let url = format!("{}/res/v1/web/search?{}", config.base_url, query_string);
    debug!("Brave Search URL: {}", url);

    // Build the request
    let mut req = Request::builder()
        .method("GET")
        .uri(&url)
        .header("Accept", "application/json")
        .header("Accept-Encoding", "gzip");

    // Add authentication header
    if let Some(api_key) = &config.api_key {
        req = req.header("X-Subscription-Token", api_key);
    }

    let req = req.body(Body::empty()).map_err(|e| {
        error!("Failed to build request: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Execute the request
    let res = client.request(req).await.map_err(|e| {
        error!("Brave Search request failed: {:?}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let status = res.status();
    let body_bytes = hyper::body::to_bytes(res.into_body())
        .await
        .map_err(|e| {
            error!("Failed to read response body: {:?}", e);
            StatusCode::BAD_GATEWAY
        })?;

    if !status.is_success() {
        let error_text = String::from_utf8_lossy(&body_bytes);
        error!("Brave Search API error: {} - {}", status, error_text);
        
        // Try to parse as JSON error response
        if let Ok(error_json) = serde_json::from_slice::<Value>(&body_bytes) {
            return Ok(error_json);
        }
        
        // Return a structured error response
        let error_response = BraveSearchError {
            error: "brave_search_error".to_string(),
            message: format!("Brave Search API returned {}: {}", status, error_text),
        };
        
        return Ok(serde_json::to_value(error_response).unwrap());
    }

    // Parse the successful response
    let response: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
        error!("Failed to parse Brave Search response: {:?}", e);
        StatusCode::BAD_GATEWAY
    })?;

    info!("Brave Search completed successfully");
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brave_search_query_deserialize() {
        let json = r#"{
            "q": "test query",
            "count": 10,
            "safesearch": "moderate",
            "extra_param": "value"
        }"#;

        let query: BraveSearchQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.q, "test query");
        assert_eq!(query.count, Some(10));
        assert_eq!(query.safesearch, Some("moderate".to_string()));
        assert_eq!(query.extra.get("extra_param").unwrap(), "value");
    }
}