use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::{ApiError, AppState};
use axum::middleware::from_fn_with_state;
use axum::{extract::State, routing::post, Extension, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

use crate::models::users::User;

// Import Kagi SDK types
use kagi_api_rust::{
    apis::{
        configuration::{ApiKey, Configuration},
        search_api,
    },
    models::SearchRequest as KagiSearchRequest,
};

#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub workflow: Option<String>, // Options: search, images, videos, news, podcasts
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/v1/search",
            post(search_handler).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<SearchRequest>,
            )),
        )
        .with_state(app_state.clone())
}

async fn search_handler(
    State(app_state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(search_request): Extension<SearchRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<SearchResponse>>, ApiError> {
    debug!("Entering search_handler function");
    info!("Search request from user {}", user.uuid);
    tracing::trace!(
        "Search query: {:?}, workflow: {:?}",
        search_request.query,
        search_request.workflow
    );

    // Check if Kagi API key is available
    let kagi_api_key = match &app_state.kagi_api_key {
        Some(key) => key,
        None => {
            error!("Kagi API key not configured");
            let response = SearchResponse {
                success: false,
                data: None,
                error: Some("Search service not configured".to_string()),
            };
            return encrypt_response(&app_state, &session_id, &response).await;
        }
    };

    // Configure Kagi API client
    let configuration = Configuration {
        base_path: "https://kagi.com/api/v1".to_string(),
        api_key: Some(ApiKey {
            prefix: None,
            key: kagi_api_key.clone(),
        }),
        ..Default::default()
    };

    // Convert workflow string to enum if provided
    let workflow = search_request
        .workflow
        .as_ref()
        .and_then(|w| match w.as_str() {
            "search" => Some(kagi_api_rust::models::search_request::Workflow::Search),
            "images" => Some(kagi_api_rust::models::search_request::Workflow::Images),
            "videos" => Some(kagi_api_rust::models::search_request::Workflow::Videos),
            "news" => Some(kagi_api_rust::models::search_request::Workflow::News),
            "podcasts" => Some(kagi_api_rust::models::search_request::Workflow::Podcasts),
            _ => None,
        });

    // Create Kagi search request
    let kagi_request = KagiSearchRequest {
        query: search_request.query.clone(),
        workflow,
    };

    // Call Kagi API
    match search_api::search(&configuration, kagi_request).await {
        Ok(kagi_response) => {
            info!("Kagi search successful for user {}", user.uuid);

            // Convert Kagi response to JSON value
            let data = serde_json::to_value(kagi_response).map_err(|e| {
                error!("Failed to serialize Kagi response: {:?}", e);
                ApiError::InternalServerError
            })?;

            let response = SearchResponse {
                success: true,
                data: Some(data),
                error: None,
            };

            trace!("Search response data: {:?}", response);
            debug!("Exiting search_handler function - success");
            encrypt_response(&app_state, &session_id, &response).await
        }
        Err(e) => {
            error!("Kagi API error: {:?}", e);
            let response = SearchResponse {
                success: false,
                data: None,
                error: Some(format!("Search failed: {:?}", e)),
            };

            debug!("Exiting search_handler function - error");
            encrypt_response(&app_state, &session_id, &response).await
        }
    }
}
