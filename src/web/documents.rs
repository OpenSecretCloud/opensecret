use crate::models::users::User;
use crate::web::encryption_middleware::{encrypt_response, EncryptedResponse};
use crate::{ApiError, AppState};
use axum::{
    extract::{Multipart, State},
    http::HeaderMap,
    routing::post,
    Json, Router,
};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error};
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
pub struct DocumentUploadResponse {
    pub text: String,
    pub filename: String,
    pub size: i64,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/v1/documents/upload", post(upload_document))
        .with_state(app_state)
}

async fn upload_document(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    mut multipart: Multipart,
) -> Result<Json<EncryptedResponse<DocumentUploadResponse>>, ApiError> {
    debug!("Entering upload_document function");

    // Prevent guest users from using the document upload feature
    if user.is_guest() {
        error!(
            "Guest user attempted to use document upload feature: {}",
            user.uuid
        );
        return Err(ApiError::Unauthorized);
    }

    // Check billing if client exists
    if let Some(billing_client) = &state.billing_client {
        debug!("Checking billing server for user {}", user.uuid);
        match billing_client.can_user_chat(user.uuid).await {
            Ok(true) => {
                debug!("Billing service passed for user {}", user.uuid);
            }
            Ok(false) => {
                error!("Usage limit reached for user: {}", user.uuid);
                return Err(ApiError::UsageLimitReached);
            }
            Err(e) => {
                error!("Billing service error, allowing request: {}", e);
            }
        }
    }

    // Get the file from multipart
    let mut file_data = Vec::new();
    let mut filename = String::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        error!("Failed to get multipart field: {:?}", e);
        ApiError::BadRequest
    })? {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            filename = field.file_name().unwrap_or("unknown").to_string();
            let data = field.bytes().await.map_err(|e| {
                error!("Failed to read file data: {:?}", e);
                ApiError::BadRequest
            })?;
            file_data = data.to_vec();
        }
    }

    if file_data.is_empty() {
        error!("No file data received");
        return Err(ApiError::BadRequest);
    }

    // Check file size (10MB limit)
    if file_data.len() > 10 * 1024 * 1024 {
        error!("File size exceeds 10MB limit");
        return Err(ApiError::BadRequest);
    }

    let _file_size = file_data.len() as i64;

    // Create multipart form data for tinfoil proxy
    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let mut body = Vec::new();

    // Add file field
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
            filename
        )
        .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(&file_data);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    // Get the Tinfoil API base URL from the proxy router
    let tinfoil_api_base = state.proxy_router.get_tinfoil_base_url().ok_or_else(|| {
        error!("Tinfoil API base not configured");
        ApiError::InternalServerError
    })?;

    // Create a new hyper client
    let https = HttpsConnector::new();
    let client = Client::builder().build::<_, Body>(https);

    // Prepare the request to tinfoil proxy
    let req = Request::builder()
        .method("POST")
        .uri(format!("{}/v1/documents/upload", tinfoil_api_base))
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body))
        .map_err(|e| {
            error!("Failed to create request body: {:?}", e);
            ApiError::InternalServerError
        })?;

    debug!("Sending document to Tinfoil proxy");

    // Send the request to Tinfoil proxy
    let res = client.request(req).await.map_err(|e| {
        error!("Failed to send request to Tinfoil proxy: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Check if the response is successful
    if !res.status().is_success() {
        error!(
            "Tinfoil proxy returned non-success status: {}",
            res.status()
        );
        return Err(ApiError::InternalServerError);
    }

    // Read the response body
    let body_bytes = hyper::body::to_bytes(res.into_body()).await.map_err(|e| {
        error!("Failed to read response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Parse the response
    let upload_response: DocumentUploadResponse =
        serde_json::from_slice(&body_bytes).map_err(|e| {
            error!("Failed to parse response: {:?}", e);
            ApiError::InternalServerError
        })?;

    debug!("Document processed successfully");

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &upload_response).await
}

