use crate::is_default_openai_domain;
use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::sqs::UsageEvent;
use crate::web::encryption_middleware::decrypt_request;
use crate::{ApiError, AppState};
use axum::http::{header, HeaderMap};
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::post,
    Router,
};
use base64::{engine::general_purpose, Engine as _};
use bigdecimal::BigDecimal;
use chrono::Utc;
use futures::stream::{self, Stream, StreamExt};
use futures::TryStreamExt;
use hyper::body::to_bytes;
use hyper::header::{HeaderName, HeaderValue};
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/v1/chat/completions", post(proxy_openai))
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            decrypt_request::<Value>,
        ))
        .with_state(app_state)
}

async fn proxy_openai(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(body): axum::Extension<Value>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    debug!("Entering proxy_openai function");

    // Prevent guest users from using the OpenAI chat feature
    if user.is_guest() {
        error!(
            "Guest user attempted to use OpenAI chat feature: {}",
            user.uuid
        );
        return Err(ApiError::Unauthorized);
    }

    // Check billing if client exists
    if let Some(billing_client) = &state.billing_client {
        debug!("Checking billing server for user {}", user.uuid);
        match billing_client.can_user_chat(user.uuid).await {
            Ok(true) => {
                // User can chat, proceed with existing logic
                debug!("Billing service passed for user {}", user.uuid);
            }
            Ok(false) => {
                error!("Usage limit reached for user: {}", user.uuid);
                return Err(ApiError::UsageLimitReached);
            }
            Err(e) => {
                // Log the error but allow the request
                error!("Billing service error, allowing request: {}", e);
            }
        }
    }

    if body.is_null() || body.as_object().map_or(true, |obj| obj.is_empty()) {
        error!("Request body is empty or invalid");
        return Err(ApiError::BadRequest);
    }

    // We already verified it's a valid object above, so this expect should never trigger
    let mut modified_body = body.as_object().expect("body was just checked").clone();
    modified_body.insert("stream_options".to_string(), json!({"include_usage": true}));
    let modified_body = Value::Object(modified_body);

    // Use the OpenAI API key and base URL from AppState
    let openai_api_key = match &state.openai_api_key {
        Some(key) if !key.is_empty() => key,
        _ => {
            if is_default_openai_domain(&state.openai_api_base) {
                error!("OpenAI API key is required for OpenAI domain");
                return Err(ApiError::InternalServerError);
            }
            "" // Empty string if not using OpenAI's domain
        }
    };
    let openai_api_base = &state.openai_api_base;

    // Create a new hyper client with extended timeouts
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(60)) // Increased from 15 to 60 seconds
        .pool_max_idle_per_host(32) // Increased connection pool size
        .http2_keep_alive_interval(Some(Duration::from_secs(20))) // Keep connections alive
        .http2_keep_alive_timeout(Duration::from_secs(10)) // Timeout for keep-alive pings
        .http1_read_buf_exact_size(65_536) // Increased buffer size
        .http1_title_case_headers(true) // Improved compatibility
        .build::<_, Body>(https);

    // Prepare the request to OpenAI
    let body_json = serde_json::to_string(&modified_body).map_err(|e| {
        error!("Failed to serialize request body: {:?}", e);
        ApiError::InternalServerError
    })?;

    let mut req = Request::builder()
        .method("POST")
        .uri(format!("{}/v1/chat/completions", openai_api_base))
        .header("Content-Type", "application/json");

    if !openai_api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", openai_api_key));
    }

    // Forward relevant headers from the original request
    for (key, value) in headers.iter() {
        if key != header::HOST && key != header::AUTHORIZATION && key != header::CONTENT_LENGTH {
            if let (Ok(name), Ok(val)) = (
                HeaderName::from_bytes(key.as_ref()),
                HeaderValue::from_str(value.to_str().unwrap_or_default()),
            ) {
                req = req.header(name, val);
            }
        }
    }

    let req = req.body(Body::from(body_json)).map_err(|e| {
        error!("Failed to create request body: {:?}", e);
        ApiError::InternalServerError
    })?;

    debug!("Sending request to OpenAI");
    // Send the request to OpenAI with retry logic
    let mut attempts = 0;
    let max_attempts = 3;
    let mut _last_error = None;

    // We need to manually clone the request since hyper::Request doesn't have try_clone
    // Extract the components we need for retrying
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let headers = parts.headers.clone();
    let version = parts.version;

    // Convert body to bytes so we can reuse it
    let body_bytes = match hyper::body::to_bytes(body).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read request body: {:?}", e);
            return Err(ApiError::InternalServerError);
        }
    };

    let res = loop {
        attempts += 1;

        // Rebuild the request from the saved components
        let mut req_builder = hyper::Request::builder()
            .method(method.clone())
            .uri(uri.clone())
            .version(version);

        // Add all headers
        for (name, value) in headers.iter() {
            req_builder = req_builder.header(name, value);
        }

        // Build the request with the body
        let request = req_builder
            .body(hyper::Body::from(body_bytes.clone()))
            .map_err(|e| {
                error!("Failed to build request: {:?}", e);
                ApiError::InternalServerError
            })?;

        match client.request(request).await {
            Ok(response) => break response,
            Err(e) => {
                error!(
                    "Attempt {}/{}: Failed to send request to OpenAI: {:?}",
                    attempts, max_attempts, e
                );
                _last_error = Some(e);

                // If we've reached max attempts, return the error
                if attempts >= max_attempts {
                    error!("Max retry attempts reached. Giving up.");
                    return Err(ApiError::InternalServerError);
                }

                // Wait before retrying (exponential backoff with max delay cap of 3 seconds)
                let backoff_ms = 500 * 2_u64.pow(attempts as u32 - 1);
                let capped_backoff_ms = std::cmp::min(backoff_ms, 3000); // Cap at 3 seconds
                let delay = std::time::Duration::from_millis(capped_backoff_ms);
                tokio::time::sleep(delay).await;
            }
        }
    };

    // Check if the response is successful
    if !res.status().is_success() {
        error!("OpenAI API returned non-success status: {}", res.status());

        // Log headers
        debug!("Response headers: {:?}", res.headers());

        // Read and log the response body
        let body_bytes = to_bytes(res.into_body()).await.map_err(|e| {
            error!("Failed to read response body: {:?}", e);
            ApiError::InternalServerError
        })?;

        let body_str = String::from_utf8_lossy(&body_bytes);
        error!("Response body: {}", body_str);

        return Err(ApiError::InternalServerError);
    }

    debug!("Successfully received response from OpenAI");

    let stream = res.into_body().into_stream();
    let buffer = Arc::new(Mutex::new(String::new()));
    let stream = stream
        .map(move |chunk| {
            let state = state.clone();
            let session_id = session_id;
            let user = user.clone();
            let buffer = buffer.clone();
            async move {
                match chunk {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        let mut events = Vec::new();
                        {
                            let mut buffer = buffer.lock().unwrap();
                            buffer.push_str(&chunk_str);
                            while let Some(event_end) = buffer.find("\n\n") {
                                let event = buffer[..event_end].to_string();
                                *buffer = buffer[event_end + 2..].to_string();
                                events.push(event);
                            }
                            if events.is_empty() {
                                trace!("No complete events in buffer. Current buffer: {}", buffer);
                            }
                        }

                        let mut processed_events = Vec::new();
                        for event in events {
                            if let Some(processed_event) =
                                encrypt_and_process_event(&state, &session_id, &user, &event).await
                            {
                                processed_events.push(Ok(processed_event));
                            }
                        }
                        processed_events
                    }
                    Err(e) => {
                        // Log the error with more details to help debug
                        error!(
                            "Error reading response body: {:?}. Current buffer: {}",
                            e,
                            buffer.lock().unwrap()
                        );

                        // Return a more specific error message that can help with debugging
                        let error_msg = format!("Error reading response: {:?}", e);

                        // Send an error event to the client
                        vec![Ok(Event::default().data(error_msg))]
                    }
                }
            }
        })
        .flat_map(stream::once)
        .flat_map(stream::iter);

    debug!("Exiting proxy_openai function");
    Ok(Sse::new(stream))
}

async fn encrypt_and_process_event(
    state: &AppState,
    session_id: &Uuid,
    user: &User,
    event: &str,
) -> Option<Event> {
    if event.trim() == "data: [DONE]" {
        return Some(Event::default().data("[DONE]"));
    }

    if let Some(data) = event.strip_prefix("data: ") {
        match serde_json::from_str::<Value>(data) {
            Ok(json) => {
                // Handle usage statistics if available
                if let Some(usage) = json.get("usage") {
                    if !usage.is_null() && usage.is_object() {
                        let input_tokens = usage
                            .get("prompt_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0) as i32;
                        let output_tokens = usage
                            .get("completion_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0) as i32;

                        // Calculate estimated cost with correct pricing
                        let input_cost = BigDecimal::from_str("0.0000053").unwrap()
                            * BigDecimal::from(input_tokens);
                        let output_cost = BigDecimal::from_str("0.0000053").unwrap()
                            * BigDecimal::from(output_tokens);
                        let total_cost = input_cost + output_cost;

                        info!(
                            "OpenAI API usage for user {}: prompt_tokens={}, completion_tokens={}, total_tokens={}, estimated_cost={}",
                            user.uuid, input_tokens, output_tokens,
                            input_tokens + output_tokens,
                            total_cost
                        );

                        // Create token usage record and post to SQS in the background
                        let state = state.clone();
                        let user_id = user.uuid;
                        tokio::spawn(async move {
                            // Create and store token usage record
                            let new_usage = NewTokenUsage::new(
                                user_id,
                                input_tokens,
                                output_tokens,
                                total_cost.clone(),
                            );

                            if let Err(e) = state.db.create_token_usage(new_usage) {
                                error!("Failed to save token usage: {:?}", e);
                            }

                            // Post event to SQS if configured
                            if let Some(publisher) = &state.sqs_publisher {
                                let event = UsageEvent {
                                    event_id: Uuid::new_v4(), // Generate new UUID for idempotency
                                    user_id,
                                    input_tokens,
                                    output_tokens,
                                    estimated_cost: total_cost,
                                    chat_time: Utc::now(),
                                };

                                match publisher.publish_event(event).await {
                                    Ok(_) => debug!("published usage event successfully"),
                                    Err(e) => error!("error publishing usage event: {e}"),
                                }
                            }
                        });
                    }
                }

                let json_str = json.to_string();
                match state
                    .encrypt_session_data(session_id, json_str.as_bytes())
                    .await
                {
                    Ok(encrypted_data) => {
                        let base64_encrypted = general_purpose::STANDARD.encode(&encrypted_data);
                        Some(process_event(&base64_encrypted))
                    }
                    Err(e) => {
                        error!("Failed to encrypt event data: {:?}", e);
                        Some(Event::default().data("Error: Encryption failed"))
                    }
                }
            }
            Err(e) => {
                error!("Received non-JSON data event. Error: {:?}", e);
                Some(Event::default().data("Error: Invalid JSON"))
            }
        }
    } else {
        error!("Received non-data event");
        Some(Event::default().data("Error: Invalid event format"))
    }
}

fn process_event(data: &str) -> Event {
    Event::default().data(data)
}
