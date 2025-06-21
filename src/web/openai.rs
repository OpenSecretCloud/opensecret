use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::sqs::UsageEvent;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::{ApiError, AppState};
use axum::http::{header, HeaderMap};
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
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
use tokio::time::sleep;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/v1/chat/completions", post(proxy_openai))
        .route("/v1/models", get(proxy_models))
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

    // Extract the model from the request - error if not specified
    let model_name = modified_body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?;

    // Get the appropriate proxy configuration for this model
    let proxy_config = state.proxy_router.get_proxy_for_model(model_name).await;

    let modified_body = Value::Object(modified_body);

    // Use the proxy configuration for this specific model
    let openai_api_key = proxy_config.api_key.as_deref().unwrap_or("");
    let openai_api_base = &proxy_config.base_url;

    // Create a new hyper client with better timeout configuration
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, Body>(https);

    // Prepare the request to OpenAI
    let body_json = serde_json::to_string(&modified_body).map_err(|e| {
        error!("Failed to serialize request body: {:?}", e);
        ApiError::InternalServerError
    })?;

    debug!("Sending request to OpenAI");

    // Retry logic: 3 attempts with exponential backoff
    let mut res = None;
    let max_attempts = 3;

    for attempt in 0..max_attempts {
        if attempt > 0 {
            let delay = attempt; // 1s after 1st failure, 2s after 2nd failure
            debug!(
                "Retrying request (attempt {} of {}) after {}s delay",
                attempt + 1,
                max_attempts,
                delay
            );
            sleep(Duration::from_secs(delay)).await;
        } else {
            debug!("Making initial request (attempt 1 of {})", max_attempts);
        }

        // Build new request for each attempt
        let mut req = Request::builder()
            .method("POST")
            .uri(format!("{}/v1/chat/completions", openai_api_base))
            .header("Content-Type", "application/json");

        if !openai_api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", openai_api_key));
        }

        // Forward relevant headers from the original request
        for (key, value) in headers.iter() {
            if key != header::HOST && key != header::AUTHORIZATION && key != header::CONTENT_LENGTH
            {
                if let (Ok(name), Ok(val)) = (
                    HeaderName::from_bytes(key.as_ref()),
                    HeaderValue::from_str(value.to_str().unwrap_or_default()),
                ) {
                    req = req.header(name, val);
                }
            }
        }

        let req = req.body(Body::from(body_json.clone())).map_err(|e| {
            error!("Failed to create request body: {:?}", e);
            ApiError::InternalServerError
        })?;

        match client.request(req).await {
            Ok(response) => {
                if response.status().is_success() {
                    res = Some(response);
                    break;
                } else if attempt == max_attempts - 1 {
                    // Only log details on last attempt
                    error!(
                        "OpenAI API returned non-success status: {}",
                        response.status()
                    );
                    debug!("Response headers: {:?}", response.headers());
                    let body_bytes = to_bytes(response.into_body()).await.map_err(|e| {
                        error!("Failed to read response body: {:?}", e);
                        ApiError::InternalServerError
                    })?;
                    let body_str = String::from_utf8_lossy(&body_bytes);
                    error!("Response body: {}", body_str);
                }
            }
            Err(e) => {
                if attempt == max_attempts - 1 {
                    error!("Failed to send request to OpenAI: {:?}", e);
                } else {
                    debug!("Request failed on attempt {}: {:?}", attempt + 1, e);
                }
            }
        }
    }

    let res = res.ok_or_else(|| {
        error!("All retry attempts failed");
        ApiError::InternalServerError
    })?;

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
                        error!(
                            "Error reading response body: {:?}. Current buffer: {}",
                            e,
                            buffer.lock().unwrap()
                        );
                        vec![Ok(Event::default().data("Error reading response"))]
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

async fn proxy_models(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    debug!("Entering proxy_models function");

    // Prevent guest users from using the models endpoint
    if user.is_guest() {
        error!(
            "Guest user attempted to access models endpoint: {}",
            user.uuid
        );
        return Err(ApiError::Unauthorized);
    }

    // Use the proxy router to get all models from all configured proxies
    // The proxy router now handles caching internally with a 5-minute TTL
    let models_response = state.proxy_router.get_all_models().await.map_err(|e| {
        error!("Failed to fetch models from proxy router: {:?}", e);
        ApiError::InternalServerError
    })?;

    debug!("Exiting proxy_models function");
    // Encrypt and return the response
    encrypt_response(&state, &session_id, &models_response).await
}
