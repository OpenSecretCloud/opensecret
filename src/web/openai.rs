use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::proxy_config::ProxyConfig;
use crate::sqs::UsageEvent;
use crate::web::audio_utils::{merge_transcriptions, AudioSplitter, TINFOIL_MAX_SIZE};
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use axum::http::{header, HeaderMap};
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use bigdecimal::BigDecimal;
use chrono::Utc;
use futures::stream::{self, StreamExt};
use futures::TryStreamExt;
use hyper::body::to_bytes;
use hyper::header::{HeaderName, HeaderValue};
use hyper::{Body as HyperBody, Client, Request};
use hyper_tls::HttpsConnector;
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, trace};
use uuid::Uuid;

// Maximum audio file size (100MB) - sanity check, CF already limits to 50MB
const MAX_AUDIO_SIZE: usize = 100 * 1024 * 1024;

/// Parameters for transcription requests
struct TranscriptionParams<'a> {
    audio_data: &'a [u8],
    filename: &'a str,
    content_type: &'a str,
    language: Option<&'a str>,
    prompt: Option<&'a str>,
    response_format: &'a str,
    temperature: Option<f64>,
}

/// Request structure for TTS (Text-to-Speech) endpoints
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct TTSRequest {
    input: String,
    #[serde(default = "default_tts_model")]
    model: String,
    #[serde(default = "default_tts_voice")]
    voice: String,
    #[serde(default = "default_tts_response_format")]
    response_format: String,
    #[serde(default = "default_tts_speed")]
    speed: f32,
}

fn default_tts_model() -> String {
    "kokoro".to_string()
}

fn default_tts_voice() -> String {
    "af_sky".to_string()
}

fn default_tts_response_format() -> String {
    "mp3".to_string()
}

fn default_tts_speed() -> f32 {
    1.0
}

/// Request structure for transcription endpoints
#[derive(Debug, Clone, serde::Deserialize)]
struct TranscriptionRequest {
    file: String, // Base64 encoded audio file
    #[serde(default = "default_transcription_filename")]
    filename: String,
    #[serde(default = "default_transcription_content_type")]
    content_type: String,
    #[serde(default = "default_transcription_model")]
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    #[serde(default = "default_transcription_response_format")]
    response_format: String,
    temperature: Option<f64>,
}

fn default_transcription_filename() -> String {
    "audio.mp3".to_string()
}

fn default_transcription_content_type() -> String {
    "audio/mpeg".to_string()
}

fn default_transcription_model() -> String {
    "whisper-large-v3".to_string()
}

fn default_transcription_response_format() -> String {
    "json".to_string()
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(proxy_openai).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<Value>,
            )),
        )
        .route(
            "/v1/models",
            get(proxy_models).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<()>,
            )),
        )
        .route(
            "/v1/audio/speech",
            post(proxy_tts).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<TTSRequest>,
            )),
        )
        .route(
            "/v1/audio/transcriptions",
            post(proxy_transcription).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<TranscriptionRequest>,
            )),
        )
        .with_state(app_state)
}

async fn proxy_openai(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(auth_method): axum::Extension<AuthMethod>,
    axum::Extension(body): axum::Extension<Value>,
) -> Result<Response, ApiError> {
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
        debug!(
            "Checking billing server for user {} (auth_method: {:?})",
            user.uuid, auth_method
        );
        let can_chat_result = if auth_method == AuthMethod::ApiKey {
            billing_client.can_user_chat_api(user.uuid).await
        } else {
            billing_client.can_user_chat(user.uuid).await
        };

        match can_chat_result {
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
    let modified_body = body.as_object().expect("body was just checked").clone();

    // Check if streaming is requested (default to false if not specified)
    let is_streaming = modified_body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Extract the model from the request - error if not specified
    let model_name = modified_body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?
        .to_string();

    // Get the model route configuration
    let route = match state.proxy_router.get_model_route(&model_name).await {
        Some(r) => r,
        None => {
            error!("Model '{}' not found in routing table", model_name);
            return Err(ApiError::BadRequest);
        }
    };

    let modified_body_json = Value::Object(modified_body);

    // Create a new hyper client with better timeout configuration
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, HyperBody>(https);

    // Prepare the request to proxies
    debug!("Sending request for model: {}", model_name);

    // All models now have routes - some with fallbacks, some without
    let (res, successful_provider) = {
        debug!(
            "Using route for model {}: primary={}, fallbacks={}",
            model_name,
            route.primary.provider_name,
            route.fallbacks.len()
        );

        // Prepare request bodies for all providers
        let primary_model_name = state
            .proxy_router
            .get_model_name_for_provider(&model_name, &route.primary.provider_name);
        let mut primary_body = modified_body_json.as_object().unwrap().clone();
        primary_body.insert("model".to_string(), json!(primary_model_name));

        // Add stream_options based on provider capabilities
        // Tinfoil supports stream_options for both streaming and non-streaming
        // Continuum only supports it for streaming requests
        if route.primary.provider_name.to_lowercase() == "tinfoil" {
            // Tinfoil always gets stream_options with include_usage
            primary_body.insert("stream_options".to_string(), json!({"include_usage": true}));
        } else if is_streaming {
            // Other providers (like continuum) only get it when streaming
            primary_body.insert("stream_options".to_string(), json!({"include_usage": true}));
        }

        let primary_body_json =
            serde_json::to_string(&Value::Object(primary_body)).map_err(|e| {
                error!("Failed to serialize request body: {:?}", e);
                ApiError::InternalServerError
            })?;

        // Prepare fallback request if available
        let fallback_request = if let Some(fallback) = route.fallbacks.first() {
            let fallback_model_name = state
                .proxy_router
                .get_model_name_for_provider(&model_name, &fallback.provider_name);
            let mut fallback_body = modified_body_json.as_object().unwrap().clone();
            fallback_body.insert("model".to_string(), json!(fallback_model_name));

            // Add stream_options based on provider capabilities
            if fallback.provider_name.to_lowercase() == "tinfoil" {
                // Tinfoil always gets stream_options with include_usage
                fallback_body.insert("stream_options".to_string(), json!({"include_usage": true}));
            } else if is_streaming {
                // Other providers (like continuum) only get it when streaming
                fallback_body.insert("stream_options".to_string(), json!({"include_usage": true}));
            }

            let fallback_body_json =
                serde_json::to_string(&Value::Object(fallback_body)).map_err(|e| {
                    error!("Failed to serialize fallback request body: {:?}", e);
                    ApiError::InternalServerError
                })?;
            Some((fallback, fallback_body_json))
        } else {
            None
        };

        // Try cycling between primary and fallback up to 3 times each
        let max_cycles = 3;
        let mut last_error = None;
        let mut found_response = None;
        let mut successful_provider_name = String::new();

        for cycle in 0..max_cycles {
            if cycle > 0 {
                // Add delay between cycles (1s after first cycle, 2s after second)
                let delay = cycle as u64;
                debug!("Starting cycle {} after {}s delay", cycle + 1, delay);
                sleep(Duration::from_secs(delay)).await;
            }

            // Try primary
            debug!(
                "Cycle {}: Trying primary provider {}",
                cycle + 1,
                route.primary.provider_name
            );
            match try_provider(&client, &route.primary, &primary_body_json, &headers).await {
                Ok(response) => {
                    info!(
                        "Successfully got response from primary provider {} on cycle {}",
                        route.primary.provider_name,
                        cycle + 1
                    );
                    found_response = Some(response);
                    successful_provider_name = route.primary.provider_name.clone();
                    break;
                }
                Err(err) => {
                    error!(
                        "Cycle {}: Primary provider {} failed: {:?}",
                        cycle + 1,
                        route.primary.provider_name,
                        err
                    );
                    last_error = Some(err);
                }
            }

            // Try fallback if available
            if let Some((fallback_provider, ref fallback_body_json)) = fallback_request {
                debug!(
                    "Cycle {}: Trying fallback provider {}",
                    cycle + 1,
                    fallback_provider.provider_name
                );
                match try_provider(&client, fallback_provider, fallback_body_json, &headers).await {
                    Ok(response) => {
                        info!(
                            "Successfully got response from fallback provider {} on cycle {}",
                            fallback_provider.provider_name,
                            cycle + 1
                        );
                        found_response = Some(response);
                        successful_provider_name = fallback_provider.provider_name.clone();
                        break;
                    }
                    Err(err) => {
                        error!(
                            "Cycle {}: Fallback provider {} failed: {:?}",
                            cycle + 1,
                            fallback_provider.provider_name,
                            err
                        );
                        last_error = Some(err);
                    }
                }
            }
        }

        match found_response {
            Some(response) => (response, successful_provider_name),
            None => {
                let error_msg = if route.fallbacks.is_empty() {
                    format!(
                        "OpenAI API returned non-success status: Provider {} failed after {} attempts for model {}. Last error: {:?}",
                        route.primary.provider_name, max_cycles, model_name, last_error
                    )
                } else {
                    format!(
                        "OpenAI API returned non-success status: All providers failed after {} cycles for model {}. Last error: {:?}",
                        max_cycles, model_name, last_error
                    )
                };
                error!("{}", error_msg);
                return Err(ApiError::InternalServerError);
            }
        }
    };

    debug!("Successfully received response from OpenAI");

    // Handle non-streaming vs streaming responses differently
    if !is_streaming {
        // For non-streaming responses, read the entire body and return as JSON
        debug!("Handling non-streaming response");
        let body_bytes = to_bytes(res.into_body()).await.map_err(|e| {
            error!("Failed to read response body: {:?}", e);
            ApiError::InternalServerError
        })?;

        let response_str = String::from_utf8_lossy(&body_bytes);
        trace!("Non-streaming response body: {}", response_str);

        // Parse the response JSON
        let response_json: Value = serde_json::from_str(&response_str).map_err(|e| {
            error!("Failed to parse response JSON: {:?}", e);
            ApiError::InternalServerError
        })?;

        // Handle usage statistics if available
        if let Some(usage) = response_json.get("usage") {
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
                let input_cost =
                    BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(input_tokens);
                let output_cost =
                    BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(output_tokens);
                let total_cost = input_cost + output_cost;

                info!(
                    "OpenAI API usage for user {}: prompt_tokens={}, completion_tokens={}, total_tokens={}, estimated_cost={}",
                    user.uuid, input_tokens, output_tokens,
                    input_tokens + output_tokens,
                    total_cost
                );

                // Create token usage record and post to SQS in the background
                let state_clone = state.clone();
                let user_id = user.uuid;
                let is_api_request = auth_method == AuthMethod::ApiKey;
                let provider_name = successful_provider.clone();
                let model_name_clone = model_name.clone();
                tokio::spawn(async move {
                    // Create and store token usage record
                    let new_usage = NewTokenUsage::new(
                        user_id,
                        input_tokens,
                        output_tokens,
                        total_cost.clone(),
                    );

                    if let Err(e) = state_clone.db.create_token_usage(new_usage) {
                        error!("Failed to save token usage: {:?}", e);
                    }

                    // Post event to SQS if configured
                    if let Some(publisher) = &state_clone.sqs_publisher {
                        let event = UsageEvent {
                            event_id: Uuid::new_v4(), // Generate new UUID for idempotency
                            user_id,
                            input_tokens,
                            output_tokens,
                            estimated_cost: total_cost,
                            chat_time: Utc::now(),
                            is_api_request,
                            provider_name,
                            model_name: model_name_clone,
                        };

                        match publisher.publish_event(event).await {
                            Ok(_) => debug!("published usage event successfully"),
                            Err(e) => error!("error publishing usage event: {e}"),
                        }
                    }
                });
            }
        }

        // Encrypt and return the response
        let encrypted_response = encrypt_response(&state, &session_id, &response_json).await?;
        debug!("Exiting proxy_openai function (non-streaming)");
        return Ok(encrypted_response.into_response());
    }

    // For streaming responses, continue with the existing SSE logic
    debug!("Handling streaming response");
    let stream = res.into_body().into_stream();
    let buffer = Arc::new(Mutex::new(String::new()));
    let stream = stream
        .map(move |chunk| {
            let state = state.clone();
            let session_id = session_id;
            let user = user.clone();
            let buffer = buffer.clone();
            let auth_method = auth_method;
            let successful_provider = successful_provider.clone();
            let model_name = model_name.clone();
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

                        let mut processed_events: Vec<Result<Event, std::convert::Infallible>> =
                            Vec::new();
                        for event in events {
                            if let Some(processed_event) = encrypt_and_process_event(
                                &state,
                                &session_id,
                                &user,
                                &event,
                                auth_method,
                                &successful_provider,
                                &model_name,
                            )
                            .await
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

    debug!("Exiting proxy_openai function (streaming)");
    Ok(Sse::new(stream).into_response())
}

async fn encrypt_and_process_event(
    state: &AppState,
    session_id: &Uuid,
    user: &User,
    event: &str,
    auth_method: AuthMethod,
    provider_name: &str,
    model_name: &str,
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
                        let is_api_request = auth_method == AuthMethod::ApiKey;
                        let provider_name = provider_name.to_string();
                        let model_name = model_name.to_string();
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
                                    is_api_request,
                                    provider_name,
                                    model_name,
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
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(_body): axum::Extension<()>,
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

/// Helper function to send transcription request with retries to primary and fallback providers
async fn send_transcription_with_retries(
    client: &Client<HttpsConnector<hyper::client::HttpConnector>, HyperBody>,
    route: &crate::proxy_config::ModelRoute,
    state: &Arc<AppState>,
    model_name: &str,
    params: &TranscriptionParams<'_>,
) -> Result<Value, String> {
    let max_cycles = 3;
    let mut last_error = None;

    for cycle in 0..max_cycles {
        if cycle > 0 {
            let delay = cycle as u64;
            debug!("Starting cycle {} after {}s delay", cycle + 1, delay);
            sleep(Duration::from_secs(delay)).await;
        }

        // Try primary
        debug!(
            "Cycle {}: Trying primary provider {} for transcription",
            cycle + 1,
            route.primary.provider_name
        );

        let primary_model = state
            .proxy_router
            .get_model_name_for_provider(model_name, &route.primary.provider_name);

        match send_transcription_request(client, &route.primary, &primary_model, params).await {
            Ok(response) => {
                info!(
                    "Successfully got transcription from primary provider {} on cycle {}",
                    route.primary.provider_name,
                    cycle + 1
                );
                return Ok(response);
            }
            Err(err) => {
                error!(
                    "Cycle {}: Primary provider {} failed: {}",
                    cycle + 1,
                    route.primary.provider_name,
                    err
                );
                last_error = Some(err);
            }
        }

        // Try fallback if available
        if let Some(fallback) = route.fallbacks.first() {
            debug!(
                "Cycle {}: Trying fallback provider {} for transcription",
                cycle + 1,
                fallback.provider_name
            );

            let fallback_model = state
                .proxy_router
                .get_model_name_for_provider(model_name, &fallback.provider_name);

            match send_transcription_request(client, fallback, &fallback_model, params).await {
                Ok(response) => {
                    info!(
                        "Successfully got transcription from fallback provider {} on cycle {}",
                        fallback.provider_name,
                        cycle + 1
                    );
                    return Ok(response);
                }
                Err(err) => {
                    error!(
                        "Cycle {}: Fallback provider {} failed: {}",
                        cycle + 1,
                        fallback.provider_name,
                        err
                    );
                    last_error = Some(err);
                }
            }
        }
    }

    Err(format!(
        "All providers failed after {} cycles. Last error: {:?}",
        max_cycles, last_error
    ))
}

async fn proxy_transcription(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(transcription_request): axum::Extension<TranscriptionRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    debug!("Entering proxy_transcription function");

    // Prevent guest users from using the transcription feature
    if user.is_guest() {
        error!(
            "Guest user attempted to use transcription feature: {}",
            user.uuid
        );
        return Err(ApiError::Unauthorized);
    }

    // Decode base64 audio file
    let file_bytes = general_purpose::STANDARD
        .decode(&transcription_request.file)
        .map_err(|e| {
            error!("Failed to decode base64 audio file: {:?}", e);
            ApiError::BadRequest
        })?;

    // Validate file size (100MB limit as sanity check, CF already limits to 50MB)
    let file_size = file_bytes.len();
    if file_size == 0 {
        error!("Audio file is empty");
        return Err(ApiError::BadRequest);
    }
    if file_size > MAX_AUDIO_SIZE {
        error!(
            "Audio file size {} bytes exceeds maximum allowed size of {} bytes",
            file_size, MAX_AUDIO_SIZE
        );
        return Err(ApiError::BadRequest);
    }
    info!("Audio file size: {} bytes", file_size);

    // Check if we need to split the audio
    let splitter = AudioSplitter::new();

    // Get the model route configuration
    let route = match state
        .proxy_router
        .get_model_route(&transcription_request.model)
        .await
    {
        Some(r) => r,
        None => {
            error!(
                "Model '{}' not found in routing table",
                transcription_request.model
            );
            return Err(ApiError::BadRequest);
        }
    };

    // Create a new hyper client
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, HyperBody>(https);

    // Always split the audio (returns single chunk if no splitting needed)
    let chunks = splitter
        .split_audio(&file_bytes, &transcription_request.content_type)
        .map_err(|e| {
            error!("Failed to split audio: {}", e);
            ApiError::InternalServerError
        })?;

    info!("Processing {} chunk(s)", chunks.len());

    // Process chunks in parallel (even if it's just one)
    let mut futures = Vec::new();

    for chunk in chunks {
        let client = client.clone();
        let mut route = route.clone();
        let state = state.clone();
        let model_name = transcription_request.model.clone();
        let filename = transcription_request.filename.clone();
        let content_type = transcription_request.content_type.clone();
        let language = transcription_request.language.clone();
        let prompt = transcription_request.prompt.clone();
        let response_format = transcription_request.response_format.clone();
        let temperature = transcription_request.temperature;

        let future = async move {
            let chunk_size = chunk.data.len();
            info!(
                "Processing chunk {} (size: {} bytes)",
                chunk.index, chunk_size
            );

            // If chunk is over 0.5MB and primary is Tinfoil, skip Tinfoil and use fallback only
            if chunk_size > TINFOIL_MAX_SIZE
                && route.primary.provider_name.to_lowercase() == "tinfoil"
            {
                info!(
                    "Chunk {} size {} bytes exceeds Tinfoil's 0.5MB limit, using fallback only",
                    chunk.index, chunk_size
                );

                // If we have a fallback (should be Continuum), use it as primary
                if let Some(fallback) = route.fallbacks.first() {
                    route.primary = fallback.clone();
                    route.fallbacks.clear();
                } else {
                    // No fallback available, this will fail
                    return Err(format!(
                        "Chunk {} size {} bytes exceeds Tinfoil's limit and no fallback available",
                        chunk.index, chunk_size
                    ));
                }
            }

            let params = TranscriptionParams {
                audio_data: &chunk.data,
                filename: &filename,
                content_type: &content_type,
                language: language.as_deref(),
                prompt: prompt.as_deref(),
                response_format: &response_format,
                temperature,
            };

            match send_transcription_with_retries(&client, &route, &state, &model_name, &params)
                .await
            {
                Ok(response) => {
                    info!("Chunk {} transcribed successfully", chunk.index);
                    Ok((chunk.index, response))
                }
                Err(err) => {
                    error!("Chunk {} failed: {}", chunk.index, err);
                    Err(format!("Chunk {} failed: {}", chunk.index, err))
                }
            }
        };

        futures.push(future);
    }

    // Execute all futures in parallel
    let results: Vec<Result<(usize, Value), String>> = futures::future::join_all(futures).await;

    // Check if all chunks succeeded
    let mut successful_results = Vec::new();
    for result in results {
        match result {
            Ok(r) => successful_results.push(r),
            Err(e) => {
                error!("Chunk processing failed: {}", e);
                return Err(ApiError::InternalServerError);
            }
        }
    }

    // Get the response (merge if multiple chunks, return as-is if single)
    let response = if successful_results.is_empty() {
        error!("No successful transcription results");
        return Err(ApiError::InternalServerError);
    } else if successful_results.len() == 1 {
        // Single chunk - return the response directly
        successful_results
            .into_iter()
            .next()
            .map(|(_, r)| r)
            .ok_or_else(|| {
                error!("Failed to get single result");
                ApiError::InternalServerError
            })?
    } else {
        // Multiple chunks - merge the results
        let merged = merge_transcriptions(successful_results).map_err(|e| {
            error!("Failed to merge transcriptions: {}", e);
            ApiError::InternalServerError
        })?;

        // Convert merged result to standard response format
        let mut response = json!({
            "text": merged.text,
        });

        if let Some(lang) = merged.language {
            response["language"] = json!(lang);
        }

        if let Some(segments) = merged.segments {
            response["segments"] = json!(segments);
        }

        response
    };

    debug!("Exiting proxy_transcription function");

    // TODO: Add SQS-based billing events for transcription usage
    // Should track: audio duration/size, model used, user ID, timestamp, provider

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &response).await
}

/// Sanitize form field values to prevent HTTP header injection attacks
/// Removes or replaces any CRLF sequences that could be used to inject headers
fn sanitize_form_field(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '\r' | '\n'))
        .collect()
}

async fn send_transcription_request(
    client: &Client<HttpsConnector<hyper::client::HttpConnector>, HyperBody>,
    provider: &ProxyConfig,
    model: &str,
    params: &TranscriptionParams<'_>,
) -> Result<Value, String> {
    // Build multipart form data
    let boundary = format!("----FormBoundary{}", Uuid::new_v4().simple());
    let mut form_data = Vec::new();

    // Add model field (sanitized to prevent header injection)
    let safe_model = sanitize_form_field(model);
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    form_data.extend_from_slice(safe_model.as_bytes());
    form_data.extend_from_slice(b"\r\n");

    // Add file field with sanitized filename to prevent header injection
    let safe_filename: String = params
        .filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();

    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
            safe_filename
        )
        .as_bytes(),
    );
    let safe_content_type = sanitize_form_field(params.content_type);
    form_data.extend_from_slice(format!("Content-Type: {}\r\n\r\n", safe_content_type).as_bytes());
    form_data.extend_from_slice(params.audio_data);
    form_data.extend_from_slice(b"\r\n");

    // Add optional fields (sanitized to prevent header injection)
    if let Some(lang) = params.language {
        let safe_lang = sanitize_form_field(lang);
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        form_data.extend_from_slice(safe_lang.as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }
    if let Some(p) = params.prompt {
        let safe_prompt = sanitize_form_field(p);
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data.extend_from_slice(b"Content-Disposition: form-data; name=\"prompt\"\r\n\r\n");
        form_data.extend_from_slice(safe_prompt.as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }
    let safe_response_format = sanitize_form_field(params.response_format);
    form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    form_data
        .extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
    form_data.extend_from_slice(safe_response_format.as_bytes());
    form_data.extend_from_slice(b"\r\n");
    if let Some(temp) = params.temperature {
        form_data.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        form_data
            .extend_from_slice(b"Content-Disposition: form-data; name=\"temperature\"\r\n\r\n");
        form_data.extend_from_slice(temp.to_string().as_bytes());
        form_data.extend_from_slice(b"\r\n");
    }

    // End boundary
    form_data.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    // Build request
    let endpoint = format!("{}/v1/audio/transcriptions", provider.base_url);
    let mut req = Request::builder().method("POST").uri(&endpoint).header(
        "Content-Type",
        format!("multipart/form-data; boundary={}", boundary),
    );

    if let Some(api_key) = &provider.api_key {
        if !api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }
    }

    let req = req
        .body(HyperBody::from(form_data))
        .map_err(|e| format!("Failed to create request: {:?}", e))?;

    // Send request
    match client.request(req).await {
        Ok(res) => {
            if res.status().is_success() {
                let body_bytes = to_bytes(res.into_body())
                    .await
                    .map_err(|e| format!("Failed to read response body: {:?}", e))?;

                let response_json: Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| format!("Failed to parse response: {:?}", e))?;

                Ok(response_json)
            } else {
                let status = res.status();
                let body_bytes = to_bytes(res.into_body()).await.ok();
                let error_msg = body_bytes
                    .map(|b| String::from_utf8_lossy(&b).to_string())
                    .unwrap_or_else(|| status.to_string());

                Err(format!(
                    "Provider {} returned error: {} - {}",
                    provider.provider_name, status, error_msg
                ))
            }
        }
        Err(e) => Err(format!(
            "Failed to send request to {}: {:?}",
            provider.provider_name, e
        )),
    }
}

async fn proxy_tts(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(tts_request): axum::Extension<TTSRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    debug!("Entering proxy_tts function");

    // Prevent guest users from using the TTS feature
    if user.is_guest() {
        error!("Guest user attempted to use TTS feature: {}", user.uuid);
        return Err(ApiError::Unauthorized);
    }

    // Validate input is not empty
    if tts_request.input.trim().is_empty() {
        error!("Input text is empty");
        return Err(ApiError::BadRequest);
    }

    // Only kokoro is supported for TTS
    if tts_request.model != "kokoro" {
        error!(
            "Unsupported TTS model: {}. Only 'kokoro' is supported",
            tts_request.model
        );
        return Err(ApiError::BadRequest);
    }

    // Build request for Tinfoil Kokoro
    let tts_api_request = TTSRequest {
        model: "kokoro".to_string(),
        voice: tts_request.voice.clone(),
        input: tts_request.input.clone(),
        response_format: tts_request.response_format.clone(),
        speed: tts_request.speed,
    };

    // Use the tinfoil proxy configuration
    // For now, we'll hardcode to use tinfoil proxy - in future could route based on model
    let base_url = state.proxy_router.get_tinfoil_base_url().ok_or_else(|| {
        error!("Tinfoil proxy not configured for TTS");
        ApiError::InternalServerError
    })?;

    let proxy_config = ProxyConfig {
        base_url,
        api_key: None, // Tinfoil proxy handles auth internally
        provider_name: "tinfoil".to_string(),
    };

    // Create a new hyper client
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, HyperBody>(https);

    // Build request
    let req = Request::builder()
        .method("POST")
        .uri(format!("{}/v1/audio/speech", proxy_config.base_url))
        .header("Content-Type", "application/json")
        .body(HyperBody::from(
            serde_json::to_string(&tts_api_request).map_err(|e| {
                error!("Failed to serialize TTS request: {:?}", e);
                ApiError::InternalServerError
            })?,
        ))
        .map_err(|e| {
            error!("Failed to create request: {:?}", e);
            ApiError::InternalServerError
        })?;

    // Send request
    let res = client.request(req).await.map_err(|e| {
        error!("Failed to send TTS request: {:?}", e);
        ApiError::InternalServerError
    })?;

    if !res.status().is_success() {
        error!("TTS proxy returned non-success status: {}", res.status());
        return Err(ApiError::InternalServerError);
    }

    // Get response body as bytes
    let body_bytes = to_bytes(res.into_body()).await.map_err(|e| {
        error!("Failed to read TTS response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Create a response object with the audio data and metadata
    let audio_response = json!({
        "content_base64": general_purpose::STANDARD.encode(&body_bytes),
        "content_type": match tts_request.response_format.as_str() {
            "mp3" => "audio/mpeg",
            "opus" => "audio/opus",
            "aac" => "audio/aac",
            "flac" => "audio/flac",
            "wav" => "audio/wav",
            "pcm" => "audio/pcm",
            _ => "audio/mpeg", // Default to MP3
        },
    });

    debug!("Exiting proxy_tts function");

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &audio_response).await
}

/// Helper function to try a provider once
async fn try_provider(
    client: &Client<HttpsConnector<hyper::client::HttpConnector>, HyperBody>,
    proxy_config: &ProxyConfig,
    body_json: &str,
    headers: &HeaderMap,
) -> Result<hyper::Response<HyperBody>, String> {
    debug!("Making request to {}", proxy_config.provider_name);

    // Build request
    let mut req = Request::builder()
        .method("POST")
        .uri(format!("{}/v1/chat/completions", proxy_config.base_url))
        .header("Content-Type", "application/json");

    if let Some(api_key) = &proxy_config.api_key {
        if !api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }
    }

    // Forward relevant headers from the original request
    for (key, value) in headers.iter() {
        if key != header::HOST
            && key != header::AUTHORIZATION
            && key != header::CONTENT_LENGTH
            && key != header::CONTENT_TYPE
        {
            if let (Ok(name), Ok(val)) = (
                HeaderName::from_bytes(key.as_ref()),
                HeaderValue::from_str(value.to_str().unwrap_or_default()),
            ) {
                req = req.header(name, val);
            }
        }
    }

    let req = req
        .body(HyperBody::from(body_json.to_string()))
        .map_err(|e| format!("Failed to create request body: {:?}", e))?;

    match client.request(req).await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(response)
            } else {
                let status = response.status();
                error!(
                    "Provider {} returned non-success status: {}",
                    proxy_config.provider_name, status
                );
                debug!("Response headers: {:?}", response.headers());

                // Try to get error body for logging
                if let Ok(body_bytes) = to_bytes(response.into_body()).await {
                    let body_str = String::from_utf8_lossy(&body_bytes);
                    error!("Response body: {}", body_str);
                    Err(format!(
                        "Provider {} returned status {}: {}",
                        proxy_config.provider_name, status, body_str
                    ))
                } else {
                    Err(format!(
                        "Provider {} returned status {}",
                        proxy_config.provider_name, status
                    ))
                }
            }
        }
        Err(e) => {
            error!(
                "Failed to send request to {}: {:?}",
                proxy_config.provider_name, e
            );
            Err(format!(
                "Failed to connect to {}: {}",
                proxy_config.provider_name, e
            ))
        }
    }
}
