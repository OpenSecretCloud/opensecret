use crate::models::token_usage::NewTokenUsage;
use crate::models::users::User;
use crate::nearai::e2ee::{
    decrypt_chat_completion_json_in_place, prepare_e2ee_request, NearAiResponseCrypto,
};
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
use futures::{StreamExt, TryStreamExt};
use hyper::body::to_bytes;
use hyper::header::{HeaderName, HeaderValue};
use hyper::{Body as HyperBody, Client, Request};
use hyper_tls::HttpsConnector;
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, trace};
use uuid::Uuid;

// Maximum audio file size (100MB) - sanity check, CF already limits to 50MB
const MAX_AUDIO_SIZE: usize = 100 * 1024 * 1024;

// Timeout constants for provider requests
const REQUEST_TIMEOUT_SECS: u64 = 120; // Request timeout (generous for large non-streaming responses)
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 120; // Per-chunk timeout for streaming reads

#[derive(Clone)]
struct PreparedChatRequest {
    body_json: String,
    extra_headers: Vec<(HeaderName, HeaderValue)>,
    near_crypto: Option<NearAiResponseCrypto>,
}

async fn prepare_chat_request_for_provider(
    state: &Arc<AppState>,
    proxy_config: &ProxyConfig,
    mut body: Value,
) -> Result<PreparedChatRequest, ApiError> {
    let mut extra_headers: Vec<(HeaderName, HeaderValue)> = Vec::new();
    let mut near_crypto: Option<NearAiResponseCrypto> = None;

    if proxy_config.provider_name.to_lowercase() == "nearai" {
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or(ApiError::BadRequest)?
            .to_string();

        trace!(
            "Near.AI E2EE: fetching verified node for model={}",
            model
        );
        let node = state
            .nearai_verifier
            .get_verified_model_node(&model)
            .await
            .map_err(|e| {
                error!("Near.AI verification failed (model={}): {}", model, e);
                ApiError::ServiceUnavailable
            })?;

        trace!(
            "Near.AI E2EE: using model node pubkey={}... (len={})",
            &node.signing_public_key[..node.signing_public_key.len().min(16)],
            node.signing_public_key.len()
        );

        let crypto = prepare_e2ee_request(&mut body, &node.signing_public_key).map_err(|e| {
            error!("Near.AI request encryption failed (model={}): {}", model, e);
            ApiError::ServiceUnavailable
        })?;

        trace!(
            "Near.AI E2EE: client ephemeral pubkey={}... (len={})",
            &crypto.client_public_key_hex[..crypto.client_public_key_hex.len().min(16)],
            crypto.client_public_key_hex.len()
        );

        extra_headers.push((
            HeaderName::from_static("x-signing-algo"),
            HeaderValue::from_static("ecdsa"),
        ));
        extra_headers.push((
            HeaderName::from_static("x-client-pub-key"),
            HeaderValue::from_str(&crypto.client_public_key_hex)
                .map_err(|_| ApiError::InternalServerError)?,
        ));
        extra_headers.push((
            HeaderName::from_static("x-model-pub-key"),
            HeaderValue::from_str(&node.signing_public_key)
                .map_err(|_| ApiError::InternalServerError)?,
        ));

        near_crypto = Some(crypto);
    }

    let body_json = serde_json::to_string(&body).map_err(|e| {
        error!("Failed to serialize request body: {:?}", e);
        ApiError::InternalServerError
    })?;

    Ok(PreparedChatRequest {
        body_json,
        extra_headers,
        near_crypto,
    })
}

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

/// Request structure for embeddings endpoints
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct EmbeddingRequest {
    input: serde_json::Value, // string or array of strings
    #[serde(default = "default_embedding_model")]
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

fn default_embedding_model() -> String {
    "nomic-embed-text".to_string()
}

// ============================================================================
// Centralized Billing Architecture - New Types
// ============================================================================

/// Context needed for billing/usage tracking
#[derive(Debug, Clone)]
pub struct BillingContext {
    pub auth_method: AuthMethod,
    pub model_name: String,
}

impl BillingContext {
    pub fn new(auth_method: AuthMethod, model_name: String) -> Self {
        Self {
            auth_method,
            model_name,
        }
    }
}

/// Usage statistics extracted from a completion
#[derive(Debug, Clone)]
pub struct CompletionUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
}

/// A chunk from the completion stream
#[derive(Clone, Debug)]
pub enum CompletionChunk {
    /// Streaming chunk with full JSON from upstream (includes all metadata)
    StreamChunk(Value),
    /// Complete response for non-streaming
    FullResponse(Value),
    /// Usage information (for streaming with include_usage)
    Usage(CompletionUsage),
    /// Stream finished
    Done,
    /// Stream error occurred
    Error(String),
}

/// Metadata about the completion
#[derive(Clone, Debug)]
pub struct CompletionMetadata {
    pub provider_name: String,
    pub model_name: String,
    pub is_streaming: bool,
}

/// Processed completion stream - billing happens automatically
pub struct CompletionStream {
    /// The actual data stream for consumers
    pub stream: mpsc::Receiver<CompletionChunk>,
    /// Metadata about the completion
    pub metadata: CompletionMetadata,
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
        .route(
            "/v1/embeddings",
            post(proxy_embeddings).layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                decrypt_request::<EmbeddingRequest>,
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

    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for chat: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use chat feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use chat without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
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

    // Extract the model from the request
    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            error!("Model not specified in request");
            ApiError::BadRequest
        })?
        .to_string();

    // Create billing context
    let billing_context = BillingContext::new(auth_method, model_name.clone());

    // Get the completion stream - billing happens automatically inside!
    let completion =
        get_chat_completion_response(&state, &user, body, &headers, billing_context).await?;

    debug!(
        "Received completion from provider: {} (streaming: {})",
        completion.metadata.provider_name, completion.metadata.is_streaming
    );

    // Handle non-streaming vs streaming responses differently
    if !completion.metadata.is_streaming {
        // For non-streaming responses, get the full response chunk
        debug!("Handling non-streaming response");
        let mut rx = completion.stream;

        // Get the FullResponse chunk
        if let Some(CompletionChunk::FullResponse(response_json)) = rx.recv().await {
            // Billing already happened in get_chat_completion_response!
            // Just encrypt and return
            let encrypted_response = encrypt_response(&state, &session_id, &response_json).await?;
            debug!("Exiting proxy_openai function (non-streaming)");
            return Ok(encrypted_response.into_response());
        } else {
            error!("Expected FullResponse chunk but got something else");
            return Err(ApiError::InternalServerError);
        }
    }

    // For streaming responses, process CompletionChunk stream
    debug!("Handling streaming response");
    let mut rx = completion.stream;

    let stream = async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            match chunk {
                CompletionChunk::StreamChunk(json) => {
                    // Pass through full JSON (includes all metadata from upstream)
                    match encrypt_sse_event(&state, &session_id, &json).await {
                        Ok(event) => yield Ok::<Event, std::convert::Infallible>(event),
                        Err(e) => {
                            error!("Failed to encrypt event data: {:?}", e);
                            yield Ok(Event::default().data("Error: Encryption failed"));
                            break;
                        }
                    }
                }
                CompletionChunk::Usage(_usage) => {
                    // Billing already handled internally, no need to send to client
                    trace!("Received usage chunk (billing already processed)");
                }
                CompletionChunk::Done => {
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                CompletionChunk::Error(error_msg) => {
                    error!("Stream error from completion API: {}", error_msg);
                    yield Ok(Event::default().data(format!("Error: {}", error_msg)));
                    break;
                }
                CompletionChunk::FullResponse(_) => {
                    // Shouldn't happen in streaming mode
                    error!("Received FullResponse in streaming mode");
                    yield Ok(Event::default().data("Error: Invalid event format"));
                    break;
                }
            }
        }
    };

    debug!("Exiting proxy_openai function (streaming)");
    Ok(Sse::new(stream).into_response())
}

/// Internal function to get chat completion response with automatic billing
/// This can be used by both the proxy_openai endpoint and the responses API
///
/// Billing happens INTERNALLY within this function - consumers just receive processed chunks
pub async fn get_chat_completion_response(
    state: &Arc<AppState>,
    user: &User,
    body: Value,
    headers: &HeaderMap,
    billing_context: BillingContext,
) -> Result<CompletionStream, ApiError> {
    debug!("Entering get_chat_completion_response with billing context");

    if body.is_null() || body.as_object().is_none_or(|obj| obj.is_empty()) {
        error!("Request body is empty or invalid");
        return Err(ApiError::BadRequest);
    }

    let modified_body = body
        .as_object()
        .ok_or_else(|| {
            error!("Request body is not a JSON object");
            ApiError::BadRequest
        })?
        .clone();

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
    let route = match state.proxy_router.get_model_route(&model_name) {
        Some(r) => r,
        None => {
            error!("Model '{}' not found in routing table", model_name);
            return Err(ApiError::BadRequest);
        }
    };

    // Create a new hyper client with better timeout configuration
    let https = HttpsConnector::new();
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .build::<_, HyperBody>(https);

    // Prepare the request to proxies
    debug!("Sending request for model: {}", model_name);

    let (res, successful_provider, successful_near_crypto) = {
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
        let mut primary_body = modified_body.clone();
        primary_body.insert("model".to_string(), json!(primary_model_name));

        // Add stream_options based on provider capabilities
        if route.primary.provider_name.to_lowercase() == "tinfoil" || is_streaming {
            primary_body.insert("stream_options".to_string(), json!({"include_usage": true}));
        }

        let primary_prepared =
            prepare_chat_request_for_provider(state, &route.primary, Value::Object(primary_body))
                .await?;

        // Prepare fallback request if available
        let fallback_request = if let Some(fallback) = route.fallbacks.first() {
            let fallback_model_name = state
                .proxy_router
                .get_model_name_for_provider(&model_name, &fallback.provider_name);
            let mut fallback_body = modified_body.clone();
            fallback_body.insert("model".to_string(), json!(fallback_model_name));

            if fallback.provider_name.to_lowercase() == "tinfoil" || is_streaming {
                fallback_body.insert("stream_options".to_string(), json!({"include_usage": true}));
            }

            let prepared =
                prepare_chat_request_for_provider(state, fallback, Value::Object(fallback_body))
                    .await?;
            Some((fallback, prepared))
        } else {
            None
        };

        // Try cycling between primary and fallback up to 3 times each
        let max_cycles = 3;
        let mut last_error = None;
        let mut found_response = None;
        let mut successful_provider_name = String::new();
        let mut successful_near_crypto: Option<NearAiResponseCrypto> = None;

        for cycle in 0..max_cycles {
            if cycle > 0 {
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
            match try_provider(
                &client,
                &route.primary,
                &primary_prepared.body_json,
                headers,
                &primary_prepared.extra_headers,
            )
            .await
            {
                Ok(response) => {
                    info!(
                        "Successfully got response from primary provider {} on cycle {}",
                        route.primary.provider_name,
                        cycle + 1
                    );
                    found_response = Some(response);
                    successful_provider_name = route.primary.provider_name.clone();
                    successful_near_crypto = primary_prepared.near_crypto.clone();
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
            if let Some((fallback_provider, fallback_prepared)) = &fallback_request {
                debug!(
                    "Cycle {}: Trying fallback provider {}",
                    cycle + 1,
                    fallback_provider.provider_name
                );
                match try_provider(
                    &client,
                    fallback_provider,
                    &fallback_prepared.body_json,
                    headers,
                    &fallback_prepared.extra_headers,
                )
                .await
                {
                    Ok(response) => {
                        info!(
                            "Successfully got response from fallback provider {} on cycle {}",
                            fallback_provider.provider_name,
                            cycle + 1
                        );
                        found_response = Some(response);
                        successful_provider_name = fallback_provider.provider_name.clone();
                        successful_near_crypto = fallback_prepared.near_crypto.clone();
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
            Some(response) => (response, successful_provider_name, successful_near_crypto),
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
                if route.primary.provider_name.to_lowercase() == "nearai"
                    && route.fallbacks.is_empty()
                {
                    return Err(ApiError::ServiceUnavailable);
                }

                return Err(ApiError::InternalServerError);
            }
        }
    };

    debug!(
        "Successfully received response from provider: {}",
        successful_provider
    );

    // NOW: Process the response internally and handle billing
    if !is_streaming {
        // NON-STREAMING: Simple case
        debug!("Processing non-streaming response with internal billing");
        let body_bytes = to_bytes(res.into_body()).await.map_err(|e| {
            error!("Failed to read response body: {:?}", e);
            ApiError::InternalServerError
        })?;

        let mut response_json: Value = serde_json::from_str(&String::from_utf8_lossy(&body_bytes))
            .map_err(|e| {
                error!("Failed to parse response JSON: {:?}", e);
                ApiError::InternalServerError
            })?;

        if successful_provider.to_lowercase() == "nearai" {
            let Some(crypto) = successful_near_crypto.as_ref() else {
                error!("Near.AI response missing request crypto");
                return Err(ApiError::InternalServerError);
            };

            decrypt_chat_completion_json_in_place(&mut response_json, crypto).map_err(|e| {
                error!("Near.AI response decryption failed: {}", e);
                ApiError::ServiceUnavailable
            })?;
        }

        // ✅ Handle billing HERE, inside completions API
        if let Some(usage) = extract_usage(&response_json) {
            publish_usage_event_internal(
                state,
                user,
                &billing_context,
                usage,
                &successful_provider,
            )
            .await;
        }

        // Return the full response as a single chunk
        let (tx, rx) = mpsc::channel(2); // Need space for FullResponse + Done
        let _ = tx.send(CompletionChunk::FullResponse(response_json)).await;
        let _ = tx.send(CompletionChunk::Done).await;

        return Ok(CompletionStream {
            stream: rx,
            metadata: CompletionMetadata {
                provider_name: successful_provider,
                model_name: billing_context.model_name.clone(),
                is_streaming: false,
            },
        });
    }

    // STREAMING: Complex case - spawn internal processor
    debug!("Processing streaming response with internal billing");
    let (tx_consumer, rx_consumer) = mpsc::channel(100);

    // Spawn INTERNAL task that handles billing
    let state_clone = state.clone();
    let user_clone = user.clone();
    let billing_ctx = billing_context.clone();
    let provider = successful_provider.clone();
    let near_crypto = successful_near_crypto.clone();
    let provider_is_near = provider.to_lowercase() == "nearai";

    tokio::spawn(async move {
        let mut body_stream = res.into_body().into_stream();
        let mut buffer = String::new();
        let mut usage_sent = false; // Track if we've already published usage for this stream

        loop {
            match timeout(
                Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS),
                body_stream.next(),
            )
            .await
            {
                Ok(Some(chunk_result)) => {
                    match chunk_result {
                        Ok(bytes) => {
                            let chunk_str = String::from_utf8_lossy(bytes.as_ref());
                            buffer.push_str(&chunk_str);

                            // Parse SSE frames
                            while let Some(frame) = extract_sse_frame(&mut buffer) {
                                if frame == "[DONE]" {
                                    if tx_consumer.send(CompletionChunk::Done).await.is_err() {
                                        // Receiver dropped, stop processing
                                        return;
                                    }
                                    return;
                                }

                                match serde_json::from_str::<Value>(&frame) {
                                    Ok(mut json) => {
                                        if provider_is_near {
                                            let Some(ref crypto) = near_crypto else {
                                                error!("Near.AI stream missing request crypto");
                                                let _ = tx_consumer
                                                    .send(CompletionChunk::Error(
                                                        "Near.AI crypto missing".to_string(),
                                                    ))
                                                    .await;
                                                return;
                                            };

                                            if let Err(e) = decrypt_chat_completion_json_in_place(
                                                &mut json, crypto,
                                            ) {
                                                error!(
                                                    "Near.AI stream chunk decryption failed: {}",
                                                    e
                                                );
                                                let _ = tx_consumer
                                                    .send(CompletionChunk::Error(
                                                        "Near.AI decrypt failed".to_string(),
                                                    ))
                                                    .await;
                                                return;
                                            }
                                        }

                                        // ✅ Extract and publish billing HERE - but ONLY on final chunk
                                        // This prevents sending usage data on intermediate chunks that vLLM now includes
                                        let has_finish = has_finish_reason(&json);

                                        if let Some(usage) = extract_usage(&json) {
                                            // Only publish usage on final chunk (has finish_reason) with actual completion tokens
                                            // Also ensure we only send usage once per stream
                                            if has_finish
                                                && usage.completion_tokens > 0
                                                && !usage_sent
                                            {
                                                publish_usage_event_internal(
                                                    &state_clone,
                                                    &user_clone,
                                                    &billing_ctx,
                                                    usage.clone(),
                                                    &provider,
                                                )
                                                .await;
                                                usage_sent = true;

                                                // Also send usage to consumer
                                                if tx_consumer
                                                    .send(CompletionChunk::Usage(usage))
                                                    .await
                                                    .is_err()
                                                {
                                                    return;
                                                }
                                            } else {
                                                trace!(
                                                    "Skipping usage publish: has_finish={}, completion_tokens={}, usage_sent={}",
                                                    has_finish, usage.completion_tokens, usage_sent
                                                );
                                            }
                                        }

                                        // Send full JSON chunk to consumer (preserves all metadata)
                                        if tx_consumer
                                            .send(CompletionChunk::StreamChunk(json))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                    Err(e) => {
                                        error!("Received non-JSON data event. Error: {:?}", e);
                                        if tx_consumer
                                            .send(CompletionChunk::Error(
                                                "Invalid JSON".to_string(),
                                            ))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Stream error: {:?}", e);
                            if tx_consumer
                                .send(CompletionChunk::Error(e.to_string()))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            break;
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended without explicit [DONE]
                    if tx_consumer.send(CompletionChunk::Done).await.is_err() {
                        return;
                    }
                    break;
                }
                Err(_) => {
                    // Timeout waiting for next chunk
                    error!("Stream chunk timeout after {}s", STREAM_CHUNK_TIMEOUT_SECS);
                    if tx_consumer
                        .send(CompletionChunk::Error("Stream timeout".to_string()))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    break;
                }
            }
        }
    });

    Ok(CompletionStream {
        stream: rx_consumer,
        metadata: CompletionMetadata {
            provider_name: successful_provider,
            model_name: billing_context.model_name.clone(),
            is_streaming: true,
        },
    })
}

// ============================================================================
// Centralized Billing Architecture - Internal Functions
// ============================================================================

/// Helper to extract usage from response JSON
fn extract_usage(json: &Value) -> Option<CompletionUsage> {
    let usage_json = json.get("usage")?;
    if usage_json.is_null() || !usage_json.is_object() {
        return None;
    }

    Some(CompletionUsage {
        prompt_tokens: usage_json
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        completion_tokens: usage_json
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
    })
}

/// Helper to check if a streaming chunk has a finish_reason
/// This indicates it's the final chunk in the stream
fn has_finish_reason(json: &Value) -> bool {
    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            if let Some(finish_reason) = choice.get("finish_reason") {
                // finish_reason is present and not null
                if !finish_reason.is_null() {
                    return true;
                }
            }
        }
    }
    false
}

/// Helper to extract SSE frame from buffer
/// Returns the data portion of "data: <content>" frames, None if no complete frame available
fn extract_sse_frame(buffer: &mut String) -> Option<String> {
    loop {
        // Look for a complete SSE frame (ends with \n\n)
        if let Some(pos) = buffer.find("\n\n") {
            let frame = buffer[..pos].to_string();
            *buffer = buffer[pos + 2..].to_string();

            // Skip empty frames
            if frame.trim().is_empty() {
                continue;
            }

            // Return data content if it's a data frame, otherwise keep looking
            if let Some(data) = frame.strip_prefix("data: ") {
                return Some(data.to_string());
            }
            // Skip non-data frames (comments, etc.) and continue looking
            continue;
        }

        // No complete frame available
        return None;
    }
}

/// Internal billing function - NEVER exposed outside this module
/// This function publishes usage events to both the database and SQS
async fn publish_usage_event_internal(
    state: &Arc<AppState>,
    user: &User,
    billing_context: &BillingContext,
    usage: CompletionUsage,
    provider_name: &str,
) {
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return;
    }

    let input_cost =
        BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(usage.prompt_tokens);
    let output_cost =
        BigDecimal::from_str("0.0000053").unwrap() * BigDecimal::from(usage.completion_tokens);
    let total_cost = input_cost + output_cost;

    info!(
        "Chat completion usage for user {}: model={}, provider={}, prompt_tokens={}, completion_tokens={}, total_tokens={}, estimated_cost={}",
        user.uuid,
        billing_context.model_name,
        provider_name,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.prompt_tokens + usage.completion_tokens,
        total_cost
    );

    // Spawn background task for DB + SQS
    let state_clone = state.clone();
    let user_id = user.uuid;
    let is_api_request = billing_context.auth_method == AuthMethod::ApiKey;
    let provider_name = provider_name.to_string();
    let model_name = billing_context.model_name.clone();

    tokio::spawn(async move {
        // Create and store token usage record
        let new_usage = NewTokenUsage::new(
            user_id,
            usage.prompt_tokens,
            usage.completion_tokens,
            total_cost.clone(),
        );

        if let Err(e) = state_clone.db.create_token_usage(new_usage) {
            error!("Failed to save token usage: {:?}", e);
        }

        // Post event to SQS if configured
        if let Some(publisher) = &state_clone.sqs_publisher {
            let event = UsageEvent {
                event_id: Uuid::new_v4(),
                user_id,
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
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

/// Helper to encrypt an SSE event
async fn encrypt_sse_event(
    state: &AppState,
    session_id: &Uuid,
    json: &Value,
) -> Result<Event, ApiError> {
    let json_str = json.to_string();
    let encrypted_data = state
        .encrypt_session_data(session_id, json_str.as_bytes())
        .await
        .map_err(|e| {
            error!("Failed to encrypt SSE event data: {:?}", e);
            ApiError::InternalServerError
        })?;

    let base64_encrypted = general_purpose::STANDARD.encode(&encrypted_data);
    Ok(Event::default().data(base64_encrypted))
}

async fn proxy_models(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(_user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(_body): axum::Extension<()>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    debug!("Entering proxy_models function");

    // Use the proxy router to get all models from all configured proxies
    // The proxy router now handles caching internally with a 5-minute TTL
    let models_response = state.proxy_router.get_all_models().map_err(|e| {
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

    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for transcription: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use transcription feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use transcription without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
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

    // Send request with timeout
    match timeout(
        Duration::from_secs(REQUEST_TIMEOUT_SECS),
        client.request(req),
    )
    .await
    {
        Ok(Ok(res)) => {
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
        Ok(Err(e)) => Err(format!(
            "Failed to send request to {}: {:?}",
            provider.provider_name, e
        )),
        Err(_) => Err(format!(
            "Request to {} timed out after {}s",
            provider.provider_name, REQUEST_TIMEOUT_SECS
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

    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for TTS: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use TTS feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use TTS without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
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

    // Send request with timeout
    let res = timeout(
        Duration::from_secs(REQUEST_TIMEOUT_SECS),
        client.request(req),
    )
    .await
    .map_err(|_| {
        error!("TTS request timed out after {}s", REQUEST_TIMEOUT_SECS);
        ApiError::InternalServerError
    })?
    .map_err(|e| {
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

async fn proxy_embeddings(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    axum::Extension(session_id): axum::Extension<Uuid>,
    axum::Extension(user): axum::Extension<User>,
    axum::Extension(_auth_method): axum::Extension<AuthMethod>,
    axum::Extension(embedding_request): axum::Extension<EmbeddingRequest>,
) -> Result<Json<EncryptedResponse<Value>>, ApiError> {
    debug!("Entering proxy_embeddings function");

    // Check if guest user is allowed (paid guests are allowed, free guests are not)
    if user.is_guest() {
        if let Some(billing_client) = &state.billing_client {
            match billing_client.is_user_paid(user.uuid).await {
                Ok(true) => {
                    debug!("Paid guest user allowed for embeddings: {}", user.uuid);
                }
                Ok(false) => {
                    error!(
                        "Free guest user attempted to use embeddings feature: {}",
                        user.uuid
                    );
                    return Err(ApiError::Unauthorized);
                }
                Err(e) => {
                    error!("Billing check failed for guest user {}: {}", user.uuid, e);
                    return Err(ApiError::Unauthorized);
                }
            }
        } else {
            error!(
                "Guest user attempted to use embeddings without billing client: {}",
                user.uuid
            );
            return Err(ApiError::Unauthorized);
        }
    }

    // Validate input is not empty
    let is_empty = match &embedding_request.input {
        Value::String(s) => s.trim().is_empty(),
        Value::Array(arr) => arr.is_empty(),
        _ => true,
    };
    if is_empty {
        error!("Input is empty or invalid");
        return Err(ApiError::BadRequest);
    }

    // Get the model route configuration
    let route = match state.proxy_router.get_model_route(&embedding_request.model) {
        Some(r) => r,
        None => {
            error!(
                "Model '{}' not found in routing table",
                embedding_request.model
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

    // Build request body
    let request_body = serde_json::to_string(&embedding_request).map_err(|e| {
        error!("Failed to serialize embedding request: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Build request to provider
    let endpoint = format!("{}/v1/embeddings", route.primary.base_url);
    let mut req = Request::builder()
        .method("POST")
        .uri(&endpoint)
        .header("Content-Type", "application/json");

    if let Some(api_key) = &route.primary.api_key {
        if !api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }
    }

    let req = req.body(HyperBody::from(request_body)).map_err(|e| {
        error!("Failed to create request: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Send request with timeout
    let res = timeout(
        Duration::from_secs(REQUEST_TIMEOUT_SECS),
        client.request(req),
    )
    .await
    .map_err(|_| {
        error!(
            "Embeddings request timed out after {}s",
            REQUEST_TIMEOUT_SECS
        );
        ApiError::InternalServerError
    })?
    .map_err(|e| {
        error!("Failed to send embeddings request: {:?}", e);
        ApiError::InternalServerError
    })?;

    if !res.status().is_success() {
        let status = res.status();
        let body_bytes = to_bytes(res.into_body()).await.ok();
        let error_msg = body_bytes
            .map(|b| String::from_utf8_lossy(&b).to_string())
            .unwrap_or_else(|| status.to_string());
        error!(
            "Embeddings proxy returned non-success status: {} - {}",
            status, error_msg
        );
        return Err(ApiError::InternalServerError);
    }

    // Parse response
    let body_bytes = to_bytes(res.into_body()).await.map_err(|e| {
        error!("Failed to read embeddings response body: {:?}", e);
        ApiError::InternalServerError
    })?;

    let response_json: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
        error!("Failed to parse embeddings response: {:?}", e);
        ApiError::InternalServerError
    })?;

    // Handle billing - embeddings only have prompt_tokens (no completion_tokens)
    if let Some(usage) = response_json.get("usage") {
        let prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;

        if prompt_tokens > 0 {
            let billing_context =
                BillingContext::new(_auth_method, embedding_request.model.clone());
            let embedding_usage = CompletionUsage {
                prompt_tokens,
                completion_tokens: 0, // Embeddings don't have completion tokens
            };
            publish_usage_event_internal(
                &state,
                &user,
                &billing_context,
                embedding_usage,
                &route.primary.provider_name,
            )
            .await;
        }
    }

    debug!("Exiting proxy_embeddings function");

    // Encrypt and return the response
    encrypt_response(&state, &session_id, &response_json).await
}

/// Helper function to try a provider once
async fn try_provider(
    client: &Client<HttpsConnector<hyper::client::HttpConnector>, HyperBody>,
    proxy_config: &ProxyConfig,
    body_json: &str,
    headers: &HeaderMap,
    extra_headers: &[(HeaderName, HeaderValue)],
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

    // Add provider-specific headers
    for (name, val) in extra_headers {
        trace!(
            "provider={} extra header: {}={}",
            proxy_config.provider_name,
            name,
            val.to_str().unwrap_or("<non-ascii>")
        );
        req = req.header(name.clone(), val.clone());
    }

    trace!(
        "Sending to provider={} url={}/v1/chat/completions body_len={}",
        proxy_config.provider_name,
        proxy_config.base_url,
        body_json.len()
    );

    let req = req
        .body(HyperBody::from(body_json.to_string()))
        .map_err(|e| format!("Failed to create request body: {:?}", e))?;

    match timeout(
        Duration::from_secs(REQUEST_TIMEOUT_SECS),
        client.request(req),
    )
    .await
    {
        Ok(Ok(response)) => {
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
        Ok(Err(e)) => {
            error!(
                "Failed to send request to {}: {:?}",
                proxy_config.provider_name, e
            );
            Err(format!(
                "Failed to connect to {}: {}",
                proxy_config.provider_name, e
            ))
        }
        Err(_) => {
            error!(
                "Request to {} timed out after {}s",
                proxy_config.provider_name, REQUEST_TIMEOUT_SECS
            );
            Err(format!(
                "Request to {} timed out",
                proxy_config.provider_name
            ))
        }
    }
}
