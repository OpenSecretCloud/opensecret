use axum::http::HeaderMap;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error};

use crate::models::users::User;
use crate::web::openai::{get_chat_completion_response, BillingContext, CompletionChunk};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};

const DEFAULT_VISION_MAX_TOKENS: u32 = 2048;

const VISION_SYSTEM_PROMPT: &str =
    "You are an image description agent. Your ONLY job is to describe the \
        image the user sent in extreme detail with as much accuracy as possible. \
        Describe everything you see: objects, people, text, colors, layout, \
        emotions, context, setting, lighting, and any other relevant details. \
        Be thorough but organized. If there is text in the image, transcribe it exactly. \
        Recent conversation context is provided so you can understand what the user \
        might be referring to - use it to make your description more relevant, \
        but your primary job is accurate visual description. \
        Output ONLY the description, nothing else.";

pub struct VisionRequest<'a> {
    pub auth_method: AuthMethod,
    pub request_model: &'a str,
    pub billing_model: &'a str,
    pub image_url: &'a str,
    pub user_message: &'a str,
    pub recent_messages: &'a str,
}

pub async fn describe_image(
    state: &Arc<AppState>,
    user: &User,
    request: VisionRequest<'_>,
) -> Result<String, ApiError> {
    if request.image_url.trim().is_empty() {
        return Err(ApiError::BadRequest);
    }

    let mut text_parts: Vec<String> = Vec::new();
    if !request.recent_messages.trim().is_empty() {
        text_parts.push(format!(
            "Recent conversation for context:\n{}",
            request.recent_messages.trim()
        ));
    }
    if !request.user_message.trim().is_empty() {
        text_parts.push(format!(
            "The user sent this message alongside the image: \"{}\"",
            request.user_message.trim()
        ));
    }
    text_parts.push("Describe this image in detail.".to_string());

    let user_content: Vec<Value> = vec![
        json!({
            "type": "image_url",
            "image_url": { "url": request.image_url }
        }),
        json!({
            "type": "text",
            "text": text_parts.join("\n\n")
        }),
    ];

    let body = json!({
        "model": request.request_model,
        "stream": false,
        "messages": [
            { "role": "system", "content": VISION_SYSTEM_PROMPT },
            { "role": "user", "content": user_content }
        ],
        "max_tokens": DEFAULT_VISION_MAX_TOKENS,
    });

    debug!(
        "Vision pre-processing: calling request model {} (billing model {})",
        request.request_model, request.billing_model
    );

    let headers = HeaderMap::new();
    let billing_context =
        BillingContext::new(request.auth_method, request.billing_model.to_string());
    let completion = get_chat_completion_response(state, user, body, &headers, billing_context)
        .await
        .map_err(|e| {
            error!("Vision model call failed: {e:?}");
            e
        })?;

    if completion.metadata.is_streaming {
        error!("Vision pre-processing returned streaming response unexpectedly");
        return Err(ApiError::InternalServerError);
    }

    let mut rx = completion.stream;
    let response_json = match rx.recv().await {
        Some(CompletionChunk::FullResponse(v)) => v,
        Some(CompletionChunk::Error(msg)) => {
            error!("Vision model error: {}", msg);
            return Err(ApiError::InternalServerError);
        }
        other => {
            error!("Unexpected vision completion chunk: {:?}", other);
            return Err(ApiError::InternalServerError);
        }
    };

    Ok(extract_assistant_content(&response_json)
        .unwrap_or_else(|| "[Could not describe image]".to_string()))
}

fn extract_assistant_content(response_json: &Value) -> Option<String> {
    response_json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}
