use crate::models::notification_events::NotificationEvent;
use crate::models::project_settings::AndroidPushSettings;
use crate::models::push_devices::PushDevice;
use crate::push::{
    CachedFcmToken, NotificationPreviewPayload, PushError, PushSendOutcome, PushTransport,
    FCM_TOKEN_CACHE_SAFETY_MARGIN_SECONDS,
};
use crate::web::platform::PROJECT_FCM_SERVICE_ACCOUNT_JSON;
use crate::AppState;
use chrono::{Duration, Utc};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

const GOOGLE_TOKEN_SCOPE: &str = "https://www.googleapis.com/auth/firebase.messaging";
const DEFAULT_GOOGLE_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

#[derive(Debug, Deserialize)]
struct FcmServiceAccount {
    pub client_email: String,
    pub private_key: String,
    pub project_id: Option<String>,
    pub token_uri: Option<String>,
}

#[derive(Debug, Serialize)]
struct FcmJwtClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    exp: i64,
    iat: i64,
}

#[derive(Debug, Deserialize)]
struct FcmAccessTokenResponse {
    access_token: String,
    expires_in: i64,
}

#[derive(Debug, Deserialize)]
struct FcmSendResponse {
    name: String,
}

#[derive(Debug, Deserialize)]
struct GoogleApiErrorEnvelope {
    error: GoogleApiError,
}

#[derive(Debug, Deserialize)]
struct GoogleApiError {
    status: Option<String>,
    message: Option<String>,
    details: Option<Vec<Value>>,
}

pub async fn send_fcm_notification(
    state: &Arc<AppState>,
    transport: &PushTransport,
    event: &NotificationEvent,
    _device: &PushDevice,
    push_token: &str,
    android_settings: &AndroidPushSettings,
    preview_payload: Option<&NotificationPreviewPayload>,
) -> Result<PushSendOutcome, PushError> {
    let service_account = load_fcm_service_account(state, event.project_id).await?;
    let project_id = if android_settings.firebase_project_id.trim().is_empty() {
        service_account
            .project_id
            .clone()
            .ok_or_else(|| PushError::InvalidSecret("missing FCM project_id".to_string()))?
    } else {
        android_settings.firebase_project_id.clone()
    };
    let access_token = get_fcm_access_token(transport, event.project_id, &service_account).await?;
    let body = build_fcm_payload(event, push_token, preview_payload);
    let response = transport
        .client
        .post(format!(
            "https://fcm.googleapis.com/v1/projects/{}/messages:send",
            project_id
        ))
        .bearer_auth(access_token)
        .json(&body)
        .send()
        .await?;

    let status_code = response.status().as_u16() as i32;
    let status = response.status();
    let response_body = response.text().await.unwrap_or_default();

    if status.is_success() {
        let parsed: FcmSendResponse = serde_json::from_str(&response_body)?;
        return Ok(PushSendOutcome::Sent {
            provider_message_id: Some(parsed.name),
            provider_status_code: Some(status_code),
        });
    }

    let parsed_error = serde_json::from_str::<GoogleApiErrorEnvelope>(&response_body).ok();
    let error_status = parsed_error
        .as_ref()
        .and_then(|body| body.error.status.clone())
        .unwrap_or_default();
    let error_message = parsed_error
        .as_ref()
        .and_then(|body| body.error.message.clone())
        .unwrap_or_else(|| response_body.clone());
    let fcm_error_code = parsed_error
        .as_ref()
        .and_then(|body| extract_fcm_error_code(body.error.details.as_ref()));

    if matches!(status.as_u16(), 401 | 403) {
        invalidate_fcm_cache(transport, event.project_id, &service_account.client_email).await;
    }

    let outcome = if matches!(fcm_error_code.as_deref(), Some("UNREGISTERED"))
        || is_invalid_registration(&error_status, &error_message)
    {
        PushSendOutcome::InvalidToken {
            provider_status_code: Some(status_code),
            error: if let Some(code) = fcm_error_code {
                format!("{}: {}", code, error_message)
            } else {
                error_message
            },
        }
    } else if matches!(status.as_u16(), 401 | 403 | 429 | 500 | 503) {
        PushSendOutcome::Retryable {
            provider_status_code: Some(status_code),
            error: error_message,
        }
    } else {
        PushSendOutcome::Failed {
            provider_status_code: Some(status_code),
            error: error_message,
        }
    };

    Ok(outcome)
}

async fn load_fcm_service_account(
    state: &Arc<AppState>,
    project_id: i32,
) -> Result<FcmServiceAccount, PushError> {
    let raw = state
        .get_project_secret_string(project_id, PROJECT_FCM_SERVICE_ACCOUNT_JSON)
        .await?
        .ok_or_else(|| PushError::InvalidSecret("missing FCM service account JSON".to_string()))?;

    serde_json::from_str(&raw).map_err(PushError::from)
}

async fn get_fcm_access_token(
    transport: &PushTransport,
    project_id: i32,
    service_account: &FcmServiceAccount,
) -> Result<String, PushError> {
    let cache_key = format!("{}:{}", project_id, service_account.client_email);
    {
        let cache = transport.fcm_tokens.read().await;
        if let Some(cached) = cache.get(&cache_key) {
            if cached.expires_at
                > Utc::now() + Duration::seconds(FCM_TOKEN_CACHE_SAFETY_MARGIN_SECONDS)
            {
                return Ok(cached.token.clone());
            }
        }
    }

    let token_uri = service_account
        .token_uri
        .as_deref()
        .unwrap_or(DEFAULT_GOOGLE_TOKEN_URI);
    let now = Utc::now().timestamp();
    let claims = FcmJwtClaims {
        iss: &service_account.client_email,
        scope: GOOGLE_TOKEN_SCOPE,
        aud: token_uri,
        exp: now + 3600,
        iat: now - 30,
    };
    let header = Header::new(Algorithm::RS256);
    let signing_key = EncodingKey::from_rsa_pem(service_account.private_key.as_bytes())
        .map_err(|e| PushError::InvalidSecret(e.to_string()))?;
    let assertion = encode(&header, &claims, &signing_key)?;

    let response = transport
        .client
        .post(token_uri)
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", assertion.as_str()),
        ])
        .send()
        .await?;
    let response_status = response.status();
    let response_body = response.text().await.unwrap_or_default();

    if !response_status.is_success() {
        return Err(PushError::ProviderError(format!(
            "FCM OAuth failed ({}): {}",
            response_status, response_body
        )));
    }

    let token_response: FcmAccessTokenResponse = serde_json::from_str(&response_body)?;
    let expires_at = Utc::now() + Duration::seconds(token_response.expires_in);
    let token = token_response.access_token;

    let mut cache = transport.fcm_tokens.write().await;
    cache.insert(
        cache_key,
        CachedFcmToken {
            token: token.clone(),
            expires_at,
        },
    );

    Ok(token)
}

async fn invalidate_fcm_cache(transport: &PushTransport, project_id: i32, client_email: &str) {
    let cache_key = format!("{}:{}", project_id, client_email);
    let mut cache = transport.fcm_tokens.write().await;
    cache.remove(&cache_key);
}

fn build_fcm_payload(
    event: &NotificationEvent,
    push_token: &str,
    preview_payload: Option<&NotificationPreviewPayload>,
) -> Value {
    let ttl_seconds = event
        .expires_at
        .map(|expires_at| (expires_at - Utc::now()).num_seconds().max(0))
        .unwrap_or(60 * 60 * 24 * 7);
    let collapse_key = event
        .collapse_key
        .clone()
        .unwrap_or_else(|| format!("notif:{}", event.uuid));

    let data = if let Some(payload) = preview_payload {
        json!({
            "notification_id": payload.notification_id.to_string(),
            "kind": payload.kind,
            "message_id": payload.message_id.to_string(),
            "thread_id": payload.thread_id,
            "deep_link": payload.deep_link,
        })
    } else {
        json!({
            "notification_id": event.uuid.to_string(),
            "kind": event.kind,
        })
    };

    json!({
        "message": {
            "token": push_token,
            "notification": {
                "title": event.fallback_title,
                "body": event.fallback_body,
            },
            "data": data,
            "android": {
                "priority": if event.priority == "high" { "HIGH" } else { "NORMAL" },
                "ttl": format!("{}s", ttl_seconds),
                "collapse_key": collapse_key,
                "notification": {
                    "channel_id": "sage_messages",
                    "tag": event.uuid.to_string(),
                    "click_action": "OPEN_THREAD",
                }
            }
        }
    })
}

fn extract_fcm_error_code(details: Option<&Vec<Value>>) -> Option<String> {
    details.and_then(|entries| {
        entries.iter().find_map(|entry| {
            entry
                .get("errorCode")
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        })
    })
}

fn is_invalid_registration(error_status: &str, error_message: &str) -> bool {
    error_status == "INVALID_ARGUMENT"
        && error_message
            .to_ascii_lowercase()
            .contains("registration token")
}
