use crate::models::notification_events::NotificationEvent;
use crate::models::project_settings::IosPushSettings;
use crate::models::push_devices::PushDevice;
use crate::push::crypto::encrypt_preview_payload;
use crate::push::{
    CachedApnsToken, NotificationPreviewPayload, PushError, PushSendOutcome, PushTransport,
    APNS_JWT_CACHE_LIFETIME_MINUTES,
};
use crate::web::platform::PROJECT_APNS_AUTH_KEY_P8;
use crate::AppState;
use chrono::{Duration, Utc};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Debug, Serialize)]
struct ApnsJwtClaims {
    iss: String,
    iat: i64,
}

pub struct ApnsSendRequest<'a> {
    pub event: &'a NotificationEvent,
    pub device: &'a PushDevice,
    pub push_token: &'a str,
    pub ios_settings: &'a IosPushSettings,
    pub preview_payload: Option<&'a NotificationPreviewPayload>,
    pub send_encrypted_preview: bool,
}

pub async fn send_apns_notification(
    state: &Arc<AppState>,
    transport: &PushTransport,
    request: ApnsSendRequest<'_>,
) -> Result<PushSendOutcome, PushError> {
    let auth_token = get_apns_auth_token(
        state,
        transport,
        request.event.project_id,
        request.ios_settings,
    )
    .await?;
    let body = build_apns_payload(
        request.event,
        request.device,
        request.preview_payload,
        request.send_encrypted_preview,
    )?;
    let endpoint = match request.ios_settings.apns_environment.as_str() {
        "dev" => format!(
            "https://api.sandbox.push.apple.com/3/device/{}",
            request.push_token
        ),
        _ => format!("https://api.push.apple.com/3/device/{}", request.push_token),
    };

    let mut http_request = transport
        .client
        .post(endpoint)
        .bearer_auth(auth_token)
        .header("apns-topic", request.ios_settings.bundle_id.as_str())
        .header("apns-push-type", "alert")
        .header(
            "apns-priority",
            if request.event.priority == "high" {
                "10"
            } else {
                "5"
            },
        );

    if let Some(collapse_key) = &request.event.collapse_key {
        http_request = http_request.header("apns-collapse-id", collapse_key);
    }

    if let Some(expires_at) = request.event.expires_at {
        http_request = http_request.header("apns-expiration", expires_at.timestamp().to_string());
    }

    let response = http_request.json(&body).send().await?;
    let status_code = response.status().as_u16() as i32;
    let apns_id = response
        .headers()
        .get("apns-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let status = response.status();
    let response_body = response.text().await.unwrap_or_default();
    let reason = parse_apns_reason(&response_body);

    if status.is_success() {
        return Ok(PushSendOutcome::Sent {
            provider_message_id: apns_id,
            provider_status_code: Some(status_code),
        });
    }

    if matches!(
        reason.as_deref(),
        Some("ExpiredProviderToken" | "InvalidProviderToken")
    ) {
        invalidate_apns_cache(transport, request.event.project_id, request.ios_settings).await;
    }

    let error_message = reason.unwrap_or_else(|| response_body.clone());
    let outcome = match status.as_u16() {
        410 => PushSendOutcome::InvalidToken {
            provider_status_code: Some(status_code),
            error: if error_message.is_empty() {
                "Unregistered".to_string()
            } else {
                error_message
            },
        },
        403 if matches!(
            error_message.as_str(),
            "ExpiredProviderToken" | "InvalidProviderToken"
        ) =>
        {
            PushSendOutcome::Retryable {
                provider_status_code: Some(status_code),
                error: error_message,
            }
        }
        429 | 500 | 503 => PushSendOutcome::Retryable {
            provider_status_code: Some(status_code),
            error: error_message,
        },
        _ => PushSendOutcome::Failed {
            provider_status_code: Some(status_code),
            error: error_message,
        },
    };

    Ok(outcome)
}

async fn get_apns_auth_token(
    state: &Arc<AppState>,
    transport: &PushTransport,
    project_id: i32,
    ios_settings: &IosPushSettings,
) -> Result<String, PushError> {
    let cache_key = format!(
        "{}:{}:{}",
        project_id, ios_settings.team_id, ios_settings.key_id
    );
    {
        let cache = transport.apns_tokens.read().await;
        if let Some(cached) = cache.get(&cache_key) {
            if cached.expires_at > Utc::now() + Duration::minutes(5) {
                return Ok(cached.token.clone());
            }
        }
    }

    let private_key_pem = state
        .get_project_secret_string(project_id, PROJECT_APNS_AUTH_KEY_P8)
        .await?
        .ok_or_else(|| PushError::InvalidSecret("missing APNs auth key".to_string()))?;

    let mut header = Header::new(Algorithm::ES256);
    header.kid = Some(ios_settings.key_id.clone());

    let claims = ApnsJwtClaims {
        iss: ios_settings.team_id.clone(),
        iat: Utc::now().timestamp() - 30,
    };
    let signing_key = EncodingKey::from_ec_pem(private_key_pem.as_bytes())
        .map_err(|e| PushError::InvalidSecret(e.to_string()))?;
    let token = encode(&header, &claims, &signing_key)?;
    let expires_at = Utc::now() + Duration::minutes(APNS_JWT_CACHE_LIFETIME_MINUTES);

    let mut cache = transport.apns_tokens.write().await;
    cache.insert(
        cache_key,
        CachedApnsToken {
            token: token.clone(),
            expires_at,
        },
    );

    Ok(token)
}

async fn invalidate_apns_cache(
    transport: &PushTransport,
    project_id: i32,
    ios_settings: &IosPushSettings,
) {
    let cache_key = format!(
        "{}:{}:{}",
        project_id, ios_settings.team_id, ios_settings.key_id
    );
    let mut cache = transport.apns_tokens.write().await;
    cache.remove(&cache_key);
}

fn build_apns_payload(
    event: &NotificationEvent,
    device: &PushDevice,
    preview_payload: Option<&NotificationPreviewPayload>,
    send_encrypted_preview: bool,
) -> Result<Value, PushError> {
    let metadata = preview_payload.map(|payload| {
        json!({
            "notification_id": payload.notification_id.to_string(),
            "kind": payload.kind,
            "message_id": payload.message_id.to_string(),
            "thread_id": payload.thread_id,
            "deep_link": payload.deep_link,
        })
    });

    let mut root = json!({
        "aps": {
            "alert": {
                "title": event.fallback_title,
                "body": event.fallback_body,
            },
            "sound": "default",
            "mutable-content": 1,
        }
    });

    if let Some(payload) = preview_payload {
        if let Some(aps) = root.get_mut("aps").and_then(|value| value.as_object_mut()) {
            aps.insert(
                "thread-id".to_string(),
                Value::String(payload.thread_id.clone()),
            );
        }
    }

    if let Some(metadata) = metadata {
        root.as_object_mut()
            .expect("root payload must be an object")
            .insert("os_meta".to_string(), metadata);
    }

    if send_encrypted_preview {
        if let Some(payload) = preview_payload {
            let envelope = encrypt_preview_payload(device, payload)?;
            root.as_object_mut()
                .expect("root payload must be an object")
                .insert("os_push".to_string(), serde_json::to_value(envelope)?);
        }
    }

    Ok(root)
}

fn parse_apns_reason(response_body: &str) -> Option<String> {
    serde_json::from_str::<Value>(response_body)
        .ok()
        .and_then(|value| {
            value
                .get("reason")
                .and_then(|reason| reason.as_str())
                .map(str::to_owned)
        })
}
