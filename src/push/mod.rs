pub mod apns;
pub mod crypto;
pub mod fcm;
pub mod worker;

use crate::db::DBError;
use crate::encrypt::encrypt_with_key;
use crate::models::notification_deliveries::{NewNotificationDelivery, NotificationDeliveryError};
use crate::models::notification_events::{
    NewNotificationEvent, NotificationEvent, NotificationEventError,
    NOTIFICATION_DELIVERY_MODE_ENCRYPTED_PREVIEW, NOTIFICATION_DELIVERY_MODE_GENERIC,
    NOTIFICATION_KIND_AGENT_MESSAGE, NOTIFICATION_PRIORITY_HIGH, NOTIFICATION_PRIORITY_NORMAL,
};
use crate::models::project_settings::ProjectSettingError;
use crate::models::push_devices::{PushDevice, PushDeviceError};
use crate::models::users::User;
use crate::{AppState, Error};
use chrono::{DateTime, Duration, Utc};
use diesel::Connection;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::warn;
use uuid::Uuid;

pub const AGENT_NOTIFICATION_FALLBACK_TITLE: &str = "New Maple message";
pub const AGENT_NOTIFICATION_FALLBACK_BODY: &str = "Open Maple to view your encrypted message";
pub const APNS_JWT_CACHE_LIFETIME_MINUTES: i64 = 50;
pub const FCM_TOKEN_CACHE_SAFETY_MARGIN_SECONDS: i64 = 60;
pub const PUSH_PREVIEW_BODY_MAX_BYTES: usize = 180;

#[derive(Debug, Error)]
pub enum PushError {
    #[error("Database connection error")]
    ConnectionError,
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("Push device error: {0}")]
    PushDeviceError(#[from] PushDeviceError),
    #[error("Notification event error: {0}")]
    NotificationEventError(#[from] NotificationEventError),
    #[error("Notification delivery error: {0}")]
    NotificationDeliveryError(#[from] NotificationDeliveryError),
    #[error("Project settings error: {0}")]
    ProjectSettingError(#[from] ProjectSettingError),
    #[error("DB error: {0}")]
    DbError(#[from] DBError),
    #[error("Application error: {0}")]
    AppError(#[from] Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Encryption error: {0}")]
    EncryptionError(#[from] crate::encrypt::EncryptError),
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("JWT error: {0}")]
    JwtError(#[from] jsonwebtoken::errors::Error),
    #[error("Crypto error: {0}")]
    CryptoError(String),
    #[error("Invalid secret: {0}")]
    InvalidSecret(String),
    #[error("Provider error: {0}")]
    ProviderError(String),
    #[error("Retryable provider error: {0}")]
    ProviderRetryable(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushDeliveryMode {
    Generic,
    EncryptedPreview,
}

impl PushDeliveryMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Generic => NOTIFICATION_DELIVERY_MODE_GENERIC,
            Self::EncryptedPreview => NOTIFICATION_DELIVERY_MODE_ENCRYPTED_PREVIEW,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushPriority {
    #[allow(dead_code)]
    Normal,
    High,
}

impl PushPriority {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => NOTIFICATION_PRIORITY_NORMAL,
            Self::High => NOTIFICATION_PRIORITY_HIGH,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentPushTarget {
    Main,
    Subagent(Uuid),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreviewPayload {
    pub v: i32,
    pub notification_id: Uuid,
    pub message_id: Uuid,
    pub kind: String,
    pub title: String,
    pub body: String,
    pub deep_link: String,
    pub thread_id: String,
    pub sent_at: i64,
}

#[derive(Debug, Clone)]
pub struct NotificationPreviewPayloadInput {
    pub message_id: Uuid,
    pub kind: String,
    pub title: String,
    pub body: String,
    pub deep_link: String,
    pub thread_id: String,
    pub sent_at: i64,
}

fn normalize_preview_body(message_text: &str) -> String {
    let normalized = message_text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    truncate_utf8_with_ellipsis(&normalized, PUSH_PREVIEW_BODY_MAX_BYTES)
}

fn truncate_utf8_with_ellipsis(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }

    let ellipsis = "…";
    if max_bytes <= ellipsis.len() {
        return String::new();
    }

    let mut end = max_bytes - ellipsis.len();
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }

    let truncated = input[..end].trim_end();
    if truncated.is_empty() {
        String::new()
    } else {
        format!("{}{}", truncated, ellipsis)
    }
}

#[derive(Debug, Clone)]
pub struct EnqueueNotificationRequest {
    pub project_id: i32,
    pub user_id: Uuid,
    pub kind: String,
    pub delivery_mode: PushDeliveryMode,
    pub priority: PushPriority,
    pub fallback_title: String,
    pub fallback_body: String,
    pub preview_payload: Option<NotificationPreviewPayloadInput>,
    pub not_before_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedNotification {
    pub notification_id: Uuid,
    pub delivery_count: usize,
}

#[derive(Debug, Clone)]
pub enum PushSendOutcome {
    Sent {
        provider_message_id: Option<String>,
        provider_status_code: Option<i32>,
    },
    Retryable {
        provider_status_code: Option<i32>,
        error: String,
    },
    InvalidToken {
        provider_status_code: Option<i32>,
        error: String,
    },
    Failed {
        provider_status_code: Option<i32>,
        error: String,
    },
}

#[derive(Clone)]
pub(crate) struct PushTransport {
    pub client: reqwest::Client,
    pub apns_client: reqwest::Client,
    pub apns_tokens: Arc<RwLock<HashMap<String, CachedApnsToken>>>,
    pub fcm_tokens: Arc<RwLock<HashMap<String, CachedFcmToken>>>,
}

impl PushTransport {
    pub fn new() -> Result<Self, PushError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()?;
        let apns_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .http2_prior_knowledge()
            .build()?;

        Ok(Self {
            client,
            apns_client,
            apns_tokens: Arc::new(RwLock::new(HashMap::new())),
            fcm_tokens: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CachedApnsToken {
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedFcmToken {
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

pub async fn enqueue_notification(
    state: &Arc<AppState>,
    request: EnqueueNotificationRequest,
) -> Result<Option<QueuedNotification>, PushError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| PushError::ConnectionError)?;

    let devices = PushDevice::list_active_for_user(&mut conn, request.user_id)?;
    if devices.is_empty() {
        return Ok(None);
    }

    let notification_id = Uuid::new_v4();
    let payload_enc = if let Some(preview_payload) = request.preview_payload {
        let payload = NotificationPreviewPayload {
            v: 1,
            notification_id,
            message_id: preview_payload.message_id,
            kind: preview_payload.kind,
            title: preview_payload.title,
            body: preview_payload.body,
            deep_link: preview_payload.deep_link,
            thread_id: preview_payload.thread_id,
            sent_at: preview_payload.sent_at,
        };
        let enclave_key = SecretKey::from_slice(&state.enclave_key)
            .map_err(|e| PushError::InvalidSecret(e.to_string()))?;
        let payload_bytes = serde_json::to_vec(&payload)?;
        Some(encrypt_with_key(&enclave_key, &payload_bytes).await)
    } else {
        None
    };

    let event = conn.transaction::<NotificationEvent, PushError, _>(|conn| {
        let event = NewNotificationEvent {
            uuid: notification_id,
            project_id: request.project_id,
            user_id: request.user_id,
            kind: request.kind,
            delivery_mode: request.delivery_mode.as_str().to_string(),
            priority: request.priority.as_str().to_string(),
            collapse_key: Some(format!("notif:{}", notification_id)),
            fallback_title: request.fallback_title,
            fallback_body: request.fallback_body,
            payload_enc,
            not_before_at: request.not_before_at,
            expires_at: request.expires_at,
        }
        .insert(conn)?;

        let new_deliveries: Vec<NewNotificationDelivery> = devices
            .iter()
            .map(|device| NewNotificationDelivery {
                event_id: event.id,
                push_device_id: device.id,
                next_attempt_at: event.not_before_at,
            })
            .collect();

        if new_deliveries.is_empty() {
            warn!(
                "Notification {} had no active push deliveries to insert",
                event.uuid
            );
        } else {
            NewNotificationDelivery::insert_many(conn, &new_deliveries)?;
        }

        Ok(event)
    })?;

    Ok(Some(QueuedNotification {
        notification_id: event.uuid,
        delivery_count: devices.len(),
    }))
}

pub async fn enqueue_agent_message_notification(
    state: &Arc<AppState>,
    user: &User,
    target: AgentPushTarget,
    message_id: Uuid,
    message_text: &str,
) -> Result<Option<QueuedNotification>, PushError> {
    if message_text.trim().is_empty() {
        return Ok(None);
    }

    let push_settings = state
        .db
        .get_project_push_settings(user.project_id)?
        .unwrap_or_default();
    let delivery_mode = if push_settings.encrypted_preview_enabled {
        PushDeliveryMode::EncryptedPreview
    } else {
        PushDeliveryMode::Generic
    };

    let (deep_link, thread_id) = match target {
        AgentPushTarget::Main => ("opensecret://agent".to_string(), "agent:main".to_string()),
        AgentPushTarget::Subagent(agent_uuid) => (
            format!("opensecret://agent/subagent/{}", agent_uuid),
            format!("agent:subagent:{}", agent_uuid),
        ),
    };

    enqueue_notification(
        state,
        EnqueueNotificationRequest {
            project_id: user.project_id,
            user_id: user.uuid,
            kind: NOTIFICATION_KIND_AGENT_MESSAGE.to_string(),
            delivery_mode,
            priority: PushPriority::High,
            fallback_title: AGENT_NOTIFICATION_FALLBACK_TITLE.to_string(),
            fallback_body: AGENT_NOTIFICATION_FALLBACK_BODY.to_string(),
            preview_payload: Some(NotificationPreviewPayloadInput {
                message_id,
                kind: NOTIFICATION_KIND_AGENT_MESSAGE.to_string(),
                title: "Maple".to_string(),
                body: normalize_preview_body(message_text),
                deep_link,
                thread_id,
                sent_at: Utc::now().timestamp(),
            }),
            not_before_at: None,
            expires_at: Some(Utc::now() + Duration::days(7)),
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::{normalize_preview_body, PUSH_PREVIEW_BODY_MAX_BYTES};

    #[test]
    fn normalize_preview_body_collapses_whitespace() {
        assert_eq!(
            normalize_preview_body("hello\n\nthere   world"),
            "hello there world"
        );
    }

    #[test]
    fn normalize_preview_body_truncates_to_budget() {
        let input = "a".repeat(PUSH_PREVIEW_BODY_MAX_BYTES + 50);
        let output = normalize_preview_body(&input);

        assert!(output.len() <= PUSH_PREVIEW_BODY_MAX_BYTES);
        assert!(output.ends_with('…'));
    }
}
