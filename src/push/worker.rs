use crate::encrypt::decrypt_with_key;
use crate::models::notification_deliveries::{
    NotificationDelivery, NotificationDeliveryWriteResult,
};
use crate::models::notification_events::NotificationEvent;
use crate::models::push_devices::{PushDevice, PUSH_PLATFORM_ANDROID, PUSH_PLATFORM_IOS};
use crate::push::apns::{send_apns_notification, ApnsSendRequest};
use crate::push::fcm::send_fcm_notification;
use crate::push::{NotificationPreviewPayload, PushError, PushSendOutcome, PushTransport};
use crate::AppState;
use chrono::Utc;
use diesel::Connection;
use futures::stream::{self, StreamExt};
use secp256k1::SecretKey;
use std::sync::Arc;
use tokio::time::{sleep, Duration as TokioDuration};
use tracing::{debug, error};
use uuid::Uuid;

const PUSH_WORKER_BATCH_SIZE: i64 = 32;
const PUSH_WORKER_LEASE_TTL_SECONDS: i32 = 60;
const PUSH_WORKER_POLL_INTERVAL_SECONDS: u64 = 3;
const PUSH_WORKER_MAX_CONCURRENCY: usize = 8;
const PUSH_WORKER_MAX_ATTEMPTS: i32 = 8;

pub fn start_push_worker(state: Arc<AppState>) {
    tokio::spawn(async move {
        let transport = match PushTransport::new() {
            Ok(transport) => transport,
            Err(error) => {
                error!("failed to initialize push transport: {:?}", error);
                return;
            }
        };

        loop {
            if let Err(error) = process_push_batch(&state, &transport).await {
                error!("push worker batch failed: {:?}", error);
            }

            sleep(TokioDuration::from_secs(PUSH_WORKER_POLL_INTERVAL_SECONDS)).await;
        }
    });
}

async fn process_push_batch(
    state: &Arc<AppState>,
    transport: &PushTransport,
) -> Result<(), PushError> {
    let lease_owner = format!("push-worker:{}:{}", std::process::id(), Uuid::new_v4());
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| PushError::ConnectionError)?;
    let deliveries = NotificationDelivery::lease_pending(
        &mut conn,
        PUSH_WORKER_BATCH_SIZE,
        &lease_owner,
        PUSH_WORKER_LEASE_TTL_SECONDS,
    )?;
    drop(conn);

    if deliveries.is_empty() {
        return Ok(());
    }

    debug!("leased {} push deliveries", deliveries.len());

    stream::iter(deliveries)
        .for_each_concurrent(PUSH_WORKER_MAX_CONCURRENCY, |delivery| {
            let state = state.clone();
            let transport = transport.clone();
            async move {
                if let Err(error) = process_leased_delivery(&state, &transport, delivery).await {
                    error!("push delivery processing failed: {:?}", error);
                }
            }
        })
        .await;

    Ok(())
}

async fn process_leased_delivery(
    state: &Arc<AppState>,
    transport: &PushTransport,
    delivery: NotificationDelivery,
) -> Result<(), PushError> {
    let Some(lease_owner) = delivery.lease_owner.clone() else {
        error!(
            "leased push delivery {} is missing lease_owner; waiting for lease expiry",
            delivery.id
        );
        return Ok(());
    };

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| PushError::ConnectionError)?;
    let event = match NotificationEvent::get_by_id(&mut conn, delivery.event_id)? {
        Some(event) => event,
        None => {
            if !record_delivery_transition(
                NotificationDelivery::mark_failed(
                    &mut conn,
                    delivery.id,
                    &lease_owner,
                    None,
                    Some("event missing"),
                )?,
                delivery.id,
                &lease_owner,
                "failed (event missing)",
            ) {
                return Ok(());
            }
            return Ok(());
        }
    };

    if event.cancelled_at.is_some() {
        if !record_delivery_transition(
            NotificationDelivery::mark_cancelled(
                &mut conn,
                delivery.id,
                &lease_owner,
                Some("event cancelled"),
            )?,
            delivery.id,
            &lease_owner,
            "cancelled (event cancelled)",
        ) {
            return Ok(());
        }
        return Ok(());
    }

    if event
        .expires_at
        .is_some_and(|expires_at| expires_at <= Utc::now())
    {
        if !record_delivery_transition(
            NotificationDelivery::mark_cancelled(
                &mut conn,
                delivery.id,
                &lease_owner,
                Some("event expired"),
            )?,
            delivery.id,
            &lease_owner,
            "cancelled (event expired)",
        ) {
            return Ok(());
        }
        return Ok(());
    }

    let device = match PushDevice::get_by_id(&mut conn, delivery.push_device_id)? {
        Some(device) => device,
        None => {
            if !record_delivery_transition(
                NotificationDelivery::mark_failed(
                    &mut conn,
                    delivery.id,
                    &lease_owner,
                    None,
                    Some("device missing"),
                )?,
                delivery.id,
                &lease_owner,
                "failed (device missing)",
            ) {
                return Ok(());
            }
            return Ok(());
        }
    };

    if device.user_id != event.user_id {
        if !record_delivery_transition(
            NotificationDelivery::mark_cancelled(
                &mut conn,
                delivery.id,
                &lease_owner,
                Some("device user does not match event user"),
            )?,
            delivery.id,
            &lease_owner,
            "cancelled (device user mismatch)",
        ) {
            return Ok(());
        }
        return Ok(());
    }

    if device.revoked_at.is_some() {
        if !record_delivery_transition(
            NotificationDelivery::mark_cancelled(
                &mut conn,
                delivery.id,
                &lease_owner,
                Some("device revoked"),
            )?,
            delivery.id,
            &lease_owner,
            "cancelled (device revoked)",
        ) {
            return Ok(());
        }
        return Ok(());
    }
    drop(conn);

    let send_outcome = match build_send_outcome(state, transport, &event, &device).await {
        Ok(outcome) => outcome,
        Err(error) => {
            error!(
                "push delivery {} encountered internal send error before provider outcome: {:?}",
                delivery.id, error
            );
            classify_internal_push_error(error)
        }
    };

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| PushError::ConnectionError)?;
    match send_outcome {
        PushSendOutcome::Sent {
            provider_message_id,
            provider_status_code,
        } => {
            record_delivery_transition(
                NotificationDelivery::mark_sent(
                    &mut conn,
                    delivery.id,
                    &lease_owner,
                    provider_message_id.as_deref(),
                    provider_status_code,
                )?,
                delivery.id,
                &lease_owner,
                "sent",
            );
        }
        PushSendOutcome::Retryable {
            provider_status_code,
            error,
        } => {
            if delivery.attempt_count + 1 >= PUSH_WORKER_MAX_ATTEMPTS {
                record_delivery_transition(
                    NotificationDelivery::mark_failed(
                        &mut conn,
                        delivery.id,
                        &lease_owner,
                        provider_status_code,
                        Some(&error),
                    )?,
                    delivery.id,
                    &lease_owner,
                    "failed (retry limit reached)",
                );
            } else {
                record_delivery_transition(
                    NotificationDelivery::mark_retry(
                        &mut conn,
                        delivery.id,
                        &lease_owner,
                        provider_status_code,
                        Some(&error),
                        retry_backoff_seconds(delivery.attempt_count + 1),
                    )?,
                    delivery.id,
                    &lease_owner,
                    "retry",
                );
            }
        }
        PushSendOutcome::InvalidToken {
            provider_status_code,
            error,
        } => {
            let invalidated = conn.transaction::<bool, PushError, _>(|conn| {
                let write_result = NotificationDelivery::mark_invalid_token(
                    conn,
                    delivery.id,
                    &lease_owner,
                    provider_status_code,
                    Some(&error),
                )?;

                if write_result.was_applied() {
                    PushDevice::invalidate(conn, device.id)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            })?;

            if !invalidated {
                debug!(
                    "push delivery {} lost lease before marking invalid_token (lease_owner={})",
                    delivery.id, lease_owner
                );
            }
        }
        PushSendOutcome::Failed {
            provider_status_code,
            error,
        } => {
            record_delivery_transition(
                NotificationDelivery::mark_failed(
                    &mut conn,
                    delivery.id,
                    &lease_owner,
                    provider_status_code,
                    Some(&error),
                )?,
                delivery.id,
                &lease_owner,
                "failed",
            );
        }
    }

    Ok(())
}

async fn build_send_outcome(
    state: &Arc<AppState>,
    transport: &PushTransport,
    event: &NotificationEvent,
    device: &PushDevice,
) -> Result<PushSendOutcome, PushError> {
    let preview_payload = decrypt_preview_payload(state, event)?;
    let push_token = decrypt_push_token(state, device)?;

    dispatch_delivery(
        state,
        transport,
        event,
        device,
        &push_token,
        preview_payload.as_ref(),
    )
    .await
}

async fn dispatch_delivery(
    state: &Arc<AppState>,
    transport: &PushTransport,
    event: &NotificationEvent,
    device: &PushDevice,
    push_token: &str,
    preview_payload: Option<&NotificationPreviewPayload>,
) -> Result<PushSendOutcome, PushError> {
    let push_settings = state
        .db
        .get_project_push_settings(event.project_id)?
        .unwrap_or_default();

    match device.platform.as_str() {
        PUSH_PLATFORM_IOS => {
            let Some(ios_settings) = push_settings
                .ios
                .as_ref()
                .filter(|settings| settings.enabled)
            else {
                return Ok(PushSendOutcome::Failed {
                    provider_status_code: None,
                    error: "iOS push is not configured for this project".to_string(),
                });
            };

            if device.app_id != ios_settings.bundle_id {
                return Ok(PushSendOutcome::Failed {
                    provider_status_code: None,
                    error: "device app_id does not match configured iOS bundle_id".to_string(),
                });
            }

            if device.environment != ios_settings.apns_environment.as_str() {
                return Ok(PushSendOutcome::Failed {
                    provider_status_code: None,
                    error: "device environment does not match project APNs environment".to_string(),
                });
            }

            let send_encrypted_preview = push_settings.encrypted_preview_enabled
                && event.delivery_mode == "encrypted_preview"
                && preview_payload.is_some()
                && device.supports_encrypted_preview;

            send_apns_notification(
                state,
                transport,
                ApnsSendRequest {
                    event,
                    device,
                    push_token,
                    ios_settings,
                    preview_payload,
                    send_encrypted_preview,
                },
            )
            .await
        }
        PUSH_PLATFORM_ANDROID => {
            let Some(android_settings) = push_settings
                .android
                .as_ref()
                .filter(|settings| settings.enabled)
            else {
                return Ok(PushSendOutcome::Failed {
                    provider_status_code: None,
                    error: "Android push is not configured for this project".to_string(),
                });
            };

            if device.app_id != android_settings.package_name {
                return Ok(PushSendOutcome::Failed {
                    provider_status_code: None,
                    error: "device app_id does not match configured Android package_name"
                        .to_string(),
                });
            }

            send_fcm_notification(
                state,
                transport,
                event,
                device,
                push_token,
                android_settings,
                preview_payload,
            )
            .await
        }
        _ => Ok(PushSendOutcome::Failed {
            provider_status_code: None,
            error: format!("unsupported push platform: {}", device.platform),
        }),
    }
}

fn decrypt_preview_payload(
    state: &Arc<AppState>,
    event: &NotificationEvent,
) -> Result<Option<NotificationPreviewPayload>, PushError> {
    let Some(payload_enc) = &event.payload_enc else {
        return Ok(None);
    };

    let secret_key = SecretKey::from_slice(&state.enclave_key)
        .map_err(|e| PushError::InvalidSecret(e.to_string()))?;
    let plaintext = decrypt_with_key(&secret_key, payload_enc)
        .map_err(|e| PushError::CryptoError(e.to_string()))?;
    let payload = serde_json::from_slice::<NotificationPreviewPayload>(&plaintext)?;

    Ok(Some(payload))
}

fn decrypt_push_token(state: &Arc<AppState>, device: &PushDevice) -> Result<String, PushError> {
    let secret_key = SecretKey::from_slice(&state.enclave_key)
        .map_err(|e| PushError::InvalidSecret(e.to_string()))?;
    let plaintext = decrypt_with_key(&secret_key, &device.push_token_enc)
        .map_err(|e| PushError::CryptoError(e.to_string()))?;

    String::from_utf8(plaintext).map_err(|e| PushError::InvalidSecret(e.to_string()))
}

fn classify_internal_push_error(error: PushError) -> PushSendOutcome {
    match error {
        PushError::ConnectionError
        | PushError::DatabaseError(_)
        | PushError::DbError(_)
        | PushError::HttpError(_)
        | PushError::ProviderRetryable(_) => PushSendOutcome::Retryable {
            provider_status_code: None,
            error: error.to_string(),
        },
        _ => PushSendOutcome::Failed {
            provider_status_code: None,
            error: error.to_string(),
        },
    }
}

fn record_delivery_transition(
    write_result: NotificationDeliveryWriteResult,
    delivery_id: i64,
    lease_owner: &str,
    transition: &str,
) -> bool {
    if write_result.was_applied() {
        true
    } else {
        debug!(
            "push delivery {} lost lease before marking {} (lease_owner={})",
            delivery_id, transition, lease_owner
        );
        false
    }
}

fn retry_backoff_seconds(attempt_count: i32) -> i32 {
    let capped_attempt = attempt_count.clamp(1, 6);
    let seconds = 15_i64 * (1_i64 << (capped_attempt - 1));
    seconds.min(15 * 60) as i32
}

#[cfg(test)]
mod tests {
    use super::{classify_internal_push_error, retry_backoff_seconds};
    use crate::push::{PushError, PushSendOutcome};

    #[test]
    fn classifies_retryable_provider_errors_as_retryable() {
        match classify_internal_push_error(PushError::ProviderRetryable(
            "temporary provider failure".to_string(),
        )) {
            PushSendOutcome::Retryable { .. } => {}
            outcome => panic!("expected retryable outcome, got {outcome:?}"),
        }
    }

    #[test]
    fn retry_backoff_is_capped() {
        assert_eq!(retry_backoff_seconds(1), 15);
        assert_eq!(retry_backoff_seconds(6), 480);
        assert_eq!(retry_backoff_seconds(10), 480);
    }
}
