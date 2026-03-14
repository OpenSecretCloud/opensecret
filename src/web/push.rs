use crate::encrypt::encrypt_with_key;
use crate::models::push_devices::{
    NewPushDevice, PushDevice, PushDeviceError, PUSH_ENV_DEV, PUSH_ENV_PROD,
    PUSH_KEY_ALGORITHM_P256_ECDH_V1, PUSH_PLATFORM_ANDROID, PUSH_PLATFORM_IOS, PUSH_PROVIDER_APNS,
    PUSH_PROVIDER_FCM,
};
use crate::models::users::User;
use crate::web::encryption_middleware::{decrypt_request, encrypt_response, EncryptedResponse};
use crate::{ApiError, AppState};
use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Extension, Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use diesel::Connection;
use p256::pkcs8::DecodePublicKey;
use p256::PublicKey;
use secp256k1::SecretKey;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize)]
struct RegisterPushDeviceRequest {
    installation_id: Uuid,
    platform: String,
    provider: String,
    environment: String,
    app_id: String,
    push_token: String,
    notification_public_key: String,
    key_algorithm: String,
    #[serde(default)]
    supports_encrypted_preview: bool,
    #[serde(default)]
    supports_background_processing: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PushDeviceResponse {
    id: Uuid,
    object: &'static str,
    installation_id: Uuid,
    platform: String,
    provider: String,
    environment: String,
    app_id: String,
    key_algorithm: String,
    supports_encrypted_preview: bool,
    supports_background_processing: bool,
    last_seen_at: DateTime<Utc>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
struct PushDeviceListResponse {
    object: &'static str,
    data: Vec<PushDeviceResponse>,
}

#[derive(Debug, Clone, Serialize)]
struct DeletedPushDeviceResponse {
    id: Uuid,
    object: &'static str,
    deleted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrationWriteMode {
    ReuseExisting,
    InsertNew,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistrationDecision {
    write_mode: RegistrationWriteMode,
    revoke_device_ids: Vec<i64>,
}

pub fn router(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/push/devices",
            post(register_push_device).layer(from_fn_with_state(
                app_state.clone(),
                decrypt_request::<RegisterPushDeviceRequest>,
            )),
        )
        .route(
            "/v1/push/devices",
            get(list_push_devices)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .route(
            "/v1/push/devices/:id",
            delete(revoke_push_device)
                .layer(from_fn_with_state(app_state.clone(), decrypt_request::<()>)),
        )
        .with_state(app_state)
}

async fn register_push_device(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(body): Extension<RegisterPushDeviceRequest>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<PushDeviceResponse>>, ApiError> {
    let public_key_bytes = general_purpose::STANDARD
        .decode(&body.notification_public_key)
        .map_err(|_| ApiError::BadRequest)?;
    validate_register_request(&body, &public_key_bytes)?;

    let token_hash = Sha256::digest(body.push_token.as_bytes()).to_vec();
    let enclave_key =
        SecretKey::from_slice(&state.enclave_key).map_err(|_| ApiError::InternalServerError)?;
    let push_token_enc = encrypt_with_key(&enclave_key, body.push_token.as_bytes()).await;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let now = Utc::now();
    let device = conn
        .transaction::<PushDevice, PushDeviceError, _>(|conn| {
            let existing_for_user = PushDevice::get_by_installation_for_user(
                conn,
                user.uuid,
                body.installation_id,
                &body.environment,
            )?;
            let active_installation =
                PushDevice::get_by_installation(conn, body.installation_id, &body.environment)?;
            let conflicting_token_owner = PushDevice::get_active_by_token_hash(
                conn,
                &body.provider,
                &body.environment,
                &token_hash,
            )?;

            let decision = plan_push_device_registration(
                user.uuid,
                existing_for_user.as_ref(),
                active_installation.as_ref(),
                conflicting_token_owner.as_ref(),
            );

            for revoke_device_id in decision.revoke_device_ids {
                PushDevice::revoke_by_id(conn, revoke_device_id)?;
            }

            if decision.write_mode == RegistrationWriteMode::ReuseExisting {
                let mut existing = existing_for_user.expect("existing device should be present");
                existing.platform = body.platform.clone();
                existing.provider = body.provider.clone();
                existing.environment = body.environment.clone();
                existing.app_id = body.app_id.clone();
                existing.push_token_enc = push_token_enc.clone();
                existing.push_token_hash = token_hash.clone();
                existing.notification_public_key = public_key_bytes.clone();
                existing.key_algorithm = body.key_algorithm.clone();
                existing.supports_encrypted_preview = body.supports_encrypted_preview;
                existing.supports_background_processing = body.supports_background_processing;
                existing.last_seen_at = now;
                existing.revoked_at = None;
                existing.update(conn)?;
                Ok(existing)
            } else {
                NewPushDevice {
                    user_id: user.uuid,
                    installation_id: body.installation_id,
                    platform: body.platform.clone(),
                    provider: body.provider.clone(),
                    environment: body.environment.clone(),
                    app_id: body.app_id.clone(),
                    push_token_enc: push_token_enc.clone(),
                    push_token_hash: token_hash.clone(),
                    notification_public_key: public_key_bytes.clone(),
                    key_algorithm: body.key_algorithm.clone(),
                    supports_encrypted_preview: body.supports_encrypted_preview,
                    supports_background_processing: body.supports_background_processing,
                    last_seen_at: now,
                }
                .insert(conn)
            }
        })
        .map_err(map_push_device_error)?;

    encrypt_response(&state, &session_id, &PushDeviceResponse::from(device)).await
}

async fn list_push_devices(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<PushDeviceListResponse>>, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let devices =
        PushDevice::list_active_for_user(&mut conn, user.uuid).map_err(map_push_device_error)?;
    let response = PushDeviceListResponse {
        object: "list",
        data: devices.into_iter().map(PushDeviceResponse::from).collect(),
    };

    encrypt_response(&state, &session_id, &response).await
}

async fn revoke_push_device(
    State(state): State<Arc<AppState>>,
    Path(device_uuid): Path<Uuid>,
    Extension(user): Extension<User>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<DeletedPushDeviceResponse>>, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let device = PushDevice::revoke_by_uuid_and_user(&mut conn, device_uuid, user.uuid)
        .map_err(map_push_device_error)?
        .ok_or(ApiError::NotFound)?;

    let response = DeletedPushDeviceResponse {
        id: device.uuid,
        object: "push.device.deleted",
        deleted: true,
    };

    encrypt_response(&state, &session_id, &response).await
}

fn validate_register_request(
    request: &RegisterPushDeviceRequest,
    public_key_bytes: &[u8],
) -> Result<(), ApiError> {
    if request.installation_id.is_nil() || request.installation_id.get_version_num() != 4 {
        return Err(ApiError::BadRequest);
    }

    if request.push_token.trim().is_empty()
        || request.app_id.trim().is_empty()
        || request.notification_public_key.trim().is_empty()
    {
        return Err(ApiError::BadRequest);
    }

    if request.app_id.len() > 255 || request.push_token.len() > 4096 {
        return Err(ApiError::BadRequest);
    }

    let valid_platform_provider = matches!(
        (request.platform.as_str(), request.provider.as_str()),
        (PUSH_PLATFORM_IOS, PUSH_PROVIDER_APNS) | (PUSH_PLATFORM_ANDROID, PUSH_PROVIDER_FCM)
    );

    if !valid_platform_provider {
        return Err(ApiError::BadRequest);
    }

    if request.environment != PUSH_ENV_DEV && request.environment != PUSH_ENV_PROD {
        return Err(ApiError::BadRequest);
    }

    if request.key_algorithm != PUSH_KEY_ALGORITHM_P256_ECDH_V1 {
        return Err(ApiError::BadRequest);
    }

    if request.notification_public_key.len() > 1024 || public_key_bytes.len() > 256 {
        return Err(ApiError::BadRequest);
    }

    PublicKey::from_public_key_der(public_key_bytes).map_err(|_| ApiError::BadRequest)?;

    Ok(())
}

fn map_push_device_error(error: PushDeviceError) -> ApiError {
    error!("Push device database error: {:?}", error);
    ApiError::InternalServerError
}

fn plan_push_device_registration(
    current_user_id: Uuid,
    existing_for_user: Option<&PushDevice>,
    active_installation: Option<&PushDevice>,
    conflicting_token_owner: Option<&PushDevice>,
) -> RegistrationDecision {
    let write_mode = if existing_for_user.is_some() {
        RegistrationWriteMode::ReuseExisting
    } else {
        RegistrationWriteMode::InsertNew
    };

    let existing_device_id = existing_for_user.map(|device| device.id);
    let mut revoke_device_ids = Vec::new();

    if let Some(active_device) = active_installation {
        if active_device.user_id != current_user_id && existing_device_id != Some(active_device.id)
        {
            revoke_device_ids.push(active_device.id);
        }
    }

    if let Some(conflict) = conflicting_token_owner {
        if existing_device_id != Some(conflict.id) && !revoke_device_ids.contains(&conflict.id) {
            revoke_device_ids.push(conflict.id);
        }
    }

    RegistrationDecision {
        write_mode,
        revoke_device_ids,
    }
}

impl From<PushDevice> for PushDeviceResponse {
    fn from(value: PushDevice) -> Self {
        Self {
            id: value.uuid,
            object: "push.device",
            installation_id: value.installation_id,
            platform: value.platform,
            provider: value.provider,
            environment: value.environment,
            app_id: value.app_id,
            key_algorithm: value.key_algorithm,
            supports_encrypted_preview: value.supports_encrypted_preview,
            supports_background_processing: value.supports_background_processing,
            last_seen_at: value.last_seen_at,
            created_at: value.created_at,
            updated_at: value.updated_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose;
    use p256::pkcs8::EncodePublicKey;
    use rand_core::OsRng;

    fn sample_push_device(
        user_id: Uuid,
        installation_id: Uuid,
        token_hash: Vec<u8>,
        revoked_at: Option<DateTime<Utc>>,
    ) -> PushDevice {
        let now = Utc::now();
        PushDevice {
            id: now.timestamp_nanos_opt().expect("valid timestamp nanos"),
            uuid: Uuid::new_v4(),
            user_id,
            installation_id,
            platform: PUSH_PLATFORM_IOS.to_string(),
            provider: PUSH_PROVIDER_APNS.to_string(),
            environment: PUSH_ENV_PROD.to_string(),
            app_id: "ai.trymaple.ios".to_string(),
            push_token_enc: vec![1, 2, 3],
            push_token_hash: token_hash,
            notification_public_key: vec![4, 5, 6],
            key_algorithm: PUSH_KEY_ALGORITHM_P256_ECDH_V1.to_string(),
            supports_encrypted_preview: true,
            supports_background_processing: true,
            last_seen_at: now,
            revoked_at,
            created_at: now,
            updated_at: now,
        }
    }

    fn sample_register_request(installation_id: Uuid) -> RegisterPushDeviceRequest {
        let secret_key = p256::SecretKey::random(&mut OsRng);
        let public_key = secret_key.public_key();
        let public_key_der = public_key
            .to_public_key_der()
            .expect("public key should encode");

        RegisterPushDeviceRequest {
            installation_id,
            platform: PUSH_PLATFORM_IOS.to_string(),
            provider: PUSH_PROVIDER_APNS.to_string(),
            environment: PUSH_ENV_PROD.to_string(),
            app_id: "ai.trymaple.ios".to_string(),
            push_token: "push-token".to_string(),
            notification_public_key: general_purpose::STANDARD.encode(public_key_der.as_bytes()),
            key_algorithm: PUSH_KEY_ALGORITHM_P256_ECDH_V1.to_string(),
            supports_encrypted_preview: true,
            supports_background_processing: true,
        }
    }

    #[test]
    fn plan_reuses_revoked_same_user_installation() {
        let user_id = Uuid::new_v4();
        let installation_id = Uuid::new_v4();
        let existing = sample_push_device(user_id, installation_id, vec![1], Some(Utc::now()));

        let decision = plan_push_device_registration(user_id, Some(&existing), None, None);

        assert_eq!(decision.write_mode, RegistrationWriteMode::ReuseExisting);
        assert!(decision.revoke_device_ids.is_empty());
    }

    #[test]
    fn plan_revokes_cross_user_installation_once() {
        let current_user_id = Uuid::new_v4();
        let installation_id = Uuid::new_v4();
        let other_user_device =
            sample_push_device(Uuid::new_v4(), installation_id, vec![1, 2, 3], None);

        let decision = plan_push_device_registration(
            current_user_id,
            None,
            Some(&other_user_device),
            Some(&other_user_device),
        );

        assert_eq!(decision.write_mode, RegistrationWriteMode::InsertNew);
        assert_eq!(decision.revoke_device_ids, vec![other_user_device.id]);
    }

    #[test]
    fn plan_revokes_conflicting_token_owner_for_handoff() {
        let current_user_id = Uuid::new_v4();
        let conflicting_device =
            sample_push_device(Uuid::new_v4(), Uuid::new_v4(), vec![9, 9, 9], None);

        let decision =
            plan_push_device_registration(current_user_id, None, None, Some(&conflicting_device));

        assert_eq!(decision.write_mode, RegistrationWriteMode::InsertNew);
        assert_eq!(decision.revoke_device_ids, vec![conflicting_device.id]);
    }

    #[test]
    fn validate_register_request_rejects_nil_installation_id() {
        let request = sample_register_request(Uuid::nil());
        let public_key_bytes = general_purpose::STANDARD
            .decode(&request.notification_public_key)
            .expect("public key should decode");

        assert!(matches!(
            validate_register_request(&request, &public_key_bytes),
            Err(ApiError::BadRequest)
        ));
    }

    #[test]
    fn validate_register_request_rejects_non_v4_installation_id() {
        let request = sample_register_request(
            Uuid::parse_str("6ba7b810-9dad-11d1-80b4-00c04fd430c8").expect("uuid should parse"),
        );
        let public_key_bytes = general_purpose::STANDARD
            .decode(&request.notification_public_key)
            .expect("public key should decode");

        assert!(matches!(
            validate_register_request(&request, &public_key_bytes),
            Err(ApiError::BadRequest)
        ));
    }
}
