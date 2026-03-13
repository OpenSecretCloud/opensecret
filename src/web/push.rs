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

            let already_revoked_installation_id = if let Some(active_device) = active_installation
                .as_ref()
                .filter(|device| device.user_id != user.uuid)
            {
                PushDevice::revoke_by_id(conn, active_device.id)?;
                Some(active_device.id)
            } else {
                None
            };

            let mut existing = existing_for_user.filter(|device| device.revoked_at.is_none());

            if let Some(conflict) = conflicting_token_owner.as_ref() {
                if existing.as_ref().map(|device| device.id) != Some(conflict.id)
                    && already_revoked_installation_id != Some(conflict.id)
                {
                    PushDevice::revoke_by_id(conn, conflict.id)?;
                }
            }

            if let Some(mut existing) = existing.take() {
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
