use crate::models::schema::push_devices;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const PUSH_PLATFORM_IOS: &str = "ios";
pub const PUSH_PLATFORM_ANDROID: &str = "android";
pub const PUSH_PROVIDER_APNS: &str = "apns";
pub const PUSH_PROVIDER_FCM: &str = "fcm";
pub const PUSH_ENV_DEV: &str = "dev";
pub const PUSH_ENV_PROD: &str = "prod";
pub const PUSH_KEY_ALGORITHM_P256_ECDH_V1: &str = "p256_ecdh_v1";

#[derive(Error, Debug)]
pub enum PushDeviceError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, Clone, Debug, Serialize, Deserialize)]
#[diesel(table_name = push_devices)]
pub struct PushDevice {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub installation_id: Uuid,
    pub platform: String,
    pub provider: String,
    pub environment: String,
    pub app_id: String,
    pub push_token_enc: Vec<u8>,
    pub push_token_hash: Vec<u8>,
    pub notification_public_key: Vec<u8>,
    pub key_algorithm: String,
    pub supports_encrypted_preview: bool,
    pub supports_background_processing: bool,
    pub last_seen_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl PushDevice {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        push_devices::table
            .filter(push_devices::id.eq(lookup_id))
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn get_by_uuid_and_user(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
        lookup_user_id: Uuid,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        push_devices::table
            .filter(push_devices::uuid.eq(lookup_uuid))
            .filter(push_devices::user_id.eq(lookup_user_id))
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn get_by_installation_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_installation_id: Uuid,
        lookup_environment: &str,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        if let Some(active_device) = push_devices::table
            .filter(push_devices::user_id.eq(lookup_user_id))
            .filter(push_devices::installation_id.eq(lookup_installation_id))
            .filter(push_devices::environment.eq(lookup_environment))
            .filter(push_devices::revoked_at.is_null())
            .order(push_devices::updated_at.desc())
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)?
        {
            return Ok(Some(active_device));
        }

        push_devices::table
            .filter(push_devices::user_id.eq(lookup_user_id))
            .filter(push_devices::installation_id.eq(lookup_installation_id))
            .filter(push_devices::environment.eq(lookup_environment))
            .order(push_devices::updated_at.desc())
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn get_by_installation(
        conn: &mut PgConnection,
        lookup_installation_id: Uuid,
        lookup_environment: &str,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        push_devices::table
            .filter(push_devices::installation_id.eq(lookup_installation_id))
            .filter(push_devices::environment.eq(lookup_environment))
            .filter(push_devices::revoked_at.is_null())
            .order(push_devices::updated_at.desc())
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn get_active_by_token_hash(
        conn: &mut PgConnection,
        lookup_provider: &str,
        lookup_environment: &str,
        lookup_token_hash: &[u8],
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        push_devices::table
            .filter(push_devices::provider.eq(lookup_provider))
            .filter(push_devices::environment.eq(lookup_environment))
            .filter(push_devices::push_token_hash.eq(lookup_token_hash))
            .filter(push_devices::revoked_at.is_null())
            .first::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn list_active_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<PushDevice>, PushDeviceError> {
        push_devices::table
            .filter(push_devices::user_id.eq(lookup_user_id))
            .filter(push_devices::revoked_at.is_null())
            .order((push_devices::last_seen_at.desc(), push_devices::id.desc()))
            .load::<PushDevice>(conn)
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn update(&self, conn: &mut PgConnection) -> Result<(), PushDeviceError> {
        diesel::update(push_devices::table.filter(push_devices::id.eq(self.id)))
            .set((
                push_devices::user_id.eq(self.user_id),
                push_devices::installation_id.eq(self.installation_id),
                push_devices::platform.eq(&self.platform),
                push_devices::provider.eq(&self.provider),
                push_devices::environment.eq(&self.environment),
                push_devices::app_id.eq(&self.app_id),
                push_devices::push_token_enc.eq(&self.push_token_enc),
                push_devices::push_token_hash.eq(&self.push_token_hash),
                push_devices::notification_public_key.eq(&self.notification_public_key),
                push_devices::key_algorithm.eq(&self.key_algorithm),
                push_devices::supports_encrypted_preview.eq(self.supports_encrypted_preview),
                push_devices::supports_background_processing
                    .eq(self.supports_background_processing),
                push_devices::last_seen_at.eq(self.last_seen_at),
                push_devices::revoked_at.eq(self.revoked_at),
                push_devices::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map(|_| ())
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn revoke_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        diesel::update(push_devices::table.filter(push_devices::id.eq(lookup_id)))
            .set((
                push_devices::revoked_at.eq(diesel::dsl::now),
                push_devices::updated_at.eq(diesel::dsl::now),
            ))
            .get_result::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }

    pub fn revoke_by_uuid_and_user(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
        lookup_user_id: Uuid,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        diesel::update(
            push_devices::table
                .filter(push_devices::uuid.eq(lookup_uuid))
                .filter(push_devices::user_id.eq(lookup_user_id)),
        )
        .set((
            push_devices::revoked_at.eq(diesel::dsl::now),
            push_devices::updated_at.eq(diesel::dsl::now),
        ))
        .get_result::<PushDevice>(conn)
        .optional()
        .map_err(PushDeviceError::DatabaseError)
    }

    pub fn invalidate(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<PushDevice>, PushDeviceError> {
        diesel::update(push_devices::table.filter(push_devices::id.eq(lookup_id)))
            .set((
                push_devices::revoked_at.eq(diesel::dsl::now),
                push_devices::updated_at.eq(diesel::dsl::now),
            ))
            .get_result::<PushDevice>(conn)
            .optional()
            .map_err(PushDeviceError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = push_devices)]
pub struct NewPushDevice {
    pub user_id: Uuid,
    pub installation_id: Uuid,
    pub platform: String,
    pub provider: String,
    pub environment: String,
    pub app_id: String,
    pub push_token_enc: Vec<u8>,
    pub push_token_hash: Vec<u8>,
    pub notification_public_key: Vec<u8>,
    pub key_algorithm: String,
    pub supports_encrypted_preview: bool,
    pub supports_background_processing: bool,
    pub last_seen_at: DateTime<Utc>,
}

impl NewPushDevice {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<PushDevice, PushDeviceError> {
        diesel::insert_into(push_devices::table)
            .values(self)
            .get_result::<PushDevice>(conn)
            .map_err(PushDeviceError::DatabaseError)
    }
}
