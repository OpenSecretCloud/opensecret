use crate::encrypt::{
    decrypt_key_deterministic, decrypt_with_key, encrypt_key_deterministic, encrypt_with_key,
};
use crate::{
    aws_credentials::AwsCredentialManager,
    models::user_kv::{NewUserKV, UserKV, UserKVCiphertextPair, UserKVError},
};
use diesel::prelude::*;
use secp256k1::SecretKey;
use serde::Serialize;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error};
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

const AES_GCM_STORAGE_OVERHEAD_BYTES: usize = 12 + 16;
const AES_SIV_STORAGE_OVERHEAD_BYTES: usize = 16;
const KV_ROW_STORAGE_OVERHEAD_BYTES: usize =
    AES_SIV_STORAGE_OVERHEAD_BYTES + AES_GCM_STORAGE_OVERHEAD_BYTES;

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("Key not found")]
    KeyNotFound,
    #[error("Unauthorized access")]
    Unauthorized,
    #[error("Decryption error")]
    DecryptionError,
    #[error("Stored output exceeds the requested limit")]
    OutputTooLarge,
    #[error("Database error: {0}")]
    DatabaseError(#[from] UserKVError),
}

impl From<diesel::result::Error> for StoreError {
    fn from(error: diesel::result::Error) -> Self {
        Self::DatabaseError(UserKVError::DatabaseError(error))
    }
}

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Clone, Serialize, Zeroize)]
pub struct KVPair {
    pub key: String,
    pub value: String,
    pub created_at: i64,
    pub updated_at: i64,
}

fn database_connection_error() -> StoreError {
    StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
}

fn checked_database_len(length: i64) -> StoreResult<usize> {
    usize::try_from(length).map_err(|_| StoreError::DecryptionError)
}

fn validate_value_ciphertext_len(
    ciphertext_len: i64,
    logical_body_limit: usize,
) -> StoreResult<usize> {
    let ciphertext_len = checked_database_len(ciphertext_len)?;
    let plaintext_len = ciphertext_len
        .checked_sub(AES_GCM_STORAGE_OVERHEAD_BYTES)
        .ok_or(StoreError::DecryptionError)?;
    if plaintext_len > logical_body_limit {
        return Err(StoreError::OutputTooLarge);
    }
    Ok(ciphertext_len)
}

fn validate_list_ciphertext_aggregate(
    row_count: i64,
    aggregate_ciphertext_len: Option<i64>,
    logical_body_limit: usize,
    row_limit: usize,
) -> StoreResult<(usize, usize)> {
    let row_count = checked_database_len(row_count)?;
    if row_count > row_limit {
        return Err(StoreError::OutputTooLarge);
    }

    let aggregate_ciphertext_len = match aggregate_ciphertext_len {
        Some(length) => checked_database_len(length)?,
        None if row_count == 0 => 0,
        None => return Err(StoreError::DecryptionError),
    };
    let aggregate_storage_overhead = row_count
        .checked_mul(KV_ROW_STORAGE_OVERHEAD_BYTES)
        .ok_or(StoreError::OutputTooLarge)?;
    let aggregate_plaintext_len = aggregate_ciphertext_len
        .checked_sub(aggregate_storage_overhead)
        .ok_or(StoreError::DecryptionError)?;
    if aggregate_plaintext_len > logical_body_limit {
        return Err(StoreError::OutputTooLarge);
    }

    Ok((row_count, aggregate_ciphertext_len))
}

fn zeroizing_utf8(bytes: Vec<u8>) -> StoreResult<Zeroizing<String>> {
    match String::from_utf8(bytes) {
        Ok(value) => Ok(Zeroizing::new(value)),
        Err(error) => {
            let mut bytes = error.into_bytes();
            bytes.zeroize();
            Err(StoreError::DecryptionError)
        }
    }
}

/// Read one KV value through a bounded transport-v2 storage path.
///
/// The length preflight and narrow ciphertext fetch share a read-only,
/// repeatable-read snapshot. The caller remains responsible for the final
/// bounded JSON representation, whose escaping can be larger than plaintext.
pub fn get_bounded(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    key: &str,
    user_secret_key: &SecretKey,
    logical_body_limit: usize,
) -> StoreResult<Option<Zeroizing<String>>> {
    let mut conn = pool.get().map_err(|_| database_connection_error())?;
    let encrypted_key = Zeroizing::new(encrypt_key_deterministic(user_secret_key, key.as_bytes()));

    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoreError, _>(|conn| {
            let Some(ciphertext_len) = UserKV::get_v2_value_ciphertext_len_by_user_and_key(
                conn,
                user_id,
                encrypted_key.as_slice(),
            )?
            else {
                return Ok(None);
            };
            let expected_ciphertext_len =
                validate_value_ciphertext_len(ciphertext_len, logical_body_limit)?;

            let Some(value_enc) = UserKV::get_v2_value_ciphertext_by_user_and_key(
                conn,
                user_id,
                encrypted_key.as_slice(),
            )?
            else {
                return Err(StoreError::DecryptionError);
            };
            let value_enc = Zeroizing::new(value_enc);
            if value_enc.len() != expected_ciphertext_len {
                return Err(StoreError::DecryptionError);
            }

            let plaintext = decrypt_with_key(user_secret_key, value_enc.as_slice())
                .map_err(|_| StoreError::DecryptionError)?;
            zeroizing_utf8(plaintext).map(Some)
        })
}

/// List KV values through a bounded transport-v2 storage path.
///
/// Row count and aggregate ciphertext size are measured before the narrow
/// projection is loaded in the same read-only, repeatable-read snapshot.
pub fn list_bounded(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    user_secret_key: &SecretKey,
    logical_body_limit: usize,
    row_limit: usize,
) -> StoreResult<Zeroizing<Vec<KVPair>>> {
    let mut conn = pool.get().map_err(|_| database_connection_error())?;

    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoreError, _>(|conn| {
            let (row_count, aggregate_ciphertext_len) =
                UserKV::get_v2_ciphertext_aggregate_for_user(conn, user_id)?;
            let (row_count, expected_ciphertext_len) = validate_list_ciphertext_aggregate(
                row_count,
                aggregate_ciphertext_len,
                logical_body_limit,
                row_limit,
            )?;

            let rows = Zeroizing::new(UserKV::get_v2_ciphertexts_for_user(conn, user_id)?);
            if rows.len() != row_count {
                return Err(StoreError::DecryptionError);
            }
            let actual_ciphertext_len = rows.iter().try_fold(0usize, |total, row| {
                total
                    .checked_add(row.key_enc.len())
                    .and_then(|total| total.checked_add(row.value_enc.len()))
                    .ok_or(StoreError::OutputTooLarge)
            })?;
            if actual_ciphertext_len != expected_ciphertext_len {
                return Err(StoreError::DecryptionError);
            }

            let mut pairs = Zeroizing::new(Vec::with_capacity(row_count));
            for UserKVCiphertextPair {
                key_enc,
                value_enc,
                created_at,
                updated_at,
            } in rows.iter()
            {
                let decrypted_key = decrypt_key_deterministic(user_secret_key, key_enc)
                    .map_err(|_| StoreError::DecryptionError)?;
                let mut key = zeroizing_utf8(decrypted_key)?;

                let decrypted_value = decrypt_with_key(user_secret_key, value_enc)
                    .map_err(|_| StoreError::DecryptionError)?;
                let mut value = zeroizing_utf8(decrypted_value)?;

                pairs.push(KVPair {
                    key: std::mem::take(&mut *key),
                    value: std::mem::take(&mut *value),
                    created_at: created_at.timestamp_millis(),
                    updated_at: updated_at.timestamp_millis(),
                });
            }

            Ok(pairs)
        })
}

// Update the get function
pub fn get(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    key: &str,
    user_secret_key: &SecretKey,
) -> StoreResult<Option<String>> {
    debug!("Getting KV pair");
    let mut conn = pool.get().map_err(|e| {
        error!("Failed to get database connection: {:?}", e);
        StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
    })?;

    let encrypted_key = encrypt_key_deterministic(user_secret_key, key.as_bytes());

    let user_kv = UserKV::get_by_user_and_key(&mut conn, user_id, &encrypted_key).map_err(|e| {
        error!("Failed to get KV pair: {:?}", e);
        e
    })?;

    if let Some(user_kv) = user_kv {
        let decrypted_value =
            decrypt_with_key(user_secret_key, &user_kv.value_enc).map_err(|e| {
                error!("Failed to decrypt value: {:?}", e);
                StoreError::DecryptionError
            })?;
        let value_str = String::from_utf8(decrypted_value).map_err(|e| {
            error!("Failed to convert decrypted value to string: {:?}", e);
            StoreError::DecryptionError
        })?;
        Ok(Some(value_str))
    } else {
        Ok(None)
    }
}

pub async fn put(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    key: &str,
    value: &str,
    encryption_key: &SecretKey,
    _aws_credential_manager: Arc<tokio::sync::RwLock<Option<AwsCredentialManager>>>,
) -> StoreResult<()> {
    let mut conn = pool.get().map_err(|_| {
        StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
    })?;

    let encrypted_key = encrypt_key_deterministic(encryption_key, key.as_bytes());
    let encrypted_value = encrypt_with_key(encryption_key, value.as_bytes()).await;

    let new_user_kv = NewUserKV {
        user_id,
        key_enc: encrypted_key,
        value_enc: encrypted_value,
    };

    new_user_kv.insert(&mut conn)?;

    Ok(())
}

pub fn delete(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    key: &str,
    user_secret_key: &SecretKey,
) -> StoreResult<()> {
    let mut conn = pool.get().map_err(|_| {
        StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
    })?;

    let encrypted_key = encrypt_key_deterministic(user_secret_key, key.as_bytes());

    let user_kv_id = UserKV::get_id_by_user_and_key(&mut conn, user_id, &encrypted_key)?;

    if let Some(user_kv_id) = user_kv_id {
        UserKV::delete_by_id(&mut conn, user_kv_id)?;
        Ok(())
    } else {
        Err(StoreError::KeyNotFound)
    }
}

pub fn delete_all(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
) -> StoreResult<()> {
    let mut conn = pool.get().map_err(|_| {
        StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
    })?;

    UserKV::delete_all_for_user(&mut conn, user_id).map_err(StoreError::DatabaseError)
}

pub fn list(
    pool: &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>,
    user_id: Uuid,
    user_secret_key: &SecretKey,
) -> StoreResult<Vec<KVPair>> {
    let mut conn = pool.get().map_err(|_| {
        StoreError::DatabaseError(UserKVError::DatabaseError(diesel::result::Error::NotFound))
    })?;
    let user_kvs = UserKV::get_all_for_user(&mut conn, user_id)?;
    let mut pairs = Vec::new();
    for user_kv in user_kvs {
        let decrypted_key = decrypt_key_deterministic(user_secret_key, &user_kv.key_enc)
            .map_err(|_| StoreError::DecryptionError)?;
        let key = String::from_utf8(decrypted_key).map_err(|_| StoreError::DecryptionError)?;

        let decrypted_value = decrypt_with_key(user_secret_key, &user_kv.value_enc)
            .map_err(|_| StoreError::DecryptionError)?;
        let value = String::from_utf8(decrypted_value).map_err(|_| StoreError::DecryptionError)?;

        let created_at = user_kv.created_at.timestamp_millis();
        let updated_at = user_kv.updated_at.timestamp_millis();

        pairs.push(KVPair {
            key,
            value,
            created_at,
            updated_at,
        });
    }
    Ok(pairs)
}

#[cfg(test)]
mod bounded_read_tests {
    use super::*;

    #[test]
    fn value_ciphertext_preflight_accounts_for_storage_overhead() {
        assert_eq!(
            validate_value_ciphertext_len(AES_GCM_STORAGE_OVERHEAD_BYTES as i64 + 7, 7).unwrap(),
            AES_GCM_STORAGE_OVERHEAD_BYTES + 7
        );
        assert!(matches!(
            validate_value_ciphertext_len(AES_GCM_STORAGE_OVERHEAD_BYTES as i64 + 8, 7),
            Err(StoreError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_value_ciphertext_len(AES_GCM_STORAGE_OVERHEAD_BYTES as i64 - 1, 7),
            Err(StoreError::DecryptionError)
        ));
    }

    #[test]
    fn list_preflight_bounds_rows_and_aggregate_plaintext() {
        assert_eq!(
            validate_list_ciphertext_aggregate(0, None, 0, 0).unwrap(),
            (0, 0)
        );
        assert_eq!(
            validate_list_ciphertext_aggregate(
                1,
                Some(KV_ROW_STORAGE_OVERHEAD_BYTES as i64 + 7),
                7,
                1,
            )
            .unwrap(),
            (1, KV_ROW_STORAGE_OVERHEAD_BYTES + 7)
        );
        assert!(matches!(
            validate_list_ciphertext_aggregate(1, Some(KV_ROW_STORAGE_OVERHEAD_BYTES as i64), 0, 0,),
            Err(StoreError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_list_ciphertext_aggregate(
                1,
                Some(KV_ROW_STORAGE_OVERHEAD_BYTES as i64 + 8),
                7,
                1,
            ),
            Err(StoreError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_list_ciphertext_aggregate(
                1,
                Some(KV_ROW_STORAGE_OVERHEAD_BYTES as i64 - 1),
                7,
                1,
            ),
            Err(StoreError::DecryptionError)
        ));
        assert!(matches!(
            validate_list_ciphertext_aggregate(1, None, 7, 1),
            Err(StoreError::DecryptionError)
        ));
    }
}
