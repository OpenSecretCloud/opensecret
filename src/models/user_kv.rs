use crate::models::schema::user_kv;
use chrono::{DateTime, Utc};
use diesel::dsl::{count_star, sql};
use diesel::prelude::*;
use diesel::sql_types::{BigInt, Nullable};
use diesel::upsert::excluded;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use zeroize::Zeroize;

#[derive(Error, Debug)]
pub enum UserKVError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = user_kv)]
pub struct UserKV {
    pub id: i64,
    pub user_id: Uuid,
    pub key_enc: Vec<u8>,
    pub value_enc: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Narrow ciphertext projection used only by bounded transport-v2 KV reads.
///
/// The custom `Zeroize` implementation deliberately wipes only encrypted
/// user-controlled bytes. Timestamps are not sensitive allocations and do not
/// implement `Zeroize` themselves.
#[derive(Queryable, Selectable)]
#[diesel(table_name = user_kv)]
pub struct UserKVCiphertextPair {
    pub key_enc: Vec<u8>,
    pub value_enc: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Zeroize for UserKVCiphertextPair {
    fn zeroize(&mut self) {
        self.key_enc.zeroize();
        self.value_enc.zeroize();
    }
}

impl Drop for UserKVCiphertextPair {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl UserKV {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<UserKV>, UserKVError> {
        user_kv::table
            .filter(user_kv::id.eq(lookup_id))
            .first::<UserKV>(conn)
            .optional()
            .map_err(UserKVError::DatabaseError)
    }

    pub fn get_by_user_and_key(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_key: &Vec<u8>,
    ) -> Result<Option<UserKV>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .filter(user_kv::key_enc.eq(lookup_key))
            .first::<UserKV>(conn)
            .optional()
            .map_err(UserKVError::DatabaseError)
    }

    pub fn get_id_by_user_and_key(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_key: &[u8],
    ) -> Result<Option<i64>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .filter(user_kv::key_enc.eq(lookup_key))
            .select(user_kv::id)
            .first::<i64>(conn)
            .optional()
            .map_err(UserKVError::DatabaseError)
    }

    pub fn get_all_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<UserKV>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .load::<UserKV>(conn)
            .map_err(UserKVError::DatabaseError)
    }

    /// Measure one stored value without loading its ciphertext.
    ///
    /// This is a transport-v2-only query seam. The existing v1 query methods
    /// intentionally continue selecting the complete `UserKV` row.
    pub fn get_v2_value_ciphertext_len_by_user_and_key(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_key: &[u8],
    ) -> Result<Option<i64>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .filter(user_kv::key_enc.eq(lookup_key))
            .select(sql::<BigInt>("octet_length(value_enc)::bigint"))
            .first::<i64>(conn)
            .optional()
            .map_err(UserKVError::DatabaseError)
    }

    /// Load only one value ciphertext after its bounded v2 preflight.
    pub fn get_v2_value_ciphertext_by_user_and_key(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_key: &[u8],
    ) -> Result<Option<Vec<u8>>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .filter(user_kv::key_enc.eq(lookup_key))
            .select(user_kv::value_enc)
            .first::<Vec<u8>>(conn)
            .optional()
            .map_err(UserKVError::DatabaseError)
    }

    /// Count rows and ciphertext octets without loading either ciphertext.
    pub fn get_v2_ciphertext_aggregate_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<(i64, Option<i64>), UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .select((
                count_star(),
                sql::<Nullable<BigInt>>(
                    "SUM(octet_length(key_enc)::bigint + octet_length(value_enc)::bigint)::bigint",
                ),
            ))
            .first::<(i64, Option<i64>)>(conn)
            .map_err(UserKVError::DatabaseError)
    }

    /// Load the narrow list projection after its bounded v2 preflight.
    pub fn get_v2_ciphertexts_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<UserKVCiphertextPair>, UserKVError> {
        user_kv::table
            .filter(user_kv::user_id.eq(lookup_user_id))
            .select(UserKVCiphertextPair::as_select())
            .load::<UserKVCiphertextPair>(conn)
            .map_err(UserKVError::DatabaseError)
    }

    pub fn update(&self, conn: &mut PgConnection) -> Result<(), UserKVError> {
        diesel::update(user_kv::table)
            .filter(user_kv::id.eq(self.id))
            .set(self)
            .execute(conn)
            .map(|_| ())
            .map_err(UserKVError::DatabaseError)
    }

    pub fn delete(&self, conn: &mut PgConnection) -> Result<(), UserKVError> {
        Self::delete_by_id(conn, self.id)
    }

    pub fn delete_by_id(conn: &mut PgConnection, lookup_id: i64) -> Result<(), UserKVError> {
        diesel::delete(user_kv::table)
            .filter(user_kv::id.eq(lookup_id))
            .execute(conn)
            .map(|_| ())
            .map_err(UserKVError::DatabaseError)
    }

    pub fn delete_all_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<(), UserKVError> {
        diesel::delete(user_kv::table)
            .filter(user_kv::user_id.eq(lookup_user_id))
            .execute(conn)
            .map(|_| ())
            .map_err(UserKVError::DatabaseError)
    }
}

#[derive(Insertable)]
#[diesel(table_name = user_kv)]
pub struct NewUserKV {
    pub user_id: Uuid,
    pub key_enc: Vec<u8>,
    pub value_enc: Vec<u8>,
}

impl NewUserKV {
    pub fn new(user_id: Uuid, key_enc: Vec<u8>, value_enc: Vec<u8>) -> Self {
        NewUserKV {
            user_id,
            key_enc,
            value_enc,
        }
    }

    pub fn insert(&self, conn: &mut PgConnection) -> Result<(), UserKVError> {
        diesel::insert_into(user_kv::table)
            .values(self)
            .on_conflict((user_kv::user_id, user_kv::key_enc))
            .do_update()
            .set(user_kv::value_enc.eq(excluded(user_kv::value_enc)))
            .execute(conn)
            .map(|_| ())
            .map_err(UserKVError::DatabaseError)
    }
}
