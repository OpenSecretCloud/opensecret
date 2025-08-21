use crate::models::schema::user_api_keys;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum UserApiKeyError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("API key with this name already exists")]
    DuplicateName,
    #[error("API key not found")]
    NotFound,
}

#[derive(Queryable, Serialize, Deserialize, Clone, Debug)]
#[diesel(check_for_backend(diesel::pg::Pg))]
#[diesel(table_name = user_api_keys)]
pub struct UserApiKey {
    pub id: i32,
    pub user_id: Uuid,
    pub key_hash: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = user_api_keys)]
pub struct NewUserApiKey {
    pub user_id: Uuid,
    pub key_hash: String,
    pub name: String,
}

impl NewUserApiKey {
    pub fn new(user_id: Uuid, key_hash: String, name: String) -> Self {
        Self {
            user_id,
            key_hash,
            name,
        }
    }

    pub fn insert(self, conn: &mut PgConnection) -> Result<UserApiKey, UserApiKeyError> {
        diesel::insert_into(user_api_keys::table)
            .values(&self)
            .get_result(conn)
            .map_err(|e| match e {
                diesel::result::Error::DatabaseError(
                    diesel::result::DatabaseErrorKind::UniqueViolation,
                    _,
                ) => UserApiKeyError::DuplicateName,
                _ => UserApiKeyError::DatabaseError(e),
            })
    }
}

impl UserApiKey {
    pub fn get_by_id(conn: &mut PgConnection, id: i32) -> Result<Option<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::id.eq(id))
            .first::<Self>(conn)
            .optional()
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn get_by_key_hash(
        conn: &mut PgConnection,
        key_hash: &str,
    ) -> Result<Option<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::key_hash.eq(key_hash))
            .first::<Self>(conn)
            .optional()
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn get_all_for_user(
        conn: &mut PgConnection,
        user_id: Uuid,
    ) -> Result<Vec<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::user_id.eq(user_id))
            .order(user_api_keys::created_at.desc())
            .load::<Self>(conn)
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn delete(self, conn: &mut PgConnection) -> Result<(), UserApiKeyError> {
        diesel::delete(user_api_keys::table.filter(user_api_keys::id.eq(self.id)))
            .execute(conn)
            .map(|_| ())
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn delete_by_id(conn: &mut PgConnection, id: i32) -> Result<(), UserApiKeyError> {
        diesel::delete(user_api_keys::table.filter(user_api_keys::id.eq(id)))
            .execute(conn)
            .map(|_| ())
            .map_err(UserApiKeyError::DatabaseError)
    }

    /// Get the user associated with this API key
    pub fn get_user(
        &self,
        conn: &mut PgConnection,
    ) -> Result<Option<crate::models::users::User>, UserApiKeyError> {
        use crate::models::users::{User, UserError};
        match User::get_by_uuid(conn, self.user_id) {
            Ok(user) => Ok(user),
            Err(UserError::DatabaseError(e)) => Err(UserApiKeyError::DatabaseError(e)),
            Err(_) => Ok(None), // User not found
        }
    }
}
