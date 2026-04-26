use crate::models::schema::user_preferences;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use thiserror::Error;
use uuid::Uuid;

pub const USER_PREFERENCE_TIMEZONE: &str = "timezone";
pub const USER_PREFERENCE_LOCALE: &str = "locale";

#[derive(Error, Debug)]
pub enum UserPreferenceError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("Invalid preference: {0}")]
    InvalidPreference(String),
}

#[derive(Queryable, Identifiable, AsChangeset, Clone, Debug)]
#[diesel(table_name = user_preferences)]
pub struct UserPreference {
    pub id: i64,
    pub user_id: Uuid,
    pub key: String,
    pub value_enc: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl UserPreference {
    pub fn get_by_user_and_key(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_key: &str,
    ) -> Result<Option<UserPreference>, UserPreferenceError> {
        user_preferences::table
            .filter(user_preferences::user_id.eq(lookup_user_id))
            .filter(user_preferences::key.eq(lookup_key))
            .first::<UserPreference>(conn)
            .optional()
            .map_err(UserPreferenceError::DatabaseError)
    }

    pub fn validate(key: &str, value: &str) -> Result<(), UserPreferenceError> {
        match key {
            USER_PREFERENCE_TIMEZONE => value.parse::<chrono_tz::Tz>().map(|_| ()).map_err(|_| {
                UserPreferenceError::InvalidPreference(format!(
                    "Invalid timezone '{value}'. Use an IANA timezone like 'America/Chicago'"
                ))
            }),
            USER_PREFERENCE_LOCALE => validate_locale(value),
            _ => Ok(()),
        }
    }
}

fn validate_locale(value: &str) -> Result<(), UserPreferenceError> {
    let trimmed = value.trim();
    if trimmed.len() < 2 || trimmed.len() > 35 {
        return Err(UserPreferenceError::InvalidPreference(format!(
            "Invalid locale '{value}'"
        )));
    }

    let starts_or_ends_with_separator =
        trimmed.chars().next().is_some_and(|c| c == '-' || c == '_')
            || trimmed.chars().last().is_some_and(|c| c == '-' || c == '_');

    if starts_or_ends_with_separator || trimmed.contains("--") || trimmed.contains("__") {
        return Err(UserPreferenceError::InvalidPreference(format!(
            "Invalid locale '{value}'"
        )));
    }

    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(UserPreferenceError::InvalidPreference(format!(
            "Invalid locale '{value}'"
        )));
    }

    Ok(())
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = user_preferences)]
pub struct NewUserPreference {
    pub user_id: Uuid,
    pub key: String,
    pub value_enc: Vec<u8>,
}

impl NewUserPreference {
    pub fn new(user_id: Uuid, key: impl Into<String>, value_enc: Vec<u8>) -> Self {
        Self {
            user_id,
            key: key.into(),
            value_enc,
        }
    }

    pub fn insert_or_update(
        &self,
        conn: &mut PgConnection,
    ) -> Result<UserPreference, UserPreferenceError> {
        diesel::insert_into(user_preferences::table)
            .values(self)
            .on_conflict((user_preferences::user_id, user_preferences::key))
            .do_update()
            .set(user_preferences::value_enc.eq(self.value_enc.clone()))
            .get_result::<UserPreference>(conn)
            .map_err(UserPreferenceError::DatabaseError)
    }
}
