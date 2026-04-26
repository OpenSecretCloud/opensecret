use crate::models::schema::notification_events;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const NOTIFICATION_KIND_AGENT_MESSAGE: &str = "agent.message";
pub const NOTIFICATION_DELIVERY_MODE_GENERIC: &str = "generic";
pub const NOTIFICATION_DELIVERY_MODE_ENCRYPTED_PREVIEW: &str = "encrypted_preview";
pub const NOTIFICATION_PRIORITY_NORMAL: &str = "normal";
pub const NOTIFICATION_PRIORITY_HIGH: &str = "high";

#[derive(Error, Debug)]
pub enum NotificationEventError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, Clone, Debug, Serialize, Deserialize)]
#[diesel(table_name = notification_events)]
pub struct NotificationEvent {
    pub id: i64,
    pub uuid: Uuid,
    pub project_id: i32,
    pub user_id: Uuid,
    pub kind: String,
    pub delivery_mode: String,
    pub priority: String,
    pub collapse_key: Option<String>,
    pub fallback_title: String,
    pub fallback_body: String,
    pub payload_enc: Option<Vec<u8>>,
    pub not_before_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub cancelled_at: Option<DateTime<Utc>>,
}

impl NotificationEvent {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<NotificationEvent>, NotificationEventError> {
        notification_events::table
            .filter(notification_events::id.eq(lookup_id))
            .first::<NotificationEvent>(conn)
            .optional()
            .map_err(NotificationEventError::DatabaseError)
    }

    pub fn cancel(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<NotificationEvent>, NotificationEventError> {
        diesel::update(notification_events::table.filter(notification_events::id.eq(lookup_id)))
            .set(notification_events::cancelled_at.eq(diesel::dsl::now))
            .get_result::<NotificationEvent>(conn)
            .optional()
            .map_err(NotificationEventError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = notification_events, treat_none_as_default_value = true)]
pub struct NewNotificationEvent {
    pub uuid: Uuid,
    pub project_id: i32,
    pub user_id: Uuid,
    pub kind: String,
    pub delivery_mode: String,
    pub priority: String,
    pub collapse_key: Option<String>,
    pub fallback_title: String,
    pub fallback_body: String,
    pub payload_enc: Option<Vec<u8>>,
    pub not_before_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl NewNotificationEvent {
    pub fn insert(
        &self,
        conn: &mut PgConnection,
    ) -> Result<NotificationEvent, NotificationEventError> {
        diesel::insert_into(notification_events::table)
            .values(self)
            .get_result::<NotificationEvent>(conn)
            .map_err(NotificationEventError::DatabaseError)
    }
}
