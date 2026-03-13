use crate::models::schema::notification_deliveries;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{BigInt, Int4, Text};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const NOTIFICATION_DELIVERY_STATUS_SENT: &str = "sent";
pub const NOTIFICATION_DELIVERY_STATUS_FAILED: &str = "failed";
pub const NOTIFICATION_DELIVERY_STATUS_INVALID_TOKEN: &str = "invalid_token";
pub const NOTIFICATION_DELIVERY_STATUS_CANCELLED: &str = "cancelled";

#[derive(Error, Debug)]
pub enum NotificationDeliveryError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, QueryableByName, Clone, Debug, Serialize, Deserialize)]
#[diesel(table_name = notification_deliveries)]
pub struct NotificationDelivery {
    pub id: i64,
    pub event_id: i64,
    pub push_device_id: i64,
    pub status: String,
    pub attempt_count: i32,
    pub next_attempt_at: DateTime<Utc>,
    pub lease_owner: Option<String>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub provider_message_id: Option<String>,
    pub provider_status_code: Option<i32>,
    pub last_error: Option<String>,
    pub sent_at: Option<DateTime<Utc>>,
    pub invalidated_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NotificationDelivery {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<NotificationDelivery>, NotificationDeliveryError> {
        notification_deliveries::table
            .filter(notification_deliveries::id.eq(lookup_id))
            .first::<NotificationDelivery>(conn)
            .optional()
            .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn lease_pending(
        conn: &mut PgConnection,
        limit: i64,
        lease_owner: &str,
        lease_seconds: i32,
    ) -> Result<Vec<NotificationDelivery>, NotificationDeliveryError> {
        let query = r#"
            WITH candidates AS (
                SELECT d.id
                FROM notification_deliveries d
                WHERE d.next_attempt_at <= NOW()
                  AND (
                        (d.status IN ('pending', 'retry')
                         AND (d.lease_expires_at IS NULL OR d.lease_expires_at < NOW()))
                     OR (d.status = 'leased'
                         AND (d.lease_expires_at IS NULL OR d.lease_expires_at < NOW()))
                  )
                ORDER BY d.next_attempt_at ASC, d.id ASC
                FOR UPDATE OF d SKIP LOCKED
                LIMIT $1
            )
            UPDATE notification_deliveries d
            SET status = 'leased',
                lease_owner = $2,
                lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
                updated_at = NOW()
            FROM candidates
            WHERE d.id = candidates.id
            RETURNING d.*
        "#;

        sql_query(query)
            .bind::<BigInt, _>(limit)
            .bind::<Text, _>(lease_owner)
            .bind::<Int4, _>(lease_seconds)
            .get_results::<NotificationDelivery>(conn)
            .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn mark_sent(
        conn: &mut PgConnection,
        lookup_id: i64,
        provider_message_id: Option<&str>,
        provider_status_code: Option<i32>,
    ) -> Result<(), NotificationDeliveryError> {
        diesel::update(
            notification_deliveries::table.filter(notification_deliveries::id.eq(lookup_id)),
        )
        .set((
            notification_deliveries::status.eq(NOTIFICATION_DELIVERY_STATUS_SENT),
            notification_deliveries::attempt_count.eq(notification_deliveries::attempt_count + 1),
            notification_deliveries::provider_message_id.eq(provider_message_id),
            notification_deliveries::provider_status_code.eq(provider_status_code),
            notification_deliveries::last_error.eq::<Option<String>>(None),
            notification_deliveries::sent_at.eq(diesel::dsl::now),
            notification_deliveries::lease_owner.eq::<Option<String>>(None),
            notification_deliveries::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            notification_deliveries::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(|_| ())
        .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn mark_retry(
        conn: &mut PgConnection,
        lookup_id: i64,
        provider_status_code: Option<i32>,
        last_error: Option<&str>,
        retry_after_seconds: i32,
    ) -> Result<(), NotificationDeliveryError> {
        let query = r#"
            UPDATE notification_deliveries
            SET status = 'retry',
                attempt_count = attempt_count + 1,
                provider_status_code = $2,
                last_error = $3,
                next_attempt_at = NOW() + ($4 * INTERVAL '1 second'),
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = NOW()
            WHERE id = $1
        "#;

        sql_query(query)
            .bind::<BigInt, _>(lookup_id)
            .bind::<diesel::sql_types::Nullable<Int4>, _>(provider_status_code)
            .bind::<diesel::sql_types::Nullable<Text>, _>(last_error)
            .bind::<Int4, _>(retry_after_seconds)
            .execute(conn)
            .map(|_| ())
            .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn mark_failed(
        conn: &mut PgConnection,
        lookup_id: i64,
        provider_status_code: Option<i32>,
        last_error: Option<&str>,
    ) -> Result<(), NotificationDeliveryError> {
        diesel::update(
            notification_deliveries::table.filter(notification_deliveries::id.eq(lookup_id)),
        )
        .set((
            notification_deliveries::status.eq(NOTIFICATION_DELIVERY_STATUS_FAILED),
            notification_deliveries::attempt_count.eq(notification_deliveries::attempt_count + 1),
            notification_deliveries::provider_status_code.eq(provider_status_code),
            notification_deliveries::last_error.eq(last_error),
            notification_deliveries::lease_owner.eq::<Option<String>>(None),
            notification_deliveries::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            notification_deliveries::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(|_| ())
        .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn mark_invalid_token(
        conn: &mut PgConnection,
        lookup_id: i64,
        provider_status_code: Option<i32>,
        last_error: Option<&str>,
    ) -> Result<(), NotificationDeliveryError> {
        diesel::update(
            notification_deliveries::table.filter(notification_deliveries::id.eq(lookup_id)),
        )
        .set((
            notification_deliveries::status.eq(NOTIFICATION_DELIVERY_STATUS_INVALID_TOKEN),
            notification_deliveries::attempt_count.eq(notification_deliveries::attempt_count + 1),
            notification_deliveries::provider_status_code.eq(provider_status_code),
            notification_deliveries::last_error.eq(last_error),
            notification_deliveries::invalidated_at.eq(diesel::dsl::now),
            notification_deliveries::lease_owner.eq::<Option<String>>(None),
            notification_deliveries::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            notification_deliveries::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(|_| ())
        .map_err(NotificationDeliveryError::DatabaseError)
    }

    pub fn mark_cancelled(
        conn: &mut PgConnection,
        lookup_id: i64,
        last_error: Option<&str>,
    ) -> Result<(), NotificationDeliveryError> {
        diesel::update(
            notification_deliveries::table.filter(notification_deliveries::id.eq(lookup_id)),
        )
        .set((
            notification_deliveries::status.eq(NOTIFICATION_DELIVERY_STATUS_CANCELLED),
            notification_deliveries::last_error.eq(last_error),
            notification_deliveries::lease_owner.eq::<Option<String>>(None),
            notification_deliveries::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            notification_deliveries::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(|_| ())
        .map_err(NotificationDeliveryError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = notification_deliveries)]
pub struct NewNotificationDelivery {
    pub event_id: i64,
    pub push_device_id: i64,
    pub next_attempt_at: DateTime<Utc>,
}

impl NewNotificationDelivery {
    pub fn insert_many(
        conn: &mut PgConnection,
        new_deliveries: &[NewNotificationDelivery],
    ) -> Result<usize, NotificationDeliveryError> {
        diesel::insert_into(notification_deliveries::table)
            .values(new_deliveries)
            .execute(conn)
            .map_err(NotificationDeliveryError::DatabaseError)
    }
}
