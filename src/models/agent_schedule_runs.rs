use crate::models::schema::agent_schedule_runs;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{BigInt, Int4, Nullable, Text};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const AGENT_SCHEDULE_RUN_STATUS_PENDING: &str = "pending";
pub const AGENT_SCHEDULE_RUN_STATUS_RETRY: &str = "retry";
pub const AGENT_SCHEDULE_RUN_STATUS_COMPLETED: &str = "completed";
pub const AGENT_SCHEDULE_RUN_STATUS_FAILED: &str = "failed";
pub const AGENT_SCHEDULE_RUN_STATUS_CANCELLED: &str = "cancelled";
pub const AGENT_SCHEDULE_RUN_STATUS_EXPIRED: &str = "expired";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentScheduleRunWriteResult {
    Updated,
    LostLease,
}

impl AgentScheduleRunWriteResult {
    fn from_updated_rows(updated_rows: usize) -> Self {
        if updated_rows == 0 {
            Self::LostLease
        } else {
            Self::Updated
        }
    }

    pub fn was_applied(self) -> bool {
        matches!(self, Self::Updated)
    }
}

#[derive(Error, Debug)]
pub enum AgentScheduleRunError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, QueryableByName, Clone, Debug, Serialize, Deserialize)]
#[diesel(table_name = agent_schedule_runs)]
pub struct AgentScheduleRun {
    pub id: i64,
    pub uuid: Uuid,
    pub schedule_id: i64,
    pub user_id: Uuid,
    pub agent_id: i64,
    pub scheduled_for: DateTime<Utc>,
    pub stale_after_at: DateTime<Utc>,
    pub status: String,
    pub attempt_count: i32,
    pub next_attempt_at: DateTime<Utc>,
    pub lease_owner: Option<String>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub first_output_at: Option<DateTime<Utc>>,
    pub first_message_id: Option<Uuid>,
    pub output_count: i32,
    pub notification_enqueued_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentScheduleRun {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<AgentScheduleRun>, AgentScheduleRunError> {
        agent_schedule_runs::table
            .filter(agent_schedule_runs::id.eq(lookup_id))
            .first::<AgentScheduleRun>(conn)
            .optional()
            .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn lease_pending(
        conn: &mut PgConnection,
        limit: i64,
        lease_owner: &str,
        lease_seconds: i32,
    ) -> Result<Vec<AgentScheduleRun>, AgentScheduleRunError> {
        let query = r#"
            WITH candidates AS (
                SELECT r.id
                FROM agent_schedule_runs r
                WHERE r.next_attempt_at <= NOW()
                  AND (
                        (r.status IN ('pending', 'retry')
                         AND (r.lease_expires_at IS NULL OR r.lease_expires_at < NOW()))
                     OR (r.status = 'leased'
                         AND (r.lease_expires_at IS NULL OR r.lease_expires_at < NOW()))
                  )
                ORDER BY r.next_attempt_at ASC, r.id ASC
                FOR UPDATE OF r SKIP LOCKED
                LIMIT $1
            )
            UPDATE agent_schedule_runs r
            SET status = 'leased',
                lease_owner = $2,
                lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
                started_at = COALESCE(r.started_at, NOW()),
                updated_at = NOW()
            FROM candidates
            WHERE r.id = candidates.id
            RETURNING r.*
        "#;

        sql_query(query)
            .bind::<BigInt, _>(limit)
            .bind::<Text, _>(lease_owner)
            .bind::<Int4, _>(lease_seconds)
            .get_results::<AgentScheduleRun>(conn)
            .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn renew_lease(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        lease_seconds: i32,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        let query = r#"
            UPDATE agent_schedule_runs
            SET lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
                updated_at = NOW()
            WHERE id = $1
              AND status = 'leased'
              AND lease_owner = $2
        "#;

        sql_query(query)
            .bind::<BigInt, _>(lookup_id)
            .bind::<Text, _>(expected_lease_owner)
            .bind::<Int4, _>(lease_seconds)
            .execute(conn)
            .map(AgentScheduleRunWriteResult::from_updated_rows)
            .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn record_output(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        first_message_id: Uuid,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        let query = r#"
            UPDATE agent_schedule_runs
            SET first_output_at = COALESCE(first_output_at, NOW()),
                first_message_id = COALESCE(first_message_id, $3),
                output_count = output_count + 1,
                updated_at = NOW()
            WHERE id = $1
              AND status = 'leased'
              AND lease_owner = $2
        "#;

        sql_query(query)
            .bind::<BigInt, _>(lookup_id)
            .bind::<Text, _>(expected_lease_owner)
            .bind::<diesel::sql_types::Uuid, _>(first_message_id)
            .execute(conn)
            .map(AgentScheduleRunWriteResult::from_updated_rows)
            .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn mark_retry(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        last_error: Option<&str>,
        retry_after_seconds: i32,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        let query = r#"
            UPDATE agent_schedule_runs
            SET status = 'retry',
                attempt_count = attempt_count + 1,
                last_error = $3,
                next_attempt_at = NOW() + ($4 * INTERVAL '1 second'),
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = NOW()
            WHERE id = $1
              AND status = 'leased'
              AND lease_owner = $2
        "#;

        sql_query(query)
            .bind::<BigInt, _>(lookup_id)
            .bind::<Text, _>(expected_lease_owner)
            .bind::<Nullable<Text>, _>(last_error)
            .bind::<Int4, _>(retry_after_seconds)
            .execute(conn)
            .map(AgentScheduleRunWriteResult::from_updated_rows)
            .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn mark_completed(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        notification_enqueued: bool,
        last_error: Option<&str>,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        diesel::update(
            agent_schedule_runs::table
                .filter(agent_schedule_runs::id.eq(lookup_id))
                .filter(agent_schedule_runs::status.eq("leased"))
                .filter(
                    agent_schedule_runs::lease_owner.eq(Some(expected_lease_owner.to_string())),
                ),
        )
        .set((
            agent_schedule_runs::status.eq(AGENT_SCHEDULE_RUN_STATUS_COMPLETED),
            agent_schedule_runs::attempt_count.eq(agent_schedule_runs::attempt_count + 1),
            agent_schedule_runs::last_error.eq(last_error),
            agent_schedule_runs::notification_enqueued_at.eq(if notification_enqueued {
                Some(Utc::now())
            } else {
                None
            }),
            agent_schedule_runs::completed_at.eq(diesel::dsl::now),
            agent_schedule_runs::lease_owner.eq::<Option<String>>(None),
            agent_schedule_runs::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            agent_schedule_runs::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(AgentScheduleRunWriteResult::from_updated_rows)
        .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn mark_failed(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        last_error: Option<&str>,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        diesel::update(
            agent_schedule_runs::table
                .filter(agent_schedule_runs::id.eq(lookup_id))
                .filter(agent_schedule_runs::status.eq("leased"))
                .filter(
                    agent_schedule_runs::lease_owner.eq(Some(expected_lease_owner.to_string())),
                ),
        )
        .set((
            agent_schedule_runs::status.eq(AGENT_SCHEDULE_RUN_STATUS_FAILED),
            agent_schedule_runs::attempt_count.eq(agent_schedule_runs::attempt_count + 1),
            agent_schedule_runs::last_error.eq(last_error),
            agent_schedule_runs::completed_at.eq(diesel::dsl::now),
            agent_schedule_runs::lease_owner.eq::<Option<String>>(None),
            agent_schedule_runs::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            agent_schedule_runs::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(AgentScheduleRunWriteResult::from_updated_rows)
        .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn mark_expired(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        last_error: Option<&str>,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        diesel::update(
            agent_schedule_runs::table
                .filter(agent_schedule_runs::id.eq(lookup_id))
                .filter(agent_schedule_runs::status.eq("leased"))
                .filter(
                    agent_schedule_runs::lease_owner.eq(Some(expected_lease_owner.to_string())),
                ),
        )
        .set((
            agent_schedule_runs::status.eq(AGENT_SCHEDULE_RUN_STATUS_EXPIRED),
            agent_schedule_runs::attempt_count.eq(agent_schedule_runs::attempt_count + 1),
            agent_schedule_runs::last_error.eq(last_error),
            agent_schedule_runs::completed_at.eq(diesel::dsl::now),
            agent_schedule_runs::lease_owner.eq::<Option<String>>(None),
            agent_schedule_runs::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            agent_schedule_runs::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(AgentScheduleRunWriteResult::from_updated_rows)
        .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn mark_cancelled(
        conn: &mut PgConnection,
        lookup_id: i64,
        expected_lease_owner: &str,
        last_error: Option<&str>,
    ) -> Result<AgentScheduleRunWriteResult, AgentScheduleRunError> {
        diesel::update(
            agent_schedule_runs::table
                .filter(agent_schedule_runs::id.eq(lookup_id))
                .filter(agent_schedule_runs::status.eq("leased"))
                .filter(
                    agent_schedule_runs::lease_owner.eq(Some(expected_lease_owner.to_string())),
                ),
        )
        .set((
            agent_schedule_runs::status.eq(AGENT_SCHEDULE_RUN_STATUS_CANCELLED),
            agent_schedule_runs::last_error.eq(last_error),
            agent_schedule_runs::completed_at.eq(diesel::dsl::now),
            agent_schedule_runs::lease_owner.eq::<Option<String>>(None),
            agent_schedule_runs::lease_expires_at.eq::<Option<DateTime<Utc>>>(None),
            agent_schedule_runs::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map(AgentScheduleRunWriteResult::from_updated_rows)
        .map_err(AgentScheduleRunError::DatabaseError)
    }

    pub fn cancel_unstarted_for_schedule(
        conn: &mut PgConnection,
        lookup_schedule_id: i64,
    ) -> Result<usize, AgentScheduleRunError> {
        diesel::update(
            agent_schedule_runs::table
                .filter(agent_schedule_runs::schedule_id.eq(lookup_schedule_id))
                .filter(agent_schedule_runs::status.eq_any(vec![
                    AGENT_SCHEDULE_RUN_STATUS_PENDING,
                    AGENT_SCHEDULE_RUN_STATUS_RETRY,
                ])),
        )
        .set((
            agent_schedule_runs::status.eq(AGENT_SCHEDULE_RUN_STATUS_CANCELLED),
            agent_schedule_runs::completed_at.eq(diesel::dsl::now),
            agent_schedule_runs::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map_err(AgentScheduleRunError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = agent_schedule_runs)]
pub struct NewAgentScheduleRun {
    pub uuid: Uuid,
    pub schedule_id: i64,
    pub user_id: Uuid,
    pub agent_id: i64,
    pub scheduled_for: DateTime<Utc>,
    pub stale_after_at: DateTime<Utc>,
    pub status: String,
    pub next_attempt_at: DateTime<Utc>,
}

impl NewAgentScheduleRun {
    pub fn insert(
        &self,
        conn: &mut PgConnection,
    ) -> Result<AgentScheduleRun, AgentScheduleRunError> {
        diesel::insert_into(agent_schedule_runs::table)
            .values(self)
            .get_result::<AgentScheduleRun>(conn)
            .map_err(AgentScheduleRunError::DatabaseError)
    }
}
