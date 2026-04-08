use crate::models::schema::agent_schedules;
use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

pub const SCHEDULE_KIND_ONE_OFF: &str = "one_off";
pub const SCHEDULE_KIND_RECURRING: &str = "recurring";

pub const RECURRENCE_TYPE_INTERVAL: &str = "interval";
pub const RECURRENCE_TYPE_DAILY: &str = "daily";
pub const RECURRENCE_TYPE_WEEKLY: &str = "weekly";

pub const SCHEDULE_TIMEZONE_MODE_FOLLOW_USER: &str = "follow_user";
pub const SCHEDULE_TIMEZONE_MODE_FIXED: &str = "fixed";

pub const SCHEDULE_STATUS_ACTIVE: &str = "active";
pub const SCHEDULE_STATUS_COMPLETED: &str = "completed";
pub const SCHEDULE_STATUS_CANCELLED: &str = "cancelled";

#[derive(Error, Debug)]
pub enum AgentScheduleError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("Invalid schedule spec: {0}")]
    InvalidSpec(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleWeekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl ScheduleWeekday {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Monday => "monday",
            Self::Tuesday => "tuesday",
            Self::Wednesday => "wednesday",
            Self::Thursday => "thursday",
            Self::Friday => "friday",
            Self::Saturday => "saturday",
            Self::Sunday => "sunday",
        }
    }

    pub fn to_chrono(&self) -> chrono::Weekday {
        match self {
            Self::Monday => chrono::Weekday::Mon,
            Self::Tuesday => chrono::Weekday::Tue,
            Self::Wednesday => chrono::Weekday::Wed,
            Self::Thursday => chrono::Weekday::Thu,
            Self::Friday => chrono::Weekday::Fri,
            Self::Saturday => chrono::Weekday::Sat,
            Self::Sunday => chrono::Weekday::Sun,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleIntervalUnit {
    Hours,
}

impl ScheduleIntervalUnit {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hours => "hours",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScheduleSpec {
    OneOff {
        local_date: String,
        local_time: String,
    },
    Interval {
        every_n: u32,
        interval_unit: ScheduleIntervalUnit,
    },
    Daily {
        local_time: String,
    },
    Weekly {
        local_time: String,
        weekdays: Vec<ScheduleWeekday>,
    },
}

impl ScheduleSpec {
    pub fn schedule_kind(&self) -> &'static str {
        match self {
            Self::OneOff { .. } => SCHEDULE_KIND_ONE_OFF,
            Self::Interval { .. } | Self::Daily { .. } | Self::Weekly { .. } => {
                SCHEDULE_KIND_RECURRING
            }
        }
    }

    pub fn recurrence_type(&self) -> Option<&'static str> {
        match self {
            Self::OneOff { .. } => None,
            Self::Interval { .. } => Some(RECURRENCE_TYPE_INTERVAL),
            Self::Daily { .. } => Some(RECURRENCE_TYPE_DAILY),
            Self::Weekly { .. } => Some(RECURRENCE_TYPE_WEEKLY),
        }
    }

    pub fn validate(&self) -> Result<(), AgentScheduleError> {
        match self {
            Self::OneOff {
                local_date,
                local_time,
            } => {
                parse_local_date(local_date)?;
                parse_local_time(local_time)?;
                Ok(())
            }
            Self::Interval {
                every_n,
                interval_unit,
            } => {
                if *every_n == 0 {
                    return Err(AgentScheduleError::InvalidSpec(
                        "interval every_n must be greater than zero".to_string(),
                    ));
                }

                match interval_unit {
                    ScheduleIntervalUnit::Hours => Ok(()),
                }
            }
            Self::Daily { local_time } => {
                parse_local_time(local_time)?;
                Ok(())
            }
            Self::Weekly {
                local_time,
                weekdays,
            } => {
                parse_local_time(local_time)?;
                if weekdays.is_empty() {
                    return Err(AgentScheduleError::InvalidSpec(
                        "weekly schedules require at least one weekday".to_string(),
                    ));
                }

                Ok(())
            }
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::OneOff {
                local_date,
                local_time,
            } => format!("once on {} at {}", local_date, local_time),
            Self::Interval {
                every_n,
                interval_unit,
            } => format!("every {} {}", every_n, interval_unit.as_str()),
            Self::Daily { local_time } => format!("every day at {}", local_time),
            Self::Weekly {
                local_time,
                weekdays,
            } => {
                let day_list = weekdays
                    .iter()
                    .map(ScheduleWeekday::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("every {} at {}", day_list, local_time)
            }
        }
    }

    pub fn to_value(&self) -> Result<Value, AgentScheduleError> {
        serde_json::to_value(self)
            .map_err(|e| AgentScheduleError::InvalidSpec(format!("serialize schedule spec: {e}")))
    }

    pub fn from_value(value: &Value) -> Result<Self, AgentScheduleError> {
        serde_json::from_value(value.clone())
            .map_err(|e| AgentScheduleError::InvalidSpec(format!("parse schedule spec: {e}")))
    }
}

pub fn parse_local_time(value: &str) -> Result<NaiveTime, AgentScheduleError> {
    NaiveTime::parse_from_str(value.trim(), "%H:%M").map_err(|_| {
        AgentScheduleError::InvalidSpec(format!(
            "invalid local_time '{value}'. Use 24-hour HH:MM format"
        ))
    })
}

pub fn parse_local_date(value: &str) -> Result<NaiveDate, AgentScheduleError> {
    NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d").map_err(|_| {
        AgentScheduleError::InvalidSpec(format!(
            "invalid local_date '{value}'. Use YYYY-MM-DD format"
        ))
    })
}

#[derive(Queryable, Identifiable, AsChangeset, Clone, Debug, Serialize, Deserialize)]
#[diesel(table_name = agent_schedules)]
pub struct AgentSchedule {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub agent_id: i64,
    pub description: String,
    pub instruction_enc: Vec<u8>,
    pub schedule_kind: String,
    pub recurrence_type: Option<String>,
    pub schedule_spec: Value,
    pub timezone_mode: String,
    pub resolved_timezone: String,
    pub fixed_timezone: Option<String>,
    pub stale_after_minutes: i32,
    pub status: String,
    pub next_scheduled_for: Option<DateTime<Utc>>,
    pub last_scheduled_for: Option<DateTime<Utc>>,
    pub last_run_at: Option<DateTime<Utc>>,
    pub run_count: i32,
    pub cancelled_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentSchedule {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<AgentSchedule>, AgentScheduleError> {
        agent_schedules::table
            .filter(agent_schedules::id.eq(lookup_id))
            .first::<AgentSchedule>(conn)
            .optional()
            .map_err(AgentScheduleError::DatabaseError)
    }

    pub fn get_by_uuid_and_agent(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
        lookup_user_id: Uuid,
        lookup_agent_id: i64,
    ) -> Result<Option<AgentSchedule>, AgentScheduleError> {
        agent_schedules::table
            .filter(agent_schedules::uuid.eq(lookup_uuid))
            .filter(agent_schedules::user_id.eq(lookup_user_id))
            .filter(agent_schedules::agent_id.eq(lookup_agent_id))
            .first::<AgentSchedule>(conn)
            .optional()
            .map_err(AgentScheduleError::DatabaseError)
    }

    pub fn list_by_agent(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_agent_id: i64,
        status_filter: Option<&str>,
    ) -> Result<Vec<AgentSchedule>, AgentScheduleError> {
        let mut query = agent_schedules::table
            .filter(agent_schedules::user_id.eq(lookup_user_id))
            .filter(agent_schedules::agent_id.eq(lookup_agent_id))
            .into_boxed();

        if let Some(status_filter) = status_filter {
            query = query.filter(agent_schedules::status.eq(status_filter));
        }

        query
            .order((
                agent_schedules::next_scheduled_for.asc().nulls_last(),
                agent_schedules::created_at.desc(),
            ))
            .load::<AgentSchedule>(conn)
            .map_err(AgentScheduleError::DatabaseError)
    }

    pub fn list_active_follow_user_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<AgentSchedule>, AgentScheduleError> {
        agent_schedules::table
            .filter(agent_schedules::user_id.eq(lookup_user_id))
            .filter(agent_schedules::status.eq(SCHEDULE_STATUS_ACTIVE))
            .filter(agent_schedules::timezone_mode.eq(SCHEDULE_TIMEZONE_MODE_FOLLOW_USER))
            .load::<AgentSchedule>(conn)
            .map_err(AgentScheduleError::DatabaseError)
    }

    pub fn spec(&self) -> Result<ScheduleSpec, AgentScheduleError> {
        ScheduleSpec::from_value(&self.schedule_spec)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = agent_schedules)]
pub struct NewAgentSchedule {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub agent_id: i64,
    pub description: String,
    pub instruction_enc: Vec<u8>,
    pub schedule_kind: String,
    pub recurrence_type: Option<String>,
    pub schedule_spec: Value,
    pub timezone_mode: String,
    pub resolved_timezone: String,
    pub fixed_timezone: Option<String>,
    pub stale_after_minutes: i32,
    pub status: String,
    pub next_scheduled_for: Option<DateTime<Utc>>,
    pub last_scheduled_for: Option<DateTime<Utc>>,
    pub last_run_at: Option<DateTime<Utc>>,
    pub run_count: i32,
    pub cancelled_at: Option<DateTime<Utc>>,
}

impl NewAgentSchedule {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<AgentSchedule, AgentScheduleError> {
        diesel::insert_into(agent_schedules::table)
            .values(self)
            .get_result::<AgentSchedule>(conn)
            .map_err(AgentScheduleError::DatabaseError)
    }
}
