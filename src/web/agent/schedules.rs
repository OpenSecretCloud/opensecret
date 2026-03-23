use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::BigInt;
use futures::stream::{self, StreamExt};
use secp256k1::SecretKey;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::time::{sleep, Duration as TokioDuration};
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

use crate::encrypt::decrypt_string;
use crate::models::agent_schedule_runs::{
    AgentScheduleRun, AgentScheduleRunError, AgentScheduleRunWriteResult, NewAgentScheduleRun,
    AGENT_SCHEDULE_RUN_STATUS_COMPLETED,
};
use crate::models::agent_schedules::{
    parse_local_date, parse_local_time, AgentSchedule, AgentScheduleError, NewAgentSchedule,
    ScheduleSpec, ScheduleWeekday, SCHEDULE_KIND_RECURRING, SCHEDULE_STATUS_ACTIVE,
    SCHEDULE_STATUS_CANCELLED, SCHEDULE_STATUS_COMPLETED, SCHEDULE_TIMEZONE_MODE_FIXED,
    SCHEDULE_TIMEZONE_MODE_FOLLOW_USER,
};
use crate::models::agents::{Agent, AGENT_KIND_MAIN, AGENT_KIND_SUBAGENT};
use crate::models::schema::agent_schedules;
use crate::models::user_preferences::{UserPreference, USER_PREFERENCE_TIMEZONE};
use crate::models::users::User;
use crate::push::{enqueue_agent_message_notification, AgentPushTarget};
use crate::web::openai::{get_chat_completion_response, BillingContext, CompletionChunk};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};

use super::runtime;
use super::tools::{Tool, ToolResult};

const DEFAULT_STALE_AFTER_MINUTES: i32 = 15;
const SCHEDULE_MATERIALIZER_BATCH_SIZE: i64 = 32;
const SCHEDULE_RUN_BATCH_SIZE: i64 = 16;
const SCHEDULE_RUN_LEASE_TTL_SECONDS: i32 = 180;
const SCHEDULE_RUN_HEARTBEAT_INTERVAL_SECONDS: u64 = 30;
const SCHEDULE_WORKER_POLL_INTERVAL_SECONDS: u64 = 5;
const SCHEDULE_RUN_MAX_CONCURRENCY: usize = 4;
const SCHEDULE_RUN_MAX_ATTEMPTS: i32 = 8;
const SCHEDULE_RUN_MAX_RETRY_BACKOFF_SECONDS: i32 = 15 * 60;

#[derive(diesel::deserialize::QueryableByName)]
struct DueScheduleIdRow {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    id: i64,
}

#[derive(Debug, Clone)]
struct PersistedScheduledMessage {
    message_id: Uuid,
    text: String,
}

#[derive(Debug, Clone)]
struct ScheduledTurnOutcome {
    persisted_messages: Vec<PersistedScheduledMessage>,
    had_error: bool,
}

pub(crate) fn start_schedule_worker(state: Arc<AppState>) {
    info!(
        "starting schedule worker (poll_interval={}s, materializer_batch_size={}, run_batch_size={}, lease_ttl={}s)",
        SCHEDULE_WORKER_POLL_INTERVAL_SECONDS,
        SCHEDULE_MATERIALIZER_BATCH_SIZE,
        SCHEDULE_RUN_BATCH_SIZE,
        SCHEDULE_RUN_LEASE_TTL_SECONDS,
    );

    tokio::spawn(async move {
        loop {
            if let Err(e) = materialize_due_schedules(&state).await {
                error!("schedule materialization failed: {e}");
            }

            if let Err(e) = process_schedule_run_batch(&state).await {
                error!("schedule run batch failed: {e}");
            }

            sleep(TokioDuration::from_secs(
                SCHEDULE_WORKER_POLL_INTERVAL_SECONDS,
            ))
            .await;
        }
    });
}

pub(crate) fn refresh_follow_user_schedules_for_user(
    conn: &mut PgConnection,
    user_id: Uuid,
    timezone_name: &str,
) -> Result<(), AgentScheduleError> {
    let timezone = parse_timezone_or_utc(timezone_name);
    let schedules = AgentSchedule::list_active_follow_user_for_user(conn, user_id)?;

    if !schedules.is_empty() {
        info!(
            "refreshing {} follow-user schedule(s) for user {} to timezone {}",
            schedules.len(),
            user_id,
            timezone.name(),
        );
    }

    let now = Utc::now();

    for schedule in schedules {
        let spec = schedule.spec()?;
        let next_scheduled_for =
            recompute_next_due_for_timezone_change(&schedule, &spec, timezone, now)?;

        diesel::update(agent_schedules::table.filter(agent_schedules::id.eq(schedule.id)))
            .set((
                agent_schedules::resolved_timezone.eq(timezone.name()),
                agent_schedules::next_scheduled_for.eq(next_scheduled_for),
                agent_schedules::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map_err(AgentScheduleError::DatabaseError)?;

        trace!(
            "refreshed follow-user schedule {} for user {} (next_scheduled_for={:?})",
            schedule.uuid,
            user_id,
            next_scheduled_for,
        );
    }

    Ok(())
}

pub struct ScheduleTaskTool {
    state: Arc<AppState>,
    user: Arc<User>,
    user_key: Arc<SecretKey>,
    agent: Agent,
}

impl ScheduleTaskTool {
    pub fn new(
        state: Arc<AppState>,
        user: Arc<User>,
        user_key: Arc<SecretKey>,
        agent: Agent,
    ) -> Self {
        Self {
            state,
            user,
            user_key,
            agent,
        }
    }
}

#[async_trait]
impl Tool for ScheduleTaskTool {
    fn name(&self) -> &str {
        "schedule_task"
    }

    fn description(&self) -> &str {
        "Schedule a future wakeup for yourself. Use structured arguments for one-off or recurring schedules. The instruction should be written for your future self, not as final notification copy."
    }

    fn args_schema(&self) -> &str {
        r#"{
  "schedule_kind": "one_off|recurring",
  "instruction": "future-agent instruction in natural language",
  "description": "optional short operational summary",
  "timezone_mode": "optional: follow_user|fixed (default follow_user)",
  "fixed_timezone": "optional IANA timezone when timezone_mode=fixed",
  "stale_after_minutes": "optional positive integer (default 15)",
  "local_date": "required for one_off: YYYY-MM-DD",
  "local_time": "required for one_off/daily/weekly: HH:MM 24-hour",
  "recurrence_type": "required for recurring: interval|daily|weekly",
  "every_n": "required for interval: positive integer",
  "interval_unit": "required for interval: hours",
  "weekdays": "required for weekly: comma-separated weekdays like monday,friday or monday,tuesday,wednesday,thursday,friday"
}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(schedule_kind) = args.get("schedule_kind").map(|s| s.trim()) else {
            return ToolResult::error("'schedule_kind' argument required");
        };
        let Some(instruction) = args.get("instruction").map(|s| s.trim()) else {
            return ToolResult::error("'instruction' argument required");
        };

        if instruction.is_empty() {
            return ToolResult::error("'instruction' must not be empty");
        }

        let spec = match parse_schedule_spec_from_args(args) {
            Ok(spec) => spec,
            Err(e) => return ToolResult::error(e),
        };

        let timezone_mode = args
            .get("timezone_mode")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or(SCHEDULE_TIMEZONE_MODE_FOLLOW_USER);

        let fixed_timezone = args
            .get("fixed_timezone")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        if timezone_mode != SCHEDULE_TIMEZONE_MODE_FOLLOW_USER
            && timezone_mode != SCHEDULE_TIMEZONE_MODE_FIXED
        {
            return ToolResult::error(
                "'timezone_mode' must be 'follow_user' or 'fixed' if provided",
            );
        }

        if timezone_mode == SCHEDULE_TIMEZONE_MODE_FIXED && fixed_timezone.is_none() {
            return ToolResult::error("'fixed_timezone' is required when timezone_mode='fixed'");
        }

        if schedule_kind == "one_off" && spec.schedule_kind() != "one_off" {
            return ToolResult::error("one_off schedules require local_date and local_time");
        }

        if schedule_kind == "recurring" && spec.schedule_kind() != SCHEDULE_KIND_RECURRING {
            return ToolResult::error(
                "recurring schedules require recurrence_type and recurrence parameters",
            );
        }

        let stale_after_minutes = match args
            .get("stale_after_minutes")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            Some(value) => match value.parse::<i32>() {
                Ok(value) if value > 0 => value,
                _ => {
                    return ToolResult::error(
                        "'stale_after_minutes' must be a positive integer when provided",
                    )
                }
            },
            None => DEFAULT_STALE_AFTER_MINUTES,
        };

        let resolved_timezone = match resolve_initial_timezone_name(
            &self.state,
            self.user.as_ref(),
            self.user_key.as_ref(),
            timezone_mode,
            fixed_timezone.as_deref(),
        )
        .await
        {
            Ok(timezone) => timezone,
            Err(e) => {
                error!("schedule_task failed to resolve timezone: {e:?}");
                return ToolResult::error("Failed to resolve timezone for schedule");
            }
        };

        let tz = parse_timezone_or_utc(&resolved_timezone);
        let now = Utc::now();
        let next_scheduled_for = match compute_initial_next_due(&spec, tz, now) {
            Ok(next) => next,
            Err(e) => return ToolResult::error(e.to_string()),
        };

        if next_scheduled_for <= now {
            return ToolResult::error("Scheduled time must be in the future");
        }

        let description = args
            .get("description")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| truncate_for_description(instruction));

        let instruction_enc =
            crate::encrypt::encrypt_with_key(self.user_key.as_ref(), instruction.as_bytes()).await;
        let schedule_spec = match spec.to_value() {
            Ok(value) => value,
            Err(e) => return ToolResult::error(e.to_string()),
        };

        let new_schedule = NewAgentSchedule {
            uuid: Uuid::new_v4(),
            user_id: self.user.uuid,
            agent_id: self.agent.id,
            description,
            instruction_enc,
            schedule_kind: spec.schedule_kind().to_string(),
            recurrence_type: spec.recurrence_type().map(str::to_string),
            schedule_spec,
            timezone_mode: timezone_mode.to_string(),
            resolved_timezone: resolved_timezone.clone(),
            fixed_timezone,
            stale_after_minutes,
            status: SCHEDULE_STATUS_ACTIVE.to_string(),
            next_scheduled_for: Some(next_scheduled_for),
            last_scheduled_for: None,
            last_run_at: None,
            run_count: 0,
            cancelled_at: None,
        };

        let mut conn = match self.state.db.get_pool().get() {
            Ok(conn) => conn,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        match new_schedule.insert(&mut conn) {
            Ok(schedule) => {
                let spec_summary = spec.summary();
                info!(
                    "created schedule {} for user {} agent {} (kind={}, recurrence={:?}, timezone_mode={}, resolved_timezone={}, next_scheduled_for={}, stale_after_minutes={})",
                    schedule.uuid,
                    self.user.uuid,
                    self.agent.uuid,
                    schedule.schedule_kind,
                    schedule.recurrence_type,
                    schedule.timezone_mode,
                    schedule.resolved_timezone,
                    next_scheduled_for.format("%Y-%m-%d %H:%M:%S UTC"),
                    stale_after_minutes,
                );

                ToolResult::success(format!(
                    "Created schedule {} ({}) in timezone {}. Next run: {} UTC.",
                    schedule.uuid,
                    spec_summary,
                    resolved_timezone,
                    next_scheduled_for.format("%Y-%m-%d %H:%M:%S")
                ))
            }
            Err(e) => {
                error!("schedule_task insert failed: {e:?}");
                ToolResult::error("Failed to create schedule")
            }
        }
    }
}

pub struct ListSchedulesTool {
    state: Arc<AppState>,
    user: Arc<User>,
    agent: Agent,
}

impl ListSchedulesTool {
    pub fn new(state: Arc<AppState>, user: Arc<User>, agent: Agent) -> Self {
        Self { state, user, agent }
    }
}

#[async_trait]
impl Tool for ListSchedulesTool {
    fn name(&self) -> &str {
        "list_schedules"
    }

    fn description(&self) -> &str {
        "List schedules for this agent. Defaults to active schedules only."
    }

    fn args_schema(&self) -> &str {
        r#"{"status": "optional: active|completed|cancelled|all (default active)"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let status_filter = match args.get("status").map(|s| s.trim()) {
            Some("all") => None,
            None | Some("") => Some(SCHEDULE_STATUS_ACTIVE),
            Some(SCHEDULE_STATUS_ACTIVE) => Some(SCHEDULE_STATUS_ACTIVE),
            Some(SCHEDULE_STATUS_COMPLETED) => Some(SCHEDULE_STATUS_COMPLETED),
            Some(SCHEDULE_STATUS_CANCELLED) => Some(SCHEDULE_STATUS_CANCELLED),
            Some(_) => {
                return ToolResult::error(
                    "'status' must be one of active, completed, cancelled, or all",
                )
            }
        };

        let mut conn = match self.state.db.get_pool().get() {
            Ok(conn) => conn,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        match AgentSchedule::list_by_agent(&mut conn, self.user.uuid, self.agent.id, status_filter)
        {
            Ok(schedules) => {
                debug!(
                    "listed {} schedule(s) for user {} agent {} (status_filter={:?})",
                    schedules.len(),
                    self.user.uuid,
                    self.agent.uuid,
                    status_filter,
                );

                if schedules.is_empty() {
                    return ToolResult::success("No schedules found.".to_string());
                }

                let mut output = format!("Found {} schedule(s):\n\n", schedules.len());
                for schedule in schedules {
                    let spec_summary = schedule
                        .spec()
                        .map(|spec| spec.summary())
                        .unwrap_or_else(|_| "[invalid spec]".to_string());
                    let next_text = schedule
                        .next_scheduled_for
                        .map(|dt| {
                            let tz = parse_timezone_or_utc(&schedule.resolved_timezone);
                            format_local_time(dt, tz)
                        })
                        .unwrap_or_else(|| "none".to_string());
                    output.push_str(&format!(
                        "- id: {}\n  status: {}\n  description: {}\n  rule: {}\n  timezone: {} ({})\n  next: {}\n\n",
                        schedule.uuid,
                        schedule.status,
                        schedule.description.trim(),
                        spec_summary,
                        schedule.timezone_mode,
                        schedule.resolved_timezone,
                        next_text,
                    ));
                }

                ToolResult::success(output)
            }
            Err(e) => {
                error!("list_schedules failed: {e:?}");
                ToolResult::error("Failed to list schedules")
            }
        }
    }
}

pub struct CancelScheduleTool {
    state: Arc<AppState>,
    user: Arc<User>,
    agent: Agent,
}

impl CancelScheduleTool {
    pub fn new(state: Arc<AppState>, user: Arc<User>, agent: Agent) -> Self {
        Self { state, user, agent }
    }
}

#[async_trait]
impl Tool for CancelScheduleTool {
    fn name(&self) -> &str {
        "cancel_schedule"
    }

    fn description(&self) -> &str {
        "Cancel an existing schedule for this agent by schedule id (UUID). Prevents future runs and cancels unstarted pending runs."
    }

    fn args_schema(&self) -> &str {
        r#"{"schedule_id": "schedule UUID from list_schedules"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(schedule_id) = args.get("schedule_id").map(|s| s.trim()) else {
            return ToolResult::error("'schedule_id' argument required");
        };

        let schedule_uuid = match Uuid::parse_str(schedule_id) {
            Ok(uuid) => uuid,
            Err(_) => return ToolResult::error("'schedule_id' must be a valid UUID"),
        };

        let mut conn = match self.state.db.get_pool().get() {
            Ok(conn) => conn,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        let result = conn.transaction::<String, AgentScheduleError, _>(|conn| {
            let Some(schedule) = AgentSchedule::get_by_uuid_and_agent(
                conn,
                schedule_uuid,
                self.user.uuid,
                self.agent.id,
            )?
            else {
                debug!(
                    "cancel_schedule did not find schedule {} for user {} agent {}",
                    schedule_uuid, self.user.uuid, self.agent.uuid,
                );
                return Ok("Schedule not found.".to_string());
            };

            if schedule.status == SCHEDULE_STATUS_CANCELLED {
                debug!(
                    "schedule {} already cancelled for user {} agent {}",
                    schedule.uuid, self.user.uuid, self.agent.uuid,
                );
                return Ok(format!("Schedule {} is already cancelled.", schedule.uuid));
            }

            diesel::update(agent_schedules::table.filter(agent_schedules::id.eq(schedule.id)))
                .set((
                    agent_schedules::status.eq(SCHEDULE_STATUS_CANCELLED),
                    agent_schedules::next_scheduled_for.eq::<Option<DateTime<Utc>>>(None),
                    agent_schedules::cancelled_at.eq(diesel::dsl::now),
                    agent_schedules::updated_at.eq(diesel::dsl::now),
                ))
                .execute(conn)
                .map_err(AgentScheduleError::DatabaseError)?;

            let cancelled_runs = AgentScheduleRun::cancel_unstarted_for_schedule(conn, schedule.id)
                .map_err(|e| match e {
                    AgentScheduleRunError::DatabaseError(err) => {
                        AgentScheduleError::DatabaseError(err)
                    }
                })?;

            info!(
                "cancelled schedule {} for user {} agent {} (cancelled_runs={})",
                schedule.uuid, self.user.uuid, self.agent.uuid, cancelled_runs,
            );

            Ok(format!(
                "Cancelled schedule {} and {} unstarted run(s).",
                schedule.uuid, cancelled_runs
            ))
        });

        match result {
            Ok(msg) => ToolResult::success(msg),
            Err(e) => {
                error!("cancel_schedule failed: {e:?}");
                ToolResult::error("Failed to cancel schedule")
            }
        }
    }
}

async fn materialize_due_schedules(state: &Arc<AppState>) -> Result<(), String> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| "database connection error".to_string())?;

    conn.transaction::<(), AgentScheduleError, _>(|conn| {
        let rows = sql_query(
            r#"
            SELECT s.id
            FROM agent_schedules s
            WHERE s.status = 'active'
              AND s.next_scheduled_for IS NOT NULL
              AND s.next_scheduled_for <= NOW()
            ORDER BY s.next_scheduled_for ASC, s.id ASC
            FOR UPDATE OF s SKIP LOCKED
            LIMIT $1
            "#,
        )
        .bind::<BigInt, _>(SCHEDULE_MATERIALIZER_BATCH_SIZE)
        .load::<DueScheduleIdRow>(conn)
        .map_err(AgentScheduleError::DatabaseError)?;

        if !rows.is_empty() {
            debug!("materializing {} due schedule(s)", rows.len());
        }

        for row in rows {
            let Some(schedule) = AgentSchedule::get_by_id(conn, row.id)? else {
                continue;
            };

            if schedule.status != SCHEDULE_STATUS_ACTIVE {
                continue;
            }

            let Some(current_due) = schedule.next_scheduled_for else {
                continue;
            };

            let spec = schedule.spec()?;
            let tz = parse_timezone_or_utc(&schedule.resolved_timezone);
            let next_scheduled_for = compute_following_next_due(&spec, tz, current_due)?;

            let stale_after_at =
                current_due + Duration::minutes(schedule.stale_after_minutes as i64);

            let new_run = NewAgentScheduleRun {
                uuid: Uuid::new_v4(),
                schedule_id: schedule.id,
                user_id: schedule.user_id,
                agent_id: schedule.agent_id,
                scheduled_for: current_due,
                stale_after_at,
                status: crate::models::agent_schedule_runs::AGENT_SCHEDULE_RUN_STATUS_PENDING
                    .to_string(),
                next_attempt_at: current_due,
            };

            let inserted_run = new_run.insert(conn).map_err(|e| match e {
                AgentScheduleRunError::DatabaseError(err) => AgentScheduleError::DatabaseError(err),
            })?;

            let new_status = if spec.schedule_kind() == "one_off" {
                SCHEDULE_STATUS_COMPLETED
            } else {
                SCHEDULE_STATUS_ACTIVE
            };

            diesel::update(agent_schedules::table.filter(agent_schedules::id.eq(schedule.id)))
                .set((
                    agent_schedules::status.eq(new_status),
                    agent_schedules::next_scheduled_for.eq(next_scheduled_for),
                    agent_schedules::last_scheduled_for.eq(Some(current_due)),
                    agent_schedules::updated_at.eq(diesel::dsl::now),
                ))
                .execute(conn)
                .map_err(AgentScheduleError::DatabaseError)?;

            debug!(
                "materialized schedule {} for user {} into run {} (scheduled_for={}, next_scheduled_for={:?})",
                schedule.uuid,
                schedule.user_id,
                inserted_run.uuid,
                current_due.format("%Y-%m-%d %H:%M:%S UTC"),
                next_scheduled_for,
            );
        }

        Ok(())
    })
    .map_err(|e| e.to_string())?;

    Ok(())
}

async fn process_schedule_run_batch(state: &Arc<AppState>) -> Result<(), String> {
    let lease_owner = format!("schedule-worker:{}:{}", std::process::id(), Uuid::new_v4());
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| "database connection error".to_string())?;

    let runs = AgentScheduleRun::lease_pending(
        &mut conn,
        SCHEDULE_RUN_BATCH_SIZE,
        &lease_owner,
        SCHEDULE_RUN_LEASE_TTL_SECONDS,
    )
    .map_err(|e| e.to_string())?;
    drop(conn);

    if runs.is_empty() {
        return Ok(());
    }

    debug!(
        "leased {} scheduled run(s) (lease_owner={})",
        runs.len(),
        lease_owner,
    );

    stream::iter(runs)
        .for_each_concurrent(SCHEDULE_RUN_MAX_CONCURRENCY, |run| {
            let state = state.clone();
            let lease_owner = lease_owner.clone();
            async move {
                if let Err(e) = process_leased_schedule_run(&state, run, &lease_owner).await {
                    error!("scheduled run processing failed: {}", e);
                }
            }
        })
        .await;

    Ok(())
}

async fn process_leased_schedule_run(
    state: &Arc<AppState>,
    run: AgentScheduleRun,
    lease_owner: &str,
) -> Result<(), String> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| "database connection error".to_string())?;

    if run.stale_after_at <= Utc::now() {
        info!(
            "expiring stale scheduled run {} for user {} (schedule_id={}, scheduled_for={}, stale_after_at={})",
            run.uuid,
            run.user_id,
            run.schedule_id,
            run.scheduled_for.format("%Y-%m-%d %H:%M:%S UTC"),
            run.stale_after_at.format("%Y-%m-%d %H:%M:%S UTC"),
        );

        record_run_transition(
            AgentScheduleRun::mark_expired(
                &mut conn,
                run.id,
                lease_owner,
                Some("schedule occurrence expired before execution"),
            )
            .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            "expired",
        );
        update_schedule_after_terminal_run(&mut conn, run.schedule_id)
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if run.first_output_at.is_some() || run.output_count > 0 {
        info!(
            "marking scheduled run {} complete after partial-output recovery (user={}, schedule_id={}, output_count={})",
            run.uuid,
            run.user_id,
            run.schedule_id,
            run.output_count,
        );

        record_run_transition(
            AgentScheduleRun::mark_completed(
                &mut conn,
                run.id,
                lease_owner,
                false,
                Some("marking completed after prior partial output recovery"),
            )
            .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            AGENT_SCHEDULE_RUN_STATUS_COMPLETED,
        );
        update_schedule_after_terminal_run(&mut conn, run.schedule_id)
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    let Some(schedule) =
        AgentSchedule::get_by_id(&mut conn, run.schedule_id).map_err(|e| e.to_string())?
    else {
        warn!(
            "scheduled run {} could not find schedule definition {}",
            run.uuid, run.schedule_id,
        );

        record_run_transition(
            AgentScheduleRun::mark_failed(
                &mut conn,
                run.id,
                lease_owner,
                Some("schedule definition missing"),
            )
            .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            "failed",
        );
        return Ok(());
    };

    if schedule.status == SCHEDULE_STATUS_CANCELLED {
        info!(
            "scheduled run {} skipped because schedule {} is cancelled",
            run.uuid, schedule.uuid,
        );

        record_run_transition(
            AgentScheduleRun::mark_cancelled(
                &mut conn,
                run.id,
                lease_owner,
                Some("schedule cancelled"),
            )
            .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            "cancelled",
        );
        return Ok(());
    }

    let Some(agent) = Agent::get_by_id(&mut conn, run.agent_id).map_err(|e| e.to_string())? else {
        warn!(
            "scheduled run {} could not find agent {} for schedule {}",
            run.uuid, run.agent_id, schedule.uuid,
        );

        record_run_transition(
            AgentScheduleRun::mark_failed(&mut conn, run.id, lease_owner, Some("agent missing"))
                .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            "failed",
        );
        return Ok(());
    };

    let Some(user) = User::get_by_uuid(&mut conn, run.user_id).map_err(|e| e.to_string())? else {
        warn!(
            "scheduled run {} could not find user {} for schedule {}",
            run.uuid, run.user_id, schedule.uuid,
        );

        record_run_transition(
            AgentScheduleRun::mark_failed(&mut conn, run.id, lease_owner, Some("user missing"))
                .map_err(|e| e.to_string())?,
            run.id,
            lease_owner,
            "failed",
        );
        return Ok(());
    };
    drop(conn);

    info!(
        "starting scheduled run {} for schedule {} user {} agent {} (attempt={}, scheduled_for={})",
        run.uuid,
        schedule.uuid,
        user.uuid,
        agent.uuid,
        run.attempt_count + 1,
        run.scheduled_for.format("%Y-%m-%d %H:%M:%S UTC"),
    );

    let user_key = state
        .get_user_key(user.uuid, None, None)
        .await
        .map_err(|e| format!("failed to derive user key: {e:?}"))?;
    let instruction = decrypt_string(&user_key, Some(&schedule.instruction_enc))
        .map_err(|e| format!("failed to decrypt schedule instruction: {e:?}"))?
        .unwrap_or_default();

    let input = build_scheduled_turn_input(&schedule, &run, &instruction);

    let (stop_tx, stop_rx) = oneshot::channel();
    let heartbeat_state = state.clone();
    let heartbeat_run_id = run.id;
    let heartbeat_owner = lease_owner.to_string();
    let heartbeat_handle = tokio::spawn(async move {
        heartbeat_schedule_run_lease(heartbeat_state, heartbeat_run_id, heartbeat_owner, stop_rx)
            .await;
    });

    let turn_result = run_scheduled_agent_turn(
        state,
        user.clone(),
        user_key,
        &agent,
        &run,
        lease_owner,
        &input,
    )
    .await;

    let _ = stop_tx.send(());
    let _ = heartbeat_handle.await;

    match turn_result {
        Ok(outcome) => {
            let mut conn = state
                .db
                .get_pool()
                .get()
                .map_err(|_| "database connection error".to_string())?;

            let push_enqueued = if !outcome.persisted_messages.is_empty() {
                let preview =
                    compose_scheduled_preview(state, &user, &outcome.persisted_messages).await;
                let target = agent_push_target(&agent);
                let first_message = &outcome.persisted_messages[0];
                match enqueue_agent_message_notification(
                    state,
                    &user,
                    target,
                    first_message.message_id,
                    &preview,
                )
                .await
                {
                    Ok(Some(_)) => true,
                    Ok(None) => false,
                    Err(e) => {
                        error!("failed to enqueue scheduled agent push: {e:?}");
                        false
                    }
                }
            } else {
                false
            };

            let terminal_error = if outcome.had_error && !outcome.persisted_messages.is_empty() {
                Some("scheduled turn ended after partial output; preserving existing output")
            } else {
                None
            };

            let message_count = outcome.persisted_messages.len();

            if outcome.had_error && outcome.persisted_messages.is_empty() {
                if run.attempt_count + 1 >= SCHEDULE_RUN_MAX_ATTEMPTS {
                    error!(
                        "scheduled run {} for schedule {} user {} failed permanently before producing output (attempt={})",
                        run.uuid,
                        schedule.uuid,
                        user.uuid,
                        run.attempt_count + 1,
                    );

                    record_run_transition(
                        AgentScheduleRun::mark_failed(
                            &mut conn,
                            run.id,
                            lease_owner,
                            Some("scheduled turn failed before producing output"),
                        )
                        .map_err(|e| e.to_string())?,
                        run.id,
                        lease_owner,
                        "failed",
                    );
                } else {
                    let retry_after_seconds = retry_backoff_seconds(run.attempt_count + 1);
                    warn!(
                        "retrying scheduled run {} for schedule {} user {} after {}s (attempt={})",
                        run.uuid,
                        schedule.uuid,
                        user.uuid,
                        retry_after_seconds,
                        run.attempt_count + 1,
                    );

                    record_run_transition(
                        AgentScheduleRun::mark_retry(
                            &mut conn,
                            run.id,
                            lease_owner,
                            Some("scheduled turn failed before producing output"),
                            retry_after_seconds,
                        )
                        .map_err(|e| e.to_string())?,
                        run.id,
                        lease_owner,
                        "retry",
                    );
                }
            } else {
                record_run_transition(
                    AgentScheduleRun::mark_completed(
                        &mut conn,
                        run.id,
                        lease_owner,
                        push_enqueued,
                        terminal_error,
                    )
                    .map_err(|e| e.to_string())?,
                    run.id,
                    lease_owner,
                    AGENT_SCHEDULE_RUN_STATUS_COMPLETED,
                );
                update_schedule_after_terminal_run(&mut conn, run.schedule_id)
                    .map_err(|e| e.to_string())?;

                info!(
                    "completed scheduled run {} for schedule {} user {} agent {} (messages={}, push_enqueued={}, had_error={})",
                    run.uuid,
                    schedule.uuid,
                    user.uuid,
                    agent.uuid,
                    message_count,
                    push_enqueued,
                    outcome.had_error,
                );
            }
        }
        Err(e) => {
            let mut conn = state
                .db
                .get_pool()
                .get()
                .map_err(|_| "database connection error".to_string())?;
            if run.attempt_count + 1 >= SCHEDULE_RUN_MAX_ATTEMPTS {
                error!(
                    "scheduled run {} for user {} failed permanently after execution error (schedule_id={}, attempt={})",
                    run.uuid,
                    run.user_id,
                    run.schedule_id,
                    run.attempt_count + 1,
                );

                record_run_transition(
                    AgentScheduleRun::mark_failed(&mut conn, run.id, lease_owner, Some(&e))
                        .map_err(|err| err.to_string())?,
                    run.id,
                    lease_owner,
                    "failed",
                );
            } else {
                let retry_after_seconds = retry_backoff_seconds(run.attempt_count + 1);
                warn!(
                    "retrying scheduled run {} for user {} after execution error in {}s (schedule_id={}, attempt={})",
                    run.uuid,
                    run.user_id,
                    retry_after_seconds,
                    run.schedule_id,
                    run.attempt_count + 1,
                );

                record_run_transition(
                    AgentScheduleRun::mark_retry(
                        &mut conn,
                        run.id,
                        lease_owner,
                        Some(&e),
                        retry_after_seconds,
                    )
                    .map_err(|err| err.to_string())?,
                    run.id,
                    lease_owner,
                    "retry",
                );
            }
        }
    }

    Ok(())
}

async fn run_scheduled_agent_turn(
    state: &Arc<AppState>,
    user: User,
    user_key: SecretKey,
    agent: &Agent,
    run: &AgentScheduleRun,
    lease_owner: &str,
    input: &str,
) -> Result<ScheduledTurnOutcome, String> {
    let runtime_result = if agent.kind == AGENT_KIND_MAIN {
        runtime::AgentRuntime::new_main(state.clone(), user.clone(), user_key).await
    } else if agent.kind == AGENT_KIND_SUBAGENT {
        runtime::AgentRuntime::new_subagent(state.clone(), user.clone(), user_key, agent.uuid).await
    } else {
        return Err(format!(
            "unsupported agent kind for schedule: {}",
            agent.kind
        ));
    };

    let mut runtime =
        runtime_result.map_err(|e| format!("failed to initialize agent runtime: {e:?}"))?;
    runtime.clear_tool_results();
    runtime
        .maybe_compact()
        .await
        .map_err(|e| format!("scheduled compaction failed: {e:?}"))?;

    let max_steps = runtime.max_steps();
    let mut persisted_messages = Vec::new();
    let mut had_error = false;

    debug!(
        "running scheduled agent turn for run {} user {} agent {} (kind={}, max_steps={})",
        run.uuid, user.uuid, agent.uuid, agent.kind, max_steps,
    );

    'steps: for step_num in 0..max_steps {
        match runtime.step(input, step_num == 0).await {
            Ok(result) => {
                trace!(
                    "scheduled run {} step {} produced {} tool call(s), {} message(s), done={}",
                    run.uuid,
                    step_num,
                    result.executed_tools.len(),
                    result.messages.len(),
                    result.done,
                );

                for executed in &result.executed_tools {
                    if let Err(e) = runtime
                        .insert_tool_call_and_output(&executed.tool_call, &executed.result)
                        .await
                    {
                        error!("failed to persist scheduled tool call: {e:?}");
                    }
                }

                for msg in result.messages {
                    match runtime.insert_assistant_message(&msg).await {
                        Ok(inserted) => {
                            persisted_messages.push(PersistedScheduledMessage {
                                message_id: inserted.uuid,
                                text: msg.clone(),
                            });

                            let mut conn = state
                                .db
                                .get_pool()
                                .get()
                                .map_err(|_| "database connection error".to_string())?;
                            let write_result = AgentScheduleRun::record_output(
                                &mut conn,
                                run.id,
                                lease_owner,
                                inserted.uuid,
                            )
                            .map_err(|e| e.to_string())?;

                            if !write_result.was_applied() {
                                return Err(
                                    "lost lease while recording scheduled output".to_string()
                                );
                            }
                        }
                        Err(e) => {
                            error!("failed to persist scheduled assistant message: {e:?}");
                            had_error = true;
                            break 'steps;
                        }
                    }
                }

                if result.done {
                    break 'steps;
                }
            }
            Err(e) => {
                error!("scheduled agent step failed: {e:?}");
                had_error = true;
                break 'steps;
            }
        }
    }

    Ok(ScheduledTurnOutcome {
        persisted_messages,
        had_error,
    })
}

async fn heartbeat_schedule_run_lease(
    state: Arc<AppState>,
    run_id: i64,
    lease_owner: String,
    mut stop_rx: oneshot::Receiver<()>,
) {
    let mut interval = tokio::time::interval(TokioDuration::from_secs(
        SCHEDULE_RUN_HEARTBEAT_INTERVAL_SECONDS,
    ));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let mut conn = match state.db.get_pool().get() {
                    Ok(conn) => conn,
                    Err(_) => {
                        warn!("stopping schedule heartbeat for run {} due to database connection error", run_id);
                        break;
                    }
                };

                match AgentScheduleRun::renew_lease(
                    &mut conn,
                    run_id,
                    &lease_owner,
                    SCHEDULE_RUN_LEASE_TTL_SECONDS,
                ) {
                    Ok(result) if result.was_applied() => {}
                    Ok(_) => {
                        warn!("stopping schedule heartbeat for run {} because lease renewal was not applied", run_id);
                        break;
                    }
                    Err(e) => {
                        warn!("stopping schedule heartbeat for run {} because lease renewal failed: {e:?}", run_id);
                        break;
                    }
                }
            }
            _ = &mut stop_rx => break,
        }
    }
}

async fn compose_scheduled_preview(
    state: &Arc<AppState>,
    user: &User,
    messages: &[PersistedScheduledMessage],
) -> String {
    let Some(first_message) = messages.first() else {
        return String::new();
    };

    trace!(
        "composing scheduled preview for user {} from {} message(s)",
        user.uuid,
        messages.len(),
    );

    let joined_messages = messages
        .iter()
        .enumerate()
        .map(|(index, message)| format!("{}. {}", index + 1, message.text.trim()))
        .collect::<Vec<_>>()
        .join("\n\n");

    let request = serde_json::json!({
        "model": "llama-3.3-70b",
        "messages": [
            {
                "role": "system",
                "content": "You compress assistant messages into a short encrypted notification preview. Preserve tone and intent. Do not add facts. Return only 1-2 short sentences, ideally under 160 characters."
            },
            {
                "role": "user",
                "content": format!("Create a notification preview from these assistant messages:\n\n{}", joined_messages)
            }
        ],
        "temperature": 0.2,
        "max_tokens": 80,
        "stream": false
    });

    let headers = axum::http::HeaderMap::new();
    let billing_context = BillingContext::new(AuthMethod::Jwt, "llama-3.3-70b".to_string());

    match get_chat_completion_response(state, user, request, &headers, billing_context).await {
        Ok(mut completion) => match completion.stream.recv().await {
            Some(CompletionChunk::FullResponse(response_json)) => response_json
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str())
                .map(|content| content.trim().to_string())
                .filter(|content| !content.is_empty())
                .unwrap_or_else(|| {
                    trace!(
                        "scheduled preview composer returned empty content for user {}, using first assistant message",
                        user.uuid,
                    );
                    first_message.text.clone()
                }),
            _ => {
                trace!(
                    "scheduled preview composer returned no full response for user {}, using first assistant message",
                    user.uuid,
                );
                first_message.text.clone()
            }
        },
        Err(e) => {
            warn!("scheduled preview composition failed, falling back to first message: {e:?}");
            first_message.text.clone()
        }
    }
}

fn build_scheduled_turn_input(
    schedule: &AgentSchedule,
    run: &AgentScheduleRun,
    instruction: &str,
) -> String {
    format!(
        "[SCHEDULED EVENT]\nThis is an internal scheduled wakeup, not a new live user message.\n\nSchedule ID: {}\nRun ID: {}\nDescription: {}\nScheduled for (UTC): {}\nCurrent resolved timezone: {}\nInstruction for your future self:\n{}\n\nDecide what, if anything, the user should see right now. You may send messages, call tools, or do nothing if the event is no longer relevant.",
        schedule.uuid,
        run.uuid,
        schedule.description.trim(),
        run.scheduled_for.format("%Y-%m-%d %H:%M:%S UTC"),
        schedule.resolved_timezone,
        instruction.trim()
    )
}

fn parse_schedule_spec_from_args(args: &HashMap<String, String>) -> Result<ScheduleSpec, String> {
    let schedule_kind = args
        .get("schedule_kind")
        .map(|s| s.trim())
        .ok_or_else(|| "'schedule_kind' argument required".to_string())?;

    let spec = match schedule_kind {
        "one_off" => {
            let local_date = args
                .get("local_date")
                .map(|s| s.trim())
                .ok_or_else(|| "'local_date' is required for one_off schedules".to_string())?;
            let local_time = args
                .get("local_time")
                .map(|s| s.trim())
                .ok_or_else(|| "'local_time' is required for one_off schedules".to_string())?;

            ScheduleSpec::OneOff {
                local_date: local_date.to_string(),
                local_time: local_time.to_string(),
            }
        }
        "recurring" => {
            let recurrence_type =
                args.get("recurrence_type")
                    .map(|s| s.trim())
                    .ok_or_else(|| {
                        "'recurrence_type' is required for recurring schedules".to_string()
                    })?;

            match recurrence_type {
                "interval" => {
                    let every_n = args
                        .get("every_n")
                        .map(|s| s.trim())
                        .ok_or_else(|| "'every_n' is required for interval schedules".to_string())?
                        .parse::<u32>()
                        .map_err(|_| "'every_n' must be a positive integer".to_string())?;
                    let interval_unit = args
                        .get("interval_unit")
                        .map(|s| s.trim())
                        .unwrap_or("hours");

                    if interval_unit != "hours" {
                        return Err(
                            "v1 interval schedules currently support interval_unit='hours' only"
                                .to_string(),
                        );
                    }

                    ScheduleSpec::Interval {
                        every_n,
                        interval_unit: crate::models::agent_schedules::ScheduleIntervalUnit::Hours,
                    }
                }
                "daily" => {
                    let local_time = args.get("local_time").map(|s| s.trim()).ok_or_else(|| {
                        "'local_time' is required for daily schedules".to_string()
                    })?;
                    ScheduleSpec::Daily {
                        local_time: local_time.to_string(),
                    }
                }
                "weekly" => {
                    let local_time = args.get("local_time").map(|s| s.trim()).ok_or_else(|| {
                        "'local_time' is required for weekly schedules".to_string()
                    })?;
                    let weekdays_value = args
                        .get("weekdays")
                        .map(|s| s.trim())
                        .ok_or_else(|| "'weekdays' is required for weekly schedules".to_string())?;
                    let weekdays = parse_weekdays_csv(weekdays_value)?;
                    ScheduleSpec::Weekly {
                        local_time: local_time.to_string(),
                        weekdays,
                    }
                }
                _ => return Err(
                    "'recurrence_type' must be interval, daily, or weekly for recurring schedules"
                        .to_string(),
                ),
            }
        }
        _ => return Err("'schedule_kind' must be 'one_off' or 'recurring'".to_string()),
    };

    spec.validate().map_err(|e| e.to_string())?;
    Ok(spec)
}

fn parse_weekdays_csv(value: &str) -> Result<Vec<ScheduleWeekday>, String> {
    let mut weekdays = Vec::new();

    for raw in value.split(',') {
        let normalized = raw.trim().to_lowercase();
        if normalized.is_empty() {
            continue;
        }

        let weekday = match normalized.as_str() {
            "weekdays" => {
                return Ok(vec![
                    ScheduleWeekday::Monday,
                    ScheduleWeekday::Tuesday,
                    ScheduleWeekday::Wednesday,
                    ScheduleWeekday::Thursday,
                    ScheduleWeekday::Friday,
                ])
            }
            "monday" | "mon" => ScheduleWeekday::Monday,
            "tuesday" | "tue" | "tues" => ScheduleWeekday::Tuesday,
            "wednesday" | "wed" => ScheduleWeekday::Wednesday,
            "thursday" | "thu" | "thurs" => ScheduleWeekday::Thursday,
            "friday" | "fri" => ScheduleWeekday::Friday,
            "saturday" | "sat" => ScheduleWeekday::Saturday,
            "sunday" | "sun" => ScheduleWeekday::Sunday,
            _ => return Err(format!("invalid weekday '{raw}'")),
        };

        if !weekdays.contains(&weekday) {
            weekdays.push(weekday);
        }
    }

    if weekdays.is_empty() {
        return Err("at least one valid weekday is required".to_string());
    }

    Ok(weekdays)
}

async fn resolve_initial_timezone_name(
    state: &Arc<AppState>,
    user: &User,
    user_key: &SecretKey,
    timezone_mode: &str,
    fixed_timezone: Option<&str>,
) -> Result<String, ApiError> {
    if timezone_mode == SCHEDULE_TIMEZONE_MODE_FIXED {
        return Ok(parse_timezone_or_utc(fixed_timezone.unwrap_or("UTC"))
            .name()
            .to_string());
    }

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    load_user_timezone_name(&mut conn, user_key, user.uuid)
        .map(|timezone| timezone.unwrap_or_else(|| "UTC".to_string()))
        .map_err(|_| ApiError::InternalServerError)
}

fn load_user_timezone_name(
    conn: &mut PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
) -> Result<Option<String>, AgentScheduleError> {
    let pref = UserPreference::get_by_user_and_key(conn, user_id, USER_PREFERENCE_TIMEZONE)
        .map_err(|e| match e {
            crate::models::user_preferences::UserPreferenceError::DatabaseError(err) => {
                AgentScheduleError::DatabaseError(err)
            }
            crate::models::user_preferences::UserPreferenceError::InvalidPreference(msg) => {
                AgentScheduleError::InvalidSpec(msg)
            }
        })?;

    let Some(pref) = pref else {
        return Ok(None);
    };

    decrypt_string(user_key, Some(&pref.value_enc))
        .map_err(|e| AgentScheduleError::InvalidSpec(format!("decrypt timezone: {e:?}")))
}

fn compute_initial_next_due(
    spec: &ScheduleSpec,
    timezone: Tz,
    now: DateTime<Utc>,
) -> Result<DateTime<Utc>, AgentScheduleError> {
    match spec {
        ScheduleSpec::OneOff {
            local_date,
            local_time,
        } => resolve_local_datetime(
            timezone,
            parse_local_date(local_date)?,
            parse_local_time(local_time)?,
        ),
        ScheduleSpec::Interval { every_n, .. } => Ok(now + Duration::hours(*every_n as i64)),
        ScheduleSpec::Daily { local_time } => {
            next_daily_occurrence(timezone, parse_local_time(local_time)?, now)
        }
        ScheduleSpec::Weekly {
            local_time,
            weekdays,
        } => next_weekly_occurrence(timezone, parse_local_time(local_time)?, weekdays, now),
    }
}

fn compute_following_next_due(
    spec: &ScheduleSpec,
    timezone: Tz,
    current_due: DateTime<Utc>,
) -> Result<Option<DateTime<Utc>>, AgentScheduleError> {
    match spec {
        ScheduleSpec::OneOff { .. } => Ok(None),
        ScheduleSpec::Interval { every_n, .. } => {
            Ok(Some(current_due + Duration::hours(*every_n as i64)))
        }
        ScheduleSpec::Daily { local_time } => Ok(Some(next_daily_occurrence(
            timezone,
            parse_local_time(local_time)?,
            current_due,
        )?)),
        ScheduleSpec::Weekly {
            local_time,
            weekdays,
        } => Ok(Some(next_weekly_occurrence(
            timezone,
            parse_local_time(local_time)?,
            weekdays,
            current_due,
        )?)),
    }
}

fn recompute_next_due_for_timezone_change(
    schedule: &AgentSchedule,
    spec: &ScheduleSpec,
    timezone: Tz,
    now: DateTime<Utc>,
) -> Result<Option<DateTime<Utc>>, AgentScheduleError> {
    match spec {
        ScheduleSpec::OneOff {
            local_date,
            local_time,
        } => resolve_local_datetime(
            timezone,
            parse_local_date(local_date)?,
            parse_local_time(local_time)?,
        )
        .map(Some),
        ScheduleSpec::Interval { .. } => Ok(schedule.next_scheduled_for),
        ScheduleSpec::Daily { local_time } => Ok(Some(next_daily_occurrence(
            timezone,
            parse_local_time(local_time)?,
            now,
        )?)),
        ScheduleSpec::Weekly {
            local_time,
            weekdays,
        } => Ok(Some(next_weekly_occurrence(
            timezone,
            parse_local_time(local_time)?,
            weekdays,
            now,
        )?)),
    }
}

fn next_daily_occurrence(
    timezone: Tz,
    local_time: chrono::NaiveTime,
    after: DateTime<Utc>,
) -> Result<DateTime<Utc>, AgentScheduleError> {
    let local_after = after.with_timezone(&timezone);
    let mut date = local_after.date_naive();

    for _ in 0..3 {
        let candidate = resolve_local_datetime(timezone, date, local_time)?;
        if candidate > after {
            return Ok(candidate);
        }
        date = date.succ_opt().ok_or_else(|| {
            AgentScheduleError::InvalidSpec("daily recurrence overflowed date range".to_string())
        })?;
    }

    Err(AgentScheduleError::InvalidSpec(
        "failed to compute next daily occurrence".to_string(),
    ))
}

fn next_weekly_occurrence(
    timezone: Tz,
    local_time: chrono::NaiveTime,
    weekdays: &[ScheduleWeekday],
    after: DateTime<Utc>,
) -> Result<DateTime<Utc>, AgentScheduleError> {
    let local_after = after.with_timezone(&timezone);
    let start_date = local_after.date_naive();

    for offset in 0..14 {
        let date = start_date
            .checked_add_signed(Duration::days(offset))
            .ok_or_else(|| {
                AgentScheduleError::InvalidSpec(
                    "weekly recurrence overflowed date range".to_string(),
                )
            })?;

        if !weekdays
            .iter()
            .any(|weekday| weekday.to_chrono() == date.weekday())
        {
            continue;
        }

        let candidate = resolve_local_datetime(timezone, date, local_time)?;
        if candidate > after {
            return Ok(candidate);
        }
    }

    Err(AgentScheduleError::InvalidSpec(
        "failed to compute next weekly occurrence".to_string(),
    ))
}

fn resolve_local_datetime(
    timezone: Tz,
    local_date: chrono::NaiveDate,
    local_time: chrono::NaiveTime,
) -> Result<DateTime<Utc>, AgentScheduleError> {
    let naive = NaiveDateTime::new(local_date, local_time);

    match timezone.from_local_datetime(&naive) {
        chrono::LocalResult::Single(dt) => Ok(dt.with_timezone(&Utc)),
        chrono::LocalResult::Ambiguous(first, _) => Ok(first.with_timezone(&Utc)),
        chrono::LocalResult::None => {
            for minute_offset in 1..=180 {
                let shifted = naive + Duration::minutes(minute_offset);
                match timezone.from_local_datetime(&shifted) {
                    chrono::LocalResult::Single(dt) => return Ok(dt.with_timezone(&Utc)),
                    chrono::LocalResult::Ambiguous(first, _) => {
                        return Ok(first.with_timezone(&Utc))
                    }
                    chrono::LocalResult::None => continue,
                }
            }

            Err(AgentScheduleError::InvalidSpec(
                "failed to resolve local datetime in timezone".to_string(),
            ))
        }
    }
}

fn parse_timezone_or_utc(value: &str) -> Tz {
    value.parse::<Tz>().unwrap_or(chrono_tz::UTC)
}

fn format_local_time(value: DateTime<Utc>, timezone: Tz) -> String {
    let localized = value.with_timezone(&timezone);
    format!(
        "{} ({})",
        localized.format("%Y-%m-%d %H:%M:%S"),
        timezone.name()
    )
}

fn agent_push_target(agent: &Agent) -> AgentPushTarget {
    if agent.kind == AGENT_KIND_SUBAGENT {
        AgentPushTarget::Subagent(agent.uuid)
    } else {
        AgentPushTarget::Main
    }
}

fn truncate_for_description(instruction: &str) -> String {
    let trimmed = instruction.trim();
    if trimmed.len() <= 80 {
        return trimmed.to_string();
    }

    let mut end = 80;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", trimmed[..end].trim_end())
}

fn retry_backoff_seconds(attempt_count: i32) -> i32 {
    let capped_attempt = attempt_count.clamp(1, 6);
    let seconds = 15_i64 * (1_i64 << (capped_attempt - 1));
    seconds.min(SCHEDULE_RUN_MAX_RETRY_BACKOFF_SECONDS as i64) as i32
}

fn update_schedule_after_terminal_run(
    conn: &mut PgConnection,
    schedule_id: i64,
) -> Result<(), AgentScheduleError> {
    diesel::update(agent_schedules::table.filter(agent_schedules::id.eq(schedule_id)))
        .set((
            agent_schedules::run_count.eq(agent_schedules::run_count + 1),
            agent_schedules::last_run_at.eq(diesel::dsl::now),
            agent_schedules::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)
        .map_err(AgentScheduleError::DatabaseError)?;
    Ok(())
}

fn record_run_transition(
    write_result: AgentScheduleRunWriteResult,
    run_id: i64,
    lease_owner: &str,
    transition: &str,
) {
    if !write_result.was_applied() {
        debug!(
            "scheduled run {} lost lease before marking {} (lease_owner={})",
            run_id, transition, lease_owner
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_weekdays_csv() {
        let weekdays = parse_weekdays_csv("monday,friday").unwrap();
        assert_eq!(
            weekdays,
            vec![ScheduleWeekday::Monday, ScheduleWeekday::Friday]
        );
    }

    #[test]
    fn rejects_invalid_weekday() {
        assert!(parse_weekdays_csv("noday").is_err());
    }

    #[test]
    fn interval_backoff_is_capped() {
        assert_eq!(retry_backoff_seconds(1), 15);
        assert_eq!(retry_backoff_seconds(6), 480);
        assert_eq!(retry_backoff_seconds(10), 480);
    }

    #[test]
    fn daily_occurrence_rolls_to_next_day_when_time_has_passed() {
        let now = Utc.with_ymd_and_hms(2026, 3, 23, 15, 0, 0).unwrap();
        let next =
            next_daily_occurrence(chrono_tz::UTC, parse_local_time("09:00").unwrap(), now).unwrap();

        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 24, 9, 0, 0).unwrap());
    }

    #[test]
    fn weekly_occurrence_picks_next_matching_weekday() {
        let now = Utc.with_ymd_and_hms(2026, 3, 23, 15, 0, 0).unwrap();
        let next = next_weekly_occurrence(
            chrono_tz::UTC,
            parse_local_time("09:00").unwrap(),
            &[ScheduleWeekday::Friday],
            now,
        )
        .unwrap();

        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 27, 9, 0, 0).unwrap());
    }

    #[test]
    fn resolve_local_datetime_rolls_forward_out_of_dst_gap() {
        let next = resolve_local_datetime(
            chrono_tz::America::New_York,
            parse_local_date("2026-03-08").unwrap(),
            parse_local_time("02:30").unwrap(),
        )
        .unwrap();

        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 8, 7, 0, 0).unwrap());
    }
}
