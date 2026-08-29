//! Bounded, transport-v2-only projections for conversations and stored responses.
//!
//! The transport-v1 loaders intentionally remain untouched. Every v2 read that
//! can materialize persisted ciphertext first measures the exact returned page
//! or object inside a repeatable-read transaction, validates the aggregate
//! plaintext and database-controlled string sizes, and only then fetches the
//! narrow ciphertext projection. List sentinels are metadata-only and are
//! removed before ciphertext sizing and fetch.
//!
//! Mutations are owner-scoped and return only the metadata required to build
//! the existing application response. Conversation updates deliberately retain
//! v1's last-writer-wins behavior; the preloaded response can be stale if two
//! updates race, but no unbounded stored value is reloaded after the write.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use diesel::dsl::sql;
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{BigInt, Integer, Nullable, Text, Timestamptz};
use diesel::{Connection, OptionalExtension, QueryableByName};
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

use crate::models::responses::{ConversationProjectFilter, ResponseStatus};
use crate::models::schema::{conversation_projects, conversations, responses};

const AES_GCM_STORAGE_OVERHEAD_BYTES: usize = 12 + 16;
const MAX_PAGE_SIZE: i64 = 100;
const MAX_CONVERSATION_BATCH_SIZE: usize = 20;

/// A hard row-count guard for the unpaginated response-retrieval contract.
///
/// The byte budget remains authoritative. This separate guard prevents a
/// database attacker from replacing a bounded response with millions of tiny
/// child rows and forcing unbounded metadata allocation before byte accounting.
const MAX_STORED_RESPONSE_ITEMS: i64 = 10_000;
const ACCOUNTED_BYTES_PER_STORED_ITEM: usize = 256;

type Pool = diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum StoredConversationError {
    #[error("conversation not found")]
    ConversationNotFound,
    #[error("conversation project not found")]
    ConversationProjectNotFound,
    #[error("conversation item not found")]
    ConversationItemNotFound,
    #[error("response not found")]
    ResponseNotFound,
    #[error("request violates the existing conversation contract")]
    Validation,
    #[error("stored output exceeds the logical response limit")]
    OutputTooLarge,
    #[error("stored output changed within a bounded snapshot")]
    InconsistentSnapshot,
    #[error("database connection unavailable")]
    Connection,
    #[error("database error: {0}")]
    Database(#[from] diesel::result::Error),
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = conversations)]
pub(crate) struct ConversationCiphertextRow {
    pub(crate) id: i64,
    pub(crate) uuid: Uuid,
    pub(crate) metadata_enc: Option<Vec<u8>>,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
    pub(crate) project_id: Option<i64>,
    pub(crate) is_pinned: bool,
    pub(crate) last_activity_at: DateTime<Utc>,
}

impl Zeroize for ConversationCiphertextRow {
    fn zeroize(&mut self) {
        if let Some(metadata) = self.metadata_enc.as_mut() {
            metadata.zeroize();
        }
    }
}

impl Drop for ConversationCiphertextRow {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub(crate) struct StoredConversation {
    pub(crate) conversation: ConversationCiphertextRow,
    pub(crate) project_uuid: Option<Uuid>,
}

pub(crate) struct ConversationMutationMetadata {
    pub(crate) uuid: Uuid,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
    pub(crate) project_uuid: Option<Uuid>,
    pub(crate) is_pinned: bool,
    pub(crate) last_activity_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectAssignmentUpdate {
    Unchanged,
    Set(Option<i64>),
}

#[derive(AsChangeset, Default)]
#[diesel(table_name = conversations)]
struct ConversationChanges<'a> {
    metadata_enc: Option<&'a Vec<u8>>,
    project_id: Option<Option<i64>>,
    is_pinned: Option<bool>,
}

#[derive(QueryableByName)]
struct ItemMeasurement {
    #[diesel(sql_type = Text)]
    message_type: String,
    #[diesel(sql_type = BigInt)]
    id: i64,
    #[diesel(sql_type = Integer)]
    type_rank: i32,
    #[diesel(sql_type = diesel::sql_types::Uuid)]
    uuid: Uuid,
    #[diesel(sql_type = Timestamptz)]
    created_at: DateTime<Utc>,
    #[diesel(sql_type = Nullable<BigInt>)]
    content_length: Option<i64>,
    #[diesel(sql_type = Nullable<BigInt>)]
    status_length: Option<i64>,
    #[diesel(sql_type = Nullable<BigInt>)]
    model_length: Option<i64>,
    #[diesel(sql_type = Nullable<BigInt>)]
    finish_reason_length: Option<i64>,
    #[diesel(sql_type = Nullable<BigInt>)]
    tool_name_length: Option<i64>,
}

#[derive(QueryableByName, Debug)]
pub(crate) struct StoredConversationItem {
    #[diesel(sql_type = Text)]
    pub(crate) message_type: String,
    #[diesel(sql_type = BigInt)]
    pub(crate) id: i64,
    #[diesel(sql_type = Integer)]
    pub(crate) type_rank: i32,
    #[diesel(sql_type = diesel::sql_types::Uuid)]
    pub(crate) uuid: Uuid,
    #[diesel(sql_type = Nullable<diesel::sql_types::Bytea>)]
    pub(crate) content_enc: Option<Vec<u8>>,
    #[diesel(sql_type = Nullable<Text>)]
    pub(crate) status: Option<String>,
    #[diesel(sql_type = Timestamptz)]
    pub(crate) created_at: DateTime<Utc>,
    #[diesel(sql_type = Nullable<Text>)]
    pub(crate) model: Option<String>,
    #[diesel(sql_type = Nullable<Integer>)]
    pub(crate) token_count: Option<i32>,
    #[diesel(sql_type = Nullable<diesel::sql_types::Uuid>)]
    pub(crate) tool_call_id: Option<Uuid>,
    #[diesel(sql_type = Nullable<Text>)]
    pub(crate) finish_reason: Option<String>,
    #[diesel(sql_type = Nullable<Text>)]
    pub(crate) tool_name: Option<String>,
}

impl Zeroize for StoredConversationItem {
    fn zeroize(&mut self) {
        if let Some(content) = self.content_enc.as_mut() {
            content.zeroize();
        }
    }
}

impl Drop for StoredConversationItem {
    fn drop(&mut self) {
        self.zeroize();
    }
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = responses)]
pub(crate) struct StoredResponseMetadata {
    pub(crate) id: i64,
    pub(crate) uuid: Uuid,
    pub(crate) conversation_id: i64,
    pub(crate) status: ResponseStatus,
    pub(crate) model: String,
    pub(crate) created_at: DateTime<Utc>,
}

pub(crate) struct StoredResponse {
    pub(crate) response: StoredResponseMetadata,
    pub(crate) items: Vec<StoredConversationItem>,
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = responses)]
pub(crate) struct ResponseMutationMetadata {
    pub(crate) uuid: Uuid,
    pub(crate) status: ResponseStatus,
    pub(crate) model: String,
    pub(crate) created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpectedItemLengths {
    content: Option<usize>,
    status: Option<usize>,
    model: Option<usize>,
    finish_reason: Option<usize>,
    tool_name: Option<usize>,
}

fn checked_length(length: i64) -> Result<usize, StoredConversationError> {
    usize::try_from(length).map_err(|_| StoredConversationError::InconsistentSnapshot)
}

fn checked_ciphertext_length(length: i64) -> Result<(usize, usize), StoredConversationError> {
    let ciphertext_length = checked_length(length)?;
    let plaintext_length = ciphertext_length
        .checked_sub(AES_GCM_STORAGE_OVERHEAD_BYTES)
        .ok_or(StoredConversationError::InconsistentSnapshot)?;
    Ok((ciphertext_length, plaintext_length))
}

fn account_bytes(
    total: &mut usize,
    bytes: usize,
    logical_body_limit: usize,
) -> Result<(), StoredConversationError> {
    *total = total
        .checked_add(bytes)
        .ok_or(StoredConversationError::OutputTooLarge)?;
    if *total > logical_body_limit {
        return Err(StoredConversationError::OutputTooLarge);
    }
    Ok(())
}

fn account_ciphertext(
    total: &mut usize,
    length: Option<i64>,
    logical_body_limit: usize,
) -> Result<Option<usize>, StoredConversationError> {
    let Some(length) = length else {
        return Ok(None);
    };
    let (ciphertext_length, plaintext_length) = checked_ciphertext_length(length)?;
    account_bytes(total, plaintext_length, logical_body_limit)?;
    Ok(Some(ciphertext_length))
}

fn account_string(
    total: &mut usize,
    length: Option<i64>,
    logical_body_limit: usize,
) -> Result<Option<usize>, StoredConversationError> {
    let Some(length) = length else {
        return Ok(None);
    };
    let length = checked_length(length)?;
    account_bytes(total, length, logical_body_limit)?;
    Ok(Some(length))
}

fn validate_page_limit(limit: i64) -> Result<(), StoredConversationError> {
    if (1..=MAX_PAGE_SIZE).contains(&limit) {
        Ok(())
    } else {
        Err(StoredConversationError::Validation)
    }
}

fn drop_page_sentinel<T>(rows: &mut Vec<T>, limit: i64) -> Result<bool, StoredConversationError> {
    let limit = usize::try_from(limit).map_err(|_| StoredConversationError::Validation)?;
    let has_more = rows.len() > limit;
    if has_more {
        rows.pop();
    }
    Ok(has_more)
}

fn validate_item_measurements(
    measurements: &[ItemMeasurement],
    logical_body_limit: usize,
    initial_total: usize,
) -> Result<Vec<ExpectedItemLengths>, StoredConversationError> {
    if initial_total > logical_body_limit {
        return Err(StoredConversationError::OutputTooLarge);
    }
    let mut total = initial_total;
    account_bytes(
        &mut total,
        measurements
            .len()
            .checked_mul(ACCOUNTED_BYTES_PER_STORED_ITEM)
            .ok_or(StoredConversationError::OutputTooLarge)?,
        logical_body_limit,
    )?;
    measurements
        .iter()
        .map(|measurement| {
            Ok(ExpectedItemLengths {
                content: account_ciphertext(
                    &mut total,
                    measurement.content_length,
                    logical_body_limit,
                )?,
                status: account_string(&mut total, measurement.status_length, logical_body_limit)?,
                model: account_string(&mut total, measurement.model_length, logical_body_limit)?,
                finish_reason: account_string(
                    &mut total,
                    measurement.finish_reason_length,
                    logical_body_limit,
                )?,
                tool_name: account_string(
                    &mut total,
                    measurement.tool_name_length,
                    logical_body_limit,
                )?,
            })
        })
        .collect()
}

fn option_len(value: Option<&Vec<u8>>) -> Option<usize> {
    value.map(Vec::len)
}

fn option_string_len(value: Option<&String>) -> Option<usize> {
    value.map(String::len)
}

fn validate_fetched_items(
    rows: &mut Vec<StoredConversationItem>,
    measurements: &[ItemMeasurement],
    expected: &[ExpectedItemLengths],
) -> Result<(), StoredConversationError> {
    if rows.len() != measurements.len() || rows.len() != expected.len() {
        rows.zeroize();
        return Err(StoredConversationError::InconsistentSnapshot);
    }

    for ((row, measurement), expected) in rows.iter().zip(measurements).zip(expected) {
        if row.message_type != measurement.message_type
            || row.id != measurement.id
            || row.type_rank != measurement.type_rank
            || row.uuid != measurement.uuid
            || row.created_at != measurement.created_at
            || option_len(row.content_enc.as_ref()) != expected.content
            || option_string_len(row.status.as_ref()) != expected.status
            || option_string_len(row.model.as_ref()) != expected.model
            || option_string_len(row.finish_reason.as_ref()) != expected.finish_reason
            || option_string_len(row.tool_name.as_ref()) != expected.tool_name
        {
            rows.zeroize();
            return Err(StoredConversationError::InconsistentSnapshot);
        }
    }

    Ok(())
}

fn project_uuid_for_internal_id(
    conn: &mut PgConnection,
    user_id: Uuid,
    project_id: Option<i64>,
) -> Result<Option<Uuid>, StoredConversationError> {
    let Some(project_id) = project_id else {
        return Ok(None);
    };
    conversation_projects::table
        .filter(conversation_projects::id.eq(project_id))
        .filter(conversation_projects::user_id.eq(user_id))
        .select(conversation_projects::uuid)
        .first::<Uuid>(conn)
        .optional()?
        .map(Some)
        .ok_or(StoredConversationError::InconsistentSnapshot)
}

/// Resolve a public project UUID without loading its encrypted name.
pub(crate) fn resolve_project_id(
    pool: &Pool,
    user_id: Uuid,
    project_uuid: Uuid,
) -> Result<i64, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conversation_projects::table
        .filter(conversation_projects::uuid.eq(project_uuid))
        .filter(conversation_projects::user_id.eq(user_id))
        .select(conversation_projects::id)
        .first::<i64>(&mut conn)
        .optional()?
        .ok_or(StoredConversationError::ConversationProjectNotFound)
}

pub(crate) fn get_conversation(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<StoredConversation, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            let (conversation_id, metadata_length, project_id) = conversations::table
                .filter(conversations::uuid.eq(conversation_uuid))
                .filter(conversations::user_id.eq(user_id))
                .select((
                    conversations::id,
                    sql::<Nullable<BigInt>>("octet_length(metadata_enc)::bigint"),
                    conversations::project_id,
                ))
                .first::<(i64, Option<i64>, Option<i64>)>(conn)
                .optional()?
                .ok_or(StoredConversationError::ConversationNotFound)?;

            let mut total = 0;
            let expected_metadata =
                account_ciphertext(&mut total, metadata_length, logical_body_limit)?;
            let project_uuid = project_uuid_for_internal_id(conn, user_id, project_id)?;

            let mut conversation = conversations::table
                .filter(conversations::id.eq(conversation_id))
                .filter(conversations::user_id.eq(user_id))
                .select(ConversationCiphertextRow::as_select())
                .first::<ConversationCiphertextRow>(conn)?;
            if option_len(conversation.metadata_enc.as_ref()) != expected_metadata
                || conversation.project_id != project_id
            {
                conversation.zeroize();
                return Err(StoredConversationError::InconsistentSnapshot);
            }

            Ok(StoredConversation {
                conversation,
                project_uuid,
            })
        })
}

fn conversation_cursor(
    conn: &mut PgConnection,
    user_id: Uuid,
    after: Option<Uuid>,
) -> Result<Option<(DateTime<Utc>, i64)>, StoredConversationError> {
    let Some(after) = after else {
        return Ok(None);
    };
    conversations::table
        .filter(conversations::uuid.eq(after))
        .filter(conversations::user_id.eq(user_id))
        .select((conversations::last_activity_at, conversations::id))
        .first::<(DateTime<Utc>, i64)>(conn)
        .optional()
        .map_err(StoredConversationError::Database)
}

type ConversationMeasurement = (
    Uuid,
    i64,
    Option<i64>,
    Option<i64>,
    Option<Uuid>,
    DateTime<Utc>,
);

fn conversation_measure_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
    project_filter: ConversationProjectFilter,
    pinned: Option<bool>,
) -> Result<Vec<ConversationMeasurement>, StoredConversationError> {
    let mut query = conversations::table
        .left_join(
            conversation_projects::table.on(conversation_projects::id
                .nullable()
                .eq(conversations::project_id)
                .and(conversation_projects::user_id.eq(user_id))),
        )
        .filter(conversations::user_id.eq(user_id))
        .into_boxed();

    query = match project_filter {
        ConversationProjectFilter::Any => query,
        ConversationProjectFilter::Assigned(project_id) => {
            query.filter(conversations::project_id.eq(Some(project_id)))
        }
        ConversationProjectFilter::Unassigned => query.filter(conversations::project_id.is_null()),
    };
    if let Some(pinned) = pinned {
        query = query.filter(conversations::is_pinned.eq(pinned));
    }
    if let Some((last_activity_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                conversations::last_activity_at.lt(last_activity_at).or(
                    conversations::last_activity_at
                        .eq(last_activity_at)
                        .and(conversations::id.lt(id)),
                ),
            )
        } else {
            query.filter(
                conversations::last_activity_at.gt(last_activity_at).or(
                    conversations::last_activity_at
                        .eq(last_activity_at)
                        .and(conversations::id.gt(id)),
                ),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            conversations::last_activity_at.desc(),
            conversations::id.desc(),
        ))
    } else {
        query.order((
            conversations::last_activity_at.asc(),
            conversations::id.asc(),
        ))
    };

    query
        .select((
            conversations::uuid,
            conversations::id,
            sql::<Nullable<BigInt>>("octet_length(conversations.metadata_enc)::bigint"),
            conversations::project_id,
            conversation_projects::uuid.nullable(),
            conversations::last_activity_at,
        ))
        .limit(limit)
        .load::<ConversationMeasurement>(conn)
        .map_err(StoredConversationError::Database)
}

fn conversation_fetch_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
    project_filter: ConversationProjectFilter,
    pinned: Option<bool>,
) -> Result<Vec<(ConversationCiphertextRow, Option<Uuid>)>, StoredConversationError> {
    let mut query = conversations::table
        .left_join(
            conversation_projects::table.on(conversation_projects::id
                .nullable()
                .eq(conversations::project_id)
                .and(conversation_projects::user_id.eq(user_id))),
        )
        .filter(conversations::user_id.eq(user_id))
        .into_boxed();

    query = match project_filter {
        ConversationProjectFilter::Any => query,
        ConversationProjectFilter::Assigned(project_id) => {
            query.filter(conversations::project_id.eq(Some(project_id)))
        }
        ConversationProjectFilter::Unassigned => query.filter(conversations::project_id.is_null()),
    };
    if let Some(pinned) = pinned {
        query = query.filter(conversations::is_pinned.eq(pinned));
    }
    if let Some((last_activity_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                conversations::last_activity_at.lt(last_activity_at).or(
                    conversations::last_activity_at
                        .eq(last_activity_at)
                        .and(conversations::id.lt(id)),
                ),
            )
        } else {
            query.filter(
                conversations::last_activity_at.gt(last_activity_at).or(
                    conversations::last_activity_at
                        .eq(last_activity_at)
                        .and(conversations::id.gt(id)),
                ),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            conversations::last_activity_at.desc(),
            conversations::id.desc(),
        ))
    } else {
        query.order((
            conversations::last_activity_at.asc(),
            conversations::id.asc(),
        ))
    };

    query
        .select((
            ConversationCiphertextRow::as_select(),
            conversation_projects::uuid.nullable(),
        ))
        .limit(limit)
        .load::<(ConversationCiphertextRow, Option<Uuid>)>(conn)
        .map_err(StoredConversationError::Database)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn list_conversations(
    pool: &Pool,
    user_id: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
    project_filter: ConversationProjectFilter,
    pinned: Option<bool>,
    logical_body_limit: usize,
) -> Result<(Vec<StoredConversation>, bool), StoredConversationError> {
    validate_page_limit(limit)?;
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            // V1 resolves the cursor by owner only, independently of filters.
            let cursor = conversation_cursor(conn, user_id, after)?;
            let sentinel_limit = limit
                .checked_add(1)
                .ok_or(StoredConversationError::OutputTooLarge)?;
            let mut measured = conversation_measure_page(
                conn,
                user_id,
                sentinel_limit,
                cursor,
                order,
                project_filter,
                pinned,
            )?;
            let has_more = drop_page_sentinel(&mut measured, limit)?;

            let mut total = 0;
            let expected_lengths = measured
                .iter()
                .map(|(_, _, length, project_id, project_uuid, _)| {
                    if project_id.is_some() != project_uuid.is_some() {
                        return Err(StoredConversationError::InconsistentSnapshot);
                    }
                    account_ciphertext(&mut total, *length, logical_body_limit)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut rows = conversation_fetch_page(
                conn,
                user_id,
                limit,
                cursor,
                order,
                project_filter,
                pinned,
            )?;
            if rows.len() != measured.len() {
                for (row, _) in &mut rows {
                    row.zeroize();
                }
                return Err(StoredConversationError::InconsistentSnapshot);
            }

            for (((row, project_uuid), measured), expected_length) in
                rows.iter().zip(&measured).zip(&expected_lengths)
            {
                if row.uuid != measured.0
                    || row.id != measured.1
                    || option_len(row.metadata_enc.as_ref()) != *expected_length
                    || row.project_id != measured.3
                    || *project_uuid != measured.4
                    || row.last_activity_at != measured.5
                {
                    for (row, _) in &mut rows {
                        row.zeroize();
                    }
                    return Err(StoredConversationError::InconsistentSnapshot);
                }
            }

            Ok((
                rows.into_iter()
                    .map(|(conversation, project_uuid)| StoredConversation {
                        conversation,
                        project_uuid,
                    })
                    .collect(),
                has_more,
            ))
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_conversation(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
    metadata_enc: Option<Vec<u8>>,
    project_update: ProjectAssignmentUpdate,
    is_pinned: Option<bool>,
) -> Result<ConversationMutationMetadata, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.transaction::<_, StoredConversationError, _>(|conn| {
        let project_id = match project_update {
            ProjectAssignmentUpdate::Unchanged => None,
            ProjectAssignmentUpdate::Set(project_id) => {
                if let Some(project_id) = project_id {
                    let exists = conversation_projects::table
                        .filter(conversation_projects::id.eq(project_id))
                        .filter(conversation_projects::user_id.eq(user_id))
                        .select(conversation_projects::id)
                        .for_update()
                        .first::<i64>(conn)
                        .optional()?;
                    if exists.is_none() {
                        return Err(StoredConversationError::ConversationProjectNotFound);
                    }
                }
                Some(project_id)
            }
        };

        let metadata_enc = metadata_enc.map(Zeroizing::new);
        let changes = ConversationChanges {
            metadata_enc: metadata_enc.as_deref(),
            project_id,
            is_pinned,
        };

        let metadata = if changes.metadata_enc.is_none()
            && changes.project_id.is_none()
            && changes.is_pinned.is_none()
        {
            conversations::table
                .filter(conversations::uuid.eq(conversation_uuid))
                .filter(conversations::user_id.eq(user_id))
                .select((
                    conversations::uuid,
                    conversations::created_at,
                    conversations::updated_at,
                    conversations::project_id,
                    conversations::is_pinned,
                    conversations::last_activity_at,
                ))
                .first::<(
                    Uuid,
                    DateTime<Utc>,
                    DateTime<Utc>,
                    Option<i64>,
                    bool,
                    DateTime<Utc>,
                )>(conn)
                .optional()?
                .ok_or(StoredConversationError::ConversationNotFound)?
        } else {
            diesel::update(
                conversations::table
                    .filter(conversations::uuid.eq(conversation_uuid))
                    .filter(conversations::user_id.eq(user_id)),
            )
            .set(&changes)
            .returning((
                conversations::uuid,
                conversations::created_at,
                conversations::updated_at,
                conversations::project_id,
                conversations::is_pinned,
                conversations::last_activity_at,
            ))
            .get_result::<(
                Uuid,
                DateTime<Utc>,
                DateTime<Utc>,
                Option<i64>,
                bool,
                DateTime<Utc>,
            )>(conn)
            .optional()?
            .ok_or(StoredConversationError::ConversationNotFound)?
        };

        let project_uuid = project_uuid_for_internal_id(conn, user_id, metadata.3)?;
        Ok(ConversationMutationMetadata {
            uuid: metadata.0,
            created_at: metadata.1,
            updated_at: metadata.2,
            project_uuid,
            is_pinned: metadata.4,
            last_activity_at: metadata.5,
        })
    })
}

pub(crate) fn delete_conversation(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
) -> Result<Uuid, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    diesel::delete(
        conversations::table
            .filter(conversations::uuid.eq(conversation_uuid))
            .filter(conversations::user_id.eq(user_id)),
    )
    .returning(conversations::uuid)
    .get_result::<Uuid>(&mut conn)
    .optional()?
    .ok_or(StoredConversationError::ConversationNotFound)
}

pub(crate) fn delete_all_conversations(
    pool: &Pool,
    user_id: Uuid,
) -> Result<usize, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    diesel::delete(conversations::table.filter(conversations::user_id.eq(user_id)))
        .execute(&mut conn)
        .map_err(StoredConversationError::Database)
}

pub(crate) fn batch_update_conversation_project(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuids: &[Uuid],
    target_project_id: Option<i64>,
) -> Result<usize, StoredConversationError> {
    if conversation_uuids.is_empty()
        || conversation_uuids.len() > MAX_CONVERSATION_BATCH_SIZE
        || conversation_uuids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != conversation_uuids.len()
    {
        return Err(StoredConversationError::Validation);
    }

    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.transaction::<_, StoredConversationError, _>(|conn| {
        if let Some(target_project_id) = target_project_id {
            let target = conversation_projects::table
                .filter(conversation_projects::id.eq(target_project_id))
                .filter(conversation_projects::user_id.eq(user_id))
                .select(conversation_projects::id)
                .for_update()
                .first::<i64>(conn)
                .optional()?;
            if target.is_none() {
                return Err(StoredConversationError::ConversationProjectNotFound);
            }
        }

        let existing = conversations::table
            .filter(conversations::user_id.eq(user_id))
            .filter(conversations::uuid.eq_any(conversation_uuids))
            .select((conversations::uuid, conversations::project_id))
            .for_update()
            .load::<(Uuid, Option<i64>)>(conn)?;
        if existing.len() != conversation_uuids.len() {
            return Err(StoredConversationError::ConversationNotFound);
        }
        let source_project = existing
            .first()
            .map(|(_, project_id)| *project_id)
            .ok_or(StoredConversationError::Validation)?;
        if existing
            .iter()
            .any(|(_, project_id)| *project_id != source_project)
        {
            return Err(StoredConversationError::Validation);
        }

        let updated = diesel::update(
            conversations::table
                .filter(conversations::user_id.eq(user_id))
                .filter(conversations::uuid.eq_any(conversation_uuids)),
        )
        .set(conversations::project_id.eq(target_project_id))
        .execute(conn)?;
        if updated != conversation_uuids.len() {
            return Err(StoredConversationError::ConversationNotFound);
        }
        Ok(updated)
    })
}

fn lookup_conversation_id_on_connection(
    conn: &mut PgConnection,
    user_id: Uuid,
    conversation_uuid: Uuid,
) -> Result<i64, StoredConversationError> {
    conversations::table
        .filter(conversations::uuid.eq(conversation_uuid))
        .filter(conversations::user_id.eq(user_id))
        .select(conversations::id)
        .first::<i64>(conn)
        .optional()?
        .ok_or(StoredConversationError::ConversationNotFound)
}

/// Resolve a public conversation UUID without loading its encrypted metadata.
pub(crate) fn lookup_conversation_id(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
) -> Result<i64, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    lookup_conversation_id_on_connection(&mut conn, user_id, conversation_uuid)
}

/// Delete a previously resolved conversation without loading any ciphertext.
///
/// Keeping lookup and delete separate lets the v1-compatible batch endpoint
/// classify lookup failures as `not_found`, delete failures as
/// `delete_failed`, and a duplicate UUID as success followed by `not_found`.
pub(crate) fn delete_conversation_by_internal_id(
    pool: &Pool,
    user_id: Uuid,
    conversation_id: i64,
) -> Result<(), StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    let deleted = diesel::delete(
        conversations::table
            .filter(conversations::id.eq(conversation_id))
            .filter(conversations::user_id.eq(user_id)),
    )
    .execute(&mut conn)?;
    if deleted == 0 {
        return Err(StoredConversationError::ConversationNotFound);
    }
    Ok(())
}

/// Source rows for the Conversations items API.
///
/// Each branch independently binds both the authenticated user and the
/// owner-scoped parent ID. The response and tool-call joins repeat the same
/// bindings instead of trusting a database-controlled foreign key.
const CONVERSATION_ITEM_SOURCE: &str = r#"
WITH item_rows AS (
    SELECT
        'user'::text AS message_type,
        um.id,
        1::integer AS type_rank,
        um.uuid,
        um.content_enc,
        'completed'::text AS status,
        um.created_at,
        r.model,
        um.prompt_tokens AS token_count,
        NULL::uuid AS tool_call_id,
        NULL::text AS finish_reason,
        NULL::text AS tool_name
    FROM user_messages um
    LEFT JOIN responses r
      ON r.id = um.response_id
     AND r.user_id = $2
     AND r.conversation_id = $1
    WHERE um.conversation_id = $1 AND um.user_id = $2

    UNION ALL

    SELECT
        'assistant'::text,
        am.id,
        2::integer,
        am.uuid,
        am.content_enc,
        am.status,
        am.created_at,
        r.model,
        am.completion_tokens,
        NULL::uuid,
        am.finish_reason,
        NULL::text
    FROM assistant_messages am
    LEFT JOIN responses r
      ON r.id = am.response_id
     AND r.user_id = $2
     AND r.conversation_id = $1
    WHERE am.conversation_id = $1 AND am.user_id = $2

    UNION ALL

    SELECT
        'tool_call'::text,
        tc.id,
        3::integer,
        tc.uuid,
        tc.arguments_enc,
        'completed'::text,
        tc.created_at,
        NULL::text,
        tc.argument_tokens,
        tc.uuid,
        NULL::text,
        tc.name
    FROM tool_calls tc
    WHERE tc.conversation_id = $1 AND tc.user_id = $2

    UNION ALL

    SELECT
        'tool_output'::text,
        tto.id,
        4::integer,
        tto.uuid,
        tto.output_enc,
        'completed'::text,
        tto.created_at,
        NULL::text,
        tto.output_tokens,
        tc.uuid,
        NULL::text,
        tc.name
    FROM tool_outputs tto
    JOIN tool_calls tc
      ON tc.id = tto.tool_call_fk
     AND tc.user_id = $2
     AND tc.conversation_id = $1
    WHERE tto.conversation_id = $1 AND tto.user_id = $2

    UNION ALL

    SELECT
        'reasoning'::text,
        ri.id,
        5::integer,
        ri.uuid,
        ri.content_enc,
        ri.status,
        ri.created_at,
        NULL::text,
        ri.reasoning_tokens,
        NULL::uuid,
        NULL::text,
        NULL::text
    FROM reasoning_items ri
    WHERE ri.conversation_id = $1 AND ri.user_id = $2
)
"#;

/// Source rows for a single stored response.
///
/// User and reasoning ciphertext are intentionally projected as NULL because
/// v1 uses those rows only for token accounting and emits no corresponding
/// plaintext in the response output. All five row families remain present.
const STORED_RESPONSE_ITEM_SOURCE: &str = r#"
WITH item_rows AS (
    SELECT
        'user'::text AS message_type,
        um.id,
        1::integer AS type_rank,
        um.uuid,
        NULL::bytea AS content_enc,
        'completed'::text AS status,
        um.created_at,
        NULL::text AS model,
        um.prompt_tokens AS token_count,
        NULL::uuid AS tool_call_id,
        NULL::text AS finish_reason,
        NULL::text AS tool_name
    FROM user_messages um
    WHERE um.response_id = $1 AND um.user_id = $2 AND um.conversation_id = $3

    UNION ALL

    SELECT
        'assistant'::text,
        am.id,
        2::integer,
        am.uuid,
        am.content_enc,
        am.status,
        am.created_at,
        NULL::text,
        am.completion_tokens,
        NULL::uuid,
        am.finish_reason,
        NULL::text
    FROM assistant_messages am
    WHERE am.response_id = $1 AND am.user_id = $2 AND am.conversation_id = $3

    UNION ALL

    SELECT
        'tool_call'::text,
        tc.id,
        3::integer,
        tc.uuid,
        tc.arguments_enc,
        'completed'::text,
        tc.created_at,
        NULL::text,
        tc.argument_tokens,
        tc.uuid,
        NULL::text,
        tc.name
    FROM tool_calls tc
    WHERE tc.response_id = $1 AND tc.user_id = $2 AND tc.conversation_id = $3

    UNION ALL

    SELECT
        'tool_output'::text,
        tto.id,
        4::integer,
        tto.uuid,
        tto.output_enc,
        'completed'::text,
        tto.created_at,
        NULL::text,
        tto.output_tokens,
        tc.uuid,
        NULL::text,
        NULL::text
    FROM tool_outputs tto
    JOIN tool_calls tc
      ON tc.id = tto.tool_call_fk
     AND tc.response_id = $1
     AND tc.user_id = $2
     AND tc.conversation_id = $3
    WHERE tto.response_id = $1 AND tto.user_id = $2 AND tto.conversation_id = $3

    UNION ALL

    SELECT
        'reasoning'::text,
        ri.id,
        5::integer,
        ri.uuid,
        NULL::bytea,
        ri.status,
        ri.created_at,
        NULL::text,
        ri.reasoning_tokens,
        NULL::uuid,
        NULL::text,
        NULL::text
    FROM reasoning_items ri
    WHERE ri.response_id = $1 AND ri.user_id = $2 AND ri.conversation_id = $3
)
"#;

const ITEM_MEASUREMENT_SELECT: &str = r#"
SELECT
    items.message_type,
    items.id,
    items.type_rank,
    items.uuid,
    items.created_at,
    octet_length(items.content_enc)::bigint AS content_length,
    octet_length(items.status)::bigint AS status_length,
    octet_length(items.model)::bigint AS model_length,
    octet_length(items.finish_reason)::bigint AS finish_reason_length,
    octet_length(items.tool_name)::bigint AS tool_name_length
FROM item_rows items
"#;

const ITEM_FETCH_SELECT: &str = r#"
SELECT
    items.message_type,
    items.id,
    items.type_rank,
    items.uuid,
    items.content_enc,
    items.status,
    items.created_at,
    items.model,
    items.token_count,
    items.tool_call_id,
    items.finish_reason,
    items.tool_name
FROM item_rows items
"#;

fn conversation_item_page_sql(after: bool, order: &str, measurement: bool) -> String {
    let select = if measurement {
        ITEM_MEASUREMENT_SELECT
    } else {
        ITEM_FETCH_SELECT
    };
    let ordering = if order == "desc" {
        "ORDER BY items.created_at DESC, items.id DESC, items.type_rank DESC"
    } else {
        "ORDER BY items.created_at ASC, items.id ASC, items.type_rank ASC"
    };
    if after {
        let predicate = if order == "desc" {
            "(items.created_at < cursor_item.created_at) OR \
             (items.created_at = cursor_item.created_at AND (\
                 items.id < cursor_item.id OR (\
                     items.id = cursor_item.id AND items.type_rank < cursor_item.type_rank\
                 )\
             ))"
        } else {
            "(items.created_at > cursor_item.created_at) OR \
             (items.created_at = cursor_item.created_at AND (\
                 items.id > cursor_item.id OR (\
                     items.id = cursor_item.id AND items.type_rank > cursor_item.type_rank\
                 )\
             ))"
        };
        format!(
            "{CONVERSATION_ITEM_SOURCE}, cursor_item AS (\
                SELECT created_at, id, type_rank FROM item_rows WHERE uuid = $3 \
                ORDER BY created_at ASC, id ASC, type_rank ASC LIMIT 1\
             ) {select}, cursor_item WHERE {predicate} {ordering} LIMIT $4"
        )
    } else {
        format!("{CONVERSATION_ITEM_SOURCE} {select} {ordering} LIMIT $3")
    }
}

fn conversation_item_get_sql(measurement: bool) -> String {
    let select = if measurement {
        ITEM_MEASUREMENT_SELECT
    } else {
        ITEM_FETCH_SELECT
    };
    format!(
        "{CONVERSATION_ITEM_SOURCE} {select} \
         WHERE items.uuid = $3 \
         ORDER BY items.created_at ASC, items.id ASC, items.type_rank ASC LIMIT 1"
    )
}

fn stored_response_items_sql(measurement: bool) -> String {
    let select = if measurement {
        ITEM_MEASUREMENT_SELECT
    } else {
        ITEM_FETCH_SELECT
    };
    format!(
        "{STORED_RESPONSE_ITEM_SOURCE} {select} \
         ORDER BY items.created_at ASC, items.id ASC, items.type_rank ASC LIMIT $4"
    )
}

fn load_conversation_item_measurements(
    conn: &mut PgConnection,
    conversation_id: i64,
    user_id: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
) -> Result<Vec<ItemMeasurement>, StoredConversationError> {
    let query = conversation_item_page_sql(after.is_some(), order, true);
    match after {
        Some(after) => sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .bind::<diesel::sql_types::Uuid, _>(user_id)
            .bind::<diesel::sql_types::Uuid, _>(after)
            .bind::<BigInt, _>(limit)
            .load::<ItemMeasurement>(conn),
        None => sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .bind::<diesel::sql_types::Uuid, _>(user_id)
            .bind::<BigInt, _>(limit)
            .load::<ItemMeasurement>(conn),
    }
    .map_err(StoredConversationError::Database)
}

fn load_conversation_items(
    conn: &mut PgConnection,
    conversation_id: i64,
    user_id: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
) -> Result<Vec<StoredConversationItem>, StoredConversationError> {
    let query = conversation_item_page_sql(after.is_some(), order, false);
    match after {
        Some(after) => sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .bind::<diesel::sql_types::Uuid, _>(user_id)
            .bind::<diesel::sql_types::Uuid, _>(after)
            .bind::<BigInt, _>(limit)
            .load::<StoredConversationItem>(conn),
        None => sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .bind::<diesel::sql_types::Uuid, _>(user_id)
            .bind::<BigInt, _>(limit)
            .load::<StoredConversationItem>(conn),
    }
    .map_err(StoredConversationError::Database)
}

pub(crate) fn list_conversation_items(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
    logical_body_limit: usize,
) -> Result<(Vec<StoredConversationItem>, bool), StoredConversationError> {
    validate_page_limit(limit)?;
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            let conversation_id =
                lookup_conversation_id_on_connection(conn, user_id, conversation_uuid)?;
            let sentinel_limit = limit
                .checked_add(1)
                .ok_or(StoredConversationError::OutputTooLarge)?;
            let mut measured = load_conversation_item_measurements(
                conn,
                conversation_id,
                user_id,
                sentinel_limit,
                after,
                order,
            )?;
            let has_more = drop_page_sentinel(&mut measured, limit)?;
            let expected = validate_item_measurements(&measured, logical_body_limit, 0)?;
            let mut rows =
                load_conversation_items(conn, conversation_id, user_id, limit, after, order)?;
            validate_fetched_items(&mut rows, &measured, &expected)?;
            Ok((rows, has_more))
        })
}

pub(crate) fn get_conversation_item(
    pool: &Pool,
    user_id: Uuid,
    conversation_uuid: Uuid,
    item_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<StoredConversationItem, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            let conversation_id =
                lookup_conversation_id_on_connection(conn, user_id, conversation_uuid)?;
            let measurement = sql_query(conversation_item_get_sql(true))
                .bind::<BigInt, _>(conversation_id)
                .bind::<diesel::sql_types::Uuid, _>(user_id)
                .bind::<diesel::sql_types::Uuid, _>(item_uuid)
                .get_result::<ItemMeasurement>(conn)
                .optional()?
                .ok_or(StoredConversationError::ConversationItemNotFound)?;
            let expected = validate_item_measurements(
                std::slice::from_ref(&measurement),
                logical_body_limit,
                0,
            )?;
            let row = sql_query(conversation_item_get_sql(false))
                .bind::<BigInt, _>(conversation_id)
                .bind::<diesel::sql_types::Uuid, _>(user_id)
                .bind::<diesel::sql_types::Uuid, _>(item_uuid)
                .get_result::<StoredConversationItem>(conn)
                .optional()?
                .ok_or(StoredConversationError::InconsistentSnapshot)?;
            let mut rows = vec![row];
            validate_fetched_items(&mut rows, std::slice::from_ref(&measurement), &expected)?;
            rows.pop()
                .ok_or(StoredConversationError::InconsistentSnapshot)
        })
}

fn load_stored_response_item_measurements(
    conn: &mut PgConnection,
    response_id: i64,
    user_id: Uuid,
    conversation_id: i64,
    limit: i64,
) -> Result<Vec<ItemMeasurement>, StoredConversationError> {
    sql_query(stored_response_items_sql(true))
        .bind::<BigInt, _>(response_id)
        .bind::<diesel::sql_types::Uuid, _>(user_id)
        .bind::<BigInt, _>(conversation_id)
        .bind::<BigInt, _>(limit)
        .load::<ItemMeasurement>(conn)
        .map_err(StoredConversationError::Database)
}

fn load_stored_response_items(
    conn: &mut PgConnection,
    response_id: i64,
    user_id: Uuid,
    conversation_id: i64,
    limit: i64,
) -> Result<Vec<StoredConversationItem>, StoredConversationError> {
    sql_query(stored_response_items_sql(false))
        .bind::<BigInt, _>(response_id)
        .bind::<diesel::sql_types::Uuid, _>(user_id)
        .bind::<BigInt, _>(conversation_id)
        .bind::<BigInt, _>(limit)
        .load::<StoredConversationItem>(conn)
        .map_err(StoredConversationError::Database)
}

pub(crate) fn get_stored_response(
    pool: &Pool,
    user_id: Uuid,
    response_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<StoredResponse, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            let (response_id, conversation_id, model_length) = responses::table
                .filter(responses::uuid.eq(response_uuid))
                .filter(responses::user_id.eq(user_id))
                .select((
                    responses::id,
                    responses::conversation_id,
                    sql::<BigInt>("octet_length(model)::bigint"),
                ))
                .first::<(i64, i64, i64)>(conn)
                .optional()?
                .ok_or(StoredConversationError::ResponseNotFound)?;

            // Do not trust a tampered response-to-conversation relationship.
            let parent_exists = conversations::table
                .filter(conversations::id.eq(conversation_id))
                .filter(conversations::user_id.eq(user_id))
                .select(conversations::id)
                .first::<i64>(conn)
                .optional()?;
            if parent_exists.is_none() {
                return Err(StoredConversationError::InconsistentSnapshot);
            }

            let model_length = checked_length(model_length)?;
            if model_length > logical_body_limit {
                return Err(StoredConversationError::OutputTooLarge);
            }

            let sentinel_limit = MAX_STORED_RESPONSE_ITEMS
                .checked_add(1)
                .ok_or(StoredConversationError::OutputTooLarge)?;
            let measured = load_stored_response_item_measurements(
                conn,
                response_id,
                user_id,
                conversation_id,
                sentinel_limit,
            )?;
            if measured.len()
                > usize::try_from(MAX_STORED_RESPONSE_ITEMS)
                    .map_err(|_| StoredConversationError::OutputTooLarge)?
            {
                return Err(StoredConversationError::OutputTooLarge);
            }
            let expected = validate_item_measurements(&measured, logical_body_limit, model_length)?;

            let response = responses::table
                .filter(responses::id.eq(response_id))
                .filter(responses::user_id.eq(user_id))
                .select(StoredResponseMetadata::as_select())
                .first::<StoredResponseMetadata>(conn)?;
            if response.conversation_id != conversation_id || response.model.len() != model_length {
                return Err(StoredConversationError::InconsistentSnapshot);
            }

            let fetch_limit = i64::try_from(measured.len())
                .map_err(|_| StoredConversationError::OutputTooLarge)?;
            let mut items = load_stored_response_items(
                conn,
                response_id,
                user_id,
                conversation_id,
                fetch_limit,
            )?;
            validate_fetched_items(&mut items, &measured, &expected)?;
            Ok(StoredResponse { response, items })
        })
}

/// Bounded metadata read for the cancel response.
///
/// The application serializes the exact cancelled-response JSON from this
/// projection before calling [`transition_stored_response_to_cancelled`]. That
/// keeps a database-controlled model string from causing a post-mutation
/// serialization/capacity failure.
pub(crate) fn get_cancelable_response_metadata(
    pool: &Pool,
    user_id: Uuid,
    response_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<ResponseMutationMetadata, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredConversationError, _>(|conn| {
            let (response_id, conversation_id, status, model_length) = responses::table
                .filter(responses::uuid.eq(response_uuid))
                .filter(responses::user_id.eq(user_id))
                .select((
                    responses::id,
                    responses::conversation_id,
                    responses::status,
                    sql::<BigInt>("octet_length(model)::bigint"),
                ))
                .first::<(i64, i64, ResponseStatus, i64)>(conn)
                .optional()?
                .ok_or(StoredConversationError::ResponseNotFound)?;

            if !matches!(status, ResponseStatus::Queued | ResponseStatus::InProgress) {
                return Err(StoredConversationError::Validation);
            }
            let model_length = checked_length(model_length)?;
            if model_length > logical_body_limit {
                return Err(StoredConversationError::OutputTooLarge);
            }
            let parent_exists = conversations::table
                .filter(conversations::id.eq(conversation_id))
                .filter(conversations::user_id.eq(user_id))
                .select(conversations::id)
                .first::<i64>(conn)
                .optional()?;
            if parent_exists.is_none() {
                return Err(StoredConversationError::InconsistentSnapshot);
            }

            let result = responses::table
                .filter(responses::id.eq(response_id))
                .filter(responses::user_id.eq(user_id))
                .select(ResponseMutationMetadata::as_select())
                .first::<ResponseMutationMetadata>(conn)?;
            if result.status != status || result.model.len() != model_length {
                return Err(StoredConversationError::InconsistentSnapshot);
            }
            Ok(result)
        })
}

/// Atomically transition a response after its exact success body was
/// preflighted by the application.
///
/// A terminal-state race remains the existing 400 validation outcome. A
/// deletion race, missing UUID, or foreign UUID remains the existing 404.
pub(crate) fn transition_stored_response_to_cancelled(
    pool: &Pool,
    user_id: Uuid,
    response_uuid: Uuid,
) -> Result<Uuid, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    let updated = diesel::update(
        responses::table
            .filter(responses::uuid.eq(response_uuid))
            .filter(responses::user_id.eq(user_id))
            .filter(
                responses::status
                    .eq(ResponseStatus::Queued)
                    .or(responses::status.eq(ResponseStatus::InProgress)),
            ),
    )
    .set((
        responses::status.eq(ResponseStatus::Cancelled),
        responses::completed_at.eq(diesel::dsl::now),
        responses::updated_at.eq(diesel::dsl::now),
    ))
    .returning(responses::uuid)
    .get_result::<Uuid>(&mut conn)
    .optional()?;
    if let Some(uuid) = updated {
        return Ok(uuid);
    }

    let exists = responses::table
        .filter(responses::uuid.eq(response_uuid))
        .filter(responses::user_id.eq(user_id))
        .select(responses::uuid)
        .first::<Uuid>(&mut conn)
        .optional()?;
    if exists.is_some() {
        Err(StoredConversationError::Validation)
    } else {
        Err(StoredConversationError::ResponseNotFound)
    }
}

pub(crate) fn delete_stored_response(
    pool: &Pool,
    user_id: Uuid,
    response_uuid: Uuid,
) -> Result<Uuid, StoredConversationError> {
    let mut conn = pool
        .get()
        .map_err(|_| StoredConversationError::Connection)?;
    diesel::delete(
        responses::table
            .filter(responses::uuid.eq(response_uuid))
            .filter(responses::user_id.eq(user_id)),
    )
    .returning(responses::uuid)
    .get_result::<Uuid>(&mut conn)
    .optional()?
    .ok_or(StoredConversationError::ResponseNotFound)
}

#[cfg(test)]
mod tests {
    use super::{
        account_ciphertext, batch_update_conversation_project, conversation_item_page_sql,
        delete_all_conversations, delete_conversation_by_internal_id, delete_stored_response,
        drop_page_sentinel, get_cancelable_response_metadata, get_conversation,
        get_conversation_item, get_stored_response, list_conversation_items, list_conversations,
        lookup_conversation_id, stored_response_items_sql, transition_stored_response_to_cancelled,
        update_conversation, validate_page_limit, ItemMeasurement, Pool, ProjectAssignmentUpdate,
        StoredConversationError, StoredConversationItem, AES_GCM_STORAGE_OVERHEAD_BYTES,
    };
    use crate::models::{
        responses::{
            ConversationProjectFilter, NewAssistantMessage, NewConversation,
            NewConversationProject, NewReasoningItem, NewResponse, NewToolCall, NewToolOutput,
            NewUserMessage, ResponseStatus,
        },
        schema::{conversations, org_projects, users},
        users::NewUser,
    };
    use chrono::{Duration, Utc};
    use diesel::prelude::*;
    use diesel::r2d2::ConnectionManager;
    use std::collections::HashSet;
    use uuid::Uuid;

    const TEST_MODEL: &str = "stored-conversation-v2-test-model";

    struct TestUsers {
        pool: Pool,
        user_ids: Vec<Uuid>,
    }

    impl Drop for TestUsers {
        fn drop(&mut self) {
            let Ok(mut conn) = self.pool.get() else {
                return;
            };
            let _ = diesel::delete(users::table.filter(users::uuid.eq_any(&self.user_ids)))
                .execute(&mut conn);
        }
    }

    struct ItemStackFixture {
        conversation_uuid: Uuid,
        response_uuid: Uuid,
        assistant_uuid: Uuid,
        foreign_message_uuid: Uuid,
    }

    fn disposable_pool() -> Option<Pool> {
        let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
            eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
            return None;
        };
        let manager = ConnectionManager::<PgConnection>::new(database_url);
        Some(
            diesel::r2d2::Pool::builder()
                .max_size(6)
                .build(manager)
                .expect("connect to disposable migrated PostgreSQL"),
        )
    }

    fn first_active_org_project_id(conn: &mut PgConnection) -> i32 {
        org_projects::table
            .filter(org_projects::status.eq("active"))
            .order(org_projects::id.asc())
            .select(org_projects::id)
            .first(conn)
            .expect("test database should contain the migrated active project")
    }

    fn insert_test_user(conn: &mut PgConnection, org_project_id: i32, label: &str) -> Uuid {
        let marker = Uuid::new_v4();
        NewUser::new(
            Some(format!("stored-v2-{label}-{marker}@example.com")),
            None,
            org_project_id,
        )
        .insert(conn)
        .expect("test user should insert")
        .uuid
    }

    fn storage_ciphertext(payload_bytes: usize, marker: u8) -> Vec<u8> {
        vec![marker; AES_GCM_STORAGE_OVERHEAD_BYTES + payload_bytes]
    }

    fn insert_conversation(
        conn: &mut PgConnection,
        user_id: Uuid,
        project_id: Option<i64>,
        payload_bytes: usize,
        marker: u8,
    ) -> crate::models::responses::Conversation {
        NewConversation {
            uuid: Uuid::new_v4(),
            user_id,
            project_id,
            is_pinned: false,
            metadata_enc: Some(storage_ciphertext(payload_bytes, marker)),
        }
        .insert(conn)
        .expect("test conversation should insert")
    }

    fn insert_item_stack(
        conn: &mut PgConnection,
        owner_id: Uuid,
        foreign_user_id: Uuid,
        status: ResponseStatus,
    ) -> ItemStackFixture {
        let conversation = insert_conversation(conn, owner_id, None, 4, 10);
        let response = NewResponse {
            uuid: Uuid::new_v4(),
            user_id: owner_id,
            conversation_id: conversation.id,
            status,
            model: TEST_MODEL.to_string(),
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            tool_choice: None,
            parallel_tool_calls: false,
            store: true,
            metadata_enc: None,
        }
        .insert(conn)
        .expect("test response should insert");

        let user_message_uuid = Uuid::new_v4();
        NewUserMessage {
            uuid: user_message_uuid,
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: owner_id,
            content_enc: storage_ciphertext(5, 11),
            prompt_tokens: 3,
        }
        .insert(conn)
        .expect("test user message should insert");

        let base = Utc::now();
        let assistant_uuid = Uuid::new_v4();
        let assistant = NewAssistantMessage {
            uuid: assistant_uuid,
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: owner_id,
            content_enc: Some(storage_ciphertext(6, 12)),
            completion_tokens: 4,
            status: "completed".to_string(),
            finish_reason: Some("stop".to_string()),
            created_at: base + Duration::milliseconds(1),
        }
        .insert(conn)
        .expect("test assistant message should insert");

        let tool_call = NewToolCall {
            uuid: Uuid::new_v4(),
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: owner_id,
            name: "lookup".to_string(),
            arguments_enc: Some(storage_ciphertext(7, 13)),
            argument_tokens: 5,
            status: "completed".to_string(),
            created_at: base + Duration::milliseconds(2),
        }
        .insert(conn)
        .expect("test tool call should insert");

        NewToolOutput {
            uuid: Uuid::new_v4(),
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: owner_id,
            tool_call_fk: tool_call.id,
            output_enc: storage_ciphertext(8, 14),
            output_tokens: 6,
            status: "completed".to_string(),
            error: None,
            created_at: base + Duration::milliseconds(3),
        }
        .insert(conn)
        .expect("test tool output should insert");

        NewReasoningItem {
            uuid: Uuid::new_v4(),
            conversation_id: conversation.id,
            response_id: Some(response.id),
            assistant_message_id: Some(assistant.id),
            user_id: owner_id,
            content_enc: Some(storage_ciphertext(9, 15)),
            summary_enc: None,
            reasoning_tokens: 7,
            status: "completed".to_string(),
            created_at: base + Duration::milliseconds(4),
        }
        .insert(conn)
        .expect("test reasoning item should insert");

        // This intentionally inconsistent child proves every item branch binds
        // the authenticated owner in addition to trusting the parent IDs.
        let foreign_message_uuid = Uuid::new_v4();
        NewUserMessage {
            uuid: foreign_message_uuid,
            conversation_id: conversation.id,
            response_id: Some(response.id),
            user_id: foreign_user_id,
            content_enc: storage_ciphertext(10, 16),
            prompt_tokens: 99,
        }
        .insert(conn)
        .expect("cross-owner tamper fixture should be representable");

        ItemStackFixture {
            conversation_uuid: conversation.uuid,
            response_uuid: response.uuid,
            assistant_uuid,
            foreign_message_uuid,
        }
    }

    #[test]
    fn ciphertext_accounting_removes_exact_storage_overhead() {
        let mut total = 0;
        let expected = account_ciphertext(
            &mut total,
            Some((AES_GCM_STORAGE_OVERHEAD_BYTES + 17) as i64),
            17,
        )
        .expect("exact limit");
        assert_eq!(expected, Some(AES_GCM_STORAGE_OVERHEAD_BYTES + 17));
        assert_eq!(total, 17);
    }

    #[test]
    fn ciphertext_accounting_rejects_short_and_oversized_values() {
        let mut total = 0;
        assert!(matches!(
            account_ciphertext(
                &mut total,
                Some((AES_GCM_STORAGE_OVERHEAD_BYTES - 1) as i64),
                usize::MAX,
            ),
            Err(StoredConversationError::InconsistentSnapshot)
        ));

        let mut total = 0;
        assert!(matches!(
            account_ciphertext(
                &mut total,
                Some((AES_GCM_STORAGE_OVERHEAD_BYTES + 2) as i64),
                1,
            ),
            Err(StoredConversationError::OutputTooLarge)
        ));
    }

    #[test]
    fn list_sentinel_is_removed_before_returned_page_accounting() {
        let mut rows = vec![1, 2, 3];
        assert!(drop_page_sentinel(&mut rows, 2).expect("valid page"));
        assert_eq!(rows, vec![1, 2]);
    }

    #[test]
    fn page_bounds_match_the_existing_normalized_contract() {
        assert!(validate_page_limit(1).is_ok());
        assert!(validate_page_limit(100).is_ok());
        assert!(matches!(
            validate_page_limit(0),
            Err(StoredConversationError::Validation)
        ));
        assert!(matches!(
            validate_page_limit(101),
            Err(StoredConversationError::Validation)
        ));
    }

    #[test]
    fn item_queries_scope_every_child_family_to_owner_and_parent() {
        let conversation_sql = conversation_item_page_sql(true, "desc", true);
        for table_alias in ["um", "am", "tc", "tto", "ri"] {
            assert!(conversation_sql.contains(&format!(
                "{table_alias}.conversation_id = $1 AND {table_alias}.user_id = $2"
            )));
        }
        assert!(conversation_sql.contains("cursor_item"));

        let response_sql = stored_response_items_sql(true);
        for table_alias in ["um", "am", "tc", "tto", "ri"] {
            assert!(response_sql.contains(&format!(
                "{table_alias}.response_id = $1 AND {table_alias}.user_id = $2 AND {table_alias}.conversation_id = $3"
            )));
        }
    }

    #[test]
    fn only_exact_desc_selects_descending_item_order() {
        assert!(conversation_item_page_sql(false, "desc", true)
            .contains("ORDER BY items.created_at DESC, items.id DESC, items.type_rank DESC"));
        assert!(conversation_item_page_sql(false, "DESC", true)
            .contains("ORDER BY items.created_at ASC, items.id ASC, items.type_rank ASC"));
    }

    #[test]
    fn item_cursor_uses_the_same_total_order_as_the_page() {
        let sql = conversation_item_page_sql(true, "desc", true);
        assert!(sql.contains("SELECT created_at, id, type_rank FROM item_rows"));
        assert!(
            sql.contains("items.id = cursor_item.id AND items.type_rank < cursor_item.type_rank")
        );
    }

    #[test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    fn database_conversation_reads_and_mutations_preserve_owner_and_batch_contracts() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut conn = pool.get().expect("test database connection");
        let org_project_id = first_active_org_project_id(&mut conn);
        let owner_id = insert_test_user(&mut conn, org_project_id, "conversation-owner");
        let foreign_id = insert_test_user(&mut conn, org_project_id, "conversation-foreign");
        let _cleanup = TestUsers {
            pool: pool.clone(),
            user_ids: vec![owner_id, foreign_id],
        };

        let source_project = NewConversationProject {
            uuid: Uuid::new_v4(),
            user_id: owner_id,
            name_enc: storage_ciphertext(2, 21),
        }
        .insert(&mut conn)
        .expect("source project should insert");
        let target_project = NewConversationProject {
            uuid: Uuid::new_v4(),
            user_id: owner_id,
            name_enc: storage_ciphertext(2, 22),
        }
        .insert(&mut conn)
        .expect("target project should insert");

        let first = insert_conversation(&mut conn, owner_id, Some(source_project.id), 3, 31);
        let second = insert_conversation(&mut conn, owner_id, Some(source_project.id), 3, 32);
        let oversized_sentinel = insert_conversation(&mut conn, owner_id, None, 4_096, 33);
        let foreign = insert_conversation(&mut conn, foreign_id, None, 3, 34);

        let base = Utc::now();
        for (conversation_id, last_activity_at) in [
            (first.id, base + Duration::seconds(3)),
            (second.id, base + Duration::seconds(2)),
            (oversized_sentinel.id, base + Duration::seconds(1)),
        ] {
            diesel::update(conversations::table.filter(conversations::id.eq(conversation_id)))
                .set(conversations::last_activity_at.eq(last_activity_at))
                .execute(&mut conn)
                .expect("test ordering timestamp should update");
        }

        // The third row is only the limit+1 sentinel. Its oversized metadata
        // must not be loaded or charged to the returned page.
        let (page, has_more) = list_conversations(
            &pool,
            owner_id,
            2,
            None,
            "desc",
            ConversationProjectFilter::Any,
            None,
            6,
        )
        .expect("sentinel ciphertext must be excluded from page accounting");
        assert!(has_more);
        assert_eq!(
            page.iter()
                .map(|row| row.conversation.uuid)
                .collect::<Vec<_>>(),
            vec![first.uuid, second.uuid]
        );

        let (baseline, baseline_more) = list_conversations(
            &pool,
            owner_id,
            100,
            None,
            "desc",
            ConversationProjectFilter::Any,
            None,
            16_384,
        )
        .expect("owner conversation list should load");
        let (missing_cursor, missing_cursor_more) = list_conversations(
            &pool,
            owner_id,
            100,
            Some(Uuid::new_v4()),
            "desc",
            ConversationProjectFilter::Any,
            None,
            16_384,
        )
        .expect("missing conversation cursor should be ignored");
        assert_eq!(baseline_more, missing_cursor_more);
        assert_eq!(
            baseline
                .iter()
                .map(|row| row.conversation.uuid)
                .collect::<Vec<_>>(),
            missing_cursor
                .iter()
                .map(|row| row.conversation.uuid)
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            get_conversation(&pool, owner_id, foreign.uuid, 1_024),
            Err(StoredConversationError::ConversationNotFound)
        ));

        let updated = update_conversation(
            &pool,
            owner_id,
            first.uuid,
            Some(storage_ciphertext(5, 35)),
            ProjectAssignmentUpdate::Set(Some(target_project.id)),
            Some(true),
        )
        .expect("owner-scoped update should succeed");
        assert_eq!(updated.project_uuid, Some(target_project.uuid));
        assert!(updated.is_pinned);
        let reloaded = get_conversation(&pool, owner_id, first.uuid, 1_024)
            .expect("updated conversation should reload");
        assert_eq!(reloaded.project_uuid, Some(target_project.uuid));
        assert_eq!(
            reloaded.conversation.metadata_enc.as_deref(),
            Some(storage_ciphertext(5, 35).as_slice())
        );

        let fourth = insert_conversation(&mut conn, owner_id, Some(source_project.id), 3, 36);
        assert!(matches!(
            batch_update_conversation_project(
                &pool,
                owner_id,
                &[second.uuid, foreign.uuid],
                Some(target_project.id),
            ),
            Err(StoredConversationError::ConversationNotFound)
        ));
        assert_eq!(
            get_conversation(&pool, owner_id, second.uuid, 1_024)
                .expect("failed batch must be atomic")
                .project_uuid,
            Some(source_project.uuid)
        );
        assert!(matches!(
            batch_update_conversation_project(
                &pool,
                owner_id,
                &[second.uuid, second.uuid],
                Some(target_project.id),
            ),
            Err(StoredConversationError::Validation)
        ));
        assert_eq!(
            batch_update_conversation_project(
                &pool,
                owner_id,
                &[second.uuid, fourth.uuid],
                Some(target_project.id),
            )
            .expect("same-source owner batch should update"),
            2
        );

        let delete_id = lookup_conversation_id(&pool, owner_id, oversized_sentinel.uuid)
            .expect("batch-delete lookup should find owner conversation");
        delete_conversation_by_internal_id(&pool, owner_id, delete_id)
            .expect("resolved owner conversation should delete");
        assert!(matches!(
            lookup_conversation_id(&pool, owner_id, oversized_sentinel.uuid),
            Err(StoredConversationError::ConversationNotFound)
        ));
        assert_eq!(
            delete_all_conversations(&pool, owner_id).expect("delete-all should be owner scoped"),
            3
        );
        assert_eq!(
            get_conversation(&pool, foreign_id, foreign.uuid, 1_024)
                .expect("owner delete-all must preserve foreign conversation")
                .conversation
                .uuid,
            foreign.uuid
        );
    }

    #[test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    fn database_conversation_items_cover_all_families_cursors_bounds_and_owner_scope() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut conn = pool.get().expect("test database connection");
        let org_project_id = first_active_org_project_id(&mut conn);
        let owner_id = insert_test_user(&mut conn, org_project_id, "items-owner");
        let foreign_id = insert_test_user(&mut conn, org_project_id, "items-foreign");
        let _cleanup = TestUsers {
            pool: pool.clone(),
            user_ids: vec![owner_id, foreign_id],
        };
        let fixture = insert_item_stack(&mut conn, owner_id, foreign_id, ResponseStatus::Completed);

        let (items, has_more) = list_conversation_items(
            &pool,
            owner_id,
            fixture.conversation_uuid,
            100,
            None,
            "asc",
            16_384,
        )
        .expect("all owner item families should load");
        assert!(!has_more);
        assert_eq!(items.len(), 5, "cross-owner child must be excluded");
        assert_eq!(
            items
                .iter()
                .map(|item| item.message_type.as_str())
                .collect::<HashSet<_>>(),
            HashSet::from(["user", "assistant", "tool_call", "tool_output", "reasoning"])
        );

        let (page, has_more) = list_conversation_items(
            &pool,
            owner_id,
            fixture.conversation_uuid,
            2,
            None,
            "asc",
            16_384,
        )
        .expect("bounded item page should load");
        assert_eq!(page.len(), 2);
        assert!(has_more);

        let cursor = page.last().expect("first page must have a cursor").uuid;
        let (after_page, after_has_more) = list_conversation_items(
            &pool,
            owner_id,
            fixture.conversation_uuid,
            100,
            Some(cursor),
            "asc",
            16_384,
        )
        .expect("existing item cursor should return the remaining ordered items");
        assert!(!after_has_more);
        assert_eq!(
            after_page.iter().map(|item| item.uuid).collect::<Vec<_>>(),
            items
                .iter()
                .skip(page.len())
                .map(|item| item.uuid)
                .collect::<Vec<_>>()
        );

        let (after_missing, has_more) = list_conversation_items(
            &pool,
            owner_id,
            fixture.conversation_uuid,
            100,
            Some(Uuid::new_v4()),
            "asc",
            16_384,
        )
        .expect("missing item cursor should produce an empty page");
        assert!(after_missing.is_empty());
        assert!(!has_more);

        assert_eq!(
            get_conversation_item(
                &pool,
                owner_id,
                fixture.conversation_uuid,
                fixture.assistant_uuid,
                4_096,
            )
            .expect("direct owner-scoped item lookup should succeed")
            .message_type,
            "assistant"
        );
        assert!(matches!(
            get_conversation_item(
                &pool,
                owner_id,
                fixture.conversation_uuid,
                fixture.foreign_message_uuid,
                4_096,
            ),
            Err(StoredConversationError::ConversationItemNotFound)
        ));
        assert!(matches!(
            list_conversation_items(
                &pool,
                foreign_id,
                fixture.conversation_uuid,
                100,
                None,
                "asc",
                16_384,
            ),
            Err(StoredConversationError::ConversationNotFound)
        ));
        assert!(matches!(
            list_conversation_items(
                &pool,
                owner_id,
                fixture.conversation_uuid,
                1,
                None,
                "asc",
                255,
            ),
            Err(StoredConversationError::OutputTooLarge)
        ));
    }

    #[test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    fn database_stored_response_retrieval_cancel_preflight_transition_and_delete_are_scoped() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut conn = pool.get().expect("test database connection");
        let org_project_id = first_active_org_project_id(&mut conn);
        let owner_id = insert_test_user(&mut conn, org_project_id, "response-owner");
        let foreign_id = insert_test_user(&mut conn, org_project_id, "response-foreign");
        let _cleanup = TestUsers {
            pool: pool.clone(),
            user_ids: vec![owner_id, foreign_id],
        };
        let fixture = insert_item_stack(&mut conn, owner_id, foreign_id, ResponseStatus::Queued);

        let stored = get_stored_response(&pool, owner_id, fixture.response_uuid, 16_384)
            .expect("owner response should load");
        assert_eq!(stored.items.len(), 5, "cross-owner child must be excluded");
        assert_eq!(stored.response.status, ResponseStatus::Queued);
        for item in &stored.items {
            if matches!(item.message_type.as_str(), "user" | "reasoning") {
                assert!(
                    item.content_enc.is_none(),
                    "user and reasoning ciphertext must not enter the stored-response projection"
                );
            }
        }
        assert!(matches!(
            get_stored_response(&pool, foreign_id, fixture.response_uuid, 16_384),
            Err(StoredConversationError::ResponseNotFound)
        ));
        assert!(matches!(
            get_stored_response(
                &pool,
                owner_id,
                fixture.response_uuid,
                TEST_MODEL.len() + 255,
            ),
            Err(StoredConversationError::OutputTooLarge)
        ));

        assert!(matches!(
            get_cancelable_response_metadata(
                &pool,
                owner_id,
                fixture.response_uuid,
                TEST_MODEL.len() - 1,
            ),
            Err(StoredConversationError::OutputTooLarge)
        ));
        assert_eq!(
            get_cancelable_response_metadata(
                &pool,
                owner_id,
                fixture.response_uuid,
                TEST_MODEL.len(),
            )
            .expect("failed preflight must not mutate status")
            .status,
            ResponseStatus::Queued
        );
        assert!(matches!(
            transition_stored_response_to_cancelled(&pool, foreign_id, fixture.response_uuid),
            Err(StoredConversationError::ResponseNotFound)
        ));
        assert_eq!(
            transition_stored_response_to_cancelled(&pool, owner_id, fixture.response_uuid)
                .expect("owner queued response should cancel"),
            fixture.response_uuid
        );
        assert!(matches!(
            transition_stored_response_to_cancelled(&pool, owner_id, fixture.response_uuid),
            Err(StoredConversationError::Validation)
        ));
        assert_eq!(
            get_stored_response(&pool, owner_id, fixture.response_uuid, 16_384)
                .expect("cancelled response should remain retrievable")
                .response
                .status,
            ResponseStatus::Cancelled
        );
        assert!(matches!(
            delete_stored_response(&pool, foreign_id, fixture.response_uuid),
            Err(StoredConversationError::ResponseNotFound)
        ));
        assert_eq!(
            delete_stored_response(&pool, owner_id, fixture.response_uuid)
                .expect("owner response should delete"),
            fixture.response_uuid
        );
        assert!(matches!(
            get_stored_response(&pool, owner_id, fixture.response_uuid, 16_384),
            Err(StoredConversationError::ResponseNotFound)
        ));
    }

    #[test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    fn item_cursor_measure_and_fetch_sql_execute_without_ambiguous_columns() {
        let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
            eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
            return;
        };
        let mut conn = diesel::PgConnection::establish(&database_url)
            .expect("connect to disposable migrated PostgreSQL");

        let measured = diesel::sql_query(conversation_item_page_sql(true, "desc", true))
            .bind::<diesel::sql_types::BigInt, _>(i64::MIN)
            .bind::<diesel::sql_types::Uuid, _>(Uuid::nil())
            .bind::<diesel::sql_types::Uuid, _>(Uuid::nil())
            .bind::<diesel::sql_types::BigInt, _>(1_i64)
            .load::<ItemMeasurement>(&mut conn)
            .expect("cursor measurement SQL must be valid");
        assert!(measured.is_empty());

        let fetched = diesel::sql_query(conversation_item_page_sql(true, "desc", false))
            .bind::<diesel::sql_types::BigInt, _>(i64::MIN)
            .bind::<diesel::sql_types::Uuid, _>(Uuid::nil())
            .bind::<diesel::sql_types::Uuid, _>(Uuid::nil())
            .bind::<diesel::sql_types::BigInt, _>(1_i64)
            .load::<StoredConversationItem>(&mut conn)
            .expect("cursor fetch SQL must be valid");
        assert!(fetched.is_empty());
    }
}
