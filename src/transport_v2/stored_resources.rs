//! Bounded, transport-v2-only storage projections for encrypted resources.
//!
//! These queries deliberately do not reuse the transport-v1 row loaders. A
//! database attacker can enlarge encrypted columns, so v2 first measures the
//! exact page or item in a repeatable-read snapshot and only then loads the
//! narrow ciphertext projection. Mutations return metadata rather than stored
//! ciphertext whenever the caller already owns the final plaintext response.

use chrono::{DateTime, Utc};
use diesel::dsl::sql;
use diesel::prelude::*;
use diesel::sql_types::BigInt;
use diesel::{Connection, OptionalExtension};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::models::responses::{NewUserInstruction, ProjectInstructionUpdate};
use crate::models::schema::{conversation_projects, user_instructions};

const AES_GCM_STORAGE_OVERHEAD_BYTES: usize = 12 + 16;

type Pool = diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum StoredResourceError {
    #[error("conversation project not found")]
    ConversationProjectNotFound,
    #[error("instruction not found")]
    InstructionNotFound,
    #[error("stored output exceeds the logical response limit")]
    OutputTooLarge,
    #[error("stored output changed within a bounded snapshot")]
    InconsistentSnapshot,
    #[error("resource changed before its mutation could be committed")]
    StaleResource,
    #[error("database connection unavailable")]
    Connection,
    #[error("database error: {0}")]
    Database(#[from] diesel::result::Error),
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = conversation_projects)]
pub(crate) struct ProjectCiphertextRow {
    pub(crate) id: i64,
    pub(crate) uuid: Uuid,
    pub(crate) name_enc: Vec<u8>,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

impl Zeroize for ProjectCiphertextRow {
    fn zeroize(&mut self) {
        self.name_enc.zeroize();
    }
}

impl Drop for ProjectCiphertextRow {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub(crate) struct ProjectWithInstructionCiphertext {
    pub(crate) project: ProjectCiphertextRow,
    pub(crate) prompt_enc: Option<Vec<u8>>,
}

impl Zeroize for ProjectWithInstructionCiphertext {
    fn zeroize(&mut self) {
        self.project.zeroize();
        if let Some(prompt) = self.prompt_enc.as_mut() {
            prompt.zeroize();
        }
    }
}

impl Drop for ProjectWithInstructionCiphertext {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub(crate) struct ProjectMutationMetadata {
    pub(crate) uuid: Uuid,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = user_instructions)]
pub(crate) struct InstructionCiphertextRow {
    pub(crate) id: i64,
    pub(crate) uuid: Uuid,
    pub(crate) name_enc: Option<Vec<u8>>,
    pub(crate) prompt_enc: Vec<u8>,
    pub(crate) prompt_tokens: i32,
    pub(crate) is_default: bool,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

impl Zeroize for InstructionCiphertextRow {
    fn zeroize(&mut self) {
        if let Some(name) = self.name_enc.as_mut() {
            name.zeroize();
        }
        self.prompt_enc.zeroize();
    }
}

impl Drop for InstructionCiphertextRow {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub(crate) struct InstructionMutationMetadata {
    pub(crate) uuid: Uuid,
    pub(crate) prompt_tokens: i32,
    pub(crate) is_default: bool,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

pub(crate) struct InstructionUpdateCiphertext {
    pub(crate) name_enc: Vec<u8>,
    pub(crate) prompt_enc: Vec<u8>,
    pub(crate) prompt_tokens: i32,
    pub(crate) is_default: bool,
}

fn map_default_instruction_conflict(error: StoredResourceError) -> StoredResourceError {
    match error {
        StoredResourceError::Database(diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::UniqueViolation,
            ref info,
        )) if info.constraint_name() == Some("idx_user_instructions_one_default") => {
            StoredResourceError::StaleResource
        }
        error => error,
    }
}

fn checked_ciphertext_plaintext_len(length: i64) -> Result<usize, StoredResourceError> {
    usize::try_from(length)
        .map_err(|_| StoredResourceError::InconsistentSnapshot)?
        .checked_sub(AES_GCM_STORAGE_OVERHEAD_BYTES)
        .ok_or(StoredResourceError::InconsistentSnapshot)
}

fn validate_ciphertext_plaintext_total(
    ciphertext_lengths: impl IntoIterator<Item = i64>,
    logical_body_limit: usize,
) -> Result<Vec<usize>, StoredResourceError> {
    let mut expected = Vec::new();
    let mut plaintext_total = 0_usize;
    for length in ciphertext_lengths {
        let ciphertext_length =
            usize::try_from(length).map_err(|_| StoredResourceError::InconsistentSnapshot)?;
        let plaintext_length = checked_ciphertext_plaintext_len(length)?;
        plaintext_total = plaintext_total
            .checked_add(plaintext_length)
            .ok_or(StoredResourceError::OutputTooLarge)?;
        if plaintext_total > logical_body_limit {
            return Err(StoredResourceError::OutputTooLarge);
        }
        expected.push(ciphertext_length);
    }
    Ok(expected)
}

pub(crate) fn get_project(
    pool: &Pool,
    user_id: Uuid,
    project_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<ProjectWithInstructionCiphertext, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredResourceError, _>(|conn| {
            let (project_id, name_length) = conversation_projects::table
                .filter(conversation_projects::uuid.eq(project_uuid))
                .filter(conversation_projects::user_id.eq(user_id))
                .select((
                    conversation_projects::id,
                    sql::<BigInt>("octet_length(name_enc)::bigint"),
                ))
                .first::<(i64, i64)>(conn)
                .optional()?
                .ok_or(StoredResourceError::ConversationProjectNotFound)?;

            let prompt_length = user_instructions::table
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.eq(Some(project_id)))
                .select(sql::<BigInt>("octet_length(prompt_enc)::bigint"))
                .first::<i64>(conn)
                .optional()?;

            let mut measured_lengths = vec![name_length];
            if let Some(prompt_length) = prompt_length {
                measured_lengths.push(prompt_length);
            }
            let expected_lengths =
                validate_ciphertext_plaintext_total(measured_lengths, logical_body_limit)?;

            let project = conversation_projects::table
                .filter(conversation_projects::id.eq(project_id))
                .filter(conversation_projects::user_id.eq(user_id))
                .select(ProjectCiphertextRow::as_select())
                .first::<ProjectCiphertextRow>(conn)?;
            if project.name_enc.len() != expected_lengths[0] {
                return Err(StoredResourceError::InconsistentSnapshot);
            }

            let mut prompt_enc = user_instructions::table
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.eq(Some(project_id)))
                .select(user_instructions::prompt_enc)
                .first::<Vec<u8>>(conn)
                .optional()?;
            match (&prompt_enc, expected_lengths.get(1)) {
                (None, None) => {}
                (Some(prompt), Some(expected)) if prompt.len() == *expected => {}
                _ => {
                    if let Some(prompt) = prompt_enc.as_mut() {
                        prompt.zeroize();
                    }
                    return Err(StoredResourceError::InconsistentSnapshot);
                }
            }

            Ok(ProjectWithInstructionCiphertext {
                project,
                prompt_enc,
            })
        })
}

fn project_cursor(
    conn: &mut PgConnection,
    user_id: Uuid,
    after: Option<Uuid>,
) -> Result<Option<(DateTime<Utc>, i64)>, StoredResourceError> {
    let Some(after) = after else {
        return Ok(None);
    };
    conversation_projects::table
        .filter(conversation_projects::uuid.eq(after))
        .filter(conversation_projects::user_id.eq(user_id))
        .select((conversation_projects::updated_at, conversation_projects::id))
        .first::<(DateTime<Utc>, i64)>(conn)
        .optional()
        .map_err(StoredResourceError::Database)
}

fn project_measure_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
) -> Result<Vec<(Uuid, i64, i64)>, StoredResourceError> {
    let mut query = conversation_projects::table
        .filter(conversation_projects::user_id.eq(user_id))
        .into_boxed();
    if let Some((updated_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                conversation_projects::updated_at.lt(updated_at).or(
                    conversation_projects::updated_at
                        .eq(updated_at)
                        .and(conversation_projects::id.lt(id)),
                ),
            )
        } else {
            query.filter(
                conversation_projects::updated_at.gt(updated_at).or(
                    conversation_projects::updated_at
                        .eq(updated_at)
                        .and(conversation_projects::id.gt(id)),
                ),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            conversation_projects::updated_at.desc(),
            conversation_projects::id.desc(),
        ))
    } else {
        query.order((
            conversation_projects::updated_at.asc(),
            conversation_projects::id.asc(),
        ))
    };
    query
        .select((
            conversation_projects::uuid,
            conversation_projects::id,
            sql::<BigInt>("octet_length(name_enc)::bigint"),
        ))
        .limit(limit)
        .load::<(Uuid, i64, i64)>(conn)
        .map_err(StoredResourceError::Database)
}

fn project_fetch_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
) -> Result<Vec<ProjectCiphertextRow>, StoredResourceError> {
    let mut query = conversation_projects::table
        .filter(conversation_projects::user_id.eq(user_id))
        .into_boxed();
    if let Some((updated_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                conversation_projects::updated_at.lt(updated_at).or(
                    conversation_projects::updated_at
                        .eq(updated_at)
                        .and(conversation_projects::id.lt(id)),
                ),
            )
        } else {
            query.filter(
                conversation_projects::updated_at.gt(updated_at).or(
                    conversation_projects::updated_at
                        .eq(updated_at)
                        .and(conversation_projects::id.gt(id)),
                ),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            conversation_projects::updated_at.desc(),
            conversation_projects::id.desc(),
        ))
    } else {
        query.order((
            conversation_projects::updated_at.asc(),
            conversation_projects::id.asc(),
        ))
    };
    query
        .select(ProjectCiphertextRow::as_select())
        .limit(limit)
        .load::<ProjectCiphertextRow>(conn)
        .map_err(StoredResourceError::Database)
}

pub(crate) fn list_projects(
    pool: &Pool,
    user_id: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
    logical_body_limit: usize,
) -> Result<(Vec<ProjectCiphertextRow>, bool), StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredResourceError, _>(|conn| {
            let cursor = project_cursor(conn, user_id, after)?;
            let sentinel_limit = limit
                .checked_add(1)
                .ok_or(StoredResourceError::OutputTooLarge)?;
            let mut measured = project_measure_page(conn, user_id, sentinel_limit, cursor, order)?;
            let has_more = measured.len()
                > usize::try_from(limit).map_err(|_| StoredResourceError::OutputTooLarge)?;
            if has_more {
                measured.pop();
            }
            let expected_lengths = validate_ciphertext_plaintext_total(
                measured.iter().map(|(_, _, length)| *length),
                logical_body_limit,
            )?;
            let rows = project_fetch_page(conn, user_id, limit, cursor, order)?;
            if rows.len() != measured.len() {
                return Err(StoredResourceError::InconsistentSnapshot);
            }
            for ((row, (expected_uuid, expected_id, _)), expected_length) in
                rows.iter().zip(measured.iter()).zip(expected_lengths)
            {
                if row.uuid != *expected_uuid
                    || row.id != *expected_id
                    || row.name_enc.len() != expected_length
                {
                    return Err(StoredResourceError::InconsistentSnapshot);
                }
            }
            Ok((rows, has_more))
        })
}

pub(crate) fn update_project(
    pool: &Pool,
    user_id: Uuid,
    project_uuid: Uuid,
    expected_updated_at: DateTime<Utc>,
    name_enc: Option<Vec<u8>>,
    instruction_update: ProjectInstructionUpdate,
) -> Result<ProjectMutationMetadata, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.transaction::<_, StoredResourceError, _>(|conn| {
        let (project_id, current_updated_at) = conversation_projects::table
            .filter(conversation_projects::uuid.eq(project_uuid))
            .filter(conversation_projects::user_id.eq(user_id))
            .select((conversation_projects::id, conversation_projects::updated_at))
            .for_update()
            .first::<(i64, DateTime<Utc>)>(conn)
            .optional()?
            .ok_or(StoredResourceError::ConversationProjectNotFound)?;
        if current_updated_at != expected_updated_at {
            return Err(StoredResourceError::StaleResource);
        }

        if let Some(name_enc) = name_enc {
            diesel::update(
                conversation_projects::table
                    .filter(conversation_projects::id.eq(project_id))
                    .filter(conversation_projects::user_id.eq(user_id)),
            )
            .set((
                conversation_projects::name_enc.eq(name_enc),
                conversation_projects::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)?;
        }

        match instruction_update {
            ProjectInstructionUpdate::Unchanged => {}
            ProjectInstructionUpdate::Set {
                prompt_enc,
                prompt_tokens,
            } => {
                let instruction_id = user_instructions::table
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.eq(Some(project_id)))
                    .select(user_instructions::id)
                    .for_update()
                    .first::<i64>(conn)
                    .optional()?;
                if let Some(instruction_id) = instruction_id {
                    diesel::update(
                        user_instructions::table
                            .filter(user_instructions::id.eq(instruction_id))
                            .filter(user_instructions::user_id.eq(user_id)),
                    )
                    .set((
                        user_instructions::prompt_enc.eq(prompt_enc),
                        user_instructions::prompt_tokens.eq(prompt_tokens),
                        user_instructions::is_default.eq(false),
                        user_instructions::updated_at.eq(diesel::dsl::now),
                    ))
                    .execute(conn)?;
                } else {
                    diesel::insert_into(user_instructions::table)
                        .values(&NewUserInstruction {
                            uuid: Uuid::new_v4(),
                            user_id,
                            project_id: Some(project_id),
                            name_enc: None,
                            prompt_enc,
                            prompt_tokens,
                            is_default: false,
                        })
                        .execute(conn)?;
                }
                diesel::update(
                    conversation_projects::table
                        .filter(conversation_projects::id.eq(project_id))
                        .filter(conversation_projects::user_id.eq(user_id)),
                )
                .set(conversation_projects::updated_at.eq(diesel::dsl::now))
                .execute(conn)?;
            }
            ProjectInstructionUpdate::Clear => {
                diesel::delete(
                    user_instructions::table
                        .filter(user_instructions::user_id.eq(user_id))
                        .filter(user_instructions::project_id.eq(Some(project_id))),
                )
                .execute(conn)?;
                diesel::update(
                    conversation_projects::table
                        .filter(conversation_projects::id.eq(project_id))
                        .filter(conversation_projects::user_id.eq(user_id)),
                )
                .set(conversation_projects::updated_at.eq(diesel::dsl::now))
                .execute(conn)?;
            }
        }

        let (uuid, created_at, updated_at) = conversation_projects::table
            .filter(conversation_projects::id.eq(project_id))
            .filter(conversation_projects::user_id.eq(user_id))
            .select((
                conversation_projects::uuid,
                conversation_projects::created_at,
                conversation_projects::updated_at,
            ))
            .first::<(Uuid, DateTime<Utc>, DateTime<Utc>)>(conn)?;
        Ok(ProjectMutationMetadata {
            uuid,
            created_at,
            updated_at,
        })
    })
}

pub(crate) fn get_instruction(
    pool: &Pool,
    user_id: Uuid,
    instruction_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<InstructionCiphertextRow, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredResourceError, _>(|conn| {
            let measured = user_instructions::table
                .filter(user_instructions::uuid.eq(instruction_uuid))
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.is_null())
                .select((
                    sql::<BigInt>("octet_length(name_enc)::bigint"),
                    sql::<BigInt>("octet_length(prompt_enc)::bigint"),
                ))
                .first::<(i64, i64)>(conn)
                .optional()?
                .ok_or(StoredResourceError::InstructionNotFound)?;
            let expected_lengths =
                validate_ciphertext_plaintext_total([measured.0, measured.1], logical_body_limit)?;

            let mut row = user_instructions::table
                .filter(user_instructions::uuid.eq(instruction_uuid))
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.is_null())
                .select(InstructionCiphertextRow::as_select())
                .first::<InstructionCiphertextRow>(conn)?;
            let Some(name) = row.name_enc.as_ref() else {
                row.zeroize();
                return Err(StoredResourceError::InconsistentSnapshot);
            };
            if name.len() != expected_lengths[0] || row.prompt_enc.len() != expected_lengths[1] {
                row.zeroize();
                return Err(StoredResourceError::InconsistentSnapshot);
            }
            Ok(row)
        })
}

fn instruction_cursor(
    conn: &mut PgConnection,
    user_id: Uuid,
    after: Option<Uuid>,
) -> Result<Option<(DateTime<Utc>, i64)>, StoredResourceError> {
    let Some(after) = after else {
        return Ok(None);
    };
    user_instructions::table
        .filter(user_instructions::uuid.eq(after))
        .filter(user_instructions::user_id.eq(user_id))
        .filter(user_instructions::project_id.is_null())
        .select((user_instructions::updated_at, user_instructions::id))
        .first::<(DateTime<Utc>, i64)>(conn)
        .optional()
        .map_err(StoredResourceError::Database)
}

fn instruction_measure_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
) -> Result<Vec<(Uuid, i64, i64, i64)>, StoredResourceError> {
    let mut query = user_instructions::table
        .filter(user_instructions::user_id.eq(user_id))
        .filter(user_instructions::project_id.is_null())
        .into_boxed();
    if let Some((updated_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                user_instructions::updated_at
                    .lt(updated_at)
                    .or(user_instructions::updated_at
                        .eq(updated_at)
                        .and(user_instructions::id.lt(id))),
            )
        } else {
            query.filter(
                user_instructions::updated_at
                    .gt(updated_at)
                    .or(user_instructions::updated_at
                        .eq(updated_at)
                        .and(user_instructions::id.gt(id))),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            user_instructions::updated_at.desc(),
            user_instructions::id.desc(),
        ))
    } else {
        query.order((
            user_instructions::updated_at.asc(),
            user_instructions::id.asc(),
        ))
    };
    query
        .select((
            user_instructions::uuid,
            user_instructions::id,
            sql::<BigInt>("octet_length(name_enc)::bigint"),
            sql::<BigInt>("octet_length(prompt_enc)::bigint"),
        ))
        .limit(limit)
        .load::<(Uuid, i64, i64, i64)>(conn)
        .map_err(StoredResourceError::Database)
}

fn instruction_fetch_page(
    conn: &mut PgConnection,
    user_id: Uuid,
    limit: i64,
    cursor: Option<(DateTime<Utc>, i64)>,
    order: &str,
) -> Result<Vec<InstructionCiphertextRow>, StoredResourceError> {
    let mut query = user_instructions::table
        .filter(user_instructions::user_id.eq(user_id))
        .filter(user_instructions::project_id.is_null())
        .into_boxed();
    if let Some((updated_at, id)) = cursor {
        query = if order == "desc" {
            query.filter(
                user_instructions::updated_at
                    .lt(updated_at)
                    .or(user_instructions::updated_at
                        .eq(updated_at)
                        .and(user_instructions::id.lt(id))),
            )
        } else {
            query.filter(
                user_instructions::updated_at
                    .gt(updated_at)
                    .or(user_instructions::updated_at
                        .eq(updated_at)
                        .and(user_instructions::id.gt(id))),
            )
        };
    }
    query = if order == "desc" {
        query.order((
            user_instructions::updated_at.desc(),
            user_instructions::id.desc(),
        ))
    } else {
        query.order((
            user_instructions::updated_at.asc(),
            user_instructions::id.asc(),
        ))
    };
    query
        .select(InstructionCiphertextRow::as_select())
        .limit(limit)
        .load::<InstructionCiphertextRow>(conn)
        .map_err(StoredResourceError::Database)
}

pub(crate) fn list_instructions(
    pool: &Pool,
    user_id: Uuid,
    limit: i64,
    after: Option<Uuid>,
    order: &str,
    logical_body_limit: usize,
) -> Result<(Vec<InstructionCiphertextRow>, bool), StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, StoredResourceError, _>(|conn| {
            let cursor = instruction_cursor(conn, user_id, after)?;
            let sentinel_limit = limit
                .checked_add(1)
                .ok_or(StoredResourceError::OutputTooLarge)?;
            let mut measured =
                instruction_measure_page(conn, user_id, sentinel_limit, cursor, order)?;
            let has_more = measured.len()
                > usize::try_from(limit).map_err(|_| StoredResourceError::OutputTooLarge)?;
            if has_more {
                measured.pop();
            }
            let expected_lengths = validate_ciphertext_plaintext_total(
                measured
                    .iter()
                    .flat_map(|(_, _, name_length, prompt_length)| [*name_length, *prompt_length]),
                logical_body_limit,
            )?;
            let mut rows = instruction_fetch_page(conn, user_id, limit, cursor, order)?;
            if rows.len() != measured.len() {
                rows.zeroize();
                return Err(StoredResourceError::InconsistentSnapshot);
            }
            for (index, (row, (expected_uuid, expected_id, _, _))) in
                rows.iter().zip(measured.iter()).enumerate()
            {
                let Some(name) = row.name_enc.as_ref() else {
                    rows.zeroize();
                    return Err(StoredResourceError::InconsistentSnapshot);
                };
                if row.uuid != *expected_uuid
                    || row.id != *expected_id
                    || name.len() != expected_lengths[index * 2]
                    || row.prompt_enc.len() != expected_lengths[index * 2 + 1]
                {
                    rows.zeroize();
                    return Err(StoredResourceError::InconsistentSnapshot);
                }
            }
            Ok((rows, has_more))
        })
}

pub(crate) fn update_instruction(
    pool: &Pool,
    user_id: Uuid,
    instruction_uuid: Uuid,
    expected_updated_at: DateTime<Utc>,
    update: InstructionUpdateCiphertext,
) -> Result<InstructionMutationMetadata, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.transaction::<_, StoredResourceError, _>(|conn| {
        let (instruction_id, current_updated_at) = user_instructions::table
            .filter(user_instructions::uuid.eq(instruction_uuid))
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null())
            .select((user_instructions::id, user_instructions::updated_at))
            .for_update()
            .first::<(i64, DateTime<Utc>)>(conn)
            .optional()?
            .ok_or(StoredResourceError::InstructionNotFound)?;
        if current_updated_at != expected_updated_at {
            return Err(StoredResourceError::StaleResource);
        }
        if update.is_default {
            diesel::update(
                user_instructions::table
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null())
                    .filter(user_instructions::is_default.eq(true))
                    .filter(user_instructions::id.ne(instruction_id)),
            )
            .set(user_instructions::is_default.eq(false))
            .execute(conn)?;
        }
        let (uuid, prompt_tokens, is_default, created_at, updated_at) = diesel::update(
            user_instructions::table
                .filter(user_instructions::id.eq(instruction_id))
                .filter(user_instructions::user_id.eq(user_id))
                .filter(user_instructions::project_id.is_null()),
        )
        .set((
            user_instructions::name_enc.eq(Some(update.name_enc)),
            user_instructions::prompt_enc.eq(update.prompt_enc),
            user_instructions::prompt_tokens.eq(update.prompt_tokens),
            user_instructions::is_default.eq(update.is_default),
            user_instructions::updated_at.eq(diesel::dsl::now),
        ))
        .returning((
            user_instructions::uuid,
            user_instructions::prompt_tokens,
            user_instructions::is_default,
            user_instructions::created_at,
            user_instructions::updated_at,
        ))
        .get_result::<(Uuid, i32, bool, DateTime<Utc>, DateTime<Utc>)>(conn)?;
        Ok(InstructionMutationMetadata {
            uuid,
            prompt_tokens,
            is_default,
            created_at,
            updated_at,
        })
    })
    .map_err(map_default_instruction_conflict)
}

pub(crate) fn set_default_instruction(
    pool: &Pool,
    user_id: Uuid,
    instruction_uuid: Uuid,
    expected_updated_at: DateTime<Utc>,
) -> Result<InstructionMutationMetadata, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    conn.transaction::<_, StoredResourceError, _>(|conn| {
        let (instruction_id, current_is_default, current_updated_at) = user_instructions::table
            .filter(user_instructions::uuid.eq(instruction_uuid))
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null())
            .select((
                user_instructions::id,
                user_instructions::is_default,
                user_instructions::updated_at,
            ))
            .for_update()
            .first::<(i64, bool, DateTime<Utc>)>(conn)
            .optional()?
            .ok_or(StoredResourceError::InstructionNotFound)?;
        if current_updated_at != expected_updated_at {
            return Err(StoredResourceError::StaleResource);
        }
        let (uuid, prompt_tokens, is_default, created_at, updated_at) = if current_is_default {
            user_instructions::table
                .filter(user_instructions::id.eq(instruction_id))
                .select((
                    user_instructions::uuid,
                    user_instructions::prompt_tokens,
                    user_instructions::is_default,
                    user_instructions::created_at,
                    user_instructions::updated_at,
                ))
                .first::<(Uuid, i32, bool, DateTime<Utc>, DateTime<Utc>)>(conn)?
        } else {
            diesel::update(
                user_instructions::table
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null())
                    .filter(user_instructions::is_default.eq(true)),
            )
            .set(user_instructions::is_default.eq(false))
            .execute(conn)?;
            diesel::update(
                user_instructions::table
                    .filter(user_instructions::id.eq(instruction_id))
                    .filter(user_instructions::user_id.eq(user_id))
                    .filter(user_instructions::project_id.is_null()),
            )
            .set((
                user_instructions::is_default.eq(true),
                user_instructions::updated_at.eq(diesel::dsl::now),
            ))
            .returning((
                user_instructions::uuid,
                user_instructions::prompt_tokens,
                user_instructions::is_default,
                user_instructions::created_at,
                user_instructions::updated_at,
            ))
            .get_result::<(Uuid, i32, bool, DateTime<Utc>, DateTime<Utc>)>(conn)?
        };
        Ok(InstructionMutationMetadata {
            uuid,
            prompt_tokens,
            is_default,
            created_at,
            updated_at,
        })
    })
    .map_err(map_default_instruction_conflict)
}

pub(crate) fn delete_instruction(
    pool: &Pool,
    user_id: Uuid,
    instruction_uuid: Uuid,
) -> Result<Uuid, StoredResourceError> {
    let mut conn = pool.get().map_err(|_| StoredResourceError::Connection)?;
    diesel::delete(
        user_instructions::table
            .filter(user_instructions::uuid.eq(instruction_uuid))
            .filter(user_instructions::user_id.eq(user_id))
            .filter(user_instructions::project_id.is_null()),
    )
    .returning(user_instructions::uuid)
    .get_result::<Uuid>(&mut conn)
    .optional()?
    .ok_or(StoredResourceError::InstructionNotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ciphertext_preflight_accounts_for_storage_overhead() {
        assert_eq!(
            validate_ciphertext_plaintext_total([28, 31], 3).unwrap(),
            vec![28, 31]
        );
        assert!(matches!(
            validate_ciphertext_plaintext_total([28, 32], 3),
            Err(StoredResourceError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_ciphertext_plaintext_total([27], usize::MAX),
            Err(StoredResourceError::InconsistentSnapshot)
        ));
        assert!(matches!(
            validate_ciphertext_plaintext_total([-1], usize::MAX),
            Err(StoredResourceError::InconsistentSnapshot)
        ));
    }
}
