use crate::models::schema::memory_blocks;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Default character limit per core memory block (mirrors Sage/Letta).
pub const DEFAULT_BLOCK_CHAR_LIMIT: i32 = 20_000;

#[derive(Error, Debug)]
pub enum MemoryBlockError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = memory_blocks)]
pub struct MemoryBlock {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub label: String,
    pub description: Option<String>,
    pub value_enc: Vec<u8>,
    pub char_limit: i32,
    pub read_only: bool,
    pub version: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl MemoryBlock {
    pub fn get_by_user_and_label(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_label: &str,
    ) -> Result<Option<MemoryBlock>, MemoryBlockError> {
        memory_blocks::table
            .filter(memory_blocks::user_id.eq(lookup_user_id))
            .filter(memory_blocks::label.eq(lookup_label))
            .first::<MemoryBlock>(conn)
            .optional()
            .map_err(MemoryBlockError::DatabaseError)
    }

    pub fn get_all_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<MemoryBlock>, MemoryBlockError> {
        memory_blocks::table
            .filter(memory_blocks::user_id.eq(lookup_user_id))
            .order(memory_blocks::label.asc())
            .load::<MemoryBlock>(conn)
            .map_err(MemoryBlockError::DatabaseError)
    }

    pub fn delete_all_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<usize, MemoryBlockError> {
        diesel::delete(memory_blocks::table.filter(memory_blocks::user_id.eq(lookup_user_id)))
            .execute(conn)
            .map_err(MemoryBlockError::DatabaseError)
    }

    pub fn delete_by_user_and_label(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_label: &str,
    ) -> Result<usize, MemoryBlockError> {
        diesel::delete(
            memory_blocks::table
                .filter(memory_blocks::user_id.eq(lookup_user_id))
                .filter(memory_blocks::label.eq(lookup_label)),
        )
        .execute(conn)
        .map_err(MemoryBlockError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = memory_blocks)]
pub struct NewMemoryBlock {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub label: String,
    pub description: Option<String>,
    pub value_enc: Vec<u8>,
    pub char_limit: i32,
    pub read_only: bool,
    pub version: i32,
}

impl NewMemoryBlock {
    pub fn new(
        user_id: Uuid,
        label: impl Into<String>,
        description: Option<String>,
        value_enc: Vec<u8>,
    ) -> Self {
        NewMemoryBlock {
            uuid: Uuid::new_v4(),
            user_id,
            label: label.into(),
            description,
            value_enc,
            char_limit: DEFAULT_BLOCK_CHAR_LIMIT,
            read_only: false,
            version: 1,
        }
    }

    pub fn insert_or_update(
        &self,
        conn: &mut PgConnection,
    ) -> Result<MemoryBlock, MemoryBlockError> {
        diesel::insert_into(memory_blocks::table)
            .values(self)
            .on_conflict((memory_blocks::user_id, memory_blocks::label))
            .do_update()
            .set((
                memory_blocks::description.eq(self.description.clone()),
                memory_blocks::value_enc.eq(self.value_enc.clone()),
                memory_blocks::char_limit.eq(self.char_limit),
                memory_blocks::read_only.eq(self.read_only),
                memory_blocks::version.eq(memory_blocks::version + 1),
            ))
            .get_result::<MemoryBlock>(conn)
            .map_err(MemoryBlockError::DatabaseError)
    }
}
