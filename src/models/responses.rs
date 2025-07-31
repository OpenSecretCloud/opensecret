use crate::models::schema::{
    assistant_messages, chat_threads, tool_calls, tool_outputs, user_messages, user_system_prompts,
};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{BigInt, Nullable};
use diesel_derive_enum::DbEnum;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// Error types
#[derive(Error, Debug)]
pub enum ResponsesError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("Chat thread not found")]
    ChatThreadNotFound,
    #[error("User message not found")]
    UserMessageNotFound,
    #[error("System prompt not found")]
    SystemPromptNotFound,
    #[error("Tool call not found")]
    ToolCallNotFound,
    #[error("Tool output not found")]
    ToolOutputNotFound,
    #[error("Assistant message not found")]
    AssistantMessageNotFound,
    #[error("Unauthorized access")]
    Unauthorized,
    #[error("Validation error")]
    ValidationError,
}

// Response status enum matching the database enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DbEnum)]
#[ExistingTypePath = "crate::models::schema::sql_types::ResponseStatus"]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

// ============================================================================
// User System Prompts
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = user_system_prompts)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct UserSystemPrompt {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub name_enc: Vec<u8>,
    pub prompt_enc: Vec<u8>,
    pub prompt_tokens: Option<i32>,
    pub is_default: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = user_system_prompts)]
pub struct NewUserSystemPrompt {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub name_enc: Vec<u8>,
    pub prompt_enc: Vec<u8>,
    pub prompt_tokens: Option<i32>,
    pub is_default: bool,
}

// ============================================================================
// Chat Threads
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = chat_threads)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ChatThread {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub system_prompt_id: Option<i64>,
    pub title_enc: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = chat_threads)]
pub struct NewChatThread {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub system_prompt_id: Option<i64>,
    pub title_enc: Option<Vec<u8>>,
}

impl ChatThread {
    pub fn get_by_id_and_user(
        conn: &mut PgConnection,
        thread_id: i64,
        user_id: Uuid,
    ) -> Result<ChatThread, ResponsesError> {
        chat_threads::table
            .filter(chat_threads::id.eq(thread_id))
            .filter(chat_threads::user_id.eq(user_id))
            .first::<ChatThread>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::ChatThreadNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn update_title(
        conn: &mut PgConnection,
        thread_id: i64,
        title_enc: Vec<u8>,
    ) -> Result<(), ResponsesError> {
        diesel::update(chat_threads::table.filter(chat_threads::id.eq(thread_id)))
            .set((
                chat_threads::title_enc.eq(title_enc),
                chat_threads::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map(|_| ())
            .map_err(ResponsesError::DatabaseError)
    }
}

impl NewChatThread {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<ChatThread, ResponsesError> {
        diesel::insert_into(chat_threads::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }

    /// Creates a new thread and its first user message in a transaction
    pub fn create_with_first_message(
        conn: &mut PgConnection,
        thread_uuid: Uuid,
        user_id: Uuid,
        system_prompt_id: Option<i64>,
        first_message: NewUserMessage,
    ) -> Result<(ChatThread, UserMessage), ResponsesError> {
        use diesel::Connection;

        conn.transaction(|tx| {
            // Create the thread
            let new_thread = NewChatThread {
                uuid: thread_uuid,
                user_id,
                system_prompt_id,
                title_enc: None,
            };
            let thread = new_thread.insert(tx)?;

            // Create the first message with the thread's ID
            let mut message_with_thread = first_message;
            message_with_thread.thread_id = thread.id;
            let user_message = message_with_thread.insert(tx)?;

            Ok((thread, user_message))
        })
    }
}

// ============================================================================
// User Messages
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = user_messages)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct UserMessage {
    pub id: i64,
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub prompt_tokens: Option<i32>,
    pub status: ResponseStatus,
    pub model: String,
    pub previous_response_id: Option<Uuid>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<i32>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub metadata: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    pub idempotency_key: Option<String>,
    pub request_hash: Option<String>,
    pub idempotency_expires_at: Option<DateTime<Utc>>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = user_messages)]
pub struct NewUserMessage {
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub prompt_tokens: Option<i32>,
    pub status: ResponseStatus,
    pub model: String,
    pub previous_response_id: Option<Uuid>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<i32>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub metadata: Option<serde_json::Value>,
    pub idempotency_key: Option<String>,
    pub request_hash: Option<String>,
    pub idempotency_expires_at: Option<DateTime<Utc>>,
}

impl UserMessage {
    pub fn get_by_id_and_user(
        conn: &mut PgConnection,
        id: i64,
        user_id: Uuid,
    ) -> Result<UserMessage, ResponsesError> {
        user_messages::table
            .filter(user_messages::id.eq(id))
            .filter(user_messages::user_id.eq(user_id))
            .first::<UserMessage>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::UserMessageNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn get_by_uuid_and_user(
        conn: &mut PgConnection,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<UserMessage, ResponsesError> {
        user_messages::table
            .filter(user_messages::uuid.eq(uuid))
            .filter(user_messages::user_id.eq(user_id))
            .first::<UserMessage>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::UserMessageNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn get_by_idempotency_key(
        conn: &mut PgConnection,
        user_id: Uuid,
        key: &str,
    ) -> Result<Option<UserMessage>, ResponsesError> {
        user_messages::table
            .filter(user_messages::user_id.eq(user_id))
            .filter(user_messages::idempotency_key.eq(key))
            .filter(user_messages::idempotency_expires_at.gt(diesel::dsl::now))
            .first::<UserMessage>(conn)
            .optional()
            .map_err(ResponsesError::DatabaseError)
    }

    pub fn update_status(
        conn: &mut PgConnection,
        id: i64,
        status: ResponseStatus,
        error: Option<String>,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), ResponsesError> {
        diesel::update(user_messages::table.filter(user_messages::id.eq(id)))
            .set((
                user_messages::status.eq(status),
                user_messages::error.eq(error),
                user_messages::completed_at.eq(completed_at),
                user_messages::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map(|_| ())
            .map_err(ResponsesError::DatabaseError)
    }

    pub fn list_for_user(
        conn: &mut PgConnection,
        user_id: Uuid,
        limit: i64,
        after: Option<(DateTime<Utc>, i64)>,
        before: Option<(DateTime<Utc>, i64)>,
    ) -> Result<Vec<UserMessage>, ResponsesError> {
        let mut query = user_messages::table
            .filter(user_messages::user_id.eq(user_id))
            .into_boxed();

        if let Some((created_at, id)) = after {
            query = query.filter(
                user_messages::created_at
                    .lt(created_at)
                    .or(user_messages::created_at
                        .eq(created_at)
                        .and(user_messages::id.lt(id))),
            );
        }

        if let Some((created_at, id)) = before {
            query = query.filter(
                user_messages::created_at
                    .gt(created_at)
                    .or(user_messages::created_at
                        .eq(created_at)
                        .and(user_messages::id.gt(id))),
            );
        }

        query
            .order(user_messages::created_at.desc())
            .limit(limit)
            .load::<UserMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }

    pub fn cleanup_expired_idempotency_keys(
        conn: &mut PgConnection,
    ) -> Result<u64, ResponsesError> {
        diesel::update(
            user_messages::table
                .filter(user_messages::idempotency_expires_at.lt(diesel::dsl::now))
                .filter(user_messages::idempotency_key.is_not_null()),
        )
        .set((
            user_messages::idempotency_key.eq(None::<String>),
            user_messages::request_hash.eq(None::<String>),
            user_messages::idempotency_expires_at.eq(None::<DateTime<Utc>>),
        ))
        .execute(conn)
        .map(|count| count as u64)
        .map_err(ResponsesError::DatabaseError)
    }

    pub fn delete_by_id_and_user(
        conn: &mut PgConnection,
        id: i64,
        user_id: Uuid,
    ) -> Result<(), ResponsesError> {
        diesel::delete(
            user_messages::table
                .filter(user_messages::id.eq(id))
                .filter(user_messages::user_id.eq(user_id)),
        )
        .execute(conn)
        .map(|rows| {
            if rows == 0 {
                Err(ResponsesError::UserMessageNotFound)
            } else {
                Ok(())
            }
        })?
    }

    pub fn cancel_by_id_and_user(
        conn: &mut PgConnection,
        id: i64,
        user_id: Uuid,
    ) -> Result<UserMessage, ResponsesError> {
        // First check the current status
        let message = user_messages::table
            .filter(user_messages::id.eq(id))
            .filter(user_messages::user_id.eq(user_id))
            .first::<UserMessage>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::UserMessageNotFound,
                _ => ResponsesError::DatabaseError(e),
            })?;

        // Only allow cancelling if in progress
        match message.status {
            ResponseStatus::InProgress | ResponseStatus::Queued => diesel::update(
                user_messages::table
                    .filter(user_messages::id.eq(id))
                    .filter(user_messages::user_id.eq(user_id)),
            )
            .set((
                user_messages::status.eq(ResponseStatus::Cancelled),
                user_messages::completed_at.eq(diesel::dsl::now),
                user_messages::updated_at.eq(diesel::dsl::now),
            ))
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError),
            _ => Err(ResponsesError::ValidationError),
        }
    }
}

impl NewUserMessage {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<UserMessage, ResponsesError> {
        diesel::insert_into(user_messages::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

// ============================================================================
// Tool Calls
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = tool_calls)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ToolCall {
    pub id: i64,
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_message_id: i64,
    pub tool_call_id: Uuid,
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: Option<i32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = tool_calls)]
pub struct NewToolCall {
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_message_id: i64,
    pub tool_call_id: Uuid,
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: Option<i32>,
}

impl NewToolCall {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<ToolCall, ResponsesError> {
        diesel::insert_into(tool_calls::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

// ============================================================================
// Tool Outputs
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = tool_outputs)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ToolOutput {
    pub id: i64,
    pub uuid: Uuid,
    pub thread_id: i64,
    pub tool_call_fk: i64,
    pub output_enc: Vec<u8>,
    pub output_tokens: Option<i32>,
    pub status: String,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = tool_outputs)]
pub struct NewToolOutput {
    pub uuid: Uuid,
    pub thread_id: i64,
    pub tool_call_fk: i64,
    pub output_enc: Vec<u8>,
    pub output_tokens: Option<i32>,
    pub status: String,
    pub error: Option<String>,
}

impl NewToolOutput {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<ToolOutput, ResponsesError> {
        diesel::insert_into(tool_outputs::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

// ============================================================================
// Assistant Messages
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = assistant_messages)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct AssistantMessage {
    pub id: i64,
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_message_id: i64,
    pub content_enc: Vec<u8>,
    pub completion_tokens: Option<i32>,
    pub finish_reason: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = assistant_messages)]
pub struct NewAssistantMessage {
    pub uuid: Uuid,
    pub thread_id: i64,
    pub user_message_id: i64,
    pub content_enc: Vec<u8>,
    pub completion_tokens: Option<i32>,
    pub finish_reason: Option<String>,
}

impl AssistantMessage {
    pub fn get_by_user_message_id(
        conn: &mut PgConnection,
        user_message_id: i64,
    ) -> Result<Vec<AssistantMessage>, ResponsesError> {
        assistant_messages::table
            .filter(assistant_messages::user_message_id.eq(user_message_id))
            .order(assistant_messages::created_at.asc())
            .load::<AssistantMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

impl NewAssistantMessage {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<AssistantMessage, ResponsesError> {
        diesel::insert_into(assistant_messages::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

// ============================================================================
// Helper structs for queries
// ============================================================================

// For the UNION ALL query to get all thread messages
#[derive(QueryableByName, Debug)]
pub struct RawThreadMessage {
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub message_type: String,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub id: i64,
    #[diesel(sql_type = diesel::sql_types::Uuid)]
    pub uuid: Uuid,
    #[diesel(sql_type = diesel::sql_types::Bytea)]
    pub content_enc: Vec<u8>,
    #[diesel(sql_type = diesel::sql_types::Timestamptz)]
    pub created_at: DateTime<Utc>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub model: Option<String>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Integer>)]
    pub token_count: Option<i32>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Uuid>)]
    pub tool_call_id: Option<Uuid>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub finish_reason: Option<String>,
}

impl RawThreadMessage {
    pub fn get_thread_context(
        conn: &mut PgConnection,
        thread_id: i64,
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        let query = r#"
            WITH thread_messages AS (
                -- User messages
                SELECT 
                    'user' as message_type,
                    um.id,
                    um.uuid,
                    um.content_enc,
                    um.created_at,
                    um.model,
                    um.prompt_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason
                FROM user_messages um
                WHERE um.thread_id = $1
                
                UNION ALL
                
                -- Assistant messages
                SELECT 
                    'assistant' as message_type,
                    am.id,
                    am.uuid,
                    am.content_enc,
                    am.created_at,
                    um.model,
                    am.completion_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    am.finish_reason
                FROM assistant_messages am
                JOIN user_messages um ON am.user_message_id = um.id
                WHERE am.thread_id = $1
                
                UNION ALL
                
                -- Tool calls
                SELECT 
                    'tool_call' as message_type,
                    tc.id,
                    tc.uuid,
                    tc.arguments_enc as content_enc,
                    tc.created_at,
                    NULL::text as model,
                    tc.argument_tokens as token_count,
                    tc.tool_call_id,
                    NULL::text as finish_reason
                FROM tool_calls tc
                WHERE tc.thread_id = $1
                
                UNION ALL
                
                -- Tool outputs
                SELECT 
                    'tool_output' as message_type,
                    tto.id,
                    tto.uuid,
                    tto.output_enc as content_enc,
                    tto.created_at,
                    NULL::text as model,
                    tto.output_tokens as token_count,
                    tc.tool_call_id,
                    NULL::text as finish_reason
                FROM tool_outputs tto
                JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                WHERE tto.thread_id = $1
            )
            SELECT * FROM thread_messages
            ORDER BY created_at ASC
        "#;

        sql_query(query)
            .bind::<BigInt, _>(thread_id)
            .load::<RawThreadMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}
