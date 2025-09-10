use crate::models::schema::{
    assistant_messages, conversations, responses, tool_calls, tool_outputs, user_messages, user_system_prompts,
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
    #[error("Conversation not found")]
    ConversationNotFound,
    #[error("Response not found")]
    ResponseNotFound,
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
// Responses (Job Tracker)
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = responses)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Response {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub status: ResponseStatus,
    pub model: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<i32>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    pub idempotency_key: Option<String>,
    pub request_hash: Option<String>,
    pub idempotency_expires_at: Option<DateTime<Utc>>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = responses)]
pub struct NewResponse {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub status: ResponseStatus,
    pub model: String,
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

impl Response {
    pub fn get_by_uuid_and_user(
        conn: &mut PgConnection,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Response, ResponsesError> {
        responses::table
            .filter(responses::uuid.eq(uuid))
            .filter(responses::user_id.eq(user_id))
            .first::<Response>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::ResponseNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn get_by_idempotency_key(
        conn: &mut PgConnection,
        user_id: Uuid,
        key: &str,
    ) -> Result<Option<Response>, ResponsesError> {
        responses::table
            .filter(responses::user_id.eq(user_id))
            .filter(responses::idempotency_key.eq(key))
            .filter(responses::idempotency_expires_at.gt(diesel::dsl::now))
            .first::<Response>(conn)
            .optional()
            .map_err(ResponsesError::DatabaseError)
    }

    pub fn update_status(
        conn: &mut PgConnection,
        id: i64,
        status: ResponseStatus,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), ResponsesError> {
        diesel::update(responses::table.filter(responses::id.eq(id)))
            .set((
                responses::status.eq(status),
                responses::completed_at.eq(completed_at),
                responses::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map(|_| ())
            .map_err(ResponsesError::DatabaseError)
    }

    pub fn cancel_by_uuid_and_user(
        conn: &mut PgConnection,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Response, ResponsesError> {
        let response = Self::get_by_uuid_and_user(conn, uuid, user_id)?;
        
        match response.status {
            ResponseStatus::InProgress | ResponseStatus::Queued => {
                diesel::update(responses::table.filter(responses::id.eq(response.id)))
                    .set((
                        responses::status.eq(ResponseStatus::Cancelled),
                        responses::completed_at.eq(diesel::dsl::now),
                        responses::updated_at.eq(diesel::dsl::now),
                    ))
                    .get_result(conn)
                    .map_err(ResponsesError::DatabaseError)
            },
            _ => Err(ResponsesError::ValidationError),
        }
    }

    pub fn delete_by_uuid_and_user(
        conn: &mut PgConnection,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<(), ResponsesError> {
        diesel::delete(
            responses::table
                .filter(responses::uuid.eq(uuid))
                .filter(responses::user_id.eq(user_id)),
        )
        .execute(conn)
        .map(|rows| {
            if rows == 0 {
                Err(ResponsesError::ResponseNotFound)
            } else {
                Ok(())
            }
        })?
    }

    pub fn cleanup_expired_idempotency_keys(
        conn: &mut PgConnection,
    ) -> Result<u64, ResponsesError> {
        diesel::update(
            responses::table
                .filter(responses::idempotency_expires_at.lt(diesel::dsl::now))
                .filter(responses::idempotency_key.is_not_null()),
        )
        .set((
            responses::idempotency_key.eq(None::<String>),
            responses::request_hash.eq(None::<String>),
            responses::idempotency_expires_at.eq(None::<DateTime<Utc>>),
        ))
        .execute(conn)
        .map(|count| count as u64)
        .map_err(ResponsesError::DatabaseError)
    }
}

impl NewResponse {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<Response, ResponsesError> {
        diesel::insert_into(responses::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
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
    pub prompt_tokens: i32,
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
    pub prompt_tokens: i32,
    pub is_default: bool,
}

// ============================================================================
// Conversations (formerly Chat Threads)
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = conversations)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Conversation {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub system_prompt_id: Option<i64>,
    pub title_enc: Option<Vec<u8>>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = conversations)]
pub struct NewConversation {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub system_prompt_id: Option<i64>,
    pub title_enc: Option<Vec<u8>>,
    pub metadata: Option<serde_json::Value>,
}

impl Conversation {
    pub fn get_by_id_and_user(
        conn: &mut PgConnection,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<Conversation, ResponsesError> {
        conversations::table
            .filter(conversations::id.eq(conversation_id))
            .filter(conversations::user_id.eq(user_id))
            .first::<Conversation>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::ConversationNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn get_by_uuid_and_user(
        conn: &mut PgConnection,
        conversation_uuid: Uuid,
        user_id: Uuid,
    ) -> Result<Conversation, ResponsesError> {
        conversations::table
            .filter(conversations::uuid.eq(conversation_uuid))
            .filter(conversations::user_id.eq(user_id))
            .first::<Conversation>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::ConversationNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }

    pub fn update_title(
        conn: &mut PgConnection,
        conversation_id: i64,
        title_enc: Vec<u8>,
    ) -> Result<(), ResponsesError> {
        diesel::update(conversations::table.filter(conversations::id.eq(conversation_id)))
            .set((
                conversations::title_enc.eq(title_enc),
                conversations::updated_at.eq(diesel::dsl::now),
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
    ) -> Result<Vec<Conversation>, ResponsesError> {
        let mut query = conversations::table
            .filter(conversations::user_id.eq(user_id))
            .into_boxed();

        if let Some((updated_at, id)) = after {
            query = query.filter(
                conversations::updated_at
                    .lt(updated_at)
                    .or(conversations::updated_at
                        .eq(updated_at)
                        .and(conversations::id.lt(id))),
            );
        }

        if let Some((updated_at, id)) = before {
            query = query.filter(
                conversations::updated_at
                    .gt(updated_at)
                    .or(conversations::updated_at
                        .eq(updated_at)
                        .and(conversations::id.gt(id))),
            );
        }

        query
            .order(conversations::updated_at.desc())
            .limit(limit)
            .load::<Conversation>(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

impl NewConversation {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<Conversation, ResponsesError> {
        diesel::insert_into(conversations::table)
            .values(self)
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }

    /// Creates a new conversation with optional response and first message in a transaction
    pub fn create_with_response_and_message(
        conn: &mut PgConnection,
        conversation_uuid: Uuid,
        user_id: Uuid,
        system_prompt_id: Option<i64>,
        title_enc: Option<Vec<u8>>,
        metadata: Option<serde_json::Value>,
        response: Option<NewResponse>,
        first_message_content: Vec<u8>,
        first_message_tokens: i32,
    ) -> Result<(Conversation, Option<Response>, UserMessage), ResponsesError> {
        use diesel::Connection;

        conn.transaction(|tx| {
            // Create the conversation
            let new_conversation = NewConversation {
                uuid: conversation_uuid,
                user_id,
                system_prompt_id,
                title_enc,
                metadata,
            };
            let conversation = new_conversation.insert(tx)?;

            // Create the response if provided
            let response_result = if let Some(mut new_response) = response {
                new_response.conversation_id = conversation.id;
                Some(new_response.insert(tx)?)
            } else {
                None
            };

            // Create the first message
            let new_message = NewUserMessage {
                uuid: Uuid::new_v4(),
                conversation_id: conversation.id,
                response_id: response_result.as_ref().map(|r| r.id),
                user_id,
                content_enc: first_message_content,
                prompt_tokens: first_message_tokens,
            };
            let user_message = new_message.insert(tx)?;

            Ok((conversation, response_result, user_message))
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
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub prompt_tokens: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = user_messages)]
pub struct NewUserMessage {
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub prompt_tokens: i32,
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

    // Note: idempotency is now handled at the Response level

    // Note: status is now tracked on Response, not UserMessage

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

    // Note: cleanup_expired_idempotency_keys is now on Response

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

    // Note: cancellation is now handled at the Response level
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
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub tool_call_id: Uuid,
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: i32,
    pub status: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = tool_calls)]
pub struct NewToolCall {
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub tool_call_id: Uuid,
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: i32,
    pub status: Option<String>,
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
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub tool_call_fk: i64,
    pub output_enc: Vec<u8>,
    pub output_tokens: i32,
    pub status: String,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = tool_outputs)]
pub struct NewToolOutput {
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub tool_call_fk: i64,
    pub output_enc: Vec<u8>,
    pub output_tokens: i32,
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
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub completion_tokens: i32,
    pub finish_reason: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = assistant_messages)]
pub struct NewAssistantMessage {
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Vec<u8>,
    pub completion_tokens: i32,
    pub finish_reason: Option<String>,
}

impl AssistantMessage {
    pub fn get_by_response_id(
        conn: &mut PgConnection,
        response_id: i64,
    ) -> Result<Vec<AssistantMessage>, ResponsesError> {
        assistant_messages::table
            .filter(assistant_messages::response_id.eq(response_id))
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
    pub fn get_conversation_context(
        conn: &mut PgConnection,
        conversation_id: i64,
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        let query = r#"
            WITH conversation_messages AS (
                -- User messages
                SELECT 
                    'user' as message_type,
                    um.id,
                    um.uuid,
                    um.content_enc,
                    um.created_at,
                    r.model,
                    um.prompt_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason
                FROM user_messages um
                LEFT JOIN responses r ON um.response_id = r.id
                WHERE um.conversation_id = $1
                
                UNION ALL
                
                -- Assistant messages
                SELECT 
                    'assistant' as message_type,
                    am.id,
                    am.uuid,
                    am.content_enc,
                    am.created_at,
                    r.model,
                    am.completion_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    am.finish_reason
                FROM assistant_messages am
                LEFT JOIN responses r ON am.response_id = r.id
                WHERE am.conversation_id = $1
                
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
                WHERE tc.conversation_id = $1
                
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
                WHERE tto.conversation_id = $1
            )
            SELECT * FROM conversation_messages
            ORDER BY created_at ASC
        "#;

        sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .load::<RawThreadMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}
