use crate::models::schema::{
    assistant_messages, conversations, reasoning_items, responses, tool_calls, tool_outputs,
    user_instructions, user_messages,
};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sql_query;
use diesel::sql_types::{Array, BigInt, Nullable};
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
    pub metadata_enc: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
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
    pub metadata_enc: Option<Vec<u8>>,
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
            }
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
// User Instructions (System Prompts)
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = user_instructions)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct UserInstruction {
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
#[diesel(table_name = user_instructions)]
pub struct NewUserInstruction {
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
    pub metadata_enc: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = conversations)]
pub struct NewConversation {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub metadata_enc: Option<Vec<u8>>,
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

    pub fn update_metadata(
        conn: &mut PgConnection,
        conversation_id: i64,
        user_id: Uuid,
        metadata_enc: Vec<u8>,
    ) -> Result<(), ResponsesError> {
        let updated = diesel::update(
            conversations::table
                .filter(conversations::id.eq(conversation_id))
                .filter(conversations::user_id.eq(user_id)),
        )
        .set((
            conversations::metadata_enc.eq(metadata_enc),
            conversations::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)?;

        if updated == 0 {
            return Err(ResponsesError::ConversationNotFound);
        }

        Ok(())
    }

    pub fn delete_by_id_and_user(
        conn: &mut PgConnection,
        conversation_id: i64,
        user_id: Uuid,
    ) -> Result<(), ResponsesError> {
        let deleted = diesel::delete(
            conversations::table
                .filter(conversations::id.eq(conversation_id))
                .filter(conversations::user_id.eq(user_id)),
        )
        .execute(conn)?;

        if deleted == 0 {
            return Err(ResponsesError::ConversationNotFound);
        }

        Ok(())
    }

    pub fn delete_all_for_user(
        conn: &mut PgConnection,
        user_id: Uuid,
    ) -> Result<(), ResponsesError> {
        diesel::delete(conversations::table.filter(conversations::user_id.eq(user_id)))
            .execute(conn)
            .map_err(ResponsesError::DatabaseError)?;

        Ok(())
    }

    pub fn list_for_user(
        conn: &mut PgConnection,
        user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<Conversation>, ResponsesError> {
        let mut query = conversations::table
            .filter(conversations::user_id.eq(user_id))
            .into_boxed();

        // If we have an after cursor, we need to find its timestamp and apply cursor-based pagination
        if let Some(after_uuid) = after {
            // Get the cursor conversation to find its updated_at and id
            let cursor_conv = conversations::table
                .filter(conversations::uuid.eq(after_uuid))
                .filter(conversations::user_id.eq(user_id))
                .select((conversations::updated_at, conversations::id))
                .first::<(DateTime<Utc>, i64)>(conn)
                .optional()?;

            if let Some((updated_at, id)) = cursor_conv {
                if order == "desc" {
                    // For desc order, get items with updated_at < cursor OR (updated_at = cursor AND id < cursor_id)
                    query = query.filter(
                        conversations::updated_at
                            .lt(updated_at)
                            .or(conversations::updated_at
                                .eq(updated_at)
                                .and(conversations::id.lt(id))),
                    );
                } else {
                    // For asc order, get items with updated_at > cursor OR (updated_at = cursor AND id > cursor_id)
                    query = query.filter(
                        conversations::updated_at
                            .gt(updated_at)
                            .or(conversations::updated_at
                                .eq(updated_at)
                                .and(conversations::id.gt(id))),
                    );
                }
            }
        }

        // Apply ordering
        if order == "desc" {
            query = query.order((conversations::updated_at.desc(), conversations::id.desc()));
        } else {
            query = query.order((conversations::updated_at.asc(), conversations::id.asc()));
        }

        query
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

    /// Creates a new conversation with optional response, first message, and placeholder assistant message in a transaction
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_response_and_message(
        conn: &mut PgConnection,
        conversation_uuid: Uuid,
        user_id: Uuid,
        metadata_enc: Option<Vec<u8>>,
        response: Option<NewResponse>,
        first_message_content: Vec<u8>,
        first_message_tokens: i32,
        message_uuid: Uuid,
        assistant_message_uuid: Option<Uuid>,
    ) -> Result<(Conversation, Option<Response>, UserMessage), ResponsesError> {
        use diesel::Connection;

        conn.transaction(|tx| {
            // Create the conversation
            let new_conversation = NewConversation {
                uuid: conversation_uuid,
                user_id,
                metadata_enc,
            };
            let conversation = new_conversation.insert(tx)?;

            // Create the response if provided
            let response_result = if let Some(mut new_response) = response {
                new_response.conversation_id = conversation.id;
                Some(new_response.insert(tx)?)
            } else {
                None
            };

            // Create the first message with specified UUID
            let new_message = NewUserMessage {
                uuid: message_uuid,
                conversation_id: conversation.id,
                response_id: response_result.as_ref().map(|r| r.id),
                user_id,
                content_enc: first_message_content,
                prompt_tokens: first_message_tokens,
            };
            let user_message = new_message.insert(tx)?;

            // Create placeholder assistant message if UUID provided (for Responses API streaming)
            if let Some(assistant_uuid) = assistant_message_uuid {
                let placeholder_assistant = NewAssistantMessage {
                    uuid: assistant_uuid,
                    conversation_id: conversation.id,
                    response_id: response_result.as_ref().map(|r| r.id),
                    user_id,
                    content_enc: None,
                    completion_tokens: 0,
                    status: "in_progress".to_string(),
                    finish_reason: None,
                };
                placeholder_assistant.insert(tx)?;
            }

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

    // Note: status is now tracked on Response, not UserMessage

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
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: i32,
    pub status: String,
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
    pub name: String,
    pub arguments_enc: Option<Vec<u8>>,
    pub argument_tokens: i32,
    pub status: String,
}

impl ToolCall {
    pub fn get_by_uuid(
        conn: &mut PgConnection,
        uuid: Uuid,
        user_id: Uuid,
    ) -> Result<ToolCall, ResponsesError> {
        tool_calls::table
            .filter(tool_calls::uuid.eq(uuid))
            .filter(tool_calls::user_id.eq(user_id))
            .first::<ToolCall>(conn)
            .map_err(|e| match e {
                diesel::result::Error::NotFound => ResponsesError::ToolCallNotFound,
                _ => ResponsesError::DatabaseError(e),
            })
    }
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
    pub content_enc: Option<Vec<u8>>,
    pub completion_tokens: i32,
    pub status: String,
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
    pub content_enc: Option<Vec<u8>>,
    pub completion_tokens: i32,
    pub status: String,
    pub finish_reason: Option<String>,
}

impl AssistantMessage {
    pub fn update(
        conn: &mut PgConnection,
        message_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        completion_tokens: i32,
        status: String,
        finish_reason: Option<String>,
    ) -> Result<AssistantMessage, ResponsesError> {
        diesel::update(assistant_messages::table.filter(assistant_messages::uuid.eq(message_uuid)))
            .set((
                assistant_messages::content_enc.eq(content_enc),
                assistant_messages::completion_tokens.eq(completion_tokens),
                assistant_messages::status.eq(status),
                assistant_messages::finish_reason.eq(finish_reason),
                assistant_messages::updated_at.eq(diesel::dsl::now),
            ))
            .get_result(conn)
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
// Reasoning Items
// ============================================================================

#[derive(Queryable, Selectable, Identifiable, Debug, Clone, Serialize, Deserialize)]
#[diesel(table_name = reasoning_items)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ReasoningItem {
    pub id: i64,
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Option<Vec<u8>>,
    pub summary_enc: Option<Vec<u8>>,
    pub reasoning_tokens: i32,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug)]
#[diesel(table_name = reasoning_items)]
pub struct NewReasoningItem {
    pub uuid: Uuid,
    pub conversation_id: i64,
    pub response_id: Option<i64>,
    pub user_id: Uuid,
    pub content_enc: Option<Vec<u8>>,
    pub summary_enc: Option<Vec<u8>>,
    pub reasoning_tokens: i32,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

impl ReasoningItem {
    pub fn update(
        conn: &mut PgConnection,
        item_uuid: Uuid,
        content_enc: Option<Vec<u8>>,
        reasoning_tokens: i32,
        status: String,
    ) -> Result<ReasoningItem, ResponsesError> {
        diesel::update(reasoning_items::table.filter(reasoning_items::uuid.eq(item_uuid)))
            .set((
                reasoning_items::content_enc.eq(content_enc),
                reasoning_items::reasoning_tokens.eq(reasoning_tokens),
                reasoning_items::status.eq(status),
                reasoning_items::updated_at.eq(diesel::dsl::now),
            ))
            .get_result(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

impl NewReasoningItem {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<ReasoningItem, ResponsesError> {
        diesel::insert_into(reasoning_items::table)
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
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Bytea>)]
    pub content_enc: Option<Vec<u8>>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub status: Option<String>,
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
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Text>)]
    pub tool_name: Option<String>,
}

impl RawThreadMessage {
    pub fn get_conversation_context(
        conn: &mut PgConnection,
        conversation_id: i64,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        // First, if we have an 'after' cursor, we need to find its created_at and id
        // We'll use a subquery to handle this
        let order_clause = if order == "desc" {
            "ORDER BY created_at DESC, id DESC"
        } else {
            "ORDER BY created_at ASC, id ASC"
        };

        let query = if after.is_some() {
            format!(
                r#"
                WITH conversation_messages AS (
                    -- User messages
                    SELECT
                        'user' as message_type,
                        um.id,
                        um.uuid,
                        um.content_enc,
                        'completed'::text as status,
                        um.created_at,
                        r.model,
                        um.prompt_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        NULL::text as finish_reason,
                        NULL::text as tool_name
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
                        am.status,
                        am.created_at,
                        r.model,
                        am.completion_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        am.finish_reason,
                        NULL::text as tool_name
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
                        'completed'::text as status,
                        tc.created_at,
                        NULL::text as model,
                        tc.argument_tokens as token_count,
                        tc.uuid as tool_call_id,
                        NULL::text as finish_reason,
                        tc.name as tool_name
                    FROM tool_calls tc
                    WHERE tc.conversation_id = $1

                    UNION ALL

                    -- Tool outputs
                    SELECT
                        'tool_output' as message_type,
                        tto.id,
                        tto.uuid,
                        tto.output_enc as content_enc,
                        'completed'::text as status,
                        tto.created_at,
                        NULL::text as model,
                        tto.output_tokens as token_count,
                        tc.uuid as tool_call_id,
                        NULL::text as finish_reason,
                        tc.name as tool_name
                    FROM tool_outputs tto
                    JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                    WHERE tto.conversation_id = $1

                    UNION ALL

                    -- Reasoning items
                    SELECT
                        'reasoning' as message_type,
                        ri.id,
                        ri.uuid,
                        ri.content_enc,
                        ri.status,
                        ri.created_at,
                        NULL::text as model,
                        ri.reasoning_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        NULL::text as finish_reason,
                        NULL::text as tool_name
                    FROM reasoning_items ri
                    WHERE ri.conversation_id = $1
                ),
                cursor_message AS (
                    SELECT created_at, id
                    FROM conversation_messages
                    WHERE uuid = $2
                )
                SELECT cm.*
                FROM conversation_messages cm, cursor_message
                WHERE {}
                {}
                LIMIT $3
                "#,
                if order == "desc" {
                    "(cm.created_at < cursor_message.created_at) OR (cm.created_at = cursor_message.created_at AND cm.id < cursor_message.id)"
                } else {
                    "(cm.created_at > cursor_message.created_at) OR (cm.created_at = cursor_message.created_at AND cm.id > cursor_message.id)"
                },
                order_clause
            )
        } else {
            format!(
                r#"
                WITH conversation_messages AS (
                    -- User messages
                    SELECT
                        'user' as message_type,
                        um.id,
                        um.uuid,
                        um.content_enc,
                        'completed'::text as status,
                        um.created_at,
                        r.model,
                        um.prompt_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        NULL::text as finish_reason,
                        NULL::text as tool_name
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
                        am.status,
                        am.created_at,
                        r.model,
                        am.completion_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        am.finish_reason,
                        NULL::text as tool_name
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
                        'completed'::text as status,
                        tc.created_at,
                        NULL::text as model,
                        tc.argument_tokens as token_count,
                        tc.uuid as tool_call_id,
                        NULL::text as finish_reason,
                        tc.name as tool_name
                    FROM tool_calls tc
                    WHERE tc.conversation_id = $1

                    UNION ALL

                    -- Tool outputs
                    SELECT
                        'tool_output' as message_type,
                        tto.id,
                        tto.uuid,
                        tto.output_enc as content_enc,
                        'completed'::text as status,
                        tto.created_at,
                        NULL::text as model,
                        tto.output_tokens as token_count,
                        tc.uuid as tool_call_id,
                        NULL::text as finish_reason,
                        tc.name as tool_name
                    FROM tool_outputs tto
                    JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                    WHERE tto.conversation_id = $1

                    UNION ALL

                    -- Reasoning items
                    SELECT
                        'reasoning' as message_type,
                        ri.id,
                        ri.uuid,
                        ri.content_enc,
                        ri.status,
                        ri.created_at,
                        NULL::text as model,
                        ri.reasoning_tokens as token_count,
                        NULL::uuid as tool_call_id,
                        NULL::text as finish_reason,
                        NULL::text as tool_name
                    FROM reasoning_items ri
                    WHERE ri.conversation_id = $1
                )
                SELECT *
                FROM conversation_messages
                {}
                LIMIT $2
                "#,
                order_clause
            )
        };

        if let Some(after_uuid) = after {
            sql_query(query)
                .bind::<BigInt, _>(conversation_id)
                .bind::<diesel::sql_types::Uuid, _>(after_uuid)
                .bind::<BigInt, _>(limit)
                .load::<RawThreadMessage>(conn)
                .map_err(ResponsesError::DatabaseError)
        } else {
            sql_query(query)
                .bind::<BigInt, _>(conversation_id)
                .bind::<BigInt, _>(limit)
                .load::<RawThreadMessage>(conn)
                .map_err(ResponsesError::DatabaseError)
        }
    }

    pub fn get_response_context(
        conn: &mut PgConnection,
        response_id: i64,
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        let query = r#"
            WITH response_messages AS (
                -- User messages
                SELECT
                    'user' as message_type,
                    um.id,
                    um.uuid,
                    um.content_enc,
                    'completed'::text as status,
                    um.created_at,
                    r.model,
                    um.prompt_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason,
                    NULL::text as tool_name
                FROM user_messages um
                LEFT JOIN responses r ON um.response_id = r.id
                WHERE um.response_id = $1

                UNION ALL

                -- Assistant messages
                SELECT
                    'assistant' as message_type,
                    am.id,
                    am.uuid,
                    am.content_enc,
                    am.status,
                    am.created_at,
                    r.model,
                    am.completion_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    am.finish_reason,
                    NULL::text as tool_name
                FROM assistant_messages am
                LEFT JOIN responses r ON am.response_id = r.id
                WHERE am.response_id = $1

                UNION ALL

                -- Tool calls
                SELECT
                    'tool_call' as message_type,
                    tc.id,
                    tc.uuid,
                    tc.arguments_enc as content_enc,
                    'completed'::text as status,
                    tc.created_at,
                    NULL::text as model,
                    tc.argument_tokens as token_count,
                    tc.uuid as tool_call_id,
                    NULL::text as finish_reason,
                    tc.name as tool_name
                FROM tool_calls tc
                WHERE tc.response_id = $1

                UNION ALL

                -- Tool outputs
                SELECT
                    'tool_output' as message_type,
                    tto.id,
                    tto.uuid,
                    tto.output_enc as content_enc,
                    'completed'::text as status,
                    tto.created_at,
                    NULL::text as model,
                    tto.output_tokens as token_count,
                    tc.uuid as tool_call_id,
                    NULL::text as finish_reason,
                    tc.name as tool_name
                FROM tool_outputs tto
                JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                WHERE tto.response_id = $1

                UNION ALL

                -- Reasoning items
                SELECT
                    'reasoning' as message_type,
                    ri.id,
                    ri.uuid,
                    ri.content_enc,
                    ri.status,
                    ri.created_at,
                    NULL::text as model,
                    ri.reasoning_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason,
                    NULL::text as tool_name
                FROM reasoning_items ri
                WHERE ri.response_id = $1
            )
            SELECT * FROM response_messages
            ORDER BY created_at ASC
        "#;

        sql_query(query)
            .bind::<BigInt, _>(response_id)
            .load::<RawThreadMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }

    /// Fetch messages by specific IDs (for targeted retrieval after metadata-based truncation)
    pub fn get_messages_by_ids(
        conn: &mut PgConnection,
        conversation_id: i64,
        message_ids: &[(String, i64)], // (message_type, id) pairs
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        if message_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Separate IDs by message type
        let user_ids: Vec<i64> = message_ids
            .iter()
            .filter(|(t, _)| t == "user")
            .map(|(_, id)| *id)
            .collect();
        let assistant_ids: Vec<i64> = message_ids
            .iter()
            .filter(|(t, _)| t == "assistant")
            .map(|(_, id)| *id)
            .collect();
        let tool_call_ids: Vec<i64> = message_ids
            .iter()
            .filter(|(t, _)| t == "tool_call")
            .map(|(_, id)| *id)
            .collect();
        let tool_output_ids: Vec<i64> = message_ids
            .iter()
            .filter(|(t, _)| t == "tool_output")
            .map(|(_, id)| *id)
            .collect();
        let reasoning_ids: Vec<i64> = message_ids
            .iter()
            .filter(|(t, _)| t == "reasoning")
            .map(|(_, id)| *id)
            .collect();

        // Build WHERE clause with proper parameter binding
        // Always use all 5 parameters ($2-$6) to maintain consistent type signature
        let query = r#"
            WITH conversation_messages AS (
                -- User messages
                SELECT
                    'user' as message_type,
                    um.id,
                    um.uuid,
                    um.content_enc,
                    'completed'::text as status,
                    um.created_at,
                    r.model,
                    um.prompt_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason,
                    NULL::text as tool_name
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
                    am.status,
                    am.created_at,
                    r.model,
                    am.completion_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    am.finish_reason,
                    NULL::text as tool_name
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
                    'completed'::text as status,
                    tc.created_at,
                    NULL::text as model,
                    tc.argument_tokens as token_count,
                    tc.uuid as tool_call_id,
                    NULL::text as finish_reason,
                    tc.name as tool_name
                FROM tool_calls tc
                WHERE tc.conversation_id = $1

                UNION ALL

                -- Tool outputs
                SELECT
                    'tool_output' as message_type,
                    tto.id,
                    tto.uuid,
                    tto.output_enc as content_enc,
                    'completed'::text as status,
                    tto.created_at,
                    NULL::text as model,
                    tto.output_tokens as token_count,
                    tc.uuid as tool_call_id,
                    NULL::text as finish_reason,
                    tc.name as tool_name
                FROM tool_outputs tto
                JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                WHERE tto.conversation_id = $1

                UNION ALL

                -- Reasoning items
                SELECT
                    'reasoning' as message_type,
                    ri.id,
                    ri.uuid,
                    ri.content_enc,
                    ri.status,
                    ri.created_at,
                    NULL::text as model,
                    ri.reasoning_tokens as token_count,
                    NULL::uuid as tool_call_id,
                    NULL::text as finish_reason,
                    NULL::text as tool_name
                FROM reasoning_items ri
                WHERE ri.conversation_id = $1
            )
            SELECT * FROM conversation_messages
            WHERE (message_type = 'user' AND id = ANY($2::bigint[]))
               OR (message_type = 'assistant' AND id = ANY($3::bigint[]))
               OR (message_type = 'tool_call' AND id = ANY($4::bigint[]))
               OR (message_type = 'tool_output' AND id = ANY($5::bigint[]))
               OR (message_type = 'reasoning' AND id = ANY($6::bigint[]))
            ORDER BY created_at ASC, id ASC
        "#;

        sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .bind::<Array<BigInt>, _>(&user_ids)
            .bind::<Array<BigInt>, _>(&assistant_ids)
            .bind::<Array<BigInt>, _>(&tool_call_ids)
            .bind::<Array<BigInt>, _>(&tool_output_ids)
            .bind::<Array<BigInt>, _>(&reasoning_ids)
            .load::<RawThreadMessage>(conn)
            .map_err(ResponsesError::DatabaseError)
    }
}

// ============================================================================
// Lightweight metadata struct for efficient truncation decisions
// ============================================================================

/// Metadata-only version of RawThreadMessage for efficient context building.
/// Excludes `content_enc` to avoid fetching and decrypting messages that will be truncated.
#[derive(QueryableByName, Debug, Clone)]
pub struct RawThreadMessageMetadata {
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub message_type: String,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub id: i64,
    #[diesel(sql_type = diesel::sql_types::Uuid)]
    pub uuid: Uuid,
    #[diesel(sql_type = diesel::sql_types::Timestamptz)]
    pub created_at: DateTime<Utc>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Integer>)]
    pub token_count: Option<i32>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Uuid>)]
    pub tool_call_id: Option<Uuid>,
}

impl RawThreadMessageMetadata {
    /// Fetch only message metadata (no content) for efficient truncation logic
    pub fn get_conversation_context_metadata(
        conn: &mut PgConnection,
        conversation_id: i64,
    ) -> Result<Vec<RawThreadMessageMetadata>, ResponsesError> {
        let query = r#"
            WITH conversation_messages AS (
                -- User messages
                SELECT
                    'user' as message_type,
                    um.id,
                    um.uuid,
                    um.created_at,
                    um.prompt_tokens as token_count,
                    NULL::uuid as tool_call_id
                FROM user_messages um
                WHERE um.conversation_id = $1

                UNION ALL

                -- Assistant messages
                SELECT
                    'assistant' as message_type,
                    am.id,
                    am.uuid,
                    am.created_at,
                    am.completion_tokens as token_count,
                    NULL::uuid as tool_call_id
                FROM assistant_messages am
                WHERE am.conversation_id = $1

                UNION ALL

                -- Tool calls
                SELECT
                    'tool_call' as message_type,
                    tc.id,
                    tc.uuid,
                    tc.created_at,
                    tc.argument_tokens as token_count,
                    tc.uuid as tool_call_id
                FROM tool_calls tc
                WHERE tc.conversation_id = $1

                UNION ALL

                -- Tool outputs
                SELECT
                    'tool_output' as message_type,
                    tto.id,
                    tto.uuid,
                    tto.created_at,
                    tto.output_tokens as token_count,
                    tc.uuid as tool_call_id
                FROM tool_outputs tto
                JOIN tool_calls tc ON tto.tool_call_fk = tc.id
                WHERE tto.conversation_id = $1

                UNION ALL

                -- Reasoning items
                -- TODO: Currently included for completeness but dropped in context_builder
                -- until we confirm how models expect reasoning to be passed back
                SELECT
                    'reasoning' as message_type,
                    ri.id,
                    ri.uuid,
                    ri.created_at,
                    ri.reasoning_tokens as token_count,
                    NULL::uuid as tool_call_id
                FROM reasoning_items ri
                WHERE ri.conversation_id = $1
            )
            SELECT * FROM conversation_messages
            ORDER BY created_at DESC, id DESC
            LIMIT 1000
        "#;

        let mut results = sql_query(query)
            .bind::<BigInt, _>(conversation_id)
            .load::<RawThreadMessageMetadata>(conn)
            .map_err(ResponsesError::DatabaseError)?;

        // Reverse to chronological order (oldest first) since query returns newest first
        results.reverse();

        Ok(results)
    }
}
