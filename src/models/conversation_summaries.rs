use crate::models::schema::conversation_summaries;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ConversationSummaryError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = conversation_summaries)]
pub struct ConversationSummary {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub from_created_at: DateTime<Utc>,
    pub to_created_at: DateTime<Utc>,
    pub message_count: i32,
    pub content_enc: Vec<u8>,
    pub content_tokens: i32,
    pub embedding_enc: Option<Vec<u8>>,
    pub previous_summary_id: Option<i64>,
    pub created_at: DateTime<Utc>,
}

impl ConversationSummary {
    pub fn get_latest_for_conversation(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_conversation_id: i64,
    ) -> Result<Option<ConversationSummary>, ConversationSummaryError> {
        conversation_summaries::table
            .filter(conversation_summaries::user_id.eq(lookup_user_id))
            .filter(conversation_summaries::conversation_id.eq(lookup_conversation_id))
            .order((
                conversation_summaries::created_at.desc(),
                conversation_summaries::id.desc(),
            ))
            .first::<ConversationSummary>(conn)
            .optional()
            .map_err(ConversationSummaryError::DatabaseError)
    }

    pub fn list_for_conversation(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_conversation_id: i64,
        limit: i64,
    ) -> Result<Vec<ConversationSummary>, ConversationSummaryError> {
        conversation_summaries::table
            .filter(conversation_summaries::user_id.eq(lookup_user_id))
            .filter(conversation_summaries::conversation_id.eq(lookup_conversation_id))
            .order((
                conversation_summaries::created_at.desc(),
                conversation_summaries::id.desc(),
            ))
            .limit(limit)
            .load::<ConversationSummary>(conn)
            .map_err(ConversationSummaryError::DatabaseError)
    }

    pub fn delete_all_for_conversation(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        lookup_conversation_id: i64,
    ) -> Result<usize, ConversationSummaryError> {
        diesel::delete(
            conversation_summaries::table
                .filter(conversation_summaries::user_id.eq(lookup_user_id))
                .filter(conversation_summaries::conversation_id.eq(lookup_conversation_id)),
        )
        .execute(conn)
        .map_err(ConversationSummaryError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = conversation_summaries)]
pub struct NewConversationSummary {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub from_created_at: DateTime<Utc>,
    pub to_created_at: DateTime<Utc>,
    pub message_count: i32,
    pub content_enc: Vec<u8>,
    pub content_tokens: i32,
    pub embedding_enc: Option<Vec<u8>>,
    pub previous_summary_id: Option<i64>,
}

impl NewConversationSummary {
    pub fn insert(
        &self,
        conn: &mut PgConnection,
    ) -> Result<ConversationSummary, ConversationSummaryError> {
        diesel::insert_into(conversation_summaries::table)
            .values(self)
            .get_result::<ConversationSummary>(conn)
            .map_err(ConversationSummaryError::DatabaseError)
    }
}
