use crate::models::responses::Conversation;
use crate::models::schema::agents;
use crate::models::schema::conversations;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const AGENT_KIND_MAIN: &str = "main";
pub const AGENT_KIND_SUBAGENT: &str = "subagent";
pub const AGENT_CREATED_BY_USER: &str = "user";
pub const AGENT_CREATED_BY_AGENT: &str = "agent";

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = agents)]
pub struct Agent {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub kind: String,
    pub parent_agent_id: Option<i64>,
    pub display_name_enc: Option<Vec<u8>>,
    pub purpose_enc: Option<Vec<u8>>,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Agent {
    pub fn get_main_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Option<Agent>, AgentError> {
        agents::table
            .filter(agents::user_id.eq(lookup_user_id))
            .filter(agents::kind.eq(AGENT_KIND_MAIN))
            .first::<Agent>(conn)
            .optional()
            .map_err(AgentError::DatabaseError)
    }

    pub fn get_by_uuid_and_user(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
        lookup_user_id: Uuid,
    ) -> Result<Option<Agent>, AgentError> {
        agents::table
            .filter(agents::uuid.eq(lookup_uuid))
            .filter(agents::user_id.eq(lookup_user_id))
            .first::<Agent>(conn)
            .optional()
            .map_err(AgentError::DatabaseError)
    }

    pub fn get_by_conversation_id_and_user(
        conn: &mut PgConnection,
        lookup_conversation_id: i64,
        lookup_user_id: Uuid,
    ) -> Result<Option<Agent>, AgentError> {
        agents::table
            .filter(agents::conversation_id.eq(lookup_conversation_id))
            .filter(agents::user_id.eq(lookup_user_id))
            .first::<Agent>(conn)
            .optional()
            .map_err(AgentError::DatabaseError)
    }

    pub fn list_subagents_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Vec<Agent>, AgentError> {
        agents::table
            .filter(agents::user_id.eq(lookup_user_id))
            .filter(agents::kind.eq(AGENT_KIND_SUBAGENT))
            .order((agents::created_at.desc(), agents::id.desc()))
            .load::<Agent>(conn)
            .map_err(AgentError::DatabaseError)
    }

    pub fn list_subagents_for_user_paginated(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        limit: i64,
        after: Option<Uuid>,
        order: &str,
        created_by_filter: Option<&str>,
    ) -> Result<Vec<(Agent, Conversation)>, AgentError> {
        let mut query = agents::table
            .inner_join(conversations::table)
            .filter(agents::user_id.eq(lookup_user_id))
            .filter(agents::kind.eq(AGENT_KIND_SUBAGENT))
            .into_boxed();

        if let Some(created_by_filter) = created_by_filter {
            query = query.filter(agents::created_by.eq(created_by_filter));
        }

        if let Some(after_uuid) = after {
            let mut cursor_query = agents::table
                .inner_join(conversations::table)
                .filter(agents::user_id.eq(lookup_user_id))
                .filter(agents::kind.eq(AGENT_KIND_SUBAGENT))
                .filter(agents::uuid.eq(after_uuid))
                .into_boxed();

            if let Some(created_by_filter) = created_by_filter {
                cursor_query = cursor_query.filter(agents::created_by.eq(created_by_filter));
            }

            let cursor_subagent = cursor_query
                .select((conversations::updated_at, agents::id))
                .first::<(DateTime<Utc>, i64)>(conn)
                .optional()
                .map_err(AgentError::DatabaseError)?;

            if let Some((updated_at, id)) = cursor_subagent {
                if order == "desc" {
                    query = query.filter(
                        conversations::updated_at
                            .lt(updated_at)
                            .or(conversations::updated_at
                                .eq(updated_at)
                                .and(agents::id.lt(id))),
                    );
                } else {
                    query = query.filter(
                        conversations::updated_at
                            .gt(updated_at)
                            .or(conversations::updated_at
                                .eq(updated_at)
                                .and(agents::id.gt(id))),
                    );
                }
            }
        }

        if order == "desc" {
            query = query.order((conversations::updated_at.desc(), agents::id.desc()));
        } else {
            query = query.order((conversations::updated_at.asc(), agents::id.asc()));
        }

        query
            .limit(limit)
            .load::<(Agent, Conversation)>(conn)
            .map_err(AgentError::DatabaseError)
    }

    pub fn delete_by_uuid_and_user(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
        lookup_user_id: Uuid,
    ) -> Result<usize, AgentError> {
        diesel::delete(
            agents::table
                .filter(agents::uuid.eq(lookup_uuid))
                .filter(agents::user_id.eq(lookup_user_id)),
        )
        .execute(conn)
        .map_err(AgentError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = agents)]
pub struct NewAgent {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: i64,
    pub kind: String,
    pub parent_agent_id: Option<i64>,
    pub display_name_enc: Option<Vec<u8>>,
    pub purpose_enc: Option<Vec<u8>>,
    pub created_by: String,
}

impl NewAgent {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<Agent, AgentError> {
        diesel::insert_into(agents::table)
            .values(self)
            .get_result::<Agent>(conn)
            .map_err(AgentError::DatabaseError)
    }
}
