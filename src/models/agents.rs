use crate::models::schema::agents;
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
