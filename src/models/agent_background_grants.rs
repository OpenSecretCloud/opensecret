use crate::models::schema::agent_background_grants;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum AgentBackgroundGrantError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, Clone, Debug)]
#[diesel(table_name = agent_background_grants)]
pub struct AgentBackgroundGrant {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub agent_id: i64,
    pub schedule_id: i64,
    pub grant_enc: Vec<u8>,
    pub seed_wrap_lookup_hash: Vec<u8>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentBackgroundGrant {
    pub fn get_by_id(
        conn: &mut PgConnection,
        lookup_id: i64,
    ) -> Result<Option<Self>, AgentBackgroundGrantError> {
        agent_background_grants::table
            .filter(agent_background_grants::id.eq(lookup_id))
            .first::<Self>(conn)
            .optional()
            .map_err(AgentBackgroundGrantError::DatabaseError)
    }

    pub fn get_active_by_schedule(
        conn: &mut PgConnection,
        lookup_schedule_id: i64,
    ) -> Result<Option<Self>, AgentBackgroundGrantError> {
        agent_background_grants::table
            .filter(agent_background_grants::schedule_id.eq(lookup_schedule_id))
            .filter(agent_background_grants::revoked_at.is_null())
            .first::<Self>(conn)
            .optional()
            .map_err(AgentBackgroundGrantError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = agent_background_grants)]
pub struct NewAgentBackgroundGrant {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub project_id: i32,
    pub agent_id: i64,
    pub schedule_id: i64,
    pub grant_enc: Vec<u8>,
    pub seed_wrap_lookup_hash: Vec<u8>,
}

impl NewAgentBackgroundGrant {
    pub fn insert(
        &self,
        conn: &mut PgConnection,
    ) -> Result<AgentBackgroundGrant, AgentBackgroundGrantError> {
        diesel::insert_into(agent_background_grants::table)
            .values(self)
            .get_result::<AgentBackgroundGrant>(conn)
            .map_err(AgentBackgroundGrantError::DatabaseError)
    }
}
