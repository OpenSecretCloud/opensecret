use crate::models::schema::agent_config;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum AgentConfigError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = agent_config)]
pub struct AgentConfig {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: Option<i64>,
    pub enabled: bool,
    pub model: String,
    pub max_context_tokens: i32,
    pub compaction_threshold: f32,
    pub system_prompt_enc: Option<Vec<u8>>,
    pub preferences_enc: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentConfig {
    pub fn get_by_user_id(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<Option<AgentConfig>, AgentConfigError> {
        agent_config::table
            .filter(agent_config::user_id.eq(lookup_user_id))
            .first::<AgentConfig>(conn)
            .optional()
            .map_err(AgentConfigError::DatabaseError)
    }

    pub fn delete_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
    ) -> Result<usize, AgentConfigError> {
        diesel::delete(agent_config::table.filter(agent_config::user_id.eq(lookup_user_id)))
            .execute(conn)
            .map_err(AgentConfigError::DatabaseError)
    }
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = agent_config)]
pub struct NewAgentConfig {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub conversation_id: Option<i64>,
    pub enabled: bool,
    pub model: String,
    pub max_context_tokens: i32,
    pub compaction_threshold: f32,
    pub system_prompt_enc: Option<Vec<u8>>,
    pub preferences_enc: Option<Vec<u8>>,
}

impl NewAgentConfig {
    pub fn insert_or_update(
        &self,
        conn: &mut PgConnection,
    ) -> Result<AgentConfig, AgentConfigError> {
        diesel::insert_into(agent_config::table)
            .values(self)
            .on_conflict(agent_config::user_id)
            .do_update()
            .set((
                agent_config::conversation_id.eq(self.conversation_id),
                agent_config::enabled.eq(self.enabled),
                agent_config::model.eq(self.model.clone()),
                agent_config::max_context_tokens.eq(self.max_context_tokens),
                agent_config::compaction_threshold.eq(self.compaction_threshold),
                agent_config::system_prompt_enc.eq(self.system_prompt_enc.clone()),
                agent_config::preferences_enc.eq(self.preferences_enc.clone()),
            ))
            .get_result::<AgentConfig>(conn)
            .map_err(AgentConfigError::DatabaseError)
    }
}
