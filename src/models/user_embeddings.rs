use crate::models::schema::user_embeddings;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum UserEmbeddingError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(Queryable, Identifiable, AsChangeset, Serialize, Deserialize, Clone, Debug)]
#[diesel(table_name = user_embeddings)]
pub struct UserEmbedding {
    pub id: i64,
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub source_type: String,
    pub user_message_id: Option<i64>,
    pub assistant_message_id: Option<i64>,
    pub conversation_id: Option<i64>,
    pub vector_enc: Vec<u8>,
    pub embedding_model: String,
    pub vector_dim: i32,
    pub content_enc: Vec<u8>,
    pub metadata_enc: Option<Vec<u8>>,
    pub token_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable, Debug, Clone)]
#[diesel(table_name = user_embeddings)]
pub struct NewUserEmbedding {
    pub uuid: Uuid,
    pub user_id: Uuid,
    pub source_type: String,
    pub user_message_id: Option<i64>,
    pub assistant_message_id: Option<i64>,
    pub conversation_id: Option<i64>,
    pub vector_enc: Vec<u8>,
    pub embedding_model: String,
    pub vector_dim: i32,
    pub content_enc: Vec<u8>,
    pub metadata_enc: Option<Vec<u8>>,
    pub token_count: i32,
}

impl NewUserEmbedding {
    pub fn insert(&self, conn: &mut PgConnection) -> Result<UserEmbedding, UserEmbeddingError> {
        diesel::insert_into(user_embeddings::table)
            .values(self)
            .get_result::<UserEmbedding>(conn)
            .map_err(UserEmbeddingError::DatabaseError)
    }
}

impl UserEmbedding {
    pub fn delete_all_for_user(
        conn: &mut PgConnection,
        user_uuid: Uuid,
    ) -> Result<usize, UserEmbeddingError> {
        diesel::delete(user_embeddings::table.filter(user_embeddings::user_id.eq(user_uuid)))
            .execute(conn)
            .map_err(UserEmbeddingError::DatabaseError)
    }

    pub fn delete_by_uuid_for_user(
        conn: &mut PgConnection,
        user_uuid: Uuid,
        embedding_uuid: Uuid,
    ) -> Result<usize, UserEmbeddingError> {
        diesel::delete(
            user_embeddings::table
                .filter(user_embeddings::user_id.eq(user_uuid))
                .filter(user_embeddings::uuid.eq(embedding_uuid)),
        )
        .execute(conn)
        .map_err(UserEmbeddingError::DatabaseError)
    }
}
