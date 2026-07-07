-- Hot RAG retrieval paths: active embedding model for a user, newest first.
CREATE INDEX idx_user_embeddings_user_model_created_id
    ON user_embeddings(user_id, embedding_model, created_at DESC, id DESC);

CREATE INDEX idx_user_embeddings_user_model_source_created_id
    ON user_embeddings(user_id, embedding_model, source_type, created_at DESC, id DESC);

CREATE INDEX idx_user_embeddings_user_model_conversation_created_id
    ON user_embeddings(user_id, embedding_model, conversation_id, created_at DESC, id DESC);
