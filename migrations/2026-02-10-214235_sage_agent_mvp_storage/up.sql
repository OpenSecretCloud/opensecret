-- Sage MVP agent storage tables
--
-- Note: update_updated_at_column() already exists from previous migrations.

CREATE TABLE memory_blocks (
    id          BIGSERIAL PRIMARY KEY,
    uuid        UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id     UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    label       TEXT NOT NULL,
    description TEXT,
    value_enc   BYTEA NOT NULL,
    char_limit  INTEGER NOT NULL DEFAULT 5000,
    read_only   BOOLEAN NOT NULL DEFAULT FALSE,
    version     INTEGER NOT NULL DEFAULT 1,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(user_id, label)
);

CREATE INDEX idx_memory_blocks_user_id ON memory_blocks(user_id);

CREATE TRIGGER update_memory_blocks_updated_at
BEFORE UPDATE ON memory_blocks
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE conversation_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    uuid                UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id             UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    conversation_id     BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    from_created_at     TIMESTAMPTZ NOT NULL,
    to_created_at       TIMESTAMPTZ NOT NULL,
    message_count       INTEGER NOT NULL,

    content_enc         BYTEA NOT NULL,
    content_tokens      INTEGER NOT NULL,

    embedding_enc       BYTEA,

    previous_summary_id BIGINT REFERENCES conversation_summaries(id) ON DELETE SET NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_time_range CHECK (from_created_at <= to_created_at)
);

CREATE INDEX idx_conversation_summaries_user_conv
    ON conversation_summaries(user_id, conversation_id, created_at DESC);

CREATE INDEX idx_conversation_summaries_chain
    ON conversation_summaries(previous_summary_id);

CREATE TABLE agent_config (
    id                  BIGSERIAL PRIMARY KEY,
    uuid                UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id             UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE UNIQUE,

    conversation_id     BIGINT REFERENCES conversations(id) ON DELETE SET NULL,

    enabled             BOOLEAN NOT NULL DEFAULT FALSE,
    model               TEXT NOT NULL DEFAULT 'deepseek-r1-0528',
    max_context_tokens  INTEGER NOT NULL DEFAULT 100000,
    compaction_threshold REAL NOT NULL DEFAULT 0.80,

    system_prompt_enc   BYTEA,
    preferences_enc     BYTEA,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_agent_config_updated_at
BEFORE UPDATE ON agent_config
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
