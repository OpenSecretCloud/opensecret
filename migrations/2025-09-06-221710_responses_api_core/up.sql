-- ===========================================================
-- Responses API Core Schema Migration
-- ===========================================================
-- This migration creates all the tables and indexes needed for
-- the OpenAI Responses API-compatible endpoint implementation.
-- ===========================================================

-- 1. Create response_status enum for tracking response lifecycle
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type WHERE typname = 'response_status'
    ) THEN
        CREATE TYPE response_status AS ENUM
          ('queued','in_progress','completed','failed','cancelled');
    END IF;
END$$;

-- 2. user_system_prompts table - Stores optional custom system prompts
CREATE TABLE IF NOT EXISTS user_system_prompts (
    id            BIGSERIAL PRIMARY KEY,
    uuid          UUID    NOT NULL UNIQUE,
    user_id       UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    name_enc      BYTEA   NOT NULL,
    prompt_enc    BYTEA   NOT NULL,
    prompt_tokens INTEGER,
    is_default    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_system_prompts
CREATE INDEX IF NOT EXISTS idx_user_system_prompts_uuid      ON user_system_prompts(uuid);
CREATE INDEX IF NOT EXISTS idx_user_system_prompts_user_id   ON user_system_prompts(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_system_prompts_one_default
    ON user_system_prompts(user_id)
    WHERE is_default;

-- 3. conversations table - Conversation containers (OpenAI Conversations API)
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGSERIAL PRIMARY KEY,
    uuid            UUID    NOT NULL UNIQUE,
    user_id         UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    system_prompt_id BIGINT REFERENCES user_system_prompts(id) ON DELETE SET NULL,
    title_enc       BYTEA,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for conversations
CREATE INDEX IF NOT EXISTS idx_conversations_uuid        ON conversations(uuid);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id     ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated     ON conversations(user_id, updated_at DESC);

-- 4. user_messages table - User inputs / Responses API requests
CREATE TABLE IF NOT EXISTS user_messages (
    id                    BIGSERIAL PRIMARY KEY,
    uuid                  UUID           NOT NULL UNIQUE,
    conversation_id       BIGINT         NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id               UUID           NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content_enc           BYTEA          NOT NULL,
    prompt_tokens         INTEGER,

    -- Responses API fields
    status                response_status NOT NULL DEFAULT 'in_progress',
    model                 TEXT            NOT NULL,
    previous_response_id  UUID            REFERENCES user_messages(uuid),
    temperature           REAL,
    top_p                 REAL,
    max_output_tokens     INTEGER,
    tool_choice           TEXT,
    parallel_tool_calls   BOOLEAN NOT NULL DEFAULT FALSE,
    store                 BOOLEAN NOT NULL DEFAULT TRUE,
    metadata              JSONB,
    error                 TEXT,

    -- Timestamps
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at          TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Idempotency fields
    idempotency_key       TEXT,
    request_hash          TEXT,
    idempotency_expires_at TIMESTAMPTZ,
    
    -- Ensure idempotency fields are set together
    CONSTRAINT idempotency_fields_check CHECK (
        (idempotency_key IS NULL AND request_hash IS NULL AND idempotency_expires_at IS NULL) OR
        (idempotency_key IS NOT NULL AND request_hash IS NOT NULL AND idempotency_expires_at IS NOT NULL)
    )
);

-- Indexes for user_messages
CREATE INDEX IF NOT EXISTS idx_user_messages_uuid             ON user_messages(uuid);
CREATE INDEX IF NOT EXISTS idx_user_messages_conversation_id  ON user_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_user_messages_user_id          ON user_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_user_messages_status           ON user_messages(status);
CREATE INDEX IF NOT EXISTS idx_user_messages_previous_uuid    ON user_messages(previous_response_id);
CREATE INDEX IF NOT EXISTS idx_user_messages_conversation_created_id 
    ON user_messages(conversation_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_user_messages_conversation_created
    ON user_messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_user_messages_idempotency
    ON user_messages(user_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- 5. tool_calls table - Tool invocations by the model
CREATE TABLE IF NOT EXISTS tool_calls (
    id             BIGSERIAL PRIMARY KEY,
    uuid           UUID    NOT NULL UNIQUE,
    conversation_id BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_message_id BIGINT NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    tool_call_id   UUID    NOT NULL,
    name           TEXT    NOT NULL,
    arguments_enc  BYTEA,
    argument_tokens INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for tool_calls
CREATE INDEX IF NOT EXISTS idx_tool_calls_uuid               ON tool_calls(uuid);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_id     ON tool_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_user_message_id    ON tool_calls(user_message_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_created_id  
    ON tool_calls(conversation_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_created
    ON tool_calls(conversation_id, created_at);

-- 6. tool_outputs table - Tool execution results
CREATE TABLE IF NOT EXISTS tool_outputs (
    id             BIGSERIAL PRIMARY KEY,
    uuid           UUID    NOT NULL UNIQUE,
    conversation_id BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tool_call_fk   BIGINT  NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    output_enc     BYTEA   NOT NULL,
    output_tokens  INTEGER,
    status         TEXT    NOT NULL DEFAULT 'succeeded' CHECK (status IN ('succeeded','failed')),
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for tool_outputs
CREATE INDEX IF NOT EXISTS idx_tool_outputs_uuid              ON tool_outputs(uuid);
CREATE INDEX IF NOT EXISTS idx_tool_outputs_conversation_id   ON tool_outputs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_outputs_tool_call_fk      ON tool_outputs(tool_call_fk);
CREATE INDEX IF NOT EXISTS idx_tool_outputs_conversation_created_id 
    ON tool_outputs(conversation_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_tool_outputs_conversation_created
    ON tool_outputs(conversation_id, created_at);

-- 7. assistant_messages table - LLM responses
CREATE TABLE IF NOT EXISTS assistant_messages (
    id                BIGSERIAL PRIMARY KEY,
    uuid              UUID    NOT NULL UNIQUE,
    conversation_id   BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_message_id   BIGINT  NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    content_enc       BYTEA   NOT NULL,
    completion_tokens INTEGER,
    finish_reason     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for assistant_messages
CREATE INDEX IF NOT EXISTS idx_assistant_messages_uuid             ON assistant_messages(uuid);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation_id  ON assistant_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_user_message_id  ON assistant_messages(user_message_id);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation_created_id
    ON assistant_messages(conversation_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation_created
    ON assistant_messages(conversation_id, created_at);

-- 8. Create triggers for updated_at columns
-- Note: update_updated_at_column() function already exists from previous migrations

CREATE TRIGGER update_user_system_prompts_updated_at
BEFORE UPDATE ON user_system_prompts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_messages_updated_at
BEFORE UPDATE ON user_messages
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_calls_updated_at
BEFORE UPDATE ON tool_calls
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_outputs_updated_at
BEFORE UPDATE ON tool_outputs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assistant_messages_updated_at
BEFORE UPDATE ON assistant_messages
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
