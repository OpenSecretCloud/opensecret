-- ===========================================================
-- Responses API Core Schema Migration
-- ===========================================================
-- This migration creates all the tables and indexes needed for
-- the OpenAI Responses API-compatible endpoint implementation.
-- ===========================================================

-- 1. Create response_status enum for tracking response lifecycle
CREATE TYPE response_status AS ENUM
    ('queued','in_progress','completed','failed','cancelled');

-- 2. user_system_prompts table - Stores optional custom system prompts
CREATE TABLE user_system_prompts (
    id            BIGSERIAL PRIMARY KEY,
    uuid          UUID    NOT NULL UNIQUE,
    user_id       UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    name_enc      BYTEA   NOT NULL,
    prompt_enc    BYTEA   NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    is_default    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_system_prompts
CREATE INDEX idx_user_system_prompts_uuid      ON user_system_prompts(uuid);
CREATE INDEX idx_user_system_prompts_user_id   ON user_system_prompts(user_id);
CREATE UNIQUE INDEX idx_user_system_prompts_one_default
    ON user_system_prompts(user_id)
    WHERE is_default;

-- 3. conversations table - Conversation containers (OpenAI Conversations API)
CREATE TABLE conversations (
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
CREATE INDEX idx_conversations_uuid        ON conversations(uuid);
CREATE INDEX idx_conversations_user_id     ON conversations(user_id);
CREATE INDEX idx_conversations_updated     ON conversations(user_id, updated_at DESC);

-- 4. responses table - Pure job/task tracker for the Responses API
-- Note: We intentionally don't support previous_response_id - use conversations instead
CREATE TABLE responses (
    id                    BIGSERIAL PRIMARY KEY,
    uuid                  UUID           NOT NULL UNIQUE,
    user_id               UUID           NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    conversation_id       BIGINT         NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    
    status                response_status NOT NULL DEFAULT 'in_progress',
    model                 TEXT            NOT NULL,
    temperature           REAL,
    top_p                 REAL,
    max_output_tokens     INTEGER,
    tool_choice           TEXT,
    parallel_tool_calls   BOOLEAN NOT NULL DEFAULT FALSE,
    store                 BOOLEAN NOT NULL DEFAULT TRUE,
    metadata              JSONB,
    
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at          TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    idempotency_key       TEXT,
    request_hash          TEXT,
    idempotency_expires_at TIMESTAMPTZ,
    
    CONSTRAINT idempotency_fields_check CHECK (
        (idempotency_key IS NULL AND request_hash IS NULL AND idempotency_expires_at IS NULL) OR
        (idempotency_key IS NOT NULL AND request_hash IS NOT NULL AND idempotency_expires_at IS NOT NULL)
    )
);

-- Indexes for responses
CREATE INDEX idx_responses_uuid             ON responses(uuid);
CREATE INDEX idx_responses_user_id          ON responses(user_id);
CREATE INDEX idx_responses_conversation_id  ON responses(conversation_id);
CREATE INDEX idx_responses_status           ON responses(status);
CREATE INDEX idx_responses_idempotency
    ON responses(user_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- 5. user_messages table - User inputs (can be created via Conversations or Responses API)
-- response_id: NULL if created via Conversations API, populated if created via Responses API
-- prompt_tokens: Token count for just this user message (not including context)
CREATE TABLE user_messages (
    id                    BIGSERIAL PRIMARY KEY,
    uuid                  UUID           NOT NULL UNIQUE,
    conversation_id       BIGINT         NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    response_id           BIGINT         REFERENCES responses(id) ON DELETE CASCADE,
    user_id               UUID           NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content_enc           BYTEA          NOT NULL,
    prompt_tokens         INTEGER        NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_messages
CREATE INDEX idx_user_messages_uuid             ON user_messages(uuid);
CREATE INDEX idx_user_messages_conversation_id  ON user_messages(conversation_id);
CREATE INDEX idx_user_messages_response_id      ON user_messages(response_id);
CREATE INDEX idx_user_messages_user_id          ON user_messages(user_id);
CREATE INDEX idx_user_messages_conversation_created_id 
    ON user_messages(conversation_id, created_at DESC, id);
CREATE INDEX idx_user_messages_conversation_created
    ON user_messages(conversation_id, created_at);

-- 6. assistant_messages table - LLM responses (can be created via Conversations or Responses API)
-- response_id: NULL if created via Conversations API, populated if created via Responses API
CREATE TABLE assistant_messages (
    id                BIGSERIAL PRIMARY KEY,
    uuid              UUID    NOT NULL UNIQUE,
    conversation_id   BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    response_id       BIGINT  REFERENCES responses(id) ON DELETE CASCADE,
    user_id           UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content_enc       BYTEA   NOT NULL,
    completion_tokens INTEGER NOT NULL,
    finish_reason     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for assistant_messages
CREATE INDEX idx_assistant_messages_uuid             ON assistant_messages(uuid);
CREATE INDEX idx_assistant_messages_conversation_id  ON assistant_messages(conversation_id);
CREATE INDEX idx_assistant_messages_response_id      ON assistant_messages(response_id);
CREATE INDEX idx_assistant_messages_user_id          ON assistant_messages(user_id);
CREATE INDEX idx_assistant_messages_conversation_created_id
    ON assistant_messages(conversation_id, created_at DESC, id);
CREATE INDEX idx_assistant_messages_conversation_created
    ON assistant_messages(conversation_id, created_at);

-- 7. tool_calls table - Tool invocations by the model (can be created via Conversations or Responses API)
-- response_id: NULL if created via Conversations API, populated if created via Responses API
-- tool_call_id: The call_id from OpenAI
-- status: in_progress, completed, incomplete
CREATE TABLE tool_calls (
    id             BIGSERIAL PRIMARY KEY,
    uuid           UUID    NOT NULL UNIQUE,
    conversation_id BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    response_id     BIGINT  REFERENCES responses(id) ON DELETE CASCADE,
    user_id        UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    tool_call_id   UUID    NOT NULL,
    name           TEXT    NOT NULL,
    arguments_enc  BYTEA,
    argument_tokens INTEGER NOT NULL,
    status         TEXT    DEFAULT 'completed',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for tool_calls
CREATE INDEX idx_tool_calls_uuid               ON tool_calls(uuid);
CREATE INDEX idx_tool_calls_conversation_id     ON tool_calls(conversation_id);
CREATE INDEX idx_tool_calls_response_id         ON tool_calls(response_id);
CREATE INDEX idx_tool_calls_user_id            ON tool_calls(user_id);
CREATE INDEX idx_tool_calls_conversation_created_id  
    ON tool_calls(conversation_id, created_at DESC, id);
CREATE INDEX idx_tool_calls_conversation_created
    ON tool_calls(conversation_id, created_at);

-- 8. tool_outputs table - Tool execution results (can be created via Conversations or Responses API)
-- response_id: NULL if created via Conversations API, populated if created via Responses API
-- status: in_progress, completed, incomplete
CREATE TABLE tool_outputs (
    id             BIGSERIAL PRIMARY KEY,
    uuid           UUID    NOT NULL UNIQUE,
    conversation_id BIGINT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    response_id     BIGINT  REFERENCES responses(id) ON DELETE CASCADE,
    user_id        UUID    NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    tool_call_fk   BIGINT  NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    output_enc     BYTEA   NOT NULL,
    output_tokens  INTEGER NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'completed' CHECK (status IN ('in_progress','completed','incomplete')),
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for tool_outputs
CREATE INDEX idx_tool_outputs_uuid              ON tool_outputs(uuid);
CREATE INDEX idx_tool_outputs_conversation_id   ON tool_outputs(conversation_id);
CREATE INDEX idx_tool_outputs_response_id       ON tool_outputs(response_id);
CREATE INDEX idx_tool_outputs_user_id           ON tool_outputs(user_id);
CREATE INDEX idx_tool_outputs_tool_call_fk      ON tool_outputs(tool_call_fk);
CREATE INDEX idx_tool_outputs_conversation_created_id 
    ON tool_outputs(conversation_id, created_at DESC, id);
CREATE INDEX idx_tool_outputs_conversation_created
    ON tool_outputs(conversation_id, created_at);

-- 9. Create triggers for updated_at columns
-- Note: update_updated_at_column() function already exists from previous migrations

CREATE TRIGGER update_user_system_prompts_updated_at
BEFORE UPDATE ON user_system_prompts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_responses_updated_at
BEFORE UPDATE ON responses
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_messages_updated_at
BEFORE UPDATE ON user_messages
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assistant_messages_updated_at
BEFORE UPDATE ON assistant_messages
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_calls_updated_at
BEFORE UPDATE ON tool_calls
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_outputs_updated_at
BEFORE UPDATE ON tool_outputs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
