-- ===========================================================
-- Rollback Responses API Core Schema Migration
-- ===========================================================
-- Drop in reverse dependency order to avoid foreign key issues
-- ===========================================================

-- Drop triggers first
DROP TRIGGER IF EXISTS update_tool_outputs_updated_at        ON tool_outputs;
DROP TRIGGER IF EXISTS update_tool_calls_updated_at          ON tool_calls;
DROP TRIGGER IF EXISTS update_assistant_messages_updated_at  ON assistant_messages;
DROP TRIGGER IF EXISTS update_user_messages_updated_at       ON user_messages;
DROP TRIGGER IF EXISTS update_responses_updated_at           ON responses;
DROP TRIGGER IF EXISTS update_conversations_updated_at       ON conversations;
DROP TRIGGER IF EXISTS update_user_system_prompts_updated_at ON user_system_prompts;

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS tool_outputs;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS assistant_messages;
DROP TABLE IF EXISTS user_messages;
DROP TABLE IF EXISTS responses;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS user_system_prompts;

-- Drop the enum type
DROP TYPE IF EXISTS response_status;
