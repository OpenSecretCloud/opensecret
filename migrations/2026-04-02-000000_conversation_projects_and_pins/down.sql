-- Reverse conversation projects and pinned chats migration

DROP TRIGGER IF EXISTS touch_conversation_updated_at_from_reasoning_items ON reasoning_items;
DROP TRIGGER IF EXISTS touch_conversation_updated_at_from_tool_outputs ON tool_outputs;
DROP TRIGGER IF EXISTS touch_conversation_updated_at_from_tool_calls ON tool_calls;
DROP TRIGGER IF EXISTS touch_conversation_updated_at_from_assistant_messages ON assistant_messages;
DROP TRIGGER IF EXISTS touch_conversation_updated_at_from_user_messages ON user_messages;
DROP FUNCTION IF EXISTS touch_conversation_updated_at_from_child();

DROP INDEX IF EXISTS idx_user_instructions_project_id_unique;
ALTER TABLE user_instructions
    DROP CONSTRAINT IF EXISTS chk_user_instructions_project_not_default,
    DROP CONSTRAINT IF EXISTS chk_user_instructions_global_name_required;
DELETE FROM user_instructions WHERE project_id IS NOT NULL;
ALTER TABLE user_instructions DROP COLUMN IF EXISTS project_id;
ALTER TABLE user_instructions ALTER COLUMN name_enc SET NOT NULL;

DROP INDEX IF EXISTS idx_conversations_pinned_updated_id;
DROP INDEX IF EXISTS idx_conversations_project_updated_id;
ALTER TABLE conversations DROP COLUMN IF EXISTS is_pinned;
ALTER TABLE conversations DROP COLUMN IF EXISTS project_id;

DROP TRIGGER IF EXISTS update_conversation_projects_updated_at ON conversation_projects;
DROP INDEX IF EXISTS idx_conversation_projects_updated_id;
DROP TABLE IF EXISTS conversation_projects;
