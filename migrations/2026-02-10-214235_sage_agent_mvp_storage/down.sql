DROP TRIGGER IF EXISTS update_agent_config_updated_at ON agent_config;
DROP TABLE IF EXISTS agent_config;

DROP INDEX IF EXISTS idx_conversation_summaries_chain;
DROP INDEX IF EXISTS idx_conversation_summaries_user_conv;
DROP TABLE IF EXISTS conversation_summaries;

DROP TRIGGER IF EXISTS update_memory_blocks_updated_at ON memory_blocks;
DROP INDEX IF EXISTS idx_memory_blocks_user_id;
DROP TABLE IF EXISTS memory_blocks;
