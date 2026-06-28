-- This file should undo anything in `up.sql`
DELETE FROM notification_deliveries;
DELETE FROM notification_events;
DELETE FROM push_devices;
DELETE FROM agent_schedule_runs;
DELETE FROM agent_background_grants;
DELETE FROM agent_schedules;

ALTER TABLE notification_events
    DROP COLUMN IF EXISTS background_grant_id,
    DROP COLUMN IF EXISTS source_request_id,
    DROP COLUMN IF EXISTS source_kind;

DROP INDEX IF EXISTS idx_push_devices_project_active;

ALTER TABLE push_devices
    ADD COLUMN push_token_enc BYTEA NOT NULL,
    ADD COLUMN notification_public_key BYTEA NOT NULL,
    DROP COLUMN IF EXISTS notification_public_key_hash,
    DROP COLUMN IF EXISTS capability_enc,
    DROP COLUMN IF EXISTS project_id;

DROP TRIGGER IF EXISTS update_agent_background_grants_updated_at ON agent_background_grants;
DROP INDEX IF EXISTS idx_agent_background_grants_user;
DROP INDEX IF EXISTS idx_agent_background_grants_schedule_active;
DROP TABLE IF EXISTS agent_background_grants;

ALTER TABLE agent_schedules
    ADD COLUMN description TEXT NOT NULL,
    DROP COLUMN IF EXISTS description_enc;

ALTER TABLE user_seed_wrappings
    DROP CONSTRAINT IF EXISTS user_seed_wrappings_credential_kind_check;

ALTER TABLE user_seed_wrappings
    ADD CONSTRAINT user_seed_wrappings_credential_kind_check
    CHECK (credential_kind IN ('password', 'oauth'));
