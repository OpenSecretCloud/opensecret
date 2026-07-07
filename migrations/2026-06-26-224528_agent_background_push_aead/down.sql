DELETE FROM notification_deliveries;
DELETE FROM notification_events;
DELETE FROM agent_schedule_runs;
DELETE FROM agent_background_grants;
DELETE FROM agent_schedules;

DROP INDEX IF EXISTS idx_notification_events_background_grant;

ALTER TABLE notification_events
    DROP COLUMN IF EXISTS background_grant_id;

ALTER TABLE notification_events
    DROP CONSTRAINT IF EXISTS notification_events_source_kind_check;

ALTER TABLE notification_events
    ADD CONSTRAINT notification_events_source_kind_check
    CHECK (source_kind IN ('request_continuation'));

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
