DELETE FROM agent_schedule_runs;
DELETE FROM agent_schedules;

ALTER TABLE user_seed_wrappings
    DROP CONSTRAINT IF EXISTS user_seed_wrappings_credential_kind_check;

ALTER TABLE user_seed_wrappings
    ADD CONSTRAINT user_seed_wrappings_credential_kind_check
    CHECK (credential_kind IN ('password', 'oauth', 'agent_background'));

ALTER TABLE agent_schedules
    ADD COLUMN description_enc BYTEA NOT NULL,
    DROP COLUMN description;

CREATE TABLE agent_background_grants (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES org_projects(id) ON DELETE CASCADE,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    schedule_id BIGINT NOT NULL REFERENCES agent_schedules(id) ON DELETE CASCADE,
    grant_enc BYTEA NOT NULL,
    seed_wrap_lookup_hash BYTEA NOT NULL,
    revoked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_agent_background_grants_schedule_active
    ON agent_background_grants(schedule_id)
    WHERE revoked_at IS NULL;

CREATE INDEX idx_agent_background_grants_user
    ON agent_background_grants(user_id, revoked_at);

CREATE TRIGGER update_agent_background_grants_updated_at
    BEFORE UPDATE ON agent_background_grants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE notification_events
    DROP CONSTRAINT IF EXISTS notification_events_source_kind_check;

ALTER TABLE notification_events
    ADD CONSTRAINT notification_events_source_kind_check
    CHECK (source_kind IN ('request_continuation', 'agent_background'));

ALTER TABLE notification_events
    ADD COLUMN background_grant_id BIGINT REFERENCES agent_background_grants(id) ON DELETE SET NULL;

CREATE INDEX idx_notification_events_background_grant
    ON notification_events(background_grant_id)
    WHERE background_grant_id IS NOT NULL;
