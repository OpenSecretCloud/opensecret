-- Removing service-side pairing and unacknowledged revocation authority can
-- strand a host's durable local allowlist. This guard must remain the first
-- executable statement. Production rollback requires a coordinated forward
-- migration; only the disposable migration-redo database may run this file.
DO $$
BEGIN
    IF current_setting(
        'opensecret.allow_destructive_maple_pairing_down',
        true
    ) IS DISTINCT FROM 'disposable-test-only' THEN
        RAISE EXCEPTION
            'maple pairing rollback is destructive and is permitted only in the disposable test database';
    END IF;
END
$$;

-- Remove every guard from tables that survive this rollback before touching
-- the authenticated marker or restoring the pre-authority foreign keys.
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_marker_commit
    ON app_data_migrations;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_marker_mutation
    ON app_data_migrations;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_marker_truncate
    ON app_data_migrations;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_user_parent_commit ON users;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_project_parent_commit
    ON org_projects;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_org_parent_commit ON orgs;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_user_truncate ON users;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_project_truncate ON org_projects;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_org_truncate ON orgs;
DROP TRIGGER IF EXISTS guard_maple_devices_truncate ON maple_devices;
DROP TRIGGER IF EXISTS guard_maple_device_operations_truncate
    ON maple_device_registration_operations;

-- Drop triggers on migration-owned tables explicitly so no trigger retains a
-- function dependency while the rollback dismantles the hierarchy.
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_global_commit
    ON maple_pairing_authority_global_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_global_mutation
    ON maple_pairing_authority_global_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_global_truncate
    ON maple_pairing_authority_global_heads;
DROP TRIGGER IF EXISTS update_maple_pairing_authority_global_heads_updated_at
    ON maple_pairing_authority_global_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_issuer_keys_truncate
    ON maple_pairing_issuer_keys;
DROP TRIGGER IF EXISTS guard_maple_pairing_issuer_key_commit
    ON maple_pairing_issuer_keys;
DROP TRIGGER IF EXISTS guard_maple_pairing_issuer_key_mutation
    ON maple_pairing_issuer_keys;

DROP TRIGGER IF EXISTS guard_maple_pairing_authority_org_head_commit
    ON maple_pairing_authority_org_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_org_head_mutation
    ON maple_pairing_authority_org_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_org_head_truncate
    ON maple_pairing_authority_org_heads;
DROP TRIGGER IF EXISTS update_maple_pairing_authority_org_heads_updated_at
    ON maple_pairing_authority_org_heads;

DROP TRIGGER IF EXISTS guard_maple_pairing_authority_project_head_commit
    ON maple_pairing_authority_project_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_project_head_mutation
    ON maple_pairing_authority_project_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_project_head_truncate
    ON maple_pairing_authority_project_heads;
DROP TRIGGER IF EXISTS update_maple_pairing_authority_project_heads_updated_at
    ON maple_pairing_authority_project_heads;

DROP TRIGGER IF EXISTS guard_maple_pairing_authority_account_head_commit
    ON maple_pairing_authority_account_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_account_head_mutation
    ON maple_pairing_authority_account_heads;
DROP TRIGGER IF EXISTS guard_maple_pairing_authority_account_head_truncate
    ON maple_pairing_authority_account_heads;
DROP TRIGGER IF EXISTS update_maple_pairing_authority_account_heads_updated_at
    ON maple_pairing_authority_account_heads;

DROP TRIGGER IF EXISTS guard_maple_pairing_events_truncate
    ON maple_pairing_revocation_events;
DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_admissions_truncate
    ON maple_pairing_reset_clear_admissions;
DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_admission_mutation
    ON maple_pairing_reset_clear_admissions;
DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_obligations_truncate
    ON maple_pairing_reset_clear_obligations;
DROP TRIGGER IF EXISTS guard_maple_pairing_reset_clear_obligation_mutation
    ON maple_pairing_reset_clear_obligations;
DROP TRIGGER IF EXISTS update_maple_pairing_reset_clear_obligations_updated_at
    ON maple_pairing_reset_clear_obligations;
DROP TRIGGER IF EXISTS guard_maple_pairing_registration_tombstones_truncate
    ON maple_pairing_registration_operation_tombstones;
DROP TRIGGER IF EXISTS guard_maple_pairing_registration_tombstone_mutation
    ON maple_pairing_registration_operation_tombstones;
DROP TRIGGER IF EXISTS guard_maple_pairing_installation_retirements_truncate
    ON maple_pairing_installation_retirements;
DROP TRIGGER IF EXISTS guard_maple_pairing_installation_retirement_mutation
    ON maple_pairing_installation_retirements;
DROP TRIGGER IF EXISTS guard_maple_pairing_highwaters_truncate
    ON maple_pairing_revocation_highwaters;
DROP TRIGGER IF EXISTS update_maple_pairing_revocation_highwaters_updated_at
    ON maple_pairing_revocation_highwaters;
DROP TRIGGER IF EXISTS guard_maple_pairing_host_states_truncate
    ON maple_pairing_host_states;
DROP TRIGGER IF EXISTS update_maple_pairing_host_states_updated_at
    ON maple_pairing_host_states;
DROP TRIGGER IF EXISTS guard_maple_pairing_operations_truncate
    ON maple_pairing_operations;
DROP TRIGGER IF EXISTS guard_maple_pairings_truncate ON maple_pairings;
DROP TRIGGER IF EXISTS update_maple_pairings_updated_at ON maple_pairings;
DROP TRIGGER IF EXISTS guard_maple_pairing_lineages_truncate
    ON maple_pairing_lineages;
DROP TRIGGER IF EXISTS update_maple_pairing_lineages_updated_at
    ON maple_pairing_lineages;

DROP FUNCTION IF EXISTS enforce_maple_pairing_authority_hierarchy_commit();
DROP FUNCTION IF EXISTS enforce_maple_pairing_authority_marker_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_authority_scoped_head_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_authority_global_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_issuer_key_mutation();
DROP FUNCTION IF EXISTS forbid_maple_pairing_authority_truncate();
DROP FUNCTION IF EXISTS enforce_maple_pairing_reset_clear_obligation_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_reset_clear_admission_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_registration_tombstone_mutation();
DROP FUNCTION IF EXISTS enforce_maple_pairing_installation_retirement_mutation();

DELETE FROM app_data_migrations
WHERE name = 'maple_pairing_authority_v1_activated';

DROP INDEX IF EXISTS idx_maple_pairing_revocations_project_id;
DROP INDEX IF EXISTS idx_maple_pairing_revocations_host_sequence;
DROP TABLE IF EXISTS maple_pairing_revocation_events;

DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_admissions_canonical;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_admissions_scope;
DROP TABLE IF EXISTS maple_pairing_reset_clear_admissions;

DROP INDEX IF EXISTS idx_maple_pairing_installation_retirements_identity;
DROP INDEX IF EXISTS idx_maple_pairing_installation_retirements_lookup;
DROP INDEX IF EXISTS idx_maple_pairing_installation_retirements_scope;
DROP TABLE IF EXISTS maple_pairing_installation_retirements;

DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_obligations_acked_by;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_obligations_identity;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_obligations_current;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_obligations_scope;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_ack_operation;
DROP INDEX IF EXISTS idx_maple_pairing_reset_clear_no_forks;
DROP TABLE IF EXISTS maple_pairing_reset_clear_obligations;

DROP INDEX IF EXISTS idx_maple_pairing_revocation_highwaters_authority_scope;
DROP INDEX IF EXISTS idx_maple_pairing_revocation_highwaters_lookup;
DROP TABLE IF EXISTS maple_pairing_revocation_highwaters;

DROP INDEX IF EXISTS idx_maple_pairing_host_states_project_id;
DROP TABLE IF EXISTS maple_pairing_host_states;

DROP INDEX IF EXISTS idx_maple_pairing_operations_project_id;
DROP INDEX IF EXISTS idx_maple_pairing_operations_pairing;
DROP TABLE IF EXISTS maple_pairing_operations;

DROP INDEX IF EXISTS idx_maple_pairings_project_id;
DROP INDEX IF EXISTS idx_maple_pairings_controller_state_uuid;
DROP INDEX IF EXISTS idx_maple_pairings_host_state_uuid;
DROP INDEX IF EXISTS idx_maple_pairings_one_live_lineage;
DROP TABLE IF EXISTS maple_pairings;

DROP INDEX IF EXISTS idx_maple_pairing_lineages_host;
DROP INDEX IF EXISTS idx_maple_pairing_lineages_controller;
DROP INDEX IF EXISTS idx_maple_pairing_lineages_project_id;
DROP TABLE IF EXISTS maple_pairing_lineages;

DROP INDEX IF EXISTS idx_maple_pairing_registration_operation_tombstones_lookup;
DROP INDEX IF EXISTS idx_maple_pairing_registration_operation_tombstones_scope;
DROP TABLE IF EXISTS maple_pairing_registration_operation_tombstones;
DROP FUNCTION IF EXISTS maple_pairing_issuer_key_ids_are_canonical(TEXT[], INTEGER);

-- The device tables belong to the preceding migration and survive this down.
-- Restore precisely the FK names and CASCADE actions from that migration.
ALTER TABLE maple_device_registration_operations
    DROP CONSTRAINT maple_device_registration_operations_authority_account_fk,
    DROP CONSTRAINT maple_device_registration_operations_authority_scope_fk,
    DROP CONSTRAINT maple_device_registration_operations_user_scope_fk,
    DROP CONSTRAINT maple_device_registration_operations_scoped_device_fk,
    DROP CONSTRAINT maple_device_registration_operations_authority_scope_length,
    DROP CONSTRAINT maple_device_registration_operations_lookup_length,
    DROP CONSTRAINT maple_device_registration_operations_operation_lookup_length,
    DROP CONSTRAINT maple_device_registration_operations_security_epoch_shape,
    DROP CONSTRAINT maple_device_registration_operations_response_kind_v1,
    DROP CONSTRAINT maple_device_registration_operations_sync_shape,
    DROP CONSTRAINT maple_device_registration_operations_operation_lookup_unique;
ALTER TABLE maple_devices
    DROP CONSTRAINT maple_devices_authority_account_fk,
    DROP CONSTRAINT maple_devices_user_scope_fk;

ALTER TABLE maple_devices
    ADD CONSTRAINT maple_devices_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE,
    ADD CONSTRAINT maple_devices_project_id_fkey
        FOREIGN KEY (project_id) REFERENCES org_projects(id) ON DELETE CASCADE;
ALTER TABLE maple_device_registration_operations
    DROP COLUMN authority_scope_digest,
    DROP COLUMN lookup_digest,
    DROP COLUMN operation_lookup_digest,
    DROP COLUMN known_security_epoch,
    DROP COLUMN accepted_security_epoch,
    DROP COLUMN response_kind,
    DROP COLUMN sync_payload_version,
    DROP COLUMN sync_payload_enc,
    DROP COLUMN sync_issuer_key_id,
    DROP COLUMN sync_digest,
    ADD CONSTRAINT maple_device_registration_operations_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE,
    ADD CONSTRAINT maple_device_registration_operations_project_id_fkey
        FOREIGN KEY (project_id) REFERENCES org_projects(id) ON DELETE CASCADE,
    ADD CONSTRAINT maple_device_registration_operations_scoped_device_fk
        FOREIGN KEY (maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE CASCADE;

DROP INDEX IF EXISTS idx_maple_pairing_authority_account_heads_org;
DROP INDEX IF EXISTS idx_maple_pairing_authority_account_heads_project;
DROP TABLE IF EXISTS maple_pairing_authority_account_heads;

DROP INDEX IF EXISTS idx_maple_pairing_authority_project_heads_org;
DROP TABLE IF EXISTS maple_pairing_authority_project_heads;

DROP TABLE IF EXISTS maple_pairing_authority_org_heads;
DROP TABLE IF EXISTS maple_pairing_issuer_keys;
DROP TABLE IF EXISTS maple_pairing_authority_global_heads;

ALTER TABLE org_projects
    DROP CONSTRAINT IF EXISTS maple_pairing_authority_projects_identity_unique,
    DROP CONSTRAINT IF EXISTS maple_pairing_authority_projects_scope_unique;
ALTER TABLE users
    DROP CONSTRAINT IF EXISTS maple_pairing_authority_users_scope_unique;

-- Deliberately preserve maple_pairing_incarnation_seq. Pairing incarnations
-- are security fence values and must not be reused after a rollback/down-up
-- cycle. A disposable database drops the sequence with the database itself.
