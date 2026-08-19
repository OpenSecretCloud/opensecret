-- Incarnations are capability fence values. They must never be reused, even
-- after a transaction rollback, an exact-operation retry, or a migration
-- down/up cycle. Gaps are consequently intentional.
CREATE SEQUENCE IF NOT EXISTS maple_pairing_incarnation_seq
    AS BIGINT
    MINVALUE 1
    NO CYCLE;

-- These composite keys are the relational scope anchors for the authenticated
-- authority hierarchy.  Independent foreign keys would permit a syntactically
-- valid head to splice a user/project into the wrong parent scope.
ALTER TABLE users
    ADD CONSTRAINT maple_pairing_authority_users_scope_unique
    UNIQUE (uuid, project_id);
ALTER TABLE org_projects
    ADD CONSTRAINT maple_pairing_authority_projects_scope_unique
    UNIQUE (id, org_id),
    ADD CONSTRAINT maple_pairing_authority_projects_identity_unique
    UNIQUE (id, org_id, uuid, client_id);

-- The singleton is inserted Pending by SQL because migrations do not possess
-- the enclave root key. Application startup may authenticate and activate it
-- exactly once, and only while every Maple authority table is empty. Once
-- Active, a missing, downgraded, or invalid-MAC root is corruption and must
-- never be recreated lazily.
CREATE TABLE maple_pairing_authority_global_heads (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    activation_state SMALLINT NOT NULL DEFAULT 1,
    org_inventory_digest BYTEA NOT NULL,
    org_count BIGINT NOT NULL DEFAULT 0,
    issuer_key_inventory_digest BYTEA NOT NULL,
    issuer_key_count BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_authority_global_state_v1
        CHECK (activation_state IN (1, 2)),
    CONSTRAINT maple_pairing_authority_global_digest_length
        CHECK (
            octet_length(org_inventory_digest) = 32
            AND octet_length(issuer_key_inventory_digest) = 32
        ),
    CONSTRAINT maple_pairing_authority_global_count_nonnegative CHECK (org_count >= 0),
    CONSTRAINT maple_pairing_authority_global_issuer_key_count_bounded
        CHECK (issuer_key_count BETWEEN 0 AND 1024),
    CONSTRAINT maple_pairing_authority_global_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairing_authority_global_activation_shape CHECK (
        (
            activation_state = 1
            AND org_inventory_digest = decode(repeat('00', 32), 'hex')
            AND org_count = 0
            AND issuer_key_inventory_digest = decode(repeat('00', 32), 'hex')
            AND issuer_key_count = 0
            AND revision = 1
            AND record_mac IS NULL
        )
        OR (
            activation_state = 2
            AND revision >= 2
            AND record_mac IS NOT NULL
            AND octet_length(record_mac) = 32
        )
    )
);

INSERT INTO maple_pairing_authority_global_heads (
    singleton,
    activation_state,
    org_inventory_digest,
    org_count,
    issuer_key_inventory_digest,
    issuer_key_count,
    revision,
    record_mac
) VALUES (
    TRUE,
    1,
    decode(repeat('00', 32), 'hex'),
    0,
    decode(repeat('00', 32), 'hex'),
    0,
    1,
    NULL
);

CREATE TRIGGER update_maple_pairing_authority_global_heads_updated_at
    BEFORE UPDATE ON maple_pairing_authority_global_heads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- An issuer key ID is a lifetime identity for exactly one algorithm/public-key
-- fingerprint. The enclave-authenticated row and global inventory make remaps,
-- deletions, and privileged storage edits fail closed across process restarts.
CREATE TABLE maple_pairing_issuer_keys (
    key_id TEXT COLLATE "C" PRIMARY KEY,
    global_singleton BOOLEAN NOT NULL DEFAULT TRUE CHECK (global_singleton),
    algorithm TEXT NOT NULL,
    public_key_digest BYTEA NOT NULL,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_issuer_keys_key_id_v1
        CHECK (key_id ~ '^[a-z0-9._:-]{1,64}$'),
    CONSTRAINT maple_pairing_issuer_keys_algorithm_v1 CHECK (algorithm = 'ed25519'),
    CONSTRAINT maple_pairing_issuer_keys_digest_lengths CHECK (
        octet_length(public_key_digest) = 32
        AND octet_length(record_mac) = 32
    ),
    CONSTRAINT maple_pairing_issuer_keys_public_key_unique
        UNIQUE (algorithm, public_key_digest),
    CONSTRAINT maple_pairing_issuer_keys_global_fk
        FOREIGN KEY (global_singleton)
        REFERENCES maple_pairing_authority_global_heads(singleton)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE maple_pairing_authority_org_heads (
    org_id INTEGER PRIMARY KEY,
    global_singleton BOOLEAN NOT NULL DEFAULT TRUE CHECK (global_singleton),
    project_inventory_digest BYTEA NOT NULL,
    project_count BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_authority_org_digest_length
        CHECK (octet_length(project_inventory_digest) = 32),
    CONSTRAINT maple_pairing_authority_org_count_nonnegative CHECK (project_count >= 0),
    CONSTRAINT maple_pairing_authority_org_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairing_authority_org_mac_length CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_authority_org_scope_fk
        FOREIGN KEY (org_id)
        REFERENCES orgs(id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_authority_org_global_fk
        FOREIGN KEY (global_singleton)
        REFERENCES maple_pairing_authority_global_heads(singleton)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE TRIGGER update_maple_pairing_authority_org_heads_updated_at
    BEFORE UPDATE ON maple_pairing_authority_org_heads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE maple_pairing_authority_project_heads (
    project_id INTEGER PRIMARY KEY,
    org_id INTEGER NOT NULL,
    project_uuid UUID NOT NULL,
    subject_project_id UUID NOT NULL,
    account_inventory_digest BYTEA NOT NULL,
    account_count BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_authority_project_digest_length
        CHECK (octet_length(account_inventory_digest) = 32),
    CONSTRAINT maple_pairing_authority_project_count_nonnegative CHECK (account_count >= 0),
    CONSTRAINT maple_pairing_authority_project_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairing_authority_project_mac_length CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_authority_project_scope_unique UNIQUE (project_id, org_id),
    CONSTRAINT maple_pairing_authority_project_identity_unique
        UNIQUE (project_id, org_id, project_uuid, subject_project_id),
    CONSTRAINT maple_pairing_authority_project_scope_fk
        FOREIGN KEY (project_id, org_id, project_uuid, subject_project_id)
        REFERENCES org_projects(id, org_id, uuid, client_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_authority_project_org_head_fk
        FOREIGN KEY (org_id)
        REFERENCES maple_pairing_authority_org_heads(org_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_authority_project_heads_org
    ON maple_pairing_authority_project_heads(org_id, project_id);
CREATE TRIGGER update_maple_pairing_authority_project_heads_updated_at
    BEFORE UPDATE ON maple_pairing_authority_project_heads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE maple_pairing_authority_account_heads (
    user_id UUID PRIMARY KEY,
    project_id INTEGER NOT NULL,
    org_id INTEGER NOT NULL,
    -- Server-controlled security generation. Destructive credential reset
    -- advances this value before any device/pair graph is removed. Device
    -- registration requests and their retained replay fences bind the value,
    -- so an operation accepted in an earlier generation can never replay as
    -- current authority.
    security_epoch BIGINT NOT NULL DEFAULT 1,
    authority_scope_digest BYTEA NOT NULL UNIQUE,
    authority_inventory_digest BYTEA NOT NULL,
    authority_row_count BIGINT NOT NULL DEFAULT 0,
    device_count BIGINT NOT NULL DEFAULT 0,
    device_operation_count BIGINT NOT NULL DEFAULT 0,
    lineage_count BIGINT NOT NULL DEFAULT 0,
    pairing_count BIGINT NOT NULL DEFAULT 0,
    pairing_operation_count BIGINT NOT NULL DEFAULT 0,
    host_state_count BIGINT NOT NULL DEFAULT 0,
    revocation_event_count BIGINT NOT NULL DEFAULT 0,
    highwater_installation_group_count BIGINT NOT NULL DEFAULT 0,
    highwater_generation_count BIGINT NOT NULL DEFAULT 0,
    registration_operation_tombstone_count BIGINT NOT NULL DEFAULT 0,
    installation_retirement_count BIGINT NOT NULL DEFAULT 0,
    reset_clear_obligation_count BIGINT NOT NULL DEFAULT 0,
    reset_clear_admission_count BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_authority_account_scope_digest_length
        CHECK (octet_length(authority_scope_digest) = 32),
    CONSTRAINT maple_pairing_authority_account_inventory_digest_length
        CHECK (octet_length(authority_inventory_digest) = 32),
    CONSTRAINT maple_pairing_authority_account_security_epoch_positive
        CHECK (security_epoch > 0),
    CONSTRAINT maple_pairing_authority_account_count_bounds CHECK (
        authority_row_count BETWEEN 0 AND 567360
        AND device_count BETWEEN 0 AND 32
        AND device_operation_count BETWEEN 0 AND 32768
        AND lineage_count BETWEEN 0 AND 128
        AND pairing_count BETWEEN 0 AND 128
        AND pairing_operation_count BETWEEN 0 AND 640
        AND host_state_count BETWEEN 0 AND 32
        AND revocation_event_count BETWEEN 0 AND 128
        AND highwater_installation_group_count BETWEEN 0 AND 1024
        AND highwater_generation_count BETWEEN 0 AND 4096
        AND registration_operation_tombstone_count BETWEEN 0 AND 32768
        AND installation_retirement_count BETWEEN 0 AND 1024
        AND reset_clear_obligation_count BETWEEN 0 AND 4096
        AND reset_clear_admission_count BETWEEN 0 AND 524288
        AND device_operation_count + registration_operation_tombstone_count <= 32768
        AND (highwater_generation_count = 0) =
            (highwater_installation_group_count = 0)
        AND highwater_generation_count >= highwater_installation_group_count
        AND highwater_generation_count <= authority_row_count
        AND authority_row_count =
            device_count + device_operation_count + lineage_count + pairing_count
            + pairing_operation_count + host_state_count + revocation_event_count
            + highwater_generation_count + registration_operation_tombstone_count
            + installation_retirement_count + reset_clear_obligation_count
            + reset_clear_admission_count
    ),
    CONSTRAINT maple_pairing_authority_account_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairing_authority_account_mac_length CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_authority_account_user_project_unique
        UNIQUE (user_id, project_id),
    CONSTRAINT maple_pairing_authority_account_scope_unique
        UNIQUE (user_id, project_id, org_id),
    CONSTRAINT maple_pairing_authority_account_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_authority_account_project_scope_fk
        FOREIGN KEY (project_id, org_id)
        REFERENCES org_projects(id, org_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_authority_account_project_head_fk
        FOREIGN KEY (project_id, org_id)
        REFERENCES maple_pairing_authority_project_heads(project_id, org_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_authority_account_heads_project
    ON maple_pairing_authority_account_heads(project_id, user_id);
CREATE INDEX idx_maple_pairing_authority_account_heads_org
    ON maple_pairing_authority_account_heads(org_id, project_id, user_id);
CREATE TRIGGER update_maple_pairing_authority_account_heads_updated_at
    BEFORE UPDATE ON maple_pairing_authority_account_heads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- The device migration predates authenticated authority heads and used
-- cascading parent deletes. Replace those edges so authority can be removed
-- only by an explicit, verified cleanup transaction.
ALTER TABLE maple_device_registration_operations
    DROP CONSTRAINT maple_device_registration_operations_user_id_fkey,
    DROP CONSTRAINT maple_device_registration_operations_project_id_fkey,
    DROP CONSTRAINT maple_device_registration_operations_scoped_device_fk;

-- Registration idempotency is scoped to a server-controlled security epoch.
-- The exact signed response is persisted for byte-identical replay while the
-- operation remains live. Reset retires each live operation into a
-- pseudonymous tombstone before deleting the device graph.
ALTER TABLE maple_device_registration_operations
    ADD COLUMN authority_scope_digest BYTEA NOT NULL,
    ADD COLUMN lookup_digest BYTEA NOT NULL,
    ADD COLUMN operation_lookup_digest BYTEA NOT NULL,
    ADD COLUMN known_security_epoch BIGINT NOT NULL,
    ADD COLUMN accepted_security_epoch BIGINT NOT NULL,
    ADD COLUMN response_kind SMALLINT NOT NULL,
    ADD COLUMN sync_payload_version SMALLINT NOT NULL,
    ADD COLUMN sync_payload_enc BYTEA NOT NULL,
    ADD COLUMN sync_issuer_key_id TEXT NOT NULL,
    ADD COLUMN sync_digest BYTEA NOT NULL,
    ADD CONSTRAINT maple_device_registration_operations_authority_scope_length
        CHECK (octet_length(authority_scope_digest) = 32),
    ADD CONSTRAINT maple_device_registration_operations_lookup_length
        CHECK (octet_length(lookup_digest) = 32),
    ADD CONSTRAINT maple_device_registration_operations_operation_lookup_length
        CHECK (octet_length(operation_lookup_digest) = 32),
    ADD CONSTRAINT maple_device_registration_operations_security_epoch_shape
        CHECK (
            known_security_epoch > 0
            AND accepted_security_epoch = known_security_epoch
        ),
    ADD CONSTRAINT maple_device_registration_operations_response_kind_v1
        -- 1 Ready, 2 RevocationsPending, 3 ResetClearRequired.
        CHECK (response_kind IN (1, 2, 3)),
    ADD CONSTRAINT maple_device_registration_operations_sync_shape
        CHECK (
            sync_payload_version = 1
            AND octet_length(sync_payload_enc) BETWEEN 1 AND 65536
            AND sync_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
            AND octet_length(sync_digest) = 32
        ),
    ADD CONSTRAINT maple_device_registration_operations_sync_issuer_fk
        FOREIGN KEY (sync_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT maple_device_registration_operations_operation_lookup_unique
        UNIQUE (authority_scope_digest, operation_lookup_digest);

CREATE INDEX idx_maple_device_registration_operations_authority_scope
    ON maple_device_registration_operations(authority_scope_digest, id);

-- Persisted issuer references must be directly auditable without decrypting
-- historical response ciphertext. Arrays are one-dimensional, bounded,
-- strictly sorted, duplicate-free, and restricted to the public key-id token
-- grammar used by the v1 wire protocol.
CREATE OR REPLACE FUNCTION maple_pairing_issuer_key_ids_are_canonical(
    key_ids TEXT[],
    maximum_count INTEGER
) RETURNS BOOLEAN AS $$
    SELECT key_ids IS NOT NULL
       AND array_ndims(key_ids) = 1
       AND cardinality(key_ids) BETWEEN 1 AND maximum_count
       AND array_position(key_ids, NULL) IS NULL
       AND key_ids = ARRAY(
            SELECT key_id COLLATE "C"
              FROM unnest(key_ids) AS key_id
             WHERE key_id ~ '^[a-z0-9._:-]{1,64}$'
             GROUP BY key_id COLLATE "C"
             ORDER BY key_id COLLATE "C"
       );
$$ LANGUAGE SQL IMMUTABLE
SET search_path = pg_catalog, public;

CREATE TABLE maple_pairing_registration_operation_tombstones (
    id BIGSERIAL PRIMARY KEY,
    authority_scope_digest BYTEA NOT NULL,
    lookup_digest BYTEA NOT NULL,
    operation_lookup_digest BYTEA NOT NULL,
    retired_security_epoch BIGINT NOT NULL,
    request_mac BYTEA NOT NULL,
    outcome_kind SMALLINT NOT NULL,
    outcome_digest BYTEA NOT NULL,
    receipt_version SMALLINT NOT NULL,
    receipt_enc BYTEA NOT NULL,
    receipt_digest BYTEA NOT NULL,
    referenced_issuer_key_ids TEXT[] NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL,
    record_mac BYTEA NOT NULL,
    retired_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_registration_operation_tombstones_id_positive
        CHECK (id > 0),
    CONSTRAINT maple_pairing_registration_operation_tombstones_digest_lengths CHECK (
        octet_length(authority_scope_digest) = 32
        AND octet_length(lookup_digest) = 32
        AND octet_length(operation_lookup_digest) = 32
        AND octet_length(request_mac) = 32
        AND octet_length(outcome_digest) = 32
        AND octet_length(receipt_digest) = 32
        AND octet_length(record_mac) = 32
    ),
    CONSTRAINT maple_pairing_registration_operation_tombstones_epoch_positive
        CHECK (retired_security_epoch > 0),
    CONSTRAINT maple_pairing_registration_operation_tombstones_outcome_v1
        -- Preserve the exact committed response class across retirement.
        CHECK (outcome_kind IN (1, 2, 3)),
    CONSTRAINT maple_pairing_registration_operation_tombstones_receipt_shape CHECK (
        receipt_version = 1
        AND octet_length(receipt_enc) BETWEEN 1 AND 65536
        AND maple_pairing_issuer_key_ids_are_canonical(referenced_issuer_key_ids, 4)
        AND accepted_at <= retired_at
    ),
    CONSTRAINT maple_pairing_registration_operation_tombstones_operation_unique
        UNIQUE (authority_scope_digest, operation_lookup_digest),
    CONSTRAINT maple_pairing_registration_operation_tombstones_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_registration_operation_tombstones_scope
    ON maple_pairing_registration_operation_tombstones(authority_scope_digest, id);
CREATE INDEX idx_maple_pairing_registration_operation_tombstones_lookup
    ON maple_pairing_registration_operation_tombstones(
        authority_scope_digest,
        lookup_digest,
        retired_security_epoch,
        id
    );

ALTER TABLE maple_devices
    DROP CONSTRAINT maple_devices_user_id_fkey,
    DROP CONSTRAINT maple_devices_project_id_fkey;
ALTER TABLE maple_devices
    ADD CONSTRAINT maple_devices_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT maple_devices_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE maple_device_registration_operations
    ADD CONSTRAINT maple_device_registration_operations_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT maple_device_registration_operations_scoped_device_fk
        FOREIGN KEY (maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT maple_device_registration_operations_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT maple_device_registration_operations_authority_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;

-- These row guards do not have an application-session bypass. The destructive
-- down migration drops them only after independently validating its disposable
-- database guard.
CREATE OR REPLACE FUNCTION enforce_maple_pairing_authority_global_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        RAISE EXCEPTION 'Maple pairing authority root cannot be recreated';
    ELSIF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Maple pairing authority root cannot be removed';
    END IF;

    IF NEW.singleton IS DISTINCT FROM OLD.singleton
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'Maple pairing authority root identity is immutable';
    END IF;

    IF OLD.activation_state = 1 THEN
        IF NEW.activation_state <> 2
           OR OLD.revision <> 1
           OR NEW.revision <> 2 THEN
            RAISE EXCEPTION 'pending Maple pairing authority root permits only revision-2 activation';
        END IF;

        -- Pairing authority may not predate authenticated activation. Scoped
        -- heads and the issuer-key registry are deliberately excluded:
        -- bootstrap creates those in the same transaction before switching
        -- the root to Active, and deferred hierarchy checks cover the final
        -- state atomically.
        IF EXISTS (SELECT 1 FROM maple_devices LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_device_registration_operations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_lineages LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairings LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_operations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_host_states LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_revocation_highwaters LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_revocation_events LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_registration_operation_tombstones LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_reset_clear_obligations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_reset_clear_admissions LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_installation_retirements LIMIT 1) THEN
            RAISE EXCEPTION 'Maple pairing authority cannot activate over existing leaf state';
        END IF;
    ELSIF OLD.activation_state = 2 THEN
        IF NEW.activation_state <> 2 THEN
            RAISE EXCEPTION 'active Maple pairing authority root cannot be downgraded';
        END IF;
        IF NEW.revision <> OLD.revision + 1 THEN
            RAISE EXCEPTION 'active Maple pairing authority root revision must advance exactly once';
        END IF;
    ELSE
        RAISE EXCEPTION 'invalid Maple pairing authority root state';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_authority_global_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_authority_global_heads
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_global_mutation();

CREATE OR REPLACE FUNCTION enforce_maple_pairing_issuer_key_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Maple pairing issuer key identity is immutable';
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_issuer_key_mutation
    BEFORE UPDATE OR DELETE ON maple_pairing_issuer_keys
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_issuer_key_mutation();

CREATE OR REPLACE FUNCTION enforce_maple_pairing_authority_scoped_head_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.revision <> 1 THEN
            RAISE EXCEPTION 'new Maple pairing authority head must start at revision 1';
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    IF NEW.revision <> OLD.revision + 1 THEN
        RAISE EXCEPTION 'Maple pairing authority head revision must advance exactly once';
    END IF;

    IF TG_ARGV[0] = 'org' THEN
        IF NEW.org_id IS DISTINCT FROM OLD.org_id
           OR NEW.global_singleton IS DISTINCT FROM OLD.global_singleton
           OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
            RAISE EXCEPTION 'Maple pairing organization head identity is immutable';
        END IF;
    ELSIF TG_ARGV[0] = 'project' THEN
        IF NEW.project_id IS DISTINCT FROM OLD.project_id
           OR NEW.org_id IS DISTINCT FROM OLD.org_id
           OR NEW.project_uuid IS DISTINCT FROM OLD.project_uuid
           OR NEW.subject_project_id IS DISTINCT FROM OLD.subject_project_id
           OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
            RAISE EXCEPTION 'Maple pairing project head identity is immutable';
        END IF;
    ELSIF TG_ARGV[0] = 'account' THEN
        IF NEW.user_id IS DISTINCT FROM OLD.user_id
           OR NEW.project_id IS DISTINCT FROM OLD.project_id
           OR NEW.org_id IS DISTINCT FROM OLD.org_id
           OR NEW.authority_scope_digest IS DISTINCT FROM OLD.authority_scope_digest
           OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
            RAISE EXCEPTION 'Maple pairing account head identity is immutable';
        END IF;
    ELSE
        RAISE EXCEPTION 'unknown Maple pairing authority head kind';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_authority_org_head_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_authority_org_heads
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_scoped_head_mutation('org');
CREATE TRIGGER guard_maple_pairing_authority_project_head_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_authority_project_heads
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_scoped_head_mutation('project');
CREATE TRIGGER guard_maple_pairing_authority_account_head_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_authority_account_heads
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_scoped_head_mutation('account');

CREATE OR REPLACE FUNCTION enforce_maple_pairing_authority_marker_mutation()
RETURNS TRIGGER AS $$
DECLARE
    marker_name CONSTANT TEXT := 'maple_pairing_authority_v1_activated';
BEGIN
    IF (TG_OP = 'INSERT' AND NEW.name = marker_name)
       OR (TG_OP = 'DELETE' AND OLD.name = marker_name)
       OR (TG_OP = 'UPDATE' AND (OLD.name = marker_name OR NEW.name = marker_name)) THEN
        IF EXISTS (
            SELECT 1
            FROM maple_pairing_authority_global_heads
            WHERE singleton AND activation_state = 2
        ) THEN
            RAISE EXCEPTION 'active Maple pairing authority marker is immutable';
        END IF;
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_authority_marker_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON app_data_migrations
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_marker_mutation();

-- Parent rows and authenticated authority heads are created/deleted in the
-- same transaction. Activation performs the one required complete inventory
-- proof. Thereafter the deferred row events preserve that established
-- invariant inductively by checking only each changed OLD/NEW scope and its
-- indexed ancestors. Bulk tenant deletion therefore does not repeat a global
-- anti-join scan for every deleted row.
CREATE OR REPLACE FUNCTION enforce_maple_pairing_authority_hierarchy_commit()
RETURNS TRIGGER AS $$
DECLARE
    root_state SMALLINT;
    root_count BIGINT;
    root_issuer_key_count BIGINT;
    actual_issuer_key_count BIGINT;
    marker_name CONSTANT TEXT := 'maple_pairing_authority_v1_activated';
    parent_exists BOOLEAN;
    head_exists BOOLEAN;
    scope_org_id INTEGER;
    scope_project_id INTEGER;
    scope_project_uuid UUID;
    scope_subject_project_id UUID;
    scope_user_id UUID;
BEGIN
    SELECT count(*), min(activation_state), max(issuer_key_count)
      INTO root_count, root_state, root_issuer_key_count
      FROM maple_pairing_authority_global_heads;
    IF root_count <> 1 THEN
        RAISE EXCEPTION 'Maple pairing authority root cardinality is invalid';
    END IF;

    IF root_state = 1 THEN
        IF EXISTS (SELECT 1 FROM app_data_migrations WHERE name = marker_name)
           OR EXISTS (SELECT 1 FROM maple_pairing_authority_org_heads LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_issuer_keys LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_authority_project_heads LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_authority_account_heads LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_devices LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_device_registration_operations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_lineages LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairings LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_operations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_host_states LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_revocation_highwaters LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_revocation_events LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_registration_operation_tombstones LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_reset_clear_obligations LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_reset_clear_admissions LIMIT 1)
           OR EXISTS (SELECT 1 FROM maple_pairing_installation_retirements LIMIT 1) THEN
            RAISE EXCEPTION 'pending Maple pairing authority must remain empty and unmarked';
        END IF;
        RETURN NULL;
    ELSIF root_state <> 2 THEN
        RAISE EXCEPTION 'Maple pairing authority root state is invalid';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM app_data_migrations WHERE name = marker_name) THEN
        RAISE EXCEPTION 'active Maple pairing authority marker is missing';
    END IF;

    IF TG_TABLE_NAME IN (
        'maple_pairing_authority_global_heads',
        'maple_pairing_issuer_keys'
    ) THEN
        SELECT count(*) INTO actual_issuer_key_count
          FROM maple_pairing_issuer_keys;
        IF root_issuer_key_count <> actual_issuer_key_count THEN
            RAISE EXCEPTION 'active Maple pairing issuer-key inventory count is inconsistent';
        END IF;
    END IF;

    -- Existing parents do not fire row events during bootstrap, so the
    -- Pending-to-Active root transition must establish the complete base case
    -- once. Every later hierarchy mutation is covered by a scoped deferred
    -- event below.
    IF TG_TABLE_NAME = 'maple_pairing_authority_global_heads'
       AND TG_OP = 'UPDATE' THEN
        IF OLD.activation_state = 1
           AND NEW.activation_state = 2
           AND (EXISTS (
                SELECT 1 FROM orgs o
                LEFT JOIN maple_pairing_authority_org_heads h ON h.org_id = o.id
                WHERE h.org_id IS NULL
           ) OR EXISTS (
                SELECT 1 FROM maple_pairing_authority_org_heads h
                LEFT JOIN orgs o ON o.id = h.org_id
                WHERE o.id IS NULL
            ) OR EXISTS (
                SELECT 1 FROM org_projects p
                LEFT JOIN maple_pairing_authority_project_heads h
                  ON h.project_id = p.id
                 AND h.org_id = p.org_id
                 AND h.project_uuid = p.uuid
                 AND h.subject_project_id = p.client_id
                WHERE h.project_id IS NULL
           ) OR EXISTS (
                SELECT 1 FROM maple_pairing_authority_project_heads h
                LEFT JOIN org_projects p
                  ON p.id = h.project_id
                 AND p.org_id = h.org_id
                 AND p.uuid = h.project_uuid
                 AND p.client_id = h.subject_project_id
                WHERE p.id IS NULL
           ) OR EXISTS (
                SELECT 1 FROM users u
                JOIN org_projects p ON p.id = u.project_id
                LEFT JOIN maple_pairing_authority_account_heads h
                  ON h.user_id = u.uuid
                 AND h.project_id = u.project_id
                 AND h.org_id = p.org_id
                WHERE h.user_id IS NULL
           ) OR EXISTS (
                SELECT 1 FROM maple_pairing_authority_account_heads h
                LEFT JOIN users u
                  ON u.uuid = h.user_id AND u.project_id = h.project_id
                LEFT JOIN org_projects p
                  ON p.id = h.project_id
                 AND p.org_id = h.org_id
                LEFT JOIN maple_pairing_authority_project_heads ph
                  ON ph.project_id = p.id
                 AND ph.org_id = p.org_id
                 AND ph.project_uuid = p.uuid
                 AND ph.subject_project_id = p.client_id
                WHERE u.uuid IS NULL OR p.id IS NULL OR ph.project_id IS NULL
           )) THEN
            RAISE EXCEPTION 'active Maple pairing authority hierarchy is incomplete';
        END IF;
    END IF;

    IF TG_TABLE_NAME IN ('orgs', 'maple_pairing_authority_org_heads') THEN
        IF TG_OP <> 'INSERT' THEN
            IF TG_TABLE_NAME = 'orgs' THEN
                scope_org_id := OLD.id;
            ELSE
                scope_org_id := OLD.org_id;
            END IF;
            SELECT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   )
              INTO parent_exists, head_exists;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing organization authority is incomplete';
            END IF;
        END IF;
        IF TG_OP <> 'DELETE' THEN
            IF TG_TABLE_NAME = 'orgs' THEN
                scope_org_id := NEW.id;
            ELSE
                scope_org_id := NEW.org_id;
            END IF;
            SELECT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   )
              INTO parent_exists, head_exists;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing organization authority is incomplete';
            END IF;
        END IF;
    ELSIF TG_TABLE_NAME IN (
        'org_projects',
        'maple_pairing_authority_project_heads'
    ) THEN
        IF TG_TABLE_NAME = 'org_projects' THEN
            IF TG_OP = 'UPDATE' THEN
                IF NEW.id IS DISTINCT FROM OLD.id
                   OR NEW.org_id IS DISTINCT FROM OLD.org_id
                   OR NEW.uuid IS DISTINCT FROM OLD.uuid
                   OR NEW.client_id IS DISTINCT FROM OLD.client_id THEN
                    RAISE EXCEPTION 'active Maple pairing project identity cannot be replaced';
                END IF;
            END IF;
        END IF;
        IF TG_OP <> 'INSERT' THEN
            IF TG_TABLE_NAME = 'org_projects' THEN
                scope_project_id := OLD.id;
                scope_project_uuid := OLD.uuid;
                scope_subject_project_id := OLD.client_id;
            ELSE
                scope_project_id := OLD.project_id;
                scope_project_uuid := OLD.project_uuid;
                scope_subject_project_id := OLD.subject_project_id;
            END IF;
            scope_org_id := OLD.org_id;
            SELECT EXISTS (
                       SELECT 1 FROM org_projects
                       WHERE id = scope_project_id
                         AND org_id = scope_org_id
                         AND uuid = scope_project_uuid
                         AND client_id = scope_subject_project_id
                   ),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_project_heads
                       WHERE project_id = scope_project_id
                         AND org_id = scope_org_id
                         AND project_uuid = scope_project_uuid
                         AND subject_project_id = scope_subject_project_id
                   )
              INTO parent_exists, head_exists;
            IF EXISTS (
                   SELECT 1 FROM org_projects
                   WHERE id = scope_project_id
                     AND (
                         org_id IS DISTINCT FROM scope_org_id
                         OR uuid IS DISTINCT FROM scope_project_uuid
                         OR client_id IS DISTINCT FROM scope_subject_project_id
                     )
               ) OR EXISTS (
                   SELECT 1 FROM maple_pairing_authority_project_heads
                   WHERE project_id = scope_project_id
                     AND (
                         org_id IS DISTINCT FROM scope_org_id
                         OR project_uuid IS DISTINCT FROM scope_project_uuid
                         OR subject_project_id IS DISTINCT FROM scope_subject_project_id
                     )
               ) THEN
                RAISE EXCEPTION 'active Maple pairing project identity cannot be replaced';
            END IF;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing project authority is incomplete';
            END IF;
            IF parent_exists THEN
                IF NOT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id)
                   OR NOT EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   ) THEN
                    RAISE EXCEPTION 'active Maple pairing project ancestry is incomplete';
                END IF;
            END IF;
        END IF;
        IF TG_OP <> 'DELETE' THEN
            IF TG_TABLE_NAME = 'org_projects' THEN
                scope_project_id := NEW.id;
                scope_project_uuid := NEW.uuid;
                scope_subject_project_id := NEW.client_id;
            ELSE
                scope_project_id := NEW.project_id;
                scope_project_uuid := NEW.project_uuid;
                scope_subject_project_id := NEW.subject_project_id;
            END IF;
            scope_org_id := NEW.org_id;
            SELECT EXISTS (
                       SELECT 1 FROM org_projects
                       WHERE id = scope_project_id
                         AND org_id = scope_org_id
                         AND uuid = scope_project_uuid
                         AND client_id = scope_subject_project_id
                   ),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_project_heads
                       WHERE project_id = scope_project_id
                         AND org_id = scope_org_id
                         AND project_uuid = scope_project_uuid
                         AND subject_project_id = scope_subject_project_id
                   )
              INTO parent_exists, head_exists;
            IF EXISTS (
                   SELECT 1 FROM org_projects
                   WHERE id = scope_project_id
                     AND (
                         org_id IS DISTINCT FROM scope_org_id
                         OR uuid IS DISTINCT FROM scope_project_uuid
                         OR client_id IS DISTINCT FROM scope_subject_project_id
                     )
               ) OR EXISTS (
                   SELECT 1 FROM maple_pairing_authority_project_heads
                   WHERE project_id = scope_project_id
                     AND (
                         org_id IS DISTINCT FROM scope_org_id
                         OR project_uuid IS DISTINCT FROM scope_project_uuid
                         OR subject_project_id IS DISTINCT FROM scope_subject_project_id
                     )
               ) THEN
                RAISE EXCEPTION 'active Maple pairing project identity cannot be replaced';
            END IF;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing project authority is incomplete';
            END IF;
            IF parent_exists THEN
                IF NOT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id)
                   OR NOT EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   ) THEN
                    RAISE EXCEPTION 'active Maple pairing project ancestry is incomplete';
                END IF;
            END IF;
        END IF;
    ELSIF TG_TABLE_NAME IN ('users', 'maple_pairing_authority_account_heads') THEN
        IF TG_OP <> 'INSERT' THEN
            scope_project_id := OLD.project_id;
            IF TG_TABLE_NAME = 'users' THEN
                scope_user_id := OLD.uuid;
                SELECT org_id INTO scope_org_id
                  FROM org_projects WHERE id = scope_project_id;
            ELSE
                scope_user_id := OLD.user_id;
                scope_org_id := OLD.org_id;
            END IF;
            SELECT EXISTS (
                       SELECT 1 FROM users
                       WHERE uuid = scope_user_id AND project_id = scope_project_id
                   ),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_account_heads
                       WHERE user_id = scope_user_id
                         AND project_id = scope_project_id
                         AND org_id = scope_org_id
                   )
              INTO parent_exists, head_exists;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing account authority is incomplete';
            END IF;
            IF parent_exists THEN
                IF NOT EXISTS (
                       SELECT 1
                       FROM org_projects p
                       JOIN maple_pairing_authority_project_heads h
                         ON h.project_id = p.id
                        AND h.org_id = p.org_id
                        AND h.project_uuid = p.uuid
                        AND h.subject_project_id = p.client_id
                       WHERE p.id = scope_project_id
                         AND p.org_id = scope_org_id
                   ) OR NOT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id)
                   OR NOT EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   ) THEN
                    RAISE EXCEPTION 'active Maple pairing account ancestry is incomplete';
                END IF;
            END IF;
        END IF;
        IF TG_OP <> 'DELETE' THEN
            scope_project_id := NEW.project_id;
            IF TG_TABLE_NAME = 'users' THEN
                scope_user_id := NEW.uuid;
                SELECT org_id INTO scope_org_id
                  FROM org_projects WHERE id = scope_project_id;
            ELSE
                scope_user_id := NEW.user_id;
                scope_org_id := NEW.org_id;
            END IF;
            SELECT EXISTS (
                       SELECT 1 FROM users
                       WHERE uuid = scope_user_id AND project_id = scope_project_id
                   ),
                   EXISTS (
                       SELECT 1 FROM maple_pairing_authority_account_heads
                       WHERE user_id = scope_user_id
                         AND project_id = scope_project_id
                         AND org_id = scope_org_id
                   )
              INTO parent_exists, head_exists;
            IF parent_exists IS DISTINCT FROM head_exists THEN
                RAISE EXCEPTION 'active Maple pairing account authority is incomplete';
            END IF;
            IF parent_exists THEN
                IF NOT EXISTS (
                       SELECT 1
                       FROM org_projects p
                       JOIN maple_pairing_authority_project_heads h
                         ON h.project_id = p.id
                        AND h.org_id = p.org_id
                        AND h.project_uuid = p.uuid
                        AND h.subject_project_id = p.client_id
                       WHERE p.id = scope_project_id
                         AND p.org_id = scope_org_id
                   ) OR NOT EXISTS (SELECT 1 FROM orgs WHERE id = scope_org_id)
                   OR NOT EXISTS (
                       SELECT 1 FROM maple_pairing_authority_org_heads
                       WHERE org_id = scope_org_id
                   ) THEN
                    RAISE EXCEPTION 'active Maple pairing account ancestry is incomplete';
                END IF;
            END IF;
        END IF;
    ELSIF TG_TABLE_NAME NOT IN (
        'maple_pairing_authority_global_heads',
        'maple_pairing_issuer_keys',
        'app_data_migrations'
    ) THEN
        RAISE EXCEPTION 'unknown Maple pairing authority hierarchy relation';
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_global_commit
    AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_authority_global_heads
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_issuer_key_commit
    AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_issuer_keys
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_org_head_commit
    AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_authority_org_heads
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_project_head_commit
    AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_authority_project_heads
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_account_head_commit
    AFTER INSERT OR UPDATE OR DELETE ON maple_pairing_authority_account_heads
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_org_parent_commit
    AFTER INSERT OR UPDATE OR DELETE ON orgs
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_project_parent_commit
    AFTER INSERT OR UPDATE OR DELETE ON org_projects
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_user_parent_commit
    AFTER INSERT OR UPDATE OR DELETE ON users
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();
CREATE CONSTRAINT TRIGGER guard_maple_pairing_authority_marker_commit
    AFTER INSERT OR UPDATE OR DELETE ON app_data_migrations
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_authority_hierarchy_commit();

CREATE OR REPLACE FUNCTION forbid_maple_pairing_authority_truncate()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'TRUNCATE of Maple pairing authority state is forbidden';
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_authority_global_truncate
    BEFORE TRUNCATE ON maple_pairing_authority_global_heads
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_issuer_keys_truncate
    BEFORE TRUNCATE ON maple_pairing_issuer_keys
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_org_head_truncate
    BEFORE TRUNCATE ON maple_pairing_authority_org_heads
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_project_head_truncate
    BEFORE TRUNCATE ON maple_pairing_authority_project_heads
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_account_head_truncate
    BEFORE TRUNCATE ON maple_pairing_authority_account_heads
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_user_truncate
    BEFORE TRUNCATE ON users EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_project_truncate
    BEFORE TRUNCATE ON org_projects EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_org_truncate
    BEFORE TRUNCATE ON orgs EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_authority_marker_truncate
    BEFORE TRUNCATE ON app_data_migrations
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();

CREATE TABLE maple_pairing_lineages (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    project_id INTEGER NOT NULL,
    controller_maple_device_id BIGINT NOT NULL,
    host_maple_device_id BIGINT NOT NULL,
    direction SMALLINT NOT NULL,
    last_pairing_incarnation BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_lineages_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_lineages_distinct_devices
        CHECK (controller_maple_device_id <> host_maple_device_id),
    CONSTRAINT maple_pairing_lineages_direction_v1 CHECK (direction = 1),
    CONSTRAINT maple_pairing_lineages_incarnation_nonnegative
        CHECK (last_pairing_incarnation >= 0),
    CONSTRAINT maple_pairing_lineages_scope_unique
        UNIQUE (
            id,
            user_id,
            project_id,
            controller_maple_device_id,
            host_maple_device_id,
            direction
        ),
    CONSTRAINT maple_pairing_lineages_direction_unique
        UNIQUE (
            user_id,
            project_id,
            controller_maple_device_id,
            host_maple_device_id,
            direction
        ),
    CONSTRAINT maple_pairing_lineages_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_lineages_controller_device_fk
        FOREIGN KEY (controller_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_lineages_host_device_fk
        FOREIGN KEY (host_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_lineages_project_id
    ON maple_pairing_lineages(project_id);
CREATE INDEX idx_maple_pairing_lineages_controller
    ON maple_pairing_lineages(controller_maple_device_id);
CREATE INDEX idx_maple_pairing_lineages_host
    ON maple_pairing_lineages(host_maple_device_id);

CREATE TRIGGER update_maple_pairing_lineages_updated_at
    BEFORE UPDATE ON maple_pairing_lineages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE maple_pairings (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE,
    pairing_request_id UUID NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    project_id INTEGER NOT NULL,
    lineage_id BIGINT NOT NULL,
    controller_maple_device_id BIGINT NOT NULL,
    host_maple_device_id BIGINT NOT NULL,
    direction SMALLINT NOT NULL,
    pairing_incarnation BIGINT NOT NULL,
    state SMALLINT NOT NULL,
    revision BIGINT NOT NULL DEFAULT 1,
    request_nonce_mac BYTEA NOT NULL,
    revocation_stream_id UUID,
    revocation_stream_generation BIGINT,
    pair_authorization_digest BYTEA,
    -- Issuer key IDs are immutable identities for one public key. Rotation
    -- adds a new ID; it must never remap an existing ID to new key material.
    ticket_issuer_key_id TEXT NOT NULL,
    authorization_issuer_key_id TEXT,
    revocation_issuer_key_id TEXT,
    payload_version SMALLINT NOT NULL DEFAULT 1,
    payload_enc BYTEA NOT NULL,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairings_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairings_uuid_non_nil
        CHECK (uuid <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairings_request_id_non_nil
        CHECK (pairing_request_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairings_distinct_devices
        CHECK (controller_maple_device_id <> host_maple_device_id),
    CONSTRAINT maple_pairings_direction_v1 CHECK (direction = 1),
    CONSTRAINT maple_pairings_incarnation_positive CHECK (pairing_incarnation > 0),
    CONSTRAINT maple_pairings_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairings_payload_version_v1 CHECK (payload_version = 1),
    CONSTRAINT maple_pairings_state_v1 CHECK (state BETWEEN 1 AND 5),
    CONSTRAINT maple_pairings_revocation_stream_shape CHECK (
        (
            revocation_stream_id IS NULL
            AND revocation_stream_generation IS NULL
        )
        OR (
            revocation_stream_id IS NOT NULL
            AND revocation_stream_id <> '00000000-0000-0000-0000-000000000000'::uuid
            AND revocation_stream_generation > 0
        )
    ),
    CONSTRAINT maple_pairings_revocation_stream_lifecycle CHECK (
        (
            state IN (1, 4)
            AND revocation_stream_id IS NULL
            AND revocation_stream_generation IS NULL
        )
        OR (
            state IN (2, 3, 5)
            AND revocation_stream_id IS NOT NULL
            AND revocation_stream_generation IS NOT NULL
        )
    ),
    CONSTRAINT maple_pairings_authorization_digest_lifecycle CHECK (
        (
            state IN (1, 4)
            AND pair_authorization_digest IS NULL
        )
        OR (
            state IN (2, 3, 5)
            AND pair_authorization_digest IS NOT NULL
            AND octet_length(pair_authorization_digest) = 32
        )
    ),
    CONSTRAINT maple_pairings_nonce_mac_length
        CHECK (octet_length(request_nonce_mac) = 32),
    CONSTRAINT maple_pairings_ticket_issuer_key_id_v1
        CHECK (
            ticket_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
        ),
    CONSTRAINT maple_pairings_authorization_issuer_key_id_v1
        CHECK (
            authorization_issuer_key_id IS NULL
            OR authorization_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
        ),
    CONSTRAINT maple_pairings_revocation_issuer_key_id_v1
        CHECK (
            revocation_issuer_key_id IS NULL
            OR revocation_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
        ),
    CONSTRAINT maple_pairings_issuer_key_lifecycle_v1
        CHECK (
            (
                state IN (1, 4)
                AND authorization_issuer_key_id IS NULL
                AND revocation_issuer_key_id IS NULL
            )
            OR (
                state IN (2, 3)
                AND authorization_issuer_key_id IS NOT NULL
                AND revocation_issuer_key_id IS NULL
            )
            OR (
                state = 5
                AND authorization_issuer_key_id IS NOT NULL
                AND revocation_issuer_key_id IS NOT NULL
            )
        ),
    CONSTRAINT maple_pairings_record_mac_length
        CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairings_payload_bounded
        CHECK (octet_length(payload_enc) <= 65536),
    CONSTRAINT maple_pairings_expiry_after_creation CHECK (expires_at > created_at),
    CONSTRAINT maple_pairings_lifecycle_timestamp_order CHECK (
        (
            state IN (1, 4)
            AND approved_at IS NULL
            AND activated_at IS NULL
            AND revoked_at IS NULL
        )
        OR (
            state = 2
            AND approved_at IS NOT NULL
            AND created_at <= approved_at
            AND activated_at IS NULL
            AND revoked_at IS NULL
        )
        OR (
            state = 3
            AND approved_at IS NOT NULL
            AND activated_at IS NOT NULL
            AND created_at <= approved_at
            AND approved_at <= activated_at
            AND revoked_at IS NULL
        )
        OR (
            state = 5
            AND revision = 3
            AND approved_at IS NOT NULL
            AND activated_at IS NULL
            AND revoked_at IS NOT NULL
            AND created_at <= approved_at
            AND approved_at <= revoked_at
        )
        OR (
            state = 5
            AND revision = 4
            AND approved_at IS NOT NULL
            AND activated_at IS NOT NULL
            AND revoked_at IS NOT NULL
            AND created_at <= approved_at
            AND approved_at <= activated_at
            AND activated_at <= revoked_at
        )
    ),
    CONSTRAINT maple_pairings_incarnation_global_unique UNIQUE (pairing_incarnation),
    CONSTRAINT maple_pairings_id_scope_unique UNIQUE (id, user_id, project_id),
    CONSTRAINT maple_pairings_host_incarnation_scope_unique
        UNIQUE (
            id,
            user_id,
            project_id,
            host_maple_device_id,
            pairing_incarnation,
            revocation_stream_id,
            revocation_stream_generation
        ),
    CONSTRAINT maple_pairings_lineage_incarnation_unique
        UNIQUE (lineage_id, pairing_incarnation),
    CONSTRAINT maple_pairings_request_nonce_unique
        UNIQUE (user_id, project_id, controller_maple_device_id, request_nonce_mac),
    CONSTRAINT maple_pairings_ticket_issuer_fk
        FOREIGN KEY (ticket_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_authorization_issuer_fk
        FOREIGN KEY (authorization_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_revocation_issuer_fk
        FOREIGN KEY (revocation_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_lineage_fk
        FOREIGN KEY (
            lineage_id,
            user_id,
            project_id,
            controller_maple_device_id,
            host_maple_device_id,
            direction
        )
        REFERENCES maple_pairing_lineages(
            id,
            user_id,
            project_id,
            controller_maple_device_id,
            host_maple_device_id,
            direction
        )
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_controller_device_fk
        FOREIGN KEY (controller_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairings_host_device_fk
        FOREIGN KEY (host_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

-- Only one pending, awaiting-host-commit, or active record may exist for a
-- directional lineage. Terminal rows remain as non-reusable incarnation and
-- audit tombstones.
CREATE UNIQUE INDEX idx_maple_pairings_one_live_lineage
    ON maple_pairings(lineage_id)
    WHERE state IN (1, 2, 3);
CREATE INDEX idx_maple_pairings_host_state_uuid
    ON maple_pairings(user_id, project_id, host_maple_device_id, state, uuid DESC);
CREATE INDEX idx_maple_pairings_controller_state_uuid
    ON maple_pairings(user_id, project_id, controller_maple_device_id, state, uuid DESC);
CREATE INDEX idx_maple_pairings_project_id ON maple_pairings(project_id);

CREATE TRIGGER update_maple_pairings_updated_at
    BEFORE UPDATE ON maple_pairings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE maple_pairing_operations (
    id BIGSERIAL PRIMARY KEY,
    operation_id UUID NOT NULL,
    user_id UUID NOT NULL,
    project_id INTEGER NOT NULL,
    actor_maple_device_id BIGINT NOT NULL,
    operation_kind SMALLINT NOT NULL,
    request_mac BYTEA NOT NULL,
    maple_pairing_id BIGINT NOT NULL,
    pairing_revision BIGINT NOT NULL,
    receipt_version SMALLINT NOT NULL DEFAULT 1,
    receipt_enc BYTEA NOT NULL,
    receipt_issuer_key_id TEXT,
    receipt_mac BYTEA NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_operations_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_operations_operation_id_non_nil
        CHECK (operation_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_operations_kind_v1 CHECK (operation_kind BETWEEN 1 AND 5),
    CONSTRAINT maple_pairing_operations_revision_positive CHECK (pairing_revision > 0),
    CONSTRAINT maple_pairing_operations_receipt_version_v1 CHECK (receipt_version = 1),
    CONSTRAINT maple_pairing_operations_request_mac_length
        CHECK (octet_length(request_mac) = 32),
    CONSTRAINT maple_pairing_operations_receipt_issuer_key_id_v1
        CHECK (
            receipt_issuer_key_id IS NULL
            OR receipt_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
        ),
    CONSTRAINT maple_pairing_operations_receipt_issuer_lifecycle_v1
        CHECK (
            (operation_kind = 5 AND receipt_issuer_key_id IS NOT NULL)
            OR (operation_kind <> 5 AND receipt_issuer_key_id IS NULL)
        ),
    CONSTRAINT maple_pairing_operations_receipt_mac_length
        CHECK (octet_length(receipt_mac) = 32),
    CONSTRAINT maple_pairing_operations_pair_kind_unique
        UNIQUE (maple_pairing_id, operation_kind),
    CONSTRAINT maple_pairing_operations_receipt_bounded
        CHECK (octet_length(receipt_enc) <= 65536),
    CONSTRAINT maple_pairing_operations_scope_unique
        UNIQUE (user_id, project_id, actor_maple_device_id, operation_id),
    CONSTRAINT maple_pairing_operations_receipt_issuer_fk
        FOREIGN KEY (receipt_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_operations_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_operations_actor_device_fk
        FOREIGN KEY (actor_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_operations_pairing_fk
        FOREIGN KEY (maple_pairing_id, user_id, project_id)
        REFERENCES maple_pairings(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_operations_pairing
    ON maple_pairing_operations(maple_pairing_id);
CREATE INDEX idx_maple_pairing_operations_project_id
    ON maple_pairing_operations(project_id);

CREATE TABLE maple_pairing_host_states (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    project_id INTEGER NOT NULL,
    host_maple_device_id BIGINT NOT NULL,
    revocation_stream_id UUID NOT NULL,
    revocation_stream_generation BIGINT NOT NULL,
    last_issued_revocation_sequence BIGINT NOT NULL DEFAULT 0,
    last_acked_revocation_sequence BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_host_states_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_host_states_stream_id_non_nil
        CHECK (revocation_stream_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_host_states_stream_generation_positive
        CHECK (revocation_stream_generation > 0),
    CONSTRAINT maple_pairing_host_states_sequences_nonnegative
        CHECK (
            last_issued_revocation_sequence >= 0
            AND last_acked_revocation_sequence >= 0
            AND last_acked_revocation_sequence <= last_issued_revocation_sequence
        ),
    CONSTRAINT maple_pairing_host_states_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_pairing_host_states_record_mac_length
        CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_host_states_scope_unique
        UNIQUE (user_id, project_id, host_maple_device_id),
    CONSTRAINT maple_pairing_host_states_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_host_states_device_fk
        FOREIGN KEY (host_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_host_states_project_id
    ON maple_pairing_host_states(project_id);

CREATE TRIGGER update_maple_pairing_host_states_updated_at
    BEFORE UPDATE ON maple_pairing_host_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- This allocation fence deliberately survives password reset, but verified-
-- clean final account/project/organization deletion consumes it. The stable
-- account/project/installation tuple is retained only as domain-separated
-- keyed digests: no raw identifier is recoverable from this tombstone. Because both lookup and
-- authentication are rooted in the enclave key, rotating that key requires an
-- explicit forward migration of these rows or a coordinated local-admission
-- clear; silently starting a new lookup namespace could reuse a sequence.
-- Generations are append-only: reset inserts generation N+1 and never updates
-- or deletes retired generations. This retains every allocated stream UUID
-- behind a global UNIQUE fence and lets a keyed full-history MAC scan detect a
-- storage-layer attempt to free an old UUID for reuse. V1 retains this history
-- across password reset, bounded by the authenticated per-account quotas. It is
-- removed only by verified-clean final account/project/org deletion after all
-- admissions have been revoked and durably acknowledged.
CREATE TABLE maple_pairing_revocation_highwaters (
    id BIGSERIAL PRIMARY KEY,
    lookup_digest BYTEA NOT NULL,
    authority_scope_digest BYTEA NOT NULL,
    revocation_stream_id UUID NOT NULL,
    revocation_stream_generation BIGINT NOT NULL,
    security_epoch BIGINT NOT NULL,
    last_issued_revocation_sequence BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 1,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_revocation_highwaters_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_revocation_highwaters_lookup_digest_length
        CHECK (octet_length(lookup_digest) = 32),
    CONSTRAINT maple_pairing_highwaters_authority_scope_digest_length
        CHECK (octet_length(authority_scope_digest) = 32),
    CONSTRAINT maple_pairing_revocation_highwaters_stream_id_non_nil
        CHECK (revocation_stream_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_revocation_highwaters_stream_generation_positive
        CHECK (revocation_stream_generation > 0),
    CONSTRAINT maple_pairing_revocation_highwaters_security_epoch_positive
        CHECK (security_epoch > 0),
    CONSTRAINT maple_pairing_revocation_highwaters_stream_id_unique
        UNIQUE (revocation_stream_id),
    CONSTRAINT maple_pairing_revocation_highwaters_lookup_generation_unique
        UNIQUE (lookup_digest, revocation_stream_generation),
    CONSTRAINT maple_pairing_revocation_highwaters_exact_namespace_unique
        UNIQUE (
            authority_scope_digest,
            lookup_digest,
            revocation_stream_id,
            revocation_stream_generation,
            security_epoch
        ),
    CONSTRAINT maple_pairing_revocation_highwaters_sequence_nonnegative
        CHECK (last_issued_revocation_sequence >= 0),
    CONSTRAINT maple_pairing_revocation_highwaters_revision_positive
        CHECK (revision > 0),
    CONSTRAINT maple_pairing_revocation_highwaters_record_mac_length
        CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_revocation_highwaters_authority_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_revocation_highwaters_lookup
    ON maple_pairing_revocation_highwaters(
        lookup_digest,
        revocation_stream_generation DESC
    );
CREATE INDEX idx_maple_pairing_revocation_highwaters_authority_scope
    ON maple_pairing_revocation_highwaters(
        authority_scope_digest,
        lookup_digest,
        revocation_stream_generation
    );

CREATE TRIGGER update_maple_pairing_revocation_highwaters_updated_at
    BEFORE UPDATE ON maple_pairing_revocation_highwaters
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Destructive reset removes the raw device/pair graph only after appending an
-- authenticated reset-clear instruction for each affected host installation.
-- These rows deliberately retain no raw account, project, installation, or
-- device identifier. The encrypted public host claim is retained solely so a
-- later registration/list transaction can materialize the issuer-signed
-- instruction without reconstructing deleted authority.
CREATE TABLE maple_pairing_reset_clear_obligations (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE,
    authority_scope_digest BYTEA NOT NULL,
    lookup_digest BYTEA NOT NULL,
    host_identity_mac BYTEA NOT NULL,
    reset_id UUID NOT NULL,
    reset_generation BIGINT NOT NULL,
    cumulative_reset_count BIGINT NOT NULL,
    previous_event_id UUID,
    previous_instruction_digest BYTEA,
    previous_chain_digest BYTEA,
    old_revocation_stream_id UUID NOT NULL,
    old_revocation_stream_generation BIGINT NOT NULL,
    source_security_epoch BIGINT NOT NULL,
    source_last_issued_revocation_sequence BIGINT NOT NULL,
    target_revocation_stream_id UUID NOT NULL,
    target_revocation_stream_generation BIGINT NOT NULL,
    target_security_epoch BIGINT NOT NULL,
    target_instruction_sequence BIGINT NOT NULL DEFAULT 1,
    clear_scope SMALLINT NOT NULL DEFAULT 1,
    admission_set_digest BYTEA NOT NULL,
    admission_count SMALLINT NOT NULL,
    host_claim_payload_version SMALLINT NOT NULL,
    host_claim_payload_enc BYTEA NOT NULL,
    host_claim_digest BYTEA NOT NULL,
    instruction_payload_version SMALLINT NOT NULL,
    instruction_payload_enc BYTEA NOT NULL,
    instruction_digest BYTEA NOT NULL,
    chain_digest BYTEA NOT NULL,
    reset_at TIMESTAMPTZ NOT NULL,
    signed_instruction_payload_version SMALLINT,
    signed_instruction_payload_enc BYTEA,
    signed_instruction_issuer_key_id TEXT,
    signed_instruction_digest BYTEA,
    sync_payload_version SMALLINT,
    sync_payload_enc BYTEA,
    sync_issuer_key_id TEXT,
    sync_digest BYTEA,
    state SMALLINT NOT NULL DEFAULT 1,
    revision BIGINT NOT NULL DEFAULT 1,
    acked_by_head_event_id UUID,
    acked_at TIMESTAMPTZ,
    ack_operation_id UUID,
    ack_host_registration_lookup_digest BYTEA,
    ack_request_mac BYTEA,
    ack_receipt_version SMALLINT,
    ack_receipt_enc BYTEA,
    ack_receipt_issuer_key_id TEXT,
    ack_receipt_digest BYTEA,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_reset_clear_obligations_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_reset_clear_obligations_uuid_non_nil
        CHECK (uuid <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_reset_clear_obligations_reset_id_non_nil
        CHECK (reset_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_reset_clear_obligations_digest_lengths CHECK (
        octet_length(authority_scope_digest) = 32
        AND octet_length(lookup_digest) = 32
        AND octet_length(host_identity_mac) = 32
        AND octet_length(admission_set_digest) = 32
        AND octet_length(host_claim_digest) = 32
        AND octet_length(instruction_digest) = 32
        AND octet_length(chain_digest) = 32
        AND octet_length(record_mac) = 32
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_generation_shape CHECK (
        reset_generation > 0
        AND cumulative_reset_count = reset_generation
        AND source_security_epoch > 0
        AND target_security_epoch = source_security_epoch + 1
        AND source_last_issued_revocation_sequence >= 0
        AND old_revocation_stream_generation > 0
        AND target_revocation_stream_generation = old_revocation_stream_generation + 1
        AND old_revocation_stream_id <> target_revocation_stream_id
        AND target_instruction_sequence = 1
        AND clear_scope = 1
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_previous_shape CHECK (
        (previous_event_id IS NULL) = (previous_instruction_digest IS NULL)
        AND (previous_event_id IS NULL) = (previous_chain_digest IS NULL)
        AND (
            previous_event_id IS NULL
            OR previous_event_id <> '00000000-0000-0000-0000-000000000000'::uuid
        )
        AND (
            previous_instruction_digest IS NULL
            OR octet_length(previous_instruction_digest) = 32
        )
        AND (
            previous_chain_digest IS NULL
            OR octet_length(previous_chain_digest) = 32
        )
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_admission_count
        CHECK (admission_count BETWEEN 0 AND 128),
    CONSTRAINT maple_pairing_reset_clear_obligations_unsigned_payload_shape CHECK (
        host_claim_payload_version = 1
        AND octet_length(host_claim_payload_enc) BETWEEN 1 AND 65536
        AND instruction_payload_version = 1
        AND octet_length(instruction_payload_enc) BETWEEN 1 AND 65536
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_signed_material_shape CHECK (
        (
            signed_instruction_payload_version IS NULL
            AND signed_instruction_payload_enc IS NULL
            AND signed_instruction_issuer_key_id IS NULL
            AND signed_instruction_digest IS NULL
            AND sync_payload_version IS NULL
            AND sync_payload_enc IS NULL
            AND sync_issuer_key_id IS NULL
            AND sync_digest IS NULL
        )
        OR (
            signed_instruction_payload_version IS NOT NULL
            AND signed_instruction_payload_enc IS NOT NULL
            AND signed_instruction_issuer_key_id IS NOT NULL
            AND signed_instruction_digest IS NOT NULL
            AND sync_payload_version IS NOT NULL
            AND sync_payload_enc IS NOT NULL
            AND sync_issuer_key_id IS NOT NULL
            AND sync_digest IS NOT NULL
            AND signed_instruction_payload_version = 1
            AND octet_length(signed_instruction_payload_enc) BETWEEN 1 AND 65536
            AND signed_instruction_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
            AND octet_length(signed_instruction_digest) = 32
            AND sync_payload_version = 1
            AND octet_length(sync_payload_enc) BETWEEN 1 AND 65536
            AND sync_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
            AND octet_length(sync_digest) = 32
        )
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_state_v1 CHECK (state IN (1, 2)),
    CONSTRAINT maple_pairing_reset_clear_obligations_state_shape CHECK (
        -- Pending and not yet materialized.
        (
            state = 1 AND revision = 1
            AND signed_instruction_payload_version IS NULL
            AND acked_by_head_event_id IS NULL AND acked_at IS NULL
            AND ack_operation_id IS NULL
            AND ack_host_registration_lookup_digest IS NULL
            AND ack_request_mac IS NULL
            AND ack_receipt_version IS NULL AND ack_receipt_enc IS NULL
            AND ack_receipt_issuer_key_id IS NULL AND ack_receipt_digest IS NULL
        )
        OR
        -- Pending with its exact signed instruction and registration sync.
        (
            state = 1 AND revision = 2
            AND signed_instruction_payload_version IS NOT NULL
            AND acked_by_head_event_id IS NULL AND acked_at IS NULL
            AND ack_operation_id IS NULL
            AND ack_host_registration_lookup_digest IS NULL
            AND ack_request_mac IS NULL
            AND ack_receipt_version IS NULL AND ack_receipt_enc IS NULL
            AND ack_receipt_issuer_key_id IS NULL AND ack_receipt_digest IS NULL
        )
        OR
        -- An unmaterialized missed-reset ancestor discharged by a newer head.
        (
            state = 2 AND revision = 2
            AND signed_instruction_payload_version IS NULL
            AND acked_by_head_event_id IS NOT NULL
            AND acked_by_head_event_id <> uuid AND acked_at IS NOT NULL
            AND ack_operation_id IS NULL
            AND ack_host_registration_lookup_digest IS NULL
            AND ack_request_mac IS NULL
            AND ack_receipt_version IS NULL AND ack_receipt_enc IS NULL
            AND ack_receipt_issuer_key_id IS NULL AND ack_receipt_digest IS NULL
        )
        OR
        -- A materialized ancestor discharged by the current head.
        (
            state = 2 AND revision = 3
            AND signed_instruction_payload_version IS NOT NULL
            AND acked_by_head_event_id IS NOT NULL
            AND acked_by_head_event_id <> uuid AND acked_at IS NOT NULL
            AND ack_operation_id IS NULL
            AND ack_host_registration_lookup_digest IS NULL
            AND ack_request_mac IS NULL
            AND ack_receipt_version IS NULL AND ack_receipt_enc IS NULL
            AND ack_receipt_issuer_key_id IS NULL AND ack_receipt_digest IS NULL
        )
        OR
        -- The exact current head stores the one replayable ACK receipt.
        (
            state = 2 AND revision = 3
            AND signed_instruction_payload_version IS NOT NULL
            AND acked_by_head_event_id IS NOT NULL
            AND acked_by_head_event_id = uuid AND acked_at IS NOT NULL
            AND ack_operation_id IS NOT NULL
            AND ack_operation_id <> '00000000-0000-0000-0000-000000000000'::uuid
            AND ack_host_registration_lookup_digest IS NOT NULL
            AND octet_length(ack_host_registration_lookup_digest) = 32
            AND ack_request_mac IS NOT NULL
            AND octet_length(ack_request_mac) = 32
            AND ack_receipt_version IS NOT NULL
            AND ack_receipt_version = 1
            AND ack_receipt_enc IS NOT NULL
            AND octet_length(ack_receipt_enc) BETWEEN 1 AND 65536
            AND ack_receipt_issuer_key_id IS NOT NULL
            AND ack_receipt_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
            AND ack_receipt_digest IS NOT NULL
            AND octet_length(ack_receipt_digest) = 32
        )
    ),
    CONSTRAINT maple_pairing_reset_clear_obligations_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_reset_lookup_unique
        UNIQUE (authority_scope_digest, reset_id, lookup_digest),
    CONSTRAINT maple_pairing_reset_clear_obligations_lookup_generation_unique
        UNIQUE (authority_scope_digest, lookup_digest, reset_generation),
    CONSTRAINT maple_pairing_reset_clear_obligations_event_scope_unique
        UNIQUE (uuid, authority_scope_digest, lookup_digest),
    CONSTRAINT maple_pairing_reset_clear_obligations_retirement_reference_unique
        UNIQUE (
            uuid,
            authority_scope_digest,
            lookup_digest,
            instruction_digest,
            chain_digest
        ),
    CONSTRAINT maple_pairing_reset_clear_obligations_chain_reference_unique
        UNIQUE (
            uuid,
            authority_scope_digest,
            lookup_digest,
            instruction_digest,
            chain_digest,
            target_revocation_stream_id,
            target_revocation_stream_generation,
            target_security_epoch
        ),
    CONSTRAINT maple_pairing_reset_clear_obligations_old_namespace_fk
        FOREIGN KEY (
            authority_scope_digest,
            lookup_digest,
            old_revocation_stream_id,
            old_revocation_stream_generation,
            source_security_epoch
        ) REFERENCES maple_pairing_revocation_highwaters(
            authority_scope_digest,
            lookup_digest,
            revocation_stream_id,
            revocation_stream_generation,
            security_epoch
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_target_namespace_fk
        FOREIGN KEY (
            authority_scope_digest,
            lookup_digest,
            target_revocation_stream_id,
            target_revocation_stream_generation,
            target_security_epoch
        ) REFERENCES maple_pairing_revocation_highwaters(
            authority_scope_digest,
            lookup_digest,
            revocation_stream_id,
            revocation_stream_generation,
            security_epoch
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_previous_fk
        FOREIGN KEY (
            previous_event_id,
            authority_scope_digest,
            lookup_digest,
            previous_instruction_digest,
            previous_chain_digest,
            old_revocation_stream_id,
            old_revocation_stream_generation,
            source_security_epoch
        ) REFERENCES maple_pairing_reset_clear_obligations(
            uuid,
            authority_scope_digest,
            lookup_digest,
            instruction_digest,
            chain_digest,
            target_revocation_stream_id,
            target_revocation_stream_generation,
            target_security_epoch
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_acked_by_fk
        FOREIGN KEY (acked_by_head_event_id, authority_scope_digest, lookup_digest)
        REFERENCES maple_pairing_reset_clear_obligations(
            uuid,
            authority_scope_digest,
            lookup_digest
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_signed_instruction_issuer_fk
        FOREIGN KEY (signed_instruction_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_sync_issuer_fk
        FOREIGN KEY (sync_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_obligations_ack_receipt_issuer_fk
        FOREIGN KEY (ack_receipt_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX idx_maple_pairing_reset_clear_no_forks
    ON maple_pairing_reset_clear_obligations(previous_event_id)
    WHERE previous_event_id IS NOT NULL;
CREATE UNIQUE INDEX idx_maple_pairing_reset_clear_ack_operation
    ON maple_pairing_reset_clear_obligations(
        authority_scope_digest,
        ack_host_registration_lookup_digest,
        ack_operation_id
    )
    WHERE ack_operation_id IS NOT NULL;
CREATE INDEX idx_maple_pairing_reset_clear_obligations_scope
    ON maple_pairing_reset_clear_obligations(authority_scope_digest, id);
CREATE INDEX idx_maple_pairing_reset_clear_obligations_current
    ON maple_pairing_reset_clear_obligations(
        authority_scope_digest,
        lookup_digest,
        state,
        reset_generation DESC,
        id DESC
    );
-- Re-enrollment must reject reuse of a retired device identity even under a
-- fresh installation lookup. This remains non-unique because every missed
-- reset successor deliberately repeats the exact retained host identity.
CREATE INDEX idx_maple_pairing_reset_clear_obligations_identity
    ON maple_pairing_reset_clear_obligations(
        authority_scope_digest,
        host_identity_mac,
        reset_generation DESC,
        id DESC
    );
CREATE INDEX idx_maple_pairing_reset_clear_obligations_acked_by
    ON maple_pairing_reset_clear_obligations(acked_by_head_event_id)
    WHERE acked_by_head_event_id IS NOT NULL;

CREATE TRIGGER update_maple_pairing_reset_clear_obligations_updated_at
    BEFORE UPDATE ON maple_pairing_reset_clear_obligations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE FUNCTION enforce_maple_pairing_reset_clear_obligation_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.state <> 1 OR NEW.revision <> 1 THEN
            RAISE EXCEPTION 'new reset-clear obligation must be pending revision 1';
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.state <> 2 THEN
            RAISE EXCEPTION 'pending reset-clear obligation cannot be deleted';
        END IF;
        RETURN OLD;
    END IF;

    -- Only lazy signed-material publication and the one terminal chain ACK
    -- may change an obligation. Every reset/namespace/identity/admission fact
    -- is append-only and remains authenticated until final clean deletion.
    IF (to_jsonb(NEW) - ARRAY[
            'signed_instruction_payload_version',
            'signed_instruction_payload_enc',
            'signed_instruction_issuer_key_id',
            'signed_instruction_digest',
            'sync_payload_version',
            'sync_payload_enc',
            'sync_issuer_key_id',
            'sync_digest',
            'state',
            'revision',
            'acked_by_head_event_id',
            'acked_at',
            'ack_operation_id',
            'ack_host_registration_lookup_digest',
            'ack_request_mac',
            'ack_receipt_version',
            'ack_receipt_enc',
            'ack_receipt_issuer_key_id',
            'ack_receipt_digest',
            'record_mac',
            'updated_at'
        ]::TEXT[])
       IS DISTINCT FROM
       (to_jsonb(OLD) - ARRAY[
            'signed_instruction_payload_version',
            'signed_instruction_payload_enc',
            'signed_instruction_issuer_key_id',
            'signed_instruction_digest',
            'sync_payload_version',
            'sync_payload_enc',
            'sync_issuer_key_id',
            'sync_digest',
            'state',
            'revision',
            'acked_by_head_event_id',
            'acked_at',
            'ack_operation_id',
            'ack_host_registration_lookup_digest',
            'ack_request_mac',
            'ack_receipt_version',
            'ack_receipt_enc',
            'ack_receipt_issuer_key_id',
            'ack_receipt_digest',
            'record_mac',
            'updated_at'
        ]::TEXT[]) THEN
        RAISE EXCEPTION 'reset-clear obligation identity is immutable';
    END IF;

    IF OLD.state = 1 AND OLD.revision = 1 THEN
        IF NOT (
            (NEW.state = 1 AND NEW.revision = 2)
            OR (NEW.state = 2 AND NEW.revision = 2)
        ) THEN
            RAISE EXCEPTION 'invalid reset-clear obligation revision-1 transition';
        END IF;
    ELSIF OLD.state = 1 AND OLD.revision = 2 THEN
        IF NEW.state <> 2 OR NEW.revision <> 3 THEN
            RAISE EXCEPTION 'materialized reset-clear obligation permits only terminal ACK';
        END IF;
        IF NEW.signed_instruction_payload_version IS DISTINCT FROM OLD.signed_instruction_payload_version
           OR NEW.signed_instruction_payload_enc IS DISTINCT FROM OLD.signed_instruction_payload_enc
           OR NEW.signed_instruction_issuer_key_id IS DISTINCT FROM OLD.signed_instruction_issuer_key_id
           OR NEW.signed_instruction_digest IS DISTINCT FROM OLD.signed_instruction_digest
           OR NEW.sync_payload_version IS DISTINCT FROM OLD.sync_payload_version
           OR NEW.sync_payload_enc IS DISTINCT FROM OLD.sync_payload_enc
           OR NEW.sync_issuer_key_id IS DISTINCT FROM OLD.sync_issuer_key_id
           OR NEW.sync_digest IS DISTINCT FROM OLD.sync_digest THEN
            RAISE EXCEPTION 'materialized reset-clear publication is immutable during ACK';
        END IF;
    ELSE
        RAISE EXCEPTION 'acked reset-clear obligation is immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_reset_clear_obligation_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_reset_clear_obligations
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_reset_clear_obligation_mutation();

-- Registration-operation history is immutable while its account authority
-- head exists. The only permitted delete is the verified-clean final account
-- teardown, after reset obligations and live registration operations have
-- already been consumed in dependency order. Define this function only after
-- every relation referenced by its body exists so migration-time validation
-- cannot depend on deferred name resolution.
CREATE OR REPLACE FUNCTION enforce_maple_pairing_registration_tombstone_mutation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'Maple registration operation tombstone is immutable';
    ELSIF TG_OP = 'INSERT' THEN
        IF NOT maple_pairing_issuer_key_ids_are_canonical(
            NEW.referenced_issuer_key_ids,
            4
        ) THEN
            RETURN NEW;
        END IF;
        IF EXISTS (
            SELECT 1
              FROM unnest(NEW.referenced_issuer_key_ids) AS referenced(key_id)
             WHERE NOT EXISTS (
                SELECT 1 FROM maple_pairing_issuer_keys registered
                 WHERE registered.key_id = referenced.key_id
             )
        ) THEN
            RAISE EXCEPTION 'Maple registration tombstone references an unknown issuer key';
        END IF;
        RETURN NEW;
    END IF;

    IF EXISTS (
        SELECT 1 FROM maple_pairing_reset_clear_obligations
         WHERE authority_scope_digest = OLD.authority_scope_digest
    ) OR EXISTS (
        SELECT 1 FROM maple_device_registration_operations
         WHERE authority_scope_digest = OLD.authority_scope_digest
    ) THEN
        RAISE EXCEPTION 'Maple registration operation tombstone deletion is out of order';
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_registration_tombstone_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_registration_operation_tombstones
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_registration_tombstone_mutation();

CREATE TABLE maple_pairing_reset_clear_admissions (
    id BIGSERIAL PRIMARY KEY,
    obligation_uuid UUID NOT NULL,
    authority_scope_digest BYTEA NOT NULL,
    lookup_digest BYTEA NOT NULL,
    pair_id UUID NOT NULL,
    pairing_incarnation BIGINT NOT NULL,
    pair_authorization_digest BYTEA NOT NULL,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_reset_clear_admissions_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_reset_clear_admissions_uuid_non_nil CHECK (
        obligation_uuid <> '00000000-0000-0000-0000-000000000000'::uuid
        AND pair_id <> '00000000-0000-0000-0000-000000000000'::uuid
    ),
    CONSTRAINT maple_pairing_reset_clear_admissions_shape CHECK (
        octet_length(authority_scope_digest) = 32
        AND octet_length(lookup_digest) = 32
        AND pairing_incarnation > 0
        AND octet_length(pair_authorization_digest) = 32
        AND octet_length(record_mac) = 32
    ),
    CONSTRAINT maple_pairing_reset_clear_admissions_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_admissions_obligation_fk
        FOREIGN KEY (obligation_uuid, authority_scope_digest, lookup_digest)
        REFERENCES maple_pairing_reset_clear_obligations(
            uuid,
            authority_scope_digest,
            lookup_digest
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_reset_clear_admissions_identity_unique
        UNIQUE (obligation_uuid, pair_id, pairing_incarnation)
);

CREATE INDEX idx_maple_pairing_reset_clear_admissions_scope
    ON maple_pairing_reset_clear_admissions(authority_scope_digest, id);
CREATE INDEX idx_maple_pairing_reset_clear_admissions_canonical
    ON maple_pairing_reset_clear_admissions(
        authority_scope_digest,
        obligation_uuid,
        pair_id,
        pairing_incarnation
    );

CREATE OR REPLACE FUNCTION enforce_maple_pairing_reset_clear_admission_mutation()
RETURNS TRIGGER AS $$
DECLARE
    obligation_state SMALLINT;
    obligation_revision BIGINT;
    obligation_materialized BOOLEAN;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'reset-clear admission is append-only';
    END IF;

    IF TG_OP = 'INSERT' THEN
        SELECT state, revision, signed_instruction_payload_version IS NOT NULL
          INTO obligation_state, obligation_revision, obligation_materialized
          FROM maple_pairing_reset_clear_obligations
         WHERE uuid = NEW.obligation_uuid
           AND authority_scope_digest = NEW.authority_scope_digest
           AND lookup_digest = NEW.lookup_digest
         FOR KEY SHARE;
        IF NOT FOUND
           OR obligation_state <> 1
           OR obligation_revision <> 1
           OR obligation_materialized THEN
            RAISE EXCEPTION 'reset-clear admission requires its unsigned pending obligation';
        END IF;
        RETURN NEW;
    END IF;

    SELECT state
      INTO obligation_state
      FROM maple_pairing_reset_clear_obligations
     WHERE uuid = OLD.obligation_uuid
       AND authority_scope_digest = OLD.authority_scope_digest
       AND lookup_digest = OLD.lookup_digest
     FOR KEY SHARE;
    IF NOT FOUND OR obligation_state <> 2 THEN
        RAISE EXCEPTION 'reset-clear admission cannot be deleted before terminal ACK';
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_reset_clear_admission_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_reset_clear_admissions
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_reset_clear_admission_mutation();

-- Terminal proof that an acknowledged enrollment lineage may never regain
-- authority. Both selectors are pseudonymous keyed digests: `lookup_digest`
-- identifies the retired installation-instance lineage, while
-- `host_identity_mac` fences reuse of the retained endpoint identity under a
-- genuinely fresh installation identifier. The keyed host-registration
-- selector scopes the retained ACK operation namespace without retaining the
-- raw registration UUID. The final obligation remains the canonical encrypted
-- ACK receipt store; this row binds its exact outcome into the account
-- inventory without retaining a raw operation identifier.
CREATE TABLE maple_pairing_installation_retirements (
    id BIGSERIAL PRIMARY KEY,
    authority_scope_digest BYTEA NOT NULL,
    lookup_digest BYTEA NOT NULL,
    host_identity_mac BYTEA NOT NULL,
    retired_security_epoch BIGINT NOT NULL,
    final_obligation_event_id UUID NOT NULL,
    final_instruction_digest BYTEA NOT NULL,
    final_chain_digest BYTEA NOT NULL,
    ack_host_registration_lookup_digest BYTEA NOT NULL,
    ack_operation_lookup_digest BYTEA NOT NULL,
    ack_request_mac BYTEA NOT NULL,
    ack_receipt_version SMALLINT NOT NULL,
    ack_receipt_issuer_key_id TEXT NOT NULL,
    ack_receipt_digest BYTEA NOT NULL,
    retired_at TIMESTAMPTZ NOT NULL,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_pairing_installation_retirements_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_installation_retirements_shape CHECK (
        octet_length(authority_scope_digest) = 32
        AND octet_length(lookup_digest) = 32
        AND octet_length(host_identity_mac) = 32
        AND retired_security_epoch > 0
        AND final_obligation_event_id <> '00000000-0000-0000-0000-000000000000'::uuid
        AND octet_length(final_instruction_digest) = 32
        AND octet_length(final_chain_digest) = 32
        AND octet_length(ack_host_registration_lookup_digest) = 32
        AND octet_length(ack_operation_lookup_digest) = 32
        AND octet_length(ack_request_mac) = 32
        AND ack_receipt_version = 1
        AND ack_receipt_issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'
        AND octet_length(ack_receipt_digest) = 32
        AND octet_length(record_mac) = 32
        AND created_at = retired_at
    ),
    CONSTRAINT maple_pairing_installation_retirements_scope_fk
        FOREIGN KEY (authority_scope_digest)
        REFERENCES maple_pairing_authority_account_heads(authority_scope_digest)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_installation_retirements_lookup_unique
        UNIQUE (authority_scope_digest, lookup_digest),
    CONSTRAINT maple_pairing_installation_retirements_identity_unique
        UNIQUE (authority_scope_digest, host_identity_mac),
    CONSTRAINT maple_pairing_installation_retirements_host_registration_unique
        UNIQUE (authority_scope_digest, ack_host_registration_lookup_digest),
    CONSTRAINT maple_pairing_installation_retirements_ack_operation_unique
        UNIQUE (authority_scope_digest, ack_operation_lookup_digest),
    CONSTRAINT maple_pairing_installation_retirements_final_obligation_fk
        FOREIGN KEY (
            final_obligation_event_id,
            authority_scope_digest,
            lookup_digest,
            final_instruction_digest,
            final_chain_digest
        ) REFERENCES maple_pairing_reset_clear_obligations(
            uuid,
            authority_scope_digest,
            lookup_digest,
            instruction_digest,
            chain_digest
        ) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_installation_retirements_ack_receipt_issuer_fk
        FOREIGN KEY (ack_receipt_issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_installation_retirements_scope
    ON maple_pairing_installation_retirements(authority_scope_digest, id);
CREATE INDEX idx_maple_pairing_installation_retirements_lookup
    ON maple_pairing_installation_retirements(authority_scope_digest, lookup_digest);
CREATE INDEX idx_maple_pairing_installation_retirements_identity
    ON maple_pairing_installation_retirements(authority_scope_digest, host_identity_mac);

CREATE OR REPLACE FUNCTION enforce_maple_pairing_installation_retirement_mutation()
RETURNS TRIGGER AS $$
DECLARE
    obligation_state SMALLINT;
    obligation_revision BIGINT;
    obligation_ack_head UUID;
    obligation_ack_host_registration_lookup_digest BYTEA;
    obligation_ack_request_mac BYTEA;
    obligation_ack_receipt_version SMALLINT;
    obligation_ack_receipt_issuer_key_id TEXT;
    obligation_ack_receipt_digest BYTEA;
    obligation_target_epoch BIGINT;
    obligation_acked_at TIMESTAMPTZ;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'Maple installation retirement is immutable';
    ELSIF TG_OP = 'DELETE' THEN
        IF EXISTS (
            SELECT 1 FROM maple_pairing_reset_clear_obligations
             WHERE authority_scope_digest = OLD.authority_scope_digest
        ) OR EXISTS (
            SELECT 1 FROM maple_pairing_registration_operation_tombstones
             WHERE authority_scope_digest = OLD.authority_scope_digest
        ) THEN
            RAISE EXCEPTION 'Maple installation retirement deletion is out of order';
        END IF;
        RETURN OLD;
    END IF;

    SELECT state, revision, acked_by_head_event_id,
           ack_host_registration_lookup_digest, ack_request_mac,
           ack_receipt_version, ack_receipt_issuer_key_id, ack_receipt_digest,
           target_security_epoch, acked_at
      INTO obligation_state, obligation_revision, obligation_ack_head,
           obligation_ack_host_registration_lookup_digest,
           obligation_ack_request_mac, obligation_ack_receipt_version,
           obligation_ack_receipt_issuer_key_id, obligation_ack_receipt_digest,
           obligation_target_epoch, obligation_acked_at
      FROM maple_pairing_reset_clear_obligations
     WHERE uuid = NEW.final_obligation_event_id
       AND authority_scope_digest = NEW.authority_scope_digest
       AND lookup_digest = NEW.lookup_digest
       AND instruction_digest = NEW.final_instruction_digest
       AND chain_digest = NEW.final_chain_digest
     FOR KEY SHARE;
    IF NOT FOUND
       OR obligation_state <> 2
       OR obligation_revision <> 3
       OR obligation_ack_head IS DISTINCT FROM NEW.final_obligation_event_id
       OR obligation_ack_host_registration_lookup_digest
          IS DISTINCT FROM NEW.ack_host_registration_lookup_digest
       OR obligation_ack_request_mac IS DISTINCT FROM NEW.ack_request_mac
       OR obligation_ack_receipt_version IS DISTINCT FROM NEW.ack_receipt_version
       OR obligation_ack_receipt_issuer_key_id IS DISTINCT FROM NEW.ack_receipt_issuer_key_id
       OR obligation_ack_receipt_digest IS DISTINCT FROM NEW.ack_receipt_digest
       OR obligation_target_epoch IS DISTINCT FROM NEW.retired_security_epoch
       OR obligation_acked_at IS DISTINCT FROM NEW.retired_at THEN
        RAISE EXCEPTION 'Maple installation retirement requires the exact terminal ACK head';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, public;

CREATE TRIGGER guard_maple_pairing_installation_retirement_mutation
    BEFORE INSERT OR UPDATE OR DELETE ON maple_pairing_installation_retirements
    FOR EACH ROW EXECUTE FUNCTION enforce_maple_pairing_installation_retirement_mutation();

CREATE TABLE maple_pairing_revocation_events (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    project_id INTEGER NOT NULL,
    recipient_host_maple_device_id BIGINT NOT NULL,
    revocation_stream_id UUID NOT NULL,
    revocation_stream_generation BIGINT NOT NULL,
    issuer_sequence BIGINT NOT NULL,
    maple_pairing_id BIGINT NOT NULL,
    pairing_incarnation BIGINT NOT NULL,
    issuer_key_id TEXT NOT NULL,
    payload_version SMALLINT NOT NULL DEFAULT 1,
    payload_enc BYTEA NOT NULL,
    event_digest BYTEA NOT NULL,
    record_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acked_at TIMESTAMPTZ,
    CONSTRAINT maple_pairing_revocation_events_id_positive CHECK (id > 0),
    CONSTRAINT maple_pairing_revocation_events_uuid_non_nil
        CHECK (uuid <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_revocation_events_stream_id_non_nil
        CHECK (revocation_stream_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_pairing_revocation_events_stream_generation_positive
        CHECK (revocation_stream_generation > 0),
    CONSTRAINT maple_pairing_revocation_events_sequence_positive CHECK (issuer_sequence > 0),
    CONSTRAINT maple_pairing_revocation_events_payload_version_v1 CHECK (payload_version = 1),
    CONSTRAINT maple_pairing_revocation_events_incarnation_positive CHECK (pairing_incarnation > 0),
    CONSTRAINT maple_pairing_revocation_events_issuer_key_id_v1
        CHECK (issuer_key_id ~ '^[a-z0-9._:-]{1,64}$'),
    CONSTRAINT maple_pairing_revocation_events_digest_length
        CHECK (octet_length(event_digest) = 32),
    CONSTRAINT maple_pairing_revocation_events_record_mac_length
        CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_pairing_revocation_events_payload_bounded
        CHECK (octet_length(payload_enc) <= 32768),
    CONSTRAINT maple_pairing_revocation_events_issuer_fk
        FOREIGN KEY (issuer_key_id)
        REFERENCES maple_pairing_issuer_keys(key_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_revocation_events_host_sequence_unique
        UNIQUE (
            recipient_host_maple_device_id,
            revocation_stream_id,
            revocation_stream_generation,
            issuer_sequence
        ),
    CONSTRAINT maple_pairing_revocation_events_user_scope_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES users(uuid, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_revocation_events_host_device_fk
        FOREIGN KEY (recipient_host_maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT maple_pairing_revocation_events_pairing_fk
        FOREIGN KEY (
            maple_pairing_id,
            user_id,
            project_id,
            recipient_host_maple_device_id,
            pairing_incarnation,
            revocation_stream_id,
            revocation_stream_generation
        )
        REFERENCES maple_pairings(
            id,
            user_id,
            project_id,
            host_maple_device_id,
            pairing_incarnation,
            revocation_stream_id,
            revocation_stream_generation
        )
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_maple_pairing_revocations_host_sequence
    ON maple_pairing_revocation_events(
        user_id,
        project_id,
        recipient_host_maple_device_id,
        revocation_stream_id,
        revocation_stream_generation,
        issuer_sequence ASC
    );
CREATE INDEX idx_maple_pairing_revocations_project_id
    ON maple_pairing_revocation_events(project_id);

ALTER TABLE maple_pairing_lineages
    ADD CONSTRAINT maple_pairing_lineages_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE maple_pairings
    ADD CONSTRAINT maple_pairings_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE maple_pairing_operations
    ADD CONSTRAINT maple_pairing_operations_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE maple_pairing_host_states
    ADD CONSTRAINT maple_pairing_host_states_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE maple_pairing_revocation_events
    ADD CONSTRAINT maple_pairing_revocation_events_authority_account_fk
        FOREIGN KEY (user_id, project_id)
        REFERENCES maple_pairing_authority_account_heads(user_id, project_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;

CREATE TRIGGER guard_maple_pairing_lineages_truncate BEFORE TRUNCATE ON maple_pairing_lineages
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairings_truncate BEFORE TRUNCATE ON maple_pairings
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_operations_truncate BEFORE TRUNCATE ON maple_pairing_operations
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_host_states_truncate BEFORE TRUNCATE ON maple_pairing_host_states
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_highwaters_truncate BEFORE TRUNCATE ON maple_pairing_revocation_highwaters
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_events_truncate BEFORE TRUNCATE ON maple_pairing_revocation_events
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_registration_tombstones_truncate
    BEFORE TRUNCATE ON maple_pairing_registration_operation_tombstones
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_installation_retirements_truncate
    BEFORE TRUNCATE ON maple_pairing_installation_retirements
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_reset_clear_obligations_truncate
    BEFORE TRUNCATE ON maple_pairing_reset_clear_obligations
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_pairing_reset_clear_admissions_truncate
    BEFORE TRUNCATE ON maple_pairing_reset_clear_admissions
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_devices_truncate BEFORE TRUNCATE ON maple_devices
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
CREATE TRIGGER guard_maple_device_operations_truncate BEFORE TRUNCATE ON maple_device_registration_operations
    EXECUTE FUNCTION forbid_maple_pairing_authority_truncate();
