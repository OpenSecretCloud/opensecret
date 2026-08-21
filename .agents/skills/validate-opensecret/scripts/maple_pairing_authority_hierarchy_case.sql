\set ON_ERROR_STOP on

SELECT :'test_case' IN (
    'activation_complete',
    'activation_missing_head',
    'project_mutable_updates',
    'valid_scoped_lifecycle',
    'parent_head_mismatch',
    'head_parent_mismatch',
    'project_parent_move_mismatch',
    'project_internal_id_mutation',
    'project_uuid_mutation',
    'project_client_id_mutation',
    'project_head_alias_mutation',
    'project_head_identity_mismatch',
    'project_alias_reinsert',
    'missing_ancestor',
    'active_marker_delete',
    'active_root_delete',
    'active_root_downgrade',
    'truncate_guard',
    'tombstone_null_issuer_key_id',
    'tombstone_unknown_issuer_key_id',
    'steady_state_scoped_no_global_scan'
) AS known_case,
:'test_case' = 'activation_missing_head' AS activation_missing_head,
:'test_case' = 'project_mutable_updates' AS project_mutable_updates,
:'test_case' = 'valid_scoped_lifecycle' AS valid_scoped_lifecycle,
:'test_case' = 'parent_head_mismatch' AS parent_head_mismatch,
:'test_case' = 'head_parent_mismatch' AS head_parent_mismatch,
:'test_case' = 'project_parent_move_mismatch' AS project_parent_move_mismatch,
:'test_case' = 'project_internal_id_mutation' AS project_internal_id_mutation,
:'test_case' = 'project_uuid_mutation' AS project_uuid_mutation,
:'test_case' = 'project_client_id_mutation' AS project_client_id_mutation,
:'test_case' = 'project_head_alias_mutation' AS project_head_alias_mutation,
:'test_case' = 'project_head_identity_mismatch' AS project_head_identity_mismatch,
:'test_case' = 'project_alias_reinsert' AS project_alias_reinsert,
:'test_case' = 'missing_ancestor' AS missing_ancestor,
:'test_case' = 'active_marker_delete' AS active_marker_delete,
:'test_case' = 'active_root_delete' AS active_root_delete,
:'test_case' = 'active_root_downgrade' AS active_root_downgrade,
:'test_case' = 'truncate_guard' AS truncate_guard,
:'test_case' = 'tombstone_null_issuer_key_id' AS tombstone_null_issuer_key_id,
:'test_case' = 'tombstone_unknown_issuer_key_id' AS tombstone_unknown_issuer_key_id,
:'test_case' = 'steady_state_scoped_no_global_scan' AS steady_state_scoped_no_global_scan
\gset

\if :known_case
\else
    \echo unknown Maple pairing authority SQL test case: :test_case
    \quit 3
\endif

BEGIN;

-- SQL migrations cannot compute enclave MACs. These structurally valid
-- 32-byte placeholders exercise only the relational trigger boundary; the
-- Rust authority audit separately authenticates every head before use.
INSERT INTO maple_pairing_authority_org_heads (
    org_id,
    project_inventory_digest,
    project_count,
    revision,
    record_mac
)
SELECT
    id,
    decode(repeat('11', 32), 'hex'),
    (SELECT count(*) FROM org_projects WHERE org_id = orgs.id),
    1,
    decode(repeat('12', 32), 'hex')
FROM orgs;

\if :activation_missing_head
\else
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    )
    SELECT
        p.id,
        p.org_id,
        p.uuid,
        p.client_id,
        decode(repeat('21', 32), 'hex'),
        (SELECT count(*) FROM users WHERE project_id = p.id),
        1,
        decode(repeat('22', 32), 'hex')
    FROM org_projects p;
\endif

\if :activation_missing_head
\else
    INSERT INTO maple_pairing_authority_account_heads (
        user_id,
        project_id,
        org_id,
        authority_scope_digest,
        authority_inventory_digest,
        revision,
        record_mac
    )
    SELECT
        u.uuid,
        u.project_id,
        p.org_id,
        decode(replace(u.uuid::text, '-', ''), 'hex')
            || decode(replace(u.uuid::text, '-', ''), 'hex'),
        decode(repeat('31', 32), 'hex'),
        1,
        decode(repeat('32', 32), 'hex')
    FROM users u
    JOIN org_projects p ON p.id = u.project_id;
\endif

-- The marker must be inserted while the singleton is Pending. Its deferred
-- event observes the final Active state when constraints are flushed.
INSERT INTO app_data_migrations (name)
VALUES ('maple_pairing_authority_v1_activated');

UPDATE maple_pairing_authority_global_heads
SET activation_state = 2,
    org_inventory_digest = decode(repeat('41', 32), 'hex'),
    org_count = (SELECT count(*) FROM orgs),
    revision = 2,
    record_mac = decode(repeat('42', 32), 'hex')
WHERE singleton;

-- This is both the successful activation proof and the expected failure point
-- for activation_missing_head.
SET CONSTRAINTS ALL IMMEDIATE;

SELECT o.id AS base_org_id, p.id AS base_project_id
FROM orgs o
JOIN org_projects p ON p.org_id = o.id
ORDER BY o.id, p.id
LIMIT 1
\gset

SET CONSTRAINTS ALL DEFERRED;

\if :project_mutable_updates
    UPDATE org_projects
    SET description = 'Maple authority SQL mutable metadata update'
    WHERE id = :base_project_id;
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit DEFERRED;

    UPDATE maple_pairing_authority_project_heads
    SET account_inventory_digest = decode(repeat('43', 32), 'hex'),
        account_count = (
            SELECT count(*) FROM users WHERE project_id = :base_project_id
        ),
        revision = revision + 1,
        record_mac = decode(repeat('44', 32), 'hex')
    WHERE project_id = :base_project_id;
    SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit IMMEDIATE;
    SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit DEFERRED;
\endif

\if :valid_scoped_lifecycle
    INSERT INTO orgs (name)
    VALUES ('Maple authority SQL valid lifecycle org')
    RETURNING id AS test_org_id
    \gset

    INSERT INTO maple_pairing_authority_org_heads (
        org_id,
        project_inventory_digest,
        project_count,
        revision,
        record_mac
    ) VALUES (
        :test_org_id,
        decode(repeat('51', 32), 'hex'),
        1,
        1,
        decode(repeat('52', 32), 'hex')
    );

    INSERT INTO org_projects (org_id, name, description, status)
    VALUES (
        :test_org_id,
        'Maple authority SQL valid lifecycle project',
        'scoped hierarchy trigger regression',
        'active'
    )
    RETURNING
        id AS test_project_id,
        uuid AS test_project_uuid,
        client_id AS test_subject_project_id
    \gset

    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        :test_project_id,
        :test_org_id,
        :'test_project_uuid',
        :'test_subject_project_id',
        decode(repeat('53', 32), 'hex'),
        1,
        1,
        decode(repeat('54', 32), 'hex')
    );

    INSERT INTO users (uuid, project_id)
    VALUES ('10000000-0000-4000-8000-000000000001', :test_project_id);

    INSERT INTO maple_pairing_authority_account_heads (
        user_id,
        project_id,
        org_id,
        authority_scope_digest,
        authority_inventory_digest,
        revision,
        record_mac
    ) VALUES (
        '10000000-0000-4000-8000-000000000001',
        :test_project_id,
        :test_org_id,
        decode(repeat('55', 32), 'hex'),
        decode(repeat('56', 32), 'hex'),
        1,
        decode(repeat('57', 32), 'hex')
    );

    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    DELETE FROM maple_pairing_authority_account_heads
    WHERE user_id = '10000000-0000-4000-8000-000000000001';
    DELETE FROM users
    WHERE uuid = '10000000-0000-4000-8000-000000000001';
    DELETE FROM maple_pairing_authority_project_heads
    WHERE project_id = :test_project_id;
    DELETE FROM org_projects WHERE id = :test_project_id;
    DELETE FROM maple_pairing_authority_org_heads
    WHERE org_id = :test_org_id;
    DELETE FROM orgs WHERE id = :test_org_id;

    SET CONSTRAINTS ALL IMMEDIATE;
\endif

\if :parent_head_mismatch
    INSERT INTO orgs (name)
    VALUES ('Maple authority SQL unmatched parent');
    SET CONSTRAINTS ALL IMMEDIATE;
\endif

\if :head_parent_mismatch
    INSERT INTO maple_pairing_authority_org_heads (
        org_id,
        project_inventory_digest,
        project_count,
        revision,
        record_mac
    ) VALUES (
        2147483000,
        decode(repeat('71', 32), 'hex'),
        0,
        1,
        decode(repeat('72', 32), 'hex')
    );
    -- Fire the hierarchy constraint deterministically before the independently
    -- deferred relational FK for this intentionally parentless head.
    SET CONSTRAINTS guard_maple_pairing_authority_org_head_commit IMMEDIATE;
\endif

\if :project_parent_move_mismatch
    INSERT INTO orgs (name)
    VALUES ('Maple authority SQL project move target')
    RETURNING id AS move_org_id
    \gset
    INSERT INTO maple_pairing_authority_org_heads (
        org_id,
        project_inventory_digest,
        project_count,
        revision,
        record_mac
    ) VALUES (
        :move_org_id,
        decode(repeat('73', 32), 'hex'),
        0,
        1,
        decode(repeat('74', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    UPDATE org_projects
    SET org_id = :move_org_id
    WHERE id = :base_project_id;
    -- The OLD scope has a head but no longer has its parent; the NEW scope has
    -- a parent but no matching head. Either side must fail this row event.
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
\endif

\if :project_uuid_mutation
    UPDATE org_projects
    SET uuid = '20000000-0000-4000-8000-000000000001'
    WHERE id = :base_project_id;
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
\endif

\if :project_internal_id_mutation
    INSERT INTO org_projects (org_id, name, description, status)
    VALUES (
        :base_org_id,
        'Maple authority SQL internal ID source project',
        'fresh identity without unrelated child rows',
        'active'
    )
    RETURNING
        id AS internal_id_project_id,
        uuid AS internal_id_project_uuid,
        client_id AS internal_id_subject_project_id
    \gset
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        :internal_id_project_id,
        :base_org_id,
        :'internal_id_project_uuid',
        :'internal_id_subject_project_id',
        decode(repeat('79', 32), 'hex'),
        0,
        1,
        decode(repeat('7a', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    DELETE FROM maple_pairing_authority_project_heads
    WHERE project_id = :internal_id_project_id;
    UPDATE org_projects
    SET id = 2147482991
    WHERE id = :internal_id_project_id;
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        2147482991,
        :base_org_id,
        :'internal_id_project_uuid',
        :'internal_id_subject_project_id',
        decode(repeat('7b', 32), 'hex'),
        0,
        1,
        decode(repeat('7c', 32), 'hex')
    );
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
\endif

\if :project_client_id_mutation
    UPDATE org_projects
    SET client_id = '20000000-0000-4000-8000-000000000002'
    WHERE id = :base_project_id;
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
\endif

\if :project_head_alias_mutation
    UPDATE maple_pairing_authority_project_heads
    SET subject_project_id = '20000000-0000-4000-8000-000000000003',
        revision = revision + 1
    WHERE project_id = :base_project_id;
\endif

\if :project_head_identity_mismatch
    INSERT INTO org_projects (org_id, name, description, status)
    VALUES (
        :base_org_id,
        'Maple authority SQL mismatched head project',
        'exact composite FK regression',
        'active'
    )
    RETURNING
        id AS mismatched_head_project_id,
        uuid AS mismatched_head_project_uuid
    \gset
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        :mismatched_head_project_id,
        :base_org_id,
        :'mismatched_head_project_uuid',
        '20000000-0000-4000-8000-000000000008',
        decode(repeat('7d', 32), 'hex'),
        0,
        1,
        decode(repeat('7e', 32), 'hex')
    );
    SET CONSTRAINTS maple_pairing_authority_project_scope_fk IMMEDIATE;
\endif

\if :project_alias_reinsert
    -- Replacing both sides with a self-consistent new alias tuple must still
    -- fail. The deferred OLD event preserves the authenticated incarnation;
    -- final-state parent/head equivalence alone would miss this ABA rewrite.
    INSERT INTO org_projects (
        uuid,
        client_id,
        org_id,
        name,
        description,
        status
    ) VALUES (
        '20000000-0000-4000-8000-000000000004',
        '20000000-0000-4000-8000-000000000005',
        :base_org_id,
        'Maple authority SQL alias source project',
        'fresh identity without unrelated child rows',
        'active'
    )
    RETURNING id AS alias_project_id
    \gset
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        :alias_project_id,
        :base_org_id,
        '20000000-0000-4000-8000-000000000004',
        '20000000-0000-4000-8000-000000000005',
        decode(repeat('75', 32), 'hex'),
        0,
        1,
        decode(repeat('76', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    DELETE FROM maple_pairing_authority_project_heads
    WHERE project_id = :alias_project_id;
    DELETE FROM org_projects WHERE id = :alias_project_id;
    INSERT INTO org_projects (
        id,
        uuid,
        client_id,
        org_id,
        name,
        description,
        status
    ) VALUES (
        :alias_project_id,
        '20000000-0000-4000-8000-000000000006',
        '20000000-0000-4000-8000-000000000007',
        :base_org_id,
        'Maple authority SQL alias reinsert project',
        'same internal scope with different aliases',
        'active'
    );
    INSERT INTO maple_pairing_authority_project_heads (
        project_id,
        org_id,
        project_uuid,
        subject_project_id,
        account_inventory_digest,
        account_count,
        revision,
        record_mac
    ) VALUES (
        :alias_project_id,
        :base_org_id,
        '20000000-0000-4000-8000-000000000006',
        '20000000-0000-4000-8000-000000000007',
        decode(repeat('77', 32), 'hex'),
        0,
        1,
        decode(repeat('78', 32), 'hex')
    );
    SET CONSTRAINTS guard_maple_pairing_authority_project_parent_commit IMMEDIATE;
\endif

\if :missing_ancestor
    -- Suppress only the changed ancestor's own deferred event so the project
    -- event must independently reject its missing ancestry before pending FK
    -- checks, which remain enabled. The expected named-constraint error aborts
    -- this per-case transaction, so rollback restores the transactional
    -- trigger disable; re-enabling after DELETE would itself fail while
    -- FK/deferred events are pending.
    ALTER TABLE maple_pairing_authority_org_heads
        DISABLE TRIGGER guard_maple_pairing_authority_org_head_commit;
    DELETE FROM maple_pairing_authority_org_heads
    WHERE org_id = :base_org_id;
    UPDATE maple_pairing_authority_project_heads
    SET revision = revision + 1
    WHERE project_id = :base_project_id;
    SET CONSTRAINTS guard_maple_pairing_authority_project_head_commit IMMEDIATE;
\endif

\if :active_marker_delete
    DELETE FROM app_data_migrations
    WHERE name = 'maple_pairing_authority_v1_activated';
\endif

\if :active_root_delete
    SET LOCAL opensecret.allow_destructive_maple_pairing_down =
        'disposable-test-only';
    DELETE FROM maple_pairing_authority_global_heads WHERE singleton;
\endif

\if :active_root_downgrade
    SET LOCAL opensecret.allow_destructive_maple_pairing_down =
        'disposable-test-only';
    UPDATE maple_pairing_authority_global_heads
    SET activation_state = 1,
        org_inventory_digest = decode(repeat('00', 32), 'hex'),
        org_count = 0,
        revision = 3,
        record_mac = NULL
    WHERE singleton;
\endif

\if :truncate_guard
    TRUNCATE maple_pairing_authority_account_heads CASCADE;
\endif

\if :tombstone_null_issuer_key_id
    INSERT INTO users (uuid, project_id)
    VALUES ('10000000-0000-4000-8000-000000000003', :base_project_id);
    INSERT INTO maple_pairing_authority_account_heads (
        user_id,
        project_id,
        org_id,
        authority_scope_digest,
        authority_inventory_digest,
        revision,
        record_mac
    ) VALUES (
        '10000000-0000-4000-8000-000000000003',
        :base_project_id,
        :base_org_id,
        decode(repeat('64', 32), 'hex'),
        decode(repeat('65', 32), 'hex'),
        1,
        decode(repeat('66', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    INSERT INTO maple_pairing_registration_operation_tombstones (
        authority_scope_digest,
        lookup_digest,
        operation_lookup_digest,
        retired_security_epoch,
        request_mac,
        outcome_kind,
        outcome_digest,
        receipt_version,
        receipt_enc,
        receipt_digest,
        referenced_issuer_key_ids,
        accepted_at,
        record_mac,
        retired_at
    ) VALUES (
        decode(repeat('64', 32), 'hex'),
        decode(repeat('67', 32), 'hex'),
        decode(repeat('68', 32), 'hex'),
        1,
        decode(repeat('69', 32), 'hex'),
        1,
        decode(repeat('6a', 32), 'hex'),
        1,
        decode('01', 'hex'),
        decode(repeat('6b', 32), 'hex'),
        ARRAY['issuer-a', NULL]::TEXT[],
        CURRENT_TIMESTAMP,
        decode(repeat('6c', 32), 'hex'),
        CURRENT_TIMESTAMP
    );
\endif

\if :tombstone_unknown_issuer_key_id
    INSERT INTO users (uuid, project_id)
    VALUES ('10000000-0000-4000-8000-000000000004', :base_project_id);
    INSERT INTO maple_pairing_authority_account_heads (
        user_id,
        project_id,
        org_id,
        authority_scope_digest,
        authority_inventory_digest,
        revision,
        record_mac
    ) VALUES (
        '10000000-0000-4000-8000-000000000004',
        :base_project_id,
        :base_org_id,
        decode(repeat('6d', 32), 'hex'),
        decode(repeat('6e', 32), 'hex'),
        1,
        decode(repeat('6f', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
    SET CONSTRAINTS ALL DEFERRED;

    INSERT INTO maple_pairing_registration_operation_tombstones (
        authority_scope_digest,
        lookup_digest,
        operation_lookup_digest,
        retired_security_epoch,
        request_mac,
        outcome_kind,
        outcome_digest,
        receipt_version,
        receipt_enc,
        receipt_digest,
        referenced_issuer_key_ids,
        accepted_at,
        record_mac,
        retired_at
    ) VALUES (
        decode(repeat('6d', 32), 'hex'),
        decode(repeat('70', 32), 'hex'),
        decode(repeat('71', 32), 'hex'),
        1,
        decode(repeat('72', 32), 'hex'),
        1,
        decode(repeat('73', 32), 'hex'),
        1,
        decode('01', 'hex'),
        decode(repeat('74', 32), 'hex'),
        ARRAY['issuer-a']::TEXT[],
        CURRENT_TIMESTAMP,
        decode(repeat('75', 32), 'hex'),
        CURRENT_TIMESTAMP
    );
\endif

\if :steady_state_scoped_no_global_scan
    -- Create an unrelated gap without queuing its row-level hierarchy event.
    -- A steady-state global anti-join would now reject the valid account pair
    -- below; scoped OLD/NEW verification must not inspect this unrelated org.
    ALTER TABLE orgs
        DISABLE TRIGGER guard_maple_pairing_authority_org_parent_commit;
    INSERT INTO orgs (name)
    VALUES ('Maple authority SQL intentionally unrelated gap');
    ALTER TABLE orgs
        ENABLE TRIGGER guard_maple_pairing_authority_org_parent_commit;

    INSERT INTO users (uuid, project_id)
    VALUES (
        '10000000-0000-4000-8000-000000000002',
        :base_project_id
    );
    INSERT INTO maple_pairing_authority_account_heads (
        user_id,
        project_id,
        org_id,
        authority_scope_digest,
        authority_inventory_digest,
        revision,
        record_mac
    ) VALUES (
        '10000000-0000-4000-8000-000000000002',
        :base_project_id,
        :base_org_id,
        decode(repeat('61', 32), 'hex'),
        decode(repeat('62', 32), 'hex'),
        1,
        decode(repeat('63', 32), 'hex')
    );
    SET CONSTRAINTS ALL IMMEDIATE;
\endif

ROLLBACK;
