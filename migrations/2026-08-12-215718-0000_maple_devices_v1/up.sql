CREATE TABLE maple_devices (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES org_projects(id) ON DELETE CASCADE,
    device_id UUID NOT NULL,
    installation_id UUID NOT NULL,
    identity_mac BYTEA NOT NULL,
    endpoint_epoch BIGINT NOT NULL,
    payload_version SMALLINT NOT NULL DEFAULT 1,
    payload_enc BYTEA NOT NULL,
    record_mac BYTEA NOT NULL,
    revision BIGINT NOT NULL DEFAULT 1,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_devices_identity_mac_length
        CHECK (octet_length(identity_mac) = 32),
    CONSTRAINT maple_devices_record_mac_length
        CHECK (octet_length(record_mac) = 32),
    CONSTRAINT maple_devices_payload_enc_bounded
        CHECK (octet_length(payload_enc) <= 16384),
    CONSTRAINT maple_devices_payload_version_v1 CHECK (payload_version = 1),
    CONSTRAINT maple_devices_endpoint_epoch_nonnegative CHECK (endpoint_epoch >= 0),
    CONSTRAINT maple_devices_revision_positive CHECK (revision > 0),
    CONSTRAINT maple_devices_id_positive CHECK (id > 0),
    CONSTRAINT maple_devices_uuid_non_nil
        CHECK (uuid <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_devices_device_id_non_nil
        CHECK (device_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_devices_installation_id_non_nil
        CHECK (installation_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_devices_user_project_device_unique
        UNIQUE (user_id, project_id, device_id),
    CONSTRAINT maple_devices_user_project_installation_unique
        UNIQUE (user_id, project_id, installation_id),
    CONSTRAINT maple_devices_user_project_identity_unique
        UNIQUE (user_id, project_id, identity_mac),
    CONSTRAINT maple_devices_id_scope_unique
        UNIQUE (id, user_id, project_id)
);

CREATE INDEX idx_maple_devices_user_project_uuid
    ON maple_devices(user_id, project_id, uuid DESC);
CREATE INDEX idx_maple_devices_project_id ON maple_devices(project_id);

CREATE TRIGGER update_maple_devices_updated_at
    BEFORE UPDATE ON maple_devices
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE maple_device_registration_operations (
    id BIGSERIAL PRIMARY KEY,
    operation_id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES org_projects(id) ON DELETE CASCADE,
    request_mac BYTEA NOT NULL,
    maple_device_id BIGINT NOT NULL,
    device_revision BIGINT NOT NULL,
    receipt_mac BYTEA NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT maple_device_registration_operations_request_mac_length
        CHECK (octet_length(request_mac) = 32),
    CONSTRAINT maple_device_registration_operations_receipt_mac_length
        CHECK (octet_length(receipt_mac) = 32),
    CONSTRAINT maple_device_registration_operations_revision_positive
        CHECK (device_revision > 0),
    CONSTRAINT maple_device_registration_operations_id_positive CHECK (id > 0),
    CONSTRAINT maple_device_registration_operations_operation_id_non_nil
        CHECK (operation_id <> '00000000-0000-0000-0000-000000000000'::uuid),
    CONSTRAINT maple_device_registration_operations_scope_unique
        UNIQUE (user_id, project_id, operation_id),
    CONSTRAINT maple_device_registration_operations_device_revision_unique
        UNIQUE (maple_device_id, device_revision),
    CONSTRAINT maple_device_registration_operations_scoped_device_fk
        FOREIGN KEY (maple_device_id, user_id, project_id)
        REFERENCES maple_devices(id, user_id, project_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_maple_device_registration_operations_device
    ON maple_device_registration_operations(maple_device_id);
CREATE INDEX idx_maple_device_registration_operations_project_id
    ON maple_device_registration_operations(project_id);
