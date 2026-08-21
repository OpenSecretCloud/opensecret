DROP INDEX IF EXISTS idx_maple_device_registration_operations_project_id;
DROP INDEX IF EXISTS idx_maple_device_registration_operations_device;
DROP TABLE IF EXISTS maple_device_registration_operations;

DROP TRIGGER IF EXISTS update_maple_devices_updated_at ON maple_devices;
DROP INDEX IF EXISTS idx_maple_devices_project_id;
DROP INDEX IF EXISTS idx_maple_devices_user_project_uuid;
DROP TABLE IF EXISTS maple_devices;
