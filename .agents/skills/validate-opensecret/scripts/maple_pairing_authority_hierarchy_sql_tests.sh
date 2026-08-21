#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
export LC_ALL=C

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fixture="$script_dir/maple_pairing_authority_hierarchy_case.sql"
database_url="${AEAD_TAMPER_TEST_DATABASE_URL:?AEAD_TAMPER_TEST_DATABASE_URL is required}"
log_dir="${MAPLE_AUTHORITY_SQL_TEST_LOG_DIR:?MAPLE_AUTHORITY_SQL_TEST_LOG_DIR is required}"

if [ ! -f "$fixture" ] || [ ! -d "$log_dir" ] || [ -L "$log_dir" ]; then
  printf 'invalid Maple authority SQL fixture or log directory\n' >&2
  exit 1
fi

passed=0

expect_success() {
  local test_case="$1"
  local log="$log_dir/maple-authority-${test_case}.log"
  if ! psql "$database_url" -X --quiet -v ON_ERROR_STOP=1 \
    -v test_case="$test_case" -f "$fixture" >"$log" 2>&1; then
    printf 'Maple authority SQL case unexpectedly failed: %s\n' "$test_case" >&2
    sed -n '1,160p' "$log" >&2
    return 1
  fi
  passed=$((passed + 1))
}

expect_failure() {
  local test_case="$1"
  local expected_pattern="$2"
  local log="$log_dir/maple-authority-${test_case}.log"
  if psql "$database_url" -X --quiet -v ON_ERROR_STOP=1 \
    -v test_case="$test_case" -f "$fixture" >"$log" 2>&1; then
    printf 'Maple authority SQL case unexpectedly succeeded: %s\n' "$test_case" >&2
    return 1
  fi
  if ! grep -Eq "$expected_pattern" "$log"; then
    printf 'Maple authority SQL case failed for the wrong reason: %s\n' "$test_case" >&2
    sed -n '1,160p' "$log" >&2
    return 1
  fi
  passed=$((passed + 1))
}

expect_success activation_complete
expect_failure activation_missing_head \
  'active Maple pairing authority hierarchy is incomplete'
expect_success project_mutable_updates
expect_success valid_scoped_lifecycle
expect_failure parent_head_mismatch \
  'active Maple pairing organization authority is incomplete'
expect_failure head_parent_mismatch \
  'active Maple pairing organization authority is incomplete'
expect_failure project_parent_move_mismatch \
  'active Maple pairing project identity cannot be replaced'
expect_failure project_internal_id_mutation \
  'active Maple pairing project identity cannot be replaced'
expect_failure project_uuid_mutation \
  'active Maple pairing project identity cannot be replaced'
expect_failure project_client_id_mutation \
  'active Maple pairing project identity cannot be replaced'
expect_failure project_head_alias_mutation \
  'Maple pairing project head identity is immutable'
expect_failure project_head_identity_mismatch \
  'violates foreign key constraint "maple_pairing_authority_project_scope_fk"'
expect_failure project_alias_reinsert \
  'active Maple pairing project identity cannot be replaced'
expect_failure missing_ancestor \
  'active Maple pairing project ancestry is incomplete'
expect_failure active_marker_delete \
  'active Maple pairing authority marker is immutable'
expect_failure active_root_delete \
  'Maple pairing authority root cannot be removed'
expect_failure active_root_downgrade \
  'active Maple pairing authority root cannot be downgraded'
expect_failure truncate_guard \
  'TRUNCATE of Maple pairing authority state is forbidden'
expect_failure tombstone_null_issuer_key_id \
  'violates check constraint "maple_pairing_registration_operation_tombstones_receipt_shape"'
expect_failure tombstone_unknown_issuer_key_id \
  'Maple registration tombstone references an unknown issuer key'
expect_success steady_state_scoped_no_global_scan

printf '%s\n' "$passed"
