#!/usr/bin/env bash

set -euo pipefail

repo_root="${REPO_ROOT_UNDER_TEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

fail() {
    echo "legacy Nitro build retirement check failed: $*" >&2
    exit 1
}

test ! -e "$repo_root/Dockerfile" || fail "ambiguous root Dockerfile still exists"
test ! -e "$repo_root/nitro-bins/kmstool_enclave_cli" || fail "legacy kmstool blob is still tracked"
test ! -e "$repo_root/nitro-bins/libnsm.so" || fail "legacy libnsm blob is still tracked"
test ! -e "$repo_root/continuum-proxy" || fail "legacy Continuum proxy blob is still tracked"
test ! -e "$repo_root/continuum-proxy-x86_64" || fail "legacy x86-64 Continuum proxy blob is still tracked"

local_dockerfile="$repo_root/Dockerfile.local"
test -f "$local_dockerfile" || fail "Dockerfile.local is missing"
grep -Fq 'ENTRYPOINT ["/app/opensecret"]' "$local_dockerfile" \
    || fail "local image does not launch the backend directly"
grep -Fq 'APP_MODE=local' "$local_dockerfile" \
    || fail "local image does not pin local mode"
if grep -Eq '(enclave_base|kmstool|libnsm|entrypoint\.sh|traffic_forwarder|vsock_helper)' "$local_dockerfile"; then
    fail "local image still consumes an enclave-only helper or entrypoint"
fi

legacy_recipe="$({
    sed -n '/^build-enclave-base:/,/^build-nitro-bins:/p' "$repo_root/justfile"
} | sed '$d')"
grep -Fq '@exit 1' <<<"$legacy_recipe" \
    || fail "legacy just recipe does not fail closed"
if grep -Eq '(\{\{container\}\}|podman|docker[[:space:]]+build)' <<<"$legacy_recipe"; then
    fail "legacy just recipe still invokes a container build"
fi

grep -Fq '{{container}} build -f Dockerfile.local' "$repo_root/justfile" \
    || fail "local Docker recipe does not select Dockerfile.local explicitly"

echo "legacy Nitro helper build is retired from active root workflows"
