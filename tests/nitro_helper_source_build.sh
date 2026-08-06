#!/usr/bin/env bash

set -euo pipefail

repo_root="${REPO_ROOT_UNDER_TEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for blob in \
    "$repo_root/nitro-bins/libnsm.so" \
    "$repo_root/nitro-bins/kmstool_enclave_cli"; do
    if [ -e "$blob" ]; then
        echo "checked-in Nitro helper blob remains: $blob" >&2
        exit 1
    fi
done

if grep -Fq 'nitro-toolkit/enclave-base-image/Dockerfile' "$repo_root/flake.nix"; then
    echo "the normal Nix graph still consumes the legacy helper Dockerfile" >&2
    exit 1
fi

if grep -Eq 'reproduce-nitro-bins|write-nitro-bins' "$repo_root/flake.nix"; then
    echo "the old Podman helper reproduction apps remain exported" >&2
    exit 1
fi

grep -Fq 'nix build .#nitro-bins' "$repo_root/justfile"
grep -Fq 'nix/nitro-bins/upstreams.nix' "$repo_root/README.md"
grep -Fq 'setupHook = ./aws-c-common-setup-hook.sh;' \
    "$repo_root/nix/nitro-bins/default.nix"
grep -Fq 'prependToVar cmakeFlags "-DCMAKE_MODULE_PATH=@out@/lib/cmake"' \
    "$repo_root/nix/nitro-bins/aws-c-common-setup-hook.sh"
grep -Fq -- '-Wno-error=implicit-function-declaration' \
    "$repo_root/nix/nitro-bins/default.nix"
grep -Fq 'legacy HTTP websocket decoder leaked into the deployed kmstool' \
    "$repo_root/nix/nitro-bins/default.nix"
grep -Fq './aws-c-auth-const-connection-manager-options.patch' \
    "$repo_root/nix/nitro-bins/default.nix"

echo "source-built Nitro helper retirement checks passed"
