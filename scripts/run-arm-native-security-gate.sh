#!/usr/bin/env bash

# Mandatory native realization gate for every deployable EIF build. Keep this
# list shared by local/OrbStack release builds and ARM CI so checks cannot drift.

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT

cd "${REPO_ROOT}"

current_system="$(nix eval --raw --impure --expr builtins.currentSystem)"
if [[ "${current_system}" != "aarch64-linux" ]]; then
  echo "ERROR: the EIF native security gate requires aarch64-linux; got ${current_system}" >&2
  exit 1
fi

lock_hash_before="$(nix hash file --type sha256 flake.lock)"

bash scripts/check-security-upstreams.sh --offline
bash scripts/check-security-upstreams.sh --live
nix flake check '.?submodules=1' \
  --all-systems \
  --no-build \
  --no-write-lock-file
nix build \
  --no-link \
  --no-write-lock-file \
  '.?submodules=1#checks.aarch64-linux.bash-runtime' \
  '.?submodules=1#checks.aarch64-linux.continuum-proxy' \
  '.?submodules=1#checks.aarch64-linux.elfutils-runtime' \
  '.?submodules=1#checks.aarch64-linux.findutils-runtime' \
  '.?submodules=1#checks.aarch64-linux.glibc-runtime' \
  '.?submodules=1#checks.aarch64-linux.iproute2-runtime' \
  '.?submodules=1#checks.aarch64-linux.kernel-source-pin' \
  '.?submodules=1#checks.aarch64-linux.legacy-nitro-build-retired' \
  '.?submodules=1#checks.aarch64-linux.nitro-helper' \
  '.?submodules=1#checks.aarch64-linux.nitro-init' \
  '.?submodules=1#checks.aarch64-linux.openssl-source-pin'

# Re-execute the harmless semantic/runtime checks for every release build even
# when their dependencies were obtained from a trusted binary cache.
nix build \
  --no-link \
  --no-write-lock-file \
  --rebuild \
  '.?submodules=1#checks.aarch64-linux.entrypoint-entropy-preflight' \
  '.?submodules=1#checks.aarch64-linux.glibc-native-smoke' \
  '.?submodules=1#checks.aarch64-linux.kernel-security-invariants' \
  '.?submodules=1#checks.aarch64-linux.vsock-helper' \
  '.?submodules=1#checks.aarch64-linux.rootfs-command-closure'

lock_hash_after="$(nix hash file --type sha256 flake.lock)"
if [[ "${lock_hash_before}" != "${lock_hash_after}" ]]; then
  echo "ERROR: native security checks mutated flake.lock" >&2
  exit 1
fi

echo "ARM native EIF security gate passed."
