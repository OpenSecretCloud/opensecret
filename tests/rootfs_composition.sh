#!/usr/bin/env bash

set -euo pipefail

repo_root="${REPO_ROOT_UNDER_TEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
flake="$repo_root/flake.nix"
entrypoint="$repo_root/entrypoint.sh"
rootfs_block="$(sed -n '/# BEGIN ENCLAVE_ROOTFS$/,/# END ENCLAVE_ROOTFS$/p' "$flake")"

if [ -z "$rootfs_block" ]; then
    echo "could not find the enclave rootfs composition block" >&2
    exit 1
fi

for package in \
    pkgs.bash pkgs.coreutils pkgs.findutils pkgs.gnused pkgs.iproute2 \
    pkgs.python3 pkgs.jq pkgs.socat pkgs.cacert; do
    if ! grep -Fq "$package" <<<"$rootfs_block"; then
        echo "required rootfs package is missing: $package" >&2
        exit 1
    fi
done

for forbidden in pkgs.busybox pkgs.postgresql pkgs.curl; do
    if grep -Fq "$forbidden" <<<"$rootfs_block"; then
        echo "unneeded package remains in the rootfs: $forbidden" >&2
        exit 1
    fi
done

grep -Fq "ln -s \${pkgs.iproute2}/sbin/ip \"\$out/bin/ip\"" <<<"$rootfs_block"
grep -Fq 'BusyBox unexpectedly entered the rootfs runtime closure' "$flake"
grep -Fq 'file -Lb "$executable"' "$flake"
grep -Fq '"$rootfs/bin/find" -L "$rootfs" -type f -perm -0100 -print0 > "$executable_list"' "$flake"
grep -Fq 'done < "$executable_list"' "$flake"
grep -Fq '/^[RWE]*E[RWE]*$/' "$flake"
grep -Fq 'rootfsValidation = mkRootfsCommandClosure' "$flake"
grep -Fq 'doCheck = true;' "$flake"
grep -Fq 'test -e ${rootfsValidation}' "$flake"

if grep -Fq '/app/libnsm.so' "$entrypoint"; then
    echo "the obsolete /app/libnsm.so placeholder remains" >&2
    exit 1
fi

echo "rootfs composition checks passed"
