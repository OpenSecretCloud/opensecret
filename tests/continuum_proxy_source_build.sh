#!/usr/bin/env bash
# shellcheck disable=SC2016 # Match literal Nix and shell interpolation syntax.

set -euo pipefail

repo_root="${REPO_ROOT_UNDER_TEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for blob in \
    "$repo_root/continuum-proxy" \
    "$repo_root/continuum-proxy-x86_64"; do
    if [ -e "$blob" ]; then
        echo "checked-in Continuum proxy blob remains: $blob" >&2
        exit 1
    fi
done

if grep -Fq '${./continuum-proxy}' "$repo_root/flake.nix"; then
    echo "the EIF still consumes a checked-in Continuum proxy blob" >&2
    exit 1
fi

grep -Fq 'version = "1.51.0";' "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'tag = "v1.51.0";' "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'owner = "edgelesssys";' "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'repo = "privatemode-public";' "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'rev = "4e625ecb28aeeabde65fe3b5d78864e02af1932c";' \
    "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'hash = "sha256-/nkkiaeKPZ/KswfM1/Nr1SBMslcijEFf2UTWSb/vwYQ=";' \
    "$repo_root/nix/continuum-proxy.nix"
grep -Fq 'vendorHash = "sha256-adHo+dzpeWVnWk3VDVohZJK4C080JJRe/9XqaieMkuI=";' \
    "$repo_root/nix/continuum-proxy.nix"

grep -Fq 'pkgs.buildGo126Module {' "$repo_root/flake.nix"
if grep -Fq 'securityToolsPkgs.buildGo' "$repo_root/flake.nix"; then
    echo "the EIF Continuum proxy must use the stock platform build toolchain" >&2
    exit 1
fi
grep -Fq 'env.CGO_ENABLED = "0";' "$repo_root/flake.nix"
grep -Fq 'subPackages = [ "privatemode-proxy" ];' "$repo_root/flake.nix"
grep -Fq 'install -m 755 ${continuum-proxy}/bin/continuum-proxy /app/' \
    "$repo_root/flake.nix"
grep -Fq '/app/continuum-proxy --port 8092 --apiKey "$continuum_proxy_api_key" --sharedPromptCache' \
    "$repo_root/entrypoint.sh"
grep -Fq 'nix build .#continuum-proxy --no-link --print-out-paths' \
    "$repo_root/justfile"

if [ -e "$repo_root/.git" ]; then
    gitlink="$(git -C "$repo_root" ls-files --stage privatemode-public | awk '{ print $2 }')"
    if [ "$gitlink" != "4e625ecb28aeeabde65fe3b5d78864e02af1932c" ]; then
        echo "the local Continuum gitlink differs from the source-build revision" >&2
        exit 1
    fi
fi

echo "source-built Continuum proxy checks passed"
