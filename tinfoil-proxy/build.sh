#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${TINFOIL_PROXY_NIX_SHELL:-} != 1 ]]; then
  exec nix develop "$script_dir" --command \
    env TINFOIL_PROXY_NIX_SHELL=1 "$script_dir/build.sh"
fi

cd "$script_dir"

if [[ $(go env GOVERSION) != go1.26.5 ]]; then
  echo "expected the pinned Go 1.26.5 toolchain, found $(go env GOVERSION)" >&2
  exit 1
fi

export CGO_ENABLED=0
export GOTOOLCHAIN=local

mkdir -p dist
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/tinfoil-proxy-build.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

build_linux() {
  local arch=$1
  local output=$2

  echo "Building Linux/$arch $output..."
  GOOS=linux GOARCH="$arch" go build \
    -buildvcs=false \
    -ldflags="-s -w" \
    -mod=readonly \
    -trimpath \
    -o "$build_dir/$output" \
    .
  install -m 0755 "$build_dir/$output" "dist/$output"
}

build_linux arm64 tinfoil-proxy
build_linux amd64 tinfoil-proxy-x86_64

go version -m dist/tinfoil-proxy | sed -n '1,8p'
go version -m dist/tinfoil-proxy-x86_64 | sed -n '1,8p'
