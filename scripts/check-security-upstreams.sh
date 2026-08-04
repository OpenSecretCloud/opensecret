#!/usr/bin/env bash

# Read-only freshness check for security-sensitive source pins. This script
# deliberately reports drift instead of rewriting the manifest: changing a pin
# remains an explicit, reviewed update to the reproducible build inputs.

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly MANIFEST="${REPO_ROOT}/nix/security-upstreams.nix"
readonly MAX_METADATA_BYTES=5242880
readonly RESPONSE_SENTINEL=$'\036'
readonly KERNEL_RELEASES_URL="https://www.kernel.org/releases.json"
readonly RUST_STABLE_CHANNEL_URL="https://static.rust-lang.org/dist/channel-rust-stable.toml"
readonly GITHUB_API_ROOT="https://api.github.com"
readonly GLIBC_RELEASES_URL="https://ftp.gnu.org/gnu/glibc/"
readonly GLIBC_STABLE_BRANCH="release/2.44/master"
readonly GLIBC_STABLE_BRANCH_URL="https://sourceware.org/git/?p=glibc.git;a=commit;h=refs/heads/release/2.44/master"
readonly GLIBC_REVIEWED_PACKAGE_REVISION="8"
readonly BASH_COMPATIBILITY_BRANCH="5.3"
readonly BASH_PATCHES_URL="https://ftp.gnu.org/gnu/bash/bash-5.3-patches/"
readonly IPROUTE2_TAGS_URL="https://git.kernel.org/pub/scm/network/iproute2/iproute2.git/refs/tags"
readonly NSS_TAGS_URL="https://hg.mozilla.org/projects/nss/json-tags"
readonly COREUTILS_RELEASES_URL="https://ftp.gnu.org/gnu/coreutils/"
readonly ELFUTILS_RELEASES_URL="https://sourceware.org/elfutils/ftp/"
readonly FINDUTILS_RELEASES_URL="https://ftp.gnu.org/gnu/findutils/"
readonly GNU_SED_RELEASES_URL="https://ftp.gnu.org/gnu/sed/"
readonly GO_COMPATIBILITY_BRANCH="1.26"
readonly GO_RELEASES_URL="https://go.dev/dl/?mode=json&include=all"
readonly KRB5_COMPATIBILITY_BRANCH="1.22"
readonly KRB5_RELEASES_URL="https://web.mit.edu/kerberos/dist/index.html"
readonly POSTGRESQL_COMPATIBILITY_BRANCH="17"
readonly POSTGRESQL_RELEASES_URL="https://www.postgresql.org/versions.json"
readonly PYTHON_COMPATIBILITY_BRANCH="3.13"
readonly PYTHON_RELEASES_URL="https://www.python.org/ftp/python/"
readonly SOCAT_COMPATIBILITY_BRANCH="1.8"
# Upstream's HTTPS certificate is currently invalid and its canonical Git host
# is intermittently unreachable. Require its official HTTP download index to
# agree with an authenticated, automatically synchronized mirror of the
# canonical Git tags; disagreement or either source being unavailable fails.
readonly SOCAT_RELEASES_URL="http://www.dest-unreach.org/socat/download/"
readonly SOCAT_AUTHENTICATED_MIRROR_URL="https://third-party-mirror.googlesource.com/socat/+refs?format=JSON"
readonly ZLIB_RELEASES_URL="https://zlib.net/"

mode="${1:---live}"
if [[ "${mode}" != "--live" && "${mode}" != "--offline" ]]; then
  echo "usage: $0 [--live|--offline]" >&2
  exit 2
fi

failures=0
manifest_json=""
http_body=""
github_body=""
github_commit_sha=""
latest_numeric_version=""

require_command() {
  local command_name="$1"

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${command_name}" >&2
    exit 2
  fi
}

bounded_curl() {
  local -a pipeline_status

  # Keep command-substitution memory bounded even when an endpoint omits
  # Content-Length or uses chunked transfer. The sentinel preserves trailing
  # newlines so the caller can measure the exact captured byte count.
  set +e
  curl \
    "$@" \
    | head -c "$((MAX_METADATA_BYTES + 1))"
  pipeline_status=( "${PIPESTATUS[@]}" )
  set -e
  printf '%s' "${RESPONSE_SENTINEL}"

  if (( pipeline_status[1] != 0 )); then
    return "${pipeline_status[1]}"
  fi
  return "${pipeline_status[0]}"
}

http_get() {
  bounded_curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 10 \
    --max-time 30 \
    --max-filesize "${MAX_METADATA_BYTES}" \
    --retry 2 \
    --retry-delay 1 \
    --retry-max-time 30 \
    "$1"
}

fetch_http_body() {
  local name="$1"
  local url="$2"
  local response fetch_status=0

  if response="$(http_get "${url}")"; then
    :
  else
    fetch_status=$?
  fi
  if [[ "${response}" != *"${RESPONSE_SENTINEL}" ]]; then
    record_failure "${name}: bounded response capture failed internally (${url})"
    return 1
  fi
  response="${response%"${RESPONSE_SENTINEL}"}"
  if (( fetch_status == 63 || ${#response} > MAX_METADATA_BYTES )); then
    record_failure "${name}: primary upstream response exceeded ${MAX_METADATA_BYTES} bytes (${url})"
    return 1
  fi
  if (( fetch_status != 0 )); then
    record_failure "${name}: could not fetch primary upstream metadata (${url})"
    return 1
  fi
  if [[ -z "${response}" ]]; then
    record_failure "${name}: primary upstream returned an empty response (${url})"
    return 1
  fi
  http_body="${response}"
}

parse_latest_numeric_version() {
  local name="$1"
  local source="$2"
  local capture_pattern="$3"
  local parsed

  if ! parsed="$({
    jq -Rers --arg pattern "${capture_pattern}" '
      [scan($pattern) | .[0]]
      | unique
      | map(select(test("^[0-9]+(\\.[0-9]+)+$")))
      | map({version: ., components: (split(".") | map(tonumber))})
      | max_by(.components)
      | .version
    ' <<<"${source}"
  })"; then
    record_failure "${name}: primary upstream response was malformed or contained no stable numeric release"
    return 1
  fi
  if [[ -z "${parsed}" ]]; then
    record_failure "${name}: parsed an empty latest release from primary upstream metadata"
    return 1
  fi

  latest_numeric_version="${parsed}"
}

compare_pinned_version() {
  local name="$1"
  local pinned_version="$2"
  local latest_version="$3"
  local source_url="$4"

  if [[ "${pinned_version}" != "${latest_version}" ]]; then
    record_failure "${name}: pinned ${pinned_version}, latest ${latest_version} (${source_url})"
  else
    echo "CURRENT: ${name} ${pinned_version}"
  fi
}

check_http_numeric_release() {
  local name="$1"
  local pinned_version="$2"
  local releases_url="$3"
  local capture_pattern="$4"

  if ! fetch_http_body "${name}" "${releases_url}"; then
    return
  fi
  if ! parse_latest_numeric_version "${name}" "${http_body}" "${capture_pattern}"; then
    return
  fi

  compare_pinned_version "${name}" "${pinned_version}" "${latest_numeric_version}" "${releases_url}"
}

github_get() {
  local path="$1"
  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  local -a headers=(
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
  )

  if [[ -n "${token}" ]]; then
    headers+=( -H "Authorization: Bearer ${token}" )
  fi

  bounded_curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --connect-timeout 10 \
    --max-time 30 \
    --max-filesize "${MAX_METADATA_BYTES}" \
    --retry 2 \
    --retry-delay 1 \
    --retry-max-time 30 \
    "${headers[@]}" \
    "${GITHUB_API_ROOT}${path}"
}

fetch_github_body() {
  local name="$1"
  local path="$2"
  local response fetch_status=0

  if response="$(github_get "${path}")"; then
    :
  else
    fetch_status=$?
  fi
  if [[ "${response}" != *"${RESPONSE_SENTINEL}" ]]; then
    record_failure "${name}: bounded GitHub response capture failed internally (${GITHUB_API_ROOT}${path})"
    return 1
  fi
  response="${response%"${RESPONSE_SENTINEL}"}"
  if (( fetch_status == 63 || ${#response} > MAX_METADATA_BYTES )); then
    record_failure "${name}: official GitHub response exceeded ${MAX_METADATA_BYTES} bytes (${GITHUB_API_ROOT}${path})"
    return 1
  fi
  if (( fetch_status != 0 )); then
    record_failure "${name}: could not fetch official GitHub metadata (${GITHUB_API_ROOT}${path})"
    return 1
  fi
  if [[ -z "${response}" ]]; then
    record_failure "${name}: official GitHub endpoint returned an empty response (${GITHUB_API_ROOT}${path})"
    return 1
  fi
  github_body="${response}"
}

urlencode() {
  jq -nr --arg value "$1" '$value | @uri'
}

record_failure() {
  echo "STALE OR INVALID: $*" >&2
  failures=$((failures + 1))
}

evaluate_manifest_restricted() {
  local manifest_json_document manifest_nix_literal

  # Copy the source into an evaluator-created store file so restricted mode
  # need not whitelist the checkout path. Double JSON encoding safely embeds
  # arbitrary source bytes in the Nix expression; escape '$' at the outer Nix
  # string layer so a manifest interpolation cannot escape that container.
  manifest_json_document="$(jq -Rs . "${MANIFEST}")"
  manifest_nix_literal="$(printf '%s' "${manifest_json_document}" | jq -Rs .)"
  manifest_nix_literal="${manifest_nix_literal//\$/\\\$}"

  nix eval \
    --offline \
    --restrict-eval \
    --json \
    --expr "import (builtins.toFile \"security-upstreams.nix\" (builtins.fromJSON ${manifest_nix_literal}))"
}

check_manifest_structure() {
  local locked_nixpkgs locked_rust_overlay locked_nitro_rust toolchain_version
  local continuum_gitlink continuum_version
  local bash_branch bash_base_version bash_version bash_patch_level bash_first_patch bash_last_patch
  local glibc_version expected_glibc_url expected_glibc_package_version expected_glibc_stable_patch
  local expected_bash_url expected_elfutils_url expected_elfutils_patch_url
  local expected_iproute2_tag expected_iproute2_url
  local expected_kernel_url expected_openssl_url
  local runtime_go runtime_krb5 runtime_postgresql runtime_python runtime_socat

  if ! jq -e '
    (.nixpkgs.rev | test("^[0-9a-f]{40}$")) and
    (.nixRuntime | type == "object") and
    ([
      .nixRuntime.glibc,
      .nixRuntime.cacert,
      .nixRuntime.coreutils,
      .nixRuntime.elfutils,
      .nixRuntime.findutils,
      .nixRuntime.gnused,
      .nixRuntime.go,
      .nixRuntime.jq,
      .nixRuntime.krb5,
      .nixRuntime.postgresql,
      .nixRuntime.python,
      .nixRuntime.socat,
      .nixRuntime.zlib
    ] | all(type == "string" and test("^[0-9]+(\\.[0-9]+)+$"))) and
    (.glibc | type == "object") and
    ((.glibc | keys) == ["hash", "packageVersion", "patches", "stableRev", "url", "version"]) and
    (.glibc.version == .nixRuntime.glibc) and
    (.glibc.version | test("^[0-9]+\\.[0-9]+$")) and
    (.glibc.packageVersion | test("^[0-9]+\\.[0-9]+-[0-9]+$")) and
    (.glibc.url | test("^https://")) and
    (.glibc.hash | test("^sha256-[A-Za-z0-9+/]{43}=$")) and
    (.glibc.stableRev | test("^[0-9a-f]{40}$")) and
    (.glibc.patches | type == "object") and
    ((.glibc.patches | keys) == ["aarch64Nvcc", "bash", "cache", "stable"]) and
    ([.glibc.patches[]]
      | all(
          ((keys) == ["file", "hash"]) and
          (.file | type == "string" and test("^[A-Za-z0-9._+-]+\\.patch$")) and
          (.hash | test("^[0-9a-f]{64}$"))
        )) and
    (.elfutils | type == "object") and
    ((.elfutils | keys) == ["hash", "patches", "url", "version"]) and
    (.elfutils.version == .nixRuntime.elfutils) and
    (.elfutils.version | test("^[0-9]+\\.[0-9]+$")) and
    (.elfutils.url | test("^https://")) and
    (.elfutils.hash | test("^sha256-[A-Za-z0-9+/]{43}=$")) and
    ((.elfutils.patches | keys) == ["i386Tlsdesc"]) and
    ((.elfutils.patches.i386Tlsdesc | keys) == ["hash", "url"]) and
    (.elfutils.patches.i386Tlsdesc.url | test("^https://")) and
    (.elfutils.patches.i386Tlsdesc.hash | test("^sha256-[A-Za-z0-9+/]{43}=$")) and
    (.bash.branch | test("^[0-9]+\\.[0-9]+$")) and
    (.bash.baseVersion | test("^[0-9]+\\.[0-9]+$")) and
    (.bash.version | test("^[0-9]+\\.[0-9]+p[0-9]+$")) and
    (.bash.url | test("^https://")) and
    (.bash.hash | startswith("sha256-")) and
    (.bash.patches | type == "array" and length > 0) and
    ([.bash.patches[]]
      | all(
          (.number | test("^[0-9]{3}$")) and
          (.hash | startswith("sha256-"))
        )) and
    (.iproute2.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$")) and
    (.iproute2.tag | test("^v[0-9]+\\.[0-9]+\\.[0-9]+$")) and
    (.iproute2.url | test("^https://")) and
    (.iproute2.hash | startswith("sha256-")) and
    (.linux.hash | startswith("sha256-")) and
    (.openssl.hash | startswith("sha256-")) and
    ([.appRust, .nitroRust][]
      | (.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$")) and
        (.overlayRev | test("^[0-9a-f]{40}$")) and
        (.overlayHash | startswith("sha256-"))) and
    (.nitroUtil.rev | test("^[0-9a-f]{40}$")) and
    (.nitroUtil.reviewedHead | test("^[0-9a-f]{40}$")) and
    (.continuumProxy.rev | test("^[0-9a-f]{40}$")) and
    (.continuumProxy.vendorHash | startswith("sha256-")) and
    (.nitro | length > 0) and
    ([.nitro[]]
      | all(
          (.version | type == "string" and length > 0) and
          (.tag | type == "string" and length > 0) and
          (.rev | test("^[0-9a-f]{40}$")) and
          (.hash | startswith("sha256-"))
        ))
  ' <<<"${manifest_json}" >/dev/null; then
    record_failure "security manifest contains a malformed version, revision, tag, or source hash"
  fi

  locked_nixpkgs="$(jq -er '.nodes.nixpkgs.locked.rev' "${REPO_ROOT}/flake.lock")"
  locked_rust_overlay="$(jq -er '.nodes["rust-overlay"].locked.rev' "${REPO_ROOT}/flake.lock")"
  locked_nitro_rust="$(jq -er '.nodes["nitro-rust-overlay"].locked.rev' "${REPO_ROOT}/flake.lock")"
  toolchain_version="$(awk -F '"' '/^channel = "/ { print $2; exit }' "${REPO_ROOT}/rust-toolchain.toml")"
  continuum_gitlink="$(git -C "${REPO_ROOT}" ls-files --stage privatemode-public | awk '{ print $2 }')"
  continuum_version="$(awk -F '"' '/version = "/ { print $2; exit }' "${REPO_ROOT}/privatemode-public/version.nix")"

  glibc_version="$(jq -er '.glibc.version' <<<"${manifest_json}")"
  expected_glibc_url="https://ftp.gnu.org/gnu/glibc/glibc-${glibc_version}.tar.xz"
  expected_glibc_package_version="${glibc_version}-${GLIBC_REVIEWED_PACKAGE_REVISION}"
  expected_glibc_stable_patch="glibc-${glibc_version}-master.patch"

  bash_branch="$(jq -er '.bash.branch' <<<"${manifest_json}")"
  bash_base_version="$(jq -er '.bash.baseVersion' <<<"${manifest_json}")"
  bash_version="$(jq -er '.bash.version' <<<"${manifest_json}")"
  bash_first_patch="$(jq -er '.bash.patches[0].number' <<<"${manifest_json}")"
  bash_last_patch="$(jq -er '.bash.patches[-1].number' <<<"${manifest_json}")"
  expected_bash_url="https://ftp.gnu.org/gnu/bash/bash-${bash_base_version}.tar.gz"
  expected_elfutils_url="https://sourceware.org/elfutils/ftp/$(jq -er '.elfutils.version' <<<"${manifest_json}")/elfutils-$(jq -er '.elfutils.version' <<<"${manifest_json}").tar.bz2"
  expected_elfutils_patch_url="https://sourceware.org/git/?p=elfutils.git;a=patch;h=bfd519cc58e190544a6785d3f0a27fcfaf7d8da3"
  expected_iproute2_tag="v$(jq -er '.iproute2.version' <<<"${manifest_json}")"
  expected_iproute2_url="https://cdn.kernel.org/pub/linux/utils/net/iproute2/iproute2-$(jq -er '.iproute2.version' <<<"${manifest_json}").tar.xz"

  [[ "${locked_nixpkgs}" == "$(jq -er '.nixpkgs.rev' <<<"${manifest_json}")" ]] \
    || record_failure "flake.lock Nixpkgs revision differs from the reviewed manifest"
  [[ "${locked_rust_overlay}" == "$(jq -er '.appRust.overlayRev' <<<"${manifest_json}")" ]] \
    || record_failure "flake.lock application rust-overlay differs from the reviewed manifest"
  [[ "${locked_nitro_rust}" == "$(jq -er '.nitroRust.overlayRev' <<<"${manifest_json}")" ]] \
    || record_failure "flake.lock Nitro rust-overlay differs from the reviewed manifest"
  [[ "${toolchain_version}" == "$(jq -er '.appRust.version' <<<"${manifest_json}")" ]] \
    || record_failure "rust-toolchain.toml differs from the reviewed application Rust version"
  [[ "${continuum_gitlink}" == "$(jq -er '.continuumProxy.rev' <<<"${manifest_json}")" ]] \
    || record_failure "privatemode-public gitlink differs from the reviewed Continuum proxy revision"
  [[ "${continuum_version}" == "v$(jq -er '.continuumProxy.version' <<<"${manifest_json}")" ]] \
    || record_failure "privatemode-public version differs from the reviewed Continuum proxy version"

  [[ "$(jq -er '.glibc.url' <<<"${manifest_json}")" == "${expected_glibc_url}" ]] \
    || record_failure "glibc source URL does not match its reviewed version"
  [[ "$(jq -er '.glibc.packageVersion' <<<"${manifest_json}")" == "${expected_glibc_package_version}" ]] \
    || record_failure "glibc package revision must remain the reviewed ${expected_glibc_package_version}"
  [[ "${GLIBC_STABLE_BRANCH}" == "release/${glibc_version}/master" ]] \
    || record_failure "glibc version differs from the independently approved ${GLIBC_STABLE_BRANCH} branch"
  [[ "${GLIBC_STABLE_BRANCH_URL}" == "https://sourceware.org/git/?p=glibc.git;a=commit;h=refs/heads/${GLIBC_STABLE_BRANCH}" ]] \
    || record_failure "glibc stable-branch metadata URL differs from the independently approved branch"
  if ! jq -e --arg stable_patch "${expected_glibc_stable_patch}" '
    .glibc.patches.aarch64Nvcc.file == "0001-aarch64-math-vector.h-add-NVCC-include-guard.patch" and
    .glibc.patches.bash.file == "0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch" and
    .glibc.patches.cache.file == "dont-use-system-ld-so-cache.patch" and
    .glibc.patches.stable.file == $stable_patch
  ' <<<"${manifest_json}" >/dev/null; then
    record_failure "glibc patch file set differs from the reviewed Nixpkgs 2.44 composition"
  fi

  [[ "${bash_branch}" == "${BASH_COMPATIBILITY_BRANCH}" ]] \
    || record_failure "Bash compatibility branch must remain the independently approved ${BASH_COMPATIBILITY_BRANCH} branch"
  [[ "${bash_base_version}" == "${BASH_COMPATIBILITY_BRANCH}" ]] \
    || record_failure "Bash base version differs from the independently approved ${BASH_COMPATIBILITY_BRANCH} branch"
  if [[ "${bash_version}" =~ ^[0-9]+\.[0-9]+p([0-9]+)$ ]]; then
    bash_patch_level="${BASH_REMATCH[1]}"
    [[ "$((10#${bash_patch_level}))" -eq "$((10#${bash_last_patch}))" ]] \
      || record_failure "Bash version patch level differs from the final pinned GNU patch"
  else
    record_failure "Bash version does not encode its applied patch level"
  fi
  [[ "${bash_version}" == "${BASH_COMPATIBILITY_BRANCH}"p* ]] \
    || record_failure "Bash version is outside the independently approved ${BASH_COMPATIBILITY_BRANCH} branch"
  [[ "${bash_first_patch}" == "010" ]] \
    || record_failure "Bash direct patch extension must begin at GNU patch 010"
  [[ "$(jq -cer '[.bash.patches[].number | tonumber] as $numbers | ($numbers == ($numbers | sort)) and (($numbers | unique | length) == ($numbers | length)) and (all(range(1; $numbers | length); . as $index | $numbers[$index] == ($numbers[$index - 1] + 1)))' <<<"${manifest_json}")" == "true" ]] \
    || record_failure "Bash patch numbers must be unique, ascending, and contiguous"
  [[ "$(jq -er '.bash.url' <<<"${manifest_json}")" == "${expected_bash_url}" ]] \
    || record_failure "Bash source URL does not match its reviewed base version"
  [[ "$(jq -er '.elfutils.url' <<<"${manifest_json}")" == "${expected_elfutils_url}" ]] \
    || record_failure "elfutils source URL does not match its reviewed version"
  [[ "$(jq -er '.elfutils.patches.i386Tlsdesc.url' <<<"${manifest_json}")" == "${expected_elfutils_patch_url}" ]] \
    || record_failure "elfutils i386 TLS descriptor patch differs from the reviewed upstream commit"
  [[ "$(jq -er '.iproute2.tag' <<<"${manifest_json}")" == "${expected_iproute2_tag}" ]] \
    || record_failure "iproute2 release tag does not match its reviewed version"
  [[ "$(jq -er '.iproute2.url' <<<"${manifest_json}")" == "${expected_iproute2_url}" ]] \
    || record_failure "iproute2 source URL does not match its reviewed version"

  runtime_go="$(jq -er '.nixRuntime.go' <<<"${manifest_json}")"
  runtime_krb5="$(jq -er '.nixRuntime.krb5' <<<"${manifest_json}")"
  runtime_postgresql="$(jq -er '.nixRuntime.postgresql' <<<"${manifest_json}")"
  runtime_python="$(jq -er '.nixRuntime.python' <<<"${manifest_json}")"
  runtime_socat="$(jq -er '.nixRuntime.socat' <<<"${manifest_json}")"
  [[ "${runtime_go}" == "${GO_COMPATIBILITY_BRANCH}."* ]] \
    || record_failure "Go version is outside the independently approved ${GO_COMPATIBILITY_BRANCH} branch"
  [[ "${runtime_krb5}" == "${KRB5_COMPATIBILITY_BRANCH}."* ]] \
    || record_failure "MIT Kerberos version is outside the independently approved ${KRB5_COMPATIBILITY_BRANCH} branch"
  [[ "${runtime_postgresql}" == "${POSTGRESQL_COMPATIBILITY_BRANCH}."* ]] \
    || record_failure "PostgreSQL version is outside the independently approved ${POSTGRESQL_COMPATIBILITY_BRANCH} branch"
  [[ "${runtime_python}" == "${PYTHON_COMPATIBILITY_BRANCH}."* ]] \
    || record_failure "Python version is outside the independently approved ${PYTHON_COMPATIBILITY_BRANCH} branch"
  [[ "${runtime_socat}" == "${SOCAT_COMPATIBILITY_BRANCH}."* ]] \
    || record_failure "socat version is outside the independently approved ${SOCAT_COMPATIBILITY_BRANCH} branch"

  expected_kernel_url="https://cdn.kernel.org/pub/linux/kernel/v$(jq -er '.linux.branch | split(".")[0]' <<<"${manifest_json}").x/linux-$(jq -er '.linux.version' <<<"${manifest_json}").tar.xz"
  expected_openssl_url="https://github.com/openssl/openssl/releases/download/openssl-$(jq -er '.openssl.version' <<<"${manifest_json}")/openssl-$(jq -er '.openssl.version' <<<"${manifest_json}").tar.gz"
  [[ "$(jq -er '.linux.url' <<<"${manifest_json}")" == "${expected_kernel_url}" ]] \
    || record_failure "Linux source URL does not match its reviewed version"
  [[ "$(jq -er '.openssl.url' <<<"${manifest_json}")" == "${expected_openssl_url}" ]] \
    || record_failure "OpenSSL source URL does not match its reviewed version"
}

github_branch_head() {
  local owner="$1"
  local repo="$2"
  local branch="$3"
  local encoded_branch parsed_sha

  encoded_branch="$(urlencode "${branch}")"
  if ! fetch_github_body \
    "${owner}/${repo} branch ${branch}" \
    "/repos/${owner}/${repo}/branches/${encoded_branch}"; then
    return 1
  fi
  if ! parsed_sha="$(jq -er '.commit.sha | select(test("^[0-9a-f]{40}$"))' <<<"${github_body}")" \
    || [[ -z "${parsed_sha}" ]]; then
    record_failure "${owner}/${repo} branch ${branch}: official GitHub response had an invalid schema or commit SHA"
    return 1
  fi

  github_commit_sha="${parsed_sha}"
}

check_nixpkgs() {
  local branch pinned_rev head

  branch="$(jq -er '.nixpkgs.branch' <<<"${manifest_json}")"
  pinned_rev="$(jq -er '.nixpkgs.rev' <<<"${manifest_json}")"
  if ! github_branch_head NixOS nixpkgs "${branch}"; then
    return
  fi
  head="${github_commit_sha}"

  if [[ ! "${pinned_rev}" =~ ^[0-9a-f]{40}$ ]]; then
    record_failure "Nixpkgs ${branch}: revision must be a full commit SHA"
  elif [[ "${pinned_rev}" != "${head}" ]]; then
    record_failure "Nixpkgs ${branch}: pinned ${pinned_rev}, branch head ${head}"
  else
    echo "CURRENT: Nixpkgs ${branch} ${pinned_rev}"
  fi
}

check_nitro_util_reviewed_head() {
  local owner repo branch pinned_rev reviewed_head head

  owner="$(jq -er '.nitroUtil.owner' <<<"${manifest_json}")"
  repo="$(jq -er '.nitroUtil.repo' <<<"${manifest_json}")"
  branch="$(jq -er '.nitroUtil.branch' <<<"${manifest_json}")"
  pinned_rev="$(jq -er '.nitroUtil.rev' <<<"${manifest_json}")"
  reviewed_head="$(jq -er '.nitroUtil.reviewedHead' <<<"${manifest_json}")"
  if ! github_branch_head "${owner}" "${repo}" "${branch}"; then
    return
  fi
  head="${github_commit_sha}"

  if [[ ! "${pinned_rev}" =~ ^[0-9a-f]{40}$ ]] || [[ ! "${reviewed_head}" =~ ^[0-9a-f]{40}$ ]]; then
    record_failure "Nitro utility: pin and reviewed head must be full commit SHAs"
  elif [[ "${reviewed_head}" != "${head}" ]]; then
    record_failure "Nitro utility: upstream moved beyond reviewed head ${reviewed_head} to ${head}"
  elif [[ "${pinned_rev}" == "${head}" ]]; then
    echo "CURRENT: Nitro utility ${pinned_rev}"
  else
    echo "REVIEWED HOLD: Nitro utility runtime pin ${pinned_rev}; upstream ${head} contains build-only changes"
  fi
}

check_kernel() {
  local branch pinned_version pinned_url releases latest_version expected_url

  branch="$(jq -er '.linux.branch' <<<"${manifest_json}")"
  pinned_version="$(jq -er '.linux.version' <<<"${manifest_json}")"
  pinned_url="$(jq -er '.linux.url' <<<"${manifest_json}")"
  if ! fetch_http_body "Linux ${branch}" "${KERNEL_RELEASES_URL}"; then
    return
  fi
  releases="${http_body}"
  if ! latest_version="$(
    jq -er --arg branch "${branch}" '
      if (.releases | type) != "array" then
        error("expected releases array")
      else
        [
          .releases[]
          | select(.version | test("^" + ($branch | split(".") | join("\\.")) + "\\.[0-9]+$"))
          | {version: .version, components: (.version | split(".") | map(tonumber))}
        ]
        | if length == 0 then error("no release on approved branch") else max_by(.components).version end
      end
    ' <<<"${releases}"
  )" || [[ -z "${latest_version}" ]]; then
    record_failure "Linux ${branch}: official releases response had an invalid schema or no stable branch release"
    return
  fi
  expected_url="https://cdn.kernel.org/pub/linux/kernel/v${branch%%.*}.x/linux-${pinned_version}.tar.xz"

  if [[ "${pinned_version}" != "${latest_version}" ]]; then
    record_failure "Linux ${branch}: pinned ${pinned_version}, latest ${latest_version} (${KERNEL_RELEASES_URL})"
  else
    echo "CURRENT: Linux ${branch} ${pinned_version}"
  fi

  if [[ "${pinned_url}" != "${expected_url}" ]]; then
    record_failure "Linux ${branch}: source URL does not match pinned version (expected ${expected_url})"
  fi
}

check_openssl() {
  local branch pinned_version pinned_url releases latest_tag latest_version expected_tag expected_url

  branch="$(jq -er '.openssl.branch' <<<"${manifest_json}")"
  pinned_version="$(jq -er '.openssl.version' <<<"${manifest_json}")"
  pinned_url="$(jq -er '.openssl.url' <<<"${manifest_json}")"
  if ! fetch_github_body "OpenSSL ${branch}" "/repos/openssl/openssl/releases?per_page=100"; then
    return
  fi
  releases="${github_body}"
  if ! latest_tag="$(
    jq -er --arg prefix "openssl-${branch}." '
      if type != "array" then
        error("expected releases array")
      else
        [
          .[]
          | select((.draft | not) and (.prerelease | not))
          | select(.tag_name | startswith($prefix))
          | select(.tag_name | test("^openssl-[0-9]+\\.[0-9]+\\.[0-9]+$"))
          | {tag: .tag_name, components: (.tag_name | sub("^openssl-"; "") | split(".") | map(tonumber))}
        ]
        | if length == 0 then error("no stable release on approved branch") else max_by(.components).tag end
      end
    ' <<<"${releases}"
  )" || [[ -z "${latest_tag}" ]]; then
    record_failure "OpenSSL ${branch}: official releases response had an invalid schema or no stable branch release"
    return
  fi
  latest_version="${latest_tag#openssl-}"
  expected_tag="openssl-${pinned_version}"
  expected_url="https://github.com/openssl/openssl/releases/download/${expected_tag}/openssl-${pinned_version}.tar.gz"

  if [[ "${pinned_version}" != "${latest_version}" ]]; then
    record_failure "OpenSSL ${branch}: pinned ${pinned_version}, latest ${latest_version} (official release ${latest_tag})"
  else
    echo "CURRENT: OpenSSL ${branch} ${pinned_version}"
  fi

  if [[ "${pinned_url}" != "${expected_url}" ]]; then
    record_failure "OpenSSL ${branch}: source URL does not match its official release tag (expected ${expected_url})"
  fi
}

check_glibc() {
  local pinned_stable_rev latest_stable_rev

  check_http_numeric_release \
    "glibc" \
    "$(jq -er '.nixRuntime.glibc' <<<"${manifest_json}")" \
    "${GLIBC_RELEASES_URL}" \
    'glibc-([0-9]+\.[0-9]+)\.tar\.xz'

  pinned_stable_rev="$(jq -er '.glibc.stableRev' <<<"${manifest_json}")"
  if ! fetch_http_body "glibc ${GLIBC_STABLE_BRANCH}" "${GLIBC_STABLE_BRANCH_URL}"; then
    return
  fi
  if ! latest_stable_rev="$(
    jq -Rers '
      [scan("<tr><td>commit</td><td class=\\\"sha1\\\">([0-9a-f]{40})</td>") | .[0]]
      | unique
      | if length != 1 then error("expected exactly one branch-head commit") else .[0] end
    ' <<<"${http_body}"
  )" || [[ -z "${latest_stable_rev}" ]]; then
    record_failure "glibc ${GLIBC_STABLE_BRANCH}: official Sourceware response had an invalid schema or branch head"
    return
  fi

  if [[ "${pinned_stable_rev}" != "${latest_stable_rev}" ]]; then
    record_failure "glibc ${GLIBC_STABLE_BRANCH}: reviewed ${pinned_stable_rev}, branch head ${latest_stable_rev} (${GLIBC_STABLE_BRANCH_URL})"
  else
    echo "CURRENT: glibc ${GLIBC_STABLE_BRANCH} ${pinned_stable_rev}"
  fi
}

check_bash() {
  local pinned_version latest_patch latest_version

  pinned_version="$(jq -er '.bash.version' <<<"${manifest_json}")"
  if ! fetch_http_body "Bash ${BASH_COMPATIBILITY_BRANCH}" "${BASH_PATCHES_URL}"; then
    return
  fi
  if ! latest_patch="$(
    jq -Rers '
      [scan("bash53-([0-9]{3})(?:[^0-9]|$)") | .[0] | tonumber]
      | unique
      | max
    ' <<<"${http_body}"
  )" || [[ -z "${latest_patch}" ]]; then
    record_failure "Bash ${BASH_COMPATIBILITY_BRANCH}: primary upstream response was malformed or contained no patch releases"
    return
  fi

  latest_version="${BASH_COMPATIBILITY_BRANCH}p${latest_patch}"
  compare_pinned_version \
    "Bash ${BASH_COMPATIBILITY_BRANCH}" \
    "${pinned_version}" \
    "${latest_version}" \
    "${BASH_PATCHES_URL}"
}

check_elfutils() {
  check_http_numeric_release \
    "elfutils" \
    "$(jq -er '.elfutils.version' <<<"${manifest_json}")" \
    "${ELFUTILS_RELEASES_URL}" \
    'href="([0-9]+\.[0-9]+)/"'
}

check_iproute2() {
  local pinned_version

  pinned_version="$(jq -er '.iproute2.version' <<<"${manifest_json}")"
  if ! fetch_http_body "iproute2" "${IPROUTE2_TAGS_URL}"; then
    return
  fi
  if ! parse_latest_numeric_version \
    "iproute2" \
    "${http_body}" \
    '>v([0-9]+\.[0-9]+\.[0-9]+)</a>'; then
    return
  fi

  compare_pinned_version "iproute2" "${pinned_version}" "${latest_numeric_version}" "${IPROUTE2_TAGS_URL}"
}

check_cacert() {
  local pinned_version latest_version

  pinned_version="$(jq -er '.nixRuntime.cacert' <<<"${manifest_json}")"
  if ! fetch_http_body "NSS/cacert" "${NSS_TAGS_URL}"; then
    return
  fi
  if ! latest_version="$(
    jq -er '
      if (.tags | type) != "array" then
        error("expected tags array")
      else
        [
          .tags[]
          | .tag
          | select(type == "string")
          | select(test("^NSS_[0-9]+_[0-9]+(_[0-9]+)?_RTM$"))
          | sub("^NSS_"; "")
          | sub("_RTM$"; "")
          | gsub("_"; ".")
          | {version: ., components: (split(".") | map(tonumber))}
        ]
        | if length == 0 then error("no stable RTM tags") else max_by(.components).version end
      end
    ' <<<"${http_body}"
  )" || [[ -z "${latest_version}" ]]; then
    record_failure "NSS/cacert: primary upstream response had an invalid schema or no stable RTM label"
    return
  fi

  compare_pinned_version "NSS/cacert" "${pinned_version}" "${latest_version}" "${NSS_TAGS_URL}"
}

check_gnu_runtime_releases() {
  check_http_numeric_release \
    "GNU coreutils" \
    "$(jq -er '.nixRuntime.coreutils' <<<"${manifest_json}")" \
    "${COREUTILS_RELEASES_URL}" \
    'coreutils-([0-9]+\.[0-9]+)\.tar\.xz'
  check_http_numeric_release \
    "GNU findutils" \
    "$(jq -er '.nixRuntime.findutils' <<<"${manifest_json}")" \
    "${FINDUTILS_RELEASES_URL}" \
    'findutils-([0-9]+\.[0-9]+\.[0-9]+)\.tar\.xz'
  check_http_numeric_release \
    "GNU sed" \
    "$(jq -er '.nixRuntime.gnused' <<<"${manifest_json}")" \
    "${GNU_SED_RELEASES_URL}" \
    'sed-([0-9]+\.[0-9]+)\.tar\.xz'
}

check_go() {
  local pinned_version latest_version

  pinned_version="$(jq -er '.nixRuntime.go' <<<"${manifest_json}")"
  if ! fetch_http_body "Go ${GO_COMPATIBILITY_BRANCH}" "${GO_RELEASES_URL}"; then
    return
  fi
  if ! latest_version="$(
    jq -er --arg branch "${GO_COMPATIBILITY_BRANCH}" '
      if type != "array" then
        error("expected release array")
      else
        [
          .[]
          | select(.stable == true)
          | .version
          | select(type == "string")
          | select(test("^go" + ($branch | gsub("\\."; "\\.")) + "\\.[0-9]+$"))
          | sub("^go"; "")
          | {version: ., components: (split(".") | map(tonumber))}
        ]
        | if length == 0 then error("no stable release on approved branch") else max_by(.components).version end
      end
    ' <<<"${http_body}"
  )" || [[ -z "${latest_version}" ]]; then
    record_failure "Go ${GO_COMPATIBILITY_BRANCH}: primary upstream response had an invalid schema or no stable branch release"
    return
  fi

  compare_pinned_version "Go ${GO_COMPATIBILITY_BRANCH}" "${pinned_version}" "${latest_version}" "${GO_RELEASES_URL}"
}

check_jq() {
  local pinned_version latest_version

  pinned_version="$(jq -er '.nixRuntime.jq' <<<"${manifest_json}")"
  if ! fetch_github_body "jq" '/repos/jqlang/jq/releases/latest'; then
    return
  fi
  if ! latest_version="$(
    jq -er '
      if type != "object" or (.draft != false) or (.prerelease != false) then
        error("expected stable release object")
      else
        .tag_name | capture("^jq-(?<version>[0-9]+\\.[0-9]+\\.[0-9]+)$").version
      end
    ' <<<"${github_body}"
  )" || [[ -z "${latest_version}" ]]; then
    record_failure "jq: official latest-release response had an invalid schema or tag"
    return
  fi

  compare_pinned_version "jq" "${pinned_version}" "${latest_version}" "https://github.com/jqlang/jq/releases/latest"
}

check_krb5() {
  local escaped_branch

  escaped_branch="${KRB5_COMPATIBILITY_BRANCH//./\\.}"
  check_http_numeric_release \
    "MIT Kerberos ${KRB5_COMPATIBILITY_BRANCH}" \
    "$(jq -er '.nixRuntime.krb5' <<<"${manifest_json}")" \
    "${KRB5_RELEASES_URL}" \
    "krb5-(${escaped_branch}\\.[0-9]+)\\.tar\\.gz"
}

check_postgresql() {
  local pinned_version latest_version

  pinned_version="$(jq -er '.nixRuntime.postgresql' <<<"${manifest_json}")"
  if ! fetch_http_body "PostgreSQL ${POSTGRESQL_COMPATIBILITY_BRANCH}" "${POSTGRESQL_RELEASES_URL}"; then
    return
  fi
  if ! latest_version="$(
    jq -er --arg branch "${POSTGRESQL_COMPATIBILITY_BRANCH}" '
      if type != "array" then
        error("expected versions array")
      else
        [.[] | select(.major == $branch and .supported == true)]
        | if length != 1 then
            error("expected exactly one supported approved branch")
          elif (.[0].latestMinor | type) != "string" or (.[0].latestMinor | test("^[0-9]+$") | not) then
            error("invalid latestMinor")
          else
            .[0].major + "." + .[0].latestMinor
          end
      end
    ' <<<"${http_body}"
  )" || [[ -z "${latest_version}" ]]; then
    record_failure "PostgreSQL ${POSTGRESQL_COMPATIBILITY_BRANCH}: primary upstream response had an invalid schema or no supported branch release"
    return
  fi

  compare_pinned_version \
    "PostgreSQL ${POSTGRESQL_COMPATIBILITY_BRANCH}" \
    "${pinned_version}" \
    "${latest_version}" \
    "${POSTGRESQL_RELEASES_URL}"
}

check_python() {
  local escaped_branch

  escaped_branch="${PYTHON_COMPATIBILITY_BRANCH//./\\.}"
  check_http_numeric_release \
    "Python ${PYTHON_COMPATIBILITY_BRANCH}" \
    "$(jq -er '.nixRuntime.python' <<<"${manifest_json}")" \
    "${PYTHON_RELEASES_URL}" \
    "href=\"(${escaped_branch}\\.[0-9]+)/\""
}

check_socat() {
  local pinned_version escaped_branch primary_version mirror_json mirror_version

  pinned_version="$(jq -er '.nixRuntime.socat' <<<"${manifest_json}")"
  escaped_branch="${SOCAT_COMPATIBILITY_BRANCH//./\\.}"
  if ! fetch_http_body "socat ${SOCAT_COMPATIBILITY_BRANCH} official releases" "${SOCAT_RELEASES_URL}"; then
    return
  fi
  if ! parse_latest_numeric_version \
    "socat ${SOCAT_COMPATIBILITY_BRANCH} official releases" \
    "${http_body}" \
    "socat-(${escaped_branch}\\.[0-9]+\\.[0-9]+)\\.tar\\.(?:gz|bz2|xz)"; then
    return
  fi
  primary_version="${latest_numeric_version}"

  if ! fetch_http_body \
    "socat ${SOCAT_COMPATIBILITY_BRANCH} authenticated tag mirror" \
    "${SOCAT_AUTHENTICATED_MIRROR_URL}"; then
    return
  fi
  if [[ "${http_body}" != ")]}'"$'\n'* ]]; then
    record_failure "socat ${SOCAT_COMPATIBILITY_BRANCH}: authenticated mirror response lacked its JSON anti-XSSI prefix"
    return
  fi
  mirror_json="${http_body#*$'\n'}"
  if ! mirror_version="$(
    jq -er --arg branch "${SOCAT_COMPATIBILITY_BRANCH}" '
      if type != "object" then
        error("expected refs object")
      else
        [
          keys[]
          | select(test("^refs/tags/tag-" + ($branch | split(".") | join("\\.")) + "\\.[0-9]+\\.[0-9]+$"))
          | sub("^refs/tags/tag-"; "")
          | {version: ., components: (split(".") | map(tonumber))}
        ]
        | if length == 0 then error("no stable tags on approved branch") else max_by(.components).version end
      end
    ' <<<"${mirror_json}"
  )" || [[ -z "${mirror_version}" ]]; then
    record_failure "socat ${SOCAT_COMPATIBILITY_BRANCH}: authenticated mirror response had an invalid schema or no stable branch tag"
    return
  fi

  if [[ "${primary_version}" != "${mirror_version}" ]]; then
    record_failure "socat ${SOCAT_COMPATIBILITY_BRANCH}: official index reports ${primary_version}, authenticated tag mirror reports ${mirror_version}"
    return
  fi

  compare_pinned_version \
    "socat ${SOCAT_COMPATIBILITY_BRANCH}" \
    "${pinned_version}" \
    "${primary_version}" \
    "${SOCAT_RELEASES_URL} corroborated by ${SOCAT_AUTHENTICATED_MIRROR_URL}"
}

check_zlib() {
  check_http_numeric_release \
    "zlib" \
    "$(jq -er '.nixRuntime.zlib' <<<"${manifest_json}")" \
    "${ZLIB_RELEASES_URL}" \
    'zlib-([0-9]+\.[0-9]+\.[0-9]+)\.tar\.(?:xz|gz)'
}

check_nix_runtime_upstreams() {
  check_glibc
  check_bash
  check_elfutils
  check_iproute2
  check_cacert
  check_gnu_runtime_releases
  check_go
  check_jq
  check_krb5
  check_postgresql
  check_python
  check_socat
  check_zlib
}

check_rust_toolchains() {
  local channel latest_version label pinned_version overlay_rev overlay_hash

  if ! fetch_http_body "Rust stable channel" "${RUST_STABLE_CHANNEL_URL}"; then
    return
  fi
  channel="${http_body}"
  latest_version="$(
    awk '
      $0 == "[pkg.rust]" { in_rust = 1; next }
      in_rust && /^\[/ { exit }
      in_rust && /^version = "/ {
        if (match($0, /[0-9]+\.[0-9]+\.[0-9]+/)) {
          print substr($0, RSTART, RLENGTH)
          exit
        }
      }
    ' <<<"${channel}"
  )"

  while IFS=$'\t' read -r label pinned_version overlay_rev overlay_hash; do
    if [[ -z "${latest_version}" ]]; then
      record_failure "${label}: could not parse the official stable Rust channel"
    elif [[ "${pinned_version}" != "${latest_version}" ]]; then
      record_failure "${label}: pinned ${pinned_version}, latest stable ${latest_version} (${RUST_STABLE_CHANNEL_URL})"
    else
      echo "CURRENT: ${label} ${pinned_version}"
    fi

    if [[ ! "${overlay_rev}" =~ ^[0-9a-f]{40}$ ]]; then
      record_failure "${label}: rust-overlay revision must be a full 40-character lowercase commit SHA"
    fi
    if [[ ! "${overlay_hash}" =~ ^sha256- ]]; then
      record_failure "${label}: rust-overlay source must use an immutable SRI sha256 hash"
    fi
  done < <(
    jq -er '
      ["Application Rust", .appRust.version, .appRust.overlayRev, .appRust.overlayHash],
      ["Nitro helper Rust", .nitroRust.version, .nitroRust.overlayRev, .nitroRust.overlayHash]
      | @tsv
    ' <<<"${manifest_json}"
  )
}

check_github_release_pin() {
  local name="$1"
  local version="$2"
  local owner="$3"
  local repo="$4"
  local pinned_tag="$5"
  local pinned_rev="$6"
  local repository
  local latest_release latest_tag encoded_tag tagged_release resolved_commit

  if [[ -z "${version}" || -z "${owner}" || -z "${repo}" || -z "${pinned_tag}" || -z "${pinned_rev}" ]] \
    || [[ "${version}" == "null" || "${owner}" == "null" || "${repo}" == "null" || "${pinned_tag}" == "null" || "${pinned_rev}" == "null" ]]; then
    record_failure "${name}: version, owner, repo, tag, and rev must all be non-empty strings"
    return
  fi

  repository="${owner}/${repo}"

  if [[ ! "${pinned_rev}" =~ ^[0-9a-f]{40}$ ]]; then
    record_failure "${name} (${repository}): rev must be a full 40-character lowercase commit SHA"
    return
  fi

  if ! fetch_github_body "${name} (${repository})" "/repos/${repository}/releases/latest"; then
    return
  fi
  latest_release="${github_body}"
  if ! latest_tag="$(
    jq -er '
      if type != "object" or (.draft != false) or (.prerelease != false) then
        error("expected stable release object")
      else
        .tag_name | select(type == "string" and length > 0)
      end
    ' <<<"${latest_release}"
  )" || [[ -z "${latest_tag}" ]]; then
    record_failure "${name} (${repository}): official latest-release response had an invalid schema or tag"
    return
  fi

  if [[ "${pinned_tag}" != "${latest_tag}" ]]; then
    record_failure "${name} (${repository}): pinned ${pinned_tag}, latest official release ${latest_tag}"

    # A stale pin must still refer to a real published release, not merely an
    # arbitrary repository tag.
    encoded_tag="$(urlencode "${pinned_tag}")"
    if ! fetch_github_body \
      "${name} (${repository}) pinned release ${pinned_tag}" \
      "/repos/${repository}/releases/tags/${encoded_tag}"; then
      return
    fi
    tagged_release="${github_body}"
    if ! jq -e --arg tag "${pinned_tag}" '
      type == "object" and
      .draft == false and
      .prerelease == false and
      .tag_name == $tag
    ' <<<"${tagged_release}" >/dev/null; then
      record_failure "${name} (${repository}): ${pinned_tag} is not an exact published release tag"
    fi
  else
    echo "CURRENT: ${name} ${pinned_tag} (${repository})"
  fi

  encoded_tag="$(urlencode "${pinned_tag}")"
  if ! fetch_github_body \
    "${name} (${repository}) release commit ${pinned_tag}" \
    "/repos/${repository}/commits/${encoded_tag}"; then
    return
  fi
  if ! resolved_commit="$(jq -er '.sha | select(test("^[0-9a-f]{40}$"))' <<<"${github_body}")" \
    || [[ -z "${resolved_commit}" ]]; then
    record_failure "${name} (${repository}): official commit response had an invalid schema or SHA"
    return
  fi

  if [[ "${pinned_rev}" != "${resolved_commit}" ]]; then
    record_failure "${name} (${repository}): ${pinned_tag} resolves to ${resolved_commit}, manifest rev is ${pinned_rev}"
  fi
}

main() {
  local nitro_count=0

  require_command git
  require_command jq
  require_command nix

  if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: security upstream manifest not found: ${MANIFEST}" >&2
    exit 2
  fi

  # The manifest is executable Nix syntax. Evaluate an exact in-store copy in
  # restricted, offline mode so a modified manifest cannot introduce ambient
  # filesystem or network access before the structural checks run.
  manifest_json="$(evaluate_manifest_restricted)"

  check_manifest_structure
  if [[ "${mode}" == "--offline" ]]; then
    if (( failures > 0 )); then
      echo "Offline security-pin validation failed with ${failures} finding(s)." >&2
      exit 1
    fi
    echo "Security-pin structure, lockfile, URLs, and toolchain declarations are consistent."
    exit 0
  fi

  require_command curl
  require_command head

  echo "Checking immutable security-source pins against primary upstream release metadata..."
  check_nixpkgs
  check_kernel
  check_openssl
  check_rust_toolchains
  check_nix_runtime_upstreams
  check_nitro_util_reviewed_head

  check_github_release_pin \
    "Continuum proxy" \
    "$(jq -er '.continuumProxy.version' <<<"${manifest_json}")" \
    "$(jq -er '.continuumProxy.owner' <<<"${manifest_json}")" \
    "$(jq -er '.continuumProxy.repo' <<<"${manifest_json}")" \
    "$(jq -er '.continuumProxy.tag' <<<"${manifest_json}")" \
    "$(jq -er '.continuumProxy.rev' <<<"${manifest_json}")"

  while IFS=$'\t' read -r name version owner repo tag rev; do
    nitro_count=$((nitro_count + 1))
    check_github_release_pin "${name}" "${version}" "${owner}" "${repo}" "${tag}" "${rev}"
  done < <(
    jq -er '
      .nitro
      | to_entries[]
      | [.key, .value.version, .value.owner, .value.repo, .value.tag, .value.rev]
      | @tsv
    ' <<<"${manifest_json}"
  )

  if (( nitro_count == 0 )); then
    record_failure "manifest contains no Nitro/CRT source entries"
  fi

  if (( failures > 0 )); then
    echo "Security upstream freshness check failed with ${failures} finding(s)." >&2
    echo "Review upstream release notes, update version/tag/rev/hash together, and rebuild before merging." >&2
    exit 1
  fi

  echo "All security-source pins are current and their release tags resolve to the pinned commits."
}

main "$@"
