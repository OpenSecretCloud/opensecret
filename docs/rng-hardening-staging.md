# RNG hardening staging gate

This change deliberately preserves all persistent cryptographic contracts. It
does not change KMS keys, operations, recipient handling, policies, key sizes,
derivations, AEAD constructions, database formats, or client wire formats. Do
not promote the candidate if any existing secret row is inserted, updated,
rewrapped, or regenerated during startup.

## Native aarch64 Linux build

Run these from a clean checkout with the reviewed `flake.lock`:

```sh
df -h /nix
export GITHUB_TOKEN="$(gh auth token)"
just build-eif-dev
```

`build-eif-dev`, `build-eif-prod`, and `build-eif-preview` all run the same
mandatory ARM gate first. It checks live upstream freshness, evaluates every
flake system without building, realizes the complete reviewed check set, and
freshly re-executes the glibc, rootfs, kernel, entropy, and VSOCK semantic
checks. It also fails if a command mutates `flake.lock`. The CI dev/prod paths
invoke this same script before building and upload artifacts only after every
gate and PCR comparison succeeds. Preview remains buildable for staging, but
promotion fails closed until a reviewed `pcrPreview.json` exists.

The helper build must pass its C tests and post-build assertions for static
AWS-LC/s2n/CRT/json-c linkage, `libnsm.so.0`, exact CLI output labels, the SDK
version, and unconditional entropy seeding. The EIF must contain Linux
6.12.101, OpenSSL 3.5.7, the guarded NixOS 26.05 userspace versions, the
source-built v1.50.0 Continuum proxy, the locally built pinned Nitro init, the
realized kernel config as its metadata, and the explicit reviewed command line
from `flake.nix`.

The rootfs is bootstrapped against glibc 2.44-8 rather than partially replacing
the final library alias. Evaluation must prove that glibc, libc, the final
stdenv/compiler, and Bash all refer to the same 2.44 store path. The native
rootfs gate exact-allowlists the reviewed glibc output paths (including
rejecting any stray `locale-glibc` output), rejects BusyBox, checks every
executable ELF interpreter and shared-library resolution, rejects executable
stacks, and harmlessly executes the startup command surface. The kernel
toolchain uses direct elfutils 0.195, whose upstream source fixes the glibc 2.44
C23 qualifier issue; every final-glibc Kerberos build carries the narrowly
guarded upstream compatibility flag. The rootfs also contains direct Bash
5.3p15, findutils 4.11.0, and iproute2 7.1.0 pins; the old incidental BusyBox
applet surface is intentionally absent while `/bin/ip` remains compatible.

glibc's AArch64 ABI audit found no removed exports, and the dynamic interpreter
name is unchanged. Newly built binaries may nevertheless use GLIBC_2.43/2.44
symbols, so rollback must always replace the complete EIF; never mix a new app,
helper, or rootfs with pieces from the previous EIF. Do not set
`GLIBC_TUNABLES=glibc.malloc.hugetlb=0` by default. Measure glibc 2.44's AArch64
transparent-huge-page behavior under Nitro memory pressure and pin that tunable
only if the staging A/B test shows a material RSS, latency, or OOM regression.

These kernel, libc, helper, and rootfs changes necessarily produce a new PCR0.
A mismatch with the checked-in operational reference is expected until the
normal measurement promotion is performed; this change does not update that
reference or any KMS runtime policy.

## Development-enclave compatibility proof

Before booting the candidate, record secret-row counts and keyed or otherwise
non-secret-bearing integrity checks for the existing encrypted dev data. Never
export plaintext keys or tokens for this comparison.

The candidate must then prove all of the following in the development enclave:

- Cold boot reports Linux 6.12.101, the reviewed `/proc/cmdline`, `/dev/nsm`,
  `rng_current` equal to `nsm-hwrng`, and a completed blocking `getrandom` check
  before credentials, KMS, networking, or OpenSecret startup.
- Startup decrypts the existing `enclave_key` and performs no `genkey`, insert,
  update, rewrap, migration, or rotation of existing cryptographic material.
- The new helper decrypts a ciphertext produced by the currently deployed
  helper to the same plaintext, and `decrypt`, `genrandom`, and `genkey` retain
  their exact stdout schemas and byte lengths.
- Existing dev users, tokens, API keys, encrypted records, wallets, and chats
  remain usable. A disposable object created by the candidate remains readable
  after rolling back to the previous dev EIF.
- Exercise both streaming and non-streaming chat error handling against the
  source-built Continuum v1.50.0 proxy. Upstream connection failures now reach
  clients as 502/504 instead of a generic 500, and an upstream stream that ends
  unexpectedly aborts the downstream connection instead of looking complete.
  Maple must show a recoverable error/retry path and must never present an
  interrupted answer as a finished assistant message.
- KMS decrypt, `GenerateRandom`, and disposable `GenerateDataKey` operations
  succeed through the existing recipient flow. The updated CRT intentionally
  offers a newer TLS policy; this wire-level negotiation change must not alter
  KMS ciphertext or stored key material.
- Repeated cold boots succeed. A controlled missing/wrong NSM or RNG-source
  test fails before KMS or application startup, without a local/plaintext
  fallback.
- The deterministic glibc 2.44 native smoke target passes, runtime closure
  contains no older glibc, DNS/socket resolution works for every backend
  dependency, and a staging A/B shows acceptable cold-boot time, RSS, latency,
  and OOM behavior. Nixpkgs marks glibc's raw full `make check` suite as known
  failing, so do not waive arbitrary failures from that suite; if it is run,
  review and pin a minimal expected-failure allowlist before treating it as a
  release gate.
- Post-test secret-row counts and integrity checks match the pre-test snapshot,
  except for explicitly created disposable test records.

Build the candidate twice on independent aarch64-linux builders and compare the
complete EIF bytes, digests, and PCR0. Only those exact bytes, after passing the
checks above, should proceed to the existing operational review and promotion
process. Do not sign, deploy, or modify measurement/KMS policy as part of this
build-validation step.
