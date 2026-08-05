# RNG hardening staging and compatibility

This runbook is for changes that affect the enclave kernel, root filesystem,
Nitro helper closure, entropy admission, or other cryptographic startup paths.
It complements the deployment recipes in the `justfile`; it does not replace
the operator's review of measurements or KMS authorization.

## Compatibility boundary

Unless a change explicitly says otherwise, an RNG-hardening release must not:

- rotate, rewrite, truncate, or replace an existing enclave key;
- change the byte representation of existing encrypted values;
- change the algorithms or derivation rules used for existing key material;
- change KMS runtime policies or secret-delivery semantics;
- silently recover from malformed cryptographic state by generating a new key;
- treat a debug-mode enclave as attestation evidence.

Failing closed on malformed or unavailable cryptographic state is compatible
with this boundary. An error is preferable to mutating an existing value.

## Evidence has three layers

Do not collapse these into one generic “build passed” claim.

1. **Source and evaluation evidence** shows that pins, assertions, and Nix
   expressions are internally consistent.
2. **Linux ARM build evidence** shows that the exact architecture and closure
   used for Nitro can be realized reproducibly.
3. **Non-debug Nitro evidence** shows that the generated EIF boots, attests,
   receives its approved secrets, opens existing state, and serves real traffic.

A macOS build is useful for fast evaluation and application-level checks, but
it is not a substitute for Linux ARM or Nitro evidence.

## Pull-request boundaries

Prefer one independently explainable behavior change per PR. A child PR is
appropriate only when its implementation genuinely requires the parent layer.
Each PR description should state:

- its exact base PR or `master`;
- whether it changes EIF bytes or expected PCR values;
- whether it can affect key identity, formats, KMS behavior, or secrets;
- which checks ran and on which platform;
- the minimum development-enclave smoke test;
- the retained rollback artifact or commit.

When a parent PR in a stack merges, verify that GitHub retargeted the child to
`master` and that its Files changed view still contains only the intended child
delta. A squash merge may require rebuilding or rebasing the child branch.

## Build and measurement flow

Run these steps from the exact commit under review.

### 1. Inspect before building

Confirm the branch, base, submodule revisions, and diff. Do not use a dirty
worktree to produce release evidence.

```bash
git status --short --branch
git submodule status --recursive
git diff --check
nix flake check --no-build --no-write-lock-file
```

The final Linux ARM build remains authoritative for Linux-only checks and EIF
construction.

### 2. Build the development EIF on Linux ARM

```bash
just build-eif-dev
```

Archive the resulting `image.eif`, `pcr.json`, source commit, submodule state,
and a digest of the EIF before copying or deploying it. A later rebuild is not
automatically the same artifact unless its digest and measurements agree.

### 3. Update reviewed PCR files when EIF bytes change

The established command remains:

```bash
just update-pcr-all
```

It builds, signs, and appends PCR entries for both development and production.
It updates the repository files locally; it does **not** review, commit, push,
merge, change KMS policies, or deploy anything by itself. Inspect all four
current/history files before committing them, and confirm that the signing key
and target environments are correct.

If a change should be artifact-neutral, compare the newly built EIF digest and
PCRs rather than assuming they are unchanged.

### 4. Verify and stage the exact development artifact

```bash
just verify-pcr-dev
just deploy-dev-nix
```

The deployment recipe rebuilds before copying. Confirm that its resulting
digest and PCRs match the artifact you reviewed. Use non-debug mode for the
attestation/KMS acceptance result. Debug mode is useful for boot diagnostics,
but its zeroed PCRs are not deployment evidence.

## Minimal development smoke test

For every EIF-changing layer, perform at least:

1. wait for the service to become ready;
2. sign in to an existing account;
3. confirm existing chats or other encrypted state are readable;
4. create a new chat and receive a response;
5. exercise one newly affected path if the PR changes more than startup.

That compact smoke crosses attestation, KMS-backed startup, existing-key reuse,
database decryption, session cryptography, new encrypted writes, and the model
path. It is not exhaustive, but it is a strong compatibility signal.

Add targeted checks only where the PR changes the relevant behavior:

- **Kernel, entropy, or Nitro helper:** cold boot, readiness timing, KMS
  startup, existing account, existing chat, and new chat.
- **Continuum source build with unchanged version:** proxy startup, attestation
  and TLS establishment, model request, streaming response, and rollback to the
  prior EIF.
- **Continuum version update:** the preceding checks plus model discovery,
  transcription if enabled, a long streaming response, interrupted streaming,
  upstream error mapping, and memory observation.
- **VSOCK bootstrap helper:** successful credential and secret retrieval for
  all configured startup requests, plus the ordinary login/chat smoke.
- **Rootfs minimization:** successful boot and startup plus any operational
  command that is intentionally supported from the enclave image.

## Soak and production promotion

Let higher-risk layers soak in development long enough to expose boot, memory,
streaming, or dependency problems. Record the exact tested EIF and the previous
known-good EIF as the rollback pair.

Before production promotion:

- confirm CI corresponds to the exact commit and PCR files under review;
- verify the development smoke used a non-debug enclave;
- confirm the production PCR entry is the intended one;
- retain the last known-good production EIF and its measurements;
- avoid combining an unrelated deployment or KMS-policy change.

After production deployment, use the same minimal smoke: existing login,
existing encrypted state, and one new chat. Roll back on startup, attestation,
KMS, decryption, or sustained model-path failures.

## Held experiments

A held experiment is not part of the merge train. Its PR should remain Draft,
use a conspicuous `HOLD` / `DO NOT MERGE` title, and state the missing decision
or evidence. It must not update production authorization or be treated as an
approved hardening requirement merely because it builds.

Current examples include changing `random.trust_cpu` policy and introducing a
fixed VSOCK response-size limit. Both require Nitro-specific compatibility and
threat-model evidence before adoption.

## Accepted RNG-hardening baseline

The initial review sequence was merged as PRs #251–#264. It retired the legacy
root Docker EIF workflow, made crypto and key-loading failures fail closed,
pinned Linux 6.12.101, required entropy readiness, source-built and modernized
the Nitro helper closure, preserved existing key identity, and refreshed the
stock Nix platform. Operational rollout evidence belongs in the release or PR
record; this runbook intentionally avoids transient branch hashes and live PR
status.

Future work should start from current `master`, not from the original combined
audit PR or its rehearsal commits.
