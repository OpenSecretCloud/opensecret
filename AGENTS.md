# OpenSecret agent guide

This file applies to the entire repository. It contains the durable rules for
developing and reviewing the OpenSecret backend. Detailed, task-specific
procedures live under `.agents/skills/`; load the matching skill before doing
that work.

## Start here

1. Confirm the checkout, branch, remote state, and worktree status. Preserve
   unrelated changes; never rewrite a dirty checkout to make it match a task.
2. Initialize submodules with `git submodule update --init --recursive`.
3. Enter the pinned toolchain with
   `OPENSECRET_DEV_CONTAINERS=0 nix develop` unless the task needs the local
   container helpers. The stateful shell hook checks any PostgreSQL listener
   on `localhost:$PGPORT` before initializing this checkout, so a responding
   listener is reused and `.pgdata`, its role, and its database are not
   guaranteed to be the active state. It also creates `.env` from
   `.env.sample` when absent. On Linux, leaving container setup enabled rewrites
   user-level container configuration.
4. Run `just diesel-migration-run-local` before starting the backend. Startup
   does not run Diesel schema migrations; `src/migrations.rs` performs a
   different application-data migration.
5. Configure the required provider credentials in ignored files or the local
   environment. Tinfoil is a hard runtime dependency. Never commit credentials,
   generated `.env`, local databases, provider captures, or decrypted traffic.
6. Read the relevant route, middleware, model, migration, and tests before
   choosing placement. Exact route, model, and provider inventories drift;
   derive them from current source rather than copying an old table.

This repository is one Rust package and binary, not a Cargo workspace. Run
Cargo commands at the repository root.

## Architecture and ownership

- `src/main.rs` composes configuration, shared state, middleware, and the
  router. Keep bootstrap and wiring here; put domain behavior in focused
  modules.
- `src/web/` owns HTTP transport, request validation, authentication context,
  encryption middleware, response/error mapping, and route orchestration.
- `src/web/responses/` owns the stateful Responses implementation: persistent
  items and conversations, tool loops, encrypted SSE projection,
  continuation, cancellation, and deletion.
- `src/models/`, `src/db.rs`, `migrations/`, and `src/models/schema.rs` own
  persistence. A schema change normally spans a reversible SQL migration,
  generated Diesel schema, model/query code, scoping, and tests.
- `src/encrypt.rs`, `src/seed_wrapping.rs`, `src/jwt.rs`, attestation/session code,
  and `src/security_invariants.rs` define cryptographic and authentication
  boundaries. Do not duplicate their policy in route handlers.
- `src/model_config.rs` owns the public model catalog and capabilities.
  `src/provider_routing.rs` owns provider selection and upstream model IDs.
  `src/proxy_config.rs` owns provider endpoints/credentials, and
  `src/provider_client.rs` owns provider transport, attestation-aware clients,
  streaming, and retry decisions. `src/web/openai.rs` and
  `src/web/responses/` own route-specific usage observation, normalization,
  metering, and event capture.
- `src/brave.rs` and `src/kagi.rs` own provider-specific search/extraction
  adapters. Keep public web routes provider-neutral.
- Billing, flags, email, OAuth providers, and event queues are external API
  boundaries. Their administrative credentials remain server-side and their
  absence/failure behavior must be explicit in the feature being changed.

OpenSecret owns authentication/session encryption, encrypted-at-rest
persistence, provider credentials/routing/model canonicalization, entitlement
enforcement, and usage capture. Clients own UI and device behavior. Published
OpenSecret SDKs own attestation, encrypted transport, and client retry policy;
do not advise Maple or another client to call protected routes with plain
`fetch`, `curl`, or `reqwest`.

## API contract rules

- Health and attestation/key-establishment endpoints are explicit public
  plaintext exceptions. Authentication bootstrap routes use a live encrypted
  session without requiring an existing JWT. Protected application routes add
  route-appropriate JWT or API-key authentication to that session. Bodyless
  protected routes must still validate and touch the session. Treat
  `/platform/*` as a separate control-plane surface and derive its policy from
  current router assembly rather than these Maple-facing rules.
- OpenAI-shaped paths describe the decrypted payload contract; they are not a
  plaintext OpenAI wire API. A valid smoke test uses an OpenSecret SDK or Maple
  to perform attestation/key exchange, sends `x-session-id`, encrypts the
  request, and decrypts the response or SSE frames.
- Derive each route's exact authentication from router assembly. JWT and API
  key auth have different contexts; API-key requests do not automatically have
  a user's storage keys. Do not add persistence to an API-key route without an
  explicit authorization and key-ownership design.
- Validate method, path/query, decrypted request schema, status, content type,
  error shape, and response schema. For streaming, also preserve event order,
  sequence semantics, usage, cancellation, encryption, and exactly one
  terminal condition.
- Reject malformed data at the boundary and return stable, sanitized errors.
  Never leak internal provider bodies, credentials, decrypted content, SQL
  details, or cryptographic errors to clients.
- Responses is stateful and persistent. Do not describe `store: false` as
  ephemeral unless current source and tests prove that behavior.
- API changes require a compatibility review of published SDKs and all affected
  Maple paths. Keep a backwards-compatible transition when clients can update
  independently.

Load `$change-opensecret-api` for route, payload, auth, SSE, persistence, or
client-contract work.

## Provider and model rules

- Keep public model IDs separate from provider IDs. Canonicalize at the
  OpenSecret boundary so upstream aliases never leak into stored or returned
  API state.
- Model availability/capabilities, provider routing weights and stickiness,
  endpoint/credential resolution, transport, and route orchestration are
  different concerns. Change each in its owning layer.
- Provider secrets, administrative headers, and raw attestation material never
  enter client-visible responses or logs. Review forwarded headers explicitly;
  prefer an allowlist for a new boundary.
- Retry only operations known to be safe. Connection establishment before a
  request is sent is different from an ambiguous failure after a POST may have
  reached a provider. Preserve Tinfoil attestation/origin binding and refresh
  only on its typed connection-establishment failure.
- Streaming adapters must handle bounded frames, cancellation, protocol errors,
  usage, and a single terminal marker. A provider-direct test does not prove
  OpenSecret authentication, E2EE, persistence, billing, routing, or Maple.
- Usage is security- and billing-relevant. Normalize it once, retain model and
  provider attribution internally, and test missing/partial provider usage.

Load `$change-opensecret-provider` for model catalog, routing, credentials,
provider transport, retries, search/extraction adapters, or usage work.

## Persistence and migration rules

- Use a new timestamped Diesel migration. Never edit an already-deployed
  migration to change history.
- Run the complete migration chain against an empty disposable database. When
  `down.sql` or data transformation changes, also test down/up and an
  upgrade-shaped database with representative encrypted rows.
- Regenerate and review `src/models/schema.rs`; do not hand-edit it into a state
  the migration does not produce.
- Preserve user/project scoping in the database trait and concrete query.
  Filtering in the handler after an unscoped read is not authorization.
- Identify the owning key domain before changing an encrypted field.
  User-private content uses the credential-derived user-key path; server-owned
  material such as OAuth access tokens, project secrets, and password
  verifiers remains in its enclave/system-secret domain. Do not make one
  domain's migration depend on a key available only in another.
- Version persisted ciphertext formats. For rows encrypted by a user key,
  deploy dual-read/new-write behavior and lazily perform the authenticated
  rewrite only after that user's key is available. A startup migration may
  rewrite enclave/system-key data only when its owning key is available there;
  a SQL migration alone cannot re-encrypt opaque ciphertext safely.
- Preserve tamper detection, purpose and row binding, seed-wrap/source
  invariants, transactionality, zeroization where present, and non-secret
  indexes/metadata only where the design permits them.
- Use only a dedicated disposable Postgres database for ignored tamper/OAuth
  suites. Fail closed on database identity, migration status, selected-test
  count, skip output, pass count, and cleanup; a successful test-process exit
  is insufficient evidence that an ignored database test executed.

## Security and privacy invariants

- Treat source-confirmed behavior, test-confirmed behavior, deployed
  configuration, and live-environment observations as distinct evidence.
  Source alone cannot prove deployed PCRs, KMS/IAM policy, logging policy,
  network placement, or which artifact is serving production.
- Never log secrets, session material, tokens, OAuth provider payloads,
  prompts, reasoning, response deltas, decrypted bodies, user email/profile
  data, provider bodies, or raw headers. Metadata must be bounded and
  allowlisted. Logs leave the enclave boundary, so `trace` is not a safe place
  for sensitive content.
- Preserve bounded, expiring, one-use or leased state for pending attestations,
  sessions, OAuth state, and other unauthenticated resources. Cleanup paths and
  failure paths receive the same bounds as success paths.
- Preserve session liveness and key ownership across every protected route.
  Cryptographic review must cover nonce generation, direction separation,
  additional authenticated data, replay behavior, bodyless requests, error
  encryption, and concurrency—not only the happy-path primitive.
- Authentication changes must review access and refresh token lifecycle,
  revocation/logout, reset and verification codes, OAuth state/nonce binding,
  account linking, native callbacks, and enumeration-resistant errors.
- Billing and flags may fail differently across registered, guest, chat,
  Responses, web, audio, and other paths. Treat that matrix as an explicit
  product/security decision; do not generalize one route's behavior.
- Provider response data and model output are untrusted. Bound sizes and loops,
  validate content types/schemas, enforce URL provenance and SSRF policy, and
  avoid replaying hidden provider state without evidence.

Load `$review-opensecret-security` before changing cryptography, auth, OAuth,
sessions, storage encryption, provider headers, billing enforcement, URL
fetching, logging, Nitro/KMS integration, or another trust boundary.

## Development workflow

For the normal local macOS stack:

```bash
git submodule update --init --recursive
OPENSECRET_DEV_CONTAINERS=0 nix develop
just build-local-proxies-macos
just diesel-migration-run-local
```

Then run the Continuum proxy and backend in separate shells:

```bash
nix develop -c just run-continuum-proxy-macos
nix develop -c just run-local-backend-macos
```

The Tinfoil client runs in-process; there is no local Tinfoil sidecar. See
`docs/local-macos-stack.md` and `$develop-opensecret` for credential files,
ports, alternative direct-provider configuration, and debugging.

For read-only or CI-style checks, prevent the Nix shell hook from starting
Postgres, writing `.env`, or changing Linux container configuration:

```bash
OPENSECRET_DEV_POSTGRES=0 \
OPENSECRET_DEV_ENV=0 \
OPENSECRET_DEV_CONTAINERS=0 \
nix develop --no-write-lock-file -c <command>
```

## Validation and evidence

The Rust CI gate is:

```bash
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check

OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c \
  env RUSTFLAGS='-D warnings' \
  cargo clippy --locked --all-targets --all-features -- -D warnings

OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c \
  env RUSTFLAGS='-D warnings' cargo test --locked --all-features
```

Run focused tests while iterating, then the full gate. Add
`nix flake check --print-build-logs` for Nix, entrypoint, Nitro helper, kernel,
or security-build changes. Current-host flake checks do not prove ARM Linux EIF
construction or PCR reproducibility.

Default CI has no PostgreSQL service and skips all ignored tests. Never run all
ignored tests as one group:

- the AEAD tamper and OAuth database subsets require a fully migrated,
  disposable database via `AEAD_TAMPER_TEST_DATABASE_URL` and serialized test
  execution;
- the Tinfoil parity test requires a live credential and network and must be
  run separately with explicit authorization.

`/health-check` is process liveness only and does not query PostgreSQL.
`/health-check-extended` checks Tinfoil model-list connectivity only. Neither
proves auth, E2EE, persistence, routing, billing, flags, or full-stack behavior.
Use `$validate-opensecret` for the exact risk matrix, ignored-test commands,
SDK-driven encrypted smoke, and evidence report.

## Review and authority

Review the changed contract end to end: route assembly, middleware order,
cryptographic ownership, authorization/scoping, persistence, provider effects,
usage, error/log content, cancellation, tests, and client compatibility. Cite
current source for claims and state what remains unverified.

Building an EIF is not deployment. PCR comparison, KMS policy changes, copying
artifacts to hosts, terminating/running enclaves, restarting proxies, writing
secrets, and invoking any `deploy-*`, `stage-*`, `run-eif-*`, or PCR mutation
recipe require explicit operator authorization. Treat PCR verification as a
gate only when it compares the built result with a reviewed, checked-in
reference through the matching workflow; a recipe that performs no comparison
is not PCR evidence.

## Skills

- `$develop-opensecret`: local setup, Postgres, environment, migrations,
  process topology, placement, and implementation loop.
- `$change-opensecret-api`: routes, auth context, encrypted payloads, errors,
  SSE, Responses persistence, and client compatibility.
- `$change-opensecret-provider`: model catalog, routing, provider credentials,
  transport/attestation, retries, usage, and search/extraction adapters.
- `$validate-opensecret`: focused tests, CI parity, disposable-database suites,
  provider probes, encrypted API/full-stack smoke, and evidence reporting.
- `$review-opensecret-security`: cryptography, attestation/session/auth/OAuth,
  persistence, provider headers, logging, billing policy, and deployment-trust
  review.

## Maintaining this guidance

Treat this guide and the repository skills as living operational documentation,
not infallible rules. Re-check prescriptive language against the current source,
tooling, and architecture. If guidance appears stale, materially wrong,
unnecessarily absolute, or repeatedly creates development friction, surface the
mismatch and confirm the intended correction with the user before changing it.
Do not churn guidance for stylistic preferences or isolated nits; do narrow
words such as "always" or "never" when they claim more than the invariant
actually requires.

Update the relevant guide or skill in the same branch when a material change
adds, removes, or rearchitects a workflow, ownership boundary, validation path,
or recurring development procedure. If you discover unrelated drift, keep the
current task scoped and propose a standalone change with evidence and reasoning.
Add a new skill when a genuinely reusable workflow does not fit an existing one;
otherwise prefer improving or consolidating current guidance. Ground every
update in current source or executed workflow experience, keep it concise, and
validate every command or path it prescribes.
