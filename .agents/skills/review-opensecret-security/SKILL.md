---
name: review-opensecret-security
description: Review security-sensitive OpenSecret backend changes and claims. Use for attestation, client-enclave sessions, AEAD or key derivation, JWT/API-key/OAuth authentication, encrypted persistence, migrations, provider routing, web tools and URL provenance, billing or flag API boundaries, secrets, logging, Nitro/KMS/PCR evidence, streaming, and release-affecting diffs.
---

# Review OpenSecret Security

Review the current source and diff as evidence. This skill defines durable
review standards; it does not catalogue current findings. Default to a
read-only review and implement changes only when the user asks.

## Establish scope and evidence

1. Read the repository `AGENTS.md`, the relevant source, tests, migrations, and
   public documentation before forming a verdict.
2. Confirm the revision and comparison base. Prefer current `master`; do not
   rebase, reset, or fetch over user work merely to obtain it.
3. Inspect the complete change, including generated schema, lockfile, submodule,
   Nix, workflow, environment, and migration changes:

   ```sh
   git status --short --branch
   git diff --stat origin/master...HEAD
   git diff origin/master...HEAD --
   git diff --cached --
   git diff --
   git ls-files --others --exclude-standard
   git submodule status
   git diff --check
   ```

4. Trace each changed value from input through authorization, decryption,
   persistence, provider calls, response encryption, and logs. Do not review a
   handler in isolation from router middleware or database scoping.
5. Label every conclusion with one of these evidence classes:

   - **source-confirmed**: the reviewed revision directly establishes the behavior;
   - **test-confirmed**: a named test exercised the relevant behavior in this run;
   - **build-confirmed**: a named reproducible build or artifact check passed;
   - **live-confirmed**: the exact deployed environment was inspected or exercised;
   - **inferred**: the conclusion follows from source plus stated assumptions;
   - **unverified**: deployment configuration or an external system was not inspected.

Keep claims within the evidence available. Local tests do not prove live DB
TLS, IAM, KMS policy, Nitro measurements, client trust policy, provider
behavior, billing decisions, or log retention.

## Map the trust boundaries

State which boundaries the change crosses:

- Maple or another client to the OpenSecret enclave;
- the public HTTP router to JWT, API-key, OAuth, and project authorization;
- enclave plaintext to host-accessible Postgres or logs;
- OpenSecret to Tinfoil, Continuum/OpenAI-compatible providers, Brave, Kagi,
  OAuth providers, billing, or flags;
- enclave to its parent instance over VSOCK for AWS credentials, secrets, and logs;
- source to Nix build, EIF/PCR evidence, CI, KMS policy, and deployed enclave.

Distinguish the client's attestation of OpenSecret from OpenSecret's attested
connection to an upstream Tinfoil enclave. A local mock attestation exercises
protocol shape only; it is not Nitro or production trust evidence.

## Enforce placement rules

Keep these responsibilities in the backend:

- authentication, authorization, project scoping, token issuance, and revocation policy;
- provider selection, provider model IDs, provider-only request fields, retries,
  provider credentials, and upstream error sanitization;
- URL authorization and SSRF policy for model-initiated tools;
- billing and feature-flag decisions at their HTTP API boundary;
- encryption-at-rest formats, migrations, destructive-reset behavior, and audit logs.

Keep presentation and user interaction in Maple. Do not make Maple select a
confidential provider, supply provider secrets, or reproduce backend policy.
Changes to attestation documents, key exchange, encrypted envelopes, headers,
streaming, token lifecycle, or error schemas are shared protocol changes: update
and test backend and client implementations together with an explicit version or
compatibility plan.

Treat billing and flags as external APIs only. Review OpenSecret's request,
response, timeout, caching, authentication, and failure semantics without
relying on server-internal implementation details. Local setup may point those
clients at approved development or production endpoints as appropriate.

## Review client-enclave transport

Inspect `src/web/attestation_routes.rs`, `src/web/encryption_middleware.rs`, the
session/cache implementation in `src/main.rs`, and every affected router.

- Require a fresh, unpredictable client nonce and one-time ephemeral key use.
- Verify nonce/body limits, TTLs, capacities, eviction, lease lifetime, and
  cleanup on malformed, abandoned, concurrent, and failed requests.
- Reject non-contributory X25519 inputs and zeroize secret/session material.
- Confirm every sensitive route validates a live encryption session, including
  GET, DELETE, and handlers with `()` bodies, before side effects.
- Separately confirm route authentication and authorization. Possession of an
  `x-session-id` is not automatically proof of JWT identity, project membership,
  or durable-credential possession.
- For AEAD, review key direction, domain separation, nonce uniqueness, canonical
  AAD, method/path/query/session/request binding, replay handling, and response
  association. Parser differences are not cryptographic domain separation.
- Keep plaintext credentials and sensitive payloads out of query strings, URLs,
  and other host-visible metadata.
- Move blocking NSM or cryptographic work off async executor threads and bound
  admission if the current path can be driven publicly.

Do not silently change the existing encrypted protocol. Require compatibility
vectors for success, tamper, wrong-session, wrong-route, replay, bodyless, error,
and streaming cases.

## Review authentication and OAuth

Inspect `src/jwt.rs`, `src/web/openai_auth.rs`, login routes, `src/oauth.rs`,
OAuth routes, seed wrapping, and the router composition in `src/main.rs`.

- Keep internal access/refresh tokens distinct from project-issued third-party
  tokens. Confirm algorithm, audience, token format, subject, project, and auth
  context for the exact token domain.
- Issue or refresh user tokens only after the signed `AuthContext` matches the
  user project and an active credential-bound seed wrap successfully verifies.
- Preserve the signed auth context during refresh; do not silently recompute it
  from mutable database fields.
- Treat OpenAI-compatible API-key auth as a separate capability. Do not derive or
  expose user-private storage keys unless an API-key-bound seed-wrap design exists.
- Check expiry, maturity, clock-skew, rotation, replay, logout, and revocation as
  separate properties. Never infer server-side revocation from client token deletion.
- Make OAuth state random, bounded, expiring, exact-match, and atomically one-use.
  Review whether it is bound to the initiating encrypted session and redirect flow.
- Derive OAuth identity from verified provider subject plus project. Distinguish
  signed claims from client hints, especially native Apple nonce, email, and name.
- Unwrap the credential-bound seed before issuing tokens. Test provider-subject,
  project, verifier, wrap, and reset-row substitution.

## Review encrypted persistence and migrations

Inspect `src/encrypt.rs`, `src/seed_wrapping.rs`, `src/db.rs`, affected models,
`src/models/schema.rs`, SQL migrations, `src/migrations.rs`,
`src/security_invariants.rs`, and `src/aead_db_tamper_tests.rs`.

- For new sensitive data, define a versioned, domain-separated envelope with a
  purpose-specific HKDF key and canonical AAD covering the necessary owner,
  project, row, field, credential kind, and version.
- Never change an existing ciphertext format in place. Design dual-read/write,
  rollback, restart, and deletion behavior explicitly, then remove temporary
  translation code after rollout.
- Test round trip, single-bit tamper, wrong key, wrong AAD, row/account/project
  substitution, credential change, destructive reset, and concurrent stale state.
- Preserve transaction boundaries around credential verifier, seed wrap, user,
  provider connection, and reset changes.
- Classify every new user-key-encrypted table in the destructive-reset invariant.
  Confirm ownership foreign keys, cascades, uniqueness, checks, and access-path indexes.
- Distinguish Diesel schema migrations from application-data migrations.
  Ordinary startup code does not have each user's credential-derived key. A
  user-private ciphertext change therefore needs versioned dual-read/new-write
  behavior and lazy rewrite after authenticated key recovery, unless a separate
  authorized backfill is explicitly designed. Startup rewriting is suitable
  only for data whose system/enclave key is actually available; it must be
  idempotent, restart-safe, bounded, and fail closed.
- Verify what API options such as `store` actually do by tracing persistence;
  do not promise ephemeral behavior from a response field or comment alone.
- Run migration and tamper checks only against a disposable, migrated local database.

## Review providers, headers, and streaming

Inspect `src/provider_routing.rs`, `src/proxy_config.rs`,
`src/provider_client.rs`, model configuration, and the calling route.

- Accept public/canonical model IDs at the API boundary. Keep provider choice,
  provider model translation, feature-flag preference, fallback, and provider-only
  fields behind the backend routing boundary.
- Load credentials only from approved runtime secret inputs. Replace inbound
  authorization with the configured upstream credential and mark secret headers
  sensitive. Never log or return provider keys.
- Review every forwarded header. Prefer an explicit allowlist for a new
  boundary, and prove that client authorization, proxy credentials, hop-by-hop
  headers, `Connection`-named headers, host, content length, and
  provider-managed fields cannot cross.
- For Tinfoil, preserve discovery, attestation, TLS pinning, single-flight refresh,
  bounded attempts, and connection reuse. Retry only failures known to occur
  before request bytes were accepted; do not replay ambiguous completion requests.
- Treat cache namespace material separately from API credentials and from a
  client-held secret. Document who can derive it and its stability requirements.
- Preserve decrypted HTTP and SSE contracts: method, path/query, body bytes,
  application content type, ordered events, numeric usage, error mapping, and
  exactly one terminal `[DONE]` where the endpoint promises it.
- Sanitize upstream errors by status, category, and bounded identifiers. Raw
  bodies and headers can contain prompts, PII, tokens, or provider diagnostics.

## Review web tools and request provenance

Inspect `src/web/web_safety.rs`, Responses tool execution and prompt rebuilding,
and direct web routes.

- Treat URL provenance as authorization. For model-initiated `open_urls`, allow
  only exact normalized URLs from visible user text or canonical structured URL
  fields emitted by trusted backend formatters.
- Never authorize URLs found in assistant/system text, titles, snippets,
  diagnostics, image parts, or extracted page bodies. Those are untrusted data.
- Preserve HTTPS-only, no-credentials, public-host checks and the private,
  reserved, metadata, IPv4, IPv6, 6to4, and NAT64 defenses.
- Keep direct authenticated extraction and model-generated extraction as separate
  trust models. Do not weaken model provenance because a direct endpoint exists.
- Reassess DNS, redirects, rebinding, and destination revalidation when changing
  which component performs the fetch. Lexical checks for a remote fetcher do not
  automatically secure a new local fetcher.
- Bound query length, URL count, URL length, result count, output size, tool turns,
  timeouts, and retained provenance. Sanitize image/HTML embeds.
- Keep untrusted-content guidance in the rebuilt prompt after tool schemas are
  removed or a continuation resumes prior tool history.
- Add adversarial tests for encoded and literal internal targets, URL variants,
  continuation history, user-versus-assistant provenance, and page-content injection.

## Review billing and flag boundaries

Inspect `src/billing.rs`, `src/os_flags.rs`, their initialization, and every
caller's decision logic.

- Verify the exact external endpoint, authentication header, product/user inputs,
  response schema, timeout, cache key, and cache TTL without describing the
  external service implementation.
- Review missing-client, missing-flag, denial, malformed response, timeout, and
  service-error behavior per route. Do not generalize one route's fail-open,
  fail-closed, fallback, or local-development exception to all features.
- Keep model-access defaults, provider-routing fallback, web-search fallback,
  paid-feature checks, guest policy, and ordinary chat quota checks distinct.
- Treat changes to entitlement failure semantics, metering order, idempotency,
  reservation/settlement, or usage events as backend API/security changes.
- Never commit service keys or production endpoints as secrets. Use documented
  environment variables and approved development/production servers.

## Review logs and errors

Assume enclave stdout/stderr can cross VSOCK to a host-visible logging system.
Treat every log level, including `trace`, as an external disclosure boundary.

- Never log decrypted request/response bodies, prompts, reasoning, instructions,
  metadata, OAuth profiles/emails, authorization headers, cookies, tokens, reset
  codes, passwords/verifiers, session/enclave/provider keys, KMS plaintext, or
  raw credential/secret-service responses.
- Prefer request IDs, operation names, provider names, status classes, bounded
  counts, durations, and sanitized trace IDs. Minimize stable user identifiers.
- Do not assume `Debug` is safe. Inspect custom `Debug` implementations and nested
  error types before logging `{:?}` or propagating an error string.
- Do not log raw upstream error bodies or response headers. Map them to bounded,
  non-sensitive categories and return generic public errors.
- Run the source logging invariants, but also inspect semantically sensitive
  payload structs: identifier-based scanners cannot prove that content is absent.

## Review Nitro, KMS, build, and release evidence

Inspect `flake.nix`, pinned Nitro helper sources, entrypoint/VSOCK code, CI,
environment-specific EIF outputs, PCR references/history tooling, and current
public deployment docs.

- Keep these claims separate:

  1. Rust tests establish local code behavior.
  2. Nix checks establish reviewed source/build invariants.
  3. An EIF build and PCR match establish a reproducible artifact measurement.
  4. Signed history establishes only what the current signature format and key policy cover.
  5. Live KMS/IAM, client trust policy, deployed EIF, parent services, and staging
     smoke establish deployment behavior.

- Verify the current repository's actual release format; do not describe an
  unmerged future signing or transparency design as deployed.
- Review KMS key/alias identity, account, region, encryption context, recipient
  attestation conditions, Secrets Manager identity, parent VSOCK provenance,
  credential permissions, signing-key custody, rollback, and revocation. Mark
  all live-policy claims unverified until inspected.
- Never run PCR copy/update/append/signing, remote migration, SCP, enclave
  terminate/run, debug-console, stage, or deploy recipes without explicit user
  authorization for the exact environment. Verification commands can still
  build artifacts; inspect their recipes first.

## Keep findings task-local

Re-derive behavior from the revision under review. Keep revision-specific
findings in task-local review notes or the requested code-review output rather
than evergreen skills or `AGENTS.md`. Promote only stable engineering rules and
repeatable review methods back into repository guidance.

## Validate proportionally

Use the Nix shell and mirror CI. Disable shell-hook Postgres and `.env` creation
for compile-only/unit checks when they are unnecessary:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c env RUSTFLAGS='-D warnings' \
  cargo clippy --locked --all-targets --all-features -- -D warnings
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c env RUSTFLAGS='-D warnings' \
  cargo test --locked --all-features
git diff --check
```

The full test command includes crypto property tests and source-level security
invariants but skips ignored integration/live tests. For security-sensitive
changes, also name the focused test module or exact test in the report.

For credential, seed-wrap, reset, or encrypted-schema changes, use the
fail-closed disposable-database procedure in `$validate-opensecret`. Do not copy
an abbreviated test command here: the validation procedure verifies the target
cluster and database, confirms the intended tests were discovered and executed,
and keeps live-provider ignored tests separate.

Run the exact live Tinfoil parity test only when the user has authorized egress
and provided an approved credential source, following
`docs/tinfoil-rust-sdk-parity.md`.

Run Linux/ARM EIF and PCR checks only when the change reaches that boundary and
the compatible builder is available. Report local, CI, artifact, and live smoke
evidence separately; do not fill an untested layer with inference.

## Report the review

Lead with the verdict and prioritized findings. For each finding include:

- evidence class and exact file/symbol;
- affected trust boundary and attacker or failure preconditions;
- concrete impact without incident language;
- the invariant or design change required;
- focused regression tests and any coordinated Maple/API migration;
- deployment evidence still needed.

Then list commands actually run, ignored or unavailable checks, and residual
risk. If there are no findings, say which boundaries were checked; do not imply
that uninspected deployment systems are secure.
