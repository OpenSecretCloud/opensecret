---
name: develop-opensecret
description: Set up, run, and navigate the open-source OpenSecret Rust backend. Use when starting work in this repository, preparing its Nix and PostgreSQL environment, initializing submodules, running Diesel migrations, configuring local provider endpoints and credentials, locating the correct implementation layer, or deciding whether routine work belongs in OpenSecret or its frontend client.
---

# Develop OpenSecret

## Establish a safe starting point

1. Read `AGENTS.md` and the files it routes to before changing code.
2. Inspect `git status --short --branch` and preserve all unrelated or pre-existing work.
3. For new work, fetch `origin` and start from current `origin/master` unless the task names another base. Do not switch, rebase, or rewrite a dirty checkout.
4. Initialize the pinned public submodules before building:

   ```sh
   git submodule update --init --recursive
   ```

5. Change a submodule pointer only when the task explicitly requires a dependency upgrade. Review the submodule diff and upstream revision as part of that change.
6. Keep deployment, shared-database migration, PCR, signing, and remote-enclave commands out of routine development. Run them only with explicit authorization for the exact environment.

## Use the repository toolchain

Use the pinned Nix shell instead of installing ad hoc system packages:

```sh
OPENSECRET_DEV_CONTAINERS=0 nix develop
```

The shell provides the pinned Rust toolchain, `rustfmt`, Clippy, PostgreSQL,
Diesel CLI, OpenSSL, `just`, compiler/linker dependencies, and repository
security tools. Use `OPENSECRET_DEV_CONTAINERS=0 nix develop -c <command>` for
reproducible one-shot commands unless the task needs the container helpers.

Treat `.env`, `.pgdata/`, and `.local/` as local, gitignored state. Never commit their contents.

By default, entering `nix develop`:

- checks any PostgreSQL listener on `localhost:$PGPORT` before inspecting
  `.pgdata`; if one responds, it reuses that listener and skips checkout-local
  cluster, role, and database initialization;
- otherwise initializes or starts PostgreSQL under `.pgdata` on port `5432`
  and creates the local role and database when needed;
- creates `.env` from `.env.sample` only when `.env` is absent, then generates
  local enclave and JWT secrets; and
- on Linux, rewrites user-level container configuration when container setup
  is enabled.

Keep `OPENSECRET_DEV_CONTAINERS=0` for routine development and one-shot
commands. Enable container setup only for a task that intentionally needs the
local container helpers.

Do not overwrite an existing `.env`; it belongs to the developer. Use these controls when the defaults conflict with another checkout or an externally managed service:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_CONTAINERS=0 nix develop
OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 nix develop
OPENSECRET_DEV_CONTAINERS=0 nix develop
PGDATA=/tmp/opensecret-feature/pgdata PGPORT=32417 \
  OPENSECRET_DEV_CONTAINERS=0 nix develop
OPENSECRET_DEV_DATABASE_URL=postgres://opensecret_user:password@localhost:32417/opensecret \
  OPENSECRET_DEV_CONTAINERS=0 nix develop
```

Keep each concurrently running checkout on distinct PostgreSQL state, database URL, and backend port. Do not point tests or migrations at a shared, preview, or production database.

## Prepare PostgreSQL correctly

Run Diesel schema migrations before starting the backend:

```sh
OPENSECRET_DEV_CONTAINERS=0 nix develop -c just diesel-migration-run-local
```

Do not mistake `src/migrations.rs` for the Diesel migration runner. The server invokes that module for application-data migration after connecting; it assumes the SQL schema and seeded `OpenSecret` organization and `Maple` project already exist.

Use the repository recipes for schema work:

```sh
OPENSECRET_DEV_CONTAINERS=0 \
  nix develop -c just diesel-migration-generate add_user_preferences
OPENSECRET_DEV_CONTAINERS=0 nix develop -c just diesel-migration-run-local
```

Replace `add_user_preferences` with a concise snake-case description of the schema change.

Add reversible `up.sql` and `down.sql` behavior. Let Diesel regenerate `src/models/schema.rs` according to `diesel.toml`; do not hand-maintain generated schema drift. Preserve existing migrations rather than editing migrations that may already have run elsewhere.

Before changing a persisted ciphertext, trace the write and read paths to its
owning key domain. User-private content uses a credential-derived user key that
is available only in an authenticated user context. Server-owned material such
as OAuth access tokens, project secrets, and password verifiers uses its
enclave/system-secret domain. Do not collapse these domains or assume a key is
available merely because startup can reach the row.

Make ciphertext formats explicitly versioned. For user-key rows, deploy
dual-read/new-write behavior, then lazily and transactionally rewrite the old
format only after authenticated access makes that user's key available. Use a
startup application-data migration only for rows whose owning key is actually
available there, such as enclave/system-key data, and make that migration
idempotent and retry-safe. Diesel SQL can evolve columns and constraints, but
cannot safely re-encrypt opaque bytes without the owning key.

Route migration verification, disposable-database setup, rollback checks, and database-backed tamper tests through `.agents/skills/validate-opensecret/SKILL.md`.

## Configure local runtime state

Use `APP_MODE=local`. Confirm these values before startup:

- `DATABASE_URL`: point to the migrated local PostgreSQL database.
- `ENCLAVE_SECRET_MOCK`: use exactly 32 random bytes encoded as hex.
- `JWT_SECRET`: use a local random secret.
- `TINFOIL_API_KEY`: provide a non-empty key; the in-process attested Tinfoil client is a runtime requirement.
- `OPENAI_API_BASE`: set the default OpenAI-compatible provider endpoint. The
  documented loopback Continuum recipe does not need `OPENAI_API_KEY`.
  Treat any other custom base as a credential boundary and review its exact
  URL, authentication headers, and forwarding behavior.
- `OPENSECRET_BIND_ADDR`: override `127.0.0.1:3000` when the default port is unavailable.

Prefer gitignored secret files for day-to-day provider credentials:

```text
.local/secrets/tinfoil_api_key
.local/secrets/continuum_api_key
```

Never print, log, paste into tracked files, or expose provider credentials in command output. Prefer a protected environment variable when a secret file is unsuitable.

Configure optional capabilities only when the task exercises them:

- `RESEND_API_KEY` for email delivery.
- GitHub or Google client ID and secret variables for the corresponding OAuth flow.
- `BRAVE_API_KEY` and `KAGI_API_KEY` for web-search providers.
- `BILLING_SERVER_URL` plus `BILLING_API_KEY` for the configurable billing API boundary.
- `OS_FLAGS_BASE_URL` plus `OS_FLAGS_API_KEY` for the configurable feature-flag API boundary.

Do not require billing or feature-flag APIs for ordinary registered-user local development. Test their unavailable, timeout, and denial behavior deliberately when changing those boundaries.

## Run the backend

For a direct local run, ensure PostgreSQL is running, migrations have completed, and the required provider configuration is present, then run:

```sh
OPENSECRET_DEV_CONTAINERS=0 nix develop -c cargo run
```

On macOS, use the documented local Continuum-compatible path when that provider is part of the task:

```sh
nix develop -c just build-local-proxies-macos   # one-time or after its submodule changes
nix develop -c just run-continuum-proxy-macos   # terminal 1
nix develop -c just diesel-migration-run-local  # after every schema change
nix develop -c just run-local-backend-macos     # terminal 2
```

Read `docs/local-macos-stack.md` for the exact local process shape and `docs/dev-shell.md` for shell-hook controls. Do not create a separate local Tinfoil sidecar; Tinfoil runs through the Rust SDK inside OpenSecret.

Use `/health-check` only as a process-liveness probe. It does not verify PostgreSQL. Use `/health-check-extended` to verify live outbound Tinfoil model discovery, and expect it to require valid credentials and network access.

Route authenticated, encrypted API and frontend smoke testing through `.agents/skills/validate-opensecret/SKILL.md`; plaintext `curl` requests do not exercise the real protected protocol.

## Place changes by responsibility

Keep this repository responsible for server-controlled security and policy:

- Put startup configuration, shared application state, and router composition in `src/main.rs`.
- Put HTTP route parsing, middleware, response envelopes, and streaming behavior in `src/web/`.
- Put domain records and Diesel mappings in `src/models/`.
- Put database operations behind `DBConnection` in `src/db.rs`.
- Put SQL schema evolution in `migrations/` and generated schema in `src/models/schema.rs`.
- Put JWT validation and claims in `src/jwt.rs`.
- Put encryption primitives and credential-bound seed handling in `src/encrypt.rs` and `src/seed_wrapping.rs`; trace each persisted field's owning key from its call sites before changing its format.
- Put model catalog and public IDs in `src/model_config.rs`.
- Put provider selection policy in `src/provider_routing.rs`, endpoint configuration in `src/proxy_config.rs`, and outbound transport in `src/provider_client.rs`.
- Put stateful Responses behavior in `src/web/responses/`; keep direct web-search contracts in `src/web/web_routes.rs` and provider adapters in `src/brave.rs` or `src/kagi.rs`.
- Put periodic in-process maintenance beside the state it maintains; do not infer a worker implementation merely from a database table.

Keep provider credentials, routing policy, authorization, durable storage, billing/flag decisions, and cryptographic operations in the backend. Keep presentation, local interaction, and device-specific behavior in the frontend. Treat public API shape, attestation, session encryption, and streamed event changes as coordinated client-and-backend work.

## Load the specialized workflow

Load the narrow sibling skill before editing a sensitive surface:

- Use `.agents/skills/change-opensecret-api/SKILL.md` for routes, request/response contracts, middleware ordering, Responses behavior, and coordinated client API changes.
- Use `.agents/skills/change-opensecret-provider/SKILL.md` for models, routing, provider endpoints, attested transport, retries, headers, and usage attribution.
- Use `.agents/skills/review-opensecret-security/SKILL.md` for authentication, encryption, key handling, privacy, logging, account deletion/reset, or threat-model review.
- Use `.agents/skills/validate-opensecret/SKILL.md` before claiming any implementation complete and whenever tests, migrations, local API smoke tests, or full frontend/backend smoke tests are required.

When a task crosses multiple surfaces, load each applicable skill and keep the change at the narrowest owning layer. Preserve existing public behavior unless the task explicitly changes the contract.
