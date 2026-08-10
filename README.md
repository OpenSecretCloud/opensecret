# OpenSecret

OpenSecret is the open-source Rust backend for confidential AI applications
such as [Maple](https://github.com/OpenSecretCloud/Maple). It owns
authentication, encrypted client sessions, encrypted persistence, model and
provider routing, usage accounting, and OpenAI-shaped/Responses APIs carried
inside the OpenSecret encrypted transport.

Production is designed to run inside AWS Nitro Enclaves. A source checkout or
local build can validate implementation and build invariants, but cannot by
itself prove the PCRs, KMS/IAM policy, logging configuration, artifact identity,
or network placement of a deployed environment.

## Local development

The Nix flake pins Rust, PostgreSQL, Diesel, native libraries, provider tooling,
and the repository's submodule dependencies.

```bash
git submodule update --init --recursive
OPENSECRET_DEV_CONTAINERS=0 nix develop
```

The default shell hook first checks for any PostgreSQL server responding on
`localhost:$PGPORT`. If one responds, the hook reuses that listener and skips
initializing this checkout's `.pgdata`, role, and database. Otherwise it starts
the checkout-local cluster. The hook also creates a gitignored `.env` from
`.env.sample` when one is absent. On Linux, enabling the container setup
rewrites user-level container configuration; keep
`OPENSECRET_DEV_CONTAINERS=0` unless the task intentionally uses those local
container helpers. For all shell-hook controls, see
[`docs/dev-shell.md`](docs/dev-shell.md).

SQL migrations are a required, separate step:

```bash
just diesel-migration-run-local
```

Backend startup does not run Diesel schema migrations. `src/migrations.rs`
contains application-data migration logic and is not a replacement for the
command above.

### Provider credentials

Tinfoil is a hard runtime dependency. On macOS, the supported local stack also
uses a native Continuum proxy. Put credentials in gitignored files so they do
not enter shell history:

```text
.local/secrets/tinfoil_api_key
.local/secrets/continuum_api_key
```

Environment variables `TINFOIL_API_KEY` and `CONTINUUM_API_KEY` override those
files. Provider credentials are secrets: never commit them, print them in
diagnostics, or copy them into Maple.

Build the local Continuum binary once:

```bash
nix develop -c just build-local-proxies-macos
```

Then use separate terminals for the provider proxy and backend:

```bash
nix develop -c just run-continuum-proxy-macos
nix develop -c just run-local-backend-macos
```

The defaults are:

```text
Continuum proxy   http://127.0.0.1:8092
OpenSecret API    http://127.0.0.1:3000
```

The attested Tinfoil client runs inside the OpenSecret process; there is no
local Tinfoil sidecar or port. See
[`docs/local-macos-stack.md`](docs/local-macos-stack.md) for the complete local
topology and provider configuration.

Point Maple's ignored `frontend/.env.local` at this API:

```dotenv
VITE_OPEN_SECRET_API_URL=http://127.0.0.1:3000
```

Keep the default Vite origin when testing OAuth or verification callbacks
unless the callback contract is the feature being changed.

## Configuration

`.env.sample` documents local configuration. The important boundaries are:

- `DATABASE_URL`, `APP_MODE`, `ENCLAVE_SECRET_MOCK`, and `JWT_SECRET` configure
  local persistence and cryptographic development state.
- `OPENSECRET_BIND_ADDR` changes the API listener from its default
  `127.0.0.1:3000`.
- `OPENAI_API_BASE` selects the default OpenAI-compatible provider. The
  documented loopback Continuum recipe does not need `OPENAI_API_KEY`.
  Arbitrary custom bases are credential boundaries: review the exact URL,
  authentication headers, and forwarding behavior rather than inferring them
  from the base URL alone.
- `TINFOIL_API_KEY` is always required.
- `BRAVE_API_KEY` and `KAGI_API_KEY` enable optional web providers.
- `BILLING_SERVER_URL`/`BILLING_API_KEY` and
  `OS_FLAGS_BASE_URL`/`OS_FLAGS_API_KEY` configure optional external APIs.

Billing and flags are independent server-side clients. Their administrative
keys never belong in Maple or another public client. An unconfigured optional
integration can have feature- and account-specific behavior; validate the
exact path rather than assuming all endpoints fail open or fail closed.

## Development and architecture

This repository is one Rust package and binary, not a Cargo workspace. The main
areas are:

- `src/main.rs`: configuration, state, middleware, and router assembly.
- `src/web/`: encrypted HTTP routes, OpenAI-shaped APIs, auth/OAuth, health,
  web search, and Responses.
- `src/models/`, `src/db.rs`, and `migrations/`: PostgreSQL persistence and
  Diesel schema.
- `src/encrypt.rs`, `src/seed_wrapping.rs`, `src/jwt.rs`, and attestation/session
  code: cryptographic and identity boundaries.
- `src/model_config.rs`, `src/provider_routing.rs`, `src/proxy_config.rs`, and
  `src/provider_client.rs`: model catalog, routing, provider configuration,
  transport, streaming, retries, and usage.
- `src/brave.rs` and `src/kagi.rs`: provider-specific web adapters.

Protected routes require OpenSecret attestation/key exchange and encrypted
sessions. “OpenAI-compatible” describes the decrypted payload shape; raw
plaintext `curl` is not an end-to-end API test. Use an OpenSecret SDK or Maple
for protected-route smoke tests.

Persisted encryption has separate key domains. User-private content follows
the credential-derived user-key path, while server-owned material such as
OAuth access tokens, project secrets, and password verifiers uses its owning
enclave/system secret. A ciphertext-format migration must preserve that
boundary: use a versioned dual-read/new-write transition and lazy authenticated
rewrite for user-key rows; reserve startup rewrites for data whose owning key
is available at startup.

Contributor and coding-agent standards live in [`AGENTS.md`](AGENTS.md).
Task-specific local development, API, provider, security, and validation
workflows live under [`.agents/skills/`](.agents/skills/).

## Validation

Match the checked-in Rust CI gate while preventing check-only shells from
starting PostgreSQL, generating `.env`, or changing Linux container settings:

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

Default CI does not start PostgreSQL or run ignored tests. Database migration,
AEAD tamper, OAuth database, live provider, encrypted API, and Maple smoke are
separate evidence tiers. In particular:

- `/health-check` is process liveness and does not query PostgreSQL.
- `/health-check-extended` checks Tinfoil model-list connectivity only.
- Ignored database evidence is valid only when the disposable database
  identity and migrations are verified, the selected tests are listed, no
  skip marker appears under `--nocapture`, and the expected pass count is
  asserted.
- The ignored live Tinfoil test bypasses OpenSecret's public API and therefore
  does not prove auth, encrypted transport, persistence, routing, or billing.

Do not run every ignored test together. Use
`.agents/skills/validate-opensecret/` for safe disposable-database commands,
provider probes, SDK-driven encrypted smoke, and evidence reporting. Add
`nix flake check --print-build-logs` for changes to Nix, entrypoints, Nitro
helpers, kernel sources, or security-build invariants.

## Nitro builds and deployment

The named Nix outputs are the supported EIF build path:

```bash
nix build '.?submodules=1#eif-dev'
nix build '.?submodules=1#eif-preview'
nix build '.?submodules=1#eif-prod'
```

EIF construction, PCR comparison, deployment, and live trust verification are
different evidence. EIF builds require the appropriate Linux/ARM environment;
current-host checks on macOS do not prove them.

Copying an EIF, mutating PCR references or KMS policy, terminating/running an
enclave, restarting a proxy, or using any `stage-*`/`deploy-*` recipe requires
explicit operator authorization. Treat a PCR recipe as verification only when
it compares the built result with a reviewed, checked-in reference through the
matching workflow.

See [`docs/nitro-deploy.md`](docs/nitro-deploy.md) for infrastructure context.
Treat its deployment commands as operator procedures, not contributor setup.

## Contributing

Keep changes focused and add regression coverage at the owning boundary. For
API changes, review encrypted transport, auth context, errors/SSE, persistence,
published SDKs, and Maple compatibility. For provider changes, keep upstream
IDs and secrets behind the provider boundary and prove retry safety and usage
handling.

Before opening a pull request, run the applicable full gates, test migrations
against a disposable database when relevant, and inspect `git diff --check`.
Security reports must separate source-confirmed facts, tests, deployed
configuration, and live observations.
