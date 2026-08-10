# OpenSecret

OpenSecret is the open-source Rust backend for confidential AI applications
such as [Maple](https://github.com/OpenSecretCloud/Maple). It owns
authentication, encrypted client sessions and persistence, provider routing,
usage accounting, and OpenAI-shaped/Responses APIs carried inside the
OpenSecret encrypted transport.

Production is designed for AWS Nitro Enclaves. A source checkout or local build
can validate implementation and build invariants, but does not by itself prove
the artifact, PCR, IAM/KMS policy, logging, or network configuration of a
deployed environment.

## Local quick start

The Nix flake pins the Rust toolchain, PostgreSQL, Diesel, native libraries, and
provider dependencies.

```sh
git submodule update --init --recursive
OPENSECRET_DEV_CONTAINERS=0 nix develop
just diesel-migration-run-local
```

The development shell may reuse a PostgreSQL listener, start `.pgdata`, and
create `.env` from `.env.sample`. On Linux, container setup also changes
user-level container configuration unless disabled. See
[`docs/dev-shell.md`](docs/dev-shell.md) before running check-only commands or
multiple checkouts.

Backend startup does not run Diesel schema migrations;
`src/migrations.rs` is separate application-data migration logic.

### Provider credentials and macOS stack

Tinfoil runs in-process and requires a credential. The supported macOS local
stack can also use a native Continuum proxy. Store credentials in these
gitignored files:

```text
.local/secrets/tinfoil_api_key
.local/secrets/continuum_api_key
```

Build the local proxy once, then run the long-lived processes in separate
terminals:

```sh
nix develop -c just build-local-proxies-macos
```

Terminal 1:

```sh
nix develop -c just run-continuum-proxy-macos
```

Terminal 2:

```sh
nix develop -c just run-local-backend-macos
```

See [`docs/local-macos-stack.md`](docs/local-macos-stack.md) for topology and
Maple wiring. There is no local Tinfoil sidecar.

## Configuration and API boundary

`.env.sample` is the configuration inventory. Keep all real credentials in
ignored files or protected environment variables. Billing and feature flags
are optional external HTTP APIs whose administrative credentials remain in the
backend; their server implementations are not part of this repository setup.

Protected routes require OpenSecret attestation/key exchange and encrypted
sessions. “OpenAI-shaped” describes decrypted payloads, not a plaintext
OpenAI-compatible wire endpoint. Use an OpenSecret SDK or Maple for
protected-route integration tests; plain `curl` is suitable only for public
health probes.

Contributor and coding-agent standards live in [`AGENTS.md`](AGENTS.md).
Task-specific development, API, provider, security, and validation workflows
live under [`.agents/skills/`](.agents/skills/).

## Validation

The Rust CI gate is formatting, strict all-target/all-feature Clippy, and locked
all-feature tests. Run it through the pinned shell without starting local
services:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c env RUSTFLAGS='-D warnings' \
  cargo clippy --locked --all-targets --all-features -- -D warnings
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c env RUSTFLAGS='-D warnings' \
  cargo test --locked --all-features
```

Default CI does not run ignored database or live-provider tests. Use the
[`validate-opensecret`](.agents/skills/validate-opensecret/SKILL.md) workflow
for disposable PostgreSQL tests, authorized provider checks, encrypted-client
smoke tests, and Nix/EIF/PCR evidence. Report those layers separately.

## Nitro builds and deployment

The supported EIF outputs are `eif-dev`, `eif-preview`, and `eif-prod`, built
with the repository's Nix flake on the appropriate Linux/ARM environment. Build,
PCR comparison, deployment, and live trust verification are distinct evidence.

See [`docs/nitro-deploy.md`](docs/nitro-deploy.md) for operator procedures.
Changing PCR references or KMS policy, copying artifacts, starting or stopping
enclaves, restarting remote services, staging, and deployment require explicit
authorization for the named environment.

## Contributing

Keep changes focused, add regression coverage at the owning boundary, and
follow the relevant repository skill. Before opening a pull request, run every
validation tier reached by the diff and state what remains unverified.
