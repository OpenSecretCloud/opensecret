# Local macOS Stack

This runbook is for developing OpenSecret locally on macOS with the native
Continuum proxy and the in-process Tinfoil Rust SDK. It is intentionally
separate from the Linux/Nitro enclave deployment path.

## Shape

Run these processes from the OpenSecret checkout:

```text
Continuum proxy   http://127.0.0.1:8092
OpenSecret API    http://127.0.0.1:3000 (includes the Tinfoil SDK)
Maple frontend    VITE_OPEN_SECRET_API_URL=http://127.0.0.1:3000
```

In local mode, OpenSecret reads `.env` through `dotenv`.

When `OPENAI_API_BASE` is not `api.openai.com`, OpenSecret treats it as the default Continuum-compatible proxy and does not require `OPENAI_API_KEY` for that route. The local Continuum proxy recipe enables Privatemode shared prompt caching at the proxy layer, so callers do not need to send a per-request `cache_salt`.

OpenSecret initializes one shared Tinfoil SDK client from `TINFOIL_API_KEY`.
The SDK verifies enclave attestation, pins TLS to the verified enclave, and
reuses its connection pool for models, chat, audio, and embeddings requests.

## One-Time Setup

Initialize submodules:

```sh
git submodule update --init --recursive
```

Build the macOS-native Continuum proxy binary:

```sh
nix develop -c just build-local-proxies-macos
```

This writes the generated binary under `.local/bin/`, which is gitignored:

```text
.local/bin/continuum-proxy-darwin
```

The checked-in Linux Continuum proxy binary is not replaced.

## Secrets

For local development, store provider keys in gitignored files:

```text
.local/secrets/continuum_api_key
.local/secrets/tinfoil_api_key
```

The run recipes also accept environment variables. The Tinfoil key is read by
OpenSecret itself, not by a local sidecar:

```sh
CONTINUUM_API_KEY=...
TINFOIL_API_KEY=...
```

Prefer the gitignored files for day-to-day use so keys do not land in shell history.

## Environment

The usual path is to enter the dev shell once and let its shell hook create
`.env`:

```sh
nix develop
```

When `.env` does not already exist, `nix develop` starts from `.env.sample`,
fills in the local Postgres URL, and generates local-only secret values.

The `run-local-backend-macos` recipe injects the local proxy URLs when it starts
the backend. If creating `.env` by hand and running `cargo run` directly,
generate local-only secrets with OpenSSL and set the proxy settings yourself:

```dotenv
APP_MODE=local
OPENAI_API_BASE=http://127.0.0.1:8092
TINFOIL_API_KEY=<your key>
ENCLAVE_SECRET_MOCK=<openssl rand -hex 32>
JWT_SECRET=<openssl rand -hex 32>
```

## Running

Use separate terminals:

```sh
nix develop -c just run-continuum-proxy-macos
nix develop -c just diesel-migration-run-local
nix develop -c just run-local-backend-macos
```

Then point Maple at the local backend from the Maple checkout:

```dotenv
VITE_OPEN_SECRET_API_URL=http://127.0.0.1:3000
```

and run:

```sh
nix develop -c just dev
```

## Notes

- Production Linux/Nitro and local macOS Continuum proxy launches enable Privatemode shared prompt caching without a fixed prompt cache salt. The cache is shared by requests handled by the same proxy instance and is reset when that proxy restarts.
- The macOS Continuum proxy binary is a generated local artifact only.
- There is no local Tinfoil proxy process or port. Tinfoil traffic originates
  from the OpenSecret process through the SDK's attested, origin-bound client.
- Device builds, TestFlight, notarization, and production app signing still need Apple developer credentials configured separately.
