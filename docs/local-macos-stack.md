# Local macOS Stack

This runbook is for developing OpenSecret locally on macOS with native macOS proxy binaries. It is intentionally separate from the Linux/Nitro enclave deployment path.

## Shape

Run these processes from the OpenSecret checkout:

```text
Continuum proxy   http://127.0.0.1:8092
Tinfoil proxy     http://127.0.0.1:8093
OpenSecret API    http://127.0.0.1:3000
Maple frontend    VITE_OPEN_SECRET_API_URL=http://127.0.0.1:3000
```

In local mode, OpenSecret reads `.env` through `dotenv`.

When `OPENAI_API_BASE` is not `api.openai.com`, OpenSecret treats it as the default Continuum-compatible proxy and does not require `OPENAI_API_KEY` for that route.

When `TINFOIL_API_BASE` is set, OpenSecret uses it for the Tinfoil route.

## One-Time Setup

Initialize submodules:

```sh
git submodule update --init --recursive
```

Build macOS-native proxy binaries:

```sh
nix develop -c just build-local-proxies-macos
```

This writes generated binaries under `.local/bin/`, which is gitignored:

```text
.local/bin/continuum-proxy-darwin
.local/bin/tinfoil-proxy-darwin
```

The checked-in Linux proxy binaries are not replaced.

## Secrets

For local development, store proxy keys in gitignored files:

```text
.local/secrets/continuum_api_key
.local/secrets/tinfoil_api_key
```

The run recipes also accept environment variables:

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
TINFOIL_API_BASE=http://127.0.0.1:8093
ENCLAVE_SECRET_MOCK=<openssl rand -hex 32>
JWT_SECRET=<openssl rand -hex 32>
```

## Running

Use separate terminals:

```sh
nix develop -c just run-continuum-proxy-macos
nix develop -c just run-tinfoil-proxy-macos
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

- Production Linux/Nitro entrypoint behavior is unchanged.
- The macOS proxy binaries are generated local artifacts only.
- Device builds, TestFlight, notarization, and production app signing still need Apple developer credentials configured separately.
