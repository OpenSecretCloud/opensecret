# Development Shell

The default `nix develop` behavior starts a local Postgres instance under
`.pgdata`, creates a local `.env` if one does not exist, and keeps the historical
port `5432` default.

For multi-workspace local development, the shell hook can be configured without
changing the default path for existing developers.

## Postgres Controls

Skip shell-hook Postgres management:

```sh
OPENSECRET_DEV_POSTGRES=0 nix develop
```

Override the local Postgres state directory and port:

```sh
PGDATA=/tmp/opensecret-feature-a/pgdata \
PGPORT=32417 \
nix develop
```

The shell hook also respects `PGSOCKETS` when set. If not set, it defaults to
`$PGDATA/sockets`.

## Environment Controls

Skip shell-hook `.env` generation:

```sh
OPENSECRET_DEV_ENV=0 nix develop
```

When `.env` does not exist and generation is enabled, the generated
`DATABASE_URL` is based on `OPENSECRET_DEV_DATABASE_URL` if set, otherwise on
the configured `PGPORT`:

```sh
OPENSECRET_DEV_DATABASE_URL=postgres://opensecret_user:password@localhost:32417/opensecret \
nix develop
```

These controls are intended for tools such as `opensecret-workspaces`, where
each feature workspace owns its own runtime state and ports.

## Backend Bind Address

The local backend listens on `127.0.0.1:3000` by default. Override it when
running multiple local backends at once:

```sh
OPENSECRET_BIND_ADDR=127.0.0.1:31417 cargo run
```
