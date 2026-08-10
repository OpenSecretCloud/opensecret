# Development Shell

The default `nix develop` behavior checks `localhost:$PGPORT` first. It reuses a
responding listener; otherwise it starts a local Postgres instance under
`.pgdata`. It also creates a local `.env` if one does not exist and keeps the
historical port `5432` default.

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

These controls are intended for isolated development environments where each
checkout owns its runtime state and ports.

## Linux Container Controls

On Linux, the default shell hook also configures user-level Podman/container
files. Keep that state unchanged when the task does not need the container
helpers:

```sh
OPENSECRET_DEV_CONTAINERS=0 nix develop
```

For a pure check, disable every stateful hook together:

```sh
OPENSECRET_DEV_POSTGRES=0 \
OPENSECRET_DEV_ENV=0 \
OPENSECRET_DEV_CONTAINERS=0 \
nix develop --no-write-lock-file -c <command>
```

## Backend Bind Address

The local backend listens on `127.0.0.1:3000` by default. Override it when
running multiple local backends at once:

```sh
OPENSECRET_BIND_ADDR=127.0.0.1:31417 cargo run
```
