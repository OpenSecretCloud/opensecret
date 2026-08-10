---
name: validate-opensecret
description: Validate OpenSecret backend changes with side-effect-free Rust checks, exact CI-parity commands, targeted security and API tests, disposable PostgreSQL migration and tamper tests, explicit live-provider contracts, encrypted frontend/backend smoke tests, and platform-specific Nix or EIF checks. Use before claiming an OpenSecret implementation complete, when reviewing test evidence, or when changes touch Rust code, authentication, encryption, Responses or OpenAI-compatible APIs, streaming, providers, migrations, Nix, entrypoints, or enclave PCRs.
---

# Validate OpenSecret

## Build an evidence plan from the diff

1. Read `AGENTS.md`, inspect `git status --short --branch`, and identify every changed trust boundary.
2. Select the union of all applicable tiers below. A higher tier supplements; it does not replace the lower tiers.
3. Run the narrowest relevant test while iterating, then run the required completion tiers.
4. Keep secrets out of commands, logs, fixtures, and tracked files. Use `.local/secrets/` or protected environment variables.
5. Never run all ignored tests with a blanket `cargo test -- --ignored`. The ignored set mixes destructive disposable-database tests with a credentialed live-network test. Invoke only an explicit module or exact test name.

| Change | Required validation |
| --- | --- |
| Documentation only | Verify every referenced path, command, environment variable, and behavior against the checkout; inspect the rendered diff. |
| Any Rust behavior or dependency | Side-effect-free targeted tests while iterating, then Tier 1 CI parity. |
| Auth, session, encryption, seed wrapping, persisted ciphertext, deletion, or reset | Tier 1 plus the relevant security/property tests; add Tier 2 database tests when persistence is involved. |
| Diesel models, schema, SQL, or application-data migration | Tiers 1 and 2; add a boot or full-stack smoke when startup/data migration behavior changes. |
| HTTP contract, middleware, SSE, OpenAI-compatible route, or Responses API | Tier 1 plus Tier 4 encrypted/authenticated smoke; include Maple when the public client contract changes. |
| Provider transport, routing, model catalog, headers, usage, or attestation | Tier 1 plus targeted provider tests; add the explicitly named Tier 3 live test and Tier 4 app smoke when credentials and egress are authorized. |
| `flake.nix`, `flake.lock`, `entrypoint.sh`, kernel, Nitro, EIF, or PCR | Tier 1 when Rust is affected plus Tier 5 on every available platform; require Linux/ARM CI evidence for Linux/Nitro claims. |

## Keep ordinary checks side-effect-free

The default `nix develop` shell hook may initialize or start `.pgdata`, create `.env` with local secrets, and on Linux configure local container state. Disable all three behaviors for formatting, linting, unit tests, and flake evaluation:

```sh
OPENSECRET_DEV_POSTGRES=0 \
OPENSECRET_DEV_ENV=0 \
OPENSECRET_DEV_CONTAINERS=0 \
nix develop --no-write-lock-file -c <command>
```

Use the stateful shell only when the selected tier genuinely needs local services. Give concurrent checkouts distinct `PGDATA`, `PGPORT`, `DATABASE_URL`, backend ports, and test databases. Never point local validation at shared, preview, or production state.

Do not treat `cargo audit`, `cargo deny`, or `cargo machete` as repository CI gates merely because the Nix shell provides them. Run one only when the task calls for it, state its configuration and network conditions, and report it separately.

## Tier 0: iterate with focused pure checks

Run focused unit tests after locating the owning module. Preserve `--locked --all-features` so local resolution and feature coverage do not silently diverge:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c \
  cargo test --locked --all-features <module_or_test_filter>
```

Useful filters include:

```sh
cargo test --locked --all-features security_invariants
cargo test --locked --all-features crypto_property_tests
cargo test --locked --all-features provider_client::tests
cargo test --locked --all-features web::openai::tests
cargo test --locked --all-features web::responses
```

Choose filters from the changed code rather than using this list mechanically. Focused success is iteration evidence, not completion evidence.

## Tier 1: reproduce Rust CI

Initialize public submodules recursively, then run the same inner commands and warning policy as `.github/workflows/rust.yml` through the pinned Nix environment:

```sh
git submodule update --init --recursive

OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check

OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c \
  env RUSTFLAGS="-D warnings" \
  cargo clippy --locked --all-targets --all-features -- -D warnings

OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c \
  env RUSTFLAGS="-D warnings" cargo test --locked --all-features
```

Read the test summary. The default suite intentionally does not execute ignored tests. Report its passed, failed, and ignored counts; do not summarize it as “all tests” when ignored tests remain.

Do not substitute an unverified aggregate recipe for the exact commands above.
Before relying on one, inspect its checked-in definition and prove that it
preserves the same targets, features, lockfile, and warning policy.

## Tier 2: validate migrations and database-backed security

The ignored tests mutate and delete rows. Run them only in a fresh PostgreSQL
cluster whose data directory, listener, role, and database are created for this
command. Do not use the default stateful shell hook: it may reuse any server
already answering on `localhost:$PGPORT` before it inspects `.pgdata`.

From the repository root, the following procedure disables all shell-hook
state, starts a temporary cluster on an OS-selected loopback port, proves the
connected data directory, port, database, user, owner, empty schema, and Diesel
migration count, discovers the ignored subsets, executes them serially with
visible output, and removes only the guarded temporary cluster on exit:

```sh
OPENSECRET_DEV_POSTGRES=0 \
OPENSECRET_DEV_ENV=0 \
OPENSECRET_DEV_CONTAINERS=0 \
nix develop --no-write-lock-file -c bash <<'BASH'
set -euo pipefail

OPENSECRET_TEST_TMPROOT="${TMPDIR:-/tmp}"
OPENSECRET_TEST_TMPROOT="${OPENSECRET_TEST_TMPROOT%/}"
PGDATA="$(mktemp -d "$OPENSECRET_TEST_TMPROOT/opensecret-db-tests.XXXXXX")"
readonly OPENSECRET_TEST_PGDATA="$PGDATA"
PGSOCKETS="$PGDATA/sockets"
PGPORT="$(python3 - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
OPENSECRET_TEST_DB="opensecret_agent_test_$(openssl rand -hex 6)"
OPENSECRET_TESTS_PASSED=0
export PGDATA PGSOCKETS PGPORT OPENSECRET_TEST_DB

cleanup() {
  status=$?
  trap - EXIT
  if [[ "${PGDATA:-}" != "$OPENSECRET_TEST_PGDATA" ||
        "$OPENSECRET_TEST_PGDATA" !=
          "$OPENSECRET_TEST_TMPROOT"/opensecret-db-tests.* ]]; then
    printf 'refusing to clean unexpected PGDATA: %s\n' \
      "${PGDATA:-<unset>}" >&2
    exit 1
  fi
  if [ -f "$PGDATA/PG_VERSION" ] && \
     pg_ctl status -D "$PGDATA" >/dev/null 2>&1 && \
     ! pg_ctl stop -D "$PGDATA" -m fast >/dev/null; then
    printf 'PostgreSQL did not stop; preserving temporary state at %s\n' \
      "$PGDATA" >&2
    exit 1
  fi
  rm -rf -- "$PGDATA"
  if [ "$status" -eq 0 ] && [ "$OPENSECRET_TESTS_PASSED" -eq 1 ]; then
    printf '%s\n' \
      'Disposable-DB evidence: 14 AEAD/database tests and 2 OAuth database tests passed; no tests skipped; temporary cluster removed.'
  fi
  exit "$status"
}
trap cleanup EXIT

initdb -D "$PGDATA" --username="${USER:?}" \
  --auth-local=trust --auth-host=scram-sha-256 >/dev/null
mkdir -p "$PGSOCKETS"
pg_ctl start -D "$PGDATA" \
  -o "-h 127.0.0.1 -p $PGPORT -k $PGSOCKETS" \
  -l "$PGDATA/postgres.log" -w >/dev/null

expected_pgdata="$(cd "$PGDATA" && pwd -P)"
reported_pgdata="$(psql -h "$PGSOCKETS" -p "$PGPORT" -U "$USER" \
  -d postgres -Atqc 'SHOW data_directory')"
reported_pgdata="$(cd "$reported_pgdata" && pwd -P)"
test "$reported_pgdata" = "$expected_pgdata"
test "$(psql -h "$PGSOCKETS" -p "$PGPORT" -U "$USER" \
  -d postgres -Atqc 'SHOW port')" = "$PGPORT"

psql -h "$PGSOCKETS" -p "$PGPORT" -U "$USER" -d postgres \
  -v ON_ERROR_STOP=1 \
  -c "CREATE USER opensecret_user WITH PASSWORD 'password'" >/dev/null
createdb -h "$PGSOCKETS" -p "$PGPORT" -U "$USER" \
  --maintenance-db=postgres --owner=opensecret_user "$OPENSECRET_TEST_DB"

AEAD_TAMPER_TEST_DATABASE_URL="postgres://opensecret_user:password@127.0.0.1:${PGPORT}/${OPENSECRET_TEST_DB}"
export AEAD_TAMPER_TEST_DATABASE_URL

db_identity="$(psql "$AEAD_TAMPER_TEST_DATABASE_URL" -Atqc \
  "SELECT current_database() || '|' || current_user || '|' ||
          pg_get_userbyid(datdba)
     FROM pg_database
    WHERE datname = current_database()")"
test "$db_identity" = \
  "$OPENSECRET_TEST_DB|opensecret_user|opensecret_user"
test "$(psql "$AEAD_TAMPER_TEST_DATABASE_URL" -Atqc \
  "SELECT count(*) FROM information_schema.tables
    WHERE table_schema = 'public'")" -eq 0

diesel migration run --database-url "$AEAD_TAMPER_TEST_DATABASE_URL"
expected_migrations="$(find migrations -type f -name up.sql | wc -l | tr -d ' ')"
applied_migrations="$(psql "$AEAD_TAMPER_TEST_DATABASE_URL" -Atqc \
  'SELECT count(*) FROM __diesel_schema_migrations')"
test "$expected_migrations" -gt 0
test "$applied_migrations" -eq "$expected_migrations"

cargo test --locked --all-features aead_db_tamper_tests \
  -- --ignored --list | tee "$PGDATA/aead-tests.list"
test "$(grep -Ec '^aead_db_tamper_tests::.*: test$' \
  "$PGDATA/aead-tests.list")" -eq 14

cargo test --locked --all-features web::oauth_routes::tests::db_ \
  -- --ignored --list | tee "$PGDATA/oauth-tests.list"
test "$(grep -Ec '^web::oauth_routes::tests::db_.*: test$' \
  "$PGDATA/oauth-tests.list")" -eq 2

cargo test --locked --all-features aead_db_tamper_tests \
  -- --ignored --test-threads=1 --nocapture 2>&1 | \
  tee "$PGDATA/aead-tests.log"
if grep -q 'skipping: AEAD_TAMPER_TEST_DATABASE_URL' \
  "$PGDATA/aead-tests.log"; then
  exit 1
fi
grep -Eq 'test result: ok\. 14 passed; 0 failed; 0 ignored;' \
  "$PGDATA/aead-tests.log"

cargo test --locked --all-features web::oauth_routes::tests::db_ \
  -- --ignored --test-threads=1 --nocapture 2>&1 | \
  tee "$PGDATA/oauth-tests.log"
if grep -q 'skipping: AEAD_TAMPER_TEST_DATABASE_URL' \
  "$PGDATA/oauth-tests.log"; then
  exit 1
fi
grep -Eq 'test result: ok\. 2 passed; 0 failed; 0 ignored;' \
  "$PGDATA/oauth-tests.log"
OPENSECRET_TESTS_PASSED=1
BASH
```

The exact discovery assertions above are 14 tests under
`aead_db_tamper_tests` and 2 under `web::oauth_routes::tests::db_` for this
checkout. If either count changes, stop and reconcile the filter and expected
count with the test source before accepting evidence. The first subset covers
database-backed seed-wrap, password/reset, tamper, deletion, and persisted
response invariants. The OAuth subset proves only its two synthetic local
database cases: subject/e-mail linking behavior and initial seed-wrap/login
behavior. It does not exercise an OAuth provider, redirect/callback transport,
state or nonce validation, token exchange, or a full encrypted client flow.

For every new migration:

1. Run the entire migration chain against an empty disposable database.
2. On that disposable database, run `diesel migration redo --database-url "$AEAD_TAMPER_TEST_DATABASE_URL"` for the new latest reversible migration, then run the full chain again.
3. Confirm `src/models/schema.rs` contains only the expected Diesel-generated change.
4. Test an upgrade-shaped database containing representative pre-change rows. A clean-database migration cannot prove data conversion, constraints, or startup application-data migration.
5. For a persisted ciphertext format, write an explicit version discriminator and deploy dual-read/new-write behavior before retiring the old reader. For user-key rows, perform the authenticated rewrite lazily and transactionally only after that user's credential-derived key is available. Do not attempt to re-encrypt those rows in SQL or at server startup.
6. Use a startup application-data migration only when the owning key is available there, such as enclave/system-key data. Make it idempotent and retry-safe, then boot the backend against an upgrade-shaped database. `src/migrations.rs` is application-data migration logic, not a replacement for `diesel migration run`.
7. Preserve old migration files. Add a new migration instead of rewriting history that another installation may already have applied.

## Tier 3: isolate live provider proof

Run live provider checks only with explicit authorization for credentials, network egress, cost, and the named provider. Keep them separate from the default suite.

For the named Tinfoil live contract, set `TINFOIL_API_KEY` or provide
`.local/secrets/tinfoil_api_key`, then run exactly:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test \
  provider_client::tests::live_tinfoil_models_and_completions_match_the_legacy_api_contract \
  -- --ignored --exact
```

This proves live Tinfoil attestation and provider-boundary models, non-streaming completion, streaming SSE framing, terminal `[DONE]`, and numeric usage for that credential and moment. It does not prove OpenSecret authentication, request encryption, database persistence, Responses behavior, Maple integration, another provider, or a deployed enclave.

For providers without an explicit live test, run their local mock/boundary tests and an authorized Tier 4 smoke through the application. If neither exists, report that live provider behavior remains unverified; do not generalize Tinfoil evidence.

## Tier 4: smoke the real encrypted application boundary

Health endpoints are preliminary probes only:

```sh
curl --fail --silent --show-error http://127.0.0.1:3000/health-check
curl --fail --silent --show-error http://127.0.0.1:3000/health-check-extended
```

- `/health-check` fabricates liveness success and does not query PostgreSQL.
- `/health-check-extended` performs a five-second outbound Tinfoil model-list check. It does not prove PostgreSQL, authentication, session encryption, persistence, or every provider.

Protected routes require attestation/key exchange, an `x-session-id`, encrypted request envelopes, and route-appropriate JWT or API-key authorization. Do not use plaintext `curl` as API evidence. OpenAI-shaped routes are compatible payloads inside the OpenSecret encrypted protocol, not ordinary plaintext OpenAI SDK endpoints.

Use the open-source Maple client or its OpenSecret client implementation for full-stack smoke:

1. Start local PostgreSQL, run Diesel migrations, configure authorized provider credentials, and start OpenSecret.
2. In the Maple checkout, follow its `AGENTS.md` and validation skill. Set `VITE_OPEN_SECRET_API_URL=http://127.0.0.1:3000`, then start Maple with its repository-native command.
3. Exercise attestation/key exchange, registration, login, session refresh or re-entry, and at least one authenticated protected read and write using a dedicated local test account.
4. Verify missing, invalid, or stale authentication and unknown sessions fail closed without creating data or leaking decrypted content.
5. Exercise the exact feature changed and verify both UI-visible behavior and backend persistence after reload or a fresh session.
6. For OpenAI-shaped chat completions, exercise non-streaming and streaming forms when the changed contract affects both. Verify decryptable ordered chunks, expected model/content, usage when promised, clean cancellation or completion, and exactly one terminal `[DONE]`.
7. For Responses changes, verify ordered lifecycle events, reasoning/text/tool event reconstruction, terminal status and usage, persistence and retrieval, continuation, cancellation, and deletion as applicable. Unit tests in `src/web/responses/` do not by themselves prove the complete encrypted HTTP/SSE path.
8. For model catalog, embeddings, speech, transcription, web search, conversations, projects, or instructions, exercise the changed route through the real client and verify its failure path as well as success.
9. Inspect logs for secrets, tokens, plaintext prompts, decrypted responses, or sensitive provider payloads before calling the smoke successful.

Derive the permitted JWT or API-key context from current router assembly and test each context changed. Do not infer that API-key requests can use user-owned persistence keys. Describe only the OpenAI or Responses fields, events, and stream modes implemented by the current route; passing the project-specific subset is not proof of complete upstream API compatibility.

If optional billing or feature-flag behavior is in scope, configure the documented API base URL and key for an authorized development or production API server. Treat it as an external boundary: test unavailable, timeout, denial, and success behavior explicitly, and do not claim it from an unconfigured local run.

When Maple cannot exercise a backend-only contract, use an existing open-source
encrypted client or add a focused integration harness. If no harness can
exercise it, mark the E2E layer unverified rather than downgrading to plaintext
requests.

## Tier 5: validate Nix, entrypoint, EIF, and PCR changes

For flake, entrypoint, or packaging changes, run on the current platform:

```sh
nix flake show --all-systems --no-write-lock-file
nix flake check --no-write-lock-file --print-build-logs
nix build --no-link --no-write-lock-file .#default
```

`nix flake check` covers the current platform's exported checks, including the entrypoint entropy preflight and kernel-source pin. On Linux it additionally exposes kernel-security invariants and the Nitro helper. GitHub's EIF workflow builds images but does not substitute for explicitly running every flake check.

EIF and PCR evidence requires the supported Linux/ARM build runner. `.github/workflows/build.yml` builds the development EIF and compares `result/pcr.json` with `pcrDev.json`; production runs only on its restricted events. A macOS package build or flake evaluation cannot prove EIF construction, Nitro boot, Linux kernel configuration, attestation, PCR equality, or deployment health.

On that supported runner, reproduce the development image/PCR gate with:

```sh
nix build '.?submodules=1#eif-dev'
cmp -s result/pcr.json pcrDev.json
```

Use `#eif-prod` and `pcrProd.json` only when production-image validation is in scope. Record the built derivation and comparison result; a successful build without the comparison is not PCR proof.

Do not run submodule-update, PCR copy/update/sign, SCP, remote enclave start/stop, or deployment recipes without explicit authorization for that exact mutation and environment. Building is not deployment authorization.

Treat a PCR recipe as verification only when it compares the built result with
a reviewed, checked-in reference through the matching workflow and the values
match exactly. A successful build or a recipe that performs no comparison is
not PCR evidence.

## Report evidence without overclaiming

Conclude with a compact validation record containing:

- checkout commit and dirty-state summary;
- host platform and whether commands ran through the pinned Nix shell;
- exact commands or named test filters;
- pass/fail counts, ignored counts, and any skip messages;
- disposable database creation, migration, executed subsets, and cleanup;
- external provider, credential source category, egress, and cost authorization without revealing secrets;
- client/backend configuration and exact smoke scenarios;
- Nix system evaluated or built, PCR comparison source, and CI job when relevant;
- every unrun, unavailable, flaky, or platform-specific layer.

Use claims no broader than the evidence:

- **Static/unit validated** means formatting, Clippy, and local unit/property tests only.
- **Disposable-DB validated** means a named ignored subset actually ran against a fresh migrated local database.
- **Live-provider validated** means an explicitly named credentialed test passed for that provider at that time.
- **Local full-stack validated** means a real encrypted client exercised the named flow against local OpenSecret and records what external services were configured.
- **Linux/Nitro/PCR validated** requires the matching Linux/ARM build, check, comparison, or runtime evidence.
- **Deployed validated** requires direct evidence from the named deployed environment; no local tier implies it.

Treat failed, ignored, skipped, interrupted, timing-dependent, and unavailable checks as such. Never convert partial proof into “fully tested,” “production ready,” or “all tests pass.”
