# Encrypted Mobile Push Notifications for Maple / OpenSecret

## Implementation-Ready Backend, Enclave, and Mobile Design

**Date:** March 2026
**Status:** Implementation target / pre-build spec
**Related Docs:**
- `sage-in-maple-architecture.md`
- `architecture-for-rag-integration.md`
- `potential-rag-integration-brute-force.md`

---

## 1. Executive Summary

This document defines the concrete v1 design for encrypted mobile push notifications in OpenSecret / Maple.

The final architectural decisions are:

1. **Use direct APNs for iOS and direct FCM HTTP v1 for Android.**
2. **Run the push sender inside the OpenSecret enclave app process.**
3. **Store provider configuration per project using `project_settings` + `org_project_secrets`.**
4. **Register one notification keypair per device, not one shared key per user.**
5. **Use a durable Postgres outbox + delivery worker, not synchronous request-path sends.**
6. **Treat successful live Sage SSE delivery as the acknowledgement for agent chat flows and skip push entirely when streaming succeeds.**
7. **Prefer standard visible-notification behavior on Android in v1; keep Android data-only encrypted preview as an optional later enhancement.**

This gives us Signal-like transport privacy for push content without pretending we are building a full Signal protocol.

---

## 2. What Problem This Solves

We want Maple / Sage to send notifications for events like:

- reminder due
- long-running agent task complete
- async follow-up from Sage
- security or account alerts

while ensuring that:

- Apple and Google do not need plaintext notification content
- minimal routing metadata may remain provider-visible in APNs / FCM payloads, and that is acceptable for v1
- raw push tokens are not stored plaintext in Postgres
- losing one device does not compromise every other device
- the design fits the current OpenSecret enclave + project-scoped secret model

---

## 3. Non-Goals

This design does **not** attempt to provide:

- a double-ratchet messaging protocol
- server blindness to the notification plaintext at creation time
- secrecy for all push metadata, timing information, routing identifiers, or provider-visible fields
- guaranteed hidden lock-screen content after the device itself decrypts it
- a single cross-device shared notification private key

This is **encrypted push delivery**, not a general secure messaging redesign.

---

## 4. Important Clarifications

### 4.1 Apple and Google do not run our decryption logic

They only deliver payloads. The only code we control at receipt time runs:

- in an iOS **Notification Service Extension**
- in an Android **FirebaseMessagingService** / `WorkManager` flow

So decryption must happen **on the device**.

### 4.2 There is no special vsock registration with Apple or Google

We do **not** register AWS vsock or Nitro plumbing with APNs or FCM.

We only need:

- outbound HTTPS access from the enclave to APNs / FCM endpoints
- parent-instance `vsock-proxy` allowlist entries
- enclave-local `/etc/hosts` entries plus `traffic_forwarder.py` mappings

### 4.3 Android and iOS do not behave the same

iOS supports the strongest version of the “messaging app” pattern:

- send a generic visible alert
- include encrypted preview payload
- let NSE replace the visible content before display
- if the NSE fails or times out, iOS shows the original generic alert
- if the app is already foregrounded, the app can suppress the visible notification via `willPresent`

Android does **not** offer the same “generic visible fallback + encrypted replacement” behavior for encrypted previews. The standard Android messaging-app approach is usually:

- send a standard visible FCM notification with non-sensitive text
- include extra `data` fields for routing / grouping / dedup
- if the app is already foregrounded, handle it in-app and avoid showing another local notification

Android **can** do encrypted preview with **high-priority data-only** FCM, but that is less standard and depends more heavily on background execution.

So the v1 system should explicitly model Android delivery as:

1. **Standard visible notification**: default v1 path, most reliable, non-sensitive visible text, app-open fetch for authoritative content
2. **Encrypted preview best-effort**: optional later path, richer UX, but depends on app background execution

This is the most important platform caveat in the whole design. We should follow the standard platform patterns rather than force parity where the OSes behave differently.

---

## 5. Final Architectural Decisions

### 5.1 Per-device notification keypairs

Each device generates its own notification keypair locally and registers only the public key.

Why:

- push routing is already device-specific
- revocation is per-device
- one compromised device does not force full-account rotation
- mobile secure storage works best with locally generated keys

### 5.2 Separate push crypto from existing user-key crypto

Push crypto must **not** reuse the current server-derived secp256k1 user-key flow.

Use a dedicated **P-256 ECDH** notification keypair instead.

Why:

- better native support on iOS and Android
- easier hardware-backed storage
- avoids importing a shared account root into device secure storage

### 5.3 Push sender runs inside the enclave app

The sender should live inside the main OpenSecret server process, behind the current enclave boundary.

Why:

- push payload plaintext can be sensitive
- APNs / FCM credentials are sensitive
- push tokens are sensitive enough to encrypt at rest
- this keeps the privacy story aligned with the rest of OpenSecret

### 5.4 Successful live Sage SSE delivery suppresses push in v1

For the common “user sent a message to Sage and is waiting on an SSE stream” flow:

- if the final Sage response is delivered successfully over the open SSE stream, do **not** enqueue push for any device
- if the SSE stream closes or errors before the final response is delivered, enqueue **exactly one** notification event for that interrupted turn
- if the interrupted turn produced multiple assistant messages, use the **first missed assistant message** as the preview source for that one notification
- this SSE success signal acts as the acknowledgement; no extra per-thread presence service or explicit ACK protocol is required in v1

This is intentionally simple and matches the normal messaging-app intuition: if the user is clearly active on one device and the live response succeeded, don’t notify every other device.

### 5.5 Outbox + worker, not inline send

Notification generation and transport delivery must be separate steps.

Why:

- APNs / FCM reject, throttle, and invalidate tokens
- we need retries and backoff
- future reminder scheduling requires durable queue semantics
- request handlers should not block on provider APIs

### 5.6 Stable notification IDs + client dedup are required

Every logical notification must have a stable `notification_id`.

Use `notification_events.uuid` as that canonical ID.

Rules:

- retries must reuse the same `notification_id`
- clients should keep a small cache of recently seen `notification_id` values and no-op duplicates
- APNs `apns-collapse-id` and FCM `collapse_key` should only be reused for retries of the same logical notification, not for every message in a thread by default

---

## 6. Existing Codebase Integration Map

This section is the implementation map for a follow-on coding agent.

### 6.1 Existing files to modify

#### Database and models

- `migrations/<new_push_migration>/up.sql`
  - Add `push_devices`, `notification_events`, and `notification_deliveries`.
  - Add indexes and `updated_at` triggers matching the existing schema style.

- `migrations/<new_push_migration>/down.sql`
  - Drop the new push tables and indexes.

- `src/models/mod.rs`
  - Export the new push model modules.

- `src/models/schema.rs`
  - Regenerate via Diesel after the migration is added.

- `src/models/project_settings.rs`
  - Extend `SettingCategory` with `Push`.
  - Add `PushSettings`, `IosPushSettings`, `AndroidPushSettings`, and related helpers.

- `src/db.rs`
  - Add trait methods and `PostgresConnection` implementations for push device CRUD, event enqueue, delivery leasing, retry, and invalidation.
  - Mirror the existing settings / secrets patterns already used for email and OAuth.

#### Web API routing

- `src/web/mod.rs`
  - Export a new `push` router module.

- `src/main.rs`
  - Merge the new push router under `validate_jwt`.
  - Start the background push worker after app state initialization and migrations.

- `src/web/agent/mod.rs`
  - Gate push enqueue on whether a live SSE response finished successfully; this is the v1 suppression signal for the normal Sage agent flow.

- `src/web/platform/common.rs`
  - Add new secret constants and request / response types for project push settings.

- `src/web/platform/project_routes.rs`
  - Add `GET` / `PUT /platform/orgs/:org_id/projects/:project_id/settings/push`.
  - Reuse the existing project secret endpoints for actual APNs / FCM credentials.

#### Deployment / runtime

- `entrypoint.sh`
  - Add `/etc/hosts` entries and `traffic_forwarder.py` processes for APNs production, APNs sandbox, and FCM.

- `docs/nitro-deploy.md`
  - Add parent-instance `vsock-proxy` instructions for the new outbound hosts.
  - This is a deployment follow-up, not required for compiling the backend.

### 6.2 New files to add

- `src/models/push_devices.rs`
  - Diesel model and helper methods for device registrations.

- `src/models/notification_events.rs`
  - Durable logical notification rows.

- `src/models/notification_deliveries.rs`
  - Per-device delivery state rows.

- `src/web/push.rs`
  - App-user authenticated routes like `POST /v1/push/devices`.

- `src/push/mod.rs`
  - Shared types, enqueue helpers, and internal interfaces.

- `src/push/crypto.rs`
  - P-256 ECDH + HKDF + AES-GCM envelope logic.

- `src/push/apns.rs`
  - APNs request building, JWT auth, and response handling.

- `src/push/fcm.rs`
  - FCM OAuth token exchange, request building, and response handling.

- `src/push/worker.rs`
  - Polling / leasing loop that sends pending deliveries.

### 6.3 Codebase gotchas to preserve

1. `src/main.rs::get_project_secret()` currently returns **base64-encoded decrypted bytes**.
   That is fine for some existing flows, but APNs `.p8` material and FCM service account JSON are easier to consume as raw bytes or raw UTF-8. Add a raw helper rather than forcing each caller to decode a second time.

2. `src/encrypt.rs` is useful for **at-rest encryption** but not for device ECDH push envelopes.
   Its current primitives take `secp256k1::SecretKey` and do symmetric encryption only.

3. `src/main.rs::create_ephemeral_key()` uses **x25519** for session establishment.
   Do not overload that codepath for push notifications.

4. New user-facing push APIs should live under **`/v1/push/*`**, not under the older `/protected/*` namespace.
   That matches the newer `agent` and `responses` route style.

---

## 7. Project Configuration Model

Push config should follow the same split already used elsewhere in the repo.

### 7.1 Non-secret settings in `project_settings`

Add a new `SettingCategory::Push` and store a JSON payload like:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PushSettings {
    pub encrypted_preview_enabled: bool,
    pub ios: Option<IosPushSettings>,
    pub android: Option<AndroidPushSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IosPushSettings {
    pub enabled: bool,
    pub bundle_id: String,
    pub apns_environment: PushEnvironment,
    pub team_id: String,
    pub key_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidPushSettings {
    pub enabled: bool,
    pub firebase_project_id: String,
    pub package_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PushEnvironment {
    Dev,
    Prod,
}
```

`encrypted_preview_enabled` controls whether the server attaches the encrypted preview envelope for supported devices. If disabled, the system still uses a generic provider-visible notification shell; it must **not** fall back to plaintext message content.

Notes:

- no new SQL table is required for settings; `project_settings` already supports new categories via JSONB
- `team_id` and `key_id` are identifiers, not secrets
- APNs environment stays explicit because sandbox vs production matters operationally

### 7.2 Secrets in `org_project_secrets`

Add constants in `src/web/platform/common.rs`:

```rust
pub const PROJECT_APNS_AUTH_KEY_P8: &str = "APNS_AUTH_KEY_P8";
pub const PROJECT_FCM_SERVICE_ACCOUNT_JSON: &str = "FCM_SERVICE_ACCOUNT_JSON";
```

Use the existing platform secret endpoints to store them.

Important:

- the existing platform secret API expects **base64-encoded raw bytes**
- APNs `.p8` content should be uploaded as raw PEM bytes, then base64-encoded for transport to the API
- FCM service account JSON should be uploaded as raw JSON bytes, then base64-encoded for transport to the API

---

## 8. Data Model

### 8.1 `push_devices`

One row per active installation binding, with revoked rows retained for re-registration history and delivery auditing.

```sql
CREATE TABLE push_devices (
    id                              BIGSERIAL PRIMARY KEY,
    uuid                            UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id                         UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    installation_id                 UUID NOT NULL,
    platform                        TEXT NOT NULL,
    provider                        TEXT NOT NULL,
    environment                     TEXT NOT NULL,
    app_id                          TEXT NOT NULL,

    push_token_enc                  BYTEA NOT NULL,
    push_token_hash                 BYTEA NOT NULL,

    notification_public_key         BYTEA NOT NULL,
    key_algorithm                   TEXT NOT NULL,

    supports_encrypted_preview      BOOLEAN NOT NULL DEFAULT FALSE,
    supports_background_processing  BOOLEAN NOT NULL DEFAULT FALSE,

    last_seen_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at                      TIMESTAMPTZ,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        (platform = 'ios' AND provider = 'apns') OR
        (platform = 'android' AND provider = 'fcm')
    ),
    CHECK (environment IN ('dev', 'prod')),
    CHECK (key_algorithm = 'p256_ecdh_v1')
);

CREATE INDEX idx_push_devices_user_active
    ON push_devices(user_id, revoked_at);

CREATE UNIQUE INDEX idx_push_devices_installation_active
    ON push_devices(installation_id, environment)
    WHERE revoked_at IS NULL;

CREATE UNIQUE INDEX idx_push_devices_token_active
    ON push_devices(provider, environment, push_token_hash)
    WHERE revoked_at IS NULL;
```

Implementation notes:

- `installation_id` is an app-generated UUIDv4 for one installed copy of the app; it is not an APNs or FCM identifier
- store `installation_id` in normal app-scoped local storage so uninstall / reinstall creates a fresh installation identity
- `push_token_enc` should be encrypted at rest with the enclave key using the same style as project secrets
- `push_token_hash` should be `SHA-256(push_token)` raw bytes for dedupe and lookup
- `notification_public_key` should store raw DER-encoded SPKI bytes, not a base64 string
- keep at most one active binding per `(installation_id, environment)` and per `(provider, environment, push_token_hash)`; older rows are revoked rather than overwritten

### 8.2 `notification_events`

Logical notification rows created by product code.

```sql
CREATE TABLE notification_events (
    id                  BIGSERIAL PRIMARY KEY,
    uuid                UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    project_id          INTEGER NOT NULL REFERENCES org_projects(id) ON DELETE CASCADE,
    user_id             UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    kind                TEXT NOT NULL,
    delivery_mode       TEXT NOT NULL,
    priority            TEXT NOT NULL DEFAULT 'normal',
    collapse_key        TEXT,

    fallback_title      TEXT NOT NULL,
    fallback_body       TEXT NOT NULL,
    payload_enc         BYTEA,

    not_before_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cancelled_at        TIMESTAMPTZ,

    CHECK (delivery_mode IN ('generic', 'encrypted_preview')),
    CHECK (priority IN ('normal', 'high'))
);

CREATE INDEX idx_notification_events_due
    ON notification_events(user_id, not_before_at, cancelled_at);
```

Implementation notes:

- `payload_enc` should contain the encrypted preview payload or deep-link metadata encrypted at rest with the enclave key
- `fallback_title` and `fallback_body` are always explicit so the worker never has to invent fallback text at send time
- `notification_events.uuid` is the canonical `notification_id` exposed to clients and reused across retries

### 8.3 `notification_deliveries`

Per-device delivery records.

```sql
CREATE TABLE notification_deliveries (
    id                  BIGSERIAL PRIMARY KEY,
    event_id            BIGINT NOT NULL REFERENCES notification_events(id) ON DELETE CASCADE,
    push_device_id      BIGINT NOT NULL REFERENCES push_devices(id) ON DELETE CASCADE,

    status              TEXT NOT NULL DEFAULT 'pending',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    next_attempt_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,

    provider_message_id TEXT,
    provider_status_code INTEGER,
    last_error          TEXT,
    sent_at             TIMESTAMPTZ,
    invalidated_at      TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (event_id, push_device_id),
    CHECK (
        status IN (
            'pending',
            'leased',
            'sent',
            'retry',
            'failed',
            'invalid_token',
            'cancelled'
        )
    )
);

CREATE INDEX idx_notification_deliveries_pending
    ON notification_deliveries(status, next_attempt_at);
```

Implementation notes:

- deliveries should be materialized when the event is enqueued, not lazily at send time
- the worker should still re-check `push_devices.revoked_at` before sending

---

## 9. API Surface

### 9.1 App-user push device API

Create a new router in `src/web/push.rs` and mount it in `main.rs` with `validate_jwt` plus the usual encrypted request / response middleware.

#### Register or rotate device

```http
POST /v1/push/devices
```

Request payload:

```json
{
  "installation_id": "uuid",
  "platform": "ios",
  "provider": "apns",
  "environment": "prod",
  "app_id": "ai.trymaple.ios",
  "push_token": "opaque platform token string",
  "notification_public_key": "base64-encoded SPKI DER",
  "key_algorithm": "p256_ecdh_v1",
  "supports_encrypted_preview": true,
  "supports_background_processing": true
}
```

Behavior:

- `installation_id` must be a non-nil UUIDv4 generated and stored by the app
- if the authenticated user already has a row for that installation, update it in place and clear `revoked_at`
- if another active row currently owns the same `installation_id` or the same push token, revoke that older binding before activating the current authenticated user
- rotate token / public key / capabilities when they change
- refresh `last_seen_at`

#### List current user's devices

```http
GET /v1/push/devices
```

Return metadata only, never the raw push token.

#### Revoke one device

```http
DELETE /v1/push/devices/:id
```

Behavior:

- user-scoped revoke only
- set `revoked_at`
- optionally return the standard deleted-object response shape already used in newer APIs

#### Client lifecycle expectations

- first install: generate a fresh UUIDv4 `installation_id`, generate the notification keypair, fetch the APNs / FCM token, then call `POST /v1/push/devices`
- app start, login, or token rotation: call `POST /v1/push/devices` again; the route is idempotent for the current authenticated user and installation
- explicit logout: while the access token is still valid, call `DELETE /v1/push/devices/:id` for the current device before dropping auth
- account switch on the same installed app: revoke the old device binding on logout, then register again after the new user logs in; if the old binding is still active, the server treats the matching installation or token as a handoff and revokes the older binding
- uninstall / reinstall: create a new `installation_id` and notification keypair, then register as a new installation on next login; stale old rows are cleaned up by explicit revoke or later invalid-token handling

#### Logout cleanup fallback

The existing `/logout` request may also include an optional `push_device_id` field so the backend can best-effort revoke the current push binding during auth teardown:

```json
{
  "refresh_token": "jwt",
  "push_device_id": "uuid"
}
```

Preferred client flow:

1. call `DELETE /v1/push/devices/:id` while the access token is still valid
2. call `/logout`
3. include `push_device_id` in `/logout` as a cleanup fallback in case the device revoke races with local auth teardown

### 9.2 Platform project settings API

Add a new project settings endpoint in `src/web/platform/project_routes.rs`:

```http
GET /platform/orgs/:org_id/projects/:project_id/settings/push
PUT /platform/orgs/:org_id/projects/:project_id/settings/push
```

Use existing platform auth, org membership checks, and the same validation style already used for email / OAuth settings.

Do **not** add a dedicated secret endpoint. Reuse the existing generic project secret endpoints with:

- `APNS_AUTH_KEY_P8`
- `FCM_SERVICE_ACCOUNT_JSON`

---

## 10. DB Trait and Service Methods

Add the following categories of methods to `src/db.rs`.

### 10.1 Push device methods

- `upsert_push_device(...)`
- `get_push_device_by_uuid_for_user(...)`
- `list_active_push_devices_for_user(...)`
- `revoke_push_device_for_user(...)`
- `invalidate_push_device(...)`

### 10.2 Push settings methods

- `get_project_push_settings(project_id)`
- `update_project_push_settings(project_id, settings)`

This mirrors the existing email / OAuth settings pattern.

### 10.3 Event and delivery methods

- `create_notification_event(...)`
- `create_notification_deliveries_for_event(...)`
- `lease_pending_notification_deliveries(limit, lease_owner, lease_ttl)`
- `mark_notification_delivery_sent(...)`
- `mark_notification_delivery_retry(...)`
- `mark_notification_delivery_failed(...)`
- `mark_notification_delivery_invalid_token(...)`

The lease query should use a transaction plus `FOR UPDATE SKIP LOCKED` semantics so the design works across multiple workers / enclaves from day one.
Use Postgres time (`now()`) for lease and retry timestamps instead of local enclave clock time.

---

## 11. Worker Architecture

### 11.1 Where the worker should start

Initialize the worker from `src/main.rs` after:

- app state build
- runtime migrations
- provider setup that already happens during boot

Do **not** initialize it from request handlers.

### 11.2 Worker structure

Create `src/push/worker.rs` with a loop like:

1. sleep / interval tick
2. lease due deliveries from Postgres
3. for each leased delivery:
   - load event + device + project settings
   - skip if device revoked or event cancelled / expired
   - send via APNs or FCM
   - update delivery status based on response

Recommended v1 behavior:

- poll every 2-5 seconds
- cap concurrent sends with a small semaphore
- exponential backoff on provider retryable failures

### 11.3 Multi-enclave operation

Run the worker in **every** OpenSecret backend / enclave instance.

Shared Postgres is the coordination plane.

Rules:

- workers claim batches using row-level leases plus `FOR UPDATE SKIP LOCKED`
- no leader election is required
- any enclave can resume work after another enclave crashes
- `lease_owner`, `lease_expires_at`, `next_attempt_at`, and `status` must all live in Postgres

### 11.4 Delivery semantics

This design is **at-least-once**, not exactly-once.

If a worker sends successfully to APNs / FCM but crashes before marking the row `sent`, another worker may retry after the lease expires.

That is acceptable if we keep these protections in place:

- stable `notification_id`
- one delivery row per `(event_id, push_device_id)`
- client-side recent-ID dedup cache
- provider collapse identifiers reused only for retries of the same logical notification

### 11.5 Why Postgres outbox instead of SQS for v1

The repo already has some SQS plumbing, but v1 push delivery should still use Postgres as the source of truth.

Why:

- fewer moving parts
- easier local/dev operation
- delivery state stays queryable in one place
- no need to solve dual-write consistency on day one

SQS can be added later as a wake-up optimization if volume requires it.

---

## 12. Cryptographic Design

### 12.1 What can be reused from the current repo

Reusable:

- `src/encrypt.rs::encrypt_with_key` for encrypting raw bytes at rest with the enclave key
- `src/encrypt.rs::decrypt_with_key` for decrypting those bytes
- existing `sha2` dependency for token hashing or HKDF backing hash

Not reusable as-is:

- current secp256k1 user-key flow
- current x25519 session setup in `AppState`
- any API that assumes a `secp256k1::SecretKey` is the transport key

### 12.2 New dependencies to add

Recommended new crates:

- `p256` for P-256 ECDH and public-key parsing
- `hkdf` for key derivation

Existing crates already cover the rest:

- `aes-gcm`
- `sha2`
- `jsonwebtoken`
- `reqwest`

### 12.3 Registered public-key format

The client should register the public key as:

- **base64-encoded DER SubjectPublicKeyInfo bytes** over the API
- stored as raw bytes in Postgres

This is the best interoperability point because:

- Android naturally exports X.509 / SPKI bytes
- iOS can export DER representation
- Rust `p256` can parse SPKI cleanly

### 12.4 Envelope algorithm

Use:

- `P-256 ECDH`
- `HKDF-SHA256`
- `AES-256-GCM`

Recommended envelope:

```json
{
  "enc_v": 1,
  "alg": "p256-hkdf-sha256-aes256gcm",
  "kid": "push-device-uuid",
  "epk": "base64-encoded ephemeral SEC1 public key",
  "salt": "base64-encoded 32-byte salt",
  "nonce": "base64-encoded 12-byte nonce",
  "ciphertext": "base64-encoded AES-GCM ciphertext"
}
```

### 12.5 Plaintext preview payload

```json
{
  "v": 1,
  "notification_id": "uuid",
  "message_id": "uuid",
  "kind": "sage.reminder",
  "title": "Reminder",
  "body": "Follow up on the deployment thread",
  "deep_link": "opensecret://agent/subagent/uuid",
  "thread_id": "agent:uuid",
  "sent_at": 1772800000
}
```

`notification_id` should be identical to `notification_events.uuid` and remain stable across retries.

Keep this intentionally small.

For the initial implementation, normalize whitespace and truncate the preview `body` to a conservative byte budget before encryption so the final APNs payload stays comfortably under provider limits.

### 12.6 At-rest encryption rules

- encrypt `push_token_enc` with the enclave key
- encrypt `notification_events.payload_enc` with the enclave key
- do **not** store preview plaintext or raw push tokens in logs

---

## 13. APNs Integration

### 13.1 Endpoint and auth

Send requests to:

- production: `https://api.push.apple.com/3/device/<token>`
- development: `https://api.sandbox.push.apple.com/3/device/<token>`

Use APNs token auth:

- sign JWT with the stored `.p8` key
- header: `alg=ES256`, `kid=<key_id>`
- claims: `iss=<team_id>`, `iat=<unix seconds>`
- refresh before one hour elapses; cache for about 50 minutes

### 13.2 Required headers

Use:

- `authorization: bearer <jwt>`
- `apns-topic: <bundle_id>`
- `apns-push-type: alert`
- `apns-priority: 10`
- `apns-collapse-id: notif:<notification_id>` when retrying the same logical notification

Do **not** reuse one collapse ID for every message in a thread by default.
If desired, use `aps.thread-id` for notification-center grouping separately from collapse behavior.

### 13.3 Payload shape for encrypted preview on iOS

```json
{
  "aps": {
    "alert": {
      "title": "New Maple update",
      "body": "Open Maple to view it"
    },
    "mutable-content": 1,
    "sound": "default"
  },
  "os_meta": {
    "notification_id": "...",
    "kind": "agent.message",
    "message_id": "...",
    "thread_id": "agent:subagent:...",
    "deep_link": "opensecret://agent/subagent/..."
  },
  "os_push": {
    "enc_v": 1,
    "alg": "p256-hkdf-sha256-aes256gcm",
    "kid": "...",
    "epk": "...",
    "salt": "...",
    "nonce": "...",
    "ciphertext": "..."
  }
}
```

Important:

- the fallback `alert.title` and `alert.body` must be non-empty
- the fallback alert must remain **generic**; never place plaintext assistant content in the provider-visible payload
- `mutable-content: 1` is required for the NSE to run
- if the NSE times out, iOS shows the original generic alert
- `os_meta` is allowed to carry minimal routing metadata (`notification_id`, `message_id`, `thread_id`, `deep_link`, `kind`), but not plaintext notification content
- this is intentional: the privacy requirement is protecting notification **content**, not hiding all routing metadata from the push provider

### 13.4 APNs error handling

Treat these as permanent invalid-token outcomes:

- `410 Unregistered`

Treat these as permanent non-retryable failures, but **do not** auto-revoke the device row based on them alone:

- `400 BadDeviceToken`
- `400 DeviceTokenNotForTopic`

Treat these as retryable:

- `429`
- `500`
- `503`

Persist:

- HTTP status
- APNs `apns-id`
- failure reason string if returned

Note: APNs token invalidation signals can be delayed. Use them for cleanup, but do not infer uninstall with certainty.

---

## 14. FCM Integration

### 14.1 Endpoint and auth

Send requests to:

```text
https://fcm.googleapis.com/v1/projects/<firebase_project_id>/messages:send
```

Use OAuth 2 service-account auth:

1. load decrypted service account JSON
2. create a JWT assertion signed with the service-account RSA private key
3. exchange it at `https://oauth2.googleapis.com/token` (or the service account `token_uri`)
4. cache the returned access token until shortly before expiry

Required scope:

```text
https://www.googleapis.com/auth/firebase.messaging
```

### 14.2 Recommended Android v1 behavior

For v1, prefer the standard messaging-app approach:

- send a normal visible FCM notification with non-sensitive text
- include `data` fields for routing, grouping, and duplicate protection
- let the app fetch or render authoritative content after open

This is the most standard and reliable path.
It avoids relying on background execution for every Sage message.

### 14.3 Standard FCM payload for v1

```json
{
  "message": {
    "token": "<fcm registration token>",
    "notification": {
      "title": "New Sage message",
      "body": "Open Maple to view it"
    },
    "data": {
      "notification_id": "<uuid>",
      "thread_id": "agent:uuid",
      "message_id": "<uuid>",
      "deep_link": "opensecret://agent/subagent/uuid"
    },
    "android": {
      "priority": "HIGH",
      "collapse_key": "sage:notif:<notification_id>",
      "ttl": "300s"
    }
  }
}
```

Notes:

- visible text should remain generic / non-sensitive
- FCM `data` values must be strings
- FCM `data` is provider-visible routing metadata, so it may include fields like `notification_id`, `message_id`, `thread_id`, `deep_link`, and `kind`, but never plaintext message content
- this is intentional in v1: the goal is to keep message content encrypted from the push provider, not to hide all metadata needed for client routing
- reuse the same `collapse_key` only for retries of the same logical notification
- `notification_id` is for exact dedup; `thread_id` is for grouping / clearing when the user opens the conversation

### 14.4 Optional Android encrypted-preview path (deferred)

If later we want closer parity with iOS:

- send a **high-priority data-only** message
- `FirebaseMessagingService.onMessageReceived()` decrypts it
- app posts a local notification after decryption
- if the process does not run, the encrypted preview may not be shown

That makes Android encrypted preview **best-effort**, not a required part of the v1 design.

### 14.5 FCM error handling

Treat these as permanent invalid-token outcomes:

- `404 UNREGISTERED`
- `400 INVALID_ARGUMENT` when the error clearly identifies token invalidity

Treat these as retryable:

- `429`
- `500`
- `503`

Persist:

- HTTP status
- returned FCM message name on success
- error code / status string on failure

---

## 15. Mobile Client Requirements

### 15.1 iOS

#### Registration flow

1. request notification permission
2. receive APNs device token
3. generate `P256.KeyAgreement.PrivateKey()` locally
4. store private key in keychain / Secure Enclave-backed flow when available
5. expose the public key in DER / SPKI form
6. register with `POST /v1/push/devices`

#### Storage requirements

The key must be accessible from both:

- main app target
- Notification Service Extension target

Recommended keychain accessibility:

- `AfterFirstUnlockThisDeviceOnly`-style semantics

This allows NSE access after first unlock while keeping the key device-bound.

#### NSE behavior

The Notification Service Extension should:

1. read `userInfo["os_push"]`
2. load the private key from shared keychain storage
3. decrypt the envelope
4. rewrite `title`, `body`, and routing metadata
5. call the completion handler promptly

If the key is unavailable or decryption fails, do nothing and let iOS show the generic fallback alert.

#### Foreground / open-thread behavior

If the app is already foregrounded and showing the target Sage conversation or active response screen:

- implement `userNotificationCenter(_:willPresent:withCompletionHandler:)`
- return `[]` to suppress banner / sound / badge presentation for that in-app event
- clear previously delivered notifications for that `thread_id` when the user opens the conversation

The NSE should not be treated as the primary “suppress if user is already looking” mechanism. For v1, successful SSE delivery is the primary suppression signal.

### 15.2 Android

#### Registration flow

1. request notification permission on Android 13+
2. obtain FCM registration token
3. generate an EC keypair in Android Keystore
4. export the public key as SPKI bytes
5. register with `POST /v1/push/devices`

#### Runtime behavior

- use `FirebaseMessagingService.onNewToken()` to rotate tokens
- for v1, assume the default incoming path is a standard visible FCM notification plus `data`
- if the app is already foregrounded and showing the target Sage conversation, update UI and do not show an extra local notification
- when the user opens a conversation, clear notifications associated with that `thread_id`
- if we later enable Android data-only encrypted preview, handle it in `onMessageReceived()` and only post a local notification after successful decrypt
- use `WorkManager` for longer processing if needed

#### Direct Boot caveat

Before first unlock after reboot, Android secure key material may be unavailable.

So:

- encrypted preview mode should degrade to app-open fetch or no preview
- if a notification must be visible pre-unlock, use generic mode instead

### 15.3 Cross-platform client guidance

#### Recent-ID dedup cache

Each client should persist a small cache of recently seen `notification_id` values for at least 1-7 days.

Rule:

- if a notification arrives with an already-seen `notification_id`, treat it as a no-op

This is the client-side protection against at-least-once retries or server bugs.

#### Thread grouping and cleanup

Each payload should include a `thread_id` when applicable.

Use it to:

- group notifications in app UI if desired
- clear thread notifications when the user opens that thread
- avoid leaving stale notifications around once the user has seen the conversation

#### Foreground suppression is defense-in-depth

If a visible notification arrives while the app is already open, suppress or no-op locally where the platform allows it.

But the primary v1 suppression rule is still server-side:

- successful live SSE delivery => no push enqueue at all

---

## 16. Nitro / Enclave Networking Changes

### 16.1 Parent-instance allowlist additions

Add these hosts to `/etc/nitro_enclaves/vsock-proxy.yaml`:

```yaml
- {address: api.push.apple.com, port: 443}
- {address: api.sandbox.push.apple.com, port: 443}
- {address: fcm.googleapis.com, port: 443}
```

`oauth2.googleapis.com` is already in the repo's current setup and can be reused for the FCM access-token exchange.

### 16.2 Parent-instance proxy services

Following the existing patterns in `docs/nitro-deploy.md`, add new parent services such as:

- `vsock-apns-prod-proxy.service`
- `vsock-apns-sandbox-proxy.service`
- `vsock-fcm-proxy.service`

Example mappings:

- APNs prod on parent port `8024`
- APNs sandbox on parent port `8025`
- FCM on parent port `8029`

Any unused ports are acceptable; they just need to stay consistent with enclave `traffic_forwarder.py` mappings.

### 16.3 Enclave `entrypoint.sh` changes

Add new `/etc/hosts` entries and forwarders, for example:

```sh
echo "127.0.0.21 api.push.apple.com" >> /etc/hosts
echo "127.0.0.22 api.sandbox.push.apple.com" >> /etc/hosts
echo "127.0.0.34 fcm.googleapis.com" >> /etc/hosts

run_forever tf_apns_prod python3 /app/traffic_forwarder.py 127.0.0.21 443 3 8024 &
run_forever tf_apns_sandbox python3 /app/traffic_forwarder.py 127.0.0.22 443 3 8025 &
run_forever tf_fcm python3 /app/traffic_forwarder.py 127.0.0.34 443 3 8029 &
```

The exact IPs and ports can be changed, but they should use currently unused slots.

---

## 17. Product Event Model

### 17.1 Live Sage agent SSE flows

For the common flow where the user sends a message to Sage and the client keeps an SSE stream open waiting for responses:

1. generate and stream the Sage response as normal
2. if the final response is successfully delivered over that SSE stream, do **not** enqueue push
3. if the SSE stream closes or errors before final response delivery completes, enqueue exactly one notification event for that interrupted turn
4. if multiple assistant messages were generated after the disconnect point, use the first missed assistant message as the preview source for that one notification
5. that push fan-out then goes to all active devices

This deliberately treats a successful live SSE stream as “the user likely saw it” and avoids more complex per-thread presence logic in v1.

### 17.2 Internal enqueue API

Do not create a public "send push" API in v1.

Instead add an internal enqueue helper in `src/push/mod.rs` shaped roughly like:

```rust
pub struct EnqueueNotificationRequest {
    pub project_id: i32,
    pub user_id: Uuid,
    pub kind: String,
    pub delivery_mode: PushDeliveryMode,
    pub priority: PushPriority,
    pub collapse_key: Option<String>,
    pub fallback_title: String,
    pub fallback_body: String,
    pub payload: Option<NotificationPreviewPayload>,
    pub not_before_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}
```

### 17.3 Initial event sources

Likely first integrations:

- `src/web/agent/runtime.rs`
  - subagent follow-up or background completion signals

- `src/web/agent/mod.rs`
  - live Sage SSE completion / failure handling

Reminder scheduling can come later; the outbox design already supports `not_before_at`.

### 17.4 Notification identity and duplicate protection

- `notification_events.uuid` is the canonical `notification_id`
- include `thread_id` and `message_id` in payloads whenever applicable
- retries must reuse the same `notification_id`
- APNs `apns-collapse-id` and FCM `collapse_key` should key off that `notification_id`
- clients should keep a recent-ID cache and no-op duplicates

---

## 18. Validation and Test Plan

### 18.1 Unit tests

Add focused tests for:

- P-256 envelope roundtrip in `src/push/crypto.rs`
- APNs JWT generation and refresh behavior
- FCM service-account assertion generation and token parsing
- push-device upsert / revoke / invalidate DB behavior
- delivery leasing and retry transitions

### 18.2 Integration tests

Add route tests for:

- `POST /v1/push/devices`
- `GET /v1/push/devices`
- `DELETE /v1/push/devices/:id`
- `GET` / `PUT` platform push settings routes

### 18.3 Provider client testability

Do not hardcode transport base URLs so deeply that they cannot be mocked.

The APNs and FCM sender code should allow test-only override of base URLs, making it possible to verify:

- request headers
- request body shapes
- retry classification
- invalid-token handling

---

## 19. Implementation Order

### Phase 1: Schema and project settings

- add migration for push tables
- add push model files and Diesel schema
- add `SettingCategory::Push`
- add project push settings routes
- add new project secret constants

### Phase 2: Device registry API

- add `src/web/push.rs`
- add push-device DB methods
- implement register / list / revoke routes
- add user-scoped tests

### Phase 3: Outbox and generic delivery

- add event / delivery DB methods
- add worker leasing loop
- wire live Sage SSE paths so successful stream completion skips push enqueue
- implement APNs generic delivery
- implement standard FCM visible delivery

### Phase 4: Client dedup + foreground suppression + iOS encrypted preview

- add stable `notification_id` handling end-to-end
- add client recent-ID cache guidance and integration
- add thread-based notification cleanup when a conversation opens
- add `p256` + `hkdf`
- implement per-device envelope encryption
- add iOS encrypted preview payload / NSE rewrite

### Phase 5: Android optional encrypted-preview path

- keep standard visible FCM delivery as the default
- optionally add Android data-only encrypted preview if product later decides it is worth the extra complexity
- if enabled, implement it in `FirebaseMessagingService` with local-notification posting only after successful decrypt

### Phase 6: Product integrations and hardening

- connect Maple / Sage event sources
- add collapse keys, TTL, and backoff policy tuning
- add metrics and invalid token cleanup dashboards
- optionally add stronger device attestation later

---

## 20. Rejected Alternatives

### 20.1 Shared per-user notification key

Rejected because it makes revocation and secure local storage worse.

### 20.2 Sender outside the enclave

Rejected for v1 because it weakens the trust boundary around payload plaintext and provider secrets.

### 20.3 Silent-only push as the only strategy

Rejected because silent / background behavior is too unreliable for user-facing alerts.

### 20.4 Reusing existing user-key encryption APIs

Rejected because those APIs are built around server-derived secp256k1 secrets and do not map cleanly to per-device mobile decryption.

---

## 21. Final Recommendation

The v1 implementation should be:

- **direct APNs + direct FCM HTTP v1**
- **push sender inside the enclave app**
- **project-scoped provider config in `project_settings` + `org_project_secrets`**
- **per-device P-256 notification keypairs**
- **Postgres outbox + multi-enclave delivery worker coordinated via Postgres row leases**
- **successful live Sage SSE delivery suppresses push entirely**
- **iOS encrypted preview with generic fallback**
- **Android v1 uses standard visible FCM notifications plus data, with app-open fetch and local foreground suppression**
- **stable `notification_id` plus client-side dedup / thread cleanup**

This is the simplest design that still matches the current OpenSecret architecture and avoids painting us into a corner later.
