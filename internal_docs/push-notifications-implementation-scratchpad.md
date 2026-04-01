# Push Notifications Implementation Scratchpad

## 2026-03-07 — Kickoff

- Started backend implementation for the mobile push notification system.
- Moved the main design doc from `internal_docs/encrypted-mobile-push-notifications.md` to `docs/encrypted-mobile-push-notifications.md` so it can be shared with app developers.
- Re-read the design doc before starting implementation.
- Current implementation constraints being followed:
  - successful live Maple SSE delivery suppresses push in v1
  - Postgres is the durable coordination plane across multiple enclaves
  - Android v1 uses standard visible FCM notifications plus `data`
  - iOS keeps encrypted preview support with generic fallback
  - stable `notification_id` is required for retry-safe client dedup
- Initial audit focus:
  - agent SSE flow in `src/web/agent/mod.rs`
  - project settings + secrets patterns already used in platform routes and DB helpers

## 2026-03-07 — Scope correction

- User clarified that implementation scope is **Maple agent APIs only** for now.
- Do **not** modify or depend on the Responses API implementation for this push work.
- Current architecture question discovered during audit:
  - agent chat generation currently runs inside the SSE stream in `src/web/agent/mod.rs`
  - if the client disconnects, generation likely stops with the stream
  - to implement “SSE success suppresses push; disconnect triggers push,” we may need to refactor agent generation to continue in a background task after disconnect

## 2026-03-07 — Section 1 complete: schema and core model scaffolding

- Re-read the push design doc sections covering data model, worker leasing, and notification identity before writing schema changes.
- Added a new migration: `2026-03-07-120000_push_notifications_v1`.
- Added schema for:
  - `push_devices`
  - `notification_events`
  - `notification_deliveries`
- Added model files:
  - `src/models/push_devices.rs`
  - `src/models/notification_events.rs`
  - `src/models/notification_deliveries.rs`
- Updated `src/models/mod.rs` and `src/models/schema.rs` to register the new tables.
- Extended `src/models/project_settings.rs` with typed push settings (`PushSettings`, `IosPushSettings`, `AndroidPushSettings`, `PushEnvironment`) and `SettingCategory::Push`.
- Added typed DB helpers in `src/db.rs` for project push settings.
- Current implementation choice:
  - use the existing project-settings DB pattern for push config
  - keep push-device / event / delivery transactional flows on direct Diesel connections so enqueue and delivery leasing can stay atomic

## 2026-03-07 — Section 2 complete: routes, config plumbing, and worker skeleton

- Added platform push settings API under:
  - `GET /platform/orgs/:org_id/projects/:project_id/settings/push`
  - `PUT /platform/orgs/:org_id/projects/:project_id/settings/push`
- Added app-user push device API under:
  - `POST /v1/push/devices`
  - `GET /v1/push/devices`
  - `DELETE /v1/push/devices/:id`
- Added raw project secret helpers in `AppState` so push senders can consume PEM / JSON secrets without the old base64 wrapper.
- Added push module structure:
  - `src/push/mod.rs`
  - `src/push/crypto.rs`
  - `src/push/apns.rs`
  - `src/push/fcm.rs`
  - `src/push/worker.rs`
- Added:
  - per-device P-256 ECDH encrypted preview envelope generation
  - APNs token-auth caching + send path
  - FCM OAuth token caching + HTTP v1 send path
  - leased delivery worker loop with retry / invalid-token handling
- Wired the push worker into server startup.

## 2026-03-07 — Section 3 complete: Maple SSE disconnect behavior

- Refactored `src/web/agent/mod.rs` so agent generation now runs in a background task behind an internal channel instead of inside the SSE stream body.
- New behavior:
  - if the SSE channel is still alive, agent message events are streamed and push is suppressed
  - if the SSE channel is dropped, generation continues and the backend enqueues exactly one notification for that interrupted turn
  - if multiple assistant messages were generated after disconnect, the first missed assistant message is used as the preview source for that one notification
- Stream success detection was tightened beyond simple in-process queueing:
  - the background task now waits for a stream-side delivery acknowledgement before treating an assistant message event as delivered
- Push enqueue helper currently wires agent-originated messages into `notification_events` + `notification_deliveries` with a stable `notification_id` and generic provider-visible fallback text.

## 2026-03-08 — Review fixups in progress

- Updated encrypted preview envelope serialization to match the main spec (`enc_v`, `p256-hkdf-sha256-aes256gcm`).
- Added conservative preview-body truncation so encrypted iOS previews stay within a small payload budget.
- Tightened worker failure handling so pre-send errors now transition deliveries to `retry` or `failed` instead of leaving rows stuck in leased retry loops.
- Tightened push registration:
  - validate registered public keys as P-256 SPKI
  - allow account-switch / stale-token recovery by moving active ownership to the latest installation registration
  - stop revoked rows from permanently occupying the global token uniqueness slot
- Tightened APNs invalidation behavior so topic/config mistakes do not auto-revoke valid devices.

## 2026-03-07 — Section 4 complete: enclave networking glue in-repo

- Updated `entrypoint.sh` with local host mappings for:
  - `api.push.apple.com`
  - `api.sandbox.push.apple.com`
  - `fcm.googleapis.com`
- Added enclave traffic forwarders for:
  - APNs prod (`8024`)
  - APNs sandbox (`8025`)
  - FCM (`8029`)
- Important remaining deployment note:
  - the parent-instance `vsock-proxy` allowlist and systemd unit changes described in the design doc are not represented as editable runtime config files in this repo, so those still need to be applied in deployment infrastructure outside this codebase.

## 2026-03-07 — Validation

- Ran `cargo fmt --all`
- Ran `cargo clippy --all-targets --all-features -- -D warnings`
- Ran `cargo test`
- Result: all validators passed (`104` tests)
- Re-ran `cargo clippy --all-targets --all-features -- -D warnings && cargo test` after the enclave networking script updates; still green.
