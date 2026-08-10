---
name: change-opensecret-api
description: Change or review OpenSecret HTTP APIs and their Maple or SDK consumers. Use for route, authentication, attestation-session, encrypted request or response, OpenAI-compatible payload, Responses or conversation persistence, error-status, SSE streaming, or cross-client compatibility work in the opensecret backend.
---

# Change the OpenSecret API

Treat the backend, released SDKs, and Maple as one protocol. Preserve the security envelope and existing public behavior unless the task explicitly changes the contract.

## Establish the live contract

1. Read the repository `AGENTS.md` and relevant local docs.
2. Confirm the checkout, base, and existing changes before editing. Preserve unrelated work.
3. Derive the current route inventory from source; never trust a copied endpoint table:

   ```sh
   rg -n -C 2 '\.route\(' src/web
   sed -n '3660,3735p' src/main.rs
   ```

4. Trace one route end to end: router method and middleware, request type, handler, storage/provider calls, response type, tests, SDK method, and Maple call site.
5. Classify the route before changing it:

   - Health and attestation bootstrap routes are unauthenticated and unencrypted.
   - Login, registration, password recovery, and OAuth bootstrap routes are not JWT-authenticated, but still use the attested encryption session.
   - Protected, Responses, conversation, instruction, project, and direct web routes require a user JWT plus the encryption session.
   - OpenAI-shaped inference routes accept a user JWT or a UUID API key plus the encryption session.
   - `/platform/*` belongs to a separate control-plane surface. Leave it out of Maple-facing work unless the task names it.

Use `src/main.rs` router composition as the authority for outer authentication middleware and each route module as the authority for inner decryption middleware.

## Preserve the transport boundary

The public OpenAI-shaped API is not a plaintext OpenAI wire API. Clients must use an OpenSecret SDK or reproduce its verified protocol.

- `GET /attestation/:nonce` binds a one-time server X25519 key to the attestation document.
- `POST /key_exchange` consumes that nonce and returns a session ID plus an encrypted session key.
- Every protected request carries `x-session-id`.
- A mutating request body is JSON shaped as `{"encrypted":"<base64>"}`. The decoded bytes are `12-byte nonce || ChaCha20-Poly1305 ciphertext and tag`.
- GET, DELETE, and typed bodyless requests omit the encrypted body, but still require a live session. They are not public simply because the body is empty.
- A successful non-streaming response is an encrypted JSON envelope. Do not return successful plaintext data from an encrypted route.
- Normal SSE `data:` payloads are independently encrypted and base64-encoded. Chat's terminal `data: [DONE]` marker is the only normal plaintext data frame.
- Keep a session lease alive through the complete response body. A streaming handler must not release or evict its session while the client is reading.

Inspect the live limits rather than copying their values:

```sh
rg -n 'MAX_ENCRYPTED_BODY_BYTES|MAX_ATTESTATION_NONCE_BYTES|MAX_PENDING_ATTESTATIONS|PENDING_ATTESTATION_TTL|MAX_ENCRYPTION_SESSIONS|ENCRYPTION_SESSION_IDLE_TTL' src
```

Security-sensitive implementation lives in:

- `src/web/attestation_routes.rs`
- `src/web/encryption_middleware.rs`
- the session-cache and encrypt/decrypt methods in `src/main.rs`
- `src/lease_aware_cache.rs` and `src/bounded_ttl_cache.rs`

Do not bypass attestation in a non-local client, weaken the contributory X25519 check, reuse a consumed nonce, log session keys or plaintext payloads, or accept an unknown session on a bodyless route.

## Preserve authentication semantics

Use `src/web/openai_auth.rs` and `src/jwt.rs` as the source of truth.

- `Authorization: Bearer <uuid>` is an API key only on the OpenAI-shaped router. The backend hashes the canonical UUID string before lookup.
- Otherwise the credential is a user access JWT. Validation includes user lookup, project binding, and active seed-wrap verification.
- General protected routes use JWT authentication, not API-key fallback.
- An API key changes usage attribution and quota selection. Keep `AuthMethod` intact through inference and usage code.
- Never accept project or user identity from a request body when middleware has already established it.

API keys are created and managed through authenticated protected routes. Do not expose stored hashes, reconstruct a key, or add provider credentials to the client API.

## Handle errors without breaking SDK recovery

Pre-stream failures use the ordinary HTTP status and a plaintext JSON body shaped as `{"status": number, "message": string}`. Successful encrypted responses remain encrypted. Derive the current mapping from `ApiError::into_response` in `src/main.rs`.

Preserve these distinctions:

- 400: malformed request, encrypted payload, or stale/unknown session.
- 401: invalid user/API authentication.
- 403: entitlement or usage denial.
- 404/409/413/422/429: their specific resource or validation conditions.
- 503: a temporarily unavailable required upstream.
- 500: an internal or non-recoverable provider failure.

Do not casually change 400 or 401 behavior. Released clients may perform one re-attestation retry after a 400 and one auth refresh after a 401. Validate all provider-free input before writes so a replay cannot duplicate side effects. For a new side-effecting route, prefer an explicit idempotency mechanism or a distinct stale-session signal over relying on ambiguous retry behavior.

Once an SSE response has started, send a typed encrypted error event. Never
insert plaintext `data: Error...`, `data: encryption_failed`, or another
non-terminal plaintext payload: strict SDK transports reject it as corrupt
rather than exposing unauthenticated inference output.

## Work with OpenAI-shaped inference routes

Derive the exact inference routes and payload structs from `src/web/openai.rs`:

```sh
rg -n -C 3 'pub fn router|struct TTSRequest|struct TranscriptionRequest|struct EmbeddingRequest' src/web/openai.rs
```

Keep these non-obvious contracts in mind:

- Chat completions intentionally accept an extensible JSON object, but require a model. Provider-owned request fields are rewritten at the backend boundary.
- Non-streaming chat returns encrypted completion JSON. Streaming chat preserves upstream JSON chunks, canonicalizes the public model ID, encrypts each data frame, and terminates once.
- `/v1/models` and `/v1/models/catalog` are backend-owned public views. They are not raw provider discovery results.
- Speech accepts a JSON payload, validates and supplies backend-owned model/voice defaults, and returns audio through an encrypted payload containing base64 bytes and content type. SDK custom-fetch adapters may present those bytes as an ordinary audio response to Maple.
- Transcription accepts a JSON request whose `file` is base64 audio. It is not the ordinary public OpenAI multipart contract, even though the backend builds multipart at the provider boundary.
- Embeddings accept a string or array input and return encrypted provider-compatible JSON.

For any changed request or response, verify status, headers, exact body shape, streaming/non-streaming behavior, size limits, cancellation, and usage. Do not infer compatibility from the route name alone.

## Work with Responses and stored conversations

The Responses surface is deliberately OpenAI-shaped but not a complete OpenAI implementation. Re-read `src/web/responses/handlers.rs`, `constants.rs`, `events.rs`, `context_builder.rs`, and the conversation/instruction/project modules before changing it.

Current behavior that must be preserved or changed deliberately:

- `conversation` is required and accepts a UUID string or `{id: UUID}`.
- `input` accepts a string or message array, but the current persistence path uses the first normalized message as the new user message. Do not claim arbitrary multi-message semantics without implementing and testing them.
- Creation always returns SSE. The `stream` field is recorded/logged but does not currently select a non-streaming response.
- The request and generated items are persisted even when `store` is false; the flag is stored on the response record. Do not describe it as a no-storage guarantee.
- User-visible web search is requested as `{"type":"web_search"}`. The backend maps it to provider function tools and forces one tool call at a time.
- Tool-loop exhaustion disables tools and lets the model finish with accumulated context; retain the internal untrusted-web safety prompt.

Keep the persistence phases ordered:

1. Validate authentication, ownership, payload features, and model access.
2. Derive the user encryption key, normalize content, count tokens, and enforce context/entitlement limits.
3. Build context without writes.
4. Persist the response and user message only after validation succeeds.
5. Persist assistant, reasoning, tool-call, and tool-output items in emitted order as the stream advances.
6. Continue the storage/orchestration work when the client disconnects unless an explicit cancellation is recorded.

Conversation content, attachment text, instructions, response metadata, and tool data that can contain user content must remain encrypted at rest with the user's derived key. Ownership checks precede decryption and mutation. Cancellation only applies to an active response; deletion must preserve the intended cascade and ownership rules.

Derive event names and sequence behavior from source:

```sh
rg -n 'EVENT_|STATUS_' src/web/responses/constants.rs
rg -n 'ResponseEvent|sequence_number|responses_sse_response' src/web/responses/events.rs src/web/responses/handlers.rs
```

Keep the SSE `event:` name and decrypted JSON `type` consistent. Sequence numbers advance only under the policy encoded in `ResponseEvent`; cancellation/error behavior is intentionally different from ordinary events. Preserve `Cache-Control: no-cache`, disabled proxy buffering, keepalives, terminal completion/cancellation/error, and durable item ordering.

## Check every client boundary

Locate consumers rather than assuming a sibling layout:

```sh
rg -n 'VITE_OPEN_SECRET_API_URL|aiCustomFetch|/v1/chat/completions|/v1/models|/v1/audio/speech' ../maple frontend 2>/dev/null
rg -n '@opensecret/react|opensecret = |name = "opensecret"' ../maple/frontend/package.json ../maple/frontend/src-tauri/Cargo.toml ../maple/frontend/src-tauri/Cargo.lock 2>/dev/null
```

At minimum, inspect these Maple seams when present:

- Browser Research chat constructs an OpenAI client around
  `useOpenSecret().aiCustomFetch` and sends stored streaming requests through
  `/v1/responses` plus the Conversations API.
- Browser TTS also uses `aiCustomFetch` and expects the SDK adapter's audio response behavior.
- Native Agent Mode separately sends caller-owned chat-completion bytes through
  the pinned Rust SDK inference transport to `/v1/chat/completions`; it does
  not internally use Responses or Maple's local proxy.
- Maple's local OpenAI-compatible proxy exposes models/chat to local tools and ultimately depends on the same backend contract.

A semantic field intended for both Research chat and Agent Mode is not one
shared wire-field change. Model it in the Responses request/context/provider
turn and independently in the chat-completions/Goose request path, then prove
each pinned client transport.

For a Responses field, trace `ResponsesCreateRequest`,
`build_model_turn_request`, and `persist_request_data` in
`src/web/responses/handlers.rs`, plus `src/models/responses.rs` and the generated
Responses tables in `src/models/schema.rs`. For a chat/provider or usage field,
trace the shared handoff and usage accumulator/publication in
`src/web/openai.rs`, `UsageEvent` in `src/sqs.rs`, and the local token-usage
model/schema. This reveals whether the change is transport-only, persisted,
metered, or all three.

Use the versions pinned by Maple, not a convenient SDK checkout. For a changed route:

- Update SDK request/response types and typed methods where applicable.
- If the Rust lossless inference transport must expose a new route, update its explicit method/path allowlist and header/body/SSE tests.
- Update the React/JavaScript SDK custom-fetch transformation if content type or envelope behavior changes.
- Update Maple call sites, mocks, fixtures, and user-facing error mapping.
- Test both old-client/new-server and new-client/old-server behavior for the
  pinned TypeScript Research path and Rust Agent path. Do not rely on unknown
  fields being rejected, ignored, or preserved unless tests prove that exact
  behavior. Prefer a server-first rollout for additive OpenSecret fields; use
  an explicit capability/version gate when either direction cannot interoperate.
  Do not silently require an unreleased SDK.

## Validate the change

Run targeted tests while iterating, then the repository validation skill. Useful discovery and targeted commands include:

```sh
rg -n '#\[cfg\(test\)\]|#\[tokio::test\]|#\[test\]' src/web src/jwt.rs
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::openai::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::responses::handlers::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::encryption_middleware::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check
git diff --check
```

Use a single Cargo test filter per command. Run the full locked all-feature test and Clippy gates before handoff.

For smoke testing, plain `curl` is appropriate for `/health-check` and `/health-check-extended` only. Exercise encrypted routes through a released/pinned SDK and test both JWT and API-key authentication when the change affects the OpenAI router. Use a disposable local database, run Diesel schema migrations first, and point Maple at the chosen local/dev/prod OpenSecret URL. Billing and flag services are external API dependencies; configure their URLs/keys only when the scenario needs their entitlement or flag behavior, and treat them solely as HTTP API boundaries.

Report exactly what was exercised: unit contract, local encrypted SDK call, provider-backed stream, Maple browser path, Maple native path, or GUI smoke. A backend test alone is not end-to-end proof.
