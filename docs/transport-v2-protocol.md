# OpenSecret Transport Protocol v2

Status: protocol contract under additive stacked implementation

Transport v2 carries a complete logical OpenSecret request inside one
attested, encrypted request record. It binds authentication to that encrypted
session, protects bodyless and bodyful operations equally, and rejects replay
before application side effects.

This is an additive network protocol. Existing routes, session behavior,
authentication middleware, response formats, and streaming behavior remain the
transport-v1 contract. A transport-v2 client never falls back to transport v1.

## 1. Goals and non-goals

Transport v2 provides these properties:

- The untrusted parent and network can see only a coarse v2 endpoint, an opaque
  session identifier, ciphertext sizes, timing, and connection metadata.
- A credential accepted for an OpenSecret operation is established through the
  same attested encrypted session that carries that operation.
- A credential cannot be paired with an attacker-controlled encryption session.
- Method, logical path, query, admitted headers, and body are authenticated as
  one request.
- Each accepted request identifier is single-use for the lifetime of its
  session key epoch.
- Every response record is encrypted by the exact session context that admitted
  its request and is bound to that request identifier.
- Streaming records are ordered and end with an authenticated terminal record.
- Existing application APIs and SDK public methods can retain their current
  shapes even though their outer network transport changes.

The first implementation deliberately does not add:

- additional transport families, TLS termination inside the enclave, or
  WebSockets;
- general application-level idempotency or transparent retry of uncertain
  mutations;
- raw API-key rotation;
- database rollback protection;
- refresh-token revocation; or
- a transport-v1 compatibility mode inside new transport-v2 SDK releases.

Those are independent features. They must not be implied by a transport-v2
success response.

## 2. Version boundary and rollout

The server keeps transport v1 and v2 as separate stacks:

- Transport-v1 routes keep their current handlers, middleware order, session
  cache, keys, ciphertexts, errors, and SSE framing.
- Transport v2 has separate pending-attestation and session caches. Transport-v2
  traffic cannot evict or acquire a transport-v1 session.
- A session identifier is valid in exactly one transport version.
- The next major JavaScript and Rust SDK releases speak transport v2 only while
  preserving their public application APIs.
- Already-published SDKs continue to use transport v1.
- A new client treats an old server as unsupported. A `404`, redirect, plaintext
  error, decryption failure, or malformed v2 response never triggers fallback.

The rollout unit is the SDK release, not a runtime feature flag. Deploy all
additive server support before publishing clients that require v2. Rolling a
client back means installing the previous SDK/application release; it does not
mean negotiating down on a live connection.

Server support may be merged in independently reviewable layers before any v2
client is published. In particular, the isolated gateway may exist while its
logical-operation allowlist is empty; such requests receive an authenticated
encrypted logical `404` and cannot reach application handlers.

The term **transport v2** is intentionally distinct from the existing
`USER_TOKEN_FORMAT_V2`, which describes first-party JWT claims rather than the
network protocol.

## 3. Outer HTTP surface

Transport v2 initially exposes three coarse endpoints:

```text
GET  /v2/attestation/:nonce
POST /v2/key_exchange
POST /v2/request
```

The v2 attestation endpoint has the same Nitro attestation purpose as the
existing endpoint but stores its one-shot ephemeral key in the separate v2
pending-attestation cache.

`POST /v2/request` carries unary and streaming operations. Its meaningful outer
inputs are:

```text
x-session-id: <opaque UUID>
content-type: application/octet-stream

<raw nonce-prefixed ChaCha20-Poly1305 request record>
```

It does not accept outer authorization, cookies, logical query parameters, or
application/provider headers. Intermediaries may add ordinary transport
headers, but those headers cannot affect application authorization or dispatch.

A unary outer success is one raw AEAD record with
`content-type: application/octet-stream`; an active stream remains
`text/event-stream` with one base64 record per SSE frame. The SDK treats
logical status, headers, errors, and stream termination as valid only after
authenticating the encrypted response record. Non-success outer status and any
unexpected content type are unauthenticated transport failures and never cause
automatic request replay.

## 4. Attested key exchange

The client generates:

- a fresh attestation nonce; and
- a fresh X25519 ephemeral key pair.

The client verifies the Nitro attestation document, including its exact
requested nonce, attested enclave ephemeral X25519 public key, signature chain,
current certificate validity, mandatory positive document timestamp, and
approved PCR policy before continuing. Handshake freshness comes from the
unpredictable client challenge signed into the document and the enclave's
one-shot, five-minute pending-key entry; the document timestamp is not an
independent client-clock age window.

The client sends its X25519 public key and the attestation nonce to
`POST /v2/key_exchange`. The enclave consumes the corresponding one-shot
pending key and rejects a non-contributory X25519 result.

The enclave generates a random 32-byte session master and a random session
UUID. X25519 is used only to protect this fresh session master; it is not used
directly as the request or response key.

The key-exchange wrapping key is:

```text
handshake_key = HKDF-SHA256(
  input_key_material = x25519_shared_secret,
  salt = empty,
  info = UTF8("opensecret/transport-v2/handshake-key"),
  length = 32
)
```

The encrypted key payload is a fixed 57-byte binary value:

```text
version_u8
|| session_uuid_bytes[16]
|| session_master[32]
|| expires_at_unix_seconds_u64_be
```

The version byte is exactly `2`. The UUID uses its RFC 9562 network-order
bytes. The expiry is an unsigned Unix timestamp in whole seconds and is a
client renewal hint; the enclave enforces the 65-minute lifetime with a
monotonic clock. The outer and encrypted session UUIDs must match. The record
uses ChaCha20-Poly1305 and this AAD:

```text
UTF8("opensecret/transport-v2/key-exchange")
```

This domain-separates v2 from the legacy direct-shared-secret wrapping format.
The exact golden bytes are frozen by the cross-language test vectors before the
endpoint is registered.

## 5. Session key schedule

Request and response keys are derived from the random session master:

```text
request_key = HKDF-SHA256(
  input_key_material = session_master,
  salt = empty,
  info = UTF8("opensecret/transport-v2/client-request"),
  length = 32
)

response_key = HKDF-SHA256(
  input_key_material = session_master,
  salt = empty,
  info = UTF8("opensecret/transport-v2/enclave-response"),
  length = 32
)
```

The enclave retains the directional keys, not the session master. Secret key
material is zeroized when the session is removed.

Each encrypted record is:

```text
random_nonce[12] || chacha20_poly1305_ciphertext_and_tag
```

The 96-bit nonce is generated independently for every record. A record shorter
than 28 bytes is invalid. Standard base64 is canonical and padded; decoders
reject alternate textual encodings.

Request-record AAD is:

```text
UTF8("opensecret/transport-v2/request-record")
|| 0x00
|| session_uuid_bytes[16]
```

Unary-response AAD is:

```text
UTF8("opensecret/transport-v2/unary-response-record")
|| 0x00
|| session_uuid_bytes[16]
|| request_id[16]
```

Streaming-response AAD is:

```text
UTF8("opensecret/transport-v2/stream-response-record")
|| 0x00
|| session_uuid_bytes[16]
|| request_id[16]
|| sequence_u64_be
```

The session UUID is serialized as its 16 RFC 9562 network-order bytes, never as
text. The request identifier and sequence are known to the client before it
opens the corresponding response record.

## 6. Complete encrypted request

The decrypted request is a strict JSON envelope:

```json
{
  "version": 2,
  "request_id": "00112233445566778899aabbccddeeff",
  "response_mode": "stream",
  "credential": null,
  "cache_namespace_root_base64": null,
  "request": {
    "method": "POST",
    "path": "/v1/responses",
    "query": null,
    "headers": [
      {
        "name": "content-type",
        "value_base64": "YXBwbGljYXRpb24vanNvbg=="
      }
    ],
    "body_base64": "eyJtb2RlbCI6Ii4uLiJ9"
  }
}
```

Fields have these meanings:

- `version` is exactly the integer `2`.
- `request_id` is lowercase hexadecimal for exactly 16 bytes generated by the
  operating-system CSPRNG. It is not a UUIDv4 because UUIDv4 exposes only 122
  random bits.
- `response_mode` is encoded as `unary`, `stream`, or `auto`, but the initial
  server accepts only the two explicit modes. `auto` is reserved for a future
  protocol revision and is rejected. The explicit mode must agree with both
  the selected route and the logical application request: Chat Completions uses
  `stream` only when its top-level JSON `stream` field is the literal Boolean
  `true`, and uses `unary` when that field is absent or Boolean `false`.
  Duplicate or non-Boolean `stream` fields are invalid. Responses create always
  uses `stream`, even when its logical body contains `"stream": false`, because
  the current OpenSecret Responses application contract always returns SSE.
  Every other initially projected operation uses `unary`.
- `credential` is normally `null`. It is used only for a permitted anonymous
  authentication transition. The initial strict variants are
  `{"kind":"api_key","value_base64":"..."}` and
  `{"kind":"resumption","value_base64":"..."}`. The value is the exact
  credential bytes in canonical padded standard base64. Password, registration,
  OAuth, and recovery credentials remain part of their logical operation body;
  they do not become generic transport credentials.
- `cache_namespace_root_base64` is a required-nullable field. When non-null it
  is canonical padded standard base64 for exactly 32 client-generated random
  bytes. The envelope codec accepts either shape, while application admission
  permits a non-null value only on the request that transitions an anonymous
  session to a user or API-key authority. It is `null` on platform transitions
  and every request made after binding.
- `method` is a supported uppercase HTTP method.
- `path` is one origin-relative application path with no query or fragment.
- `query` is either `null` or the exact query string without a leading `?`.
- `headers` preserves supported duplicate fields in order. Header names are
  lowercase ASCII and values are exact bytes encoded as canonical base64.
- `body_base64: null` means no request body. An empty string means an explicitly
  present empty body. Other values preserve exact request bytes without
  transport-layer JSON parsing or reserialization.

Unknown and duplicate JSON fields are rejected. The implementation validates
the path and query once, rejects schemes, authorities, fragments, backslashes,
dot-segments, encoded separator ambiguity, and invalid percent encoding, then
maps `(method, path)` through an exact application allowlist.

Registered routes may define an opaque dynamic segment without relaxing those
rules for any other path. Route selection first matches an exact literal prefix
and method. The initial exception is the final segment of
`GET|PUT|DELETE /protected/kv/<key>`: ASCII alphanumeric key bytes remain
literal and every other UTF-8 byte is one uppercase `%HH` triplet, matching the
released Rust SDK's `NON_ALPHANUMERIC` encoding. The enclave rejects empty,
noncanonical, malformed, over-encoded, or invalid-UTF-8 segments and decodes the
accepted value exactly once after selecting the KV route. It performs no
Unicode normalization. Thus `/`, a literal `%2F`, dot-only keys, backslashes,
and distinct Unicode byte strings remain application data rather than routing
syntax. The future TypeScript v2 SDK must implement the same UTF-8 byte codec;
generic URI encoders that leave punctuation unescaped are insufficient.

`DELETE /protected/api-keys/<name>` reuses that exact opaque-segment codec and
then applies the existing API-key name contract after decoding: 1–50 ASCII
alphanumeric, space, hyphen, or underscore characters, with no edge spaces.
For example, `Production Key-1_test` has the single v2 spelling
`Production%20Key%2D1%5Ftest`. Other encodings of the same name are rejected;
transport-v1 path handling is unchanged.

The gateway never hands an arbitrary URI to the full transport-v1 router.
Application operations are projected deliberately onto shared application
services/handlers.

The encrypted-header policy is route scoped. Typed application operations use
explicit per-operation allowlists. Raw inference operations preserve the
current SDK contract by authenticating and admitting arbitrary valid
end-to-end extension headers and duplicates, but apply a strict denylist.
Chat Completions and Responses forward those admitted headers through the
existing provider-header path; models, catalog, audio, embeddings, and web
operations authenticate but discard extension headers that their v1 handlers
also ignore. Encrypted inference query strings are likewise authenticated and
accepted for compatibility but ignored by the current application operations.
Every policy rejects at least:

- `authorization`, `cookie`, `set-cookie`, `host`, and `x-session-id`;
- connection, proxy, transfer, upgrade, and other hop-by-hop/framing headers;
- client-supplied provider credentials; and
- unsafe fields selected by a `connection` header and any field the existing
  raw-inference contract already strips.

Body-bearing inference routes require one unambiguous JSON content type. GET
inference routes allow no content type or one canonical JSON content type.

The implementation PR owns the exact typed allowlists and raw-inference
denylist. It tests them against the current Rust and JavaScript raw-request
contracts, including safe extension headers such as `x-provider-beta`. Outer
headers are never forwarded into application or provider logic.

## 7. Request admission and replay

The enclave processes every v2 request in this order:

1. Read the bounded outer body.
2. Parse the outer session UUID and acquire one lease from the v2 cache.
3. Reject a session past absolute expiry, bound-authentication expiry, or record
   budget. Authentication expiry closes the bound session fail closed.
4. Decode and decrypt with that lease's request key and request AAD.
5. Strictly parse and structurally validate the envelope.
6. Validate method, path, query, headers, body presence, credential use, and
   response mode for the selected application operation.
7. Reserve response capacity before dispatch: one record for unary, or start
   and terminal records atomically for streaming.
8. Atomically claim the decoded 16-byte request identifier.
9. Only then authenticate, dispatch, mutate state, bill, send email, or call a
   provider.

Unused pre-dispatch response reservations are released. Once encryption of a
reserved record is attempted, its slot remains charged even if encryption
fails. Later stream chunks charge capacity individually while the terminal
reservation remains unavailable to them.

After a request has passed route and mode classification as streaming, the
gateway reserves the start and terminal slots together before the replay gate
or application dispatch. Any later authenticated failure before `Start` is
emitted returns as an encrypted unary response, not as a partial SSE stream.
The gateway converts the unstarted two-slot reservation into a one-slot unary
reservation atomically: it retains one already-reserved slot and releases the
other without a release-and-reacquire race. This includes
provider-working-set admission, replay, authentication transition, and
application setup failures. If the two-slot reservation cannot be made but one
response slot remains, the gateway uses that final slot for an authenticated
unary `session_exhausted` error. Earlier route or mode rejection uses the
ordinary authenticated unary-error path because no stream reservation exists.
Once `Start` is emitted, section 11 owns every remaining outcome.

Replay rules are:

- Identifiers are scoped to one v2 session/key epoch.
- Distinct requests may arrive and complete in any order.
- The enclave stores exact identifiers, not a highest sequence counter or a
  packet-order assumption.
- An identifier is never evicted while its session keys can admit requests.
- Two concurrent requests with one identifier have exactly one winner at the
  replay gate.
- A duplicate returns an encrypted `replay_detected` error. It does not return a
  cached result and does not reveal whether the first execution completed.
- If the per-session or global replay budget is exhausted, the session becomes
  exhausted and admits no more application requests. The client establishes a
  new attested session for later requests.
- Malformed, undecryptable, structurally invalid, and replayed records do not
  refresh idle state and never dispatch.

The first release uses a 65-minute absolute key lifetime. A lease may allow an
already-admitted response stream to finish, but it does not allow new requests
after absolute expiry. Exact session and replay capacities are chosen and
memory-accounted in the v2-core implementation PR; they are not borrowed from
the transport-v1 cache budget.

Replay detection is not application idempotency. Once request bytes may have
reached the enclave, a transport failure is an uncertain outcome. The v2 SDK
does not transparently resend that application request under either the same or
a new session. A caller may retry only according to an operation's explicit
idempotency contract.

## 8. Authentication and immutable session binding

A v2 session has one of these authority states:

```text
Anonymous
  -> Authenticating(request_id)
  -> Bound(User | Platform | ApiKey)
  -> Closing
```

The transition to `Authenticating` is atomic and happens before asynchronous
credential verification or authentication side effects. Failure and
cancellation return the session to `Anonymous`. Exactly one successful
transition can bind a session.

Once bound:

- the principal kind and identity are immutable;
- account switching requires a new attested session;
- login, registration, resumption, and API-key rebinding are rejected;
- protected authorization comes from the bound session, never an outer bearer;
  and
- a successful terminal operation moves the exact session to `Closing` before
  encrypting its final response through the already-admitted lease. Closing
  fences new work but does not invalidate the held response key or reservation;
  secret state is retired after active leases drop and cache cleanup detaches
  it.

The SDK owns each v2 session within exactly one authentication/client context;
it never reuses the current origin-and-PCR-only global cache across users,
platform users, API keys, or anonymous clients. User/account changes and
`set_api_key` or `clear_api_key` abandon the affected bound session and establish
a fresh one. Distinct API keys can have concurrent sessions against one origin
without sharing authority.

A password change never replaces a bound session's credential-derived
`AuthContext` in place. It prepares a fresh access descriptor and resumption
credential for the replacement `AuthContext`, commits the password and seed
wrap atomically, and terminally closes the old session. The client establishes
a fresh attested session before using the replacement credentials.

### 8.1 User sessions

A user binding retains immutable user/project identity, the current
credential-bound `AuthContext`, and authentication expiry. Authorization keeps
the existing active seed-wrap and live user/project checks so password changes,
destructive resets, and account deletion invalidate old authority.

Fresh password, registration, and OAuth success bind the anonymous session that
carried the operation. Resumption binds a new anonymous v2 session only after
validating a transport-v2 resumption credential inside ciphertext.

### 8.2 Platform sessions

A platform binding retains platform-user identity and authentication expiry.
Organization membership and Owner/Admin authority remain live database checks;
they are not cached in the transport session.

### 8.3 API-key sessions

The existing raw API key is sent once inside the encrypted initial credential
transition. It is not rotated. A successful transition binds a distinct
inference-only API-key principal and the triggering request may proceed.

The binding retains only the database key identity, owning user identity, and
derived provider-cache namespace. It rechecks that exact key-to-owner
relationship before every operation so deletion or ownership changes remain
immediately effective. API-key sessions can reach only model catalog,
chat completions (unary or streaming), speech synthesis, audio transcription,
and embeddings. They never become general user sessions.

### 8.4 Provider cache namespace binding

User and API-key authentication transitions carry a client-generated cache
namespace root in `cache_namespace_root_base64`. The client persists this root
inside the corresponding local authentication/API-key context so a newly
attested session can recover provider cache hits after application or enclave
restart. Losing or deliberately replacing the root only creates a fresh cache
namespace; it does not change account authority or data access.

The enclave derives the namespace only after it has verified the user UUID (or
the owning user UUID for an API key):

```text
HMAC-SHA256(
  cache_namespace_root[32],
  UTF8("opensecret/provider-cache/tinfoil/user-cache-namespace/v1")
  || 0x00
  || verified_user_uuid_bytes[16]
)
```

UUID bytes are the 16 RFC 9562 network-order bytes, not UUID text. The fixed,
versioned label domain-separates this value from every other use of the client
root. Tinfoil receives the 32-byte result as 64 lowercase hexadecimal
characters in its provider-managed `user_cache_secret` field. OpenSecret never
logs or intentionally includes the client root, derived bytes, or encoded
provider secret in an application response. Provider responses remain
untrusted and could echo arbitrary request fields; any such echo remains inside
the response encrypted to the same authenticated client and is never accepted
as authority.

The bound user or API-key authority retains only an `Arc`-backed derived
namespace. Clones share that allocation and its bytes are zeroized when the
last clone drops. The input root is zeroized when the transition request is
released and is never retained in session state. Platform authorities retain
neither value. This makes a user cache namespace stable for a cooperating
client while preventing the parent/operator, which sees neither the root
plaintext nor the enclave-held derivation result, from computing it from a
UUID.

### 8.5 Session-bound user OAuth

Transport v2 admits exactly seven user OAuth operations:

| Logical operation | Authority before request | Cache namespace root | Success effect |
| --- | --- | --- | --- |
| `POST /auth/github` | anonymous | null | retain anonymous session |
| `POST /auth/google` | anonymous | null | retain anonymous session |
| `POST /auth/apple` | anonymous | null | retain anonymous session |
| `POST /auth/github/callback` | anonymous | required | bind verified user |
| `POST /auth/google/callback` | anonymous | required | bind verified user |
| `POST /auth/apple/callback` | anonymous | required | bind verified user |
| `POST /auth/apple/native` | anonymous | required | bind verified user |

Every operation is unary, carries one non-empty JSON body, and rejects logical
query fields, generic credentials, or additional logical headers. Initiation
accepts the existing `{client_id}` shape; existing ignored SDK compatibility
fields remain tolerated by the application decoder. Callback and native
success use the same transport-v2 token issuance and immutable user-binding
path as password login. V2 applies bounded JSON-shape and credential-field
limits before decoding state or calling an external provider, and callback or
native operations reserve provider-output working-set capacity before replay
claim or dispatch.

The public OAuth `state` encoding remains compatible. Its server-side entry is
additionally tagged as either `LegacyV1` or `TransportV2(session_id)`. A v2
callback must match both the stored state and the exact attested session that
initiated the redirect. A v1 callback, or a callback on another v2 session,
fails without consuming the legitimate continuation. Successful matching is
still atomic and one-time.

The gateway reserves the anonymous session's authentication transition before
callback state consumption, provider exchange, or database work. A failed
callback releases that reservation back to anonymous. A verified callback
commits the user authority only after its complete logical success response has
been constructed. The gateway then encrypts it through the same held session
lease that authenticated the callback request; encryption failure closes the
newly bound session through the normal gateway rule.

OAuth transport keys are never moved between clients. If the initiating v2
session is lost or expires while the browser is away, the client establishes a
new anonymous session and restarts OAuth. It cannot replay the old callback on
the replacement session. Transport-v1 state storage, callbacks, token issuance,
and encrypted-body routes retain their existing behavior during migration.

### 8.6 Unary inference projection

The initial unary-inference layer admits exactly these logical operations:

| Logical operation | Bound user | API-key session | Anonymous |
| --- | :---: | :---: | :---: |
| `GET /v1/models` | yes | yes | yes |
| `GET /v1/models/catalog` | yes | yes | no |
| `POST /v1/chat/completions` with streaming absent or false | yes | yes | no |
| `POST /v1/audio/speech` | yes | yes | no |
| `POST /v1/audio/transcriptions` | yes | yes | no |
| `POST /v1/embeddings` | yes | yes | no |
| `POST /v1/web/search` | yes | no | no |
| `POST /v1/web/extract` | yes | no | no |

`GET /v1/models` remains public when no credential is present. When a caller
explicitly supplies an API key, it may also perform the one-time encrypted
API-key transition so the released SDK/proxy behavior still rejects an invalid
key rather than silently treating it as anonymous. The other five API-key
operations may perform the same transition; web operations require a bound user. Platform
authority is rejected for every operation in this layer. Chat streaming and
the Responses create API are projected by the authenticated-streaming contract
in section 11; this table intentionally describes only the unary portion of the
inference surface.

Every admitted unary provider operation reserves 128 MiB of aggregate provider
output working set before claiming replay identity or dispatching. V2 caps
provider JSON at 28 MiB and structurally preflights it before deserialization.
Speech synthesis uses a smaller raw-byte ceiling so base64 plus logical JSON
still fits the 28 MiB response bound. Audio transcription additionally caps
retained chunks at four and 64 MiB aggregate, caps each multipart provider
request at 32 MiB, divides both raw response bytes and JSON structural-token
allowance across the admitted chunks, processes chunks sequentially, and does
not retry an ambiguous provider attempt. V2 provider failures log only provider
identity and status while draining a small bounded prefix; provider-controlled
error bodies never enter enclave logs. These are v2-only admission rules;
transport v1 retains its existing limits, concurrency, retry behavior, error
logging, and provider-cache derivation.

### 8.7 Password recovery and platform authentication lifecycle

Transport v2 completes the anonymous password-recovery paths needed by a
v2-only SDK and establishes a distinct platform authority before projecting
the larger organization control plane:

| Logical operation | Authority before request | Generic credential | Success effect |
| --- | --- | --- | --- |
| `POST /password-reset/request` | anonymous | none | retain anonymous |
| `POST /password-reset/confirm` | anonymous | none | retain anonymous |
| `POST /platform/login` | anonymous | none | bind platform user |
| `POST /platform/register` | anonymous | none | bind platform user |
| `POST /platform/refresh` | anonymous | platform resumption | bind platform user |
| `GET /platform/verify-email/{code}` | anonymous or same-kind platform binding | none | retain authority |
| `POST /platform/password-reset/request` | anonymous | none | retain anonymous |
| `POST /platform/password-reset/confirm` | anonymous | none | retain anonymous |
| `POST /platform/logout` | bound platform user | none | close exact session |
| `POST /platform/request_verification` | bound platform user | none | retain binding |
| `POST /platform/change-password` | bound platform user | none | close exact session |

Except for refresh, bodyful operations retain their existing logical JSON
shapes. Platform refresh carries no logical body or header: its v2-only
resumption secret is the envelope credential. Every platform transition
requires `cache_namespace_root_base64: null`; a platform session never obtains
a provider-cache namespace.

Platform access-descriptor and resumption tokens use distinct platform
audiences and `pk = platform`. Legacy platform JWTs, user-v2 credentials,
access descriptors, wrong-signature tokens, and wrong-purpose tokens cannot
resume a platform session. Successful login, registration, or resumption
constructs the complete logical response before atomically committing
`Bound(Platform { platform_user_id })`; the gateway then encrypts that response
through the same held session lease. The binding's authentication deadline is
the access-descriptor expiry capped by the session's absolute expiry.

Every bound platform account operation reloads the exact platform user. A
deleted platform user produces an authenticated unauthorized response and
closes the session; transient database failure retains the otherwise valid
session. Logout intentionally preserves current behavior: it closes this
transport session but does not globally revoke resumption credentials. A
successful password change also closes this session. Global credential epochs
and all-device logout remain separate application-authentication work, not an
implicit transport promise.

The following stack layer projects `/platform/me` and the complete
organization, project, membership, invite, secret, and settings control plane.
Those operations require live role checks and v2-only bounded database output;
they are deliberately separated from principal establishment.

### 8.8 Platform resource control

Every platform resource request requires the immutable platform authority on
the admitting v2 session. The logical request carries no credential, cache
namespace, or query. GET and DELETE operations, plus invite acceptance, carry
no logical headers or body. POST, PATCH, and PUT resource operations carry one
nonempty JSON body and exactly one logical JSON content-type header. The
gateway claims the request ID before any read or mutation and encrypts the
logical result through the same held session lease.

Transport v2 projects the complete currently implemented control plane:

| Resource | Read authority | Mutation authority |
| --- | --- | --- |
| `/platform/me` and `/platform/orgs` | current platform user; organization lists contain only memberships | any platform user may create an organization and becomes Owner; only Owner deletes it |
| organization projects | any organization member | Owner or Admin creates, updates, and deletes |
| project secret metadata | any organization member; values are never returned or loaded by a metadata read | Owner or Admin upserts and deletes |
| project email and OAuth settings | any organization member | Owner or Admin updates |
| organization memberships | any organization member | Owner changes roles or removes members; the final Owner cannot be demoted or removed |
| organization invites | Owner or Admin lists and deletes; the exact recipient may read its invite | Owner or Admin creates non-Owner invites; only Owner creates an Owner invite |
| `/platform/accept_invite/{code}` | exact bound recipient | unused, unexpired invite is consumed atomically with membership creation |

Every operation rechecks that the platform-user row still exists. Organization
roles are never cached in session state: reads recheck live membership, and a
write checks authority in the same transaction as its mutation. Project paths
also prove that the selected project belongs to the selected organization.
Membership changes serialize the actor, target, and final-Owner invariant.
Invite acceptance locks and revalidates the invite before consuming it.

V2 closes two demonstrated invitation elevation paths without changing the
transport-v1 contract. An Admin cannot issue an Owner invite. Accepting an
Owner invite additionally requires a currently verified platform email, so an
account registered under a visible pending recipient address cannot claim
ownership before proving mailbox control. During v1 coexistence, an authorized
legacy Admin can still use the unchanged v1 invitation behavior; global closure
therefore requires eventual v1 platform-control retirement or a separately
approved v1 security correction.

Database-controlled platform output uses a v2-only bounded storage projection.
Reads measure length and count in a repeatable-read snapshot before loading
variable strings or JSON, use narrow columns, and fail with encrypted 413
rather than truncating. Collections have a 65,536-row sentinel and must also
fit the 28 MiB logical-response ceiling. Secret listing and deletion never load
the encrypted secret value. Operations whose logical success depends on stored
variable data obtain the larger stored-output working-set reservation before
the replay claim and database dispatch.

Response status and JSON shape otherwise remain the current application
contract: every success is 200, delete operations retain their message object,
secret creation remains an upsert, absent email settings remain 404, absent
OAuth settings return the disabled default, and the SDK-only push-settings
stubs are not treated as backend routes. V2 also applies the already-declared
project-update and invite-request validators that their v1 handlers omit; this
prevents a v2 write from creating stored values outside the bounded v2 read
contract while leaving the v1 handlers unchanged.

## 9. First-party and third-party tokens

Transport v2 removes first-party JWTs from steady-state OpenSecret request
authorization. It can temporarily preserve the SDK's current token-shaped
login result for Maple compatibility:

- `access_token` is a signed, v2-only compatibility/session descriptor with a
  distinct audience that transport-v1 middleware rejects. The v2 SDK never
  sends it as OpenSecret authorization.
- `refresh_token` is a v2-only resumption credential accepted only inside an
  encrypted v2 transition. Transport-v1 refresh rejects it.
- Existing v1 access and refresh JWTs cannot bootstrap a v2 session. Initial
  adoption therefore requires the agreed one-time fresh login.
- The returned fields remain encrypted and retain current public SDK response
  shapes while Maple's auth-state bridge is migrated separately later.

All transport-v2 access and resumption audiences are reserved internal
audiences. Third-party token issuance rejects them, just as it rejects existing
first-party audiences. Resumption validates the expected v2 audience, token
kind, principal kind and identity, project/authentication context where
applicable, issue/expiry times, and active credential state; an audience match
alone is never sufficient.

User and platform credentials occupy separate v2 domains. Platform credentials
carry no user project or seed-wrap authentication context; platform resumption
instead revalidates that exact platform-user row before binding. Neither kind
can be parsed as or substituted for the other.

These credentials remain bearer secrets if stolen from the client. Transport
v2 protects them from the parent/network threat; it does not claim to protect a
fully compromised same-origin application or device.

Third-party JWT issuance remains an ordinary bound-user application operation.
Its existing audience, project signing configuration, claims, expiry, and
portability for billing and other microservices remain unchanged. Third-party
JWTs never authenticate OpenSecret v2 and cannot bind a v2 session. Platform and
API-key principals cannot issue them.

## 10. Unary responses and errors

Once a valid session decrypts a request identifier, responses use a strict
encrypted unary envelope. This includes structural/pre-dispatch errors and
`replay_detected`; success and side-effecting application errors additionally
require the identifier to have passed the replay gate.

```json
{
  "version": 2,
  "request_id": "00112233445566778899aabbccddeeff",
  "status": 200,
  "headers": [
    {
      "name": "content-type",
      "value_base64": "YXBwbGljYXRpb24vanNvbg=="
    }
  ],
  "body_base64": "eyJvayI6dHJ1ZX0="
}
```

The response uses the exact leased crypto context that admitted the request. It
does not look up a session again from an outer UUID. The SDK verifies response
AAD and exact `request_id` before accepting status, headers, or body.

Response headers use a strict allowlist and exclude transport framing, cookies,
server internals, and credentials. Application errors preserve the established
`x-opensecret-error-contract` header and optional
`x-opensecret-error-code` header inside this authenticated logical envelope.

Failures before a valid session can be leased and the request identifier can be
recovered are generic bounded plaintext transport errors. They are untrusted
and terminal for that attempt. They cannot authorize automatic replay or
fallback.

## 11. Streaming responses

Transport v2 preserves the SDK's caller-visible streaming response, including
the application SSE used by Chat Completions and Responses. Response shape is
never selected implicitly:

| Logical operation | Required `response_mode` | Application result |
| --- | --- | --- |
| `POST /v1/chat/completions` with top-level `"stream": true` | `stream` | Chat Completions SSE |
| `POST /v1/chat/completions` with `stream` absent or `false` | `unary` | JSON |
| `POST /v1/responses` for every accepted request body | `stream` | Responses SSE |
| Every other initially projected operation | `unary` | Route-specific unary body |

Responses create remains streaming when its logical body contains
`"stream": false`; that preserves the existing OpenSecret application
contract. A missing Chat Completions `stream` field selects unary; a duplicate
or non-Boolean field is rejected. The `auto` mode is reserved and rejected.

### 11.1 Outer carrier and authenticated records

The outer HTTP streaming response is only an SSE carrier. It uses HTTP status
`200` and outer `content-type: text/event-stream`, but the client does not treat
that outer status or those headers as the logical response. Each outer SSE
event has exactly one `data` field and exactly this byte shape:

```text
data: <canonical padded standard-base64 encrypted record>\n\n
```

There is one ASCII space after the colon. The carrier contains no outer
`event`, `id`, or `retry` fields, no comments, no multiline `data` fields, and
no plaintext application bytes. Each field decodes to exactly one encrypted
`StreamRecord`; records are not combined in one event.

Decrypted records are one of these strict JSON shapes:

```json
{"version":2,"request_id":"00112233445566778899aabbccddeeff","sequence":0,"kind":"start","status":200,"headers":[{"name":"content-type","value_base64":"dGV4dC9ldmVudC1zdHJlYW0="}]}
{"version":2,"request_id":"00112233445566778899aabbccddeeff","sequence":1,"kind":"chunk","body_base64":"ZGF0YTogaGkKCg=="}
{"version":2,"request_id":"00112233445566778899aabbccddeeff","sequence":2,"kind":"end"}
{"version":2,"request_id":"00112233445566778899aabbccddeeff","sequence":2,"kind":"error","status":500,"body_base64":"eyJlcnJvciI6eyJjb2RlIjoic3RyZWFtX2ZhaWxlZCJ9fQ=="}
```

`End` and `Error` are alternatives at the same next sequence; a stream never
contains both. `Start` is always sequence zero, carries a logical 2xx status,
and authenticates the route's allowlisted logical headers, including
`content-type: text/event-stream`. `Error` carries a logical 4xx or 5xx status
and at most 16 KiB of sanitized decoded body bytes.

The gateway alone assigns sequence numbers, copies the admitted request ID into
every record, serializes each record, and encrypts it. Sequence starts at zero
and increments by exactly one for each encrypted record. Streaming-response AAD
binds the record to the directional protocol domain, exact session UUID,
16-byte request ID, and sequence as defined in section 5. Sequence also appears
inside the plaintext record. A client accepts a record only when its AAD,
request ID, and next expected sequence all match.

A `Chunk` contains arbitrary application bytes, not an application event. Its
decoded body is at most 64 KiB. When an application item is larger, the gateway
splits it across consecutive records without interpreting UTF-8, JSON, or SSE
boundaries. The SDK authenticates and concatenates chunk bodies in order before
its existing application-level parser consumes them. Thus a Responses `event:`
line or a Chat Completions `data:` line may cross transport records without
changing its caller-visible bytes.

### 11.2 Start, termination, and failure boundary

The gateway reserves both the `Start` and terminal record slots atomically
before the replay gate and dispatch. Failures before `Start` use the
authenticated unary response path and the reservation conversion described in
section 7. A client that requested streaming must therefore also accept a valid
encrypted unary error before any outer SSE carrier begins.

After a successful `Start`, application bytes become `Chunk` records, an
explicit application `Complete` becomes `End`, and an explicit application
failure becomes `Error`. Application-source EOF without `Complete` is a
failure, not successful termination; the gateway emits a sanitized `Error` when
the reserved terminal slot and AEAD state remain usable. Exactly one
authenticated `End` or `Error` is required for client success or an
authenticated stream error.

If AEAD encryption or its randomness fails while producing `Start`, no
authenticated carrier begins and the attempt ends as an untrusted transport
failure. If it fails for any later record, the already-started outer response
ends in abrupt EOF. The gateway does not emit plaintext fallback inside the
carrier or attempt another encrypted terminal after the cryptographic sequence
state has failed. EOF before a valid terminal, duplicate or out-of-order
sequence, malformed carrier framing, plaintext data, or an undecryptable record
is an unauthenticated transport failure at the client.

The exact session lease, request-bound stream reservation, and promoted
provider-output working-set permit remain held by the response body through
terminal delivery. Detached v2 producer work that can retain provider or
application output holds a clone of the same opaque permit; disconnecting the
network body cannot release admission capacity while that work still owns
retained output. Dropping the body stops delivery and releases its held lease
and any unused terminal reservation. It does not synthesize a terminal record
and does not implicitly cancel a persisted Responses operation. Responses
cancellation remains the explicit application endpoint; background
orchestration and storage retain their existing disconnect semantics. A Chat
producer observes its closed receiver and stops forwarding later output.

Once a request ID has passed replay admission or application dispatch may have
begun, a dropped or failed stream is an uncertain outcome. The SDK never
replays it automatically under the same ID and never resends it with a fresh ID
or session. Any caller retry remains governed by the explicit application
idempotency contract in section 7.

### 11.3 Application terminal semantics

For v2 Chat Completions, the provider stream must have a valid SSE content type
and must produce its terminal `[DONE]` event. A clean provider EOF without
`[DONE]` is an error. After observing that event, OpenSecret emits the existing
caller-visible `data: [DONE]\n\n` bytes as an ordinary encrypted `Chunk`, then
signals application completion so the gateway emits `End`. Provider error,
timeout, malformed or oversized provider framing, or unterminated EOF becomes a
sanitized transport `Error`; none becomes plaintext SSE. These stricter checks
are v2-only.

For v2 Responses, every application event is serialized into the existing
caller-visible `event: ...\ndata: ...\n\n` bytes and carried inside one or more
encrypted chunks. The application terminal events `response.completed`,
`response.cancelled`, and `response.error` are ordinary final application SSE
events; after emitting one, the application signals completion and the gateway
emits `End`. Responses does not emit `[DONE]`. A failure that prevents the
application from producing one of those terminal events instead becomes the
transport `Error` record. Image-description preprocessing may fall through
after a known pre-acceptance failure, explicit retryable response, or completed
but unusable response, but never after an ambiguous post-send failure; v1 keeps
its existing fallback behavior.

This record layer is not specific to SSE application semantics. It can later
carry other ordered server-to-client byte streams. Bidirectional transports
such as WebSocket, WebTransport, or WebRTC require a separate framing profile
and are not added by this protocol version. All of these rules are additive to
transport v2; transport-v1 routing, middleware, provider behavior, response
encryption, SSE framing, terminal behavior, and disconnect behavior remain
unchanged.

## 12. Parsing and resource limits

All limits are checked before allocation or work at the corresponding boundary.
At minimum v2 bounds:

- outer encrypted request bytes;
- decrypted envelope bytes;
- path and query bytes;
- header count, individual name/value bytes, and aggregate header bytes;
- logical body bytes;
- individual encrypted/decrypted streaming record bytes;
- concurrent sessions and pending attestations;
- absolute session age;
- request records per session;
- response records per request and session; and
- per-session and global replay identifiers.

The raw outer request ceiling is the 67 MiB request-envelope limit plus the
28-byte AEAD nonce/tag overhead. Inner base64 expansion and simultaneous
encrypted/decrypted/decoded buffers must be included in memory accounting
rather than described as only wire overhead. The request working-set
reservation remains held while the final encrypted response is serialized and
until its HTTP body is consumed or dropped. Operations whose
maximum output is not correlated with their request size promote that held
reservation to a route-specific output target before dispatch.

The initial v2-core limits are:

| Resource | Limit |
| --- | ---: |
| Absolute session lifetime | 3,900 seconds |
| Pending attestations | 65,536 |
| Live sessions | 65,536 |
| Request records per session | 65,536 |
| Response records per session | 65,536 |
| Replay identifiers per session | 65,536 |
| Replay identifiers globally | 2,097,152 |
| Decrypted request envelope | 67 MiB |
| Raw outer request record | 70,254,620 bytes (67 MiB + 28 bytes) |
| Logical request body | 50 MiB |
| Decrypted unary-response envelope | 50 MiB |
| Raw unary-response record | 52,428,828 bytes (50 MiB + 28 bytes) |
| Logical unary-response body | 28 MiB |
| Decoded application bytes per stream chunk | 64 KiB |
| Decoded sanitized stream-error body | 16 KiB |
| V2 Chat unfinished provider SSE frame | 1 MiB |
| V2 Chat provider stream aggregate | 64 MiB |
| V2 Chat caller-visible logical stream aggregate | 64 MiB |
| V2 Responses caller-visible logical stream aggregate | 64 MiB |
| Streaming provider working-set reservation | 128 MiB |

The separate v2 cache budget is 256 MiB. Capacity accounting reserves 512
bytes per live session, 64 bytes per replay identifier, and 192 bytes per
pending attestation entry: approximately 172 MiB at the independent hard
limits, leaving headroom for allocator and cache metadata. Replay sets allocate
lazily. The dormant-core stack layer defines and tests these limits without
allocating a cache; the gateway layer allocates the separate pending-attestation
and live-session caches when `AppState` is built.

The gateway also applies a separate 320 MiB aggregate admission budget to
in-flight v2 request buffers. It reserves a conservative four times the outer
content length in 64 KiB units before reading the body and holds that permit
through decryption, parsing, and response encryption. A missing
content length reserves the maximum request allowance. The actual body remains
independently capped at 67 MiB plus 28 bytes. Reserving the full working set up front permits
normal small-request concurrency without allowing several maximum-size base64
and plaintext buffer pipelines to grow concurrently.

The asymmetric request/response limits are deliberate. A request body is
base64-encoded inside the JSON envelope, then the envelope is encrypted and
carried as raw bytes. A maximally shaped 50 MiB logical request is approximately
66.79 MiB on the wire. The 67 MiB envelope ceiling accepts the released proxy's
50 MiB logical cap without adding a second outer base64 layer. Its four-times
admission charge permits at most one maximum request inside the 320 MiB request
budget while preserving the prior provider/stored-output concurrency classes.
Stored/provider-output promotion raises a smaller request reservation to its
bounded output target; it is not additive when the request already holds more
permits. Response and stored/provider-output ceilings remain unchanged.

No outer content encoding or decompression is accepted. Application-layer
payload validation remains owned by the existing application operation.

## 13. Required compatibility and security tests

Before a v2 client is published, backend, TypeScript, and Rust tests share
golden vectors for:

- handshake HKDF and encrypted key payload;
- request and response directional keys;
- UUID and request-ID byte encodings;
- request, unary-response, and streaming AAD;
- fixed nonces and expected ChaCha20-Poly1305 records;
- no body versus an explicitly empty body;
- exact query/header/body bytes; and
- wrong key/direction/session/request ID/sequence, malformed encodings, and tag
  tampering.

The backend additionally proves:

- concurrent identical request IDs have exactly one replay-gate winner;
- distinct request IDs are admitted out of order;
- replay is claimed before a mocked mutation/provider/billing/email side effect;
- malformed, undecryptable, and replayed requests never dispatch;
- replay/session exhaustion fails closed;
- authentication transitions are atomic and cancellation safe;
- bound principals cannot switch kind or identity;
- API-key deletion remains effective for an already-bound session;
- Chat and Responses select only their exact explicit response modes and reject
  `auto` or a mode/body mismatch;
- unary outer responses contain only the raw authenticated record, while the
  SSE carrier accepts only one canonical padded-base64 encrypted record per
  exact `data: ...\n\n` frame;
- stream records stay bound to the admitted session, request ID, and exact next
  sequence, and application bytes are split at the 64 KiB decoded limit;
- every post-`Start` application outcome has exactly one encrypted ordered
  terminal, including authenticated `Error` for unexpected source EOF;
- a cryptographic failure after `Start` ends in abrupt EOF without plaintext or
  a second encryption attempt;
- pre-`Start` failures atomically convert the stream reservation into an
  authenticated unary error;
- Chat emits application `[DONE]` before `End`, while each Responses terminal
  application event precedes `End` and never adds `[DONE]`;
- body drop releases delivery resources without turning disconnect into
  Responses cancellation, while shared producer permits remain held until all
  output-retaining work ends; and
- v2 load, leases, and cache exhaustion cannot alter a v1 session.

Every server PR runs current released-style JavaScript and Rust v1 clients
against the new server and characterizes existing route, status, error,
encryption, bodyless, and SSE behavior. Client cutover proves:

- old client against new server still uses unchanged v1;
- new client against old server fails as unsupported with no fallback;
- new client sends no outer authorization or application metadata;
- an ambiguous post-send failure is surfaced without automatic resend; and
- third-party JWT issuance remains usable downstream but cannot authenticate
  OpenSecret;
- two client instances or API keys at one origin never share a bound session;
  and
- deployment rehearsal verifies that every mandatory ingress accepts at least
  the 70,254,620-byte raw request cap, forwards it without buffering the full
  body in a memory-constrained intermediary, and preserves
  `application/octet-stream` without content transformation.

## 14. Planned pull-request stack

The implementation is intentionally consolidated into at most fifteen
capability PRs. Each PR targets the branch immediately below it:

1. **Protocol and v1-neutral seam**: freeze the contract and introduce shared
   application seams without changing transport-v1 behavior.
2. **Dormant crypto, session, and replay core**: codecs, key schedule,
   directional AEAD, separate caches, exact unordered replay registry, budgets,
   and golden vectors without a public v2 route.
3. **Isolated gateway**: v2 attestation, key exchange, bounded request
   admission, same-lease response encryption, and encrypted unsupported-route
   responses.
4. **User authority and sensitive crypto**: password registration/login,
   resumption, immutable user binding, sensitive key operations, and bounded
   encrypt/decrypt utilities.
5. **Complete bounded KV**: canonical key routing, mutations, bounded reads,
   and held output reservations.
6. **User credentials and account lifecycle**: API-key administration,
   verification, logout, password changes, and account deletion.
7. **Conversation projects and instructions**: owner-scoped project and
   instruction CRUD.
8. **Conversations, items, and stored response control**: bounded encrypted
   storage operations and response cancellation/deletion.
9. **Unary providers and API-key binding**: the exact unary route table in
   section 8.6, one-time encrypted API-key authority transition, live key-owner
   revalidation, and client-random Tinfoil cache namespaces.
10. **Session-bound user OAuth**: bind OAuth continuation to its originating
    v2 session without exposing credentials outside ciphertext.
11. **Platform authentication and account lifecycle**: distinct platform-v2
    credentials, immutable platform-session binding, password recovery, email
    verification, logout, and password change.
12. **Platform resource control**: bounded `/platform/me` and complete
    organization, project, membership, invite, secret, and settings projection
    with live role checks.
13. **Authenticated streaming**: ordered, request-bound records and an
    authenticated terminal event while preserving caller-visible streaming.
14. **Dormant TypeScript and Rust SDK engines**: shared vectors and private v2
    session/transport implementations that public calls do not yet select.
15. **Atomic v2 cutover and packaging**: switch new SDK majors to v2-only,
    update Maple and maple-proxy dependencies, and run compatibility/release
    rehearsal. Publication and deployment remain separately authorized.

Each implementation PR receives one focused security review and one focused
compatibility review. Reviews should report credible vulnerabilities,
behavioral regressions, or small correctness fixes; unrelated redesign and
speculative edge cases stay out of this stack.
