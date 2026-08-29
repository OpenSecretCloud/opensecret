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

- bHTTP, EHBP, DPoP, HPKE, TLS termination inside the enclave, or WebSockets;
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
content-type: application/json

{"encrypted":"<canonical padded standard-base64 record>"}
```

It does not accept outer authorization, cookies, logical query parameters, or
application/provider headers. Intermediaries may add ordinary transport
headers, but those headers cannot affect application authorization or dispatch.

The outer response may use HTTP status and content type for transport framing.
The SDK treats logical status, headers, errors, and stream termination as valid
only after authenticating the encrypted response record.

## 4. Attested key exchange

The client generates:

- a fresh attestation nonce; and
- a fresh X25519 ephemeral key pair.

The client verifies the Nitro attestation document, including its requested
nonce, attested enclave ephemeral X25519 public key, signature chain, freshness
policy, and approved PCR policy, before continuing.

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
  "response_mode": "auto",
  "credential": null,
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
- `response_mode` is `unary`, `stream`, or `auto`. `auto` lets raw SDK/proxy
  APIs preserve a server-selected unary or streaming response. Explicit modes
  must agree with the selected route and application request.
- `credential` is normally `null`. It is used only for a permitted anonymous
  authentication transition. The initial strict variants are
  `{"kind":"api_key","value_base64":"..."}` and
  `{"kind":"resumption","value_base64":"..."}`. The value is the exact
  credential bytes in canonical padded standard base64. Password, registration,
  OAuth, and recovery credentials remain part of their logical operation body;
  they do not become generic transport credentials.
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
current SDK contract for arbitrary valid end-to-end extension headers and
duplicates, but apply a strict denylist. Every policy rejects at least:

- `authorization`, `cookie`, `set-cookie`, `host`, and `x-session-id`;
- connection, proxy, transfer, upgrade, and other hop-by-hop/framing headers;
- client-supplied provider credentials; and
- unsafe fields selected by a `connection` header and any field the existing
  raw-inference contract already strips.

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

The binding retains a non-secret key identity/hash and rechecks current key
existence before every operation so deletion remains immediately effective.
API-key sessions can reach only the same inference operation set that accepts
API keys today; they never become general user sessions.

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
server internals, and credentials.

Failures before a valid session can be leased and the request identifier can be
recovered are generic bounded plaintext transport errors. They are untrusted
and terminal for that attempt. They cannot authorize automatic replay or
fallback.

## 11. Streaming responses

Transport v2 preserves the SDK's caller-visible streaming response, including
SSE used by chat and Responses APIs. The outer response is an SSE carrier whose
`data` field contains one canonical-base64 encrypted record.

Decrypted records are one of:

```json
{"version":2,"request_id":"...","sequence":0,"kind":"start","status":200,"headers":[]}
{"version":2,"request_id":"...","sequence":1,"kind":"chunk","body_base64":"..."}
{"version":2,"request_id":"...","sequence":2,"kind":"end"}
{"version":2,"request_id":"...","sequence":2,"kind":"error","status":500,"body_base64":"..."}
```

Rules are:

- Sequence starts at zero and increments by one for every encrypted record.
- Sequence is included in both AAD and plaintext and must equal the client's
  next expected value.
- `start` is first and authenticates logical status and response headers.
- `chunk` carries exact raw response bytes. The SDK reconstructs the current
  SSE or other streaming body without parsing application events in the
  transport layer.
- Exactly one authenticated `end` or `error` terminal record is required.
- The enclave reserves both `start` and terminal record capacity before
  dispatch; later chunks cannot consume either reservation.
- EOF before a terminal record, a duplicate/out-of-order sequence, plaintext
  application data, or an undecryptable record fails the stream.
- The exact request lease remains held until terminal delivery or body drop.

This record layer is not specific to SSE application semantics. It can later
carry other ordered server-to-client byte streams. Bidirectional transports
such as WebSocket, WebTransport, or WebRTC require a separate framing profile
and are not added by this protocol version.

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

The initial outer ceiling may retain the current 50 MiB limit. Base64 expansion
and simultaneous outer/decoded/decrypted buffers must be included in memory
accounting rather than described as only wire overhead. The request working-set
reservation remains held while the final encrypted response is JSON/base64
serialized and until its HTTP body is consumed or dropped. Operations whose
maximum output is not correlated with their request size require an additional
route-specific output reservation before dispatch.

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
| Decrypted envelope | 50 MiB |
| Logical body | 28 MiB |

The separate v2 cache budget is 256 MiB. Capacity accounting reserves 512
bytes per live session, 64 bytes per replay identifier, and 192 bytes per
pending attestation entry: approximately 172 MiB at the independent hard
limits, leaving headroom for allocator and cache metadata. Replay sets allocate
lazily. The dormant-core stack layer defines and tests these limits without
allocating a cache; the gateway layer allocates the separate pending-attestation
and live-session caches when `AppState` is built.

The gateway also applies a separate 256 MiB aggregate admission budget to
in-flight v2 request buffers. It reserves a conservative four times the outer
content length in 64 KiB units before reading the body and holds that permit
through decoding, decryption, parsing, and response encryption. A missing
content length reserves the maximum request allowance. The actual body remains
independently capped at 50 MiB. Reserving the full working set up front permits
normal small-request concurrency without allowing several maximum-size base64
and plaintext buffer pipelines to grow concurrently.

The 28 MiB logical-body limit is deliberate. A body is base64-encoded inside
the JSON envelope, then the envelope is encrypted and base64-encoded again for
the outer request. Keeping a 50 MiB outer ceiling therefore cannot preserve
transport v1's larger effective plaintext ceiling. Raising the v2 outer limit
requires separate enclave-memory measurement rather than an implicit change.

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
- every stream outcome has encrypted ordered terminal framing; and
- v2 load, leases, and cache exhaustion cannot alter a v1 session.

Every server PR runs current released-style JavaScript and Rust v1 clients
against the new server and characterizes existing route, status, error,
encryption, bodyless, and SSE behavior. Client cutover proves:

- old client against new server still uses unchanged v1;
- new client against old server fails as unsupported with no fallback;
- new client sends no outer authorization or application metadata;
- an ambiguous post-send failure is surfaced without automatic resend; and
- third-party JWT issuance remains usable downstream but cannot authenticate
  OpenSecret.
- two client instances or API keys at one origin never share a bound session.

## 14. Planned pull-request stack

1. **Protocol contract**: this document and byte-level review decisions.
2. **V1 transport seam**: behavior-neutral response/session context and
   characterization tests; no v2 routes.
3. **Dormant v2 core**: codecs, key schedule, directional AEAD, separate session
   state/cache, absolute expiry, replay registry, budgets, and vectors; no
   public routes.
4. **Isolated v2 gateway**: separate attestation/key-exchange/request endpoints,
   bounded session allocation, strict outer parsing, exact-session decryption,
   and encrypted unsupported-operation responses; no application dispatch.
5. **User binding and first unary slice**: password login, registration,
   v2-only resumption, immutable user/project/`AuthContext` authority, live
   seed-wrap revalidation, and bodyless `GET /protected/user`. This layer also
   activates exact unordered replay claims for supported operations. Platform,
   API-key, OAuth, and all other application paths still receive the encrypted
   unsupported-operation response.
6. **Sensitive user-key projection**: project root/derived mnemonic and private
   key export, public-key derivation, signing, and third-party token issuance
   through the live bound-user check. Sensitive intermediate response values
   are zeroized, decoded byte fields and prepared sensitive operations wipe on
   drop across rejection/cancellation paths, and serialized logical responses
   are bounded before gateway encryption. Large encrypt/decrypt utilities
   remain unsupported until their base64 and JSON expansion has an exact
   pre-dispatch limit.
7. **Bounded user crypto utilities**: project user-key encrypt/decrypt with an
   exact v2-only plaintext ceiling derived from AES-GCM, base64, and JSON
   expansion. Logical JSON serialization writes through a bounded buffer so a
   highly escaped decrypted string cannot allocate beyond the response limit.
8. **Dynamic-path and response-resource seams**: admit only route-scoped,
   canonical KV item segments and retain request working-set reservations
   through final response serialization/body lifetime. Tiny-request operations
   with uncorrelated stored output remain unsupported until they reserve that
   output before dispatch.
9. **KV mutations**: project PUT item, DELETE item, and DELETE all through
   transport-neutral helpers. Preserve existing response and error behavior,
   claim replay identifiers before every mutation, sanitize storage errors, and
   wipe decoded keys and values. GET item and list remain unsupported here.
10. **Bounded KV reads**: project GET item and list through a v2-only,
   read-only repeatable-read storage seam. Promote each admitted read to a
   conservative 200 MiB reservation from the shared 256 MiB request/response
   pool before replay claim or dispatch; contention returns an authenticated
   503 without consuming the request identifier. Bound item ciphertext, list
   aggregate plaintext, and list rows before loading narrow ciphertext
   projections, then retain the merged reservation through final HTTP body
   consumption. Preserve `null`, string, bare-array, timestamp, ordering, and
   whole-list failure behavior from v1 while leaving every v1 query untouched.
11. **Remaining protected user operations**: project account-lifecycle and user
   API-key-management families in reviewable sub-stacks. Start API-key
   administration with bounded create/delete mutations: retain the existing
   raw UUID key format without rotation, return it only through the original
   bound-session response, wipe its plaintext response value on drop, and use
   the canonical validated name segment for deletion. Add list in a following
   slice using the same conservative stored-output admission as KV reads and a
   v2-only read-only repeatable-read database path. Preflight the user-scoped
   row count and aggregate name bytes, cap the list at 65,536 rows, fetch only
   `name` and `created_at`, recheck the snapshot totals, retain unspecified
   server ordering, and bound final JSON serialization. The v1 full-row query
   remains untouched. Project verification-email resend as a bound-user unary
   mutation with no logical body or metadata; preserve its existing 200 JSON
   outcomes while claiming replay state before any database or email side
   effect. Project `GET /verify-email/{code}` as a code-authorized operation
   that accepts either an anonymous or already-bound session without changing
   its authority. Require one lowercase hyphenated UUID path spelling, reject
   all logical metadata, and preserve the existing invalid/expired 400 plus
   success/already-verified 200 outcomes. Project account-deletion request as
   an exact bound-user JSON mutation, preserving the generic success response,
   independent fresh attempts, and background email behavior while claiming
   replay state before request creation. Account-deletion confirmation must
   pre-serialize its fixed success response before committing deletion, then
   explicitly close the now-invalid bound session only after that commit.
   Invalid codes, secrets, requests, expiry, and database failures return their
   existing encrypted errors without closing an otherwise valid session.
   Project user logout as an exact bound-user JSON operation with the existing
   refresh-token request and success body, then close the admitted session only
   after the response is prepared. This remains session-local logout: matching
   transport-v1 behavior, the submitted resumption credential is not revoked
   server-side. The v2 SDK must treat the request as terminal, never
   transparently retry or resume the logout attempt, and use generation-safe
   local cleanup after the final response attempt. Exact cleanup behavior for
   an ambiguous outcome is fixed in the SDK cutover PR. Project password change
   as an exact bound-user JSON mutation with its existing logical request and
   response fields. Prepare and locally verify the replacement password wrap,
   issue both replacement v2 credentials, and bound-serialize the success
   response before attempting the existing password-ciphertext CAS transaction.
   A successful commit or any failure after the commit attempt terminally
   closes the old session; definite parse, current-password, preparation, and
   serialization failures retain a still-valid session. A lost final response
   is never retried: the client recovers through a fresh login with the
   submitted new password.
12. **Stored user unary operations**: project conversations, conversation
   projects, instructions, response control, and web-provider unary routes in
   ownership-preserving families before the client cutover.
13. **API-key and platform binding**: bind existing raw API keys inside
   ciphertext without rotating them, enforce their current inference-only
   scope, and add platform authentication/authorization with live organization
   checks. OAuth continuation receives an explicit session-binding design in
   this layer rather than being inferred from password login.
14. **Streaming projection**: project current inference streams with ordered,
   request-bound, authenticated terminal records while preserving the SDK's
   caller-visible SSE behavior.
15. **Additive SDK v2 internals**: TypeScript and Rust codecs/session managers
   behind private seams, still not selected by public calls.
16. **Atomic SDK cutover**: v2-only network behavior, no downgrade, one-time
   fresh login, and Maple/proxy integration.
17. **SDK major/version packaging**: package metadata, locks, integration pin,
   compatibility matrix, and release rehearsal. Publication/deployment remain
   separate authorized actions.

Each implementation PR receives one focused security review and one focused
compatibility review. Reviews should report credible vulnerabilities,
behavioral regressions, or small correctness fixes; unrelated redesign and
speculative edge cases stay out of this stack.
