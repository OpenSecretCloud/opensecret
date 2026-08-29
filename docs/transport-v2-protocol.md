# OpenSecret Transport Protocol v2

Status: pre-implementation protocol contract

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

The encrypted key payload contains the protocol version, session UUID, session
master, and absolute expiry. The outer and encrypted session UUIDs must match.
The record uses ChaCha20-Poly1305 and this AAD:

```text
UTF8("opensecret/transport-v2/key-exchange")
```

This domain-separates v2 from the legacy direct-shared-secret wrapping format.
The exact payload encoding and golden bytes are frozen by the cross-language
test vectors before the endpoint is registered.

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
  authentication transition such as initial API-key binding.
- `method` is a supported uppercase HTTP method.
- `path` is one origin-relative application path with no query or fragment.
- `query` is either `null` or the exact query string without a leading `?`.
- `headers` preserves supported duplicate fields in order. Header names are
  lowercase ASCII and values are exact bytes encoded as canonical base64.
- `body_base64: null` means no request body. An empty string means an explicitly
  present empty body. Other values preserve exact request bytes without
  transport-layer JSON parsing or reserialization.

Unknown and duplicate JSON fields are rejected. The implementation parses the
path and query once, rejects schemes, authorities, fragments, backslashes,
dot-segments, encoded separator ambiguity, and invalid percent encoding, then
maps `(method, path)` through an exact application allowlist.

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
7. Atomically claim the decoded 16-byte request identifier.
8. Only then authenticate, dispatch, mutate state, bill, send email, or call a
   provider.

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
- logout encrypts its response with the admitting lease before closing and
  zeroizing the session.

The SDK owns each v2 session within exactly one authentication/client context;
it never reuses the current origin-and-PCR-only global cache across users,
platform users, API keys, or anonymous clients. User/account changes and
`set_api_key` or `clear_api_key` abandon the affected bound session and establish
a fresh one. Distinct API keys can have concurrent sessions against one origin
without sharing authority.

A password change may replace the same user's credential-bound `AuthContext`.
That is credential-context evolution for one principal, not principal rebinding.

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
accounting rather than described as only wire overhead.

No outer content encoding or decompression is accepted. Application-layer
payload validation remains owned by the existing application operation.

## 13. Required compatibility and security tests

Before registering public v2 routes, backend, TypeScript, and Rust tests share
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
4. **V2 gateway and binding**: attestation/key exchange/request endpoints,
   anonymous authentication transitions, v2-only resumption, and API-key
   binding.
5. **Application projection and streaming**: existing operation families over
   v2, live authorization checks, unary responses, and ordered streaming.
6. **Additive SDK v2 internals**: TypeScript and Rust codecs/session managers
   behind private seams, still not selected by public calls.
7. **Atomic SDK cutover**: v2-only network behavior, no downgrade, one-time
   fresh login, and Maple/proxy integration.
8. **SDK major/version packaging**: package metadata, locks, integration pin,
   compatibility matrix, and release rehearsal. Publication/deployment remain
   separate authorized actions.

Each implementation PR receives one focused security review and one focused
compatibility review. Reviews should report credible vulnerabilities,
behavioral regressions, or small correctness fixes; unrelated redesign and
speculative edge cases stay out of this stack.
