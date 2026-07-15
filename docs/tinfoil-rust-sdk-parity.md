# Tinfoil Rust SDK migration parity

This migration removes the localhost Tinfoil Go process and sends Tinfoil
requests from the OpenSecret backend through the attested `tinfoil-rs` client.
The public OpenSecret API is unchanged.

## What parity means

Literal packets cannot be identical after removing a network hop. TLS record
bytes are randomized, the HTTP implementation changed, and the Rust client can
choose different header casing, ordering, HTTP framing, or connection reuse.
The relevant compatibility boundary is the decrypted HTTP request and response:

- method, path, and query string;
- backend-supplied end-to-end request headers, with inbound authorization
  replaced by the configured Tinfoil bearer token;
- removal of standard hop-by-hop headers and headers named by `Connection`;
- exact request body bytes;
- upstream status, application `Content-Type`, and response body schema;
- ordered Server-Sent Event payloads, numeric usage, and exactly one terminal
  `[DONE]` frame.

Library-generated defaults such as `User-Agent`, `Accept-Encoding`, header
casing/order, and message framing are not part of that boundary. Likewise,
`Date`, `Server`, the selected `Tinfoil-Enclave`, proof metadata, and
`Content-Length` versus chunked transfer reflect the live transport or selected
enclave rather than the OpenSecret API contract. The before and after captures
retain those headers so the distinction is reviewable; stable JSON/SSE content
types are asserted automatically.

The Rust client intentionally sends directly to an attested enclave over
origin-bound TLS. The removed Go process accepted localhost HTTP and then used
ordinary CA-validated TLS for the effective request path. Requests with a
known byte-vector body may also carry `Content-Length` instead of the previous
unknown-length framing. Those are transport and security changes, not public
API changes.

## Before and after evidence

The baseline was captured from the checked-in Go proxy before it was removed.
The same live Tinfoil credentials, model, prompts, temperature, token limit,
and streaming options were then exercised through the Rust SDK.

| Contract | Go baseline | Rust SDK result |
| --- | --- | --- |
| Tinfoil upstream `GET /v1/models` | HTTP 200, JSON, `object: list`, 14 model IDs | HTTP 200, JSON, same object and exact model-ID set |
| Non-streaming `POST /v1/chat/completions` | HTTP 200, `chat.completion`, model `glm-5-2`, exact text `parity-ok`, numeric usage | Same status, object, model, exact text, and usage shape |
| Streaming `POST /v1/chat/completions` | HTTP 200 SSE, ordered `chat.completion.chunk` frames, numeric final usage, one terminal `[DONE]` | Same status/content type, event schema, model, numeric final usage, and one terminal `[DONE]` |
| OpenSecret extended health check | Tinfoil models reachable through localhost sidecar | Tinfoil models reachable directly; 14 models reported and no sidecar port |
| Maple paid-user chat | Not applicable to the isolated proxy baseline | Built desktop app, Pro entitlement and 100,000 credits verified; exact GUI responses `maple-sdk-parity-ok` and `maple-sdk-final-ok` before and after the final backend restart |

Both completion request bodies were byte-identical across implementations. The
non-streaming body SHA-256 was
`1815c5ecd2c36834c59f7ce354737ad1b5bc2b93f176068c561f451957c6a717`;
the streaming body SHA-256 was
`d15a8b974409567cc3fef3d708406dedb99ee5578edeaf05ecc9dca9168120c8`.
The `GET` had the standard empty-body SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Raw Go and Rust response artifacts plus redacted manifests are kept outside the
repository workspace under `evidence/before/` and `evidence/after/`, so live
response data is not committed. The manifests record the implementation,
revision state, target, exact request JSON and hashes, stable response headers,
and normalized results. Their normalized request sections have no diff. The
live Rust assertions are executable and fail on a changed contract.

Completion text and streaming chunk boundaries are model-generated and are not
treated as packet-stable evidence. The same exact non-streaming text was
observed in both captures. Streaming parity uses the identical request bytes
and compares response status/content type, ordered event schema, numeric final
usage, and terminal framing; it does not require an identical number of deltas
or identical model-generated reasoning.

OpenSecret's public encrypted `/v1/models` response remains its existing static
catalog. The live upstream models call above exercises the provider boundary
used by extended health, which is the path that changed in this migration.

## Automated boundary checks

`provider_client::tests::tinfoil_request_matches_contract_at_the_network_boundary`
uses a real local HTTP listener to inspect the request emitted by the new
client. It proves the method, path/query, Tinfoil authorization replacement,
content type, preserved end-to-end header, stripped hop headers, and exact body
bytes at the receiving socket.

`provider_client::tests::live_tinfoil_models_and_completions_match_the_legacy_api_contract`
is ignored by default because it requires live credentials and egress. It
attests a real Tinfoil enclave and checks the models, JSON/SSE content types,
non-streaming completion, and streaming completion contracts listed above. It
also checks that the only `[DONE]` is terminal and that the final usage frame
contains numeric prompt, completion, and total token counts.

Run the live check with either `TINFOIL_API_KEY` or the normal gitignored local
secret file configured:

```sh
nix develop -c cargo test \
  provider_client::tests::live_tinfoil_models_and_completions_match_the_legacy_api_contract \
  -- --ignored --exact
```

Set `TINFOIL_PARITY_EVIDENCE_DIR` to write response bodies, headers, and a
redacted manifest during that test. `TINFOIL_PARITY_CREDENTIAL_LABEL` records a
non-secret operator assertion that the same test credential was used for both
captures; the credential itself and its fingerprint are never written.

## Deployment and build boundary

`entrypoint.sh` still decrypts the historical `tinfoil_proxy_*` Secrets Manager
value, but exports it as `TINFOIL_API_KEY` to the backend instead of launching a
second binary. The existing OpenSecret startup, five-second wait, VSOCK
exposure, and traffic-forwarder behavior are otherwise unchanged. Tinfoil
discovery and attestation retry in a single background loop inside OpenSecret,
so an outage at that boundary does not prevent login, billing,
stored-conversation, or non-Tinfoil routes from starting. Tinfoil-dependent
routes return HTTP 503 until an attested client is ready. Each discovery or
attestation attempt is bounded to 30 seconds before the background loop backs
off and retries. The EIF root filesystem no longer contains the Go proxy.
Existing Tinfoil egress forwarders remain unchanged for this minimal migration.

The shared attested client reuses connections. A typed DNS/TCP/TLS connection
failure triggers a single-flight router rediscovery and attestation, then one
retry rebuilt with identical request bytes. Request/body/response/status errors
are never replayed by the rotation layer, avoiding ambiguous completion POST
retries. The underlying HTTP implementation may still perform protocol-defined
safe retries, such as an HTTP/2 stream explicitly rejected before processing.
Only one request owns a router refresh; concurrent requests fail fast with HTTP
503 instead of retaining their bodies in a refresh wait queue. Recovery runs in
an owned task that captures no request body, so a short caller timeout does not
cancel the shared certificate/router refresh.

The root Nix development shell still includes Go only for the independently
vendored Continuum proxy build. All Tinfoil-specific Go source, module files,
build recipes, nested flake, binary artifacts, and root-filesystem wiring are
removed.

The companion `OpenSecretCloud/opensecret-workspaces` compatibility change must
be rolled out first. The updated manager supports both sidecar and SDK backend
branches, while the legacy manager still expects the recipes removed here.

The released Rust SDK currently verifies AMD SEV-SNP enclaves; its TDX verifier
is not implemented. This matches the Tinfoil deployment selected by the current
SDK defaults and is an explicit constraint for future provider changes.
