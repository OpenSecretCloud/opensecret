---
name: change-opensecret-provider
description: Change or review OpenSecret inference and web-provider integrations. Use for model catalog entries or aliases, Tinfoil or Continuum routing, provider credentials and attestation, request or response canonicalization, retries, headers, cache fields, usage accounting, health checks, web search or extraction providers, and direct provider parity testing.
---

# Change an OpenSecret Provider

Keep providers behind the backend boundary. Maple and public API clients choose public capabilities and model IDs; OpenSecret owns provider selection, credentials, transport security, provider model translation, request policy, response normalization, and usage attribution.

## Establish the live provider graph

Read `AGENTS.md`, inspect the checkout, and derive the current graph from source. Provider and model inventories change frequently; never copy a static list from this skill into code or documentation.

```sh
rg -n 'MODEL_CONFIGS|MODEL_ALIAS_ENTRIES|QUICK_MODEL_ID|POWERFUL_MODEL_ID|openai_models_response|model_catalog_response' src/model_config.rs
rg -n 'PROVIDERS|MODEL_ROUTES|ModelProviderRoute|default_provider|flag_key|select_completion_route' src/provider_routing.rs
rg -n 'get_completion_proxy|get_default_proxy|get_tinfoil_proxy|provider_name' src/proxy_config.rs
rg -n 'ProviderClient|ProviderRequest|TinfoilUnavailable|send_standard|forwarded_headers|skipped_forward_headers' src/provider_client.rs
```

Trace the requested model through:

1. Public alias and access resolution in `src/model_config.rs`.
2. User-specific model-access and routing flags in `src/main.rs` and `src/os_flags.rs`.
3. Provider route selection in `src/provider_routing.rs`.
4. Provider-model translation and managed request fields in `src/web/openai.rs`.
5. Transport construction in `src/provider_client.rs`.
6. Public response-model canonicalization and usage publication.
7. Responses-specific sampling, reasoning history, and tool behavior.

The checked-in code is authoritative. Design notes and old parity evidence explain intent but may describe a previous provider set or launch phase.

## Preserve the public model contract

Centralize model behavior in `src/model_config.rs`. A model change may affect all of these together:

- canonical public ID and provider ID;
- display/catalog metadata and ordering;
- listed versus API-only visibility;
- enabled and feature-gated access;
- access tier and chat/vision/reasoning/tool-use capabilities;
- context window and prompt budget;
- Responses sampling and current-turn thinking behavior;
- prior-reasoning replay strategy;
- public aliases and quick/powerful defaults;
- `/v1/models`, `/v1/models/catalog`, audio/embedding entries, and Maple selectors.

Do not add an ID in one response builder or route handler. Add it to the centralized configuration and test alias resolution, access filtering, both model-list surfaces, completion validation, Responses context limits, and the actual upstream provider ID.

Keep public IDs provider-neutral and pinned. Do not expose `latest`, provider hostnames, routing flags, weights, or provider-specific model spellings. Translate only immediately before sending and canonicalize returned `model` fields to the public ID for non-streaming and streaming responses.

Reasoning is model-specific. Distinguish:

- emitting reasoning in the current turn;
- preserving prior assistant reasoning in later prompt history;
- provider-specific template kwargs needed for either behavior.

Never infer replay support from visible answer quality. Prove it with direct provider probes that compare prompt-token deltas and a recall prompt, then encode the result in the model strategy and unit tests.

## Preserve routing semantics

Use `src/provider_routing.rs` as the sole completion-routing policy. The current architecture supports model-specific eligible routes, provider and route weights, an explicit default, stable UUID bucketing when no preference wins, and user-specific flag preferences.

Do not describe configured weights as active traffic distribution without tracing precedence. An explicit flag preference or model default can bypass weighted selection entirely. Missing flag configuration, missing flag values, timeouts, and flag-service errors fall back to the model's default route rather than exposing control-plane failure to the client.

Maintain these boundaries:

- Route using the authenticated account UUID, not a request-provided identity.
- Keep aliases resolved before routing.
- Keep feature-flag keys canonical in `src/os_flags.rs`.
- Keep selection deterministic for the same account/configuration.
- Record the actual provider but bill and respond with the canonical public model.
- Return a client error for an unsupported public model and an internal error for a valid model with no eligible configured route.
- Do not accept a client-specified provider or provider-model ID.

Chat completion currently selects one route and tries it once. Do not add backend-level cross-provider replay after a timeout, ambiguous body error, partial stream, or provider status. A POST may already have caused computation and usage. If product requirements call for failover, design idempotency, side-effect accounting, stream semantics, and rollout observability first.

## Preserve provider transport security

Tinfoil and ordinary OpenAI-compatible transports are intentionally different.

### Tinfoil

- `TINFOIL_API_KEY` is a hard backend startup requirement. Local mode may read the gitignored `.local/secrets/tinfoil_api_key`.
- The in-process Tinfoil SDK discovers and attests an enclave in the background, pins TLS to the verified enclave, and shares its connection pool.
- Backend startup and non-Tinfoil routes do not wait for discovery. Tinfoil-dependent requests return 503 while the attested client is unavailable.
- Discovery attempts are bounded and back off. Derive current timings from `provider_client.rs`.
- A typed DNS/TCP/TLS connect failure may trigger one single-flight rediscovery/attestation and one identical-byte retry. Request, body, response, timeout, provider-status, and partially streamed failures are not replayed by this layer.
- Recovery tasks must not retain a request body or be cancelled when one caller times out.

Do not replace this path with an unauthenticated localhost sidecar, ordinary CA-only TLS, a caller-selected enclave, or a fallback that skips attestation.

### Continuum or another standard provider

- `OPENAI_API_BASE` configures the ordinary compatible base. The supported
  local Continuum recipe uses the loopback URL documented in
  `docs/local-macos-stack.md` and does not need an OpenAI credential. Treat any
  other custom base as a credential boundary: derive URL classification and
  header behavior from current source and cover it with exact-host and
  adversarial-host tests before using credentials.
- Local macOS Continuum runs as its own native proxy; Tinfoil does not have a local proxy port.
- Adding another provider requires a clear transport/authentication policy. Do not route a confidential workload through the ordinary client merely because its API is OpenAI-shaped.

Provider secrets stay server-side, in environment/gitignored local secret files or the existing non-local secret path. Never place a real key in `.env.sample`, source, tests, evidence, logs, URLs, client headers, Maple configuration, or shell history.

## Sanitize and own request fields

Both transports must strip inbound authorization, host, content length/type controlled by the backend, hop-by-hop headers, and every header named by `Connection`. Preserve only safe end-to-end headers, then set the provider credential and content type explicitly. Extend both Tinfoil and standard sanitizers together and add network-boundary tests for any header-policy change.

OpenSecret also owns provider cache fields:

- Strip caller-supplied `cache_salt`.
- For Tinfoil, replace `user_cache_secret` with the stable SHA-256 namespace derived from the authenticated user UUID.
- For other providers, strip caller-supplied `user_cache_secret`.

Do not expose the raw user UUID to a provider when a derived namespace is sufficient. Do not let a caller select another user's cache namespace.

Log bounded request metadata, not prompts, images, audio, credentials, full
headers, or full responses. Redact provider diagnostics before returning them
publicly, and add semantic tests or source invariants for newly introduced
sensitive fields where practical.

## Preserve response, stream, and usage semantics

For chat completions:

- Force `stream_options.include_usage=true` on streaming upstream requests.
- Parse SSE at complete frame boundaries and support LF and CRLF framing.
- Preserve complete upstream JSON chunks while replacing provider model IDs with the canonical public ID.
- Treat `[DONE]` as the authoritative terminal marker. A finish reason is terminal evidence, but a later usage-only frame may still arrive.
- If the upstream ends without `[DONE]`, use the existing terminal-evidence policy; do not publish partial usage as successful usage without that evidence.
- Emit exactly one public `[DONE]` and exactly one usage publication.

Usage parsing clamps numeric values, tracks prompt and completion tokens, and bounds cached prompt tokens by total prompt tokens. Preserve:

- actual provider name;
- canonical public model name;
- JWT versus API-key attribution;
- cached input tokens when present;
- asynchronous local persistence and optional event publication;
- no event when both token counts are zero.

The local cost field is observability, not the authoritative billing price.
When entitlement behavior matters, configure the external billing API URL/key
and test the backend's public outcomes: allowed, denied, unavailable, and guest
behavior. Treat billing solely as an HTTP API boundary.

Responses calls the shared completion sender, so provider changes can affect stored conversations, tool loops, title generation, reasoning history, and usage aggregation. Ensure the Responses orchestrator does not publish duplicate usage for a multi-turn tool response.

## Treat audio and embeddings as provider contracts

Derive the current provider choice, model translations, size limits, and retries from `src/web/openai.rs` and `src/web/audio_utils.rs`.

- Speech is currently provider-owned through the Tinfoil transport and has paid-entitlement, timeout, model, voice, and encrypted base64-response behavior.
- Transcription accepts base64 input at the public boundary, splits audio when necessary, maps model IDs per provider, and has explicit primary/fallback behavior. This is separate from chat's no-failover rule.
- Embeddings use their configured provider, validate non-empty input, encrypt the public response, and publish prompt-token usage.

When changing any of these, test both public payload compatibility and the exact provider request. Do not generalize transcription retry behavior to chat or speech.

## Keep web providers and web content untrusted

There are two related but distinct surfaces:

1. Direct authenticated `/v1/web/search` and `/v1/web/extract` are provider-neutral public APIs implemented by `src/web/web_routes.rs`. Their current adapter is Kagi.
2. The Responses `web_search` tool chooses an available Brave or Kagi tool implementation. Kagi additionally exposes internal `open_urls` extraction to the model.

Derive the exact schemas, limits, and selection rules:

```sh
rg -n 'WEB_SEARCH_PATH|WEB_EXTRACT_PATH|MAX_|DEFAULT_|deny_unknown_fields' src/web/web_routes.rs src/kagi.rs src/brave.rs
rg -n 'choose_web_search_provider|ToolRegistry|web_search|open_urls|allowed_urls' src/web/responses/handlers.rs src/web/responses/tools.rs
```

Preserve these security properties:

- Keep search and page extraction separate. Search metadata is not page content.
- Accept extraction only for normalized public HTTPS URLs. Reject credentials, local/reserved/metadata hosts, control characters, non-HTTPS schemes, and duplicates.
- Treat titles, snippets, diagnostics, and extracted Markdown as untrusted data. Strip image embeds and bound text before putting it into model context.
- Authorize Kagi `open_urls` from exact normalized URLs visibly supplied by the user or canonical URL fields generated by trusted Kagi formatters.
- Never authorize a URL merely because it appears in assistant text, a title/snippet, diagnostics, Brave free-form output, or an extracted page body.
- Reconstruct authorized URLs from visible persisted history on continuation; do not rely only on request-local discovery state.
- Keep provider traces sanitized and bounded. Do not return provider-specific experimental payloads or arbitrary personalization controls.

If the external flag API is absent or fails, preserve the documented provider
fallback. Configure its base URL/key for a specific dev/prod environment when
testing flag behavior and treat it solely as an HTTP API boundary.

## Interpret health correctly

`/health-check` is process liveness only and does not test the database. `/health-check-extended` performs a bounded direct Tinfoil model-list request. It demonstrates current Tinfoil discovery/attestation/connectivity, not database readiness, billing, flags, Brave, Kagi, Continuum, migrations, or Maple compatibility.

Use the successful Tinfoil discovery/attestation log plus extended health as provider-readiness evidence. Repeated discovery failures, attestation/certificate errors, timeouts, 503s, or connection failures are actionable. Normal HTTP/2 connection rotation or graceful shutdown logs alone are not failure proof.

## Validate safely

Start with deterministic unit and boundary tests:

```sh
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features provider_routing::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features model_config::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features provider_client::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::openai::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::web_routes::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::web_safety::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo test --locked --all-features web::responses::tools::tests
OPENSECRET_DEV_POSTGRES=0 OPENSECRET_DEV_ENV=0 OPENSECRET_DEV_CONTAINERS=0 \
  nix develop --no-write-lock-file -c cargo fmt --all -- --check
git diff --check
```

Use one Cargo filter per command. Then run the repository's full validation skill, including locked all-feature tests and strict Clippy.

Run live provider probes only when the task needs them and credentials/egress are authorized. Prefer deterministic prompts and paid/free fixtures that make routing and usage observable without recording user data. Keep raw evidence outside the repository and redact credentials, authorization headers, enclave proof metadata, and sensitive response content.

For the existing ignored Tinfoil parity test, derive the exact test name and evidence variables from `docs/tinfoil-rust-sdk-parity.md` and `src/provider_client.rs`; do not paste a possibly stale command. Its parity boundary is decrypted method/path/query, safe end-to-end headers, exact request bytes, status/content type/schema, ordered SSE semantics, numeric usage, and one terminal marker—not TLS record bytes, header order/casing, connection reuse, or generated text/chunk boundaries.

For a model reasoning claim, probe the provider directly before changing backend policy. For an application contract claim, also smoke the encrypted OpenSecret route through the SDK. For a Maple compatibility claim, run the relevant browser or native Maple path. Report each layer separately and never call a provider-only check end-to-end proof.
