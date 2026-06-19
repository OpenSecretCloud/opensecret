# Provider Routing Design

This draft describes an internal provider router for chat completion traffic.
The first concrete use case is splitting Kimi K2.6 traffic across Tinfoil and
Continuum without changing the public OpenSecret API surface.

## Decisions

- Public model IDs, catalog responses, request shapes, and client behavior stay
  unchanged.
- Users cannot request an explicit provider.
- Provider-specific model IDs are translated only at request-send time.
- Kimi is pinned to the current provider-specific model version. Do not route
  public Kimi traffic to a moving `latest` alias.
- The first implementation should be a configurable, sticky, weighted router.
- The first production rollout target is 70/30, with 30% of eligible Kimi
  traffic sent to Continuum. The router still supports 50/50 and other future
  percentages through static weights.
- The initial config is static in code. Runtime health and availability can
  override selection, but they do not rewrite the base config.
- Weights should exist at both provider and model-route levels.
- OpenSecret should not auto-retry failed completion requests. Clients can retry
  if they want.
- Health-based routing, circuit breakers, polling, and provider recovery should
  be built as a second layer after the basic sticky router is easy to reason
  about.

## Goals

- Reduce provider load concentration, especially on Tinfoil.
- Keep routing stable for a user session so users do not bounce between providers
  unpredictably.
- Support per-model provider equivalence because not every provider supports
  every public OpenSecret model.
- Allow future routing percentages other than 50/50.
- Leave room for a third provider without redesigning the completion path.
- Make provider health visible enough to disable unhealthy routes automatically
  later.
- Keep most of the new routing code in new modules so ownership and future
  engineering work are cleanly separated from the existing completion handlers.

## Non-Goals

- No public API changes.
- No client-controlled provider selection.
- No automatic in-process request retry/failover for a single completion call.
- No broad rewrite of chat completions or Responses. Responses already uses
  completions under the hood, so the router should sit in the shared completion
  send path.

## Current Shape

OpenSecret currently has:

- Public model catalog and model alias resolution in `src/model_config.rs`.
- Proxy base URL handling in `src/proxy_config.rs`.
- Chat completion request sending in `src/web/openai.rs`.
- Responses request handling in `src/web/responses/handlers.rs`, which builds a
  chat completion request and calls the same completion send function.

The router should sit between model resolution and provider request sending.
That keeps the Responses path covered without duplicating routing logic.

## Proposed Internal Model

Separate public model identity from provider route identity:

```text
Public model:
  id: kimi-k2-6
  display_name: Kimi K2.6

Provider routes:
  tinfoil:
    provider_model_id: kimi-k2-6
    weight: 50
  continuum:
    provider_model_id: kimi-k2.6
    weight: 50
```

Only the provider route table knows provider-specific model names. Public model
catalog responses continue to expose the existing public IDs.

## Sticky Weighted Routing

For a public model with multiple eligible provider routes:

1. Resolve the public model ID and aliases as OpenSecret does today.
2. Find enabled provider routes for that public model.
3. Filter out routes whose provider is unavailable or circuit-open.
4. Select a route by hashing the user's stable account UUID into a 0-99 bucket.
5. Apply configured weights to that bucket.
6. Replace the request body model with the selected provider model ID only when
   building the outbound provider request.

For a 50/50 split, buckets `0..49` can route to provider A and `50..99` can
route to provider B. Other percentages use the same bucket table:

```text
70/30:
  0..69   provider A
  70..99  provider B
```

The first routing key should be the user's account UUID. Anonymous and guest
users also have UUID-backed accounts, so they can use the same stable bucketing
path. If model mix creates skew later, we can consider a routing key that
includes the public model ID, but that should be a deliberate product/ops
decision.

## Route Eligibility

A route is eligible only when all of these are true:

- The public model has an equivalent provider model configured for that
  provider.
- The provider is configured and enabled.
- The provider proxy base URL is present.
- The provider:model route is not circuit-open.
- The route weight is greater than zero.

If only one route is eligible, use that route. If no routes are eligible, return
the existing unsupported/provider-unavailable error shape rather than changing
the public contract.

## Static Configuration

The router should start with static in-code configuration, either as `const`
data or once-built global structs. Do not add a TOML config layer for the first
version.

The base config defines intended routing. In-flight health and load-balancing
state may override route eligibility at selection time, but should not mutate
the configured defaults.

Conceptual shape:

```rust
static PROVIDER_ROUTES: OnceLock<ProviderRoutingConfig> = OnceLock::new();

struct ProviderRoutingConfig {
    providers: &'static [ProviderConfig],
    models: &'static [ModelRoutingConfig],
}

struct ProviderConfig {
    provider: ProviderName,
    weight: u8,
    enabled: bool,
}

struct ModelRoutingConfig {
    public_model_id: &'static str,
    routes: &'static [ModelProviderRoute],
}

struct ModelProviderRoute {
    provider: ProviderName,
    provider_model_id: &'static str,
    weight: u8,
    enabled: bool,
}
```

Current rollout config:

```text
providers:
  tinfoil:
    enabled: true
    weight: 70
  continuum:
    enabled: true
    weight: 30

models:
  kimi-k2-6:
    tinfoil:
      enabled: true
      weight: 100
      provider_model_id: kimi-k2-6
    continuum:
      enabled: true
      weight: 100
      provider_model_id: kimi-k2.6
```

Both provider-level and model-route-level weights participate in selection. The
exact formula can be simple in v1, for example multiplying or normalizing the
enabled provider and model route weights into one effective route weight. Tests
should cover the resulting bucket distribution so future changes do not
silently change traffic splits.

For the first rollout, provider-level weights carry the overall traffic split
and model-route weights stay at `100` for providers that should be eligible for
Kimi. Later model-specific rollouts can lower a route weight without changing
global provider posture.

## Failure Classification

Circuit breakers should count provider failures, not user/client errors.

Likely provider failures:

- Connection errors.
- Request timeout.
- Stream chunk timeout or stalled stream.
- Unexpected upstream EOF before a terminal stream chunk.
- 5xx responses from the provider.
- 429 or provider overload responses, especially with retry/backoff hints.
- Malformed provider response for a request shape OpenSecret knows is valid.

Observed transient examples:

- On June 19, 2026, a small streaming `kimi-k2-6` comparison through
  maple-proxy produced a Tinfoil-backed `500 Internal server error` for a
  valid exact-format chat completion request while the Continuum-backed route
  succeeded for the same public request shape. A focused repeat of the same
  request shape immediately afterwards passed on both providers. Treat this
  kind of 5xx as a provider:model failure metric increment, not as a user error
  and not as proof of persistent provider outage by itself.

Likely non-provider failures:

- OpenSecret authentication or authorization failure.
- Billing denial.
- Invalid user request body.
- Unsupported public model ID.
- Payload too large before provider send.
- Provider 400 caused by OpenSecret sending a request shape unsupported by that
  provider. This is usually a route/model mapping bug, and should be surfaced
  separately from transient provider health.

Ambiguous failures should be logged with enough metadata to classify later, but
the first version should avoid overfitting the breaker to unclear cases.
False positives are also harmful: taking a healthy route out of rotation shifts
load elsewhere and can hide real application bugs. Start with loose but
reasonable classification, then tighten it only when real provider error cases
are observed.

## Continuum Prompt Caching

Continuum/Privatemode prompt caching should be enabled globally at the
Continuum proxy layer with `--sharedPromptCache`. OpenSecret should not expose
or require Privatemode-specific `cache_salt` request fields for this route.

With no fixed prompt cache salt configured, the proxy uses one random salt for
all requests handled by that proxy instance. This gives us the intended latency
benefit for long shared prefixes while keeping OpenSecret's public API unchanged.
The tradeoff is that cache reuse is per proxy process and is lost on restart. If
we later need cache sharing across multiple Continuum proxy instances or across
proxy restarts, we can add a deployment-managed `--promptCacheSalt` secret
without changing client requests.

## Follow-Up: Cached Token Billing Discounts

Continuum exposes cache-hit accounting in the OpenAI-compatible usage payload as
`usage.prompt_tokens_details.cached_tokens`. In a June 19, 2026 100k-token
follow-up test, the Continuum route reported `cached_tokens: 125120` on a
`125260` prompt-token request, which is enough signal for billing discounts if
we preserve and price the field internally.

This should be a follow-up feature after routing and global Continuum prompt
caching are stable. It should not require any public API changes, client
request changes, explicit provider selection, or per-request `cache_salt`.

Suggested behavior:

- Trust cached-token accounting only for the selected `continuum` route.
- Treat providers that do not report cached-token accounting, including Tinfoil,
  as `cached_input_tokens = 0`.
- Clamp cached tokens to `0..=prompt_tokens` before billing.
- Charge normal input pricing for `prompt_tokens - cached_input_tokens`.
- Charge cached-input pricing for `cached_input_tokens`, using the contracted
  Continuum cached-token rate or product discount. Do not invent this rate in
  OpenSecret.
- Charge output tokens normally.
- Apply the same discounted credit calculation to API-credit usage and
  subscription usage so usage limits match customer-visible pricing.
- Keep estimated provider cost separate from customer credits. Estimated cost
  should reflect the actual provider cost model, while credits should reflect
  the customer-facing discount policy.

Implementation sketch:

- Extend OpenSecret's internal `CompletionUsage` with `cached_prompt_tokens`.
- Update usage extraction to read
  `usage.prompt_tokens_details.cached_tokens`, defaulting to zero.
- Preserve the existing terminal-stream-only usage publishing behavior so
  cached-token discounts are based on final provider usage, not intermediate
  chunks.
- Add `cached_input_tokens` to the SQS `UsageEvent` with a serde default for
  backward compatibility.
- Add `cached_input_tokens` to billing `token_usage` via migration, with a
  non-negative check and default zero.
- Update billing's cost and credit calculation to split uncached input, cached
  input, and output tokens.
- Add tests for non-streaming usage, streaming terminal usage, absent
  `prompt_tokens_details`, over-reported cached tokens, Continuum cached hits,
  and non-Continuum providers with the same field present.

Open questions:

- What exact customer discount should cached Continuum input tokens receive?
- Should customer-facing usage screens show cached-token savings, or should the
  discount only appear through lower credits used?
- Should billing store both `cached_input_tokens` and computed
  `billable_input_tokens`, or store cached tokens only and derive billable input
  as needed?

## Circuit Breaker Direction

The circuit breaker can be added after sticky routing works.

Suggested states:

- `closed`: route is healthy and participates in weighted routing.
- `open`: route is removed from selection for a cooldown window.
- `half_open`: route can be probed or sampled with limited traffic.

Circuit breaker dimensions:

- Provider plus public model health.

Provider plus model health matters because one model can be overloaded or
misconfigured while another model on the same provider is healthy.

Do not add a global provider blacklist in the initial design. If Tinfoil Kimi is
unhealthy, only the `tinfoil:kimi-k2-6` route should leave rotation. Other
Tinfoil models remain eligible unless they independently cross their own
provider:model threshold.

Circuit state should be in-memory per OpenSecret instance. OpenSecret may run
behind distributed load balancing, but adding Redis, database tables, or another
shared state mechanism is out of scope for this feature. Per-instance routing
will not be perfect, and that is acceptable for v1.

The breaker should affect future routing decisions only. It should not retry the
current user request inside OpenSecret.

## Weighted Degradation Option

An alternative to a simple closed/open/half-open breaker is a weighted
degradation model. This may fit provider overload better than immediately
removing a provider:model route from rotation.

In this model, each provider:model route has:

- A configured base weight from static routing config.
- A dynamic health multiplier derived from recent provider-classified failures.
- A hard-disabled state only for severe or repeated failures.

Example conceptual levels:

```text
healthy:
  dynamic multiplier: 1.0
  effective route weight: base weight

watch:
  trigger: isolated provider 5xx, timeout, or overload signal
  dynamic multiplier: 0.75
  behavior: keep routing, but shift some load away

degraded:
  trigger: repeated provider failures above a short-window threshold
  dynamic multiplier: 0.25
  behavior: provider still receives some traffic/probes, but most traffic shifts

open:
  trigger: sustained failures or severe unavailable response
  dynamic multiplier: 0.0
  behavior: remove provider:model route for a cooldown window
```

This keeps single transient failures from causing a hard failover while still
responding to load-related instability. For example, one valid `kimi-k2-6`
request returning a provider 500 could move the route to `watch` briefly, while
several 5xx/timeouts in a rolling window could step it down to `degraded` or
`open`.

Important constraints:

- Degradation should be per provider:model, not global provider blacklist.
- User/client errors must not affect the dynamic multiplier.
- Recovery should be gradual, either through successful real traffic or tiny
  safe probes.
- Sticky routing should remain stable within the currently eligible weighted
  table, but the table itself can change as dynamic multipliers change.
- The first implementation does not need this. It is a future layer once static
  sticky routing is well tested.

## Health Recovery

Later versions can restore providers through background checks rather than user
request retries.

Tiny safe test prompts are allowed for health recovery. Cost is acceptable, but
probe behavior must be deliberately conservative so a bug cannot spam providers.

Possible probes:

- Provider `/v1/models` check.
- Provider-specific lightweight health endpoint, if available.
- A small synthetic completion against a safe test prompt, only if cost and
  provider policy make that acceptable.

Recovery should require more than one green signal before restoring normal
traffic. A route can move from `open` to `half_open`, then back to `closed`
after consecutive healthy probes or successful sampled requests.

Probe safeguards:

- Bound probe frequency per provider:model route.
- Use jittered intervals so multiple OpenSecret instances do not probe at the
  exact same time.
- Cap concurrent probes.
- Make the probe prompt tiny and deterministic.
- Make it easy to disable probes entirely through a code/config switch if they
  behave unexpectedly.
- Never use user traffic retries as the recovery mechanism.

## Observability

Log or emit metrics for:

- Public model ID.
- Selected provider.
- Provider model ID.
- Routing bucket.
- Route weight version or config version.
- Provider result class.
- Latency to first byte.
- Total completion latency.
- Stream timeout or terminal chunk status.
- Circuit state transitions.

Do not log prompts, generated content, secrets, or raw user payloads as part of
router observability.

## Implementation Sketch

Possible new internal types:

```rust
struct ProviderRoute {
    provider: ProviderName,
    provider_model_id: String,
    effective_weight: u8,
}

struct SelectedProviderRoute {
    provider: ProviderName,
    provider_model_id: String,
    proxy: ProxyConfig,
    bucket: u8,
}

struct ProviderRouter {
    config: ProviderRoutingConfig,
    provider_registry: ProviderRegistry,
    health: ProviderHealthState,
}
```

Most new logic should live in new files, for example:

```text
src/provider_routing.rs
src/provider_routing/config.rs
src/provider_routing/health.rs
```

`src/model_config.rs` should continue to own public model catalog behavior.
Existing completion code should only call a small router interface, receive a
selected route, and send the outbound request.

Completion flow:

1. Keep `body["model"]` as the public model until the shared send path.
2. Ask the router for a `SelectedProviderRoute`.
3. Clone the outbound request body.
4. Replace only the outbound `model` value with `provider_model_id`.
5. Send to the selected provider proxy.
6. Record provider result metadata.
7. Return provider errors without in-process retrying.

Responses flow:

1. Keep `ResponsesCreateRequest.model` as the public model while building
   context, sampling defaults, and internal storage metadata.
2. Build the chat completion request using the public model.
3. Let the shared completion send path translate to the selected provider model.

This avoids leaking provider IDs into public response state and preserves model
config lookup behavior.

## Phased Plan

Phase 1: Sticky weighted routing

- Add static in-code provider route table for models with provider equivalents.
- Select routes by stable account UUID hash and configured weights.
- Translate provider model IDs only at request send time.
- Keep public model catalog unchanged.
- Add focused unit tests for 50/50 bucketing and provider model translation.

Phase 2: Metrics and classification

- Record selected provider, provider model, bucket, latency, and result class.
- Classify failures without changing routing behavior yet.

Phase 3: Circuit breaker

- Add in-memory provider:model circuit state.
- Remove open routes from future weighted routing.
- Use cooldowns and half-open probes.
- Keep request-level behavior no-retry.

Phase 4: Operational tuning

- Add background provider polling.
- Consider per-model provider health dashboards.
- Consider carefully whether runtime-configurable weights are needed later. The
  starting point is static config plus health-based route eligibility.

## Open Questions

- Should the static route table live in a single `provider_routing.rs` file or a
  small `provider_routing/` module tree from the start?
- What exact formula should combine provider-level and model-route-level weights?
- Which existing feature-flag/user-bucketing helper logic can be reused for
  stable UUID bucketing?
- What provider error bodies/statuses should be counted as overload versus
  invalid request versus configuration bug?
