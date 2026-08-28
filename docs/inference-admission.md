# Inference admission policy

OpenSecret applies bounded, enclave-local admission control before starting a
completion provider turn. It limits concurrent work, bounds waiting queues,
gives interactive requests fair access across accounts, and immediately sheds
background work when capacity is unavailable. It does not coordinate state
between replicas and it does not replay an accepted provider request.

The first release uses one policy compiled into the OpenSecret binary. It does
not read an environment variable, database row, or encrypted enclave secret.
Changing the policy requires code review, a new build, and a process or enclave
restart. The scheduler still consumes a typed, validated `AdmissionPolicy` so a
future authenticated admin control plane can replace the source without
rewriting the admission machinery.

## Compiled baseline

The source constants bound local concurrency and memory use:

| Internal field | Baseline | Meaning |
| --- | ---: | --- |
| `deployment_in_flight` | `4` | Concurrent turns per provider-model deployment |
| `per_account_in_flight` | `2` | Concurrent logical responses per account |
| `pool_queue` | `16` | Waiting turns per provider quota pool |
| `per_account_queue` | `2` | Waiting logical responses or turns per account and pool |
| `interactive_wait_ms` | `5000` | Maximum interactive wait for local capacity |
| `background_wait_ms` | `0` | Background work is immediate-only; this must remain zero |
| `local_retry_ms` | `1000` | Client retry hint for local-capacity rejection |
| `rolling_window_seconds` | `60` | Local provider-budget accounting window |
| `completion_default_reservation` | `4096` | Per-choice completion reservation and provider-visible cap when the request has no limit |
| `max_completion_choices` | `8` | Maximum provider sequences (`n`) admitted for one request |

The compiled baseline deliberately does **not** assign an RPM, prompt-token,
completion-token, or cached-token quota to any provider. Those entitlements can
drift and may differ by environment, so this release does not promote example
or public-plan numbers into production policy. Upstream enforcement and the
local 429/503 health gates still apply. Add a source-level provider budget only
from a confirmed current limit, with rollout headroom and explicit review.

## Internal policy topology

The typed policy retains provider quota and deployment-override maps as the
future control-plane seam. Their keys are derived from the provider registry:

- `provider:account` is a provider-account quota shared by all mapped models,
  for example `continuum:account`.
- `provider:model:<provider_model_id>` is a provider-model quota, for example
  `tinfoil:model:kimi-k3`.

Quota budgets must use the scope declared by the registry. Deployment
overrides always use the provider-model form because concurrency is tracked per
deployment even when quota is provider-account scoped. The compiled baseline
leaves both maps empty and uses the global deployment limit above.

A future source may supply any combination of request, prompt-token,
completion-token, and cached-token budgets. They are evaluated over the rolling
window, and omitted dimensions remain unmetered. Policy validation rejects
unknown topology keys or invalid limits before a controller can be created.

## Reservation accounting

Prompt reservations use the UTF-8 byte length of the complete serialized
provider-visible request after replacing message image payloads with small
sentinels, plus conservative per-request, per-message, and per-tool template
allowances and the full worst-case expansion of provider-rendered `&amp;` and
`&quot;` attribute entities. It also accounts for normalized JSON delimiter
spacing, Unicode escaping, model-visible tool-call tags, and parsed argument
entries. A shared tokenizer is not used for admission because provider
tokenizers can split identical text very differently. The byte bound counts
JSON keys, numbers, structure, tool schemas, response-format schemas,
documents, and provider template arguments without tokenizing base64 image
bytes. Each explicit or shorthand image part instead uses the selected
deployment's registry bound: Kimi K3 reserves 16,384 prompt tokens per image
and Continuum Kimi K2.6 reserves 4,096. Vision bounds are route metadata, so a
provider preprocessor or model change must update the registry and its tests in
the same release. Video content and provider-internal cached image/embed
carriers are rejected on these image-only public routes; they are not forwarded
with an unbounded or guessed media-token reservation.

An explicit completion limit is multiplied by the request's positive `n`
value; `n` defaults to one. If both completion-limit fields are absent or null,
OpenSecret forwards `completion_default_reservation` as `max_tokens` and
reserves that cap per generated choice and model turn. This structurally bounds
provider extensions such as `ignore_eos` and token allowlists. Forwarded
`min_tokens` raises both the injected cap and reservation whenever it exceeds
the compiled baseline. Non-null explicit limits remain provider-validated and
are never silently repaired.

Output maxima, `n`, `min_tokens`, and `stream` must use their canonical JSON
integer/boolean types. Provider-coercible strings, floats, and numeric stream
flags are rejected before admission so local reservations and response parsing
cannot disagree with the provider. `n` is also capped by
`max_completion_choices` independently of provider quotas, preventing
one deployment slot from expanding into thousands of provider sequences.
Provider-internal KV-transfer parameters
are stripped before estimation and forwarding; clients cannot replace the
rendered conversation with pre-tokenized prompt IDs.

Message, function, and named tool-choice names follow the OpenAI-compatible
1-64 byte `[A-Za-z0-9_-]` contract. Enforcing that boundary prevents provider
templates from entity-expanding or duplicating an unbounded function name
across tool-result messages. Historical assistant tool arguments must be valid
JSON strings no larger than 1 MiB; their normalized JSON rendering is included
in the reservation so exponent-form numbers and other canonicalization cannot
expand behind admission.

The controller initially reserves the full prompt estimate against both prompt
and cached-token dimensions. Terminal usage reconciles prompt, completion, and
cached observations independently. A provider-omitted dimension retains a
conservative reservation; it is never interpreted as an authoritative zero.

OpenSecret is the sole owner of model selection and tool-loop execution.
Tinfoil router-only options (including built-in tool profiles and per-function
auto-continue) are stripped before the provider send so one admitted turn
cannot fan out into hidden provider generations.

## Replica budgeting

Admission state and accounting are intentionally local to one OpenSecret
process or enclave. If a confirmed provider budget is later compiled into the
policy, assign each enclave only its intended share, including headroom for
uneven sticky-routing traffic. Recalculate and review those constants whenever
the replica count, provider entitlement, model mapping, or observed traffic
distribution changes.

This is a rollout-safety boundary, not a distributed quota guarantee. One
enclave can temporarily leave capacity unused while another is saturated.

## Deployment

There is no admission-policy provisioning step for local or Nitro deployments.
OpenSecret constructs the compiled baseline while building `AppState`; a stray
environment variable or `enclave_secrets` row cannot alter it. To change the
policy, edit the baseline constants and typed policy construction, run the
admission and routing test suites, build a new artifact, and restart or roll the
affected processes or enclaves.

This deliberately temporary ownership model avoids treating the encrypted
secrets table as a manual control plane. A future admin API should authenticate
and authorize policy changes, validate the complete policy against the fixed
provider registry, and make rollout/audit semantics explicit before it replaces
the compiled source.
