# Application logs and request correlation

The binary and enclave entrypoint default to `RUST_LOG=warn,opensecret=info`.
Dependencies stay at warning level; application events remain visible. For a
focused investigation, keep the application root enabled and add a module:

```sh
RUST_LOG=warn,opensecret=info,opensecret::web::responses=debug
```

An inherited `RUST_LOG` overrides the default too. If trace IDs are missing,
check the launching shell's filter: `RUST_LOG=warn` disables the info-level
request spans. Set `RUST_LOG=warn,opensecret=info` on the managed backend start
command to retain them without changing generated environment files.

Logs remain line-oriented tracing text with ANSI disabled. Local stdout is
line-buffered. This change does not add a metrics endpoint, exporter, collector,
or JSON log format. `trace` and `debug` are still host-visible logs, not private
channels for decrypted content.

## Follow one request

Every incoming HTTP request receives a fresh server-generated, 32-character
hexadecimal `trace_id` on its `http_request` span. Search for that field across
the selected time window and all relevant origin streams. Never use an account,
session, token, client-supplied request ID, or provider ID as a trace ID.
Incoming tracing headers are not trusted or copied into these fields, and the
ID is not added to client or provider HTTP headers.

The span records a standard method and the router's matched route template.
Unknown methods become `OTHER`, unmatched routes become `unmatched`, and raw
paths, query strings, headers, and bodies are excluded. Transport V2's internal
dispatch adds a `logical_request` child span under the same outer trace. Its
logical route template is backend diagnostic metadata, not decrypted content.

The request span surrounds middleware and handlers, then follows response-body
polling and destruction. Request-owned background inference, storage, title,
usage, recovery, and email tasks capture the current span when spawned, so
their events remain correlated after the client body ends. Existing inference
request/execution/attempt IDs and response IDs keep their separate meanings.
Periodic service maintenance and startup jobs do not invent request IDs.

`HTTP response body finished` records `status`, `elapsed_ms`, and one outcome:

| Outcome | Meaning |
| --- | --- |
| `body_complete` | The application produced the full HTTP body. |
| `body_error` | Body production returned an error. Its payload is not logged. |
| `body_dropped` | The body was dropped before completion. |

These events do not prove client receipt, inference success, cancellation of
background execution, or storage commit. Use the corresponding execution and
storage terminal events for those conclusions. Successful health probes and
early drops are debug-level summaries; body errors and server-error statuses
remain warnings. A normal response otherwise has one info-level body summary.
Transport V2 has a summary for its logical response and its outer encrypted
response, distinguishable by span context.

## Add or change logging

- The confidentiality boundary is content and secret material, not PII in
  general. User/project/client/response IDs, email addresses, API-key display
  names (never the actual key), model/provider names, status, counts, timings,
  and other non-content request parameters are useful diagnostic metadata.
  Retain them when relevant, with bounded/escaped string fields.
- Never log user/enclave/session keys, mnemonics, passwords, tokens, credentials,
  or plaintext that belongs encrypted at rest: prompts, completions, assistant
  messages/reasoning, conversation titles, instructions, KV keys/values,
  tool arguments/results, or media content. Being encrypted during transport
  does not make the decrypted value safe to log.
- Select metadata fields explicitly. Whole records, claims, headers, OAuth
  documents, provider bodies, KMS output, and arbitrary error text may mix
  allowed metadata with prohibited content. Avoid dumping those containers;
  keep the useful fields without retaining their payloads.
- Use `observability::error_kind` for supported Rust error chains, or a narrow
  typed classifier for a particular boundary. It deliberately returns a fixed
  fallback for unknown types. Add a typed category when more detail is needed;
  do not fall back to `Display`/`Debug` or parse arbitrary error messages.
  `Result::expect`/`unwrap` and a failed `main -> Result` also print error
  `Debug`; erase retained secret buffers and parse text before those paths.
- Use `#[tracing::instrument(skip_all)]` with explicit safe fields for operation
  spans. Normal awaited calls inherit context. Detached futures need
  `.in_current_span()`; blocking closures need a captured span and `in_scope`.
  Never hold an entered-span guard across an `await`.
- Expected malformed input, expired credentials, missing records, and client
  disconnects are not automatically infrastructure errors. Preserve warnings
  for actual provider unavailability, storage failure, and incomplete execution.
- Keep one useful failure at the owning boundary when possible. Default info
  should not emit per-token/per-delta progress. Temporary debug overrides must
  retain the same privacy rules.

The database startup span records configured maximum/minimum pool sizes,
elapsed initialization time, and actual connected/idle counts on success. On
failure it records a bounded class and time before the existing startup panic.
The pool's worker error handler also emits typed categories instead of raw
PostgreSQL connection or validation diagnostics.
Pool limits, retries, timeouts, and proxy behavior are unchanged. A pool timeout
alone does not establish PostgreSQL saturation or a connection leak.

## Validation boundaries

The correlation tests exercise concurrent routing, untrusted metadata,
background work, logical dispatch, frame/trailer preservation, early drop, and
body errors. Privacy tests use synthetic payload-bearing OAuth, JWT, Resend,
database, JSON, UTF-8, KMS, and derivation-path failures. Run the backend's
normal validation skill before submitting changes; use a pinned SDK against
the managed local backend for encrypted protocol checks.

The parent logger in the pinned Nitro toolkit still emits socket chunks as
CloudWatch events. A CloudWatch event is therefore not necessarily one complete
application line, and this backend change does not repair event framing or
delivery loss. Reassemble per stream before counting lines or extracting trace
fields. Deployment/PCR validation and live log verification remain separate.

The async span rules follow the upstream [tracing Instrument documentation](https://docs.rs/tracing/latest/tracing/trait.Instrument.html).
