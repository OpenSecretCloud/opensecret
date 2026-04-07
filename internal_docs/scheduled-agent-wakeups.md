# Scheduled Agent Wakeups and Background Turns

## Status

- **Date:** March 2026
- **Status:** v1 design reviewed; initial implementation landed
- **Audience:** backend / agent / mobile product design
- **Purpose:** capture the agreed architecture for scheduled agent wakeups and record the initial implementation status

This document started as a forward-looking design draft. It now also records the initial implementation that landed on 2026-03-23.

Historical research notes are still preserved below. Any "open questions" inside the section 17 research log should be read as point-in-time notes from before implementation unless they are restated later as still-open items.

## Implementation update (2026-03-23)

The agreed v1 scheduler shape is now wired into the codebase.

Implemented pieces:

- Postgres-backed schedule definitions plus per-occurrence run records via `agent_schedules` and `agent_schedule_runs`
- Structured recurrence for `one_off`, `interval`, `daily`, and `weekly` schedules
- Timezone mode support for `follow_user` versus `fixed`, including recomputation when the user's timezone preference changes
- Agent tools for `schedule_task`, `list_schedules`, and `cancel_schedule`
- Background schedule worker startup in `main.rs`
- Lease-based execution with retries, heartbeat renewal, stale-expiry handling, and partial-output completion behavior
- Scheduled turns reusing the normal agent runtime for tool calls and assistant-message persistence, without inserting fake user messages
- Scheduled reminder push enqueueing with one encrypted preview per turn, using `llama-3.3-70b` summarization plus first-message fallback
- Coverage for recurrence parsing / calculation edge cases including DST-gap handling

Still intentionally out of scope for v1:

- direct client schedule-management APIs beyond the agent tools
- strict per-agent turn serialization beyond the accepted v1 parallel-turn tradeoff
- advanced calendar rules like third Monday / last business day

---

## 1. Core Product Idea

The key design decision is:

**scheduled tasks should wake the agent, not directly send notifications.**

That means a scheduled event is not fundamentally a push job. It is a future agent-triggered turn.

The LLM remains the brain:

- the agent decides whether to message the user at all
- the agent decides what tone and wording to use
- the agent may use normal tools before replying
- the agent may decide the event is no longer relevant and do nothing

Push delivery is a downstream consequence of that background turn, not the primary product abstraction.

This is the main behavioral difference from a traditional reminder system. We are not storing “notification text to send later.” We are storing “instructions for the future agent to consider when that time arrives.”

---

## 2. Example Product Behavior

### 2.1 Wake-up example

User says:

> I want a wake up notification at 8am tomorrow.

Agent says:

> Okay.

The agent schedules a one-off event for the next day at 8am local time. The scheduled payload is **not** the notification text. It is a future-facing instruction for the agent, such as:

> At 8am tomorrow, wake the user up. Check whether they have already been active in the conversation this morning before sending anything. If there is relevant recent context about why they need to wake up, use it.

At 8am, the event fires and wakes the agent in the background.

The agent then has normal context available:

- recent thread history
- memory blocks
- archival memory
- conversation search
- normal tools
- current local time / timezone context

At that point the agent might:

- send nothing because the user already woke up and has been chatting since 7am
- send a personalized wake-up message based on newer context
- check the weather first, then send a more useful message
- send several messages if that is what the normal turn logic decides

### 2.2 Weather-aware wake-up example

User says:

> Wake me up at 8 tomorrow and let me know if I need a coat.

At 8am, the agent wakes, sees the scheduled instruction, calls `web_search`, gets the weather, and then sends:

> Hey, it’s 30 degrees out. Wake up and put on a coat.

This is exactly why the scheduled event should enter the normal agent runtime rather than bypassing it with prewritten push text.

---

## 3. Primary v1 Decisions

### 3.1 Best-effort agent wakeup

The system is **best-effort**, not alarm-clock-grade guaranteed delivery.

If the model stack is down or a dependency fails, the system should retry according to durable queue semantics. But if the full turn never succeeds, we accept that no user-visible notification may be delivered.

This is consistent with the product philosophy:

- the LLM is the decision-maker
- the scheduled event exists to wake the agent
- there is no separate deterministic fallback reminder engine in v1

### 3.2 Internal event, not fake user or tool-result persistence

When a scheduled event fires, it should enter the runtime as a **new internal event type**.

It should **not** be persisted as:

- a fake `user_message`
- a fake `assistant_message`
- a fake `tool_output`
- a synthetic row in existing message tables pretending a tool call stayed open for hours

The internal event should still reuse almost all of the normal runtime behavior, but it is not itself normal chat history.

### 3.3 Reuse the normal agent runtime as much as possible

The scheduled-event turn should reuse the same major machinery as a normal user-driven agent turn:

- thread loading
- memory loading
- prompt assembly
- tool registry
- tool execution chain
- assistant message persistence
- post-turn delivery / push planning

In other words, the internal trigger is new, but nearly all of the downstream execution model should stay aligned with the existing agent flow.

### 3.4 Normal outputs are persisted normally

The **results** of the scheduled turn should be treated as ordinary assistant behavior.

If the agent sends messages, those messages should be stored as real assistant messages.

If the agent calls tools, those tool calls and tool outputs should follow the normal persistence rules used by the existing runtime.

If the agent sends three messages, then calls tools, then sends three more messages, that is allowed. The scheduled trigger should not artificially force a single-message reminder format.

### 3.5 No explicit push tool

There should be no “send push notification” tool in this design.

The agent should keep doing normal things:

- send messages
- search
- consult memory
- reason over recent context

If the scheduled background turn produces assistant messages and there is no live SSE consumer for that turn, then the delivery system should treat those outputs as eligible for push, subject to the user’s push settings.

Push remains delivery plumbing, not agent-facing product logic.

### 3.6 One push per scheduled turn by default

For v1, scheduled background turns should usually produce **at most one push notification event**.

Even if the agent emits multiple assistant messages during the turn, we should derive a single encrypted preview summary for notification delivery.

This keeps scheduled events from producing noisy notification bursts while still preserving the full assistant output in conversation history.

### 3.7 No separate notifier agent

We should **not** create a second autonomous “notifier” agent.

Instead, after the main scheduled turn completes, we should run a lightweight **notification-preview composer** over the turn’s assistant outputs.

That post-processing step is allowed to use a cheap model, but it is not another reasoning agent with independent product behavior.

---

## 4. Scheduling Model

### 4.1 Agent-scoped future wakeups

Scheduled events should belong to an agent identity, not to a generic global notification system.

That means the event targets:

- the main agent, or
- a specific subagent

When the event fires, that exact agent/thread is what wakes up.

### 4.2 Store instructions for the future agent

The stored payload should be agent-oriented instructions, not notification copy.

Good examples:

- “At 8am tomorrow, wake the user up for yoga class. If they have already been active this morning, avoid a redundant wake-up message. If weather seems relevant, check it first.”
- “Tomorrow at 6pm, remind the user to leave for dinner. If recent conversation changed the plan, adapt to the latest context.”

Bad examples:

- “Wake up! Time for yoga!”
- “It’s cold outside, bring a coat!”
- “Reminder: leave now.”

Those bad examples are candidate notification text, not the right stored instruction shape.

### 4.3 One-off first, with limited recurring support in v1

The primary implementation milestone should still be **one-off scheduled events first**.

However, based on current product discussion, the broader v1 design should also be able to support a **limited structured recurrence model** for common cases like:

- every hour
- every morning at 8am
- every Friday at 5pm
- weekdays at 7:30am

What remains out of scope is not recurrence altogether, but rather:

- raw free-form cron syntax as the main product surface
- advanced calendar-style recurrences like “third Monday” or “last business day”

### 4.4 Schedule time is authoritative; message wording is not

The scheduler is responsible for firing the event at the right time.

The agent is responsible for deciding whether and how to turn that event into user-visible output.

This division of responsibility is important:

- time semantics are deterministic infrastructure
- language semantics stay inside the agent runtime

---

## 5. Tool Contract for Scheduling

### 5.1 Agent-facing scheduling tool goal

The scheduling tool should explicitly teach the model that it is writing instructions for its **future self**.

The tool prompt / schema guidance should say that the agent must:

- write in normal human language
- describe the goal and relevant context for the future turn
- mention conditional behavior when appropriate
- avoid drafting final notification copy
- avoid assuming the future state will match the current one exactly

### 5.2 Recommended instruction-writing guidance

The tool contract should push the agent toward guidance like this:

> You are scheduling a future wakeup for yourself. Write instructions for the future agent turn, not the final notification text. Be specific about the goal, timing, relevant context, and any useful conditional behavior. Use normal natural language. Do not optimize for brevity if detail would help the future agent make a better decision.

### 5.3 Suggested base scheduling arguments

The exact schema can be finalized later, but the base shape is roughly:

- `run_at`: absolute scheduled time
- `timezone`: optional user-local timezone override if needed
- `instruction`: detailed future-agent instruction in natural language
- `description`: short operational description for listing/debugging
- `expires_at`: optional staleness cutoff for cases where old reminders stop being useful

The important field is `instruction`, and it should be modeled as the sensitive core payload.

Recurring-specific structured fields are described later in section `19.15`.

### 5.4 Example of a good scheduled instruction

> At 8am local time tomorrow, wake the user up. They want this as a real wake-up nudge, not just a passive reminder. Before sending anything, consider whether they have already been active in chat this morning. If there is recent context about what they need to wake up for, use that context. If helpful, you may check weather or other relevant information before replying.

This is the right shape because it tells the future agent what job it is trying to do without locking it into exact wording.

---

## 6. Runtime Trigger Model

### 6.1 Scheduled event wakeup path

When a scheduled event becomes due:

1. a worker leases the event row
2. the worker claims execution responsibility for that target agent/thread on that server
3. the runtime starts a background agent turn using a scheduled-event trigger
4. the runtime injects hidden event context
5. the agent executes its normal loop
6. normal outputs are persisted
7. post-turn delivery planning decides whether to enqueue push
8. the scheduled event row is completed or retried

### 6.2 Hidden event context

The runtime should inject hidden context that includes things like:

- scheduled event id
- scheduled-for time
- actual fired-at time
- local timezone / local wall-clock representation
- the stored instruction
- optional source message reference or creation context

This hidden context should behave similarly to other runtime-only inputs, but it should not create user-visible history rows by itself.

### 6.3 Prompt shape

The scheduled event should enter prompt assembly in a form that clearly communicates:

- this is an internal scheduled wakeup
- the user did not just send a new live message right now
- the agent should consider the latest context before acting
- it may decide to no-op

The system should make it easy for the agent to understand that the stored instruction came from an earlier planning moment and may need reinterpretation based on newer context.

---

## 7. Delivery and Push Behavior

### 7.1 Push is downstream of the background turn

The scheduled event itself should not enqueue push directly.

Instead:

- the agent finishes its background turn
- the system inspects the assistant outputs from that turn
- if there are user-visible assistant messages and push is enabled, a notification event may be enqueued

### 7.2 Notification preview generation

For the encrypted notification preview text:

- use the existing available **Llama** model as a compression/summarization step
- its job is to compress the turn’s assistant outputs into a short preview while preserving tone and intent
- it must not invent facts not present in the assistant outputs

The summarizer should receive the assistant outputs from that scheduled turn and produce a compact 1-2 sentence preview suitable for encrypted notification display.

### 7.3 Preview fallback behavior

If the Llama summarizer fails, fall back to:

- the **first assistant message** produced in that scheduled turn

This fallback does not need to inspect which substep or tool boundary produced it. If it is the first normal assistant message from the turn, it is the fallback preview source.

### 7.4 Preserve full assistant output in chat history

The push preview is only a delivery summary.

The full assistant outputs of the scheduled turn should still be persisted normally. The notification preview should not replace or reshape the underlying conversation history.

### 7.5 One encrypted preview per scheduled turn

If the agent produced multiple messages, the summarizer should compress across the whole turn and yield one preview.

That preview becomes the encrypted notification text. The provider-visible shell can remain generic, consistent with the current encrypted push model.

---

## 8. Retry, Leasing, and Recovery

### 8.1 Durable queue semantics

Scheduled events need real durable worker semantics, not an in-memory timer.

At minimum this requires:

- due-time polling
- row leasing
- lease expiry recovery
- explicit retry states
- backoff
- attempt counters
- terminal failure handling

### 8.2 Use both lease expiry and explicit retry

We want both failure mechanisms:

1. **lease expiry** for crash recovery
2. **retry status with next attempt time** for known transient failures

That means:

- if a worker crashes mid-turn, another worker can recover the event after lease expiry
- if the runtime returns a transient failure, the current worker should explicitly schedule a retry with backoff

### 8.3 Extend or heartbeat the lease while the turn runs

Scheduled turns may take longer than a small lease interval, especially if they involve tool calls.

The worker should therefore renew the lease while the turn is active so another worker does not mistakenly steal the same event during a long-running but healthy execution.

### 8.4 Per-agent turn serialization

The long-term safest model is still:

- every agent has one serialized turn lane
- scheduled turns join that same lane
- user turns still take precedence where appropriate, but concurrency is controlled centrally

This avoids context corruption, duplicate outputs, and confusing interleaving.

However, the current v1 decision is more relaxed:

- a scheduled wakeup may run with its own lifetime even if a live turn for that same agent is already in progress

So serialization remains the preferred end-state, but it is **not** a blocker for the first implementation pass.

### 8.5 Staleness and expiry

Many scheduled events are time-sensitive.

Examples:

- a wake-up reminder is useful at 8:00am, much less useful at 2:00pm
- a “leave now for dinner” reminder may be stale shortly after it misses its target

So scheduled events should support an explicit expiry or stale-after threshold. Once stale, the system should stop retrying and mark the event terminal.

### 8.6 Best-effort failure model

Because this feature is explicitly best-effort, there is no requirement in v1 to synthesize a system-generated reminder if the LLM never completes successfully.

If all retries fail or the event expires, the event simply fails.

---

## 9. Data Model Intent

### 9.1 Separate scheduled-events table

This feature should use a dedicated scheduled-events table rather than overloading `notification_events`.

`notification_events` should remain the push outbox.

The new scheduled-events table is the source of truth for:

- when to wake the agent
- which agent to wake
- what instruction to inject
- current lease / retry / completion state

### 9.2 Plaintext fields

We should keep operational fields plaintext so PostgreSQL can index, filter, and lease them efficiently.

That likely includes:

- row id / uuid
- user id
- agent id
- conversation id or agent-thread reference if needed
- status
- run time
- next attempt time
- lease owner / lease expiry
- attempt count
- created at / updated at
- optional expiry time
- short operational description
- source message id references where useful

These are infrastructure fields, not the sensitive semantic payload.

### 9.3 Encrypted fields

The main sensitive field is likely:

- `instruction_enc`

That is the actual future-agent instruction and is the most obvious content that should be encrypted at rest.

We may also want an optional encrypted metadata blob for future-sensitive context if we later decide we need more than just the instruction text.

### 9.4 Plaintext minimum principle

We should keep only the minimum necessary fields encrypted.

The goal is:

- preserve queryability, scheduling efficiency, and operational simplicity in Postgres
- encrypt only the sensitive content-bearing parts of the event

At the current intent level, that likely means **the instruction is the primary encrypted field**, while scheduling and operational metadata stay plaintext.

### 9.5 Error storage should be sanitized

Retry / failure diagnostics should avoid storing highly sensitive prompt or model output content in plaintext error columns.

Plaintext `last_error` fields should stay operational and sanitized.

---

## 10. Execution Semantics

### 10.1 The scheduled event itself is not chat history

The scheduled trigger should not appear in user-visible history as if the user just sent a new message.

The user should see only the outputs the agent actually chose to send.

### 10.2 The scheduled event can still lead to rich behavior

Even though the trigger itself is hidden, the turn can still produce:

- multiple assistant messages
- tool calls
- tool outputs
- follow-up reasoning using the normal runtime

This is intentional. Scheduled turns should not be artificially crippled compared with ordinary turns.

### 10.3 No-op is valid behavior

The agent should be allowed to conclude that no user-visible action is needed.

Examples:

- the user is already awake and active
- the plan changed in later conversation
- the event is now irrelevant given newer context

In those cases the event should complete successfully with no assistant message and no push.

---

## 11. Notification Preview Composer

### 11.1 Purpose

The notification-preview composer exists only to create a compact encrypted push preview after a scheduled turn.

It is not a new product-level agent.

### 11.2 Inputs

The composer should consume:

- the assistant messages produced by the scheduled turn
- optional turn metadata like thread target or message ids

It should not independently search, reason, or reinterpret the user’s world beyond compressing the already-produced assistant output.

### 11.3 Model choice

Use the existing available **Llama** model because it is already present and is a strong fit for compression / summarization.

### 11.4 Behavior requirements

The preview composer should:

- preserve tone and intent
- stay faithful to the assistant outputs
- avoid adding new facts
- keep the result short
- target notification-preview length rather than full chat prose

### 11.5 Deterministic fallback

If the composer fails for any reason, the fallback is deterministic:

- use the first assistant message from the turn as the encrypted preview text

This keeps the delivery path robust without adding another complex branch.

---

## 12. Security and Privacy Boundaries

### 12.1 Sensitive data should remain encrypted

The instruction payload is sensitive because it contains semantic user intent and planning context.

That content should be encrypted at rest in the scheduled-events table.

### 12.2 Operational metadata may remain plaintext

We do not need to encrypt every field just because it belongs to a scheduled event.

In particular, fields required for:

- polling
- leasing
- retry management
- indexing
- expiry checks
- list views

should remain plaintext unless they themselves carry sensitive user content.

### 12.3 No plaintext instruction logging

The system should avoid logging:

- decrypted scheduled instructions
- raw notification preview plaintext
- full assistant turn content in operational logs

Logs should stay operational rather than content-rich.

### 12.4 Push encryption remains unchanged in principle

Scheduled turns should continue to use the same encrypted-push architecture already adopted for mobile push delivery.

The only new part is how the preview text is sourced:

- from the assistant outputs of the scheduled turn,
- compressed by Llama,
- with first-message fallback.

---

## 13. Product Scope for the First Implementation Pass

### In scope

- one-off scheduled agent wakeups
- limited structured recurring schedules for common cases (interval / daily / weekly)
- agent-scoped scheduled events for main agent and subagents
- encrypted-at-rest scheduled instructions
- durable leasing and retries
- hidden internal event trigger into the normal agent runtime
- normal persistence of assistant outputs
- one encrypted push preview derived after turn completion
- Llama preview compression with first-message fallback
- no-op outcomes when the agent decides nothing should be sent

### Out of scope for the first pass

- raw cron as the primary product/API surface
- advanced calendar-style recurrences (e.g. third Monday, last business day)
- a separate notifier agent
- a user-facing generic push tool
- persisting the internal scheduled trigger into chat history
- guaranteed alarm-clock delivery semantics
- a deterministic system-generated reminder fallback when the LLM never succeeds

---

## 14. Main Rejected Alternatives

### 14.1 Scheduled events directly enqueue notifications

Rejected because it prevents the agent from adapting to newer context and reduces reminders to prewritten notification copy.

### 14.2 Persist the fired event as a fake user message

Rejected because it pollutes chat history and misrepresents what actually happened.

### 14.3 Persist the fired event as a delayed tool result

Rejected because it is structurally misleading and creates awkward long-lived pseudo-tool state.

### 14.4 Second autonomous notifier agent

Rejected because it adds a second reasoning layer when a lightweight preview composer is enough.

### 14.5 Explicit push-notification tool

Rejected because push should remain a delivery concern, not a first-class agent product surface.

---

## 15. Open Questions for the Next Pass

These are intentionally left for a later code-aware design pass:

- exact table schema and naming
- exact worker / lease heartbeat implementation
- exact agent-turn locking strategy relative to live user turns
- whether `conversation_id` should be stored on the scheduled row or derived from `agent_id`
- how user-visible schedule listing / cancellation UX should work
- how much scheduling metadata should be exposed to clients
- whether multiple due scheduled events for one agent should coalesce before execution
- whether future versions should support richer recurrence families or additional event classes beyond the initial interval/daily/weekly set

Note: the default stale / expiry policy for v1 is now set to **15 minutes**; what remains open is whether some schedule families should override that by default.

---

## 16. Bottom Line

The intended architecture is:

**schedule now -> wake the agent later -> let the agent decide what to do -> optionally push the result**

That preserves the core Maple product idea:

- the LLM is the brain
- reminders are context-aware agent behaviors, not canned notifications
- push is a downstream transport layer
- scheduled instructions are written for the future agent, not for the lock screen

This should be the starting point for the implementation design pass.

---

## 17. Code Research Log

This section is intentionally incremental. It captures code-informed findings as they are discovered, rather than waiting for one final summary.

### 17.1 Research pass 1 — current live agent turn flow and push hook

Relevant files reviewed in this pass:

- `src/web/agent/mod.rs`
- `src/web/agent/runtime.rs`
- `src/push/mod.rs`
- `src/models/notification_events.rs`
- `src/models/notification_deliveries.rs`
- `src/push/worker.rs`

#### What the code does today

The current `agent.message` push path is tightly coupled to the live SSE chat flow.

In `src/web/agent/mod.rs`, `run_agent_chat_task(...)` currently does this:

1. initialize `AgentRuntime`
2. call `runtime.prepare(&input_content)`
3. run `runtime.step(...)` in a loop
4. persist any tool call + tool output rows
5. persist assistant messages with `runtime.insert_assistant_message(...)`
6. attempt to stream each assistant message over SSE
7. if SSE delivery fails, remember only the **first persisted-but-undelivered** assistant message
8. after the turn completes, enqueue **one** push notification from that first missed assistant message

This means the current implementation already has the important product behavior of:

- persisting assistant output first
- treating push as downstream delivery
- sending at most one push per interrupted turn

That is directionally aligned with the scheduled-wakeup design.

#### Important code-level constraints discovered

##### `prepare()` currently persists a user message

`AgentRuntime::prepare(...)` in `src/web/agent/runtime.rs` currently inserts a real `user_messages` row and schedules embedding work.

That is correct for live user chat, but it is **not** correct for scheduled wakeups if the internal scheduled trigger must stay hidden.

This strongly suggests the scheduled-turn implementation will need either:

- a parallel `prepare_internal_event(...)` path, or
- a refactor that separates “build step input” from “persist user message.”

##### Assistant/tool persistence is already reusable

`AgentRuntime` already exposes reusable persistence helpers for the outputs we do want:

- `insert_assistant_message(...)`
- `insert_tool_call_and_output(...)`

Those helpers also already spawn background embedding work for assistant messages, which is good: scheduled turns should benefit from the same memory/indexing behavior as ordinary turns.

##### Current push enqueue expects one `message_id` and one text body

`enqueue_agent_push_for_disconnect(...)` calls `enqueue_agent_message_notification(...)`, which takes:

- a single `message_id`
- a single `message_text`

That is sufficient for the current disconnect flow, but it is narrower than the scheduled-turn design, where one push preview may need to summarize **multiple** assistant messages from one background turn.

##### The push outbox is already durable and leased

The push system already has a strong distributed-worker pattern:

- durable `notification_events`
- per-device `notification_deliveries`
- leasing via `FOR UPDATE SKIP LOCKED`
- explicit retry transitions
- lease ownership checks on writeback
- expiry/cancellation handling

This is a strong conceptual template for scheduled-event execution, even though the scheduled-event table and worker do not exist yet.

#### Early implications for the feature

- scheduled wakeups should likely reuse the existing assistant/tool persistence paths
- scheduled wakeups should **not** reuse the live `prepare(...)` path unchanged
- push for scheduled turns should likely happen **after the whole background turn finishes**, not on the first produced assistant message
- the current push helper shape probably needs either a lower-level enqueue path or a new scheduled-turn-specific wrapper

#### Open questions discovered in this pass

- What is the cleanest hidden-trigger entrypoint into `AgentRuntime`?
- Should scheduled turns share most of `run_agent_chat_task(...)` through an extracted non-SSE turn runner?
- What should `message_id` mean when one push preview summarizes several assistant messages from one turn?
- How do we make one-push-per-turn idempotent across scheduled-worker retries?
- I did **not** find an obvious existing per-agent turn lock in the reviewed code, so where should scheduled-turn serialization live relative to live user turns?

### 17.2 Research pass 2 — context assembly, model hooks, and concurrency gaps

Relevant files reviewed in this pass:

- `src/web/agent/runtime.rs`
- `src/web/agent/signatures.rs`
- `src/web/agent/compaction.rs`
- `src/proxy_config.rs`
- `src/web/openai.rs`
- `src/web/responses/handlers.rs`
- `src/main.rs`

#### What the runtime already gives us for a scheduled turn

`AgentRuntime::build_context(...)` already assembles most of the context we want a scheduled wakeup to inherit.

It currently loads:

- the user’s timezone from `user_preferences`
- the user’s locale from `user_preferences`
- a formatted `current_time` string localized to that timezone
- decrypted persona + human memory blocks
- memory metadata counts
- the latest conversation summary
- recent conversation history
- main-agent onboarding state
- subagent purpose

This is important because it means a scheduled wakeup does **not** need a special prompt stack to feel “normal.” Once it reaches the runtime correctly, it can already inherit most of the same state as a live user turn.

#### Tool-result continuation behavior is already encoded in the runtime

`AgentRuntime::step(...)` already has a continuation mode for follow-up steps after tools run.

It tracks:

- `current_tool_results`
- `previous_step_summary`

and then injects tool-result instructions like:

- this is a continuation of the previous turn
- previous messages are already visible to the user
- silence is the default unless there is genuinely new information

This does **not** directly implement scheduled events, but it shows the runtime already understands “non-fresh-user-message continuation” as a first-class pattern.

That is encouraging for the scheduled-event design, which also wants a hidden internal trigger rather than a fake new user message.

#### `prepare()` currently does more than validation

This pass confirmed that `prepare(...)` currently does all of the following together:

- validates / normalizes the user message
- runs vision pre-processing when needed
- clears tool-result state
- inserts the user message
- triggers background embedding
- runs compaction checks

This means a hidden scheduled-event entrypoint will need a deliberate decision about which of those behaviors to keep.

Most likely:

- keep tool-result reset
- probably keep compaction behavior
- skip user-message insertion
- skip any semantics that assume a new live user-authored message exists

#### LLM helper infrastructure is already flexible enough for a preview composer

There are two useful existing LLM helper patterns in the codebase.

##### Pattern A: direct small non-streaming helper calls

`src/web/responses/handlers.rs::spawn_title_generation_task(...)` already runs a background helper call using:

- model: `llama-3.3-70b`
- `get_chat_completion_response(...)`
- non-streaming completion extraction
- best-effort logging-only failure behavior

This is a strong template for a post-turn preview-composition step.

##### Pattern B: typed DSRS summarization with correction

`src/web/agent/compaction.rs` already implements typed summarization with correction retries.

That path uses:

- `build_lm(...)` from `src/web/agent/signatures.rs`
- typed signatures
- malformed-output correction

This means we already have a reusable pattern if we decide the notification-preview composer should be typed rather than raw string extraction.

#### Llama availability is real, but conditional

`src/proxy_config.rs` confirms that `llama-3.3-70b` is a routed canonical model in this repo.

However, it is only available when `TINFOIL_API_BASE` is configured. Without Tinfoil, the router falls back to Continuum-only models and Llama is not included.

That means the current product decision of “use Llama for preview composition” is viable, but there is an implementation question about what to do in environments where Tinfoil is absent.

#### Current agent model defaults

The main agent runtime currently builds its LM with:

- request model: `kimi-k2-5-agent`
- billing model: `kimi-k2-5`

So the preview composer would be a genuine secondary helper model path, not just a reuse of the main-agent LM.

#### I still did not find an agent turn lock

I searched the agent runtime and broader app state for agent/conversation turn serialization and did **not** find an obvious existing lock, semaphore, or in-flight turn registry for agent chat.

This remains one of the most important implementation gaps discovered so far.

#### Additional structural finding for subagents

`AgentRuntime::new_subagent(...)` already reconstructs the runtime from the subagent UUID by loading:

- the subagent row
- its conversation
- its decrypted purpose

That means scheduled events targeting subagents can likely anchor on subagent identity cleanly without needing to persist every prompt-level field on the scheduled row.

#### Open questions discovered in this pass

- Should the scheduled-event row store `agent_id`, `agent_uuid`, `conversation_id`, or some combination?
- Should the scheduled-event entrypoint run compaction before the turn, the same way live `prepare(...)` does?
- Should the preview composer use the lightweight raw-completion pattern like title generation, or a typed DSRS summarizer with correction?
- What should happen in environments where `llama-3.3-70b` is unavailable because Tinfoil is not configured?
- Where should agent-turn serialization live if the current runtime has no obvious lock/queue for it?

### 17.3 Research pass 3 — agent signature shape and queue-schema implications

Relevant files reviewed in this pass:

- `src/web/agent/signatures.rs`
- `src/web/agent/runtime.rs`
- `migrations/2026-03-07-120000_push_notifications_v1/up.sql`

#### Current agent signature shape is input-centric

The main agent signature in `src/web/agent/signatures.rs` currently takes these top-level inputs:

- `input`
- `current_time`
- `agent_kind`
- `subagent_purpose`
- `persona_block`
- `human_block`
- `memory_metadata`
- `previous_context_summary`
- `recent_conversation`
- `available_tools`
- `is_first_time_user`

There is currently **no explicit field for trigger kind** such as:

- live user message
- tool-result continuation
- scheduled internal event

So if we want scheduled events to be a first-class trigger type at the prompt layer, we will likely need one of two approaches:

1. encode the scheduled-event semantics entirely inside the `input` string, or
2. extend the agent signature with an explicit field such as `turn_source`, `event_kind`, or equivalent

This is an important design choice because it affects how visible and stable the scheduled-event concept becomes in prompt assembly.

#### The runtime already supports first-step raw string input

`AgentRuntime::step(...)` accepts a plain `user_message: &str` for the first step after `prepare(...)` returns the normalized text.

That means a scheduled internal event could probably enter the runtime as a first-step string without needing to fake a `MessageContent` object.

So there is already a plausible implementation path where the scheduled-event worker:

- skips `prepare(...)`
- builds a hidden event input string
- feeds that directly into `step(..., is_first_step = true)`

Whether that is the final shape depends on how much explicit structure we want in the signature.

#### Retry and correction behavior already exists for agent outputs

`call_agent_response_with_retry_and_correction(...)` in `src/web/agent/signatures.rs` already provides:

- up to 3 LM attempts
- correction on parse failures
- a fallback correction prompt that reshapes malformed outputs rather than generating new content

This means scheduled turns would inherit the same output-hardening behavior as live turns if they reuse the same signature path.

That is useful for the best-effort design: the scheduled worker should not need a separate output-repair mechanism if it is truly using the same runtime.

#### Push schema confirms a separation we should preserve

The push migration confirms that `notification_events` is structurally a delivery/outbox table, not a scheduler table.

Notable fields:

- `payload_enc`
- `not_before_at`
- `expires_at`
- `cancelled_at`
- one delivery row per device with lease/retry state

This reinforces the design decision already captured earlier:

- `notification_events` should remain the push outbox
- scheduled-agent wakeups should have their own source-of-truth table

Even though `notification_events.not_before_at` exists, it is the wrong abstraction for “wake the agent and let it decide what to do.”

#### New open questions from this pass

- Is it better to add an explicit `event_kind` / `turn_source` field to the agent signature now, or keep scheduled-event semantics hidden in the first-step input text?
- If scheduled turns bypass `prepare(...)`, what is the cleanest place to run the subset of prepare-time behavior we still want, like compaction and tool-state reset?
- Should the scheduled-event worker persist a turn-level execution record for idempotency / observability, even if the trigger itself stays out of chat history?

### 17.4 Research pass 4 — preview payload and delivery-shape details

Relevant files reviewed in this pass:

- `src/push/mod.rs`
- `src/web/agent/mod.rs`

#### The current encrypted preview payload is message-centric

`NotificationPreviewPayload` currently contains:

- `notification_id`
- `message_id`
- `kind`
- `title`
- `body`
- `deep_link`
- `thread_id`
- `sent_at`

This is important for scheduled turns because the current payload model assumes a notification is anchored to **one message id**.

That works naturally for the current disconnect flow, where the preview is derived from one first-undelivered assistant message.

It is a bit more awkward for scheduled turns, where the preview may summarize an entire background turn.

#### Current preview text budget is very small

`src/push/mod.rs` sets:

- `PUSH_PREVIEW_BODY_MAX_BYTES = 180`

and `normalize_preview_body(...)` currently:

- collapses whitespace
- truncates to that byte budget
- adds an ellipsis when needed

This means the Llama preview composer should target a genuinely short result. Even if we ask for 1-2 sentences, the implementation still needs to respect this small body budget.

#### Deep-linking already distinguishes main vs subagent

The current push helper already derives:

- main agent deep link: `opensecret://agent`
- subagent deep link: `opensecret://agent/subagent/<uuid>`

and matching `thread_id` values.

That is useful because a scheduled-turn push can likely reuse the same targeting rules without inventing a new notification routing system.

#### Current scheduled-turn preview implication

For scheduled turns, if we keep the existing preview payload shape, then we probably need to choose a canonical message id for the turn, most likely:

- the first assistant message of the scheduled turn, or
- the last assistant message of the scheduled turn

The first-message option aligns better with the already-planned fallback behavior and with the current disconnect path, which also anchors on the first user-visible missed message.

#### Additional open questions from this pass

- Should scheduled-turn notifications keep the existing message-centric payload shape and simply use the first assistant message as the canonical anchor?
- Do we want a future turn-level identifier in preview payloads, or is that unnecessary if one notification maps to one scheduled turn?
- Should the preview composer enforce a tighter output limit than “1-2 sentences” so it naturally fits the 180-byte cap?

### 17.5 Research pass 5 — concrete worker numbers and missing lease-heartbeat support

Relevant files reviewed in this pass:

- `src/push/worker.rs`
- `src/models/notification_deliveries.rs`

#### Current push worker defaults

The existing push worker uses these concrete values:

- batch size: `32`
- lease TTL: `60` seconds
- poll interval: `3` seconds
- max concurrency: `8`
- max attempts: `8`

Retry backoff is exponential and currently caps at `480` seconds:

- attempt 1 -> `15s`
- attempt 2 -> `30s`
- attempt 3 -> `60s`
- attempt 4 -> `120s`
- attempt 5 -> `240s`
- attempt 6+ -> `480s`

These values are useful reference points for scheduled-event execution, but scheduled agent turns will probably need longer-running lease behavior than push deliveries do.

#### Important limitation: the existing worker does not heartbeat or extend leases

The push worker leases a delivery row once and then processes it. There is no lease-renewal path because provider send attempts are expected to be short.

That is likely **not** enough for scheduled agent turns, which may:

- call tools
- wait on network operations
- run multiple LLM steps
- take meaningfully longer than a push send

So the scheduled-event worker probably needs one of these patterns:

- an explicit lease heartbeat / renewal update, or
- a substantially longer lease window plus separate stuck-turn handling

The first option still looks cleaner.

#### No advisory-lock pattern found in current code

I searched for advisory-lock / turn-lock patterns and did **not** find existing Postgres advisory-lock usage or an equivalent in-memory agent turn registry in the current agent code.

This strengthens the earlier conclusion that scheduled turns will likely need a brand-new serialization mechanism rather than plugging into an existing one.

#### Additional open questions from this pass

- Should scheduled-event execution use the same general lease-state machine shape as push deliveries, but with a lease-renewal API added?
- What should the initial lease TTL be for a scheduled turn, given that tool-augmented agent turns can be much longer than provider sends?
- Should turn serialization be implemented at the database level, in app memory, or both?

### 17.6 Research pass 6 — current agent route surface

Relevant files reviewed in this pass:

- `src/web/agent/mod.rs`

#### Current agent API surface

The current agent router exposes:

- `/v1/agent`
- `/v1/agent/init`
- `/v1/agent/items`
- `/v1/agent/chat`
- `/v1/agent/subagents`
- `/v1/agent/subagents/:id/chat`
- related item and delete routes

There is currently **no** schedule-specific API surface yet for:

- create scheduled task
- list scheduled tasks
- cancel scheduled task

That is expected, but it is now confirmed from code.

#### Current routing is gated to local/dev

The current agent router is only enabled when `app_mode` is `Local` or `Dev`.

That means any initial schedule API work in this area will naturally follow the same experimental surface unless we intentionally broaden that exposure later.

#### Implication for the feature

If we introduce schedule endpoints as part of the agent API, the most natural home appears to be under the existing agent surface, likely alongside main-agent and subagent routes rather than under push routes.

That fits the product model already captured earlier: scheduled events are agent behavior, not push-device plumbing.

#### Open questions from this pass

- Should scheduling routes live under `/v1/agent/...` and `/v1/agent/subagents/:id/...`, or should creation happen only via agent tool use at first?
- Do we want a direct client-management surface for list/cancel in v1, or is the first pass tool-driven only?
- If these remain under the current experimental agent router, is that sufficient for the intended rollout path?

---

## 18. Decisions Captured During Review

These are user-confirmed decisions made after the research pass.

### 18.1 Concurrency with live turns in v1

For v1, a scheduled wakeup may run with its **own lifetime** even if that same agent is already mid-turn on a live conversation.

That means:

- we are **not** blocking v1 on solving agent-turn merging
- we are **not** blocking v1 on building a strong per-agent serialization mechanism first
- parallel execution for this edge case is acceptable in v1

This should be treated as an explicit temporary product tradeoff, not an accidental bug.

### 18.2 Partial-failure policy after user-visible output

Normal completion-level retries inside the agent runtime can stay as they are.

However, if a scheduled turn has already produced and persisted user-visible assistant output, and then later fails after exhausting its normal retries, v1 should:

- keep the already-produced output
- treat the scheduled event as effectively **completed**
- **not** retry the entire scheduled event from the top

Rationale:

- retrying after user-visible output risks duplicate or confusing messages
- preserving partial useful output is better than replaying the whole scheduled wakeup

This implies an important execution rule for v1:

- scheduled-event retries are appropriate when the turn remains effectively silent
- once the user has already received meaningful output from that scheduled turn, the system should prefer completion over replay

### 18.3 Default expiry policy

The default stale/expiry cutoff for scheduled wakeups in v1 should be:

- **15 minutes**

This is especially appropriate for wake-up-style events where usefulness drops quickly after the intended time.

### 18.4 Client/API surface for v1

The first pass should be:

- **tool-driven only**

That means we do **not** need a broad direct schedule-management API surface in the initial implementation just to validate the scheduled-agent-wakeup model.

---

## 19. Recurring Schedule Model

Recurring schedules are importantly different from one-off wakeups.

For a one-off event, the system stores one future wakeup and runs it once.

For a recurring event like:

> Every morning at 8am, look up the weather and tell me.

the system should **not** be modeled as an infinite stream of pre-created future tasks. Instead, it should be modeled as a durable recurring schedule definition that produces concrete occurrences over time.

### 19.1 Do not use literal OS cron jobs

We should not depend on host-level cron or per-user OS cron entries.

This should stay an application-level scheduler with Postgres as the durable source of truth, just like the broader scheduled-agent-wakeup design.

The recurrence engine may use cron-like logic internally, but the product abstraction should remain:

- store schedule definition
- compute next due occurrence
- wake the agent for that occurrence
- update the schedule to the next occurrence

### 19.2 Two-layer model: schedule definition plus occurrences

The clean recurring model is:

1. **Recurring schedule definition**
   - which agent it belongs to
   - recurrence rule
   - timezone
   - encrypted future-agent instruction template
   - default expiry / stale window
   - active / cancelled state

2. **Concrete occurrence / run**
   - this specific 8am firing
   - scheduled-for timestamp
   - lease / retry state
   - execution result / terminal status

This matters for several reasons:

- recurring schedules need stable long-lived identity
- each occurrence needs its own retry/lease behavior
- occurrence-level state should not mutate the canonical recurrence definition
- observability and idempotency are easier when each firing is concrete

### 19.3 Recurrence should be stored in local-time semantics

For human schedules like:

- every day at 8am
- weekdays at 7:30am
- every Monday at 9am

the important semantic is **wall-clock local time**, not a fixed UTC interval.

So the durable schedule definition should preserve:

- the user’s IANA timezone
- the local recurrence rule

Example:

- timezone: `America/Chicago`
- recurrence: daily at `08:00`

The resulting UTC timestamp will naturally change across DST transitions, but the user still experiences “8am every morning,” which is what we want.

### 19.4 Prefer structured recurrence over raw cron as the primary product shape

For v1, I would strongly prefer a structured recurrence model over exposing raw cron expressions to the agent or user.

For example, represent recurring schedules as fields like:

- frequency: daily / weekly
- local_time: `08:00`
- weekdays: optional set
- timezone: IANA timezone

rather than asking the model to write raw cron strings.

Reasons:

- easier for the agent to author correctly
- easier to validate
- clearer DST behavior
- fewer parsing edge cases
- better product ergonomics for later client surfaces

If needed, we can still compile that structured rule into cron-like evaluation logic internally.

### 19.5 Compute the next occurrence from the schedule, not from “now”

The next firing time should be computed from the schedule definition and the last scheduled occurrence, not by repeatedly adding a fixed duration to the current time.

This avoids drift.

For example, “every morning at 8am” should not gradually wander because a worker happened to process yesterday’s run at 8:02am.

The schedule should still target the next proper local 8:00am occurrence.

### 19.6 Materialize occurrences one at a time

We should not eagerly create months of future runs.

Instead, the scheduler should typically:

- maintain the next due occurrence for the recurring schedule
- when that occurrence is claimed, create or mark a concrete run
- after terminal completion / skip, compute and persist the next due occurrence

This keeps state smaller and makes edits/cancellation much simpler.

### 19.7 Recurring schedules should still wake the agent, not send canned reminders

The recurring model does **not** change the core product philosophy.

Each occurrence should still wake the agent as a normal background turn.

So for:

> Every morning at 8am, look up the weather and tell me.

the recurring schedule should store a future-agent instruction template like:

> Each morning at 8am local time, check the weather for the user and send a concise useful morning weather update. If there is newer context that changes what would be helpful, adapt to it.

At each occurrence, the runtime should inject occurrence-specific metadata like:

- this occurrence was scheduled for 2026-03-24 08:00 America/Chicago
- this is part of a recurring schedule
- current local time is ...

The agent then decides what to do with current context just like a one-off scheduled wakeup.

### 19.8 Backfill policy matters a lot for recurring schedules

Recurring schedules raise an important question:

If the system is down or delayed, should it run missed occurrences later?

For the kind of recurring schedules we are discussing, the default v1 answer should be:

- **no large backfills by default**

Instead:

- if the occurrence is still inside its stale window, it can run late
- if it is past the stale window, mark that occurrence skipped/expired and move on to the next recurrence

This prevents nonsense behavior like sending several stale morning weather notifications after the fact.

### 19.9 DST and ambiguous-time policy must be explicit

Recurring schedules have timezone edge cases that one-off schedules mostly avoid.

At minimum, the implementation needs explicit behavior for:

- daylight saving time spring-forward gaps
- daylight saving time fall-back repeated hours

For common cases like 8am daily reminders, the behavior is straightforward: fire at local 8am and let the UTC time shift with DST.

For rarer ambiguous times, we should document a policy rather than leave it accidental.

### 19.10 Cancellation semantics

Cancelling a recurring schedule should mean:

- stop generating future occurrences
- cancel any pending unstarted occurrence rows if they already exist
- do not affect already completed historical runs

This is another reason the schedule-definition / occurrence split is helpful.

### 19.11 Relationship to v1 tool-driven scope

Even if v1 is tool-driven only, recurring schedules still need enough internal structure for the agent to create them safely.

That means the schedule tool should be able to distinguish:

- one-off wakeup
- recurring wakeup

without forcing the agent to encode every recurrence as a raw cron string.

### 19.12 Recommended v1 stance on recurrence

Given the current product discussion, the safest v1 recurrence shape is:

- daily / weekly recurring schedules only
- timezone-aware local-time execution
- structured recurrence fields, not raw cron strings
- one concrete occurrence at a time
- stale occurrences skipped instead of heavily backfilled
- same background-agent execution model as one-off events

That would cover common product cases like:

- every morning at 8am, send weather
- weekdays at 7am, remind me to wake up
- every Sunday evening, remind me to plan the week

without taking on the full complexity of arbitrary cron semantics immediately.

### 19.13 Distributed scheduling across multiple OpenSecret servers

This system must assume that multiple OpenSecret servers may be running the same code concurrently against the same Postgres database.

That means recurring scheduling must be **database-coordinated**, not process-local.

Important implications:

- no server-local in-memory timers as the source of truth
- no assumption that one server “owns” a user’s schedules
- no schedule state that only exists in process memory

Postgres must remain the durable coordination plane.

#### Recommended distributed model

For both one-off and recurring schedules:

- workers on any server may poll for due rows
- claiming work must happen through row leasing / transactional state changes
- lease ownership must be checked again when writing terminal results

For recurring schedules specifically, there are two related coordination problems:

1. **Who claims the due occurrence for execution?**
2. **Who advances the recurring schedule to its next occurrence?**

Both need to be safe under concurrent workers on different servers.

#### Suggested safety pattern for recurring schedules

The safest shape is:

- keep the recurring schedule definition row in Postgres
- when an occurrence is due, claim/update it transactionally
- generate a concrete occurrence/run row with a unique occurrence key
- advance the schedule definition to the next due local-time occurrence in the same transaction, or under the same row lock

To avoid duplicates, recurring occurrences should have a uniqueness guarantee like:

- unique `(schedule_id, scheduled_for_at)`

or another stable occurrence identity derived from the recurrence.

That way, even if two servers race, the database prevents duplicate occurrence creation.

#### Reuse the existing push-worker philosophy

The current push system already uses the right distributed philosophy:

- `FOR UPDATE SKIP LOCKED`
- lease owner fields
- lease expiry
- compare-and-set style writeback

Scheduled-agent wakeups should follow the same philosophy, even if the exact schema differs.

#### Important consequence for future turn serialization

Because multiple OpenSecret servers may execute turns, any future “one active turn per agent” policy cannot rely only on in-memory locks.

If we later tighten serialization beyond the v1 parallel-turn tradeoff, that coordination will need to be database-backed or otherwise distributed-safe.

### 19.14 Timezone-following versus fixed-timezone schedules

User timezone updates matter a lot for human schedules like:

- wake me up at 8am
- every morning at 8am, tell me the weather
- remind me at 6pm to leave

These often mean:

- “8am where I am”

not:

- “8am forever in the timezone I happened to be in when I created the schedule”

So the scheduler should support two distinct timezone behaviors:

1. **Floating / follow-user-timezone schedules**
   - the schedule follows the user’s current timezone preference
   - best default for personal routine reminders and wake-up-style schedules

2. **Fixed-timezone schedules**
   - the schedule stays anchored to a specific IANA timezone
   - appropriate when the user explicitly anchors it, e.g. “9am Eastern”

This distinction is standard and important. Calendaring systems often make the same separation between floating local-time events and explicitly zoned events.

#### Recommended product default

For the kind of personal schedules we are discussing, the default should usually be:

- **follow the user’s timezone preference**

unless the user or agent explicitly anchors the schedule to a fixed timezone.

#### What the timezone update tool should do

If the user’s timezone preference is updated, the system should proactively recompute future schedules that are marked as:

- active, and
- follow-user-timezone

It should **not** touch:

- historical completed occurrences
- cancelled schedules
- schedules explicitly pinned to a fixed timezone

Because the design already prefers one concrete occurrence at a time, this recalculation should stay relatively cheap:

- update the schedule definition’s `next_due_at`
- update or replace any unclaimed pending future occurrence for that schedule if one already exists

There is no need to rewrite a large history of future rows if we keep materialization bounded.

#### Important schema implication

For follow-user-timezone schedules, we cannot store only an absolute UTC timestamp and expect timezone changes to work correctly.

We also need to preserve the **local-time intent**, for example:

- local recurrence rule
- local clock time
- timezone mode (`follow_user` vs `fixed`)
- fixed timezone value when applicable

Then `next_due_at` becomes a computed operational field derived from that intent.

#### One-off schedules

This distinction also applies to one-off schedules.

If the product wants a one-off “8am tomorrow” wakeup to follow the user when they travel, then that one-off event also needs to preserve local-time intent rather than only a resolved UTC timestamp.

If instead a one-off is meant to stay pinned to the original timezone, it should be marked fixed.

So the schedule tool should ideally decide this explicitly when creating the event, rather than leaving the system to guess later.

### 19.15 Agent-friendly v1 recurrence parameters

The scheduling interface should be simple enough that the agent can reliably map common user intent into correct parameters without being forced to invent raw cron syntax.

The right v1 model is not “free-form cron strings.” It is a small structured recurrence vocabulary that covers the common cases well.

#### Recommended v1 recurrence families

Support these recurring families in v1:

1. **Interval**
   - example: every hour
   - fields like:
     - `every_n`
     - `unit` = `hours`

2. **Daily wall-clock**
   - examples: every morning at 8am, every day at 6pm
   - fields like:
     - `local_time`
     - timezone behavior

3. **Weekly wall-clock**
   - examples: every Friday at 9am, weekdays at 7:30am
   - fields like:
     - `weekdays`
     - `local_time`
     - timezone behavior

This already covers most of the common recurring schedules we care about.

#### Example parameter shape

Conceptually, a tool/API shape like this is enough for v1:

- `schedule_kind`: `one_off` | `recurring`
- `instruction`: future-agent instruction text
- `description`: short operational summary
- `timezone_mode`: `follow_user` | `fixed`
- `fixed_timezone`: optional IANA timezone when pinned
- `stale_after_minutes`: optional override

For one-off:

- `local_date`
- `local_time`

For recurring interval:

- `recurrence_type`: `interval`
- `every_n`
- `interval_unit`: `hours`

For recurring daily:

- `recurrence_type`: `daily`
- `local_time`

For recurring weekly:

- `recurrence_type`: `weekly`
- `weekdays`
- `local_time`

That is much easier for the agent to use correctly than arbitrary cron.

#### How common examples map

- “every morning at 8am”
  - recurring
  - daily
  - `local_time = 08:00`
  - `timezone_mode = follow_user`

- “every Friday at 5pm”
  - recurring
  - weekly
  - `weekdays = [friday]`
  - `local_time = 17:00`

- “every hour”
  - recurring
  - interval
  - `every_n = 1`
  - `interval_unit = hours`

- “every weekday at 7:30am”
  - recurring
  - weekly
  - `weekdays = [monday, tuesday, wednesday, thursday, friday]`
  - `local_time = 07:30`

#### Important distinction: interval versus wall-clock recurrence

“Every hour” is fundamentally different from “every day at 8am.”

- interval schedules are duration-based
- daily/weekly schedules are wall-clock based

That distinction should be explicit in the schedule definition, because it affects DST handling, drift, and how the next occurrence is computed.

#### Is this standard practice?

Yes.

It is very normal to:

- own the schedule schema and distributed execution model
- use a limited structured recurrence model for product ergonomics
- defer complex calendar recurrences until later
- rely on a recurrence library internally rather than exposing raw recurrence syntax directly

This is a sound and scalable engineering approach, especially for a multi-server application using Postgres as the coordination plane.
