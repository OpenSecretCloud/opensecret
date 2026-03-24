# Maple Emoji Reactions

**Date:** March 2026  
**Status:** Implemented  
**Related Docs:**
- `internal_docs/maple-agent-memory-architecture.md`
- `docs/encrypted-mobile-push-notifications.md`

## Implementation Notes

- `reply_reaction` is only available on the first assistant step of a turn.
- Continuation turns after tool results omit the `reply_reaction` field from the model output template entirely.
- User reaction mutation routes rely on the HTTP response; separate SSE fan-out for those mutations is out of scope in v1.

---

## 1. Goal

Add first-class emoji reactions to Maple conversations for both directions:

- users reacting to assistant messages
- assistants reacting to the user message they are replying to

This should be implemented as real message metadata in storage, API objects, and SSE, while also being rendered into the Maple prompt transcript so the model can see reaction context.

---

## 2. Product Shape

### 2.1 Supported v1 Behavior

Each message can have at most one reaction from the opposite side:

- a `user_messages` row may hold one assistant reaction
- an `assistant_messages` row may hold one user reaction

This matches the current Maple chat model and keeps the first implementation small.

### 2.2 Prompt Rendering

Reactions should appear in the generated prompt transcript header, not as separate text in the message body.

Example:

```text
[user @ 10:42 AM | ❤️]: i got the job
[assistant @ 10:43 AM | 🎉]: NO WAY
```

The reaction glyph by itself is enough. We do not need extra wording like `assistant reacted`.

---

## 3. Why Reactions Should Be First-Class

Reactions should not be hidden inside prompt-only formatting or conversation-level metadata.

They should be first-class because we need them to work consistently across:

- database persistence
- normal item fetch/list APIs
- Maple SSE live delivery
- prompt construction for future turns

The prompt rendering is a consumer of reaction data, not the source of truth.

---

## 4. Storage Plan

Use direct nullable columns on the existing shared message tables:

```sql
ALTER TABLE user_messages
    ADD COLUMN assistant_reaction TEXT;

ALTER TABLE assistant_messages
    ADD COLUMN user_reaction TEXT;
```

### 4.1 Why Direct Columns

This is the smallest change that fits the current product requirements:

- one reaction per message
- only the opposite side may react
- no reaction history required
- no multi-user aggregation required

A separate `message_reactions` table can be introduced later if Maple needs multiple reactions, audit history, or multi-actor support.

### 4.2 Encoding

Use PostgreSQL `TEXT`.

- PostgreSQL already supports UTF-8 text well
- emoji do not require a special column type
- we do not need encryption for reactions in v1

The main complexity is validation and normalization of multi-codepoint emoji, not storage type.

---

## 5. API Model Plan

Extend `ConversationItem::Message` with a first-class reaction field.

Conceptually:

```rust
Message {
    id,
    status,
    role,
    content,
    reaction: Option<String>,
    created_at,
}
```

Interpretation:

- for `role = "user"`, `reaction` means the assistant's reaction
- for `role = "assistant"`, `reaction` means the user's reaction

This keeps the API simple for clients and avoids separate reaction wrapper objects in v1.

### 5.1 Surfaces Covered

Once threaded through shared message conversion, reactions will flow through:

- `/v1/agent/items`
- `/v1/agent/items/:item_id`
- `/v1/agent/subagents/:id/items`
- `/v1/agent/subagents/:id/items/:item_id`
- shared conversation item APIs where those conversation rows are visible

---

## 6. Assistant Reaction Generation

Assistant reactions should be a first-class Maple agent output, not a tool call.

Current Maple output is:

```rust
messages: Vec<String>
tool_calls: Vec<AgentToolCall>
```

Implemented first-step Maple output:

```rust
messages: Vec<String>
reply_reaction: String
tool_calls: Vec<AgentToolCall>
```

The model uses `""` when no first-step reaction is intended.

Continuation turns after tool results use a reduced output shape with no `reply_reaction` field.

### 6.1 Why Not a Tool

Normal Maple tools currently imply all of the following:

- execute a tool implementation
- inject a `[Tool Result: ...]` block into the next model turn
- persist `tool_call` and `tool_output` rows into transcript history

That behavior is correct for real tools, but it is the wrong shape for emoji reactions.

An assistant emoji reaction is closer to a first-class chat output, like a message, than to a tool execution.

### 6.2 Targeting Semantics

`reply_reaction` should implicitly target the current user message for the turn.

No explicit message id needs to be exposed to the model in v1.

---

## 7. Runtime Plan

### 7.1 User -> Assistant Flow

When a user sends a Maple message:

1. persist the user message as normal
2. retain the inserted user message id/uuid in runtime state for the current turn
3. run the Maple agent loop
4. persist assistant `messages` as normal assistant message rows
5. on the first assistant step only, if `reply_reaction` is present, update the just-inserted user message row with `assistant_reaction`

### 7.2 Why This Works Well

- the model only needs to decide whether to react and which emoji to use
- runtime already knows which user message is being replied to
- no transcript noise is created from fake tool rows

---

## 8. User Reaction Mutation Plan

Add Maple-specific mutation endpoints for reacting to assistant messages.

Suggested shape:

- `POST /v1/agent/items/:item_id/reaction`
- `DELETE /v1/agent/items/:item_id/reaction`
- `POST /v1/agent/subagents/:id/items/:item_id/reaction`
- `DELETE /v1/agent/subagents/:id/items/:item_id/reaction`

Expected behavior:

- only assistant message items can receive a user reaction through these routes
- `POST` sets or replaces the reaction
- `DELETE` clears it

Main-agent mutation routes should remain under `/v1/agent/...` because the public conversations API intentionally hides the main-agent thread.

---

## 9. SSE Plan

Maple SSE needs to become more structured so reactions can update live.

### 9.1 Current Limitation

Current `agent.message` events only send:

```json
{ "messages": [...], "step": N }
```

That is too thin for live reactions because clients do not receive message item ids.

### 9.2 Proposed Changes

#### `agent.message`

Include the persisted assistant message id with each delivered assistant message.

Conceptually:

```json
{
  "message_id": "uuid",
  "messages": ["..."],
  "step": 0
}
```

If Maple continues to emit one assistant message per event, this shape is sufficient.

#### `agent.reaction`

Add a dedicated reaction event:

```json
{
  "item_id": "uuid",
  "emoji": "🫡"
}
```

We do not need an explicit `actor` field. In Maple, the target message role already implies who reacted:

- reaction on a user message means assistant reacted
- reaction on an assistant message means user reacted

In the implemented v1 scope, live `agent.reaction` SSE is used for assistant-generated reactions during chat turns. User `POST/DELETE .../reaction` mutations return the updated item over HTTP; separate multi-client fan-out is intentionally out of scope.

---

## 10. Prompt Construction Plan

Prompt building should include reaction metadata in the transcript header.

This applies when rendering `recent_conversation` from stored thread items.

Example rendering rules:

- no reaction:
  ```text
  [user @ 10:42 AM]: hey
  ```
- with reaction:
  ```text
  [user @ 10:42 AM | ❤️]: hey
  ```

This keeps the signal lightweight while still letting the model perceive social context.

---

## 11. Validation / Normalization

V1 accepts arbitrary emoji input without a fixed palette, but the server still applies lightweight normalization:

- trim surrounding whitespace
- unwrap quoted JSON-string values
- reject empty strings
- enforce a 16-character maximum
- require a single grapheme cluster
- reject component-only values such as bare `1`, `#`, `*`, joiners, or variation selectors without an emoji base

We should treat the stored value as a single reaction value, even if the emoji is composed of multiple Unicode codepoints.

We do not need a fixed reaction palette in v1.

---

## 12. File-Level Impact

Expected implementation areas:

- migrations for `user_messages` and `assistant_messages`
- `src/models/schema.rs`
- `src/models/responses.rs`
- `src/db.rs`
- `src/web/responses/conversions.rs`
- `src/web/responses/conversations.rs`
- `src/web/agent/runtime.rs`
- `src/web/agent/signatures.rs`
- `src/web/agent/mod.rs`

Potentially also:

- client-facing request/response structs for Maple reaction mutation routes
- Responses API item serializers if reaction support should also appear there

---

## 13. Implementation Order

Recommended order:

1. add DB columns and model/schema wiring
2. extend raw conversation message queries to carry reaction fields
3. extend `ConversationItem::Message` with `reaction`
4. wire reactions through Maple item list/get routes
5. add Maple user-reaction mutation endpoints
6. add `reply_reaction` to Maple DSR output
7. update runtime to persist assistant reactions onto the current user message
8. extend Maple SSE with `message_id` and `agent.reaction`
9. render reactions into prompt transcript headers

---

## 14. Non-Goals for v1

Not included in the first implementation:

- multiple reactions per message
- reaction history / audit log
- arbitrary target-item reactions by the assistant
- cross-user aggregated reactions
- reaction counts
- custom reaction metadata objects beyond a single emoji string

---

## 15. Summary

The v1 Maple reaction design should be:

- direct nullable reaction columns on message rows
- first-class `reaction` field on message items
- first-class first-step agent output `reply_reaction`
- continuation-step templates with no `reply_reaction` field
- live `agent.reaction` SSE events for assistant-generated reactions during chat
- prompt rendering using compact header metadata like `| ❤️`

This keeps the feature native to Maple chat semantics without abusing the tool-call system.
