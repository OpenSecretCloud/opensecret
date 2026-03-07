# Sage-in-Maple: Agent Memory Architecture

## Main Agent + Shared-Memory Subagents

**Date:** March 2026  
**Status:** Target architecture (local-only reset; no backwards-compatibility constraints)  
**Related Docs:**
- `potential-rag-integration-brute-force.md` -- RAG/vector storage layer
- `architecture-for-rag-integration.md` -- OpenSecret encryption and data model reference
- Sage V2 Design Doc (`~/Dev/Personal/sage/docs/SAGE_V2_DESIGN.md`) -- proven prototype
- Sage V2 Codebase (`~/Dev/Personal/sage/crates/sage-core`) -- DSRs signatures + BAML parsing + multi-step tool loop prototype

**Implementation note:** the current local codebase still reflects an older single-agent MVP in places. This document is now the source of truth for the rewrite.

---

## 1. Goal

Bring Sage's proven 4-tier memory architecture (core, recall, archival, summary) into Maple as a first-class product experience, without breaking the existing Responses API that third-party developers already understand.

The target product shape is:
- **one main persistent agent** as the app's home surface
- **unlimited subagents** as topic-specific chats/workspaces
- **shared memory** across the main agent and all subagents
- **separate conversation history** per agent thread
- **no end-user configuration burden** for prompts, models, memory block definitions, or tuning knobs

The end state is not “users manually configure an agent.” The end state is “users talk to Maple, Maple manages the agent system for them.”

### 1.1 The User Model

**One user : one main agent : unlimited subagents.**

Each agent owns exactly one conversation thread (`conversations.id`):
- The **main agent** is the user's long-running relationship with Maple.
- A **subagent** is a topic-specific extension of the main agent with its **own conversation history**, but access to the **same shared memory** and **same recall surface**.
- A subagent can be created either by the **user** or by the **main agent**.
- A subagent can remain long-lived forever if the user keeps using it.

This means the durable model is:
1. **Agent identity**
2. **One conversation thread per agent**
3. **User-shared memory across all of that user's agents**

**Implementation note:** do not infer agent identity from encrypted `conversations.metadata_enc`. Agent identity must be explicit in a dedicated `agents` table.

### 1.2 Shared Memory, Isolated Transcripts

All agents belonging to a user share:
- `memory_blocks` (core memory)
- archival memory in `user_embeddings`
- recall search over embedded messages in `user_embeddings`

Each individual agent keeps its own:
- recent conversation history
- `conversation_summaries` chain
- active thread-local context window

A subagent does **not** inherit the main agent's recent transcript. If it needs older context, it should use `conversation_search` just like the main agent would.

### 1.3 Conversation List Model

The **main agent** is the app's first-class home surface, not just another conversation row.

The **conversation list** should show:
- legacy Responses API threads
- subagent chats

The conversation list should **not** show:
- the main agent thread

So the product model becomes:
- **Main agent:** always-available home agent surface
- **Subagents:** the new first-class “chat/workspace” objects
- **Responses threads:** legacy/stateless threads that still exist alongside the new system

### 1.4 Cross-Thread Memory Visibility

**All agents for a user see the same memory layer.**

When `conversation_search` runs, it should search across all eligible embedded messages for that user by default:
- main agent thread messages
- subagent thread messages
- Responses API thread messages

Rationale: subagents are extensions of the same agent system, not separate privacy domains. If the user or the main agent creates a subagent for taxes, writing, or deployment planning, it should be able to benefit from the same user-level memory and searchable history.

The tool can still accept an optional `conversation_id` to scope search to one thread, but the default should be broad.

**Future:**
- `private` / incognito conversations excluded from embedding
- `store=false` suppressing embedding/indexing

Those remain opt-out boundaries for recall visibility.

### 1.5 Relationship to the Responses API

The Responses API remains a separate surface:
- good for stateless or developer-facing usage
- preserved for backwards compatibility at the product/API level
- still eligible for auto-embedding into `user_embeddings` when policy allows

Responses threads are **not** subagents. They remain distinct objects. But they can still contribute to the user's shared recall memory.

### 1.6 Configuration Philosophy

The product should be **agent-managed**, not **user-configured**.

That means the following should be owned by code / rollout config, not by per-user mutable database rows:
- default model selection
- prompt templates
- context windows
- compaction thresholds
- fixed memory block definitions and limits

The database should persist **durable state**, not **tuning policy**.

---

## 2. Core Architectural Decisions

### 2.1 Separate Runtime, Shared Storage

The agent runtime remains separate from the Responses API runtime.

What stays shared:
- `conversations`
- `user_messages`
- `assistant_messages`
- `tool_calls`
- `tool_outputs`
- `reasoning_items`

What becomes agent-specific:
- `agents`
- `memory_blocks`
- `conversation_summaries`
- agent-only tools / orchestration logic

This keeps the OpenAI-compatible Responses API stable while allowing Sage-style regenerated context and memory behavior for the main agent and subagents.

### 2.2 Introduce Explicit Agent Identity Now

The old MVP's `agent_config` shape bakes in “one agent per user.” That is no longer the right abstraction.

The architecture should move directly to an explicit `agents` table now.

Why:
- one user must have exactly one **main** agent
- one user can have many **subagents**
- each agent maps to exactly one conversation thread
- subagents must be distinguishable from Responses API threads in list views and routing
- the main agent must be hideable from `/v1/conversations/*` while subagents remain visible

### 2.3 Keep Shared Memory Per User

For the subagent model described above, the correct v1 choice is:
- **shared per-user core memory**
- **shared per-user archival memory**
- **shared per-user recall search**
- **per-agent thread-local conversation history**

This means we should **not** add `agent_id` to `memory_blocks` or `user_embeddings` in v1. Doing so would complicate the schema for a capability we explicitly do not want yet.

### 2.4 Treat the Main Agent as Special, but Not All Agents as Hidden

Only the **main agent** should be hidden from the generic conversation list.

Subagents should appear in `/v1/conversations/*` because they are the new product-level chat objects. They are exactly what users should browse, reopen, delete, and continue.

So the rule is:
- **hide the main agent thread**
- **show subagent threads**
- **show Responses API threads**

### 2.5 No Backwards Compatibility in Local-Only Schema

This work has not been deployed anywhere yet. We do **not** need compatibility-preserving additive migrations for the current Sage work.

So for the rewrite:
- edit the existing local-only Sage migrations directly
- replace `agent_config` with `agents`
- simplify `memory_blocks` now instead of carrying legacy fields forward
- move code-owned defaults out of SQL and into Rust/config

---

## 3. Why Reuse the Message Tables

This decision does not change.

**Maple's split message tables are still the right storage layer** for Sage-style agents. The main difference is no longer “one persistent agent thread per user”; it is now “one persistent thread per agent identity.”

| Operation | Agent requirement | Maple storage choice |
|---|---|---|
| Read recent messages in order | Load messages for the active agent's thread | Existing `RawThreadMessage` UNION query filtered by that agent's `conversation_id` |
| Store new messages | Persist by role and message type | Existing `user_messages` / `assistant_messages` / etc. |
| Search semantically | Search across all eligible user messages | `user_embeddings` keyed by `user_id` with optional `conversation_id` filter |
| Track compacted history | Keep summaries per thread | `conversation_summaries` keyed by `conversation_id` |

Nothing about the subagent model requires changing the core message tables themselves. The agent system still needs to **read them differently**, not **store them differently**.

---

## 4. Data Model

### 4.1 `agents` -- Explicit Agent Identity

This replaces the old `agent_config`-as-identity model.

```sql
CREATE TABLE agents (
    id               BIGSERIAL PRIMARY KEY,
    uuid             UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id          UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    -- One conversation thread per agent
    conversation_id  BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    -- 'main' or 'subagent'
    kind             TEXT NOT NULL,

    -- Main agent has NULL parent. Subagents should point at the main agent.
    parent_agent_id  BIGINT REFERENCES agents(id) ON DELETE SET NULL,

    -- User-visible metadata; encrypted because names/purposes may be sensitive.
    display_name_enc BYTEA,
    purpose_enc      BYTEA,

    -- Provenance only: 'user' or 'agent'
    created_by       TEXT NOT NULL DEFAULT 'user',

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT agents_kind_check CHECK (kind IN ('main', 'subagent')),
    CONSTRAINT agents_created_by_check CHECK (created_by IN ('user', 'agent')),
    UNIQUE(conversation_id)
);

CREATE UNIQUE INDEX idx_agents_one_main_per_user
    ON agents(user_id)
    WHERE kind = 'main';

CREATE INDEX idx_agents_user_kind_created
    ON agents(user_id, kind, created_at DESC);

CREATE INDEX idx_agents_parent_agent_id
    ON agents(parent_agent_id);

CREATE TRIGGER update_agents_updated_at
BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**Notes:**
- There is exactly **one** `kind='main'` agent per user.
- There may be **many** `kind='subagent'` rows per user.
- `conversation_id` is the canonical pointer to the agent's thread.
- `display_name_enc` and `purpose_enc` are encrypted because they may contain user-sensitive content.
- `parent_agent_id` exists so a subagent is explicitly modeled as an extension of the main agent.
- There is intentionally **no per-agent model/prompt/tuning config** here.

### 4.2 `memory_blocks` -- Shared Core Memory

Core memory remains per-user and shared across the main agent plus all subagents.

```sql
CREATE TABLE memory_blocks (
    id          BIGSERIAL PRIMARY KEY,
    uuid        UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id     UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    -- Fixed built-in labels for now: 'persona', 'human'
    label       TEXT NOT NULL,
    value_enc   BYTEA NOT NULL,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, label)
);

CREATE INDEX idx_memory_blocks_user_id ON memory_blocks(user_id);

CREATE TRIGGER update_memory_blocks_updated_at
BEFORE UPDATE ON memory_blocks
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**V1 choices:**
- Shared per user, **not** per agent
- No `description`
- No persisted `char_limit`
- No `read_only`
- No `version`
- No public manual CRUD requirement

**Built-in blocks for v1:**
- `persona`
- `human`

Any limits, formatting rules, or block semantics should live in code. If we later add more built-in blocks, we can do that by code first without needing a new table shape.

### 4.3 `user_embeddings` -- Shared Recall + Archival Store

This remains the shared embedding layer for the user.

- `source_type = 'message'`: embedded chat history
- `source_type = 'archival'`: long-term memory passages inserted by the agent
- `source_type = 'document'`: document chunks

**V1 rule:** do **not** add `agent_id` here.

Why:
- recall memory is intentionally shared across the main agent and all subagents
- archival memory is also intentionally shared in v1
- `conversation_id` already provides the necessary filter for message-scope search when needed

If we ever decide to support isolated subagent memory later, we can add explicit scope then. We should not pre-complicate the current design for a behavior we do not want yet.

See `potential-rag-integration-brute-force.md` for the full schema and search behavior.

### 4.4 `conversation_summaries` -- Per-Thread Compaction Artifacts

This table stays conceptually the same.

```sql
CREATE TABLE conversation_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    uuid                UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id             UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    conversation_id     BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    from_created_at     TIMESTAMPTZ NOT NULL,
    to_created_at       TIMESTAMPTZ NOT NULL,
    message_count       INTEGER NOT NULL,

    content_enc         BYTEA NOT NULL,
    content_tokens      INTEGER NOT NULL,
    embedding_enc       BYTEA,

    previous_summary_id BIGINT REFERENCES conversation_summaries(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_time_range CHECK (from_created_at <= to_created_at)
);

CREATE INDEX idx_conversation_summaries_user_conv
    ON conversation_summaries(user_id, conversation_id, created_at DESC);

CREATE INDEX idx_conversation_summaries_chain
    ON conversation_summaries(previous_summary_id);
```

**Key point:** summaries are per conversation thread, which naturally means per agent. No `agent_id` is needed because `conversation_id` already maps back to `agents`.

### 4.5 Remove `agent_config`

`agent_config` should not exist in the target architecture.

Why it should go away:
- it mixes durable identity with tuning policy
- it assumes one agent per user
- it stores values that should now be code-owned (`model`, `system_prompt`, `max_context_tokens`, `compaction_threshold`)
- it encourages user-facing configuration we do not want as a product

The durable pieces of state we actually need are:
- **agent identity** -> `agents`
- **shared core memory** -> `memory_blocks`
- **shared recall/archival storage** -> `user_embeddings`
- **thread-local compaction state** -> `conversation_summaries`

---

## 5. Agent Context Assembly

The main agent and every subagent use the same high-level regenerated-context pattern.

### 5.1 Code-Owned Prompting, Not User-Owned Prompting

The active agent runtime should load:
1. the `agents` row for the active agent
2. code-owned prompt template(s) for that agent kind
3. shared memory blocks for the user
4. the latest summary for the active conversation
5. recent messages for the active conversation

**Prompt/model selection should be code-owned.**

That means:
- no `system_prompt_enc` per user
- no per-user model selection row
- no per-user compaction threshold row
- no per-user max context row

Instead, code chooses:
- default main-agent instruction
- default subagent instruction
- default model(s)
- context window / packing rules
- compaction thresholds

### 5.2 Main Agent vs Subagent Context

The main agent and subagents share memory, but they do **not** share recent transcript.

The context builder should behave like this:

1. **Load active agent** from `agents`
2. **Build system prompt**
   - start with code-owned instruction for `kind='main'` or `kind='subagent'`
   - if subagent, inject decrypted `purpose_enc`
   - inject shared `persona` and `human` blocks
   - inject available tools
   - inject memory metadata
3. **Load latest summary** for the active `conversation_id`
4. **Load recent messages** for the active `conversation_id`
5. **Pack within token budget**
6. **Compact old messages** in that same conversation when thresholds are reached

A subagent should never silently get the main agent's recent transcript. If it needs information from another thread, it should use recall tools intentionally.

### 5.3 DSRs Signature Shape

The DSRs pattern still fits well. Conceptually the active agent call becomes:

```text
AgentResponse (inputs)
  - input
  - current_time
  - agent_kind
  - subagent_purpose
  - persona_block
  - human_block
  - memory_metadata
  - previous_context_summary
  - recent_conversation
  - available_tools
  - is_first_time_user

AgentResponse (outputs)
  - messages: string[]
  - tool_calls: {name: string, args: map<string,string>}[]
```

### 5.4 Compaction vs Truncation

This also stays the same conceptually:
- Responses API truncates
- agents compact

Each agent thread owns its own summary chain. Main agent compaction and subagent compaction are independent because they are keyed by different `conversation_id`s.

### 5.5 Vision / Image Handling

No architectural change is needed here.

The Sage-style approach still fits:
- preprocess image input into text
- inject the derived text into the conversation flow
- persist the derived text so it can be embedded and recalled later

This should work the same way for the main agent and for subagents.

---

## 6. Tools

These remain the core tools for the shared-memory model:

| Tool | Action | Storage |
|---|---|---|
| `memory_replace` | Replace text in a core memory block | `memory_blocks` |
| `memory_append` | Append text to a core memory block | `memory_blocks` |
| `memory_insert` | Insert text into a core memory block | `memory_blocks` |
| `archival_insert` | Store long-term memory | `user_embeddings` (`source_type='archival'`) |
| `archival_search` | Search long-term memory | `user_embeddings` (`source_type='archival'`) |
| `conversation_search` | Search embedded message history | `user_embeddings` (`source_type='message'`) |
| `spawn_subagent` | Create a topic-specific subagent chat | `conversations` + `agents` |
| `done` | Stop signal after tool-result continuation | no-op |

### 6.1 Memory Tool Semantics

For v1, core memory tools should operate on the built-in shared blocks only.

That means:
- `persona`
- `human`

This is still enough to get the product benefits of shared always-in-context memory without exposing a general-purpose user-editable block system.

### 6.2 `conversation_search` Visibility

By default, `conversation_search` should search across all eligible message embeddings for the user:
- main agent thread
- all subagent threads
- Responses API threads

The tool may optionally accept a `conversation_id` filter to scope to one thread.

### 6.3 `spawn_subagent`

Subagent creation should be a first-class operation.

A `spawn_subagent` tool should:
- create a new conversation row
- create a new `agents` row with `kind='subagent'`
- link it to the main agent via `parent_agent_id`
- store a display name and/or purpose
- return identifiers the client can use to hand off the user into that chat

This is how the main agent can proactively create a focused workspace for the user.

---

## 7. API Surface

### 7.1 Public Agent-Facing Routes

The public product surface should focus on the main agent and subagent lifecycle, not user configuration.

```text
POST   /v1/agent/chat                     -- chat with the main agent (request-scoped SSE)
POST   /v1/agent/subagents               -- create a new subagent
POST   /v1/agent/subagents/:id/chat      -- chat with a subagent (request-scoped SSE)
DELETE /v1/agent/subagents/:id           -- delete a subagent
GET    /v1/agent/events                  -- long-lived SSE for proactive delivery (post-MVP)
```

### 7.2 Routes That Should Not Be Public Product API

These may exist temporarily as local debugging surfaces, but they should not be part of the target user-facing architecture:
- manual agent config update endpoints
- manual memory block CRUD endpoints
- manual archival insert/delete endpoints for normal product usage

The product intent is **agent-managed memory**, not “settings panels for the user to tune Sage.”

### 7.3 Conversation List Integration

`/v1/conversations/*` should become the list/read/delete surface for:
- Responses API threads
- subagent chats

It should **exclude**:
- the main agent thread

The API should expose enough metadata to distinguish conversation kinds in the UI, e.g.:
- `response`
- `subagent`

That distinction should come from the `agents` join / derived server response, not from making encrypted conversation metadata the canonical source of truth.

### 7.4 Chat Event Model

The SSE event model can stay the same for both main agent and subagents:
- `agent.typing`
- `agent.message`
- `agent.done`
- `agent.error`

The difference is only which agent identity / conversation thread is being driven.

---

## 8. Relationship to Other API Surfaces

```text
/v1/responses/*
  Stateless chat surface
  Uses conversations + message tables
  Threads remain visible in /v1/conversations
  Eligible for shared recall embedding when policy allows

/v1/agent/chat
  Main agent home surface
  Uses agents + shared memory + thread-local summaries
  Main thread is hidden from /v1/conversations

/v1/agent/subagents/*
  Subagent lifecycle + chat surface
  Uses agents + shared memory + thread-local summaries
  Subagent threads are visible in /v1/conversations

/v1/conversations/*
  Unified list/read/delete surface for response threads + subagent chats
  Does not expose the main agent thread
```

The key data-flow rule is:
- all eligible stored messages from all three surfaces feed the same user-level recall memory
- all agents for that user can search that recall memory
- only thread-local history and summaries stay isolated per agent

---

## 9. What We Are Not Deciding Yet

These remain intentionally deferred:
- isolated subagent memory
- nested subagent trees beyond the “main agent -> subagent” model
- per-agent custom model selection
- per-agent custom prompt editing by users
- reminders / scheduling / background task execution
- code sandbox execution
- private/incognito threads excluding embedding
- agent-backed Responses API threads

The important v1 decision is already made: **subagents share memory and search, but not transcript history.**

---

## 10. Implementation Ordering

Because this is still local-only, we should update the existing Sage docs and schema directly instead of preserving the old MVP shape.

### Phase 1: RAG Foundation
- Keep `user_embeddings`
- Keep brute-force search + cache
- Keep experimental `/v1/rag/*` endpoints for validation if useful
- Add Responses API auto-embedding when the agent rewrite lands

### Phase 2: Schema Reset (edit local-only migrations in place)
- Replace `agent_config` with `agents`
- Simplify `memory_blocks`
- Keep `conversation_summaries`
- Keep `user_embeddings` shared per user
- Do **not** add per-agent memory scoping columns

### Phase 3: Main Agent Runtime Rewrite
- Load the main agent from `agents`
- Move model/prompt/tuning defaults into code
- Remove user-facing config assumptions
- Keep regenerated context + compaction

### Phase 4: Subagent Lifecycle
- Create subagents through API and tool flow
- Give each subagent its own conversation + summary chain
- Inject subagent purpose into prompt assembly

### Phase 5: Conversation List Integration
- Hide the main agent thread from `/v1/conversations/*`
- Show subagent threads there
- Continue showing Responses API threads
- Add derived conversation kind metadata for UI rendering

### Phase 6: Remove Non-Product Surfaces
- Remove or internalize manual config endpoints
- Remove or internalize manual memory block endpoints
- Keep only the public surfaces that match the product model

### Phase 7: Post-MVP
- Long-lived SSE event channel
- reminders / scheduling
- background delegation patterns
- optional future scoped memory if product requirements change

**Overall implementation status:** target architecture documented; code rewrite still required.
