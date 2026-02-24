# Sage-in-Maple: Agent Memory Architecture

## Bringing Persistent Agent Memory into OpenSecret

**Date:** February 2026
**Status:** MVP implemented (Local/Dev only; `/v1/agent/chat` streams message-level SSE; Responses API auto-embedding + production feature flags pending)
**Related Docs:**
- `potential-rag-integration-brute-force.md` -- RAG/vector storage layer (prerequisite)
- `architecture-for-rag-integration.md` -- OpenSecret encryption and data model reference
- Sage V2 Design Doc (`~/Dev/Personal/sage/docs/SAGE_V2_DESIGN.md`) -- proven prototype
- Sage V2 Codebase (`~/Dev/Personal/sage/crates/sage-core`) -- DSRs signatures + BAML parsing + multi-step tool loop prototype

**Overall implementation status:** Partially implemented (MVP shipped behind Local/Dev gating; Responses API auto-embedding + production feature flags not implemented yet).

---

## 1. Goal

Bring Sage's proven 4-tier memory architecture (core, recall, archival, summary) into Maple as a first-class feature, without disrupting the existing Responses API that third-party developers build against.

The end state: Maple users can have a persistent AI agent that remembers across conversations, has editable personality/user memory blocks, can store and search long-term memories, and auto-compacts conversation history when context windows fill up -- all with the same per-user encryption guarantees that exist today.

### 1.1 The User Model

**One user : one main agent : one persistent conversation.**

The agent lives in a single long-running conversation thread. Unlike Responses API threads (which are isolated, stateless, and disposable), the agent conversation is the user's ongoing relationship with Maple. Memory blocks, archival memory, and conversation summaries all attach to this single thread.

**Implementation note:** this mapping must be explicit in the database. We should not infer "the agent thread" from encrypted `conversations.metadata_enc`. Instead, store the thread pointer in `agent_config.conversation_id` (BIGINT FK to `conversations.id`) when the agent is initialized.

The Responses API continues to exist alongside the agent. Users can still have one-off stateless threads for quick questions. But the agent is the primary interface for users who want memory and persistence.

**Implementation status:** Complete (agent_config.conversation_id is the canonical pointer to the main agent thread).

### 1.2 Cross-Thread Memory Visibility

**The agent sees everything.** When the agent's `conversation_search` tool fires, it searches across all of the user's embedded messages -- both the agent's own conversation and any Responses API threads. The `user_embeddings` table indexes messages from all conversations by default.

Rationale: if a user had a Responses API thread about a topic, then asks the agent about that topic, the agent should know about it. The user opted into the agent system; that's the trust boundary. Individual threads are not a privacy boundary within the same user account.

The agent can still scope searches to its own thread via `conversation_id` filtering, but the default is broad. See the RAG proposal's "Data Visibility and Isolation" section for the full filtering model.

**Future: incognito threads.** A `private` flag on conversations will let users create Responses API threads that are excluded from embedding entirely. The agent cannot see them. This is deferred to post-v1.

Additionally, requests made with `store=false` should not be embedded/indexed into `user_embeddings` (per-request opt-out), even if the Responses API continues to persist messages today.

**Implementation status:** Partially implemented (cross-thread search works where embeddings exist; Responses API auto-embedding + private/store=false exclusions not implemented yet).

### 1.3 Future Vision: Agent-Backed Responses API

Eventually, the Responses API threads themselves could be backed by the agent's memory system. Instead of stateless threads with no memory, a Responses API thread would:
- Start with a fresh conversation history (no prior messages in context)
- But still have access to memory blocks, archival search, and conversation search
- Effectively: the agent "knows" the user (from core + archival memory), but the thread is a clean slate

This would give users the best of both: the lightweight feel of a new thread, with the accumulated knowledge of the persistent agent. This is a post-v1 evolution that requires proving out the agent system first -- particularly that memory blocks and search produce consistently good results without the agent's own conversation context.

**Implementation status:** Not implemented yet.

### 1.4 Future: Subagents (Multi-Agent per User)

Subagents are out of scope for the MVP, but we should keep the storage design compatible with them.

**Key idea:** each agent is just:
1) an **agent identity/config** record, and
2) a **single long-running conversation thread** (`conversations.id`) that the agent reads/writes.

The existing message tables remain unchanged; different agents simply write to different `conversation_id`s.

We also want subagents to support **configurable memory scoping**:
- **Shared/inherited memory:** subagents read/write the user's shared memory blocks and archival memory.
- **Isolated memory:** subagents have their own blocks/archival memory (no visibility into the user's shared memory unless explicitly allowed).

The cleanest way to support this is to introduce an `agents` table and add an `agent_id` FK to *agent-specific* tables (Section 4.5). Shared vs isolated memory becomes a query policy (and optionally a config field) rather than a schema rewrite.

**Implementation status:** Not implemented yet.

**Overall implementation status:** Partially implemented (MVP main-agent model shipped; agent-backed Responses API + subagents not implemented yet).

---

## 2. Key Architectural Decision: Separate API, Shared Storage

### What stays the same

The existing Responses API (`/v1/responses/*`, `/v1/conversations/*`, `/v1/instructions/*`) is **untouched**. It continues to be a stateless OpenAI-compatible chat API. Third-party developers who use it see no changes.

The existing database tables for conversations and messages are **reused** by the agent system. The agent reads from and writes to the same `conversations`, `user_messages`, `assistant_messages`, `tool_calls`, `tool_outputs`, and `reasoning_items` tables. The encrypted content format is identical.

**Implementation status:** Complete (agent reuses existing message tables and encryption format).

### What's new

A new `/v1/agent/*` API surface with its own handler module (`src/web/agent/`), its own step loop, and its own context assembly logic. This API implements the Sage-style regenerated-context pattern instead of the Responses API's middle-truncation pattern.

New database tables for agent-specific concerns (detailed in Section 4):
- `memory_blocks` -- core memory (persona, human, custom blocks)
- `user_embeddings` -- archival memory + chat embeddings (from the RAG proposal)
- `conversation_summaries` -- compaction artifacts
- `agent_config` -- per-user agent settings

**Implementation status:** Complete (MVP /v1/agent surface + new tables + RAG foundation implemented; Local/Dev gated).

### 2.1 Feature Flags and Rollout Safety (MVP requirement)

Sage is a large feature that touches storage, tool execution, and LLM calls. We need a lightweight, production-friendly feature flag system to safely ship in enclaves.

At minimum, ship with **two levels of gating**:
- **Global kill switch** (env/config): hard-disable all `/v1/agent/*` routes and background jobs (auto-embedding, compaction, reminders).
- **Per-user opt-in** (`agent_config.enabled`): enables the persistent agent for that user.

Flags should also gate sub-features independently (even if `agent_config.enabled=true`):
- auto-embedding into `user_embeddings` (recall memory)
- agent tool execution (memory tools, web search)
- compaction/summarization
- GEPA runs (optimizer execution) vs simply *consuming* an optimized instruction

**Implementation status:** Not implemented yet (only Local/Dev AppMode gating today; os_flags client exists but is unused for agent rollout).

### 2.2 Conversations API Isolation (Hide the Main Agent Thread)

Even though the agent's persistent thread is stored in the shared `conversations`/message tables, the **main agent conversation should be treated as an internal implementation detail** and **excluded from the public Conversations API** (`/v1/conversations/*`).

Rationale:
- Prevents confusion for developers using the stateless Responses/Conversations APIs.
- Avoids muddying thread lists with an always-on, long-running agent thread.
- Keeps `/v1/agent/*` as the single supported interface for agent state.

**Status: Implemented (2026-02-11).** The agent thread is hidden from all public Conversations API endpoints:
- `agent_config.conversation_id` stores the agent thread pointer.
- `ConversationContext::load()` returns `404` when the requested conversation is the agent thread (guards `GET/POST/DELETE /v1/conversations/:id` and `/items`).
- `GET /v1/conversations` filters out the agent thread before pagination.
- `DELETE /v1/conversations` (delete-all) excludes the agent conversation via `id.ne(agent_conversation_id)`.
- `POST /v1/conversations/batch-delete` returns `not_found` for the agent thread instead of deleting it.

**Implementation status:** Complete.

**Overall implementation status:** Partially implemented (API split + conversation isolation complete; feature flags not implemented yet).

---

## 3. Why Reuse the Message Tables

This was the hardest decision. We did a column-by-column comparison between Sage's `messages` table and Maple's split message tables. The conclusion:

**Maple's schema is a strict superset of Sage's.** Sage stores `(id, role, content, agent_id, sequence_id, embedding, tool_calls, tool_results, created_at)` in a single table. Maple stores the same information across typed tables with more metadata: token counts per message, response status tracking, response_id linking, conversation_id scoping, and UUIDs for external references.

**What Sage does with messages at runtime:**

| Operation | Sage approach | Maple equivalent |
|---|---|---|
| Read recent messages in order | `SELECT * FROM messages WHERE agent_id = ? ORDER BY sequence_id DESC LIMIT N` | The existing `RawThreadMessage::get_conversation_context` UNION ALL query across all message tables |
| Track which messages are "in context" | `agents.message_ids` UUID array | Compute dynamically from conversation + summaries (or add field to `agent_config`) |
| Store new messages | `INSERT INTO messages` with role | `INSERT INTO user_messages` / `assistant_messages` / etc. (existing flow) |
| Search messages semantically | pgvector on messages.embedding | `user_embeddings` table with in-process brute-force search (from RAG proposal) |
| Mark messages as summarized | Doesn't mark messages; records sequence ranges in summaries table | Same approach: `conversation_summaries` stores ranges, messages are untouched |

**Nothing in Sage's message access patterns requires modifying the existing Maple message table columns.** The agent needs to *read* from those tables differently (regenerated context with memory blocks instead of middle-truncation), but the storage format is identical.

**Potential future addition:** An `is_in_context` boolean or a `summary_id` FK on messages would be an optimization for fast filtering. But it's not required -- the summaries table's ranges can determine this. If needed, it would be an additive nullable column with a migration default, not a structural change.

**Overall implementation status:** Complete (agent uses existing message tables via RawThreadMessage UNION queries; no message-table schema changes required).

---

## 4. New Database Tables

### 4.1 `memory_blocks` -- Core Memory

The always-in-context memory blocks that define agent personality and user information. Directly modeled on Sage's `blocks` table, adapted for Maple's encryption model.

```sql
CREATE TABLE memory_blocks (
    id          BIGSERIAL PRIMARY KEY,
    uuid        UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id     UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    label       TEXT NOT NULL,           -- plaintext: "persona", "human", custom labels
    description TEXT,                    -- plaintext: how this block should be used
    value_enc   BYTEA NOT NULL,          -- AES-256-GCM encrypted block content
    char_limit  INTEGER NOT NULL DEFAULT 5000,
    read_only   BOOLEAN NOT NULL DEFAULT FALSE,
    version     INTEGER NOT NULL DEFAULT 1,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, label)
);

CREATE INDEX idx_memory_blocks_user_id ON memory_blocks(user_id);

CREATE TRIGGER update_memory_blocks_updated_at
BEFORE UPDATE ON memory_blocks
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**Key differences from Sage's `blocks` table:**
- `value_enc` (BYTEA) instead of `value` (TEXT) -- content encrypted with user's per-user key
- `user_id` instead of `agent_id` -- MVP keeps blocks per-user. For multi-agent/subagents, add a nullable `agent_id` so blocks can be shared (`agent_id IS NULL`) or agent-scoped (`agent_id = <agent>`).
- `label` and `description` are plaintext -- these are structural identifiers ("persona", "human"), not user content. They don't need encryption. This allows querying by label without decryption.

**Default blocks created on agent initialization:**
- `persona`: "I am a helpful AI assistant." (editable by agent)
- `human`: "" (populated by agent as it learns about the user)

**Implementation status:** Complete (table + models + CRUD endpoints + tools; code default char_limit=20000).

### 4.2 `user_embeddings` -- General-Purpose Embedding Store

Defined in the RAG proposal. A single table that serves all embedding use cases through a `source_type` discriminator:

- `source_type = 'message'`: auto-indexed chat history (recall memory)
- `source_type = 'archival'`: agent-inserted long-term memories (archival memory)
- `source_type = 'document'`: user-uploaded document chunks (document RAG)

- Future source types added without migration

Key columns for the agent system: `embedding_model` (tracks which model produced the vector, enables re-embedding on model upgrade), `content_enc` (always present -- the embedded text itself, encrypted), and `tags_enc` (deterministically-encrypted, base64 tags enabling SQL-indexable filtering for archival memories; extracted from `metadata.tags` when present).

**Future:** add `chunk_index` (ordering within multi-chunk sources like documents).

**Future (multi-agent/subagents):** consider adding an optional `agent_id` column for `source_type='archival'` (and possibly `source_type='document'`) rows so archival/document memory can be shared or isolated per agent. Recall/message embeddings remain scoped by `conversation_id` and can still default to cross-thread search.

See `potential-rag-integration-brute-force.md` for full schema and API design.

**Implementation status:** Partially implemented (table + in-process search/cache + archival tags implemented; Responses API auto-embedding not implemented yet).

### 4.3 `conversation_summaries` -- Compaction Artifacts

Stores rolling summaries when conversation context exceeds limits. Modeled on Sage's `summaries` table, adapted for Maple's conversation model.

```sql
CREATE TABLE conversation_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    uuid                UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id             UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    conversation_id     BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    -- Range of messages this summary covers
    -- Uses message created_at timestamps as boundaries
    from_created_at     TIMESTAMPTZ NOT NULL,
    to_created_at       TIMESTAMPTZ NOT NULL,
    message_count       INTEGER NOT NULL,   -- how many messages were summarized

    -- Encrypted summary content
    content_enc         BYTEA NOT NULL,     -- AES-256-GCM encrypted summary text
    content_tokens      INTEGER NOT NULL,   -- plaintext token count of summary

    -- Encrypted embedding for semantic search over summaries
    embedding_enc       BYTEA,              -- AES-256-GCM encrypted float32 array

    -- Chain: previous summary that was absorbed into this one
    previous_summary_id BIGINT REFERENCES conversation_summaries(id) ON DELETE SET NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_time_range CHECK (from_created_at <= to_created_at)
);

CREATE INDEX idx_conversation_summaries_user_conv
    ON conversation_summaries(user_id, conversation_id, created_at DESC);

CREATE INDEX idx_conversation_summaries_chain
    ON conversation_summaries(previous_summary_id);
```

**Key differences from Sage:**
- Uses `from_created_at` / `to_created_at` timestamp ranges instead of Sage's `from_sequence_id` / `to_sequence_id`. Maple's message tables don't have a monotonic sequence_id across the UNION -- they have per-table auto-increment IDs. Timestamps are the reliable ordering key that works across all message types.
- `embedding_enc` instead of pgvector `embedding` column. Follows the same encrypted pattern as everything else.
- `content_tokens` plaintext for context budgeting.

**Implementation status:** Complete (compaction writes summaries + embeds them for search).

### 4.4 `agent_config` -- Per-User Agent Settings

```sql
CREATE TABLE agent_config (
    id              BIGSERIAL PRIMARY KEY,
    uuid            UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id         UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE UNIQUE,

    -- Which conversation thread is the user's "main agent" thread
    -- Nullable until agent initialization creates/assigns a thread.
    conversation_id BIGINT REFERENCES conversations(id) ON DELETE SET NULL,

    -- Agent behavior settings (plaintext, not sensitive)
    enabled         BOOLEAN NOT NULL DEFAULT FALSE,
    model           TEXT NOT NULL DEFAULT 'deepseek-r1-0528',
    max_context_tokens  INTEGER NOT NULL DEFAULT 100000,
    compaction_threshold REAL NOT NULL DEFAULT 0.80,

    -- Agent system prompt (encrypted, user-customizable)
    system_prompt_enc   BYTEA,

    -- User preferences (encrypted JSON: timezone, response style, etc.)
    preferences_enc     BYTEA,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER update_agent_config_updated_at
BEFORE UPDATE ON agent_config
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**Notes:**
- One agent config per user (MVP). The `UNIQUE(user_id)` constraint enforces this.
- **Future (multi-agent/subagents):** prefer introducing an `agents` table and making `agent_config` keyed by `agent_id` (or merging `agent_config` into `agents`). This makes agent identity explicit and enables agent-scoped memory via `agent_id` FKs (Section 4.5). The simpler alternative (remove `UNIQUE(user_id)` + add a `name` column) is viable but makes shared vs isolated memory harder to represent cleanly.
- `conversation_id` is the canonical pointer to the user's persistent agent thread. This avoids relying on encrypted conversation metadata to locate the agent conversation.
- `enabled` allows opt-in activation. Users who don't want agent features continue using the stateless Responses API.
- `system_prompt_enc` is the base agent instruction. Equivalent to Sage's `agents.system_prompt`, but encrypted. If NULL, a default system prompt is used.
- `preferences_enc` absorbs what Sage stores in the separate `user_preferences` table. A single encrypted JSON blob is simpler than a KV table for a small number of preferences (timezone, response style, language).

**Implementation status:** Partially implemented (config + conversation_id + system_prompt implemented; preferences + robust opt-in/flags not implemented yet).

### 4.5 Future: `agents` Table + `agent_id` FKs (Multi-Agent/Subagents)

If we want multiple agents per user (subagents), the most extensible model is to make agent identity explicit.

**Proposed `agents` table (future):**

```sql
CREATE TABLE agents (
    id              BIGSERIAL PRIMARY KEY,
    uuid            UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id         UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,

    -- User-visible identifier (plaintext)
    name            TEXT NOT NULL DEFAULT 'main',

    -- Each agent has exactly one long-running conversation thread
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    enabled         BOOLEAN NOT NULL DEFAULT FALSE,

    -- Optional: policy for shared vs isolated memory
    -- Examples: 'shared', 'isolated', 'overlay'
    memory_mode     TEXT NOT NULL DEFAULT 'shared',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, name),
    UNIQUE(conversation_id)
);

CREATE TRIGGER update_agents_updated_at
BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**Agent-scoped foreign keys (future):** add `agent_id` (nullable or required) to agent-specific tables:
- `memory_blocks.agent_id` (nullable)
  - `agent_id IS NULL` => shared blocks for the user
  - `agent_id = X` => blocks scoped to agent X
- `conversation_summaries.agent_id` (optional; can be derived from `conversation_id`)
- `user_embeddings.agent_id` (optional; most useful for `source_type='archival'` / `'document'`)
- `scheduled_tasks.agent_id` (for reminders)
- `agent_config.agent_id` (if/when `agent_config` becomes per-agent)

**Uniqueness note (Postgres):** if we use `memory_blocks.agent_id IS NULL` to represent shared blocks, we likely want **partial unique indexes** to enforce:
- exactly one shared block per `(user_id, label)` and
- exactly one agent-scoped block per `(user_id, agent_id, label)`.

For MVP, we can start with the simpler per-user `agent_config` model, and only introduce `agents` once subagents are in scope.

**Implementation status:** Not implemented yet.

**Overall implementation status:** Partially implemented (MVP agent tables shipped; multi-agent/subagent schema not implemented yet).

---

## 5. Agent Context Assembly

This is the fundamental difference between the Responses API and the agent system. Instead of the Responses API's approach (load all messages, middle-truncate to fit context window), the agent regenerates a structured context on every turn.

### 5.1 Sage V2 Prototype Pattern (DSRs Signatures + BAML Parsing)

The current Sage V2 prototype uses DSRs signatures for typed input/output and BAML parsing for structured extraction, rather than provider-native tool calling.

Conceptually, the agent is called with a set of **separated context fields** (so GEPA can optimize them independently), and returns **two typed outputs**: `messages[]` and `tool_calls[]`.

```
AgentResponse (inputs)
  - input                       (user message OR tool-result continuation)
  - current_time
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

```
[System Prompt]
  - Base instruction (GEPA-optimized or user-customized)
  - <memory_blocks>
      <persona>...</persona>
      <human>...</human>
  - <available_tools>
      memory_replace, memory_append, archival_insert, archival_search,
      conversation_search, web_search, ...
  - <memory_metadata>
      archival_count, recall_count, last_compaction, ...

[Previous Context Summary]   (if conversation was compacted)

[Recent Messages]            (last N messages that fit in token budget)

[Current User Message]
```

**Implementation status:** Complete (DSRs-style signature inputs/outputs implemented via dspy-rs + structured parsing).

### 5.2 Maple Agent Context Builder

A new `src/web/agent/context_builder.rs` that replaces the Responses API's `context_builder.rs` for agent requests:

1. **Load agent config** -- model, max_context_tokens, compaction_threshold
2. **Build system prompt:**
   - Start with base instruction from `agent_config.system_prompt_enc` (or default)
   - Decrypt and inject memory blocks from `memory_blocks` table
   - Inject tool descriptions (from agent tool registry)
   - Inject memory metadata (counts from user_embeddings, memory_blocks, conversation_summaries)
3. **Load summary** -- most recent `conversation_summaries` entry for this conversation, if any
4. **Load recent messages** -- from the existing message tables (same UNION ALL query), but:
   - Start from after the summary's `to_created_at` timestamp
   - Load messages until token budget is consumed
   - No middle-truncation -- if budget exceeded, trigger compaction instead
5. **Check compaction threshold** -- if total tokens > threshold, compact oldest messages into a new summary before proceeding
6. **Assemble final prompt** -- system + summary + recent messages + current input

In practice, if we follow the Sage V2 prototype, the "assembled prompt" is expressed as the DSRs signature inputs above (plus an instruction string), and BAML parses the structured outputs.

**Implementation status:** Partially implemented (context assembly implemented in `src/web/agent/runtime.rs`; does not strictly pack messages to a token budget).

### 5.3 Compaction vs Truncation

The Responses API truncates: it drops middle messages and inserts `[Previous messages truncated due to context limits]`. Information is lost.

The agent system compacts: it summarizes old messages into a `conversation_summaries` entry using an LLM call, then removes those messages from the in-context window (but they remain in the database and are searchable via recall memory). Information is preserved in compressed form.

Compaction uses a DSRs-style signature (following Sage's `SummarizeConversation` pattern):

```
Input:
  - previous_summary (if chaining summaries)
  - messages_to_summarize
Output:
  - summary (100 word limit)
```

The LLM call for summarization uses a fast/cheap model (e.g., `gpt-oss-120b`) to minimize latency and cost.

**Implementation status:** Partially implemented (compaction + summaries implemented; summarization currently uses the agent model, not a dedicated cheap model).

**Overall implementation status:** Partially implemented (regenerated context + compaction shipped; strict token-budget packing + cheap summarizer model not implemented yet).

---

## 6. Agent Step Loop

The agent processes messages through a multi-step loop, following Sage's proven pattern:

```
User sends message
  -> Step 1: Build context, call LLM
  -> LLM returns: messages to user + tool calls
  -> Execute tool calls (memory_replace, archival_insert, web_search, etc.)
  -> If tools were called: inject results, go to Step 2
  -> Step 2: Build context (with tool results), call LLM
  -> LLM returns: messages to user + tool calls
  -> ... repeat until LLM returns messages with no tool calls, or max steps reached
  -> Send final messages to user
```

**Max steps:** 10 (same as Sage). Prevents infinite loops.

**Tool execution:** Each tool call is persisted to the existing `tool_calls` and `tool_outputs` tables. Memory tools (`memory_replace`, `archival_insert`, etc.) also modify the memory tables directly.

### 6.2 Structured Output + Parse Correction (Recommended)

To maximize reliability across providers/models (and to avoid native tool calling issues), align with the Sage V2 prototype:

- The agent returns structured `messages[]` and `tool_calls[]` via DSRs signatures + BAML parsing.
- If the LLM returns malformed output that fails to parse, run a **correction signature** that reshapes the raw text into the expected structure (preserving intent, not generating new content).
- Include a `done` tool as a no-op stop signal for tool-result continuation steps.

This pattern dramatically reduces sensitivity to provider-specific tool calling bugs, while still supporting multi-step tool loops.

**Implementation status:** Complete (AgentResponse + correction signatures + done tool implemented).

### 6.3 DSRs Integration in OpenSecret (Nitro-safe LLM routing)

DSRs must not call providers directly. All agent LLM calls (main steps + summarization + correction) must route through OpenSecret's existing LLM pipeline (billing, auth, retries, provider routing): `web::openai::get_chat_completion_response()`.

Implementation options:
- **Preferred:** implement a custom DSRs LM backend/hook that calls `get_chat_completion_response()` and returns the model text + usage.
- **Fallback:** vendor/fork DSRs at a pinned commit to add the hook (so we control upgrades and avoid billing bypass regressions).

**Current status (implemented 2026-02-10):** OpenSecret now uses a locally-patched `dspy-rs` (DSRs) with a `LMClient::Custom(CustomCompletionModel)` hook that routes completions through `get_chat_completion_response()` (see `src/web/agent/signatures.rs`). Default `temperature=0.7` and `max_tokens=32768` match the Sage prototype.

**Billing note:** embedding calls used by the agent system (archival insert/search, auto-embedding of agent messages, summary embeddings) must also route through OpenSecret's billing-aware embedding pipeline (`web::get_embedding_vector()`), and must never call providers directly.

**Implementation status:** Complete (DSRs LM hook routes via `get_chat_completion_response()`; embeddings route via billing-aware `get_embedding_vector()`).

### 6.1 Memory Tools

These are the agent's interface to the memory system. Following Sage's design:

| Tool | Action | Storage |
|---|---|---|
| `memory_replace` | Replace text in a core memory block | UPDATE `memory_blocks` |
| `memory_append` | Append text to a core memory block | UPDATE `memory_blocks` |
| `memory_insert` | Insert text at a specific line in a block | UPDATE `memory_blocks` |
| `archival_insert` | Store information in long-term memory | INSERT into `user_embeddings` (source_type='archival') |
| `archival_search` | Search long-term memory semantically (optional tag filter) | Query `user_embeddings` (source_type='archival') with brute-force cosine similarity |
| `conversation_search` | Search conversation history semantically | Query `user_embeddings` (source_type='message') -- **all conversations by default**, not just the agent's own thread |

All memory tool inputs/outputs are encrypted before storage. The tool execution happens in-enclave where the user key is available.

**`conversation_search` visibility note:** By default, this tool searches across all of the user's embedded messages, including Responses API threads. This is intentional -- the agent should surface relevant context regardless of which API surface generated it. The tool can accept an optional `conversation_id` parameter to scope to a specific thread if needed, but the default is broad. See the RAG proposal's "Data Visibility and Isolation" section.

**Archival tags (implemented 2026-02-11):** `archival_insert` supports `metadata.tags` (string or string[]). Tags are normalized (trim + lowercase), deterministically encrypted per-tag (base64) and stored in `user_embeddings.tags_enc`. `archival_search` supports a `tags` argument and applies an ANY-match SQL filter (`tags_enc && <encrypted-tags>`) backed by a partial GIN index.

**Implementation status:** Partially implemented (core memory + archival + conversation_search implemented; web_search tool not implemented yet).

**Overall implementation status:** Partially implemented (step loop + tool persistence implemented; web_search + other post-MVP tools pending).

---

## 7. API Surface

### 7.1 New Routes (`/v1/agent/*`)

**Current status (implemented 2026-02-11):** the MVP `/v1/agent/*` surface below is implemented (Local/Dev only). `POST /v1/agent/chat` streams message-level SSE (not token streaming); other endpoints are encrypted JSON.

```
POST   /v1/agent/chat                -- Send a message to the agent (step loop, request-scoped SSE)
GET    /v1/agent/config               -- Get agent settings
PUT    /v1/agent/config               -- Update agent settings (model, system prompt, etc.)

GET    /v1/agent/memory/blocks        -- List all memory blocks
GET    /v1/agent/memory/blocks/:label -- Get a specific block
PUT    /v1/agent/memory/blocks/:label -- Manually edit a block

POST   /v1/agent/memory/archival      -- Manually insert archival memory
POST   /v1/agent/memory/search        -- Search archival + recall memory
DELETE /v1/agent/memory/archival/:id  -- Delete specific archival entry

GET    /v1/agent/conversations        -- List agent conversations (reuses conversations table)
GET    /v1/agent/conversations/:id/items -- Get conversation items (reuses existing item format)
DELETE /v1/agent/conversations/:id    -- Delete conversation + associated summaries

GET    /v1/agent/events               -- Long-lived SSE channel for proactive agent delivery + fan-out (post-MVP)
```

**Implementation status:** Partially implemented (all MVP routes except `GET /v1/agent/events`; Local/Dev only).

### 7.2 Chat Endpoint Detail

**Current implementation (2026-02-11):** `POST /v1/agent/chat` accepts `{ "input": "..." }` (encrypted request body like other endpoints) and returns request-scoped SSE (message-level events, not token streaming).

Event types (payloads are JSON, encrypted per-session and base64-encoded in the `data:` field):

| Event type | Meaning |
|---|---|
| `agent.typing` | Agent is working on step `step` |
| `agent.message` | One or more user-visible messages for step `step` |
| `agent.done` | Turn completed (`total_steps`, `total_messages`) |
| `agent.error` | Turn failed (`error`) |

```
POST /v1/agent/chat
{
  "input": "What did we discuss about deployment strategies last week?"
}
```

**Implementation status:** Complete (request-scoped, message-level SSE with typing/message/done/error events).

### 7.3 Proactive Agent Delivery: Long-Lived SSE Channel (Post-MVP)

`POST /v1/agent/chat` uses request-scoped SSE: the stream opens on request, delivers step-loop events (`agent.typing`, `agent.message`, `agent.done`), and closes. This is correct for request-response interactions but insufficient for proactive agent behavior (reminders, scheduled task results, agent-initiated messages).

**Decision: long-lived SSE over WebSockets.** Three reasons specific to our architecture:

1. **Encryption model fit.** The per-session encryption middleware operates on HTTP request/response cycles. SSE is still HTTP and slots in naturally. WebSockets would require rethinking per-message encryption (WS frames aren't HTTP responses, so `encrypt_event` / session key lookup needs a different path).
2. **Proxy chain simplicity.** tinfoil-proxy and continuum-proxy sit in front of the enclave via vsock. HTTP/SSE flows through standard reverse proxies. WebSocket upgrade handling through that chain adds operational complexity -- sticky sessions, connection draining, timeout tuning at every hop.
3. **Unidirectionality is sufficient.** Proactive agent messages are server-to-client. The client already has `POST /v1/agent/chat` for the other direction.

**Proposed endpoint:** `GET /v1/agent/events`

The client opens this connection once and keeps it open. The server pushes **the same `agent.*` SSE events used by `/v1/agent/chat`** (`agent.typing`, `agent.message`, `agent.done`, `agent.error`) whenever the agent produces output outside of an active chat request.

**Resilience requirements:**
- **At-least-once delivery.** Clients MUST dedupe by SSE event `id`.
- **SSE `id` support + `Last-Event-ID` resumption.** Every non-heartbeat event MUST include an SSE `id:` field. On reconnect, clients send `Last-Event-ID` to resume from the next event.
- **DB-backed event log (fan-out).** Events are persisted so clients that were offline can catch up on reconnect. Because this is fan-out, events are retained by TTL (e.g., 24h) rather than deleted-on-delivery.
- **Heartbeat keepalive.** Emit SSE comment frames (e.g., `: ping\n\n`) every ~30s to prevent proxy idle timeouts.

**Relationship to `POST /v1/agent/chat`:** The two channels are independent. Chat continues to use request-scoped SSE for immediate step-loop delivery. The long-lived channel handles async/proactive events only. A client that only uses chat (no proactive features) never needs to open `/v1/agent/events`.

**Implementation status:** Not implemented yet.

### 7.4 Relationship to Responses API

The two API surfaces are independent but share storage:

```
/v1/responses/*        -- Stateless chat API (existing, unchanged)
  reads/writes: conversations, user_messages, assistant_messages,
                tool_calls, tool_outputs, reasoning_items, user_instructions
  triggers: (TODO) async embedding into user_embeddings (store=true and non-private threads only)

/v1/agent/*            -- Persistent agent API (new)
  reads/writes: conversations, user_messages, assistant_messages,
                tool_calls, tool_outputs, reasoning_items
  also reads/writes: memory_blocks, user_embeddings,
                     conversation_summaries, agent_config
  triggers: async embedding into user_embeddings for agent messages (implemented)
  searches: user_embeddings across ALL user conversations (broad default)
```

A user can use both APIs simultaneously. The key data flow:

- **Responses API -> Agent visibility (pending):** Responses API threads will be auto-embedded into `user_embeddings` when eligible (stored + not-private). Today, only the agent's own messages are auto-embedded.
- **Agent conversation visibility:** The agent conversation is stored in the shared tables, but the **main agent thread should be hidden from `/v1/conversations/*`** (Section 2.2). Clients should use `/v1/agent/conversations/*` to access agent threads.
- **Isolation boundary:** `store=false` requests and future incognito/private threads on the Responses API will NOT be embedded. The agent cannot see them. This is the opt-out mechanism.

**Implementation status:** Partially implemented (agent messages auto-embedded; Responses API auto-embedding + private/store=false exclusions not implemented yet).

**Overall implementation status:** Partially implemented (MVP /v1/agent surface shipped Local/Dev; proactive events channel pending).

---

## 8. What We're Not Deciding Yet

These are intentionally deferred:

- **Native tool calling in the agent loop.** The Responses API will continue using native tool calling via the Tinfoil proxy. For the agent loop, prefer DSRs signatures + BAML parsing (Section 6.2) for reliability; native tool calling can be revisited later.

- **GEPA operationalization in production.** MVP should be GEPA-*ready*: consume a GEPA-optimized instruction stored in `agent_config.system_prompt_enc`. Running GEPA itself can be feature-flagged/offline until we have safe evaluation + trace capture.

- **Multi-agent per user (subagents).** MVP assumes one agent per user. When subagents are in scope, prefer an `agents` table + `agent_id` FKs on agent-specific tables (Section 4.5) so memory can be shared or isolated per agent. The simpler approach (remove `UNIQUE(user_id)` + add `name` on `agent_config`) is viable but less expressive for memory scoping.

- **Group/shared memory.** All memory is per-user. Shared memory blocks between users in the same project would require a new sharing model. Deferred.

- **Voice/image input handling.** Sage has a vision pipeline for Signal image attachments. Maple's Responses API already handles multimodal input. The agent system inherits this capability from the shared message tables.

- **Reminders / scheduling.** Post-MVP. Sage V2 has a working scheduler + tools (`schedule_task`, `list_schedules`, `cancel_schedule`) backed by a `scheduled_tasks` table. We can port this once the core agent loop is stable. Delivery of fired reminders will use the long-lived SSE channel (`GET /v1/agent/events`, Section 7.3).

- **Code sandbox execution.** Long-lead, deferred. Requires a secure execution substrate (likely isolated runtime) and careful Nitro threat modeling.

- **Incognito / private threads.** A `private` flag on Responses API conversations that excludes their messages from embedding (and thus from agent visibility). In addition, `store=false` should suppress embedding as a per-request opt-out. The conversation-level `private` flag is deferred to post-v1. A UI toggle to set incognito as the default for new threads is further out.

- **Agent-backed Responses API threads.** The long-term vision where Responses API threads inherit agent memory (memory blocks + search) but start with a clean conversation history. Requires proving out the agent system first. See Section 1.3.

**Overall implementation status:** Not implemented yet (intentionally deferred).

---

## 9. Implementation Ordering

A suggested sequence, building on the RAG layer as the foundation:

### Phase 1: RAG Foundation
- [x] Implement `user_embeddings` table + migration
- [x] Implement brute-force search + LRU cache
- [x] Implement `/v1/rag/*` API endpoints
- [ ] Wire up async embedding generation after message creation in Responses API (respect `store=false` and future `private` threads)
- **Milestone:** Recall + archival memory storage/search exists (even before the agent loop ships)

**Implementation status:** Partially implemented (RAG + /v1/rag done; Responses API auto-embedding not implemented yet).

### Phase 2: Agent Storage + Feature Flags
- [x] Implement `memory_blocks`, `conversation_summaries`, `agent_config` tables + migrations
- [x] Implement Diesel models for new tables
- [x] Add `agent_config.conversation_id` pointer to the user's persistent agent thread
- [x] Implement memory block CRUD operations
- [ ] Implement global + per-user feature flagging for `/v1/agent/*` and background jobs
- **Milestone:** Agent storage layer exists and is tested

**Implementation status:** Partially implemented (storage tables/models done; feature flags not implemented yet).

### Phase 3: Agent LLM Runtime (DSRs + BAML)
- [x] Integrate DSRs signatures + BAML parsing (AgentResponse)
- [x] Integrate correction signatures for parse repair
- [x] Implement Nitro-safe DSRs LM routing via `get_chat_completion_response()`
- [x] Implement summarization signatures for compaction
- **Milestone:** Agent can reliably produce `messages[]` + `tool_calls[]` without native tool calling

**Implementation status:** Complete.

### Phase 4: Agent Context Builder + Compaction
- [x] Implement regenerated agent context builder (currently in `src/web/agent/runtime.rs`)
- [x] Implement compaction (LLM summarization of old messages) into `conversation_summaries`
- **Milestone:** Agent can sustain long-running conversations without truncation

**Implementation status:** Complete.

### Phase 5: Memory Tools + Multi-Step Agent Loop
- [x] Implement agent tool registry (core memory + archival + conversation search + web search)
- [x] Implement multi-step execution loop (max steps, tool-result continuation)
- **Milestone:** DSRs-based agent can have multi-turn conversations with memory tool use

**Implementation status:** Partially implemented (memory tools + step loop done; web_search tool not implemented yet).

### Phase 6: Agent API Surface
- [x] Implement `/v1/agent/chat` (request-scoped SSE)
- [x] Implement remaining `/v1/agent/*` endpoints (config, memory management, conversations)
- [x] Wire up memory block initialization + agent thread creation on opt-in
- [x] Wire up async embedding generation after agent message creation
- **Milestone:** Full agent API surface available

**Implementation status:** Complete (Local/Dev only).

### Phase 7: Post-MVP
- Long-lived SSE event channel (`GET /v1/agent/events`) -- proactive notification delivery (Section 7.3)
- Reminders/scheduling (`scheduled_tasks` + scheduler loop + tools), delivered via the event channel
- Monitoring/metrics + tuning
- (Optional) agent-backed Responses API threads (Section 1.3)
- Code sandboxes (separate major project)

**Implementation status:** Not implemented yet.

**Overall implementation status:** Partially implemented (Phases 1-6 mostly complete; Phase 7 items pending).
