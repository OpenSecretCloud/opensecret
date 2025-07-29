# OpenAI Responses API Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Database Schema](#database-schema)
3. [API Endpoints](#api-endpoints)
4. [Message Encryption Strategy](#message-encryption-strategy)
5. [Idempotency](#idempotency)
6. [Request Processing Architecture](#request-processing-architecture)
7. [Usage Tracking & Analytics](#usage-tracking--analytics)
8. [Token Management Strategy](#token-management-strategy)
9. [Tool Calling Framework](#tool-calling-framework)
10. [Authentication & Authorization](#authentication--authorization)
11. [Error Handling](#error-handling)
12. [Database Migrations](#database-migrations)
13. [Security Considerations](#security-considerations)
14. [Performance Optimizations](#performance-optimizations)
15. [Testing Strategy](#testing-strategy)
16. [Implementation Checklist](#implementation-checklist)

## Overview

This document outlines the implementation of an OpenAI Responses API-compatible endpoint for OpenSecret. The Responses API is different from the Chat Completions API - it provides server-side conversation state management with SSE streaming and a single entry point where the request contains the entire conversation state.

**Key Differences from Chat Completions API**:
- **Chat Completions** (`/v1/chat/completions`): Stateless, requires full conversation history each request
- **Responses** (`/v1/responses`): Server-managed conversation state, status tracking, server-side tools

### Objectives

1. **OpenAI Responses API Compatibility**: Implement `/v1/responses` endpoint that matches OpenAI's Responses API specification (including the `queued` status added in April 2025)
2. **Dual Streaming Architecture**: Stream responses to client while simultaneously storing to database
3. **SSE Streaming**: Server-sent events with proper event types (response.created, response.delta, tool.call, response.done, etc.)
4. **Status Lifecycle**: Track response status through queued → in_progress → completed (though queued is not currently used)
5. **Tool Calling Support**: Server-side tool execution with proper status tracking
6. **Integration with Existing Chat API**: Use the existing `/v1/chat/completions` internally for model calls
7. **Security First**: All user content encrypted at rest using existing patterns

### Key Features

- Single POST /v1/responses endpoint (no thread/run management)
- Simultaneous streaming to client and database storage
- SSE streaming with OpenAI-compatible event format
- Status lifecycle management (in_progress → completed)
- Server-side tool execution integrated into response flow
- Leverages existing /v1/chat/completions for model interaction
- Model-specific token context window support

### Implementation TODO List

**Phase 1: Foundation (Database & Models)**
- [x] Create and run database migrations
  - [x] Create response_status enum (includes 'queued' for future use, but MVP will skip directly to 'in_progress')
  - [x] Create user_system_prompts table
  - [x] Create chat_threads table 
  - [x] Create user_messages table with idempotency columns
  - [x] Create tool_calls table
  - [x] Create tool_outputs table
  - [x] Create assistant_messages table
  - [x] Add all necessary indexes
- [x] Generate schema.rs with diesel run command (fix any migration errors that come up)
- [x] Create Diesel model structs
  - [x] ResponseStatus enum
  - [x] UserSystemPrompt model
  - [x] ChatThread model
  - [x] UserMessage model (with idempotency fields)
  - [x] ToolCall model
  - [x] ToolOutput model
  - [x] AssistantMessage model
- [x] Add query methods to DBConnection trait
  - [x] get_thread_by_id_and_user()
  - [x] create_thread() - renamed from create_thread_with_id()
  - [x] get_user_message_by_uuid() - added instead of get_user_message_by_previous_id()
  - [x] get_thread_context_messages() - renamed from get_thread_messages_for_context()
  - [x] get_user_message_by_idempotency_key() - query includes NOW() check for expiry
  - [x] Additional methods added: update_thread_title(), create_user_message(), update_user_message_status(), get_user_message(), list_user_messages(), create_assistant_message(), create_tool_call(), create_tool_output(), cleanup_expired_idempotency_keys()

**Phase 2: Basic Responses Endpoint (No Streaming)**
- [x] Create ResponsesCreateRequest struct matching OpenAI spec
- [x] Create ResponsesCreateResponse struct
- [x] Implement POST /v1/responses handler
  - [x] JWT validation (reuse existing middleware)
  - [x] Request validation
  - [x] Thread creation logic (thread.uuid = message.uuid for new threads)
  - [x] Thread lookup for previous_response_id
  - [x] Store user message with status='in_progress'
  - [x] Return immediate response with ID
- [x] Add route to web server
- [x] Idempotency support with header checking
- [ ] Test basic request/response flow

**Phase 3: Context Building & Chat Integration**
- [ ] Implement conversation context builder
  - [ ] Query all message types from thread
  - [ ] Merge and sort by timestamp
  - [ ] Decrypt message content
  - [ ] Format into ChatCompletionRequest messages array
- [ ] Implement token counting
  - [ ] Integrate tiktoken-rs
  - [ ] Add per-model token limits (ask for supported models and their limits)
  - [ ] Implement context truncation strategy
- [ ] Call internal /v1/chat/completions
  - [ ] Build request from context
  - [ ] Handle non-streaming response
  - [ ] Store assistant message
  - [ ] Update user message status to 'completed'
- [ ] Test end-to-end flow

**Phase 4: SSE Streaming**
- [ ] Update handler to support stream=true
- [ ] Implement SSE response format
  - [ ] Add Content-Type: text/event-stream header
  - [ ] Format events: response.delta, response.done
  - [ ] Add encryption for SSE chunks
- [ ] Implement streaming from chat API
  - [ ] Handle streaming response from internal endpoint
  - [ ] Forward chunks to client as SSE
  - [ ] Accumulate content for storage
- [ ] Add heartbeat support
  - [ ] Send comment frames every 30s
  - [ ] Handle client disconnection gracefully
- [ ] Test streaming functionality

**Phase 5: Dual Streaming (Simultaneous DB Storage)**
- [ ] Implement dual stream architecture
  - [ ] Create separate channels for client and storage
  - [ ] Spawn storage task
  - [ ] Send chunks to both streams
- [ ] Implement storage accumulator
  - [ ] Accumulate content as it streams
  - [ ] Store complete assistant message on completion
  - [ ] No partial storage on error
- [ ] Add proper error handling
  - [ ] Continue streaming even if storage fails
  - [ ] Log storage errors for retry
  - [ ] No partial content on stream error
- [ ] Test concurrent streaming and storage

**Phase 6: Tool Calling Framework**
- [ ] Define Tool and FunctionDefinition structs
- [ ] Create ToolExecutor trait
- [ ] Implement ToolRegistry
- [ ] Create example tool: current_time
  - [ ] Implement CurrentTimeExecutor (UTC)
  - [ ] Test execution
- [ ] Integrate tool calling into stream processing
  - [ ] Detect tool calls in stream
  - [ ] Execute tools immediately via ToolRegistry
  - [ ] Store tool_calls records
  - [ ] Store tool outputs
  - [ ] Format tool results as messages
  - [ ] Send back to LLM to continue
  - [ ] Stream continued response to client
- [ ] Test tool calling flow

**Phase 7: Additional Endpoints**
- [ ] Implement GET /v1/responses/{id}
  - [ ] Query user message by ID
  - [ ] Verify user ownership
  - [ ] Build response with usage data
  - [ ] Return formatted response
- [ ] Implement GET /v1/responses (list) [NOT openai standard but needed in our app)
  - [ ] Add pagination support
  - [ ] Query user's responses
  - [ ] Format list response
- [ ] Implement POST /v1/responses/{id}/cancel
  - [ ] Check if status is 'in_progress'
  - [ ] Update status to 'cancelled'
  - [ ] Return updated response object
- [ ] Implement DELETE /v1/responses/{id}
  - [ ] Verify user ownership
  - [ ] Delete response record
  - [ ] Let cascade deletes handle related records
- [ ] Test all endpoints

**Phase 8: Idempotency Support**
- [ ] Add idempotency handling to POST /v1/responses
  - [ ] Check Idempotency-Key header
  - [ ] Hash request body for comparison
  - [ ] Handle in-progress requests (409 Conflict)
  - [ ] Return cached responses
  - [ ] Handle different parameters (422 error)
- [ ] Add cleanup job for expired keys
- [ ] Test idempotency behavior

**Phase 9: Error Handling & Edge Cases**
- [ ] Implement comprehensive error types
- [ ] Add provider error mapping
- [ ] Handle streaming errors gracefully
- [ ] Add timeout handling
- [ ] Test error scenarios

**Phase 10: Performance & Polish**
- [ ] Add caching layer
  - [ ] Cache tiktoken encoders
  - [ ] Cache user encryption keys (LRU)
  - [ ] Cache thread metadata
- [ ] Optimize database queries
  - [ ] Review query plans
  - [ ] Add missing indexes
  - [ ] Batch operations where possible
- [ ] Add metrics and logging
  - [ ] Request duration
  - [ ] Token usage
  - [ ] Error rates
- [ ] Load testing
  - [ ] Test concurrent streams
  - [ ] Measure latencies
  - [ ] Identify bottlenecks

**Phase 11: Documentation & Integration**
- [ ] Update OpenAPI specification
- [ ] Create integration examples
  - [ ] Python client example
  - [ ] TypeScript client example
  - [ ] Curl examples
- [ ] Document migration from chat API
- [ ] Add troubleshooting guide

### Architecture Overview

The Responses API builds on top of the existing infrastructure:
- Requests authenticated via JWT middleware
- Immediate response with SSE streaming if requested
- Calls internal /v1/chat/completions with streaming
- Dual stream: responses sent to client while storing to database
- All data encrypted at rest using existing patterns
- Thread context automatically managed server-side

## Database Schema

The database schema is designed to support the Responses API while maintaining clean separation between different message types. All tables work together to reconstruct conversation history when needed.

### Core Schema Design

The schema splits messages into distinct tables by type, allowing us to:
1. Track each message type with appropriate metadata
2. Reconstruct conversations by ordering all records by timestamp
3. Support the Responses API's status tracking on user messages
4. Maintain clean separation of concerns

### user_system_prompts Table

Stores optional custom system prompts for users.

```sql
-- Table: user_system_prompts (optional custom system prompts)
CREATE TABLE user_system_prompts (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    name_enc BYTEA NOT NULL, -- Encrypted system prompt name (binary ciphertext)
    prompt_enc BYTEA NOT NULL, -- Encrypted system prompt (binary ciphertext)
    prompt_tokens INTEGER, -- Token count for the system prompt
    is_default BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_system_prompts_uuid ON user_system_prompts(uuid);
CREATE INDEX idx_user_system_prompts_user_id ON user_system_prompts(user_id);
-- Ensure only one default prompt per user
CREATE UNIQUE INDEX idx_user_system_prompts_one_default 
    ON user_system_prompts(user_id) 
    WHERE is_default = true;
```

### chat_threads Table

Conversation containers that group related messages. For new conversations (where `previous_response_id` is null), the thread ID matches the first message ID.

```sql
-- Table: chat_threads (conversation containers)
CREATE TABLE chat_threads (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE, -- Set to first user_message uuid for new threads
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    system_prompt_id BIGINT REFERENCES user_system_prompts(id) ON DELETE SET NULL,
    title_enc BYTEA, -- Encrypted title (binary ciphertext)
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_threads_uuid ON chat_threads(uuid);
CREATE INDEX idx_chat_threads_user_id ON chat_threads(user_id);
CREATE INDEX idx_chat_threads_updated ON chat_threads(user_id, updated_at DESC);
```

### user_messages Table

Stores user inputs and tracks Responses API request lifecycle.

**Note on model storage**: The `model` field is only stored in `user_messages` since assistant messages are always generated using the model specified in their parent user message. This avoids redundancy and ensures consistency.

```sql
-- Table: user_messages (user inputs / Responses API requests)
-- Note: 'queued' status is included for future compatibility with async processing,
-- but MVP implementation will skip directly to 'in_progress' status
CREATE TYPE response_status AS ENUM 
  ('queued', 'in_progress', 'completed', 'failed', 'cancelled');

CREATE TABLE user_messages (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE, -- This is the response_id in the API. For new threads (previous_response_id=null), this UUID also becomes the chat_thread(uuid)
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content_enc BYTEA NOT NULL, -- Encrypted user input (binary ciphertext)
    prompt_tokens INTEGER, -- Total tokens in the prompt sent to the model (includes system + all context + user message)
    
    -- Responses API fields
    status response_status NOT NULL DEFAULT 'in_progress',
    model TEXT NOT NULL,
    previous_response_id UUID REFERENCES user_messages(uuid),
    temperature REAL,
    top_p REAL,
    max_output_tokens INTEGER,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN NOT NULL DEFAULT false,
    store BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB,
    error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Idempotency fields
    idempotency_key TEXT,
    request_hash TEXT,
    idempotency_expires_at TIMESTAMP WITH TIME ZONE
);

-- Indexes
CREATE INDEX idx_user_messages_uuid ON user_messages(uuid);
CREATE INDEX idx_user_messages_thread_id ON user_messages(thread_id);
CREATE INDEX idx_user_messages_user_id ON user_messages(user_id);
CREATE INDEX idx_user_messages_status ON user_messages(status);
CREATE INDEX idx_user_messages_previous_uuid ON user_messages(previous_response_id);
-- Composite index for pagination
CREATE INDEX idx_user_messages_thread_created_id 
    ON user_messages(thread_id, created_at DESC, id);

-- Index for efficient conversation reconstruction
CREATE INDEX idx_user_messages_thread_created 
    ON user_messages(thread_id, created_at);
    
-- Index for efficient idempotency lookups (not a unique constraint)
CREATE INDEX idx_user_messages_idempotency 
    ON user_messages(user_id, idempotency_key) 
    WHERE idempotency_key IS NOT NULL;
```

### tool_calls Table

Tracks tool calls requested by the model.

```sql
-- Table: tool_calls (tool invocations by the model)
CREATE TABLE tool_calls (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE, -- Backend generates before streaming
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id BIGINT NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    tool_call_id UUID NOT NULL, -- Unique identifier for the tool call
    name TEXT NOT NULL,
    arguments_enc BYTEA, -- Encrypted arguments (binary ciphertext)
    argument_tokens INTEGER, -- Token count for the tool arguments
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_calls_uuid ON tool_calls(uuid);
CREATE INDEX idx_tool_calls_thread_id ON tool_calls(thread_id);
CREATE INDEX idx_tool_calls_user_message_id ON tool_calls(user_message_id);
CREATE INDEX idx_tool_calls_thread_created_id 
    ON tool_calls(thread_id, created_at DESC, id);

-- Index for efficient conversation reconstruction
CREATE INDEX idx_tool_calls_thread_created 
    ON tool_calls(thread_id, created_at);
```

### tool_outputs Table  

Stores results from tool executions.

```sql
-- Table: tool_outputs (tool execution results)
CREATE TABLE tool_outputs (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE, -- Backend generates before streaming
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    tool_call_fk BIGINT NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    output_enc BYTEA NOT NULL, -- Encrypted output (binary ciphertext) - use {} for empty responses
    output_tokens INTEGER, -- Token count for the tool output
    status TEXT NOT NULL DEFAULT 'succeeded' CHECK (status IN ('succeeded', 'failed')),
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_outputs_uuid ON tool_outputs(uuid);
CREATE INDEX idx_tool_outputs_thread_id ON tool_outputs(thread_id);
CREATE INDEX idx_tool_outputs_tool_call_fk ON tool_outputs(tool_call_fk);
CREATE INDEX idx_tool_outputs_thread_created_id 
    ON tool_outputs(thread_id, created_at DESC, id);

-- Index for efficient conversation reconstruction
CREATE INDEX idx_tool_outputs_thread_created 
    ON tool_outputs(thread_id, created_at);
```

### assistant_messages Table

Stores LLM responses (non-tool responses). The model is not stored here since it can be derived from the parent `user_message_id` - this avoids redundancy and potential inconsistencies.

```sql
-- Table: assistant_messages (LLM responses)
CREATE TABLE assistant_messages (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE, -- Backend generates before streaming
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id BIGINT NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    content_enc BYTEA NOT NULL, -- Encrypted assistant response (binary ciphertext)
    completion_tokens INTEGER, -- Token count for this assistant response
    finish_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_assistant_messages_uuid ON assistant_messages(uuid);
CREATE INDEX idx_assistant_messages_thread_id ON assistant_messages(thread_id);
CREATE INDEX idx_assistant_messages_user_message_id ON assistant_messages(user_message_id);
CREATE INDEX idx_assistant_messages_thread_created_id 
    ON assistant_messages(thread_id, created_at DESC, id);

-- Index for efficient conversation reconstruction
CREATE INDEX idx_assistant_messages_thread_created 
    ON assistant_messages(thread_id, created_at);

-- Apply triggers for updated_at (using existing update_updated_at_column function)
CREATE TRIGGER update_user_system_prompts_updated_at BEFORE UPDATE
    ON user_system_prompts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_chat_threads_updated_at BEFORE UPDATE
    ON chat_threads FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_user_messages_updated_at BEFORE UPDATE
    ON user_messages FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_tool_calls_updated_at BEFORE UPDATE
    ON tool_calls FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_tool_outputs_updated_at BEFORE UPDATE
    ON tool_outputs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_assistant_messages_updated_at BEFORE UPDATE
    ON assistant_messages FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### Migration Management

**Running Migrations:**
```bash
# Run migrations
diesel migration run

# Rollback if needed
diesel migration revert

# Generate schema.rs updates
diesel print-schema > src/schema.rs
```

**Down Migration (`down.sql`):**
```sql
DROP TRIGGER IF EXISTS update_assistant_messages_updated_at ON assistant_messages;
DROP TRIGGER IF EXISTS update_tool_outputs_updated_at ON tool_outputs;
DROP TRIGGER IF EXISTS update_tool_calls_updated_at ON tool_calls;
DROP TRIGGER IF EXISTS update_user_messages_updated_at ON user_messages;
DROP TRIGGER IF EXISTS update_chat_threads_updated_at ON chat_threads;
DROP TRIGGER IF EXISTS update_user_system_prompts_updated_at ON user_system_prompts;
DROP TABLE IF EXISTS assistant_messages;
DROP TABLE IF EXISTS tool_outputs;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS user_messages;
DROP TABLE IF EXISTS chat_threads;
DROP TABLE IF EXISTS user_system_prompts;
DROP TYPE IF EXISTS response_status;
```

## API Endpoints

The following endpoints implement OpenAI Responses API compatibility, with some OpenSecret-specific extensions noted.

### POST /v1/responses

Creates a new response request. This is the single entry point for the Responses API.

**Note**: This endpoint complements the existing `/v1/chat/completions` endpoint. Use `/v1/chat/completions` for simple stateless requests. Use `/v1/responses` when you need:
- Server-managed conversation state with status tracking
- Server-side tool execution
- Conversation continuity via `previous_response_id`
- OpenAI Responses API compatibility

**Thread Management**:
- When `previous_response_id` is null, a new thread is created with `thread.id = message.id`
- When `previous_response_id` is provided, the conversation continues in the existing thread
- Thread titles are auto-generated after the first few messages and stored in `chat_threads.title_enc`

**Headers:**
- `Authorization: Bearer <token>` (required)
- `Idempotency-Key: <unique-key>` (optional, recommended for retries)

**Request Body:**
```json
{
  "model": "gpt-4",
  "input": "Explain quantum computing",
  "tools": [
    {
      "type": "function", 
      "function": {
        "name": "current_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0.7,
  "top_p": 1.0,
  "max_output_tokens": 150,
  "parallel_tool_calls": true,
  "store": true,
  "metadata": {},
  "stream": true,
  "previous_response_id": null // null for new thread, or UUID to continue
}
```

**Response (Immediate):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "object": "response",
  "created": 1677652288,
  "model": "gpt-4",
  "status": "in_progress"
}
```

**SSE Stream Events (if stream=true):**
```
event: response.delta
data: {"content": "Let me check the current time for you."}

event: tool.call
data: {"id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "name": "current_time", "arguments": "{}"}

event: tool.result
data: {"tool_call_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "content": "{\"time\": \"2024-01-15T10:30:00Z\", \"timezone\": \"UTC\"}"}

event: response.delta
data: {"content": "The current time is 10:30 AM UTC on January 15, 2024."}

event: response.done
data: {"id": "550e8400-e29b-41d4-a716-446655440000", "status": "completed", "usage": {"prompt_tokens": 25, "completion_tokens": 45}}

// Example: Cancelled response
event: response.done
data: {"id": "550e8400-e29b-41d4-a716-446655440000", "status": "cancelled"}

### GET /v1/responses/{response_id}

Get the status and result of a response (for polling if not using SSE).

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "object": "response",
  "created": 1677652288,
  "status": "completed",
  "model": "gpt-4",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "output": "Quantum computing is a revolutionary approach to computation that leverages quantum mechanical phenomena..."
}
```

### GET /v1/responses

List responses for the authenticated user with pagination.

**Note**: This is a custom OpenSecret endpoint for convenience, not part of the standard OpenAI Responses API. OpenAI only supports GET /v1/responses/{response_id} for retrieving individual responses.

**Query Parameters:**
- `limit` (integer, default: 20, max: 100): Number of responses to return
- `after` (string): Cursor for pagination (response ID to start after)
- `before` (string): Cursor for pagination (response ID to start before)
- `order` (string, default: "desc"): Sort order by created_at ("asc" or "desc")

**Pagination Design**: We use `after`/`before` parameters following OpenAI's standard list endpoint patterns (e.g., list assistants, list files), not to be confused with the `starting_after` parameter in GET /v1/responses/{response_id} which is for streaming event sequences.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "object": "response",
      "created": 1677652288,
      "status": "completed",
      "model": "gpt-4",
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60
      }
    }
  ],
  "has_more": true,
  "first_id": "550e8400-e29b-41d4-a716-446655440000",
  "last_id": "6ba7b814-9dad-11d1-80b4-00c04fd430c8"
}
```

### POST /v1/responses/{response_id}/cancel

Cancel a model response with the given ID. Only responses with status `in_progress` can be cancelled.

**Response (Success - 200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "object": "response",
  "created": 1677652288,
  "status": "cancelled",
  "model": "gpt-4"
}
```

**Response (Error - 400 Bad Request):**
```json
{
  "error": {
    "message": "Cannot cancel response with status 'completed'",
    "type": "invalid_request_error",
    "code": "response_not_cancelable"
  }
}
```

**Implementation Notes:**
- Only responses with `status = 'in_progress'` can be cancelled
- Updates status to 'cancelled' if current status allows
- Returns the updated response object
- No version/optimistic locking needed - simple status check is sufficient
- For streaming responses: Send `response.done` event with `"status": "cancelled"` before closing the SSE connection

### DELETE /v1/responses/{response_id}

Delete a model response with the given ID. This permanently removes the response and all associated data.

**Response (Success - 200 OK):**
```json
{
  "id": "resp_677efb5139a88190b512bc3fef8e535d",
  "object": "response.deleted",
  "deleted": true
}
```

**Response (Error - 404 Not Found):**
```json
{
  "error": {
    "message": "Response not found",
    "type": "invalid_request_error",
    "code": "resource_not_found"
  }
}
```

**Implementation Notes:**
- Permanently deletes the response record from the database
- Cascading deletes automatically clean up related tool_calls, tool_outputs, and assistant_messages
- Returns 200 OK with deletion confirmation on successful deletion
- Returns 404 if the response doesn't exist or doesn't belong to the user

## Message Encryption Strategy

All user content is encrypted at rest using the same patterns as the existing KV store implementation.

### Encryption Keys

1. **User Master Key**: Derived from `users.seed_enc` using KDF
2. **Encryption Key**: AES-256 key derived from master key
3. **No per-thread keys**: All messages for a user use the same encryption key

### Encryption Methods

**Non-deterministic Encryption (AES-256-GCM)**
- Used for ALL encrypted fields: `content_enc`, `title_enc`, `arguments_enc`, `output_enc`, `name_enc`, `prompt_enc`
- Random nonce for each encryption
- Same plaintext produces different ciphertext each time
- More secure than deterministic encryption
- Use existing standards for user-based encryption in the codebase currently. Like the value in the KV store.

Note: Unlike the KV store (src/kv.rs), the Responses API does NOT use deterministic encryption. Thread and message lookups use unencrypted UUIDs, not encrypted fields.

**Unencrypted Fields**
- UUIDs (thread_id, message_id, user_id)
- Metadata (timestamps, status, model names)
- These contain no sensitive user content - just identifiers and operational data

### Implementation Pattern

```rust
// Encryption pattern for message content (following src/kv.rs pattern)
use crate::encrypt::{encrypt_with_key, decrypt_with_key};
use secp256k1::SecretKey;

// User's encryption key from seed_enc
let user_key: &SecretKey = &user.seed_enc;

// All encryption uses AES-256-GCM (non-deterministic) via encrypt_with_key
let content_enc = encrypt_with_key(user_key, message_content.as_bytes()).await;
let title_enc = encrypt_with_key(user_key, thread_title.as_bytes()).await;
let args_enc = encrypt_with_key(user_key, tool_args.as_bytes()).await;

// Decrypting on retrieval
let content = decrypt_with_key(user_key, &content_enc)?;
let content_str = String::from_utf8(content)?;
```

### API Layer Encryption

1. **Request Flow**:
   - Client sends encrypted request via encryption middleware
   - Server decrypts request using session key
   - Server processes and stores data with user encryption key
   - Server encrypts response with session key

2. **Streaming Encryption**:
   - Each SSE chunk is encrypted before sending
   - Uses the same session encryption as regular responses
   - Client decrypts chunks in real-time

### Security Considerations

- All message content uses non-deterministic encryption
- User encryption keys are stored encrypted in database with the KMS key that only the enclave server has access to
- Keys derived on-demand from user seed
- Follows existing `encryption_middleware` patterns

## Idempotency

The Responses API supports idempotent requests to ensure safe retries.

### Implementation

Idempotency is handled at the application level with the help of columns in the `user_messages` table:

**Application-Level Approach**:
- Check for existing idempotency key before inserting
- Include expiry time in the query to ignore expired keys
- No complex database constraints needed
- Simple and predictable behavior

```sql
-- Idempotency columns are already included in the user_messages table:
-- idempotency_key TEXT
-- request_hash TEXT  
-- idempotency_expires_at TIMESTAMP WITH TIME ZONE

-- Index for efficient idempotency lookups
CREATE INDEX idx_user_messages_idempotency 
    ON user_messages(user_id, idempotency_key) 
    WHERE idempotency_key IS NOT NULL;
```

**Query Pattern**:
```sql
-- Check for existing non-expired idempotency key
SELECT * FROM user_messages 
WHERE user_id = $1 
  AND idempotency_key = $2 
  AND idempotency_expires_at > NOW()  -- Evaluated at query time
LIMIT 1;
```

### Request Handling

```rust
// Idempotency key handling - simple application-level check
if let Some(idempotency_key) = headers.get("idempotency-key") {
    // Canonicalize and hash the request body
    let request_json = serde_json::to_string(&request)?;
    let request_hash = sha256::digest(request_json);
    
    // Check if we've seen this key before for this user within the expiry window
    if let Some(existing) = db.get_user_message_by_idempotency_key(
        &idempotency_key, 
        user.id
    ).await? {
        // Verify the request hasn't changed
        if existing.request_hash != Some(request_hash.clone()) {
            return Err(ApiError::IdempotencyKeyReused {
                message: "Different request body with same idempotency key".into(),
                status: StatusCode::UNPROCESSABLE_ENTITY, // 422
            });
        }
        
        // Check if request is still in progress
        match existing.status {
            ResponseStatus::InProgress => {
                // Request is still being processed
                return Err(ApiError::RequestInProgress {
                    message: "Request with this idempotency key is already in progress".into(),
                    status: StatusCode::CONFLICT, // 409
                    retry_after: Some(5), // seconds
                });
            }
            _ => {
                // Return cached response (completed, failed, or cancelled)
                return Ok(Json(build_response_from_message(existing).await?).into_response());
            }
        }
    }
    
    // Include idempotency info when creating the user message
    let message = create_user_message(
        request,
        Some(idempotency_key),
        Some(request_hash),
        Some(Utc::now() + Duration::hours(24))
    ).await?;
}
```

### Key Properties

- **One execution per key**: Each idempotency key triggers exactly one LLM call
- **Conflict on concurrent**: Returns 409 if request is still processing
- **Parameter validation**: Different parameters with same key returns 422 error
- **24-hour expiration**: Keys automatically expire and can be reused after 24 hours
- **User-scoped**: Keys are unique per user, not globally

### Response Behavior by Status

| Status | Behavior |
|--------|----------|
| `in_progress` | Return 409 Conflict - request in progress |
| `completed` | Return cached successful response |
| `failed`, `cancelled` | Return cached error response |
| Key not found | Process new request |
| Different parameters | Return 422 Unprocessable Entity |

### Cleanup

```sql
-- Optional periodic cleanup to remove old idempotency data and save storage
UPDATE user_messages 
SET idempotency_key = NULL, request_hash = NULL, idempotency_expires_at = NULL
WHERE idempotency_expires_at < CURRENT_TIMESTAMP
  AND idempotency_key IS NOT NULL;
```

**Note**: Since we handle expiry checks in the application code when querying, this cleanup
is optional and only serves to reduce storage usage.

## Request Processing Architecture

The Responses API processes requests synchronously with dual streaming for real-time responses and persistent storage.

### Efficient Conversation Reconstruction

Building conversation context is the most performance-critical operation, happening on every request. For MVP, we'll use a single UNION ALL query which is more efficient than 4 separate queries.

**Note**: For advanced optimization patterns (caching, parallel decryption, materialized views), see the [Future Enhancements](#future-enhancements) section.

#### Query Strategy: Single UNION ALL Query

```sql
-- Single query to get all message types with token counts ordered by timestamp
WITH thread_messages AS (
    -- User messages
    SELECT 
        'user' as message_type,
        um.id,
        um.uuid,
        um.content_enc,
        um.created_at,
        um.model,
        um.prompt_tokens as token_count,
        NULL as tool_call_id,
        NULL as finish_reason
    FROM user_messages um
    WHERE um.thread_id = $1
    
    UNION ALL
    
    -- Assistant messages
    SELECT 
        'assistant' as message_type,
        am.id,
        am.uuid,
        am.content_enc,
        am.created_at,
        um.model, -- Get model from parent user message
        am.completion_tokens as token_count,
        NULL as tool_call_id,
        am.finish_reason
    FROM assistant_messages am
    JOIN user_messages um ON am.user_message_id = um.id
    WHERE am.thread_id = $1
    
    UNION ALL
    
    -- Tool calls
    SELECT 
        'tool_call' as message_type,
        tc.id,
        tc.uuid,
        tc.arguments_enc as content_enc,
        tc.created_at,
        NULL as model,
        tc.argument_tokens as token_count,
        tc.tool_call_id,
        NULL as finish_reason
    FROM tool_calls tc
    WHERE tc.thread_id = $1
    
    UNION ALL
    
    -- Tool outputs
    SELECT 
        'tool_output' as message_type,
        tto.id,
        tto.uuid,
        tto.output_enc as content_enc,
        tto.created_at,
        NULL as model,
        tto.output_tokens as token_count,
        tc.tool_call_id,
        NULL as finish_reason
    FROM tool_outputs tto
    JOIN tool_calls tc ON tto.tool_call_fk = tc.id
    WHERE tto.thread_id = $1
)
SELECT * FROM thread_messages
ORDER BY created_at ASC;
```

#### Basic Implementation

```rust
// MVP implementation - efficient but not overly complex
impl DBConnection for AsyncPgConnection {
    async fn get_thread_context(
        &mut self,
        thread_id: i64,
        user_id: Uuid,
    ) -> Result<Vec<ChatCompletionMessage>> {
        // 1. Verify thread ownership
        let thread = chat_threads::table
            .filter(chat_threads::id.eq(thread_id))
            .filter(chat_threads::user_id.eq(user_id))
            .first::<ChatThread>(self)
            .await?;
            
        // 2. Get user encryption key
        let user_key: &SecretKey = &user.seed_enc;
        
        // 3. Run the UNION ALL query to get ALL messages with token counts
        const THREAD_MESSAGES_QUERY: &str = include_str!("../sql/thread_messages.sql");
        let raw_messages = diesel::sql_query(THREAD_MESSAGES_QUERY)
            .bind::<BigInt, _>(thread_id)
            .load::<RawThreadMessage>(self)
            .await?;
            
        // 4. Convert all messages to chat format (no truncation here)
        let mut messages = Vec::new();
        let mut token_counts = Vec::new();
        
        // Add system prompt if exists
        if let Some(system_prompt) = thread.system_prompt {
            let decrypted_prompt = decrypt_content(&system_prompt.prompt_enc, &user_key)?;
            messages.push(ChatCompletionMessage {
                role: "system".to_string(),
                content: decrypted_prompt,
            });
            // Use stored token count if available
            if let Some(tokens) = system_prompt.prompt_tokens {
                token_counts.push(tokens);
            }
        }
        
        // Process all messages
        for raw_msg in raw_messages {
            let decrypted_content = decrypt_content(&raw_msg.content_enc, &user_key)?;
            
            // Convert to chat format based on message type
            let chat_msg = match raw_msg.message_type.as_str() {
                "user" => ChatCompletionMessage {
                    role: "user".to_string(),
                    content: decrypted_content,
                },
                "assistant" => ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: decrypted_content,
                },
                "tool_output" => ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: decrypted_content,
                    tool_call_id: Some(raw_msg.tool_call_id.unwrap()),
                },
                _ => continue, // Skip tool_call for now
            };
            
            messages.push(chat_msg);
            
            // Store token count if available
            if let Some(tokens) = raw_msg.token_count {
                token_counts.push(tokens);
            }
        }
        
        Ok((messages, token_counts))
    }
}
```

#### Required Indexes

```sql
-- Essential indexes for the UNION ALL query
CREATE INDEX idx_user_messages_thread_created 
ON user_messages(thread_id, created_at);

CREATE INDEX idx_assistant_messages_thread_created 
ON assistant_messages(thread_id, created_at);

CREATE INDEX idx_tool_calls_thread_created 
ON tool_calls(thread_id, created_at);

CREATE INDEX idx_tool_outputs_thread_created 
ON tool_outputs(thread_id, created_at);
```

### Request Flow

1. **Initial Request**:
   - Validate JWT and decrypt request
   - Check for idempotency key if provided
   - Generate message ID (or reuse from idempotency check)
   - Determine thread handling:
     - If `previous_response_id` is null: Create new thread with `thread.id = message.id`
     - If `previous_response_id` is provided: Look up thread from previous message
   - Insert user_message record with status='in_progress' (MVP skips 'queued' status for synchronous processing)
   - Build conversation context from thread history
   - If stream=true, upgrade to SSE connection
   - Call internal chat API and process response

2. **Synchronous Processing**:
   ```rust
   // Inside the request handler
   async fn process_response(
       state: &AppState,
       user: &User,
       user_message: &UserMessage,
       context: Vec<Message>,
   ) -> Result<()> {
       // Create dual streams - one for client, one for storage
       let (client_tx, client_rx) = channel(1024);
       let (storage_tx, storage_rx) = channel(1024);
       
       // Spawn storage task
       let storage_task = tokio::spawn(async move {
           let mut content = String::new();
           let mut tool_calls = Vec::new();
           
           while let Some(chunk) = storage_rx.recv().await {
               match chunk {
                   StreamChunk::Content(text) => content.push_str(&text),
                   StreamChunk::ToolCall(call) => {
                       store_tool_call(&db, user_message, &call).await?;
                       tool_calls.push(call);
                   }
                   StreamChunk::Done(usage) => {
                       store_assistant_message(&db, user_message, &content, usage).await?;
                       update_user_message_status(user_message.id, "completed").await?;
                       break;
                   }
               }
           }
           Ok::<(), Error>(())
       });
       
       // Stream from internal chat API
       let stream = call_internal_chat_api(&state.llm, context).await?;
       
       // Process stream - send to both client and storage
       while let Some(chunk) = stream.next().await {
           // Send to client
           if let Some(sse_event) = format_sse_event(&chunk) {
               client_tx.send(sse_event).await?;
           }
           
           // Send to storage
           storage_tx.send(chunk).await?;
       }
       
       // Wait for storage to complete
       storage_task.await??;
       Ok(())
   }
   ```

3. **Message Storage**:
   - User message: Stored immediately on request
   - Assistant message: Stored after streaming completes
   - Tool calls: Stored as they arrive
   - Metadata: Updated when stream ends

### SSE Event Format

Following OpenAI Responses API event format:

#### Heartbeat Frames

To prevent browser timeouts (typically ~120s), the server sends periodic heartbeat frames:

```
: heartbeat\n\n
```

These are SSE comment frames that keep the connection alive without affecting the data stream.

#### Data Events

```
event: response.created
data: {"id": "550e8400-e29b-41d4-a716-446655440000", "status": "in_progress"}

event: response.delta
data: {"content": "Here's what I know about"}

event: tool.call
data: {"id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}

event: response.done
data: {"id": "550e8400-e29b-41d4-a716-446655440000", "status": "completed", "usage": {"prompt_tokens": 25, "completion_tokens": 150}}
```

### Error Handling During Streaming

1. **Upstream Errors**: 
   - Log error and store partial response
   - Send error event to client
   - Mark message with error status

2. **Database Errors**:
   - Continue streaming to client
   - Log error for later retry
   - Try a couple times, it's important to try to persist this

3. **Client Disconnection**:
   - Continue processing for database
   - Complete message storage
   - Clean up resources

### Implementation Considerations

- Use tokio broadcast channel for fan-out
- Buffer size tuning for performance
- Graceful shutdown handling
- Metrics for stream latency
- Connection pooling for database writes

## Usage Tracking & Analytics

Token usage data is not stored in the database but is broadcast via SQS for centralized analytics and billing:

```rust
// Usage is broadcast via SQS, not stored in assistant_messages
let usage_event = UsageEvent {
    event_id: Uuid::new_v4(),
    user_id: user.id,
    input_tokens: prompt_tokens,
    output_tokens: completion_tokens,
    estimated_cost: calculate_cost(model, prompt_tokens, completion_tokens),
    chat_time: Utc::now(),
};

sqs_publisher.publish_event(usage_event).await?;
```

This pattern:
- Keeps the response tables focused on conversation data
- Centralizes usage tracking for all API endpoints
- Enables real-time billing and analytics
- Avoids the need for expensive JSONB indexes on usage data
- We already do this for completions API so I don't think we need to add anything specifically since we should be usig completions under the hood

## Token Management Strategy

Manage conversation context within model-specific token limits using intelligent truncation.

### Model Token Limits

```rust
// Model-specific context window limits
pub const MODEL_LIMITS: &[(&str, usize)] = &[
    // Free Tier
    ("ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 70_000), // Llama 3.3 70B
    
    // Starter Tier
    ("google/gemma-3-27b-it", 70_000), // Gemma 3 27B
    ("leon-se/gemma-3-27b-it-fp8-dynamic", 70_000), // Gemma 3 27B (vision)
    
    // Pro Tier
    ("deepseek-r1-70b", 64_000), // DeepSeek R1 70B
    ("mistral-small-3-1-24b", 128_000), // Mistral Small 3.1 24B (vision)
    ("qwen2-5-72b", 128_000), // Qwen 2.5 72B
];

pub fn get_model_max_tokens(model: &str) -> usize {
    MODEL_LIMITS.iter()
        .find(|(m, _)| model.starts_with(m))
        .map(|(_, limit)| *limit)
        .unwrap_or(64_000) // Default for unknown models
}
```

### Token Counting

1. **Libraries**:
   - Use `tiktoken-rs` for accurate OpenAI token counting
   - Cache encoder instances per model
   - Count tokens during message storage

2. **Storage Strategy**:
   - Store token counts in each message table as they're created
   - `user_messages.prompt_tokens`: Total tokens in the prompt (system + context + user message)
   - `assistant_messages.completion_tokens`: Tokens in the assistant response
   - `tool_calls.argument_tokens`: Tokens in the tool arguments
   - `tool_outputs.output_tokens`: Tokens in the tool output
   - `user_system_prompts.prompt_tokens`: Tokens in custom system prompts

3. **Implementation Pattern**:
   ```rust
   use tiktoken::cl100k_base;
   use std::sync::OnceLock;
   
   // Cache the encoder at the module level
   static ENCODER: OnceLock<tiktoken::CoreBPE> = OnceLock::new();
   
   pub fn count_tokens(text: &str) -> Result<i32, Error> {
       let encoder = ENCODER.get_or_init(|| cl100k_base().unwrap());
       let tokens = encoder.encode_with_special_tokens(text);
       Ok(tokens.len() as i32)
   }
   
   // When storing a user message
   async fn store_user_message(
       conn: &mut AsyncPgConnection,
       user_id: Uuid,
       content: &str,
       prompt_tokens: i32, // Already calculated from full prompt
   ) -> Result<UserMessage> {
       let encrypted_content = encrypt_with_key(&user_key, content.as_bytes()).await;
       
       let new_message = NewUserMessage {
           uuid: Uuid::new_v4(),
           user_id,
           content_enc: encrypted_content,
           prompt_tokens: Some(prompt_tokens),
           // ... other fields
       };
       
       diesel::insert_into(user_messages::table)
           .values(&new_message)
           .get_result(conn)
           .await
   }
   
   // When storing assistant messages from streaming
   async fn store_assistant_message(
       conn: &mut AsyncPgConnection,
       user_message_id: i64,
       content: &str,
       completion_tokens: Option<i32>, // From stream or calculated
   ) -> Result<AssistantMessage> {
       let tokens = completion_tokens.unwrap_or_else(|| count_tokens(content).unwrap_or(0));
       
       let encrypted_content = encrypt_with_key(&user_key, content.as_bytes()).await;
       
       let new_message = NewAssistantMessage {
           uuid: Uuid::new_v4(),
           user_message_id,
           content_enc: encrypted_content,
           completion_tokens: Some(tokens),
           // ... other fields
       };
       
       diesel::insert_into(assistant_messages::table)
           .values(&new_message)
           .get_result(conn)
           .await
   }
   ```

4. **Token Counting During Streaming**:
   - Extract token counts from SSE `usage` events when available
   - Fall back to manual counting if not provided by the model
   - Store counts immediately when creating database records

### Context Window Management

1. **Building Prompts**:
   ```rust
   // Build the final prompt for the LLM with context window management
   async fn build_conversation_prompt(
       thread_id: i64,
       user_id: Uuid,
       model: &str,
       new_message: &str,
       new_message_tokens: i32,
   ) -> Result<Vec<ChatCompletionMessage>> {
       let max_tokens = get_model_max_tokens(model);
       
       // Calculate tokens available for context using explicit formula
       // available = max_context - response_reserve - tool_reserve - safety_margin
       let response_reserve = 4096; // Max tokens we'll allow for response
       let tool_reserve = if has_tools { 1000 } else { 0 }; // Space for tool calls/results
       let safety_margin = 500; // Buffer to avoid edge cases
       
       let context_limit = max_tokens
           .saturating_sub(response_reserve)
           .saturating_sub(tool_reserve)
           .saturating_sub(safety_margin);
       
       // Get ALL messages from the thread efficiently using the UNION ALL query
       // This now returns both messages and their token counts
       let (all_messages, token_counts) = get_thread_context(thread_id, user_id).await?;
       
       // Calculate total tokens using stored counts
       let mut total_tokens = new_message_tokens as usize;
       
       // Sum up existing message tokens using stored counts
       for &count in &token_counts {
           total_tokens += count as usize;
       }
       
       // Apply middle truncation strategy if needed
       if total_tokens > context_limit {
           let mut included_messages = Vec::new();
           let mut included_tokens = Vec::new();
           let mut running_tokens = new_message_tokens as usize;
           
           // Step 1: Keep first 2-3 messages (system prompt + initial exchange)
           let mut preserved_count = 0;
           let mut first_messages_end_idx = 0;
           
           for (idx, (msg, &tokens)) in all_messages.iter().zip(&token_counts).enumerate() {
               // Always include system prompt
               if msg.role == "system" {
                   included_messages.push(msg.clone());
                   included_tokens.push(tokens);
                   running_tokens += tokens as usize;
                   preserved_count += 1;
               } 
               // Include first user message
               else if preserved_count == 1 && msg.role == "user" {
                   included_messages.push(msg.clone());
                   included_tokens.push(tokens);
                   running_tokens += tokens as usize;
                   preserved_count += 1;
               }
               // Include first assistant response
               else if preserved_count == 2 && msg.role == "assistant" {
                   included_messages.push(msg.clone());
                   included_tokens.push(tokens);
                   running_tokens += tokens as usize;
                   preserved_count += 1;
                   first_messages_end_idx = idx;
                   break;
               }
               // If no system prompt, start with first user message
               else if preserved_count == 0 && msg.role == "user" {
                   included_messages.push(msg.clone());
                   included_tokens.push(tokens);
                   running_tokens += tokens as usize;
                   preserved_count += 1;
               }
           }
           
           // Step 2: Keep most recent messages, maintaining role alternation
           let mut recent_messages = Vec::new();
           let mut recent_tokens = Vec::new();
           let mut last_role = None;
           
           // Iterate through messages in reverse with their token counts
           for ((msg, &tokens)) in all_messages.iter().zip(&token_counts).rev() {
               // Don't duplicate already included messages
               if included_messages.iter().any(|m| m == msg) {
                   continue;
               }
               
               // Check if we can include this message
               if running_tokens + tokens as usize <= context_limit {
                   // Ensure role alternation is maintained
                   if let Some(ref prev_role) = last_role {
                       if prev_role == &msg.role {
                           // Skip to maintain alternation
                           continue;
                       }
                   }
                   
                   recent_messages.push(msg.clone());
                   recent_tokens.push(tokens);
                   running_tokens += tokens as usize;
                   last_role = Some(msg.role.clone());
               } else {
                   break;
               }
           }
           
           // Step 3: Ensure we have proper role alternation
           recent_messages.reverse();
           
           // Check if truncation occurred
           let total_original = all_messages.len();
           let total_preserved = included_messages.len() + recent_messages.len();
           
           if total_preserved < total_original {
               // Find the first role in recent_messages
               if let Some(first_recent) = recent_messages.first() {
                   // If the last included message has the same role as first recent, we need a truncation message
                   if let Some(last_included) = included_messages.last() {
                       if last_included.role == first_recent.role || 
                          (last_included.role == "assistant" && first_recent.role == "assistant") {
                           // Insert user truncation message
                           included_messages.push(ChatCompletionMessage {
                               role: "user".to_string(),
                               content: "[Previous messages truncated due to context limits]".to_string(),
                               name: None,
                               tool_calls: None,
                               tool_call_id: None,
                           });
                       }
                   }
               }
           }
           
           // Combine all messages
           included_messages.extend(recent_messages);
           
           Ok(included_messages)
       } else {
           // Include all messages if under limit
           Ok(all_messages)
       }
   }
   ```

2. **Middle Truncation Strategy**:
   - Keep first 2-3 messages:
     - System prompt (if exists)
     - First user message
     - First assistant response
   - Keep most recent N messages while maintaining role alternation
   - Remove middle messages if over limit
   - Add truncation indicator as user message: `[Previous messages truncated due to context limits]`
   - Ensure proper role alternation (no consecutive same-role messages)

3. **Role Alternation Algorithm**:
   ```rust
   // Example of proper truncation with role alternation
   // Original messages:
   // [system] You are helpful
   // [user] Hello
   // [assistant] Hi there!
   // [user] Question 1
   // [assistant] Answer 1
   // [user] Question 2     <- Start removal here
   // [assistant] Answer 2  <- Remove
   // [user] Question 3     <- Remove
   // [assistant] Answer 3  <- Remove
   // [user] Latest question
   // [assistant] Latest answer
   
   // Result after truncation:
   // [system] You are helpful
   // [user] Hello
   // [assistant] Hi there!
   // [user] [Previous messages truncated due to context limits]
   // [assistant] Answer 3  <- This ensures alternation
   // [user] Latest question
   // [assistant] Latest answer
   ```

4. **Advanced Truncation** (Future):
   ```rust
   // Smart truncation: summarize removed messages
   if needs_truncation {
       let summary = summarize_messages(truncated_messages).await?;
       messages.insert(1, Message {
           role: "system",
           content: format!("Summary of earlier conversation: {}", summary),
           ..
       });
   }
   ```

### Token Estimation

**Token Budget Formula**:
```
available_for_context = max_context_window - response_reserve - tool_reserve - safety_margin
available_for_response = response_reserve (capped at 4096)
```

**Token Reserves**:
- **Response reserve**: 4096 tokens (maximum response length)
- **Tool call reserve**: 1000 tokens when tools are enabled (for arguments + results)
- **Safety margin**: 500 tokens (buffer for token counting variations)

**Implementation Notes**:
- Use `saturating_sub` to prevent underflow
- Token counts are estimates and may vary slightly between models
- Always validate context_limit > 0 before building prompt
- Set `max_tokens` parameter in completion request to `min(4096, available_response_tokens)`

## Tool Calling Framework

Extensible framework for function calling with OpenAI-compatible format.

### Tool Calling Modes: Single vs Parallel

#### MVP: Single Tool Calling Only

For the initial MVP implementation, we will **only support single tool calling** (parallel_tool_calls = false). This simplifies the implementation while still providing core functionality.

**Single Tool Calling Behavior:**
- Model calls one tool at a time
- Each tool call is processed sequentially
- Model waits for tool result before calling next tool
- Simpler state management and error handling

**Request Configuration (MVP):**
```json
{
  "model": "gpt-4",
  "input": "What's the weather in NYC and London?",
  "tools": [...],
  "parallel_tool_calls": false  // Always false in MVP
}
```

**Streaming Pattern (Single Tool):**
```
event: response.delta
data: {"content": "I'll check the weather for you."}

event: tool.call
data: {"id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}

event: tool.result
data: {"tool_call_id": "call_1", "content": "{\"temp\": 72, \"condition\": \"sunny\"}"}

event: response.delta
data: {"content": "In New York City, it's 72°F and sunny. Now let me check London."}

event: tool.call
data: {"id": "call_2", "name": "get_weather", "arguments": "{\"city\": \"London\"}"}

event: tool.result
data: {"tool_call_id": "call_2", "content": "{\"temp\": 59, \"condition\": \"cloudy\"}"}

event: response.delta
data: {"content": "In London, it's 59°F and cloudy."}

event: response.done
data: {"status": "completed", "usage": {...}}
```

#### Future: Parallel Tool Calling Support

When we add parallel tool calling support (parallel_tool_calls = true), the implementation will need to handle:

**Parallel Tool Calling Behavior:**
- Model can request multiple tools in one response
- All tools execute concurrently
- Results collected before continuing
- More complex state management required

**Request Configuration (Future):**
```json
{
  "model": "gpt-4",
  "input": "What's the weather in NYC, London, and Tokyo?",
  "tools": [...],
  "parallel_tool_calls": true  // Enable parallel execution
}
```

**Streaming Pattern (Parallel Tools):**
```
event: response.delta
data: {"content": "I'll check the weather in all three cities for you."}

event: tool.call
data: {"id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}

event: tool.call
data: {"id": "call_2", "name": "get_weather", "arguments": "{\"city\": \"London\"}"}

event: tool.call
data: {"id": "call_3", "name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}

// Tools execute in parallel
event: tool.result
data: {"tool_call_id": "call_2", "content": "{\"temp\": 59, \"condition\": \"cloudy\"}"}

event: tool.result
data: {"tool_call_id": "call_1", "content": "{\"temp\": 72, \"condition\": \"sunny\"}"}

event: tool.result
data: {"tool_call_id": "call_3", "content": "{\"temp\": 68, \"condition\": \"rainy\"}"}

event: response.delta
data: {"content": "Here's the weather in all three cities:\n- NYC: 72°F, sunny\n- London: 59°F, cloudy\n- Tokyo: 68°F, rainy"}

event: response.done
data: {"status": "completed", "usage": {...}}
```

**Implementation Changes for Parallel Support:**

1. **Concurrent Execution:**
   ```rust
   // Future implementation for parallel tools
   let tool_futures: Vec<_> = tool_calls
       .iter()
       .map(|call| {
           let executor = tool_registry.get(&call.name)?;
           tokio::spawn(async move {
               executor.execute(&call.arguments_enc, &user_context).await
           })
       })
       .collect();
   
   // Wait for all tools to complete
   let results = futures::future::join_all(tool_futures).await;
   ```

2. **State Management:**
   - Track multiple in-flight tool calls
   - Handle partial failures gracefully
   - Maintain order for result association

3. **Database Considerations:**
   - Batch insert tool_calls records
   - Handle concurrent tool_outputs inserts
   - Maintain consistency with transactions

4. **Error Handling:**
   - Decide whether to fail all or continue with successful tools
   - Report individual tool failures to model
   - Let model decide how to proceed

**Migration Path:**
1. Start with single tool calling (MVP)
2. Validate core functionality works correctly
3. Add parallel execution infrastructure
4. Enable based on model capabilities and user settings
5. Maintain backward compatibility

### Tool Definition

Tools are defined in the request or at the thread level:

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String, // "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

// Example: current_time tool
pub fn current_time_tool() -> Tool {
    Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: "current_time".to_string(),
            description: "Get the current time in a specific timezone".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone (e.g., 'UTC', 'America/New_York')",
                        "default": "UTC"
                    }
                },
                "required": []
            }),
        },
    }
}
```

### Tool Execution

```rust
// Tool executor trait and registry implementation
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        name: &str,
        arguments: serde_json::Value,
        user_context: &UserContext,
    ) -> Result<serde_json::Value, ToolError>;
}

pub struct ToolRegistry {
    executors: HashMap<String, Box<dyn ToolExecutor>>,
}

// Implementation for current_time
pub struct CurrentTimeExecutor;

#[async_trait]
impl ToolExecutor for CurrentTimeExecutor {
    async fn execute(
        &self,
        _name: &str,
        arguments: serde_json::Value,
        _user_context: &UserContext,
    ) -> Result<serde_json::Value, ToolError> {
        let timezone = arguments.get("timezone")
            .and_then(|v| v.as_str())
            .unwrap_or("UTC");
        
        let tz: chrono_tz::Tz = timezone.parse()
            .map_err(|_| ToolError::InvalidArgument("Invalid timezone".into()))?;
        
        let now = chrono::Utc::now().with_timezone(&tz);
        
        Ok(json!({
            "time": now.to_rfc3339(),
            "timezone": timezone,
            "unix_timestamp": now.timestamp()
        }))
    }
}

// Note: When tools fail or return no data, store {} as the output
// Example for failed tool execution:
// output_enc = encrypt_with_key(user_key, b"{}")
```

### Tool Call Flow

1. **Model requests tool call** (detected in SSE stream):
   ```json
   {
     "tool_calls": [{
       "id": "7ba7b815-9dad-11d1-80b4-00c04fd430c8",
       "type": "function",
       "function": {
         "name": "current_time",
         "arguments": "{\"timezone\": \"America/New_York\"}"
       }
     }]
   }
   ```

2. **Server executes tool automatically**:
   - Parse tool call from stream
   - Execute via ToolRegistry immediately
   - Store tool call and output in database (use `{}` for empty responses)
   - Format tool response as message
   - Send back to LLM to continue conversation

3. **LLM continues with tool result**:
   - Tool result included in conversation context
   - Model processes tool output and responds
   - May call more tools or provide final answer
   - All happens within same request/response cycle

### Security Considerations

1. **Input Validation**:
   - Validate against JSON Schema
   - Sanitize user inputs
   - Rate limit tool calls (max consecutive tool calls) 

2. **Execution Isolation**:
   - Run in separate tokio task
   - Timeout enforcement
   - Resource limits

3. **Audit Trail**:
   - Log all executions
   - Track in tool_executions table
   - Monitor for abuse

### Future Tool Examples

```rust
// Database query tool (with permissions)
database_query_tool()

// Web search tool
web_search_tool()

// Email sending tool
send_email_tool()

// Custom user tools
user_defined_tool()
```

## Request/Response Flow

### Responses API Flow

```
Client              API Gateway         Responses API          Chat API         Database
  |                     |                    |                    |                |
  |--POST /v1/--------->|                    |                    |                |
  |  responses          |--JWT Validate----->|                    |                |
  |                     |--Decrypt Request-->|                    |                |
  |                     |                    |--Insert User----------------------->|
  |                     |                    |  Message            |                |
  |                     |                    |  (in_progress)      |                |
  |<--Response ID-------|<--Return ID--------|<-------------------|                |
  |                     |                    |                    |                |
  |<----SSE Connect-----|<--Upgrade to SSE--|                    |                |
  |                     |                    |--Build Context-----|--------------->|
  |                     |                    |--POST /v1/chat/---->|                |
  |                     |                    |  completions        |                |
  |<--response.delta----|<--Stream-----------|<--LLM Response-----|                |
  |                     |                    |--Store Assistant------------------->|
  |                     |                    |  Message            |                |
  |<--response.done-----|<--Complete---------|--Update Status-------------------->|
  |                     |                    |  (completed)        |                |
```

### Tool Calling Flow

```
Client              Responses API      Chat API         Tool Executor    Database
  |                     |                 |                |               |
  |<--response.delta----|--Stream LLM---->|                |               |
  |                     |<--Tool Call-----|                |               |
  |<--tool.call---------|                 |                |               |
  |                     |--Execute Tool---|--------------->|               |
  |                     |                |                |--Store Call-->|
  |                     |<--Tool Result---|<---------------|<--------------|
  |<--tool.result-------|                |                |--Store Output->|
  |                     |--Send Result--->|                |               |
  |                     |  back to LLM    |                |               |
  |<--response.delta----|<--Continue------|                |               |
  |                     |  with Result    |                |               |
  |<--response.done-----|--Complete---------|--------------|-------------->|
```

### Error Recovery Flow

```
Client              Chat Service         Database            LLM Provider
  |                     |                   |                     |
  |--Request----------->|                   |                     |
  |                     |--Store User------>|                     |
  |                     |   Message         |                     |
  |                     |<------------------|                     |
  |                     |--Stream Request------------------------->|
  |                     |                   |                     |
  |<---Partial Stream---|<----------------------------------------|<---LLM Stream---|
  |                     |                   |                     |
  |                     |                   |      X Error X     |
  |                     |                   |                     |
  |                     |--Store Partial--->|                     |
  |                     |   Response        |                     |
  |                     |<------------------|                     |
  |<---Error Event------|                   |                     |
  |                     |                   |                     |
  |                     |--Mark Error------->|                     |
  |                     |   Status          |                     |
```

### Key Flow Characteristics

1. **Asynchronous Processing**: All database operations are non-blocking
2. **Dual Stream Handling**: Responses stream to client while storing to DB
3. **Error Resilience**: Partial responses are saved on error
4. **Stateful Conversation**: Thread context maintained across requests
5. **Encryption at Every Layer**: Request, storage, and response encryption

## Authentication & Authorization

### Authentication Flow

Uses the existing JWT-based authentication:

1. **JWT Validation**:
   ```rust
   // Applied via user_middleware
   pub async fn chat_completions_handler(
       State(state): State<AppState>,
       Extension(user): Extension<User>, // Injected by middleware
       Extension(session): Extension<Session>,
       Json(request): Json<ChatCompletionRequest>,
   ) -> Result<Response> {
       // User is already authenticated
   }
   ```

2. **Session Encryption**:
   - Uses `encryption_middleware` for E2E encryption
   - Session key derived from JWT claims
   - All requests/responses encrypted
   - Ephemeral

### Authorization Rules

1. **Thread Access**:
   - Users can only access their own threads
   - Enforced at database query level:
   ```rust
   let thread = db.get_thread_by_id_and_user(thread_id, user.id)?;
   ```

## Error Handling

### Error Types

Use existing error types when possible. Here's a few additional ones we might want.

```rust
// Error types for the Responses API
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Tool execution failed: {0}")]
    ToolExecutionError(String),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded { retry_after: u64 },
    
    #[error("Request in progress")]
    RequestInProgress {
        message: String,
        retry_after: Option<u64>,
    },
    
    #[error("Idempotency key reused with different parameters")]
    IdempotencyKeyReused {
        message: String,
    },
    
    #[error("Invalid request: {0}")]
    ValidationError(String),
    
    #[error("Database error")]
    DatabaseError(#[from] DBError),
}
```

### Error Response Format

Following OpenAI's error format with provider code preservation:

```json
{
  "error": {
    "message": "Context window exceeded",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded",
    "details": {
      "current_tokens": 75000,
      "max_tokens": 128000
    }
  }
}
```

### Streaming Error Handling

For SSE streams, errors are sent as special events:

```
data: {"error": {"message": "Tool execution failed", "type": "tool_error", "code": "tool_execution_failed"}}

data: [ERROR]
```

## Database Migrations

### Running Migrations

```bash
# Run migrations
diesel migration run

# Rollback if needed (or about to change the schema while in development)
diesel migration revert

# Generate schema.rs updates
diesel run
```

### Model Definitions

After running migrations, examine the generated `src/schema.rs` file and create matching Diesel models in `src/models/responses.rs`. You'll need these structs:

- `ResponseStatus` enum (queued, in_progress, completed, failed, cancelled)
- `UserSystemPrompt` - custom system prompts
- `ChatThread` - conversation containers
- `UserMessage` - user inputs with Responses API lifecycle tracking
- `ToolCall` - model-requested tool invocations
- `ToolOutput` - tool execution results
- `AssistantMessage` - LLM responses

All `_enc` fields are `Vec<u8>` for BYTEA encrypted content. Use existing encryption patterns from the codebase.

## Security Considerations

### Threat Model

1. **Data Confidentiality**:
   - **Threat**: Unauthorized access to conversation history
   - **Mitigation**: All message content encrypted with user-specific keys
   - **Implementation**: AES-256-GCM for content, no plaintext storage

2. **Cross-User Data Leakage**:
   - **Threat**: User A accessing User B's conversations
   - **Mitigation**: Strict user_id filtering at database level
   - **Implementation**: All queries include user_id constraint

3. **Tool Execution Risks**:
   - **Threat**: Malicious tool calls compromising system
   - **Mitigation**: Sandboxed execution, timeouts, resource limits
   - **Implementation**: Separate tokio tasks with limits

### API Security

1. **Request Validation**:
   - Maximum message size limits
   - Tool permission checks

2. **Response Security**:
   - No sensitive data in errors
   - Sanitized error messages
   - Request IDs for tracing

### Monitoring & Alerting

1. **Security Events**:
   - Failed authentication attempts
   - Tool execution failures

### Compliance Considerations

1. **Data Retention**:
   - User-initiated deletion supported

2. **Privacy**:
   - No logging of message content
   - Encrypted data unreadable outside enclave
   - User isolation enforced

## Risk Mitigation

1. **Performance Risk**: Start with conservative limits, monitor closely
2. **Security Risk**: Security review before deployment
3. **Compatibility Risk**: Test with OpenAI client libraries
4. **Scale Risk**: Design for horizontal scaling from day one

## Future Enhancements

### Caching Strategy

1. **Token Count Cache**:
   ```rust
   // Use cl100k_base encoder for all models (GPT-4, GPT-3.5-turbo, etc.)
   use std::sync::OnceLock;
   use tiktoken::cl100k_base;
   
   static ENCODER: OnceLock<tiktoken::CoreBPE> = OnceLock::new();
   
   pub fn count_tokens(text: &str) -> usize {
       let encoder = ENCODER.get_or_init(|| cl100k_base().unwrap());
       encoder.encode_with_special_tokens(text).len()
   }
   ```
   
   **Note**: We use `cl100k_base` for all models since:
   - All modern OpenAI models (GPT-4, GPT-3.5-turbo) use this encoder
   - It simplifies the implementation (no need for model-specific caching)
   - Token counts are used only for context window management, not billing
   - Claude models would need different counting logic anyway

2. **User Key Cache**:
   ```rust
   // LRU cache for user encryption keys (in-memory only)
   type UserKeyCache = Arc<Mutex<LruCache<Uuid, Arc<UserKey>>>>;
   
   pub async fn get_user_key_cached(
       user_id: Uuid,
       cache: &UserKeyCache,
       db: &dyn DBConnection,
   ) -> Result<Arc<UserKey>> {
       // Check cache first
       if let Some(key) = cache.lock().unwrap().get(&user_id) {
           return Ok(Arc::clone(key));
       }
       
       // Load and cache
       let user = db.get_user_by_id(user_id)?;
       let key = Arc::new(derive_user_key(&user.seed_enc)?);
       cache.lock().unwrap().put(user_id, Arc::clone(&key));
       Ok(key)
   }
   ```

3. **Thread Metadata Cache**:
   - Cache recent thread metadata
   - Invalidate on updates
   - TTL-based expiration

### Database Optimizations

1. **Query Optimization**:
   ```sql
   -- Composite index for common query pattern
   CREATE INDEX idx_messages_thread_created 
   ON chat_messages(thread_id, created_at DESC);
   
   -- Partial index for active threads
   CREATE INDEX idx_threads_active 
   ON chat_threads(user_id, updated_at DESC)
   WHERE updated_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
   ```

2. **Batch Operations**:
   ```rust
   // Batch insert messages
   pub async fn insert_messages_batch(
       conn: &mut AsyncPgConnection,
       messages: Vec<NewChatMessage>,
   ) -> Result<Vec<ChatMessage>> {
       diesel::insert_into(chat_messages::table)
           .values(&messages)
           .get_results(conn)
           .await
   }
   ```

### Streaming Optimizations

1. **Buffer Management**:
   ```rust
   // Tuned buffer sizes
   const SSE_BUFFER_SIZE: usize = 8192;
   const BROADCAST_BUFFER_SIZE: usize = 1024;
   
   // Adaptive buffering based on client speed
   let (tx, rx) = broadcast::channel(BROADCAST_BUFFER_SIZE);
   ```

2. **Compression**:
   - Enable gzip for non-streaming responses
   - Consider SSE compression for supported clients

### Resource Management

1. **Memory Management**:
   - Stream large responses instead of buffering
   - Limit concurrent message processing
   - Clear caches on memory pressure


### Advanced Features

1. **Multi-Modal Support**:
   - Image inputs (should already be supported, this is base64'd in the user message)
   - Audio transcription
   - File attachments with encryption
   - Rich content rendering

2. **Advanced Context Management**:
   - Intelligent message summarization
   - Semantic importance scoring
   - Dynamic context window adjustment
   - Cross-thread memory

3. **Enhanced Tool Ecosystem**:
   ```rust
   // Plugin system for custom tools
   pub trait ToolPlugin: Send + Sync {
       fn metadata(&self) -> ToolMetadata;
       fn validate_permissions(&self, user: &User) -> bool;
       fn execute(&self, args: Value) -> BoxFuture<Result<Value>>;
   }
   
   // Built-in tools
   - web_search: Search the internet
   - code_interpreter: Execute Python code
   - sql_query: Query user's authorized databases
   - email_send: Send emails on behalf of user
   ```

4. **Conversation Intelligence**:
   - Automatic thread titling
   - Topic extraction and tagging
   - Conversation analytics
   - Suggested follow-ups

### Performance Enhancements

1. **Advanced Conversation Reconstruction**:
   
   **Thread Context Caching**:
   ```rust
   pub struct ThreadContextCache {
       // Thread ID -> (messages, last_updated, token_count)
       cache: Arc<RwLock<LruCache<i64, CachedContext>>>,
   }
   
   struct CachedContext {
       messages: Vec<ChatCompletionMessage>,
       last_message_id: i64,
       token_count: usize,
       cached_at: Instant,
   }
   
   impl ThreadContextCache {
       pub async fn get_or_build(
           &self,
           thread_id: i64,
           db: &mut AsyncPgConnection,
       ) -> Result<Vec<ChatCompletionMessage>> {
           // Check cache first
           if let Some(cached) = self.get_cached(thread_id).await {
               // Only fetch new messages since last cache
               let new_messages = self.get_messages_after(
                   thread_id, 
                   cached.last_message_id,
                   db
               ).await?;
               
               if new_messages.is_empty() {
                   return Ok(cached.messages);
               }
               
               // Append and update cache
               let mut updated = cached.messages;
               updated.extend(new_messages);
               self.update_cache(thread_id, updated.clone()).await;
               return Ok(updated);
           }
           
           // Full rebuild if not cached
           let messages = db.get_thread_context(thread_id, user_id).await?;
           self.update_cache(thread_id, messages.clone()).await;
           Ok(messages)
       }
   }
   ```
   
   **Parallel Decryption**:
   ```rust
   use rayon::prelude::*;
   
   let decrypted_messages: Vec<Result<DecryptedMessage, _>> = raw_messages
       .par_iter()
       .map(|msg| decrypt_message_parallel(msg, &user_key))
       .collect();
   ```
   
   **Query Optimization for Token-Based Selection**:
   ```sql
   -- Use stored token counts for efficient message selection
   WITH token_budget AS (
       SELECT $1::INTEGER as max_tokens
   ),
   selected_messages AS (
       SELECT *, 
              SUM(token_count) OVER (ORDER BY created_at DESC) as running_total
       FROM thread_messages
       WHERE thread_id = $2
   )
   SELECT * FROM selected_messages
   WHERE running_total <= (SELECT max_tokens FROM token_budget)
   ORDER BY created_at ASC;
   ```
   
   **Materialized Views for Hot Threads**:
   ```sql
   CREATE MATERIALIZED VIEW thread_context_cache AS
   WITH recent_threads AS (
       SELECT DISTINCT thread_id 
       FROM user_messages 
       WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
   )
   SELECT /* the UNION ALL query from basic implementation */
   FROM recent_threads rt
   WHERE thread_id = rt.thread_id;
   
   -- Refresh periodically
   REFRESH MATERIALIZED VIEW CONCURRENTLY thread_context_cache;
   ```
   
   **Selective Time-Based Loading**:
   ```sql
   -- For long conversations, only load recent messages
   WHERE um.thread_id = $1 
     AND um.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
   ```

2. **Distributed Architecture**:
   - Redis for distributed caching
   - Kafka for event streaming
   - Horizontal scaling with load balancing
   - Geographic distribution

3. **Optimized Storage**:
   - Compressed message storage
   - Tiered storage (hot/cold)
   - Message deduplication
   - Incremental backups

4. **Smart Routing**:
   - Model selection based on query complexity
   - Cost optimization routing
   - Latency-based provider selection
   - Fallback chain configuration

### Integration Capabilities

1. **Webhook System**:
   ```rust
   pub struct WebhookConfig {
       url: String,
       events: Vec<EventType>,
       secret: String,
       retry_policy: RetryPolicy,
   }
   
   // Events: message.created, thread.completed, tool.executed
   ```

2. **Real-time Updates**:
   - WebSocket support for live updates
   - Server-sent events for thread changes
   - Presence indicators
   - Collaborative conversations

3. **Export/Import**:
   - Thread export in multiple formats
   - Bulk data import
   - Conversation templates
   - Knowledge base integration


### Async Job Queue Architecture

For scaling beyond single-server capacity, implement job queue processing:

1. **Queue Infrastructure**:
   ```rust
   // Reuse existing SqsEventPublisher pattern
   let sqs_publisher = SqsEventPublisher::new(
       sqs_client,
       env::var("RESPONSES_QUEUE_URL")?,
       project_id,
   );
   
   // Queue long-running operations
   let job = ResponseJob {
       response_id: response.id,
       job_type: JobType::ToolExecution { tool_call_id },
       payload: encrypted_request,
   };
   
   sqs_publisher.publish_event(Event::ResponseJob(job)).await?;
   ```

2. **Use Cases**:
   - Tools requiring >30 second execution
   - Batch processing operations
   - Rate-limited external API calls
   - Thread summarization
   - Export generation

3. **Worker Architecture**:
   - Poll SQS for jobs
   - Process in background
   - Update status via WebSocket/webhooks
   - Handle retries with exponential backoff
   - Dead letter queue for failures

4. **Benefits**:
   - Non-blocking API responses
   - Horizontal scaling of workers
   - Fault tolerance and retry handling
   - Load distribution across servers

These enhancements maintain backward compatibility while enabling powerful new capabilities for future growth.
