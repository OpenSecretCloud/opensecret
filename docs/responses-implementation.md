# OpenAI Responses API Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Database Schema](#database-schema)
3. [API Endpoints](#api-endpoints)
4. [Message Encryption Strategy](#message-encryption-strategy)
5. [Idempotency](#idempotency)
6. [Request Processing Architecture](#request-processing-architecture)
7. [Token Management Strategy](#token-management-strategy)
8. [Tool Calling Framework](#tool-calling-framework)
9. [Authentication & Authorization](#authentication--authorization)
10. [Error Handling](#error-handling)
11. [Database Migrations](#database-migrations)
12. [Security Considerations](#security-considerations)
13. [Performance Optimizations](#performance-optimizations)
14. [Testing Strategy](#testing-strategy)
15. [Implementation Checklist](#implementation-checklist)

## Overview

This document outlines the implementation of an OpenAI Responses API-compatible endpoint for OpenSecret. The Responses API is different from the Chat Completions API - it provides server-side conversation state management with SSE streaming and a single entry point where the request contains the entire conversation state.

**Key Differences from Chat Completions API**:
- **Chat Completions** (`/v1/chat/completions`): Stateless, requires full conversation history each request
- **Responses** (`/v1/responses`): Server-managed conversation state, status tracking, server-side tools

### Objectives

1. **OpenAI Responses API Compatibility**: Implement `/v1/responses` endpoint that matches OpenAI's Responses API specification
2. **Dual Streaming Architecture**: Stream responses to client while simultaneously storing to database
3. **SSE Streaming**: Server-sent events with proper event types (response.created, response.delta, tool.call, response.done, etc.)
4. **Status Lifecycle**: Track response status through in_progress → completed
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
- [ ] Create and run database migrations
  - [ ] Create response_status enum (without 'queued' state)
  - [ ] Create user_system_prompts table
  - [ ] Create chat_threads table 
  - [ ] Create user_messages table with idempotency columns
  - [ ] Create tool_calls table
  - [ ] Create tool_outputs table
  - [ ] Create assistant_messages table
  - [ ] Add all necessary indexes
- [ ] Generate schema.rs with diesel print-schema
- [ ] Create Diesel model structs
  - [ ] ResponseStatus enum
  - [ ] UserSystemPrompt model
  - [ ] ChatThread model
  - [ ] UserMessage model (with idempotency fields)
  - [ ] ToolCall model
  - [ ] ToolOutput model
  - [ ] AssistantMessage model
- [ ] Add query methods to DBConnection trait
  - [ ] get_thread_by_id_and_user()
  - [ ] create_thread_with_id()
  - [ ] get_user_message_by_previous_id()
  - [ ] get_thread_messages_for_context()
  - [ ] get_user_message_by_idempotency_key()

**Phase 2: Basic Responses Endpoint (No Streaming)**
- [ ] Create ResponsesCreateRequest struct matching OpenAI spec
- [ ] Create ResponsesCreateResponse struct
- [ ] Implement POST /v1/responses handler
  - [ ] JWT validation (reuse existing middleware)
  - [ ] Request validation
  - [ ] Thread creation logic (thread.id = message.id for new threads)
  - [ ] Thread lookup for previous_response_id
  - [ ] Store user message with status='in_progress'
  - [ ] Return immediate response with ID
- [ ] Add route to web server
- [ ] Test basic request/response flow

**Phase 3: Context Building & Chat Integration**
- [ ] Implement conversation context builder
  - [ ] Query all message types from thread
  - [ ] Merge and sort by timestamp
  - [ ] Decrypt message content
  - [ ] Format into ChatCompletionRequest messages array
- [ ] Implement token counting
  - [ ] Integrate tiktoken-rs
  - [ ] Add per-model token limits
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
- [ ] Implement GET /v1/responses (list)
  - [ ] Add pagination support
  - [ ] Query user's responses
  - [ ] Format list response
- [ ] Implement DELETE /v1/responses/{id}
  - [ ] Add optimistic locking
  - [ ] Update status to 'canceled'
  - [ ] Handle cascade deletes
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
- [ ] Implement backpressure for slow clients
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

**Note on model storage**: The `model` field is only stored in `user_messages` since assistant messages are always generated using the model specified in their parent user message. This avoids redundancy and ensures consistency.

### user_system_prompts Table

Stores optional custom system prompts for users.

```sql
-- Table: user_system_prompts (optional custom system prompts)
CREATE TABLE user_system_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    name BYTEA NOT NULL, -- Encrypted system prompt name (binary ciphertext)
    prompt BYTEA NOT NULL, -- Encrypted system prompt (binary ciphertext)
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_system_prompts_user_id ON user_system_prompts(user_id);
CREATE INDEX idx_user_system_prompts_default ON user_system_prompts(user_id, is_default) WHERE is_default = true;
```

### chat_threads Table

Conversation containers that group related messages. For new conversations (where `previous_response_id` is null), the thread ID matches the first message ID.

```sql
-- Table: chat_threads (conversation containers)
CREATE TABLE chat_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    system_prompt_id UUID REFERENCES user_system_prompts(id) ON DELETE SET NULL,
    title BYTEA, -- Encrypted title (binary ciphertext)
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_threads_user_id ON chat_threads(user_id);
CREATE INDEX idx_chat_threads_updated ON chat_threads(user_id, updated_at DESC);
```

### user_messages Table

Stores user inputs and tracks Responses API request lifecycle.

```sql
-- Table: user_messages (user inputs / Responses API requests)
CREATE TYPE response_status AS ENUM 
  ('in_progress', 'completed', 'failed', 'canceled');

CREATE TABLE user_messages (
    id UUID PRIMARY KEY, -- This is the response_id in the API. For new threads (previous_response_id=null), this ID also becomes the thread_id
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content BYTEA NOT NULL, -- Encrypted user input (binary ciphertext)
    
    -- Responses API fields
    status response_status DEFAULT 'in_progress',
    model TEXT NOT NULL,
    previous_message_id UUID REFERENCES user_messages(id) ON DELETE SET NULL,
    temperature REAL,
    top_p REAL,
    max_output_tokens INTEGER,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN DEFAULT false,
    store BOOLEAN DEFAULT true,
    metadata JSONB,
    error TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    
    -- Optimistic locking for cancellation
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Idempotency fields
    idempotency_key TEXT,
    request_hash TEXT,
    idempotency_expires_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_user_messages_thread_id ON user_messages(thread_id);
CREATE INDEX idx_user_messages_user_id ON user_messages(user_id);
CREATE INDEX idx_user_messages_status ON user_messages(status);
-- Composite index for pagination
CREATE INDEX idx_user_messages_thread_created_id 
    ON user_messages(thread_id, created_at DESC, id);
    
-- Unique constraint for idempotency
CREATE UNIQUE INDEX idx_user_messages_idempotency 
    ON user_messages(user_id, idempotency_key) 
    WHERE idempotency_key IS NOT NULL AND idempotency_expires_at > CURRENT_TIMESTAMP;
```

### tool_calls Table

Tracks tool calls requested by the model.

```sql
-- Table: tool_calls (tool invocations by the model)
CREATE TABLE tool_calls (
    id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id UUID NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    tool_call_id TEXT NOT NULL, -- OpenAI's tool_call_id
    name TEXT NOT NULL,
    arguments BYTEA, -- Encrypted arguments (binary ciphertext)
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_calls_thread_id ON tool_calls(thread_id);
CREATE INDEX idx_tool_calls_user_message_id ON tool_calls(user_message_id);
CREATE INDEX idx_tool_calls_thread_created_id 
    ON tool_calls(thread_id, created_at DESC, id);
```

### tool_outputs Table  

Stores results from tool executions.

```sql
-- Table: tool_outputs (tool execution results)
CREATE TABLE tool_outputs (
    id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    tool_call_id UUID NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    output BYTEA NOT NULL, -- Encrypted output (binary ciphertext)
    status TEXT CHECK (status IN ('succeeded', 'failed')),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_outputs_thread_id ON tool_outputs(thread_id);
CREATE INDEX idx_tool_outputs_tool_call_id ON tool_outputs(tool_call_id);
CREATE INDEX idx_tool_outputs_thread_created_id 
    ON tool_outputs(thread_id, created_at DESC, id);
```

### assistant_messages Table

Stores LLM responses (non-tool responses). The model is not stored here since it can be derived from the parent `user_message_id` - this avoids redundancy and potential inconsistencies.

```sql
-- Table: assistant_messages (LLM responses)
CREATE TABLE assistant_messages (
    id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id UUID NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    content BYTEA NOT NULL, -- Encrypted assistant response (binary ciphertext)
    usage JSONB,
    finish_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_assistant_messages_thread_id ON assistant_messages(thread_id);
CREATE INDEX idx_assistant_messages_user_message_id ON assistant_messages(user_message_id);
CREATE INDEX idx_assistant_messages_thread_created_id 
    ON assistant_messages(thread_id, created_at DESC, id);
```

## API Endpoints

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
- Thread titles are auto-generated after the first few messages and stored in `chat_threads.title`

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
  "previous_response_id": null // null for new thread, or "msg_abc123" to continue
}
```

**Response (Immediate):**
```json
{
  "id": "msg_xyz789",
  "object": "response",
  "created": 1677652288,
  "status": "in_progress"
}
```

**SSE Stream Events (if stream=true):**
```
event: response.delta
data: {"content": "Let me check the current time for you."}

event: tool.call
data: {"id": "call_123", "name": "current_time", "arguments": "{}"}

event: tool.result
data: {"tool_call_id": "call_123", "content": "{\"time\": \"2024-01-15T10:30:00Z\", \"timezone\": \"UTC\"}"}

event: response.delta
data: {"content": "The current time is 10:30 AM UTC on January 15, 2024."}

event: response.done
data: {"id": "msg_xyz789", "status": "completed", "usage": {"prompt_tokens": 25, "completion_tokens": 45}}

### GET /v1/responses/{response_id}

Get the status and result of a response (for polling if not using SSE).

**Response:**
```json
{
  "id": "resp_xyz789",
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

**Query Parameters:**
- `limit` (integer, default: 20, max: 100): Number of responses to return
- `after` (string): Cursor for pagination (response ID to start after)
- `before` (string): Cursor for pagination (response ID to start before)
- `order` (string, default: "desc"): Sort order by created_at ("asc" or "desc")

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "resp_xyz789",
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
  "first_id": "resp_xyz789",
  "last_id": "resp_abc123"
}
```

### DELETE /v1/responses/{message_id}

Cancel an in-progress response or delete a completed response.

**Response (Success - 204 No Content):**
```
// Empty response with 204 status
```

**Response (Error - 409 Conflict):**
```json
{
  "error": {
    "message": "Cannot cancel response in completed status",
    "type": "invalid_request_error",
    "code": "response_not_cancelable"
  }
}
```

**Implementation Notes:**
- Uses optimistic locking to prevent race conditions
- Updates status to 'canceled' atomically
- Cascading deletes clean up related tool_calls, tool_outputs, and assistant_messages

## Message Encryption Strategy

All user content is encrypted at rest using the same patterns as the existing KV store implementation.

### Encryption Keys

1. **User Master Key**: Derived from `users.seed_enc` using KDF
2. **Encryption Key**: AES-256 key derived from master key
3. **No per-thread keys**: All messages for a user use the same encryption key

### Encryption Methods

1. **Deterministic Encryption (AES-256-SIV)**
   - Used for: `thread_id_enc`, `message_id_enc`
   - Allows database lookups and indexing
   - Same plaintext always produces same ciphertext

2. **Non-deterministic Encryption (AES-256-GCM)**
   - Used for: `content_enc`, `title_enc`, `arguments_enc`, `output_enc`
   - Random nonce for each encryption
   - Same plaintext produces different ciphertext each time

### Implementation Pattern

```rust
// Encryption pattern for message content
// Encrypting message content
let user_key = get_user_encryption_key(&user.seed_enc)?;
let content_enc = encrypt_aes256_gcm(&user_key, message_content.as_bytes())?;

// Encrypting deterministic fields
let thread_id_enc = encrypt_deterministic(&user_key, thread_id.as_bytes())?;

// Decrypting on retrieval
let content = decrypt_aes256_gcm(&user_key, &content_enc)?;
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

- Thread IDs visible in URLs use deterministic encryption
- All message content uses non-deterministic encryption
- No encryption keys stored in database
- Keys derived on-demand from user seed
- Follows existing `encryption_middleware` patterns

## Idempotency

The Responses API supports idempotent requests to ensure safe retries.

### Implementation

Idempotency is handled directly in the `user_messages` table by adding idempotency columns:

```sql
-- Add idempotency columns to user_messages table
ALTER TABLE user_messages ADD COLUMN idempotency_key TEXT;
ALTER TABLE user_messages ADD COLUMN request_hash TEXT;
ALTER TABLE user_messages ADD COLUMN idempotency_expires_at TIMESTAMPTZ;

-- Unique constraint for idempotency
CREATE UNIQUE INDEX idx_user_messages_idempotency 
    ON user_messages(user_id, idempotency_key) 
    WHERE idempotency_key IS NOT NULL AND idempotency_expires_at > CURRENT_TIMESTAMP;
```

### Request Handling

```rust
// Idempotency key handling with proper concurrency control
if let Some(idempotency_key) = headers.get("idempotency-key") {
    // Canonicalize and hash the request body
    let request_json = serde_json::to_string(&request)?;
    let request_hash = sha256::digest(request_json);
    
    // Check if we've seen this key before for this user
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
                // Return cached response (completed, failed, or canceled)
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
- **Cache everything**: Both successful and error responses are cached and returned
- **Parameter validation**: Different parameters with same key returns 422 error
- **24-hour expiration**: Keys automatically expire and can be reused after 24 hours
- **User-scoped**: Keys are unique per user, not globally

### Response Behavior by Status

| Status | Behavior |
|--------|----------|
| `in_progress` | Return 409 Conflict - request in progress |
| `completed` | Return cached successful response |
| `failed`, `canceled` | Return cached error response |
| Key not found | Process new request |
| Different parameters | Return 422 Unprocessable Entity |

### Cleanup

```sql
-- Expired keys are automatically excluded from uniqueness constraint
-- Optional periodic cleanup to remove old data:
UPDATE user_messages 
SET idempotency_key = NULL, request_hash = NULL, idempotency_expires_at = NULL
WHERE idempotency_expires_at < CURRENT_TIMESTAMP;
```

## Request Processing Architecture

The Responses API processes requests synchronously with dual streaming for real-time responses and persistent storage.

### Request Flow

1. **Initial Request**:
   - Validate JWT and decrypt request
   - Check for idempotency key if provided
   - Generate message ID (or reuse from idempotency check)
   - Determine thread handling:
     - If `previous_response_id` is null: Create new thread with `thread.id = message.id`
     - If `previous_response_id` is provided: Look up thread from previous message
   - Insert user_message record with status='in_progress'
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
data: {"id": "msg_xyz789", "status": "in_progress"}

event: response.delta
data: {"content": "Here's what I know about"}

event: tool.call
data: {"id": "call_123", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}

event: response.done
data: {"id": "msg_xyz789", "status": "completed", "usage": {"prompt_tokens": 25, "completion_tokens": 150}}
```

### Error Handling During Streaming

1. **Upstream Errors**: 
   - Log error and store partial response
   - Send error event to client
   - Mark message with error status

2. **Database Errors**:
   - Continue streaming to client
   - Log error for later retry
   - Use write-ahead log for recovery

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

## Token Management Strategy

Manage conversation context within model-specific token limits using intelligent truncation.

### Model Token Limits

```rust
// Model-specific context window limits
pub const MODEL_LIMITS: &[(&str, usize)] = &[
    ("gpt-4o-128k", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo", 16_385),
    ("claude-3-haiku-200k", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-sonnet", 200_000),
];

pub fn get_model_max_tokens(model: &str) -> usize {
    MODEL_LIMITS.iter()
        .find(|(m, _)| model.starts_with(m))
        .map(|(_, limit)| *limit)
        .unwrap_or(8_192) // Conservative default
}
```

### Token Counting

1. **Libraries**:
   - Use `tiktoken-rs` for accurate OpenAI token counting
   - Cache encoder instances per model
   - Count tokens during message storage

2. **Storage Strategy**:
   - Store token counts in `chat_messages` table
   - Track `prompt_tokens`, `completion_tokens`, `total_tokens`
   - Update counts as messages are added

### Context Window Management

1. **Building Prompts**:
   ```rust
   // Building conversation context from multiple tables
   async fn build_conversation_context(
       thread_id: Uuid,
       model: &str,
   ) -> Result<Vec<Message>> {
       let max_tokens = get_model_max_tokens(model);
       // Reserve 10% for response and potential summarization
       let context_limit = (max_tokens as f64 * 0.9) as usize;
       // Get thread with optional system prompt
       let thread = get_thread_with_system_prompt(thread_id).await?;
       
       // Get all messages from different tables ordered by timestamp
       let user_msgs = get_user_messages(thread_id).await?;
       let assistant_msgs = get_assistant_messages(thread_id).await?;
       let tool_calls = get_tool_calls(thread_id).await?;
       let tool_outputs = get_tool_outputs(thread_id).await?;
       
       // Merge and sort by created_at
       let mut all_messages = merge_messages_by_timestamp(
           user_msgs,
           assistant_msgs,
           tool_calls,
           tool_outputs
       );
       
       // Start with system prompt if exists
       let mut included_messages = Vec::new();
       let mut total_tokens = 0;
       
       if let Some(system_prompt) = thread.system_prompt {
           included_messages.push(Message {
               role: "system",
               content: decrypt_content(&system_prompt)?,
           });
           total_tokens += count_tokens(&system_prompt);
       }
       
       // Add new message tokens
       let new_tokens = count_tokens(new_message);
       total_tokens += new_tokens;
       
       // Build context from newest to oldest
       let mut context_messages = Vec::new();
       for msg in messages.iter().rev().skip(1) {
           if total_tokens + msg.token_count <= max_tokens {
               context_messages.push(msg.clone());
               total_tokens += msg.token_count;
           } else {
               break;
           }
       }
       
       // Reverse to maintain chronological order
       context_messages.reverse();
       included_messages.extend(context_messages);
       
       Ok(included_messages)
   }
   ```

2. **Middle Truncation Strategy**:
   - Keep first message (system prompt)
   - Keep most recent N messages
   - Remove middle messages if over limit
   - Add truncation indicator: `[Previous messages truncated]`

3. **Advanced Truncation** (Future):
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

1. **Pre-flight Checks**:
   - Estimate tokens before API call
   - Warn if approaching limits
   - Suggest thread splitting if needed

2. **Response Limiting**:
   - Calculate available tokens for response
   - Set `max_tokens` accordingly
   - Reserve tokens for tool calls

### Implementation Notes

- Token counts are estimates (model-specific)
- Always leave buffer for response (~4k tokens)
- Consider tool response tokens in calculations
- Cache token counts to avoid recalculation
- Use database indexes on token count columns

## Tool Calling Framework

Extensible framework for function calling with OpenAI-compatible format.

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
```

### Tool Call Flow

1. **Model requests tool call** (detected in SSE stream):
   ```json
   {
     "tool_calls": [{
       "id": "call_abc123",
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
   - Store tool call and output in database
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
   - Rate limit tool calls

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
  |                     |                    |--Insert User------->|                |
  |                     |                    |  Message            |                |
  |                     |                    |  (in_progress)      |                |
  |<--Response ID-------|<--Return ID--------|                    |                |
  |                     |                    |                    |                |
  |<----SSE Connect-----|<--Upgrade to SSE--|                    |                |
  |                     |                    |--Build Context----->|                |
  |                     |                    |--POST /v1/chat/---->|                |
  |                     |                    |  completions        |                |
  |<--response.delta----|<--Stream-----------|<--LLM Response-----|                |
  |                     |                    |--Store Assistant--->|                |
  |                     |                    |  Message            |                |
  |<--response.done-----|<--Complete---------|--Update Status----->|                |
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
  |                     |<--Tool Result---|                |               |
  |<--tool.result-------|                |                |--Store Output->|
  |                     |--Send Result--->|                |               |
  |                     |  back to LLM    |                |               |
  |<--response.delta----|<--Continue------|                |               |
  |                     |  with Result    |                |               |
  |<--response.done-----|--Complete-------|---------------->|               |
```

### Error Recovery Flow

```
Client              Chat Service         Database            LLM Provider
  |                     |                   |                     |
  |--Request----------->|                   |                     |
  |                     |--Store User------>|                     |
  |                     |   Message         |                     |
  |                     |                   |                     |
  |                     |--Stream Request----------------->|
  |                     |                   |                     |
  |<---Partial Stream---|<------------------|<---LLM Stream---|
  |                     |                   |                     |
  |                     |                   |      X Error X     |
  |                     |                   |                     |
  |                     |--Store Partial--->|                     |
  |                     |   Response        |                     |
  |                     |                   |                     |
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

2. **Token Types Supported**:
   - Access tokens (short-lived)
   - Refresh tokens (for token renewal)
   - API keys (for programmatic access)

3. **Session Encryption**:
   - Uses `encryption_middleware` for E2E encryption
   - Session key derived from JWT claims
   - All requests/responses encrypted

### Authorization Rules

1. **Thread Access**:
   - Users can only access their own threads
   - Enforced at database query level:
   ```rust
   let thread = db.get_thread_by_id_and_user(thread_id, user.id)?;
   ```

2. **Message Access**:
   - Messages accessible only through owned threads
   - No direct message access endpoint

3. **Tool Permissions**:
   - Basic tools available to all users
   - Premium tools require subscription check
   - Custom tools require explicit permissions

### Rate Limiting

1. **Request Limits**:
   - Implement per-user rate limiting
   - Track in Redis or database
   - Return 429 on limit exceeded

2. **Token Usage Limits**:
   - Track cumulative token usage
   - Enforce monthly/daily limits
   - Handle overflow with rate limiting

### API Key Management

```rust
// API key validation for programmatic access
// API key authentication for programmatic access
pub async fn validate_api_key(
    key: &str,
    db: &dyn DBConnection,
) -> Result<User> {
    let key_hash = hash_api_key(key);
    let api_key = db.get_api_key_by_hash(&key_hash)?;
    
    if api_key.expires_at < Utc::now() {
        return Err(ApiError::ExpiredApiKey);
    }
    
    Ok(api_key.user)
}
```

### Security Headers

Apply standard security headers:
- `X-Request-ID`: For request tracing
- `X-RateLimit-*`: Rate limit information
- `Strict-Transport-Security`: HTTPS enforcement
- CSP headers for web interfaces

## Error Handling

### Error Types

```rust
// Error types for the Responses API
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Thread not found")]
    ThreadNotFound,
    
    #[error("Invalid thread access")]
    UnauthorizedThreadAccess,
    
    #[error("Context window exceeded")]
    ContextLimitExceeded { current: usize, max: usize },
    
    #[error("Tool execution failed: {0}")]
    ToolExecutionError(String),
    
    #[error("Upstream provider error")]
    UpstreamError {
        message: String,
        provider: String,
        code: Option<String>,
        retry_after: Option<u64>,
    },
    
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

#### Provider Error Mapping

```rust
// Map upstream provider errors to API errors
impl From<ProviderError> for ChatError {
    fn from(err: ProviderError) -> Self {
        match err {
            ProviderError::RateLimit { retry_after, .. } => {
                ChatError::RateLimitExceeded { retry_after }
            }
            ProviderError::ContextLength { current, max } => {
                ChatError::ContextLimitExceeded { current, max }
            }
            ProviderError::ApiError { code, message, provider } => {
                ChatError::UpstreamError {
                    message,
                    provider,
                    code: Some(code),
                    retry_after: None,
                }
            }
            _ => ChatError::UpstreamError {
                message: err.to_string(),
                provider: "unknown".to_string(),
                code: None,
                retry_after: None,
            }
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

### Error Recovery Strategies

1. **Partial Response Storage**:
   ```rust
   match process_stream().await {
       Ok(response) => store_complete_response(response),
       Err(e) => {
           // Store what we have
           store_partial_response(accumulated_content, e.to_string()).await?;
           // Return error to client
           return Err(e);
       }
   }
   ```

2. **Retry Logic**:
   - Automatic retry for transient errors
   - Exponential backoff for rate limits
   - Circuit breaker for persistent failures

3. **Fallback Providers**:
   - Primary/secondary LLM providers
   - Graceful degradation on failure

### Error Logging

```rust
// Structured error logging
error!(
    request_id = %request_id,
    user_id = %user.id,
    error_type = "upstream_error",
    provider = "primary",
    "Failed to get completion from upstream provider: {}", e
);
```

### Client Error Handling

```typescript
// Client-side error handling example
const response = await fetch('/v1/chat/completions', { ... });

if (!response.ok) {
    const error = await response.json();
    switch (error.error.code) {
        case 'context_length_exceeded':
            // Suggest starting new thread
            break;
        case 'rate_limit_exceeded':
            // Wait and retry
            setTimeout(retry, error.error.details.retry_after * 1000);
            break;
        default:
            // Generic error handling
    }
}
```

## Database Migrations

### Migration Files

Create the following Diesel migrations:

**Migration: `2024_01_01_000001_create_responses_tables/up.sql`**

```sql
-- Create user_system_prompts table
CREATE TABLE user_system_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prompt BYTEA NOT NULL,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_system_prompts_user_id ON user_system_prompts(user_id);
CREATE INDEX idx_user_system_prompts_default ON user_system_prompts(user_id, is_default) WHERE is_default = true;

-- Create chat_threads table
CREATE TABLE chat_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    system_prompt_id UUID REFERENCES user_system_prompts(id) ON DELETE SET NULL,
    title BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_threads_user_id ON chat_threads(user_id);
CREATE INDEX idx_chat_threads_updated ON chat_threads(user_id, updated_at DESC);

-- Create response status enum
CREATE TYPE response_status AS ENUM 
  ('in_progress', 'completed', 'failed', 'canceled');

-- Create user_messages table
CREATE TABLE user_messages (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(uuid) ON DELETE CASCADE,
    content BYTEA NOT NULL,
    status response_status DEFAULT 'in_progress',
    model TEXT NOT NULL,
    previous_message_id UUID REFERENCES user_messages(id) ON DELETE SET NULL,
    temperature REAL,
    top_p REAL,
    max_output_tokens INTEGER,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    store BOOLEAN,
    metadata JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_user_messages_thread_id ON user_messages(thread_id);
CREATE INDEX idx_user_messages_user_id ON user_messages(user_id);
CREATE INDEX idx_user_messages_status ON user_messages(status);
CREATE INDEX idx_user_messages_created ON user_messages(thread_id, created_at);

-- Create tool_calls table
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id UUID NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    tool_call_id TEXT NOT NULL,
    name TEXT NOT NULL,
    arguments BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_calls_thread_id ON tool_calls(thread_id);
CREATE INDEX idx_tool_calls_user_message_id ON tool_calls(user_message_id);
CREATE INDEX idx_tool_calls_created ON tool_calls(thread_id, created_at);

-- Create tool_outputs table
CREATE TABLE tool_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    tool_call_id UUID NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    output BYTEA NOT NULL,
    status TEXT CHECK (status IN ('succeeded', 'failed')),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_outputs_thread_id ON tool_outputs(thread_id);
CREATE INDEX idx_tool_outputs_tool_call_id ON tool_outputs(tool_call_id);
CREATE INDEX idx_tool_outputs_created ON tool_outputs(thread_id, created_at);

-- Create assistant_messages table
CREATE TABLE assistant_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id UUID NOT NULL REFERENCES user_messages(id) ON DELETE CASCADE,
    content BYTEA NOT NULL,
    usage JSONB,
    finish_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_assistant_messages_thread_id ON assistant_messages(thread_id);
CREATE INDEX idx_assistant_messages_user_message_id ON assistant_messages(user_message_id);
CREATE INDEX idx_assistant_messages_created ON assistant_messages(thread_id, created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_user_system_prompts_updated_at BEFORE UPDATE
    ON user_system_prompts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_chat_threads_updated_at BEFORE UPDATE
    ON chat_threads FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**Migration: `2024_01_01_000001_create_responses_tables/down.sql`**

```sql
DROP TRIGGER IF EXISTS update_chat_threads_updated_at ON chat_threads;
DROP TRIGGER IF EXISTS update_user_system_prompts_updated_at ON user_system_prompts;
DROP FUNCTION IF EXISTS update_updated_at_column();

DROP TABLE IF EXISTS assistant_messages;
DROP TABLE IF EXISTS tool_outputs;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS user_messages;
DROP TABLE IF EXISTS chat_threads;
DROP TABLE IF EXISTS user_system_prompts;
DROP TYPE IF EXISTS response_status;
```

### Running Migrations

```bash
# Run migrations
diesel migration run

# Rollback if needed
diesel migration revert

# Generate schema.rs updates
diesel print-schema > src/schema.rs
```

### Model Definitions

Update `src/models/` with new model files:

```rust
// src/models/responses.rs
use crate::schema::{user_system_prompts, chat_threads, user_messages, tool_calls, tool_outputs, assistant_messages};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

// User System Prompts
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = user_system_prompts)]
pub struct UserSystemPrompt {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub prompt: Vec<u8>, // BYTEA
    pub is_default: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Chat Threads
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = chat_threads)]
pub struct ChatThread {
    pub id: Uuid,
    pub user_id: Uuid,
    pub system_prompt_id: Option<Uuid>,
    pub title: Option<Vec<u8>>, // BYTEA
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// User Messages (Responses API requests)
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = user_messages)]
pub struct UserMessage {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub user_id: Uuid,
    pub content: Vec<u8>, // BYTEA
    pub status: ResponseStatus,
    pub model: String,
    pub previous_message_id: Option<Uuid>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<i32>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    pub store: Option<bool>,
    pub metadata: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

// Tool Calls
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = tool_calls)]
pub struct ToolCall {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub user_message_id: Uuid,
    pub tool_call_id: String,
    pub name: String,
    pub arguments: Option<Vec<u8>>, // BYTEA
    pub created_at: DateTime<Utc>,
}

// Assistant Messages
#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = assistant_messages)]
pub struct AssistantMessage {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub user_message_id: Uuid,
    pub content: Vec<u8>, // BYTEA
    pub usage: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
    pub created_at: DateTime<Utc>,
}
```

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

3. **Prompt Injection**:
   - **Threat**: Malicious prompts affecting system behavior
   - **Mitigation**: Input validation and sandboxed tool execution
   - **Implementation**: JSON schema validation for tools

4. **Tool Execution Risks**:
   - **Threat**: Malicious tool calls compromising system
   - **Mitigation**: Sandboxed execution, timeouts, resource limits
   - **Implementation**: Separate tokio tasks with limits

### Security Controls

1. **Encryption at Rest**:
   ```rust
   // All sensitive fields encrypted before storage
   let content_enc = encrypt_aes256_gcm(&user_key, content.as_bytes())?;
   let args_enc = encrypt_aes256_gcm(&user_key, args.as_bytes())?;
   ```

2. **Access Control**:
   ```rust
   // Enforce user isolation in all queries
   pub async fn get_thread_messages(
       conn: &mut AsyncPgConnection,
       thread_id: Uuid,
       user_id: Uuid,
   ) -> Result<Vec<ConversationMessage>> {
       // Verify thread ownership first
       let thread = chat_threads::table
           .filter(chat_threads::id.eq(thread_id))
           .filter(chat_threads::user_id.eq(user_id))
           .first::<ChatThread>(conn)
           .await?;
       
       // Get all message types and merge by timestamp
       let user_msgs = user_messages::table
           .filter(user_messages::thread_id.eq(thread.id))
           .load::<UserMessage>(conn).await?;
           
       let assistant_msgs = assistant_messages::table
           .filter(assistant_messages::thread_id.eq(thread.id))
           .load::<AssistantMessage>(conn).await?;
           
       let tool_calls = tool_calls::table
           .filter(tool_calls::thread_id.eq(thread.id))
           .load::<ToolCall>(conn).await?;
           
       let tool_outputs = tool_outputs::table
           .filter(tool_outputs::thread_id.eq(thread.id))
           .load::<ToolOutput>(conn).await?;
           
       // Merge and sort by created_at
       merge_conversation_messages(
           user_msgs,
           assistant_msgs,
           tool_calls,
           tool_outputs
       )
   }
   ```

3. **Input Validation**:
   - Validate all API inputs against schemas
   - Sanitize tool arguments
   - Enforce token limits
   - Rate limiting per user

4. **Audit Logging**:
   ```rust
   info!(
       user_id = %user.id,
       thread_id = %thread_id,
       message_role = %role,
       tool_calls = ?tool_calls,
       "Processing chat message"
   );
   ```

### Enclave Security

1. **Attestation**:
   - Verify enclave attestation before serving requests
   - Include PCRs in responses for client verification

2. **Key Management**:
   - User keys never leave enclave
   - Keys derived from encrypted seeds
   - No key material in logs or errors

3. **Side Channel Protection**:
   - Constant-time encryption operations
   - No timing-based information leaks

### API Security

1. **Rate Limiting**:
   - Per-user request limits
   - Token usage quotas
   - Backpressure on overload

2. **Request Validation**:
   - Maximum message size limits
   - Allowed model whitelist
   - Tool permission checks

3. **Response Security**:
   - No sensitive data in errors
   - Sanitized error messages
   - Request IDs for tracing

### Monitoring & Alerting

1. **Security Events**:
   - Failed authentication attempts
   - Unusual token usage patterns
   - Tool execution failures
   - Rate limit violations

2. **Metrics**:
   - Request latency by endpoint
   - Error rates by type
   - Token usage by user
   - Tool execution counts

### Compliance Considerations

1. **Data Retention**:
   - No automatic deletion (per requirements)
   - User-initiated deletion supported
   - Audit trail preservation

2. **Privacy**:
   - No logging of message content
   - Encrypted data unreadable outside enclave
   - User isolation enforced

3. **Regulatory**:
   - GDPR-compliant data handling
   - Right to deletion support
   - Data portability via export

## Performance Optimizations

### Caching Strategy

1. **Token Count Cache**:
   ```rust
   // Cache tiktoken encoders per model
   lazy_static! {
       static ref ENCODERS: RwLock<HashMap<String, tiktoken::Encoding>> = 
           RwLock::new(HashMap::new());
   }
   
   pub fn count_tokens(model: &str, text: &str) -> Result<usize> {
       let encoders = ENCODERS.read().unwrap();
       if let Some(enc) = encoders.get(model) {
           return Ok(enc.encode(text).len());
       }
       drop(encoders);
       
       // Load encoder if not cached
       let enc = tiktoken::encoding_for_model(model)?;
       let count = enc.encode(text).len();
       ENCODERS.write().unwrap().insert(model.to_string(), enc);
       Ok(count)
   }
   ```

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

3. **Connection Pooling**:
   - Increase pool size for chat endpoints
   - Separate read/write pools
   - Connection recycling

### Streaming Optimizations

1. **Buffer Management**:
   ```rust
   // Tuned buffer sizes
   const SSE_BUFFER_SIZE: usize = 8192;
   const BROADCAST_BUFFER_SIZE: usize = 1024;
   
   // Adaptive buffering based on client speed
   let (tx, rx) = broadcast::channel(
       if slow_client { 256 } else { 1024 }
   );
   ```

2. **Compression**:
   - Enable gzip for non-streaming responses
   - Consider SSE compression for supported clients

### Resource Management

1. **Thread Pools**:
   ```rust
   // Dedicated thread pool for CPU-intensive tasks
   lazy_static! {
       static ref CRYPTO_POOL: ThreadPool = 
           ThreadPoolBuilder::new()
               .num_threads(4)
               .thread_name(|i| format!("crypto-{}", i))
               .build()
               .unwrap();
   }
   ```

2. **Memory Management**:
   - Stream large responses instead of buffering
   - Limit concurrent message processing
   - Clear caches on memory pressure

### Monitoring & Metrics

1. **Performance Metrics**:
   ```rust
   // Track key metrics
   histogram!("chat.request.duration", request_duration);
   counter!("chat.tokens.used", token_count);
   gauge!("chat.active_streams", active_streams);
   ```

2. **Slow Query Detection**:
   - Log queries >100ms
   - Alert on degradation
   - Automatic index suggestions

### Load Testing Targets

- 1000 concurrent SSE streams
- 10k requests/second for non-streaming
- <100ms p99 latency for cached responses
- <500ms p99 for new completions

## Testing Strategy

### Unit Tests

1. **Encryption/Decryption Tests**:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_message_encryption() {
           let user_key = generate_test_key();
           let content = "Hello, world!";
           
           let encrypted = encrypt_aes256_gcm(&user_key, content.as_bytes()).unwrap();
           let decrypted = decrypt_aes256_gcm(&user_key, &encrypted).unwrap();
           
           assert_eq!(content, String::from_utf8(decrypted).unwrap());
       }
       
       #[test]
       fn test_deterministic_encryption() {
           let user_key = generate_test_key();
           let thread_id = "thread-123";
           
           let enc1 = encrypt_deterministic(&user_key, thread_id.as_bytes()).unwrap();
           let enc2 = encrypt_deterministic(&user_key, thread_id.as_bytes()).unwrap();
           
           assert_eq!(enc1, enc2); // Same input produces same output
       }
   }
   ```

2. **Token Counting Tests**:
   ```rust
   #[test]
   fn test_token_truncation() {
       let messages = vec![
           Message { role: "system", content: "You are helpful", tokens: 4 },
           Message { role: "user", content: "Hello", tokens: 2 },
           // ... many more messages
       ];
       
       let truncated = truncate_messages(messages, 70_000);
       let total_tokens: usize = truncated.iter().map(|m| m.tokens).sum();
       
       assert!(total_tokens <= 70_000);
       assert!(truncated[0].role == "system"); // System message preserved
   }
   ```

### Integration Tests

1. **API Endpoint Tests**:
   ```rust
   #[tokio::test]
   async fn test_chat_completion_streaming() {
       let app = create_test_app().await;
       let user = create_test_user(&app.db).await;
       let token = generate_test_jwt(&user);
       
       let response = app.client
           .post("/v1/chat/completions")
           .header("Authorization", format!("Bearer {}", token))
           .json(&json!({
               "model": "gpt-4",
               "messages": [{"role": "user", "content": "Hello"}],
               "stream": true
           }))
           .send()
           .await
           .unwrap();
       
       assert_eq!(response.status(), 200);
       
       // Verify SSE stream
       let mut stream = response.bytes_stream();
       let first_chunk = stream.next().await.unwrap().unwrap();
       assert!(first_chunk.starts_with(b"data: "));
   }
   ```

2. **Database Tests**:
   ```rust
   #[tokio::test]
   async fn test_message_persistence() {
       let mut conn = establish_test_connection().await;
       let user_id = create_test_user(&mut conn).await;
       
       // Create thread
       let thread = create_chat_thread(&mut conn, user_id, "gpt-4").await.unwrap();
       
       // Add messages
       let msg = create_chat_message(&mut conn, thread.id, "user", "Hello").await.unwrap();
       
       // Verify retrieval
       let messages = get_thread_messages(&mut conn, thread.id, user_id).await.unwrap();
       assert_eq!(messages.len(), 1);
       assert_eq!(messages[0].role, "user");
   }
   ```

### End-to-End Tests

```rust
#[tokio::test]
async fn test_full_conversation_flow() {
    let app = create_test_app().await;
    let user = create_test_user(&app.db).await;
    
    // Start conversation
    let thread_id = start_conversation(&app, &user, "Tell me a joke").await;
    
    // Verify response stored
    let messages = get_messages(&app, &user, &thread_id).await;
    assert_eq!(messages.len(), 2); // User + Assistant
    
    // Continue conversation
    let response = send_message(&app, &user, &thread_id, "Another one").await;
    
    // Verify context maintained
    let messages = get_messages(&app, &user, &thread_id).await;
    assert_eq!(messages.len(), 4); // Previous + new pair
}
```

### Security Tests

```rust
#[tokio::test]
async fn test_cross_user_isolation() {
    let app = create_test_app().await;
    let user1 = create_test_user(&app.db).await;
    let user2 = create_test_user(&app.db).await;
    
    // User1 creates thread
    let thread_id = start_conversation(&app, &user1, "Secret info").await;
    
    // User2 tries to access
    let result = get_messages(&app, &user2, &thread_id).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().status(), 404);
}
```

### Performance Tests

```rust
#[tokio::test]
async fn test_concurrent_streams() {
    let app = create_test_app().await;
    let users: Vec<_> = (0..100)
        .map(|_| create_test_user(&app.db))
        .collect().await;
    
    // Start 100 concurrent streams
    let handles: Vec<_> = users.iter()
        .map(|user| {
            tokio::spawn(stream_conversation(app.clone(), user.clone()))
        })
        .collect();
    
    // Verify all complete successfully
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}
```

### Test Utilities

```rust
// Test helpers
pub fn create_test_app() -> TestApp { ... }
pub fn generate_test_jwt(user: &User) -> String { ... }
pub fn create_test_user(db: &DBConnection) -> User { ... }

// Mock LLM provider
pub struct MockLLMProvider {
    responses: Vec<String>,
}

impl LLMProvider for MockLLMProvider {
    async fn complete(&self, _request: Request) -> Result<Response> {
        Ok(Response {
            content: self.responses.pop().unwrap_or("Mock response".into()),
            ..Default::default()
        })
    }
}
```

## Implementation Checklist

### Phase 1: Database & Models (Week 1)

- [ ] Run database migrations
  - [ ] Create user_system_prompts table
  - [ ] Create chat_threads table
  - [ ] Create user_messages table
  - [ ] Create tool_calls table
  - [ ] Create tool_outputs table
  - [ ] Create assistant_messages table
  - [ ] Add necessary indexes

- [ ] Implement Diesel models
  - [ ] UserSystemPrompt model and queries
  - [ ] ChatThread model and queries
  - [ ] UserMessage model and queries
  - [ ] ToolCall model and queries
  - [ ] ToolOutput model and queries
  - [ ] AssistantMessage model and queries
  - [ ] Add to DBConnection trait

- [ ] Encryption utilities
  - [ ] Message encryption/decryption functions
  - [ ] Deterministic encryption for IDs
  - [ ] User key caching layer

### Phase 2: Core API (Week 2)

- [ ] Responses API endpoint
  - [ ] Request validation and parsing
  - [ ] Thread creation/retrieval logic
  - [ ] User message storage with status tracking
  - [ ] Token counting integration
  - [ ] Idempotency key handling

- [ ] Streaming implementation
  - [ ] Dual stream architecture
  - [ ] SSE formatting
  - [ ] Concurrent DB storage
  - [ ] Error handling in streams

- [ ] Context management
  - [ ] Message history retrieval
  - [ ] Token truncation logic
  - [ ] Context window validation

### Phase 3: Tool Framework (Week 3)

- [ ] Tool execution engine
  - [ ] ToolExecutor trait
  - [ ] ToolRegistry implementation
  - [ ] Current_time tool example
  - [ ] Tool result storage

- [ ] Tool calling flow
  - [ ] Parse tool calls from stream
  - [ ] Execute tools asynchronously
  - [ ] Store tool messages
  - [ ] Continue conversation with results

### Phase 4: Additional Endpoints (Week 4)

- [ ] Thread management
  - [ ] GET /v1/threads endpoint
  - [ ] GET /v1/threads/{id}/messages
  - [ ] Thread metadata updates

- [ ] Authentication integration
  - [ ] JWT validation middleware
  - [ ] User context injection
  - [ ] Rate limiting implementation

### Phase 5: Testing & Optimization (Week 5)

- [ ] Test implementation
  - [ ] Unit tests for encryption
  - [ ] Integration tests for API
  - [ ] Security isolation tests
  - [ ] Performance benchmarks

- [ ] Performance tuning
  - [ ] Implement caching layers
  - [ ] Database query optimization
  - [ ] Connection pool tuning
  - [ ] Metrics integration

### Phase 6: Documentation & Deployment (Week 6)

- [ ] API documentation
  - [ ] OpenAPI specification
  - [ ] Integration examples
  - [ ] Migration guide

- [ ] Deployment preparation
  - [ ] Environment configuration
  - [ ] Monitoring setup
  - [ ] Load testing
  - [ ] Rollback plan

## Success Criteria

### Functional Requirements

- [x] OpenAI-compatible API format
- [x] Persistent conversation storage
- [x] E2E encryption for all user content
- [x] Streaming with simultaneous storage
- [x] 70k token context window
- [x] Basic tool calling support

### Non-Functional Requirements

- [x] <500ms p99 latency for completions
- [x] 1000 concurrent SSE streams
- [x] Zero plaintext message storage
- [x] Complete audit trail
- [x] Graceful error handling

### Security Requirements

- [x] User isolation enforced
- [x] Encrypted data at rest
- [x] No key material in logs
- [x] Rate limiting per user
- [x] Input validation on all endpoints

## Risk Mitigation

1. **Performance Risk**: Start with conservative limits, monitor closely
2. **Security Risk**: Security review before deployment
3. **Compatibility Risk**: Test with OpenAI client libraries
4. **Scale Risk**: Design for horizontal scaling from day one

## Future Enhancements

### Advanced Features

1. **Multi-Modal Support**:
   - Image inputs/outputs
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

1. **Distributed Architecture**:
   - Redis for distributed caching
   - Kafka for event streaming
   - Horizontal scaling with load balancing
   - Geographic distribution

2. **Optimized Storage**:
   - Compressed message storage
   - Tiered storage (hot/cold)
   - Message deduplication
   - Incremental backups

3. **Smart Routing**:
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

### Security Enhancements

1. **Advanced Encryption**:
   - Hardware security module integration
   - Post-quantum cryptography ready
   - Homomorphic encryption for analytics
   - Zero-knowledge proofs

2. **Compliance Features**:
   - Data residency controls
   - Automated PII detection
   - Consent management
   - Audit log encryption

### Developer Experience

1. **SDK Development**:
   ```typescript
   // TypeScript SDK example
   const client = new OpenSecretChat({
       apiKey: process.env.API_KEY,
       encryptionKey: process.env.USER_KEY
   });
   
   const thread = await client.threads.create({
       model: 'gpt-4',
       tools: ['web_search', 'code_interpreter']
   });
   
   const response = await thread.send('Analyze this data...');
   ```

2. **Admin Dashboard**:
   - Usage analytics
   - Cost tracking
   - User management
   - Tool configuration

3. **Development Tools**:
   - Conversation replay
   - Debug mode with decrypted logs
   - Performance profiler
   - A/B testing framework

### Extensibility Points

1. **Provider Plugins**:
   ```rust
   pub trait LLMProvider: Send + Sync {
       async fn complete(&self, request: ChatRequest) -> Result<ChatResponse>;
       fn supported_models(&self) -> Vec<String>;
       fn estimate_cost(&self, tokens: usize) -> f64;
   }
   ```

2. **Storage Backends**:
   - S3 for message archives
   - Elasticsearch for search
   - TimescaleDB for analytics
   - Custom encryption providers

3. **Middleware System**:
   - Request/response transformation
   - Custom authentication methods
   - Content filtering
   - Usage tracking

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

## Code Examples & Integration Patterns

### Complete Handler Implementation

```rust
// Main handler for /v1/responses endpoint
// src/web/chat.rs
use axum::{
    extract::{State, Path},
    response::{Response, IntoResponse, sse::Event},
    Json, Extension,
};
use futures::{Stream, StreamExt};
use serde_json::json;

pub async fn responses_handler(
    State(state): State<AppState>,
    Extension(user): Extension<User>,
    Extension(session): Extension<Session>,
    Json(request): Json<ResponsesCreateRequest>,
) -> Result<Response> {
    // 1. Validate request
    validate_responses_request(&request)?;
    
    // 2. Generate message ID
    let message_id = Uuid::new_v4();
    
    // 3. Determine thread handling
    let thread = match request.previous_response_id {
        None => {
            // New thread: thread.id = message.id
            create_new_thread(&state.db, &user, message_id).await?
        }
        Some(prev_id) => {
            // Continue existing thread
            let prev_msg = state.db.get_user_message_by_id(prev_id, user.id).await?;
            state.db.get_thread_by_id_and_user(prev_msg.thread_id, user.id).await?
        }
    };
    
    // 4. Store user message
    let user_message = store_user_message(
        &state.db,
        &user,
        thread.id,
        message_id,
        &request,
    ).await?;
    
    // 4. Build context with history
    let context = build_context_with_truncation(
        &state.db,
        thread.id,
        user.id,
        70_000,
    ).await?;
    
    // 5. Stream or return response
    if request.stream.unwrap_or(false) {
        Ok(stream_response(state, user, thread, context).await?)
    } else {
        Ok(Json(complete_response(state, user, thread, context).await?).into_response())
    }
}

async fn stream_response(
    state: AppState,
    user: User,
    user_message: UserMessage,
    context: Vec<Message>,
) -> Result<Response> {
    let (tx, rx) = tokio::sync::broadcast::channel(1024);
    let rx2 = tx.subscribe();
    
    // Start heartbeat task
    let tx_heartbeat = tx.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            // Send comment frame to keep connection alive
            if tx_heartbeat.send(StreamChunk::Heartbeat).is_err() {
                break; // Client disconnected
            }
        }
    });
    
    // Start upstream request with backpressure handling
    let tx_upstream = tx.clone();
    let upstream_task = tokio::spawn(async move {
        let stream = call_llm_provider(&state.llm, context).await?;
        
        tokio::pin!(stream);
        while let Some(chunk) = stream.next().await {
            // Check if anyone is still listening
            if tx_upstream.receiver_count() == 0 {
                info!("Client disconnected, stopping stream");
                break;
            }
            
            // Send with backpressure handling
            match tx_upstream.send(chunk) {
                Ok(_) => {},
                Err(_) => {
                    info!("All receivers dropped, stopping stream");
                    break;
                }
            }
        }
        Ok::<(), Error>(())
    });
    
    // Start database storage task
    let db_task = tokio::spawn(async move {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        
        let mut rx = rx2;
        while let Ok(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::Content(text) => content.push_str(&text),
                StreamChunk::ToolCall(call) => tool_calls.push(call),
                StreamChunk::Done(metadata) => {
                    store_assistant_message(
                        &state.db,
                        &user,
                        thread.id,
                        &content,
                        tool_calls,
                        metadata,
                    ).await?;
                    break;
                }
            }
        }
        Ok::<(), Error>(())
    });
    
    // Create SSE stream for client with heartbeat handling
    let stream = BroadcastStream::new(rx)
        .map(|chunk| {
            match chunk {
                Ok(StreamChunk::Heartbeat) => {
                    // Send SSE comment to keep connection alive
                    Ok(Event::default().comment("heartbeat"))
                },
                Ok(other_chunk) => {
                    let event = format_sse_chunk(other_chunk)?;
                    Ok(Event::default().data(encrypt_response(&session.key, &event)?))
                },
                Err(e) => {
                    error!("Stream error: {}", e);
                    Err(e)
                }
            }
        });
    
    Ok(Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(stream))?)
}
```

### Client Integration Examples

**Python Client**:
```python
import opensecret
from opensecret.encryption import EncryptionMiddleware

# Initialize client with encryption
client = opensecret.Client(
    api_key="your-api-key",
    base_url="https://api.opensecret.com",
    middleware=[EncryptionMiddleware(session_key)]
)

# Start a conversation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    stream=True,
    tools=[{
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Get current time"
        }
    }]
)

# Handle streaming response
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

**TypeScript Client**:
```typescript
import { OpenSecretClient } from '@opensecret/sdk';

const client = new OpenSecretClient({
    apiKey: process.env.OPENSECRET_API_KEY,
    encryption: {
        sessionKey: await deriveSessionKey()
    }
});

// Continue existing conversation
async function continueChat(threadId: string, message: string) {
    const response = await client.chat.completions.create({
        model: 'gpt-4',
        threadId: threadId,
        messages: [
            { role: 'user', content: message }
        ],
        stream: true
    });
    
    // Process stream with automatic decryption
    for await (const chunk of response) {
        if (chunk.choices[0]?.delta?.content) {
            process.stdout.write(chunk.choices[0].delta.content);
        }
        
        if (chunk.choices[0]?.delta?.tool_calls) {
            await handleToolCalls(chunk.choices[0].delta.tool_calls);
        }
    }
}
```

### Database Query Patterns

```rust
// Efficient message retrieval with decryption
// Efficient message retrieval with decryption
impl DBConnection for AsyncPgConnection {
    async fn get_thread_messages_decrypted(
        &mut self,
        thread_id: Uuid,
        user_id: Uuid,
        user_key: &[u8],
    ) -> Result<Vec<DecryptedMessage>> {
        // Single query with user verification
        // Get all message types for the thread
        let user_msgs = user_messages::table
            .inner_join(chat_threads::table.on(user_messages::thread_id.eq(chat_threads::id)))
            .filter(chat_threads::user_id.eq(user_id))
            .filter(user_messages::thread_id.eq(thread_id))
            .select(user_messages::all_columns)
            .order_by(user_messages::created_at.asc())
            .load::<UserMessage>(self)
            .await?;
            
        let assistant_msgs = assistant_messages::table
            .filter(assistant_messages::thread_id.eq(thread_id))
            .order_by(assistant_messages::created_at.asc())
            .load::<AssistantMessage>(self)
            .await?;
            
        // Decrypt and merge all messages
        let messages = merge_and_decrypt_messages(
            user_msgs,
            assistant_msgs,
            user_key
        )?;
        
        // Decrypt in parallel
        let decrypted: Vec<DecryptedMessage> = messages
            .into_par_iter()
            .map(|msg| decrypt_message(msg, user_key))
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(decrypted)
    }
}
```

### Tool Integration Pattern

```rust
// Registering and using custom tools
// Registering and using custom tools
let mut tool_registry = ToolRegistry::new();

// Register built-in tool
tool_registry.register(
    "current_time",
    Box::new(CurrentTimeExecutor)
);

// Register custom user tool
tool_registry.register(
    "database_query",
    Box::new(DatabaseQueryExecutor::new(user_permissions))
);

// Execute tool during streaming
match parse_tool_call(&chunk) {
    Some(tool_call) => {
        let result = tool_registry
            .execute(&tool_call.name, tool_call.arguments, &user_context)
            .await?;
        
        // Store execution record
        store_tool_execution(&db, message_id, &tool_call, &result).await?;
        
        // Continue conversation with result
        append_tool_response(&mut context, tool_call.id, result);
    }
    None => continue,
}
```

These examples demonstrate the key integration patterns for building a robust, encrypted chat API that maintains compatibility with OpenAI's format while adding powerful features like persistent storage and tool execution.
