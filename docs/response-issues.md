# Response Failure Handling Fix

**Date**: 2025-10-03
**Issue**: Assistant messages left in `in_progress` state when streaming fails

## Problem

When the streaming pipeline setup fails (e.g., provider connection refused), the database records created in Phase 3 are left in an inconsistent state:
- `responses.status` = `'in_progress'`
- `assistant_messages.status` = `'in_progress'`
- `assistant_messages.content_enc` = `NULL`

This happens because the error occurs in Phase 4 (`setup_streaming_pipeline`) after Phase 3 has already persisted the records.

## Root Causes

### 1. Streaming Pipeline Setup Failure (handlers.rs ~line 1393)
When `get_chat_completion_response` fails (e.g., connection refused), the error propagates up but doesn't clean up database records.

### 2. Unexpected Stream Closure (handlers.rs ~line 1258)
When the completion stream closes without a Done signal, the processor task wasn't explicitly notifying the storage task.

## Fixes Applied

### Fix 1: Error Handling in create_response_stream
**File**: `src/web/responses/handlers.rs`
**Location**: Phase 4 setup

Changed from:
```rust
let (mut rx_client, response) = setup_streaming_pipeline(
    &state, &user, &body, &context, &prepared, &persisted, &headers,
)
.await?;
```

To:
```rust
let (mut rx_client, response) = match setup_streaming_pipeline(
    &state, &user, &body, &context, &prepared, &persisted, &headers,
)
.await
{
    Ok(result) => result,
    Err(e) => {
        // Clean up database records
        error!("Failed to setup streaming pipeline for response {}: {:?}", 
               persisted.response.uuid, e);
        
        // Update response status to failed
        if let Err(db_err) = state.db.update_response_status(
            persisted.response.id,
            ResponseStatus::Failed,
            Some(Utc::now()),
        ) {
            error!("Failed to update response status: {:?}", db_err);
        }
        
        // Update assistant message to incomplete
        if let Err(db_err) = state.db.update_assistant_message(
            prepared.assistant_message_id,
            None, // No content
            0,    // No tokens
            STATUS_INCOMPLETE.to_string(),
            None, // No finish_reason
        ) {
            error!("Failed to update assistant message: {:?}", db_err);
        }
        
        return Err(e);
    }
};
```

### Fix 2: Explicit Error Notification on Stream Closure
**File**: `src/web/responses/handlers.rs`
**Location**: Processor task in `setup_streaming_pipeline`

Changed from:
```rust
let Some(chunk) = chunk_opt else {
    debug!("Completion stream ended without explicit Done");
    break;
};
```

To:
```rust
let Some(chunk) = chunk_opt else {
    error!("Completion stream closed unexpectedly without Done signal");
    // Explicitly notify storage and client of the failure
    let msg = StorageMessage::Error("Stream closed unexpectedly".to_string());
    let _ = tx_storage.send(msg.clone()).await;
    let _ = tx_client.send(msg).await;
    break;
};
```

## Testing

To verify the fix works:

1. **Test provider connection failure**:
   ```bash
   # Stop the provider service
   # Make a request to the Responses API
   # Check database:
   SELECT uuid, status FROM responses ORDER BY created_at DESC LIMIT 1;
   SELECT uuid, status, finish_reason FROM assistant_messages ORDER BY created_at DESC LIMIT 1;
   ```
   
   Expected:
   - `responses.status` = `'failed'`
   - `assistant_messages.status` = `'incomplete'`

2. **Test stream closure during streaming**:
   ```bash
   # Kill the provider mid-stream
   # Check database
   ```
   
   Expected:
   - `responses.status` = `'failed'`
   - `assistant_messages.status` = `'incomplete'`
   - `assistant_messages.content_enc` = encrypted partial content (if any was received)

3. **Test cancellation**:
   ```bash
   # POST /v1/responses/{id}/cancel during streaming
   ```
   
   Expected:
   - `responses.status` = `'cancelled'`
   - `assistant_messages.status` = `'incomplete'`
   - `assistant_messages.finish_reason` = `'cancelled'`

## State Transitions

### Before Fix
```
[Phase 3: Persist] → response.status='in_progress', assistant.status='in_progress'
[Phase 4: Setup fails] → ERROR returned, records left unchanged ❌
```

### After Fix
```
[Phase 3: Persist] → response.status='in_progress', assistant.status='in_progress'
[Phase 4: Setup fails] → response.status='failed', assistant.status='incomplete' ✅
[Phase 4: Setup succeeds]
  → [Stream fails] → response.status='failed', assistant.status='incomplete' ✅
  → [Stream cancelled] → response.status='cancelled', assistant.status='incomplete' ✅
  → [Stream completes] → response.status='completed', assistant.status='completed' ✅
```

---

# Security & DoS Issues in Responses API

**Date**: 2024
**File Analyzed**: `src/web/responses/handlers.rs` and related files

## 🔴 HIGH SEVERITY

### 1. Unbounded Memory Growth (DoS) ✅ MITIGATED BY EXISTING PROTECTIONS

**Locations**:
- `handlers.rs` lines 1389-1505 (client stream accumulator)
- `storage.rs` ContentAccumulator

**Original Concern**: Response content could grow unbounded, causing memory exhaustion.

**Status**: ✅ **MITIGATED** - Multiple layers of protection exist:
1. **Trusted Provider Assumption**: LLM providers are not malicious actors
2. **Natural Token Limits**: Text responses bounded by `max_output_tokens` parameter (default 10,000 tokens ≈ 40-50KB max)
3. **Upstream Input Limits**: 50MB middleware limit + Cloudflare limits prevent malicious clients from triggering massive responses via huge prompts
4. **Content Type**: Responses are text-only (images only in input); even with generous token limits, text responses remain bounded

**Rationale**: The combination of trusted providers, natural token limits, and upstream input validation makes unbounded growth impractical. A 100K token response (~400KB) would be an extreme outlier and still well within reasonable memory bounds per request.

**No action needed.**

---

### 2. No Size Limits on Request Metadata ✅ MITIGATED BY EXISTING PROTECTIONS

**Location**: `handlers.rs` lines 1032-1037 in `persist_request_data`

**Original Concern**: Gigabytes of metadata could exhaust memory during deserialization/encryption.

**Status**: ✅ **MITIGATED** - Protected by existing infrastructure:
1. **50MB Middleware Limit**: `encryption_middleware.rs` enforces `MAX_ENCRYPTED_BODY_BYTES = 50MB` on all request bodies (includes metadata)
2. **Cloudflare Limits**: Upstream WAF provides additional size enforcement
3. **JSON Deserialization**: Axum's JSON deserialization happens within the 50MB budget

**Rationale**: The entire encrypted request body (including all metadata) cannot exceed 50MB. This is enforced before any application-level deserialization or processing occurs.

**No action needed.**

---

## 🟡 MEDIUM SEVERITY

### 3. Integer Overflow on Token Counting

**Location**: `handlers.rs` line ~905 in `validate_and_normalize_input`

**Code**:
```rust
let user_message_tokens = count_tokens(&input_text_for_tokens) as i32;
```

**Impact**: If token count exceeds i32::MAX (2,147,483,647), this will wrap around or truncate, leading to incorrect billing and potentially bypassing token limits.

**Recommendation**: Use checked conversion:
```rust
let token_count = count_tokens(&input_text_for_tokens);
let user_message_tokens = i32::try_from(token_count).map_err(|_| {
    error!("Token count {} exceeds i32::MAX", token_count);
    ApiError::PayloadTooLarge
})?;
```

---

### 4. Missing Parameter Validation

**Location**: `handlers.rs` lines 1189-1191 in `setup_streaming_pipeline`

**Code**:
```rust
"temperature": body.temperature.unwrap_or(DEFAULT_TEMPERATURE),
"top_p": body.top_p.unwrap_or(DEFAULT_TOP_P),
"max_tokens": body.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
```

**Impact**: Invalid parameters could be passed to upstream APIs, potentially causing errors or unexpected behavior. Could also allow negative `max_output_tokens`.

**Status**: Ignored by product decision — not a big deal; leaving current behavior.

**Recommendation**: Add validation in Phase 1 (`validate_and_normalize_input`):
```rust
// Validate temperature
if let Some(temp) = body.temperature {
    if temp < 0.0 || temp > 2.0 {
        error!("Invalid temperature: {}", temp);
        return Err(ApiError::BadRequest);
    }
}

// Validate top_p
if let Some(top_p) = body.top_p {
    if top_p < 0.0 || top_p > 1.0 {
        error!("Invalid top_p: {}", top_p);
        return Err(ApiError::BadRequest);
    }
}

// Validate max_output_tokens
if let Some(max_tokens) = body.max_output_tokens {
    if max_tokens <= 0 || max_tokens > 100_000 {
        error!("Invalid max_output_tokens: {}", max_tokens);
        return Err(ApiError::BadRequest);
    }
}
```

---

### 5. Conversation History DoS ✅ FIXED

**Location**: `context_builder.rs` line 63

**Original Code**:
```rust
let raw = db.get_conversation_context_messages(
    conversation_id,
    i64::MAX, // No limit - fetch all messages!
    None,
    "asc",
)
```

**Impact**: A conversation with 10,000+ messages will load ALL into memory, decrypt them all, then truncate. This wastes resources and could be exploited for DoS.

**Status**: ✅ **FIXED** - Implemented two-pass optimization to only decrypt messages that will be used.

## Solution: Two-Pass Metadata-Based Optimization

### Problem Analysis

The original flow was extremely inefficient:
1. Fetch ALL messages from DB (UNION of 4 tables: user_messages, assistant_messages, tool_calls, tool_outputs)
2. Decrypt ALL `content_enc` fields (CPU intensive encryption operations)
3. Build ChatMsg objects with token counts for ALL messages
4. Run truncation logic to determine which messages fit in context window
5. **Discard 90%+ of messages** that don't fit in the context budget

**Example**: For a 1000-message conversation where only 50 messages fit in context:
- ❌ Old: Decrypt 1000 messages, discard 950
- ✅ New: Fetch metadata for 1000, decrypt only 50 needed

### Implementation

#### Step 1: New Lightweight Metadata Struct

**File**: `src/models/responses.rs`

Created `RawThreadMessageMetadata` with only fields needed for truncation decisions:

```rust
#[derive(QueryableByName, Debug)]
pub struct RawThreadMessageMetadata {
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub message_type: String,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub id: i64,
    #[diesel(sql_type = diesel::sql_types::Uuid)]
    pub uuid: Uuid,
    #[diesel(sql_type = diesel::sql_types::Timestamptz)]
    pub created_at: DateTime<Utc>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Integer>)]
    pub token_count: Option<i32>,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Uuid>)]
    pub tool_call_id: Option<Uuid>,
}
```

**Key difference**: No `content_enc` field - avoids fetching encrypted bytea columns.

#### Step 2: New DB Methods

**File**: `src/models/responses.rs`

**Method 1**: `get_conversation_context_metadata()` - Fetch only metadata
```rust
impl RawThreadMessageMetadata {
    pub fn get_conversation_context_metadata(
        conn: &mut PgConnection,
        conversation_id: i64,
    ) -> Result<Vec<RawThreadMessageMetadata>, ResponsesError> {
        // Same UNION query structure but SELECT only:
        // - message_type, id, uuid, created_at, token_count, tool_call_id
        // - EXCLUDE content_enc/arguments_enc/output_enc (heavy bytea fields)
    }
}
```

**Method 2**: `get_messages_by_ids()` - Fetch full messages for specific IDs
```rust
impl RawThreadMessage {
    pub fn get_messages_by_ids(
        conn: &mut PgConnection,
        conversation_id: i64,
        message_ids: &[(String, i64)], // (message_type, id) pairs
    ) -> Result<Vec<RawThreadMessage>, ResponsesError> {
        // Targeted fetch with WHERE clauses:
        // (message_type = 'user' AND id IN (...))
        // OR (message_type = 'assistant' AND id IN (...))
        // AND conversation_id = $1  -- for safety
    }
}
```

#### Step 3: Refactored Context Building

**File**: `src/web/responses/context_builder.rs`

**New Flow**:
```rust
pub fn build_prompt<D: DBConnection + ?Sized>(
    db: &D,
    conversation_id: i64,
    user_id: uuid::Uuid,
    user_key: &secp256k1::SecretKey,
    model: &str,
    override_instructions: Option<&str>,
) -> Result<(Vec<serde_json::Value>, usize), crate::ApiError> {
    // 1. Get user instructions (decrypt - always needed)
    let mut msgs: Vec<ChatMsg> = Vec::new();
    // ... handle system message ...

    // 2. PASS 1: Fetch only metadata (lightweight, no decryption)
    let metadata = db.get_conversation_context_metadata(conversation_id)?;

    // 3. Run truncation logic on metadata to determine needed IDs
    let needed_ids = determine_needed_message_ids(
        metadata,
        model_max_ctx(model),
        msgs.iter().map(|m| m.tok).sum(), // system message tokens
    )?;

    // 4. PASS 2: Fetch and decrypt ONLY the messages we need
    let raw = db.get_messages_by_ids(conversation_id, &needed_ids)?;

    // 5. Decrypt only the needed messages
    for r in raw {
        let content_enc = match &r.content_enc {
            Some(enc) => enc,
            None => continue,
        };
        let plain = decrypt_with_key(user_key, content_enc)?;
        // ... build ChatMsg ...
    }

    // 6. Format for LLM API
    build_prompt_from_chat_messages(msgs, model)
}
```

**New Helper Function**: `determine_needed_message_ids()`
```rust
fn determine_needed_message_ids(
    metadata: Vec<RawThreadMessageMetadata>,
    model: &str,
    system_tokens: usize,
) -> Result<Vec<(String, i64)>, crate::ApiError> {
    let max_ctx = model_max_ctx(model);
    let ctx_budget = max_ctx.saturating_sub(4096 + 500); // response_reserve + safety
    
    let total_tokens: usize = metadata.iter()
        .filter_map(|m| m.token_count.map(|t| t as usize))
        .sum();
    
    if system_tokens + total_tokens <= ctx_budget {
        // All messages fit - return all IDs
        return Ok(metadata.iter()
            .map(|m| (m.message_type.clone(), m.id))
            .collect());
    }
    
    // Apply middle truncation logic on metadata
    // Keep: first user message + recent messages that fit in budget
    let mut needed: Vec<(String, i64)> = Vec::new();
    
    // ... implement same truncation logic as before, but on metadata ...
    // ... collect IDs of messages we need to keep ...
    
    Ok(needed)
}
```

### Performance Improvements

**Before (1000 message conversation, 50 messages fit in context)**:
- Query 1: Fetch 1000 full messages (including bytea `content_enc`) ≈ 10MB+
- Decrypt 1000 messages (CPU intensive) ≈ 1000 decrypt operations
- Truncate: Keep 50, discard 950
- **Wasted**: 950 decrypt operations + 9.5MB+ data transfer

**After**:
- Query 1: Fetch 1000 metadata rows (no bytea) ≈ 100KB
- Run truncation logic on metadata
- Query 2: Fetch 50 full messages (targeted `WHERE id IN (...)`) ≈ 500KB
- Decrypt 50 messages ≈ 50 decrypt operations
- **Savings**: 950 decrypt operations avoided, ~10MB less data transfer

### Database Query Optimization

The `get_messages_by_ids()` query uses existing indexes efficiently:

```sql
-- Leverages these existing indexes:
-- idx_user_messages_conversation_created_id (conversation_id, created_at DESC, id)
-- idx_assistant_messages_conversation_created_id (conversation_id, created_at DESC, id)
-- idx_tool_calls_conversation_created_id (conversation_id, created_at DESC, id)
-- idx_tool_outputs_conversation_created_id (conversation_id, created_at DESC, id)

WHERE conversation_id = $1 
  AND (
    (message_type = 'user' AND id IN (...)) OR
    (message_type = 'assistant' AND id IN (...)) OR
    (message_type = 'tool_call' AND id IN (...)) OR
    (message_type = 'tool_output' AND id IN (...))
  )
```

### Trade-offs

**Pros**:
- ✅ Massive reduction in decryption operations (950 avoided in example)
- ✅ Significant memory savings (avoid loading unnecessary bytea fields)
- ✅ Faster response times for large conversations
- ✅ Maintains exact same truncation logic and behavior

**Cons**:
- ⚠️ Two DB queries instead of one (metadata + targeted fetch)
  - Mitigated: Second query is highly targeted with WHERE id IN clause
  - Mitigated: Metadata query is very lightweight (no bytea columns)
- ⚠️ Slightly more complex code
  - Mitigated: Truncation logic extracted to pure function
  - Mitigated: Well-tested with existing test suite

**Net Result**: The overhead of a second lightweight query is negligible compared to avoiding hundreds/thousands of decrypt operations.

---

### 6. No Input Content Size Limit ✅ MITIGATED BY EXISTING PROTECTIONS

**Location**: `handlers.rs` line ~897 in `validate_and_normalize_input`

**Original Concern**: Extremely large user messages could consume excessive memory/CPU during encryption.

**Status**: ✅ **MITIGATED** - Protected by existing infrastructure:
1. **50MB Middleware Limit**: `encryption_middleware.rs` enforces `MAX_ENCRYPTED_BODY_BYTES = 50MB` on all incoming request bodies
2. **Cloudflare Limits**: Upstream WAF provides additional request size enforcement
3. **Pre-Decryption Check**: Size validation happens before decryption/deserialization

**Rationale**: The entire encrypted request (including all message content and attachments) is capped at 50MB before any application processing. This provides sufficient protection against resource exhaustion from oversized inputs.

**No action needed.**

---

## 🟢 LOW SEVERITY

### 7. Debug Assertions Could Panic

**Location**: `context_builder.rs` line 219

**Code**:
```rust
debug_assert!(
    total <= ctx_budget,
    "Token count {} exceeds budget {}",
    total,
    ctx_budget
);
```

**Impact**: In debug builds, this could panic if the assertion fails. Production builds are fine, but it indicates a logic error.

**Status**: Ignored by product decision — not a big deal.

**Recommendation**: Replace with a runtime check and error return:
```rust
if total > ctx_budget {
    error!("Token count {} exceeds budget {}", total, ctx_budget);
    return Err(crate::ApiError::PayloadTooLarge);
}
```

---

## Good Security Practices Observed

✅ Guest users are blocked (line ~866)  
✅ Billing checks happen BEFORE persistence (Phase 2)  
✅ Safe JSON chaining with Option methods (no panicking unwraps)  
✅ Cancellation support via broadcast channels  
✅ Encryption key retrieval errors are handled properly  
✅ Channel buffer sizes are reasonable (1024)  
✅ Error types are properly mapped and don't leak internal details  

---

## Priority Recommendations

**STATUS UPDATE**: Most HIGH severity issues are already mitigated by existing protections (50MB middleware limit, Cloudflare, trusted providers, natural token limits).

### Remaining Issues Worth Addressing:

1. **Medium**: Add parameter validation (#4) — Ignored by product decision
2. **Medium**: Fix integer overflow on token counting (#3) — Already fixed
3. **Low**: Replace debug assertions with runtime checks (#7) — Ignored by product decision

### Already Protected:
- ✅ #1: Unbounded memory growth (mitigated by trusted providers + token limits + upstream limits)
- ✅ #2: Metadata size (mitigated by 50MB middleware limit)
- ✅ #5: Conversation history DoS (fixed with two-pass optimization)
- ✅ #6: Input content size (mitigated by 50MB middleware limit)
- ✅ #8: Encrypted body size (fixed with 50MB limit)
- ✅ #9: Sensitive logs (uses trace! only)
- ✅ #10: Provider timeouts (fixed with request and stream timeouts)
- ✅ #14: Client backpressure blocking storage (fixed with try_send)

---

## Implementation Checklist

### ✅ Completed (Mitigated by Existing Infrastructure)
- [x] **#1 MITIGATED**: Content size limits - Protected by trusted providers, natural token limits, and upstream 50MB limits
- [x] **#2 MITIGATED**: Metadata size validation - Protected by 50MB middleware limit
- [x] **#6 MITIGATED**: Input size validation - Protected by 50MB middleware limit
- [x] **#8 FIXED**: Limit encrypted body size in `decrypt_request` to 50MB (2025-01-XX)
  - Added `ApiError::PayloadTooLarge` error variant
  - Added `MAX_ENCRYPTED_BODY_BYTES` constant (50MB) in `encryption_middleware.rs`
  - Changed `to_bytes(body, usize::MAX)` to use size limit
- [x] **#5 FIXED**: Optimize conversation history loading with two-pass metadata approach
  - Created `RawThreadMessageMetadata` struct (lightweight, no `content_enc` field)
  - Added `get_conversation_context_metadata()` DB method  
  - Added `get_messages_by_ids()` DB method for targeted retrieval
  - Refactored `build_prompt()` to fetch metadata first, run truncation logic, then fetch only needed messages
  - All existing tests pass - behavior unchanged
  - For 1000-message conversations, avoids 950+ unnecessary decrypt operations
- [x] **#9 VERIFIED**: Sensitive data in logs - All sensitive logging uses `trace!` level only (disabled in production)
- [x] **#10 FIXED**: Add connect/request/read timeouts in provider calls
  - Added `REQUEST_TIMEOUT_SECS = 120` for request timeout (generous for large non-streaming responses)
  - Added `STREAM_CHUNK_TIMEOUT_SECS = 120` for streaming chunk timeout
  - Wrapped all provider requests with `timeout()` in `try_provider()`, `send_transcription_request()`, and `proxy_tts()`
  - Wrapped streaming loop in `get_chat_completion_response()` with per-chunk timeout

### 🔄 Remaining Work
- [x] **#3 FIXED**: Token counting integer overflow protection
  - Changed all `count_tokens() as i32` casts to clamping logic
  - If token count exceeds i32::MAX, clamps to i32::MAX with warning log
  - Ensures billing still happens and no panics/errors
  - Fixed in 4 locations: handlers.rs (user messages), storage.rs (completion fallback), instructions.rs (create/update)
- [ ] **#4**: Add parameter validation (temperature, top_p, max_tokens) in Phase 1
- [ ] **#4**: Add parameter validation (temperature, top_p, max_tokens) in Phase 1 — Ignored by product decision
- [ ] **#7**: Replace debug_assert with runtime error in context_builder — Ignored by product decision

### Optional Improvements (Lower Priority)
- [ ] **#11**: Title generation is desired and already enforced by per-user quotas — no action needed
- [ ] **#12**: Lower `DEFAULT_MAX_TOKENS` and enforce per-plan caps
- [ ] **#13**: Switch header forwarding to allowlist in `try_provider`
- [x] **#14 FIXED**: Use `try_send` for client channel; keep awaited storage sends
- [x] **#15 FIXED**: Removed unnecessary mutex from token encoder - `CoreBPE` is immutable and thread-safe

---

## Additional 🔴 HIGH SEVERITY

### 8. Unbounded Encrypted Request Body Read (DoS)

**Location**: `web/encryption_middleware.rs` in `decrypt_request`

**Code**:
```rust
let body_bytes = axum::body::to_bytes(body, usize::MAX)
    .await
    .map_err(|_| ApiError::BadRequest)?;
```

**Impact**: Allows arbitrarily large request bodies to be read into memory prior to decryption/deserialization. A malicious client can exhaust memory and crash the server despite upstream CF limits (e.g., via chunked uploads or misconfiguration).

**Recommendation**: Impose a strict limit and return a 413-equivalent error.
```rust
const MAX_ENCRYPTED_BODY_BYTES: usize = 1 * 1024 * 1024; // 1MB
let body_bytes = axum::body::to_bytes(body, MAX_ENCRYPTED_BODY_BYTES)
    .await
    .map_err(|_| ApiError::PayloadTooLarge)?;
```
Also consider validating decrypted payload size before deserialization.

---

### 9. Sensitive Data in Logs (PII/Secrets Leakage) ✅ NOT A CONCERN

**Locations**:
- `responses/handlers.rs` line 1340: logs full `body` 
- `responses/handlers.rs` lines 1197-1202: logs pretty-printed `chat_request` (includes messages)
- `responses/handlers.rs` line 1594: logs content deltas
- `storage.rs` line 37: logs content delta sizes

**Code**:
```rust
trace!("Request body: {:?}", body);
trace!(
    "Chat completion request to model {}: {}",
    body.model,
    serde_json::to_string_pretty(&chat_request).unwrap_or_else(|_| "failed to serialize".to_string())
);
trace!("Client stream received content delta: {}", content);
```

**Status**: ✅ **All sensitive data logging uses `trace!` level only.** This is disabled in production by default, so there is no actual risk. Debug/info/warn/error logs only contain UUIDs, counts, and error messages—no user content.

**No action needed.**

---

### 10. Missing Provider Request/Stream Timeouts (Resource Pinning) ✅ FIXED

**Location**: `web/openai.rs` (provider HTTP calls and SSE read loop)

**Status**: ✅ **Fixed** - Added timeouts to all provider HTTP calls and streaming reads.

**Implementation**:
- Added `REQUEST_TIMEOUT_SECS = 120` constant for request timeout (generous for large non-streaming responses)
- Added `STREAM_CHUNK_TIMEOUT_SECS = 120` constant for per-chunk streaming timeout
- Wrapped all `client.request()` calls with `timeout()` in:
  - `try_provider()` - chat completions
  - `send_transcription_request()` - audio transcriptions  
  - `proxy_tts()` - text-to-speech
- Wrapped streaming loop in `get_chat_completion_response()` with per-chunk timeout
- Proper error handling and logging for all timeout cases

**Changes Made**:
```rust
// Constants added
const REQUEST_TIMEOUT_SECS: u64 = 120;
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 120;

// Request timeout example
match timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS), client.request(req)).await {
    Ok(Ok(response)) => { /* success */ }
    Ok(Err(e)) => { /* request error */ }
    Err(_) => { /* timeout error */ }
}

// Streaming timeout
loop {
    match timeout(Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS), body_stream.next()).await {
        Ok(Some(chunk)) => { /* process */ }
        Ok(None) => break,
        Err(_) => { /* timeout - send error and break */ }
    }
}
```

---

## Additional 🟡 MEDIUM SEVERITY

### 11. Cost Amplification via Automatic Title Generation

**Location**: `responses/handlers.rs` `spawn_title_generation_task`

**Impact**: Each first message triggers an extra LLM call (70B model). Attackers can create many new conversations to multiply costs and background workloads.

**Status**: Accepted by design — title generation is desired and already governed by per-user quotas; no action needed.

**Recommendation**: No action needed beyond existing per-user quotas.

---

### 12. Excessive Default Max Output Tokens

**Location**: `responses/constants.rs`

**Code**:
```rust
pub const DEFAULT_MAX_TOKENS: i32 = 10_000;
```

**Impact**: Encourages long generations and increases provider cost and stream duration by default.

**Recommendation**: Reduce default (e.g., 2048) and enforce per-plan upper bounds server-side. Validate user-supplied `max_output_tokens` as noted in issue #4.

---

### 13. Header Forwarding May Leak Client/Internal Headers

**Location**: `web/openai.rs` `try_provider`

**Code**:
```rust
for (key, value) in headers.iter() {
    if key != header::HOST && key != header::AUTHORIZATION &&
       key != header::CONTENT_LENGTH && key != header::CONTENT_TYPE {
        req = req.header(name, val);
    }
}
```

**Impact**: Forwards nearly all headers to third-party providers, potentially leaking internal correlation or experimental headers.

**Recommendation**: Switch to an allowlist (e.g., `User-Agent`, `X-Request-Id`) and drop everything else. Avoid forwarding cookie/session headers entirely.

---

### 14. Client SSE Channel Backpressure Can Stall Upstream Processor

**Location**: `responses/handlers.rs` `setup_streaming_pipeline` processor task

**Code**:
```rust
// Best-effort send to client
let _ = tx_client.send(msg).await; // still awaits if buffer is full
```

**Impact**: If the client channel’s buffer fills (slow client), `.send().await` waits, which can add backpressure to the processor loop and delay storage sends.

**Recommendation**: Use `try_send` for client channel (drop or coalesce deltas) while preserving `await` for storage channel only.

---

## Additional 🟢 LOW SEVERITY

### 15. Potential Panic on Mutex Poisoning in Token Encoder ✅ FIXED

**Location**: `tokens.rs`

**Original Code**:
```rust
static ENCODER: Lazy<Mutex<tiktoken_rs::CoreBPE>> =
    Lazy::new(|| Mutex::new(cl100k_base().expect("init cl100k encoder")));

pub fn count_tokens(text: &str) -> usize {
    ENCODER.lock().expect("encoder lock")
        .encode_with_special_tokens(text)
        .len()
}
```

**Impact**: A poisoned mutex would panic future callers. Rare, but could crash a worker and cause cascading failures.

**Status**: ✅ **FIXED** - Removed unnecessary mutex entirely.

**New Implementation**:
```rust
static ENCODER: Lazy<tiktoken_rs::CoreBPE> =
    Lazy::new(|| cl100k_base().expect("init cl100k encoder"));

pub fn count_tokens(text: &str) -> usize {
    ENCODER.encode_with_special_tokens(text).len()
}
```

**Rationale**: `tiktoken_rs::CoreBPE` is immutable and thread-safe by design. The mutex was unnecessary overhead - `Lazy` already handles thread-safe initialization, and `CoreBPE` can be safely shared across threads without additional locking. This eliminates the mutex poisoning risk entirely and improves performance by removing lock contention.

---

### 16. SSE Payload Amplification

**Location**: `responses/handlers.rs` event emission

**Impact**: Full assistant content is included in multiple “done” events, increasing bandwidth and client processing time.

**Recommendation**: Consider slimming final events (e.g., send only final snapshot once, or include references) to reduce bandwidth for very long outputs.

---

## Summary of Security Posture

**Overall Status**: The application is **well-protected** against most DoS and security threats through multiple layers:

1. **Infrastructure Layer**: Cloudflare + 50MB middleware limit catches malicious input
2. **Application Layer**: Trusted provider assumption + natural token limits bound outputs  
3. **Architecture**: Proper timeout handling, encryption, and billing checks

**Key Remaining Work**: Only defensive improvements (#3, #4, #7) and optional optimizations (#11-15) remain. All critical DoS vectors are already mitigated.
