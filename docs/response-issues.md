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

### 1. Unbounded Memory Growth (DoS)

**Locations**:
- `handlers.rs` lines 1389-1505 (client stream accumulator)
- `storage.rs` ContentAccumulator

**Code**:
```rust
let mut assistant_content = String::new();
// ...
assistant_content.push_str(&content); // No size check!
```

**Impact**: An attacker could send a request that generates extremely long responses (or a malicious LLM provider could return huge responses), causing unbounded memory growth and potentially crashing the server.

**Recommendation**: Add a maximum content size limit (e.g., 1MB) and reject/truncate responses that exceed it:
```rust
const MAX_CONTENT_SIZE: usize = 1_000_000; // 1MB

if assistant_content.len() + content.len() > MAX_CONTENT_SIZE {
    error!("Content size limit exceeded");
    // Send error event to client and break
    let error_event = ResponseErrorEvent {
        event_type: EVENT_RESPONSE_ERROR,
        error: ResponseError {
            error_type: "content_too_large".to_string(),
            message: "Response content exceeded maximum size limit".to_string(),
        },
    };
    yield Ok(ResponseEvent::Error(error_event).to_sse_event(&mut emitter).await);
    break;
}
assistant_content.push_str(&content);
```

Apply the same check in `storage.rs` ContentAccumulator:
```rust
StorageMessage::ContentDelta(delta) => {
    if self.content.len() + delta.len() > MAX_CONTENT_SIZE {
        error!("Storage: content size limit exceeded");
        return AccumulatorState::Failed(FailureData {
            error: "Content size limit exceeded".to_string(),
            partial_content: self.content.clone(),
            completion_tokens: self.completion_tokens,
        });
    }
    self.content.push_str(&delta);
    AccumulatorState::Continue
}
```

---

### 2. No Size Limits on Request Metadata

**Location**: `handlers.rs` lines 1032-1037 in `persist_request_data`

**Code**:
```rust
let metadata_enc = if let Some(metadata) = &body.metadata {
    let metadata_json = serde_json::to_string(metadata)?; // No size check!
    Some(encrypt_with_key(&prepared.user_key, metadata_json.as_bytes()).await)
}
```

**Impact**: Attacker could send gigabytes of metadata, exhausting memory during deserialization/encryption.

**Recommendation**: Add size validation in Phase 1 (`validate_and_normalize_input`):
```rust
const MAX_METADATA_SIZE: usize = 10_000; // 10KB

// Add this check in validate_and_normalize_input
if let Some(metadata) = &body.metadata {
    let metadata_json = serde_json::to_string(metadata)
        .map_err(|_| error_mapping::map_serialization_error("metadata"))?;
    if metadata_json.len() > MAX_METADATA_SIZE {
        error!("Metadata too large: {} bytes", metadata_json.len());
        return Err(ApiError::PayloadTooLarge);
    }
}
```

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

### 5. Conversation History DoS

**Location**: `context_builder.rs` line 63

**Code**:
```rust
let raw = db.get_conversation_context_messages(
    conversation_id,
    i64::MAX, // No limit - fetch all messages!
    None,
    "asc",
)
```

**Impact**: A conversation with 10,000+ messages will load ALL into memory, decrypt them all, then truncate. This wastes resources and could be exploited for DoS.

**Recommendation**: Implement smarter pagination or limit the initial fetch:
```rust
// Fetch only recent messages up to a reasonable limit
// The truncation logic will handle keeping the right messages
const MAX_HISTORY_MESSAGES: i64 = 1000;

let raw = db.get_conversation_context_messages(
    conversation_id,
    MAX_HISTORY_MESSAGES,
    None,
    "desc", // Get most recent first
)
.map_err(|_| crate::ApiError::InternalServerError)?;

// Reverse to chronological order
let raw: Vec<_> = raw.into_iter().rev().collect();
```

Alternatively, implement a smarter two-pass approach:
1. Count total messages and tokens
2. If over budget, fetch only first N and last M messages (skip middle)

---

### 6. No Input Content Size Limit

**Location**: `handlers.rs` line ~897 in `validate_and_normalize_input`

**Code**:
```rust
let content_for_storage = serde_json::to_string(&message_content)?;
let content_enc = encrypt_with_key(&user_key, content_for_storage.as_bytes()).await;
```

**Impact**: No validation on the size of user input before encryption. Extremely large messages could consume excessive memory/CPU during encryption.

**Recommendation**: Add size check after serialization:
```rust
const MAX_INPUT_SIZE: usize = 500_000; // 500KB

let content_for_storage = serde_json::to_string(&message_content)
    .map_err(|_| error_mapping::map_serialization_error("message content"))?;

if content_for_storage.len() > MAX_INPUT_SIZE {
    error!("Input message too large: {} bytes", content_for_storage.len());
    return Err(ApiError::PayloadTooLarge);
}

let content_enc = encrypt_with_key(&user_key, content_for_storage.as_bytes()).await;
```

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

1. **Immediate**: Implement content size limits (#1)
2. **High**: Add input/metadata size validation (#2, #6)
3. **Medium**: Add parameter validation (#4)
4. **Medium**: Fix integer overflow on token counting (#3)
5. **Medium**: Optimize conversation history loading (#5)
6. **Low**: Replace debug assertions with runtime checks (#7)

---

## Implementation Checklist

- [ ] Define constants for size limits in `constants.rs`
- [ ] Add content size check in client stream loop
- [ ] Add content size check in storage accumulator
- [ ] Add metadata size validation in Phase 1
- [ ] Add input size validation in Phase 1
- [ ] Add parameter validation (temperature, top_p, max_tokens) in Phase 1
- [ ] Fix token counting integer overflow with try_from
- [ ] Optimize conversation history loading with pagination
- [ ] Replace debug_assert with runtime error in context_builder
- [ ] Add `ApiError::PayloadTooLarge` variant if not exists
- [ ] Add tests for size limit edge cases

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

### 9. Sensitive Data in Logs (PII/Secrets Leakage)

**Locations**:
- `responses/handlers.rs` `create_response_stream`: logs full `body` and flags
- `responses/handlers.rs` `setup_streaming_pipeline`: logs pretty-printed `chat_request` (includes messages)
- Various trace logs include content deltas and generated titles

**Code**:
```rust
trace!("Request body: {:?}", body);
trace!(
    "Chat completion request to model {}: {}",
    body.model,
    serde_json::to_string_pretty(&chat_request).unwrap_or_else(|_| "failed to serialize".to_string())
);
```

**Impact**: Persists user prompts, responses, and metadata into logs. This is a privacy and compliance risk and can leak secrets inadvertently pasted by users.

**Recommendation**: Redact or remove content-bearing logs in production. Log only metadata (IDs, sizes, counts). Guard verbose logs behind feature flags or non-prod builds.

---

### 10. Missing Provider Request/Stream Timeouts (Resource Pinning)

**Location**: `web/openai.rs` (provider HTTP calls and SSE read loop)

**Code**:
```rust
let client = Client::builder()
    .pool_idle_timeout(Duration::from_secs(30))
    .pool_max_idle_per_host(10)
    .build::<_, HyperBody>(https);
// ... request + streaming without explicit per-request/read timeouts
```

**Impact**: A slow or stalled upstream can hold connections, tasks, and memory indefinitely, degrading capacity (DoS).

**Recommendation**: Apply timeouts around request execution and streaming reads.
```rust
use tokio::time::{timeout, Duration};

let res = timeout(Duration::from_secs(30), client.request(req))
    .await
    .map_err(|_| ApiError::GatewayTimeout)??;

// In streaming loop
while let Ok(Some(chunk_result)) = timeout(Duration::from_secs(60), body_stream.next()).await {
    // handle chunk_result ...
}
```
Use sensible defaults and make them configurable per model/provider.

---

## Additional 🟡 MEDIUM SEVERITY

### 11. Cost Amplification via Automatic Title Generation

**Location**: `responses/handlers.rs` `spawn_title_generation_task`

**Impact**: Each first message triggers an extra LLM call (70B model). Attackers can create many new conversations to multiply costs and background workloads.

**Recommendation**: Gate behind feature flag/plan, throttle per user, use a cheaper model, or defer to a queued background job with rate controls. Consider caching/dedup (don’t re-title similar openings).

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

### 15. Potential Panic on Mutex Poisoning in Token Encoder

**Location**: `tokens.rs`

**Code**:
```rust
ENCODER.lock().expect("encoder lock")
```

**Impact**: A poisoned mutex will panic future callers. Rare, but can crash a worker.

**Recommendation**: Handle poisoning gracefully (e.g., `unwrap_or_else(|e| e.into_inner())`) or reinitialize encoder.

---

### 16. SSE Payload Amplification

**Location**: `responses/handlers.rs` event emission

**Impact**: Full assistant content is included in multiple “done” events, increasing bandwidth and client processing time.

**Recommendation**: Consider slimming final events (e.g., send only final snapshot once, or include references) to reduce bandwidth for very long outputs.

---

## Updated Priority Recommendations

1. Immediate: Enforce body/content size limits (#1, #8)
2. High: Redact sensitive logs and add upstream timeouts (#9, #10)
3. High: Validate input/metadata size and params (#2, #4, #6)
4. Medium: Clamp token counts and optimize history loading (#3, #5)
5. Medium: Reduce default max tokens; header allowlist; throttle title generation (#11, #12, #13)
6. Medium: Use `try_send` for client channel (#14)
7. Low: Replace debug asserts; handle encoder poisoning; reduce SSE amplification (#7, #15, #16)

## Updated Implementation Checklist

- [ ] Limit encrypted body size in `decrypt_request`
- [ ] Redact/remove content-bearing logs in prod (requests, deltas, chat_request)
- [ ] Add connect/request/read timeouts and idle chunk timeouts in provider calls
- [ ] Throttle/flag or cheapen title generation path; add per-user quotas
- [ ] Lower `DEFAULT_MAX_TOKENS` and enforce per-plan caps
- [ ] Switch header forwarding to allowlist in `try_provider`
- [ ] Use `try_send` for client channel; keep awaited storage sends
- [ ] Handle encoder mutex poisoning gracefully
