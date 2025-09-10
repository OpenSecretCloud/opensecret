## Future Enhancements

### Idempotency Support

The Responses API could support idempotent requests to ensure safe retries. This was removed from the initial implementation to simplify the API and make it more OpenAI-compatible, but could be added in the future.

#### Implementation Approach

Idempotency could be handled at the application level with the help of columns in the `responses` table:

**Database Schema**:
```sql
-- Idempotency fields in responses table
idempotency_key TEXT,
request_hash TEXT,
idempotency_expires_at TIMESTAMPTZ,

CONSTRAINT idempotency_fields_check CHECK (
    (idempotency_key IS NULL AND request_hash IS NULL AND idempotency_expires_at IS NULL) OR
    (idempotency_key IS NOT NULL AND request_hash IS NOT NULL AND idempotency_expires_at IS NOT NULL)
)

-- Index for efficient idempotency lookups
CREATE INDEX idx_responses_idempotency 
    ON responses(user_id, idempotency_key) 
    WHERE idempotency_key IS NOT NULL;
```

**Request Handling**:
```rust
// Idempotency key handling - simple application-level check
if let Some(idempotency_key) = headers.get("idempotency-key") {
    // Hash the request body for comparison
    let request_hash = sha256(request_body);
    
    // Check for existing response with this key
    if let Some(existing) = db.get_response_by_idempotency_key(
        &idempotency_key, 
        user_id
    ).await? {
        // Compare request hashes
        if existing.request_hash != request_hash {
            return Err(ApiError::IdempotencyKeyReused {
                message: "Different request body with same idempotency key".into(),
            });
        }
        
        // Return existing response based on status
        match existing.status {
            ResponseStatus::InProgress | ResponseStatus::Queued => {
                // Return 409 for in-progress requests
                return Err(ApiError::Conflict {
                    message: "Request with this idempotency key is already in progress".into(),
                });
            }
            _ => {
                // Return the completed response
                return Ok(existing);
            }
        }
    }
    
    // Include idempotency info when creating the response
    new_response.idempotency_key = Some(idempotency_key);
    new_response.request_hash = Some(request_hash);
    new_response.idempotency_expires_at = Some(now + 24.hours());
}
```

**Key Features**:
- **One execution per key**: Each idempotency key triggers exactly one LLM call
- **Request validation**: Same key with different request body is rejected
- **Automatic expiry**: Keys expire after 24 hours by default
- **In-progress protection**: Returns 409 if the same key is used while processing
- **Status-aware**: Returns completed responses immediately

**Cleanup Strategy**:
```sql
-- Optional periodic cleanup to remove old idempotency data and save storage
UPDATE responses 
SET idempotency_key = NULL, request_hash = NULL, idempotency_expires_at = NULL
WHERE idempotency_expires_at < CURRENT_TIMESTAMP
  AND idempotency_key IS NOT NULL;
```

**Benefits**:
1. **Safe retries**: Clients can safely retry failed requests without duplicating work
2. **Cost savings**: Prevents duplicate LLM calls for the same request
3. **Network resilience**: Handles network interruptions gracefully
4. **Client simplicity**: Clients don't need complex retry logic

**Considerations**:
1. **Storage overhead**: Additional columns and index for idempotency data
2. **Complexity**: More logic in request handling
3. **OpenAI compatibility**: OpenAI doesn't use idempotency keys in their Responses API
4. **Cache management**: Need to decide on expiry strategy and cleanup frequency

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

3. **SSE Heartbeat Support**:
   ```rust
   // Send comment frames every 30s to keep connection alive
   let heartbeat_interval = Duration::from_secs(30);
   let mut heartbeat_timer = interval(heartbeat_interval);
   
   loop {
       tokio::select! {
           _ = heartbeat_timer.tick() => {
               // Send SSE comment frame
               yield Ok(Event::default().comment("heartbeat"));
           }
           chunk = body_stream.next() => {
               // Process normal chunks
           }
       }
   }
   ```
   - Prevents proxy/firewall timeouts
   - Detects client disconnection early
   - No impact on client processing (comments are ignored)

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
