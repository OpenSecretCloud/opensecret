Below is a pragmatic, end-to-end blueprint for a green-field "Responses-compatible" chat server with Rust on the backend, TypeScript on the client, and PostgreSQL for storage. It is deliberately opinionated but every moving part can be swapped to suit your stack.

**Note**: This complements the existing `/v1/chat/completions` endpoint. Use that for simple stateless requests. Use `/v1/responses` when you need job-based async processing, server-side tool execution, or OpenAI Responses API compatibility.

?

1. What "Responses-compatible" really means

Aspect	OpenAI Responses API	Implication for your server
Single entry-point	POST /v1/responses - no threads/runs; the request contains the entire conversation state (input, previous_response_id if you want continuity).	You must rebuild the full prompt every time a user sends a new message.
Key request params	model, input (string or image array), tools, tool_choice, temperature, top_p, max_output_tokens, parallel_tool_calls, store, metadata, stream	Your Rust DTOs must match these exactly so the TypeScript SDK can hit your server unchanged.
Streaming	Server-Sent Events (text/event-stream). Each chunk is an event object (event, id, data). Common events are response.delta, response.done, tool-call events, etc.	Re-use the same SSE framing so the browser side can pipe chunks straight into your existing stream handler. Note: SSE typically doesn't support gzip compression; document your compression strategy.
Tool calls	Model returns JSON: {"type":"tool_call","id":"toolcall_...","name":"function_name","arguments":{...}}. Client (or server) POSTs /v1/responses/{id}/tool_outputs.	Persist pending tool calls; spin a worker that resolves them and then PATCHes the response record with output.
Status lifecycle	queued → in_progress → requires_action (for tool calls) → completed / failed / canceled.	Keep the same enum so UI libraries such as the OpenAI TS SDK render correctly.


?

2. High-level architecture

graph TD
    subgraph Rust API (Axum)
      G1[POST /v1/responses] --> Q((Job Queue))
      Q --> W1[Worker: build ChatCompletion & stream]
      W1 --> DB[(PostgreSQL)]
      W1 --> SSE[[SSE Hub]]
      SSE --> FE
      W1 -->|tool_call| ToolQ
      ToolQ--> ToolWorker[Run server-side fn / micro-service]
      ToolWorker --> DB
      ToolWorker --> SSE
    end
    FE[React / TS client] --fetch/SSE--> Rust API

	• Axum (or Actix) handles thin HTTP routing + auth (JWT middleware you already have).
	• AWS SQS acts as a durable job queue so that long-running completions do not block request threads.
	• SSE Hub - small in-process publisher that multiplexes chunks to each subscribed browser tab.
	• Workers are async Tokio tasks.
	• Everything that matters is also persisted in PostgreSQL so you can resume crashed jobs or audit later.

?

3. Database schema (DDL snippets)

```sql
-- Table: chat_threads (optional for UI convenience)
create table chat_threads (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(uuid) on delete cascade,
  created_at timestamptz default now()
);

-- Table: chat_messages (stores conversation history)
create table chat_messages (
  id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references chat_threads(id) on delete cascade,
  role text check (role in ('user','assistant','tool')),
  content bytea not null,  -- Encrypted content (binary ciphertext)
  created_at timestamptz default now()
);

-- Type: response status enum
create type response_status as enum
  ('queued','in_progress','requires_action','completed','failed','canceled');

-- Table: responses (core Responses API table)
create table responses (
  id uuid primary key,
  user_id uuid not null references users(uuid) on delete cascade,
  model text not null,
  status response_status default 'queued',
  input bytea,                    -- Encrypted last user turn (binary ciphertext)
  output bytea,                   -- Encrypted full assistant response (binary ciphertext)
  usage jsonb,
  previous_response_id uuid references responses(id) on delete set null,
  temperature real,
  top_p real,
  max_output_tokens int,
  tool_choice text,
  parallel_tool_calls boolean,
  store boolean,
  metadata jsonb,
  error text,
  created_at timestamptz default now(),
  updated_at timestamptz
);

-- Indices for performance
create index idx_responses_user_id on responses(user_id);
create index idx_responses_status on responses(status);
create index idx_responses_created_at on responses(created_at);
create index idx_responses_user_created on responses(user_id, created_at desc); -- Hot path

-- Table: tool_calls
create table tool_calls (
  id uuid primary key default gen_random_uuid(),
  response_id uuid references responses(id) on delete cascade,
  name text not null,
  arguments bytea,  -- Encrypted arguments (binary ciphertext)
  output bytea,     -- Encrypted output (binary ciphertext)
  status text check (status in ('pending','running','succeeded','failed')),
  error text,
  created_at timestamptz default now()
);

create index idx_tool_calls_response_id on tool_calls(response_id);
create index idx_tool_calls_status on tool_calls(status);

-- Table: idempotency_keys (for safe retries)
create table idempotency_keys (
  idempotency_key text not null,
  user_id uuid not null references users(uuid) on delete cascade,
  response_id uuid not null references responses(id) on delete cascade,
  created_at timestamptz default now(),
  primary key (idempotency_key, user_id)
);
```

You still get a "thread" table even though the Responses API is threadless - that's for your convenience in the UI.

?

4. Rust surface types (DTO)

```rust
// Request DTO matching OpenAI spec
#[derive(Deserialize)]
pub struct ResponsesCreate {
    pub model: String,
    pub input: Input,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub previous_response_id: Option<Uuid>,
}
```

(Mirror exactly what the PHP and TS SDKs expect.)

?

5. Request flow in detail

1. Frontend POST /v1/responses with the new user turn and (optionally) previous_response_id.
2. HTTP handler:
   - Validates JWT - identifies user_id.
   - Checks for idempotency key if provided.
   - Inserts placeholder responses row (status='queued').
   - Publishes job-ID to SQS and immediately returns 202 + "id".
   - If stream=true, it also upgrades to an SSE stream before returning so it can proxy chunks later.
3. Worker:
   1. Pulls job; resolves conversation history by querying chat_messages for that thread_id.
   2. Converts into Chat Completions request tailored to your model endpoints.
   3. Calls your model with stream=true and pipes each chunk:
      - Emits SSE: `event: response.delta\ndata: { ... }\n\n`
      - Optionally stores partial chunks in Postgres for recovery (up to you).
   4. On done, writes the assistant message row + updates responses.status='completed'.
   5. If the final chunk contains tool calls - write rows in tool_calls and mark responses.status='requires_action'.
4. Tool worker picks pending tool calls, executes, then POSTs /responses/{id}/tool_outputs which:
   - Appends ROLE=tool message row,
   - updates responses.status back to queued so the main worker continues (loop until no requires_action).

?

6. Streaming implementation (Rust, Axum)

```rust
// SSE handler for streaming responses
async fn sse_handler(
    Extension(db): Extension<PgPool>,
    Extension(broker): Extension<Broadcast<ServerEvent>>,
    req: Json<ResponsesCreate>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let response_id = insert_placeholder(&db, &req).await?;
    let _ = broker.send(ServerEvent::JobQueued(response_id));
    let stream = broker
        .subscribe()
        .filter_map(move |ev| async move {
            match ev {
                ServerEvent::Delta { id, chunk } if id == response_id => {
                    Some(Ok(Event::default()
                        .event("response.delta")
                        .json_data(chunk).unwrap()))
                }
                ServerEvent::Done { id } if id == response_id => {
                    Some(Ok(Event::default().event("response.done")))
                }
                _ => None,
            }
        });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
}
```


?

7. TypeScript client helper (works with the official SDK)

```typescript
// Client code using OpenAI SDK
import OpenAI from "openai";

const openai = new OpenAI({ baseURL: "/v1", apiKey: "dummy" /*ignored*/ });

export async function sendMessage(threadId: string, text: string) {
  const resp = await openai.responses.create({
    model: "local/gemma-2b-it",
    input: text,
    metadata: { threadId },
    stream: true
  });

  for await (const event of resp) {
    switch (event.event) {
      case "response.delta":
        ui.appendDelta(event.data);
        break;
      case "response.done":
        ui.finishMessage();
        break;
    }
  }
}
```

Because our server mirrors the wire format, the SDK "just works".

?

8. Background job worker (simplified)

```rust
// Worker loop processing SQS jobs
loop {
    let job = dequeue_job(&sqs).await;
    let resp = fetch_response(job.response_id).await?;
    let history = load_history(resp.user_id).await?;
    let chat_req = to_chat_completion(history, &resp);
    let mut stream = call_model(chat_req).await?;

    while let Some(delta) = stream.next().await {
        broker.send(ServerEvent::Delta { id: resp.id, chunk: &delta })?;
        append_delta(&db, resp.id, &delta).await?;
    }
    finalize_response(&db, resp.id).await?;
    broker.send(ServerEvent::Done { id: resp.id })?;
}
```


?

9. Why a queue & background worker?

- Frees HTTP threads - large local models can take >30s even on a GPU.
- Retries & exponential back-off on transient GPU/IO failures.
- Makes it trivial to scale horizontally: run N identical workers off the queue.
- Enables graceful shutdown and job recovery on crashes.

?

10. Additional endpoints

### GET /v1/responses
List responses for the authenticated user:
```json
{
  "object": "list",
  "data": [...],
  "has_more": true,
  "first_id": "resp_xyz",
  "last_id": "resp_abc"
}
```

### Extending to uploads & custom tools

- **File uploads** - expose POST /v1/files that writes to S3 / Cloudflare R2, store metadata in Postgres. Attach file_id in tools just like OpenAI.
- **New server-side tools** - add rows in tool_calls with name = "my_custom_tool". The tool worker maintains a registry:

```rust
// Tool registry pattern
match call.name.as_str() {
    "web_search_preview" => run_web_search(call.arguments).await,
    "my_custom_tool"     => my_tool(call.arguments).await,
    _ => mark_failed(),
}
```

- **Realtime / WebSocket** - keep SSE for now; once browsers support WebTransport widely, you can switch transports without touching the DB layer.

?

11. Checklist for "gotchas"

| Item | Tip |
|------|-----|
| Token counting | When you rebuild the prompt, always re-count tokens against the target model context window and apply truncation. |
| Idempotency | Accept an optional Idempotency-Key header so the client can safely retry dropped requests. Store (key, user_id) -> response_id mapping. |
| Multi-tenancy | Store org_id in every table row; index (org_id, id) so a single Postgres cluster can serve several tenants. |
| Billing | Insert a row into your existing cost table right after you record usage returned by the model. |
| Encryption | Use BYTEA columns for encrypted content, not TEXT. Encrypted binary data isn't valid UTF-8. |
| Compression | SSE typically doesn't support gzip. Document your compression strategy or use WebSockets for large payloads. |
| Key rotation | All content encrypted with user keys - plan for key rotation scenarios. |


?

## You can start small

Skip the job queue and tool worker on day-1; just launch the worker task inline. Everything else remains the same, and you can promote to the full architecture later without breaking the API contract.

Good luck - this layout will let your Rust server look and feel exactly like the official OpenAI Responses endpoint while keeping full control over prompt-building, model routing, and data ownership.

## Security considerations

- **Encryption at rest**: All user content (input, output, tool arguments) must be encrypted
- **User isolation**: Every query must include user_id constraints
- **Rate limiting**: Implement per-user rate limits at the API gateway
- **Audit logging**: Log all tool executions and response status changes
- **Key management**: User encryption keys derived from seed, never stored directly
