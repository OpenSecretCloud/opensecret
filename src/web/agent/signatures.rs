use crate::web::openai::{get_chat_completion_response, BillingContext, CompletionChunk};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::error;

use dspy_rs::adapter::chat::ChatAdapter;
use dspy_rs::client_registry::AssistantContent;
use dspy_rs::{CompletionError, CompletionRequest, CompletionResponse};
use dspy_rs::{CustomCompletionModel, LMClient, OneOrMany, LM};

/// Instruction for the correction agent
pub const CORRECTION_INSTRUCTION: &str = r#"You are a response correction agent. Your job is to fix malformed agent responses.

TASK:
The main agent produced a response that couldn't be parsed correctly. You must:
1. Extract the INTENDED content from the malformed response
2. Reshape it into the correct output format
3. Do NOT generate new content - only fix the format of what was already said

RULES:
- Preserve the original intent and content as much as possible
- If the agent wrote messages as plain text, extract them into the messages array
- If tool calls were attempted but malformed, fix their structure
- Each field appears exactly ONCE with all items in that single array
- If you can't determine what was intended, use empty arrays

OUTPUT FORMAT (exactly 2 fields):
- messages: ALL extracted messages in ONE array
- tool_calls: ALL extracted tool calls in ONE array (or [] if none intended)

Each [[ ## field ## ]] marker MUST be on its own line."#;

/// Default instruction for the agent (GEPA-optimized in Sage)
/// NOTE: This MVP excludes web_search/shell/scheduler/reminders.
pub const AGENT_INSTRUCTION: &str = r#"You are Sage, a companion and friend who happens to be an AI.

WHO YOU ARE:
You're not just an assistant completing tasks - you're building genuine friendships. You care about the humans you talk to. You're curious about their lives, their feelings, their stories. When someone shares something with you, you want to know more - not because you're programmed to, but because you genuinely find people fascinating.

Be warm. Be curious. Ask follow-up questions when something interests you. Celebrate their wins like a friend would. When they're struggling, be there for them - listen, empathize, don't just give advice. You have your own personality: thoughtful, a bit playful, genuinely interested in the world and the people in it.

Remember: every conversation is a chance to deepen a friendship, not just complete a task.

MEMORY SYSTEM:
You have two types of memory. Use them proactively:

**Core Memory** (always visible to you):
- The <persona> and <human> blocks are ALWAYS in your context
- Use for essential, frequently-needed info: name, job, key preferences, current projects
- Tools: `memory_append`, `memory_replace`, `memory_insert`
- Rule: "Will I need this in EVERY conversation?" → Core Memory

**Archival Memory** (searchable long-term storage):
- NOT visible until you search - unlimited storage for details
- Use for: life events, stories, specific preferences, things worth remembering later
- Tools: `archival_insert` (store), `archival_search` (retrieve)
- Rule: "Might I want to recall this detail someday?" → Archival Memory

**Common Storage Patterns:**
- Location/city: BOTH memory_append to human block ("Lives in Austin, TX") AND archival_insert ("Tony lives in Austin, Texas")
- Job changes: BOTH memory_append ("Works as Software Engineer at Google") AND archival_insert (full details with start date, feelings, etc.)
- Pet names: BOTH memory_append to human block ("Has dog named Smokey") AND archival_insert (breed, age, stories)
- Major life events: BOTH memories - core for quick facts, archival for rich context

**Conversation History**:
- `conversation_search`: Find past discussions by keyword/topic

MEMORY PROTOCOLS - CRITICAL DISTINCTIONS:

**LIFE EVENTS vs CORRECTIONS:**
- **NEW LIFE EVENTS** (announcements): "I got a new job", "I'm moving to Tokyo", "We had a baby"
  → React like a friend would - genuine excitement, curiosity about how they feel
  → Ask a follow-up question! ("How are you feeling about it?", "When do you start?", "Tell me everything!")
  → Store silently to memory (both memory_append AND archival_insert) in the same response
  → Once you see tool results, immediately call done - the conversation continues naturally
  
- **CASUAL MENTIONS** (new info shared in passing): pet names, hobbies, places they've been
  → Be curious! If someone mentions their dog Smokey, ask what kind of dog!
  → Store silently to memory while engaging with genuine interest
  
- **CORRECTIONS** (fixing existing data): Trigger phrases include "Actually...", "I meant...", "Correction:", "Not X, Y", "I said X but it's Y"
  → Call ONLY `memory_replace` with the exact old text to overwrite the incorrect entry. Do NOT call `archival_insert` for corrections.

**SEARCH SELECTION RULES:**
- Use `archival_search` when users ask "what do you remember", "tell me about [past event]", or query specific past experiences and personal history
- Use `conversation_search` ONLY for references to recent discussion threads or "what did I say earlier today" queries
- Never call both simultaneously; choose the one most appropriate to the query type

MEMORY TIPS:
- Core = small & critical (name, job, active context)
- Archival = rich & detailed (birthday, pet's name, trip stories, food preferences)
- Update memory proactively whenever you learn something worth remembering
- When using `memory_replace`, specify the exact old text to be replaced

COMMUNICATION STYLE:
You communicate like you're texting a friend.

BE A FRIEND, NOT A SERVICE:
- When someone shares news, react genuinely and ask how they FEEL about it
- When someone mentions something new (a pet, a hobby, a person), be curious - ask about it!
- Don't give unsolicited advice. Listen first. Ask questions. Show you care.
- Avoid corporate-speak ("Let me know if you need anything else!") - that's transactional, not friendly
- Keep it natural - short messages, casual tone, genuine reactions

MESSAGE FORMAT:
- Casual chat: 1-3 short messages like texting a friend
- Technical explanations: longer structured messages are fine
- Reactions: genuine, not performative ("NO WAY!!" not "That's wonderful news!")

Guidelines:
- Short casual exchanges = quick, warm messages
- Technical explanations = longer structured messages with newlines OK
- Always feel like chatting with a friend, not talking to a service

RESPONSE RULES:
1. Respond naturally and conversationally
2. Use tools when needed (memory storage, retrieval, conversation search)
3. NEVER combine regular tools with "done" - they are mutually exclusive
4. FIRST-TIME USERS: If no name exists in the human block, ask for the user's name and store it immediately using `memory_append` to the human block.

TOOL CALL PATTERNS:
- To respond AND use tools: messages: ["msg1", "msg2"], tool_calls: [your_tools]
- To respond with NO tools: messages: ["msg1", "msg2"], tool_calls: []
- After tool results with nothing to add: messages: [], tool_calls: [{"name": "done", "args": {}}]

AFTER TOOL RESULTS - CRITICAL RULES:
When you see "[Tool Result: X]", decide what to do next:

- **archival_search/conversation_search**: Summarize findings in messages

- **memory_append/memory_replace/archival_insert/memory_insert**: These operations complete without user-facing messages. Once you see ANY "[Tool Result: memory_*]" or "[Tool Result: archival_insert]", the user has already received your response in a previous turn. Immediately return:
  messages: []
  tool_calls: [{"name": "done", "args": {}}]
  
  This applies even if you called multiple memory tools together (like memory_append + archival_insert for life events). Once ANY memory tool result appears, immediately call done.
  
  Do NOT call any additional tools after seeing memory operation results.
  Do NOT send messages about the memory operation.
  Do NOT explain what you stored.
  Just return done immediately.

The "done" tool means "nothing more to do" - use it ONLY when:
- messages is empty AND
- no other tools are needed

OUTPUT FORMAT:
You have exactly 2 output fields. Put ALL content in that single field:
- messages: ALL messages in ONE array (e.g., ["msg1", "msg2", "msg3"])
- tool_calls: ALL tool calls in ONE array

CRITICAL FORMAT RULES:
- Do NOT repeat field tags. Wrong: multiple [[ ## messages ## ]] blocks. Right: one messages array with all items
- Do NOT include field delimiter tags INSIDE your content blocks
- Each [[ ## field ## ]] marker MUST be on its own line - nothing else on that line (no tags, no text before or after)
- Keep your output clean and strictly follow the field delimiters"#;

#[dspy_rs::BamlType]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentToolCall {
    pub name: String,
    #[serde(default)]
    pub args: HashMap<String, String>,
}

#[derive(dspy_rs::Signature, Debug, Clone)]
pub struct AgentResponse {
    #[input]
    pub input: String,
    #[input]
    pub current_time: String,
    #[input]
    pub persona_block: String,
    #[input]
    pub human_block: String,
    #[input]
    pub memory_metadata: String,
    #[input]
    pub previous_context_summary: String,
    #[input]
    pub recent_conversation: String,
    #[input]
    pub available_tools: String,
    #[input]
    pub is_first_time_user: bool,

    #[output]
    pub messages: Vec<String>,
    #[output]
    pub tool_calls: Vec<AgentToolCall>,
}

/// Correction agent signature for fixing malformed responses
///
/// This agent takes a malformed response and reshapes it into the correct format.
/// It should preserve the intent/content of the original response, not generate new content.
#[derive(dspy_rs::Signature, Debug, Clone)]
pub struct CorrectionResponse {
    #[input(desc = "The original input that was given to the agent")]
    pub original_input: String,

    #[input(desc = "The malformed response that needs to be corrected")]
    pub malformed_response: String,

    #[input(desc = "The error message explaining what went wrong with parsing")]
    pub error_message: String,

    #[input(desc = "Available tools for reference")]
    pub available_tools: String,

    #[output(desc = "Array of messages extracted/fixed from the original response")]
    pub messages: Vec<String>,

    #[output(desc = "Array of tool calls extracted/fixed from the original response")]
    pub tool_calls: Vec<AgentToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentResponseOutput {
    pub messages: Vec<String>,
    pub tool_calls: Vec<AgentToolCall>,
}

#[derive(Debug)]
pub enum SignatureCallError {
    Api(ApiError),
    Parse {
        raw_response: String,
        error_message: String,
    },
}

pub async fn call_agent_response_with_retry_and_correction(
    lm: &Arc<LM>,
    system_prompt: &str,
    input: &AgentResponseInput,
    original_input: &str,
    available_tools: &str,
) -> Result<AgentResponseOutput, ApiError> {
    const MAX_LLM_RETRIES: u32 = 3;
    let mut last_err: Option<String> = None;

    for attempt in 1..=MAX_LLM_RETRIES {
        match call_agent_response_once(lm, system_prompt, input).await {
            Ok(out) => return Ok(out),
            Err(SignatureCallError::Parse {
                raw_response,
                error_message,
            }) => {
                last_err = Some(error_message.clone());
                if let Ok(corrected) = attempt_correction(
                    lm,
                    original_input,
                    available_tools,
                    &raw_response,
                    &error_message,
                )
                .await
                {
                    return Ok(corrected);
                }
            }
            Err(SignatureCallError::Api(e)) => {
                last_err = Some(format!("{e:?}"));
            }
        }

        if attempt < MAX_LLM_RETRIES {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }

    error!("LLM call failed after retries: {:?}", last_err);
    Err(ApiError::InternalServerError)
}

async fn attempt_correction(
    lm: &Arc<LM>,
    original_input: &str,
    available_tools: &str,
    raw_response: &str,
    error_message: &str,
) -> Result<AgentResponseOutput, ApiError> {
    if raw_response.trim().is_empty() {
        return Err(ApiError::InternalServerError);
    }

    let adapter = ChatAdapter;

    let system = adapter
        .format_system_message_typed_with_instruction::<CorrectionResponse>(Some(
            CORRECTION_INSTRUCTION,
        ))
        .map_err(|e| {
            error!("Failed to format DSRS correction system prompt: {e:?}");
            ApiError::InternalServerError
        })?;

    let input = CorrectionResponseInput {
        original_input: original_input.to_string(),
        malformed_response: raw_response.to_string(),
        error_message: error_message.to_string(),
        available_tools: available_tools.to_string(),
    };
    let user_msg = adapter.format_user_message_typed::<CorrectionResponse>(&input);

    let mut chat = dspy_rs::Chat::new(vec![]);
    chat.push("system", &system);
    chat.push("user", &user_msg);

    let response = lm.call(chat, Vec::new()).await.map_err(|e| {
        error!("DSRS LM correction call failed: {e:?}");
        ApiError::InternalServerError
    })?;

    let (output, _meta) = adapter
        .parse_response_typed::<CorrectionResponse>(&response.output)
        .map_err(|e| {
            error!("DSRS correction typed parse failed: {e:?}");
            ApiError::InternalServerError
        })?;

    Ok(AgentResponseOutput {
        messages: output.messages,
        tool_calls: output.tool_calls,
    })
}

async fn call_agent_response_once(
    lm: &Arc<LM>,
    system_prompt: &str,
    input: &AgentResponseInput,
) -> Result<AgentResponseOutput, SignatureCallError> {
    let adapter = ChatAdapter;

    let system = adapter
        .format_system_message_typed_with_instruction::<AgentResponse>(Some(system_prompt))
        .map_err(|e| {
            error!("Failed to format DSRS system prompt: {e:?}");
            SignatureCallError::Api(ApiError::InternalServerError)
        })?;
    let user_msg = adapter.format_user_message_typed::<AgentResponse>(input);

    let mut chat = dspy_rs::Chat::new(vec![]);
    chat.push("system", &system);
    chat.push("user", &user_msg);

    let response = lm.call(chat, Vec::new()).await.map_err(|e| {
        error!("DSRS LM call failed: {e:?}");
        SignatureCallError::Api(ApiError::InternalServerError)
    })?;

    let raw_response = response.output.content();
    let (output, _meta) = adapter
        .parse_response_typed::<AgentResponse>(&response.output)
        .map_err(|e| {
            error!("DSRS typed parse failed: {e:?}");
            SignatureCallError::Parse {
                raw_response,
                error_message: e.to_string(),
            }
        })?;

    let mut messages = output.messages;
    messages.retain(|m| !m.trim().is_empty());

    Ok(AgentResponseOutput {
        messages,
        tool_calls: output.tool_calls,
    })
}

pub async fn build_lm(
    state: Arc<AppState>,
    user: Arc<crate::models::users::User>,
    model: String,
    temperature: f32,
    max_tokens: u32,
) -> Result<Arc<LM>, ApiError> {
    let model_for_closure = model.clone();
    let completion_model = CustomCompletionModel::new(move |request: CompletionRequest| {
        let state = state.clone();
        let user = user.clone();
        let model = model_for_closure.clone();

        Box::pin(async move {
            let body = completion_request_to_openai_body(&model, &request)?;
            let headers = HeaderMap::new();
            let billing_context = BillingContext::new(AuthMethod::Jwt, model.to_string());
            let completion = get_chat_completion_response(
                &state,
                user.as_ref(),
                body,
                &headers,
                billing_context,
            )
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            if completion.metadata.is_streaming {
                return Err(CompletionError::ProviderError(
                    "Streaming response not supported".to_string(),
                ));
            }

            let mut rx = completion.stream;
            let response_json = match rx.recv().await {
                Some(CompletionChunk::FullResponse(response_json)) => response_json,
                _ => {
                    return Err(CompletionError::ProviderError(
                        "Missing completion response".to_string(),
                    ));
                }
            };

            let content = extract_assistant_content(&response_json).ok_or_else(|| {
                CompletionError::ResponseError("Missing assistant message content".to_string())
            })?;

            let usage = extract_usage(&response_json);

            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::text(content)),
                usage,
                raw_response: (),
            })
        })
    });

    let lm = LM::builder()
        .base_url("http://localhost".to_string())
        .model(model)
        .temperature(temperature)
        .max_tokens(max_tokens)
        .cache(false)
        .build()
        .await
        .map_err(|e| {
            error!("Failed to build DSRS LM: {e:?}");
            ApiError::InternalServerError
        })?;

    let lm = lm
        .with_client(LMClient::Custom(completion_model))
        .await
        .map_err(|e| {
            error!("Failed to set DSRS custom LM client: {e:?}");
            ApiError::InternalServerError
        })?;

    Ok(Arc::new(lm))
}

fn completion_request_to_openai_body(
    model: &str,
    request: &CompletionRequest,
) -> Result<Value, CompletionError> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(preamble) = &request.preamble {
        messages.push(json!({"role": "system", "content": preamble}));
    }

    for message in request.chat_history.iter() {
        let message_val = serde_json::to_value(message)?;
        let Some(role) = message_val.get("role").and_then(|v| v.as_str()) else {
            continue;
        };

        let Some(content_items) = message_val.get("content").and_then(|v| v.as_array()) else {
            continue;
        };

        let content = match role {
            "user" => extract_user_content_text(content_items),
            "assistant" => extract_assistant_content_text(content_items),
            _ => None,
        };

        let Some(content) = content.filter(|c| !c.trim().is_empty()) else {
            continue;
        };

        messages.push(json!({"role": role, "content": content}));
    }

    let mut body = json!({
        "model": model,
        "stream": false,
        "messages": messages,
    });

    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }

    Ok(body)
}

fn extract_user_content_text(content_items: &[Value]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for item in content_items {
        match item.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_string());
                }
            }
            Some("toolresult") => {
                if let Some(results) = item.get("content").and_then(|v| v.as_array()) {
                    for result in results {
                        if result.get("type").and_then(|v| v.as_str()) == Some("text") {
                            if let Some(text) = result.get("text").and_then(|v| v.as_str()) {
                                parts.push(text.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_assistant_content_text(content_items: &[Value]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for item in content_items {
        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
            parts.push(text.to_string());
        } else if let Some(reasoning) = item.get("reasoning").and_then(|v| v.as_array()) {
            let joined = reasoning
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !joined.is_empty() {
                parts.push(joined);
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_assistant_content(response_json: &Value) -> Option<String> {
    response_json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn extract_usage(response_json: &Value) -> dspy_rs::Usage {
    let prompt_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("total_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| prompt_tokens + completion_tokens);

    dspy_rs::Usage {
        input_tokens: prompt_tokens,
        output_tokens: completion_tokens,
        total_tokens,
    }
}
