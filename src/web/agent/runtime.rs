use secp256k1::SecretKey;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, error, warn};
use uuid::Uuid;

use diesel::prelude::*;

use crate::encrypt::{decrypt_string, encrypt_with_key};
use crate::models::agent_config::{AgentConfig, NewAgentConfig};
use crate::models::conversation_summaries::{ConversationSummary, NewConversationSummary};
use crate::models::memory_blocks::{MemoryBlock, NewMemoryBlock, DEFAULT_BLOCK_CHAR_LIMIT};
use crate::models::responses::{
    AssistantMessage, Conversation, NewAssistantMessage, NewConversation, NewToolCall,
    NewToolOutput, NewUserMessage, RawThreadMessage, RawThreadMessageMetadata, ToolCall,
    ToolOutput, UserMessage,
};
use crate::models::schema::{memory_blocks, user_embeddings};
use crate::rag::{
    insert_message_embedding, serialize_f32_le, SOURCE_TYPE_ARCHIVAL, SOURCE_TYPE_MESSAGE,
};
use crate::tokens::count_tokens;
use crate::web::openai_auth::AuthMethod;
use crate::web::responses::{MessageContent, MessageContentConverter};
use crate::{ApiError, AppState};

use super::compaction::CompactionManager;
use super::signatures::{
    build_lm, call_agent_response_with_retry_and_correction, AgentResponseInput, AgentToolCall,
    AGENT_INSTRUCTION,
};
use super::tools::{
    ArchivalInsertTool, ArchivalSearchTool, ConversationSearchTool, DoneTool, MemoryAppendTool,
    MemoryInsertTool, MemoryReplaceTool, ToolRegistry, ToolResult,
};

// Mirrors Sage defaults
const DEFAULT_PERSONA_DESCRIPTION: &str = "The persona block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.";
const DEFAULT_HUMAN_DESCRIPTION: &str = "The human block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.";
const DEFAULT_PERSONA_VALUE: &str = "I am Sage, a helpful AI companion. I maintain long-term memory across our conversations and strive to be friendly, concise, and genuinely helpful.";
const DEFAULT_CONTEXT_WINDOW: i32 = 256_000;
const DEFAULT_COMPACTION_THRESHOLD: f32 = 0.80;
const MIN_MESSAGES_IN_CONTEXT: usize = 20;

#[derive(Clone, Debug, Default)]
struct AgentContext {
    current_time: String,
    persona_block: String,
    human_block: String,
    memory_metadata: String,
    previous_context_summary: String,
    recent_conversation: String,
    is_first_time_user: bool,
}

#[derive(Clone, Debug)]
pub struct ExecutedTool {
    pub tool_call: AgentToolCall,
    pub result: ToolResult,
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub messages: Vec<String>,
    #[allow(dead_code)]
    pub tool_calls: Vec<AgentToolCall>,
    pub executed_tools: Vec<ExecutedTool>,
    pub done: bool,
}

pub struct AgentRuntime {
    state: Arc<AppState>,
    user: Arc<crate::models::users::User>,
    user_key: Arc<SecretKey>,
    agent_config: AgentConfig,
    conversation: Conversation,
    system_prompt: String,
    lm: Arc<dspy_rs::LM>,
    tools: ToolRegistry,
    available_tools: String,
    compaction: CompactionManager,
    current_tool_results: Vec<String>,
    previous_step_summary: Option<(Vec<String>, Vec<String>)>,
    max_steps: usize,
}

impl AgentRuntime {
    pub async fn new(
        state: Arc<AppState>,
        user: crate::models::users::User,
        user_key: SecretKey,
    ) -> Result<Self, ApiError> {
        let user = Arc::new(user);
        let user_key = Arc::new(user_key);

        let mut conn = state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        // Load or create config
        let agent_config = match AgentConfig::get_by_user_id(&mut conn, user.uuid).map_err(|e| {
            error!("Failed to load agent config: {e:?}");
            ApiError::InternalServerError
        })? {
            Some(cfg) => cfg,
            None => {
                let new_cfg = NewAgentConfig {
                    uuid: Uuid::new_v4(),
                    user_id: user.uuid,
                    conversation_id: None,
                    enabled: true,
                    model: "kimi-k2".to_string(),
                    max_context_tokens: DEFAULT_CONTEXT_WINDOW,
                    compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
                    system_prompt_enc: None,
                    preferences_enc: None,
                };

                new_cfg.insert_or_update(&mut conn).map_err(|e| {
                    error!("Failed to create agent config: {e:?}");
                    ApiError::InternalServerError
                })?
            }
        };

        if !agent_config.enabled {
            return Err(ApiError::Unauthorized);
        }

        // Ensure conversation exists
        let conversation = if let Some(conversation_id) = agent_config.conversation_id {
            Conversation::get_by_id_and_user(&mut conn, conversation_id, user.uuid).map_err(
                |e| {
                    error!("Failed to load agent conversation: {e:?}");
                    ApiError::InternalServerError
                },
            )?
        } else {
            let metadata = json!({"type":"agent_main"});
            let metadata_enc = encrypt_with_key(&user_key, metadata.to_string().as_bytes()).await;

            let new_conversation = NewConversation {
                uuid: Uuid::new_v4(),
                user_id: user.uuid,
                metadata_enc: Some(metadata_enc),
            };
            let conversation = new_conversation.insert(&mut conn).map_err(|e| {
                error!("Failed to create agent conversation: {e:?}");
                ApiError::InternalServerError
            })?;

            let updated_cfg = NewAgentConfig {
                uuid: agent_config.uuid,
                user_id: agent_config.user_id,
                conversation_id: Some(conversation.id),
                enabled: agent_config.enabled,
                model: agent_config.model.clone(),
                max_context_tokens: agent_config.max_context_tokens,
                compaction_threshold: agent_config.compaction_threshold,
                system_prompt_enc: agent_config.system_prompt_enc.clone(),
                preferences_enc: agent_config.preferences_enc.clone(),
            };
            let _ = updated_cfg.insert_or_update(&mut conn).map_err(|e| {
                error!("Failed to update agent config with conversation_id: {e:?}");
                ApiError::InternalServerError
            })?;

            conversation
        };

        // Ensure default memory blocks exist
        Self::ensure_default_blocks(&state, &mut conn, &user_key, user.uuid).await?;

        // System prompt override
        let system_prompt = match decrypt_string(&user_key, agent_config.system_prompt_enc.as_ref())
            .map_err(|e| {
                error!("Failed to decrypt system_prompt_enc: {e:?}");
                ApiError::InternalServerError
            })? {
            Some(s) if !s.trim().is_empty() => s,
            _ => AGENT_INSTRUCTION.to_string(),
        };

        // Tool registry
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(MemoryReplaceTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(MemoryAppendTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(MemoryInsertTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(ArchivalInsertTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
        )));
        tools.register(Arc::new(ArchivalSearchTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
        )));
        tools.register(Arc::new(ConversationSearchTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
            conversation.id,
        )));
        tools.register(Arc::new(DoneTool));

        let available_tools = tools.generate_description();

        let lm = build_lm(
            state.clone(),
            user.clone(),
            agent_config.model.clone(),
            0.7,
            32768,
        )
        .await?;

        Ok(Self {
            state,
            user,
            user_key,
            agent_config,
            conversation,
            system_prompt,
            lm,
            tools,
            available_tools,
            compaction: CompactionManager::new(),
            current_tool_results: Vec::new(),
            previous_step_summary: None,
            max_steps: 10,
        })
    }

    pub(crate) async fn ensure_default_blocks(
        _state: &Arc<AppState>,
        conn: &mut diesel::PgConnection,
        user_key: &SecretKey,
        user_id: Uuid,
    ) -> Result<(), ApiError> {
        let persona =
            MemoryBlock::get_by_user_and_label(conn, user_id, "persona").map_err(|e| {
                error!("Failed to load persona block: {e:?}");
                ApiError::InternalServerError
            })?;

        if persona.is_none() {
            let value_enc = encrypt_with_key(user_key, DEFAULT_PERSONA_VALUE.as_bytes()).await;
            let block = NewMemoryBlock {
                uuid: Uuid::new_v4(),
                user_id,
                label: "persona".to_string(),
                description: Some(DEFAULT_PERSONA_DESCRIPTION.to_string()),
                value_enc,
                char_limit: DEFAULT_BLOCK_CHAR_LIMIT,
                read_only: false,
                version: 1,
            };
            block.insert_or_update(conn).map_err(|e| {
                error!("Failed to create persona block: {e:?}");
                ApiError::InternalServerError
            })?;
        }

        let human = MemoryBlock::get_by_user_and_label(conn, user_id, "human").map_err(|e| {
            error!("Failed to load human block: {e:?}");
            ApiError::InternalServerError
        })?;

        if human.is_none() {
            let value_enc = encrypt_with_key(user_key, "".as_bytes()).await;
            let block = NewMemoryBlock {
                uuid: Uuid::new_v4(),
                user_id,
                label: "human".to_string(),
                description: Some(DEFAULT_HUMAN_DESCRIPTION.to_string()),
                value_enc,
                char_limit: DEFAULT_BLOCK_CHAR_LIMIT,
                read_only: false,
                version: 1,
            };
            block.insert_or_update(conn).map_err(|e| {
                error!("Failed to create human block: {e:?}");
                ApiError::InternalServerError
            })?;
        }

        Ok(())
    }

    pub fn clear_tool_results(&mut self) {
        self.current_tool_results.clear();
        self.previous_step_summary = None;
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Prepare the runtime for a new message: validate, persist user message, compact if needed.
    /// Call this once before driving the step loop.
    pub async fn prepare(&mut self, user_message: &str) -> Result<(), ApiError> {
        let trimmed = user_message.trim();
        if trimmed.is_empty() {
            return Err(ApiError::BadRequest);
        }

        self.clear_tool_results();
        self.insert_user_message(trimmed).await?;
        self.maybe_compact().await?;
        Ok(())
    }

    /// Execute a single step of the agent loop.
    /// The caller (chat handler) drives the loop, persists messages, and emits SSE events.
    pub async fn step(
        &mut self,
        user_message: &str,
        is_first_step: bool,
    ) -> Result<StepResult, ApiError> {
        if is_first_step {
            self.current_tool_results.clear();
        }

        debug!("Agent step (first={})", is_first_step);

        let ctx = self.build_context().await?;

        let input_content = if is_first_step {
            user_message.to_string()
        } else {
            let tool_results: Vec<&str> = self
                .current_tool_results
                .iter()
                .map(|s| s.as_str())
                .collect();

            if tool_results.is_empty() {
                user_message.to_string()
            } else {
                let already_sent = if let Some((sent_messages, tool_names)) =
                    &self.previous_step_summary
                {
                    let tools_str = tool_names.join(", ");
                    let msgs_preview = if sent_messages.is_empty() {
                        String::new()
                    } else {
                        let msgs_text = sent_messages
                            .iter()
                            .enumerate()
                            .map(|(i, m)| format!("  {}. \"{}\"", i + 1, m))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("\nMessages you already sent to user:\n{}\n", msgs_text)
                    };

                    format!(
                        "[You already sent {} message(s) and called {} this turn.{}Tools have executed:]\n\n",
                        sent_messages.len(),
                        tools_str,
                        msgs_preview
                    )
                } else {
                    String::new()
                };

                let tool_result_instructions = r#"

=== TOOL RESULT PROCESSING MODE ===
This is a CONTINUATION of your previous turn, NOT a new conversation.
Your previous messages are already visible to the user in recent_conversation.

RULES:
1. SILENCE IS DEFAULT - You do NOT need to acknowledge the tool result
2. DO NOT say: "I see the results", "Let me analyze", "Based on what I found", "Here's what the tool returned"
3. DO NOT repeat or rephrase what you already said
4. If the tool was for YOUR benefit (memory ops, archival), call 'done' immediately
5. Only send messages if you have GENUINELY NEW information the user hasn't seen

SELF-CHECK: Before ANY message, ask: "Is this new info the user hasn't seen?" If no → call 'done'"#;

                let result = if tool_results.len() == 1 {
                    format!(
                        "{}=== TOOL RESULT ===\n{}\n=== END TOOL RESULT ==={}",
                        already_sent, tool_results[0], tool_result_instructions
                    )
                } else {
                    let results_text = tool_results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| format!("--- Tool {} ---\n{}", i + 1, r))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    format!(
                        "{}=== TOOL RESULTS ({} tools) ===\n{}\n=== END TOOL RESULTS ==={}",
                        already_sent,
                        tool_results.len(),
                        results_text,
                        tool_result_instructions
                    )
                };

                self.current_tool_results.clear();

                result
            }
        };

        let input = AgentResponseInput {
            input: input_content.clone(),
            current_time: ctx.current_time,
            persona_block: ctx.persona_block,
            human_block: ctx.human_block,
            memory_metadata: ctx.memory_metadata,
            previous_context_summary: ctx.previous_context_summary,
            recent_conversation: ctx.recent_conversation,
            available_tools: self.available_tools.clone(),
            is_first_time_user: ctx.is_first_time_user,
        };

        let response = call_agent_response_with_retry_and_correction(
            &self.lm,
            &self.system_prompt,
            &input,
            &input_content,
            &self.available_tools,
        )
        .await?;

        // Unwrap nested JSON arrays (Sage compatibility)
        let messages: Vec<String> = response
            .messages
            .iter()
            .flat_map(|m| {
                let trimmed = m.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    if let Ok(inner) = serde_json::from_str::<Vec<String>>(trimmed) {
                        return inner;
                    }
                }
                vec![m.clone()]
            })
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .collect();

        // Execute tools and inject results for next step.
        // Persistence is handled by the caller (chat handler) to match Sage's
        // "send first, store synchronously, embed async" pattern.
        let mut executed_tools = Vec::new();
        for tool_call in &response.tool_calls {
            let result = if tool_call.name == "done" {
                ToolResult::success("Done".to_string())
            } else if let Some(tool) = self.tools.get(&tool_call.name) {
                tool.execute(&tool_call.args).await
            } else {
                ToolResult::error(format!("Unknown tool: {}", tool_call.name))
            };

            self.inject_tool_result(tool_call, &result);

            if tool_call.name != "done" {
                executed_tools.push(ExecutedTool {
                    tool_call: tool_call.clone(),
                    result,
                });
            }
        }

        let done = response.tool_calls.is_empty()
            || (response.tool_calls.len() == 1 && response.tool_calls[0].name == "done");

        if !messages.is_empty() || !response.tool_calls.is_empty() {
            let tool_names: Vec<String> = response
                .tool_calls
                .iter()
                .map(|tc| tc.name.clone())
                .collect();
            self.previous_step_summary = Some((messages.clone(), tool_names));
        }

        Ok(StepResult {
            messages,
            tool_calls: response.tool_calls,
            executed_tools,
            done,
        })
    }

    fn inject_tool_result(&mut self, tool_call: &AgentToolCall, result: &ToolResult) {
        let args_str = if tool_call.args.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = tool_call
                .args
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("\nArgs: {}", pairs.join(", "))
        };

        let result_text = format!(
            "[Tool Result: {}]{}\nStatus: {}\nOutput: {}",
            tool_call.name,
            args_str,
            if result.success { "OK" } else { "ERROR" },
            if result.success {
                result.output.as_str()
            } else {
                result.error.as_deref().unwrap_or("Unknown error")
            }
        );

        self.current_tool_results.push(result_text);
    }

    async fn build_context(&self) -> Result<AgentContext, ApiError> {
        let mut ctx = AgentContext::default();

        let now = chrono::Utc::now();
        ctx.current_time = format!("{} UTC", now.format("%m/%d/%Y %H:%M:%S (%A)"));

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let persona = MemoryBlock::get_by_user_and_label(&mut conn, self.user.uuid, "persona")
            .map_err(|_| ApiError::InternalServerError)?;
        let human = MemoryBlock::get_by_user_and_label(&mut conn, self.user.uuid, "human")
            .map_err(|_| ApiError::InternalServerError)?;

        ctx.persona_block = decrypt_string(&self.user_key, persona.as_ref().map(|b| &b.value_enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_default();
        ctx.human_block = decrypt_string(&self.user_key, human.as_ref().map(|b| &b.value_enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_default();

        // Memory metadata
        let last_modified = memory_blocks::table
            .filter(memory_blocks::user_id.eq(self.user.uuid))
            .select(diesel::dsl::max(memory_blocks::updated_at))
            .first::<Option<chrono::DateTime<chrono::Utc>>>(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let recall_count: i64 = user_embeddings::table
            .filter(user_embeddings::user_id.eq(self.user.uuid))
            .filter(user_embeddings::source_type.eq(SOURCE_TYPE_MESSAGE))
            .filter(user_embeddings::conversation_id.eq(Some(self.conversation.id)))
            .count()
            .get_result(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let archival_count: i64 = user_embeddings::table
            .filter(user_embeddings::user_id.eq(self.user.uuid))
            .filter(user_embeddings::source_type.eq(SOURCE_TYPE_ARCHIVAL))
            .count()
            .get_result(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let mut metadata = String::new();
        if let Some(ts) = last_modified {
            metadata.push_str(&format!(
                "- Memory blocks last modified: {} UTC\n",
                ts.format("%Y-%m-%d %H:%M:%S")
            ));
        }
        metadata.push_str(&format!(
            "- {} messages in recall memory (use conversation_search to access)\n",
            recall_count
        ));
        metadata.push_str(&format!(
            "- {} passages in archival memory (use archival_search to access)",
            archival_count
        ));
        ctx.memory_metadata = metadata;

        // Conversation summary + recent messages
        let summary = ConversationSummary::get_latest_for_conversation(
            &mut conn,
            self.user.uuid,
            self.conversation.id,
        )
        .map_err(|_| ApiError::InternalServerError)?;

        if let Some(s) = &summary {
            ctx.previous_context_summary = decrypt_string(&self.user_key, Some(&s.content_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default();
        }

        let mut messages = RawThreadMessage::get_conversation_context(
            &mut conn,
            self.conversation.id,
            10_000,
            None,
            "desc",
        )
        .map_err(|e| {
            error!("Failed to load conversation context: {e:?}");
            ApiError::InternalServerError
        })?;
        messages.reverse();

        if let Some(s) = &summary {
            messages.retain(|m| m.created_at > s.to_created_at);

            if messages.len() < MIN_MESSAGES_IN_CONTEXT {
                // Ensure at least a small slice of recent conversation
                let mut recent = RawThreadMessage::get_conversation_context(
                    &mut conn,
                    self.conversation.id,
                    MIN_MESSAGES_IN_CONTEXT as i64,
                    None,
                    "desc",
                )
                .map_err(|_| ApiError::InternalServerError)?;
                recent.reverse();
                messages = recent;
            }
        }

        // First-time user heuristic (matches Sage)
        let has_summary = summary.is_some();
        if messages.len() <= 1 && !has_summary {
            ctx.is_first_time_user = true;
        }

        // Render conversation history
        let mut conversation = String::new();
        for msg in &messages {
            if msg.message_type == "reasoning" {
                continue;
            }

            let role = match msg.message_type.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                "tool_call" | "tool_output" => "tool",
                other => other,
            };

            let timestamp = format!("{} UTC", msg.created_at.format("%m/%d/%Y %H:%M:%S"));
            let content = self.render_raw_message(msg)?;
            conversation.push_str(&format!("[{} @ {}]: {}\n", role, timestamp, content));
        }

        ctx.recent_conversation = if conversation.trim().is_empty() {
            "No previous conversation.".to_string()
        } else {
            conversation
        };

        Ok(ctx)
    }

    fn render_raw_message(&self, msg: &RawThreadMessage) -> Result<String, ApiError> {
        let Some(enc) = msg.content_enc.as_ref() else {
            return Ok(String::new());
        };

        let raw = decrypt_string(&self.user_key, Some(enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_default();

        let content = if msg.message_type == "user" {
            let parsed: Result<MessageContent, _> = serde_json::from_str(&raw);
            match parsed {
                Ok(c) => MessageContentConverter::extract_text_for_token_counting(&c),
                Err(_) => raw,
            }
        } else if msg.message_type == "tool_call" {
            match msg.tool_name.as_deref() {
                Some(name) => {
                    if raw.trim().is_empty() {
                        format!("CALL {}", name)
                    } else {
                        format!("CALL {} args={}", name, raw)
                    }
                }
                None => raw,
            }
        } else if msg.message_type == "tool_output" {
            match msg.tool_name.as_deref() {
                Some(name) => format!("OUTPUT {}: {}", name, raw),
                None => raw,
            }
        } else {
            raw
        };

        if (msg.message_type == "tool_call" || msg.message_type == "tool_output")
            && content.len() > 2000
        {
            let mut end = 2000;
            while !content.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            return Ok(format!("{}...", &content[..end]));
        }

        Ok(content)
    }

    pub async fn insert_user_message(&self, text: &str) -> Result<UserMessage, ApiError> {
        let content = super::tools::normalize_user_message_content(text);
        let prompt_tokens = super::tools::count_user_message_tokens(&content);
        let content_enc = encrypt_with_key(&self.user_key, content.as_bytes()).await;

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let msg = NewUserMessage {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            content_enc,
            prompt_tokens,
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert user message: {e:?}");
            ApiError::InternalServerError
        })?;

        // Mirror Sage: store message synchronously, update embedding in background.
        let state = self.state.clone();
        let user = self.user.clone();
        let user_key = self.user_key.clone();
        let text = text.to_string();
        let conversation_id = self.conversation.id;
        let user_message_id = Some(msg.id);

        tokio::spawn(async move {
            let _ = insert_message_embedding(
                &state,
                user.as_ref(),
                AuthMethod::Jwt,
                user_key.as_ref(),
                &text,
                conversation_id,
                user_message_id,
                None,
            )
            .await;
        });

        Ok(msg)
    }

    pub async fn insert_assistant_message(&self, text: &str) -> Result<AssistantMessage, ApiError> {
        let completion_tokens = count_tokens(text) as i32;
        let content_enc = Some(encrypt_with_key(&self.user_key, text.as_bytes()).await);

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let msg = NewAssistantMessage {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            content_enc,
            completion_tokens,
            status: "completed".to_string(),
            finish_reason: None,
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert assistant message: {e:?}");
            ApiError::InternalServerError
        })?;

        // Mirror Sage: store message synchronously, update embedding in background.
        let state = self.state.clone();
        let user = self.user.clone();
        let user_key = self.user_key.clone();
        let text = text.to_string();
        let conversation_id = self.conversation.id;
        let assistant_message_id = Some(msg.id);

        tokio::spawn(async move {
            let _ = insert_message_embedding(
                &state,
                user.as_ref(),
                AuthMethod::Jwt,
                user_key.as_ref(),
                &text,
                conversation_id,
                None,
                assistant_message_id,
            )
            .await;
        });

        Ok(msg)
    }

    pub async fn insert_tool_call_and_output(
        &self,
        tool_call: &AgentToolCall,
        result: &ToolResult,
    ) -> Result<(ToolCall, ToolOutput), ApiError> {
        let args_json = serde_json::to_string(&tool_call.args).unwrap_or_else(|_| "{}".to_string());
        let arguments_enc = Some(encrypt_with_key(&self.user_key, args_json.as_bytes()).await);
        let argument_tokens = count_tokens(&args_json) as i32;

        let output_text = if result.success {
            result.output.clone()
        } else {
            result
                .error
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string())
        };
        let output_tokens = count_tokens(&output_text) as i32;
        let output_enc = encrypt_with_key(&self.user_key, output_text.as_bytes()).await;

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let call = NewToolCall {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            name: tool_call.name.clone(),
            arguments_enc,
            argument_tokens,
            status: "completed".to_string(),
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert tool call: {e:?}");
            ApiError::InternalServerError
        })?;

        let out = NewToolOutput {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            tool_call_fk: call.id,
            output_enc,
            output_tokens,
            status: "completed".to_string(),
            error: result.error.clone(),
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert tool output: {e:?}");
            ApiError::InternalServerError
        })?;

        Ok((call, out))
    }

    async fn maybe_compact(&self) -> Result<(), ApiError> {
        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let latest_summary = ConversationSummary::get_latest_for_conversation(
            &mut conn,
            self.user.uuid,
            self.conversation.id,
        )
        .map_err(|_| ApiError::InternalServerError)?;

        let summary_tokens = latest_summary
            .as_ref()
            .map(|s| s.content_tokens)
            .unwrap_or(0);
        let summary_text = if let Some(s) = &latest_summary {
            decrypt_string(&self.user_key, Some(&s.content_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default()
        } else {
            String::new()
        };

        let mut metadata = RawThreadMessageMetadata::get_conversation_context_metadata(
            &mut conn,
            self.conversation.id,
        )
        .map_err(|e| {
            error!("Failed to load context metadata: {e:?}");
            ApiError::InternalServerError
        })?;

        if let Some(s) = &latest_summary {
            metadata.retain(|m| m.created_at > s.to_created_at);
        }

        metadata.retain(|m| m.message_type != "reasoning");

        let current_tokens: i32 = summary_tokens
            + metadata
                .iter()
                .map(|m| m.token_count.unwrap_or(0))
                .sum::<i32>();

        if !self.compaction.should_compact(
            current_tokens as usize,
            self.agent_config.max_context_tokens as usize,
            self.agent_config.compaction_threshold,
        ) {
            return Ok(());
        }

        if metadata.len() <= MIN_MESSAGES_IN_CONTEXT {
            return Ok(());
        }

        // Summarize the oldest half, keep at least MIN_MESSAGES_IN_CONTEXT recent messages
        let keep_count = (metadata.len() / 2).max(MIN_MESSAGES_IN_CONTEXT);
        let to_summarize_count = metadata.len().saturating_sub(keep_count);
        if to_summarize_count == 0 {
            return Ok(());
        }

        let to_summarize: Vec<(String, i64)> = metadata
            .iter()
            .take(to_summarize_count)
            .map(|m| (m.message_type.clone(), m.id))
            .collect();

        let raw_messages =
            RawThreadMessage::get_messages_by_ids(&mut conn, self.conversation.id, &to_summarize)
                .map_err(|e| {
                error!("Failed to load messages for compaction: {e:?}");
                ApiError::InternalServerError
            })?;

        if raw_messages.is_empty() {
            return Ok(());
        }

        let mut formatted: Vec<String> = Vec::new();
        for m in &raw_messages {
            if m.message_type == "reasoning" {
                continue;
            }
            let role = match m.message_type.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                _ => "tool",
            };
            let content = self.render_raw_message(m)?;
            formatted.push(format!("[{}]: {}", role, content));
        }

        let new_messages = formatted.join("\n---\n");

        let summary = self
            .compaction
            .summarize(&self.lm, &summary_text, &new_messages)
            .await?;

        let content_tokens = count_tokens(&summary) as i32;
        let content_enc = encrypt_with_key(&self.user_key, summary.as_bytes()).await;

        // Embed summary for conversation_search over summaries
        let embedding = crate::web::get_embedding_vector(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            crate::rag::DEFAULT_EMBEDDING_MODEL,
            &summary,
            Some(crate::rag::DEFAULT_EMBEDDING_DIM),
        )
        .await;

        let embedding_enc = match embedding {
            Ok((vec, _tok)) => {
                let bytes = serialize_f32_le(&vec);
                Some(encrypt_with_key(&self.user_key, &bytes).await)
            }
            Err(e) => {
                warn!("Failed to embed summary for conversation_search: {e:?}");
                None
            }
        };

        let from_created_at = raw_messages.first().unwrap().created_at;
        let to_created_at = raw_messages.last().unwrap().created_at;
        let previous_summary_id = latest_summary.as_ref().map(|s| s.id);

        let new_summary = NewConversationSummary {
            uuid: Uuid::new_v4(),
            user_id: self.user.uuid,
            conversation_id: self.conversation.id,
            from_created_at,
            to_created_at,
            message_count: raw_messages.len() as i32,
            content_enc,
            content_tokens,
            embedding_enc,
            previous_summary_id,
        };

        let _ = new_summary.insert(&mut conn).map_err(|e| {
            error!("Failed to insert conversation summary: {e:?}");
            ApiError::InternalServerError
        })?;

        Ok(())
    }
}
