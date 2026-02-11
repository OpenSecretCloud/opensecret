use async_trait::async_trait;
use secp256k1::SecretKey;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::{error, warn};
use uuid::Uuid;

use crate::encrypt::{decrypt_string, encrypt_with_key};
use crate::models::conversation_summaries::ConversationSummary;
use crate::models::memory_blocks::{MemoryBlock, NewMemoryBlock};
use crate::models::responses::Conversation;
use crate::models::users::User;
use crate::rag::{cosine_similarity, deserialize_f32_le, search_user_embeddings};
use crate::tokens::count_tokens;
use crate::web::openai_auth::AuthMethod;
use crate::web::responses::MessageContentConverter;
use crate::web::responses::{MessageContent, MessageContentPart};
use crate::{ApiError, AppState};

// ============================================================================
// Shared types
// ============================================================================

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        let msg = error.into();
        Self {
            success: false,
            output: msg.clone(),
            error: Some(msg),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> &str;
    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult;
}

/// Registry of available tools
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn generate_description(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available.".to_string();
        }

        let mut desc = String::from("Available tools (add to tool_calls array to use):\n\n");
        for tool in self.tools.values() {
            desc.push_str(&format!(
                "{}:\n  Description: {}\n  Args: {}\n\n",
                tool.name(),
                tool.description(),
                tool.args_schema()
            ));
        }
        desc
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Core memory tools
// ============================================================================

fn parse_line_arg(args: &HashMap<String, String>) -> i32 {
    args.get("line").and_then(|l| l.parse().ok()).unwrap_or(-1)
}

fn insert_at_line(value: &str, content: &str, line: i32) -> String {
    let lines: Vec<&str> = value.lines().collect();
    let mut new_lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();

    let insert_idx = if line < 0 {
        new_lines.len()
    } else {
        (line as usize).min(new_lines.len())
    };

    new_lines.insert(insert_idx, content.to_string());
    new_lines.join("\n")
}

async fn update_block_value(
    state: &Arc<AppState>,
    user_id: Uuid,
    user_key: &SecretKey,
    label: &str,
    new_value: &str,
    existing: &MemoryBlock,
) -> Result<(), ApiError> {
    if existing.read_only {
        return Err(ApiError::BadRequest);
    }

    if new_value.len() > existing.char_limit.max(0) as usize {
        return Err(ApiError::BadRequest);
    }

    let value_enc = encrypt_with_key(user_key, new_value.as_bytes()).await;

    let new_block = NewMemoryBlock {
        uuid: existing.uuid,
        user_id,
        label: label.to_string(),
        description: existing.description.clone(),
        value_enc,
        char_limit: existing.char_limit,
        read_only: existing.read_only,
        version: existing.version,
    };

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    new_block.insert_or_update(&mut conn).map_err(|e| {
        error!("Failed to update memory block '{}': {e:?}", label);
        ApiError::InternalServerError
    })?;

    Ok(())
}

pub struct MemoryReplaceTool {
    state: Arc<AppState>,
    user_id: Uuid,
    user_key: Arc<SecretKey>,
}

impl MemoryReplaceTool {
    pub fn new(state: Arc<AppState>, user_id: Uuid, user_key: Arc<SecretKey>) -> Self {
        Self {
            state,
            user_id,
            user_key,
        }
    }
}

#[async_trait]
impl Tool for MemoryReplaceTool {
    fn name(&self) -> &str {
        "memory_replace"
    }

    fn description(&self) -> &str {
        "Replace text in a memory block. Requires exact match of old text."
    }

    fn args_schema(&self) -> &str {
        r#"{"block": "block label (e.g., 'persona', 'human')", "old": "exact text to find", "new": "replacement text"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(block) = args.get("block") else {
            return ToolResult::error("'block' argument required");
        };
        let Some(old) = args.get("old") else {
            return ToolResult::error("'old' argument required");
        };
        let Some(new) = args.get("new") else {
            return ToolResult::error("'new' argument required");
        };

        let mut conn = match self.state.db.get_pool().get() {
            Ok(c) => c,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        let existing = match MemoryBlock::get_by_user_and_label(&mut conn, self.user_id, block) {
            Ok(Some(b)) => b,
            Ok(None) => return ToolResult::error(format!("Block '{}' not found", block)),
            Err(e) => {
                error!("Failed to load block '{}': {e:?}", block);
                return ToolResult::error("Failed to load memory block");
            }
        };

        let value = match decrypt_string(&self.user_key, Some(&existing.value_enc)) {
            Ok(Some(v)) => v,
            Ok(None) => String::new(),
            Err(e) => {
                error!("Failed to decrypt block '{}': {e:?}", block);
                return ToolResult::error("Failed to decrypt memory block");
            }
        };

        if !value.contains(old) {
            return ToolResult::error(format!(
                "Old content '{}' not found in memory block '{}'",
                old, block
            ));
        }

        let new_value = value.replace(old, new);
        if let Err(e) = update_block_value(
            &self.state,
            self.user_id,
            &self.user_key,
            block,
            &new_value,
            &existing,
        )
        .await
        {
            error!("Failed to update block '{}': {e:?}", block);
            return ToolResult::error("Failed to update memory block");
        }

        ToolResult::success(format!("Successfully replaced text in '{}' block.", block))
    }
}

pub struct MemoryAppendTool {
    state: Arc<AppState>,
    user_id: Uuid,
    user_key: Arc<SecretKey>,
}

impl MemoryAppendTool {
    pub fn new(state: Arc<AppState>, user_id: Uuid, user_key: Arc<SecretKey>) -> Self {
        Self {
            state,
            user_id,
            user_key,
        }
    }
}

#[async_trait]
impl Tool for MemoryAppendTool {
    fn name(&self) -> &str {
        "memory_append"
    }

    fn description(&self) -> &str {
        "Append text to the end of a memory block."
    }

    fn args_schema(&self) -> &str {
        r#"{"block": "block label (e.g., 'persona', 'human')", "content": "text to append"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(block) = args.get("block") else {
            return ToolResult::error("'block' argument required");
        };
        let Some(content) = args.get("content") else {
            return ToolResult::error("'content' argument required");
        };

        let mut conn = match self.state.db.get_pool().get() {
            Ok(c) => c,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        let existing = match MemoryBlock::get_by_user_and_label(&mut conn, self.user_id, block) {
            Ok(Some(b)) => b,
            Ok(None) => return ToolResult::error(format!("Block '{}' not found", block)),
            Err(e) => {
                error!("Failed to load block '{}': {e:?}", block);
                return ToolResult::error("Failed to load memory block");
            }
        };

        let value = match decrypt_string(&self.user_key, Some(&existing.value_enc)) {
            Ok(Some(v)) => v,
            Ok(None) => String::new(),
            Err(e) => {
                error!("Failed to decrypt block '{}': {e:?}", block);
                return ToolResult::error("Failed to decrypt memory block");
            }
        };

        let new_value = if value.is_empty() {
            content.to_string()
        } else {
            format!("{}\n{}", value, content)
        };

        if let Err(e) = update_block_value(
            &self.state,
            self.user_id,
            &self.user_key,
            block,
            &new_value,
            &existing,
        )
        .await
        {
            error!("Failed to update block '{}': {e:?}", block);
            return ToolResult::error("Failed to update memory block");
        }

        ToolResult::success(format!("Successfully appended to '{}' block.", block))
    }
}

pub struct MemoryInsertTool {
    state: Arc<AppState>,
    user_id: Uuid,
    user_key: Arc<SecretKey>,
}

impl MemoryInsertTool {
    pub fn new(state: Arc<AppState>, user_id: Uuid, user_key: Arc<SecretKey>) -> Self {
        Self {
            state,
            user_id,
            user_key,
        }
    }
}

#[async_trait]
impl Tool for MemoryInsertTool {
    fn name(&self) -> &str {
        "memory_insert"
    }

    fn description(&self) -> &str {
        "Insert text at a specific line in a memory block. Use line=-1 for end."
    }

    fn args_schema(&self) -> &str {
        r#"{"block": "block label", "content": "text to insert", "line": "line number (0-indexed, -1 for end)"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(block) = args.get("block") else {
            return ToolResult::error("'block' argument required");
        };
        let Some(content) = args.get("content") else {
            return ToolResult::error("'content' argument required");
        };
        let line = parse_line_arg(args);

        let mut conn = match self.state.db.get_pool().get() {
            Ok(c) => c,
            Err(_) => return ToolResult::error("Database connection error"),
        };

        let existing = match MemoryBlock::get_by_user_and_label(&mut conn, self.user_id, block) {
            Ok(Some(b)) => b,
            Ok(None) => return ToolResult::error(format!("Block '{}' not found", block)),
            Err(e) => {
                error!("Failed to load block '{}': {e:?}", block);
                return ToolResult::error("Failed to load memory block");
            }
        };

        let value = match decrypt_string(&self.user_key, Some(&existing.value_enc)) {
            Ok(Some(v)) => v,
            Ok(None) => String::new(),
            Err(e) => {
                error!("Failed to decrypt block '{}': {e:?}", block);
                return ToolResult::error("Failed to decrypt memory block");
            }
        };

        let new_value = insert_at_line(&value, content, line);

        if let Err(e) = update_block_value(
            &self.state,
            self.user_id,
            &self.user_key,
            block,
            &new_value,
            &existing,
        )
        .await
        {
            error!("Failed to update block '{}': {e:?}", block);
            return ToolResult::error("Failed to update memory block");
        }

        ToolResult::success(format!(
            "Successfully inserted text into '{}' block at line {}.",
            block,
            if line < 0 {
                "end".to_string()
            } else {
                line.to_string()
            }
        ))
    }
}

// ============================================================================
// Recall + summary search tool
// ============================================================================

pub struct ConversationSearchTool {
    state: Arc<AppState>,
    user: Arc<User>,
    user_key: Arc<SecretKey>,
    agent_conversation_id: i64,
}

impl ConversationSearchTool {
    pub fn new(
        state: Arc<AppState>,
        user: Arc<User>,
        user_key: Arc<SecretKey>,
        conversation_id: i64,
    ) -> Self {
        Self {
            state,
            user,
            user_key,
            agent_conversation_id: conversation_id,
        }
    }

    async fn search_summaries(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(ConversationSummary, f32, String)>, ApiError> {
        let (query_vec, _tok) = crate::web::get_embedding_vector(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            crate::rag::DEFAULT_EMBEDDING_MODEL,
            query,
            Some(crate::rag::DEFAULT_EMBEDDING_DIM),
        )
        .await?;

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        // Fetch latest summaries. Bound this to keep runtime predictable.
        let summaries = ConversationSummary::list_for_conversation(
            &mut conn,
            self.user.uuid,
            self.agent_conversation_id,
            200,
        )
        .map_err(|e| {
            error!("Failed to list conversation summaries: {e:?}");
            ApiError::InternalServerError
        })?;

        let mut scored: Vec<(ConversationSummary, f32, String)> = Vec::new();
        for s in summaries {
            let Some(embedding_enc) = &s.embedding_enc else {
                continue;
            };
            let embedding_bytes = crate::encrypt::decrypt_with_key(&self.user_key, embedding_enc)
                .map_err(|_| ApiError::InternalServerError)?;
            let embedding = deserialize_f32_le(&embedding_bytes)?;
            let score = cosine_similarity(&query_vec, &embedding)?;

            let content = decrypt_string(&self.user_key, Some(&s.content_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default();

            scored.push((s, score, content));
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);
        Ok(scored)
    }
}

#[async_trait]
impl Tool for ConversationSearchTool {
    fn name(&self) -> &str {
        "conversation_search"
    }

    fn description(&self) -> &str {
        "Search through past conversation history, including older summarized conversations. Returns matching messages and summaries with relevance scores."
    }

    fn args_schema(&self) -> &str {
        r#"{"query": "search query", "limit": "max results (default 5)", "conversation_id": "optional conversation UUID to scope message search"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(query) = args.get("query") else {
            return ToolResult::error("'query' argument required");
        };
        let limit: usize = args.get("limit").and_then(|l| l.parse().ok()).unwrap_or(5);

        let conversation_id_filter: Option<i64> = match args.get("conversation_id") {
            Some(raw) if !raw.trim().is_empty() => {
                let Ok(conversation_uuid) = raw.trim().parse::<Uuid>() else {
                    return ToolResult::error("invalid 'conversation_id' (expected UUID)");
                };

                let mut conn = match self.state.db.get_pool().get() {
                    Ok(c) => c,
                    Err(_) => return ToolResult::error("database connection error"),
                };

                match Conversation::get_by_uuid_and_user(
                    &mut conn,
                    conversation_uuid,
                    self.user.uuid,
                ) {
                    Ok(c) => Some(c.id),
                    Err(_) => return ToolResult::error("conversation not found"),
                }
            }
            _ => None,
        };

        let mut output = String::new();
        let mut total_results = 0usize;

        // Search recall messages (all conversations by default)
        let source_types = vec![crate::rag::SOURCE_TYPE_MESSAGE.to_string()];
        match search_user_embeddings(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            &self.user_key,
            query,
            limit,
            None,
            Some(&source_types),
            conversation_id_filter,
            None,
        )
        .await
        {
            Ok(results) => {
                if !results.is_empty() {
                    total_results += results.len();
                    output.push_str(&format!("=== Messages ({}) ===\n\n", results.len()));
                    for (i, r) in results.iter().enumerate() {
                        output.push_str(&format!("{}. {}\n\n", i + 1, r.content.trim()));
                    }
                }
            }
            Err(e) => {
                warn!("Message search failed: {e:?}");
            }
        }

        // Search summaries
        match self.search_summaries(query, limit).await {
            Ok(results) => {
                if !results.is_empty() {
                    total_results += results.len();
                    output.push_str(&format!(
                        "=== Conversation Summaries ({}) ===\n\n",
                        results.len()
                    ));
                    for (i, (summary, score, content)) in results.iter().enumerate() {
                        output.push_str(&format!(
                            "{}. [Summary of messages {}-{}] (relevance: {:.2})\n{}\n\n",
                            i + 1,
                            summary.from_created_at.format("%Y-%m-%d"),
                            summary.to_created_at.format("%Y-%m-%d"),
                            *score,
                            content.trim()
                        ));
                    }
                }
            }
            Err(e) => {
                warn!("Summary search failed: {e:?}");
            }
        }

        if total_results == 0 {
            return ToolResult::success("No matching messages or summaries found.".to_string());
        }

        ToolResult::success(output)
    }
}

// ============================================================================
// Archival tools
// ============================================================================

pub struct ArchivalInsertTool {
    state: Arc<AppState>,
    user: Arc<User>,
    user_key: Arc<SecretKey>,
}

impl ArchivalInsertTool {
    pub fn new(state: Arc<AppState>, user: Arc<User>, user_key: Arc<SecretKey>) -> Self {
        Self {
            state,
            user,
            user_key,
        }
    }
}

#[async_trait]
impl Tool for ArchivalInsertTool {
    fn name(&self) -> &str {
        "archival_insert"
    }

    fn description(&self) -> &str {
        "Store information in long-term archival memory for future recall. Good for important facts, preferences, and details you want to remember."
    }

    fn args_schema(&self) -> &str {
        r#"{"content": "text to store", "tags": "optional comma-separated tags"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(content) = args.get("content") else {
            return ToolResult::error("'content' argument required");
        };

        let tags = args.get("tags").map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        });

        let metadata = tags
            .as_ref()
            .filter(|t| !t.is_empty())
            .map(|t| json!({"tags": t}));
        let metadata_ref = metadata.as_ref();

        match crate::rag::insert_archival_embedding(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            &self.user_key,
            content,
            metadata_ref,
        )
        .await
        {
            Ok(inserted) => ToolResult::success(format!(
                "Successfully stored in archival memory (id: {}).",
                inserted.uuid
            )),
            Err(e) => {
                error!("archival_insert failed: {e:?}");
                ToolResult::error("Failed to store in archival memory")
            }
        }
    }
}

pub struct ArchivalSearchTool {
    state: Arc<AppState>,
    user: Arc<User>,
    user_key: Arc<SecretKey>,
}

impl ArchivalSearchTool {
    pub fn new(state: Arc<AppState>, user: Arc<User>, user_key: Arc<SecretKey>) -> Self {
        Self {
            state,
            user,
            user_key,
        }
    }
}

#[async_trait]
impl Tool for ArchivalSearchTool {
    fn name(&self) -> &str {
        "archival_search"
    }

    fn description(&self) -> &str {
        "Search long-term archival memory using semantic similarity. Returns most relevant stored memories."
    }

    fn args_schema(&self) -> &str {
        r#"{"query": "search query", "top_k": "max results (default 5)", "tags": "optional comma-separated tags to filter by"}"#
    }

    async fn execute(&self, args: &HashMap<String, String>) -> ToolResult {
        let Some(query) = args.get("query") else {
            return ToolResult::error("'query' argument required");
        };
        let top_k: usize = args.get("top_k").and_then(|k| k.parse().ok()).unwrap_or(5);

        let tags = args
            .get("tags")
            .map(|t| {
                t.split(',')
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|t| !t.is_empty());

        let source_types = vec![crate::rag::SOURCE_TYPE_ARCHIVAL.to_string()];
        match search_user_embeddings(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            &self.user_key,
            query,
            top_k,
            None,
            Some(&source_types),
            None,
            tags.as_deref(),
        )
        .await
        {
            Ok(results) => {
                if results.is_empty() {
                    return ToolResult::success("No matching memories found.".to_string());
                }

                let mut output = format!("Found {} matching memories:\n\n", results.len());
                for (i, r) in results.iter().enumerate() {
                    output.push_str(&format!("{}. {}\n\n", i + 1, r.content.trim()));
                }
                ToolResult::success(output)
            }
            Err(e) => {
                error!("archival_search failed: {e:?}");
                ToolResult::error("Archival search failed")
            }
        }
    }
}

// ============================================================================
// Done tool
// ============================================================================

pub struct DoneTool;

#[async_trait]
impl Tool for DoneTool {
    fn name(&self) -> &str {
        "done"
    }

    fn description(&self) -> &str {
        "No-op signal. Use ONLY when messages is [] AND no other tools needed. Indicates nothing to do this turn."
    }

    fn args_schema(&self) -> &str {
        r#"{}"#
    }

    async fn execute(&self, _args: &HashMap<String, String>) -> ToolResult {
        ToolResult::success("Done.".to_string())
    }
}

// ============================================================================
// Formatting helpers
// ============================================================================

pub fn normalize_user_message_content(text: &str) -> String {
    // Store user messages in the DB as MessageContent JSON, like Responses API.
    // Use input_text part for compatibility with Conversations API.
    let content = MessageContent::Parts(vec![MessageContentPart::InputText {
        text: text.to_string(),
    }]);
    serde_json::to_string(&content).unwrap_or_else(|_| format!("\"{}\"", text))
}

pub fn count_user_message_tokens(content_json: &str) -> i32 {
    let parsed: Result<MessageContent, _> = serde_json::from_str(content_json);
    match parsed {
        Ok(content) => count_tokens(&MessageContentConverter::extract_text_for_token_counting(
            &content,
        )) as i32,
        Err(_) => count_tokens(content_json) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_at_line_end() {
        let v = "a\nb";
        assert_eq!(insert_at_line(v, "c", -1), "a\nb\nc");
    }

    #[test]
    fn insert_at_line_middle() {
        let v = "a\nc";
        assert_eq!(insert_at_line(v, "b", 1), "a\nb\nc");
    }
}
