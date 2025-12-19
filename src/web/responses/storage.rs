//! Storage task components for accumulating and persisting streaming responses

use crate::{
    encrypt::encrypt_with_key,
    models::responses::ResponseStatus,
    tokens::count_tokens,
    web::responses::constants::{FINISH_REASON_CANCELLED, STATUS_COMPLETED, STATUS_INCOMPLETE},
    DBConnection,
};
use chrono::Utc;
use secp256k1::SecretKey;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, trace, warn};
use uuid::Uuid;

use super::handlers::StorageMessage;

/// Accumulates streaming content and metadata
pub(crate) struct ContentAccumulator {
    content: String,
    reasoning_content: String,
    reasoning_item_id: Option<Uuid>,
    completion_tokens: i32,
}

impl ContentAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::with_capacity(4096),
            reasoning_content: String::with_capacity(4096),
            reasoning_item_id: None,
            completion_tokens: 0,
        }
    }

    /// Get the reasoning item ID if one was created
    pub fn reasoning_item_id(&self) -> Option<Uuid> {
        self.reasoning_item_id
    }

    /// Get accumulated reasoning content
    pub fn reasoning_content(&self) -> &str {
        &self.reasoning_content
    }

    /// Handle a storage message and return the accumulator state
    pub fn handle_message(&mut self, msg: StorageMessage) -> AccumulatorState {
        match msg {
            StorageMessage::ContentDelta(delta) => {
                trace!("Storage: received content delta: {} chars", delta.len());
                self.content.push_str(&delta);
                AccumulatorState::Continue
            }
            StorageMessage::ReasoningDelta { item_id, delta } => {
                trace!("Storage: received reasoning delta: {} chars", delta.len());
                self.reasoning_content.push_str(&delta);

                // Use the item_id from the message - same UUID as SSE events
                if self.reasoning_item_id.is_none() {
                    self.reasoning_item_id = Some(item_id);
                    return AccumulatorState::CreateReasoningItem { item_id };
                }

                AccumulatorState::Continue
            }
            StorageMessage::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                debug!(
                    "Storage: received usage - prompt_tokens={}, completion_tokens={}",
                    prompt_tokens, completion_tokens
                );
                // Note: prompt_tokens already tracked when user message was created in DB
                // Only store completion_tokens for the assistant message
                self.completion_tokens = completion_tokens;
                AccumulatorState::Continue
            }
            StorageMessage::Done {
                finish_reason,
                message_id,
            } => {
                debug!(
                    "Storage: received Done signal with finish_reason={}, message_id={}",
                    finish_reason, message_id
                );
                AccumulatorState::Complete(CompleteData {
                    content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                    finish_reason,
                    message_id,
                    reasoning_content: self.reasoning_content.clone(),
                    reasoning_item_id: self.reasoning_item_id,
                })
            }
            StorageMessage::Cancelled => {
                debug!("Storage: received cancellation signal");
                AccumulatorState::Cancelled(PartialData {
                    content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                    reasoning_content: self.reasoning_content.clone(),
                    reasoning_item_id: self.reasoning_item_id,
                })
            }
            StorageMessage::Error(e) => {
                error!("Storage: received error: {}", e);
                AccumulatorState::Failed(FailureData {
                    error: e,
                    partial_content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                    reasoning_content: self.reasoning_content.clone(),
                    reasoning_item_id: self.reasoning_item_id,
                })
            }
            StorageMessage::ToolCall {
                tool_call_id,
                name,
                arguments,
            } => {
                trace!(
                    "Storage: received tool_call - id={}, name={}",
                    tool_call_id,
                    name
                );
                // Signal immediate persistence
                AccumulatorState::PersistToolCall {
                    tool_call_id,
                    name,
                    arguments,
                }
            }
            StorageMessage::ToolOutput {
                tool_output_id,
                tool_call_id,
                output,
            } => {
                trace!(
                    "Storage: received tool_output - id={}, tool_call_id={}, output_len={}",
                    tool_output_id,
                    tool_call_id,
                    output.len()
                );
                // Signal immediate persistence
                AccumulatorState::PersistToolOutput {
                    tool_output_id,
                    tool_call_id,
                    output,
                }
            }
            StorageMessage::AssistantMessageStarting => {
                trace!("Storage: received assistant message starting signal (no-op for storage)");
                // This is a signal for the client stream only, storage doesn't need to act on it
                AccumulatorState::Continue
            }
        }
    }
}

/// State transitions for the accumulator
pub enum AccumulatorState {
    Continue,
    Complete(CompleteData),
    Cancelled(PartialData),
    Failed(FailureData),
    CreateReasoningItem {
        item_id: Uuid,
    },
    PersistToolCall {
        tool_call_id: Uuid,
        name: String,
        arguments: serde_json::Value,
    },
    PersistToolOutput {
        tool_output_id: Uuid,
        tool_call_id: Uuid,
        output: String,
    },
}

/// Data for a completed response
pub struct CompleteData {
    pub content: String,
    pub completion_tokens: i32,
    pub finish_reason: String,
    pub message_id: Uuid,
    pub reasoning_content: String,
    pub reasoning_item_id: Option<Uuid>,
}

/// Data for a partial/cancelled response
pub struct PartialData {
    pub content: String,
    pub completion_tokens: i32,
    pub reasoning_content: String,
    pub reasoning_item_id: Option<Uuid>,
}

/// Data for a failed response
pub struct FailureData {
    pub error: String,
    pub partial_content: String,
    pub completion_tokens: i32,
    pub reasoning_content: String,
    pub reasoning_item_id: Option<Uuid>,
}

/// Handles persistence of responses in various states
pub(crate) struct ResponsePersister {
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    message_id: Uuid,
    user_key: SecretKey,
}

impl ResponsePersister {
    pub fn new(
        db: Arc<dyn DBConnection + Send + Sync>,
        response_id: i64,
        message_id: Uuid,
        user_key: SecretKey,
    ) -> Self {
        Self {
            db,
            response_id,
            message_id,
            user_key,
        }
    }

    /// Persist a completed response
    pub async fn persist_completed(&self, data: CompleteData) -> Result<(), String> {
        // Fallback token counting if not provided
        let completion_tokens = if data.completion_tokens == 0 && !data.content.is_empty() {
            let token_count = count_tokens(&data.content);
            if token_count > i32::MAX as usize {
                warn!(
                    "Completion token count {} exceeds i32::MAX, clamping to i32::MAX",
                    token_count
                );
                i32::MAX
            } else {
                token_count as i32
            }
        } else {
            data.completion_tokens
        };

        // Encrypt and store assistant message
        let content_enc = encrypt_with_key(&self.user_key, data.content.as_bytes()).await;

        if let Err(e) = self.db.update_assistant_message(
            data.message_id,
            Some(content_enc),
            completion_tokens,
            STATUS_COMPLETED.to_string(),
            Some(data.finish_reason),
        ) {
            error!("Failed to update assistant message: {:?}", e);
            return Err(format!("Failed to update assistant message: {:?}", e));
        }

        // Update response status
        if let Err(e) = self.db.update_response_status(
            self.response_id,
            ResponseStatus::Completed,
            Some(Utc::now()),
        ) {
            error!("Failed to update response status to completed: {:?}", e);
            return Err(format!("Failed to update response status: {:?}", e));
        }

        // Update reasoning item if present
        if let Some(reasoning_item_id) = data.reasoning_item_id {
            if !data.reasoning_content.is_empty() {
                let reasoning_enc =
                    encrypt_with_key(&self.user_key, data.reasoning_content.as_bytes()).await;
                let reasoning_tokens = count_tokens(&data.reasoning_content);
                let reasoning_tokens_i32 = if reasoning_tokens > i32::MAX as usize {
                    warn!(
                        "Reasoning token count {} exceeds i32::MAX, clamping",
                        reasoning_tokens
                    );
                    i32::MAX
                } else {
                    reasoning_tokens as i32
                };

                if let Err(e) = self.db.update_reasoning_item(
                    reasoning_item_id,
                    Some(reasoning_enc),
                    reasoning_tokens_i32,
                    STATUS_COMPLETED.to_string(),
                ) {
                    error!("Failed to update reasoning item: {:?}", e);
                    // Non-fatal: assistant message was saved successfully
                }
            }
        }

        debug!("Successfully persisted completed response");
        Ok(())
    }

    /// Persist a cancelled response
    pub async fn persist_cancelled(&self, data: PartialData) -> Result<(), String> {
        // Update response status
        if let Err(e) = self.db.update_response_status(
            self.response_id,
            ResponseStatus::Cancelled,
            Some(Utc::now()),
        ) {
            error!("Failed to update response status to cancelled: {:?}", e);
            return Err(format!("Failed to update response status: {:?}", e));
        }

        // Update assistant message to incomplete status with partial content
        let content_enc = if !data.content.is_empty() {
            Some(encrypt_with_key(&self.user_key, data.content.as_bytes()).await)
        } else {
            None
        };

        if let Err(e) = self.db.update_assistant_message(
            self.message_id,
            content_enc,
            data.completion_tokens,
            STATUS_INCOMPLETE.to_string(),
            Some(FINISH_REASON_CANCELLED.to_string()),
        ) {
            error!(
                "Failed to update assistant message after cancellation: {:?}",
                e
            );
            return Err(format!("Failed to update assistant message: {:?}", e));
        }

        // Update reasoning item to incomplete if present
        if let Some(reasoning_item_id) = data.reasoning_item_id {
            if !data.reasoning_content.is_empty() {
                let reasoning_enc =
                    encrypt_with_key(&self.user_key, data.reasoning_content.as_bytes()).await;
                let reasoning_tokens = count_tokens(&data.reasoning_content) as i32;

                if let Err(e) = self.db.update_reasoning_item(
                    reasoning_item_id,
                    Some(reasoning_enc),
                    reasoning_tokens,
                    STATUS_INCOMPLETE.to_string(),
                ) {
                    error!(
                        "Failed to update reasoning item after cancellation: {:?}",
                        e
                    );
                    // Non-fatal
                }
            }
        }

        debug!(
            "Persisted cancelled response {} with {} tokens",
            self.response_id, data.completion_tokens
        );
        Ok(())
    }

    /// Persist a failed response
    pub async fn persist_failed(&self, data: FailureData) -> Result<(), String> {
        // Update response status
        if let Err(e) = self.db.update_response_status(
            self.response_id,
            ResponseStatus::Failed,
            Some(Utc::now()),
        ) {
            error!("Failed to update response status to failed: {:?}", e);
            return Err(format!("Failed to update response status: {:?}", e));
        }

        // Update assistant message to incomplete status with partial content
        let content_enc = if !data.partial_content.is_empty() {
            Some(encrypt_with_key(&self.user_key, data.partial_content.as_bytes()).await)
        } else {
            None
        };

        if let Err(e) = self.db.update_assistant_message(
            self.message_id,
            content_enc,
            data.completion_tokens,
            STATUS_INCOMPLETE.to_string(),
            None,
        ) {
            error!("Failed to update assistant message to incomplete: {:?}", e);
            return Err(format!("Failed to update assistant message: {:?}", e));
        }

        // Update reasoning item to incomplete if present
        if let Some(reasoning_item_id) = data.reasoning_item_id {
            if !data.reasoning_content.is_empty() {
                let reasoning_enc =
                    encrypt_with_key(&self.user_key, data.reasoning_content.as_bytes()).await;
                let reasoning_tokens = count_tokens(&data.reasoning_content) as i32;

                if let Err(e) = self.db.update_reasoning_item(
                    reasoning_item_id,
                    Some(reasoning_enc),
                    reasoning_tokens,
                    STATUS_INCOMPLETE.to_string(),
                ) {
                    error!("Failed to update reasoning item after failure: {:?}", e);
                    // Non-fatal
                }
            }
        }

        debug!("Persisted failed response with error: {}", data.error);
        Ok(())
    }
}

/// Main storage task that orchestrates accumulation and persistence
#[allow(clippy::too_many_arguments)]
pub async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    tool_persist_ack: Option<tokio::sync::oneshot::Sender<Result<(), String>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    reasoning_base_timestamp: chrono::DateTime<chrono::Utc>,
    conversation_id: i64,
    user_id: Uuid,
    user_key: SecretKey,
    message_id: Uuid,
) {
    let mut accumulator = ContentAccumulator::new();
    let persister = ResponsePersister::new(db.clone(), response_id, message_id, user_key);

    // Track tool acknowledgment channel
    let mut tool_ack = tool_persist_ack;

    // Accumulate messages until completion or error
    while let Some(msg) = rx.recv().await {
        match accumulator.handle_message(msg) {
            AccumulatorState::Continue => continue,

            AccumulatorState::CreateReasoningItem { item_id } => {
                // Create reasoning item record with in_progress status
                use crate::models::responses::NewReasoningItem;

                // Look up assistant_message database ID from UUID
                // The assistant_message should exist by now (created before streaming starts)
                let assistant_message_db_id = match db.get_assistant_message_by_uuid(message_id) {
                    Ok(Some(am)) => Some(am.id),
                    Ok(None) => {
                        warn!(
                            "Assistant message {} not found when creating reasoning item",
                            message_id
                        );
                        None
                    }
                    Err(e) => {
                        warn!(
                            "Failed to look up assistant message {}: {:?}",
                            message_id, e
                        );
                        None
                    }
                };

                let new_reasoning_item = NewReasoningItem {
                    uuid: item_id,
                    conversation_id,
                    response_id: Some(response_id),
                    assistant_message_id: assistant_message_db_id,
                    user_id,
                    content_enc: None,
                    summary_enc: None,
                    reasoning_tokens: 0,
                    status: "in_progress".to_string(),
                    created_at: reasoning_base_timestamp,
                };

                match db.create_reasoning_item(new_reasoning_item) {
                    Ok(reasoning_item) => {
                        debug!(
                            "Created reasoning item {} (db id: {}, assistant_message_id: {:?})",
                            item_id, reasoning_item.id, assistant_message_db_id
                        );
                    }
                    Err(e) => {
                        error!("Failed to create reasoning item {}: {:?}", item_id, e);
                        // Non-fatal: reasoning will be lost but response can continue
                    }
                }
            }

            AccumulatorState::PersistToolCall {
                tool_call_id,
                name,
                arguments,
            } => {
                // Persist tool call immediately to database
                use crate::models::responses::NewToolCall;

                let arguments_json = match serde_json::to_string(&arguments) {
                    Ok(json) => json,
                    Err(e) => {
                        error!("Failed to serialize tool arguments: {:?}", e);
                        if let Some(ack) = tool_ack.take() {
                            let _ = ack
                                .send(Err(format!("Failed to serialize tool arguments: {:?}", e)));
                        }
                        continue;
                    }
                };
                let arguments_enc = encrypt_with_key(&user_key, arguments_json.as_bytes()).await;
                let token_count = count_tokens(&arguments_json);
                let argument_tokens = if token_count > i32::MAX as usize {
                    warn!(
                        "Tool argument token count {} exceeds i32::MAX, clamping",
                        token_count
                    );
                    i32::MAX
                } else {
                    token_count as i32
                };

                let new_tool_call = NewToolCall {
                    uuid: tool_call_id,
                    conversation_id,
                    response_id: Some(response_id),
                    user_id,
                    name,
                    arguments_enc: Some(arguments_enc),
                    argument_tokens,
                    status: "completed".to_string(),
                };

                match db.create_tool_call(new_tool_call) {
                    Ok(tool_call) => {
                        debug!(
                            "Persisted tool_call {} (db id: {})",
                            tool_call_id, tool_call.id
                        );
                        // No need to track the ID in memory - we'll look it up when needed
                    }
                    Err(e) => {
                        error!("Failed to persist tool_call {}: {:?}", tool_call_id, e);
                        if let Some(ack) = tool_ack.take() {
                            let _ = ack.send(Err(format!("Failed to persist tool_call: {:?}", e)));
                        }
                    }
                }
            }

            AccumulatorState::PersistToolOutput {
                tool_output_id,
                tool_call_id,
                output,
            } => {
                // Persist tool output immediately to database
                use crate::models::responses::NewToolOutput;

                // Look up the tool_call by UUID to get its database ID (primary key)
                // This is more reliable than tracking in memory across async operations
                // Also validates that the tool_call belongs to this user (security check)
                let tool_call_fk = match db.get_tool_call_by_uuid(tool_call_id, user_id) {
                    Ok(tool_call) => tool_call.id,
                    Err(e) => {
                        error!(
                            "Failed to find tool_call {} for tool_output: {:?}",
                            tool_call_id, e
                        );
                        if let Some(ack) = tool_ack.take() {
                            let _ =
                                ack.send(Err(format!("Tool call not found in database: {:?}", e)));
                        }
                        continue;
                    }
                };

                let output_enc = encrypt_with_key(&user_key, output.as_bytes()).await;
                let token_count = count_tokens(&output);
                let output_tokens = if token_count > i32::MAX as usize {
                    warn!(
                        "Tool output token count {} exceeds i32::MAX, clamping",
                        token_count
                    );
                    i32::MAX
                } else {
                    token_count as i32
                };

                let new_tool_output = NewToolOutput {
                    uuid: tool_output_id,
                    conversation_id,
                    response_id: Some(response_id),
                    user_id,
                    tool_call_fk,
                    output_enc,
                    output_tokens,
                    status: "completed".to_string(),
                    error: None,
                };

                match db.create_tool_output(new_tool_output) {
                    Ok(_) => {
                        debug!(
                            "Persisted tool_output {} for tool_call {}",
                            tool_output_id, tool_call_id
                        );

                        // Send acknowledgment after tool output is persisted
                        if let Some(ack) = tool_ack.take() {
                            let _ = ack.send(Ok(()));
                        }
                    }
                    Err(e) => {
                        error!("Failed to persist tool_output {}: {:?}", tool_output_id, e);
                        if let Some(ack) = tool_ack.take() {
                            let _ =
                                ack.send(Err(format!("Failed to persist tool_output: {:?}", e)));
                        }
                    }
                }
            }

            AccumulatorState::Complete(data) => {
                if let Err(e) = persister.persist_completed(data).await {
                    error!("Failed to persist completed response: {}", e);
                }
                return;
            }

            AccumulatorState::Cancelled(data) => {
                if let Err(e) = persister.persist_cancelled(data).await {
                    error!("Failed to persist cancelled response: {}", e);
                }
                return;
            }

            AccumulatorState::Failed(data) => {
                if let Err(e) = persister.persist_failed(data).await {
                    error!("Failed to persist failed response: {}", e);
                }
                return;
            }
        }
    }

    // Channel closed without Done or Error - treat as failure
    warn!("Storage channel closed before receiving Done signal");
    if let Err(e) = persister
        .persist_failed(FailureData {
            error: "Channel closed prematurely".to_string(),
            partial_content: String::new(),
            completion_tokens: 0,
            reasoning_content: accumulator.reasoning_content().to_string(),
            reasoning_item_id: accumulator.reasoning_item_id(),
        })
        .await
    {
        error!("Failed to persist incomplete response: {}", e);
    }
}
