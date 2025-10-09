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

/// Tool message to be persisted
#[derive(Debug, Clone)]
pub(crate) struct ToolMessage {
    tool_call_id: Uuid,
    name: String,
    arguments: serde_json::Value,
}

/// Tool output to be persisted
#[derive(Debug, Clone)]
pub(crate) struct ToolOutputMessage {
    tool_output_id: Uuid,
    tool_call_id: Uuid,
    output: String,
}

/// Accumulates streaming content and metadata
pub(crate) struct ContentAccumulator {
    content: String,
    completion_tokens: i32,
    tool_calls: Vec<ToolMessage>,
    tool_outputs: Vec<ToolOutputMessage>,
}

impl ContentAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::with_capacity(4096),
            completion_tokens: 0,
            tool_calls: Vec::new(),
            tool_outputs: Vec::new(),
        }
    }

    /// Handle a storage message and return the accumulator state
    pub fn handle_message(&mut self, msg: StorageMessage) -> AccumulatorState {
        match msg {
            StorageMessage::ContentDelta(delta) => {
                trace!("Storage: received content delta: {} chars", delta.len());
                self.content.push_str(&delta);
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
                    tool_calls: self.tool_calls.clone(),
                    tool_outputs: self.tool_outputs.clone(),
                })
            }
            StorageMessage::Cancelled => {
                debug!("Storage: received cancellation signal");
                AccumulatorState::Cancelled(PartialData {
                    content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                    tool_calls: self.tool_calls.clone(),
                    tool_outputs: self.tool_outputs.clone(),
                })
            }
            StorageMessage::Error(e) => {
                error!("Storage: received error: {}", e);
                AccumulatorState::Failed(FailureData {
                    error: e,
                    partial_content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                    tool_calls: self.tool_calls.clone(),
                    tool_outputs: self.tool_outputs.clone(),
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
                // Accumulate tool call for later persistence
                self.tool_calls.push(ToolMessage {
                    tool_call_id,
                    name,
                    arguments,
                });
                AccumulatorState::Continue
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
                // Accumulate tool output for later persistence
                self.tool_outputs.push(ToolOutputMessage {
                    tool_output_id,
                    tool_call_id,
                    output,
                });
                AccumulatorState::Continue
            }
            StorageMessage::PersistTools { ack } => {
                debug!(
                    "Storage: received PersistTools barrier - {} tool_calls, {} tool_outputs",
                    self.tool_calls.len(),
                    self.tool_outputs.len()
                );
                AccumulatorState::PersistToolsNow { ack }
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
    PersistToolsNow {
        ack: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

/// Data for a completed response
pub struct CompleteData {
    pub content: String,
    pub completion_tokens: i32,
    pub finish_reason: String,
    pub message_id: Uuid,
    pub tool_calls: Vec<ToolMessage>,
    pub tool_outputs: Vec<ToolOutputMessage>,
}

/// Data for a partial/cancelled response
pub struct PartialData {
    pub content: String,
    pub completion_tokens: i32,
    pub tool_calls: Vec<ToolMessage>,
    pub tool_outputs: Vec<ToolOutputMessage>,
}

/// Data for a failed response
pub struct FailureData {
    pub error: String,
    pub partial_content: String,
    pub completion_tokens: i32,
    pub tool_calls: Vec<ToolMessage>,
    pub tool_outputs: Vec<ToolOutputMessage>,
}

/// Handles persistence of responses in various states
pub(crate) struct ResponsePersister {
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    conversation_id: i64,
    user_id: Uuid,
    message_id: Uuid,
    user_key: SecretKey,
}

impl ResponsePersister {
    pub fn new(
        db: Arc<dyn DBConnection + Send + Sync>,
        response_id: i64,
        conversation_id: i64,
        user_id: Uuid,
        message_id: Uuid,
        user_key: SecretKey,
    ) -> Self {
        Self {
            db,
            response_id,
            conversation_id,
            user_id,
            message_id,
            user_key,
        }
    }

    /// Persist tool messages to database
    async fn persist_tools(
        &self,
        tool_calls: &[ToolMessage],
        tool_outputs: &[ToolOutputMessage],
    ) -> Result<(), String> {
        use crate::models::responses::{NewToolCall, NewToolOutput};

        // Persist tool calls and build a map of UUID -> database ID
        let mut tool_call_id_map = std::collections::HashMap::new();

        for tool_msg in tool_calls {
            // Encrypt arguments
            let arguments_json = serde_json::to_string(&tool_msg.arguments)
                .map_err(|e| format!("Failed to serialize tool arguments: {:?}", e))?;
            let arguments_enc = encrypt_with_key(&self.user_key, arguments_json.as_bytes()).await;
            let argument_tokens = count_tokens(&arguments_json) as i32;

            let new_tool_call = NewToolCall {
                uuid: tool_msg.tool_call_id,
                conversation_id: self.conversation_id,
                response_id: Some(self.response_id),
                user_id: self.user_id,
                name: tool_msg.name.clone(),
                arguments_enc: Some(arguments_enc),
                argument_tokens,
                status: "completed".to_string(),
            };

            match self.db.create_tool_call(new_tool_call) {
                Ok(tool_call) => {
                    debug!(
                        "Persisted tool_call {} (db id: {})",
                        tool_msg.tool_call_id, tool_call.id
                    );
                    tool_call_id_map.insert(tool_msg.tool_call_id, tool_call.id);
                }
                Err(e) => {
                    error!(
                        "Failed to persist tool_call {}: {:?}",
                        tool_msg.tool_call_id, e
                    );
                    return Err(format!("Failed to persist tool_call: {:?}", e));
                }
            }
        }

        // Persist tool outputs
        for tool_output_msg in tool_outputs {
            // Get the database ID for this tool_call
            let tool_call_fk = tool_call_id_map
                .get(&tool_output_msg.tool_call_id)
                .ok_or_else(|| {
                    format!(
                        "Tool output references unknown tool_call: {}",
                        tool_output_msg.tool_call_id
                    )
                })?;

            // Encrypt output
            let output_enc =
                encrypt_with_key(&self.user_key, tool_output_msg.output.as_bytes()).await;
            let output_tokens = count_tokens(&tool_output_msg.output) as i32;

            let new_tool_output = NewToolOutput {
                uuid: tool_output_msg.tool_output_id,
                conversation_id: self.conversation_id,
                response_id: Some(self.response_id),
                user_id: self.user_id,
                tool_call_fk: *tool_call_fk,
                output_enc,
                output_tokens,
                status: "completed".to_string(),
                error: None,
            };

            if let Err(e) = self.db.create_tool_output(new_tool_output) {
                error!(
                    "Failed to persist tool_output {}: {:?}",
                    tool_output_msg.tool_output_id, e
                );
                return Err(format!("Failed to persist tool_output: {:?}", e));
            }

            debug!(
                "Persisted tool_output {} for tool_call {}",
                tool_output_msg.tool_output_id, tool_output_msg.tool_call_id
            );
        }

        Ok(())
    }

    /// Persist a completed response
    pub async fn persist_completed(&self, data: CompleteData) -> Result<(), String> {
        // Persist tool messages first (if any)
        if !data.tool_calls.is_empty() || !data.tool_outputs.is_empty() {
            debug!(
                "Persisting {} tool_calls and {} tool_outputs",
                data.tool_calls.len(),
                data.tool_outputs.len()
            );
            self.persist_tools(&data.tool_calls, &data.tool_outputs)
                .await?;
        }

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

        debug!("Successfully persisted completed response");
        Ok(())
    }

    /// Persist a cancelled response
    pub async fn persist_cancelled(&self, data: PartialData) -> Result<(), String> {
        // Persist tool messages first (if any)
        if !data.tool_calls.is_empty() || !data.tool_outputs.is_empty() {
            debug!(
                "Persisting {} tool_calls and {} tool_outputs (cancelled)",
                data.tool_calls.len(),
                data.tool_outputs.len()
            );
            self.persist_tools(&data.tool_calls, &data.tool_outputs)
                .await?;
        }

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

        debug!(
            "Persisted cancelled response {} with {} tokens",
            self.response_id, data.completion_tokens
        );
        Ok(())
    }

    /// Persist a failed response
    pub async fn persist_failed(&self, data: FailureData) -> Result<(), String> {
        // Persist tool messages first (if any)
        if !data.tool_calls.is_empty() || !data.tool_outputs.is_empty() {
            debug!(
                "Persisting {} tool_calls and {} tool_outputs (failed)",
                data.tool_calls.len(),
                data.tool_outputs.len()
            );
            self.persist_tools(&data.tool_calls, &data.tool_outputs)
                .await?;
        }

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

        debug!("Persisted failed response with error: {}", data.error);
        Ok(())
    }
}

/// Main storage task that orchestrates accumulation and persistence
pub async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    conversation_id: i64,
    user_id: Uuid,
    user_key: SecretKey,
    message_id: Uuid,
) {
    let mut accumulator = ContentAccumulator::new();
    let persister = ResponsePersister::new(
        db.clone(),
        response_id,
        conversation_id,
        user_id,
        message_id,
        user_key,
    );

    // Accumulate messages until completion or error
    while let Some(msg) = rx.recv().await {
        match accumulator.handle_message(msg) {
            AccumulatorState::Continue => continue,

            AccumulatorState::PersistToolsNow { ack } => {
                // Persist accumulated tools immediately
                let result = persister
                    .persist_tools(&accumulator.tool_calls, &accumulator.tool_outputs)
                    .await;

                // Clear accumulated tools after persistence (whether success or failure)
                accumulator.tool_calls.clear();
                accumulator.tool_outputs.clear();

                // Send acknowledgment back to caller
                let _ = ack.send(result);

                // Continue processing other messages
                continue;
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
            tool_calls: accumulator.tool_calls.clone(),
            tool_outputs: accumulator.tool_outputs.clone(),
        })
        .await
    {
        error!("Failed to persist incomplete response: {}", e);
    }
}
