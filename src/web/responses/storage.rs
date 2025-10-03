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
    completion_tokens: i32,
}

impl ContentAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::with_capacity(4096),
            completion_tokens: 0,
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
                })
            }
            StorageMessage::Cancelled => {
                debug!("Storage: received cancellation signal");
                AccumulatorState::Cancelled(PartialData {
                    content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                })
            }
            StorageMessage::Error(e) => {
                error!("Storage: received error: {}", e);
                AccumulatorState::Failed(FailureData {
                    error: e,
                    partial_content: self.content.clone(),
                    completion_tokens: self.completion_tokens,
                })
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
}

/// Data for a completed response
pub struct CompleteData {
    pub content: String,
    pub completion_tokens: i32,
    pub finish_reason: String,
    pub message_id: Uuid,
}

/// Data for a partial/cancelled response
pub struct PartialData {
    pub content: String,
    pub completion_tokens: i32,
}

/// Data for a failed response
pub struct FailureData {
    pub error: String,
    pub partial_content: String,
    pub completion_tokens: i32,
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

        debug!("Persisted failed response with error: {}", data.error);
        Ok(())
    }
}

/// Main storage task that orchestrates accumulation and persistence
pub async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    user_key: SecretKey,
    message_id: Uuid,
) {
    let mut accumulator = ContentAccumulator::new();
    let persister = ResponsePersister::new(db.clone(), response_id, message_id, user_key);

    // Accumulate messages until completion or error
    while let Some(msg) = rx.recv().await {
        match accumulator.handle_message(msg) {
            AccumulatorState::Continue => continue,

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
        })
        .await
    {
        error!("Failed to persist incomplete response: {}", e);
    }
}
