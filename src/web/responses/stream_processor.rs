//! Upstream stream processor for handling SSE responses from chat completion API

use crate::web::responses::constants::FINISH_REASON_STOP;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, trace};
use uuid::Uuid;

use super::handlers::StorageMessage;

/// Processes upstream SSE stream and broadcasts to storage and client channels
///
/// This struct encapsulates the complex SSE parsing, buffer management, and
/// channel broadcasting logic, making it testable and reusable.
pub struct UpstreamStreamProcessor {
    buffer: String,
    message_id: Uuid,
    response_uuid: Uuid,
    tx_storage: mpsc::Sender<StorageMessage>,
    tx_client: mpsc::Sender<StorageMessage>,
    finish_reason: Option<String>,
}

impl UpstreamStreamProcessor {
    /// Create a new upstream stream processor
    pub fn new(
        message_id: Uuid,
        response_uuid: Uuid,
        tx_storage: mpsc::Sender<StorageMessage>,
        tx_client: mpsc::Sender<StorageMessage>,
    ) -> Self {
        Self {
            buffer: String::with_capacity(8192),
            message_id,
            response_uuid,
            tx_storage,
            tx_client,
            finish_reason: None,
        }
    }

    /// Process a chunk of bytes from the upstream stream
    pub async fn process_chunk(&mut self, bytes: &[u8]) -> Result<(), ProcessorError> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        // Process complete SSE frames
        while let Some(frame) = self.extract_sse_frame() {
            self.handle_sse_frame(&frame).await?;
        }

        Ok(())
    }

    /// Extract a complete SSE frame from the buffer
    fn extract_sse_frame(&mut self) -> Option<String> {
        if let Some(pos) = self.buffer.find("\n\n") {
            let frame = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            // Skip empty frames
            if frame.trim().is_empty() {
                return None;
            }

            Some(frame)
        } else {
            None
        }
    }

    /// Handle a single SSE frame
    async fn handle_sse_frame(&mut self, frame: &str) -> Result<(), ProcessorError> {
        // Skip non-data frames
        if !frame.starts_with("data: ") {
            return Ok(());
        }

        let data = frame.strip_prefix("data: ").unwrap().trim();

        // Handle [DONE] signal
        if data == "[DONE]" {
            trace!("Upstream processor: received [DONE], sending completion");
            self.send_completion().await?;
            return Ok(());
        }

        // Parse JSON data
        let json_data: Value =
            serde_json::from_str(data).map_err(|e| ProcessorError::JsonParse(e.to_string()))?;

        // Extract content delta
        if let Some(content) = json_data["choices"][0]["delta"]["content"].as_str() {
            trace!("Upstream processor: found content delta: {}", content);
            self.send_content_delta(content).await?;
        }

        // Extract usage
        if let Some(usage) = json_data.get("usage") {
            let prompt_tokens = usage["prompt_tokens"].as_i64().unwrap_or(0) as i32;
            let completion_tokens = usage["completion_tokens"].as_i64().unwrap_or(0) as i32;

            debug!(
                "Upstream processor: found usage - prompt_tokens={}, completion_tokens={}",
                prompt_tokens, completion_tokens
            );
            self.send_usage(prompt_tokens, completion_tokens).await?;
        }

        // Extract finish reason
        if let Some(finish_reason) = json_data["choices"][0]["finish_reason"].as_str() {
            trace!("Upstream processor: found finish_reason: {}", finish_reason);
            self.finish_reason = Some(finish_reason.to_string());
        }

        Ok(())
    }

    /// Send content delta to both channels
    ///
    /// IMPORTANT: Storage send must succeed (returns error if it fails).
    /// Client send failures are ignored (client may have disconnected).
    async fn send_content_delta(&self, content: &str) -> Result<(), ProcessorError> {
        let msg = StorageMessage::ContentDelta(content.to_string());

        // Storage is critical - must succeed
        self.tx_storage
            .send(msg.clone())
            .await
            .map_err(|_| ProcessorError::ChannelClosed)?;

        // Client send failure is OK (they might have disconnected)
        let _ = self.tx_client.send(msg).await;

        Ok(())
    }

    /// Send usage data to both channels
    ///
    /// IMPORTANT: Storage send must succeed (returns error if it fails).
    /// Client send failures are ignored (client may have disconnected).
    async fn send_usage(
        &self,
        prompt_tokens: i32,
        completion_tokens: i32,
    ) -> Result<(), ProcessorError> {
        let msg = StorageMessage::Usage {
            prompt_tokens,
            completion_tokens,
        };

        // Storage is critical - must succeed
        self.tx_storage
            .send(msg.clone())
            .await
            .map_err(|_| ProcessorError::ChannelClosed)?;

        // Client send failure is OK (they might have disconnected)
        let _ = self.tx_client.send(msg).await;

        Ok(())
    }

    /// Send completion signal to both channels
    ///
    /// IMPORTANT: Storage send must succeed (returns error if it fails).
    /// Client send failures are ignored (client may have disconnected).
    async fn send_completion(&self) -> Result<(), ProcessorError> {
        let msg = StorageMessage::Done {
            finish_reason: self
                .finish_reason
                .clone()
                .unwrap_or_else(|| FINISH_REASON_STOP.to_string()),
            message_id: self.message_id,
        };

        // Storage is critical - must succeed
        self.tx_storage
            .send(msg.clone())
            .await
            .map_err(|_| ProcessorError::ChannelClosed)?;

        // Client send failure is OK (they might have disconnected)
        let _ = self.tx_client.send(msg).await;

        Ok(())
    }

    /// Send error to both channels
    ///
    /// Best effort for both channels - errors are logged but not returned
    pub async fn send_error(&self, error: String) -> Result<(), ProcessorError> {
        let msg = StorageMessage::Error(error);
        let _ = self.tx_storage.send(msg.clone()).await;
        let _ = self.tx_client.send(msg).await;
        Ok(())
    }

    /// Send cancellation signal to both channels
    ///
    /// Best effort for both channels - errors are logged but not returned
    pub async fn send_cancellation(&self) -> Result<(), ProcessorError> {
        debug!(
            "Upstream processor received cancellation for response {}",
            self.response_uuid
        );
        let msg = StorageMessage::Cancelled;
        let _ = self.tx_storage.send(msg.clone()).await;
        let _ = self.tx_client.send(msg).await;
        Ok(())
    }
}

/// Errors that can occur during stream processing
#[derive(Debug)]
pub enum ProcessorError {
    JsonParse(String),
    ChannelClosed,
}

impl std::fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessorError::JsonParse(msg) => write!(f, "JSON parse error: {}", msg),
            ProcessorError::ChannelClosed => write!(f, "Channel closed"),
        }
    }
}

impl std::error::Error for ProcessorError {}
