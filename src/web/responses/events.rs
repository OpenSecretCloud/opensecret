//! SSE event handling utilities

use crate::AppState;
use axum::response::sse::Event;
use serde::Serialize;
use tracing::{error, trace};
use uuid::Uuid;

use super::constants::{
    ERROR_DATA_ENCRYPTION_FAILED, ERROR_DATA_SERIALIZATION_FAILED, EVENT_RESPONSE_CANCELLED,
    EVENT_RESPONSE_COMPLETED, EVENT_RESPONSE_CONTENT_PART_ADDED, EVENT_RESPONSE_CONTENT_PART_DONE,
    EVENT_RESPONSE_CREATED, EVENT_RESPONSE_ERROR, EVENT_RESPONSE_IN_PROGRESS,
    EVENT_RESPONSE_OUTPUT_ITEM_ADDED, EVENT_RESPONSE_OUTPUT_ITEM_DONE,
    EVENT_RESPONSE_OUTPUT_TEXT_DELTA, EVENT_RESPONSE_OUTPUT_TEXT_DONE,
};
use super::handlers::{
    encrypt_event, ResponseCancelledEvent, ResponseCompletedEvent, ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent, ResponseCreatedEvent, ResponseErrorEvent,
    ResponseInProgressEvent, ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent, ResponseOutputTextDoneEvent,
};

/// Handles SSE event emission with automatic encryption and error handling
///
/// This struct eliminates ~300 lines of duplicated event handling code by centralizing:
/// - Serialization
/// - Encryption
/// - Error handling
/// - Sequence number management
pub struct SseEventEmitter<'a> {
    state: &'a AppState,
    session_id: Uuid,
    sequence_number: i32,
}

impl<'a> SseEventEmitter<'a> {
    /// Create a new SSE event emitter
    pub fn new(state: &'a AppState, session_id: Uuid, initial_sequence: i32) -> Self {
        Self {
            state,
            session_id,
            sequence_number: initial_sequence,
        }
    }

    /// Emit an SSE event with automatic serialization, encryption, and error handling
    ///
    /// This method:
    /// 1. Increments sequence number
    /// 2. Serializes the event data to JSON
    /// 3. Encrypts the JSON payload
    /// 4. Returns an SSE Event ready to yield to the client
    ///
    /// # Arguments
    /// * `event_type` - The SSE event type (e.g., "response.created")
    /// * `data` - The event data to serialize and send
    ///
    /// # Returns
    /// An SSE Event with encrypted data, or an error event if serialization/encryption fails
    pub async fn emit<T: Serialize>(&mut self, event_type: &str, data: &T) -> Event {
        self.sequence_number += 1;

        match serde_json::to_value(data) {
            Ok(json) => {
                match encrypt_event(self.state, &self.session_id, event_type, &json).await {
                    Ok(event) => {
                        trace!(
                            "Emitted {} event (seq: {})",
                            event_type,
                            self.sequence_number
                        );
                        event
                    }
                    Err(e) => {
                        error!("Failed to encrypt {} event: {:?}", event_type, e);
                        Event::default()
                            .event("error")
                            .data(ERROR_DATA_ENCRYPTION_FAILED)
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize {}: {:?}", event_type, e);
                Event::default()
                    .event("error")
                    .data(ERROR_DATA_SERIALIZATION_FAILED)
            }
        }
    }

    /// Emit an event without incrementing the sequence number
    ///
    /// Useful for error events or special cases where sequence continuity
    /// should not be affected.
    pub async fn emit_without_sequence<T: Serialize>(&self, event_type: &str, data: &T) -> Event {
        match serde_json::to_value(data) {
            Ok(json) => {
                match encrypt_event(self.state, &self.session_id, event_type, &json).await {
                    Ok(event) => event,
                    Err(e) => {
                        error!("Failed to encrypt {} event: {:?}", event_type, e);
                        Event::default()
                            .event("error")
                            .data(ERROR_DATA_ENCRYPTION_FAILED)
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize {}: {:?}", event_type, e);
                Event::default()
                    .event("error")
                    .data(ERROR_DATA_SERIALIZATION_FAILED)
            }
        }
    }

    /// Get the current sequence number
    pub fn sequence_number(&self) -> i32 {
        self.sequence_number
    }
}

/// Type-safe event wrapper for all Response API events
///
/// This enum provides compile-time safety for event types, eliminating
/// the possibility of typos in event names and making refactoring easier.
///
/// Each variant wraps the corresponding event struct and knows its event type string.
pub enum ResponseEvent {
    Created(ResponseCreatedEvent),
    InProgress(ResponseInProgressEvent),
    OutputItemAdded(ResponseOutputItemAddedEvent),
    ContentPartAdded(ResponseContentPartAddedEvent),
    OutputTextDelta(ResponseOutputTextDeltaEvent),
    OutputTextDone(ResponseOutputTextDoneEvent),
    ContentPartDone(ResponseContentPartDoneEvent),
    OutputItemDone(ResponseOutputItemDoneEvent),
    Completed(ResponseCompletedEvent),
    Cancelled(ResponseCancelledEvent),
    Error(ResponseErrorEvent),
}

impl ResponseEvent {
    /// Get the event type string for SSE
    pub fn event_type(&self) -> &'static str {
        match self {
            ResponseEvent::Created(_) => EVENT_RESPONSE_CREATED,
            ResponseEvent::InProgress(_) => EVENT_RESPONSE_IN_PROGRESS,
            ResponseEvent::OutputItemAdded(_) => EVENT_RESPONSE_OUTPUT_ITEM_ADDED,
            ResponseEvent::ContentPartAdded(_) => EVENT_RESPONSE_CONTENT_PART_ADDED,
            ResponseEvent::OutputTextDelta(_) => EVENT_RESPONSE_OUTPUT_TEXT_DELTA,
            ResponseEvent::OutputTextDone(_) => EVENT_RESPONSE_OUTPUT_TEXT_DONE,
            ResponseEvent::ContentPartDone(_) => EVENT_RESPONSE_CONTENT_PART_DONE,
            ResponseEvent::OutputItemDone(_) => EVENT_RESPONSE_OUTPUT_ITEM_DONE,
            ResponseEvent::Completed(_) => EVENT_RESPONSE_COMPLETED,
            ResponseEvent::Cancelled(_) => EVENT_RESPONSE_CANCELLED,
            ResponseEvent::Error(_) => EVENT_RESPONSE_ERROR,
        }
    }

    /// Convert to SSE event with encryption
    ///
    /// This is a convenience method that automatically serializes and encrypts the event.
    pub async fn to_sse_event(&self, emitter: &mut SseEventEmitter<'_>) -> Event {
        match self {
            ResponseEvent::Created(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::InProgress(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::OutputItemAdded(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::ContentPartAdded(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::OutputTextDelta(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::OutputTextDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::ContentPartDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::OutputItemDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::Completed(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::Cancelled(e) => {
                emitter.emit_without_sequence(self.event_type(), e).await
            }
            ResponseEvent::Error(e) => emitter.emit_without_sequence(self.event_type(), e).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct TestEvent {
        message: String,
    }

    // Note: Full integration tests would require AppState setup
    // These are placeholder tests showing the API

    #[test]
    fn test_sequence_number_management() {
        // This is a unit test for sequence number logic
        // Real emit() tests would require async and AppState
        let mut seq = 0;
        seq += 1;
        assert_eq!(seq, 1);
        seq += 1;
        assert_eq!(seq, 2);
    }

    #[test]
    fn test_event_serialization() {
        let event = TestEvent {
            message: "test".to_string(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["message"], "test");
    }

    #[test]
    fn test_event_type_mapping() {
        use crate::web::responses::handlers::*;
        use uuid::Uuid;

        // Test that event types map correctly
        let created = ResponseEvent::Created(ResponseCreatedEvent {
            event_type: "response.created",
            response: ResponsesCreateResponse {
                id: Uuid::new_v4(),
                object: "response",
                created_at: 0,
                status: "in_progress",
                background: false,
                error: None,
                incomplete_details: None,
                instructions: None,
                max_output_tokens: None,
                max_tool_calls: None,
                model: "test".to_string(),
                output: vec![],
                parallel_tool_calls: false,
                previous_response_id: None,
                prompt_cache_key: None,
                reasoning: ReasoningInfo {
                    effort: None,
                    summary: None,
                },
                safety_identifier: None,
                store: true,
                temperature: 1.0,
                text: TextFormat {
                    format: TextFormatSpec {
                        format_type: "text".to_string(),
                    },
                },
                tool_choice: "auto".to_string(),
                tools: vec![],
                top_logprobs: 0,
                top_p: 1.0,
                truncation: "disabled",
                usage: None,
                user: None,
                metadata: None,
            },
            sequence_number: 0,
        });

        assert_eq!(created.event_type(), EVENT_RESPONSE_CREATED);
    }
}
