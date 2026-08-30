//! SSE event handling utilities

use crate::{web::encryption_middleware::TransportSession, AppState};
use axum::response::sse::Event;
use bytes::Bytes;
use serde::Serialize;
use tracing::{error, trace};

use super::constants::{
    ERROR_DATA_ENCRYPTION_FAILED, ERROR_DATA_SERIALIZATION_FAILED, EVENT_RESPONSE_CANCELLED,
    EVENT_RESPONSE_COMPLETED, EVENT_RESPONSE_CONTENT_PART_ADDED, EVENT_RESPONSE_CONTENT_PART_DONE,
    EVENT_RESPONSE_CREATED, EVENT_RESPONSE_ERROR, EVENT_RESPONSE_IN_PROGRESS,
    EVENT_RESPONSE_OUTPUT_ITEM_ADDED, EVENT_RESPONSE_OUTPUT_ITEM_DONE,
    EVENT_RESPONSE_OUTPUT_TEXT_DELTA, EVENT_RESPONSE_OUTPUT_TEXT_DONE,
    EVENT_RESPONSE_REASONING_TEXT_DELTA, EVENT_RESPONSE_REASONING_TEXT_DONE,
    EVENT_TOOL_CALL_CREATED, EVENT_TOOL_OUTPUT_CREATED,
};
use super::handlers::{
    encrypt_event, ResponseCancelledEvent, ResponseCompletedEvent, ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent, ResponseCreatedEvent, ResponseErrorEvent,
    ResponseInProgressEvent, ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent, ResponseOutputTextDoneEvent, ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent, ToolCallCreatedEvent, ToolOutputCreatedEvent,
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
    transport_session: TransportSession,
    sequence_number: i32,
}

impl<'a> SseEventEmitter<'a> {
    /// Create a new SSE event emitter
    pub fn new(
        state: &'a AppState,
        transport_session: TransportSession,
        initial_sequence: i32,
    ) -> Self {
        Self {
            state,
            transport_session,
            sequence_number: initial_sequence,
        }
    }

    /// Emit an SSE event with automatic serialization, encryption, and error handling
    ///
    /// This method:
    /// 1. Serializes the event data to JSON
    /// 2. Encrypts the JSON payload
    /// 3. Increments sequence number on success
    /// 4. Returns an SSE Event ready to yield to the client
    ///
    /// # Arguments
    /// * `event_type` - The SSE event type (e.g., "response.created")
    /// * `data` - The event data to serialize and send
    ///
    /// # Returns
    /// An SSE Event with encrypted data, or an error event if serialization/encryption fails
    pub async fn emit<T: Serialize>(&mut self, event_type: &str, data: &T) -> Event {
        match serde_json::to_value(data) {
            Ok(json) => {
                match encrypt_event(self.state, &self.transport_session, event_type, &json).await {
                    Ok(event) => {
                        self.sequence_number += 1;
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
                match encrypt_event(self.state, &self.transport_session, event_type, &json).await {
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
    ReasoningTextDelta(ResponseReasoningTextDeltaEvent),
    ReasoningTextDone(ResponseReasoningTextDoneEvent),
    ContentPartDone(ResponseContentPartDoneEvent),
    OutputItemDone(ResponseOutputItemDoneEvent),
    Completed(ResponseCompletedEvent),
    Cancelled(ResponseCancelledEvent),
    Error(ResponseErrorEvent),
    ToolCallCreated(ToolCallCreatedEvent),
    ToolOutputCreated(ToolOutputCreatedEvent),
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
            ResponseEvent::ReasoningTextDelta(_) => EVENT_RESPONSE_REASONING_TEXT_DELTA,
            ResponseEvent::ReasoningTextDone(_) => EVENT_RESPONSE_REASONING_TEXT_DONE,
            ResponseEvent::ContentPartDone(_) => EVENT_RESPONSE_CONTENT_PART_DONE,
            ResponseEvent::OutputItemDone(_) => EVENT_RESPONSE_OUTPUT_ITEM_DONE,
            ResponseEvent::Completed(_) => EVENT_RESPONSE_COMPLETED,
            ResponseEvent::Cancelled(_) => EVENT_RESPONSE_CANCELLED,
            ResponseEvent::Error(_) => EVENT_RESPONSE_ERROR,
            ResponseEvent::ToolCallCreated(_) => EVENT_TOOL_CALL_CREATED,
            ResponseEvent::ToolOutputCreated(_) => EVENT_TOOL_OUTPUT_CREATED,
        }
    }

    /// Whether this event completes the application-level Responses stream.
    ///
    /// Transport v2 carries these events as ordinary plaintext SSE bytes and
    /// emits its authenticated `End` record only after the terminal event has
    /// been delivered. `response.error` is an application terminal, not a
    /// transport failure.
    pub(crate) fn is_terminal(&self) -> bool {
        matches!(
            self,
            ResponseEvent::Completed(_) | ResponseEvent::Cancelled(_) | ResponseEvent::Error(_)
        )
    }

    /// Match the legacy application sequence-number behavior. Cancellation
    /// and error payloads do not carry a sequence number and historically did
    /// not advance the v1 emitter's counter.
    pub(crate) fn advances_sequence(&self) -> bool {
        !matches!(self, ResponseEvent::Cancelled(_) | ResponseEvent::Error(_))
    }

    /// Assign the application-level Responses sequence number immediately
    /// before projection. V1 does this before encryption and commits the
    /// emitter counter only after encryption succeeds; v2 commits only after
    /// plaintext serialization succeeds. Keeping assignment in the adapters
    /// preserves the legacy encryption-failure behavior without coupling the
    /// application producer to either transport.
    pub(crate) fn set_sequence_number(&mut self, sequence_number: i32) {
        match self {
            ResponseEvent::Created(event) => event.sequence_number = sequence_number,
            ResponseEvent::InProgress(event) => event.sequence_number = sequence_number,
            ResponseEvent::OutputItemAdded(event) => event.sequence_number = sequence_number,
            ResponseEvent::ContentPartAdded(event) => event.sequence_number = sequence_number,
            ResponseEvent::OutputTextDelta(event) => event.sequence_number = sequence_number,
            ResponseEvent::OutputTextDone(event) => event.sequence_number = sequence_number,
            ResponseEvent::ReasoningTextDelta(event) => event.sequence_number = sequence_number,
            ResponseEvent::ReasoningTextDone(event) => event.sequence_number = sequence_number,
            ResponseEvent::ContentPartDone(event) => event.sequence_number = sequence_number,
            ResponseEvent::OutputItemDone(event) => event.sequence_number = sequence_number,
            ResponseEvent::Completed(event) => event.sequence_number = sequence_number,
            ResponseEvent::ToolCallCreated(event) => event.sequence_number = sequence_number,
            ResponseEvent::ToolOutputCreated(event) => event.sequence_number = sequence_number,
            ResponseEvent::Cancelled(_) | ResponseEvent::Error(_) => {}
        }
    }

    /// Serialize this application event as an ordinary OpenAI-compatible SSE
    /// frame. The bytes are plaintext at the logical application layer; the v2
    /// gateway authenticates and encrypts them as `StreamRecord::Chunk` data.
    pub(crate) fn to_plaintext_sse_frame(&self) -> Result<Bytes, serde_json::Error> {
        fn payload<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
            serde_json::to_vec(value)
        }

        let payload = match self {
            ResponseEvent::Created(event) => payload(event),
            ResponseEvent::InProgress(event) => payload(event),
            ResponseEvent::OutputItemAdded(event) => payload(event),
            ResponseEvent::ContentPartAdded(event) => payload(event),
            ResponseEvent::OutputTextDelta(event) => payload(event),
            ResponseEvent::OutputTextDone(event) => payload(event),
            ResponseEvent::ReasoningTextDelta(event) => payload(event),
            ResponseEvent::ReasoningTextDone(event) => payload(event),
            ResponseEvent::ContentPartDone(event) => payload(event),
            ResponseEvent::OutputItemDone(event) => payload(event),
            ResponseEvent::Completed(event) => payload(event),
            ResponseEvent::Cancelled(event) => payload(event),
            ResponseEvent::Error(event) => payload(event),
            ResponseEvent::ToolCallCreated(event) => payload(event),
            ResponseEvent::ToolOutputCreated(event) => payload(event),
        }?;

        let event_type = self.event_type().as_bytes();
        let mut frame = Vec::with_capacity(event_type.len() + payload.len() + 16);
        frame.extend_from_slice(b"event: ");
        frame.extend_from_slice(event_type);
        frame.extend_from_slice(b"\ndata: ");
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(b"\n\n");
        Ok(Bytes::from(frame))
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
            ResponseEvent::ReasoningTextDelta(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::ReasoningTextDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::ContentPartDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::OutputItemDone(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::Completed(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::Cancelled(e) => {
                emitter.emit_without_sequence(self.event_type(), e).await
            }
            ResponseEvent::Error(e) => emitter.emit_without_sequence(self.event_type(), e).await,
            ResponseEvent::ToolCallCreated(e) => emitter.emit(self.event_type(), e).await,
            ResponseEvent::ToolOutputCreated(e) => emitter.emit(self.event_type(), e).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::responses::constants::{
        OBJECT_TYPE_RESPONSE, STATUS_IN_PROGRESS, TEXT_FORMAT_TYPE, TOOL_CHOICE_AUTO,
        TRUNCATION_DISABLED,
    };

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
        let mut created = ResponseEvent::Created(ResponseCreatedEvent {
            event_type: EVENT_RESPONSE_CREATED,
            response: ResponsesCreateResponse {
                id: Uuid::new_v4(),
                object: OBJECT_TYPE_RESPONSE,
                created_at: 0,
                status: STATUS_IN_PROGRESS.to_string(),
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
                        format_type: TEXT_FORMAT_TYPE.to_string(),
                    },
                },
                tool_choice: TOOL_CHOICE_AUTO.to_string(),
                tools: vec![],
                top_logprobs: 0,
                top_p: 1.0,
                truncation: TRUNCATION_DISABLED,
                usage: None,
                user: None,
                metadata: None,
            },
            sequence_number: 0,
        });

        assert_eq!(created.event_type(), EVENT_RESPONSE_CREATED);
        assert!(!created.is_terminal());
        assert!(created.advances_sequence());
        created.set_sequence_number(7);

        let frame = created
            .to_plaintext_sse_frame()
            .expect("created event should serialize");
        let frame = std::str::from_utf8(&frame).expect("SSE frame should be UTF-8");
        assert!(frame.starts_with("event: response.created\ndata: {"));
        assert!(frame.ends_with("\n\n"));
        let payload = frame
            .strip_prefix("event: response.created\ndata: ")
            .and_then(|frame| frame.strip_suffix("\n\n"))
            .expect("canonical plaintext SSE framing");
        let payload: serde_json::Value =
            serde_json::from_str(payload).expect("SSE data should contain JSON");
        assert_eq!(payload["type"], EVENT_RESPONSE_CREATED);
        assert_eq!(payload["sequence_number"], 7);
    }

    #[test]
    fn test_plaintext_error_event_is_an_application_terminal() {
        let event = ResponseEvent::Error(ResponseErrorEvent {
            event_type: EVENT_RESPONSE_ERROR,
            error: crate::web::responses::handlers::ResponseError {
                error_type: "stream_error".to_string(),
                message: "Stream failed".to_string(),
            },
        });

        assert!(event.is_terminal());
        assert!(!event.advances_sequence());
        assert_eq!(
            event.to_plaintext_sse_frame().unwrap(),
            Bytes::from_static(
                b"event: response.error\ndata: {\"type\":\"response.error\",\"error\":{\"type\":\"stream_error\",\"message\":\"Stream failed\"}}\n\n"
            )
        );
    }
}
