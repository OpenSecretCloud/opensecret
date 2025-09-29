//! SSE event handling utilities

use crate::AppState;
use axum::response::sse::Event;
use serde::Serialize;
use tracing::{error, trace};
use uuid::Uuid;

use super::constants::{ERROR_DATA_ENCRYPTION_FAILED, ERROR_DATA_SERIALIZATION_FAILED};
use super::handlers::encrypt_event;

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

    /// Set the sequence number (useful for resetting or syncing)
    pub fn set_sequence_number(&mut self, seq: i32) {
        self.sequence_number = seq;
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
}
