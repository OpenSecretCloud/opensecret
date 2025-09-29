//! Modular Responses API implementation
//!
//! This module contains the refactored components of the Responses API,
//! separated by concern for better maintainability and testability.

pub mod constants;
pub mod conversions;
pub mod errors;
pub mod events;
pub mod handlers;
pub mod storage;
pub mod stream_processor;

// Re-export commonly used types
pub use constants::*;
pub use conversions::MessageContentConverter;
pub use errors::error_mapping;
pub use events::SseEventEmitter;
pub use storage::{storage_task, BillingEventPublisher, ContentAccumulator, ResponsePersister};
pub use stream_processor::UpstreamStreamProcessor;

// Re-export the router from handlers (legacy file)
pub use handlers::router;
