//! Modular Responses API implementation
//!
//! This module contains the refactored components of the Responses API,
//! separated by concern for better maintainability and testability.

pub mod builders;
pub mod constants;
pub mod context_builder;
pub mod conversions;
pub mod errors;
pub mod events;
pub mod handlers;
pub mod storage;
pub mod stream_processor;

// Re-export commonly used types
pub use builders::{build_usage, ContentPartBuilder, OutputItemBuilder, ResponseBuilder};
pub use context_builder::build_prompt;
pub use conversions::MessageContentConverter;
pub use errors::error_mapping;
pub use events::{ResponseEvent, SseEventEmitter};
pub use storage::storage_task;
pub use stream_processor::UpstreamStreamProcessor;

// Re-export the router from handlers (legacy file)
pub use handlers::router;
