//! Modular Responses API implementation
//!
//! This module contains the refactored components of the Responses API,
//! separated by concern for better maintainability and testability.

pub mod builders;
pub mod constants;
pub mod context_builder;
pub mod conversations;
pub mod conversions;
pub mod errors;
pub mod events;
pub mod handlers;
pub mod pagination;
pub mod storage;
pub mod stream_processor;
pub mod types;

// Re-export commonly used types
pub use builders::{
    build_usage, ContentPartBuilder, ConversationBuilder, OutputItemBuilder, ResponseBuilder,
};
pub use context_builder::build_prompt;
pub use conversions::{ConversationItem, ConversationItemConverter, MessageContentConverter};
pub use errors::error_mapping;
pub use events::{ResponseEvent, SseEventEmitter};
pub use pagination::Paginator;
pub use storage::storage_task;
pub use stream_processor::UpstreamStreamProcessor;
pub use types::{MessageContent, MessageContentPart};

// Re-export routers
pub use conversations::router as conversations_router;
pub use handlers::router as responses_router;
