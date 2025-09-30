//! Constants used throughout the Responses API

/// Channel buffer sizes
pub const STORAGE_CHANNEL_BUFFER: usize = 1024;
pub const CLIENT_CHANNEL_BUFFER: usize = 1024;

/// SSE buffer sizing
pub const SSE_BUFFER_CAPACITY: usize = 8192;

/// Default values
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const DEFAULT_MAX_TOKENS: i32 = 10_000;

/// Cost per token (in dollars)
pub const COST_PER_TOKEN: &str = "0.0000053";

/// Response object types
pub const OBJECT_TYPE_RESPONSE: &str = "response";
pub const OBJECT_TYPE_RESPONSE_DELETED: &str = "response.deleted";
pub const OBJECT_TYPE_CONVERSATION: &str = "conversation";
pub const OBJECT_TYPE_LIST: &str = "list";

/// Event types for SSE streaming
pub const EVENT_RESPONSE_CREATED: &str = "response.created";
pub const EVENT_RESPONSE_IN_PROGRESS: &str = "response.in_progress";
pub const EVENT_RESPONSE_OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub const EVENT_RESPONSE_CONTENT_PART_ADDED: &str = "response.content_part.added";
pub const EVENT_RESPONSE_OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
pub const EVENT_RESPONSE_OUTPUT_TEXT_DONE: &str = "response.output_text.done";
pub const EVENT_RESPONSE_CONTENT_PART_DONE: &str = "response.content_part.done";
pub const EVENT_RESPONSE_OUTPUT_ITEM_DONE: &str = "response.output_item.done";
pub const EVENT_RESPONSE_COMPLETED: &str = "response.completed";
pub const EVENT_RESPONSE_CANCELLED: &str = "response.cancelled";
pub const EVENT_RESPONSE_ERROR: &str = "response.error";

/// Error event data
pub const ERROR_DATA_ENCRYPTION_FAILED: &str = "encryption_failed";
pub const ERROR_DATA_SERIALIZATION_FAILED: &str = "serialization_failed";

/// Message statuses
pub const STATUS_IN_PROGRESS: &str = "in_progress";
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_INCOMPLETE: &str = "incomplete";
pub const STATUS_CANCELLED: &str = "cancelled";

/// Finish reasons
pub const FINISH_REASON_STOP: &str = "stop";
pub const FINISH_REASON_CANCELLED: &str = "cancelled";

/// Text format types
pub const TEXT_FORMAT_TYPE: &str = "text";

/// Truncation strategies
pub const TRUNCATION_DISABLED: &str = "disabled";

/// Tool choice strategies
pub const TOOL_CHOICE_AUTO: &str = "auto";

/// Output item types
pub const OUTPUT_TYPE_MESSAGE: &str = "message";

/// Content part types
pub const CONTENT_PART_TYPE_OUTPUT_TEXT: &str = "output_text";

/// Message roles
pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_SYSTEM: &str = "system";
