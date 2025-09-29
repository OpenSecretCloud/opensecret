//! Builder patterns for complex response structures

use crate::{
    models::responses::Response,
    web::responses::{
        constants::*,
        handlers::{
            ContentPart, InputTokenDetails, OutputItem, OutputTokenDetails, ReasoningInfo,
            ResponseUsage, ResponsesCreateResponse, TextFormat, TextFormatSpec,
        },
    },
};
use serde_json::Value;
use uuid::Uuid;

/// Builder for ResponsesCreateResponse with sensible defaults
///
/// This builder eliminates ~80 lines of duplicated response construction code
/// by providing a fluent API with all the boilerplate pre-filled.
///
/// # Example
/// ```ignore
/// let response_obj = ResponseBuilder::from_response(&response)
///     .status(STATUS_COMPLETED)
///     .output(output_items)
///     .usage(usage)
///     .metadata(metadata)
///     .build();
/// ```
pub struct ResponseBuilder {
    response: ResponsesCreateResponse,
}

impl ResponseBuilder {
    /// Create a new builder from a database Response model
    ///
    /// Sets all fields with sensible defaults based on the Response record.
    /// Most fields can be overridden with builder methods.
    ///
    /// # Arguments
    /// * `response` - The database Response model to build from
    pub fn from_response(response: &Response) -> Self {
        Self {
            response: ResponsesCreateResponse {
                id: response.uuid,
                object: OBJECT_TYPE_RESPONSE,
                created_at: response.created_at.timestamp(),
                status: STATUS_IN_PROGRESS, // Default to in_progress
                background: false,
                error: None,
                incomplete_details: None,
                instructions: None,
                max_output_tokens: response.max_output_tokens,
                max_tool_calls: None,
                model: response.model.clone(),
                output: vec![], // Default to empty output
                parallel_tool_calls: response.parallel_tool_calls,
                previous_response_id: None,
                prompt_cache_key: None,
                reasoning: ReasoningInfo {
                    effort: None,
                    summary: None,
                },
                safety_identifier: None,
                store: response.store,
                temperature: response.temperature.unwrap_or(1.0),
                text: TextFormat {
                    format: TextFormatSpec {
                        format_type: TEXT_FORMAT_TYPE.to_string(),
                    },
                },
                tool_choice: response
                    .tool_choice
                    .clone()
                    .unwrap_or_else(|| TOOL_CHOICE_AUTO.to_string()),
                tools: vec![],
                top_logprobs: 0,
                top_p: response.top_p.unwrap_or(1.0),
                truncation: TRUNCATION_DISABLED,
                usage: None, // Default to None
                user: None,
                metadata: None, // Default to None
            },
        }
    }

    /// Set the status of the response
    ///
    /// # Arguments
    /// * `status` - Status constant (e.g., STATUS_IN_PROGRESS, STATUS_COMPLETED)
    pub fn status(mut self, status: &'static str) -> Self {
        self.response.status = status;
        self
    }

    /// Set the output items
    ///
    /// # Arguments
    /// * `output` - Vector of OutputItem objects
    pub fn output(mut self, output: Vec<OutputItem>) -> Self {
        self.response.output = output;
        self
    }

    /// Set the usage statistics
    ///
    /// # Arguments
    /// * `usage` - ResponseUsage with token counts
    pub fn usage(mut self, usage: ResponseUsage) -> Self {
        self.response.usage = Some(usage);
        self
    }

    /// Set the metadata
    ///
    /// # Arguments
    /// * `metadata` - Optional JSON metadata
    pub fn metadata(mut self, metadata: Option<Value>) -> Self {
        self.response.metadata = metadata;
        self
    }

    /// Build the final ResponsesCreateResponse
    ///
    /// Consumes the builder and returns the constructed response.
    pub fn build(self) -> ResponsesCreateResponse {
        self.response
    }
}

/// Builder for OutputItem (assistant message output)
///
/// Provides a convenient way to construct output items for the response.
pub struct OutputItemBuilder {
    item: OutputItem,
}

impl OutputItemBuilder {
    /// Create a new message output item
    ///
    /// # Arguments
    /// * `message_id` - UUID of the assistant message
    pub fn new_message(message_id: Uuid) -> Self {
        Self {
            item: OutputItem {
                id: message_id.to_string(),
                output_type: OUTPUT_TYPE_MESSAGE.to_string(),
                status: STATUS_IN_PROGRESS.to_string(),
                role: Some(ROLE_ASSISTANT.to_string()),
                content: Some(vec![]),
            },
        }
    }

    /// Set the status of the output item
    pub fn status(mut self, status: &str) -> Self {
        self.item.status = status.to_string();
        self
    }

    /// Set the content parts
    pub fn content(mut self, content: Vec<ContentPart>) -> Self {
        self.item.content = Some(content);
        self
    }

    /// Build the final OutputItem
    pub fn build(self) -> OutputItem {
        self.item
    }
}

/// Builder for ContentPart (output text)
///
/// Simplifies creation of content parts with consistent defaults.
pub struct ContentPartBuilder {
    part: ContentPart,
}

impl ContentPartBuilder {
    /// Create a new output text content part
    ///
    /// # Arguments
    /// * `text` - The text content
    pub fn new_output_text(text: String) -> Self {
        Self {
            part: ContentPart {
                part_type: CONTENT_PART_TYPE_OUTPUT_TEXT.to_string(),
                annotations: vec![],
                logprobs: vec![],
                text,
            },
        }
    }

    /// Build the final ContentPart
    pub fn build(self) -> ContentPart {
        self.part
    }
}

/// Helper function to build ResponseUsage from token counts
///
/// # Arguments
/// * `prompt_tokens` - Number of input tokens
/// * `completion_tokens` - Number of output tokens
///
/// # Returns
/// ResponseUsage with proper structure and totals
pub fn build_usage(prompt_tokens: i32, completion_tokens: i32) -> ResponseUsage {
    ResponseUsage {
        input_tokens: prompt_tokens,
        input_tokens_details: InputTokenDetails { cached_tokens: 0 },
        output_tokens: completion_tokens,
        output_tokens_details: OutputTokenDetails {
            reasoning_tokens: 0,
        },
        total_tokens: prompt_tokens + completion_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::responses::ResponseStatus;
    use chrono::Utc;

    fn create_test_response() -> Response {
        Response {
            id: 1,
            uuid: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            conversation_id: 1,
            status: ResponseStatus::InProgress,
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            top_p: Some(1.0),
            max_output_tokens: Some(1000),
            tool_choice: Some("auto".to_string()),
            parallel_tool_calls: false,
            store: true,
            metadata_enc: None,
            created_at: Utc::now(),
            completed_at: None,
            updated_at: Utc::now(),
            input_tokens: None,
            output_tokens: None,
        }
    }

    #[test]
    fn test_builder_defaults() {
        let response = create_test_response();
        let built = ResponseBuilder::from_response(&response).build();

        assert_eq!(built.id, response.uuid);
        assert_eq!(built.status, STATUS_IN_PROGRESS);
        assert_eq!(built.model, "gpt-4");
        assert_eq!(built.output.len(), 0);
        assert!(built.usage.is_none());
        assert!(built.metadata.is_none());
    }

    #[test]
    fn test_builder_fluent_api() {
        let response = create_test_response();
        let usage = build_usage(100, 50);

        let built = ResponseBuilder::from_response(&response)
            .status(STATUS_COMPLETED)
            .output(vec![])
            .usage(usage.clone())
            .metadata(Some(serde_json::json!({"test": "data"})))
            .build();

        assert_eq!(built.status, STATUS_COMPLETED);
        assert!(built.usage.is_some());
        assert_eq!(built.usage.as_ref().unwrap().input_tokens, 100);
        assert_eq!(built.usage.as_ref().unwrap().output_tokens, 50);
        assert_eq!(built.usage.as_ref().unwrap().total_tokens, 150);
        assert!(built.metadata.is_some());
    }

    #[test]
    fn test_output_item_builder() {
        let message_id = Uuid::new_v4();
        let content_part = ContentPartBuilder::new_output_text("Hello".to_string()).build();

        let item = OutputItemBuilder::new_message(message_id)
            .status(STATUS_COMPLETED)
            .content(vec![content_part])
            .build();

        assert_eq!(item.id, message_id.to_string());
        assert_eq!(item.status, STATUS_COMPLETED);
        assert_eq!(item.output_type, OUTPUT_TYPE_MESSAGE);
        assert_eq!(item.content.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_content_part_builder() {
        let part = ContentPartBuilder::new_output_text("Test content".to_string()).build();

        assert_eq!(part.part_type, CONTENT_PART_TYPE_OUTPUT_TEXT);
        assert_eq!(part.text, "Test content");
        assert_eq!(part.annotations.len(), 0);
        assert_eq!(part.logprobs.len(), 0);
    }

    #[test]
    fn test_build_usage_helper() {
        let usage = build_usage(100, 50);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }
}
