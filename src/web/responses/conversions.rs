//! Message content conversion utilities

use crate::{
    encrypt::decrypt_string,
    models::responses::RawThreadMessage,
    web::responses::{constants::*, error_mapping, types::*},
    ApiError,
};
use secp256k1::SecretKey;
use serde_json::{json, Value};
use tracing::error;
use uuid::Uuid;

/// Centralized message content conversion utilities
///
/// This struct provides a single source of truth for converting between
/// different message content formats used throughout the API:
/// - User input normalization
/// - OpenAI API format conversion
/// - Conversation API format conversion
/// - Token counting text extraction
pub struct MessageContentConverter;

impl MessageContentConverter {
    /// Validate MessageContent parts to ensure unsupported features are rejected
    ///
    /// Currently validates:
    /// - file_id is not supported in InputImage (only image_url)
    ///
    /// # Arguments
    /// * `content` - The content to validate
    ///
    /// # Returns
    /// Ok(()) if valid, Err(ApiError) if validation fails
    pub fn validate_content(content: &MessageContent) -> Result<(), ApiError> {
        if let MessageContent::Parts(parts) = content {
            for part in parts {
                if let MessageContentPart::InputImage {
                    file_id, image_url, ..
                } = part
                {
                    if file_id.is_some() {
                        return Err(ApiError::BadRequest);
                    }
                    if image_url.is_none() {
                        return Err(ApiError::BadRequest);
                    }
                }
            }
        }
        Ok(())
    }

    /// Normalize MessageContent to always use Parts format
    ///
    /// Converts simple Text format to Parts with InputText, ensuring
    /// consistent internal representation.
    ///
    /// # Arguments
    /// * `content` - The content to normalize
    ///
    /// # Returns
    /// MessageContent in Parts format
    pub fn normalize_content(content: MessageContent) -> MessageContent {
        match content {
            MessageContent::Text(text) => {
                MessageContent::Parts(vec![MessageContentPart::InputText { text }])
            }
            MessageContent::Parts(parts) => MessageContent::Parts(parts),
        }
    }

    /// Convert MessageContent to OpenAI API format for chat completions
    ///
    /// Transforms our internal MessageContent representation into the format
    /// expected by OpenAI's Chat Completions API.
    ///
    /// # Arguments
    /// * `content` - The content to convert
    ///
    /// # Returns
    /// JSON Value in OpenAI format
    pub fn to_openai_format(content: &MessageContent) -> Value {
        match content {
            MessageContent::Text(text) => json!(text),
            MessageContent::Parts(parts) => {
                let openai_parts: Vec<Value> =
                    parts.iter().map(Self::content_part_to_openai).collect();
                json!(openai_parts)
            }
        }
    }

    /// Convert a single MessageContentPart to OpenAI format
    fn content_part_to_openai(part: &MessageContentPart) -> Value {
        match part {
            MessageContentPart::Text { text } | MessageContentPart::InputText { text } => {
                json!({
                    "type": "text",
                    "text": text
                })
            }
            MessageContentPart::InputImage {
                image_url, detail, ..
            } => {
                let mut image_obj = json!({
                    "url": image_url.as_ref().unwrap_or(&"".to_string())
                });
                if let Some(d) = detail {
                    image_obj["detail"] = json!(d);
                }
                json!({
                    "type": "image_url",
                    "image_url": image_obj
                })
            }
            MessageContentPart::InputFile {
                filename,
                file_data,
            } => {
                json!({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": file_data
                    }
                })
            }
        }
    }

    /// Convert assistant text to conversation content
    ///
    /// Helper specifically for assistant messages which are stored as plain text
    /// rather than structured MessageContent.
    ///
    /// # Arguments
    /// * `text` - The assistant's text response
    ///
    /// # Returns
    /// Vector containing a single OutputText content part
    pub fn assistant_text_to_content(text: String) -> Vec<ConversationContent> {
        vec![ConversationContent::OutputText { text }]
    }

    /// Extract text content for token counting purposes only
    ///
    /// Concatenates all text parts (ignoring images/files) to produce a string
    /// suitable for token counting. This does NOT represent the full content
    /// structure - use only for token estimation.
    ///
    /// # Arguments
    /// * `content` - The content to extract text from
    ///
    /// # Returns
    /// Concatenated text from all text content parts
    pub fn extract_text_for_token_counting(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    MessageContentPart::Text { text } => Some(text.clone()),
                    MessageContentPart::InputText { text } => Some(text.clone()),
                    MessageContentPart::InputImage { .. } => None, // Ignore images
                    MessageContentPart::InputFile { .. } => None,  // Ignore files
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

// ============================================================================
// Conversation Item Converter
// ============================================================================

/// Conversation item used in the Conversations API
///
/// This is defined here to avoid circular dependencies since it's only used
/// for conversion logic.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ConversationItem {
    #[serde(rename = "message")]
    Message {
        id: Uuid,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        role: String,
        content: Vec<ConversationContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: Uuid,
        call_id: Uuid,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
    #[serde(rename = "function_tool_call_output")]
    FunctionToolCallOutput {
        id: Uuid,
        call_id: Uuid,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        created_at: Option<i64>,
    },
}

/// Centralized conversation item conversion utilities
///
/// Converts database Message models to ConversationItem API types,
/// handling decryption and format conversion for all message types.
pub struct ConversationItemConverter;

impl ConversationItemConverter {
    /// Convert database message to ConversationItem
    ///
    /// Handles decryption and format conversion for all message types.
    ///
    /// # Arguments
    /// * `msg` - The database message to convert
    /// * `user_key` - User's encryption key for decrypting content
    ///
    /// # Returns
    /// ConversationItem ready for API response
    ///
    /// # Errors
    /// Returns ApiError if decryption or deserialization fails
    pub fn message_to_item(
        msg: &RawThreadMessage,
        user_key: &SecretKey,
    ) -> Result<ConversationItem, ApiError> {
        // Decrypt content (handle nullable content_enc)
        let content = decrypt_string(user_key, msg.content_enc.as_ref())
            .map_err(|_| error_mapping::map_decryption_error("message content"))?
            .unwrap_or_default();

        match msg.message_type.as_str() {
            "user" => Self::user_message_to_item(msg, content),
            "assistant" => Self::assistant_message_to_item(msg, content),
            "tool_call" => Self::tool_call_to_item(msg, content),
            "tool_output" => Self::tool_output_to_item(msg, content),
            unknown => {
                error!("Unknown message type: {}", unknown);
                Err(ApiError::InternalServerError)
            }
        }
    }

    /// Convert user message to item
    fn user_message_to_item(
        msg: &RawThreadMessage,
        content: String,
    ) -> Result<ConversationItem, ApiError> {
        // User messages MUST be stored as MessageContent
        let message_content: MessageContent = serde_json::from_str(&content).map_err(|e| {
            error!("Failed to deserialize message content: {:?}", e);
            ApiError::InternalServerError
        })?;

        Ok(ConversationItem::Message {
            id: msg.uuid,
            status: msg.status.clone(),
            role: ROLE_USER.to_string(),
            content: Vec::<ConversationContent>::from(message_content),
            created_at: Some(msg.created_at.timestamp()),
        })
    }

    /// Convert assistant message to item
    fn assistant_message_to_item(
        msg: &RawThreadMessage,
        content: String,
    ) -> Result<ConversationItem, ApiError> {
        // Assistant messages are plain text strings
        // If content is empty (in_progress), return empty content array
        let content_parts = if content.is_empty() {
            vec![]
        } else {
            MessageContentConverter::assistant_text_to_content(content)
        };

        Ok(ConversationItem::Message {
            id: msg.uuid,
            status: msg.status.clone(),
            role: ROLE_ASSISTANT.to_string(),
            content: content_parts,
            created_at: Some(msg.created_at.timestamp()),
        })
    }

    /// Convert tool call to item
    fn tool_call_to_item(
        msg: &RawThreadMessage,
        content: String,
    ) -> Result<ConversationItem, ApiError> {
        Ok(ConversationItem::FunctionToolCall {
            id: msg.uuid,
            call_id: msg.tool_call_id.ok_or_else(|| {
                error!("tool_call_id missing for tool call");
                ApiError::InternalServerError
            })?,
            name: DEFAULT_TOOL_FUNCTION_NAME.to_string(),
            arguments: content,
            status: msg.status.clone(),
            created_at: Some(msg.created_at.timestamp()),
        })
    }

    /// Convert tool output to item
    fn tool_output_to_item(
        msg: &RawThreadMessage,
        content: String,
    ) -> Result<ConversationItem, ApiError> {
        Ok(ConversationItem::FunctionToolCallOutput {
            id: msg.uuid,
            call_id: msg.tool_call_id.ok_or_else(|| {
                error!("tool_call_id missing for tool output");
                ApiError::InternalServerError
            })?,
            output: content,
            status: msg.status.clone(),
            created_at: Some(msg.created_at.timestamp()),
        })
    }

    /// Convert multiple messages to items with pagination
    ///
    /// Processes messages starting from an offset and applies limit.
    ///
    /// # Arguments
    /// * `raw_messages` - All messages from the conversation
    /// * `user_key` - User's encryption key
    /// * `start_index` - Index to start from (for cursor-based pagination)
    /// * `limit` - Maximum number of items to return
    ///
    /// # Returns
    /// Vector of ConversationItems
    ///
    /// # Errors
    /// Returns ApiError if any message conversion fails
    pub fn messages_to_items(
        raw_messages: &[RawThreadMessage],
        user_key: &SecretKey,
        start_index: usize,
        limit: usize,
    ) -> Result<Vec<ConversationItem>, ApiError> {
        let mut items = Vec::new();

        for msg in raw_messages.iter().skip(start_index).take(limit) {
            items.push(Self::message_to_item(msg, user_key)?);
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text_to_parts() {
        let content = MessageContent::Text("hello".to_string());
        let normalized = MessageContentConverter::normalize_content(content);

        match normalized {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                match &parts[0] {
                    MessageContentPart::InputText { text } => {
                        assert_eq!(text, "hello");
                    }
                    _ => panic!("Expected InputText"),
                }
            }
            _ => panic!("Expected Parts"),
        }
    }

    #[test]
    fn test_normalize_parts_unchanged() {
        let content = MessageContent::Parts(vec![MessageContentPart::InputText {
            text: "hello".to_string(),
        }]);
        let normalized = MessageContentConverter::normalize_content(content);

        match normalized {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
            }
            _ => panic!("Expected Parts"),
        }
    }

    #[test]
    fn test_extract_text_for_token_counting() {
        let content = MessageContent::Parts(vec![
            MessageContentPart::InputText {
                text: "hello".to_string(),
            },
            MessageContentPart::InputText {
                text: "world".to_string(),
            },
        ]);

        let text = MessageContentConverter::extract_text_for_token_counting(&content);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_extract_text_ignores_images() {
        let content = MessageContent::Parts(vec![
            MessageContentPart::InputText {
                text: "hello".to_string(),
            },
            MessageContentPart::InputImage {
                image_url: Some("http://example.com/image.jpg".to_string()),
                file_id: None,
                detail: None,
            },
            MessageContentPart::InputText {
                text: "world".to_string(),
            },
        ]);

        let text = MessageContentConverter::extract_text_for_token_counting(&content);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_to_openai_format_text() {
        let content = MessageContent::Text("hello".to_string());
        let openai = MessageContentConverter::to_openai_format(&content);
        assert_eq!(openai, json!("hello"));
    }

    #[test]
    fn test_to_openai_format_parts() {
        let content = MessageContent::Parts(vec![MessageContentPart::InputText {
            text: "hello".to_string(),
        }]);
        let openai = MessageContentConverter::to_openai_format(&content);

        assert!(openai.is_array());
        assert_eq!(openai[0]["type"], "text");
        assert_eq!(openai[0]["text"], "hello");
    }

    #[test]
    fn test_assistant_text_to_content() {
        let content = MessageContentConverter::assistant_text_to_content("response".to_string());
        assert_eq!(content.len(), 1);
        match &content[0] {
            ConversationContent::OutputText { text } => {
                assert_eq!(text, "response");
            }
            _ => panic!("Expected OutputText"),
        }
    }
}
