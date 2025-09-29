//! Message content conversion utilities

use crate::web::conversations::{ConversationContent, MessageContent, MessageContentPart};
use serde_json::{json, Value};

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

    /// Convert MessageContent to ConversationContent array for API responses
    ///
    /// Transforms content based on the message role (user vs assistant) to
    /// produce the appropriate ConversationContent variants.
    ///
    /// # Arguments
    /// * `content` - The content to convert
    /// * `role` - The message role ("user", "assistant", etc.)
    ///
    /// # Returns
    /// Vector of ConversationContent items
    pub fn to_conversation_content(
        content: MessageContent,
        role: &str,
    ) -> Vec<ConversationContent> {
        match (content, role) {
            (MessageContent::Text(text), "user") => {
                vec![ConversationContent::InputText { text }]
            }
            (MessageContent::Text(text), "assistant") => {
                vec![ConversationContent::OutputText { text }]
            }
            (MessageContent::Parts(parts), role) => parts
                .into_iter()
                .map(|part| Self::content_part_to_conversation(part, role))
                .collect(),
            _ => vec![],
        }
    }

    /// Convert a single MessageContentPart to ConversationContent
    fn content_part_to_conversation(part: MessageContentPart, role: &str) -> ConversationContent {
        match (part, role) {
            (
                MessageContentPart::Text { text } | MessageContentPart::InputText { text },
                "user",
            ) => ConversationContent::InputText { text },
            (
                MessageContentPart::Text { text } | MessageContentPart::InputText { text },
                "assistant",
            ) => ConversationContent::OutputText { text },
            (MessageContentPart::InputImage { image_url, .. }, _) => {
                ConversationContent::InputImage {
                    image_url: image_url.unwrap_or_else(|| "[No URL]".to_string()),
                }
            }
            (MessageContentPart::InputFile { filename, .. }, _) => {
                ConversationContent::InputFile { filename }
            }
            _ => ConversationContent::InputText {
                text: "".to_string(),
            },
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
