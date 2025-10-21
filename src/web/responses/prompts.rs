//! Prompt templates for the Responses API
//!
//! This module contains all prompt templates used for intent classification,
//! query extraction, and other AI-driven features of the Responses API.

use serde_json::{json, Value};

/// System prompt for intent classification
///
/// This prompt instructs the LLM to classify whether a user's message requires
/// web search or can be handled as a regular chat conversation.
pub const INTENT_CLASSIFIER_PROMPT: &str = "\
Classify whether to search the web before responding: \"web_search\" or \"chat\". Return \"web_search\" if web search is needed, \"chat\" if you can answer directly. \
Pay special attention to the context of the conversation history and the nature of the inquiry. If the user asks for current events, factual information, or specific data that may have changed recently, classify as \"web_search\". \
For casual conversations or subjective inquiries, classify as \"chat\". Identify inquiries that may seem casual but actually require factual information, such as popular culture references or specific names that are currently relevant. \
Recognize when a casual inquiry is part of a broader context that may require factual information, and adjust the classification accordingly.";

/// System prompt for search query extraction
///
/// This prompt instructs the LLM to extract a clean search query from the user's
/// natural language question, using conversation history for context.
pub const SEARCH_QUERY_EXTRACTOR_PROMPT: &str = "\
Extract the main search query from the user's question. Use the conversation history to understand context and references.
Return only the search terms, nothing else. Be concise and specific.
If the user's question refers to something mentioned earlier in the conversation, include that context in your search query.

Examples:
- \"What's the weather in San Francisco today?\" → weather San Francisco today
- \"Who is the current president of the United States?\" → current president United States
- \"Tell me about the latest SpaceX launch\" → latest SpaceX launch
- After discussing \"iPhone 15\", user asks \"when was it released?\" → iPhone 15 release date";

/// Build a chat completion request for intent classification
///
/// Uses a fast, cheap model (gpt-oss-120b) with temperature=0 for deterministic results.
///
/// # Arguments
/// * `conversation_history` - Recent conversation messages (will use last 6, truncated to 200 chars each)
/// * `user_message` - The current user's message to classify
///
/// # Returns
/// A JSON request ready to be sent to `get_chat_completion_response`
pub fn build_intent_classification_request(
    conversation_history: &[Value],
    user_message: &str,
) -> Value {
    // Format conversation history as text for context
    let history_text = if !conversation_history.is_empty() {
        let formatted_messages: Vec<String> = conversation_history
            .iter()
            .rev()
            .take(6)
            .rev()
            .filter_map(|msg| {
                let role = msg.get("role")?.as_str()?;
                let content = extract_text_from_content(msg.get("content")?);
                let truncated_content: String = content.chars().take(200).collect();
                Some(format!("{}: {}", role, truncated_content))
            })
            .collect();

        if formatted_messages.is_empty() {
            String::new()
        } else {
            format!(
                "Conversation history:\n{}\n\n",
                formatted_messages.join("\n")
            )
        }
    } else {
        String::new()
    };

    // Build single user message with history + current query
    let user_prompt = format!("{}Current user query: {}", history_text, user_message);

    json!({
        "model": "gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": INTENT_CLASSIFIER_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 250,
        "stream": false
    })
}

/// Helper function to extract text from content (handles both string and array formats)
fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Build a chat completion request for search query extraction
///
/// Uses the same fast model as classification to extract a clean search query.
///
/// # Arguments
/// * `conversation_history` - Recent conversation messages for context
/// * `user_message` - The user's message to extract a query from
///
/// # Returns
/// A JSON request ready to be sent to `get_chat_completion_response`
pub fn build_query_extraction_request(conversation_history: &[Value], user_message: &str) -> Value {
    // Format conversation history as text for context
    let history_text = if !conversation_history.is_empty() {
        let formatted_messages: Vec<String> = conversation_history
            .iter()
            .rev()
            .take(6)
            .rev()
            .filter_map(|msg| {
                let role = msg.get("role")?.as_str()?;
                let content = extract_text_from_content(msg.get("content")?);
                let truncated_content: String = content.chars().take(200).collect();
                Some(format!("{}: {}", role, truncated_content))
            })
            .collect();

        if formatted_messages.is_empty() {
            String::new()
        } else {
            format!(
                "Conversation history:\n{}\n\n",
                formatted_messages.join("\n")
            )
        }
    } else {
        String::new()
    };

    // Build single user message with history + current query
    let user_prompt = format!("{}Current user question: {}", history_text, user_message);

    json!({
        "model": "gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": SEARCH_QUERY_EXTRACTOR_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 250,
        "stream": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_intent_classification_request() {
        let request = build_intent_classification_request(&[], "What's the weather?");

        assert_eq!(request["model"], "gpt-oss-120b");
        assert_eq!(request["temperature"], 0.0);
        assert_eq!(request["stream"], false);

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        // Should have formatted user query
        let user_content = messages[1]["content"].as_str().unwrap();
        assert_eq!(user_content, "Current user query: What's the weather?");
    }

    #[test]
    fn test_build_intent_classification_with_history() {
        let history = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there!"}),
            json!({"role": "user", "content": "How are you?"}),
        ];

        let request = build_intent_classification_request(&history, "What's the weather?");

        let messages = request["messages"].as_array().unwrap();
        // Should have only 2 messages: system + single user message with history
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");

        let user_content = messages[1]["content"].as_str().unwrap();
        // Should contain formatted history
        assert!(user_content.contains("Conversation history:"));
        assert!(user_content.contains("user: Hello"));
        assert!(user_content.contains("assistant: Hi there!"));
        assert!(user_content.contains("user: How are you?"));
        assert!(user_content.contains("Current user query: What's the weather?"));
    }

    #[test]
    fn test_build_intent_classification_truncates_long_messages() {
        let long_text = "a".repeat(500);
        let history = vec![json!({"role": "user", "content": long_text})];

        let request = build_intent_classification_request(&history, "test");

        let messages = request["messages"].as_array().unwrap();
        let user_content = messages[1]["content"].as_str().unwrap();

        // History should be truncated to 200 chars per message
        // Format is: "Conversation history:\nuser: <200 chars>\n\nCurrent user query: test"
        assert!(user_content.contains("Conversation history:"));
        assert!(user_content.contains("user: "));
        assert!(user_content.contains("Current user query: test"));

        // Extract just the history part to verify truncation
        let history_part = user_content.split("Current user query:").next().unwrap();
        // The 'aaa...' part should be truncated (much less than original 500 chars)
        let a_count = history_part.chars().filter(|&c| c == 'a').count();
        // Should be around 200 (within reasonable bounds, accounting for any formatting)
        assert!(
            (190..=210).contains(&a_count),
            "Expected around 200 'a' characters, got {}",
            a_count
        );
        // Definitely should be much less than the original 500
        assert!(
            a_count < 300,
            "Truncation failed: got {} 'a' characters (original was 500)",
            a_count
        );
    }

    #[test]
    fn test_build_intent_classification_limits_to_6_messages() {
        let history: Vec<Value> = (0..10)
            .map(|i| json!({"role": "user", "content": format!("Message {}", i)}))
            .collect();

        let request = build_intent_classification_request(&history, "test");

        let messages = request["messages"].as_array().unwrap();
        // Should have only 2 messages: system + single user message
        assert_eq!(messages.len(), 2);

        let user_content = messages[1]["content"].as_str().unwrap();
        // Should have last 6 messages (4-9)
        assert!(user_content.contains("user: Message 4"));
        assert!(user_content.contains("user: Message 9"));
        // Should not have earlier messages
        assert!(!user_content.contains("user: Message 0"));
        assert!(!user_content.contains("user: Message 3"));
    }

    #[test]
    fn test_build_query_extraction_request() {
        let request = build_query_extraction_request(&[], "What's the weather in New York?");

        assert_eq!(request["model"], "gpt-oss-120b");
        assert_eq!(request["max_tokens"], 250);

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);

        let user_content = messages[1]["content"].as_str().unwrap();
        assert_eq!(
            user_content,
            "Current user question: What's the weather in New York?"
        );
    }

    #[test]
    fn test_build_query_extraction_with_context() {
        let history = vec![
            json!({"role": "user", "content": "Tell me about iPhone 15"}),
            json!({"role": "assistant", "content": "The iPhone 15 was announced in September 2023..."}),
        ];

        let request = build_query_extraction_request(&history, "when was it released?");

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);

        let user_content = messages[1]["content"].as_str().unwrap();
        assert!(user_content.contains("Conversation history:"));
        assert!(user_content.contains("iPhone 15"));
        assert!(user_content.contains("Current user question: when was it released?"));
    }

    #[test]
    fn test_prompts_contain_examples() {
        assert!(INTENT_CLASSIFIER_PROMPT.contains("web_search"));
        assert!(INTENT_CLASSIFIER_PROMPT.contains("chat"));
        assert!(INTENT_CLASSIFIER_PROMPT.contains("conversation history"));

        assert!(SEARCH_QUERY_EXTRACTOR_PROMPT.contains("Examples:"));
    }
}
