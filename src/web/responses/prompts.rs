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
Classify the user's intent. Return ONLY one of these exact values:
- \"web_search\" if the user needs current information, facts, news, real-time data, or web search
- \"chat\" if the user wants casual conversation, greetings, explanations, or general discussion

Examples:
- \"What's the weather today?\" → web_search
- \"Who is the current president?\" → web_search
- \"What happened in the news today?\" → web_search
- \"Hello, how are you?\" → chat
- \"Explain how photosynthesis works\" → chat
- \"Tell me a joke\" → chat";

/// System prompt for search query extraction
///
/// This prompt instructs the LLM to extract a clean search query from the user's
/// natural language question.
pub const SEARCH_QUERY_EXTRACTOR_PROMPT: &str = "\
Extract the main search query from the user's question.
Return only the search terms, nothing else. Be concise and specific.

Examples:
- \"What's the weather in San Francisco today?\" → weather San Francisco today
- \"Who is the current president of the United States?\" → current president United States
- \"Tell me about the latest SpaceX launch\" → latest SpaceX launch";

/// Build a chat completion request for intent classification
///
/// Uses a fast, cheap model (llama-3.3-70b) with temperature=0 for deterministic results.
///
/// # Arguments
/// * `user_message` - The user's message to classify
///
/// # Returns
/// A JSON request ready to be sent to `get_chat_completion_response`
pub fn build_intent_classification_request(user_message: &str) -> Value {
    json!({
        "model": "llama-3.3-70b",
        "messages": [
            {
                "role": "system",
                "content": INTENT_CLASSIFIER_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.0,
        "max_tokens": 10,
        "stream": false
    })
}

/// Build a chat completion request for search query extraction
///
/// Uses the same fast model as classification to extract a clean search query.
///
/// # Arguments
/// * `user_message` - The user's message to extract a query from
///
/// # Returns
/// A JSON request ready to be sent to `get_chat_completion_response`
pub fn build_query_extraction_request(user_message: &str) -> Value {
    json!({
        "model": "llama-3.3-70b",
        "messages": [
            {
                "role": "system",
                "content": SEARCH_QUERY_EXTRACTOR_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "stream": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_intent_classification_request() {
        let request = build_intent_classification_request("What's the weather?");

        assert_eq!(request["model"], "llama-3.3-70b");
        assert_eq!(request["temperature"], 0.0);
        assert_eq!(request["stream"], false);

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What's the weather?");
    }

    #[test]
    fn test_build_query_extraction_request() {
        let request = build_query_extraction_request("What's the weather in New York?");

        assert_eq!(request["model"], "llama-3.3-70b");
        assert_eq!(request["max_tokens"], 50);

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["content"], "What's the weather in New York?");
    }

    #[test]
    fn test_prompts_contain_examples() {
        assert!(INTENT_CLASSIFIER_PROMPT.contains("Examples:"));
        assert!(INTENT_CLASSIFIER_PROMPT.contains("web_search"));
        assert!(INTENT_CLASSIFIER_PROMPT.contains("chat"));

        assert!(SEARCH_QUERY_EXTRACTOR_PROMPT.contains("Examples:"));
    }
}
