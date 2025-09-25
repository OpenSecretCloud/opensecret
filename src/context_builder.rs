//! Build the ChatCompletion prompt array respecting token limits.

use crate::encrypt::decrypt_with_key;
use crate::tokens::{count_tokens, model_max_ctx};
use crate::DBConnection;
use serde_json::json;
use tracing::error;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub role: &'static str,
    pub content: String,
    /// Only tool_output messages need an ID.
    pub tool_call_id: Option<Uuid>,
    pub tok: usize,
}

/// Return (messages, total_prompt_tokens)
pub fn build_prompt<D: DBConnection + ?Sized>(
    db: &D,
    conversation_id: i64,
    user_key: &secp256k1::SecretKey,
    model: &str,
) -> Result<(Vec<serde_json::Value>, usize), crate::ApiError> {
    // 1. Pull every stored message (already in chrono order ASC)
    let raw = db
        .get_conversation_context_messages(conversation_id)
        .map_err(|_| crate::ApiError::InternalServerError)?;

    // 2. Decrypt + map to ChatMsg
    let mut msgs: Vec<ChatMsg> = Vec::with_capacity(raw.len() + 1);
    for r in raw {
        let plain = decrypt_with_key(user_key, &r.content_enc)
            .map_err(|_| crate::ApiError::InternalServerError)?;
        let content = String::from_utf8_lossy(&plain).into_owned();
        let role = match r.message_type.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            "tool_output" => "tool",
            _ => continue, // Skip tool_call itself
        };
        let t = r
            .token_count
            .map(|v| v as usize)
            .unwrap_or_else(|| count_tokens(&content));
        msgs.push(ChatMsg {
            role,
            content,
            tool_call_id: r.tool_call_id,
            tok: t,
        });
    }

    // Note: In the Responses API flow, the current user message is already persisted
    // to the database before this function is called, so it's included in the
    // get_conversation_context_messages result above.

    // Delegate to the pure function for testing
    build_prompt_from_chat_messages(msgs, model)
}

/// Build prompt from chat messages - pure function for testing
pub fn build_prompt_from_chat_messages(
    mut msgs: Vec<ChatMsg>,
    model: &str,
) -> Result<(Vec<serde_json::Value>, usize), crate::ApiError> {
    // 4. Truncation
    let max_ctx = model_max_ctx(model);
    let response_reserve = 4096usize;
    let safety = 500usize;
    let ctx_budget = max_ctx.saturating_sub(response_reserve + safety);

    let mut total: usize = msgs.iter().map(|m| m.tok).sum();

    if total > ctx_budget {
        // Middle truncation: keep first 3 + truncate middle
        let mut head: Vec<ChatMsg> = Vec::new();
        let mut tail: Vec<ChatMsg> = Vec::new();
        let mut tail_tokens = 0usize;

        // Always keep the first messages (system + first user + first assistant if present)
        let mut has_system = false;
        let mut has_user = false;
        let mut has_assistant = false;

        for m in &msgs {
            if (m.role == "system" && !has_system)
                || (m.role == "user" && !has_user)
                || (m.role == "assistant" && !has_assistant)
            {
                head.push(m.clone());
                match m.role {
                    "system" => has_system = true,
                    "user" => has_user = true,
                    "assistant" => has_assistant = true,
                    _ => {}
                }
            }
            if has_system && has_user && has_assistant {
                break;
            }
        }

        let head_tokens: usize = head.iter().map(|m| m.tok).sum();
        let truncation_msg_tokens =
            count_tokens("[Previous messages truncated due to context limits]");

        // Calculate how many tokens we have left for the tail
        let available_for_tail = ctx_budget.saturating_sub(head_tokens + truncation_msg_tokens);

        // Collect messages from the end until we hit the budget
        for m in msgs.iter().rev() {
            if tail_tokens + m.tok > available_for_tail {
                break;
            }
            tail.push(m.clone());
            tail_tokens += m.tok;
        }
        tail.reverse();

        // Reconstruct with truncation message
        msgs = head;

        // Only add truncation message if we're actually truncating something
        if !tail.is_empty() || msgs.len() < msgs.len() {
            // Check last role to avoid duplicate user messages
            let last_role = msgs.last().map(|m| m.role).unwrap_or("");
            let truncation_role = if last_role == "user" {
                "assistant"
            } else {
                "user"
            };

            msgs.push(ChatMsg {
                role: truncation_role,
                content: "[Previous messages truncated due to context limits]".to_string(),
                tool_call_id: None,
                tok: truncation_msg_tokens,
            });
        }

        msgs.extend(tail);
    }

    // Recalculate total after potential truncation
    total = msgs.iter().map(|m| m.tok).sum();

    // Safety check
    debug_assert!(
        total <= ctx_budget,
        "Token count {} exceeds budget {}",
        total,
        ctx_budget
    );

    // 5. Convert to JSON array required by chat API
    let mut final_msgs = Vec::new();
    for m in msgs {
        let msg = if m.role == "tool" {
            // tool_call_id should always be present for tool messages
            debug_assert!(
                m.tool_call_id.is_some(),
                "tool_call_id must be present for tool output messages"
            );

            json!({
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| {
                        error!("tool_call_id missing for tool output message");
                        "error-missing-tool-call-id".to_string()
                    })
            })
        } else {
            // Deserialize stored MessageContent and convert to OpenAI format
            let content = if m.role == "user" {
                // User messages are stored as MessageContent - convert to OpenAI format
                use crate::web::conversations::MessageContent;
                let mc: MessageContent = serde_json::from_str(&m.content).map_err(|e| {
                    error!("Failed to deserialize user message content: {:?}", e);
                    crate::ApiError::InternalServerError
                })?;
                mc.to_openai_format()
            } else {
                // Assistant messages are plain strings
                serde_json::Value::String(m.content.clone())
            };

            json!({
                "role": m.role,
                "content": content
            })
        };
        final_msgs.push(msg);
    }

    Ok((final_msgs, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a ChatMsg
    fn create_chat_msg(role: &'static str, content: &str, tokens: Option<usize>) -> ChatMsg {
        ChatMsg {
            role,
            content: content.to_string(),
            tool_call_id: None,
            tok: tokens.unwrap_or_else(|| count_tokens(content)),
        }
    }

    #[test]
    fn test_build_prompt_empty_conversation() {
        // Empty conversation with just new user input
        let msgs = vec![create_chat_msg("user", "Hello, assistant!", None)];

        let result = build_prompt_from_chat_messages(msgs, "test-model");

        assert!(result.is_ok());
        let (messages, total_tokens) = result.unwrap();

        // Should have just the user message
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello, assistant!");
        assert!(total_tokens > 0);
    }

    #[test]
    fn test_build_prompt_with_history() {
        let msgs = vec![
            create_chat_msg("user", "Hello", Some(1)),
            create_chat_msg("assistant", "Hi there!", Some(3)),
            create_chat_msg("user", "How are you?", Some(4)),
            create_chat_msg("assistant", "I'm doing well, thanks!", Some(6)),
            create_chat_msg("user", "Great!", Some(1)),
        ];

        let result = build_prompt_from_chat_messages(msgs, "test-model");

        let (messages, _total_tokens) = result.expect("Failed to build prompt");

        // Should have all messages
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[3]["role"], "assistant");
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(messages[4]["content"], "Great!");
    }

    #[test]
    fn test_truncation_preserves_first_and_last() {
        // Create many messages that will exceed token limit
        let mut msgs = vec![];

        // First exchange
        msgs.push(create_chat_msg("user", "First user message", Some(5)));
        msgs.push(create_chat_msg(
            "assistant",
            "First assistant reply",
            Some(5),
        ));

        // Many middle messages (each with high token count to trigger truncation)
        // We need to exceed 59,404 token budget
        for i in 0..35 {
            msgs.push(create_chat_msg(
                "user",
                &format!("Middle user message {}", i),
                Some(1000),
            ));
            msgs.push(create_chat_msg(
                "assistant",
                &format!("Middle assistant message {}", i),
                Some(1000),
            ));
        }

        // Last messages
        msgs.push(create_chat_msg("user", "Recent user message", Some(5)));
        msgs.push(create_chat_msg(
            "assistant",
            "Recent assistant reply",
            Some(5),
        ));
        msgs.push(create_chat_msg("user", "Final message", Some(3)));

        let original_count = msgs.len(); // 2 + 70 + 3 = 75

        let result = build_prompt_from_chat_messages(
            msgs,
            "deepseek-r1-70b", // 64k context limit
        );

        assert!(result.is_ok());
        let (messages, total_tokens) = result.unwrap();

        // Should have truncated middle messages
        assert!(
            messages.len() < original_count,
            "Expected truncation to occur"
        );

        // First messages should be preserved
        assert_eq!(messages[0]["content"], "First user message");
        assert_eq!(messages[1]["content"], "First assistant reply");

        // Should have truncation message
        let has_truncation = messages
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("truncated"));
        assert!(has_truncation);

        // Last message should be the new input
        assert_eq!(messages.last().unwrap()["content"], "Final message");

        // Total tokens should be within budget
        let budget = 64_000 - 4096 - 500; // max - response_reserve - safety
        assert!(total_tokens <= budget);
    }

    #[test]
    fn test_role_alternation_with_truncation() {
        // Force truncation by using massive token counts
        let msgs = vec![
            create_chat_msg("user", "First", Some(20000)),
            create_chat_msg("user", "Second user message", Some(20000)), // Duplicate user
            create_chat_msg("assistant", "Reply", Some(20000)),
            create_chat_msg("user", "New message", Some(5)),
        ];

        let result = build_prompt_from_chat_messages(msgs, "deepseek-r1-70b");

        assert!(result.is_ok());
        let (messages, _) = result.unwrap();

        // Check that truncation message has appropriate role
        let truncation_idx = messages
            .iter()
            .position(|m| m["content"].as_str().unwrap_or("").contains("truncated"));

        if let Some(idx) = truncation_idx {
            // Check the role before truncation message
            if idx > 0 {
                let prev_role = messages[idx - 1]["role"].as_str().unwrap();
                let truncation_role = messages[idx]["role"].as_str().unwrap();

                // If previous was user, truncation should be assistant
                if prev_role == "user" {
                    assert_eq!(
                        truncation_role, "assistant",
                        "Truncation message should alternate roles"
                    );
                }
            }
        }
    }
}
