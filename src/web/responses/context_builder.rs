//! Build the ChatCompletion prompt array respecting token limits.

use super::constants::{ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER};
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
    user_id: uuid::Uuid,
    user_key: &secp256k1::SecretKey,
    model: &str,
    override_instructions: Option<&str>,
) -> Result<(Vec<serde_json::Value>, usize), crate::ApiError> {
    // 1. Get default user instructions if they exist (unless override provided)
    let mut msgs: Vec<ChatMsg> = Vec::new();

    if let Some(instruction_text) = override_instructions {
        // Use override instructions if provided
        let tok = count_tokens(instruction_text);
        msgs.push(ChatMsg {
            role: ROLE_SYSTEM,
            content: instruction_text.to_string(),
            tool_call_id: None,
            tok,
        });
    } else if let Ok(Some(default_instruction)) = db.get_default_user_instruction(user_id) {
        // Otherwise use default user instructions
        let plain = decrypt_with_key(user_key, &default_instruction.prompt_enc)
            .map_err(|_| crate::ApiError::InternalServerError)?;
        let content = String::from_utf8_lossy(&plain).into_owned();
        let tok = default_instruction.prompt_tokens as usize;
        msgs.push(ChatMsg {
            role: ROLE_SYSTEM,
            content,
            tool_call_id: None,
            tok,
        });
    }

    // 2. Pull every stored message (already in chrono order ASC)
    let raw = db
        .get_conversation_context_messages(
            conversation_id,
            i64::MAX, // No limit - fetch all messages for context building
            None,     // No cursor
            "asc",    // Chronological order
        )
        .map_err(|_| crate::ApiError::InternalServerError)?;

    // 3. Decrypt + map to ChatMsg
    for r in raw {
        // Skip messages with no content (in_progress assistant messages)
        let content_enc = match &r.content_enc {
            Some(enc) => enc,
            None => continue,
        };

        let plain = decrypt_with_key(user_key, content_enc)
            .map_err(|_| crate::ApiError::InternalServerError)?;
        let content = String::from_utf8_lossy(&plain).into_owned();
        let role = match r.message_type.as_str() {
            "user" => ROLE_USER,
            "assistant" => ROLE_ASSISTANT,
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

    // 4. Delegate to the pure function for truncation and formatting
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

    let raw_msg_count = msgs.len();
    let mut total: usize = msgs.iter().map(|m| m.tok).sum();

    if total > ctx_budget {
        // Middle truncation: keep first messages + truncate middle
        let mut head: Vec<ChatMsg> = Vec::new();
        let mut tail: Vec<ChatMsg> = Vec::new();

        // Always preserve the first system message (user instructions) if present
        // Then keep first user message only (first assistant gets removed)
        let mut has_system = false;

        for m in &msgs {
            // Always keep the first system message (this is the user's default instructions)
            if m.role == ROLE_SYSTEM && !has_system {
                head.push(m.clone());
                has_system = true;
                continue;
            }
            // Keep first user message
            if m.role == ROLE_USER {
                head.push(m.clone());
                break; // Stop after first user - don't keep first assistant
            }
        }

        let head_tokens: usize = head.iter().map(|m| m.tok).sum();
        let truncation_msg_tokens =
            count_tokens("[Previous messages truncated due to context limits]");

        // Calculate how many tokens we have left for the tail
        let available_for_tail = ctx_budget.saturating_sub(head_tokens + truncation_msg_tokens);

        // Collect messages from the end until we hit the budget
        // We need to ensure tail starts with a user message (for proper alternation after assistant truncation)
        let mut potential_tail: Vec<ChatMsg> = Vec::new();
        let mut potential_tail_tokens = 0usize;

        for m in msgs.iter().rev() {
            if potential_tail_tokens + m.tok > available_for_tail {
                break;
            }
            potential_tail.push(m.clone());
            potential_tail_tokens += m.tok;
        }
        potential_tail.reverse();

        // Ensure tail starts with a user message (drop leading assistant messages if needed)
        let mut found_user = false;
        for m in potential_tail {
            if !found_user {
                if m.role == ROLE_USER {
                    found_user = true;
                    tail.push(m);
                }
                // Skip any leading non-user messages
            } else {
                tail.push(m);
            }
        }

        // Reconstruct with truncation message
        // Only add truncation message if we're actually truncating something
        // The truncation message is always an assistant message, representing the removed assistant→...→assistant block
        let total_msgs = head.len() + tail.len();
        let did_truncate = total_msgs < raw_msg_count;

        msgs = head;

        if did_truncate {
            msgs.push(ChatMsg {
                role: ROLE_ASSISTANT,
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
            let content = if m.role == ROLE_USER {
                // User messages are stored as MessageContent - convert to OpenAI format
                use crate::web::responses::{MessageContent, MessageContentConverter};
                let mc: MessageContent = serde_json::from_str(&m.content).map_err(|e| {
                    error!("Failed to deserialize user message content: {:?}", e);
                    crate::ApiError::InternalServerError
                })?;
                MessageContentConverter::to_openai_format(&mc)
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
        use crate::web::responses::MessageContent;

        // User messages need to be stored as MessageContent JSON
        let content_str = if role == ROLE_USER {
            // Serialize as MessageContent::Text (which is just a JSON string)
            let mc = MessageContent::Text(content.to_string());
            serde_json::to_string(&mc).unwrap()
        } else {
            content.to_string()
        };

        ChatMsg {
            role,
            content: content_str,
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

        let (messages, total_tokens) = result.expect("Failed to build prompt");

        // Should have truncated middle messages
        assert!(
            messages.len() < original_count,
            "Expected truncation to occur"
        );

        // First user message should be preserved
        assert_eq!(messages[0]["content"], "First user message");
        assert_eq!(messages[0]["role"], "user");

        // Find truncation message
        let truncation_idx = messages
            .iter()
            .position(|m| m["content"].as_str().unwrap_or("").contains("truncated"))
            .expect("Should have truncation message");

        // Truncation message must ALWAYS be assistant role
        assert_eq!(
            messages[truncation_idx]["role"], "assistant",
            "Truncation message must always be assistant role"
        );

        // Message before truncation should be user (the first user message we kept)
        assert_eq!(
            messages[truncation_idx - 1]["role"], "user",
            "Message before truncation should be user"
        );
        assert_eq!(
            messages[truncation_idx - 1]["content"], "First user message",
            "Should be the first user message"
        );

        // Message after truncation should be user (start of tail)
        assert_eq!(
            messages[truncation_idx + 1]["role"], "user",
            "Message after truncation should be user"
        );

        // Last message should be the new input
        assert_eq!(messages.last().unwrap()["content"], "Final message");
        assert_eq!(messages.last().unwrap()["role"], "user");

        // Verify proper role alternation throughout
        for i in 0..messages.len() - 1 {
            let curr_role = messages[i]["role"].as_str().unwrap();
            let next_role = messages[i + 1]["role"].as_str().unwrap();

            // Exception: user → assistant truncation message
            if i == truncation_idx - 1 {
                assert_eq!(curr_role, "user");
                assert_eq!(next_role, "assistant");
                continue;
            }

            // Exception: assistant truncation → user (start of tail)
            if i == truncation_idx {
                assert_eq!(curr_role, "assistant");
                assert_eq!(next_role, "user");
                continue;
            }

            // Otherwise roles should alternate
            assert_ne!(
                curr_role, next_role,
                "Roles should alternate at position {}",
                i
            );
        }

        // Total tokens should be within budget
        let budget = 64_000 - 4096 - 500; // max - response_reserve - safety
        assert!(total_tokens <= budget);
    }

    #[test]
    fn test_truncation_with_system_message() {
        // Force truncation with a system message at the start
        let msgs = vec![
            create_chat_msg("system", "You are a helpful assistant", Some(10)),
            create_chat_msg("user", "First", Some(20000)),
            create_chat_msg("assistant", "First reply", Some(20000)),
            create_chat_msg("user", "Second", Some(20000)),
            create_chat_msg("assistant", "Second reply", Some(20000)),
            create_chat_msg("user", "New message", Some(5)),
        ];

        let result = build_prompt_from_chat_messages(msgs, "deepseek-r1-70b");

        let (messages, _) = result.expect("Failed to build prompt");

        // Find truncation message
        let truncation_idx = messages
            .iter()
            .position(|m| m["content"].as_str().unwrap_or("").contains("truncated"))
            .expect("Should have truncation message");

        // Should have pattern: system, user, assistant (truncation), user
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a helpful assistant");

        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "First");

        assert_eq!(messages[2]["role"], "assistant");
        assert!(messages[2]["content"]
            .as_str()
            .unwrap()
            .contains("truncated"));
        assert_eq!(truncation_idx, 2);

        assert_eq!(messages[3]["role"], "user");
        assert_eq!(messages[3]["content"], "New message");
    }
}
