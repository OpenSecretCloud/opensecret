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
///
/// Optimized two-pass approach:
/// 1. Fetch metadata only (lightweight, no decryption)
/// 2. Run truncation logic on metadata to determine which messages to keep
/// 3. Fetch and decrypt only the needed messages
pub fn build_prompt<D: DBConnection + ?Sized>(
    db: &D,
    conversation_id: i64,
    user_id: uuid::Uuid,
    user_key: &secp256k1::SecretKey,
    model: &str,
    override_instructions: Option<&str>,
) -> Result<(Vec<serde_json::Value>, usize), crate::ApiError> {
    // 1. Get default user instructions if they exist (unless override provided)
    let mut system_tokens = 0usize;
    let system_msg_opt = if let Some(instruction_text) = override_instructions {
        // Use override instructions if provided
        let tok = count_tokens(instruction_text);
        system_tokens = tok;
        Some((ROLE_SYSTEM, instruction_text.to_string(), tok))
    } else if let Ok(Some(default_instruction)) = db.get_default_user_instruction(user_id) {
        // Otherwise use default user instructions
        let plain = decrypt_with_key(user_key, &default_instruction.prompt_enc)
            .map_err(|_| crate::ApiError::InternalServerError)?;
        let content = String::from_utf8_lossy(&plain).into_owned();
        let tok = default_instruction.prompt_tokens as usize;
        system_tokens = tok;
        Some((ROLE_SYSTEM, content, tok))
    } else {
        None
    };

    // 2. PASS 1: Fetch only metadata (lightweight, no content/decryption)
    let metadata = db
        .get_conversation_context_metadata(conversation_id)
        .map_err(|_| crate::ApiError::InternalServerError)?;

    // 3. Run truncation logic on metadata to determine which messages we need
    let (needed_ids, did_truncate) = determine_needed_message_ids(&metadata, model, system_tokens)?;

    // 4. PASS 2: Fetch and decrypt ONLY the messages we need
    let raw = if needed_ids.is_empty() {
        Vec::new()
    } else {
        db.get_messages_by_ids(conversation_id, &needed_ids)
            .map_err(|_| crate::ApiError::InternalServerError)?
    };

    // 5. Build ChatMsg vector with system message + needed messages
    let mut msgs: Vec<ChatMsg> = Vec::new();

    // Track if we have a system message for truncation insert position
    let has_system_msg = system_msg_opt.is_some();

    // Add system message if present
    if let Some((role, content, tok)) = system_msg_opt {
        msgs.push(ChatMsg {
            role,
            content,
            tool_call_id: None,
            tok,
        });
    }

    // Decrypt and add the messages we fetched
    for r in raw {
        match r.message_type.as_str() {
            "user" => {
                // User messages have encrypted MessageContent
                let content_enc = match &r.content_enc {
                    Some(enc) => enc,
                    None => continue,
                };
                let plain = decrypt_with_key(user_key, content_enc)
                    .map_err(|_| crate::ApiError::InternalServerError)?;
                let content = String::from_utf8_lossy(&plain).into_owned();
                let t = r
                    .token_count
                    .map(|v| v as usize)
                    .unwrap_or_else(|| count_tokens(&content));
                msgs.push(ChatMsg {
                    role: ROLE_USER,
                    content,
                    tool_call_id: None,
                    tok: t,
                });
            }
            "assistant" => {
                // Skip in_progress assistant messages (no content yet)
                let content_enc = match &r.content_enc {
                    Some(enc) => enc,
                    None => continue,
                };
                let plain = decrypt_with_key(user_key, content_enc)
                    .map_err(|_| crate::ApiError::InternalServerError)?;
                let content = String::from_utf8_lossy(&plain).into_owned();
                let t = r
                    .token_count
                    .map(|v| v as usize)
                    .unwrap_or_else(|| count_tokens(&content));
                msgs.push(ChatMsg {
                    role: ROLE_ASSISTANT,
                    content,
                    tool_call_id: None,
                    tok: t,
                });
            }
            "tool_call" => {
                // Tool calls are stored with encrypted arguments
                // We need to format these as assistant messages with tool_calls array
                let content_enc = match &r.content_enc {
                    Some(enc) => enc,
                    None => continue,
                };
                let plain = decrypt_with_key(user_key, content_enc)
                    .map_err(|_| crate::ApiError::InternalServerError)?;
                let arguments_str = String::from_utf8_lossy(&plain).into_owned();

                // Parse arguments as JSON - if malformed, use empty object but continue safely
                let arguments: serde_json::Value =
                    serde_json::from_str(&arguments_str).unwrap_or_else(|e| {
                        error!("Failed to parse tool call arguments as JSON: {:?}. Using empty object.", e);
                        serde_json::json!({})
                    });

                // Get tool name from database
                let tool_name = r.tool_name.as_deref().unwrap_or("function");

                // Serialize arguments back to string for OpenAI format
                // OpenAI expects arguments as a JSON string, not a JSON object
                let arguments_string = serde_json::to_string(&arguments).unwrap_or_else(|e| {
                    error!(
                        "Failed to serialize tool arguments: {:?}. Using empty object string.",
                        e
                    );
                    "{}".to_string()
                });

                // Format as assistant message with tool_calls
                let tool_call_msg = serde_json::json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": r.tool_call_id.unwrap_or_else(uuid::Uuid::new_v4).to_string(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments_string
                        }
                    }]
                });

                // Serialize tool_call_msg for storage in ChatMsg
                // This should never fail since we're serializing a well-formed JSON structure
                let content = match serde_json::to_string(&tool_call_msg) {
                    Ok(s) => s,
                    Err(e) => {
                        error!(
                            "Failed to serialize tool_call message: {:?}. Skipping this tool call.",
                            e
                        );
                        // If this fails, skip this message entirely rather than corrupting the conversation
                        continue;
                    }
                };

                let t = r
                    .token_count
                    .map(|v| v as usize)
                    .unwrap_or_else(|| count_tokens(&arguments_str));

                msgs.push(ChatMsg {
                    role: ROLE_ASSISTANT,
                    content,
                    tool_call_id: None,
                    tok: t,
                });
            }
            "tool_output" => {
                // Tool outputs have encrypted output content
                let content_enc = match &r.content_enc {
                    Some(enc) => enc,
                    None => continue,
                };
                let plain = decrypt_with_key(user_key, content_enc)
                    .map_err(|_| crate::ApiError::InternalServerError)?;
                let content = String::from_utf8_lossy(&plain).into_owned();
                let t = r
                    .token_count
                    .map(|v| v as usize)
                    .unwrap_or_else(|| count_tokens(&content));
                msgs.push(ChatMsg {
                    role: "tool",
                    content,
                    tool_call_id: r.tool_call_id,
                    tok: t,
                });
            }
            _ => {
                // Unknown message type, skip
                continue;
            }
        }
    }

    // Insert truncation message if we truncated
    if did_truncate {
        // Insert after system message (if exists) + first kept message
        // The first kept message is always a user message (from our truncation logic)
        let insert_pos = if has_system_msg { 2 } else { 1 };

        let truncation_content = "[Previous messages truncated due to context limits]";
        let truncation_msg = ChatMsg {
            role: ROLE_ASSISTANT,
            content: truncation_content.to_string(),
            tool_call_id: None,
            tok: count_tokens(truncation_content),
        };

        // Only insert if we have enough messages (safety check)
        if msgs.len() > insert_pos {
            msgs.insert(insert_pos, truncation_msg);
        }
    }

    // Note: In the Responses API flow, the current user message is already persisted
    // to the database before this function is called, so it's included in the
    // get_conversation_context_metadata result above.

    // 6. Format for LLM API (no further truncation needed - already done in step 3)
    build_prompt_from_chat_messages(msgs, model)
}

/// Determine which message IDs are needed based on truncation logic
///
/// Runs the same middle-truncation algorithm as `build_prompt_from_chat_messages`,
/// but operates on lightweight metadata to avoid fetching/decrypting unnecessary messages.
///
/// Returns (needed_ids, did_truncate)
fn determine_needed_message_ids(
    metadata: &[crate::models::responses::RawThreadMessageMetadata],
    model: &str,
    system_tokens: usize,
) -> Result<(Vec<(String, i64)>, bool), crate::ApiError> {
    use tracing::debug;

    // Calculate token budget
    let max_ctx = model_max_ctx(model);
    let response_reserve = 4096usize;
    let safety = 500usize;
    let ctx_budget = max_ctx.saturating_sub(response_reserve + safety);

    // Calculate total tokens in all messages
    let total_msg_tokens: usize = metadata
        .iter()
        .filter_map(|m| m.token_count.map(|t| t as usize))
        .sum();

    // If everything fits, return all IDs with no truncation
    if system_tokens + total_msg_tokens <= ctx_budget {
        debug!(
            "All {} messages fit in context budget ({}+ {} = {} <= {})",
            metadata.len(),
            system_tokens,
            total_msg_tokens,
            system_tokens + total_msg_tokens,
            ctx_budget
        );
        return Ok((
            metadata
                .iter()
                .map(|m| (m.message_type.clone(), m.id))
                .collect(),
            false, // did not truncate
        ));
    }

    // Need to truncate - apply middle truncation logic
    debug!(
        "Truncating {} messages: total tokens {}+{} = {} exceeds budget {}",
        metadata.len(),
        system_tokens,
        total_msg_tokens,
        system_tokens + total_msg_tokens,
        ctx_budget
    );

    let mut needed_ids: Vec<(String, i64)> = Vec::new();

    // Always preserve the first system message (already accounted for in system_tokens)
    // Then keep first user message only (first assistant gets removed)
    for m in metadata {
        if m.message_type == "user" {
            needed_ids.push((m.message_type.clone(), m.id));
            break; // Stop after first user - don't keep first assistant
        }
    }

    let head_tokens: usize = needed_ids
        .iter()
        .filter_map(|(msg_type, id)| {
            metadata
                .iter()
                .find(|m| &m.message_type == msg_type && m.id == *id)
                .and_then(|m| m.token_count.map(|t| t as usize))
        })
        .sum();

    let truncation_msg_tokens = count_tokens("[Previous messages truncated due to context limits]");

    // Calculate how many tokens we have left for the tail
    let available_for_tail =
        ctx_budget.saturating_sub(system_tokens + head_tokens + truncation_msg_tokens);

    // Collect messages from the end until we hit the budget
    // CRITICAL: Always include the most recent message (even if it exceeds budget)
    // to ensure the current user query is never lost
    let mut potential_tail: Vec<(String, i64, usize)> = Vec::new(); // (type, id, tokens)
    let mut potential_tail_tokens = 0usize;

    for (i, m) in metadata.iter().rev().enumerate() {
        let tok = m.token_count.map(|t| t as usize).unwrap_or(0);

        // Always include the most recent message (first iteration), even if it exceeds budget
        if i == 0 {
            potential_tail.push((m.message_type.clone(), m.id, tok));
            potential_tail_tokens += tok;
            continue;
        }

        // For subsequent messages, respect the budget
        if potential_tail_tokens + tok > available_for_tail {
            break;
        }
        potential_tail.push((m.message_type.clone(), m.id, tok));
        potential_tail_tokens += tok;
    }
    potential_tail.reverse();

    // Ensure tail starts with a user message (drop leading assistant messages if needed)
    let mut found_user = false;
    for (msg_type, id, _tok) in potential_tail {
        if !found_user {
            if msg_type == "user" {
                found_user = true;
                needed_ids.push((msg_type, id));
            }
            // Skip any leading non-user messages
        } else {
            needed_ids.push((msg_type, id));
        }
    }

    // Add the truncation indicator to the needed IDs (it will be inserted by build_prompt_from_chat_messages)
    // Note: The truncation message itself is inserted by build_prompt_from_chat_messages if truncation occurred

    debug!(
        "Truncation: keeping {} out of {} messages",
        needed_ids.len(),
        metadata.len()
    );

    Ok((needed_ids, true)) // did truncate
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
        // CRITICAL: Always include the most recent message (even if it exceeds budget)
        // to ensure the current user query is never lost
        let mut potential_tail: Vec<ChatMsg> = Vec::new();
        let mut potential_tail_tokens = 0usize;

        for (i, m) in msgs.iter().rev().enumerate() {
            // Always include the most recent message (first iteration), even if it exceeds budget
            if i == 0 {
                potential_tail.push(m.clone());
                potential_tail_tokens += m.tok;
                continue;
            }

            // For subsequent messages, respect the budget
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

        msgs.extend(tail.clone());

        // Check if head + truncation + tail still exceeds budget
        // This can happen when the last message is very large
        let total_after_truncation: usize = msgs.iter().map(|m| m.tok).sum();
        if total_after_truncation > ctx_budget {
            // Try to fit just tail (possibly with system message)
            // First, try system + tail if we have a system message
            if has_system {
                let system_msg = msgs.iter().find(|m| m.role == ROLE_SYSTEM).cloned();
                if let Some(sys) = system_msg {
                    let tail_tokens: usize = tail.iter().map(|m| m.tok).sum();
                    if sys.tok + tail_tokens <= ctx_budget {
                        // System + tail fits
                        msgs = vec![sys];
                        msgs.extend(tail);
                        // No truncation message in this case
                    } else {
                        // Only tail fits
                        msgs = tail;
                    }
                } else {
                    // Only tail fits
                    msgs = tail;
                }
            } else {
                // No system message, just use tail
                msgs = tail;
            }
        }
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
                    .ok_or_else(|| {
                        error!("tool_call_id missing for tool output message");
                        crate::ApiError::InternalServerError
                    })?
                    .to_string()
            })
        } else if m.role == ROLE_ASSISTANT {
            // Check if this is a tool_call message (JSON with tool_calls field) or regular assistant message
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&m.content) {
                if parsed.get("tool_calls").is_some() {
                    // This is a tool_call message - use the JSON directly
                    parsed
                } else {
                    // Regular assistant message - plain string
                    json!({
                        "role": ROLE_ASSISTANT,
                        "content": m.content
                    })
                }
            } else {
                // Not valid JSON, treat as regular assistant message
                json!({
                    "role": ROLE_ASSISTANT,
                    "content": m.content
                })
            }
        } else {
            // User messages
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
                // Fallback for any other role
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
            messages[truncation_idx - 1]["role"],
            "user",
            "Message before truncation should be user"
        );
        assert_eq!(
            messages[truncation_idx - 1]["content"],
            "First user message",
            "Should be the first user message"
        );

        // Message after truncation should be user (start of tail)
        assert_eq!(
            messages[truncation_idx + 1]["role"],
            "user",
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

    #[test]
    fn test_only_last_message_fits() {
        // Scenario: Budget is so tight that ONLY the last user message can fit
        // No room for system, first user, or truncation message
        // Expected: Just return the last user message alone

        let msgs = vec![
            create_chat_msg("user", "First", Some(20000)),
            create_chat_msg("assistant", "First reply", Some(20000)),
            create_chat_msg("user", "Second", Some(20000)),
            create_chat_msg("assistant", "Second reply", Some(20000)),
            // Last message fits, but adding anything else (system, first, truncation) would exceed
            create_chat_msg("user", "Final message that barely fits", Some(58000)),
        ];

        let result = build_prompt_from_chat_messages(msgs, "deepseek-r1-70b");

        let (messages, _) = result.expect("Failed to build prompt");

        // Should have ONLY the last user message
        assert_eq!(messages.len(), 1, "Should only have the last user message");
        assert_eq!(messages[0]["role"], "user");
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("Final message that barely fits"));
    }

    #[test]
    fn test_last_message_plus_system_fits() {
        // Scenario: Last message + system message fits, but not first user + truncation
        // Expected: system + last user message (no truncation, no first user)

        let msgs = vec![
            create_chat_msg("system", "You are helpful", Some(10)),
            create_chat_msg("user", "First", Some(20000)),
            create_chat_msg("assistant", "First reply", Some(20000)),
            create_chat_msg("user", "Second", Some(20000)),
            create_chat_msg("assistant", "Second reply", Some(20000)),
            // Last message is large enough that first+truncation won't fit
            create_chat_msg("user", "Final", Some(58000)),
        ];

        let result = build_prompt_from_chat_messages(msgs, "deepseek-r1-70b");

        let (messages, _) = result.expect("Failed to build prompt");

        // Should have system + last user only (no truncation because we couldn't fit head)
        assert_eq!(messages.len(), 2, "Should have system + last user");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Final");
    }

    #[test]
    fn test_oversized_recent_message_survives_truncation() {
        // Regression test for critical bug: ensure the most recent user message
        // is ALWAYS included, even if it alone exceeds the available tail budget.
        //
        // Scenario:
        // - First user message: 100 tokens (kept in head)
        // - Middle messages: 50,000 tokens total (will be truncated)
        // - Recent user message: 10,000 tokens (HUGE - exceeds available tail budget)
        //
        // The recent message must survive even though it's larger than available_for_tail

        let mut msgs = vec![];

        // First user message (will be in head)
        msgs.push(create_chat_msg("user", "First message", Some(100)));

        // Many middle messages to consume most of the budget
        for i in 0..25 {
            msgs.push(create_chat_msg(
                "user",
                &format!("Middle user {}", i),
                Some(1000),
            ));
            msgs.push(create_chat_msg(
                "assistant",
                &format!("Middle assistant {}", i),
                Some(1000),
            ));
        }

        // Recent HUGE user message that exceeds available_for_tail
        // This is the critical message that must not be lost
        msgs.push(create_chat_msg(
            "user",
            "This is a very long recent user message that exceeds the available tail budget",
            Some(10000),
        ));

        let result = build_prompt_from_chat_messages(
            msgs.clone(),
            "deepseek-r1-70b", // 64k context
        );

        let (messages, _total_tokens) = result.expect("Failed to build prompt");

        // CRITICAL: The recent huge message MUST be in the output
        let recent_msg_found = messages.iter().any(|m| {
            m["content"]
                .as_str()
                .unwrap_or("")
                .contains("very long recent user message")
        });

        assert!(
            recent_msg_found,
            "Recent user message must be preserved even if it exceeds tail budget. Messages: {:?}",
            messages
        );

        // The last message in the output should be our recent message
        assert_eq!(messages.last().unwrap()["role"], "user");
        assert!(messages.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("very long recent user message"));

        // Should have truncated (has truncation message)
        let has_truncation = messages
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("truncated"));
        assert!(has_truncation, "Should have truncated middle messages");

        // First message should be the first user message
        assert_eq!(messages[0]["content"], "First message");
    }
}
