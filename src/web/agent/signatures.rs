use crate::web::openai::{get_chat_completion_response, BillingContext, CompletionChunk};
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::error;

use dspy_rs::adapter::chat::ChatAdapter;
use dspy_rs::client_registry::AssistantContent;
use dspy_rs::{CompletionError, CompletionRequest, CompletionResponse};
use dspy_rs::{CustomCompletionModel, LMClient, OneOrMany, LM};

pub const DEFAULT_AGENT_SYSTEM_PROMPT: &str = r#"You are Maple, a helpful AI assistant.

Return `tool_calls` as an empty JSON array when no tools are needed."#;

#[dspy_rs::BamlType]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentToolCall {
    pub name: String,
    #[serde(default)]
    pub args: HashMap<String, String>,
}

#[derive(dspy_rs::Signature, Debug, Clone)]
pub struct AgentResponse {
    #[input]
    pub input: String,
    #[input]
    pub current_time: String,
    #[input]
    pub persona_block: String,
    #[input]
    pub human_block: String,
    #[input]
    pub memory_metadata: String,
    #[input]
    pub previous_context_summary: String,
    #[input]
    pub recent_conversation: String,
    #[input]
    pub available_tools: String,
    #[input]
    pub is_first_time_user: bool,

    #[output]
    pub messages: Vec<String>,
    #[output]
    pub tool_calls: Vec<AgentToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentResponseOutput {
    pub messages: Vec<String>,
    pub tool_calls: Vec<AgentToolCall>,
}

pub async fn run_agent_response_signature(
    state: &Arc<AppState>,
    user: &crate::models::users::User,
    model: &str,
    system_prompt: &str,
    input: &AgentResponseInput,
) -> Result<AgentResponseOutput, ApiError> {
    let lm = build_lm(state.clone(), Arc::new(user.clone()), model.to_string()).await?;
    let adapter = ChatAdapter;

    let system = adapter
        .format_system_message_typed_with_instruction::<AgentResponse>(Some(system_prompt))
        .map_err(|e| {
            error!("Failed to format DSRS system prompt: {e:?}");
            ApiError::InternalServerError
        })?;
    let user_msg = adapter.format_user_message_typed::<AgentResponse>(input);

    let mut chat = dspy_rs::Chat::new(vec![]);
    chat.push("system", &system);
    chat.push("user", &user_msg);

    let response = lm.call(chat, Vec::new()).await.map_err(|e| {
        error!("DSRS LM call failed: {e:?}");
        ApiError::InternalServerError
    })?;

    let (output, _meta) = adapter
        .parse_response_typed::<AgentResponse>(&response.output)
        .map_err(|e| {
            error!("DSRS typed parse failed: {e:?}");
            ApiError::InternalServerError
        })?;

    let mut messages = output.messages;
    messages.retain(|m| !m.trim().is_empty());

    Ok(AgentResponseOutput {
        messages,
        tool_calls: output.tool_calls,
    })
}

async fn build_lm(
    state: Arc<AppState>,
    user: Arc<crate::models::users::User>,
    model: String,
) -> Result<Arc<LM>, ApiError> {
    let model_for_closure = model.clone();
    let completion_model = CustomCompletionModel::new(move |request: CompletionRequest| {
        let state = state.clone();
        let user = user.clone();
        let model = model_for_closure.clone();

        Box::pin(async move {
            let body = completion_request_to_openai_body(&model, &request)?;
            let headers = HeaderMap::new();
            let billing_context = BillingContext::new(AuthMethod::Jwt, model.to_string());
            let completion = get_chat_completion_response(
                &state,
                user.as_ref(),
                body,
                &headers,
                billing_context,
            )
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            if completion.metadata.is_streaming {
                return Err(CompletionError::ProviderError(
                    "Streaming response not supported".to_string(),
                ));
            }

            let mut rx = completion.stream;
            let response_json = match rx.recv().await {
                Some(CompletionChunk::FullResponse(response_json)) => response_json,
                _ => {
                    return Err(CompletionError::ProviderError(
                        "Missing completion response".to_string(),
                    ));
                }
            };

            let content = extract_assistant_content(&response_json).ok_or_else(|| {
                CompletionError::ResponseError("Missing assistant message content".to_string())
            })?;

            let usage = extract_usage(&response_json);

            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::text(content)),
                usage,
                raw_response: (),
            })
        })
    });

    let lm = LM::builder()
        .base_url("http://localhost".to_string())
        .model(model)
        .temperature(0.7)
        .max_tokens(32768)
        .cache(false)
        .build()
        .await
        .map_err(|e| {
            error!("Failed to build DSRS LM: {e:?}");
            ApiError::InternalServerError
        })?;

    let lm = lm
        .with_client(LMClient::Custom(completion_model))
        .await
        .map_err(|e| {
            error!("Failed to set DSRS custom LM client: {e:?}");
            ApiError::InternalServerError
        })?;

    Ok(Arc::new(lm))
}

fn completion_request_to_openai_body(
    model: &str,
    request: &CompletionRequest,
) -> Result<Value, CompletionError> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(preamble) = &request.preamble {
        messages.push(json!({"role": "system", "content": preamble}));
    }

    for message in request.chat_history.iter() {
        let message_val = serde_json::to_value(message)?;
        let Some(role) = message_val.get("role").and_then(|v| v.as_str()) else {
            continue;
        };

        let Some(content_items) = message_val.get("content").and_then(|v| v.as_array()) else {
            continue;
        };

        let content = match role {
            "user" => extract_user_content_text(content_items),
            "assistant" => extract_assistant_content_text(content_items),
            _ => None,
        };

        let Some(content) = content.filter(|c| !c.trim().is_empty()) else {
            continue;
        };

        messages.push(json!({"role": role, "content": content}));
    }

    let mut body = json!({
        "model": model,
        "stream": false,
        "messages": messages,
    });

    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }

    Ok(body)
}

fn extract_user_content_text(content_items: &[Value]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for item in content_items {
        match item.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_string());
                }
            }
            Some("toolresult") => {
                if let Some(results) = item.get("content").and_then(|v| v.as_array()) {
                    for result in results {
                        if result.get("type").and_then(|v| v.as_str()) == Some("text") {
                            if let Some(text) = result.get("text").and_then(|v| v.as_str()) {
                                parts.push(text.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_assistant_content_text(content_items: &[Value]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for item in content_items {
        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
            parts.push(text.to_string());
        } else if let Some(reasoning) = item.get("reasoning").and_then(|v| v.as_array()) {
            let joined = reasoning
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !joined.is_empty() {
                parts.push(joined);
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_assistant_content(response_json: &Value) -> Option<String> {
    response_json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn extract_usage(response_json: &Value) -> dspy_rs::Usage {
    let prompt_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_tokens = response_json
        .get("usage")
        .and_then(|v| v.get("total_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| prompt_tokens + completion_tokens);

    dspy_rs::Usage {
        input_tokens: prompt_tokens,
        output_tokens: completion_tokens,
        total_tokens,
    }
}
