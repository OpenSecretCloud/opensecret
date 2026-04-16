use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum TinfoilWebSearchError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
    #[error("JSON-RPC error: {message}")]
    Rpc { message: String },
    #[error("Invalid response format: {message}")]
    InvalidResponse { message: String },
}

#[derive(Clone)]
pub struct TinfoilWebSearchClient {
    client: reqwest::Client,
    base_url: Arc<String>,
    api_key: Arc<String>,
}

impl TinfoilWebSearchClient {
    pub fn new(
        base_url: String,
        api_key: String,
        allow_insecure_tls: bool,
    ) -> Result<Self, TinfoilWebSearchError> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(100)
            .danger_accept_invalid_certs(allow_insecure_tls)
            .user_agent("OpenSecret/0.1.0")
            .build()?;

        Ok(Self {
            client,
            base_url: Arc::new(base_url.trim_end_matches('/').to_string()),
            api_key: Arc::new(api_key),
        })
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: Option<u32>,
    ) -> Result<Value, TinfoilWebSearchError> {
        self.call_tool(
            "search",
            json!({
                "query": query,
                "max_results": max_results.unwrap_or(5),
            }),
        )
        .await
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, TinfoilWebSearchError> {
        let response = self
            .client
            .post(format!("{}/mcp", self.base_url.as_str()))
            .bearer_auth(self.api_key.as_str())
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments,
                }
            }))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(TinfoilWebSearchError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let body = response.text().await?;
        let payload = parse_response_body(&body)?;
        if let Some(error) = payload.get("error") {
            return Err(TinfoilWebSearchError::Rpc {
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| error.to_string()),
            });
        }

        Ok(normalize_tool_result(
            payload.get("result").cloned().unwrap_or(payload),
        ))
    }
}

fn parse_response_body(body: &str) -> Result<Value, TinfoilWebSearchError> {
    serde_json::from_str(body).or_else(|json_error| {
        let sse_data =
            extract_first_sse_data(body).ok_or_else(|| TinfoilWebSearchError::InvalidResponse {
                message: format!(
                    "could not parse JSON or SSE payload: {json_error}; body prefix: {}",
                    truncate_for_error(body)
                ),
            })?;

        serde_json::from_str(&sse_data).map_err(|sse_error| {
            TinfoilWebSearchError::InvalidResponse {
                message: format!(
                    "failed to parse SSE data as JSON: {sse_error}; data prefix: {}",
                    truncate_for_error(&sse_data)
                ),
            }
        })
    })
}

fn extract_first_sse_data(body: &str) -> Option<String> {
    let mut current_block = Vec::new();

    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            let trimmed = data.trim_start();
            if trimmed != "[DONE]" {
                current_block.push(trimmed.to_string());
            }
            continue;
        }

        if line.trim().is_empty() && !current_block.is_empty() {
            return Some(current_block.join("\n"));
        }
    }

    if current_block.is_empty() {
        None
    } else {
        Some(current_block.join("\n"))
    }
}

fn normalize_tool_result(result: Value) -> Value {
    if let Some(structured) = result
        .get("structuredContent")
        .cloned()
        .or_else(|| result.get("structured_content").cloned())
    {
        return structured;
    }

    if let Some(text) = result
        .get("content")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find_map(|item| item.get("text").and_then(Value::as_str))
        })
    {
        if let Ok(parsed) = serde_json::from_str::<Value>(text) {
            return parsed;
        }
    }

    result
}

fn truncate_for_error(value: &str) -> String {
    const MAX_LEN: usize = 200;
    let mut truncated = value.chars().take(MAX_LEN).collect::<String>();
    if value.chars().count() > MAX_LEN {
        truncated.push_str("...");
    }
    truncated
}

impl std::fmt::Debug for TinfoilWebSearchClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinfoilWebSearchClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response_body_accepts_plain_json() {
        let payload = parse_response_body(r#"{"jsonrpc":"2.0","id":1,"result":{"results":[]}}"#)
            .expect("plain json");
        assert_eq!(payload["result"]["results"], json!([]));
    }

    #[test]
    fn test_parse_response_body_accepts_sse_wrapped_json() {
        let payload = parse_response_body(
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"results\":[]}}\n\n",
        )
        .expect("sse json");
        assert_eq!(payload["result"]["results"], json!([]));
    }

    #[test]
    fn test_normalize_tool_result_parses_json_text_content() {
        let normalized = normalize_tool_result(json!({
            "content": [{
                "type": "text",
                "text": "{\"results\":[{\"title\":\"Example\"}]}"
            }]
        }));

        assert_eq!(normalized["results"][0]["title"], "Example");
    }
}
