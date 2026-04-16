//! Tool execution for the Responses API
//!
//! This module handles tool execution including web search, with a clean
//! architecture that can be extended for additional tools in the future.

use crate::brave::{BraveClient, SearchRequest as BraveSearchRequest};
use crate::tinfoil_websearch::TinfoilWebSearchClient;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

/// Execute web search using the configured backend.
pub async fn execute_web_search(
    query: &str,
    max_results: Option<u32>,
    tinfoil_web_search_client: Option<&Arc<TinfoilWebSearchClient>>,
    brave_client: Option<&Arc<BraveClient>>,
) -> Result<String, String> {
    trace!("Executing web search for query: {}", query);
    info!("Executing web search");

    if let Some(client) = tinfoil_web_search_client {
        return execute_tinfoil_search(query, max_results, client).await;
    }

    if let Some(client) = brave_client {
        execute_brave_search(query, max_results, client).await
    } else {
        error!("No search client configured");
        Err("No search client configured".to_string())
    }
}

async fn execute_tinfoil_search(
    query: &str,
    max_results: Option<u32>,
    client: &Arc<TinfoilWebSearchClient>,
) -> Result<String, String> {
    trace!("Executing Tinfoil web search for query: {}", query);

    let result = client
        .search(query, max_results.map(|value| value.clamp(1, 30)))
        .await
        .map_err(|e| {
            error!("Tinfoil web search API error: {:?}", e);
            format!("Search API error: {:?}", e)
        })?;

    let formatted = format_tinfoil_search_result(&result);
    if formatted.trim().is_empty() {
        warn!("No Tinfoil web search results found");
        return Ok(format!("No results found for query: '{}'", query));
    }

    Ok(formatted)
}

/// Execute web search using Brave Search API
async fn execute_brave_search(
    query: &str,
    max_results: Option<u32>,
    client: &Arc<BraveClient>,
) -> Result<String, String> {
    trace!("Executing Brave search for query: {}", query);
    let result_limit = max_results.unwrap_or(5).clamp(1, 30) as usize;

    // Create search request with summary enabled
    let mut search_request = BraveSearchRequest::new(query.to_string());
    search_request.count = Some(result_limit as u32);
    search_request.summary = Some(true);

    // Execute search
    let response = client.search(search_request).await.map_err(|e| {
        error!("Brave search API error: {:?}", e);
        format!("Search API error: {:?}", e)
    })?;

    // Format results
    let mut result_text = String::new();

    // Add web search results
    if let Some(web) = response.web {
        if let Some(results) = web.results {
            result_text.push_str("Search Results:\n\n");
            for (i, result) in results.iter().take(result_limit).enumerate() {
                result_text.push_str(&format!(
                    "{}. {}\n   URL: {}\n   {}\n\n",
                    i + 1,
                    result.title,
                    result.url,
                    result.description.as_ref().unwrap_or(&String::new())
                ));
            }
        }
    }

    // Add infobox if available
    if let Some(infobox) = response.infobox {
        if let Some(title) = infobox.title {
            result_text.push_str(&format!("Information:\n\n{}\n", title));
            if let Some(desc) = infobox.long_desc.or(infobox.description) {
                result_text.push_str(&format!("   {}\n\n", desc));
            }
        }
    }

    // Add news results if available
    if let Some(news) = response.news {
        if let Some(news_results) = news.results {
            if !news_results.is_empty() {
                result_text.push_str("\nNews:\n\n");
                for (i, result) in news_results.iter().take(result_limit.min(3)).enumerate() {
                    result_text.push_str(&format!(
                        "{}. {}\n   URL: {}\n   {}\n\n",
                        i + 1,
                        result.title,
                        result.url,
                        result.description.as_ref().unwrap_or(&String::new())
                    ));
                }
            }
        }
    }

    if result_text.is_empty() {
        warn!("No search results found");
        return Ok(format!("No results found for query: '{}'", query));
    }

    // Check if we have a summarizer key and fetch the summary
    if let Some(summarizer) = response.summarizer {
        debug!("Summarizer key found, fetching summary");
        match client.summarizer(&summarizer.key).await {
            Ok(summarizer_response) => {
                if let Some(summary_items) = summarizer_response.summary {
                    if !summary_items.is_empty() {
                        result_text.push_str("\n--- Search Summary ---\n\n");
                        for item in summary_items.iter() {
                            // Extract text from "token" type items
                            if item.item_type == "token" {
                                if let Some(data) = &item.data {
                                    if let Some(text) = data.as_str() {
                                        result_text.push_str(text);
                                    }
                                }
                            }
                        }
                        result_text.push_str("\n\n");
                        debug!("Successfully added summary to results");
                    }
                }
            }
            Err(e) => {
                // Best effort - log but don't fail the entire request
                warn!("Failed to fetch summarizer content: {:?}", e);
            }
        }
    }

    Ok(result_text)
}

fn format_tinfoil_search_result(result: &Value) -> String {
    let content_text = result
        .get("content")
        .and_then(Value::as_array)
        .map(|content| {
            content
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .filter(|text| !text.is_empty());

    if let Some(text) = content_text {
        return text;
    }

    let structured_content = result
        .get("structuredContent")
        .or_else(|| result.get("structured_content"))
        .unwrap_or(result);

    let formatted_results = structured_content
        .get("results")
        .and_then(Value::as_array)
        .map(|results| format_search_results(results))
        .filter(|text| !text.is_empty());

    if let Some(text) = formatted_results {
        return text;
    }

    if let Some(text) = structured_content.as_str() {
        return text.to_string();
    }

    serde_json::to_string_pretty(structured_content)
        .unwrap_or_else(|_| structured_content.to_string())
}

fn format_search_results(results: &[Value]) -> String {
    let mut formatted = String::new();

    if !results.is_empty() {
        formatted.push_str("Search Results:\n\n");
    }

    for (index, result) in results.iter().enumerate() {
        let title = result
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("Untitled result");
        let url = result
            .get("url")
            .or_else(|| result.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let snippet = [
            "text",
            "snippet",
            "summary",
            "description",
            "content",
            "highlights",
        ]
        .iter()
        .find_map(|field| result.get(field).and_then(Value::as_str))
        .unwrap_or("");

        formatted.push_str(&format!("{}. {}\n", index + 1, title));
        if !url.is_empty() {
            formatted.push_str(&format!("   URL: {}\n", url));
        }
        if !snippet.is_empty() {
            formatted.push_str(&format!("   {}\n", snippet));
        }
        formatted.push('\n');
    }

    formatted.trim().to_string()
}
/// Execute a tool by name with the given arguments
///
/// This is the main entry point for tool execution. It routes to the appropriate
/// tool implementation based on the tool name.
///
/// # Arguments
/// * `tool_name` - The name of the tool to execute (e.g., "web_search")
/// * `arguments` - JSON object containing the tool's arguments
/// * `tinfoil_web_search_client` - Optional Tinfoil web search client
/// * `brave_client` - Optional Brave client (with connection pooling)
///
/// # Returns
/// * `Ok(String)` - The tool's output as a string
/// * `Err(String)` - An error message if the tool execution failed
pub async fn execute_tool(
    tool_name: &str,
    arguments: &Value,
    tinfoil_web_search_client: Option<&Arc<TinfoilWebSearchClient>>,
    brave_client: Option<&Arc<BraveClient>>,
) -> Result<String, String> {
    trace!(
        "Executing tool: {} with arguments: {}",
        tool_name,
        arguments
    );
    debug!("Executing tool: {}", tool_name);

    match tool_name {
        "web_search" => {
            // Extract the query from arguments
            let query = arguments
                .get("query")
                .and_then(|q| q.as_str())
                .ok_or_else(|| "Missing 'query' argument for web_search".to_string())?;
            let max_results = arguments
                .get("max_results")
                .and_then(|value| value.as_u64())
                .map(|value| value as u32);

            execute_web_search(query, max_results, tinfoil_web_search_client, brave_client).await
        }
        _ => {
            error!("Unknown tool requested: {}", tool_name);
            Err(format!("Unknown tool: {}", tool_name))
        }
    }
}

/// Tool registry for managing available tools and their schemas
///
/// This will be expanded in the future to support dynamic tool registration,
/// tool schemas, and validation.
pub struct ToolRegistry {
    // Future: Add tool metadata, schemas, validation rules
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {}
    }

    /// Get the schema for a specific tool
    ///
    /// Returns the JSON schema that describes the tool's parameters and usage.
    /// This can be used for validation or for passing to LLMs that support function calling.
    #[allow(dead_code)]
    pub fn get_tool_schema(&self, tool_name: &str) -> Option<Value> {
        match tool_name {
            "web_search" => Some(json!({
                "name": "web_search",
                "description": "Search the web for current information, facts, and real-time data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Optional maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            })),
            _ => None,
        }
    }

    /// Check if a tool is available
    #[allow(dead_code)]
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        matches!(tool_name, "web_search")
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_web_search_no_client() {
        let result = execute_web_search("test query", None, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No search client configured"));
    }

    #[tokio::test]
    async fn test_execute_tool_missing_args() {
        // Test with None client - should fail on missing args before client check
        let args = json!({});
        let result = execute_tool("web_search", &args, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'query'"));
    }

    #[tokio::test]
    async fn test_execute_tool_unknown() {
        let args = json!({"query": "test"});
        let result = execute_tool("unknown_tool", &args, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tool"));
    }

    #[test]
    fn test_format_tinfoil_search_result_prefers_text_content() {
        let result = json!({
            "content": [
                {
                    "type": "text",
                    "text": "Search Results:\n\n1. Example\n   URL: https://example.com"
                }
            ]
        });

        assert_eq!(
            format_tinfoil_search_result(&result),
            "Search Results:\n\n1. Example\n   URL: https://example.com"
        );
    }

    #[test]
    fn test_format_tinfoil_search_result_formats_structured_results() {
        let result = json!({
            "structuredContent": {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "text": "Example summary"
                    }
                ]
            }
        });

        let formatted = format_tinfoil_search_result(&result);
        assert!(formatted.contains("Search Results:"));
        assert!(formatted.contains("Example"));
        assert!(formatted.contains("https://example.com"));
        assert!(formatted.contains("Example summary"));
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_tool_available("web_search"));
        assert!(!registry.is_tool_available("unknown_tool"));

        let schema = registry.get_tool_schema("web_search");
        assert!(schema.is_some());
        assert_eq!(schema.unwrap()["name"], "web_search");
    }
}
