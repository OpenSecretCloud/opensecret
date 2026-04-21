//! Tool execution for the Responses API
//!
//! This module handles tool execution including web search, with a clean
//! architecture that can be extended for additional tools in the future.

use crate::brave::{BraveClient, BraveError, SearchRequest as BraveSearchRequest};
use serde_json::{json, Value};
use std::{sync::Arc, time::Instant};
use tracing::{debug, error, info, warn};

fn summarize_brave_error(error: &BraveError) -> String {
    match error {
        BraveError::Api { status, .. } => format!("api_status={status}"),
        BraveError::Request(err) if err.is_timeout() => "request_timeout".to_string(),
        BraveError::Request(err) if err.is_connect() => "request_connect".to_string(),
        BraveError::Request(err) if err.is_decode() => "request_decode".to_string(),
        BraveError::Request(err) if err.is_request() => "request_error".to_string(),
        BraveError::Request(_) => "request_other".to_string(),
    }
}

/// Execute web search using Brave Search API
///
/// Requires a Brave client to be provided (initialized at startup with connection pooling).
pub async fn execute_web_search(
    query: &str,
    brave_client: Option<&Arc<BraveClient>>,
) -> Result<String, String> {
    info!("Executing web search");
    let search_started = Instant::now();
    debug!("Starting web_search tool execution");

    let result = if let Some(client) = brave_client {
        execute_brave_search(query, client).await
    } else {
        error!("No search client configured");
        Err("No search client configured".to_string())
    };

    debug!(
        "Finished web_search tool execution in {} ms with status={}",
        search_started.elapsed().as_millis(),
        if result.is_ok() { "ok" } else { "error" }
    );

    result
}

/// Execute web search using Brave Search API
async fn execute_brave_search(query: &str, client: &Arc<BraveClient>) -> Result<String, String> {
    let brave_search_started = Instant::now();
    debug!("Starting Brave search request");

    // Create search request with summary enabled
    let mut search_request = BraveSearchRequest::new(query.to_string());
    search_request.summary = Some(true);

    // Execute search
    let response = client.search(search_request).await.map_err(|e| {
        error!(
            "Brave search API error during web_search ({})",
            summarize_brave_error(&e)
        );
        format!("Search API error: {:?}", e)
    })?;
    debug!(
        "Finished Brave search request in {} ms",
        brave_search_started.elapsed().as_millis()
    );

    // Format results
    let mut result_text = String::new();

    // Add web search results
    if let Some(web) = response.web {
        if let Some(results) = web.results {
            result_text.push_str("Search Results:\n\n");
            for (i, result) in results.iter().take(5).enumerate() {
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
                for (i, result) in news_results.iter().take(3).enumerate() {
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
        let summarizer_started = Instant::now();
        debug!("Starting Brave summarizer request");
        match client.summarizer(&summarizer.key).await {
            Ok(summarizer_response) => {
                debug!(
                    "Finished Brave summarizer request in {} ms",
                    summarizer_started.elapsed().as_millis()
                );
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
                warn!(
                    "Failed to fetch Brave summarizer content after {} ms ({})",
                    summarizer_started.elapsed().as_millis(),
                    summarize_brave_error(&e)
                );
            }
        }
    }

    Ok(result_text)
}
/// Execute a tool by name with the given arguments
///
/// This is the main entry point for tool execution. It routes to the appropriate
/// tool implementation based on the tool name.
///
/// # Arguments
/// * `tool_name` - The name of the tool to execute (e.g., "web_search")
/// * `arguments` - JSON object containing the tool's arguments
/// * `brave_client` - Optional Brave client (with connection pooling)
///
/// # Returns
/// * `Ok(String)` - The tool's output as a string
/// * `Err(String)` - An error message if the tool execution failed
pub async fn execute_tool(
    tool_name: &str,
    arguments: &Value,
    brave_client: Option<&Arc<BraveClient>>,
) -> Result<String, String> {
    debug!("Executing tool: {}", tool_name);

    match tool_name {
        "web_search" => {
            // Extract the query from arguments
            let query = arguments
                .get("query")
                .and_then(|q| q.as_str())
                .ok_or_else(|| "Missing 'query' argument for web_search".to_string())?;

            execute_web_search(query, brave_client).await
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
        let result = execute_web_search("test query", None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No search client configured"));
    }

    #[tokio::test]
    async fn test_execute_tool_missing_args() {
        // Test with None client - should fail on missing args before client check
        let args = json!({});
        let result = execute_tool("web_search", &args, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'query'"));
    }

    #[tokio::test]
    async fn test_execute_tool_unknown() {
        let args = json!({"query": "test"});
        let result = execute_tool("unknown_tool", &args, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tool"));
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
