//! Tool execution for the Responses API
//!
//! This module handles tool execution including web search, with a clean
//! architecture that can be extended for additional tools in the future.

use crate::brave::{BraveClient, SearchRequest as BraveSearchRequest};
use crate::kagi::{KagiClient, SearchRequest as KagiSearchRequest};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

/// Execute web search using Brave or Kagi Search API
///
/// Prefers Brave if available, falls back to Kagi if Brave is not configured.
/// Requires at least one client to be provided (initialized at startup with connection pooling).
pub async fn execute_web_search(
    query: &str,
    brave_client: Option<&Arc<BraveClient>>,
    kagi_client: Option<&Arc<KagiClient>>,
) -> Result<String, String> {
    trace!("Executing web search for query: {}", query);
    info!("Executing web search");

    // Try Brave first, then fall back to Kagi
    if let Some(client) = brave_client {
        execute_brave_search(query, client).await
    } else if let Some(client) = kagi_client {
        execute_kagi_search(query, client).await
    } else {
        error!("No search client configured");
        Err("No search client configured".to_string())
    }
}

/// Execute web search using Brave Search API
async fn execute_brave_search(query: &str, client: &Arc<BraveClient>) -> Result<String, String> {
    trace!("Executing Brave search for query: {}", query);

    // Create search request
    let search_request = BraveSearchRequest::new(query.to_string());

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
        warn!("No search results found for query: {}", query);
        return Ok(format!("No results found for query: '{}'", query));
    }

    Ok(result_text)
}

/// Execute web search using Kagi Search API
async fn execute_kagi_search(query: &str, client: &Arc<KagiClient>) -> Result<String, String> {
    trace!("Executing Kagi search for query: {}", query);

    // Create search request
    let search_request = KagiSearchRequest::new(query.to_string());

    // Execute search
    let response = client.search(search_request).await.map_err(|e| {
        error!("Kagi search API error: {:?}", e);
        format!("Search API error: {:?}", e)
    })?;

    // Format results
    let mut result_text = String::new();

    if let Some(data) = response.data {
        // Prioritize direct answers
        if let Some(direct_answers) = data.direct_answer {
            for answer in direct_answers {
                result_text.push_str(&format!(
                    "Direct Answer: {}\n\n",
                    answer.snippet.unwrap_or_default()
                ));
            }
        }

        // Add weather information if available
        if let Some(weather_results) = data.weather {
            if !weather_results.is_empty() {
                result_text.push_str("Weather:\n\n");
                for result in weather_results.iter().take(1) {
                    result_text.push_str(&format!(
                        "{}\n   {}\n\n",
                        result.title,
                        result.snippet.as_ref().unwrap_or(&String::new())
                    ));
                }
            }
        }

        // Add infobox if available (detailed entity information)
        if let Some(infobox_results) = data.infobox {
            if !infobox_results.is_empty() {
                result_text.push_str("Information:\n\n");
                for result in infobox_results.iter().take(1) {
                    result_text.push_str(&format!(
                        "{}\n   {}\n",
                        result.title,
                        result.snippet.as_ref().unwrap_or(&String::new())
                    ));

                    // Add URL if available for more details
                    if !result.url.is_empty() {
                        result_text.push_str(&format!("   More info: {}\n", result.url));
                    }
                    result_text.push('\n');
                }
            }
        }

        // Add search results
        if let Some(search_results) = data.search {
            result_text.push_str("Search Results:\n\n");
            for (i, result) in search_results.iter().take(5).enumerate() {
                result_text.push_str(&format!(
                    "{}. {}\n   URL: {}\n   {}\n\n",
                    i + 1,
                    result.title,
                    result.url,
                    result.snippet.as_ref().unwrap_or(&String::new())
                ));
            }
        }

        // Add news results if available
        if let Some(news_results) = data.news {
            if !news_results.is_empty() {
                result_text.push_str("\nNews:\n\n");
                for (i, result) in news_results.iter().take(3).enumerate() {
                    result_text.push_str(&format!(
                        "{}. {}\n   URL: {}\n   {}\n\n",
                        i + 1,
                        result.title,
                        result.url,
                        result.snippet.as_ref().unwrap_or(&String::new())
                    ));
                }
            }
        }
    }

    if result_text.is_empty() {
        warn!("No search results found for query: {}", query);
        return Ok(format!("No results found for query: '{}'", query));
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
/// * `kagi_client` - Optional Kagi client (with connection pooling)
///
/// # Returns
/// * `Ok(String)` - The tool's output as a string
/// * `Err(String)` - An error message if the tool execution failed
pub async fn execute_tool(
    tool_name: &str,
    arguments: &Value,
    brave_client: Option<&Arc<BraveClient>>,
    kagi_client: Option<&Arc<KagiClient>>,
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

            execute_web_search(query, brave_client, kagi_client).await
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
        let result = execute_web_search("test query", None, None).await;
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
    fn test_tool_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_tool_available("web_search"));
        assert!(!registry.is_tool_available("unknown_tool"));

        let schema = registry.get_tool_schema("web_search");
        assert!(schema.is_some());
        assert_eq!(schema.unwrap()["name"], "web_search");
    }
}
