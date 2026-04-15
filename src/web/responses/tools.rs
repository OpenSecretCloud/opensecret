//! Tool execution for the Responses API
//!
//! This module handles tool execution including web search, with a clean
//! architecture that can be extended for additional tools in the future.

use super::prompts;
use crate::brave::{BraveClient, SearchRequest as BraveSearchRequest};
use crate::kagi::{KagiClient, SearchRequest as KagiSearchRequest};
use crate::models::users::User;
use crate::web::openai::{get_chat_completion_response, BillingContext, CompletionChunk};
use crate::web::openai_auth::AuthMethod;
use crate::AppState;
use axum::http::HeaderMap;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

pub(crate) struct WebSearchExecutionContext<'a> {
    pub state: &'a Arc<AppState>,
    pub user: &'a User,
    pub conversation_history: &'a [Value],
    pub user_message: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UrlCitation {
    title: String,
    url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockedSearch {
    query: String,
    reason: String,
}

pub(crate) fn has_web_search_backend(state: &Arc<AppState>) -> bool {
    state.proxy_router.get_tinfoil_proxy().is_some()
        || state.brave_client.is_some()
        || state.kagi_client.is_some()
}

pub(crate) async fn prepare_web_search_query(context: &WebSearchExecutionContext<'_>) -> String {
    trace!(
        "Preparing web search subagent query from user message: {}",
        context.user_message
    );

    let request = prompts::build_web_search_prompt_drafting_request(
        context.conversation_history,
        context.user_message,
    );
    let headers = HeaderMap::new();
    let billing_context = BillingContext::new(
        AuthMethod::Jwt,
        prompts::WEB_SEARCH_PROMPT_DRAFTER_MODEL.to_string(),
    );

    match get_chat_completion_response(
        context.state,
        context.user,
        request,
        &headers,
        billing_context,
    )
    .await
    {
        Ok(mut completion) => match completion.stream.recv().await {
            Some(CompletionChunk::FullResponse(response_json)) => extract_completion_text(
                response_json
                    .get("choices")
                    .and_then(|choices| choices.get(0))
                    .and_then(|choice| choice.get("message")),
            )
            .map(|query| query.trim().to_string())
            .filter(|query| !query.is_empty())
            .unwrap_or_else(|| {
                warn!("Failed to prepare web search query, using original message");
                context.user_message.to_string()
            }),
            Some(CompletionChunk::Error(err)) => {
                warn!(
                    "Web search prompt drafting returned an error, using original message: {}",
                    err
                );
                context.user_message.to_string()
            }
            _ => {
                warn!("Unexpected web search prompt drafting response, using original message");
                context.user_message.to_string()
            }
        },
        Err(e) => {
            warn!(
                "Web search prompt drafting failed, using original message: {:?}",
                e
            );
            context.user_message.to_string()
        }
    }
}

/// Execute web search using Brave or Kagi Search API.
///
/// This remains the fallback path when TinFoil web search is unavailable or
/// fails. Prefers Brave if available, then falls back to Kagi.
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

async fn execute_web_search_with_context(
    context: &WebSearchExecutionContext<'_>,
    query: &str,
    brave_client: Option<&Arc<BraveClient>>,
    kagi_client: Option<&Arc<KagiClient>>,
) -> Result<String, String> {
    trace!("Executing context-aware web search for query: {}", query);

    let mut tinfoil_error = None;

    if context.state.proxy_router.get_tinfoil_proxy().is_some() {
        match execute_tinfoil_web_search(context, query).await {
            Ok(output) => return Ok(output),
            Err(err) => {
                warn!(
                    "TinFoil web search failed, falling back to legacy providers if available: {}",
                    err
                );
                tinfoil_error = Some(err);
            }
        }
    }

    if brave_client.is_some() || kagi_client.is_some() {
        let fallback_query = extract_fallback_search_query(context).await;
        return execute_web_search(&fallback_query, brave_client, kagi_client).await;
    }

    if let Some(err) = tinfoil_error {
        Err(err)
    } else {
        error!("No search client configured");
        Err("No search client configured".to_string())
    }
}

async fn execute_tinfoil_web_search(
    context: &WebSearchExecutionContext<'_>,
    subagent_prompt: &str,
) -> Result<String, String> {
    trace!(
        "Executing TinFoil web search with prepared subagent prompt: {}",
        subagent_prompt
    );

    let request = prompts::build_web_search_subagent_request(subagent_prompt);
    let headers = HeaderMap::new();
    let billing_context = BillingContext::new(
        AuthMethod::Jwt,
        prompts::WEB_SEARCH_SUBAGENT_MODEL.to_string(),
    );

    let mut completion = get_chat_completion_response(
        context.state,
        context.user,
        request,
        &headers,
        billing_context,
    )
    .await
    .map_err(|e| format!("TinFoil web search request failed: {:?}", e))?;

    match completion.stream.recv().await {
        Some(CompletionChunk::FullResponse(response_json)) => {
            trace!(
                "TinFoil web search response: {}",
                serde_json::to_string_pretty(&response_json)
                    .unwrap_or_else(|_| "failed to serialize".to_string())
            );
            format_tinfoil_web_search_response(&response_json)
        }
        Some(CompletionChunk::Error(err)) => Err(format!("TinFoil web search error: {}", err)),
        _ => Err("TinFoil web search returned unexpected response".to_string()),
    }
}

async fn extract_fallback_search_query(context: &WebSearchExecutionContext<'_>) -> String {
    let query_request =
        prompts::build_query_extraction_request(context.conversation_history, context.user_message);
    let headers = HeaderMap::new();
    let billing_context =
        BillingContext::new(AuthMethod::Jwt, prompts::QUERY_EXTRACTOR_MODEL.to_string());

    match get_chat_completion_response(
        context.state,
        context.user,
        query_request,
        &headers,
        billing_context,
    )
    .await
    {
        Ok(mut completion) => match completion.stream.recv().await {
            Some(CompletionChunk::FullResponse(response_json)) => extract_completion_text(
                response_json
                    .get("choices")
                    .and_then(|choices| choices.get(0))
                    .and_then(|choice| choice.get("message")),
            )
            .map(|query| query.trim().to_string())
            .filter(|query| !query.is_empty())
            .unwrap_or_else(|| {
                warn!("Failed to extract fallback search query, using original message");
                context.user_message.to_string()
            }),
            Some(CompletionChunk::Error(err)) => {
                warn!(
                    "Fallback query extraction returned an error, using original message: {}",
                    err
                );
                context.user_message.to_string()
            }
            _ => {
                warn!("Unexpected fallback query extraction response, using original message");
                context.user_message.to_string()
            }
        },
        Err(e) => {
            warn!(
                "Fallback query extraction failed, using original message: {:?}",
                e
            );
            context.user_message.to_string()
        }
    }
}

/// Execute web search using Brave Search API
async fn execute_brave_search(query: &str, client: &Arc<BraveClient>) -> Result<String, String> {
    trace!("Executing Brave search for query: {}", query);

    // Create search request with summary enabled
    let mut search_request = BraveSearchRequest::new(query.to_string());
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
        warn!("No search results found");
        return Ok(format!("No results found for query: '{}'", query));
    }

    Ok(result_text)
}

fn format_tinfoil_web_search_response(response_json: &Value) -> Result<String, String> {
    let message = response_json
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .ok_or_else(|| "TinFoil web search returned no assistant message".to_string())?;

    let content =
        sanitize_tinfoil_summary(&extract_completion_text(Some(message)).unwrap_or_default());
    let blocked_searches = extract_blocked_searches(message);
    let citations = extract_url_citations(message);

    let mut sections = Vec::new();

    if !content.trim().is_empty() {
        sections.push(format!("Search Summary:\n\n{}", content.trim()));
    }

    if !blocked_searches.is_empty() {
        let blocked = blocked_searches
            .iter()
            .map(|blocked| format!("- {} ({})", blocked.query, blocked.reason))
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("Blocked Searches:\n{}", blocked));
    }

    if !citations.is_empty() {
        let sources = citations
            .iter()
            .enumerate()
            .map(|(index, citation)| {
                format!("{}. {} — {}", index + 1, citation.title, citation.url)
            })
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("Sources:\n{}", sources));
    }

    if sections.is_empty() {
        Err("TinFoil web search returned no content".to_string())
    } else {
        Ok(sections.join("\n\n"))
    }
}

fn sanitize_tinfoil_summary(content: &str) -> String {
    let mut cleaned_lines = Vec::new();

    for line in content.lines() {
        if is_sources_heading(line) {
            break;
        }

        cleaned_lines.push(line);
    }

    cleaned_lines.join("\n").trim().to_string()
}

fn is_sources_heading(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.eq_ignore_ascii_case("sources:")
        || trimmed
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("sources:"))
}

fn extract_completion_text(message: Option<&Value>) -> Option<String> {
    let content = message?.get("content")?;

    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let joined = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(|text| text.as_str())
                        .or_else(|| part.get("content").and_then(|text| text.as_str()))
                        .map(|text| text.to_string())
                })
                .collect::<Vec<_>>()
                .join("\n");

            if joined.trim().is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

fn extract_url_citations(message: &Value) -> Vec<UrlCitation> {
    let mut seen_urls = HashSet::new();
    let mut citations = Vec::new();

    let Some(annotations) = message
        .get("annotations")
        .and_then(|annotations| annotations.as_array())
    else {
        return citations;
    };

    for annotation in annotations {
        let Some(citation) = annotation.get("url_citation") else {
            continue;
        };

        let Some(url) = citation.get("url").and_then(|url| url.as_str()) else {
            continue;
        };

        let url = url.trim();
        if url.is_empty() || !seen_urls.insert(url.to_string()) {
            continue;
        }

        let title = citation
            .get("title")
            .and_then(|title| title.as_str())
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .unwrap_or(url);

        citations.push(UrlCitation {
            title: title.to_string(),
            url: url.to_string(),
        });
    }

    citations
}

fn extract_blocked_searches(message: &Value) -> Vec<BlockedSearch> {
    message
        .get("blocked_searches")
        .and_then(|blocked_searches| blocked_searches.as_array())
        .map(|blocked_searches| {
            blocked_searches
                .iter()
                .filter_map(|blocked| {
                    let query = blocked.get("query")?.as_str()?.trim();
                    let reason = blocked.get("reason")?.as_str()?.trim();

                    if query.is_empty() || reason.is_empty() {
                        return None;
                    }

                    Some(BlockedSearch {
                        query: query.to_string(),
                        reason: reason.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Execute a tool by name with the given arguments
///
/// This is the main entry point for tool execution. It routes to the appropriate
/// tool implementation based on the tool name.
///
/// # Arguments
/// * `tool_name` - The name of the tool to execute (e.g., "web_search")
/// * `arguments` - JSON object containing the tool's arguments
/// * `web_search_context` - Optional context for TinFoil-backed search execution
/// * `brave_client` - Optional Brave client (with connection pooling)
/// * `kagi_client` - Optional Kagi client (with connection pooling)
///
/// # Returns
/// * `Ok(String)` - The tool's output as a string
/// * `Err(String)` - An error message if the tool execution failed
pub async fn execute_tool(
    tool_name: &str,
    arguments: &Value,
    web_search_context: Option<&WebSearchExecutionContext<'_>>,
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

            if let Some(context) = web_search_context {
                execute_web_search_with_context(context, query, brave_client, kagi_client).await
            } else {
                execute_web_search(query, brave_client, kagi_client).await
            }
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
        let result = execute_tool("web_search", &args, None, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'query'"));
    }

    #[tokio::test]
    async fn test_execute_tool_unknown() {
        let args = json!({"query": "test"});
        let result = execute_tool("unknown_tool", &args, None, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tool"));
    }

    #[test]
    fn test_format_tinfoil_web_search_response_includes_sources() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "SpaceX generated about $15.6 billion in revenue in 2025.【1】",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "title": "SpaceX Revenue Report",
                            "url": "https://example.com/spacex-revenue"
                        }
                    }]
                }
            }]
        });

        let result = format_tinfoil_web_search_response(&response).unwrap();
        assert!(result.contains("Search Summary:"));
        assert!(result.contains("SpaceX generated about $15.6 billion"));
        assert!(result.contains("Sources:"));
        assert!(result.contains("SpaceX Revenue Report"));
        assert!(result.contains("https://example.com/spacex-revenue"));
    }

    #[test]
    fn test_format_tinfoil_web_search_response_includes_blocked_searches() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "I could not use web search for this request.",
                    "blocked_searches": [{
                        "query": "search for account number 1234567890",
                        "reason": "Bank account number detected"
                    }]
                }
            }]
        });

        let result = format_tinfoil_web_search_response(&response).unwrap();
        assert!(result.contains("Blocked Searches:"));
        assert!(result.contains("Bank account number detected"));
    }

    #[test]
    fn test_extract_completion_text_from_content_array() {
        let message = json!({
            "content": [
                {"type": "text", "text": "First line"},
                {"type": "text", "text": "Second line"}
            ]
        });

        let text = extract_completion_text(Some(&message)).unwrap();
        assert_eq!(text, "First line\nSecond line");
    }

    #[test]
    fn test_sanitize_tinfoil_summary_strips_embedded_sources_section() {
        let summary = sanitize_tinfoil_summary(
            "It will be cloudy today.\n\nSources:\nExample Weather: https://example.com/weather",
        );

        assert_eq!(summary, "It will be cloudy today.");
    }

    #[test]
    fn test_format_tinfoil_web_search_response_avoids_duplicate_sources_sections() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "It will be cloudy today.【1】\n\nSources:\nExample Weather: https://example.com/weather",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "title": "Example Weather",
                            "url": "https://example.com/weather"
                        }
                    }]
                }
            }]
        });

        let result = format_tinfoil_web_search_response(&response).unwrap();
        assert_eq!(result.matches("Sources:").count(), 1);
        assert!(result.contains("It will be cloudy today."));
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
