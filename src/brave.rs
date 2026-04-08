use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

const BRAVE_API_BASE: &str = "https://api.search.brave.com/res/v1";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
pub enum BraveError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub count: Option<u32>,
    pub freshness: Option<String>,
    pub location: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Clone)]
pub struct BraveClient {
    client: reqwest::Client,
    api_key: Arc<String>,
}

impl BraveClient {
    pub fn new(api_key: String) -> Result<Self, BraveError> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(100)
            .user_agent("OpenSecret/0.1.0")
            .build()
            .map_err(BraveError::Request)?;

        Ok(Self {
            client,
            api_key: Arc::new(api_key),
        })
    }

    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse, BraveError> {
        let url = format!("{}/web/search", BRAVE_API_BASE);

        let mut query_params = vec![("q", request.query.clone())];

        if let Some(country) = &request.country {
            query_params.push(("country", country.clone()));
        }
        if let Some(search_lang) = &request.search_lang {
            query_params.push(("search_lang", search_lang.clone()));
        }
        if let Some(count) = request.count {
            query_params.push(("count", count.min(20).to_string()));
        }
        if let Some(offset) = request.offset {
            query_params.push(("offset", offset.to_string()));
        }
        if let Some(safesearch) = &request.safesearch {
            query_params.push(("safesearch", safesearch.clone()));
        }
        if let Some(freshness) = &request.freshness {
            query_params.push(("freshness", freshness.clone()));
        }
        if let Some(summary) = request.summary {
            query_params.push(("summary", if summary { "1" } else { "0" }.to_string()));
        }
        if let Some(extra_snippets) = request.extra_snippets {
            query_params.push((
                "extra_snippets",
                if extra_snippets { "true" } else { "false" }.to_string(),
            ));
        }
        if let Some(enable_rich_callback) = request.enable_rich_callback {
            query_params.push((
                "enable_rich_callback",
                if enable_rich_callback { "1" } else { "0" }.to_string(),
            ));
        }
        if let Some(spellcheck) = request.spellcheck {
            query_params.push((
                "spellcheck",
                if spellcheck { "true" } else { "false" }.to_string(),
            ));
        }

        let mut request_builder = self
            .client
            .get(&url)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json");

        if let Some(timezone) = &request.timezone {
            request_builder = request_builder.header("x-loc-timezone", timezone);
        }

        if let Some(location) = &request.location {
            let parts: Vec<&str> = location.split(',').map(|s| s.trim()).collect();
            if let Some(city) = parts.first().filter(|s| !s.is_empty()) {
                request_builder = request_builder.header("x-loc-city", *city);
            }
            if let Some(state_name) = parts.get(1).filter(|s| !s.is_empty()) {
                request_builder = request_builder.header("x-loc-state-name", *state_name);
            }
        }

        let response = request_builder.query(&query_params).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(BraveError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        response
            .json::<SearchResponse>()
            .await
            .map_err(BraveError::Request)
    }

    pub async fn search_with_options(
        &self,
        query: &str,
        options: Option<SearchOptions>,
    ) -> Result<SearchResponse, BraveError> {
        let options = options.unwrap_or_default();
        let request = SearchRequest {
            query: query.to_string(),
            country: None,
            search_lang: None,
            count: options.count.or(Some(10)),
            offset: None,
            safesearch: None,
            freshness: options.freshness,
            summary: Some(true),
            extra_snippets: Some(true),
            enable_rich_callback: Some(true),
            spellcheck: Some(true),
            location: options.location,
            timezone: options.timezone,
        };

        let mut search_response = self.search(request).await?;

        let summarizer_key = search_response.summarizer.as_ref().map(|s| s.key.clone());
        if let Some(key) = summarizer_key {
            debug!("Fetching Brave AI summary");
            match self.summarizer(&key).await {
                Ok(summary_response) => {
                    search_response.summary_text = summary_response.extract_text();
                }
                Err(err) => {
                    warn!("Failed to fetch Brave summary: {err}");
                }
            }
        }

        let rich_callback = search_response
            .rich
            .as_ref()
            .map(|r| (r.hint.vertical.clone(), r.hint.callback_key.clone()));
        if let Some((vertical, callback_key)) = rich_callback {
            debug!("Fetching Brave rich data for vertical: {vertical}");
            match self.fetch_rich(&callback_key).await {
                Ok(rich_response) => {
                    search_response.rich_data = Some(rich_response);
                }
                Err(err) => {
                    warn!("Failed to fetch Brave rich data: {err}");
                }
            }
        }

        Ok(search_response)
    }

    pub async fn summarizer(&self, key: &str) -> Result<SummarizerSearchResponse, BraveError> {
        let url = format!("{}/summarizer/search", BRAVE_API_BASE);

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json")
            .query(&[("key", key)])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(BraveError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        response
            .json::<SummarizerSearchResponse>()
            .await
            .map_err(BraveError::Request)
    }

    async fn fetch_rich(&self, callback_key: &str) -> Result<RichResponse, BraveError> {
        let url = format!("{}/web/rich", BRAVE_API_BASE);

        let response = self
            .client
            .get(&url)
            .header("X-Subscription-Token", self.api_key.as_str())
            .header("Accept", "application/json")
            .query(&[("callback_key", callback_key)])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(BraveError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        response
            .json::<RichResponse>()
            .await
            .map_err(BraveError::Request)
    }
}

impl std::fmt::Debug for BraveClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BraveClient")
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safesearch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_snippets: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rich_callback: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spellcheck: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl SearchRequest {
    pub fn new(query: String) -> Self {
        Self {
            query,
            country: None,
            search_lang: None,
            count: Some(10),
            offset: None,
            safesearch: None,
            freshness: None,
            summary: None,
            extra_snippets: None,
            enable_rich_callback: None,
            spellcheck: None,
            location: None,
            timezone: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct SearchResponse {
    #[serde(rename = "type")]
    pub response_type: Option<String>,
    pub query: Option<QueryInfo>,
    pub web: Option<WebResults>,
    pub news: Option<NewsResults>,
    pub faq: Option<FaqResults>,
    pub discussions: Option<DiscussionResults>,
    pub infobox: Option<Infobox>,
    pub summarizer: Option<Summarizer>,
    pub rich: Option<RichHint>,
    #[serde(skip)]
    pub summary_text: Option<String>,
    #[serde(skip)]
    pub rich_data: Option<RichResponse>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct QueryInfo {
    pub original: Option<String>,
    pub altered: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct WebResults {
    #[serde(rename = "type")]
    pub results_type: Option<String>,
    pub results: Option<Vec<SearchResult>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct NewsResults {
    #[serde(rename = "type")]
    pub results_type: Option<String>,
    pub results: Option<Vec<NewsResult>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct FaqResults {
    pub results: Option<Vec<FaqResult>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct DiscussionResults {
    pub results: Option<Vec<DiscussionResult>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub description: Option<String>,
    pub age: Option<String>,
    pub extra_snippets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct NewsResult {
    pub title: String,
    pub url: String,
    pub description: Option<String>,
    pub age: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct FaqResult {
    pub question: String,
    pub answer: String,
    pub title: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DiscussionResult {
    pub title: String,
    pub url: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct Infobox {
    #[serde(rename = "type")]
    pub infobox_type: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub long_desc: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Summarizer {
    #[serde(rename = "type")]
    pub summarizer_type: Option<String>,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RichHint {
    #[serde(rename = "type")]
    pub rich_type: Option<String>,
    pub hint: RichHintDetails,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RichHintDetails {
    pub vertical: String,
    pub callback_key: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct SummarizerSearchResponse {
    pub status: Option<String>,
    pub summary: Option<Vec<SummaryItem>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SummaryItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub data: Option<serde_json::Value>,
}

impl SummarizerSearchResponse {
    pub fn extract_text(&self) -> Option<String> {
        let items = self.summary.as_ref()?;
        let mut text = String::new();

        for item in items {
            if item.item_type == "token" {
                if let Some(data) = &item.data {
                    if let Some(value) = data.as_str() {
                        text.push_str(value);
                    }
                }
            }
        }

        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct RichResponse {
    #[serde(rename = "type")]
    pub response_type: Option<String>,
    pub results: Option<Vec<RichResult>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RichResult {
    #[serde(rename = "type")]
    pub result_type: Option<String>,
    pub subtype: Option<String>,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

impl RichResponse {
    pub fn format(&self) -> Option<String> {
        let results = self.results.as_ref()?;
        let first = results.first()?;
        first.format()
    }
}

impl RichResult {
    pub fn format(&self) -> Option<String> {
        match self.subtype.as_deref()? {
            "weather" => self.format_weather(),
            "stock" => self.format_stock(),
            "currency" => self.format_currency(),
            "cryptocurrency" => self.format_crypto(),
            "calculator" => self.format_calculator(),
            "unit_conversion" => self.format_unit_conversion(),
            "definitions" => self.format_definition(),
            subtype => Some(format!(
                "{}\n{}",
                subtype,
                serde_json::to_string_pretty(&self.data).unwrap_or_default()
            )),
        }
    }

    fn format_weather(&self) -> Option<String> {
        let mut output = String::new();

        if let Some(weather) = self.data.get("weather") {
            if let Some(location) = weather.get("location") {
                let name = location
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("Unknown");
                let state = location
                    .get("state")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                output.push_str(&format!("Weather for {}, {}\n\n", name, state));
            }

            if let Some(current) = weather.get("current_weather") {
                output.push_str("Current Conditions:\n");
                if let Some(temp_c) = current.get("temp").and_then(|value| value.as_f64()) {
                    output.push_str(&format!(
                        "  Temperature: {:.0}°F\n",
                        temp_c * 9.0 / 5.0 + 32.0
                    ));
                }
                if let Some(feels_c) = current.get("feels_like").and_then(|value| value.as_f64()) {
                    output.push_str(&format!(
                        "  Feels like: {:.0}°F\n",
                        feels_c * 9.0 / 5.0 + 32.0
                    ));
                }
                if let Some(desc) = current
                    .get("weather")
                    .and_then(|value| value.get("description"))
                    .and_then(|value| value.as_str())
                {
                    output.push_str(&format!("  Conditions: {}\n", desc));
                }
                if let Some(humidity) = current.get("humidity") {
                    output.push_str(&format!("  Humidity: {}%\n", humidity));
                }
                if let Some(wind_ms) = current
                    .get("wind")
                    .and_then(|value| value.get("speed"))
                    .and_then(|value| value.as_f64())
                {
                    output.push_str(&format!("  Wind: {:.0} mph\n", wind_ms * 2.237));
                }
                output.push('\n');
            }

            if let Some(alerts) = weather.get("alerts").and_then(|value| value.as_array()) {
                if !alerts.is_empty() {
                    output.push_str("Weather Alerts:\n");
                    for alert in alerts.iter().take(3) {
                        if let Some(event) = alert.get("event").and_then(|value| value.as_str()) {
                            output.push_str(&format!("  - {}\n", event));
                            if let Some(desc) =
                                alert.get("description").and_then(|value| value.as_str())
                            {
                                let short_desc: String = desc.chars().take(200).collect();
                                output.push_str(&format!(
                                    "    {}{}\n",
                                    short_desc,
                                    if desc.len() > 200 { "..." } else { "" }
                                ));
                            }
                        }
                    }
                    output.push('\n');
                }
            }

            if let Some(daily) = weather.get("daily").and_then(|value| value.as_array()) {
                output.push_str("Forecast:\n");
                for (idx, day) in daily.iter().take(5).enumerate() {
                    let day_name = day
                        .get("date_i18n")
                        .and_then(|value| value.as_str())
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| match idx {
                            0 => "Today".to_string(),
                            1 => "Tomorrow".to_string(),
                            _ => format!("Day {}", idx + 1),
                        });

                    let high = day
                        .get("temperature")
                        .and_then(|value| value.get("max"))
                        .and_then(|value| value.as_f64())
                        .map(|value| format!("{:.0}°F", value * 9.0 / 5.0 + 32.0))
                        .unwrap_or_default();
                    let low = day
                        .get("temperature")
                        .and_then(|value| value.get("min"))
                        .and_then(|value| value.as_f64())
                        .map(|value| format!("{:.0}°F", value * 9.0 / 5.0 + 32.0))
                        .unwrap_or_default();
                    let desc = day
                        .get("weather")
                        .and_then(|value| value.get("description"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("");

                    output.push_str(&format!(
                        "  {} - High: {}, Low: {} - {}\n",
                        day_name, high, low, desc
                    ));
                }
            }
        } else {
            output.push_str(&serde_json::to_string_pretty(&self.data).unwrap_or_default());
        }

        if output.is_empty() {
            None
        } else {
            Some(output)
        }
    }

    fn format_stock(&self) -> Option<String> {
        let mut output = String::from("Stock:\n\n");

        if let Some(symbol) = self.data.get("symbol").and_then(|value| value.as_str()) {
            output.push_str(&format!("Symbol: {}\n", symbol));
        }
        if let Some(name) = self.data.get("name").and_then(|value| value.as_str()) {
            output.push_str(&format!("Name: {}\n", name));
        }
        if let Some(price) = self.data.get("price") {
            output.push_str(&format!("Price: ${}\n", price));
        }
        if let Some(change) = self.data.get("change") {
            output.push_str(&format!("Change: {}\n", change));
        }
        if let Some(change_pct) = self.data.get("change_percent") {
            output.push_str(&format!("Change %: {}%\n", change_pct));
        }

        Some(output)
    }

    fn format_currency(&self) -> Option<String> {
        Some(format!(
            "Currency Conversion:\n\n{}",
            serde_json::to_string_pretty(&self.data).unwrap_or_default()
        ))
    }

    fn format_crypto(&self) -> Option<String> {
        let mut output = String::from("Cryptocurrency:\n\n");

        if let Some(name) = self.data.get("name").and_then(|value| value.as_str()) {
            output.push_str(&format!("Name: {}\n", name));
        }
        if let Some(symbol) = self.data.get("symbol").and_then(|value| value.as_str()) {
            output.push_str(&format!("Symbol: {}\n", symbol));
        }
        if let Some(price) = self.data.get("price") {
            output.push_str(&format!("Price: ${}\n", price));
        }
        if let Some(change) = self.data.get("change_24h") {
            output.push_str(&format!("24h Change: {}%\n", change));
        }

        Some(output)
    }

    fn format_calculator(&self) -> Option<String> {
        self.data.get("result").map(|result| match result.as_str() {
            Some(value) => format!("Calculator: {}", value),
            None => format!("Calculator: {}", result),
        })
    }

    fn format_unit_conversion(&self) -> Option<String> {
        let result = self.data.get("result").and_then(|value| value.as_str())?;
        Some(format!("Unit Conversion:\n{}", result))
    }

    fn format_definition(&self) -> Option<String> {
        let mut output = String::from("Definition:\n\n");

        if let Some(word) = self.data.get("word").and_then(|value| value.as_str()) {
            output.push_str(&format!("{}\n", word));
        }
        if let Some(definitions) = self
            .data
            .get("definitions")
            .and_then(|value| value.as_array())
        {
            for (idx, definition) in definitions.iter().take(3).enumerate() {
                if let Some(text) = definition
                    .get("definition")
                    .and_then(|value| value.as_str())
                {
                    output.push_str(&format!("{}. {}\n", idx + 1, text));
                }
            }
        }

        Some(output)
    }
}

impl SearchResponse {
    pub fn format_results(&self) -> String {
        let mut output = String::new();

        if let Some(query) = &self.query {
            if let Some(altered) = &query.altered {
                if query.original.as_ref() != Some(altered) {
                    output.push_str(&format!("Showing results for: {}\n\n", altered));
                }
            }
        }

        if let Some(rich) = &self.rich_data {
            if let Some(formatted) = rich.format() {
                output.push_str(&formatted);
                output.push_str("\n\n---\n\n");
            }
        }

        if let Some(summary) = &self.summary_text {
            output.push_str("AI Summary:\n");
            output.push_str(summary);
            output.push_str("\n\n---\n\n");
        }

        if let Some(infobox) = &self.infobox {
            if let Some(title) = &infobox.title {
                output.push_str(title);
                output.push('\n');
                if let Some(desc) = infobox.long_desc.as_ref().or(infobox.description.as_ref()) {
                    output.push_str(desc);
                    output.push_str("\n\n");
                }
            }
        }

        if let Some(faq) = &self.faq {
            if let Some(results) = &faq.results {
                if !results.is_empty() {
                    output.push_str("FAQ:\n\n");
                    for item in results.iter().take(3) {
                        output.push_str(&format!("Q: {}\nA: {}\n\n", item.question, item.answer));
                    }
                }
            }
        }

        if let Some(web) = &self.web {
            if let Some(results) = &web.results {
                if !results.is_empty() {
                    output.push_str("Search Results:\n\n");
                    for (idx, result) in results.iter().take(5).enumerate() {
                        let age = result
                            .age
                            .as_deref()
                            .map(|value| format!(" ({})", value))
                            .unwrap_or_default();
                        output.push_str(&format!(
                            "{}. {}{}\n   URL: {}\n   {}\n",
                            idx + 1,
                            result.title,
                            age,
                            result.url,
                            result.description.as_deref().unwrap_or("")
                        ));
                        if let Some(extra_snippets) = &result.extra_snippets {
                            for snippet in extra_snippets.iter().take(2) {
                                output.push_str(&format!("   > {}\n", snippet));
                            }
                        }
                        output.push('\n');
                    }
                }
            }
        }

        if let Some(news) = &self.news {
            if let Some(results) = &news.results {
                if !results.is_empty() {
                    output.push_str("Recent News:\n\n");
                    for (idx, result) in results.iter().take(3).enumerate() {
                        let age = result
                            .age
                            .as_deref()
                            .map(|value| format!(" ({})", value))
                            .unwrap_or_default();
                        output.push_str(&format!(
                            "{}. {}{}\n   URL: {}\n   {}\n\n",
                            idx + 1,
                            result.title,
                            age,
                            result.url,
                            result.description.as_deref().unwrap_or("")
                        ));
                    }
                }
            }
        }

        if let Some(discussions) = &self.discussions {
            if let Some(results) = &discussions.results {
                if !results.is_empty() {
                    output.push_str("Discussions:\n\n");
                    for result in results.iter().take(2) {
                        output.push_str(&format!("- {}\n  {}\n\n", result.title, result.url));
                    }
                }
            }
        }

        if output.is_empty() {
            "No results found.".to_string()
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn summarizer_extract_text_combines_token_items() {
        let response = SummarizerSearchResponse {
            status: Some("complete".to_string()),
            summary: Some(vec![
                SummaryItem {
                    item_type: "token".to_string(),
                    data: Some(json!("hello ")),
                },
                SummaryItem {
                    item_type: "enum_item".to_string(),
                    data: Some(json!({"ignored": true})),
                },
                SummaryItem {
                    item_type: "token".to_string(),
                    data: Some(json!("world")),
                },
            ]),
        };

        assert_eq!(response.extract_text().as_deref(), Some("hello world"));
    }

    #[test]
    fn format_results_includes_rich_summary_and_result_sections() {
        let response = SearchResponse {
            query: Some(QueryInfo {
                original: Some("weathr austin".to_string()),
                altered: Some("weather austin".to_string()),
            }),
            web: Some(WebResults {
                results_type: None,
                results: Some(vec![SearchResult {
                    title: "Austin forecast".to_string(),
                    url: "https://example.com/weather".to_string(),
                    description: Some("Sunny all week".to_string()),
                    age: Some("2h".to_string()),
                    extra_snippets: Some(vec!["High of 75F".to_string()]),
                }]),
            }),
            news: Some(NewsResults {
                results_type: None,
                results: Some(vec![NewsResult {
                    title: "Austin weather update".to_string(),
                    url: "https://example.com/news".to_string(),
                    description: Some("Cold front incoming".to_string()),
                    age: Some("1h".to_string()),
                }]),
            }),
            faq: Some(FaqResults {
                results: Some(vec![FaqResult {
                    question: "Will it rain?".to_string(),
                    answer: "No rain expected today.".to_string(),
                    title: None,
                    url: None,
                }]),
            }),
            discussions: Some(DiscussionResults {
                results: Some(vec![DiscussionResult {
                    title: "Austin weather thread".to_string(),
                    url: "https://example.com/discuss".to_string(),
                    description: None,
                }]),
            }),
            infobox: Some(Infobox {
                infobox_type: None,
                title: Some("Austin, TX".to_string()),
                description: Some("Current weather conditions".to_string()),
                long_desc: None,
            }),
            summarizer: None,
            rich: None,
            summary_text: Some("Warm and sunny today.".to_string()),
            rich_data: Some(RichResponse {
                response_type: None,
                results: Some(vec![RichResult {
                    result_type: Some("rich_result".to_string()),
                    subtype: Some("calculator".to_string()),
                    data: json!({"result": "72 + 3 = 75"}),
                }]),
            }),
            response_type: None,
        };

        let formatted = response.format_results();

        assert!(formatted.contains("Showing results for: weather austin"));
        assert!(formatted.contains("Calculator: 72 + 3 = 75"));
        assert!(formatted.contains("AI Summary:"));
        assert!(formatted.contains("Search Results:"));
        assert!(formatted.contains("Recent News:"));
        assert!(formatted.contains("Discussions:"));
    }
}
