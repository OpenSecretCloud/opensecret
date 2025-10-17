//! DSPy adapter for OpenSecret's completions API
//!
//! This module provides a custom Adapter implementation that integrates DSRs (DSPy Rust)
//! with our existing completions API infrastructure, ensuring billing, routing,
//! and auth are handled correctly.

use crate::{
    models::users::User,
    web::openai::{get_chat_completion_response, BillingContext, CompletionChunk},
    ApiError, AppState,
};
use async_trait::async_trait;
use axum::http::HeaderMap;
use dspy_rs::{adapter::Adapter, Chat, ChatAdapter, Example, LMResponse, LmUsage, Message, MetaSignature, Prediction, LM};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, trace, warn};

/// Custom LM implementation that wraps our completions API
///
/// This ensures all DSPy calls go through our centralized billing,
/// routing (primary/fallback), retry logic, and auth handling.
///
/// IMPORTANT: This implements the v0.6.0 API where LM takes &self (not &mut self)
/// and is wrapped in Arc<LM> (not Arc<Mutex<LM>>).
#[derive(Clone)]
pub struct OpenSecretLM {
    state: Arc<AppState>,
    user: User,
    billing_context: BillingContext,
}

impl OpenSecretLM {
    pub fn new(state: Arc<AppState>, user: User, billing_context: BillingContext) -> Self {
        Self {
            state,
            user,
            billing_context,
        }
    }

    /// Call our completions API (non-streaming)
    ///
    /// Converts DSRs Chat format → our API → back to DSRs Message format.
    /// Billing, routing, retries, and auth all happen inside get_chat_completion_response.
    ///
    /// NOTE: v0.6.0 signature - takes &self (not &mut), returns LMResponse (not tuple)
    pub async fn call(&self, messages: Chat) -> Result<LMResponse, ApiError> {
        debug!("OpenSecretLM: Starting DSPy LM call");

        // 1. Convert DSRs Chat → JSON
        let messages_json = messages.to_json();
        trace!("OpenSecretLM: Converted messages to JSON: {:?}", messages_json);

        // 2. Build request body
        let body = json!({
            "model": self.billing_context.model_name,
            "messages": messages_json,
            "stream": false  // Non-streaming for classification
        });

        debug!(
            "OpenSecretLM: Calling completions API with model {}",
            self.billing_context.model_name
        );

        // 3. Call OUR API (billing, routing, retries all handled!)
        let completion = get_chat_completion_response(
            &self.state,
            &self.user,
            body,
            &HeaderMap::new(),
            self.billing_context.clone(),
        )
        .await?;

        debug!("OpenSecretLM: Received completion stream");

        // 4. Extract response from CompletionChunk stream
        let mut rx = completion.stream;
        if let Some(CompletionChunk::FullResponse(response_json)) = rx.recv().await {
            trace!("OpenSecretLM: Received full response: {:?}", response_json);

            // 5. Parse response content
            let content = response_json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .ok_or_else(|| {
                    error!("OpenSecretLM: Failed to extract content from response");
                    ApiError::InternalServerError
                })?;

            debug!(
                "OpenSecretLM: Extracted content: {}",
                content.chars().take(100).collect::<String>()
            );

            // 6. Extract usage (DSPy uses u32 for token counts)
            let usage = if let Some(usage_json) = response_json.get("usage") {
                LmUsage {
                    prompt_tokens: usage_json
                        .get("prompt_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as u32,
                    completion_tokens: usage_json
                        .get("completion_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as u32,
                    total_tokens: usage_json
                        .get("total_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as u32,
                    reasoning_tokens: Some(0),
                }
            } else {
                LmUsage::default()
            };

            trace!("OpenSecretLM: Extracted usage: {:?}", usage);

            // 7. Create output message
            let output = Message::assistant(content);

            // 8. Build full chat history (input + output)
            let mut full_chat = messages.clone();
            full_chat.push_message(output.clone());

            // 9. Return v0.6.0 LMResponse struct
            debug!("OpenSecretLM: Call completed successfully");
            Ok(LMResponse {
                output,
                usage,
                chat: full_chat,
            })
        } else {
            error!("OpenSecretLM: Did not receive FullResponse chunk");
            Err(ApiError::InternalServerError)
        }
    }
}

/// Custom Adapter that uses OpenSecret's completions API instead of DSRs's default LM
///
/// This adapter implements the DSRs Adapter trait, allowing us to use DSRs's `Predict`
/// module and other patterns while maintaining our existing infrastructure for billing,
/// routing, and auth.
///
/// The adapter delegates formatting and parsing to ChatAdapter but overrides the `call`
/// method to use our custom OpenSecretLM instead of the standard DSRs LM.
#[derive(Clone)]
pub struct OpenSecretAdapter {
    state: Arc<AppState>,
    user: User,
    billing_context: BillingContext,
    /// We use ChatAdapter for standard formatting and parsing
    chat_adapter: ChatAdapter,
}

impl OpenSecretAdapter {
    pub fn new(state: Arc<AppState>, user: User, billing_context: BillingContext) -> Self {
        Self {
            state,
            user,
            billing_context,
            chat_adapter: ChatAdapter::default(),
        }
    }
}

#[async_trait]
impl Adapter for OpenSecretAdapter {
    /// Format the signature and inputs into a Chat (reuses ChatAdapter's logic)
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        self.chat_adapter.format(signature, inputs)
    }

    /// Parse the LM response back into structured output (reuses ChatAdapter's logic)
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        self.chat_adapter.parse_response(signature, response)
    }

    /// Call the LM - THIS is where we use our custom OpenSecretLM
    ///
    /// Note: The `lm` parameter is ignored because DSRs passes Arc<LM> but we need
    /// to use our custom OpenSecretLM that wraps our infrastructure. This is a
    /// necessary bridge to make DSRs work with our existing systems.
    async fn call(
        &self,
        _lm: Arc<LM>, // Ignored - we use our own LM
        signature: &dyn MetaSignature,
        inputs: Example,
    ) -> anyhow::Result<Prediction> {
        debug!(
            "OpenSecretAdapter: Calling {} signature",
            std::any::type_name_of_val(signature)
        );

        // Format inputs into Chat messages using standard ChatAdapter logic
        let messages = self.format(signature, inputs);

        // Use our custom LM that integrates with our infrastructure
        let our_lm = OpenSecretLM::new(
            self.state.clone(),
            self.user.clone(),
            self.billing_context.clone(),
        );

        // Call our LM
        let response = our_lm.call(messages).await.map_err(|e| {
            error!("OpenSecretAdapter: LM call failed: {:?}", e);
            anyhow::anyhow!("LM call failed: {:?}", e)
        })?;

        // Parse the response using standard ChatAdapter logic
        let output = self.parse_response(signature, response.output);

        debug!("OpenSecretAdapter: Successfully parsed response");

        // Return prediction with usage stats
        Ok(Prediction {
            data: output,
            lm_usage: response.usage,
        })
    }
}
