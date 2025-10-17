//! DSPy modules for classification and query extraction
//!
//! This module provides DSPy-style wrappers around our signatures, following
//! the standard DSPy pattern of encapsulating Predict-like modules in domain-specific structs.
//!
//! These modules use our custom OpenSecretAdapter to integrate with our existing
//! infrastructure while maintaining the DSPy API style.

use crate::{
    models::users::User,
    web::{
        openai::BillingContext,
        responses::{prompts, OpenSecretAdapter},
    },
    ApiError, AppState,
};
use dspy_rs::{adapter::Adapter, core::lm::LM, example, MetaSignature};
use std::sync::Arc;
use tracing::{debug, warn};

/// IntentClassifier - Classifies user intent as "web_search" or "chat"
///
/// This follows the DSPy pattern of wrapping a signature with domain-specific logic.
/// Uses a fast, cheap model with temperature=0 for deterministic classification.
pub struct IntentClassifier {
    signature: Box<dyn MetaSignature>,
    adapter: OpenSecretAdapter,
    /// Dummy LM required by DSRs API but ignored by our adapter
    dummy_lm: Arc<dspy_rs::LM>,
}

impl IntentClassifier {
    /// Create a new intent classifier
    ///
    /// # Arguments
    /// * `state` - Application state for API access
    /// * `user` - User making the request
    pub async fn new(state: Arc<AppState>, user: User) -> Self {
        let billing_context = BillingContext::new(
            crate::web::openai_auth::AuthMethod::Jwt,
            "llama-3.3-70b".to_string(),
        );

        let adapter = OpenSecretAdapter::new(state, user, billing_context);
        let signature = Box::new(prompts::new_intent_classifier());

        // Create a dummy LM - won't be used because our adapter ignores it
        // We need a real LM instance to satisfy the API, but our adapter ignores it
        let dummy_lm = Arc::new(
            LM::builder()
                .api_key("dummy_key".into())
                .build()
                .await,
        );

        Self {
            signature,
            adapter,
            dummy_lm,
        }
    }

    /// Classify a user message as "web_search" or "chat"
    ///
    /// # Arguments
    /// * `message` - The user's message to classify
    ///
    /// # Returns
    /// - "web_search" if the user needs current information, facts, or search
    /// - "chat" if the user wants casual conversation or general discussion
    pub async fn classify(&self, message: &str) -> Result<String, ApiError> {
        debug!("IntentClassifier: Classifying message");

        let input = example! {
            "user_message": "input" => message,
        };

        // Call our adapter directly (similar to how Predict::forward works)
        // The dummy_lm is ignored by our adapter
        let result = self
            .adapter
            .call(self.dummy_lm.clone(), self.signature.as_ref(), input)
            .await
            .map_err(|e| {
                warn!("IntentClassifier: Classification failed: {:?}", e);
                ApiError::InternalServerError
            })?;

        let intent = result
            .get("intent", None)
            .as_str()
            .unwrap_or("chat")
            .trim()
            .to_lowercase();

        debug!("IntentClassifier: Classified as '{}'", intent);

        // Normalize to expected values
        let normalized = if intent.contains("web_search") || intent.contains("search") {
            "web_search".to_string()
        } else {
            "chat".to_string()
        };

        Ok(normalized)
    }
}

/// QueryExtractor - Extracts clean search queries from natural language
///
/// This follows the DSPy pattern of wrapping a signature with domain-specific logic.
/// Uses a fast, cheap model with temperature=0 for consistent extraction.
pub struct QueryExtractor {
    signature: Box<dyn MetaSignature>,
    adapter: OpenSecretAdapter,
    /// Dummy LM required by DSRs API but ignored by our adapter
    dummy_lm: Arc<dspy_rs::LM>,
}

impl QueryExtractor {
    /// Create a new query extractor
    ///
    /// # Arguments
    /// * `state` - Application state for API access
    /// * `user` - User making the request
    pub async fn new(state: Arc<AppState>, user: User) -> Self {
        let billing_context = BillingContext::new(
            crate::web::openai_auth::AuthMethod::Jwt,
            "llama-3.3-70b".to_string(),
        );

        let adapter = OpenSecretAdapter::new(state, user, billing_context);
        let signature = Box::new(prompts::new_query_extractor());

        // Create a dummy LM - won't be used because our adapter ignores it
        // We need a real LM instance to satisfy the API, but our adapter ignores it
        let dummy_lm = Arc::new(
            LM::builder()
                .api_key("dummy_key".into())
                .build()
                .await,
        );

        Self {
            signature,
            adapter,
            dummy_lm,
        }
    }

    /// Extract a clean search query from a natural language question
    ///
    /// # Arguments
    /// * `user_message` - The user's natural language question
    ///
    /// # Returns
    /// A concise search query extracted from the message
    pub async fn extract(&self, user_message: &str) -> Result<String, ApiError> {
        debug!("QueryExtractor: Extracting query from message");

        let input = example! {
            "user_message": "input" => user_message,
        };

        // Call our adapter directly (similar to how Predict::forward works)
        // The dummy_lm is ignored by our adapter
        let result = self
            .adapter
            .call(self.dummy_lm.clone(), self.signature.as_ref(), input)
            .await
            .map_err(|e| {
                warn!("QueryExtractor: Extraction failed: {:?}", e);
                ApiError::InternalServerError
            })?;

        let query = result
            .get("search_query", None)
            .as_str()
            .unwrap_or(user_message)
            .trim()
            .to_string();

        debug!("QueryExtractor: Extracted query: '{}'", query);

        Ok(query)
    }
}
