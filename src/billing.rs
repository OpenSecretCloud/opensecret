use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct UsageResponse {
    pub can_use: bool,
    #[serde(default)]
    pub is_free: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BillingError {
    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("Failed to parse response: {0}")]
    ParseError(String),
    #[error("Service error: {0}")]
    ServiceError(String),
    #[error("Usage limit reached")]
    UsageLimitExceeded,
    #[error("Token limit exceeded on free plan")]
    FreeTokenLimitExceeded,
}

#[derive(Clone)]
pub struct BillingClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl BillingClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    async fn check_usage(
        &self,
        user_id: Uuid,
        is_api: bool,
    ) -> Result<UsageResponse, BillingError> {
        let mut request = self
            .client
            .get(format!("{}/v1/admin/check-usage", self.base_url))
            .query(&[
                ("user_id", user_id.to_string()),
                ("product", "maple".to_string()),
            ])
            .header("x-api-key", &self.api_key);

        if is_api {
            request = request.query(&[("api", "true".to_string())]);
        }

        let response = request.send().await?;

        if response.status().is_success() {
            response
                .json::<UsageResponse>()
                .await
                .map_err(|e| BillingError::ParseError(e.to_string()))
        } else {
            let error = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(BillingError::ServiceError(error))
        }
    }

    pub async fn can_user_chat(&self, user_id: Uuid) -> Result<bool, BillingError> {
        self.check_usage(user_id, false)
            .await
            .map(|usage| usage.can_use)
    }

    pub async fn can_user_chat_api(&self, user_id: Uuid) -> Result<bool, BillingError> {
        self.check_usage(user_id, true)
            .await
            .map(|usage| usage.can_use)
    }

    pub async fn check_user_chat_with_tokens(
        &self,
        user_id: Uuid,
        is_api: bool,
        input_tokens: i32,
    ) -> Result<(), BillingError> {
        let usage = self.check_usage(user_id, is_api).await?;

        if !usage.can_use {
            return Err(BillingError::UsageLimitExceeded);
        }

        // Check free tier token limit (20k tokens)
        if usage.is_free && input_tokens > 20_000 {
            return Err(BillingError::FreeTokenLimitExceeded);
        }

        Ok(())
    }

    /// Check if a user is on a paid plan (not free)
    pub async fn is_user_paid(&self, user_id: Uuid) -> Result<bool, BillingError> {
        let usage = self.check_usage(user_id, false).await?;
        Ok(!usage.is_free)
    }
}
