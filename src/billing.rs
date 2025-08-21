use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct UsageResponse {
    pub can_use: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BillingError {
    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("Failed to parse response: {0}")]
    ParseError(String),
    #[error("Service error: {0}")]
    ServiceError(String),
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

    async fn check_usage(&self, user_id: Uuid, is_api: bool) -> Result<bool, BillingError> {
        let mut url = format!(
            "{}/v1/admin/check-usage?user_id={}&product=maple",
            self.base_url, user_id
        );

        if is_api {
            url.push_str("&api=true");
        }

        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .send()
            .await?;

        if response.status().is_success() {
            response
                .json::<UsageResponse>()
                .await
                .map(|usage| usage.can_use)
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
        self.check_usage(user_id, false).await
    }

    pub async fn can_user_chat_api(&self, user_id: Uuid) -> Result<bool, BillingError> {
        self.check_usage(user_id, true).await
    }
}
