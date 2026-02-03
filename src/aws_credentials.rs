use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use vsock::{VsockAddr, VsockStream};

pub(crate) const PARENT_CID: u32 = 3;
pub(crate) const CREDENTIAL_REQUESTER_PORT: u32 = 8003;
pub(crate) const MAX_VSOCK_RESPONSE_BYTES: u64 = 256 * 1024;

pub(crate) fn set_vsock_timeouts(
    stream: &VsockStream,
    timeout: Duration,
) -> Result<(), AwsCredentialError> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

pub(crate) fn read_vsock_response_string_limited(
    stream: &mut VsockStream,
    max_bytes: u64,
) -> Result<String, AwsCredentialError> {
    let mut limited = (&mut *stream).take(max_bytes.saturating_add(1));
    let mut response = String::new();
    limited.read_to_string(&mut response)?;

    if limited.limit() == 0 {
        return Err(AwsCredentialError::ResponseTooLarge { max_bytes });
    }

    Ok(response)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AwsCredentials {
    #[serde(rename = "AccessKeyId")]
    pub access_key_id: String,
    #[serde(rename = "SecretAccessKey")]
    pub secret_access_key: String,
    #[serde(rename = "Token")]
    pub token: String,
    #[serde(rename = "Region")]
    pub region: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnclaveRequest {
    pub request_type: String,
    pub key_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParentResponse {
    pub response_type: String,
    pub response_value: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct ParentVsockPingStatus {
    pub last_attempt_unix_ms: Option<u64>,
    pub last_success_unix_ms: Option<u64>,
    pub last_pong_unix_ms: Option<u64>,
    pub last_rtt_ms: Option<u64>,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

pub fn unix_ms_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
        .min(u64::MAX as u128) as u64
}

#[derive(Debug, thiserror::Error)]
pub enum AwsCredentialError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Authentication error")]
    Authentication,

    #[error("Timed out waiting for credentials")]
    Timeout,

    #[error("VSOCK response exceeded {max_bytes} bytes")]
    ResponseTooLarge { max_bytes: u64 },

    #[error("Ping timed out after {0:?}")]
    PingTimeout(Duration),

    #[error("Ping failed: {0}")]
    PingFailed(String),

    #[error("Unexpected response type: {0}")]
    UnexpectedResponseType(String),

    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Clone, Default)]
pub struct AwsCredentialManager {
    credentials: Arc<RwLock<Option<AwsCredentials>>>,
    parent_vsock_ping_status: Arc<RwLock<ParentVsockPingStatus>>,
}

impl AwsCredentialManager {
    pub fn new() -> Self {
        Self {
            credentials: Arc::new(RwLock::new(None)),
            parent_vsock_ping_status: Arc::new(RwLock::new(Default::default())),
        }
    }

    pub async fn get_parent_vsock_ping_status(&self) -> ParentVsockPingStatus {
        self.parent_vsock_ping_status.read().await.clone()
    }

    pub async fn ping_credential_requester_and_update_status(&self, timeout: Duration) {
        let attempt_unix_ms = unix_ms_now();
        let res = self.ping_credential_requester(timeout).await;

        let mut status = self.parent_vsock_ping_status.write().await;
        status.last_attempt_unix_ms = Some(attempt_unix_ms);

        match res {
            Ok((pong_unix_ms, rtt_ms)) => {
                status.last_success_unix_ms = Some(unix_ms_now());
                status.last_pong_unix_ms = pong_unix_ms;
                status.last_rtt_ms = Some(rtt_ms);
                status.consecutive_failures = 0;
                status.last_error = None;
            }
            Err(e) => {
                status.consecutive_failures = status.consecutive_failures.saturating_add(1);
                status.last_error = Some(e.to_string());
            }
        }
    }

    async fn ping_credential_requester(
        &self,
        timeout: Duration,
    ) -> Result<(Option<u64>, u64), AwsCredentialError> {
        let start = Instant::now();

        let handle = tokio::task::spawn_blocking(move || {
            let res: Result<Option<u64>, AwsCredentialError> = (|| {
                let cid = PARENT_CID;
                let port = CREDENTIAL_REQUESTER_PORT;

                let sock_addr = VsockAddr::new(cid, port);
                let mut stream = VsockStream::connect(&sock_addr)?;
                set_vsock_timeouts(&stream, timeout)?;

                let request = EnclaveRequest {
                    request_type: "ping".to_string(),
                    key_name: None,
                };
                let request_json = serde_json::to_string(&request)?;
                stream.write_all(request_json.as_bytes())?;

                let response =
                    read_vsock_response_string_limited(&mut stream, MAX_VSOCK_RESPONSE_BYTES)?;

                let parent_response: ParentResponse = serde_json::from_str(&response)?;
                match parent_response.response_type.as_str() {
                    "pong" => Ok(parent_response
                        .response_value
                        .get("unix_ms")
                        .and_then(|v| v.as_u64())),
                    "error" => Err(AwsCredentialError::PingFailed(
                        parent_response
                            .response_value
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string(),
                    )),
                    other => Err(AwsCredentialError::UnexpectedResponseType(
                        other.to_string(),
                    )),
                }
            })();

            match res {
                Err(AwsCredentialError::Io(e))
                    if matches!(e.kind(), std::io::ErrorKind::TimedOut)
                        || matches!(e.kind(), std::io::ErrorKind::WouldBlock) =>
                {
                    Err(AwsCredentialError::PingTimeout(timeout))
                }
                other => other,
            }
        });

        // We rely on VSOCK socket timeouts (set on the stream) rather than an async timeout.
        // An async timeout does not cancel the blocking task.
        let pong_unix_ms = handle.await??;

        let rtt_ms = start.elapsed().as_millis().min(u64::MAX as u128) as u64;

        Ok((pong_unix_ms, rtt_ms))
    }

    pub async fn get_credentials(&self) -> Option<AwsCredentials> {
        let creds = self.credentials.read().await;
        creds.clone()
    }

    pub async fn set_credentials(&self, credentials: AwsCredentials) {
        let mut creds = self.credentials.write().await;
        *creds = Some(credentials);
    }

    pub async fn fetch_credentials(&self) -> Result<AwsCredentials, AwsCredentialError> {
        tracing::debug!("Entering fetch_credentials");

        let creds = Self::fetch_credentials_from_vsock().await?;
        self.set_credentials(creds.clone()).await;

        tracing::debug!("Exiting fetch_credentials");
        Ok(creds)
    }

    async fn fetch_credentials_from_vsock() -> Result<AwsCredentials, AwsCredentialError> {
        let cid = PARENT_CID;
        let port = CREDENTIAL_REQUESTER_PORT;

        let sock_addr = VsockAddr::new(cid, port);
        let mut stream = VsockStream::connect(&sock_addr)?;
        set_vsock_timeouts(&stream, Duration::from_secs(30))?;

        let request = EnclaveRequest {
            request_type: "credentials".to_string(),
            key_name: None,
        };
        let request_json = serde_json::to_string(&request)?;
        stream.write_all(request_json.as_bytes())?;

        let response = read_vsock_response_string_limited(&mut stream, MAX_VSOCK_RESPONSE_BYTES)?;

        let parent_response: ParentResponse = serde_json::from_str(&response)?;
        if parent_response.response_type == "credentials" {
            let creds: AwsCredentials = serde_json::from_value(parent_response.response_value)?;
            Ok(creds)
        } else {
            Err(AwsCredentialError::Authentication)
        }
    }

    pub async fn wait_for_credentials(&self) -> AwsCredentials {
        tracing::info!("Waiting for initial AWS credentials");
        let max_retries = 12; // 1 minute total with 5s delay
        let mut attempts = 0;

        loop {
            match self.fetch_credentials().await {
                Ok(c) => return c,
                Err(e) => {
                    attempts += 1;
                    if attempts >= max_retries {
                        tracing::error!(
                            "Failed to get credentials after {} attempts, giving up",
                            max_retries
                        );
                        panic!("Could not obtain AWS credentials after maximum retries");
                    }
                    tracing::error!("Failed to refresh AWS credentials: {:?}", e);
                    tracing::info!(
                        "Retrying in 5 seconds... (attempt {}/{})",
                        attempts,
                        max_retries
                    );
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
}
