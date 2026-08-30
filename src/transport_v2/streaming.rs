//! Transport-neutral application streaming contract for encrypted transport v2.
//!
//! Application adapters emit ordinary response bytes plus an explicit terminal
//! signal. The gateway alone assigns transport sequence numbers, serializes
//! `StreamRecord`s, and encrypts them under the exact admitted session.

use std::pin::Pin;
use std::sync::Arc;

use axum::http::StatusCode;
use bytes::Bytes;
use futures::Stream;
use tokio::sync::OwnedSemaphorePermit;
use zeroize::Zeroizing;

use super::envelope::{EncodedBytes, HeaderField};

pub(crate) type LogicalByteStream = Pin<Box<dyn Stream<Item = LogicalStreamItem> + Send + 'static>>;

pub(crate) enum LogicalStreamItem {
    Bytes(Bytes),
    Complete,
    Failure(LogicalStreamFailure),
}

pub(crate) struct LogicalStreamFailure {
    pub(crate) status: StatusCode,
    pub(crate) body: Zeroizing<Vec<u8>>,
}

impl LogicalStreamFailure {
    pub(crate) fn protocol(status: StatusCode, code: &str, message: &str) -> Self {
        #[derive(serde::Serialize)]
        struct ErrorBody<'a> {
            error: ErrorDetails<'a>,
        }

        #[derive(serde::Serialize)]
        struct ErrorDetails<'a> {
            code: &'a str,
            message: &'a str,
        }

        let body = serde_json::to_vec(&ErrorBody {
            error: ErrorDetails { code, message },
        })
        .expect("fixed transport-v2 stream error body must serialize");
        Self {
            status,
            body: Zeroizing::new(body),
        }
    }

    pub(crate) fn internal() -> Self {
        Self::protocol(
            StatusCode::INTERNAL_SERVER_ERROR,
            "stream_failed",
            "Stream failed",
        )
    }
}

pub(crate) struct LogicalStreamResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: Vec<HeaderField>,
    pub(crate) stream: LogicalByteStream,
}

impl LogicalStreamResponse {
    pub(crate) fn sse(stream: LogicalByteStream) -> Self {
        Self {
            status: StatusCode::OK,
            headers: vec![HeaderField {
                name: "content-type".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"text/event-stream".to_vec()),
            }],
            stream,
        }
    }
}

/// Cloneable lifetime token for the single promoted provider working-set
/// permit. The response body and every detached v2 task that can retain
/// provider/application output hold the same permit rather than double-counting
/// or releasing admission capacity when only the network carrier disconnects.
#[derive(Clone)]
pub(crate) struct StreamExecutionGuard {
    permit: Arc<OwnedSemaphorePermit>,
}

impl StreamExecutionGuard {
    pub(super) fn new(permit: OwnedSemaphorePermit) -> Self {
        Self {
            permit: Arc::new(permit),
        }
    }

    #[cfg(test)]
    pub(crate) fn permits(&self) -> usize {
        self.permit.num_permits()
    }
}
