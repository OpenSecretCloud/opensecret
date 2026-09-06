//! Request correlation without exporting credentials, content, or client trace context.
//!
//! The root is server-generated. Logical Transport V2 requests are child spans,
//! and body polling/drop stays in the originating span after the handler returns.
use axum::{
    body::{Body, Bytes, HttpBody},
    extract::{MatchedPath, Request},
    http::{Method, StatusCode},
    middleware::Next,
    response::Response,
};
use http_body::{Frame, SizeHint};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};
use tracing::{Instrument, Span};
use uuid::Uuid;

/// A bounded diagnostic category, never an error's Display/Debug payload.
/// Walk typed sources so DB model wrappers retain useful SQL failure classes.
pub(crate) fn error_kind(mut error: &(dyn std::error::Error + 'static)) -> &'static str {
    for _ in 0..8 {
        if let Some(error) = error.downcast_ref::<crate::db::DBError>() {
            use crate::db::DBError;
            match error {
                DBError::ConnectionError => return "db_pool_acquire",
                DBError::UserNotFound
                | DBError::EmailVerificationNotFound
                | DBError::PasswordResetRequestNotFound
                | DBError::AccountDeletionRequestNotFound
                | DBError::OrgNotFound
                | DBError::OrgProjectNotFound
                | DBError::OrgProjectSecretNotFound
                | DBError::InviteCodeNotFound
                | DBError::PlatformInviteCodeNotFound
                | DBError::PlatformUserNotFound
                | DBError::PlatformEmailVerificationNotFound
                | DBError::PlatformPasswordResetRequestNotFound
                | DBError::OrgMembershipNotFound
                | DBError::ProjectSettingNotFound => return "db_not_found",
                _ => {}
            }
        }
        if let Some(error) = error.downcast_ref::<crate::provider_client::ProviderRequestError>() {
            use crate::provider_client::ProviderRequestError;
            return match error {
                ProviderRequestError::TinfoilUnavailable => "provider_unavailable",
                ProviderRequestError::Timeout(_) => "provider_timeout",
                ProviderRequestError::Build(_) => "provider_build",
                ProviderRequestError::Connect(_) => "provider_connect",
                ProviderRequestError::Send(_) => "provider_send",
                ProviderRequestError::Upstream(_) => "provider_status",
            };
        }
        if let Some(error) = error.downcast_ref::<diesel::result::Error>() {
            use diesel::result::{DatabaseErrorKind as Kind, Error};
            return match error {
                Error::DatabaseError(kind, _) => match kind {
                    Kind::UniqueViolation => "db_unique_violation",
                    Kind::ForeignKeyViolation => "db_foreign_key_violation",
                    Kind::NotNullViolation => "db_not_null_violation",
                    Kind::CheckViolation => "db_check_violation",
                    Kind::SerializationFailure => "db_serialization_failure",
                    Kind::ReadOnlyTransaction => "db_read_only_transaction",
                    Kind::ClosedConnection => "db_closed_connection",
                    _ => "db_query",
                },
                Error::NotFound => "db_not_found",
                Error::DeserializationError(_) => "db_deserialization",
                Error::SerializationError(_) => "db_serialization",
                _ => "db_query",
            };
        }
        if let Some(error) = error.downcast_ref::<serde_json::Error>() {
            return match error.classify() {
                serde_json::error::Category::Io => "json_io",
                serde_json::error::Category::Syntax => "json_syntax",
                serde_json::error::Category::Data => "json_data",
                serde_json::error::Category::Eof => "json_eof",
            };
        }
        if let Some(error) = error.downcast_ref::<reqwest::Error>() {
            return if error.is_timeout() {
                "http_timeout"
            } else if error.is_connect() {
                "http_connect"
            } else if error.is_decode() {
                "http_decode"
            } else if error.is_status() {
                "http_status"
            } else {
                "http_request"
            };
        }
        if error.is::<std::string::FromUtf8Error>() {
            return "invalid_utf8";
        }
        if error.is::<diesel::r2d2::PoolError>() {
            return "db_pool_timeout";
        }
        match error.source() {
            Some(source) => error = source,
            None => break,
        }
    }
    "operation_failed"
}

pub(crate) async fn trace_request(request: Request, next: Next) -> Response {
    // Never accept a caller's trace ID or log a raw path/query (including 404s).
    let span = tracing::info_span!(
        parent: None,
        "http_request",
        trace_id = %Uuid::new_v4().simple(),
        method = method_label(request.method()),
        route = route_label(&request),
    );
    trace_response(request, next, span).await
}

pub(crate) async fn trace_logical_request(request: Request, next: Next) -> Response {
    let span = tracing::info_span!(
        "logical_request",
        method = method_label(request.method()),
        route = route_label(&request),
    );
    trace_response(request, next, span).await
}

fn route_label(request: &Request) -> &str {
    request
        .extensions()
        .get::<MatchedPath>()
        .map_or("unmatched", MatchedPath::as_str)
}

fn method_label(method: &Method) -> &'static str {
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::PATCH => "PATCH",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::CONNECT => "CONNECT",
        Method::TRACE => "TRACE",
        _ => "OTHER",
    }
}

async fn trace_response(request: Request, next: Next, span: Span) -> Response {
    let quiet = matches!(
        route_label(&request),
        "/health-check" | "/health-check-extended"
    );
    let started = Instant::now();
    let response = next.run(request).instrument(span.clone()).await;
    let (parts, body) = response.into_parts();
    let mut body = TracedBody {
        inner: Some(body),
        span,
        started,
        status: parts.status,
        quiet,
        finished: false,
    };
    if body.inner.as_ref().expect("body present").is_end_stream() {
        body.finish("body_complete");
    }
    Response::from_parts(parts, Body::new(body))
}

struct TracedBody {
    // An Option lets Drop dispose of the inner stream in its span as well.
    inner: Option<Body>,
    span: Span,
    started: Instant,
    status: StatusCode,
    quiet: bool,
    finished: bool,
}

impl TracedBody {
    fn finish(&mut self, outcome: &'static str) {
        if self.finished {
            return;
        }
        self.finished = true;
        let _entered = self.span.enter();
        let elapsed_ms = self.started.elapsed().as_millis() as u64;
        let status = self.status.as_u16();
        // This describes HTTP body production, not client receipt, inference
        // success, or a durable storage commit. Those have their own events.
        if outcome == "body_error" || self.status.is_server_error() {
            tracing::warn!(status, elapsed_ms, outcome, "HTTP response body finished");
        } else if self.quiet || outcome == "body_dropped" {
            tracing::debug!(status, elapsed_ms, outcome, "HTTP response body finished");
        } else {
            tracing::info!(status, elapsed_ms, outcome, "HTTP response body finished");
        }
    }
}

impl HttpBody for TracedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let span = self.span.clone();
        let _entered = span.enter();
        let inner = self.inner.as_mut().expect("body present until drop");
        let result = Pin::new(&mut *inner).poll_frame(cx);
        match &result {
            Poll::Ready(Some(Err(_))) => self.finish("body_error"),
            Poll::Ready(None) => self.finish("body_complete"),
            Poll::Ready(Some(Ok(_))) if inner.is_end_stream() => self.finish("body_complete"),
            _ => {}
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.inner
            .as_ref()
            .expect("body present until drop")
            .is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner
            .as_ref()
            .expect("body present until drop")
            .size_hint()
    }
}

impl Drop for TracedBody {
    fn drop(&mut self) {
        self.finish("body_dropped");
        self.span.in_scope(|| drop(self.inner.take()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        http::{HeaderMap, Request as HttpRequest},
        middleware::from_fn,
        routing::get,
        Router,
    };
    use std::{
        collections::{HashSet, VecDeque},
        io::{self, Write},
        sync::{Arc, Mutex},
    };
    use tower::ServiceExt;

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn subscriber(&self) -> impl tracing::Subscriber + Send + Sync {
            let output = self.clone();
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .without_time()
                .with_writer(move || output.clone())
                .finish()
        }
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    struct FixtureBody(VecDeque<Result<Frame<Bytes>, io::Error>>);
    impl HttpBody for FixtureBody {
        type Data = Bytes;
        type Error = io::Error;
        fn poll_frame(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, io::Error>>> {
            tracing::info!("fixture body poll");
            Poll::Ready(self.0.pop_front())
        }
        fn is_end_stream(&self) -> bool {
            self.0.is_empty()
        }
        fn size_hint(&self) -> SizeHint {
            let mut hint = SizeHint::new();
            hint.set_exact(
                self.0
                    .iter()
                    .filter_map(|f| f.as_ref().ok()?.data_ref())
                    .map(|b| b.len() as u64)
                    .sum(),
            );
            hint
        }
    }
    impl Drop for FixtureBody {
        fn drop(&mut self) {
            tracing::info!("fixture body drop");
        }
    }

    fn trace_ids(text: &str) -> HashSet<String> {
        regex::Regex::new(r"trace_id=([0-9a-f]{32})")
            .unwrap()
            .captures_iter(text)
            .map(|capture| capture[1].to_string())
            .collect()
    }

    #[test]
    fn diagnostics_classify_wrapped_errors_without_formatting_sensitive_values() {
        let capture = Capture::default();
        let _subscriber = tracing::subscriber::set_default(capture.subscriber());
        let json = serde_json::from_str::<u64>("\"private-json-content\"").unwrap_err();
        let invalid_utf8 =
            String::from_utf8([b"private-kv-content".as_slice(), &[255]].concat()).unwrap_err();
        let database = crate::db::DBError::QueryError(diesel::result::Error::DatabaseError(
            diesel::result::DatabaseErrorKind::UniqueViolation,
            Box::new("private-db-message".to_owned()),
        ));
        let provider = crate::provider_client::ProviderRequestError::Connect(
            "private-provider-url".to_owned(),
        );
        for (error, kind) in [
            (&json as &dyn std::error::Error, "json_data"),
            (&invalid_utf8, "invalid_utf8"),
            (&database, "db_unique_violation"),
            (&provider, "provider_connect"),
        ] {
            assert_eq!(error_kind(error), kind);
            tracing::warn!(error_kind = error_kind(error), "fixture diagnostic");
        }
        assert!(!capture.text().contains("private-"));
        assert_eq!(
            error_kind(&crate::db::DBError::UserNotFound),
            "db_not_found"
        );
    }

    async fn poll_frame(body: &mut Body) -> Option<Result<Frame<Bytes>, axum::Error>> {
        std::future::poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await
    }

    #[tokio::test]
    async fn trace_covers_handler_background_task_body_and_drop_without_changing_frames() {
        let capture = Capture::default();
        let _subscriber = tracing::subscriber::set_default(capture.subscriber());
        let released = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(tokio::sync::Notify::new());
        let app = Router::new()
            .route(
                "/resource/:id",
                get({
                    let released = released.clone();
                    let completed = completed.clone();
                    move || async move {
                        tracing::info!("fixture handler");
                        tokio::spawn(
                            async move {
                                released.notified().await;
                                tracing::info!("fixture background after body");
                                completed.notify_one();
                            }
                            .in_current_span(),
                        );
                        let mut trailers = HeaderMap::new();
                        trailers.insert("x-fixture", "trailer".parse().unwrap());
                        let body = Body::new(FixtureBody(VecDeque::from([
                            Ok(Frame::data(Bytes::from_static(b"a"))),
                            Ok(Frame::data(Bytes::from_static(b"b"))),
                            Ok(Frame::trailers(trailers)),
                        ])));
                        Response::builder()
                            .status(StatusCode::CREATED)
                            .header("x-fixture", "header")
                            .body(body)
                            .unwrap()
                    }
                }),
            )
            .layer(from_fn(trace_request));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/resource/private-path?token=private-query")
                    .header("authorization", "Bearer private-token")
                    .header("traceparent", "private-traceparent")
                    .header("x-request-id", "private-request-id")
                    .body(Body::from("private-body"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()["x-fixture"], "header");
        let mut body = response.into_body();
        assert_eq!(body.size_hint().exact(), Some(2));
        assert_eq!(
            poll_frame(&mut body)
                .await
                .unwrap()
                .unwrap()
                .into_data()
                .unwrap(),
            "a"
        );
        assert_eq!(
            poll_frame(&mut body)
                .await
                .unwrap()
                .unwrap()
                .into_data()
                .unwrap(),
            "b"
        );
        assert_eq!(
            poll_frame(&mut body)
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap()["x-fixture"],
            "trailer"
        );
        assert!(body.is_end_stream());
        drop(body);
        released.notify_one();
        completed.notified().await;
        let logs = capture.text();
        assert_eq!(trace_ids(&logs).len(), 1, "{logs}");
        for marker in [
            "fixture handler",
            "fixture background after body",
            "fixture body poll",
            "fixture body drop",
            "body_complete",
        ] {
            assert!(
                logs.lines()
                    .filter(|line| line.contains(marker))
                    .all(|line| trace_ids(line).len() == 1),
                "{logs}"
            );
            assert!(logs.contains(marker), "{logs}");
        }
        assert!(logs.contains("/resource/:id"));
        assert!(!logs.contains("private-"), "{logs}");
        assert_eq!(logs.matches("HTTP response body finished").count(), 1);
        assert!(logs.contains("status=201"));
    }

    #[tokio::test]
    async fn unknown_routes_methods_and_client_trace_ids_cannot_pollute_or_merge_traces() {
        let capture = Capture::default();
        let _subscriber = tracing::subscriber::set_default(capture.subscriber());
        let app = Router::new()
            .fallback(|| async {
                tokio::task::yield_now().await;
                tracing::info!("fixture fallback");
                StatusCode::NOT_FOUND
            })
            .layer(from_fn(trace_request));
        let requests = (0..20).map(|i| {
            app.clone().oneshot(
                HttpRequest::builder()
                    .method("PRIVATE-METHOD")
                    .uri(format!("/private-{i}?secret=private-value"))
                    .header("traceparent", "same-private-id")
                    .body(Body::empty())
                    .unwrap(),
            )
        });
        for response in futures::future::join_all(requests).await {
            assert_eq!(response.unwrap().status(), StatusCode::NOT_FOUND);
        }
        let logs = capture.text();
        assert_eq!(trace_ids(&logs).len(), 20, "{logs}");
        assert!(logs.contains("method=\"OTHER\"") || logs.contains("method=OTHER"));
        assert!(logs.contains("route=\"unmatched\"") || logs.contains("route=unmatched"));
        assert!(!logs.contains("private"));
        assert!(!logs.contains("PRIVATE-METHOD"));
    }

    #[tokio::test]
    async fn logical_dispatch_shares_outer_trace_after_handler_returns() {
        let capture = Capture::default();
        let _subscriber = tracing::subscriber::set_default(capture.subscriber());
        let inner = Router::new()
            .route(
                "/logical/:id",
                get(|| async {
                    Body::new(FixtureBody(VecDeque::from([Ok(Frame::data(
                        Bytes::from_static(b"unchanged"),
                    ))])))
                }),
            )
            .layer(from_fn(trace_logical_request));
        let outer = Router::new()
            .route(
                "/v2/request",
                get(move || async move {
                    inner
                        .oneshot(
                            HttpRequest::builder()
                                .uri("/logical/private-id")
                                .body(Body::empty())
                                .unwrap(),
                        )
                        .await
                        .unwrap()
                }),
            )
            .layer(from_fn(trace_request));
        let response = outer
            .oneshot(
                HttpRequest::builder()
                    .uri("/v2/request")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 100)
                .await
                .unwrap(),
            "unchanged"
        );
        let logs = capture.text();
        assert_eq!(trace_ids(&logs).len(), 1, "{logs}");
        assert!(
            logs.lines().any(|line| line.contains("fixture body poll")
                && line.contains("logical_request")
                && line.contains("http_request")),
            "{logs}"
        );
        assert!(!logs.contains("private-id"));
    }

    #[tokio::test]
    async fn dropping_or_failing_a_body_is_not_reported_as_success_and_errors_stay_private() {
        for fail in [false, true] {
            let capture = Capture::default();
            let _subscriber = tracing::subscriber::set_default(capture.subscriber());
            let app = Router::new()
                .route(
                    "/stream",
                    get(move || async move {
                        Body::new(FixtureBody(VecDeque::from([Err(io::Error::other(
                            "private-provider-body",
                        ))])))
                    }),
                )
                .layer(from_fn(trace_request));
            let mut body = app
                .oneshot(
                    HttpRequest::builder()
                        .uri("/stream")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .into_body();
            if fail {
                assert!(poll_frame(&mut body).await.unwrap().is_err());
            }
            drop(body);
            let logs = capture.text();
            assert!(
                logs.contains(if fail { "body_error" } else { "body_dropped" }),
                "{logs}"
            );
            assert!(!logs.contains("body_complete"));
            assert!(!logs.contains("private-provider-body"));
            assert_eq!(logs.matches("HTTP response body finished").count(), 1);
            assert_eq!(trace_ids(&logs).len(), 1);
        }
    }

    #[tokio::test]
    async fn default_filter_keeps_each_concurrent_pending_body_and_task_in_its_own_trace() {
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(crate::DEFAULT_RUST_LOG_FILTER)
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .finish();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        struct BodyDrop(usize);
        impl Drop for BodyDrop {
            fn drop(&mut self) {
                tracing::info!(fixture_request = self.0, "concurrent body drop");
            }
        }
        let app = Router::new()
            .route(
                "/stream/:id",
                get(
                    |axum::extract::Path(id): axum::extract::Path<usize>| async move {
                        tracing::info!(fixture_request = id, "concurrent handler");
                        tokio::spawn(
                            async move {
                                tokio::task::yield_now().await;
                                tracing::info!(fixture_request = id, "concurrent background");
                            }
                            .in_current_span(),
                        )
                        .await
                        .unwrap();
                        let guard = BodyDrop(id);
                        Body::from_stream(async_stream::stream! {
                            let _guard = guard;
                            tracing::info!(fixture_request = id, "concurrent before pending");
                            tokio::task::yield_now().await;
                            tracing::info!(fixture_request = id, "concurrent after pending");
                            yield Ok::<_, io::Error>(Bytes::from_static(b"frame"));
                        })
                    },
                ),
            )
            .layer(from_fn(trace_request));
        futures::future::join_all((0..20).map(|id| {
            let app = app.clone();
            async move {
                let response = app
                    .oneshot(
                        HttpRequest::builder()
                            .uri(format!("/stream/{id}"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    axum::body::to_bytes(response.into_body(), 100)
                        .await
                        .unwrap(),
                    "frame"
                );
            }
        }))
        .await;
        let logs = capture.text();
        let id_pattern = regex::Regex::new(r"fixture_request=(\d+)\b").unwrap();
        let mut observed = std::collections::HashMap::<usize, (HashSet<String>, usize)>::new();
        for line in logs.lines() {
            if let Some(id) = id_pattern.captures(line) {
                let entry = observed.entry(id[1].parse().unwrap()).or_default();
                let ids = trace_ids(line);
                assert_eq!(ids.len(), 1, "{line}");
                entry.0.extend(ids);
                entry.1 += 1;
            }
        }
        assert_eq!(observed.len(), 20, "{logs}");
        for (ids, events) in observed.values() {
            assert_eq!(ids.len(), 1, "a request changed trace: {logs}");
            assert_eq!(
                *events, 5,
                "handler/task/pending/resume/drop each occurs once: {logs}"
            );
        }
        assert_eq!(
            trace_ids(&logs).len(),
            20,
            "requests shared a trace: {logs}"
        );
    }

    #[tokio::test]
    async fn cancellation_before_headers_drops_handler_inside_its_trace() {
        let capture = Capture::default();
        let _subscriber = tracing::subscriber::set_default(capture.subscriber());
        struct HandlerDrop;
        impl Drop for HandlerDrop {
            fn drop(&mut self) {
                tracing::info!("cancelled handler drop");
            }
        }
        let app = Router::new()
            .route(
                "/pending",
                get(|| async {
                    let _guard = HandlerDrop;
                    tracing::info!("pending handler entered");
                    std::future::pending::<StatusCode>().await
                }),
            )
            .layer(from_fn(trace_request));
        let mut future = Box::pin(
            app.oneshot(
                HttpRequest::builder()
                    .uri("/pending")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        std::future::poll_fn(|cx| {
            assert!(std::future::Future::poll(future.as_mut(), cx).is_pending());
            Poll::Ready(())
        })
        .await;
        drop(future);
        let logs = capture.text();
        assert_eq!(trace_ids(&logs).len(), 1, "{logs}");
        for line in logs.lines().filter(|line| line.contains("handler")) {
            assert_eq!(trace_ids(line).len(), 1, "{line}");
        }
        assert!(logs.contains("cancelled handler drop"));
        assert!(!logs.contains("body_complete"));
    }
}
