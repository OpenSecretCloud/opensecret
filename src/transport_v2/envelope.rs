use std::fmt;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::provider_cache::CacheNamespaceRoot;

const MIB: usize = 1024 * 1024;
const KIB: usize = 1024;

/// Resource ceilings for one decrypted transport-v2 envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EnvelopeLimits {
    pub(crate) envelope_bytes: usize,
    pub(crate) logical_body_bytes: usize,
    pub(crate) path_bytes: usize,
    pub(crate) query_bytes: usize,
    pub(crate) header_count: usize,
    pub(crate) header_name_bytes: usize,
    pub(crate) header_value_bytes: usize,
    pub(crate) aggregate_header_bytes: usize,
    pub(crate) credential_bytes: usize,
}

impl EnvelopeLimits {
    /// Request limits preserve the released proxy's 50 MiB logical-body
    /// contract. The larger envelope allowance accounts for the body's inner
    /// base64 representation plus the bounded request metadata.
    pub(crate) const REQUEST: Self = Self {
        envelope_bytes: 67 * MIB,
        logical_body_bytes: 50 * MIB,
        path_bytes: 4096,
        query_bytes: 8192,
        header_count: 64,
        header_name_bytes: 128,
        header_value_bytes: 16 * KIB,
        aggregate_header_bytes: 64 * KIB,
        credential_bytes: 16 * KIB,
    };

    /// Response limits remain deliberately smaller. In particular, widening
    /// request admission must not silently increase database/provider output
    /// retained inside the enclave or returned to a client.
    pub(crate) const RESPONSE: Self = Self {
        envelope_bytes: 50 * MIB,
        logical_body_bytes: 28 * MIB,
        path_bytes: 4096,
        query_bytes: 8192,
        header_count: 64,
        header_name_bytes: 128,
        header_value_bytes: 16 * KIB,
        aggregate_header_bytes: 64 * KIB,
        credential_bytes: 16 * KIB,
    };

    /// Compatibility alias for response-side/application-output call sites.
    /// Request parsing must opt into `REQUEST` explicitly.
    pub(crate) const DEFAULT: Self = Self::RESPONSE;
}

impl Default for EnvelopeLimits {
    fn default() -> Self {
        Self::RESPONSE
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EnvelopeError {
    #[error("transport-v2 envelope exceeds the configured {field} limit of {limit} bytes")]
    LimitExceeded { field: &'static str, limit: usize },
    #[error("invalid transport-v2 JSON")]
    InvalidJson(#[source] serde_json::Error),
    #[error("invalid transport-v2 path: {0}")]
    InvalidPath(&'static str),
    #[error("invalid transport-v2 query: {0}")]
    InvalidQuery(&'static str),
    #[error("invalid transport-v2 header name")]
    InvalidHeaderName,
    #[error("invalid transport-v2 header value")]
    InvalidHeaderValue,
    #[error("invalid transport-v2 response status")]
    InvalidStatus,
    #[error("invalid transport-v2 stream sequence")]
    InvalidStreamSequence,
}

/// The protocol version has no invalid in-memory representation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Version2;

impl Version2 {
    pub(crate) const VALUE: u8 = 2;
}

impl Serialize for Version2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(Self::VALUE)
    }
}

impl<'de> Deserialize<'de> for Version2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let version = u8::deserialize(deserializer)?;
        if version == Self::VALUE {
            Ok(Self)
        } else {
            Err(de::Error::custom("transport version must be exactly 2"))
        }
    }
}

/// A full 128-bit, per-session replay identifier.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RequestId([u8; 16]);

impl RequestId {
    pub(crate) const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub(crate) const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub(crate) const fn into_bytes(self) -> [u8; 16] {
        self.0
    }
}

impl fmt::Debug for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&hex::encode(self.0))
    }
}

impl Serialize for RequestId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for RequestId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RequestIdVisitor;

        impl de::Visitor<'_> for RequestIdVisitor {
            type Value = RequestId;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("exactly 32 lowercase hexadecimal characters")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                parse_request_id(value).ok_or_else(|| E::custom("non-canonical request ID"))
            }
        }

        deserializer.deserialize_str(RequestIdVisitor)
    }
}

fn parse_request_id(value: &str) -> Option<RequestId> {
    let encoded = value.as_bytes();
    if encoded.len() != 32
        || !encoded
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return None;
    }

    let mut decoded = [0_u8; 16];
    for (destination, pair) in decoded.iter_mut().zip(encoded.chunks_exact(2)) {
        *destination = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Some(RequestId(decoded))
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Exact bytes represented on the wire as padded standard base64.
#[derive(Clone, Eq, PartialEq, Zeroize, ZeroizeOnDrop)]
pub(crate) struct EncodedBytes(Vec<u8>);

impl EncodedBytes {
    pub(crate) fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for EncodedBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncodedBytes")
            .field("len", &self.0.len())
            .finish_non_exhaustive()
    }
}

impl Serialize for EncodedBytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for EncodedBytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EncodedBytesVisitor;

        impl de::Visitor<'_> for EncodedBytesVisitor {
            type Value = EncodedBytes;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("canonical padded standard base64")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let bytes = STANDARD
                    .decode(value)
                    .map_err(|_| E::custom("invalid standard base64"))?;
                if STANDARD.encode(&bytes) != value {
                    return Err(E::custom("non-canonical standard base64"));
                }
                Ok(EncodedBytes(bytes))
            }
        }

        deserializer.deserialize_str(EncodedBytesVisitor)
    }
}

/// Authentication material permitted only during an anonymous session transition.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Credential {
    ApiKey { value_base64: EncodedBytes },
    Resumption { value_base64: EncodedBytes },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseMode {
    Unary,
    Stream,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum LogicalMethod {
    #[serde(rename = "GET")]
    Get,
    #[serde(rename = "POST")]
    Post,
    #[serde(rename = "PUT")]
    Put,
    #[serde(rename = "PATCH")]
    Patch,
    #[serde(rename = "DELETE")]
    Delete,
}

impl LogicalMethod {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeaderField {
    pub(crate) name: String,
    pub(crate) value_base64: EncodedBytes,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LogicalRequest {
    pub(crate) method: LogicalMethod,
    pub(crate) path: String,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) query: Option<String>,
    pub(crate) headers: Vec<HeaderField>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) body_base64: Option<EncodedBytes>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RequestEnvelope {
    pub(crate) version: Version2,
    pub(crate) request_id: RequestId,
    pub(crate) response_mode: ResponseMode,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) credential: Option<Credential>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) cache_namespace_root_base64: Option<CacheNamespaceRoot>,
    pub(crate) request: LogicalRequest,
}

impl RequestEnvelope {
    pub(crate) fn from_json_slice(
        input: &[u8],
        limits: &EnvelopeLimits,
    ) -> Result<Self, EnvelopeError> {
        check_limit(input.len(), limits.envelope_bytes, "envelope")?;
        let envelope: Self = serde_json::from_slice(input).map_err(EnvelopeError::InvalidJson)?;
        envelope.validate(limits)?;
        Ok(envelope)
    }

    pub(crate) fn validate(&self, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
        self.request.validate(limits)?;
        if let Some(credential) = &self.credential {
            let credential_bytes = match credential {
                Credential::ApiKey { value_base64 } | Credential::Resumption { value_base64 } => {
                    value_base64.len()
                }
            };
            check_limit(credential_bytes, limits.credential_bytes, "credential")?;
        }
        Ok(())
    }
}

impl LogicalRequest {
    pub(crate) fn validate(&self, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
        validate_logical_path(self.method, &self.path, limits)?;
        if let Some(query) = self.query.as_deref() {
            validate_query(query, limits)?;
        }
        validate_headers(&self.headers, limits)?;
        if let Some(body) = &self.body_base64 {
            check_limit(body.len(), limits.logical_body_bytes, "logical body")?;
        }
        Ok(())
    }
}

const KV_ITEM_PATH_PREFIX: &str = "/protected/kv/";
const API_KEY_ITEM_PATH_PREFIX: &str = "/protected/api-keys/";
const VERIFY_EMAIL_PATH_PREFIX: &str = "/verify-email/";
const PLATFORM_VERIFY_EMAIL_PATH_PREFIX: &str = "/platform/verify-email/";
const PLATFORM_ORG_PATH_PREFIX: &str = "/platform/orgs/";
const PLATFORM_ACCEPT_INVITE_PATH_PREFIX: &str = "/platform/accept_invite/";
const CONVERSATION_PROJECT_ITEM_PATH_PREFIX: &str = "/v1/conversation-projects/";
const CONVERSATION_ITEM_PATH_PREFIX: &str = "/v1/conversations/";
const INSTRUCTION_ITEM_PATH_PREFIX: &str = "/v1/instructions/";
const RESPONSE_ITEM_PATH_PREFIX: &str = "/v1/responses/";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstructionItemPath {
    Item(Uuid),
    SetDefault(Uuid),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConversationItemPath {
    Conversation(Uuid),
    Items(Uuid),
    Item {
        conversation_id: Uuid,
        item_id: Uuid,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResponseItemPath {
    Item(Uuid),
    Cancel(Uuid),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PlatformResourcePath {
    Organization(Uuid),
    Projects(Uuid),
    Project {
        org_id: Uuid,
        project_id: Uuid,
    },
    Secrets {
        org_id: Uuid,
        project_id: Uuid,
    },
    Secret {
        org_id: Uuid,
        project_id: Uuid,
        key_name: Zeroizing<String>,
    },
    EmailSettings {
        org_id: Uuid,
        project_id: Uuid,
    },
    OAuthSettings {
        org_id: Uuid,
        project_id: Uuid,
    },
    Memberships(Uuid),
    Membership {
        org_id: Uuid,
        user_id: Uuid,
    },
    Invites(Uuid),
    Invite {
        org_id: Uuid,
        invite_code: Uuid,
    },
    AcceptInvite(Uuid),
}

fn validate_logical_path(
    method: LogicalMethod,
    path: &str,
    limits: &EnvelopeLimits,
) -> Result<(), EnvelopeError> {
    check_limit(path.len(), limits.path_bytes, "path")?;
    if decode_canonical_kv_item_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_api_key_name_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_verify_email_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_platform_verify_email_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_platform_resource_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_conversation_project_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_conversation_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_instruction_path(method, path)?.is_some() {
        return Ok(());
    }
    if decode_canonical_response_path(method, path)?.is_some() {
        return Ok(());
    }
    validate_path(path, limits)
}

/// Decodes the one opaque UTF-8 segment admitted by the released KV item API.
///
/// Literal separators select the route before this function runs. Every
/// non-alphanumeric key byte must use one uppercase `%HH` triplet, matching the
/// Rust SDK's `NON_ALPHANUMERIC` encoder. The result is decoded exactly once.
pub(crate) fn decode_canonical_kv_item_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<Zeroizing<String>>, EnvelopeError> {
    if !matches!(
        method,
        LogicalMethod::Get | LogicalMethod::Put | LogicalMethod::Delete
    ) {
        return Ok(None);
    }
    let Some(segment) = path.strip_prefix(KV_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    decode_canonical_opaque_segment(segment).map(Some)
}

/// Decodes the validated API-key name carried by the DELETE item route.
///
/// API-key names are ASCII alphanumeric characters, spaces, hyphens, and
/// underscores. V2 uses the same route-scoped opaque-segment spelling as KV:
/// alphanumeric bytes remain literal and every other byte is one uppercase
/// `%HH` triplet. Rejecting equivalent aliases gives the operation one path
/// spelling across clients before the name reaches the database lookup.
pub(crate) fn decode_canonical_api_key_name_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<Zeroizing<String>>, EnvelopeError> {
    if method != LogicalMethod::Delete {
        return Ok(None);
    }
    let Some(segment) = path.strip_prefix(API_KEY_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    let decoded = decode_canonical_opaque_segment(segment)?;
    if decoded.len() > 50
        || decoded.starts_with(' ')
        || decoded.ends_with(' ')
        || !decoded
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b' ' | b'-' | b'_'))
    {
        return Err(EnvelopeError::InvalidPath(
            "API-key name segment violates the name contract",
        ));
    }
    Ok(Some(decoded))
}

/// Decodes the one canonical UUID credential admitted by email verification.
///
/// Axum's v1 `Path<Uuid>` parser remains untouched. Transport v2 deliberately
/// accepts only the lowercase hyphenated UUID spelling so one verification
/// action has one authenticated logical path.
pub(crate) fn decode_canonical_verify_email_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<Uuid>, EnvelopeError> {
    if method != LogicalMethod::Get {
        return Ok(None);
    }
    let Some(segment) = path.strip_prefix(VERIFY_EMAIL_PATH_PREFIX) else {
        return Ok(None);
    };
    let code = Uuid::parse_str(segment).map_err(|_| {
        EnvelopeError::InvalidPath("email verification code must be a canonical UUID")
    })?;
    if code.hyphenated().to_string() != segment {
        return Err(EnvelopeError::InvalidPath(
            "email verification code must use lowercase hyphenated UUID spelling",
        ));
    }
    Ok(Some(code))
}

/// Decodes the canonical UUID credential admitted by platform email
/// verification without changing the more permissive transport-v1 parser.
pub(crate) fn decode_canonical_platform_verify_email_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<Uuid>, EnvelopeError> {
    if method != LogicalMethod::Get {
        return Ok(None);
    }
    let Some(segment) = path.strip_prefix(PLATFORM_VERIFY_EMAIL_PATH_PREFIX) else {
        return Ok(None);
    };
    let code = Uuid::parse_str(segment).map_err(|_| {
        EnvelopeError::InvalidPath("platform email verification code must be a canonical UUID")
    })?;
    if code.hyphenated().to_string() != segment {
        return Err(EnvelopeError::InvalidPath(
            "platform email verification code must use lowercase hyphenated UUID spelling",
        ));
    }
    Ok(Some(code))
}

/// Decodes the dynamic platform control-plane paths admitted by transport v2.
///
/// UUIDs have one lowercase-hyphenated spelling. Project-secret names use the
/// released `[A-Za-z0-9_]+` route contract directly; unlike general opaque
/// paths, an underscore is literal because existing TypeScript clients place
/// it in the URL without percent encoding.
pub(crate) fn decode_canonical_platform_resource_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<PlatformResourcePath>, EnvelopeError> {
    if let Some(code) = path.strip_prefix(PLATFORM_ACCEPT_INVITE_PATH_PREFIX) {
        if method != LogicalMethod::Post {
            return Ok(None);
        }
        return decode_canonical_uuid_segment(
            code,
            "platform invite code must use lowercase hyphenated UUID spelling",
        )
        .map(PlatformResourcePath::AcceptInvite)
        .map(Some);
    }

    let Some(suffix) = path.strip_prefix(PLATFORM_ORG_PATH_PREFIX) else {
        return Ok(None);
    };
    let mut segments = suffix.split('/');
    let org = segments.next().unwrap_or_default();
    let org_id = decode_canonical_uuid_segment(
        org,
        "platform organization ID must use lowercase hyphenated UUID spelling",
    )?;
    let remainder = segments.collect::<Vec<_>>();

    let decoded = match remainder.as_slice() {
        [] if method == LogicalMethod::Delete => PlatformResourcePath::Organization(org_id),
        ["projects"] if matches!(method, LogicalMethod::Get | LogicalMethod::Post) => {
            PlatformResourcePath::Projects(org_id)
        }
        ["projects", project]
            if matches!(
                method,
                LogicalMethod::Get | LogicalMethod::Patch | LogicalMethod::Delete
            ) =>
        {
            PlatformResourcePath::Project {
                org_id,
                project_id: decode_canonical_uuid_segment(
                    project,
                    "platform project ID must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        ["projects", project, "secrets"]
            if matches!(method, LogicalMethod::Get | LogicalMethod::Post) =>
        {
            PlatformResourcePath::Secrets {
                org_id,
                project_id: decode_canonical_uuid_segment(
                    project,
                    "platform project ID must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        ["projects", project, "secrets", key_name] if method == LogicalMethod::Delete => {
            if key_name.is_empty()
                || key_name.len() > 50
                || !key_name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            {
                return Err(EnvelopeError::InvalidPath(
                    "platform secret name violates the route contract",
                ));
            }
            PlatformResourcePath::Secret {
                org_id,
                project_id: decode_canonical_uuid_segment(
                    project,
                    "platform project ID must use lowercase hyphenated UUID spelling",
                )?,
                key_name: Zeroizing::new((*key_name).to_owned()),
            }
        }
        ["projects", project, "settings", "email"]
            if matches!(method, LogicalMethod::Get | LogicalMethod::Put) =>
        {
            PlatformResourcePath::EmailSettings {
                org_id,
                project_id: decode_canonical_uuid_segment(
                    project,
                    "platform project ID must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        ["projects", project, "settings", "oauth"]
            if matches!(method, LogicalMethod::Get | LogicalMethod::Put) =>
        {
            PlatformResourcePath::OAuthSettings {
                org_id,
                project_id: decode_canonical_uuid_segment(
                    project,
                    "platform project ID must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        ["memberships"] if method == LogicalMethod::Get => {
            PlatformResourcePath::Memberships(org_id)
        }
        ["memberships", user] if matches!(method, LogicalMethod::Patch | LogicalMethod::Delete) => {
            PlatformResourcePath::Membership {
                org_id,
                user_id: decode_canonical_uuid_segment(
                    user,
                    "platform membership user ID must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        ["invites"] if matches!(method, LogicalMethod::Get | LogicalMethod::Post) => {
            PlatformResourcePath::Invites(org_id)
        }
        ["invites", invite] if matches!(method, LogicalMethod::Get | LogicalMethod::Delete) => {
            PlatformResourcePath::Invite {
                org_id,
                invite_code: decode_canonical_uuid_segment(
                    invite,
                    "platform invite code must use lowercase hyphenated UUID spelling",
                )?,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(decoded))
}

/// Decodes the canonical UUID segment used by one conversation-project item.
///
/// Axum's v1 UUID aliases remain untouched. Transport v2 accepts one lowercase
/// hyphenated spelling for GET, POST, and DELETE so the authenticated logical
/// path identifies exactly one project operation.
pub(crate) fn decode_canonical_conversation_project_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<Uuid>, EnvelopeError> {
    if !matches!(
        method,
        LogicalMethod::Get | LogicalMethod::Post | LogicalMethod::Delete
    ) {
        return Ok(None);
    }
    let Some(segment) = path.strip_prefix(CONVERSATION_PROJECT_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    let project_id = Uuid::parse_str(segment).map_err(|_| {
        EnvelopeError::InvalidPath("conversation project ID must be a canonical UUID")
    })?;
    if project_id.hyphenated().to_string() != segment {
        return Err(EnvelopeError::InvalidPath(
            "conversation project ID must use lowercase hyphenated UUID spelling",
        ));
    }
    Ok(Some(project_id))
}

/// Decodes canonical general-instruction item and set-default paths.
///
/// The fixed `/set-default` suffix is recognized before the ordinary item
/// form, preventing the action route from being confused with an alternate
/// UUID spelling. Project-linked instructions remain outside this route
/// family and are filtered by the owner-scoped storage projection.
pub(crate) fn decode_canonical_instruction_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<InstructionItemPath>, EnvelopeError> {
    let Some(segment) = path.strip_prefix(INSTRUCTION_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    let (segment, set_default) = if let Some(segment) = segment.strip_suffix("/set-default") {
        if method != LogicalMethod::Post {
            return Ok(None);
        }
        (segment, true)
    } else {
        if !matches!(
            method,
            LogicalMethod::Get | LogicalMethod::Post | LogicalMethod::Delete
        ) {
            return Ok(None);
        }
        (segment, false)
    };
    let instruction_id = Uuid::parse_str(segment)
        .map_err(|_| EnvelopeError::InvalidPath("instruction ID must be a canonical UUID"))?;
    if instruction_id.hyphenated().to_string() != segment {
        return Err(EnvelopeError::InvalidPath(
            "instruction ID must use lowercase hyphenated UUID spelling",
        ));
    }
    Ok(Some(if set_default {
        InstructionItemPath::SetDefault(instruction_id)
    } else {
        InstructionItemPath::Item(instruction_id)
    }))
}

/// Decodes canonical conversation, item-collection, and individual-item paths.
///
/// Fixed collection actions are excluded before UUID parsing. Item suffixes are
/// recognized before the ordinary conversation form so an admitted operation
/// has one unambiguous lowercase-hyphenated spelling.
pub(crate) fn decode_canonical_conversation_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<ConversationItemPath>, EnvelopeError> {
    let Some(suffix) = path.strip_prefix(CONVERSATION_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    if matches!(suffix, "batch-delete" | "batch-update-project") {
        return Ok(None);
    }

    if method == LogicalMethod::Get {
        if let Some((conversation, item)) = suffix.split_once("/items/") {
            if item.contains('/') {
                return Err(EnvelopeError::InvalidPath(
                    "conversation item path has extra segments",
                ));
            }
            return Ok(Some(ConversationItemPath::Item {
                conversation_id: decode_canonical_uuid_segment(
                    conversation,
                    "conversation ID must use lowercase hyphenated UUID spelling",
                )?,
                item_id: decode_canonical_uuid_segment(
                    item,
                    "conversation item ID must use lowercase hyphenated UUID spelling",
                )?,
            }));
        }
        if let Some(conversation) = suffix.strip_suffix("/items") {
            return Ok(Some(ConversationItemPath::Items(
                decode_canonical_uuid_segment(
                    conversation,
                    "conversation ID must use lowercase hyphenated UUID spelling",
                )?,
            )));
        }
    }

    if !matches!(
        method,
        LogicalMethod::Get | LogicalMethod::Post | LogicalMethod::Delete
    ) {
        return Ok(None);
    }
    Ok(Some(ConversationItemPath::Conversation(
        decode_canonical_uuid_segment(
            suffix,
            "conversation ID must use lowercase hyphenated UUID spelling",
        )?,
    )))
}

/// Decodes canonical stored-response item and cancellation paths.
pub(crate) fn decode_canonical_response_path(
    method: LogicalMethod,
    path: &str,
) -> Result<Option<ResponseItemPath>, EnvelopeError> {
    let Some(suffix) = path.strip_prefix(RESPONSE_ITEM_PATH_PREFIX) else {
        return Ok(None);
    };
    if let Some(response) = suffix.strip_suffix("/cancel") {
        if method != LogicalMethod::Post {
            return Ok(None);
        }
        return Ok(Some(ResponseItemPath::Cancel(
            decode_canonical_uuid_segment(
                response,
                "response ID must use lowercase hyphenated UUID spelling",
            )?,
        )));
    }
    if !matches!(method, LogicalMethod::Get | LogicalMethod::Delete) {
        return Ok(None);
    }
    Ok(Some(ResponseItemPath::Item(decode_canonical_uuid_segment(
        suffix,
        "response ID must use lowercase hyphenated UUID spelling",
    )?)))
}

fn decode_canonical_uuid_segment(
    segment: &str,
    error: &'static str,
) -> Result<Uuid, EnvelopeError> {
    let id = Uuid::parse_str(segment).map_err(|_| EnvelopeError::InvalidPath(error))?;
    if id.hyphenated().to_string() != segment {
        return Err(EnvelopeError::InvalidPath(error));
    }
    Ok(id)
}

fn decode_canonical_opaque_segment(segment: &str) -> Result<Zeroizing<String>, EnvelopeError> {
    if segment.is_empty() {
        return Err(EnvelopeError::InvalidPath(
            "opaque path segment must not be empty",
        ));
    }

    let encoded = segment.as_bytes();
    let mut decoded = Zeroizing::new(Vec::with_capacity(encoded.len()));
    let mut index = 0;
    while index < encoded.len() {
        let byte = encoded[index];
        if byte.is_ascii_alphanumeric() {
            decoded.push(byte);
            index += 1;
            continue;
        }
        if byte != b'%' {
            return Err(EnvelopeError::InvalidPath(
                "opaque path segment is not canonically encoded",
            ));
        }
        let high = *encoded
            .get(index + 1)
            .ok_or(EnvelopeError::InvalidPath("invalid percent encoding"))?;
        let low = *encoded
            .get(index + 2)
            .ok_or(EnvelopeError::InvalidPath("invalid percent encoding"))?;
        let decoded_byte = (canonical_uri_hex_nibble(high)? << 4) | canonical_uri_hex_nibble(low)?;
        if decoded_byte.is_ascii_alphanumeric() {
            return Err(EnvelopeError::InvalidPath(
                "opaque path segment over-encodes an alphanumeric byte",
            ));
        }
        decoded.push(decoded_byte);
        index += 3;
    }

    match String::from_utf8(std::mem::take(&mut *decoded)) {
        Ok(value) => Ok(Zeroizing::new(value)),
        Err(error) => {
            let mut bytes = error.into_bytes();
            bytes.zeroize();
            Err(EnvelopeError::InvalidPath(
                "opaque path segment is not valid UTF-8",
            ))
        }
    }
}

fn canonical_uri_hex_nibble(byte: u8) -> Result<u8, EnvelopeError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(EnvelopeError::InvalidPath(
            "opaque path segment requires uppercase percent encoding",
        )),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnaryResponseEnvelope {
    pub(crate) version: Version2,
    pub(crate) request_id: RequestId,
    pub(crate) status: u16,
    pub(crate) headers: Vec<HeaderField>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) body_base64: Option<EncodedBytes>,
}

impl UnaryResponseEnvelope {
    pub(crate) fn from_json_slice(
        input: &[u8],
        limits: &EnvelopeLimits,
    ) -> Result<Self, EnvelopeError> {
        check_limit(input.len(), limits.envelope_bytes, "envelope")?;
        let envelope: Self = serde_json::from_slice(input).map_err(EnvelopeError::InvalidJson)?;
        envelope.validate(limits)?;
        Ok(envelope)
    }

    pub(crate) fn validate(&self, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
        validate_status(self.status)?;
        validate_headers(&self.headers, limits)?;
        if let Some(body) = &self.body_base64 {
            check_limit(body.len(), limits.logical_body_bytes, "logical body")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum StreamRecord {
    Start {
        version: Version2,
        request_id: RequestId,
        sequence: u64,
        status: u16,
        headers: Vec<HeaderField>,
    },
    Chunk {
        version: Version2,
        request_id: RequestId,
        sequence: u64,
        body_base64: EncodedBytes,
    },
    End {
        version: Version2,
        request_id: RequestId,
        sequence: u64,
    },
    Error {
        version: Version2,
        request_id: RequestId,
        sequence: u64,
        status: u16,
        body_base64: EncodedBytes,
    },
}

pub(crate) const MAX_STREAM_CHUNK_BYTES: usize = 64 * 1024;
pub(crate) const MAX_STREAM_ERROR_BYTES: usize = 16 * 1024;

impl StreamRecord {
    pub(crate) fn from_json_slice(
        input: &[u8],
        limits: &EnvelopeLimits,
    ) -> Result<Self, EnvelopeError> {
        check_limit(input.len(), limits.envelope_bytes, "envelope")?;
        let record: Self = serde_json::from_slice(input).map_err(EnvelopeError::InvalidJson)?;
        record.validate(limits)?;
        Ok(record)
    }

    pub(crate) fn validate(&self, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
        match self {
            Self::Start {
                sequence,
                status,
                headers,
                ..
            } => {
                if *sequence != 0 {
                    return Err(EnvelopeError::InvalidStreamSequence);
                }
                if !(200..=299).contains(status) {
                    return Err(EnvelopeError::InvalidStatus);
                }
                validate_headers(headers, limits)
            }
            Self::Chunk {
                sequence,
                body_base64,
                ..
            } => {
                validate_non_initial_sequence(*sequence)?;
                check_limit(body_base64.len(), MAX_STREAM_CHUNK_BYTES, "stream chunk")
            }
            Self::End { sequence, .. } => validate_non_initial_sequence(*sequence),
            Self::Error {
                sequence,
                status,
                body_base64,
                ..
            } => {
                validate_non_initial_sequence(*sequence)?;
                if !(400..=599).contains(status) {
                    return Err(EnvelopeError::InvalidStatus);
                }
                check_limit(body_base64.len(), MAX_STREAM_ERROR_BYTES, "stream error")
            }
        }
    }
}

fn deserialize_required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn check_limit(actual: usize, limit: usize, field: &'static str) -> Result<(), EnvelopeError> {
    if actual > limit {
        Err(EnvelopeError::LimitExceeded { field, limit })
    } else {
        Ok(())
    }
}

fn validate_status(status: u16) -> Result<(), EnvelopeError> {
    if (100..=599).contains(&status) {
        Ok(())
    } else {
        Err(EnvelopeError::InvalidStatus)
    }
}

fn validate_non_initial_sequence(sequence: u64) -> Result<(), EnvelopeError> {
    if sequence == 0 {
        Err(EnvelopeError::InvalidStreamSequence)
    } else {
        Ok(())
    }
}

fn validate_path(path: &str, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
    check_limit(path.len(), limits.path_bytes, "path")?;
    if !path.starts_with('/') {
        return Err(EnvelopeError::InvalidPath("path must start with '/'"));
    }
    if path.starts_with("//") {
        return Err(EnvelopeError::InvalidPath(
            "path must not contain an authority",
        ));
    }
    if path.contains('?') || path.contains('#') {
        return Err(EnvelopeError::InvalidPath(
            "path must not contain a query or fragment",
        ));
    }
    if path.contains('\\') {
        return Err(EnvelopeError::InvalidPath(
            "path must not contain a backslash",
        ));
    }

    let bytes = path.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                let decoded = decode_percent_triplet(bytes, index)
                    .ok_or(EnvelopeError::InvalidPath("invalid percent encoding"))?;
                if matches!(decoded, b'/' | b'\\') {
                    return Err(EnvelopeError::InvalidPath(
                        "path must not contain an encoded separator",
                    ));
                }
                index += 3;
            }
            byte if is_path_character(byte) => index += 1,
            _ => return Err(EnvelopeError::InvalidPath("invalid URI path character")),
        }
    }

    for segment in path.split('/') {
        if is_dot_segment(segment)? {
            return Err(EnvelopeError::InvalidPath(
                "path must not contain dot-segments",
            ));
        }
    }

    Ok(())
}

fn validate_query(query: &str, limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
    check_limit(query.len(), limits.query_bytes, "query")?;
    if query.starts_with('?') || query.starts_with('#') {
        return Err(EnvelopeError::InvalidQuery(
            "query must not include a leading delimiter",
        ));
    }
    if query.contains('#') {
        return Err(EnvelopeError::InvalidQuery(
            "query must not contain a fragment",
        ));
    }

    let bytes = query.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                decode_percent_triplet(bytes, index)
                    .ok_or(EnvelopeError::InvalidQuery("invalid percent encoding"))?;
                index += 3;
            }
            byte if is_query_character(byte) => index += 1,
            _ => return Err(EnvelopeError::InvalidQuery("invalid URI query character")),
        }
    }
    Ok(())
}

fn decode_percent_triplet(bytes: &[u8], percent_index: usize) -> Option<u8> {
    let high = *bytes.get(percent_index + 1)?;
    let low = *bytes.get(percent_index + 2)?;
    Some((uri_hex_nibble(high)? << 4) | uri_hex_nibble(low)?)
}

fn uri_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_dot_segment(segment: &str) -> Result<bool, EnvelopeError> {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let byte = decode_percent_triplet(bytes, index)
                .ok_or(EnvelopeError::InvalidPath("invalid percent encoding"))?;
            decoded.push(byte);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    Ok(matches!(decoded.as_slice(), b"." | b".."))
}

fn is_path_character(byte: u8) -> bool {
    byte == b'/' || is_uri_pchar(byte)
}

fn is_query_character(byte: u8) -> bool {
    matches!(byte, b'/' | b'?') || is_uri_pchar(byte)
}

fn is_uri_pchar(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b':'
                | b'@'
        )
}

fn validate_headers(headers: &[HeaderField], limits: &EnvelopeLimits) -> Result<(), EnvelopeError> {
    if headers.len() > limits.header_count {
        return Err(EnvelopeError::LimitExceeded {
            field: "header count",
            limit: limits.header_count,
        });
    }

    let mut aggregate_bytes = 0_usize;
    for header in headers {
        check_limit(header.name.len(), limits.header_name_bytes, "header name")?;
        if header.name.is_empty() || !header.name.bytes().all(is_lowercase_http_token) {
            return Err(EnvelopeError::InvalidHeaderName);
        }

        check_limit(
            header.value_base64.len(),
            limits.header_value_bytes,
            "header value",
        )?;
        if header
            .value_base64
            .as_slice()
            .iter()
            .any(|byte| matches!(byte, b'\r' | b'\n' | 0))
        {
            return Err(EnvelopeError::InvalidHeaderValue);
        }

        aggregate_bytes = aggregate_bytes
            .checked_add(header.name.len())
            .and_then(|total| total.checked_add(header.value_base64.len()))
            .ok_or(EnvelopeError::LimitExceeded {
                field: "aggregate headers",
                limit: limits.aggregate_header_bytes,
            })?;
        check_limit(
            aggregate_bytes,
            limits.aggregate_header_bytes,
            "aggregate headers",
        )?;
    }
    Ok(())
}

fn is_lowercase_http_token(byte: u8) -> bool {
    byte.is_ascii_lowercase()
        || byte.is_ascii_digit()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST_ID: &str = "00112233445566778899aabbccddeeff";

    #[test]
    fn decoded_byte_fields_zeroize_on_drop() {
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}

        assert_zeroize_on_drop::<EncodedBytes>();
        assert_zeroize_on_drop::<CacheNamespaceRoot>();
    }

    fn request_json(body: &str) -> String {
        format!(
            r#"{{
                "version":2,
                "request_id":"{REQUEST_ID}",
                "response_mode":"auto",
                "credential":null,
                "cache_namespace_root_base64":null,
                "request":{{
                    "method":"POST",
                    "path":"/v1/responses",
                    "query":null,
                    "headers":[{{"name":"content-type","value_base64":"YXBwbGljYXRpb24vanNvbg=="}}],
                    "body_base64":{body}
                }}
            }}"#
        )
    }

    #[test]
    fn request_id_requires_exact_lowercase_hex() {
        let request_id: RequestId = serde_json::from_str(&format!(r#""{REQUEST_ID}""#)).unwrap();
        assert_eq!(request_id.as_bytes(), &hex::decode(REQUEST_ID).unwrap()[..]);
        assert_eq!(
            serde_json::to_string(&request_id).unwrap(),
            format!(r#""{REQUEST_ID}""#)
        );

        for invalid in [
            "00112233445566778899AABBCCDDEEFF",
            "00112233445566778899aabbccddeef",
            "00112233445566778899aabbccddeeff00",
            "00112233445566778899aabbccddeefg",
        ] {
            assert!(serde_json::from_str::<RequestId>(&format!(r#""{invalid}""#)).is_err());
        }
    }

    #[test]
    fn encoded_bytes_require_canonical_padded_standard_base64() {
        let bytes: EncodedBytes = serde_json::from_str(r#""YQ==""#).unwrap();
        assert_eq!(bytes.as_slice(), b"a");
        assert_eq!(serde_json::to_string(&bytes).unwrap(), r#""YQ==""#);
        assert!(serde_json::from_str::<EncodedBytes>(r#""""#)
            .unwrap()
            .is_empty());

        for invalid in [r#""YQ""#, r#""YR==""#, r#""YQ==\n""#, r#""_w==""#] {
            assert!(serde_json::from_str::<EncodedBytes>(invalid).is_err());
        }
    }

    #[test]
    fn required_nullable_body_distinguishes_absent_null_and_empty() {
        let no_body = RequestEnvelope::from_json_slice(
            request_json("null").as_bytes(),
            &EnvelopeLimits::default(),
        )
        .unwrap();
        assert!(no_body.request.body_base64.is_none());

        let empty_body = RequestEnvelope::from_json_slice(
            request_json(r#""""#).as_bytes(),
            &EnvelopeLimits::default(),
        )
        .unwrap();
        assert!(empty_body.request.body_base64.as_ref().unwrap().is_empty());

        let missing =
            request_json("null").replace(",\n                    \"body_base64\":null", "");
        assert!(
            RequestEnvelope::from_json_slice(missing.as_bytes(), &EnvelopeLimits::default())
                .is_err()
        );

        let missing_query = request_json("null").replace("\"query\":null,", "");
        assert!(RequestEnvelope::from_json_slice(
            missing_query.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());

        let missing_credential = request_json("null").replace("\"credential\":null,", "");
        assert!(RequestEnvelope::from_json_slice(
            missing_credential.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());

        let missing_cache_namespace_root =
            request_json("null").replace("\"cache_namespace_root_base64\":null,", "");
        assert!(RequestEnvelope::from_json_slice(
            missing_cache_namespace_root.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());
    }

    #[test]
    fn cache_namespace_root_requires_canonical_padded_base64_for_exactly_32_bytes() {
        let encoded_root = STANDARD.encode([0x5a; 32]);
        let with_root = request_json("null").replace(
            "\"cache_namespace_root_base64\":null",
            &format!("\"cache_namespace_root_base64\":\"{encoded_root}\""),
        );
        let parsed =
            RequestEnvelope::from_json_slice(with_root.as_bytes(), &EnvelopeLimits::default())
                .expect("canonical 32-byte cache namespace root");
        assert_eq!(
            serde_json::to_value(parsed.cache_namespace_root_base64.as_ref().unwrap()).unwrap(),
            serde_json::Value::String(encoded_root.clone())
        );

        for invalid_root in [
            STANDARD.encode([0x5a; 31]),
            STANDARD.encode([0x5a; 33]),
            encoded_root.trim_end_matches('=').to_string(),
            format!("{}A", encoded_root.trim_end_matches('=')),
            format!("{}p=", &encoded_root[..42]),
            "___________________________________________=".to_string(),
        ] {
            let invalid = request_json("null").replace(
                "\"cache_namespace_root_base64\":null",
                &format!("\"cache_namespace_root_base64\":\"{invalid_root}\""),
            );
            assert!(
                RequestEnvelope::from_json_slice(invalid.as_bytes(), &EnvelopeLimits::default(),)
                    .is_err(),
                "accepted invalid cache namespace root: {invalid_root}"
            );
        }
    }

    #[test]
    fn typed_json_rejects_duplicate_and_unknown_fields_at_every_level() {
        let duplicate =
            request_json("null").replace("\"version\":2,", "\"version\":2,\"version\":2,");
        assert!(
            RequestEnvelope::from_json_slice(duplicate.as_bytes(), &EnvelopeLimits::default())
                .is_err()
        );

        let duplicate_nested = request_json("null").replace(
            "\"method\":\"POST\",",
            "\"method\":\"POST\",\"method\":\"POST\",",
        );
        assert!(RequestEnvelope::from_json_slice(
            duplicate_nested.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());

        let unknown = request_json("null").replace(
            "\"response_mode\":\"auto\",",
            "\"response_mode\":\"auto\",\"extra\":true,",
        );
        assert!(
            RequestEnvelope::from_json_slice(unknown.as_bytes(), &EnvelopeLimits::default())
                .is_err()
        );

        let unknown_header = request_json("null").replace(
            "\"name\":\"content-type\",",
            "\"name\":\"content-type\",\"extra\":true,",
        );
        assert!(RequestEnvelope::from_json_slice(
            unknown_header.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());

        let credential_unknown = request_json("null").replace(
            "\"credential\":null",
            "\"credential\":{\"kind\":\"api_key\",\"value_base64\":\"YQ==\",\"extra\":true}",
        );
        assert!(RequestEnvelope::from_json_slice(
            credential_unknown.as_bytes(),
            &EnvelopeLimits::default()
        )
        .is_err());
    }

    #[test]
    fn methods_paths_and_queries_are_structurally_strict() {
        for invalid_method in ["get", "HEAD", "OPTIONS"] {
            let json = request_json("null").replace(
                "\"method\":\"POST\"",
                &format!("\"method\":\"{invalid_method}\""),
            );
            assert!(
                RequestEnvelope::from_json_slice(json.as_bytes(), &EnvelopeLimits::default())
                    .is_err()
            );
        }

        for invalid_path in [
            "https://example.test/x",
            "//example.test/x",
            "/a?b",
            "/a#b",
            "/a\\b",
            "/a/../b",
            "/a/%2e%2E/b",
            "/a%2fb",
            "/a%5Cb",
            "/a%zz",
        ] {
            assert!(
                validate_path(invalid_path, &EnvelopeLimits::default()).is_err(),
                "{invalid_path}"
            );
        }
        validate_path("/v1/a%20b/~ok", &EnvelopeLimits::default()).unwrap();

        for invalid_query in ["?a=b", "#fragment", "a=b#fragment", "a=%zz"] {
            assert!(validate_query(invalid_query, &EnvelopeLimits::default()).is_err());
        }
        validate_query(
            "a=b&next=%2Fv1%2Fmodels?raw=true",
            &EnvelopeLimits::default(),
        )
        .unwrap();
    }

    #[test]
    fn kv_item_paths_use_one_route_scoped_canonical_utf8_segment() {
        for (encoded, decoded) in [
            ("simple123", "simple123"),
            ("key%2Fwith%2Fslashes", "key/with/slashes"),
            ("key%3Fwith%3Dquery%26params", "key?with=query&params"),
            ("key%23with%23hash", "key#with#hash"),
            ("key%20with%20spaces", "key with spaces"),
            ("key%25with%25percents", "key%with%percents"),
            ("key%40with%21special%24chars", "key@with!special$chars"),
            ("%2D%5F%2E%21%7E%2A%27%28%29", "-_.!~*'()"),
            ("%2E", "."),
            ("%2E%2E", ".."),
            ("%2F", "/"),
            ("%5C", "\\"),
            ("%252F", "%2F"),
            ("caf%C3%A9", "café"),
            ("%F0%9F%94%90", "🔐"),
        ] {
            for method in [
                LogicalMethod::Get,
                LogicalMethod::Put,
                LogicalMethod::Delete,
            ] {
                let path = format!("{KV_ITEM_PATH_PREFIX}{encoded}");
                let value = decode_canonical_kv_item_path(method, &path)
                    .unwrap()
                    .expect("KV item route must decode");
                assert_eq!(&*value, decoded, "{method:?} {encoded}");
                validate_logical_path(method, &path, &EnvelopeLimits::default()).unwrap();
            }
        }

        let json = request_json("null")
            .replace("\"method\":\"POST\"", "\"method\":\"GET\"")
            .replace("/v1/responses", "/protected/kv/key%2Fpart");
        let envelope =
            RequestEnvelope::from_json_slice(json.as_bytes(), &EnvelopeLimits::default()).unwrap();
        assert_eq!(envelope.request.path, "/protected/kv/key%2Fpart");
    }

    #[test]
    fn kv_item_paths_reject_noncanonical_or_ambiguous_segments() {
        for invalid in [
            "/protected/kv/",
            "/protected/kv/a/b",
            "/protected/kv/%2f",
            "/protected/kv/%5c",
            "/protected/kv/%2e",
            "/protected/kv/%41",
            "/protected/kv/%",
            "/protected/kv/%0",
            "/protected/kv/%GG",
            "/protected/kv/raw_punctuation",
            "/protected/kv/café",
            "/protected/kv/%FF",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Get, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }

        assert!(
            decode_canonical_kv_item_path(LogicalMethod::Post, "/protected/kv/key%2Fpart")
                .unwrap()
                .is_none()
        );
        assert!(validate_logical_path(
            LogicalMethod::Post,
            "/protected/kv/key%2Fpart",
            &EnvelopeLimits::default()
        )
        .is_err());
        assert!(
            decode_canonical_kv_item_path(LogicalMethod::Get, "/protected/kvx/key")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn api_key_delete_paths_use_one_restricted_canonical_name_segment() {
        for (encoded, decoded) in [
            ("Production%20Key", "Production Key"),
            ("agent%2Dproxy%5F42", "agent-proxy_42"),
            ("a%20%20b", "a  b"),
            ("%2D", "-"),
            ("%5F", "_"),
            (
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
        ] {
            let path = format!("{API_KEY_ITEM_PATH_PREFIX}{encoded}");
            let value = decode_canonical_api_key_name_path(LogicalMethod::Delete, &path)
                .unwrap()
                .expect("API-key item route must decode");
            assert_eq!(&*value, decoded, "{encoded}");
            validate_logical_path(LogicalMethod::Delete, &path, &EnvelopeLimits::default())
                .unwrap();
        }

        assert!(decode_canonical_api_key_name_path(
            LogicalMethod::Get,
            "/protected/api-keys/Production%20Key",
        )
        .unwrap()
        .is_none());
        assert!(decode_canonical_api_key_name_path(
            LogicalMethod::Delete,
            "/protected/api-keysx/Production%20Key",
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn api_key_delete_paths_reject_aliases_and_invalid_names() {
        let overlong = format!("{API_KEY_ITEM_PATH_PREFIX}{}", "a".repeat(51));
        for invalid in [
            "/protected/api-keys/",
            "/protected/api-keys/%20leading",
            "/protected/api-keys/trailing%20",
            "/protected/api-keys/literal-hyphen",
            "/protected/api-keys/literal_underscore",
            "/protected/api-keys/%2d",
            "/protected/api-keys/%5f",
            "/protected/api-keys/%41",
            "/protected/api-keys/plus+alias",
            "/protected/api-keys/literal space",
            "/protected/api-keys/%252D",
            "/protected/api-keys/%2F",
            "/protected/api-keys/%5C",
            "/protected/api-keys/%2E",
            "/protected/api-keys/%FF",
            "/protected/api-keys/%",
            "/protected/api-keys/%2",
            "/protected/api-keys/%GG",
            "/protected/api-keys/name/part",
            "/protected/api-keys/caf%C3%A9",
            overlong.as_str(),
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Delete, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn email_verification_path_uses_one_canonical_uuid_spelling() {
        let encoded = "123e4567-e89b-12d3-a456-426614174000";
        let path = format!("{VERIFY_EMAIL_PATH_PREFIX}{encoded}");
        assert_eq!(
            decode_canonical_verify_email_path(LogicalMethod::Get, &path).unwrap(),
            Some(Uuid::parse_str(encoded).unwrap())
        );
        validate_logical_path(LogicalMethod::Get, &path, &EnvelopeLimits::default()).unwrap();

        assert!(
            decode_canonical_verify_email_path(LogicalMethod::Post, &path)
                .unwrap()
                .is_none()
        );
        assert!(decode_canonical_verify_email_path(
            LogicalMethod::Get,
            "/verify-emails/123e4567-e89b-12d3-a456-426614174000",
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn email_verification_path_rejects_uuid_aliases_and_malformed_codes() {
        for invalid in [
            "/verify-email/",
            "/verify-email/123E4567-E89B-12D3-A456-426614174000",
            "/verify-email/123e4567e89b12d3a456426614174000",
            "/verify-email/{123e4567-e89b-12d3-a456-426614174000}",
            "/verify-email/urn:uuid:123e4567-e89b-12d3-a456-426614174000",
            "/verify-email/123e4567%2De89b%2D12d3%2Da456%2D426614174000",
            "/verify-email/123e4567-e89b-12d3-a456-426614174000/extra",
            "/verify-email/not-a-uuid",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Get, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn platform_email_verification_path_uses_one_canonical_uuid_spelling() {
        let encoded = "123e4567-e89b-12d3-a456-426614174000";
        let path = format!("{PLATFORM_VERIFY_EMAIL_PATH_PREFIX}{encoded}");
        assert_eq!(
            decode_canonical_platform_verify_email_path(LogicalMethod::Get, &path).unwrap(),
            Some(Uuid::parse_str(encoded).unwrap())
        );
        validate_logical_path(LogicalMethod::Get, &path, &EnvelopeLimits::default()).unwrap();

        assert!(
            decode_canonical_platform_verify_email_path(LogicalMethod::Post, &path)
                .unwrap()
                .is_none()
        );
        for invalid in [
            "/platform/verify-email/",
            "/platform/verify-email/123E4567-E89B-12D3-A456-426614174000",
            "/platform/verify-email/123e4567e89b12d3a456426614174000",
            "/platform/verify-email/123e4567-e89b-12d3-a456-426614174000/extra",
            "/platform/verify-email/not-a-uuid",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Get, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn platform_resource_paths_have_one_canonical_route_spelling() {
        let org = Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let project = Uuid::parse_str("223e4567-e89b-12d3-a456-426614174000").unwrap();
        let user = Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap();
        let invite = Uuid::parse_str("423e4567-e89b-12d3-a456-426614174000").unwrap();

        for (method, path, expected) in [
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}"),
                PlatformResourcePath::Organization(org),
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects"),
                PlatformResourcePath::Projects(org),
            ),
            (
                LogicalMethod::Patch,
                format!("/platform/orgs/{org}/projects/{project}"),
                PlatformResourcePath::Project {
                    org_id: org,
                    project_id: project,
                },
            ),
            (
                LogicalMethod::Post,
                format!("/platform/orgs/{org}/projects/{project}/secrets"),
                PlatformResourcePath::Secrets {
                    org_id: org,
                    project_id: project,
                },
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/projects/{project}/secrets/API_KEY_2"),
                PlatformResourcePath::Secret {
                    org_id: org,
                    project_id: project,
                    key_name: Zeroizing::new("API_KEY_2".to_owned()),
                },
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/projects/{project}/settings/email"),
                PlatformResourcePath::EmailSettings {
                    org_id: org,
                    project_id: project,
                },
            ),
            (
                LogicalMethod::Put,
                format!("/platform/orgs/{org}/projects/{project}/settings/oauth"),
                PlatformResourcePath::OAuthSettings {
                    org_id: org,
                    project_id: project,
                },
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/memberships"),
                PlatformResourcePath::Memberships(org),
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{org}/memberships/{user}"),
                PlatformResourcePath::Membership {
                    org_id: org,
                    user_id: user,
                },
            ),
            (
                LogicalMethod::Post,
                format!("/platform/orgs/{org}/invites"),
                PlatformResourcePath::Invites(org),
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{org}/invites/{invite}"),
                PlatformResourcePath::Invite {
                    org_id: org,
                    invite_code: invite,
                },
            ),
            (
                LogicalMethod::Post,
                format!("/platform/accept_invite/{invite}"),
                PlatformResourcePath::AcceptInvite(invite),
            ),
        ] {
            assert_eq!(
                decode_canonical_platform_resource_path(method, &path).unwrap(),
                Some(expected),
                "{method:?} {path}"
            );
            validate_logical_path(method, &path, &EnvelopeLimits::default()).unwrap();
        }
    }

    #[test]
    fn platform_resource_paths_reject_aliases_and_route_transplants() {
        let canonical_org = "123e4567-e89b-12d3-a456-426614174000";
        let canonical_project = "223e4567-e89b-12d3-a456-426614174000";
        for (method, invalid) in [
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{}/projects", canonical_org.to_uppercase()),
            ),
            (
                LogicalMethod::Get,
                format!("/platform/orgs/{canonical_org}/projects/{canonical_project}/"),
            ),
            (
                LogicalMethod::Delete,
                format!("/platform/orgs/{canonical_org}/projects/{canonical_project}/secrets/a-b"),
            ),
            (
                LogicalMethod::Delete,
                format!(
                    "/platform/orgs/{canonical_org}/projects/{canonical_project}/secrets/a%5Fb"
                ),
            ),
            (
                LogicalMethod::Post,
                "/platform/accept_invite/423E4567-E89B-12D3-A456-426614174000".to_owned(),
            ),
        ] {
            assert!(
                !matches!(
                    decode_canonical_platform_resource_path(method, &invalid),
                    Ok(Some(_))
                ),
                "{method:?} {invalid}"
            );
        }

        assert!(decode_canonical_platform_resource_path(
            LogicalMethod::Post,
            &format!("/platform/orgs/{canonical_org}/memberships")
        )
        .unwrap()
        .is_none());
        assert!(decode_canonical_platform_resource_path(
            LogicalMethod::Get,
            "/platform/accept_invite/423e4567-e89b-12d3-a456-426614174000"
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn conversation_project_item_paths_use_one_canonical_uuid_spelling() {
        let encoded = "123e4567-e89b-12d3-a456-426614174000";
        let path = format!("{CONVERSATION_PROJECT_ITEM_PATH_PREFIX}{encoded}");
        for method in [
            LogicalMethod::Get,
            LogicalMethod::Post,
            LogicalMethod::Delete,
        ] {
            assert_eq!(
                decode_canonical_conversation_project_path(method, &path).unwrap(),
                Some(Uuid::parse_str(encoded).unwrap())
            );
            validate_logical_path(method, &path, &EnvelopeLimits::default()).unwrap();
        }

        assert!(
            decode_canonical_conversation_project_path(LogicalMethod::Put, &path)
                .unwrap()
                .is_none()
        );
        assert!(decode_canonical_conversation_project_path(
            LogicalMethod::Get,
            "/v1/conversation-project/123e4567-e89b-12d3-a456-426614174000",
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn conversation_project_item_paths_reject_uuid_aliases_and_malformed_ids() {
        for method in [
            LogicalMethod::Get,
            LogicalMethod::Post,
            LogicalMethod::Delete,
        ] {
            for invalid in [
                "/v1/conversation-projects/",
                "/v1/conversation-projects/123E4567-E89B-12D3-A456-426614174000",
                "/v1/conversation-projects/123e4567e89b12d3a456426614174000",
                "/v1/conversation-projects/{123e4567-e89b-12d3-a456-426614174000}",
                "/v1/conversation-projects/123e4567%2De89b%2D12d3%2Da456%2D426614174000",
                "/v1/conversation-projects/123e4567-e89b-12d3-a456-426614174000/extra",
                "/v1/conversation-projects/not-a-uuid",
            ] {
                assert!(
                    validate_logical_path(method, invalid, &EnvelopeLimits::default()).is_err(),
                    "{method:?} {invalid}"
                );
            }
        }
    }

    #[test]
    fn instruction_item_and_set_default_paths_are_canonical_and_distinct() {
        let encoded = "123e4567-e89b-12d3-a456-426614174000";
        let id = Uuid::parse_str(encoded).unwrap();
        let item = format!("{INSTRUCTION_ITEM_PATH_PREFIX}{encoded}");
        for method in [
            LogicalMethod::Get,
            LogicalMethod::Post,
            LogicalMethod::Delete,
        ] {
            assert_eq!(
                decode_canonical_instruction_path(method, &item).unwrap(),
                Some(InstructionItemPath::Item(id))
            );
            validate_logical_path(method, &item, &EnvelopeLimits::default()).unwrap();
        }

        let set_default = format!("{item}/set-default");
        assert_eq!(
            decode_canonical_instruction_path(LogicalMethod::Post, &set_default).unwrap(),
            Some(InstructionItemPath::SetDefault(id))
        );
        validate_logical_path(
            LogicalMethod::Post,
            &set_default,
            &EnvelopeLimits::default(),
        )
        .unwrap();
        assert!(
            decode_canonical_instruction_path(LogicalMethod::Get, &set_default)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn instruction_paths_reject_uuid_aliases_and_suffix_ambiguity() {
        for invalid in [
            "/v1/instructions/",
            "/v1/instructions/123E4567-E89B-12D3-A456-426614174000",
            "/v1/instructions/123e4567e89b12d3a456426614174000",
            "/v1/instructions/{123e4567-e89b-12d3-a456-426614174000}",
            "/v1/instructions/123e4567-e89b-12d3-a456-426614174000/extra",
            "/v1/instructions/123e4567-e89b-12d3-a456-426614174000/set-default/extra",
            "/v1/instructions/not-a-uuid",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Post, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn conversation_and_item_paths_are_canonical_and_distinct() {
        let conversation = "123e4567-e89b-12d3-a456-426614174000";
        let item = "223e4567-e89b-12d3-a456-426614174000";
        let conversation_id = Uuid::parse_str(conversation).unwrap();
        let item_id = Uuid::parse_str(item).unwrap();
        let conversation_path = format!("{CONVERSATION_ITEM_PATH_PREFIX}{conversation}");

        for method in [
            LogicalMethod::Get,
            LogicalMethod::Post,
            LogicalMethod::Delete,
        ] {
            assert_eq!(
                decode_canonical_conversation_path(method, &conversation_path).unwrap(),
                Some(ConversationItemPath::Conversation(conversation_id))
            );
        }

        let items_path = format!("{conversation_path}/items");
        assert_eq!(
            decode_canonical_conversation_path(LogicalMethod::Get, &items_path).unwrap(),
            Some(ConversationItemPath::Items(conversation_id))
        );
        let item_path = format!("{items_path}/{item}");
        assert_eq!(
            decode_canonical_conversation_path(LogicalMethod::Get, &item_path).unwrap(),
            Some(ConversationItemPath::Item {
                conversation_id,
                item_id,
            })
        );
        validate_logical_path(LogicalMethod::Get, &item_path, &EnvelopeLimits::default()).unwrap();

        for action in ["batch-delete", "batch-update-project"] {
            assert!(decode_canonical_conversation_path(
                LogicalMethod::Post,
                &format!("{CONVERSATION_ITEM_PATH_PREFIX}{action}"),
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn conversation_paths_reject_uuid_aliases_and_suffix_ambiguity() {
        for invalid in [
            "/v1/conversations/",
            "/v1/conversations/123E4567-E89B-12D3-A456-426614174000",
            "/v1/conversations/123e4567e89b12d3a456426614174000",
            "/v1/conversations/{123e4567-e89b-12d3-a456-426614174000}",
            "/v1/conversations/123e4567-e89b-12d3-a456-426614174000/extra",
            "/v1/conversations/123e4567-e89b-12d3-a456-426614174000/items/",
            "/v1/conversations/123e4567-e89b-12d3-a456-426614174000/items/not-a-uuid",
            "/v1/conversations/123e4567-e89b-12d3-a456-426614174000/items/223e4567-e89b-12d3-a456-426614174000/extra",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Get, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn stored_response_item_and_cancel_paths_are_canonical() {
        let encoded = "123e4567-e89b-12d3-a456-426614174000";
        let id = Uuid::parse_str(encoded).unwrap();
        let item = format!("{RESPONSE_ITEM_PATH_PREFIX}{encoded}");
        assert_eq!(
            decode_canonical_response_path(LogicalMethod::Get, &item).unwrap(),
            Some(ResponseItemPath::Item(id))
        );
        assert_eq!(
            decode_canonical_response_path(LogicalMethod::Delete, &item).unwrap(),
            Some(ResponseItemPath::Item(id))
        );
        let cancel = format!("{item}/cancel");
        assert_eq!(
            decode_canonical_response_path(LogicalMethod::Post, &cancel).unwrap(),
            Some(ResponseItemPath::Cancel(id))
        );
        validate_logical_path(LogicalMethod::Post, &cancel, &EnvelopeLimits::default()).unwrap();
    }

    #[test]
    fn stored_response_paths_reject_uuid_aliases_and_suffix_ambiguity() {
        for invalid in [
            "/v1/responses/",
            "/v1/responses/123E4567-E89B-12D3-A456-426614174000",
            "/v1/responses/123e4567e89b12d3a456426614174000",
            "/v1/responses/{123e4567-e89b-12d3-a456-426614174000}",
            "/v1/responses/123e4567-e89b-12d3-a456-426614174000/extra",
            "/v1/responses/123e4567-e89b-12d3-a456-426614174000/cancel/extra",
        ] {
            assert!(
                validate_logical_path(LogicalMethod::Get, invalid, &EnvelopeLimits::default())
                    .is_err(),
                "{invalid}"
            );
        }
        assert!(validate_logical_path(
            LogicalMethod::Post,
            "/v1/responses/not-a-uuid/cancel",
            &EnvelopeLimits::default(),
        )
        .is_err());
    }

    #[test]
    fn header_syntax_and_each_header_limit_are_enforced() {
        let valid = HeaderField {
            name: "x-provider-beta".to_owned(),
            value_base64: EncodedBytes::from_bytes(b"one".to_vec()),
        };
        validate_headers(std::slice::from_ref(&valid), &EnvelopeLimits::default()).unwrap();

        for name in ["Content-Type", "bad:name", "", "café"] {
            let header = HeaderField {
                name: name.to_owned(),
                value_base64: EncodedBytes::from_bytes(Vec::new()),
            };
            assert!(validate_headers(&[header], &EnvelopeLimits::default()).is_err());
        }
        for value in [b"bad\rvalue".as_slice(), b"bad\nvalue", b"bad\0value"] {
            let header = HeaderField {
                name: "x-test".to_owned(),
                value_base64: EncodedBytes::from_bytes(value.to_vec()),
            };
            assert!(validate_headers(&[header], &EnvelopeLimits::default()).is_err());
        }

        let limits = EnvelopeLimits {
            header_count: 1,
            header_name_bytes: 3,
            header_value_bytes: 2,
            aggregate_header_bytes: 4,
            ..EnvelopeLimits::default()
        };
        assert!(validate_headers(&[valid.clone(), valid.clone()], &limits).is_err());
        assert!(validate_headers(
            &[HeaderField {
                name: "long".to_owned(),
                value_base64: EncodedBytes::from_bytes(Vec::new()),
            }],
            &limits
        )
        .is_err());
        assert!(validate_headers(
            &[HeaderField {
                name: "x".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"abc".to_vec()),
            }],
            &limits
        )
        .is_err());
        assert!(validate_headers(
            &[HeaderField {
                name: "abc".to_owned(),
                value_base64: EncodedBytes::from_bytes(b"ab".to_vec()),
            }],
            &limits
        )
        .is_err());
    }

    #[test]
    fn path_body_and_envelope_limits_are_checked() {
        let limits = EnvelopeLimits {
            envelope_bytes: 1024,
            logical_body_bytes: 1,
            path_bytes: 3,
            query_bytes: 2,
            ..EnvelopeLimits::default()
        };
        validate_path("/ab", &limits).unwrap();
        assert!(validate_path("/abc", &limits).is_err());
        validate_query("ab", &limits).unwrap();
        assert!(validate_query("abc", &limits).is_err());

        let oversized_body = request_json(r#""YWI=""#);
        assert!(RequestEnvelope::from_json_slice(oversized_body.as_bytes(), &limits).is_err());
        assert!(RequestEnvelope::from_json_slice(&vec![b' '; 1025], &limits).is_err());
    }

    #[test]
    fn credential_bytes_have_an_independent_limit() {
        let limits = EnvelopeLimits {
            credential_bytes: 2,
            ..EnvelopeLimits::default()
        };
        for kind in ["api_key", "resumption"] {
            let at_limit = request_json("null").replace(
                "\"credential\":null",
                &format!("\"credential\":{{\"kind\":\"{kind}\",\"value_base64\":\"YWI=\"}}"),
            );
            assert!(RequestEnvelope::from_json_slice(at_limit.as_bytes(), &limits).is_ok());

            let oversized = request_json("null").replace(
                "\"credential\":null",
                &format!("\"credential\":{{\"kind\":\"{kind}\",\"value_base64\":\"YWJj\"}}"),
            );
            assert!(RequestEnvelope::from_json_slice(oversized.as_bytes(), &limits).is_err());
        }
    }

    #[test]
    fn unary_and_stream_records_are_strict_typed_objects() {
        let limits = EnvelopeLimits::default();
        let unary = format!(
            r#"{{"version":2,"request_id":"{REQUEST_ID}","status":200,"headers":[],"body_base64":null}}"#
        );
        assert!(UnaryResponseEnvelope::from_json_slice(unary.as_bytes(), &limits).is_ok());
        assert!(UnaryResponseEnvelope::from_json_slice(
            unary.replace("\"status\":200", "\"status\":99").as_bytes(),
            &limits
        )
        .is_err());
        assert!(UnaryResponseEnvelope::from_json_slice(
            unary.replace("\"version\":2", "\"version\":3").as_bytes(),
            &limits
        )
        .is_err());

        let records = [
            format!(
                r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":0,"kind":"start","status":200,"headers":[]}}"#
            ),
            format!(
                r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":1,"kind":"chunk","body_base64":"YQ=="}}"#
            ),
            format!(r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":2,"kind":"end"}}"#),
            format!(
                r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":2,"kind":"error","status":500,"body_base64":"YQ=="}}"#
            ),
        ];
        for record in &records {
            assert!(StreamRecord::from_json_slice(record.as_bytes(), &limits).is_ok());
        }

        let unknown = format!(
            r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":2,"kind":"end","extra":true}}"#
        );
        assert!(StreamRecord::from_json_slice(unknown.as_bytes(), &limits).is_err());
        let duplicate = format!(
            r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":2,"sequence":3,"kind":"end"}}"#
        );
        assert!(StreamRecord::from_json_slice(duplicate.as_bytes(), &limits).is_err());

        let non_initial_start = format!(
            r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":1,"kind":"start","status":200,"headers":[]}}"#
        );
        assert!(StreamRecord::from_json_slice(non_initial_start.as_bytes(), &limits).is_err());
        assert!(StreamRecord::from_json_slice(
            records[0]
                .replace("\"status\":200", "\"status\":400")
                .as_bytes(),
            &limits
        )
        .is_err());
        assert!(StreamRecord::from_json_slice(
            records[3]
                .replace("\"status\":500", "\"status\":200")
                .as_bytes(),
            &limits
        )
        .is_err());
        for kind_and_fields in [
            r#""kind":"chunk","body_base64":"YQ==""#,
            r#""kind":"end""#,
            r#""kind":"error","status":500,"body_base64":"YQ==""#,
        ] {
            let invalid = format!(
                r#"{{"version":2,"request_id":"{REQUEST_ID}","sequence":0,{kind_and_fields}}}"#
            );
            assert!(StreamRecord::from_json_slice(invalid.as_bytes(), &limits).is_err());
        }

        let at_chunk_limit = StreamRecord::Chunk {
            version: Version2,
            request_id: RequestId::from_bytes([0x11; 16]),
            sequence: 1,
            body_base64: EncodedBytes::from_bytes(vec![0; MAX_STREAM_CHUNK_BYTES]),
        };
        assert!(at_chunk_limit.validate(&limits).is_ok());
        let oversized_chunk = StreamRecord::Chunk {
            version: Version2,
            request_id: RequestId::from_bytes([0x11; 16]),
            sequence: 1,
            body_base64: EncodedBytes::from_bytes(vec![0; MAX_STREAM_CHUNK_BYTES + 1]),
        };
        assert!(oversized_chunk.validate(&limits).is_err());

        let at_error_limit = StreamRecord::Error {
            version: Version2,
            request_id: RequestId::from_bytes([0x11; 16]),
            sequence: 1,
            status: 500,
            body_base64: EncodedBytes::from_bytes(vec![0; MAX_STREAM_ERROR_BYTES]),
        };
        assert!(at_error_limit.validate(&limits).is_ok());
    }

    #[test]
    fn default_limits_match_the_protocol_contract() {
        assert_eq!(EnvelopeLimits::REQUEST.envelope_bytes, 67 * 1024 * 1024);
        assert_eq!(EnvelopeLimits::REQUEST.logical_body_bytes, 50 * 1024 * 1024);
        assert_eq!(EnvelopeLimits::RESPONSE.envelope_bytes, 50 * 1024 * 1024);
        assert_eq!(
            EnvelopeLimits::RESPONSE.logical_body_bytes,
            28 * 1024 * 1024
        );
        assert_eq!(EnvelopeLimits::RESPONSE.path_bytes, 4096);
        assert_eq!(EnvelopeLimits::RESPONSE.query_bytes, 8192);
        assert_eq!(EnvelopeLimits::RESPONSE.header_count, 64);
        assert_eq!(EnvelopeLimits::RESPONSE.header_name_bytes, 128);
        assert_eq!(EnvelopeLimits::RESPONSE.header_value_bytes, 16 * 1024);
        assert_eq!(EnvelopeLimits::RESPONSE.aggregate_header_bytes, 64 * 1024);
    }

    #[test]
    fn maximum_structural_request_shape_fits_the_request_envelope_limit() {
        let limits = EnvelopeLimits::REQUEST;
        let mut headers = Vec::with_capacity(limits.header_count);
        for _ in 0..60 {
            headers.push(HeaderField {
                name: "x".to_owned(),
                value_base64: EncodedBytes::from_bytes(vec![b'a'; 1]),
            });
        }
        for _ in 0..3 {
            headers.push(HeaderField {
                name: "x".to_owned(),
                value_base64: EncodedBytes::from_bytes(vec![b'a'; 16_381]),
            });
        }
        headers.push(HeaderField {
            name: "x".to_owned(),
            value_base64: EncodedBytes::from_bytes(vec![b'a'; 16_269]),
        });

        let path_prefix = "/protected/kv/";
        let envelope = RequestEnvelope {
            version: Version2,
            request_id: RequestId::from_bytes([0xff; 16]),
            response_mode: ResponseMode::Stream,
            credential: Some(Credential::Resumption {
                value_base64: EncodedBytes::from_bytes(vec![0_u8; limits.credential_bytes]),
            }),
            cache_namespace_root_base64: Some(CacheNamespaceRoot::from_bytes([0xff; 32])),
            request: LogicalRequest {
                method: LogicalMethod::Delete,
                path: format!(
                    "{path_prefix}{}",
                    "A".repeat(limits.path_bytes - path_prefix.len())
                ),
                query: Some("q".repeat(limits.query_bytes)),
                headers,
                body_base64: Some(EncodedBytes::from_bytes(Vec::new())),
            },
        };
        envelope.validate(&limits).unwrap();

        let empty_body_json = serde_json::to_vec(&envelope).unwrap();
        let maximum_body_base64_bytes = limits.logical_body_bytes.div_ceil(3) * 4;
        let projected_maximum_envelope_bytes = empty_body_json
            .len()
            .checked_add(maximum_body_base64_bytes)
            .unwrap();

        assert_eq!(projected_maximum_envelope_bytes, 70_028_948);
        assert!(projected_maximum_envelope_bytes <= limits.envelope_bytes);
    }
}
