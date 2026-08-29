use std::fmt;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use zeroize::{Zeroize, ZeroizeOnDrop};

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
    pub(crate) const DEFAULT: Self = Self {
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
}

impl Default for EnvelopeLimits {
    fn default() -> Self {
        Self::DEFAULT
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
        validate_path(&self.path, limits)?;
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
                validate_status(*status)?;
                validate_headers(headers, limits)
            }
            Self::Chunk {
                sequence,
                body_base64,
                ..
            } => {
                validate_non_initial_sequence(*sequence)?;
                check_limit(body_base64.len(), limits.logical_body_bytes, "logical body")
            }
            Self::End { sequence, .. } => validate_non_initial_sequence(*sequence),
            Self::Error {
                sequence,
                status,
                body_base64,
                ..
            } => {
                validate_non_initial_sequence(*sequence)?;
                validate_status(*status)?;
                check_limit(body_base64.len(), limits.logical_body_bytes, "logical body")
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
    }

    fn request_json(body: &str) -> String {
        format!(
            r#"{{
                "version":2,
                "request_id":"{REQUEST_ID}",
                "response_mode":"auto",
                "credential":null,
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
        for record in records {
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
    }

    #[test]
    fn default_limits_match_the_protocol_contract() {
        assert_eq!(EnvelopeLimits::DEFAULT.envelope_bytes, 50 * 1024 * 1024);
        assert_eq!(EnvelopeLimits::DEFAULT.logical_body_bytes, 28 * 1024 * 1024);
        assert_eq!(EnvelopeLimits::DEFAULT.path_bytes, 4096);
        assert_eq!(EnvelopeLimits::DEFAULT.query_bytes, 8192);
        assert_eq!(EnvelopeLimits::DEFAULT.header_count, 64);
        assert_eq!(EnvelopeLimits::DEFAULT.header_name_bytes, 128);
        assert_eq!(EnvelopeLimits::DEFAULT.header_value_bytes, 16 * 1024);
        assert_eq!(EnvelopeLimits::DEFAULT.aggregate_header_bytes, 64 * 1024);
    }
}
