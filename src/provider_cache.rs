use std::fmt;
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use hmac::{Hmac, Mac};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use sha2::Sha256;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

pub(crate) const CACHE_NAMESPACE_ROOT_BYTES: usize = 32;
const CACHE_NAMESPACE_ROOT_BASE64_BYTES: usize = 44;

/// Versioned domain separation for the Tinfoil user-cache namespace.
///
/// The trailing zero byte makes the UTF-8 label self-delimiting before the
/// fixed-width UUID bytes.
const TINFOIL_CACHE_NAMESPACE_V1_LABEL: &[u8] =
    b"opensecret/provider-cache/tinfoil/user-cache-namespace/v1";

type HmacSha256 = Hmac<Sha256>;

/// Client-generated root material used only while binding a v2 authority.
///
/// The root is never retained in a bound session. Its wire representation is
/// canonical padded standard base64 for exactly 32 bytes.
#[derive(Clone, Eq, PartialEq, Zeroize, ZeroizeOnDrop)]
pub(crate) struct CacheNamespaceRoot([u8; CACHE_NAMESPACE_ROOT_BYTES]);

impl CacheNamespaceRoot {
    #[cfg(test)]
    pub(crate) const fn from_bytes(bytes: [u8; CACHE_NAMESPACE_ROOT_BYTES]) -> Self {
        Self(bytes)
    }

    const fn as_bytes(&self) -> &[u8; CACHE_NAMESPACE_ROOT_BYTES] {
        &self.0
    }
}

impl fmt::Debug for CacheNamespaceRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CacheNamespaceRoot([REDACTED])")
    }
}

impl Serialize for CacheNamespaceRoot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = Zeroizing::new(STANDARD.encode(self.0));
        serializer.serialize_str(encoded.as_str())
    }
}

impl<'de> Deserialize<'de> for CacheNamespaceRoot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CacheNamespaceRootVisitor;

        impl de::Visitor<'_> for CacheNamespaceRootVisitor {
            type Value = CacheNamespaceRoot;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("canonical padded standard base64 encoding exactly 32 bytes")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.len() != CACHE_NAMESPACE_ROOT_BASE64_BYTES {
                    return Err(E::custom("cache namespace root must decode to 32 bytes"));
                }

                let mut decoded = Zeroizing::new([0_u8; CACHE_NAMESPACE_ROOT_BYTES]);
                let decoded_bytes = STANDARD
                    .decode_slice(value.as_bytes(), &mut decoded[..])
                    .map_err(|_| E::custom("invalid standard base64 cache namespace root"))?;
                if decoded_bytes != CACHE_NAMESPACE_ROOT_BYTES {
                    return Err(E::custom("cache namespace root must decode to 32 bytes"));
                }

                let mut canonical = Zeroizing::new([0_u8; CACHE_NAMESPACE_ROOT_BASE64_BYTES]);
                let encoded_bytes = STANDARD
                    .encode_slice(&decoded[..], &mut canonical[..])
                    .map_err(|_| E::custom("invalid cache namespace root"))?;
                if encoded_bytes != value.len() || &canonical[..encoded_bytes] != value.as_bytes() {
                    return Err(E::custom(
                        "non-canonical standard base64 cache namespace root",
                    ));
                }

                Ok(CacheNamespaceRoot(*decoded))
            }
        }

        deserializer.deserialize_str(CacheNamespaceRootVisitor)
    }
}

#[derive(Eq, PartialEq, Zeroize, ZeroizeOnDrop)]
struct DerivedCacheNamespaceBytes([u8; 32]);

/// Cloneable provider cache namespace retained by a bound user or API key.
///
/// Clones share one allocation. The final `Arc` drop zeroizes the derived
/// bytes, while `Debug` never exposes them.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct DerivedCacheNamespace(Arc<DerivedCacheNamespaceBytes>);

impl DerivedCacheNamespace {
    /// Tinfoil's `user_cache_secret` representation.
    ///
    /// The returned lowercase hexadecimal string is provider-bound secret
    /// material and must not be logged or returned to the client.
    pub(crate) fn tinfoil_user_cache_secret(&self) -> String {
        hex::encode(self.0.as_ref().0)
    }
}

impl fmt::Debug for DerivedCacheNamespace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DerivedCacheNamespace([REDACTED])")
    }
}

/// Derive a stable Tinfoil cache namespace for one verified user identity.
///
/// The exact derivation is:
///
/// ```text
/// HMAC-SHA256(
///   cache_namespace_root,
///   UTF8("opensecret/provider-cache/tinfoil/user-cache-namespace/v1")
///   || 0x00
///   || verified_user_uuid_bytes[16]
/// )
/// ```
pub(crate) fn derive_tinfoil_cache_namespace(
    root: &CacheNamespaceRoot,
    verified_user_id: Uuid,
) -> DerivedCacheNamespace {
    let mut hmac = HmacSha256::new_from_slice(root.as_bytes())
        .expect("HMAC-SHA256 accepts cache namespace roots of any length");
    hmac.update(TINFOIL_CACHE_NAMESPACE_V1_LABEL);
    hmac.update(&[0]);
    hmac.update(verified_user_id.as_bytes());

    let bytes = Zeroizing::new(<[u8; 32]>::from(hmac.finalize().into_bytes()));
    DerivedCacheNamespace(Arc::new(DerivedCacheNamespaceBytes(*bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT_BYTES: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    const USER_ID: Uuid = Uuid::from_bytes([
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ]);

    #[test]
    fn cache_namespace_root_and_final_derived_allocation_zeroize_on_drop() {
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}

        assert_zeroize_on_drop::<CacheNamespaceRoot>();
        assert_zeroize_on_drop::<DerivedCacheNamespaceBytes>();
    }

    #[test]
    fn derivation_is_deterministic_and_domain_versioned() {
        let root = CacheNamespaceRoot::from_bytes(ROOT_BYTES);
        let first = derive_tinfoil_cache_namespace(&root, USER_ID);
        let second = derive_tinfoil_cache_namespace(&root, USER_ID);

        assert_eq!(first, second);
        assert_eq!(
            first.tinfoil_user_cache_secret(),
            "1e30544852f2ef1db03bbd8d3a34bb106120d89b0769d4e199a40a99d7150927"
        );

        let other_user = derive_tinfoil_cache_namespace(&root, Uuid::from_u128(1));
        assert_ne!(first, other_user);

        let other_root = CacheNamespaceRoot::from_bytes([0x55; 32]);
        assert_ne!(first, derive_tinfoil_cache_namespace(&other_root, USER_ID));
    }

    #[test]
    fn derived_namespace_is_redacted_and_clones_share_final_drop_allocation() {
        let root = CacheNamespaceRoot::from_bytes(ROOT_BYTES);
        assert_eq!(format!("{root:?}"), "CacheNamespaceRoot([REDACTED])");
        let namespace = derive_tinfoil_cache_namespace(&root, USER_ID);
        let expected = namespace.tinfoil_user_cache_secret();
        let clone = namespace.clone();

        assert!(Arc::ptr_eq(&namespace.0, &clone.0));
        assert_eq!(Arc::strong_count(&namespace.0), 2);
        assert_eq!(
            format!("{namespace:?}"),
            "DerivedCacheNamespace([REDACTED])"
        );
        assert!(!format!("{namespace:?}").contains(&expected));

        drop(namespace);
        assert_eq!(Arc::strong_count(&clone.0), 1);
        assert_eq!(clone.tinfoil_user_cache_secret(), expected);
    }
}
