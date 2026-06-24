use crate::{encrypt::generate_random, ApiError, AppMode, AppState};
use aws_nitro_enclaves_nsm_api::{
    api::{Request, Response},
    driver::{nsm_exit, nsm_init, nsm_process_request},
};
use axum::routing::post;
use axum::{extract::State, routing::get, Json};
use axum::{http::StatusCode, Router};
use base64::{engine::general_purpose, Engine as _};
use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, Key, KeyInit, Nonce};
use chrono::{Duration, Utc};
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use serde_cbor::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{error, trace};
use uuid::Uuid;
use yasna::models::ObjectIdentifier;
use yasna::{construct_der, Tag};

pub struct SessionState {
    session_key: [u8; 32],
}

impl SessionState {
    pub fn new(session_key: [u8; 32]) -> Self {
        Self { session_key }
    }

    pub fn get_session_key(&self) -> [u8; 32] {
        self.session_key
    }

    pub fn decrypt(&self, encrypted_data: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>, ApiError> {
        tracing::trace!("decrypting encrypted data");
        tracing::trace!("session key: {:?}", self.session_key);
        tracing::trace!("nonce: {:?}", nonce);
        tracing::trace!("encrypted data length: {}", encrypted_data.len());

        let key = Key::from_slice(self.session_key.as_ref());
        let cipher = ChaCha20Poly1305::new(key);
        let nonce = Nonce::from_slice(nonce);

        cipher.decrypt(nonce, encrypted_data).map_err(|e| {
            tracing::error!("could not decrypt data: {e}");
            ApiError::InternalServerError
        })
    }
}

#[derive(Deserialize)]
struct KeyExchangeRequest {
    nonce: String,
    client_public_key: String,
}

#[derive(Serialize)]
struct KeyExchangeResponse {
    session_id: Uuid,
    encrypted_session_key: String,
}

#[derive(Serialize)]
struct AttestationResponse {
    attestation_document: String,
}

pub fn router(app_state: Arc<AppState>) -> Router<()> {
    Router::new()
        .route("/attestation/:nonce", get(get_attestation))
        .route("/key_exchange", post(key_exchange))
        .with_state(app_state)
}

async fn get_attestation(
    State(data): State<Arc<AppState>>,
    axum::extract::Path(nonce): axum::extract::Path<String>,
) -> Result<(StatusCode, Json<AttestationResponse>), ApiError> {
    // Create an ephemeral key pair for this request
    trace!("Creating ephemeral key");
    let enclave_public_key = data.create_ephemeral_key(nonce.clone()).await;
    trace!("Ephemeral key created");

    // Create a request for the attestation document
    let request = Request::Attestation {
        user_data: None,
        public_key: Some(ByteBuf::from(enclave_public_key.as_bytes().to_vec())),
        nonce: Some(ByteBuf::from(nonce.into_bytes())),
    };

    trace!("Generating attestation based on app mode");
    let result = match data.app_mode {
        AppMode::Local => generate_mock_attestation(data.clone(), request).await,
        _ => generate_real_attestation(data, request).await,
    };
    result
}

async fn generate_mock_attestation(
    data: Arc<AppState>,
    request: Request,
) -> Result<(StatusCode, Json<AttestationResponse>), ApiError> {
    let (user_data, nonce, public_key) = match request {
        Request::Attestation {
            user_data,
            nonce,
            public_key,
        } => (user_data, nonce, public_key),
        _ => unreachable!(),
    };

    // Create a mock attestation document
    trace!("Creating mock attestation document");
    let mock_document =
        create_mock_attestation_document(data.clone(), user_data, nonce, public_key).await;
    trace!("Mock attestation document created");

    // Encode the mock document
    trace!("Encoding mock document");
    let encoded_document = serde_cbor::to_vec(&mock_document).map_err(|e| {
        error!("Failed to encode mock document: {}", e);
        ApiError::InternalServerError
    })?;
    trace!("Mock document encoded");

    // Sign the mock document
    trace!("Signing mock document");
    let (signature, _) = sign_mock_document(&encoded_document).map_err(|e| {
        error!("Failed to sign mock document: {}", e);
        ApiError::InternalServerError
    })?;
    trace!("Mock document signed");

    // Create the COSE_Sign1 structure
    trace!("Creating COSE_Sign1 structure");
    let cose_sign1 = create_cose_sign1(encoded_document, signature);
    trace!("COSE_Sign1 structure created");

    // Encode the COSE_Sign1 structure
    trace!("Encoding COSE_Sign1 structure");
    let final_document = serde_cbor::to_vec(&cose_sign1).map_err(|e| {
        error!("Failed to encode COSE_Sign1 structure: {}", e);
        ApiError::InternalServerError
    })?;
    trace!("COSE_Sign1 structure encoded");

    // Convert to base64
    trace!("Converting to base64");
    let attestation_doc_base64 = general_purpose::STANDARD.encode(&final_document);
    trace!("Converted to base64");
    Ok((
        StatusCode::OK,
        Json(AttestationResponse {
            attestation_document: attestation_doc_base64,
        }),
    ))
}

async fn create_mock_attestation_document(
    data: Arc<AppState>,
    user_data: Option<ByteBuf>,
    nonce: Option<ByteBuf>,
    public_key: Option<ByteBuf>,
) -> Value {
    let mut pcrs = BTreeMap::new();
    for i in 0..3 {
        trace!("Generating random bytes for PCR {}", i);
        let random_bytes = generate_random::<48>();
        pcrs.insert(
            Value::Integer(i.into()),
            Value::Bytes(random_bytes.to_vec()),
        );
    }

    trace!("Generating module_id");
    let module_id = format!("i-{}", hex::encode(generate_random::<8>()));

    // Create a mock certificate
    trace!("Creating mock certificate");
    let mock_cert = create_mock_certificate(data.clone()).await;
    trace!("Creating cabundle");
    let cabundle = vec![
        create_mock_certificate(data.clone()).await,
        create_mock_certificate(data.clone()).await,
    ];

    trace!("Building attestation document");
    let mut document = BTreeMap::new();
    document.insert(Value::Text("module_id".into()), Value::Text(module_id));
    document.insert(Value::Text("digest".into()), Value::Text("SHA384".into()));
    document.insert(
        Value::Text("timestamp".into()),
        Value::Integer(chrono::Utc::now().timestamp().into()),
    );
    document.insert(Value::Text("pcrs".into()), Value::Map(pcrs));
    document.insert(Value::Text("certificate".into()), Value::Bytes(mock_cert));
    document.insert(
        Value::Text("cabundle".into()),
        Value::Array(cabundle.into_iter().map(Value::Bytes).collect()),
    );

    // Always include public_key, even if None
    document.insert(
        Value::Text("public_key".into()),
        public_key.map_or(Value::Null, |p| Value::Bytes(p.to_vec())),
    );

    // Always include user_data, even if None
    document.insert(
        Value::Text("user_data".into()),
        user_data.map_or(Value::Null, |u| Value::Bytes(u.into_vec())),
    );

    // Always include nonce, even if None
    document.insert(
        Value::Text("nonce".into()),
        nonce.map_or(Value::Null, |n| Value::Bytes(n.into_vec())),
    );
    Value::Map(document)
}

async fn create_mock_certificate(_data: Arc<AppState>) -> Vec<u8> {
    trace!("Generating random bytes");
    let random_8_bytes = generate_random::<8>();
    let random_32_bytes = generate_random::<32>();

    trace!("Constructing DER");
    let result = construct_der(|writer| {
        writer.write_sequence(|writer| {
            // TBSCertificate
            writer.next().write_sequence(|writer| {
                // Version
                writer.next().write_tagged(Tag::context(0), |writer| {
                    writer.write_i32(2) // v3
                });
                // SerialNumber
                writer.next().write_u64(u64::from_be_bytes(random_8_bytes));
                // Signature Algorithm
                writer.next().write_sequence(|writer| {
                    writer
                        .next()
                        .write_oid(&ObjectIdentifier::from_slice(&[1, 2, 840, 10045, 4, 3, 2]));
                    // ecdsa-with-SHA256
                });
                // Issuer
                writer.next().write_sequence(|writer| {
                    writer.next().write_set(|writer| {
                        writer.next().write_sequence(|writer| {
                            writer
                                .next()
                                .write_oid(&ObjectIdentifier::from_slice(&[2, 5, 4, 3])); // commonName
                            writer.next().write_utf8_string("Mock CA");
                        });
                    });
                });
                // Validity
                writer.next().write_sequence(|writer| {
                    let now = Utc::now();
                    let not_after = now + Duration::days(365);

                    // Write dates as bytes
                    writer
                        .next()
                        .write_bytes(&now.format("%y%m%d%H%M%SZ").to_string().into_bytes());
                    writer
                        .next()
                        .write_bytes(&not_after.format("%y%m%d%H%M%SZ").to_string().into_bytes());
                });
                // Subject
                writer.next().write_sequence(|writer| {
                    writer.next().write_set(|writer| {
                        writer.next().write_sequence(|writer| {
                            writer
                                .next()
                                .write_oid(&ObjectIdentifier::from_slice(&[2, 5, 4, 3])); // commonName
                            writer.next().write_utf8_string("Mock Enclave");
                        });
                    });
                });
                // SubjectPublicKeyInfo
                writer.next().write_sequence(|writer| {
                    writer.next().write_sequence(|writer| {
                        writer
                            .next()
                            .write_oid(&ObjectIdentifier::from_slice(&[1, 2, 840, 10045, 2, 1])); // ecPublicKey
                        writer
                            .next()
                            .write_oid(&ObjectIdentifier::from_slice(&[1, 3, 132, 0, 34]));
                        // secp384r1
                    });
                    writer
                        .next()
                        .write_bitvec_bytes(&random_32_bytes, random_32_bytes.len() * 8);
                });
            });
            // SignatureAlgorithm
            writer.next().write_sequence(|writer| {
                writer
                    .next()
                    .write_oid(&ObjectIdentifier::from_slice(&[1, 2, 840, 10045, 4, 3, 2]));
                // ecdsa-with-SHA256
            });
            // SignatureValue
            writer
                .next()
                .write_bitvec_bytes(&random_32_bytes, random_32_bytes.len() * 8);
        })
    });
    result
}

fn sign_mock_document(document: &[u8]) -> Result<(Vec<u8>, PublicKey), String> {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(&[0x42; 32])
        .map_err(|e| format!("Failed to create secret key: {}", e))?;
    let public_key = PublicKey::from_secret_key(&secp, &secret_key);

    let mut hasher = Sha256::new();
    hasher.update(document);
    let message_hash = hasher.finalize();

    let message = secp256k1::Message::from_digest_slice(&message_hash)
        .map_err(|e| format!("Failed to create message from digest: {}", e))?;

    let signature = secp.sign_ecdsa(&message, &secret_key);

    Ok((signature.serialize_compact().to_vec(), public_key))
}

fn create_cose_sign1(payload: Vec<u8>, signature: Vec<u8>) -> Value {
    Value::Array(vec![
        Value::Bytes(vec![]),        // Protected header (empty)
        Value::Map(BTreeMap::new()), // Unprotected header (empty)
        Value::Bytes(payload),
        Value::Bytes(signature),
    ])
}

async fn generate_real_attestation(
    _data: Arc<AppState>,
    request: Request,
) -> Result<(StatusCode, Json<AttestationResponse>), ApiError> {
    // Initialize the Nitro Secure Module (NSM) driver
    let nsm_fd = nsm_init();
    if nsm_fd < 0 {
        return Err(ApiError::InternalServerError);
    }

    // Process the request and get the response
    let response = nsm_process_request(nsm_fd, request);

    // Close the NSM file descriptor
    nsm_exit(nsm_fd);

    // Handle the response
    match response {
        Response::Attestation { document } => {
            // Convert the attestation document to a base64 encoded string
            let attestation_doc_base64 = general_purpose::STANDARD.encode(&document);

            Ok((
                StatusCode::OK,
                Json(AttestationResponse {
                    attestation_document: attestation_doc_base64,
                }),
            ))
        }
        Response::Error(_) => {
            error!("NSM returned an error response");
            Err(ApiError::InternalServerError)
        }
        _ => {
            error!("Unexpected response from NSM");
            Err(ApiError::InternalServerError)
        }
    }
}

async fn key_exchange(
    State(data): State<Arc<AppState>>,
    Json(payload): Json<KeyExchangeRequest>,
) -> Result<Json<KeyExchangeResponse>, ApiError> {
    trace!("Starting key exchange");

    let client_public_key_bytes = general_purpose::STANDARD
        .decode(&payload.client_public_key)
        .map_err(|_| ApiError::BadRequest)?;

    let client_public_key = x25519_dalek::PublicKey::from(
        <[u8; 32]>::try_from(client_public_key_bytes.as_slice())
            .map_err(|_| ApiError::BadRequest)?,
    );

    let ephemeral_secret = data
        .get_and_remove_ephemeral_secret(&payload.nonce)
        .await
        .ok_or(ApiError::BadRequest)?;

    let shared_secret = ephemeral_secret.diffie_hellman(&client_public_key);

    // Generate a random session key using your secure random function
    let session_key: [u8; 32] = crate::encrypt::generate_random();

    // Encrypt the session key using the shared secret
    let nonce_bytes: [u8; 12] = crate::encrypt::generate_random();
    let nonce = Nonce::from_slice(&nonce_bytes);
    let cipher = ChaCha20Poly1305::new(shared_secret.as_bytes().into());
    let mut encrypted_session_key = nonce_bytes.to_vec();
    encrypted_session_key.extend_from_slice(
        &cipher
            .encrypt(nonce, session_key.as_ref())
            .map_err(|_| ApiError::InternalServerError)?,
    );

    // Generate a new UUID for the session
    let session_id = Uuid::new_v4();

    trace!(
        "Generated session key {:?} for nonce {:?}",
        session_key,
        nonce
    );

    // Store the session state
    data.session_states
        .write()
        .await
        .insert(session_id, SessionState::new(session_key));
    Ok(Json(KeyExchangeResponse {
        session_id,
        encrypted_session_key: general_purpose::STANDARD.encode(&encrypted_session_key),
    }))
}
