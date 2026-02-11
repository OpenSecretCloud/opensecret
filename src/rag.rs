use crate::encrypt::{decrypt_with_key, encrypt_with_key};
use crate::models::schema::user_embeddings;
use crate::models::user_embeddings::NewUserEmbedding;
use crate::models::users::User;
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use diesel::prelude::*;
use secp256k1::SecretKey;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error};
use uuid::Uuid;

pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";
pub const DEFAULT_EMBEDDING_DIM: i32 = 768;

const CACHE_MAX_USERS: usize = 100;
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

const DB_SCAN_BATCH_SIZE: i64 = 1000;

#[allow(dead_code)]
pub const SOURCE_TYPE_MESSAGE: &str = "message";
pub const SOURCE_TYPE_ARCHIVAL: &str = "archival";

#[derive(Debug, Clone, Serialize)]
pub struct RagSearchResult {
    pub content: String,
    pub score: f32,
    pub token_count: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RagEmbeddingsStatus {
    pub total_embeddings: i64,
    pub by_model: HashMap<String, i64>,
    pub stale_count: i64,
}

#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub source_type: String,
    pub conversation_id: Option<i64>,
    pub vector: Vec<f32>,
    pub content_enc: Vec<u8>,
    pub token_count: i32,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    loaded_at: Instant,
    embeddings: Arc<Vec<CachedEmbedding>>,
}

#[derive(Debug)]
pub struct RagCache {
    max_users: usize,
    ttl: Duration,
    entries: HashMap<Uuid, CacheEntry>,
    lru: VecDeque<Uuid>,
}

impl Default for RagCache {
    fn default() -> Self {
        Self::new(CACHE_MAX_USERS, CACHE_TTL)
    }
}

impl RagCache {
    pub fn new(max_users: usize, ttl: Duration) -> Self {
        Self {
            max_users,
            ttl,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn evict_user(&mut self, user_id: Uuid) {
        self.entries.remove(&user_id);
        self.lru.retain(|u| *u != user_id);
    }

    pub fn get(&mut self, user_id: Uuid) -> Option<Arc<Vec<CachedEmbedding>>> {
        self.evict_expired();

        let (loaded_at, embeddings) = {
            let entry = self.entries.get(&user_id)?;
            (entry.loaded_at, entry.embeddings.clone())
        };

        if loaded_at.elapsed() > self.ttl {
            self.evict_user(user_id);
            return None;
        }

        self.touch(user_id);
        Some(embeddings)
    }

    pub fn put(&mut self, user_id: Uuid, embeddings: Arc<Vec<CachedEmbedding>>) {
        self.entries.insert(
            user_id,
            CacheEntry {
                loaded_at: Instant::now(),
                embeddings,
            },
        );
        self.touch(user_id);

        while self.entries.len() > self.max_users {
            if let Some(lru_user) = self.lru.pop_back() {
                self.entries.remove(&lru_user);
            } else {
                break;
            }
        }
    }

    fn touch(&mut self, user_id: Uuid) {
        self.lru.retain(|u| *u != user_id);
        self.lru.push_front(user_id);
    }

    fn evict_expired(&mut self) {
        let ttl = self.ttl;
        let expired: Vec<Uuid> = self
            .entries
            .iter()
            .filter_map(|(user_id, entry)| {
                if entry.loaded_at.elapsed() > ttl {
                    Some(*user_id)
                } else {
                    None
                }
            })
            .collect();

        for user_id in expired {
            self.evict_user(user_id);
        }
    }
}

pub fn serialize_f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn deserialize_f32_le(bytes: &[u8]) -> Result<Vec<f32>, ApiError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ApiError::BadRequest);
    }

    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().map_err(|_| ApiError::BadRequest)?;
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, ApiError> {
    if a.len() != b.len() {
        return Err(ApiError::BadRequest);
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

#[derive(Debug, Clone)]
struct HeapItem {
    score: f32,
    token_count: i32,
    content_enc: Vec<u8>,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.token_count == other.token_count
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.token_count.cmp(&self.token_count))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn top_k_candidates(
    query: &[f32],
    embeddings: &[CachedEmbedding],
    top_k: usize,
    source_types: Option<&[String]>,
    conversation_id: Option<i64>,
) -> Result<Vec<HeapItem>, ApiError> {
    let allowed_source_types: Option<std::collections::HashSet<&str>> = source_types.map(|v| {
        v.iter()
            .map(|s| s.as_str())
            .collect::<std::collections::HashSet<_>>()
    });

    let mut heap: BinaryHeap<std::cmp::Reverse<HeapItem>> = BinaryHeap::new();

    for e in embeddings {
        if let Some(allowed) = &allowed_source_types {
            if !allowed.contains(e.source_type.as_str()) {
                continue;
            }
        }

        if let Some(conv_id) = conversation_id {
            if e.conversation_id != Some(conv_id) {
                continue;
            }
        }

        if e.vector.len() != query.len() {
            continue;
        }

        let score = cosine_similarity(query, &e.vector)?;
        let item = HeapItem {
            score,
            token_count: e.token_count,
            content_enc: e.content_enc.clone(),
        };

        if heap.len() < top_k {
            heap.push(std::cmp::Reverse(item));
            continue;
        }

        if let Some(std::cmp::Reverse(min)) = heap.peek() {
            if item.cmp(min) == Ordering::Greater {
                heap.pop();
                heap.push(std::cmp::Reverse(item));
            }
        }
    }

    let mut out: Vec<HeapItem> = heap.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| b.cmp(a));
    Ok(out)
}

fn apply_token_budget(results: Vec<RagSearchResult>, budget: i32) -> Vec<RagSearchResult> {
    let mut total: i32 = 0;
    let mut limited: Vec<RagSearchResult> = Vec::new();
    for r in results {
        if total + r.token_count > budget {
            break;
        }
        total += r.token_count;
        limited.push(r);
    }
    limited
}

async fn embed_text_via_tinfoil(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    text: &str,
) -> Result<(Vec<f32>, i32), ApiError> {
    crate::web::get_embedding_vector(
        state,
        user,
        auth_method,
        DEFAULT_EMBEDDING_MODEL,
        text,
        Some(DEFAULT_EMBEDDING_DIM),
    )
    .await
}

pub async fn insert_archival_embedding(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    user_key: &SecretKey,
    text: &str,
    metadata: Option<&serde_json::Value>,
) -> Result<crate::models::user_embeddings::UserEmbedding, ApiError> {
    let user_id = user.uuid;
    let (vector, token_count) = embed_text_via_tinfoil(state, user, auth_method, text).await?;

    let vector_bytes = serialize_f32_le(&vector);
    let vector_enc = encrypt_with_key(user_key, &vector_bytes).await;
    let content_enc = encrypt_with_key(user_key, text.as_bytes()).await;

    let metadata_enc = if let Some(m) = metadata {
        let m_bytes = serde_json::to_vec(m).map_err(|_| ApiError::BadRequest)?;
        Some(encrypt_with_key(user_key, &m_bytes).await)
    } else {
        None
    };

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let inserted = NewUserEmbedding {
        uuid: Uuid::new_v4(),
        user_id,
        source_type: SOURCE_TYPE_ARCHIVAL.to_string(),
        user_message_id: None,
        assistant_message_id: None,
        conversation_id: None,
        vector_enc,
        embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
        vector_dim: DEFAULT_EMBEDDING_DIM,
        content_enc,
        metadata_enc,
        token_count,
    }
    .insert(&mut conn)
    .map_err(|e| {
        error!("Failed to insert archival embedding: {:?}", e);
        ApiError::InternalServerError
    })?;

    state.rag_cache.lock().await.evict_user(user_id);
    Ok(inserted)
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_message_embedding(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    user_key: &SecretKey,
    text: &str,
    conversation_id: i64,
    user_message_id: Option<i64>,
    assistant_message_id: Option<i64>,
) -> Result<crate::models::user_embeddings::UserEmbedding, ApiError> {
    let user_id = user.uuid;
    let text = text.trim();
    if text.is_empty() {
        return Err(ApiError::BadRequest);
    }

    let (vector, token_count) = embed_text_via_tinfoil(state, user, auth_method, text).await?;

    let vector_bytes = serialize_f32_le(&vector);
    let vector_enc = encrypt_with_key(user_key, &vector_bytes).await;
    let content_enc = encrypt_with_key(user_key, text.as_bytes()).await;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let inserted = NewUserEmbedding {
        uuid: Uuid::new_v4(),
        user_id,
        source_type: SOURCE_TYPE_MESSAGE.to_string(),
        user_message_id,
        assistant_message_id,
        conversation_id: Some(conversation_id),
        vector_enc,
        embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
        vector_dim: DEFAULT_EMBEDDING_DIM,
        content_enc,
        metadata_enc: None,
        token_count,
    }
    .insert(&mut conn)
    .map_err(|e| {
        error!("Failed to insert message embedding: {:?}", e);
        ApiError::InternalServerError
    })?;

    state.rag_cache.lock().await.evict_user(user_id);
    Ok(inserted)
}

async fn load_all_user_embeddings(
    state: &AppState,
    user_id: Uuid,
    user_key: &SecretKey,
) -> Result<Arc<Vec<CachedEmbedding>>, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let mut last_id: i64 = 0;
    let mut out: Vec<CachedEmbedding> = Vec::new();

    #[derive(Queryable)]
    struct EmbeddingScanRow {
        source_type: String,
        conversation_id: Option<i64>,
        vector_enc: Vec<u8>,
        content_enc: Vec<u8>,
        token_count: i32,
        vector_dim: i32,
        id: i64,
    }

    loop {
        let rows: Vec<EmbeddingScanRow> = user_embeddings::table
            .filter(user_embeddings::user_id.eq(user_id))
            .filter(user_embeddings::id.gt(last_id))
            .order(user_embeddings::id.asc())
            .select((
                user_embeddings::source_type,
                user_embeddings::conversation_id,
                user_embeddings::vector_enc,
                user_embeddings::content_enc,
                user_embeddings::token_count,
                user_embeddings::vector_dim,
                user_embeddings::id,
            ))
            .limit(DB_SCAN_BATCH_SIZE)
            .load(&mut conn)
            .map_err(|e| {
                error!(
                    "Failed to load embeddings batch for user={} after id={}: {:?}",
                    user_id, last_id, e
                );
                ApiError::InternalServerError
            })?;

        if rows.is_empty() {
            break;
        }

        for row in rows {
            let EmbeddingScanRow {
                source_type,
                conversation_id,
                vector_enc,
                content_enc,
                token_count,
                vector_dim,
                id,
            } = row;

            let vector_bytes = decrypt_with_key(user_key, &vector_enc)
                .map_err(|_| ApiError::InternalServerError)?;
            let vector = deserialize_f32_le(&vector_bytes)?;

            if vector.len() != vector_dim as usize {
                debug!(
                    "Skipping embedding id={} for user={} due to dim mismatch (expected={}, got={})",
                    id,
                    user_id,
                    vector_dim,
                    vector.len()
                );
                continue;
            }

            out.push(CachedEmbedding {
                source_type,
                conversation_id,
                vector,
                content_enc,
                token_count,
            });
            last_id = id;
        }
    }

    Ok(Arc::new(out))
}

#[allow(clippy::too_many_arguments)]
pub async fn search_user_embeddings(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    user_key: &SecretKey,
    query: &str,
    top_k: usize,
    max_tokens: Option<i32>,
    source_types: Option<&[String]>,
    conversation_id: Option<i64>,
) -> Result<Vec<RagSearchResult>, ApiError> {
    let top_k = top_k.clamp(1, 20);

    let user_id = user.uuid;

    let (query_vec, _query_tokens) =
        embed_text_via_tinfoil(state, user, auth_method, query).await?;

    let cached = {
        let mut cache = state.rag_cache.lock().await;
        cache.get(user_id)
    };

    let embeddings = if let Some(hit) = cached {
        hit
    } else {
        let loaded = load_all_user_embeddings(state, user_id, user_key).await?;
        state.rag_cache.lock().await.put(user_id, loaded.clone());
        loaded
    };

    let candidates = top_k_candidates(
        &query_vec,
        &embeddings,
        top_k,
        source_types,
        conversation_id,
    )?;

    let mut results: Vec<RagSearchResult> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let plaintext = decrypt_with_key(user_key, &c.content_enc)
            .map_err(|_| ApiError::InternalServerError)?;
        let content = String::from_utf8(plaintext).map_err(|_| ApiError::InternalServerError)?;
        results.push(RagSearchResult {
            content,
            score: c.score,
            token_count: c.token_count,
        });
    }

    if let Some(budget) = max_tokens {
        results = apply_token_budget(results, budget);
    }

    Ok(results)
}

pub async fn delete_all_user_embeddings(state: &AppState, user_id: Uuid) -> Result<(), ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    diesel::delete(user_embeddings::table.filter(user_embeddings::user_id.eq(user_id)))
        .execute(&mut conn)
        .map_err(|e| {
            error!(
                "Failed to delete all embeddings for user={}: {:?}",
                user_id, e
            );
            ApiError::InternalServerError
        })?;

    state.rag_cache.lock().await.evict_user(user_id);
    Ok(())
}

pub async fn delete_user_embedding_by_uuid(
    state: &AppState,
    user_id: Uuid,
    embedding_uuid: Uuid,
) -> Result<(), ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let affected = diesel::delete(
        user_embeddings::table
            .filter(user_embeddings::user_id.eq(user_id))
            .filter(user_embeddings::uuid.eq(embedding_uuid)),
    )
    .execute(&mut conn)
    .map_err(|e| {
        error!(
            "Failed to delete embedding user={} uuid={}: {:?}",
            user_id, embedding_uuid, e
        );
        ApiError::InternalServerError
    })?;

    if affected == 0 {
        return Err(ApiError::NotFound);
    }

    state.rag_cache.lock().await.evict_user(user_id);
    Ok(())
}

pub async fn embeddings_status(
    state: &AppState,
    user_id: Uuid,
) -> Result<RagEmbeddingsStatus, ApiError> {
    use diesel::dsl::count_star;

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let total_embeddings: i64 = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user_id))
        .select(count_star())
        .first(&mut conn)
        .map_err(|e| {
            error!(
                "Failed to count embeddings for user={} (total): {:?}",
                user_id, e
            );
            ApiError::InternalServerError
        })?;

    let grouped: Vec<(String, i64)> = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user_id))
        .group_by(user_embeddings::embedding_model)
        .select((user_embeddings::embedding_model, count_star()))
        .load(&mut conn)
        .map_err(|e| {
            error!(
                "Failed to group embeddings by model for user={}: {:?}",
                user_id, e
            );
            ApiError::InternalServerError
        })?;

    let mut by_model: HashMap<String, i64> = HashMap::new();
    for (model, count) in grouped {
        by_model.insert(model, count);
    }

    let stale_count: i64 = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user_id))
        .filter(user_embeddings::embedding_model.ne(DEFAULT_EMBEDDING_MODEL))
        .select(count_star())
        .first(&mut conn)
        .map_err(|e| {
            error!(
                "Failed to count stale embeddings for user={}: {:?}",
                user_id, e
            );
            ApiError::InternalServerError
        })?;

    Ok(RagEmbeddingsStatus {
        total_embeddings,
        by_model,
        stale_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_roundtrip() {
        let v = vec![0.0f32, 1.5, -2.25, 42.0];
        let bytes = serialize_f32_le(&v);
        let decoded = deserialize_f32_le(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[tokio::test]
    async fn vector_encrypt_roundtrip() {
        let key = SecretKey::from_slice(&[7u8; 32]).unwrap();
        let v = vec![0.0f32, 1.5, -2.25, 42.0];

        let bytes = serialize_f32_le(&v);
        let enc = encrypt_with_key(&key, &bytes).await;
        let dec = decrypt_with_key(&key, &enc).unwrap();
        let decoded = deserialize_f32_le(&dec).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn cosine_similarity_known_values() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let c = vec![0.0f32, 1.0, 0.0];

        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_mismatched_dimensions_errors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cosine_similarity(&a, &b),
            Err(ApiError::BadRequest)
        ));
    }

    #[test]
    fn cosine_similarity_zero_vectors_return_zero() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let nonzero = vec![1.0f32, 2.0, 3.0];

        assert_eq!(cosine_similarity(&zero, &nonzero).unwrap(), 0.0);
        assert_eq!(cosine_similarity(&nonzero, &zero).unwrap(), 0.0);
        assert_eq!(cosine_similarity(&zero, &zero).unwrap(), 0.0);
    }

    #[test]
    fn deserialize_invalid_byte_length_errors() {
        let bytes = vec![0u8; 3];
        assert!(matches!(
            deserialize_f32_le(&bytes),
            Err(ApiError::BadRequest)
        ));
    }

    #[test]
    fn apply_token_budget_is_prefix() {
        let results = vec![
            RagSearchResult {
                content: "a".to_string(),
                score: 1.0,
                token_count: 6,
            },
            RagSearchResult {
                content: "b".to_string(),
                score: 0.9,
                token_count: 6,
            },
            RagSearchResult {
                content: "c".to_string(),
                score: 0.8,
                token_count: 1,
            },
        ];

        let limited = apply_token_budget(results, 10);
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].content, "a");
    }

    #[test]
    fn apply_token_budget_exact_fit_keeps_all() {
        let results = vec![
            RagSearchResult {
                content: "a".to_string(),
                score: 1.0,
                token_count: 3,
            },
            RagSearchResult {
                content: "b".to_string(),
                score: 0.9,
                token_count: 7,
            },
        ];

        let limited = apply_token_budget(results, 10);
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn apply_token_budget_zero_budget_returns_empty() {
        let results = vec![RagSearchResult {
            content: "a".to_string(),
            score: 1.0,
            token_count: 1,
        }];

        let limited = apply_token_budget(results, 0);
        assert!(limited.is_empty());
    }

    #[test]
    fn top_k_heap_ranking_and_filters() {
        let query = vec![1.0f32, 0.0];

        let embeddings = vec![
            CachedEmbedding {
                source_type: SOURCE_TYPE_ARCHIVAL.to_string(),
                conversation_id: None,
                vector: vec![1.0, 0.0],
                content_enc: b"a".to_vec(),
                token_count: 5,
            },
            CachedEmbedding {
                source_type: SOURCE_TYPE_MESSAGE.to_string(),
                conversation_id: Some(123),
                vector: vec![0.0, 1.0],
                content_enc: b"b".to_vec(),
                token_count: 7,
            },
            CachedEmbedding {
                source_type: SOURCE_TYPE_MESSAGE.to_string(),
                conversation_id: Some(123),
                vector: vec![0.8, 0.2],
                content_enc: b"c".to_vec(),
                token_count: 9,
            },
        ];

        let items = top_k_candidates(
            &query,
            &embeddings,
            2,
            Some(&[SOURCE_TYPE_MESSAGE.to_string()]),
            Some(123),
        )
        .unwrap();

        assert_eq!(items.len(), 2);
        assert!(items[0].score >= items[1].score);
        assert_eq!(items[0].content_enc, b"c");
        assert_eq!(items[1].content_enc, b"b");
    }

    #[test]
    fn top_k_tie_break_prefers_fewer_tokens() {
        let query = vec![1.0f32, 0.0];

        let embeddings = vec![
            CachedEmbedding {
                source_type: SOURCE_TYPE_ARCHIVAL.to_string(),
                conversation_id: None,
                vector: vec![1.0, 0.0],
                content_enc: b"a".to_vec(),
                token_count: 10,
            },
            CachedEmbedding {
                source_type: SOURCE_TYPE_ARCHIVAL.to_string(),
                conversation_id: None,
                vector: vec![1.0, 0.0],
                content_enc: b"b".to_vec(),
                token_count: 5,
            },
        ];

        let items = top_k_candidates(&query, &embeddings, 1, None, None).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content_enc, b"b");
    }

    #[test]
    fn top_k_when_k_gt_embeddings_returns_all() {
        let query = vec![1.0f32, 0.0];

        let embeddings = vec![
            CachedEmbedding {
                source_type: SOURCE_TYPE_ARCHIVAL.to_string(),
                conversation_id: None,
                vector: vec![1.0, 0.0],
                content_enc: b"a".to_vec(),
                token_count: 5,
            },
            CachedEmbedding {
                source_type: SOURCE_TYPE_MESSAGE.to_string(),
                conversation_id: Some(123),
                vector: vec![0.0, 1.0],
                content_enc: b"b".to_vec(),
                token_count: 7,
            },
        ];

        let total = embeddings.len();
        let items = top_k_candidates(&query, &embeddings, 10, None, None).unwrap();
        assert_eq!(items.len(), total);
    }

    #[test]
    fn top_k_with_empty_embeddings_returns_empty() {
        let query = vec![1.0f32, 0.0];
        let embeddings: Vec<CachedEmbedding> = vec![];
        let items = top_k_candidates(&query, &embeddings, 10, None, None).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn rag_cache_evict_user_removes_entry() {
        let mut cache = RagCache::new(10, Duration::from_secs(60));
        let user_id = Uuid::new_v4();

        cache.put(user_id, Arc::new(vec![]));
        assert!(cache.entries.contains_key(&user_id));

        cache.evict_user(user_id);
        assert!(!cache.entries.contains_key(&user_id));
        assert!(!cache.lru.contains(&user_id));
        assert!(cache.get(user_id).is_none());
    }

    #[tokio::test]
    async fn rag_cache_lru_eviction() {
        let mut cache = RagCache::new(2, Duration::from_secs(60));

        let v1 = Arc::new(vec![]);
        let v2 = Arc::new(vec![]);
        let v3 = Arc::new(vec![]);

        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let u3 = Uuid::new_v4();

        cache.put(u1, v1);
        cache.put(u2, v2);
        // touch u1 so u2 becomes LRU
        cache.get(u1);
        cache.put(u3, v3);

        assert!(cache.entries.contains_key(&u1));
        assert!(!cache.entries.contains_key(&u2));
        assert!(cache.entries.contains_key(&u3));
    }

    #[tokio::test]
    async fn rag_cache_ttl_expiration() {
        let mut cache = RagCache::new(10, Duration::from_millis(5));
        let user = Uuid::new_v4();

        cache.put(user, Arc::new(vec![]));
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(cache.get(user).is_none());
    }
}
