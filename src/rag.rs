#![allow(dead_code)]

use crate::encrypt::{decrypt_with_key, encrypt_key_deterministic, encrypt_with_key};
use crate::models::schema::user_embeddings;
use crate::models::user_embeddings::NewUserEmbedding;
use crate::models::users::User;
use crate::web::openai_auth::AuthMethod;
use crate::{ApiError, AppState};
use base64::{engine::general_purpose::STANDARD as B64_STANDARD, Engine as _};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::QueryableByName;
use secp256k1::SecretKey;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";
pub const DEFAULT_EMBEDDING_DIM: i32 = 768;

const DEFAULT_CACHE_MAX_BYTES: usize = 4 * 1024 * 1024 * 1024;
const DEFAULT_CACHE_MAX_USER_BYTES: usize = 128 * 1024 * 1024;
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

const DEFAULT_SCAN_LIMIT: i64 = 25_000;
const DEFAULT_MAX_INSERT_TEXT_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_SEARCH_QUERY_BYTES: usize = 4 * 1024;
const DEFAULT_MAX_USER_EMBEDDINGS: i64 = 250_000;
const DEFAULT_MAX_USER_STORED_BYTES: i64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_INSERTS_PER_USER_PER_HOUR: i64 = 10_000;
const DEFAULT_MAX_PROJECT_EMBEDDINGS: i64 = 2_500_000;
const DEFAULT_MAX_PROJECT_STORED_BYTES: i64 = 20 * 1024 * 1024 * 1024;

const CACHE_LOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
const CACHE_LOAD_TIMEOUT_BACKOFF: Duration = Duration::from_millis(250);
const FINAL_RESULT_OVERFETCH_MULTIPLIER: usize = 3;
const FINAL_RESULT_OVERFETCH_MIN_EXTRA: usize = 10;
const FINAL_RESULT_OVERFETCH_MAX: usize = 100;

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
pub struct RagSearchOutcome {
    pub results: Vec<RagSearchResult>,
    pub feedback: Option<String>,
    pub scan_limit_hit: bool,
    pub scanned_rows: usize,
    pub skipped_rows: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RagSearchFilters {
    pub source_types: Option<Vec<String>>,
    pub conversation_id: Option<i64>,
    pub tags: Option<Vec<String>>,
    pub begin_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct RagSearchOptions {
    pub limit: usize,
    pub max_tokens: Option<i32>,
    pub filters: RagSearchFilters,
}

impl Default for RagSearchOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            max_tokens: None,
            filters: RagSearchFilters::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RagEmbeddingsStatus {
    pub total_embeddings: i64,
    pub by_model: HashMap<String, i64>,
    pub stale_count: i64,
}

#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub id: i64,
    pub uuid: Uuid,
    pub source_type: String,
    pub conversation_id: Option<i64>,
    pub vector: Vec<f32>,
    pub token_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    loaded_at: Instant,
    embeddings: Arc<Vec<CachedEmbedding>>,
    bytes: usize,
    scan_limit_hit: bool,
}

#[derive(Debug)]
struct InFlightLoad {
    notify: Arc<Notify>,
    timeout_duplicate_started: bool,
}

#[derive(Debug, Clone)]
enum CacheLoadPermit {
    Start,
    Wait(Arc<Notify>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheAppendResult {
    Appended,
    Missing,
    EvictedOverLimit,
}

#[derive(Debug)]
pub struct RagCache {
    max_bytes: usize,
    max_user_bytes: usize,
    ttl: Duration,
    total_bytes: usize,
    entries: HashMap<Uuid, CacheEntry>,
    lru: VecDeque<Uuid>,
    in_flight_loads: HashMap<Uuid, InFlightLoad>,
}

impl Default for RagCache {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RagCache {
    pub fn from_env() -> Self {
        Self::new(
            read_env_usize("RAG_CACHE_MAX_BYTES", DEFAULT_CACHE_MAX_BYTES),
            read_env_usize("RAG_CACHE_MAX_USER_BYTES", DEFAULT_CACHE_MAX_USER_BYTES),
            CACHE_TTL,
        )
    }

    pub fn new(max_bytes: usize, max_user_bytes: usize, ttl: Duration) -> Self {
        Self {
            max_bytes,
            max_user_bytes,
            ttl,
            total_bytes: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            in_flight_loads: HashMap::new(),
        }
    }

    pub fn evict_user(&mut self, user_id: Uuid) {
        self.evict_user_with_reason(user_id, "manual");
    }

    fn evict_user_with_reason(&mut self, user_id: Uuid, reason: &'static str) {
        if let Some(entry) = self.entries.remove(&user_id) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            info!(
                target: "rag",
                user_id = %user_id,
                bytes = entry.bytes,
                total_cache_bytes = self.total_bytes,
                reason,
                "rag_cache_evict"
            );
        }
        self.lru.retain(|u| *u != user_id);
    }

    pub fn get(&mut self, user_id: Uuid) -> Option<(Arc<Vec<CachedEmbedding>>, bool)> {
        self.evict_expired();

        let (loaded_at, embeddings, scan_limit_hit) = {
            let entry = self.entries.get(&user_id)?;
            (
                entry.loaded_at,
                entry.embeddings.clone(),
                entry.scan_limit_hit,
            )
        };

        if loaded_at.elapsed() > self.ttl {
            self.evict_user_with_reason(user_id, "ttl");
            return None;
        }

        self.touch(user_id);
        Some((embeddings, scan_limit_hit))
    }

    pub fn put(
        &mut self,
        user_id: Uuid,
        embeddings: Arc<Vec<CachedEmbedding>>,
        scan_limit_hit: bool,
    ) -> bool {
        let bytes = cached_embeddings_bytes(&embeddings);
        if bytes > self.max_user_bytes || bytes > self.max_bytes {
            info!(
                target: "rag",
                user_id = %user_id,
                bytes,
                max_user_bytes = self.max_user_bytes,
                max_cache_bytes = self.max_bytes,
                reason = "entry_over_limit",
                "rag_cache_skip"
            );
            self.evict_user_with_reason(user_id, "entry_over_limit");
            return false;
        }

        if self.entries.contains_key(&user_id) {
            self.evict_user_with_reason(user_id, "replace");
        }

        while self.total_bytes.saturating_add(bytes) > self.max_bytes {
            if let Some(lru_user) = self.lru.pop_back() {
                self.evict_user_with_reason(lru_user, "global_byte_limit");
            } else {
                break;
            }
        }

        if self.total_bytes.saturating_add(bytes) > self.max_bytes {
            info!(
                target: "rag",
                user_id = %user_id,
                bytes,
                total_cache_bytes = self.total_bytes,
                max_cache_bytes = self.max_bytes,
                reason = "global_byte_limit",
                "rag_cache_skip"
            );
            return false;
        }

        self.entries.insert(
            user_id,
            CacheEntry {
                loaded_at: Instant::now(),
                embeddings,
                bytes,
                scan_limit_hit,
            },
        );
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.touch(user_id);
        true
    }

    pub fn append(&mut self, user_id: Uuid, embedding: CachedEmbedding) -> CacheAppendResult {
        self.evict_expired();
        let Some(entry) = self.entries.get(&user_id) else {
            return CacheAppendResult::Missing;
        };

        if entry.loaded_at.elapsed() > self.ttl {
            self.evict_user_with_reason(user_id, "ttl");
            return CacheAppendResult::Missing;
        }

        let row_bytes = embedding.estimated_cache_bytes();
        let new_entry_bytes = entry.bytes.saturating_add(row_bytes);
        let new_total_bytes = self.total_bytes.saturating_add(row_bytes);
        if new_entry_bytes > self.max_user_bytes || new_total_bytes > self.max_bytes {
            self.evict_user_with_reason(user_id, "append_over_limit");
            return CacheAppendResult::EvictedOverLimit;
        }

        if let Some(entry) = self.entries.get_mut(&user_id) {
            Arc::make_mut(&mut entry.embeddings).push(embedding);
            entry.bytes = new_entry_bytes;
            self.total_bytes = new_total_bytes;
        }
        self.touch(user_id);
        CacheAppendResult::Appended
    }

    pub fn remove_embedding_by_uuid(&mut self, user_id: Uuid, embedding_uuid: Uuid) {
        self.evict_expired();
        let Some(entry) = self.entries.get(&user_id) else {
            return;
        };

        let before_len = entry.embeddings.len();
        let old_bytes = entry.bytes;
        let Some(entry) = self.entries.get_mut(&user_id) else {
            return;
        };
        Arc::make_mut(&mut entry.embeddings).retain(|e| e.uuid != embedding_uuid);
        let after_len = entry.embeddings.len();
        let new_bytes = cached_embeddings_bytes(&entry.embeddings);
        entry.bytes = new_bytes;

        if after_len == before_len {
            return;
        }

        self.total_bytes = self
            .total_bytes
            .saturating_sub(old_bytes.saturating_sub(new_bytes));
        self.touch(user_id);
    }

    fn begin_load(&mut self, user_id: Uuid) -> CacheLoadPermit {
        if let Some(load) = self.in_flight_loads.get(&user_id) {
            return CacheLoadPermit::Wait(load.notify.clone());
        }

        self.in_flight_loads.insert(
            user_id,
            InFlightLoad {
                notify: Arc::new(Notify::new()),
                timeout_duplicate_started: false,
            },
        );
        CacheLoadPermit::Start
    }

    fn try_start_timeout_duplicate(&mut self, user_id: Uuid) -> bool {
        let Some(load) = self.in_flight_loads.get_mut(&user_id) else {
            return true;
        };

        if load.timeout_duplicate_started {
            return false;
        }

        load.timeout_duplicate_started = true;
        true
    }

    fn finish_load(&mut self, user_id: Uuid) {
        if let Some(load) = self.in_flight_loads.remove(&user_id) {
            load.notify.notify_waiters();
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
            self.evict_user_with_reason(user_id, "ttl");
        }
    }
}

fn read_env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.parse::<usize>() {
            Ok(parsed) if parsed > 0 => parsed,
            _ => {
                warn!(
                    target: "rag",
                    name,
                    value,
                    default,
                    "Invalid RAG usize env var; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn read_env_i64(name: &str, default: i64) -> i64 {
    match std::env::var(name) {
        Ok(value) => match value.parse::<i64>() {
            Ok(parsed) if parsed > 0 => parsed,
            _ => {
                warn!(
                    target: "rag",
                    name,
                    value,
                    default,
                    "Invalid RAG i64 env var; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

#[derive(Debug, Clone, Copy)]
struct RagLimits {
    scan_limit: i64,
    max_insert_text_bytes: usize,
    max_search_query_bytes: usize,
    max_user_embeddings: i64,
    max_user_stored_bytes: i64,
    max_inserts_per_user_per_hour: i64,
    max_project_embeddings: i64,
    max_project_stored_bytes: i64,
}

impl RagLimits {
    fn from_env() -> Self {
        Self {
            scan_limit: read_env_i64("RAG_SCAN_LIMIT", DEFAULT_SCAN_LIMIT),
            max_insert_text_bytes: read_env_usize(
                "RAG_MAX_INSERT_TEXT_BYTES",
                DEFAULT_MAX_INSERT_TEXT_BYTES,
            ),
            max_search_query_bytes: read_env_usize(
                "RAG_MAX_SEARCH_QUERY_BYTES",
                DEFAULT_MAX_SEARCH_QUERY_BYTES,
            ),
            max_user_embeddings: read_env_i64(
                "RAG_MAX_USER_EMBEDDINGS",
                DEFAULT_MAX_USER_EMBEDDINGS,
            ),
            max_user_stored_bytes: read_env_i64(
                "RAG_MAX_USER_STORED_BYTES",
                DEFAULT_MAX_USER_STORED_BYTES,
            ),
            max_inserts_per_user_per_hour: read_env_i64(
                "RAG_MAX_INSERTS_PER_USER_PER_HOUR",
                DEFAULT_MAX_INSERTS_PER_USER_PER_HOUR,
            ),
            max_project_embeddings: read_env_i64(
                "RAG_MAX_PROJECT_EMBEDDINGS",
                DEFAULT_MAX_PROJECT_EMBEDDINGS,
            ),
            max_project_stored_bytes: read_env_i64(
                "RAG_MAX_PROJECT_STORED_BYTES",
                DEFAULT_MAX_PROJECT_STORED_BYTES,
            ),
        }
    }
}

impl CachedEmbedding {
    fn estimated_cache_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.source_type.len())
            .saturating_add(self.vector.len().saturating_mul(std::mem::size_of::<f32>()))
    }
}

fn cached_embeddings_bytes(embeddings: &[CachedEmbedding]) -> usize {
    embeddings
        .iter()
        .map(CachedEmbedding::estimated_cache_bytes)
        .sum()
}

fn validate_text_size(text: &str, max_bytes: usize, label: &'static str) -> Result<(), ApiError> {
    if text.len() > max_bytes {
        warn!(
            target: "rag",
            label,
            bytes = text.len(),
            max_bytes,
            "RAG text exceeds configured byte limit"
        );
        return Err(ApiError::BadRequest);
    }
    Ok(())
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
    id: i64,
    uuid: Uuid,
    score: f32,
    token_count: i32,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.token_count == other.token_count
            && self.id == other.id
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.token_count.cmp(&self.token_count))
            .then_with(|| other.id.cmp(&self.id))
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

        if heap.len() < top_k {
            let score = cosine_similarity(query, &e.vector)?;
            let item = HeapItem {
                id: e.id,
                uuid: e.uuid,
                score,
                token_count: e.token_count,
            };
            heap.push(std::cmp::Reverse(item));
            continue;
        }

        let score = cosine_similarity(query, &e.vector)?;
        if let Some(std::cmp::Reverse(min)) = heap.peek() {
            let ordering = score
                .total_cmp(&min.score)
                .then_with(|| min.token_count.cmp(&e.token_count));
            if ordering == Ordering::Greater {
                heap.pop();
                let item = HeapItem {
                    id: e.id,
                    uuid: e.uuid,
                    score,
                    token_count: e.token_count,
                };
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

fn normalize_tags<'a>(tags: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();

    for tag in tags {
        let normalized = tag.trim().to_lowercase();
        if normalized.is_empty() {
            continue;
        }

        if seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }

    out
}

fn encrypt_tags_b64(user_key: &SecretKey, tags: &[String]) -> Vec<Option<String>> {
    tags.iter()
        .map(|tag| {
            let ciphertext = encrypt_key_deterministic(user_key, tag.as_bytes());
            Some(B64_STANDARD.encode(ciphertext))
        })
        .collect()
}

fn extract_tags_from_metadata(metadata: Option<&serde_json::Value>) -> Vec<String> {
    let Some(metadata) = metadata else {
        return Vec::new();
    };

    let Some(tags) = metadata.get("tags") else {
        return Vec::new();
    };

    match tags {
        serde_json::Value::Array(arr) => normalize_tags(arr.iter().filter_map(|v| v.as_str())),
        serde_json::Value::String(s) => normalize_tags(s.split(',')),
        _ => Vec::new(),
    }
}

#[derive(Debug, QueryableByName)]
struct StorageStats {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    row_count: i64,
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    stored_bytes: i64,
}

fn encrypted_embedding_bytes(
    vector_enc: &[u8],
    content_enc: &[u8],
    metadata_enc: Option<&Vec<u8>>,
) -> i64 {
    vector_enc
        .len()
        .saturating_add(content_enc.len())
        .saturating_add(metadata_enc.map_or(0, Vec::len)) as i64
}

fn load_user_embedding_storage_stats(
    conn: &mut diesel::PgConnection,
    user_id: Uuid,
) -> Result<StorageStats, ApiError> {
    diesel::sql_query(
        r#"
        SELECT
            COUNT(*)::BIGINT AS row_count,
            COALESCE(
                SUM(
                    octet_length(vector_enc)
                    + octet_length(content_enc)
                    + COALESCE(octet_length(metadata_enc), 0)
                ),
                0
            )::BIGINT AS stored_bytes
        FROM user_embeddings
        WHERE user_id = $1
        "#,
    )
    .bind::<diesel::sql_types::Uuid, _>(user_id)
    .get_result(conn)
    .map_err(|e| {
        error!(
            "Failed to load user embedding storage stats for user={}: {:?}",
            user_id, e
        );
        ApiError::InternalServerError
    })
}

fn load_project_embedding_storage_stats(
    conn: &mut diesel::PgConnection,
    project_id: i32,
) -> Result<StorageStats, ApiError> {
    diesel::sql_query(
        r#"
        SELECT
            COUNT(e.*)::BIGINT AS row_count,
            COALESCE(
                SUM(
                    octet_length(e.vector_enc)
                    + octet_length(e.content_enc)
                    + COALESCE(octet_length(e.metadata_enc), 0)
                ),
                0
            )::BIGINT AS stored_bytes
        FROM user_embeddings e
        INNER JOIN users u ON u.uuid = e.user_id
        WHERE u.project_id = $1
        "#,
    )
    .bind::<diesel::sql_types::Integer, _>(project_id)
    .get_result(conn)
    .map_err(|e| {
        error!(
            "Failed to load project embedding storage stats for project={}: {:?}",
            project_id, e
        );
        ApiError::InternalServerError
    })
}

fn ensure_storage_limits(
    conn: &mut diesel::PgConnection,
    user: &User,
    projected_new_bytes: i64,
    limits: RagLimits,
) -> Result<(), ApiError> {
    use diesel::dsl::count_star;

    let recent_insert_count: i64 = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user.uuid))
        .filter(user_embeddings::created_at.gt(Utc::now() - chrono::Duration::hours(1)))
        .select(count_star())
        .first(conn)
        .map_err(|e| {
            error!(
                "Failed to load recent embedding insert count for user={}: {:?}",
                user.uuid, e
            );
            ApiError::InternalServerError
        })?;
    if recent_insert_count >= limits.max_inserts_per_user_per_hour {
        warn!(
            target: "rag",
            user_id = %user.uuid,
            recent_insert_count,
            max_inserts_per_user_per_hour = limits.max_inserts_per_user_per_hour,
            "RAG per-user insert rate limit reached"
        );
        return Err(ApiError::BadRequest);
    }

    let user_stats = load_user_embedding_storage_stats(conn, user.uuid)?;
    if user_stats.row_count.saturating_add(1) > limits.max_user_embeddings
        || user_stats.stored_bytes.saturating_add(projected_new_bytes)
            > limits.max_user_stored_bytes
    {
        warn!(
            target: "rag",
            user_id = %user.uuid,
            rows = user_stats.row_count,
            stored_bytes = user_stats.stored_bytes,
            projected_new_bytes,
            max_rows = limits.max_user_embeddings,
            max_bytes = limits.max_user_stored_bytes,
            "RAG user storage limit reached"
        );
        return Err(ApiError::BadRequest);
    }

    let project_stats = load_project_embedding_storage_stats(conn, user.project_id)?;
    if project_stats.row_count.saturating_add(1) > limits.max_project_embeddings
        || project_stats
            .stored_bytes
            .saturating_add(projected_new_bytes)
            > limits.max_project_stored_bytes
    {
        warn!(
            target: "rag",
            project_id = user.project_id,
            rows = project_stats.row_count,
            stored_bytes = project_stats.stored_bytes,
            projected_new_bytes,
            max_rows = limits.max_project_embeddings,
            max_bytes = limits.max_project_stored_bytes,
            "RAG project storage limit reached"
        );
        return Err(ApiError::BadRequest);
    }

    Ok(())
}

fn cached_embedding_from_inserted(
    inserted: &crate::models::user_embeddings::UserEmbedding,
    vector: Vec<f32>,
) -> CachedEmbedding {
    CachedEmbedding {
        id: inserted.id,
        uuid: inserted.uuid,
        source_type: inserted.source_type.clone(),
        conversation_id: inserted.conversation_id,
        vector,
        token_count: inserted.token_count.max(0),
        created_at: inserted.created_at,
        updated_at: inserted.updated_at,
    }
}

pub async fn insert_archival_embedding(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    user_key: &SecretKey,
    text: &str,
    metadata: Option<&serde_json::Value>,
) -> Result<crate::models::user_embeddings::UserEmbedding, ApiError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(ApiError::BadRequest);
    }
    let limits = RagLimits::from_env();
    validate_text_size(text, limits.max_insert_text_bytes, "insert")?;

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

    let tags = extract_tags_from_metadata(metadata);
    let tags_enc = encrypt_tags_b64(user_key, &tags);
    let projected_new_bytes =
        encrypted_embedding_bytes(&vector_enc, &content_enc, metadata_enc.as_ref());

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;
    ensure_storage_limits(&mut conn, user, projected_new_bytes, limits)?;

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
        tags_enc,
        token_count,
    }
    .insert(&mut conn)
    .map_err(|e| {
        error!("Failed to insert archival embedding: {:?}", e);
        ApiError::InternalServerError
    })?;

    let cached = cached_embedding_from_inserted(&inserted, vector);
    let append_result = state.rag_cache.lock().await.append(user_id, cached);
    debug!(
        target: "rag",
        user_id = %user_id,
        embedding_uuid = %inserted.uuid,
        ?append_result,
        "rag_cache_append_after_insert"
    );
    Ok(inserted)
}

#[allow(dead_code)]
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
    let limits = RagLimits::from_env();
    validate_text_size(text, limits.max_insert_text_bytes, "insert")?;

    let (vector, token_count) = embed_text_via_tinfoil(state, user, auth_method, text).await?;

    let vector_bytes = serialize_f32_le(&vector);
    let vector_enc = encrypt_with_key(user_key, &vector_bytes).await;
    let content_enc = encrypt_with_key(user_key, text.as_bytes()).await;
    let projected_new_bytes = encrypted_embedding_bytes(&vector_enc, &content_enc, None);

    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;
    ensure_storage_limits(&mut conn, user, projected_new_bytes, limits)?;

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
        tags_enc: Vec::new(),
        token_count,
    }
    .insert(&mut conn)
    .map_err(|e| {
        error!("Failed to insert message embedding: {:?}", e);
        ApiError::InternalServerError
    })?;

    let cached = cached_embedding_from_inserted(&inserted, vector);
    let append_result = state.rag_cache.lock().await.append(user_id, cached);
    debug!(
        target: "rag",
        user_id = %user_id,
        embedding_uuid = %inserted.uuid,
        ?append_result,
        "rag_cache_append_after_insert"
    );
    Ok(inserted)
}

#[derive(Debug)]
struct LoadedEmbeddings {
    embeddings: Arc<Vec<CachedEmbedding>>,
    scan_limit_hit: bool,
    scanned_rows: usize,
    skipped_rows: usize,
    db_read_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct LoadFilters<'a> {
    source_types: Option<&'a [String]>,
    conversation_id: Option<i64>,
    tags_enc_filter: Option<&'a [Option<String>]>,
    begin_date: Option<DateTime<Utc>>,
    end_date: Option<DateTime<Utc>>,
}

impl LoadFilters<'_> {
    fn is_cacheable_broad_load(&self) -> bool {
        self.source_types.is_none()
            && self.conversation_id.is_none()
            && self.tags_enc_filter.is_none()
            && self.begin_date.is_none()
            && self.end_date.is_none()
    }
}

#[derive(Queryable)]
struct EmbeddingScanRow {
    id: i64,
    uuid: Uuid,
    source_type: String,
    conversation_id: Option<i64>,
    vector_enc: Vec<u8>,
    token_count: i32,
    vector_dim: i32,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

async fn load_user_embeddings_for_search(
    state: &AppState,
    user_id: Uuid,
    user_key: &SecretKey,
    filters: LoadFilters<'_>,
    scan_limit: i64,
) -> Result<LoadedEmbeddings, ApiError> {
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let started_at = Instant::now();
    let mut query = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user_id))
        .filter(user_embeddings::embedding_model.eq(DEFAULT_EMBEDDING_MODEL))
        .into_boxed();

    if let Some(source_types) = filters.source_types {
        query = query.filter(user_embeddings::source_type.eq_any(source_types));
    }

    if let Some(conversation_id) = filters.conversation_id {
        query = query.filter(user_embeddings::conversation_id.eq(Some(conversation_id)));
    }

    if let Some(tags_enc_filter) = filters.tags_enc_filter {
        query = query.filter(user_embeddings::tags_enc.overlaps_with(tags_enc_filter.to_vec()));
    }

    if let Some(begin_date) = filters.begin_date {
        query = query.filter(user_embeddings::created_at.ge(begin_date));
    }

    if let Some(end_date) = filters.end_date {
        query = query.filter(user_embeddings::created_at.le(end_date));
    }

    let mut rows: Vec<EmbeddingScanRow> = query
        .order((
            user_embeddings::created_at.desc(),
            user_embeddings::id.desc(),
        ))
        .select((
            user_embeddings::id,
            user_embeddings::uuid,
            user_embeddings::source_type,
            user_embeddings::conversation_id,
            user_embeddings::vector_enc,
            user_embeddings::token_count,
            user_embeddings::vector_dim,
            user_embeddings::created_at,
            user_embeddings::updated_at,
        ))
        .limit(scan_limit.saturating_add(1))
        .load(&mut conn)
        .map_err(|e| {
            error!("Failed to load embeddings for user={}: {:?}", user_id, e);
            ApiError::InternalServerError
        })?;

    let scan_limit_hit = rows.len() as i64 > scan_limit;
    if scan_limit_hit {
        rows.truncate(scan_limit as usize);
    }

    let db_read_bytes = rows.iter().map(|row| row.vector_enc.len()).sum();
    let scanned_rows = rows.len();
    let mut skipped_rows = 0usize;
    let mut out: Vec<CachedEmbedding> = Vec::with_capacity(rows.len());
    let decrypt_started_at = Instant::now();

    for row in rows {
        let vector_bytes = match decrypt_with_key(user_key, &row.vector_enc) {
            Ok(bytes) => bytes,
            Err(_) => {
                skipped_rows = skipped_rows.saturating_add(1);
                warn!(
                    target: "rag",
                    user_id = %user_id,
                    embedding_id = row.id,
                    embedding_uuid = %row.uuid,
                    reason = "vector_decrypt",
                    "Skipping corrupt RAG row"
                );
                continue;
            }
        };

        let vector = match deserialize_f32_le(&vector_bytes) {
            Ok(vector) => vector,
            Err(_) => {
                skipped_rows = skipped_rows.saturating_add(1);
                warn!(
                    target: "rag",
                    user_id = %user_id,
                    embedding_id = row.id,
                    embedding_uuid = %row.uuid,
                    reason = "vector_deserialize",
                    "Skipping corrupt RAG row"
                );
                continue;
            }
        };

        if vector.len() != row.vector_dim as usize {
            skipped_rows = skipped_rows.saturating_add(1);
            warn!(
                target: "rag",
                user_id = %user_id,
                embedding_id = row.id,
                embedding_uuid = %row.uuid,
                expected_dim = row.vector_dim,
                actual_dim = vector.len(),
                reason = "dimension_mismatch",
                "Skipping corrupt RAG row"
            );
            continue;
        }

        out.push(CachedEmbedding {
            id: row.id,
            uuid: row.uuid,
            source_type: row.source_type,
            conversation_id: row.conversation_id,
            vector,
            token_count: row.token_count.max(0),
            created_at: row.created_at,
            updated_at: row.updated_at,
        });
    }

    info!(
        target: "rag",
        user_id = %user_id,
        cacheable = filters.is_cacheable_broad_load(),
        scanned_rows,
        loaded_rows = out.len(),
        skipped_rows,
        db_read_bytes,
        scan_limit,
        scan_limit_hit,
        db_and_decrypt_ms = started_at.elapsed().as_millis() as u64,
        decrypt_ms = decrypt_started_at.elapsed().as_millis() as u64,
        "rag_load_embeddings"
    );

    Ok(LoadedEmbeddings {
        embeddings: Arc::new(out),
        scan_limit_hit,
        scanned_rows,
        skipped_rows,
        db_read_bytes,
    })
}

#[derive(Queryable)]
struct ContentFetchRow {
    id: i64,
    uuid: Uuid,
    content_enc: Vec<u8>,
}

fn overfetch_limit(limit: usize) -> usize {
    limit
        .saturating_mul(FINAL_RESULT_OVERFETCH_MULTIPLIER)
        .saturating_add(FINAL_RESULT_OVERFETCH_MIN_EXTRA)
        .clamp(limit, FINAL_RESULT_OVERFETCH_MAX)
}

async fn load_cacheable_embeddings_with_coordination(
    state: &Arc<AppState>,
    user_id: Uuid,
    user_key: &SecretKey,
    limits: RagLimits,
) -> Result<LoadedEmbeddings, ApiError> {
    loop {
        let permit = {
            let mut cache = state.rag_cache.lock().await;
            if let Some((embeddings, scan_limit_hit)) = cache.get(user_id) {
                info!(
                    target: "rag",
                    user_id = %user_id,
                    cache_hit = true,
                    loaded_rows = embeddings.len(),
                    cache_bytes_per_user = cached_embeddings_bytes(&embeddings),
                    total_cache_bytes = cache.total_bytes,
                    scan_limit_hit,
                    "rag_cache_lookup"
                );
                return Ok(LoadedEmbeddings {
                    embeddings,
                    scan_limit_hit,
                    scanned_rows: 0,
                    skipped_rows: 0,
                    db_read_bytes: 0,
                });
            }

            info!(
                target: "rag",
                user_id = %user_id,
                cache_hit = false,
                total_cache_bytes = cache.total_bytes,
                "rag_cache_lookup"
            );
            cache.begin_load(user_id)
        };

        match permit {
            CacheLoadPermit::Start => {
                let filters = LoadFilters {
                    source_types: None,
                    conversation_id: None,
                    tags_enc_filter: None,
                    begin_date: None,
                    end_date: None,
                };
                let loaded = load_user_embeddings_for_search(
                    state,
                    user_id,
                    user_key,
                    filters,
                    limits.scan_limit,
                )
                .await;

                let mut cache = state.rag_cache.lock().await;
                match &loaded {
                    Ok(loaded) => {
                        let cached =
                            cache.put(user_id, loaded.embeddings.clone(), loaded.scan_limit_hit);
                        info!(
                            target: "rag",
                            user_id = %user_id,
                            cached,
                            loaded_rows = loaded.embeddings.len(),
                            cache_bytes_per_user = cached_embeddings_bytes(&loaded.embeddings),
                            total_cache_bytes = cache.total_bytes,
                            scan_limit_hit = loaded.scan_limit_hit,
                            "rag_cache_store_after_load"
                        );
                    }
                    Err(_) => {
                        warn!(
                            target: "rag",
                            user_id = %user_id,
                            "RAG cacheable load failed"
                        );
                    }
                }
                cache.finish_load(user_id);
                return loaded;
            }
            CacheLoadPermit::Wait(notify) => {
                if timeout(CACHE_LOAD_WAIT_TIMEOUT, notify.notified())
                    .await
                    .is_ok()
                {
                    continue;
                }

                let should_duplicate = {
                    let mut cache = state.rag_cache.lock().await;
                    cache.try_start_timeout_duplicate(user_id)
                };

                warn!(
                    target: "rag",
                    user_id = %user_id,
                    should_duplicate,
                    wait_timeout_secs = CACHE_LOAD_WAIT_TIMEOUT.as_secs(),
                    "RAG cache load wait timed out"
                );

                if !should_duplicate {
                    sleep(CACHE_LOAD_TIMEOUT_BACKOFF).await;
                    continue;
                }

                let filters = LoadFilters {
                    source_types: None,
                    conversation_id: None,
                    tags_enc_filter: None,
                    begin_date: None,
                    end_date: None,
                };
                let loaded = load_user_embeddings_for_search(
                    state,
                    user_id,
                    user_key,
                    filters,
                    limits.scan_limit,
                )
                .await?;

                let mut cache = state.rag_cache.lock().await;
                let cached = cache.put(user_id, loaded.embeddings.clone(), loaded.scan_limit_hit);
                cache.finish_load(user_id);
                info!(
                    target: "rag",
                    user_id = %user_id,
                    cached,
                    loaded_rows = loaded.embeddings.len(),
                    cache_bytes_per_user = cached_embeddings_bytes(&loaded.embeddings),
                    total_cache_bytes = cache.total_bytes,
                    scan_limit_hit = loaded.scan_limit_hit,
                    reason = "timeout_duplicate",
                    "rag_cache_store_after_load"
                );
                return Ok(loaded);
            }
        }
    }
}

async fn fetch_ranked_content(
    state: &AppState,
    user_id: Uuid,
    user_key: &SecretKey,
    candidates: Vec<HeapItem>,
    limit: usize,
) -> Result<(Vec<RagSearchResult>, usize), ApiError> {
    if candidates.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let ids: Vec<i64> = candidates.iter().map(|candidate| candidate.id).collect();
    let mut conn = state
        .db
        .get_pool()
        .get()
        .map_err(|_| ApiError::InternalServerError)?;

    let rows: Vec<ContentFetchRow> = user_embeddings::table
        .filter(user_embeddings::user_id.eq(user_id))
        .filter(user_embeddings::id.eq_any(&ids))
        .select((
            user_embeddings::id,
            user_embeddings::uuid,
            user_embeddings::content_enc,
        ))
        .load(&mut conn)
        .map_err(|e| {
            error!(
                "Failed to fetch RAG top-k content for user={}: {:?}",
                user_id, e
            );
            ApiError::InternalServerError
        })?;

    let content_by_id: HashMap<i64, ContentFetchRow> =
        rows.into_iter().map(|row| (row.id, row)).collect();
    let mut results: Vec<RagSearchResult> = Vec::with_capacity(limit);
    let mut skipped_rows = 0usize;

    for candidate in candidates {
        let Some(row) = content_by_id.get(&candidate.id) else {
            skipped_rows = skipped_rows.saturating_add(1);
            warn!(
                target: "rag",
                user_id = %user_id,
                embedding_id = candidate.id,
                embedding_uuid = %candidate.uuid,
                reason = "content_missing",
                "Skipping RAG candidate"
            );
            continue;
        };

        let plaintext = match decrypt_with_key(user_key, &row.content_enc) {
            Ok(plaintext) => plaintext,
            Err(_) => {
                skipped_rows = skipped_rows.saturating_add(1);
                warn!(
                    target: "rag",
                    user_id = %user_id,
                    embedding_id = candidate.id,
                    embedding_uuid = %row.uuid,
                    reason = "content_decrypt",
                    "Skipping corrupt RAG candidate"
                );
                continue;
            }
        };

        let content = match String::from_utf8(plaintext) {
            Ok(content) => content,
            Err(_) => {
                skipped_rows = skipped_rows.saturating_add(1);
                warn!(
                    target: "rag",
                    user_id = %user_id,
                    embedding_id = candidate.id,
                    embedding_uuid = %row.uuid,
                    reason = "content_utf8",
                    "Skipping corrupt RAG candidate"
                );
                continue;
            }
        };

        results.push(RagSearchResult {
            content,
            score: candidate.score,
            token_count: candidate.token_count,
        });

        if results.len() >= limit {
            break;
        }
    }

    Ok((results, skipped_rows))
}

pub async fn search_user_embeddings_with_options(
    state: &Arc<AppState>,
    user: &User,
    auth_method: AuthMethod,
    user_key: &SecretKey,
    query: &str,
    options: RagSearchOptions,
) -> Result<RagSearchOutcome, ApiError> {
    let started_at = Instant::now();
    let limit = options.limit.clamp(1, 20);
    let candidate_limit = overfetch_limit(limit);
    let limits = RagLimits::from_env();
    let query = query.trim();
    if query.is_empty() {
        return Err(ApiError::BadRequest);
    }
    validate_text_size(query, limits.max_search_query_bytes, "search")?;

    let user_id = user.uuid;

    let (query_vec, _query_tokens) =
        embed_text_via_tinfoil(state, user, auth_method, query).await?;

    let tags_enc_filter = options
        .filters
        .tags
        .as_ref()
        .map(|t| normalize_tags(t.iter().map(|s| s.as_str())))
        .filter(|t| !t.is_empty())
        .map(|t| encrypt_tags_b64(user_key, &t));

    let load_filters = LoadFilters {
        source_types: options.filters.source_types.as_deref(),
        conversation_id: options.filters.conversation_id,
        tags_enc_filter: tags_enc_filter.as_deref(),
        begin_date: options.filters.begin_date,
        end_date: options.filters.end_date,
    };

    let loaded = if load_filters.is_cacheable_broad_load() {
        load_cacheable_embeddings_with_coordination(state, user_id, user_key, limits).await?
    } else {
        load_user_embeddings_for_search(state, user_id, user_key, load_filters, limits.scan_limit)
            .await?
    };

    let score_started_at = Instant::now();
    let candidates = top_k_candidates(
        &query_vec,
        &loaded.embeddings,
        candidate_limit,
        options.filters.source_types.as_deref(),
        options.filters.conversation_id,
    )?;

    let score_ms = score_started_at.elapsed().as_millis() as u64;
    let content_started_at = Instant::now();
    let (mut results, content_skipped_rows) =
        fetch_ranked_content(state, user_id, user_key, candidates, limit).await?;
    let content_ms = content_started_at.elapsed().as_millis() as u64;

    if let Some(budget) = options.max_tokens {
        results = apply_token_budget(results, budget);
    }

    let feedback = if loaded.scan_limit_hit {
        Some(format!(
            "RAG search reached the internal scan limit of {} candidate rows. Older or out-of-window matches may exist; retry with begin_date/end_date for a narrower time range.",
            limits.scan_limit
        ))
    } else {
        None
    };

    let skipped_rows = loaded.skipped_rows.saturating_add(content_skipped_rows);
    info!(
        target: "rag",
        user_id = %user_id,
        returned_results = results.len(),
        scanned_rows = loaded.scanned_rows,
        skipped_rows,
        db_read_bytes = loaded.db_read_bytes,
        scan_limit = limits.scan_limit,
        scan_limit_hit = loaded.scan_limit_hit,
        score_ms,
        content_fetch_ms = content_ms,
        total_search_ms = started_at.elapsed().as_millis() as u64,
        "rag_search_complete"
    );

    Ok(RagSearchOutcome {
        results,
        feedback,
        scan_limit_hit: loaded.scan_limit_hit,
        scanned_rows: loaded.scanned_rows,
        skipped_rows,
    })
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
    tags: Option<&[String]>,
) -> Result<Vec<RagSearchResult>, ApiError> {
    let options = RagSearchOptions {
        limit: top_k,
        max_tokens,
        filters: RagSearchFilters {
            source_types: source_types.map(|s| s.to_vec()),
            conversation_id,
            tags: tags.map(|t| t.to_vec()),
            begin_date: None,
            end_date: None,
        },
    };
    let outcome =
        search_user_embeddings_with_options(state, user, auth_method, user_key, query, options)
            .await?;
    Ok(outcome.results)
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

    state
        .rag_cache
        .lock()
        .await
        .remove_embedding_by_uuid(user_id, embedding_uuid);
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

    fn cached_embedding(
        id: i64,
        source_type: &str,
        conversation_id: Option<i64>,
        vector: Vec<f32>,
        token_count: i32,
    ) -> CachedEmbedding {
        CachedEmbedding {
            id,
            uuid: Uuid::new_v4(),
            source_type: source_type.to_string(),
            conversation_id,
            vector,
            token_count,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

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
            cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5),
            cached_embedding(2, SOURCE_TYPE_MESSAGE, Some(123), vec![0.0, 1.0], 7),
            cached_embedding(3, SOURCE_TYPE_MESSAGE, Some(123), vec![0.8, 0.2], 9),
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
        assert_eq!(items[0].id, 3);
        assert_eq!(items[1].id, 2);
    }

    #[test]
    fn top_k_tie_break_prefers_fewer_tokens() {
        let query = vec![1.0f32, 0.0];

        let embeddings = vec![
            cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 10),
            cached_embedding(2, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5),
        ];

        let items = top_k_candidates(&query, &embeddings, 1, None, None).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, 2);
    }

    #[test]
    fn top_k_when_k_gt_embeddings_returns_all() {
        let query = vec![1.0f32, 0.0];

        let embeddings = vec![
            cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5),
            cached_embedding(2, SOURCE_TYPE_MESSAGE, Some(123), vec![0.0, 1.0], 7),
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
        let mut cache = RagCache::new(1024 * 1024, 1024 * 1024, Duration::from_secs(60));
        let user_id = Uuid::new_v4();

        cache.put(user_id, Arc::new(vec![]), false);
        assert!(cache.entries.contains_key(&user_id));

        cache.evict_user(user_id);
        assert!(!cache.entries.contains_key(&user_id));
        assert!(!cache.lru.contains(&user_id));
        assert!(cache.get(user_id).is_none());
    }

    #[tokio::test]
    async fn rag_cache_byte_lru_eviction() {
        let row = cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5);
        let row_bytes = row.estimated_cache_bytes();
        let mut cache = RagCache::new(row_bytes * 2, row_bytes * 10, Duration::from_secs(60));

        let v1 = Arc::new(vec![row.clone()]);
        let v2 = Arc::new(vec![row.clone()]);
        let v3 = Arc::new(vec![row]);

        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let u3 = Uuid::new_v4();

        assert!(cache.put(u1, v1, false));
        assert!(cache.put(u2, v2, false));
        // touch u1 so u2 becomes LRU
        cache.get(u1);
        assert!(cache.put(u3, v3, false));

        assert!(cache.entries.contains_key(&u1));
        assert!(!cache.entries.contains_key(&u2));
        assert!(cache.entries.contains_key(&u3));
    }

    #[tokio::test]
    async fn rag_cache_ttl_expiration() {
        let mut cache = RagCache::new(1024 * 1024, 1024 * 1024, Duration::from_millis(5));
        let user = Uuid::new_v4();

        cache.put(user, Arc::new(vec![]), false);
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(cache.get(user).is_none());
    }

    #[test]
    fn rag_cache_append_updates_present_entry() {
        let mut cache = RagCache::new(1024 * 1024, 1024 * 1024, Duration::from_secs(60));
        let user = Uuid::new_v4();
        let first = cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5);
        let second = cached_embedding(2, SOURCE_TYPE_ARCHIVAL, None, vec![0.0, 1.0], 7);

        assert!(cache.put(user, Arc::new(vec![first]), false));
        assert_eq!(cache.append(user, second), CacheAppendResult::Appended);

        let (cached, _) = cache.get(user).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[1].id, 2);
    }

    #[test]
    fn rag_cache_append_evicts_when_over_user_byte_cap() {
        let first = cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5);
        let second = cached_embedding(2, SOURCE_TYPE_ARCHIVAL, None, vec![0.0, 1.0], 7);
        let row_bytes = first.estimated_cache_bytes();
        let mut cache = RagCache::new(row_bytes * 10, row_bytes + 1, Duration::from_secs(60));
        let user = Uuid::new_v4();

        assert!(cache.put(user, Arc::new(vec![first]), false));
        assert_eq!(
            cache.append(user, second),
            CacheAppendResult::EvictedOverLimit
        );
        assert!(cache.get(user).is_none());
    }

    #[test]
    fn rag_cache_remove_embedding_by_uuid_updates_entry() {
        let mut cache = RagCache::new(1024 * 1024, 1024 * 1024, Duration::from_secs(60));
        let user = Uuid::new_v4();
        let first = cached_embedding(1, SOURCE_TYPE_ARCHIVAL, None, vec![1.0, 0.0], 5);
        let second = cached_embedding(2, SOURCE_TYPE_ARCHIVAL, None, vec![0.0, 1.0], 7);
        let second_uuid = second.uuid;

        assert!(cache.put(user, Arc::new(vec![first, second]), false));
        cache.remove_embedding_by_uuid(user, second_uuid);

        let (cached, _) = cache.get(user).unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, 1);
    }
}
