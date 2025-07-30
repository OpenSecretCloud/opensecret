//! Thin wrapper around tiktoken for fast, cached token counting.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use tiktoken_rs::cl100k_base;

/// Cached encoder – created once per process.
static ENCODER: Lazy<Mutex<tiktoken_rs::CoreBPE>> =
    Lazy::new(|| Mutex::new(cl100k_base().expect("init cl100k encoder")));

/// Count tokens for a piece of UTF‑8 text.
pub fn count_tokens(text: &str) -> usize {
    ENCODER
        .lock()
        .expect("encoder lock")
        .encode_with_special_tokens(text)
        .len()
}

/// Per‑model context windows (hard limits).  
/// Any unknown model defaults to 64 k tokens.
pub fn model_max_ctx(model: &str) -> usize {
    const LIMITS: &[(&str, usize)] = &[
        // Free Tier
        ("ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 70_000), // Llama 3.3 70B
        
        // Starter Tier
        ("leon-se/gemma-3-27b-it-fp8-dynamic", 70_000), // Gemma 3 27B (vision)
        
        // Pro Tier
        ("deepseek-r1-70b", 64_000), // DeepSeek R1 70B
        ("mistral-small-3-1-24b", 128_000), // Mistral Small 3.1 24B (vision)
        ("qwen2-5-72b", 128_000), // Qwen 2.5 72B
    ];

    LIMITS
        .iter()
        .find(|(prefix, _)| model.starts_with(*prefix))
        .map(|(_, l)| *l)
        .unwrap_or(64_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_basic() {
        // Basic English text
        assert_eq!(count_tokens("Hello, world!"), 4); // "Hello", ",", " world", "!"
        
        // Empty string
        assert_eq!(count_tokens(""), 0);
        
        // Single token
        assert_eq!(count_tokens("Hello"), 1);
    }

    #[test]
    fn test_count_tokens_consistency() {
        // Same text should always produce same count
        let text = "The quick brown fox jumps over the lazy dog";
        let count1 = count_tokens(text);
        let count2 = count_tokens(text);
        assert_eq!(count1, count2);
    }

    #[test]
    fn test_model_max_ctx_known_models() {
        // Test known models
        assert_eq!(model_max_ctx("ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"), 70_000);
        assert_eq!(model_max_ctx("leon-se/gemma-3-27b-it-fp8-dynamic"), 70_000);
        assert_eq!(model_max_ctx("deepseek-r1-70b"), 64_000);
        assert_eq!(model_max_ctx("mistral-small-3-1-24b"), 128_000);
        assert_eq!(model_max_ctx("qwen2-5-72b"), 128_000);
    }

    #[test]
    fn test_model_max_ctx_unknown_models() {
        // Unknown models should default to 64k
        assert_eq!(model_max_ctx("gpt-4"), 64_000);
        assert_eq!(model_max_ctx("claude-3"), 64_000);
        assert_eq!(model_max_ctx("unknown-model-xyz"), 64_000);
        assert_eq!(model_max_ctx(""), 64_000);
    }

    #[test]
    fn test_model_max_ctx_prefix_matching() {
        // Should match by prefix
        assert_eq!(model_max_ctx("deepseek-r1-70b-instruct"), 64_000);
        assert_eq!(model_max_ctx("qwen2-5-72b-chat"), 128_000);
    }
}