//! Thin wrapper around tiktoken for fast, cached token counting.

use once_cell::sync::Lazy;
use tiktoken_rs::cl100k_base;

/// Cached encoder – created once per process.
/// No mutex needed: CoreBPE is immutable and thread-safe.
static ENCODER: Lazy<tiktoken_rs::CoreBPE> =
    Lazy::new(|| cl100k_base().expect("init cl100k encoder"));

/// Count tokens for a piece of UTF‑8 text.
pub fn count_tokens(text: &str) -> usize {
    ENCODER.encode_with_special_tokens(text).len()
}

/// Per‑model context windows (hard limits).
/// Any unknown model defaults to 64k tokens.
pub fn model_max_ctx(model: &str) -> usize {
    const LIMITS: &[(&str, usize)] = &[
        // Canonical names
        ("llama-3.3-70b", 128_000),
        ("gpt-oss-120b", 128_000),
        ("qwen3-coder-480b", 128_000),
        ("qwen3-vl-30b", 256_000), // Vision-language model
        // Provider-specific equivalents
        ("llama3-3-70b", 128_000), // Tinfoil alias
        // Chat models
        ("deepseek-r1-0528", 128_000),
        ("deepseek-v31-terminus", 128_000),
        // Gemma 3 27B (vision) — capped at 20k
        ("leon-se/gemma-3-27b-it-fp8-dynamic", 20_000),
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
        assert_eq!(model_max_ctx("leon-se/gemma-3-27b-it-fp8-dynamic"), 20_000);
        assert_eq!(model_max_ctx("deepseek-r1-0528"), 128_000);
        assert_eq!(model_max_ctx("deepseek-v31-terminus"), 128_000);

        // Canonical and tinfoil aliases
        assert_eq!(model_max_ctx("llama-3.3-70b"), 128_000);
        assert_eq!(model_max_ctx("llama3-3-70b"), 128_000);
        assert_eq!(model_max_ctx("gpt-oss-120b"), 128_000);
        assert_eq!(model_max_ctx("qwen3-coder-480b"), 128_000);
        assert_eq!(model_max_ctx("qwen3-vl-30b"), 256_000);
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
    }
}
