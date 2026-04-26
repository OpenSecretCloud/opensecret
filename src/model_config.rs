//! Central model-specific configuration.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub context_window: usize,
}

#[derive(Debug, Clone, Copy)]
struct ModelConfigEntry {
    prefix: &'static str,
    config: ModelConfig,
}

pub const DEFAULT_CONTEXT_WINDOW: usize = 64_000;

const DEFAULT_MODEL_CONFIG: ModelConfig = ModelConfig {
    context_window: DEFAULT_CONTEXT_WINDOW,
};

const MODEL_CONFIGS: &[ModelConfigEntry] = &[
    ModelConfigEntry {
        prefix: "llama3-3-70b",
        config: ModelConfig {
            context_window: 128_000,
        },
    },
    ModelConfigEntry {
        prefix: "gpt-oss-120b",
        config: ModelConfig {
            context_window: 128_000,
        },
    },
    ModelConfigEntry {
        prefix: "qwen3-vl-30b",
        config: ModelConfig {
            context_window: 256_000,
        },
    },
    ModelConfigEntry {
        prefix: "kimi-k2-5",
        config: ModelConfig {
            context_window: 256_000,
        },
    },
    ModelConfigEntry {
        prefix: "kimi-k2-6",
        config: ModelConfig {
            context_window: 256_000,
        },
    },
    ModelConfigEntry {
        prefix: "gemma4-31b",
        config: ModelConfig {
            context_window: 256_000,
        },
    },
    ModelConfigEntry {
        prefix: "glm-5-1",
        config: ModelConfig {
            context_window: 202_000,
        },
    },
    ModelConfigEntry {
        prefix: "deepseek-r1-0528",
        config: ModelConfig {
            context_window: 128_000,
        },
    },
    ModelConfigEntry {
        prefix: "gemma-3-27b",
        config: ModelConfig {
            context_window: 20_000,
        },
    },
];

pub fn model_config(model: &str) -> ModelConfig {
    MODEL_CONFIGS
        .iter()
        .find(|entry| model.starts_with(entry.prefix))
        .map(|entry| entry.config)
        .unwrap_or(DEFAULT_MODEL_CONFIG)
}

pub fn model_context_window(model: &str) -> usize {
    model_config(model).context_window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_context_window_known_models() {
        assert_eq!(model_context_window("gemma-3-27b"), 20_000);
        assert_eq!(model_context_window("deepseek-r1-0528"), 128_000);
        assert_eq!(model_context_window("llama3-3-70b"), 128_000);
        assert_eq!(model_context_window("gpt-oss-120b"), 128_000);
        assert_eq!(model_context_window("qwen3-vl-30b"), 256_000);
        assert_eq!(model_context_window("kimi-k2-5"), 256_000);
        assert_eq!(model_context_window("kimi-k2-6"), 256_000);
        assert_eq!(model_context_window("gemma4-31b"), 256_000);
        assert_eq!(model_context_window("glm-5-1"), 202_000);
    }

    #[test]
    fn test_model_context_window_unknown_models() {
        assert_eq!(model_context_window("gpt-4"), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(model_context_window("claude-3"), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(
            model_context_window("unknown-model-xyz"),
            DEFAULT_CONTEXT_WINDOW
        );
        assert_eq!(model_context_window(""), DEFAULT_CONTEXT_WINDOW);
    }

    #[test]
    fn test_model_config_prefix_matching() {
        assert_eq!(model_context_window("deepseek-r1-0528-instruct"), 128_000);
        assert_eq!(
            model_context_window("deepseek-r1-70b-instruct"),
            DEFAULT_CONTEXT_WINDOW
        );
    }
}
