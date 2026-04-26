//! Central model-specific configuration.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    pub context_window: usize,
    pub responses: ResponsesModelConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponsesModelConfig {
    pub sampling: SamplingConfig,
    pub include_reasoning: bool,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
}

impl SamplingConfig {
    pub fn with_overrides(self, temperature: Option<f32>, top_p: Option<f32>) -> Self {
        Self {
            temperature: temperature.unwrap_or(self.temperature),
            top_p: top_p.unwrap_or(self.top_p),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelConfigEntry {
    prefix: &'static str,
    config: ModelConfig,
}

pub const DEFAULT_CONTEXT_WINDOW: usize = 64_000;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 1.0;

const DEFAULT_SAMPLING_CONFIG: SamplingConfig = SamplingConfig {
    temperature: DEFAULT_TEMPERATURE,
    top_p: DEFAULT_TOP_P,
};

const DEFAULT_RESPONSES_MODEL_CONFIG: ResponsesModelConfig = ResponsesModelConfig {
    sampling: DEFAULT_SAMPLING_CONFIG,
    include_reasoning: false,
    enable_thinking: false,
};

impl ModelConfig {
    const fn new(context_window: usize) -> Self {
        Self {
            context_window,
            responses: DEFAULT_RESPONSES_MODEL_CONFIG,
        }
    }

    const fn with_responses(context_window: usize, responses: ResponsesModelConfig) -> Self {
        Self {
            context_window,
            responses,
        }
    }
}

impl ModelConfigEntry {
    const fn new(prefix: &'static str, context_window: usize) -> Self {
        Self {
            prefix,
            config: ModelConfig::new(context_window),
        }
    }

    const fn with_responses(
        prefix: &'static str,
        context_window: usize,
        responses: ResponsesModelConfig,
    ) -> Self {
        Self {
            prefix,
            config: ModelConfig::with_responses(context_window, responses),
        }
    }
}

const DEFAULT_MODEL_CONFIG: ModelConfig = ModelConfig::new(DEFAULT_CONTEXT_WINDOW);

const GEMMA4_RESPONSES_MODEL_CONFIG: ResponsesModelConfig = ResponsesModelConfig {
    sampling: DEFAULT_SAMPLING_CONFIG,
    include_reasoning: true,
    enable_thinking: true,
};

const DEEPSEEK_V4_PRO_RESPONSES_MODEL_CONFIG: ResponsesModelConfig = ResponsesModelConfig {
    sampling: SamplingConfig {
        temperature: 1.0,
        top_p: 1.0,
    },
    include_reasoning: false,
    enable_thinking: false,
};

const MODEL_CONFIGS: &[ModelConfigEntry] = &[
    ModelConfigEntry::new("llama3-3-70b", 128_000),
    ModelConfigEntry::new("gpt-oss-120b", 128_000),
    ModelConfigEntry::new("qwen3-vl-30b", 256_000),
    ModelConfigEntry::new("kimi-k2-5", 256_000),
    ModelConfigEntry::new("kimi-k2-6", 256_000),
    ModelConfigEntry::with_responses("gemma4-31b", 256_000, GEMMA4_RESPONSES_MODEL_CONFIG),
    ModelConfigEntry::new("glm-5-1", 202_000),
    ModelConfigEntry::with_responses(
        "deepseek-v4-pro",
        800_000,
        DEEPSEEK_V4_PRO_RESPONSES_MODEL_CONFIG,
    ),
    ModelConfigEntry::new("deepseek-r1-0528", 128_000),
    ModelConfigEntry::new("gemma-3-27b", 20_000),
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
        assert_eq!(model_context_window("deepseek-v4-pro"), 800_000);
    }

    #[test]
    fn test_existing_models_use_default_sampling_config() {
        for model in [
            "gemma-3-27b",
            "deepseek-r1-0528",
            "llama3-3-70b",
            "gpt-oss-120b",
            "qwen3-vl-30b",
            "kimi-k2-5",
            "kimi-k2-6",
            "gemma4-31b",
            "glm-5-1",
        ] {
            let config = model_config(model);

            assert_eq!(config.responses.sampling.temperature, DEFAULT_TEMPERATURE);
            assert_eq!(config.responses.sampling.top_p, DEFAULT_TOP_P);
        }
    }

    #[test]
    fn test_gemma4_responses_config_enables_thinking() {
        let responses_config = model_config("gemma4-31b").responses;

        assert!(responses_config.include_reasoning);
        assert!(responses_config.enable_thinking);
    }

    #[test]
    fn test_deepseek_v4_pro_responses_config() {
        let config = model_config("deepseek-v4-pro");

        assert_eq!(config.context_window, 800_000);
        assert_eq!(config.responses.sampling.temperature, 1.0);
        assert_eq!(config.responses.sampling.top_p, 1.0);
    }

    #[test]
    fn test_sampling_config_applies_overrides() {
        let sampling = DEFAULT_SAMPLING_CONFIG.with_overrides(Some(0.5), None);

        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_p, DEFAULT_TOP_P);
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
