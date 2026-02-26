use crate::ApiError;
use dspy_rs::adapter::chat::ChatAdapter;
use dspy_rs::LM;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::error;

/// Instruction for summarization DSRs signature
pub const SUMMARY_INSTRUCTION: &str = r#"You are a conversation summarizer. Your job is to create a concise summary that allows an AI agent to resume a conversation without disruption, even after older messages are replaced with this summary.

Your summary should be structured and actionable. Include:
1. Task/Conversation overview: What is the user working on? Key clarifications or constraints?
2. Current State: What has been completed or discussed? Any files/resources referenced?
3. Next Steps: What would logically come next in this conversation?

Keep your summary under 100 words. Be specific and preserve key details like names, preferences, and decisions made."#;

/// Instruction for correction DSRs signature
pub const SUMMARY_CORRECTION_INSTRUCTION: &str = r#"You are a correction agent. The summarizer produced a malformed response that couldn't be parsed. Your job is to extract the summary from the malformed response and return it in the correct format.

Preserve the original intent and content - do NOT generate new content. Just reshape the malformed response into the expected output format."#;

/// DSRs signature for conversation summarization
#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct SummarizeConversation {
    #[input(desc = "Previous summary to build upon (empty if first summarization)")]
    pub previous_summary: String,

    #[input(desc = "New conversation messages to incorporate into the summary")]
    pub new_messages: String,

    #[output(desc = "Updated summary incorporating all context (100 word limit)")]
    pub summary: String,
}

/// DSRs signature for correcting malformed summarization responses
#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct SummarizationCorrection {
    #[input(desc = "The previous summary that was being built upon")]
    pub previous_summary: String,

    #[input(desc = "The new messages that were being summarized")]
    pub new_messages: String,

    #[input(desc = "The malformed response that needs correction")]
    pub malformed_response: String,

    #[input(desc = "The error message explaining what went wrong")]
    pub error_message: String,

    #[output(desc = "Corrected summary (100 word limit)")]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummarizeConversationOutput {
    pub summary: String,
}

pub struct CompactionManager {
    max_retries: usize,
}

impl CompactionManager {
    pub fn new() -> Self {
        Self { max_retries: 2 }
    }

    pub fn should_compact(&self, current_tokens: usize, max_tokens: usize, threshold: f32) -> bool {
        current_tokens > ((max_tokens as f32 * threshold) as usize)
    }

    pub async fn summarize(
        &self,
        lm: &Arc<LM>,
        previous_summary: &str,
        new_messages: &str,
    ) -> Result<String, ApiError> {
        let input = SummarizeConversationInput {
            previous_summary: previous_summary.to_string(),
            new_messages: new_messages.to_string(),
        };

        // First attempt
        match call_summarize_conversation(lm, &input).await {
            Ok(out) => return Ok(out.summary),
            Err(SummarizationError::Parse {
                raw_response,
                error_message,
            }) => {
                if let Ok(corrected) = self
                    .try_correction(
                        lm,
                        previous_summary,
                        new_messages,
                        &raw_response,
                        &error_message,
                    )
                    .await
                {
                    return Ok(corrected);
                }
            }
            Err(SummarizationError::Api(e)) => return Err(e),
        };

        for _attempt in 1..=self.max_retries {
            match call_summarize_conversation(lm, &input).await {
                Ok(out) => return Ok(out.summary),
                Err(SummarizationError::Parse {
                    raw_response,
                    error_message,
                }) => {
                    if let Ok(corrected) = self
                        .try_correction(
                            lm,
                            previous_summary,
                            new_messages,
                            &raw_response,
                            &error_message,
                        )
                        .await
                    {
                        return Ok(corrected);
                    }
                }
                Err(SummarizationError::Api(e)) => {
                    return Err(e);
                }
            };
        }

        Err(ApiError::InternalServerError)
    }

    async fn try_correction(
        &self,
        lm: &Arc<LM>,
        previous_summary: &str,
        new_messages: &str,
        raw_response: &str,
        error_message: &str,
    ) -> Result<String, ApiError> {
        let input = SummarizationCorrectionInput {
            previous_summary: previous_summary.to_string(),
            new_messages: new_messages.to_string(),
            malformed_response: raw_response.to_string(),
            error_message: error_message.to_string(),
        };

        let out = call_summarization_correction(lm, &input).await?;
        Ok(out.summary)
    }
}

impl Default for CompactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum SummarizationError {
    Api(ApiError),
    Parse {
        raw_response: String,
        error_message: String,
    },
}

async fn call_summarize_conversation(
    lm: &Arc<LM>,
    input: &SummarizeConversationInput,
) -> Result<SummarizeConversationOutput, SummarizationError> {
    let adapter = ChatAdapter;

    let system = adapter
        .format_system_message_typed_with_instruction::<SummarizeConversation>(Some(
            SUMMARY_INSTRUCTION,
        ))
        .map_err(|e| {
            error!("Failed to format DSRS system prompt: {e:?}");
            SummarizationError::Api(ApiError::InternalServerError)
        })?;
    let user_msg = adapter.format_user_message_typed::<SummarizeConversation>(input);

    let mut chat = dspy_rs::Chat::new(vec![]);
    chat.push("system", &system);
    chat.push("user", &user_msg);

    let response = lm.call(chat, Vec::new()).await.map_err(|e| {
        error!("DSRS LM call failed: {e:?}");
        SummarizationError::Api(ApiError::InternalServerError)
    })?;

    let raw_response = response.output.content();
    let (output, _meta) = adapter
        .parse_response_typed::<SummarizeConversation>(&response.output)
        .map_err(|e| {
            error!("DSRS typed parse failed: {e:?}");
            SummarizationError::Parse {
                raw_response,
                error_message: e.to_string(),
            }
        })?;

    Ok(SummarizeConversationOutput {
        summary: output.summary,
    })
}

async fn call_summarization_correction(
    lm: &Arc<LM>,
    input: &SummarizationCorrectionInput,
) -> Result<SummarizeConversationOutput, ApiError> {
    let adapter = ChatAdapter;

    let system = adapter
        .format_system_message_typed_with_instruction::<SummarizationCorrection>(Some(
            SUMMARY_CORRECTION_INSTRUCTION,
        ))
        .map_err(|e| {
            error!("Failed to format DSRS system prompt: {e:?}");
            ApiError::InternalServerError
        })?;
    let user_msg = adapter.format_user_message_typed::<SummarizationCorrection>(input);

    let mut chat = dspy_rs::Chat::new(vec![]);
    chat.push("system", &system);
    chat.push("user", &user_msg);

    let response = lm.call(chat, Vec::new()).await.map_err(|e| {
        error!("DSRS LM call failed: {e:?}");
        ApiError::InternalServerError
    })?;

    let (output, _meta) = adapter
        .parse_response_typed::<SummarizationCorrection>(&response.output)
        .map_err(|e| {
            error!("DSRS typed parse failed: {e:?}");
            ApiError::InternalServerError
        })?;

    Ok(SummarizeConversationOutput {
        summary: output.summary,
    })
}
