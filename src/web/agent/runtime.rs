use chrono_tz::Tz;
use secp256k1::SecretKey;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, error, warn};
use uuid::Uuid;

use diesel::prelude::*;

use crate::encrypt::{decrypt_string, encrypt_with_key};
use crate::models::agents::{
    Agent, NewAgent, AGENT_CREATED_BY_USER, AGENT_KIND_MAIN, AGENT_KIND_SUBAGENT,
};
use crate::models::conversation_summaries::{ConversationSummary, NewConversationSummary};
use crate::models::memory_blocks::{
    MemoryBlock, MEMORY_BLOCK_LABEL_HUMAN, MEMORY_BLOCK_LABEL_PERSONA,
};
use crate::models::responses::{
    AssistantMessage, Conversation, NewAssistantMessage, NewConversation, NewToolCall,
    NewToolOutput, NewUserMessage, RawThreadMessage, RawThreadMessageMetadata, ToolCall,
    ToolOutput, UserMessage,
};
use crate::models::schema::{
    agents, assistant_messages, memory_blocks, user_embeddings, user_messages,
};
use crate::models::user_preferences::{
    NewUserPreference, UserPreference, UserPreferenceError, USER_PREFERENCE_LOCALE,
    USER_PREFERENCE_TIMEZONE,
};
use crate::rag::{
    insert_message_embedding, serialize_f32_le, SOURCE_TYPE_ARCHIVAL, SOURCE_TYPE_MESSAGE,
};
use crate::tokens::count_tokens;
use crate::web::openai_auth::AuthMethod;
use crate::web::responses::{MessageContent, MessageContentConverter, MessageContentPart};
use crate::{ApiError, AppState};

use super::compaction::CompactionManager;
use super::schedules::{
    refresh_follow_user_schedules_for_user, CancelScheduleTool, ListSchedulesTool, ScheduleTaskTool,
};
use super::signatures::{
    build_lm, call_agent_response_with_retry_and_correction, AgentResponseInput, AgentToolCall,
    AGENT_INSTRUCTION,
};
use super::tools::{
    ArchivalInsertTool, ArchivalSearchTool, ConversationSearchTool, DoneTool, MemoryAppendTool,
    MemoryInsertTool, MemoryReplaceTool, SpawnSubagentTool, ToolRegistry, ToolResult,
    WebSearchTool,
};
use super::vision;

pub(crate) const DEFAULT_PERSONA_VALUE: &str = "I am Maple, a helpful AI companion. I maintain long-term memory across our conversations and strive to be friendly, concise, and genuinely helpful.";
pub const DEFAULT_MODEL: &str = "kimi-k2-5";
pub const DEFAULT_CONTEXT_WINDOW: i32 = 256_000;
pub const DEFAULT_COMPACTION_THRESHOLD: f32 = 0.80;
const MIN_MESSAGES_IN_CONTEXT: usize = 20;
const MAIN_AGENT_METADATA_TYPE: &str = "agent_main";
const SUBAGENT_METADATA_TYPE: &str = "subagent";
const MAIN_AGENT_ONBOARDING_TURN_LIMIT: i64 = 15;
const MAIN_AGENT_ONBOARDING_MESSAGES: [&str; 3] = [
    "Hey, I'm Maple. 👋",
    "Nice to meet you.",
    "What should I call you?",
];

#[derive(Clone, Debug, Default)]
struct AgentContext {
    current_time: String,
    user_locale: String,
    agent_kind: String,
    subagent_purpose: String,
    persona_block: String,
    human_block: String,
    memory_metadata: String,
    previous_context_summary: String,
    recent_conversation: String,
    main_agent_user_message_count: i64,
    is_first_time_user: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MainAgentInitOptions {
    pub timezone: Option<String>,
    pub locale: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct SeededOnboardingMessage {
    pub id: Uuid,
    pub content: String,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub(crate) struct MainAgentInitResult {
    pub agent: Agent,
    pub conversation: Conversation,
    pub onboarding_messages: Vec<SeededOnboardingMessage>,
}

#[derive(Clone, Debug)]
pub struct ExecutedTool {
    pub tool_call: AgentToolCall,
    pub result: ToolResult,
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub messages: Vec<String>,
    #[allow(dead_code)]
    pub tool_calls: Vec<AgentToolCall>,
    pub executed_tools: Vec<ExecutedTool>,
    pub done: bool,
}

pub struct AgentRuntime {
    state: Arc<AppState>,
    user: Arc<crate::models::users::User>,
    user_key: Arc<SecretKey>,
    agent: Agent,
    conversation: Conversation,
    subagent_purpose: String,
    system_prompt: String,
    lm: Arc<dspy_rs::LM>,
    tools: ToolRegistry,
    available_tools: String,
    compaction: CompactionManager,
    current_tool_results: Vec<String>,
    previous_step_summary: Option<(Vec<String>, Vec<String>)>,
    max_steps: usize,
}

fn build_system_prompt(agent: &Agent, subagent_purpose: &str) -> String {
    let mut prompt = AGENT_INSTRUCTION.to_string();

    if agent.kind == AGENT_KIND_MAIN {
        prompt.push_str(
            "\n\nMAIN AGENT MODE:\nYou are the user's primary persistent agent and the home surface of Maple, Maple AI's secure and encrypted communications app.",
        );
    } else {
        prompt.push_str(
            "\n\nSUBAGENT MODE:\nYou are operating inside a focused subagent chat. You share the user's memory with the main agent, but your recent conversation history is only this thread. Stay tightly focused on your assigned purpose and do not behave like a separate identity.",
        );

        if !subagent_purpose.trim().is_empty() {
            prompt.push_str("\n\nSUBAGENT PURPOSE:\n");
            prompt.push_str(subagent_purpose.trim());
        }
    }

    prompt
}

fn derive_subagent_display_name(display_name: Option<&str>, purpose: &str) -> String {
    if let Some(name) = display_name.map(str::trim).filter(|s| !s.is_empty()) {
        let trimmed: String = name.chars().take(80).collect();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }

    let mut derived = purpose
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ");
    if derived.is_empty() {
        derived = "New Subagent".to_string();
    }
    derived.chars().take(80).collect()
}

async fn main_agent_metadata_enc(user_key: &SecretKey) -> Vec<u8> {
    encrypt_with_key(
        user_key,
        json!({"type": MAIN_AGENT_METADATA_TYPE})
            .to_string()
            .as_bytes(),
    )
    .await
}

async fn subagent_metadata_enc(
    user_key: &SecretKey,
    agent_uuid: Uuid,
    display_name: &str,
    purpose: &str,
) -> Vec<u8> {
    encrypt_with_key(
        user_key,
        json!({
            "type": SUBAGENT_METADATA_TYPE,
            "agent_id": agent_uuid,
            "display_name": display_name,
            "purpose": purpose,
        })
        .to_string()
        .as_bytes(),
    )
    .await
}

fn normalize_optional_preference(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn validate_optional_preference(
    key: &str,
    value: Option<&str>,
) -> Result<Option<String>, ApiError> {
    let Some(value) = normalize_optional_preference(value) else {
        return Ok(None);
    };

    UserPreference::validate(key, &value).map_err(|e| match e {
        UserPreferenceError::InvalidPreference(_) => ApiError::BadRequest,
        UserPreferenceError::DatabaseError(_) => ApiError::InternalServerError,
    })?;

    Ok(Some(value))
}

fn onboarding_message_texts() -> &'static [&'static str; 3] {
    &MAIN_AGENT_ONBOARDING_MESSAGES
}

fn load_user_preference(
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
    key: &str,
) -> Result<Option<String>, ApiError> {
    let preference = UserPreference::get_by_user_and_key(conn, user_id, key).map_err(|e| {
        error!("Failed to load user preference '{key}': {e:?}");
        ApiError::InternalServerError
    })?;

    decrypt_string(
        user_key,
        preference.as_ref().map(|preference| &preference.value_enc),
    )
    .map_err(|e| {
        error!("Failed to decrypt user preference '{key}': {e:?}");
        ApiError::InternalServerError
    })
}

fn load_user_timezone(
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
) -> Result<Option<Tz>, ApiError> {
    let Some(timezone) = load_user_preference(conn, user_key, user_id, USER_PREFERENCE_TIMEZONE)?
    else {
        return Ok(None);
    };

    match timezone.parse::<Tz>() {
        Ok(timezone) => Ok(Some(timezone)),
        Err(_) => {
            warn!("Ignoring invalid stored timezone '{timezone}' for user {user_id}");
            Ok(None)
        }
    }
}

fn format_current_time(now: chrono::DateTime<chrono::Utc>, timezone: Option<&Tz>) -> String {
    if let Some(timezone) = timezone {
        let local_time = now.with_timezone(timezone);
        format!(
            "{} ({})",
            local_time.format("%m/%d/%Y %H:%M:%S (%A)"),
            timezone.name()
        )
    } else {
        format!("{} UTC", now.format("%m/%d/%Y %H:%M:%S (%A)"))
    }
}

fn format_message_timestamp(
    created_at: chrono::DateTime<chrono::Utc>,
    timezone: Option<&Tz>,
) -> String {
    if let Some(timezone) = timezone {
        let local_time = created_at.with_timezone(timezone);
        format!(
            "{} ({})",
            local_time.format("%m/%d/%Y %H:%M:%S"),
            timezone.name()
        )
    } else {
        format!("{} UTC", created_at.format("%m/%d/%Y %H:%M:%S"))
    }
}

async fn upsert_user_preference(
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
    key: &str,
    value: Option<String>,
) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };

    let value_enc = encrypt_with_key(user_key, value.as_bytes()).await;
    NewUserPreference::new(user_id, key, value_enc)
        .insert_or_update(conn)
        .map_err(|e| {
            error!("Failed to upsert user preference '{key}': {e:?}");
            ApiError::InternalServerError
        })?;

    if key == USER_PREFERENCE_TIMEZONE {
        refresh_follow_user_schedules_for_user(conn, user_id, &value).map_err(|e| {
            error!("Failed to refresh follow-user schedules for timezone update: {e:?}");
            ApiError::InternalServerError
        })?;
    }

    Ok(())
}

async fn seed_main_agent_onboarding_messages(
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
    conversation_id: i64,
) -> Result<Vec<SeededOnboardingMessage>, ApiError> {
    let user_message_count: i64 = user_messages::table
        .filter(user_messages::conversation_id.eq(conversation_id))
        .count()
        .get_result(conn)
        .map_err(|e| {
            error!("Failed to count user messages for onboarding seed: {e:?}");
            ApiError::InternalServerError
        })?;

    let assistant_message_count: i64 = assistant_messages::table
        .filter(assistant_messages::conversation_id.eq(conversation_id))
        .count()
        .get_result(conn)
        .map_err(|e| {
            error!("Failed to count assistant messages for onboarding seed: {e:?}");
            ApiError::InternalServerError
        })?;

    if user_message_count > 0 || assistant_message_count > 0 {
        return Ok(vec![]);
    }

    let mut seeded_messages = Vec::with_capacity(onboarding_message_texts().len());

    for message in onboarding_message_texts() {
        let content_enc = Some(encrypt_with_key(user_key, message.as_bytes()).await);
        let completion_tokens = count_tokens(message) as i32;

        let inserted = NewAssistantMessage {
            uuid: Uuid::new_v4(),
            conversation_id,
            response_id: None,
            user_id,
            content_enc,
            completion_tokens,
            status: "completed".to_string(),
            finish_reason: None,
        }
        .insert(conn)
        .map_err(|e| {
            error!("Failed to seed main agent onboarding message: {e:?}");
            ApiError::InternalServerError
        })?;

        seeded_messages.push(SeededOnboardingMessage {
            id: inserted.uuid,
            content: message.to_string(),
            created_at: inserted.created_at.timestamp(),
        });
    }

    Ok(seeded_messages)
}

pub(crate) fn load_main_agent(
    conn: &mut diesel::PgConnection,
    user_id: Uuid,
) -> Result<Option<(Agent, Conversation)>, ApiError> {
    let Some(agent) = Agent::get_main_for_user(conn, user_id).map_err(|e| {
        error!("Failed to load main agent: {e:?}");
        ApiError::InternalServerError
    })?
    else {
        return Ok(None);
    };

    let conversation = match Conversation::get_by_id_and_user(conn, agent.conversation_id, user_id)
    {
        Ok(conversation) => conversation,
        Err(crate::models::responses::ResponsesError::ConversationNotFound) => {
            warn!(
                "Main agent {} for user {} points to missing conversation {}; treating as uninitialized",
                agent.uuid,
                user_id,
                agent.conversation_id,
            );
            return Ok(None);
        }
        Err(e) => {
            error!("Failed to load main agent conversation: {e:?}");
            return Err(ApiError::InternalServerError);
        }
    };

    Ok(Some((agent, conversation)))
}

pub(crate) async fn init_main_agent(
    _state: &Arc<AppState>,
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
    init_options: &MainAgentInitOptions,
) -> Result<MainAgentInitResult, ApiError> {
    let timezone =
        validate_optional_preference(USER_PREFERENCE_TIMEZONE, init_options.timezone.as_deref())?;
    let locale =
        validate_optional_preference(USER_PREFERENCE_LOCALE, init_options.locale.as_deref())?;

    let existing = Agent::get_main_for_user(conn, user_id).map_err(|e| {
        error!("Failed to load main agent: {e:?}");
        ApiError::InternalServerError
    })?;

    let (agent, conversation) = if let Some(agent) = existing {
        match Conversation::get_by_id_and_user(conn, agent.conversation_id, user_id) {
            Ok(conversation) => (agent, conversation),
            Err(_) => {
                let conversation = NewConversation {
                    uuid: Uuid::new_v4(),
                    user_id,
                    metadata_enc: Some(main_agent_metadata_enc(user_key).await),
                }
                .insert(conn)
                .map_err(|e| {
                    error!("Failed to recreate main agent conversation: {e:?}");
                    ApiError::InternalServerError
                })?;

                let agent = diesel::update(agents::table.filter(agents::id.eq(agent.id)))
                    .set(agents::conversation_id.eq(conversation.id))
                    .get_result::<Agent>(conn)
                    .map_err(|e| {
                        error!("Failed to repair main agent conversation_id: {e:?}");
                        ApiError::InternalServerError
                    })?;

                (agent, conversation)
            }
        }
    } else {
        let conversation = NewConversation {
            uuid: Uuid::new_v4(),
            user_id,
            metadata_enc: Some(main_agent_metadata_enc(user_key).await),
        }
        .insert(conn)
        .map_err(|e| {
            error!("Failed to create main agent conversation: {e:?}");
            ApiError::InternalServerError
        })?;

        let agent = NewAgent {
            uuid: Uuid::new_v4(),
            user_id,
            conversation_id: conversation.id,
            kind: AGENT_KIND_MAIN.to_string(),
            parent_agent_id: None,
            display_name_enc: None,
            purpose_enc: None,
            created_by: AGENT_CREATED_BY_USER.to_string(),
        }
        .insert(conn)
        .map_err(|e| {
            error!("Failed to create main agent row: {e:?}");
            ApiError::InternalServerError
        })?;

        (agent, conversation)
    };

    upsert_user_preference(conn, user_key, user_id, USER_PREFERENCE_TIMEZONE, timezone).await?;
    upsert_user_preference(conn, user_key, user_id, USER_PREFERENCE_LOCALE, locale).await?;
    let onboarding_messages =
        seed_main_agent_onboarding_messages(conn, user_key, user_id, conversation.id).await?;

    Ok(MainAgentInitResult {
        agent,
        conversation,
        onboarding_messages,
    })
}

pub(crate) async fn create_subagent(
    conn: &mut diesel::PgConnection,
    user_key: &SecretKey,
    user_id: Uuid,
    parent_agent: &Agent,
    display_name: Option<&str>,
    purpose: &str,
    created_by: &str,
) -> Result<(Agent, Conversation, String), ApiError> {
    if parent_agent.kind != AGENT_KIND_MAIN {
        return Err(ApiError::BadRequest);
    }

    let purpose = purpose.trim();
    if purpose.is_empty() {
        return Err(ApiError::BadRequest);
    }

    let resolved_display_name = derive_subagent_display_name(display_name, purpose);
    let agent_uuid = Uuid::new_v4();

    let conversation = NewConversation {
        uuid: Uuid::new_v4(),
        user_id,
        metadata_enc: Some(
            subagent_metadata_enc(user_key, agent_uuid, &resolved_display_name, purpose).await,
        ),
    }
    .insert(conn)
    .map_err(|e| {
        error!("Failed to create subagent conversation: {e:?}");
        ApiError::InternalServerError
    })?;

    let display_name_enc = Some(encrypt_with_key(user_key, resolved_display_name.as_bytes()).await);
    let purpose_enc = Some(encrypt_with_key(user_key, purpose.as_bytes()).await);

    let agent = NewAgent {
        uuid: agent_uuid,
        user_id,
        conversation_id: conversation.id,
        kind: AGENT_KIND_SUBAGENT.to_string(),
        parent_agent_id: Some(parent_agent.id),
        display_name_enc,
        purpose_enc,
        created_by: created_by.to_string(),
    }
    .insert(conn)
    .map_err(|e| {
        error!("Failed to create subagent row: {e:?}");
        ApiError::InternalServerError
    })?;

    Ok((agent, conversation, resolved_display_name))
}

impl AgentRuntime {
    pub async fn new_main(
        state: Arc<AppState>,
        user: crate::models::users::User,
        user_key: SecretKey,
    ) -> Result<Self, ApiError> {
        let user = Arc::new(user);
        let user_key = Arc::new(user_key);

        let mut conn = state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let (agent, conversation) =
            load_main_agent(&mut conn, user.uuid)?.ok_or(ApiError::NotFound)?;

        Self::from_loaded(state, user, user_key, agent, conversation).await
    }

    pub async fn new_subagent(
        state: Arc<AppState>,
        user: crate::models::users::User,
        user_key: SecretKey,
        agent_uuid: Uuid,
    ) -> Result<Self, ApiError> {
        let user = Arc::new(user);
        let user_key = Arc::new(user_key);

        let mut conn = state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let agent = Agent::get_by_uuid_and_user(&mut conn, agent_uuid, user.uuid)
            .map_err(|e| {
                error!("Failed to load subagent: {e:?}");
                ApiError::InternalServerError
            })?
            .ok_or(ApiError::NotFound)?;

        if agent.kind != AGENT_KIND_SUBAGENT {
            return Err(ApiError::NotFound);
        }

        let conversation =
            Conversation::get_by_id_and_user(&mut conn, agent.conversation_id, user.uuid).map_err(
                |e| {
                    error!("Failed to load subagent conversation: {e:?}");
                    ApiError::InternalServerError
                },
            )?;

        Self::from_loaded(state, user, user_key, agent, conversation).await
    }

    async fn from_loaded(
        state: Arc<AppState>,
        user: Arc<crate::models::users::User>,
        user_key: Arc<SecretKey>,
        agent: Agent,
        conversation: Conversation,
    ) -> Result<Self, ApiError> {
        let subagent_purpose = decrypt_string(&user_key, agent.purpose_enc.as_ref())
            .map_err(|e| {
                error!("Failed to decrypt subagent purpose: {e:?}");
                ApiError::InternalServerError
            })?
            .unwrap_or_default();

        let system_prompt = build_system_prompt(&agent, &subagent_purpose);

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(MemoryReplaceTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(MemoryAppendTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(MemoryInsertTool::new(
            state.clone(),
            user.uuid,
            user_key.clone(),
        )));
        tools.register(Arc::new(ArchivalInsertTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
        )));
        tools.register(Arc::new(ArchivalSearchTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
        )));
        tools.register(Arc::new(ConversationSearchTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
            conversation.id,
        )));
        if let Some(brave_client) = state.brave_client.clone() {
            tools.register(Arc::new(WebSearchTool::new(brave_client)));
        }
        tools.register(Arc::new(ScheduleTaskTool::new(
            state.clone(),
            user.clone(),
            user_key.clone(),
            agent.clone(),
        )));
        tools.register(Arc::new(ListSchedulesTool::new(
            state.clone(),
            user.clone(),
            agent.clone(),
        )));
        tools.register(Arc::new(CancelScheduleTool::new(
            state.clone(),
            user.clone(),
            agent.clone(),
        )));
        if agent.kind == AGENT_KIND_MAIN {
            tools.register(Arc::new(SpawnSubagentTool::new(
                state.clone(),
                user.clone(),
                user_key.clone(),
                agent.clone(),
            )));
        }
        tools.register(Arc::new(DoneTool));

        let available_tools = tools.generate_description();
        let lm = build_lm(
            state.clone(),
            user.clone(),
            DEFAULT_MODEL.to_string(),
            0.7,
            32768,
        )
        .await?;

        Ok(Self {
            state,
            user,
            user_key,
            agent,
            conversation,
            subagent_purpose,
            system_prompt,
            lm,
            tools,
            available_tools,
            compaction: CompactionManager::new(),
            current_tool_results: Vec::new(),
            previous_step_summary: None,
            max_steps: 10,
        })
    }

    pub fn clear_tool_results(&mut self) {
        self.current_tool_results.clear();
        self.previous_step_summary = None;
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    fn build_step_system_prompt(&self, ctx: &AgentContext) -> String {
        let mut prompt = self.system_prompt.clone();

        if self.agent.kind == AGENT_KIND_MAIN
            && ctx.main_agent_user_message_count > 0
            && ctx.main_agent_user_message_count <= MAIN_AGENT_ONBOARDING_TURN_LIMIT
        {
            prompt.push_str(&format!(
                "\n\nMAIN AGENT ONBOARDING WINDOW:\nThis is still early in your relationship with the user (main-agent user message {} of {}). Be especially warm, welcoming, and natural. Focus on getting to know them gradually rather than interrogating them. Ask at most one thoughtful follow-up at a time. At a natural point early on, briefly explain who Maple is, what this app is good for, and what the user can expect from chatting here. Do that conversationally, not as a canned pitch, and do not force it into the second or third message if the moment is awkward. Prioritize learning their name, important relationships, routines, goals, preferences, and current life context when it feels natural. Save useful facts to memory proactively. Avoid transactional, overly clinical, or assistant-like phrasing. Make Maple feel like a welcoming, steady presence, not a service desk.",
                ctx.main_agent_user_message_count,
                MAIN_AGENT_ONBOARDING_TURN_LIMIT,
            ));

            if ctx.human_block.trim().is_empty() {
                prompt.push_str(
                    "\nIf you still do not know the user's name, ask what they would like to be called and store it as soon as they answer.",
                );
            }
        }

        if !ctx.user_locale.trim().is_empty() {
            prompt.push_str("\n\nUSER LOCALE HINT:\nThe user's preferred locale is '");
            prompt.push_str(ctx.user_locale.trim());
            prompt.push_str("'. If it is natural and the user has not indicated otherwise, prefer replying in that locale/language.");
        }

        prompt
    }

    /// Prepare the runtime for a new message: validate, persist user message, compact if needed.
    /// Call this once before driving the step loop.
    pub async fn prepare(&mut self, user_message: &MessageContent) -> Result<String, ApiError> {
        MessageContentConverter::validate_content(user_message)?;
        let normalized = MessageContentConverter::normalize_content(user_message.clone());

        let user_text = MessageContentConverter::extract_text_for_token_counting(&normalized);
        let user_text = user_text.trim().to_string();

        let image_url = match &normalized {
            MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                MessageContentPart::InputImage {
                    image_url: Some(url),
                    ..
                } => Some(url.clone()),
                _ => None,
            }),
            MessageContent::Text(_) => None,
        };

        let attachment_text = if let Some(image_url) = &image_url {
            let recent_context = self
                .get_recent_messages_for_vision(6)
                .await
                .unwrap_or_else(|e| {
                    warn!("Failed to build vision context: {e:?}");
                    String::new()
                });

            match vision::describe_image(
                &self.state,
                self.user.as_ref(),
                AuthMethod::Jwt,
                DEFAULT_MODEL,
                image_url,
                &user_text,
                &recent_context,
            )
            .await
            {
                Ok(desc) => Some(desc),
                Err(e) => {
                    warn!("Vision pre-processing failed: {e:?}");
                    Some("[Could not describe image]".to_string())
                }
            }
        } else {
            None
        };

        let mut parts: Vec<String> = Vec::new();
        if !user_text.is_empty() {
            parts.push(user_text);
        }
        if let Some(att) = attachment_text.as_ref().filter(|s| !s.trim().is_empty()) {
            parts.push(format!("[Uploaded Image: {}]", att.trim()));
        }

        let embed_text = parts.join("\n\n");
        if embed_text.trim().is_empty() {
            return Err(ApiError::BadRequest);
        }

        self.clear_tool_results();
        self.insert_user_message(normalized, attachment_text, &embed_text)
            .await?;
        self.maybe_compact().await?;
        Ok(embed_text)
    }

    async fn get_recent_messages_for_vision(&self, limit: usize) -> Result<String, ApiError> {
        if limit == 0 {
            return Ok(String::new());
        }

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        // Load more than needed in case tool calls/outputs are interleaved.
        let fetch_limit = (limit as i64).saturating_mul(10).max(limit as i64);
        let mut messages = RawThreadMessage::get_conversation_context(
            &mut conn,
            self.conversation.id,
            fetch_limit,
            None,
            "desc",
        )
        .map_err(|e| {
            error!("Failed to load recent messages for vision: {e:?}");
            ApiError::InternalServerError
        })?;

        messages.reverse();

        let mut formatted_rev: Vec<String> = Vec::new();
        for msg in messages.iter().rev() {
            if msg.message_type != "user" && msg.message_type != "assistant" {
                continue;
            }

            let content = self.render_raw_message(msg)?;
            let truncated: String = content.chars().take(300).collect();
            formatted_rev.push(format!("[{}]: {}", msg.message_type, truncated));
            if formatted_rev.len() >= limit {
                break;
            }
        }

        formatted_rev.reverse();
        Ok(formatted_rev.join("\n"))
    }

    /// Execute a single step of the agent loop.
    /// The caller (chat handler) drives the loop, persists messages, and emits SSE events.
    pub async fn step(
        &mut self,
        user_message: &str,
        is_first_step: bool,
    ) -> Result<StepResult, ApiError> {
        if is_first_step {
            self.current_tool_results.clear();
        }

        debug!("Agent step (first={})", is_first_step);

        let ctx = self.build_context().await?;

        let input_content = if is_first_step {
            user_message.to_string()
        } else {
            let tool_results: Vec<&str> = self
                .current_tool_results
                .iter()
                .map(|s| s.as_str())
                .collect();

            if tool_results.is_empty() {
                user_message.to_string()
            } else {
                let already_sent = if let Some((sent_messages, tool_names)) =
                    &self.previous_step_summary
                {
                    let tools_str = tool_names.join(", ");
                    let msgs_preview = if sent_messages.is_empty() {
                        String::new()
                    } else {
                        let msgs_text = sent_messages
                            .iter()
                            .enumerate()
                            .map(|(i, m)| format!("  {}. \"{}\"", i + 1, m))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("\nMessages you already sent to user:\n{}\n", msgs_text)
                    };

                    format!(
                        "[You already sent {} message(s) and called {} this turn.{}Tools have executed:]\n\n",
                        sent_messages.len(),
                        tools_str,
                        msgs_preview
                    )
                } else {
                    String::new()
                };

                let tool_result_instructions = r#"

=== TOOL RESULT PROCESSING MODE ===
This is a CONTINUATION of your previous turn, NOT a new conversation.
Your previous messages are already visible to the user in recent_conversation.

RULES:
1. SILENCE IS DEFAULT - You do NOT need to acknowledge the tool result
2. DO NOT say: "I see the results", "Let me analyze", "Based on what I found", "Here's what the tool returned"
3. DO NOT repeat or rephrase what you already said
4. If the tool was for YOUR benefit (memory ops, archival), call 'done' immediately
5. Only send messages if you have GENUINELY NEW information the user hasn't seen

SELF-CHECK: Before ANY message, ask: "Is this new info the user hasn't seen?" If no → call 'done'"#;

                let result = if tool_results.len() == 1 {
                    format!(
                        "{}=== TOOL RESULT ===\n{}\n=== END TOOL RESULT ==={}",
                        already_sent, tool_results[0], tool_result_instructions
                    )
                } else {
                    let results_text = tool_results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| format!("--- Tool {} ---\n{}", i + 1, r))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    format!(
                        "{}=== TOOL RESULTS ({} tools) ===\n{}\n=== END TOOL RESULTS ==={}",
                        already_sent,
                        tool_results.len(),
                        results_text,
                        tool_result_instructions
                    )
                };

                self.current_tool_results.clear();

                result
            }
        };

        let system_prompt = self.build_step_system_prompt(&ctx);

        let input = AgentResponseInput {
            input: input_content.clone(),
            current_time: ctx.current_time,
            agent_kind: ctx.agent_kind,
            subagent_purpose: ctx.subagent_purpose,
            persona_block: ctx.persona_block,
            human_block: ctx.human_block,
            memory_metadata: ctx.memory_metadata,
            previous_context_summary: ctx.previous_context_summary,
            recent_conversation: ctx.recent_conversation,
            available_tools: self.available_tools.clone(),
            is_first_time_user: ctx.is_first_time_user,
        };

        let response = call_agent_response_with_retry_and_correction(
            &self.lm,
            &system_prompt,
            &input,
            &input_content,
            &self.available_tools,
        )
        .await?;

        // Unwrap nested JSON arrays emitted by the model.
        let messages: Vec<String> = response
            .messages
            .iter()
            .flat_map(|m| {
                let trimmed = m.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    if let Ok(inner) = serde_json::from_str::<Vec<String>>(trimmed) {
                        return inner;
                    }
                }
                vec![m.clone()]
            })
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .collect();

        // Execute tools and inject results for next step.
        // Persistence is handled by the caller (chat handler) so messages are
        // sent first, stored synchronously, then embedded asynchronously.
        let mut executed_tools = Vec::new();
        for tool_call in &response.tool_calls {
            let result = if tool_call.name == "done" {
                ToolResult::success("Done".to_string())
            } else if let Some(tool) = self.tools.get(&tool_call.name) {
                tool.execute(&tool_call.args).await
            } else {
                ToolResult::error(format!("Unknown tool: {}", tool_call.name))
            };

            self.inject_tool_result(tool_call, &result);

            if tool_call.name != "done" {
                executed_tools.push(ExecutedTool {
                    tool_call: tool_call.clone(),
                    result,
                });
            }
        }

        let done = response.tool_calls.is_empty()
            || (response.tool_calls.len() == 1 && response.tool_calls[0].name == "done");

        if !messages.is_empty() || !response.tool_calls.is_empty() {
            let tool_names: Vec<String> = response
                .tool_calls
                .iter()
                .map(|tc| tc.name.clone())
                .collect();
            self.previous_step_summary = Some((messages.clone(), tool_names));
        }

        Ok(StepResult {
            messages,
            tool_calls: response.tool_calls,
            executed_tools,
            done,
        })
    }

    fn inject_tool_result(&mut self, tool_call: &AgentToolCall, result: &ToolResult) {
        let args_str = if tool_call.args.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = tool_call
                .args
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("\nArgs: {}", pairs.join(", "))
        };

        let result_text = format!(
            "[Tool Result: {}]{}\nStatus: {}\nOutput: {}",
            tool_call.name,
            args_str,
            if result.success { "OK" } else { "ERROR" },
            if result.success {
                result.output.as_str()
            } else {
                result.error.as_deref().unwrap_or("Unknown error")
            }
        );

        self.current_tool_results.push(result_text);
    }

    async fn build_context(&self) -> Result<AgentContext, ApiError> {
        let mut ctx = AgentContext {
            agent_kind: self.agent.kind.clone(),
            subagent_purpose: self.subagent_purpose.clone(),
            ..AgentContext::default()
        };

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let user_timezone = load_user_timezone(&mut conn, &self.user_key, self.user.uuid)?;
        ctx.user_locale = load_user_preference(
            &mut conn,
            &self.user_key,
            self.user.uuid,
            USER_PREFERENCE_LOCALE,
        )?
        .unwrap_or_default();

        let now = chrono::Utc::now();
        ctx.current_time = format_current_time(now, user_timezone.as_ref());

        let persona = MemoryBlock::get_by_user_and_label(
            &mut conn,
            self.user.uuid,
            MEMORY_BLOCK_LABEL_PERSONA,
        )
        .map_err(|_| ApiError::InternalServerError)?;
        let human =
            MemoryBlock::get_by_user_and_label(&mut conn, self.user.uuid, MEMORY_BLOCK_LABEL_HUMAN)
                .map_err(|_| ApiError::InternalServerError)?;

        ctx.persona_block = decrypt_string(&self.user_key, persona.as_ref().map(|b| &b.value_enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_else(|| DEFAULT_PERSONA_VALUE.to_string());
        ctx.human_block = decrypt_string(&self.user_key, human.as_ref().map(|b| &b.value_enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_default();

        // Memory metadata
        let last_modified = memory_blocks::table
            .filter(memory_blocks::user_id.eq(self.user.uuid))
            .select(diesel::dsl::max(memory_blocks::updated_at))
            .first::<Option<chrono::DateTime<chrono::Utc>>>(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let recall_count: i64 = user_embeddings::table
            .filter(user_embeddings::user_id.eq(self.user.uuid))
            .filter(user_embeddings::source_type.eq(SOURCE_TYPE_MESSAGE))
            .count()
            .get_result(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let archival_count: i64 = user_embeddings::table
            .filter(user_embeddings::user_id.eq(self.user.uuid))
            .filter(user_embeddings::source_type.eq(SOURCE_TYPE_ARCHIVAL))
            .count()
            .get_result(&mut conn)
            .map_err(|_| ApiError::InternalServerError)?;

        let mut metadata = String::new();
        if let Some(ts) = last_modified {
            metadata.push_str(&format!(
                "- Memory blocks last modified: {} UTC\n",
                ts.format("%Y-%m-%d %H:%M:%S")
            ));
        }
        metadata.push_str(&format!(
            "- {} messages in recall memory (use conversation_search to access)\n",
            recall_count
        ));
        metadata.push_str(&format!(
            "- {} passages in archival memory (use archival_search to access)",
            archival_count
        ));
        ctx.memory_metadata = metadata;

        // Conversation summary + recent messages
        let summary = ConversationSummary::get_latest_for_conversation(
            &mut conn,
            self.user.uuid,
            self.conversation.id,
        )
        .map_err(|_| ApiError::InternalServerError)?;

        if let Some(s) = &summary {
            ctx.previous_context_summary = decrypt_string(&self.user_key, Some(&s.content_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default();
        }

        if self.agent.kind == AGENT_KIND_MAIN {
            ctx.main_agent_user_message_count = user_messages::table
                .filter(user_messages::conversation_id.eq(self.conversation.id))
                .count()
                .get_result(&mut conn)
                .map_err(|e| {
                    error!("Failed to count main agent user messages: {e:?}");
                    ApiError::InternalServerError
                })?;

            if ctx.main_agent_user_message_count <= 1 && ctx.human_block.trim().is_empty() {
                ctx.is_first_time_user = true;
            }
        }

        let mut messages = RawThreadMessage::get_conversation_context(
            &mut conn,
            self.conversation.id,
            10_000,
            None,
            "desc",
        )
        .map_err(|e| {
            error!("Failed to load conversation context: {e:?}");
            ApiError::InternalServerError
        })?;
        messages.reverse();

        if let Some(s) = &summary {
            messages.retain(|m| m.created_at > s.to_created_at);

            if messages.len() < MIN_MESSAGES_IN_CONTEXT {
                // Ensure at least a small slice of recent conversation
                let mut recent = RawThreadMessage::get_conversation_context(
                    &mut conn,
                    self.conversation.id,
                    MIN_MESSAGES_IN_CONTEXT as i64,
                    None,
                    "desc",
                )
                .map_err(|_| ApiError::InternalServerError)?;
                recent.reverse();
                messages = recent;
            }
        }

        // Render conversation history
        let mut conversation = String::new();
        for msg in &messages {
            if msg.message_type == "reasoning" {
                continue;
            }

            let role = match msg.message_type.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                "tool_call" | "tool_output" => "tool",
                other => other,
            };

            let timestamp = format_message_timestamp(msg.created_at, user_timezone.as_ref());
            let content = self.render_raw_message(msg)?;
            conversation.push_str(&format!("[{} @ {}]: {}\n", role, timestamp, content));
        }

        ctx.recent_conversation = if conversation.trim().is_empty() {
            "No previous conversation.".to_string()
        } else {
            conversation
        };

        Ok(ctx)
    }

    fn render_raw_message(&self, msg: &RawThreadMessage) -> Result<String, ApiError> {
        let Some(enc) = msg.content_enc.as_ref() else {
            return Ok(String::new());
        };

        let raw = decrypt_string(&self.user_key, Some(enc))
            .map_err(|_| ApiError::InternalServerError)?
            .unwrap_or_default();

        let mut content = if msg.message_type == "user" {
            let parsed: Result<MessageContent, _> = serde_json::from_str(&raw);
            match parsed {
                Ok(c) => MessageContentConverter::extract_text_for_token_counting(&c),
                Err(_) => raw,
            }
        } else if msg.message_type == "tool_call" {
            match msg.tool_name.as_deref() {
                Some(name) => {
                    if raw.trim().is_empty() {
                        format!("CALL {}", name)
                    } else {
                        format!("CALL {} args={}", name, raw)
                    }
                }
                None => raw,
            }
        } else if msg.message_type == "tool_output" {
            match msg.tool_name.as_deref() {
                Some(name) => format!("OUTPUT {}: {}", name, raw),
                None => raw,
            }
        } else {
            raw
        };

        if msg.message_type == "user" {
            let attachment_text = decrypt_string(&self.user_key, msg.attachment_text_enc.as_ref())
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default();

            if !attachment_text.trim().is_empty() {
                let suffix = format!("[Uploaded Image: {}]", attachment_text.trim());
                if content.trim().is_empty() {
                    content = suffix;
                } else {
                    content = format!("{}\n\n{}", content.trim(), suffix);
                }
            }
        }

        if (msg.message_type == "tool_call" || msg.message_type == "tool_output")
            && content.len() > 2000
        {
            let mut end = 2000;
            while !content.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            return Ok(format!("{}...", &content[..end]));
        }

        Ok(content)
    }

    pub async fn insert_user_message(
        &self,
        content: MessageContent,
        attachment_text: Option<String>,
        embed_text: &str,
    ) -> Result<UserMessage, ApiError> {
        let embed_text = embed_text.trim();
        if embed_text.is_empty() {
            return Err(ApiError::BadRequest);
        }

        let normalized = MessageContentConverter::normalize_content(content);
        let content_json = serde_json::to_string(&normalized).map_err(|e| {
            error!("Failed to serialize user MessageContent: {e:?}");
            ApiError::InternalServerError
        })?;

        let prompt_tokens = count_tokens(embed_text);
        let prompt_tokens = if prompt_tokens > i32::MAX as usize {
            i32::MAX
        } else {
            prompt_tokens as i32
        };

        let content_enc = encrypt_with_key(&self.user_key, content_json.as_bytes()).await;
        let attachment_text_enc = match attachment_text.as_ref().filter(|s| !s.trim().is_empty()) {
            Some(text) => Some(encrypt_with_key(&self.user_key, text.trim().as_bytes()).await),
            None => None,
        };

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let msg = NewUserMessage {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            content_enc,
            attachment_text_enc,
            prompt_tokens,
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert user message: {e:?}");
            ApiError::InternalServerError
        })?;

        // Store the message synchronously and update the embedding in background.
        let state = self.state.clone();
        let user = self.user.clone();
        let user_key = self.user_key.clone();
        let text = embed_text.to_string();
        let conversation_id = self.conversation.id;
        let user_message_id = Some(msg.id);

        tokio::spawn(async move {
            let _ = insert_message_embedding(
                &state,
                user.as_ref(),
                AuthMethod::Jwt,
                user_key.as_ref(),
                &text,
                conversation_id,
                user_message_id,
                None,
            )
            .await;
        });

        Ok(msg)
    }

    pub async fn insert_assistant_message(&self, text: &str) -> Result<AssistantMessage, ApiError> {
        let completion_tokens = count_tokens(text) as i32;
        let content_enc = Some(encrypt_with_key(&self.user_key, text.as_bytes()).await);

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let msg = NewAssistantMessage {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            content_enc,
            completion_tokens,
            status: "completed".to_string(),
            finish_reason: None,
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert assistant message: {e:?}");
            ApiError::InternalServerError
        })?;

        // Store the message synchronously and update the embedding in background.
        let state = self.state.clone();
        let user = self.user.clone();
        let user_key = self.user_key.clone();
        let text = text.to_string();
        let conversation_id = self.conversation.id;
        let assistant_message_id = Some(msg.id);

        tokio::spawn(async move {
            let _ = insert_message_embedding(
                &state,
                user.as_ref(),
                AuthMethod::Jwt,
                user_key.as_ref(),
                &text,
                conversation_id,
                None,
                assistant_message_id,
            )
            .await;
        });

        Ok(msg)
    }

    pub async fn insert_tool_call_and_output(
        &self,
        tool_call: &AgentToolCall,
        result: &ToolResult,
    ) -> Result<(ToolCall, ToolOutput), ApiError> {
        let args_json = serde_json::to_string(&tool_call.args).unwrap_or_else(|_| "{}".to_string());
        let arguments_enc = Some(encrypt_with_key(&self.user_key, args_json.as_bytes()).await);
        let argument_tokens = count_tokens(&args_json) as i32;

        let output_text = if result.success {
            result.output.clone()
        } else {
            result
                .error
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string())
        };
        let output_tokens = count_tokens(&output_text) as i32;
        let output_enc = encrypt_with_key(&self.user_key, output_text.as_bytes()).await;

        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let call = NewToolCall {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            name: tool_call.name.clone(),
            arguments_enc,
            argument_tokens,
            status: "completed".to_string(),
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert tool call: {e:?}");
            ApiError::InternalServerError
        })?;

        let out = NewToolOutput {
            uuid: Uuid::new_v4(),
            conversation_id: self.conversation.id,
            response_id: None,
            user_id: self.user.uuid,
            tool_call_fk: call.id,
            output_enc,
            output_tokens,
            status: "completed".to_string(),
            error: result.error.clone(),
        }
        .insert(&mut conn)
        .map_err(|e| {
            error!("Failed to insert tool output: {e:?}");
            ApiError::InternalServerError
        })?;

        Ok((call, out))
    }

    pub(crate) async fn maybe_compact(&self) -> Result<(), ApiError> {
        let mut conn = self
            .state
            .db
            .get_pool()
            .get()
            .map_err(|_| ApiError::InternalServerError)?;

        let latest_summary = ConversationSummary::get_latest_for_conversation(
            &mut conn,
            self.user.uuid,
            self.conversation.id,
        )
        .map_err(|_| ApiError::InternalServerError)?;

        let summary_tokens = latest_summary
            .as_ref()
            .map(|s| s.content_tokens)
            .unwrap_or(0);
        let summary_text = if let Some(s) = &latest_summary {
            decrypt_string(&self.user_key, Some(&s.content_enc))
                .map_err(|_| ApiError::InternalServerError)?
                .unwrap_or_default()
        } else {
            String::new()
        };

        let mut metadata = RawThreadMessageMetadata::get_conversation_context_metadata(
            &mut conn,
            self.conversation.id,
        )
        .map_err(|e| {
            error!("Failed to load context metadata: {e:?}");
            ApiError::InternalServerError
        })?;

        if let Some(s) = &latest_summary {
            metadata.retain(|m| m.created_at > s.to_created_at);
        }

        metadata.retain(|m| m.message_type != "reasoning");

        let current_tokens: i32 = summary_tokens
            + metadata
                .iter()
                .map(|m| m.token_count.unwrap_or(0))
                .sum::<i32>();

        if !self.compaction.should_compact(
            current_tokens as usize,
            DEFAULT_CONTEXT_WINDOW as usize,
            DEFAULT_COMPACTION_THRESHOLD,
        ) {
            return Ok(());
        }

        if metadata.len() <= MIN_MESSAGES_IN_CONTEXT {
            return Ok(());
        }

        // Summarize the oldest half, keep at least MIN_MESSAGES_IN_CONTEXT recent messages
        let keep_count = (metadata.len() / 2).max(MIN_MESSAGES_IN_CONTEXT);
        let to_summarize_count = metadata.len().saturating_sub(keep_count);
        if to_summarize_count == 0 {
            return Ok(());
        }

        let to_summarize: Vec<(String, i64)> = metadata
            .iter()
            .take(to_summarize_count)
            .map(|m| (m.message_type.clone(), m.id))
            .collect();

        let raw_messages =
            RawThreadMessage::get_messages_by_ids(&mut conn, self.conversation.id, &to_summarize)
                .map_err(|e| {
                error!("Failed to load messages for compaction: {e:?}");
                ApiError::InternalServerError
            })?;

        if raw_messages.is_empty() {
            return Ok(());
        }

        let mut formatted: Vec<String> = Vec::new();
        for m in &raw_messages {
            if m.message_type == "reasoning" {
                continue;
            }
            let role = match m.message_type.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                _ => "tool",
            };
            let content = self.render_raw_message(m)?;
            formatted.push(format!("[{}]: {}", role, content));
        }

        let new_messages = formatted.join("\n---\n");

        let summary = self
            .compaction
            .summarize(&self.lm, &summary_text, &new_messages)
            .await?;

        let content_tokens = count_tokens(&summary) as i32;
        let content_enc = encrypt_with_key(&self.user_key, summary.as_bytes()).await;

        // Embed summary for conversation_search over summaries
        let embedding = crate::web::get_embedding_vector(
            &self.state,
            self.user.as_ref(),
            AuthMethod::Jwt,
            crate::rag::DEFAULT_EMBEDDING_MODEL,
            &summary,
            Some(crate::rag::DEFAULT_EMBEDDING_DIM),
        )
        .await;

        let embedding_enc = match embedding {
            Ok((vec, _tok)) => {
                let bytes = serialize_f32_le(&vec);
                Some(encrypt_with_key(&self.user_key, &bytes).await)
            }
            Err(e) => {
                warn!("Failed to embed summary for conversation_search: {e:?}");
                None
            }
        };

        let from_created_at = raw_messages.first().unwrap().created_at;
        let to_created_at = raw_messages.last().unwrap().created_at;
        let previous_summary_id = latest_summary.as_ref().map(|s| s.id);

        let new_summary = NewConversationSummary {
            uuid: Uuid::new_v4(),
            user_id: self.user.uuid,
            conversation_id: self.conversation.id,
            from_created_at,
            to_created_at,
            message_count: raw_messages.len() as i32,
            content_enc,
            content_tokens,
            embedding_enc,
            previous_summary_id,
        };

        let _ = new_summary.insert(&mut conn).map_err(|e| {
            error!("Failed to insert conversation summary: {e:?}");
            ApiError::InternalServerError
        })?;

        Ok(())
    }
}
