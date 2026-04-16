//! Storage task components for persisting streaming response items.

use crate::{
    encrypt::encrypt_with_key,
    models::responses::{
        NewAssistantMessage, NewReasoningItem, NewToolCall, NewToolOutput, ResponseStatus,
    },
    tokens::count_tokens,
    web::responses::constants::{
        FINISH_REASON_CANCELLED, STATUS_COMPLETED, STATUS_INCOMPLETE, STATUS_IN_PROGRESS,
    },
    DBConnection,
};
use chrono::Utc;
use secp256k1::SecretKey;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;
use tracing::{debug, error, trace, warn};
use uuid::Uuid;

use super::handlers::StorageMessage;

#[derive(Default)]
struct PendingAssistantMessage {
    content: String,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Default)]
struct PendingReasoningItem {
    content: String,
}

fn clamp_token_count(text: &str, label: &str) -> i32 {
    let token_count = count_tokens(text);
    if token_count > i32::MAX as usize {
        warn!(
            "{} token count {} exceeds i32::MAX, clamping",
            label, token_count
        );
        i32::MAX
    } else {
        token_count as i32
    }
}

fn allocate_created_at(
    next_created_at: &mut chrono::DateTime<chrono::Utc>,
) -> chrono::DateTime<chrono::Utc> {
    let created_at = next_created_at.to_owned();
    *next_created_at = created_at + chrono::Duration::microseconds(1);
    created_at
}

fn create_assistant_message_if_missing(
    db: &Arc<dyn DBConnection + Send + Sync>,
    conversation_id: i64,
    response_id: i64,
    user_id: Uuid,
    item_id: Uuid,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Result<(), String> {
    match db.get_assistant_message_by_uuid(item_id) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => db
            .create_assistant_message(NewAssistantMessage {
                uuid: item_id,
                conversation_id,
                response_id: Some(response_id),
                user_id,
                content_enc: None,
                completion_tokens: 0,
                status: STATUS_IN_PROGRESS.to_string(),
                finish_reason: None,
                created_at,
            })
            .map(|_| ())
            .map_err(|e| format!("Failed to create assistant message: {:?}", e)),
        Err(e) => Err(format!("Failed to look up assistant message: {:?}", e)),
    }
}

async fn finalize_assistant_message(
    db: &Arc<dyn DBConnection + Send + Sync>,
    user_key: &SecretKey,
    item_id: Uuid,
    content: String,
    status: &str,
    finish_reason: Option<String>,
) -> Result<(), String> {
    let content_enc = if content.is_empty() {
        None
    } else {
        Some(encrypt_with_key(user_key, content.as_bytes()).await)
    };

    db.update_assistant_message(
        item_id,
        content_enc,
        clamp_token_count(&content, "assistant message"),
        status.to_string(),
        finish_reason,
    )
    .map(|_| ())
    .map_err(|e| format!("Failed to update assistant message: {:?}", e))
}

fn create_reasoning_item(
    db: &Arc<dyn DBConnection + Send + Sync>,
    conversation_id: i64,
    response_id: i64,
    user_id: Uuid,
    item_id: Uuid,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Result<(), String> {
    db.create_reasoning_item(NewReasoningItem {
        uuid: item_id,
        conversation_id,
        response_id: Some(response_id),
        assistant_message_id: None,
        user_id,
        content_enc: None,
        summary_enc: None,
        reasoning_tokens: 0,
        status: STATUS_IN_PROGRESS.to_string(),
        created_at,
    })
    .map(|_| ())
    .map_err(|e| format!("Failed to create reasoning item: {:?}", e))
}

async fn finalize_reasoning_item(
    db: &Arc<dyn DBConnection + Send + Sync>,
    user_key: &SecretKey,
    item_id: Uuid,
    content: String,
    status: &str,
) -> Result<(), String> {
    let content_enc = if content.is_empty() {
        None
    } else {
        Some(encrypt_with_key(user_key, content.as_bytes()).await)
    };

    db.update_reasoning_item(
        item_id,
        content_enc,
        clamp_token_count(&content, "reasoning"),
        status.to_string(),
    )
    .map(|_| ())
    .map_err(|e| format!("Failed to update reasoning item: {:?}", e))
}

#[allow(clippy::too_many_arguments)]
async fn persist_tool_call(
    db: &Arc<dyn DBConnection + Send + Sync>,
    user_key: &SecretKey,
    conversation_id: i64,
    response_id: i64,
    user_id: Uuid,
    tool_call_id: Uuid,
    name: String,
    arguments: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Result<(), String> {
    let arguments_json = serde_json::to_string(&arguments)
        .map_err(|e| format!("Failed to serialize tool arguments: {:?}", e))?;
    let arguments_enc = encrypt_with_key(user_key, arguments_json.as_bytes()).await;

    db.create_tool_call(NewToolCall {
        uuid: tool_call_id,
        conversation_id,
        response_id: Some(response_id),
        user_id,
        name,
        arguments_enc: Some(arguments_enc),
        argument_tokens: clamp_token_count(&arguments_json, "tool arguments"),
        status: STATUS_COMPLETED.to_string(),
        created_at,
    })
    .map(|_| ())
    .map_err(|e| format!("Failed to persist tool_call: {:?}", e))
}

#[allow(clippy::too_many_arguments)]
async fn persist_tool_output(
    db: &Arc<dyn DBConnection + Send + Sync>,
    user_key: &SecretKey,
    conversation_id: i64,
    response_id: i64,
    user_id: Uuid,
    tool_output_id: Uuid,
    tool_call_id: Uuid,
    output: String,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Result<(), String> {
    let tool_call_fk = db
        .get_tool_call_by_uuid(tool_call_id, user_id)
        .map_err(|e| format!("Tool call not found in database: {:?}", e))?
        .id;
    let output_enc = encrypt_with_key(user_key, output.as_bytes()).await;

    db.create_tool_output(NewToolOutput {
        uuid: tool_output_id,
        conversation_id,
        response_id: Some(response_id),
        user_id,
        tool_call_fk,
        output_enc,
        output_tokens: clamp_token_count(&output, "tool output"),
        status: STATUS_COMPLETED.to_string(),
        error: None,
        created_at,
    })
    .map(|_| ())
    .map_err(|e| format!("Failed to persist tool_output: {:?}", e))
}

async fn mark_pending_items_incomplete(
    db: &Arc<dyn DBConnection + Send + Sync>,
    user_key: &SecretKey,
    pending_messages: &mut HashMap<Uuid, PendingAssistantMessage>,
    pending_reasoning: &mut HashMap<Uuid, PendingReasoningItem>,
    message_finish_reason: Option<String>,
) {
    for (item_id, pending) in pending_messages.drain() {
        if let Err(e) = finalize_assistant_message(
            db,
            user_key,
            item_id,
            pending.content,
            STATUS_INCOMPLETE,
            message_finish_reason.clone(),
        )
        .await
        {
            error!(
                "Failed to finalize pending assistant message {}: {}",
                item_id, e
            );
        }
    }

    for (item_id, pending) in pending_reasoning.drain() {
        if let Err(e) =
            finalize_reasoning_item(db, user_key, item_id, pending.content, STATUS_INCOMPLETE).await
        {
            error!(
                "Failed to finalize pending reasoning item {}: {}",
                item_id, e
            );
        }
    }
}

/// Main storage task that orchestrates per-item persistence.
#[allow(clippy::too_many_arguments)]
pub async fn storage_task(
    mut rx: mpsc::Receiver<StorageMessage>,
    tool_persist_ack: Option<mpsc::Sender<Result<(), String>>>,
    db: Arc<dyn DBConnection + Send + Sync>,
    response_id: i64,
    first_item_created_at: chrono::DateTime<chrono::Utc>,
    conversation_id: i64,
    user_id: Uuid,
    user_key: SecretKey,
    _message_id: Uuid,
) {
    let tool_ack = tool_persist_ack;
    let mut pending_messages: HashMap<Uuid, PendingAssistantMessage> = HashMap::new();
    let mut pending_reasoning: HashMap<Uuid, PendingReasoningItem> = HashMap::new();
    let mut next_item_created_at = first_item_created_at;

    while let Some(msg) = rx.recv().await {
        match msg {
            StorageMessage::MessageStarted { item_id } => {
                trace!("Storage: message started {}", item_id);
                let pending = pending_messages.entry(item_id).or_default();
                let created_at = pending
                    .created_at
                    .get_or_insert_with(|| allocate_created_at(&mut next_item_created_at))
                    .to_owned();
                if let Err(e) = create_assistant_message_if_missing(
                    &db,
                    conversation_id,
                    response_id,
                    user_id,
                    item_id,
                    created_at,
                ) {
                    error!("{}", e);
                }
            }
            StorageMessage::ContentDelta { item_id, delta } => {
                trace!(
                    "Storage: content delta for {} ({} chars)",
                    item_id,
                    delta.len()
                );
                pending_messages
                    .entry(item_id)
                    .or_default()
                    .content
                    .push_str(&delta);
            }
            StorageMessage::MessageDone {
                item_id,
                finish_reason,
            } => {
                debug!(
                    "Storage: message done {} with finish_reason={}",
                    item_id, finish_reason
                );
                let pending = pending_messages.remove(&item_id).unwrap_or_default();
                let created_at = pending
                    .created_at
                    .unwrap_or_else(|| allocate_created_at(&mut next_item_created_at));
                if let Err(e) = create_assistant_message_if_missing(
                    &db,
                    conversation_id,
                    response_id,
                    user_id,
                    item_id,
                    created_at,
                ) {
                    error!("{}", e);
                }
                if let Err(e) = finalize_assistant_message(
                    &db,
                    &user_key,
                    item_id,
                    pending.content,
                    STATUS_COMPLETED,
                    Some(finish_reason),
                )
                .await
                {
                    error!("{}", e);
                }
            }
            StorageMessage::ReasoningStarted { item_id } => {
                trace!("Storage: reasoning started {}", item_id);
                pending_reasoning.entry(item_id).or_default();
                let created_at = allocate_created_at(&mut next_item_created_at);
                if let Err(e) = create_reasoning_item(
                    &db,
                    conversation_id,
                    response_id,
                    user_id,
                    item_id,
                    created_at,
                ) {
                    error!("{}", e);
                }
            }
            StorageMessage::ReasoningDelta { item_id, delta } => {
                trace!(
                    "Storage: reasoning delta for {} ({} chars)",
                    item_id,
                    delta.len()
                );
                pending_reasoning
                    .entry(item_id)
                    .or_default()
                    .content
                    .push_str(&delta);
            }
            StorageMessage::ReasoningDone { item_id } => {
                debug!("Storage: reasoning done {}", item_id);
                let pending = pending_reasoning.remove(&item_id).unwrap_or_default();
                if let Err(e) = finalize_reasoning_item(
                    &db,
                    &user_key,
                    item_id,
                    pending.content,
                    STATUS_COMPLETED,
                )
                .await
                {
                    error!("{}", e);
                }
            }
            StorageMessage::Usage { .. } => {
                trace!("Storage: usage message ignored for item persistence");
            }
            StorageMessage::ResponseDone { finish_reason } => {
                debug!(
                    "Storage: response done {} with finish_reason={}",
                    response_id, finish_reason
                );
                if let Err(e) = db.update_response_status(
                    response_id,
                    ResponseStatus::Completed,
                    Some(Utc::now()),
                ) {
                    error!("Failed to update response status to completed: {:?}", e);
                }
                return;
            }
            StorageMessage::Cancelled => {
                debug!(
                    "Storage: cancellation received for response {}",
                    response_id
                );
                if let Err(e) = db.update_response_status(
                    response_id,
                    ResponseStatus::Cancelled,
                    Some(Utc::now()),
                ) {
                    error!("Failed to update response status to cancelled: {:?}", e);
                }
                mark_pending_items_incomplete(
                    &db,
                    &user_key,
                    &mut pending_messages,
                    &mut pending_reasoning,
                    Some(FINISH_REASON_CANCELLED.to_string()),
                )
                .await;
                return;
            }
            StorageMessage::Error(error_msg) => {
                error!("Storage: received error: {}", error_msg);
                if let Err(e) =
                    db.update_response_status(response_id, ResponseStatus::Failed, Some(Utc::now()))
                {
                    error!("Failed to update response status to failed: {:?}", e);
                }
                mark_pending_items_incomplete(
                    &db,
                    &user_key,
                    &mut pending_messages,
                    &mut pending_reasoning,
                    None,
                )
                .await;
                return;
            }
            StorageMessage::ToolCall {
                tool_call_id,
                name,
                arguments,
            } => {
                trace!("Storage: persisting tool_call {}", tool_call_id);
                let created_at = allocate_created_at(&mut next_item_created_at);
                match persist_tool_call(
                    &db,
                    &user_key,
                    conversation_id,
                    response_id,
                    user_id,
                    tool_call_id,
                    name,
                    arguments,
                    created_at,
                )
                .await
                {
                    Ok(()) => debug!("Persisted tool_call {}", tool_call_id),
                    Err(e) => {
                        error!("{}", e);
                        if let Some(ack) = &tool_ack {
                            let _ = ack.send(Err(e)).await;
                        }
                    }
                }
            }
            StorageMessage::ToolOutput {
                tool_output_id,
                tool_call_id,
                output,
            } => {
                trace!("Storage: persisting tool_output {}", tool_output_id);
                let created_at = allocate_created_at(&mut next_item_created_at);
                match persist_tool_output(
                    &db,
                    &user_key,
                    conversation_id,
                    response_id,
                    user_id,
                    tool_output_id,
                    tool_call_id,
                    output,
                    created_at,
                )
                .await
                {
                    Ok(()) => {
                        debug!("Persisted tool_output {}", tool_output_id);
                        if let Some(ack) = &tool_ack {
                            let _ = ack.send(Ok(())).await;
                        }
                    }
                    Err(e) => {
                        error!("{}", e);
                        if let Some(ack) = &tool_ack {
                            let _ = ack.send(Err(e)).await;
                        }
                    }
                }
            }
        }
    }

    warn!("Storage channel closed before receiving ResponseDone signal");
    if let Err(e) = db.update_response_status(response_id, ResponseStatus::Failed, Some(Utc::now()))
    {
        error!(
            "Failed to update response status after premature channel close: {:?}",
            e
        );
    }
    mark_pending_items_incomplete(
        &db,
        &user_key,
        &mut pending_messages,
        &mut pending_reasoning,
        None,
    )
    .await;
}
