// [Previous imports and code remain the same until the accept_invite function]

async fn accept_invite(
    State(data): State<Arc<AppState>>,
    Extension(platform_user): Extension<PlatformUser>,
    Path(code): Path<Uuid>,
    Extension(session_id): Extension<Uuid>,
) -> Result<Json<EncryptedResponse<serde_json::Value>>, ApiError> {
    debug!("Accepting invite");

    // Get and validate the invite code
    let invite = data
        .db
        .get_invite_code_by_code(code)
        .map_err(|_| ApiError::NotFound)?;

    if invite.used {
        error!("Attempt to accept already used invite code: {}", code);
        return Err(ApiError::BadRequest);
    }

    if invite.expires_at < chrono::Utc::now() {
        error!("Attempt to accept expired invite code: {}", code);
        return Err(ApiError::InviteExpired);
    }

    if invite.email != platform_user.email {
        error!("Unauthorized attempt to accept invite code: {}", code);
        return Err(ApiError::Unauthorized);
    }

    // Create the membership and mark invite as used in a single transaction
    let new_membership = NewOrgMembership::new(
        platform_user.uuid,
        invite.org_id,
        invite.role.clone().into(),
    );
    data.db
        .accept_invite_transaction(&invite, new_membership)
        .map_err(|e| {
            error!("Failed to accept invite: {:?}", e);
            match e {
                DBError::InviteCodeError(InviteCodeError::AlreadyUsed) => ApiError::BadRequest,
                DBError::InviteCodeError(InviteCodeError::Expired) => ApiError::InviteExpired,
                _ => ApiError::InternalServerError,
            }
        })?;

    let response = serde_json::json!({
        "message": "Invite accepted successfully"
    });

    encrypt_response(&data, &session_id, &response).await
}

// [Rest of the file remains the same]