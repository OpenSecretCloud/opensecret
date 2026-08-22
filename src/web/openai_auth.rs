use crate::jwt::{validate_access_token_for_auth, AuthContext, USER_ACCESS};
use crate::ApiError;
use axum::{
    body::Body,
    extract::State,
    http::{header, Request},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuthMethod {
    Jwt,
    ApiKey,
}

pub async fn validate_openai_auth(
    State(data): State<Arc<crate::AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // Extract Authorization header
    if let Some(auth_header) = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        // Check if it's a UUID (API key) or JWT (contains dots)
        if !auth_header.contains('.') {
            // Try to parse as UUID (API key)
            if let Ok(api_key_uuid) = Uuid::parse_str(auth_header) {
                // Hash the API key (UUID string with dashes)
                let mut hasher = Sha256::new();
                hasher.update(api_key_uuid.to_string().as_bytes()); // to_string() includes dashes
                let key_hash = format!("{:x}", hasher.finalize());

                // Look up the API key in database
                let db_result = data.db.get_user_by_api_key_hash(&key_hash);

                match db_result {
                    Ok(Some(user)) => {
                        req.extensions_mut().insert(user);
                        req.extensions_mut().insert(AuthMethod::ApiKey);
                        return next.run(req).await;
                    }
                    Ok(None) => {
                        tracing::debug!("API key not found in database");
                        return ApiError::Unauthorized.into_response();
                    }
                    Err(e) => {
                        tracing::error!("Database error during API key lookup: {:?}", e);
                        return ApiError::InternalServerError.into_response();
                    }
                }
            }
        }
    }

    // Fall back to JWT validation
    // Try to validate as JWT
    let token = match req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|auth_header| auth_header.to_str().ok())
        .and_then(|auth_value| auth_value.strip_prefix("Bearer ").map(ToString::to_string))
    {
        Some(token) => token,
        None => return ApiError::InvalidJwt.into_response(),
    };

    let (claims, access_token_expired) =
        match validate_access_token_for_auth(&token, &data, USER_ACCESS) {
            Ok(validation) => validation,
            Err(_) => return ApiError::InvalidJwt.into_response(),
        };

    let auth_context = match AuthContext::from_claims(&claims) {
        Ok(auth_context) => auth_context,
        Err(_) => return ApiError::InvalidJwt.into_response(),
    };

    let user_uuid: Uuid = match Uuid::parse_str(&claims.sub) {
        Ok(uuid) => uuid,
        Err(e) => {
            tracing::error!("Error parsing user uuid: {:?}", e);
            return ApiError::InvalidJwt.into_response();
        }
    };

    let user = match data.get_user(user_uuid).await {
        Ok(user) => user,
        Err(e) => {
            tracing::error!("Error getting user: {:?}", e);
            return ApiError::InternalServerError.into_response();
        }
    };

    if user.project_id != auth_context.project_id {
        tracing::error!("JWT auth context project does not match user project");
        return ApiError::InvalidJwt.into_response();
    }

    if let Err(e) = data.verify_seed_wrap_for_auth_context(&user, &auth_context) {
        tracing::error!(
            "OpenAI JWT auth context no longer unwraps an active seed wrap: {:?}",
            e
        );
        return ApiError::InvalidJwt.into_response();
    }

    if access_token_expired {
        return ApiError::AccessTokenExpired.into_response();
    }

    req.extensions_mut().insert(auth_context);
    req.extensions_mut().insert(user);
    req.extensions_mut().insert(AuthMethod::Jwt);
    next.run(req).await
}

/// Authenticate the dedicated models route when current OpenAI credentials
/// are present. The route's attested encryption session is validated
/// independently by its encryption middleware.
pub async fn validate_optional_openai_auth(
    state: State<Arc<crate::AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if !req.headers().contains_key(header::AUTHORIZATION) {
        return next.run(req).await;
    }

    validate_openai_auth(state, req, next).await
}
