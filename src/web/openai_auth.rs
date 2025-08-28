use crate::jwt::{validate_token, USER_ACCESS};
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

    let claims = match validate_token(&token, &data, USER_ACCESS) {
        Ok(claims) => claims,
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

    req.extensions_mut().insert(user);
    req.extensions_mut().insert(AuthMethod::Jwt);
    next.run(req).await
}
