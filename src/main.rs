// [Previous imports remain the same...]

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Invalid email, password, or login method")]
    InvalidUsernameOrPassword,

    #[error("Invalid JWT")]
    InvalidJwt,

    #[error("Internal server error")]
    InternalServerError,

    #[error("Bad Request")]
    BadRequest,

    #[error("Encryption error")]
    EncryptionError,

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Invalid invite code")]
    InvalidInviteCode,

    #[error("Invite has expired")]
    InviteExpired,

    #[error("Token refresh failed")]
    RefreshFailed,

    #[error("User is already verified")]
    UserAlreadyVerified,

    #[error("No valid email found for the Oauth account")]
    NoEmailFound,

    #[error("User exists but Oauth not linked")]
    UserExistsNotLinked,

    #[error("User not found")]
    UserNotFound,

    #[error("Email already registered")]
    EmailAlreadyExists,

    #[error("Usage limit reached")]
    UsageLimitReached,

    #[error("Resource not found")]
    NotFound,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            ApiError::InvalidUsernameOrPassword => StatusCode::UNAUTHORIZED,
            ApiError::InvalidJwt => StatusCode::UNAUTHORIZED,
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::InternalServerError => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::BadRequest => StatusCode::BAD_REQUEST,
            ApiError::InvalidInviteCode => StatusCode::UNAUTHORIZED,
            ApiError::InviteExpired => StatusCode::GONE,
            ApiError::RefreshFailed => StatusCode::UNAUTHORIZED,
            ApiError::UserAlreadyVerified => StatusCode::BAD_REQUEST,
            ApiError::EncryptionError => StatusCode::BAD_REQUEST,
            ApiError::NoEmailFound => StatusCode::BAD_REQUEST,
            ApiError::UserExistsNotLinked => StatusCode::CONFLICT,
            ApiError::UserNotFound => StatusCode::NOT_FOUND,
            ApiError::EmailAlreadyExists => StatusCode::CONFLICT,
            ApiError::UsageLimitReached => StatusCode::FORBIDDEN,
            ApiError::NotFound => StatusCode::NOT_FOUND,
        };
        (
            status,
            Json(ErrorResponse {
                status: status.as_u16(),
                message: self.to_string(),
            }),
        )
            .into_response()
    }
}

// [Rest of the file remains the same...]