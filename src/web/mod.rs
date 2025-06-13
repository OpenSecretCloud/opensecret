pub mod attestation_routes;
mod documents;
pub mod encryption_middleware;
mod health_routes;
pub mod login_routes;
mod oauth_routes;
mod openai;
pub mod platform;
pub mod protected_routes;

pub use documents::router as document_routes;
pub use health_routes::router as health_routes;
pub use login_routes::router as login_routes;
pub use oauth_routes::router as oauth_routes;
pub use openai::router as openai_routes;
pub use platform::router as platform_routes;
pub use protected_routes::router as protected_routes;

pub use platform::login_routes as platform_login_routes;
