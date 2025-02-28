use crate::models::enclave_secrets::{EnclaveSecret, EnclaveSecretError, NewEnclaveSecret};
use crate::models::invite_codes::{InviteCode, InviteCodeError, NewInviteCode};
use crate::models::oauth::{
    NewOAuthProvider, NewUserOAuthConnection, OAuthError, OAuthProvider, UserOAuthConnection,
};
use crate::models::org_memberships::NewOrgMembership;
use crate::models::org_memberships::{OrgMembership, OrgMembershipError, OrgMembershipWithUser};
use crate::models::org_project_secrets::{
    NewOrgProjectSecret, OrgProjectSecret, OrgProjectSecretError,
};
use crate::models::org_projects::{NewOrgProject, OrgProject, OrgProjectError};
use crate::models::orgs::{NewOrg, Org, OrgError};
use crate::models::password_reset::{
    NewPasswordResetRequest, PasswordResetError, PasswordResetRequest,
};
use crate::models::platform_email_verification::{
    NewPlatformEmailVerification, PlatformEmailVerification, PlatformEmailVerificationError,
};
use crate::models::platform_users::{NewPlatformUser, PlatformUser, PlatformUserError};
use crate::models::project_settings::OAuthSettings;
use crate::models::project_settings::{
    EmailSettings, NewProjectSetting, ProjectSetting, ProjectSettingError, SettingCategory,
};
use crate::models::token_usage::{NewTokenUsage, TokenUsage, TokenUsageError};
use crate::models::users::{NewUser, User, UserError};
use crate::models::{
    email_verification::{EmailVerification, EmailVerificationError, NewEmailVerification},
    org_memberships::OrgRole,
};
use diesel::Connection;
use diesel::{
    pg::PgConnection,
    r2d2::{ConnectionManager, Pool},
};
use std::sync::Arc;
use tracing::{debug, error, info};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum DBError {
    #[error("Database connection error")]
    ConnectionError,
    #[error("Database query error: {0}")]
    QueryError(#[from] diesel::result::Error),
    #[error("User error: {0}")]
    UserError(#[from] UserError),
    #[error("User not found")]
    UserNotFound,
    #[error("Enclave secret error: {0}")]
    EnclaveSecretError(#[from] EnclaveSecretError),
    #[error("Email verification error: {0}")]
    EmailVerificationError(#[from] EmailVerificationError),
    #[error("Email verification not found")]
    EmailVerificationNotFound,
    #[error("Password reset error: {0}")]
    PasswordResetError(#[from] PasswordResetError),
    #[error("Password reset request not found")]
    PasswordResetRequestNotFound,
    #[error("Encryption error: {0}")]
    EncryptionError(#[from] crate::encrypt::EncryptError),
    #[error("OAuth error: {0}")]
    OAuthError(#[from] OAuthError),
    #[error("Token usage error: {0}")]
    TokenUsageError(#[from] TokenUsageError),
    #[error("Org error: {0}")]
    OrgError(#[from] OrgError),
    #[error("Org not found")]
    OrgNotFound,
    #[error("Org project error: {0}")]
    OrgProjectError(#[from] OrgProjectError),
    #[error("Org project not found")]
    OrgProjectNotFound,
    #[error("Org project secret error: {0}")]
    OrgProjectSecretError(#[from] OrgProjectSecretError),
    #[error("Org project secret not found")]
    OrgProjectSecretNotFound,
    #[error("Invite code error: {0}")]
    InviteCodeError(#[from] InviteCodeError),
    #[error("Invite code not found")]
    InviteCodeNotFound,
    #[error("Platform user error: {0}")]
    PlatformUserError(#[from] PlatformUserError),
    #[error("Platform user not found")]
    PlatformUserNotFound,
    #[error("Platform email verification error: {0}")]
    PlatformEmailVerificationError(#[from] PlatformEmailVerificationError),
    #[error("Platform email verification not found")]
    PlatformEmailVerificationNotFound,
    #[error("Org membership error: {0}")]
    OrgMembershipError(#[from] OrgMembershipError),
    #[error("Org membership not found")]
    OrgMembershipNotFound,
    #[error("Project setting error: {0}")]
    ProjectSettingError(#[from] ProjectSettingError),
    #[error("Project setting not found")]
    ProjectSettingNotFound,
}

#[allow(dead_code)]
pub trait DBConnection {
    fn create_user(&self, new_user: NewUser) -> Result<User, DBError>;
    fn get_user_by_uuid(&self, uuid: Uuid) -> Result<User, DBError>;
    fn get_user_by_email(&self, email: String, project_id: i32) -> Result<User, DBError>;
    fn set_user_key(&self, user: User, private_key: Vec<u8>) -> Result<(), DBError>;
    fn get_pool(&self) -> &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>;
    fn create_enclave_secret(&self, new_secret: NewEnclaveSecret)
        -> Result<EnclaveSecret, DBError>;
    fn get_enclave_secret_by_id(&self, id: i32) -> Result<Option<EnclaveSecret>, DBError>;
    fn get_enclave_secret_by_key(&self, key: &str) -> Result<Option<EnclaveSecret>, DBError>;
    fn get_all_enclave_secrets(&self) -> Result<Vec<EnclaveSecret>, DBError>;
    fn update_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError>;
    fn delete_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError>;
    fn create_email_verification(
        &self,
        new_verification: NewEmailVerification,
    ) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_id(&self, id: i32) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_user_id(
        &self,
        user_id: Uuid,
    ) -> Result<EmailVerification, DBError>;
    fn get_email_verification_by_code(&self, code: Uuid) -> Result<EmailVerification, DBError>;
    fn update_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError>;
    fn delete_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError>;
    fn verify_email(&self, verification: &mut EmailVerification) -> Result<(), DBError>;
    fn create_password_reset_request(
        &self,
        new_request: NewPasswordResetRequest,
    ) -> Result<PasswordResetRequest, DBError>;
    fn get_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PasswordResetRequest>, DBError>;
    fn update_user_password(
        &self,
        user: &User,
        new_password_enc: Option<Vec<u8>>,
    ) -> Result<(), DBError>;
    fn mark_password_reset_as_complete(
        &self,
        request: &PasswordResetRequest,
    ) -> Result<(), DBError>;

    // OAuth Provider methods
    fn create_oauth_provider(
        &self,
        new_provider: NewOAuthProvider,
    ) -> Result<OAuthProvider, DBError>;
    fn get_oauth_provider_by_id(&self, id: i32) -> Result<Option<OAuthProvider>, DBError>;
    fn get_oauth_provider_by_name(&self, name: &str) -> Result<Option<OAuthProvider>, DBError>;
    fn get_all_oauth_providers(&self) -> Result<Vec<OAuthProvider>, DBError>;
    fn update_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError>;
    fn delete_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError>;

    // User OAuth Connection methods
    fn create_user_oauth_connection(
        &self,
        new_connection: NewUserOAuthConnection,
    ) -> Result<UserOAuthConnection, DBError>;
    fn get_user_oauth_connection_by_id(
        &self,
        id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError>;
    fn get_user_oauth_connection_by_user_and_provider(
        &self,
        user_id: Uuid,
        provider_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError>;
    fn get_all_user_oauth_connections_for_user(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<UserOAuthConnection>, DBError>;
    fn update_user_oauth_connection(&self, connection: &UserOAuthConnection)
        -> Result<(), DBError>;
    fn delete_user_oauth_connection(&self, connection: &UserOAuthConnection)
        -> Result<(), DBError>;

    fn create_token_usage(&self, new_usage: NewTokenUsage) -> Result<TokenUsage, DBError>;

    fn update_user(&self, user: &User) -> Result<(), DBError>;

    // New org-related methods
    fn create_org(&self, new_org: NewOrg) -> Result<Org, DBError>;
    fn get_org_by_id(&self, id: i32) -> Result<Org, DBError>;
    fn get_org_by_uuid(&self, uuid: Uuid) -> Result<Org, DBError>;
    fn get_org_by_name(&self, name: &str) -> Result<Option<Org>, DBError>;
    fn get_all_orgs(&self) -> Result<Vec<Org>, DBError>;
    fn update_org(&self, org: &Org) -> Result<(), DBError>;
    fn delete_org(&self, org: &Org) -> Result<(), DBError>;

    // Org project methods
    fn create_org_project(&self, new_project: NewOrgProject) -> Result<OrgProject, DBError>;
    fn get_org_project_by_id(&self, id: i32) -> Result<OrgProject, DBError>;
    fn get_org_project_by_uuid(&self, uuid: Uuid) -> Result<OrgProject, DBError>;
    fn get_org_project_by_client_id(&self, client_id: Uuid) -> Result<OrgProject, DBError>;
    fn get_org_project_by_name_and_org(
        &self,
        name: &str,
        org_id: i32,
    ) -> Result<Option<OrgProject>, DBError>;
    fn get_all_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError>;
    fn get_active_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError>;
    fn update_org_project(&self, project: &OrgProject) -> Result<(), DBError>;
    fn delete_org_project(&self, project: &OrgProject) -> Result<(), DBError>;

    // Org project secret methods
    fn create_org_project_secret(
        &self,
        new_secret: NewOrgProjectSecret,
    ) -> Result<OrgProjectSecret, DBError>;
    fn get_org_project_secret_by_id(&self, id: i32) -> Result<OrgProjectSecret, DBError>;
    fn get_org_project_secret_by_key_name_and_project(
        &self,
        key_name: &str,
        project_id: i32,
    ) -> Result<Option<OrgProjectSecret>, DBError>;
    fn get_all_org_project_secrets_for_project(
        &self,
        project_id: i32,
    ) -> Result<Vec<OrgProjectSecret>, DBError>;
    fn update_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError>;
    fn delete_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError>;

    // Invite code methods
    fn create_invite_code(&self, new_invite: NewInviteCode) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_id(&self, id: i32) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_code(&self, code: Uuid) -> Result<InviteCode, DBError>;
    fn get_invite_code_by_email_and_org(
        &self,
        email: &str,
        org_id: i32,
    ) -> Result<Option<InviteCode>, DBError>;
    fn get_all_invite_codes_for_org(&self, org_id: i32) -> Result<Vec<InviteCode>, DBError>;
    fn mark_invite_code_as_used(&self, invite: &InviteCode) -> Result<(), DBError>;
    fn update_invite_code(&self, invite: &InviteCode) -> Result<(), DBError>;
    fn delete_invite_code(&self, invite: &InviteCode) -> Result<(), DBError>;

    // Platform user methods
    fn create_platform_user(&self, new_user: NewPlatformUser) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_id(&self, id: i32) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_uuid(&self, uuid: Uuid) -> Result<PlatformUser, DBError>;
    fn get_platform_user_by_email(&self, email: &str) -> Result<Option<PlatformUser>, DBError>;
    fn update_platform_user(&self, user: &PlatformUser) -> Result<(), DBError>;
    fn update_platform_user_password(
        &self,
        user: &PlatformUser,
        new_password_enc: Vec<u8>,
    ) -> Result<(), DBError>;

    // Org membership methods
    fn create_org_membership(
        &self,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError>;
    fn get_org_membership_by_platform_user_and_org(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembership, DBError>;

    fn get_org_membership_by_platform_user_and_org_with_user(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembershipWithUser, DBError>;
    fn get_all_org_memberships_for_platform_user(
        &self,
        platform_user_id: Uuid,
    ) -> Result<Vec<OrgMembership>, DBError>;
    fn get_all_org_memberships_for_org(&self, org_id: i32) -> Result<Vec<OrgMembership>, DBError>;
    fn get_all_org_memberships_with_users_for_org(
        &self,
        org_id: i32,
    ) -> Result<Vec<OrgMembershipWithUser>, DBError>;
    fn update_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError>;
    fn delete_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError>;
    fn update_membership_role(
        &self,
        membership: &mut OrgMembership,
        new_role: OrgRole,
    ) -> Result<(), DBError>;
    fn delete_membership_with_owner_check(&self, membership: &OrgMembership)
        -> Result<(), DBError>;

    // project-scoped methods
    fn get_users_for_project(
        &self,
        project_id: i32,
        page: Option<i64>,
        per_page: Option<i64>,
    ) -> Result<(Vec<User>, i64), DBError>;

    fn create_org_with_owner(&self, new_org: NewOrg, owner_id: Uuid) -> Result<Org, DBError>;

    fn accept_invite_transaction(
        &self,
        invite: &InviteCode,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError>;

    // Project settings methods
    fn get_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
    ) -> Result<Option<ProjectSetting>, DBError>;

    fn update_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
        settings: serde_json::Value,
    ) -> Result<ProjectSetting, DBError>;

    fn get_project_email_settings(&self, project_id: i32)
        -> Result<Option<EmailSettings>, DBError>;

    fn update_project_email_settings(
        &self,
        project_id: i32,
        settings: EmailSettings,
    ) -> Result<ProjectSetting, DBError>;

    fn get_project_oauth_settings(&self, project_id: i32)
        -> Result<Option<OAuthSettings>, DBError>;

    fn update_project_oauth_settings(
        &self,
        project_id: i32,
        settings: OAuthSettings,
    ) -> Result<ProjectSetting, DBError>;

    // Platform email verification methods
    fn create_platform_email_verification(
        &self,
        new_verification: NewPlatformEmailVerification,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_id(
        &self,
        id: i32,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_platform_user_id(
        &self,
        platform_user_id: Uuid,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn get_platform_email_verification_by_code(
        &self,
        code: Uuid,
    ) -> Result<PlatformEmailVerification, DBError>;

    fn update_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError>;

    fn delete_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError>;

    fn verify_platform_email(
        &self,
        verification: &mut PlatformEmailVerification,
    ) -> Result<(), DBError>;
}

pub(crate) struct PostgresConnection {
    db: Pool<ConnectionManager<PgConnection>>,
}

impl DBConnection for PostgresConnection {
    fn create_user(&self, new_user: NewUser) -> Result<User, DBError> {
        debug!("Creating new user");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_user.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create user: {:?}", e);
        }
        result
    }

    fn get_user_by_uuid(&self, uuid: Uuid) -> Result<User, DBError> {
        debug!("Getting user by UUID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = User::get_by_uuid(conn, uuid)?.ok_or(DBError::UserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get user by UUID: {:?}", e);
        }
        result
    }

    fn get_user_by_email(&self, email: String, project_id: i32) -> Result<User, DBError> {
        debug!("Getting user by email and project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = User::get_by_email(conn, email, project_id)?.ok_or(DBError::UserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get user by email: {:?}", e);
        }
        result
    }

    fn set_user_key(&self, user: User, private_key: Vec<u8>) -> Result<(), DBError> {
        debug!("Setting user key");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user.set_key(conn, private_key).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to set user key: {:?}", e);
        }
        result
    }

    fn get_pool(&self) -> &diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>> {
        &self.db
    }

    fn create_enclave_secret(
        &self,
        new_secret: NewEnclaveSecret,
    ) -> Result<EnclaveSecret, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_secret.insert(conn).map_err(DBError::from)
    }

    fn get_enclave_secret_by_id(&self, id: i32) -> Result<Option<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_enclave_secret_by_key(&self, key: &str) -> Result<Option<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_by_key(conn, key).map_err(DBError::from)
    }

    fn get_all_enclave_secrets(&self) -> Result<Vec<EnclaveSecret>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EnclaveSecret::get_all(conn).map_err(DBError::from)
    }

    fn update_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        secret.update(conn).map_err(DBError::from)
    }

    fn delete_enclave_secret(&self, secret: &EnclaveSecret) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        secret.delete(conn).map_err(DBError::from)
    }

    fn create_email_verification(
        &self,
        new_verification: NewEmailVerification,
    ) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_verification.insert(conn).map_err(DBError::from)
    }

    fn get_email_verification_by_id(&self, id: i32) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_id(conn, id)?.ok_or(DBError::EmailVerificationNotFound)
    }

    fn get_email_verification_by_user_id(
        &self,
        user_id: Uuid,
    ) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_user_id(conn, user_id)?.ok_or(DBError::EmailVerificationNotFound)
    }

    fn get_email_verification_by_code(&self, code: Uuid) -> Result<EmailVerification, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        EmailVerification::get_by_verification_code(conn, code)?
            .ok_or(DBError::EmailVerificationNotFound)
    }

    fn update_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        verification.update(conn).map_err(DBError::from)
    }

    fn delete_email_verification(&self, verification: &EmailVerification) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        verification.delete(conn).map_err(DBError::from)
    }

    fn verify_email(&self, verification: &mut EmailVerification) -> Result<(), DBError> {
        debug!("Verifying email");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.verify(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to verify email: {:?}", e);
        }
        result
    }

    fn create_password_reset_request(
        &self,
        new_request: NewPasswordResetRequest,
    ) -> Result<PasswordResetRequest, DBError> {
        debug!("Creating new password reset request");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_request.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create password reset request: {:?}", e);
        }
        result
    }

    fn get_password_reset_request_by_user_id_and_code(
        &self,
        user_id: Uuid,
        encrypted_code: Vec<u8>,
    ) -> Result<Option<PasswordResetRequest>, DBError> {
        debug!("Getting password reset request by user_id and encrypted code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PasswordResetRequest::get_by_user_id_and_code(conn, user_id, &encrypted_code)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get password reset request: {:?}", e);
        }
        result
    }

    fn update_user_password(
        &self,
        user: &User,
        new_password_enc: Option<Vec<u8>>,
    ) -> Result<(), DBError> {
        debug!("Updating user password");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user
            .update_password(conn, new_password_enc)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update user password: {:?}", e);
        }
        result
    }

    fn mark_password_reset_as_complete(
        &self,
        request: &PasswordResetRequest,
    ) -> Result<(), DBError> {
        debug!("Marking password reset request as complete");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = request.mark_as_reset(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to mark password reset request as complete: {:?}", e);
        }
        result
    }

    // OAuth Provider method implementations
    fn create_oauth_provider(
        &self,
        new_provider: NewOAuthProvider,
    ) -> Result<OAuthProvider, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_provider.insert(conn).map_err(DBError::from)
    }

    fn get_oauth_provider_by_id(&self, id: i32) -> Result<Option<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_oauth_provider_by_name(&self, name: &str) -> Result<Option<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_by_name(conn, name).map_err(DBError::from)
    }

    fn get_all_oauth_providers(&self) -> Result<Vec<OAuthProvider>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        OAuthProvider::get_all(conn).map_err(DBError::from)
    }

    fn update_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        provider.update(conn).map_err(DBError::from)
    }

    fn delete_oauth_provider(&self, provider: &OAuthProvider) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        provider.delete(conn).map_err(DBError::from)
    }

    // User OAuth Connection method implementations
    fn create_user_oauth_connection(
        &self,
        new_connection: NewUserOAuthConnection,
    ) -> Result<UserOAuthConnection, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_connection.insert(conn).map_err(DBError::from)
    }

    fn get_user_oauth_connection_by_id(
        &self,
        id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_by_id(conn, id).map_err(DBError::from)
    }

    fn get_user_oauth_connection_by_user_and_provider(
        &self,
        user_id: Uuid,
        provider_id: i32,
    ) -> Result<Option<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_by_user_and_provider(conn, user_id, provider_id)
            .map_err(DBError::from)
    }

    fn get_all_user_oauth_connections_for_user(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<UserOAuthConnection>, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        UserOAuthConnection::get_all_for_user(conn, user_id).map_err(DBError::from)
    }

    fn update_user_oauth_connection(
        &self,
        connection: &UserOAuthConnection,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        connection.update(conn).map_err(DBError::from)
    }

    fn delete_user_oauth_connection(
        &self,
        connection: &UserOAuthConnection,
    ) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        connection.delete(conn).map_err(DBError::from)
    }

    fn create_token_usage(&self, new_usage: NewTokenUsage) -> Result<TokenUsage, DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        new_usage.insert(conn).map_err(DBError::from)
    }

    fn update_user(&self, user: &User) -> Result<(), DBError> {
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        user.update(conn).map_err(DBError::from)
    }

    // Org implementations
    fn create_org(&self, new_org: NewOrg) -> Result<Org, DBError> {
        debug!("Creating new org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_org.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create org: {:?}", e);
        }
        result
    }

    fn get_org_by_id(&self, id: i32) -> Result<Org, DBError> {
        debug!("Getting org by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_id(conn, id)?.ok_or(DBError::OrgNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org by ID: {:?}", e);
        }
        result
    }

    fn get_org_by_uuid(&self, uuid: Uuid) -> Result<Org, DBError> {
        debug!("Getting org by UUID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_uuid(conn, uuid)?.ok_or(DBError::OrgNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org by UUID: {:?}", e);
        }
        result
    }

    fn get_org_by_name(&self, name: &str) -> Result<Option<Org>, DBError> {
        debug!("Getting org by name");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_by_name(conn, name).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get org by name: {:?}", e);
        }
        result
    }

    fn get_all_orgs(&self) -> Result<Vec<Org>, DBError> {
        debug!("Getting all orgs");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = Org::get_all(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all orgs: {:?}", e);
        }
        result
    }

    fn update_org(&self, org: &Org) -> Result<(), DBError> {
        debug!("Updating org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = org.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org: {:?}", e);
        }
        result
    }

    fn delete_org(&self, org: &Org) -> Result<(), DBError> {
        debug!("Deleting org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = org.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org: {:?}", e);
        }
        result
    }

    // Org project implementations
    fn create_org_project(&self, new_project: NewOrgProject) -> Result<OrgProject, DBError> {
        debug!("Creating new org project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_project.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create org project: {:?}", e);
        }
        result
    }

    fn get_org_project_by_id(&self, id: i32) -> Result<OrgProject, DBError> {
        debug!("Getting org project by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_id(conn, id)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by ID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_uuid(&self, uuid: Uuid) -> Result<OrgProject, DBError> {
        debug!("Getting org project by UUID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_uuid(conn, uuid)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by UUID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_client_id(&self, client_id: Uuid) -> Result<OrgProject, DBError> {
        debug!("Getting org project by client ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgProject::get_by_client_id(conn, client_id)?.ok_or(DBError::OrgProjectNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project by client ID: {:?}", e);
        }
        result
    }

    fn get_org_project_by_name_and_org(
        &self,
        name: &str,
        org_id: i32,
    ) -> Result<Option<OrgProject>, DBError> {
        debug!("Getting org project by name and org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_by_name_and_org(conn, name, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get org project by name and org: {:?}", e);
        }
        result
    }

    fn get_all_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError> {
        debug!("Getting all org projects for org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org projects for org: {:?}", e);
        }
        result
    }

    fn get_active_org_projects_for_org(&self, org_id: i32) -> Result<Vec<OrgProject>, DBError> {
        debug!("Getting active org projects for org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProject::get_active_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get active org projects for org: {:?}", e);
        }
        result
    }

    fn update_org_project(&self, project: &OrgProject) -> Result<(), DBError> {
        debug!("Updating org project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = project.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org project: {:?}", e);
        }
        result
    }

    fn delete_org_project(&self, project: &OrgProject) -> Result<(), DBError> {
        debug!("Deleting org project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = project.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org project: {:?}", e);
        }
        result
    }

    // Org project secret implementations
    fn create_org_project_secret(
        &self,
        new_secret: NewOrgProjectSecret,
    ) -> Result<OrgProjectSecret, DBError> {
        debug!("Creating new org project secret");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_secret.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create org project secret: {:?}", e);
        }
        result
    }

    fn get_org_project_secret_by_id(&self, id: i32) -> Result<OrgProjectSecret, DBError> {
        debug!("Getting org project secret by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgProjectSecret::get_by_id(conn, id)?.ok_or(DBError::OrgProjectSecretNotFound);
        if let Err(ref e) = result {
            error!("Failed to get org project secret by ID: {:?}", e);
        }
        result
    }

    fn get_org_project_secret_by_key_name_and_project(
        &self,
        key_name: &str,
        project_id: i32,
    ) -> Result<Option<OrgProjectSecret>, DBError> {
        debug!("Getting org project secret by key name and project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProjectSecret::get_by_key_name_and_project(conn, key_name, project_id)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org project secret by key name and project: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_project_secrets_for_project(
        &self,
        project_id: i32,
    ) -> Result<Vec<OrgProjectSecret>, DBError> {
        debug!("Getting all org project secrets for project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgProjectSecret::get_all_for_project(conn, project_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org project secrets for project: {:?}", e);
        }
        result
    }

    fn update_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError> {
        debug!("Updating org project secret");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = secret.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org project secret: {:?}", e);
        }
        result
    }

    fn delete_org_project_secret(&self, secret: &OrgProjectSecret) -> Result<(), DBError> {
        debug!("Deleting org project secret");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = secret.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org project secret: {:?}", e);
        }
        result
    }

    // Invite code implementations
    fn create_invite_code(&self, new_invite: NewInviteCode) -> Result<InviteCode, DBError> {
        debug!("Creating new invite code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_invite.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create invite code: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_id(&self, id: i32) -> Result<InviteCode, DBError> {
        debug!("Getting invite code by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_id(conn, id)?.ok_or(DBError::InviteCodeNotFound);
        if let Err(ref e) = result {
            error!("Failed to get invite code by ID: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_code(&self, code: Uuid) -> Result<InviteCode, DBError> {
        debug!("Getting invite code by code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_code(conn, code)?.ok_or(DBError::InviteCodeNotFound);
        if let Err(ref e) = result {
            error!("Failed to get invite code by code: {:?}", e);
        }
        result
    }

    fn get_invite_code_by_email_and_org(
        &self,
        email: &str,
        org_id: i32,
    ) -> Result<Option<InviteCode>, DBError> {
        debug!("Getting invite code by email and org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_by_email_and_org(conn, email, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get invite code by email and org: {:?}", e);
        }
        result
    }

    fn get_all_invite_codes_for_org(&self, org_id: i32) -> Result<Vec<InviteCode>, DBError> {
        debug!("Getting all invite codes for org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = InviteCode::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all invite codes for org: {:?}", e);
        }
        result
    }

    fn mark_invite_code_as_used(&self, invite: &InviteCode) -> Result<(), DBError> {
        debug!("Marking invite code as used");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.mark_as_used(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to mark invite code as used: {:?}", e);
        }
        result
    }

    fn update_invite_code(&self, invite: &InviteCode) -> Result<(), DBError> {
        debug!("Updating invite code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update invite code: {:?}", e);
        }
        result
    }

    fn delete_invite_code(&self, invite: &InviteCode) -> Result<(), DBError> {
        debug!("Deleting invite code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = invite.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete invite code: {:?}", e);
        }
        result
    }

    // Platform user methods
    fn create_platform_user(&self, new_user: NewPlatformUser) -> Result<PlatformUser, DBError> {
        debug!("Creating new platform user");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_user.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create platform user: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_id(&self, id: i32) -> Result<PlatformUser, DBError> {
        debug!("Getting platform user by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_id(conn, id)?.ok_or(DBError::PlatformUserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform user by ID: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_uuid(&self, uuid: Uuid) -> Result<PlatformUser, DBError> {
        debug!("Getting platform user by UUID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_uuid(conn, uuid)?.ok_or(DBError::PlatformUserNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform user by UUID: {:?}", e);
        }
        result
    }

    fn get_platform_user_by_email(&self, email: &str) -> Result<Option<PlatformUser>, DBError> {
        debug!("Getting platform user by email");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformUser::get_by_email(conn, email).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get platform user by email: {:?}", e);
        }
        result
    }

    fn update_platform_user(&self, user: &PlatformUser) -> Result<(), DBError> {
        debug!("Updating platform user");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform user: {:?}", e);
        }
        result
    }

    fn update_platform_user_password(
        &self,
        user: &PlatformUser,
        new_password_enc: Vec<u8>,
    ) -> Result<(), DBError> {
        debug!("Updating platform user password");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = user
            .update_password(conn, new_password_enc)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform user password: {:?}", e);
        }
        result
    }

    // Org membership methods
    fn create_org_membership(
        &self,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError> {
        debug!("Creating new org membership");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_membership
            .insert(conn)
            .map_err(DBError::OrgMembershipError);
        if let Err(ref e) = result {
            error!("Failed to create org membership: {:?}", e);
        }
        result
    }

    fn get_org_membership_by_platform_user_and_org(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembership, DBError> {
        debug!("Getting org membership by platform user and org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_by_platform_user_and_org(conn, platform_user_id, org_id)
            .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org membership by platform user and org: {:?}",
                e
            );
        }
        result
    }

    fn get_org_membership_by_platform_user_and_org_with_user(
        &self,
        platform_user_id: Uuid,
        org_id: i32,
    ) -> Result<OrgMembershipWithUser, DBError> {
        debug!("Getting org membership with user info by platform user and org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::get_by_platform_user_and_org_with_user(conn, platform_user_id, org_id)
                .map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get org membership with user info by platform user and org: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_memberships_for_platform_user(
        &self,
        platform_user_id: Uuid,
    ) -> Result<Vec<OrgMembership>, DBError> {
        debug!("Getting all org memberships for platform user");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::get_all_for_platform_user(conn, platform_user_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get all org memberships for platform user: {:?}",
                e
            );
        }
        result
    }

    fn get_all_org_memberships_for_org(&self, org_id: i32) -> Result<Vec<OrgMembership>, DBError> {
        debug!("Getting all org memberships for org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_all_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to get all org memberships for org: {:?}", e);
        }
        result
    }

    fn get_all_org_memberships_with_users_for_org(
        &self,
        org_id: i32,
    ) -> Result<Vec<OrgMembershipWithUser>, DBError> {
        debug!("Getting all org memberships with users for org");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::get_all_with_users_for_org(conn, org_id).map_err(DBError::from);
        if let Err(ref e) = result {
            error!(
                "Failed to get all org memberships with users for org: {:?}",
                e
            );
        }
        result
    }

    fn update_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError> {
        debug!("Updating org membership");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = membership.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update org membership: {:?}", e);
        }
        result
    }

    fn delete_org_membership(&self, membership: &OrgMembership) -> Result<(), DBError> {
        debug!("Deleting org membership");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = membership.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete org membership: {:?}", e);
        }
        result
    }

    fn update_membership_role(
        &self,
        membership: &mut OrgMembership,
        new_role: OrgRole,
    ) -> Result<(), DBError> {
        debug!("Updating org membership role with owner check");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = OrgMembership::update_role_with_owner_check(conn, membership, new_role)
            .map_err(|e| match e {
                OrgMembershipError::DatabaseError(diesel::result::Error::RollbackTransaction) => {
                    DBError::OrgMembershipError(OrgMembershipError::DatabaseError(
                        diesel::result::Error::RollbackTransaction,
                    ))
                }
                _ => DBError::from(e),
            });
        if let Err(ref e) = result {
            error!("Failed to update org membership role: {:?}", e);
        }
        result
    }

    fn delete_membership_with_owner_check(
        &self,
        membership: &OrgMembership,
    ) -> Result<(), DBError> {
        debug!("Deleting org membership with owner check");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result =
            OrgMembership::delete_with_owner_check(conn, membership).map_err(|e| match e {
                OrgMembershipError::DatabaseError(diesel::result::Error::RollbackTransaction) => {
                    DBError::OrgMembershipError(OrgMembershipError::DatabaseError(
                        diesel::result::Error::RollbackTransaction,
                    ))
                }
                _ => DBError::from(e),
            });
        if let Err(ref e) = result {
            error!("Failed to delete org membership: {:?}", e);
        }
        result
    }

    // New project-scoped methods
    fn get_users_for_project(
        &self,
        project_id: i32,
        page: Option<i64>,
        per_page: Option<i64>,
    ) -> Result<(Vec<User>, i64), DBError> {
        debug!("Getting all users for project");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Get total count first
        let total = User::get_count_for_project(conn, project_id)?;

        // Default to first page with 10 items per page
        let page = page.unwrap_or(0);
        let per_page = per_page.unwrap_or(10);

        let users = User::get_all_for_project(conn, project_id, page, per_page)?;

        Ok((users, total))
    }

    fn create_org_with_owner(&self, new_org: NewOrg, owner_id: Uuid) -> Result<Org, DBError> {
        debug!("Creating new organization with owner");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        conn.transaction(|conn| {
            // Create the organization
            let org = new_org.insert(conn).map_err(DBError::from)?;

            // Create ownership membership
            let new_membership = NewOrgMembership::new(owner_id, org.id, OrgRole::Owner);
            new_membership.insert(conn)?;

            Ok(org)
        })
    }

    fn accept_invite_transaction(
        &self,
        invite: &InviteCode,
        new_membership: NewOrgMembership,
    ) -> Result<OrgMembership, DBError> {
        debug!("Starting invite acceptance transaction");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        conn.transaction(|conn| {
            // Create the membership
            let membership = new_membership.insert(conn)?;

            // Mark invite as used
            invite.mark_as_used(conn)?;

            Ok(membership)
        })
    }

    // Project settings methods
    fn get_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
    ) -> Result<Option<ProjectSetting>, DBError> {
        debug!("Getting project settings");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        ProjectSetting::get_by_project_and_category(conn, project_id, category)
            .map_err(DBError::from)
    }

    fn update_project_settings(
        &self,
        project_id: i32,
        category: SettingCategory,
        settings: serde_json::Value,
    ) -> Result<ProjectSetting, DBError> {
        debug!("Updating project settings");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, category.clone())?
        {
            existing.settings = settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            let new_settings = NewProjectSetting {
                project_id,
                category: category.as_str().to_string(),
                settings,
            };
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    fn get_project_email_settings(
        &self,
        project_id: i32,
    ) -> Result<Option<EmailSettings>, DBError> {
        debug!("Getting project email settings");
        let settings = self.get_project_settings(project_id, SettingCategory::Email)?;

        match settings {
            Some(s) => s.get_email_settings().map(Some).map_err(DBError::from),
            None => Ok(None),
        }
    }

    fn update_project_email_settings(
        &self,
        project_id: i32,
        settings: EmailSettings,
    ) -> Result<ProjectSetting, DBError> {
        debug!("Updating project email settings");
        let new_settings = NewProjectSetting::new_email_settings(project_id, settings)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, SettingCategory::Email)?
        {
            existing.settings = new_settings.settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    fn get_project_oauth_settings(
        &self,
        project_id: i32,
    ) -> Result<Option<OAuthSettings>, DBError> {
        debug!("Getting project OAuth settings");
        let settings = self.get_project_settings(project_id, SettingCategory::OAuth)?;

        match settings {
            Some(s) => s.get_oauth_settings().map(Some).map_err(DBError::from),
            None => Ok(None),
        }
    }

    fn update_project_oauth_settings(
        &self,
        project_id: i32,
        settings: OAuthSettings,
    ) -> Result<ProjectSetting, DBError> {
        debug!("Updating project OAuth settings");
        let new_settings = NewProjectSetting::new_oauth_settings(project_id, settings)?;
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;

        // Check if settings exist
        if let Some(mut existing) =
            ProjectSetting::get_by_project_and_category(conn, project_id, SettingCategory::OAuth)?
        {
            existing.settings = new_settings.settings;
            existing.update(conn)?;
            Ok(existing)
        } else {
            // Create new settings
            new_settings.insert(conn).map_err(DBError::from)
        }
    }

    // Platform email verification implementations
    fn create_platform_email_verification(
        &self,
        new_verification: NewPlatformEmailVerification,
    ) -> Result<PlatformEmailVerification, DBError> {
        debug!("Creating new platform email verification");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = new_verification.insert(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to create platform email verification: {:?}", e);
        }
        result
    }

    fn get_platform_email_verification_by_id(
        &self,
        id: i32,
    ) -> Result<PlatformEmailVerification, DBError> {
        debug!("Getting platform email verification by ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_id(conn, id)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform email verification by ID: {:?}", e);
        }
        result
    }

    fn get_platform_email_verification_by_platform_user_id(
        &self,
        platform_user_id: Uuid,
    ) -> Result<PlatformEmailVerification, DBError> {
        debug!("Getting platform email verification by platform user ID");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_platform_user_id(conn, platform_user_id)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!(
                "Failed to get platform email verification by platform user ID: {:?}",
                e
            );
        }
        result
    }

    fn get_platform_email_verification_by_code(
        &self,
        code: Uuid,
    ) -> Result<PlatformEmailVerification, DBError> {
        debug!("Getting platform email verification by code");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = PlatformEmailVerification::get_by_verification_code(conn, code)?
            .ok_or(DBError::PlatformEmailVerificationNotFound);
        if let Err(ref e) = result {
            error!("Failed to get platform email verification by code: {:?}", e);
        }
        result
    }

    fn update_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError> {
        debug!("Updating platform email verification");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.update(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to update platform email verification: {:?}", e);
        }
        result
    }

    fn delete_platform_email_verification(
        &self,
        verification: &PlatformEmailVerification,
    ) -> Result<(), DBError> {
        debug!("Deleting platform email verification");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.delete(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to delete platform email verification: {:?}", e);
        }
        result
    }

    fn verify_platform_email(
        &self,
        verification: &mut PlatformEmailVerification,
    ) -> Result<(), DBError> {
        debug!("Verifying platform email");
        let conn = &mut self.db.get().map_err(|_| DBError::ConnectionError)?;
        let result = verification.verify(conn).map_err(DBError::from);
        if let Err(ref e) = result {
            error!("Failed to verify platform email: {:?}", e);
        }
        result
    }
}

pub(crate) fn setup_db(url: String) -> Arc<dyn DBConnection + Send + Sync> {
    info!("Connecting to database...");
    let manager = ConnectionManager::<PgConnection>::new(url);
    // TODO make pool size bigger, just for testing connection issues
    let pool = Pool::builder()
        .max_size(1) // should be a multiple of 100, our database connection limit
        .test_on_check_out(true)
        .build(manager)
        .expect("Unable to build DB connection pool");
    info!("Connected to database");
    Arc::new(PostgresConnection { db: pool })
}
